//! Compile-step rules for cylindrical grids (design §5.15).
//!
//! Most of the model maps as on a Cartesian grid: csx discretises in the
//! grid's `(ρ, α, z)` coordinates (with the Jacobian in its volume
//! weights), and the engine takes the geometry from the grid. Features
//! whose models assume a Cartesian mesh are not available yet and are
//! dropped with a warning: conducting sheets (treated as PEC), dispersive
//! poles (the static part `ε∞`, `μ∞` stays), plane-wave excitations, SAR
//! dumps and waveguide mode probes.

use csx::discretize::MaterialGrid;
use csx::properties::PropertyKind;

use super::ProbeSpec;
use super::sar::SarKind;
use super::sheet::SheetMap;
use crate::engine::Grid;
use crate::fdtd::{Cpml, Face};

/// Conducting-sheet edges as PEC edges (with a warning).
pub(super) fn sheets_as_pec(mg: &MaterialGrid, warnings: &mut Vec<String>) -> SheetMap {
    let mut out = SheetMap::default();
    for d in 0..3 {
        for (idx, &s) in mg.sheet[d].iter().enumerate() {
            if s != 0 {
                out.pec.push((d, idx));
            }
        }
    }
    if !out.pec.is_empty() {
        warnings.push(format!(
            "conducting sheets on a cylindrical grid are treated as PEC ({} edges)",
            out.pec.len()
        ));
    }
    out
}

/// Warns about the dispersive properties of `model`, whose poles are not
/// mapped on cylindrical grids.
pub(super) fn no_dispersive(model: &csx::ContinuousStructure, warnings: &mut Vec<String>) {
    for p in model.properties() {
        if matches!(
            p.kind,
            PropertyKind::LorentzMaterial(_) | PropertyKind::DebyeMaterial(_)
        ) && !p.primitives.is_empty()
        {
            warnings.push(format!(
                "dispersive material {:?}: poles are not supported on cylindrical grids; \
                 only its static values are used",
                p.name
            ));
        }
    }
}

/// Removes the SAR dumps from `dumps` (with a warning each).
pub(super) fn no_sar(
    dumps: &mut Vec<ProbeSpec<csx::properties::DumpBox>>,
    warnings: &mut Vec<String>,
) {
    dumps.retain(|d| {
        let sar = SarKind::of(d.settings.dump_type).is_some();
        if sar {
            warnings.push(format!(
                "SAR dump {:?}: SAR needs a Cartesian grid; ignored",
                d.name
            ));
        }
        !sar
    });
}

/// Drops (with a warning) the absorbing boundaries that a cylindrical
/// grid ignores: on the α faces of a closed grid (periodic) and on the
/// `ρ_min` face of a grid with an axis.
pub(super) fn drop_ignored_boundaries(
    grid: &Grid,
    mur: &mut Vec<(Face, f64)>,
    pml: &mut Vec<(Face, Cpml)>,
    warnings: &mut Vec<String>,
) {
    let ignored = |f: Face| {
        (grid.is_closed() && matches!(f, Face::YMin | Face::YMax))
            || (grid.has_axis() && f == Face::XMin)
    };
    let mut warn = |f: Face| {
        warnings.push(format!(
            "absorbing boundary on {f} ignored: on this cylindrical grid it is {}",
            if f == Face::XMin {
                "the axis"
            } else {
                "periodic (the azimuth spans 2π)"
            }
        ));
    };
    mur.retain(|m| {
        let keep = !ignored(m.0);
        if !keep {
            warn(m.0);
        }
        keep
    });
    pml.retain(|p| {
        let keep = !ignored(p.0);
        if !keep {
            warn(p.0);
        }
        keep
    });
}

/// Applies the [`Fdtd::multigrid`](crate::Fdtd::multigrid) radii (drawing
/// units) to `grid`: each radius snaps to the nearest `ρ` line of the csx
/// grid (with a warning if it is not on one). Cartesian grids ignore them
/// with a warning.
///
/// # Errors
/// [`Error::Invalid`](crate::Error::Invalid) from
/// [`Grid::with_multigrid`] (zone boundary at the first or last line, two
/// radii on the same line, α cells not a multiple of `2^L`, …).
pub(crate) fn multigrid(
    grid: Grid,
    csx_grid: &csx::RectilinearGrid,
    radii: &[f64],
    warnings: &mut Vec<String>,
) -> crate::Result<Grid> {
    if radii.is_empty() {
        return Ok(grid);
    }
    if !grid.is_cylindrical() {
        warnings.push(format!(
            "multigrid radii {radii:?} ignored: multigrid needs a cylindrical grid"
        ));
        return Ok(grid);
    }
    let rho = csx_grid.lines(0);
    let rings: Vec<usize> = radii
        .iter()
        .map(|&r| {
            let i = (0..rho.len())
                .min_by(|&a, &b| (rho[a] - r).abs().total_cmp(&(rho[b] - r).abs()))
                .unwrap_or(0);
            if (rho[i] - r).abs() > 1e-9 * r.abs().max(rho[i].abs()) {
                warnings.push(format!(
                    "multigrid radius {r} snapped to the rho line {} (index {i})",
                    rho[i]
                ));
            }
            i
        })
        .collect();
    grid.with_multigrid(&rings)
}

/// Warns about excitation and lumped-element edges that the multigrid
/// zones of `grid` remove (they have no effect, or make the element fail)
/// or couple (α edges of a coarse arc, slave z edges on a zone boundary:
/// the element then acts through the zone projection).
pub(super) fn multigrid_edge_warnings(
    grid: &Grid,
    node_map: &super::NodeMap,
    excitations: &[csx::discretize::ExcitationField],
    lumped: &[csx::discretize::LumpedEdge],
    warnings: &mut Vec<String>,
) {
    use crate::Axis;
    if !grid.has_multigrid() {
        return;
    }
    let coupled = |axis: Axis, [i, j, _]: [usize; 3]| match axis {
        Axis::Y => !(i == 0 && grid.has_axis()) && grid.alpha_factor(i) > 1,
        Axis::Z => {
            let (inn, out) = grid.ring_factors(i);
            !(i == 0 && grid.has_axis()) && inn > out && j % inn != 0
        }
        Axis::X => false,
    };
    for ex in excitations {
        let magnetic = ex.excitation_type.is_magnetic();
        let removed = ex
            .amplitudes
            .iter()
            .filter(|(e, _)| {
                let (axis, node) = node_map.edge(*e);
                if magnetic {
                    grid.mg_removed_h(axis, node)
                } else {
                    grid.mg_removed_e(axis, node)
                }
            })
            .count();
        if removed > 0 {
            warnings.push(format!(
                "excitation {:?}: {removed} of its {} edges lie off the alpha lines kept by the \
                 multigrid zones and have no effect",
                ex.name,
                ex.amplitudes.len()
            ));
        }
    }
    let mut names: Vec<(&str, usize, usize)> = Vec::new();
    for le in lumped {
        let (axis, node) = node_map.edge(le.edge);
        let (r, c) = (grid.mg_removed_e(axis, node), coupled(axis, node));
        if r || c {
            match names.iter_mut().find(|n| n.0 == le.name) {
                Some(n) => {
                    n.1 += usize::from(r);
                    n.2 += usize::from(c);
                }
                None => names.push((&le.name, usize::from(r), usize::from(c))),
            }
        }
    }
    for (name, r, c) in names {
        warnings.push(format!(
            "lumped element {name:?}: {r} edge(s) removed and {c} coupled by the multigrid \
             zones (place lumped elements on the rho / z edges of kept alpha lines, or outside \
             the zones)"
        ));
    }
}
