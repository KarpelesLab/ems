//! Compile-step rules for cylindrical grids (design §5.15).
//!
//! Most of the model maps as on a Cartesian grid: csx discretises in the
//! grid's `(ρ, α, z)` coordinates (with the Jacobian in its volume
//! weights), and the engine takes the geometry from the grid. Features
//! whose models assume a Cartesian mesh are not available yet and are
//! dropped with a warning: plane-wave excitations (openEMS ignores them on
//! cylindrical grids too) and mass-averaged SAR dumps (local SAR works,
//! with the `ρ` metric in the cell volumes). Conducting sheets run their slab
//! model with the cylindrical metric ([`sheet_edges`]); conformal PEC
//! handles solid metal (`compile::conformal`, cut z faces with the `ρ`
//! metric), not thin conformal sheets. Dispersive poles are mapped (midpoint rule in
//! `(ρ, α, z)`, physical edge areas; within 1e-3 of openEMS on a
//! Drude-filled coax). Waveguide mode probes work with a `z` normal (exact
//! sampling, see `compile/mode.rs`).

use super::sheet::SheetSpec;
use crate::engine::Grid;
use crate::fdtd::{Cpml, Face};

/// Completes csx's sheet edges on a cylindrical grid.
///
/// csx selects the edges of a planar metal or conducting-sheet primitive
/// within its bounding box, whose α range it normalises to `[−π, π)`: on
/// a grid with α lines in `[0, 2π]`, a sheet over the full circle then only
/// gets the edges with `α ≤ π`, and a sheet at `α > π` none (with a "not
/// on a mesh plane" warning). Here every planar (2-D, flat in one of
/// `ρ`, `α`, `z`) `Metal` or `ConductingSheet` primitive is selected again
/// on its mesh plane: the edges of the plane whose midpoints the primitive
/// contains (tested at `α`, `α ± 2π`) are added to the PEC mask or the
/// sheet ids (PEC and csx's own assignment win), and csx's warnings about
/// such a primitive are removed when it got edges here.
///
/// Returns the sheets the ids of `mg.sheet` refer to (csx's, plus one per
/// property found here only).
pub(super) fn wrapped_sheets(
    model: &csx::ContinuousStructure,
    mg: &mut csx::discretize::MaterialGrid,
    node_map: &super::NodeMap,
    snap_tol: f64,
    warnings: &mut Vec<String>,
) -> Vec<SheetSpec> {
    use csx::{CoordSystem, PropertyKind, Vec3};
    use std::f64::consts::TAU;
    let mut specs = SheetSpec::of(mg);
    let lines = [0, 1, 2].map(|a| model.grid.lines(a).to_vec());
    let n = lines.clone().map(|l| l.len());
    for p in model.properties() {
        let sheet = match &p.kind {
            PropertyKind::Metal => None,
            PropertyKind::ConductingSheet(c) => Some((c.conductivity.value(), c.thickness.value())),
            _ => continue,
        };
        for (pi, prim) in p.primitives.iter().enumerate() {
            if prim.dimension() != 2 {
                continue;
            }
            let b = prim.bounding_box_in(CoordSystem::Cylindrical, model.coord_system);
            if b.is_empty() {
                continue;
            }
            let flat: Vec<usize> = (0..3)
                .filter(|&a| {
                    let k = model.grid.nearest_line(a, b.min[a]).unwrap_or(0);
                    let l = &lines[a];
                    let cell = if l.len() > 1 {
                        (l[k.min(l.len() - 2) + 1] - l[k.min(l.len() - 2)]).abs()
                    } else {
                        0.0
                    };
                    b.max[a] - b.min[a] <= snap_tol * cell + 1e-12 * b.min[a].abs()
                })
                .collect();
            let [nrm] = flat[..] else { continue };
            let Some(k) = super::sheet::plane_line(model, nrm, b.min[nrm], snap_tol) else {
                continue;
            };
            let id = sheet.map(|(s, t)| {
                let found = specs
                    .iter()
                    .position(|x| x.name == p.name && x.conductivity == s && x.thickness == t);
                found.unwrap_or_else(|| {
                    specs.push(SheetSpec {
                        name: p.name.clone(),
                        conductivity: s,
                        thickness: t,
                    });
                    specs.len() - 1
                }) as u32
                    + 1
            });
            let mut hits = 0usize;
            for d in (0..3).filter(|&d| d != nrm) {
                let t = 3 - d - nrm;
                for i in 0..n[d] - 1 {
                    for j in 0..n[t] {
                        let mut node = [0; 3];
                        node[d] = i;
                        node[t] = j;
                        node[nrm] = k;
                        let mut q = [0.0; 3];
                        q[d] = 0.5 * (lines[d][i] + lines[d][i + 1]);
                        q[t] = lines[t][j];
                        q[nrm] = lines[nrm][k];
                        let inside = [0.0, TAU, -TAU].iter().any(|&s| {
                            prim.contains_in(
                                Vec3::new(q[0], q[1] + s, q[2]),
                                CoordSystem::Cylindrical,
                                model.coord_system,
                            )
                        });
                        if !inside {
                            continue;
                        }
                        hits += 1;
                        let idx = node_map.csx_index(node);
                        match id {
                            None => mg.pec[d][idx] = true,
                            Some(id) if mg.sheet[d][idx] == 0 => mg.sheet[d][idx] = id,
                            Some(_) => {}
                        }
                    }
                }
            }
            if hits > 0 {
                let tag = format!("property {:?} primitive #{pi} ", p.name);
                warnings.retain(|w| {
                    !(w.starts_with(&tag)
                        && (w.contains("is not on a mesh plane")
                            || w.contains("covers no grid edge")))
                });
            }
        }
    }
    specs
}

/// Cylindrical rules for conducting-sheet edges (design §5.9, §5.15): the
/// sheet model runs on `ρ = const`, `α = const` and `z = const` sheets with
/// the physical edge lengths and strip widths of the grid
/// ([`ConductingSheet`](crate::engine::ConductingSheet)). Removed here
/// (with a warning): z edges on the axis, whose dual face is the whole disc
/// around the axis (they stay ordinary edges). Sheet edges inside the
/// multigrid zones (§5.15.1) are kept with a warning: the zone projection
/// acts on the face voltage only.
pub(super) fn sheet_edges(
    grid: &Grid,
    edges: &mut Vec<crate::engine::SheetEdge>,
    warnings: &mut Vec<String>,
) {
    use crate::Axis;
    let before = edges.len();
    edges.retain(|e| !(grid.has_axis() && e.axis == Axis::Z && e.node[0] == 0));
    if edges.len() < before {
        warnings.push(format!(
            "{} conducting-sheet edge(s) on the axis are left as ordinary edges",
            before - edges.len()
        ));
    }
    let zoned = edges
        .iter()
        .filter(|e| grid.alpha_factor(e.node[0]) > 1)
        .count();
    if zoned > 0 {
        warnings.push(format!(
            "{zoned} conducting-sheet edge(s) inside the multigrid zones: the zones couple only \
             the face voltage of the sheet (approximate)"
        ));
    }
}

/// Drops (with a warning) the absorbing boundaries that a cylindrical
/// grid ignores: on the α faces of a closed grid (periodic) and on the
/// `ρ_min` face of a grid with an axis.
pub(super) fn drop_ignored_boundaries(
    grid: &Grid,
    mur: &mut Vec<(Face, Option<f64>)>,
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
