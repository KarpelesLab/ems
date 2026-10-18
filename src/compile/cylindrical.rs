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
