//! SAR dumps: csx `DumpBox` properties of type `SARLocal`, `SAR1g`,
//! `SAR10g` and `SARRaw`, and the per-cell conductivity and mass density
//! they need.
//!
//! Each SAR dump covers the cells of its (snapped) box; a box that is flat
//! along an axis is widened to one cell there ([`CellBox::fitted`]). The
//! conductivity `κ` (per axis, weighting functions applied) and the mass
//! density `ρ` (`Density` of the material, with its weighting function) of
//! every cell are the averages over `n³` sample points per cell (`n` the
//! compile [subsampling](super::CompileOptions::subsampling)), taken with
//! the csx priority query: metal and the background count as `ρ = 0`
//! (the background with its own conductivity), so they are not tissue.
//!
//! Only the conductivity `κ` contributes to the absorbed power; the losses
//! of dispersive (Lorentz/Debye) poles are not included.

use csx::properties::{DumpFileType, DumpType, MaterialQuantity};
use csx::{CoordSystem, PropertyKind, PropertyType, TypeMask, Vec3};

use super::ProbeSpec;
use crate::engine::CellBox;

/// How a SAR dump evaluates its field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SarKind {
    /// Local SAR (`SARLocal`).
    Local,
    /// Averaged over a tissue mass in kg (`SAR1g`: 1e-3, `SAR10g`: 1e-2).
    Averaged(f64),
    /// Raw data (`SARRaw`): the power and density per cell; its field is
    /// the local SAR.
    Raw,
}

impl SarKind {
    /// The kind of a csx dump type, `None` for non-SAR types.
    pub fn of(t: DumpType) -> Option<Self> {
        match t {
            DumpType::SarLocal => Some(Self::Local),
            DumpType::Sar1g => Some(Self::Averaged(1e-3)),
            DumpType::Sar10g => Some(Self::Averaged(1e-2)),
            DumpType::SarRaw => Some(Self::Raw),
            _ => None,
        }
    }
}

/// A SAR dump: its cells, frequencies and per-cell materials.
#[derive(Debug, Clone, PartialEq)]
pub struct SarSpec {
    /// Property name.
    pub name: String,
    /// csx property id.
    pub id: u32,
    /// What to evaluate.
    pub kind: SarKind,
    /// The DumpBox `FileType`.
    pub file_type: DumpFileType,
    /// Analysis frequencies \[Hz\].
    pub frequencies: Vec<f64>,
    /// The cells (engine node indices).
    pub cells: CellBox,
    /// Conductivity per cell and axis \[S/m\] (z fastest within the box).
    pub sigma: Vec<[f64; 3]>,
    /// Mass density per cell \[kg/m³\].
    pub density: Vec<f64>,
}

/// The SAR specs of the SAR dumps among `dumps`. Dumps without
/// frequencies are skipped (the compile warnings report them).
pub(super) fn sar_specs(
    model: &csx::ContinuousStructure,
    dumps: &[ProbeSpec<csx::properties::DumpBox>],
    subsampling: usize,
    warnings: &mut Vec<String>,
) -> Vec<SarSpec> {
    let mut out = Vec::new();
    let mut index = None;
    let nodes = [0, 1, 2].map(|a| model.grid.lines(a).len());
    for d in dumps {
        let Some(kind) = SarKind::of(d.settings.dump_type) else {
            continue;
        };
        if d.settings.probe.frequencies.is_empty() {
            continue;
        }
        let cells = match CellBox::new(d.start, d.stop).fitted(nodes) {
            Ok(c) => c,
            Err(e) => {
                warnings.push(format!("SAR dump {:?}: {e}; ignored", d.name));
                continue;
            }
        };
        let index = index.get_or_insert_with(|| {
            model.spatial_index(
                CoordSystem::Cartesian,
                TypeMask::MATERIALS | TypeMask::of(PropertyType::Metal),
            )
        });
        let (mut sigma, density) = cell_media(model, index, &cells, subsampling.max(1));
        let mut bad = 0usize;
        for s in sigma.iter_mut().flatten() {
            if !(s.is_finite() && *s >= 0.0) {
                *s = 0.0;
                bad += 1;
            }
        }
        if bad > 0 {
            warnings.push(format!(
                "SAR dump {:?}: {bad} negative or non-finite cell conductivities set to 0",
                d.name
            ));
        }
        if density.iter().all(|&r| r == 0.0) {
            warnings.push(format!(
                "SAR dump {:?}: no material with a density in its box; the SAR is zero",
                d.name
            ));
        }
        out.push(SarSpec {
            name: d.name.clone(),
            id: d.id,
            kind,
            file_type: d.settings.file_type,
            frequencies: d.settings.probe.frequencies.clone(),
            cells,
            sigma,
            density,
        });
    }
    out
}

/// Cell-averaged conductivity and density over `cells` (see the module
/// docs).
fn cell_media(
    model: &csx::ContinuousStructure,
    index: &csx::structure::SpatialIndex<'_>,
    cells: &CellBox,
    n: usize,
) -> (Vec<[f64; 3]>, Vec<f64>) {
    let lines = [0, 1, 2].map(|a| model.grid.lines(a));
    let bg = model.background.kappa.value();
    let (lo, c) = (cells.lo(), cells.cells());
    let mut sigma = Vec::with_capacity(cells.num_cells());
    let mut density = Vec::with_capacity(cells.num_cells());
    let w = 1.0 / (n * n * n) as f64;
    for i in 0..c[0] {
        for j in 0..c[1] {
            for k in 0..c[2] {
                let cell = [lo[0] + i, lo[1] + j, lo[2] + k];
                let mut s = [0.0; 3];
                let mut r = 0.0;
                for a in 0..n {
                    for b in 0..n {
                        for cc in 0..n {
                            let t = [a, b, cc];
                            let p: [f64; 3] = core::array::from_fn(|d| {
                                let (x0, x1) = (lines[d][cell[d]], lines[d][cell[d] + 1]);
                                x0 + (x1 - x0) * (t[d] as f64 + 0.5) / n as f64
                            });
                            let p = Vec3::new(p[0], p[1], p[2]);
                            match index.query(p).map(|h| &h.property.kind) {
                                Some(PropertyKind::Metal) => {}
                                Some(k) if let Some(m) = k.material() => {
                                    let kap = m.values_at(MaterialQuantity::Kappa, p);
                                    for d in 0..3 {
                                        s[d] += w * kap[d];
                                    }
                                    r += w * m.density_at(p);
                                }
                                _ => {
                                    for x in &mut s {
                                        *x += w * bg;
                                    }
                                }
                            }
                        }
                    }
                }
                sigma.push(s);
                density.push(if r.is_finite() { r.max(0.0) } else { 0.0 });
            }
        }
    }
    (sigma, density)
}
