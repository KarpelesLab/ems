//! csx plane-wave excitations (`ExcitationType::PlaneWave`, type 10) →
//! [`PlaneWaveSource`]s, installed by [`crate::Simulation`] as
//! [`PlaneWave`](crate::engine::PlaneWave) operators.
//!
//! Mapping rules:
//!
//! * Every primitive of an enabled plane-wave excitation is snapped to the
//!   mesh like a probe box ([`super`] on snapping) and becomes one
//!   total-field box; it must have a volume on the mesh (otherwise it is
//!   ignored with a warning). Several primitives give several independent
//!   sources, so they should not overlap.
//! * `PropDir` is the propagation direction (normalised; a zero vector
//!   ignores the excitation with a warning), the excitation vector
//!   `Excite` the incident E field in V/m at amplitude 1 of the waveform.
//!   A component of `Excite` along `PropDir` is removed (warning); weight
//!   functions are not used for plane waves (warning if set).
//! * The incident wave is `Excite·a(t − t_lag − ξ/v)` with `ξ` the
//!   distance along `PropDir` from the box corner the wave enters first
//!   (see [`crate::engine::planewave`]) and `t_lag` one time step plus the
//!   travel time across the box's first cell (`openems_lag`): openEMS's
//!   phase reference, measured from its output (design §5.16; the
//!   polarisation and direction conventions agree). The excitation's
//!   `Delay` is ignored with a warning, as openEMS ignores it for plane
//!   waves (its output does not change).
//! * The background medium is the material on the box surface
//!   (after rasterisation): the effective edge
//!   materials of the faces that do not lie on the domain boundary must be
//!   one homogeneous, isotropic medium (relative tolerance 1e-4, PEC edges
//!   ignored); its `ε_r`, `μ_r`, `κ` and `σ_m` go into the spec (lossy
//!   backgrounds included). Natively (not for openEMS files,
//!   [`Fdtd::openems_run`]) a surface whose medium varies along one axis
//!   only is a **layered** background
//!   ([`PlaneWaveSpec::with_layers`], ends from the domain boundaries along
//!   that axis: PEC, PMC, else open), and the compiled dispersive poles are
//!   handed to the spec ([`PlaneWaveSpec::with_dispersion`]), so a
//!   dispersive surface gets the exact incident field of its medium
//!   ([`crate::engine::planewave`], layered and dispersive backgrounds).
//!   Otherwise compiling fails with [`Error::Invalid`] naming the plane
//!   wave and the offending edges. For openEMS files dispersive materials
//!   on the surface are not detected (only their static part is seen) and
//!   layered surfaces are an error, as before: openEMS itself injects a
//!   vacuum plane wave whatever the materials (design §5.16.2).
//! * The auxiliary grid's dispersion is matched at the centre of the
//!   waveform: `f0` of a Gaussian (`fc/2` for `f0 = 0`), `f0` of a
//!   sinusoid, otherwise half the maximum frequency of the settings, or
//!   the operator's default (20 cells per wavelength) without one.

use csx::properties::ExcitationType;
use csx::{ContinuousStructure, PropertyKind};

use super::snap_primitive;
use crate::axis::Axis;
use crate::engine::dispersive::Scheme;
use crate::engine::medium::{interior_faces, layered_axis, surface_medium};
use crate::engine::{EdgeMaterials, EdgePole, Grid, LayerEnd, PlaneWaveSpec};
use crate::fdtd::{Boundary, Face, Fdtd};
use crate::waveform::Waveform;
use crate::{Error, Result};

/// A plane-wave excitation of the model (see the [module docs](self)).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PlaneWaveSource {
    /// Property name.
    pub name: String,
    /// Excitation number.
    pub number: u32,
    /// Delay \[s\].
    pub delay: f64,
    /// Geometry and physics of the engine operator (engine node indices).
    pub spec: PlaneWaveSpec,
}

/// Whether `w` is a csx discretisation warning about plane wave `name`
/// (csx does not discretise type-10 excitations; the compile step does).
pub(super) fn is_csx_skip_warning(w: &str, names: &[&str]) -> bool {
    w.contains("is not discretised")
        && names
            .iter()
            .any(|n| w.starts_with(&format!("property {n:?} primitive #")))
}

/// The matching frequency for the settings' waveform (0: operator default).
fn match_frequency(fdtd: &Fdtd) -> f64 {
    match fdtd.get_waveform() {
        Some(w) if w.gaussian_parameters().is_some() => {
            let (f0, fc) = w.gaussian_parameters().expect("checked");
            if f0 > 0.0 { f0 } else { 0.5 * fc }
        }
        Some(&Waveform::Sinusoid { f0 }) => f0,
        _ => fdtd.get_max_frequency().map_or(0.0, |f| 0.5 * f),
    }
}

/// The plane-wave sources of `model` (see the [module docs](self)).
pub(super) fn plane_wave_sources(
    model: &ContinuousStructure,
    fdtd: &Fdtd,
    warnings: &mut Vec<String>,
) -> Vec<PlaneWaveSource> {
    let mut out = Vec::new();
    for p in model.properties() {
        let PropertyKind::Excitation(ex) = &p.kind else {
            continue;
        };
        if ex.excitation_type != ExcitationType::PlaneWave || !ex.enabled {
            continue;
        }
        let name = &p.name;
        let dir = ex.prop_dir.each_ref().map(|s| s.value());
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if !(len.is_finite() && len > 0.0) {
            warnings.push(format!(
                "plane wave {name:?}: no propagation direction (PropDir {dir:?}); ignored"
            ));
            continue;
        }
        let k = dir.map(|d| d / len);
        let e = ex.excite.each_ref().map(|s| s.value());
        let along = e[0] * k[0] + e[1] * k[1] + e[2] * k[2];
        let pol = [0, 1, 2].map(|a| e[a] - along * k[a]);
        let e_len = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        let p_len = (pol[0] * pol[0] + pol[1] * pol[1] + pol[2] * pol[2]).sqrt();
        if !(p_len.is_finite() && p_len > 1e-12 * e_len.max(f64::MIN_POSITIVE)) {
            warnings.push(format!(
                "plane wave {name:?}: excitation vector {e:?} has no component across \
                 the propagation direction; ignored"
            ));
            continue;
        }
        if along.abs() > 1e-9 * e_len {
            warnings.push(format!(
                "plane wave {name:?}: excitation vector component along the propagation \
                 direction removed ({e:?} → {pol:?})"
            ));
        }
        if ex.weight.iter().any(|w| !w.is_unity()) {
            warnings.push(format!(
                "plane wave {name:?}: weighting functions are not used for plane waves"
            ));
        }
        if p.primitives.is_empty() {
            warnings.push(format!("plane wave {name:?}: no primitive; ignored"));
        }
        let delay = ex.delay.value();
        if delay != 0.0 {
            warnings.push(format!(
                "plane wave {name:?}: Delay = {delay:e} s ignored (openEMS does not delay \
                 plane waves)"
            ));
        }
        for prim in &p.primitives {
            let (a, b) = snap_primitive(&model.grid, prim, "plane wave", name, warnings);
            let start = [0, 1, 2].map(|d| a[d].min(b[d]));
            let stop = [0, 1, 2].map(|d| a[d].max(b[d]));
            if (0..3).any(|d| start[d] == stop[d]) {
                warnings.push(format!(
                    "plane wave {name:?}: primitive has no volume on the mesh; ignored"
                ));
                continue;
            }
            out.push(PlaneWaveSource {
                name: name.clone(),
                number: ex.number,
                delay: 0.0,
                spec: PlaneWaveSpec::new(start, stop, k, pol)
                    .with_match_frequency(match_frequency(fdtd)),
            });
        }
    }
    out
}

/// How a layered medium along `axis` ends at the domain faces.
fn layer_ends(fdtd: &Fdtd, axis: Axis) -> [LayerEnd; 2] {
    [false, true].map(|max| match fdtd.get_boundaries()[Face::new(axis, max)] {
        Boundary::Pec => LayerEnd::Pec,
        Boundary::Pmc => LayerEnd::Pmc,
        _ => LayerEnd::Open,
    })
}

/// The stratification axis of the box surface of `spec`, if its medium
/// varies along one axis only and the wave crosses the layers.
fn crosses_layers(spec: &PlaneWaveSpec, grid: &Grid, materials: &EdgeMaterials) -> Option<Axis> {
    let (lo, hi) = (spec.start, spec.stop);
    let ax = layered_axis(grid, materials, lo, hi, interior_faces(grid, lo, hi))?;
    let d = spec.direction;
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    (d[ax.index()].abs() >= 1e-3 * len).then_some(ax)
}

/// Sets the background of every source to the medium on its box surface
/// (see the [module docs](self)); natively (not for openEMS files) also
/// stratified backgrounds and the dispersive poles on the surface.
///
/// # Errors
/// [`Error::Invalid`] if the surface of a box is neither homogeneous nor
/// (natively) stratified along one axis.
pub(super) fn set_backgrounds(
    sources: &mut [PlaneWaveSource],
    grid: &Grid,
    materials: &EdgeMaterials,
    fdtd: &Fdtd,
    poles: &[EdgePole],
    scheme: Scheme,
) -> Result<()> {
    let native = !fdtd.get_openems_run();
    for s in sources {
        let (lo, hi) = (s.spec.start, s.spec.stop);
        let faces = interior_faces(grid, lo, hi);
        if native && !poles.is_empty() {
            s.spec = s.spec.clone().with_dispersion(poles, scheme);
        }
        match surface_medium(grid, materials, lo, hi, faces) {
            Ok(Some(m)) => s.spec = s.spec.clone().with_medium(m),
            Ok(None) => {}
            Err(_) if native && let Some(ax) = crosses_layers(&s.spec, grid, materials) => {
                let [a, b] = layer_ends(fdtd, ax);
                s.spec = s.spec.clone().with_layers(ax).with_layer_ends(a, b);
            }
            Err(e) => {
                return Err(Error::Invalid(format!(
                    "plane wave {:?} (box {lo:?}..={hi:?}): {e}; the total-field box must \
                     lie in a homogeneous background, with scatterers strictly inside",
                    s.name
                )));
            }
        }
    }
    Ok(())
}

/// Extra delay \[s\] of openEMS's incident wave relative to ems's
/// reference (`E_inc = Excite·a(t − Delay)` at the entry corner `r_ref`),
/// measured from openEMS output (design §5.16): one time step plus the
/// travel time across the first cell of the box along the propagation
/// direction, `dt + Σ_a |k̂_a|·Δ_a / v` with `Δ_a` the box's cell at the
/// entry corner on axis `a` and `v` the background phase velocity.
/// [`crate::Simulation`] adds it to the plane wave's `Delay`, so the
/// incident field matches openEMS's in phase (to its dispersion error).
pub(crate) fn openems_lag(spec: &PlaneWaveSpec, grid: &Grid, dt: f64) -> f64 {
    let k = spec.direction;
    let len = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
    let d: f64 = crate::axis::Axis::ALL
        .iter()
        .map(|&ax| {
            let a = ax.index();
            let cell = if k[a] >= 0.0 {
                spec.start[a]
            } else {
                spec.stop[a] - 1
            };
            (k[a] / len).abs() * grid.delta(ax, cell)
        })
        .sum();
    let v = crate::constants::C0 / (spec.eps_r * spec.mu_r).sqrt();
    dt + d / v
}
