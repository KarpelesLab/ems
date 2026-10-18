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
//! * The incident wave is `Excite·a(t − Delay − ξ/c)` with `ξ` the distance
//!   along `PropDir` from the box corner the wave enters first (see
//!   [`crate::engine::planewave`]); the background around the box is taken
//!   to be vacuum.
//! * The auxiliary grid's dispersion is matched at the centre of the
//!   waveform: `f0` of a Gaussian (`fc/2` for `f0 = 0`), `f0` of a
//!   sinusoid, otherwise half the maximum frequency of the settings, or
//!   the operator's default (20 cells per wavelength) without one.

use csx::properties::ExcitationType;
use csx::{ContinuousStructure, PropertyKind};

use super::snap_primitive;
use crate::engine::PlaneWaveSpec;
use crate::fdtd::Fdtd;
use crate::waveform::Waveform;

/// A plane-wave excitation of the model (see the [module docs](self)).
#[derive(Debug, Clone, PartialEq)]
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
        Some(&Waveform::Gaussian { f0, .. }) if f0 > 0.0 => f0,
        Some(&Waveform::Gaussian { fc, .. }) => 0.5 * fc,
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
                delay: ex.delay.value(),
                spec: PlaneWaveSpec::new(start, stop, k, pol)
                    .with_match_frequency(match_frequency(fdtd)),
            });
        }
    }
    out
}
