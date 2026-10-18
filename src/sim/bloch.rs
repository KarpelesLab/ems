//! The Bloch mode of [`Simulation`](super::Simulation) (design §5.6.2):
//! selection of the transverse wavevector, and the mapping of plane-wave
//! excitations onto [`PeriodicPlaneWave`] sources.

use crate::axis::Axis;
use crate::compile::PlaneWaveSource;
use crate::constants::C0;
use crate::engine::{PeriodicPlaneWaveSpec, PlaneWaveSpec};
use crate::fdtd::Fdtd;
use crate::{Error, Result};

/// How a plane-wave box sits in a periodic cell: it spans every periodic
/// axis and all but one other axis completely; that one is the normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CellPlaneWave {
    pub axis: Axis,
    pub start: usize,
    pub stop: usize,
}

/// The periodic axes of the settings.
fn periodic_axes(fdtd: &Fdtd) -> [bool; 3] {
    Axis::ALL.map(|a| fdtd.get_boundaries().is_periodic(a))
}

/// The normal axis and planes of a plane-wave box spanning a periodic
/// cell, or `None` if the box does not (or there is no periodic axis).
pub(crate) fn cell_geometry(
    spec: &PlaneWaveSpec,
    nodes: [usize; 3],
    periodic: [bool; 3],
) -> Option<CellPlaneWave> {
    if !periodic.iter().any(|&p| p) {
        return None;
    }
    let full = |a: usize| spec.start[a] == 0 && spec.stop[a] == nodes[a] - 1;
    if (0..3).any(|a| periodic[a] && !full(a)) {
        return None;
    }
    let partial: Vec<usize> = (0..3).filter(|&a| !full(a)).collect();
    let &[n] = partial.as_slice() else {
        return None;
    };
    Some(CellPlaneWave {
        axis: Axis::ALL[n],
        start: spec.start[n],
        stop: spec.stop[n],
    })
}

/// The Bloch wavevector of the run, if it is a Bloch run: the explicit
/// [`Fdtd::bloch_wavevector`], else the one of the oblique plane waves
/// spanning a periodic cell, `k_t = (2π f_d/c)·k̂_t` at the design
/// frequency `f_d` ([`Fdtd::bloch_frequency`], else the plane wave's
/// dispersion-matching frequency).
///
/// # Errors
/// [`Error::Invalid`] if the settings fail [`Fdtd::validate`]'s Bloch
/// checks, an oblique plane wave has no design frequency or a transverse
/// direction along a non-periodic axis, or several oblique plane waves
/// give different wavevectors.
pub(crate) fn wavevector(
    fdtd: &Fdtd,
    plane_waves: &[PlaneWaveSource],
    nodes: [usize; 3],
) -> Result<Option<[f64; 3]>> {
    if let Some(k) = fdtd.get_bloch_wavevector() {
        return Ok(Some(k));
    }
    let periodic = periodic_axes(fdtd);
    let mut found: Option<([f64; 3], &str)> = None;
    for pw in plane_waves {
        let Some(cell) = cell_geometry(&pw.spec, nodes, periodic) else {
            continue;
        };
        let d = unit(pw.spec.direction);
        let n = cell.axis.index();
        let mut t = d;
        t[n] = 0.0;
        if t.iter().all(|x| x.abs() < 1e-12) {
            continue;
        }
        if let Some(a) = (0..3).find(|&a| a != n && !periodic[a] && t[a].abs() >= 1e-12) {
            return Err(Error::Invalid(format!(
                "plane wave {:?}: oblique direction along {}, whose faces are not periodic",
                pw.name,
                Axis::ALL[a]
            )));
        }
        let f = fdtd
            .get_bloch_frequency()
            .unwrap_or(pw.spec.match_frequency);
        if !(f.is_finite() && f > 0.0) {
            return Err(Error::Invalid(format!(
                "plane wave {:?}: an oblique plane wave in a periodic cell needs a design \
                 frequency (Fdtd::bloch_frequency or a waveform with a centre frequency)",
                pw.name
            )));
        }
        let k0 = 2.0 * core::f64::consts::PI * f * (pw.spec.eps_r * pw.spec.mu_r).sqrt() / C0;
        let k = t.map(|x| k0 * x);
        if let Some((k1, other)) = found {
            let scale = k1.iter().chain(&k).fold(0.0f64, |m, x| m.max(x.abs()));
            if (0..3).any(|a| (k1[a] - k[a]).abs() > 1e-9 * scale) {
                return Err(Error::Invalid(format!(
                    "plane waves {other:?} and {:?} need different Bloch wavevectors \
                     ({k1:?} and {k:?} rad/m); one run has one",
                    pw.name
                )));
            }
        } else {
            found = Some((k, &pw.name));
        }
    }
    Ok(found.map(|(k, _)| k))
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|x| x / l)
}

/// The periodic-cell source of plane wave `pw` in a Bloch run with
/// wavevector `k`, or a warning text if it cannot be one.
pub(crate) fn periodic_spec(
    pw: &PlaneWaveSource,
    fdtd: &Fdtd,
    nodes: [usize; 3],
    k: [f64; 3],
) -> core::result::Result<PeriodicPlaneWaveSpec, String> {
    let Some(cell) = cell_geometry(&pw.spec, nodes, periodic_axes(fdtd)) else {
        return Err(format!(
            "plane wave {:?}: in a Bloch run a plane-wave box must span the periodic \
             cell (whole extent of every periodic axis, and of all but one other axis); \
             ignored",
            pw.name
        ));
    };
    let n = cell.axis.index();
    let d = unit(pw.spec.direction);
    if d[n] == 0.0 {
        return Err(format!(
            "plane wave {:?}: direction parallel to the cell planes; ignored",
            pw.name
        ));
    }
    if k[n] != 0.0 {
        return Err(format!(
            "plane wave {:?}: the Bloch wavevector has a component along the normal \
             axis {}; ignored",
            pw.name, cell.axis
        ));
    }
    let mut p = pw.spec.polarization;
    p[n] = 0.0;
    if p.iter().all(|&x| x == 0.0) {
        return Err(format!(
            "plane wave {:?}: excitation vector has no component tangential to the \
             cell planes; ignored",
            pw.name
        ));
    }
    Ok(
        PeriodicPlaneWaveSpec::new(cell.axis, cell.start, cell.stop, d[n] > 0.0, k, p)
            .with_background(pw.spec.eps_r, pw.spec.mu_r),
    )
}

/// The warning for an explicit Bloch wavevector that differs from the
/// one plane wave `pw` would give at the design frequency (`None` if they
/// agree, or if there is no design frequency).
pub(crate) fn direction_mismatch(
    pw: &PlaneWaveSource,
    fdtd: &Fdtd,
    spec: &PeriodicPlaneWaveSpec,
) -> Option<String> {
    let f = fdtd
        .get_bloch_frequency()
        .unwrap_or(pw.spec.match_frequency);
    let n = spec.axis.index();
    let k0 = 2.0 * core::f64::consts::PI * f * (pw.spec.eps_r * pw.spec.mu_r).sqrt() / C0;
    let mut own = unit(pw.spec.direction).map(|x| k0 * x);
    own[n] = 0.0;
    let scale = own
        .iter()
        .chain(&spec.k_t)
        .fold(0.0f64, |m, x| m.max(x.abs()));
    let agree =
        !(f.is_finite() && f > 0.0) || (0..3).all(|a| (own[a] - spec.k_t[a]).abs() <= 1e-9 * scale);
    (!agree).then(|| {
        format!(
            "plane wave {:?}: the explicit Bloch wavevector {:?} rad/m sets the incidence \
             (its direction gives {own:?} rad/m at {f} Hz)",
            pw.name, spec.k_t
        )
    })
}
