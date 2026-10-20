//! Shared machinery of the transmission-line ports (microstrip, coaxial,
//! coplanar and strip line): the choice of three equally
//! spaced voltage planes, the extraction of `γ` and `Z_L` from three
//! voltages and two currents, and the result type.
//!
//! The line model and the extraction formulas are documented in the
//! [`msl`](super::msl) module: they hold for any line whose mode is
//! spatially exponential along the propagation axis (TEM or quasi-TEM),
//! whatever its cross section.

use std::ops::Deref;

use crate::constants::C0;
use crate::post::{Complex64, PortResult};
use crate::sim::Results;
use crate::{Error, Result};

/// Probe layout chosen by a line port's `add_to`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Layout {
    /// Voltage probe positions along the propagation axis (drawing units),
    /// in propagation order.
    pub(crate) v_pos: [f64; 3],
    /// Their spacing \[m\].
    pub(crate) spacing: f64,
    /// Middle voltage probe and reference plane, in metres from the feed
    /// along the propagation direction.
    pub(crate) s_mid: f64,
    pub(crate) s_ref: f64,
}

/// The names of a line port's probes, in propagation order.
pub(crate) struct ProbeNames<'a> {
    pub(crate) voltages: [&'a str; 3],
    pub(crate) currents: [&'a str; 2],
}

/// The line analysis of a port with probes `names` and layout `l`: the
/// waves at the reference plane, with reference impedance `z_ref` or
/// (`None`) the extracted `Z_L(f)`. `kind` and `number` name the port in
/// errors.
pub(crate) fn analyze(
    results: &Results,
    freqs: &[f64],
    names: &ProbeNames<'_>,
    l: &Layout,
    z_ref: Option<Complex64>,
    kind: &str,
    number: u32,
) -> Result<LinePortResult> {
    let get = |name: &str| {
        results
            .probe(name)
            .ok_or_else(|| Error::Invalid(format!("no probe {name:?} in the results")))?
            .dft(freqs)
    };
    let [v1, v2, v3] = names.voltages.map(get);
    let [ia, ib] = names.currents.map(get);
    let (v1, v2, v3, ia, ib) = (v1?, v2?, v3?, ia?, ib?);
    let n = freqs.len();
    let (mut gamma, mut z_line, mut v, mut i) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let delta = l.s_ref - l.s_mid;
    for k in 0..n {
        let (g, z, vk, ik) = extract([v1[k], v2[k], v3[k]], [ia[k], ib[k]], l.spacing, delta);
        gamma.push(g);
        z_line.push(z);
        v.push(vk);
        i.push(ik);
    }
    let port = match z_ref {
        Some(z) => PortResult::new(freqs.to_vec(), v, i, z)?,
        None => {
            PortResult::new_per_frequency(freqs.to_vec(), v, i, z_line.clone()).map_err(|e| {
                Error::Invalid(format!(
                    "{kind} port {number}: unusable line impedance ({e}); is every \
                     frequency inside the excited band?"
                ))
            })?
        }
    };
    Ok(LinePortResult {
        port,
        gamma,
        z_line,
    })
}

/// The line quantities from three voltages `v` at spacing `d` \[m\] and two
/// midpoint currents `i` (see the [module documentation](self)):
/// `(γ, Z_L, V, I)` with `V`, `I` at `delta` metres past the middle probe.
pub(crate) fn extract(
    v: [Complex64; 3],
    i: [Complex64; 2],
    d: f64,
    delta: f64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let sh2 = (v[0] + v[2] - v[1] * 2.0) / (v[1] * 4.0);
    let mut sh = sh2.sqrt();
    // x = γd/2 = asinh(sh), on the branch with β ≥ 0 (asinh is odd).
    let mut x = (sh + (sh * sh + 1.0).sqrt()).ln();
    if x.im < 0.0 {
        x = -x;
        sh = -sh;
    }
    let gamma = x * (2.0 / d);
    let ch = (x.exp() + (-x).exp()) * 0.5;
    let isum = i[0] + i[1];
    let z = -(v[2] - v[0]) / (sh * isum * 2.0);
    let i_mid = isum / (ch * 2.0);
    let a = (v[1] + z * i_mid) * 0.5 * (-gamma * delta).exp();
    let b = (v[1] - z * i_mid) * 0.5 * (gamma * delta).exp();
    (gamma, z, a + b, (a - b) / z)
}

/// Index of the mesh line within a relative tolerance of `x`.
pub(crate) fn line_index(lines: &[f64], x: f64) -> Option<usize> {
    let (k, &l) = lines
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - x).abs().total_cmp(&(b.1 - x).abs()))?;
    let cell = [k.checked_sub(1), Some(k + 1)]
        .into_iter()
        .flatten()
        .filter_map(|j| lines.get(j))
        .map(|&o| (o - l).abs())
        .fold(f64::INFINITY, f64::min);
    let tol = if cell.is_finite() {
        1e-6 * cell
    } else {
        1e-9 * (1.0 + x.abs())
    };
    ((l - x).abs() <= tol).then_some(k)
}

/// Picks three equally spaced consecutive lines past `feed`, not beyond
/// `end`, with the middle one nearest the middle of the port.
pub(crate) fn choose_probes(
    lines: &[f64],
    feed: f64,
    end: f64,
    shift: f64,
    du: f64,
) -> Option<Layout> {
    let sign = (end - feed).signum();
    // Propagation coordinate (drawing units) of a line.
    let s = |x: f64| (x - feed) * sign;
    let len = s(end);
    let tol = 1e-6 * len;
    let mut cand: Vec<f64> = lines
        .iter()
        .copied()
        .filter(|&x| s(x) > tol && s(x) <= len + tol)
        .collect();
    if sign < 0.0 {
        cand.reverse();
    }
    let mid = 0.5 * len;
    cand.windows(3)
        .filter(|t| {
            let (d1, d2) = (s(t[1]) - s(t[0]), s(t[2]) - s(t[1]));
            (d1 - d2).abs() <= 1e-6 * d1.max(d2)
        })
        .min_by(|a, b| (s(a[1]) - mid).abs().total_cmp(&(s(b[1]) - mid).abs()))
        .map(|t| Layout {
            v_pos: [t[0], t[1], t[2]],
            spacing: 0.5 * (s(t[2]) - s(t[0])) * du,
            s_mid: s(t[1]) * du,
            s_ref: (len + shift) * du,
        })
}

/// The analysis of a transmission-line port (such as
/// [`MslPort`](super::MslPort)): the [`PortResult`] at the reference
/// plane (also reachable through `Deref`, so it can be passed to
/// [`s_param`](crate::post::s_param) directly) plus the extracted line
/// parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct LinePortResult {
    port: PortResult,
    gamma: Vec<Complex64>,
    z_line: Vec<Complex64>,
}

impl LinePortResult {
    /// Voltage and current at the reference plane.
    pub fn port(&self) -> &PortResult {
        &self.port
    }

    /// Converts into the [`PortResult`].
    pub fn into_port(self) -> PortResult {
        self.port
    }

    /// Propagation constant `γ = α + jβ` \[1/m\] per frequency.
    pub fn gamma(&self) -> &[Complex64] {
        &self.gamma
    }

    /// Phase constant `β` \[rad/m\] per frequency.
    pub fn beta(&self) -> Vec<f64> {
        self.gamma.iter().map(|g| g.im).collect()
    }

    /// Attenuation constant `α` \[Np/m\] per frequency.
    pub fn alpha(&self) -> Vec<f64> {
        self.gamma.iter().map(|g| g.re).collect()
    }

    /// Effective permittivity `(c0·β/ω)²` per frequency (it includes the
    /// numerical dispersion of the mesh).
    pub fn eps_eff(&self) -> Vec<f64> {
        self.port
            .freqs()
            .iter()
            .zip(&self.gamma)
            .map(|(&f, g)| (C0 * g.im / (std::f64::consts::TAU * f)).powi(2))
            .collect()
    }

    /// Extracted line (characteristic) impedance `Z_L` \[Ω\] per frequency.
    pub fn z_line(&self) -> &[Complex64] {
        &self.z_line
    }
}

impl Deref for LinePortResult {
    type Target = PortResult;

    fn deref(&self) -> &PortResult {
        &self.port
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// A lossy line with forward and backward waves is recovered exactly.
    #[test]
    fn extraction_is_exact_for_exponential_modes() {
        let gamma = c(0.3, 50.0);
        let z = c(48.0, -2.0);
        let (a, b) = (c(1.0, 0.2), c(-0.3, 0.1));
        let (d, s2) = (1e-3, 7e-3);
        let v = |s: f64| a * (-gamma * s).exp() + b * (gamma * s).exp();
        let i = |s: f64| (a * (-gamma * s).exp() - b * (gamma * s).exp()) / z;
        let (gx, zx, vx, ix) = extract(
            [v(s2 - d), v(s2), v(s2 + d)],
            [i(s2 - 0.5 * d), i(s2 + 0.5 * d)],
            d,
            3e-3,
        );
        assert!((gx - gamma).norm() < 1e-9 * gamma.norm(), "{gx}");
        assert!((zx - z).norm() < 1e-9 * z.norm(), "{zx}");
        assert!((vx - v(s2 + 3e-3)).norm() < 1e-9);
        assert!((ix - i(s2 + 3e-3)).norm() < 1e-11);
    }

    #[test]
    fn probe_choice() {
        let lines: Vec<f64> = (0..=10).map(f64::from).collect();
        let l = choose_probes(&lines, 0.0, 10.0, 0.0, 1e-3).unwrap();
        assert_eq!(l.v_pos, [4.0, 5.0, 6.0]);
        assert!((l.spacing - 1e-3).abs() < 1e-15);
        assert!((l.s_mid - 5e-3).abs() < 1e-15 && (l.s_ref - 10e-3).abs() < 1e-15);
        // Reversed port.
        let l = choose_probes(&lines, 10.0, 2.0, 1.0, 1e-3).unwrap();
        assert_eq!(l.v_pos, [7.0, 6.0, 5.0]);
        assert!((l.s_mid - 4e-3).abs() < 1e-15 && (l.s_ref - 9e-3).abs() < 1e-15);
        // Graded mesh: only one equally spaced triple.
        let lines = [0.0, 1.0, 3.0, 4.0, 5.0, 7.0, 10.0];
        let l = choose_probes(&lines, 0.0, 10.0, 0.0, 1.0).unwrap();
        assert_eq!(l.v_pos, [3.0, 4.0, 5.0]);
        let lines = [0.0, 1.0, 3.0, 6.0, 10.0];
        assert!(choose_probes(&lines, 0.0, 10.0, 0.0, 1.0).is_none());
        // The feed line itself is excluded.
        assert!(choose_probes(&[0.0, 1.0, 2.0], 0.0, 2.0, 0.0, 1.0).is_none());
    }

    #[test]
    fn mesh_snapping() {
        let lines = [0.0, 0.5, 1.0];
        assert_eq!(line_index(&lines, 0.5 + 1e-9), Some(1));
        assert_eq!(line_index(&lines, 0.51), None);
        assert_eq!(line_index(&[], 0.0), None);
    }
}
