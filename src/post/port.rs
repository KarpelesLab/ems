//! Port analysis: power waves, S-parameters, impedance and power.
//!
//! # Power waves
//!
//! For a port with voltage `V`, current `I` (flowing *into* the structure)
//! and reference impedance `Z_ref` with `Re Z_ref > 0`, the power waves of
//! K. Kurokawa, "Power waves and the scattering matrix", IEEE Trans. MTT
//! 13(2), 1965 (also D. M. Pozar, *Microwave Engineering*, power waves
//! and generalized scattering parameters) are
//!
//! ```text
//! a = (V + Z_ref·I) / (2·√Re Z_ref)        (incident)
//! b = (V − Z_ref*·I) / (2·√Re Z_ref)       (reflected)
//! ```
//!
//! For a real `Z_ref` they reduce to the usual travelling-wave amplitudes.
//! `Z_ref` may depend on frequency ([`PortResult::new_per_frequency`]),
//! e.g. the characteristic impedance of a transmission-line port.
//! The derived quantities are `Z_in = V/I`, `S_11 = b/a` for the excited
//! port and `S_ij = b_i/a_j` between ports ([`s_param`]), with all ports
//! but `j` passive (terminated in their reference impedance) in the run
//! that excites `j`.
//!
//! # Powers
//!
//! With peak phasors, `P_inc = ½|a|²`, `P_ref = ½|b|²` and
//! `P_acc = ½·Re(V·I*) = P_inc − P_ref` (the last identity is exact for any
//! complex `Z_ref`, Kurokawa 1965). When `V` and `I` are the DFTs of
//! transients (as produced by [`LumpedPort::analyze`](crate::engine::LumpedPort::analyze)),
//! these are spectral densities (V²·s²/Ω); ratios between them, and the
//! S-parameters, are independent of that normalisation.
//!
//! # Voltage and current waves
//!
//! [`PortResult::v_inc`] and friends split `V` and `I` into the
//! travelling waves of a line with characteristic impedance `Z_ref`:
//!
//! ```text
//! V_inc = (V + Z_ref·I)/2,   V_ref = (V − Z_ref·I)/2,
//! I_inc = V_inc/Z_ref,       I_ref = −V_ref/Z_ref,
//! ```
//!
//! so that `V = V_inc + V_ref` and `I = I_inc + I_ref` for any `Z_ref`. For
//! a real `Z_ref`, `V_inc = √Z_ref·a` and `V_ref = √Z_ref·b`.

use super::complex::Complex64;
use crate::{Error, Result};

/// Frequency-domain voltage and current of one port, with derived
/// quantities; see the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
pub struct PortResult {
    freqs: Vec<f64>,
    v: Vec<Complex64>,
    i: Vec<Complex64>,
    /// One per frequency.
    z_ref: Vec<Complex64>,
}

impl PortResult {
    /// From spectra `v`, `i` (one per frequency in `freqs`) and reference
    /// impedance `z_ref`.
    ///
    /// Fails if the lengths differ or `Re z_ref` is not positive and
    /// finite.
    pub fn new(
        freqs: Vec<f64>,
        v: Vec<Complex64>,
        i: Vec<Complex64>,
        z_ref: Complex64,
    ) -> Result<Self> {
        let z_ref = vec![z_ref; freqs.len().max(1)];
        Self::checked(freqs, v, i, z_ref)
    }

    /// Like [`new`](Self::new), with a frequency-dependent reference
    /// impedance (one value per frequency).
    ///
    /// Fails if the lengths differ or any `Re z_ref` is not positive and
    /// finite.
    pub fn new_per_frequency(
        freqs: Vec<f64>,
        v: Vec<Complex64>,
        i: Vec<Complex64>,
        z_ref: Vec<Complex64>,
    ) -> Result<Self> {
        if z_ref.len() != freqs.len() {
            return Err(Error::Invalid(format!(
                "port result: {} frequencies but {} reference impedances",
                freqs.len(),
                z_ref.len()
            )));
        }
        Self::checked(freqs, v, i, z_ref)
    }

    fn checked(
        freqs: Vec<f64>,
        v: Vec<Complex64>,
        i: Vec<Complex64>,
        mut z_ref: Vec<Complex64>,
    ) -> Result<Self> {
        if v.len() != freqs.len() || i.len() != freqs.len() {
            return Err(Error::Invalid(format!(
                "port result: {} frequencies but {} voltages, {} currents and {} \
                 reference impedances",
                freqs.len(),
                v.len(),
                i.len(),
                z_ref.len()
            )));
        }
        if let Some(z) = z_ref.iter().find(|z| !(z.is_finite() && z.re > 0.0)) {
            return Err(Error::Invalid(format!(
                "reference impedance {z} must have a positive real part"
            )));
        }
        z_ref.truncate(freqs.len());
        Ok(Self { freqs, v, i, z_ref })
    }

    /// Returns the same spectra with another (constant) reference
    /// impedance.
    pub fn with_z_ref(&self, z_ref: Complex64) -> Result<Self> {
        Self::new(self.freqs.clone(), self.v.clone(), self.i.clone(), z_ref)
    }

    /// Returns the same spectra with a frequency-dependent reference
    /// impedance (one value per frequency).
    pub fn with_z_ref_per_frequency(&self, z_ref: Vec<Complex64>) -> Result<Self> {
        Self::new_per_frequency(self.freqs.clone(), self.v.clone(), self.i.clone(), z_ref)
    }

    /// Analysis frequencies \[Hz\].
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// Voltage spectrum `V(f)`.
    pub fn voltage(&self) -> &[Complex64] {
        &self.v
    }

    /// Current spectrum `I(f)` (into the structure).
    pub fn current(&self) -> &[Complex64] {
        &self.i
    }

    /// Reference impedance \[Ω\], one value per frequency (all equal
    /// unless built with [`new_per_frequency`](Self::new_per_frequency)).
    pub fn z_ref(&self) -> &[Complex64] {
        &self.z_ref
    }

    /// `(V, I, Z_ref)` per frequency.
    fn viz(&self) -> impl Iterator<Item = (Complex64, Complex64, Complex64)> + '_ {
        self.v
            .iter()
            .zip(&self.i)
            .zip(&self.z_ref)
            .map(|((&v, &i), &z)| (v, i, z))
    }

    /// Incident power wave `a = (V + Z_ref·I)/(2√Re Z_ref)`.
    pub fn a(&self) -> Vec<Complex64> {
        self.viz()
            .map(|(v, i, z)| (v + z * i) * (0.5 / z.re.sqrt()))
            .collect()
    }

    /// Reflected power wave `b = (V − Z_ref*·I)/(2√Re Z_ref)`.
    pub fn b(&self) -> Vec<Complex64> {
        self.viz()
            .map(|(v, i, z)| (v - z.conj() * i) * (0.5 / z.re.sqrt()))
            .collect()
    }

    /// Incident voltage wave `V_inc = (V + Z_ref·I)/2` (see
    /// [Voltage and current waves](self#voltage-and-current-waves)).
    pub fn v_inc(&self) -> Vec<Complex64> {
        self.viz().map(|(v, i, z)| (v + z * i) * 0.5).collect()
    }

    /// Reflected voltage wave `V_ref = (V − Z_ref·I)/2`.
    pub fn v_ref(&self) -> Vec<Complex64> {
        self.viz().map(|(v, i, z)| (v - z * i) * 0.5).collect()
    }

    /// Incident current wave `I_inc = V_inc/Z_ref` (into the structure).
    pub fn i_inc(&self) -> Vec<Complex64> {
        self.viz().map(|(v, i, z)| (v / z + i) * 0.5).collect()
    }

    /// Reflected current wave `I_ref = −V_ref/Z_ref` (into the structure,
    /// so `I = I_inc + I_ref`).
    pub fn i_ref(&self) -> Vec<Complex64> {
        self.viz().map(|(v, i, z)| (i - v / z) * 0.5).collect()
    }

    /// Input impedance `Z_in = V/I` \[Ω\].
    pub fn z_in(&self) -> Vec<Complex64> {
        self.v.iter().zip(&self.i).map(|(&v, &i)| v / i).collect()
    }

    /// Reflection coefficient `S_11 = b/a` (for the excited port).
    pub fn s11(&self) -> Vec<Complex64> {
        s_param(self, self)
    }

    /// Incident power `½|a|²`.
    pub fn incident_power(&self) -> Vec<f64> {
        self.a().iter().map(|a| 0.5 * a.norm_sqr()).collect()
    }

    /// Reflected power `½|b|²`.
    pub fn reflected_power(&self) -> Vec<f64> {
        self.b().iter().map(|b| 0.5 * b.norm_sqr()).collect()
    }

    /// Accepted power `½·Re(V·I*)` (= incident − reflected).
    pub fn accepted_power(&self) -> Vec<f64> {
        self.v
            .iter()
            .zip(&self.i)
            .map(|(&v, &i)| 0.5 * (v.re * i.re + v.im * i.im))
            .collect()
    }
}

/// Scattering parameter `S_oi = b_o / a_i` from the result of the observed
/// port `out` and the excited port `inc` of the same run.
///
/// # Panics
/// If the two results have different frequencies.
pub fn s_param(out: &PortResult, inc: &PortResult) -> Vec<Complex64> {
    assert_eq!(
        out.freqs, inc.freqs,
        "port results at different frequencies"
    );
    out.b().iter().zip(inc.a()).map(|(&b, a)| b / a).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn load_reflection_and_power_balance() {
        // Source V_s = 2, R = 50 into Z_L = 100 + 30j: I = V_s/(R + Z_L).
        let zl = c(100.0, 30.0);
        let i = c(2.0, 0.0) / (zl + 50.0);
        let v = zl * i;
        let p = PortResult::new(vec![1e9], vec![v], vec![i], c(50.0, 0.0)).unwrap();
        let s = p.s11()[0];
        let want = (zl - 50.0) / (zl + 50.0);
        assert!((s - want).norm() < 1e-14);
        assert!((p.z_in()[0] - zl).norm() < 1e-12);
        // Incident wave of a Thevenin source with R = Z_ref is V_s/(2√R).
        assert!((p.a()[0] - c(1.0 / 50f64.sqrt(), 0.0)).norm() < 1e-14);
        let (pi, pr, pa) = (
            p.incident_power()[0],
            p.reflected_power()[0],
            p.accepted_power()[0],
        );
        assert!((pi - pr - pa).abs() < 1e-14 * pi);
        // Conjugate-matched complex reference: no reflection.
        let q = p.with_z_ref(c(100.0, -30.0)).unwrap();
        assert!(q.s11()[0].norm() < 1e-14);
        assert!((q.incident_power()[0] - q.accepted_power()[0]).abs() < 1e-14);
        // Power balance also holds for complex references.
        let q = p.with_z_ref(c(20.0, 70.0)).unwrap();
        let d = q.incident_power()[0] - q.reflected_power()[0] - q.accepted_power()[0];
        assert!(d.abs() < 1e-13 * q.incident_power()[0]);
    }

    #[test]
    fn frequency_dependent_reference() {
        // Each frequency uses its own reference: matched loads give no
        // reflection.
        let z = vec![c(50.0, 0.0), c(80.0, 0.0)];
        let i = vec![c(1.0, 0.0), c(0.5, 0.2)];
        let v: Vec<Complex64> = i.iter().zip(&z).map(|(&i, &z)| z * i).collect();
        let p = PortResult::new_per_frequency(vec![1e9, 2e9], v, i, z.clone()).unwrap();
        assert!(p.s11().iter().all(|s| s.norm() < 1e-15));
        assert_eq!(p.z_ref(), &z[..]);
        let bad = vec![c(50.0, 0.0), c(0.0, 1.0)];
        assert!(PortResult::new_per_frequency(vec![1e9, 2e9], z.clone(), z.clone(), bad).is_err());
        assert!(PortResult::new_per_frequency(vec![1e9], z.clone(), z.clone(), z).is_err());
    }

    #[test]
    fn waves_and_per_frequency_reference() {
        let zl = c(100.0, 30.0);
        let i = vec![c(0.01, 0.002), c(-0.003, 0.02)];
        let v: Vec<_> = i.iter().map(|&i| zl * i).collect();
        let z = vec![c(50.0, 0.0), c(60.0, -5.0)];
        let p =
            PortResult::new_per_frequency(vec![1e9, 2e9], v.clone(), i.clone(), z.clone()).unwrap();
        assert_eq!(p.z_ref(), &z[..]);
        let (vi, vr, ii, ir) = (p.v_inc(), p.v_ref(), p.i_inc(), p.i_ref());
        for k in 0..2 {
            assert!((vi[k] + vr[k] - v[k]).norm() < 1e-14);
            assert!((ii[k] + ir[k] - i[k]).norm() < 1e-16);
            assert!((vi[k] / ii[k] - z[k]).norm() < 1e-12);
            assert!((vr[k] / ir[k] + z[k]).norm() < 1e-12);
            let want = (zl - z[k]) / (zl + z[k]);
            if k == 0 {
                // Real reference: S11 is also the voltage reflection.
                assert!((vr[k] / vi[k] - want).norm() < 1e-14);
                assert!((p.s11()[k] - want).norm() < 1e-14);
            }
        }
        let one = p.with_z_ref(c(50.0, 0.0)).unwrap();
        assert_eq!(one.z_ref(), &[c(50.0, 0.0); 2]);
        assert!(p.with_z_ref_per_frequency(vec![c(50.0, 0.0)]).is_err());
        assert!(
            p.with_z_ref_per_frequency(vec![c(50.0, 0.0), c(-1.0, 0.0)])
                .is_err()
        );
    }

    #[test]
    fn validation() {
        let z = vec![Complex64::ZERO];
        assert!(PortResult::new(vec![1.0], z.clone(), z.clone(), c(0.0, 1.0)).is_err());
        assert!(PortResult::new(vec![1.0, 2.0], z.clone(), z.clone(), c(50.0, 0.0)).is_err());
        assert!(PortResult::new(vec![1.0], z.clone(), z, c(f64::NAN, 0.0)).is_err());
    }
}
