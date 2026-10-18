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

use super::complex::Complex64;
use crate::{Error, Result};

/// Frequency-domain voltage and current of one port, with derived
/// quantities; see the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
pub struct PortResult {
    freqs: Vec<f64>,
    v: Vec<Complex64>,
    i: Vec<Complex64>,
    z_ref: Complex64,
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
        if v.len() != freqs.len() || i.len() != freqs.len() {
            return Err(Error::Invalid(format!(
                "port result: {} frequencies but {} voltages and {} currents",
                freqs.len(),
                v.len(),
                i.len()
            )));
        }
        if !(z_ref.is_finite() && z_ref.re > 0.0) {
            return Err(Error::Invalid(format!(
                "reference impedance {z_ref} must have a positive real part"
            )));
        }
        Ok(Self { freqs, v, i, z_ref })
    }

    /// Returns the same spectra with another reference impedance.
    pub fn with_z_ref(&self, z_ref: Complex64) -> Result<Self> {
        Self::new(self.freqs.clone(), self.v.clone(), self.i.clone(), z_ref)
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

    /// Reference impedance \[Ω\].
    pub fn z_ref(&self) -> Complex64 {
        self.z_ref
    }

    fn norm(&self) -> f64 {
        0.5 / self.z_ref.re.sqrt()
    }

    /// Incident power wave `a = (V + Z_ref·I)/(2√Re Z_ref)`.
    pub fn a(&self) -> Vec<Complex64> {
        let k = self.norm();
        self.v
            .iter()
            .zip(&self.i)
            .map(|(&v, &i)| (v + self.z_ref * i) * k)
            .collect()
    }

    /// Reflected power wave `b = (V − Z_ref*·I)/(2√Re Z_ref)`.
    pub fn b(&self) -> Vec<Complex64> {
        let k = self.norm();
        let zc = Complex64::new(self.z_ref.re, -self.z_ref.im);
        self.v
            .iter()
            .zip(&self.i)
            .map(|(&v, &i)| (v - zc * i) * k)
            .collect()
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
    fn validation() {
        let z = vec![Complex64::ZERO];
        assert!(PortResult::new(vec![1.0], z.clone(), z.clone(), c(0.0, 1.0)).is_err());
        assert!(PortResult::new(vec![1.0, 2.0], z.clone(), z.clone(), c(50.0, 0.0)).is_err());
        assert!(PortResult::new(vec![1.0], z.clone(), z, c(f64::NAN, 0.0)).is_err());
    }
}
