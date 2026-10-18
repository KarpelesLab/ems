//! Excitation waveforms.
//!
//! A [`Waveform`] is the time function `s(t)` that scales every excitation of
//! a simulation (the openEMS `<Excitation>` element of the `<FDTD>` section).
//! Time is in seconds; `t = 0` is the start of the simulation.
//!
//! # Gaussian pulse
//!
//! [`Waveform::Gaussian`] is a Gaussian-modulated cosine
//!
//! ```text
//! s(t) = cos(2π f0 (t − T)) · exp(−((t − T)/τ)²)     for 0 ≤ t ≤ 2T, else 0
//! ```
//!
//! Using `∫ exp(−t²/τ²) e^{−j2πft} dt = τ√π · exp(−(πfτ)²)` and the
//! modulation and shift theorems (any signals text, e.g. Bracewell, *The
//! Fourier Transform and Its Applications*), its continuous Fourier transform
//! is
//!
//! ```text
//! S(f) = e^{−j2πfT} · (τ√π/2) · [exp(−(πτ(f − f0))²) + exp(−(πτ(f + f0))²)].
//! ```
//!
//! **Width.** `fc` is the 20 dB half-bandwidth: each spectral lobe falls to
//! 1/10 of its peak (−20 dB in amplitude) at `f0 ± fc`, i.e.
//! `exp(−(πτ·fc)²) = 1/10`, which gives
//!
//! ```text
//! τ = √(ln 10) / (π·fc).
//! ```
//!
//! The statement is exact for each lobe on its own; the mirror lobe at
//! `−f0` adds `exp(−(πτ(2f0 ± fc))²)` relative to the peak, which is below
//! 1e-4 as soon as `f0 ≥ fc` (and for `f0 = 0` both lobes coincide, so the
//! −20 dB point is exactly at `fc`).
//!
//! **Delay.** The pulse is centred at `T`, chosen so that the envelope at
//! `t = 0` is 1e-6 of its peak: `exp(−(T/τ)²) = 1e-6`, hence
//!
//! ```text
//! T = τ·√(6 ln 10) = √6 · ln 10 / (π·fc) ≈ 1.796 / fc.
//! ```
//!
//! The pulse is symmetric and considered over at `2T`
//! ([`Waveform::duration`]); outside `[0, 2T]` it is exactly zero. The
//! truncation changes the spectrum by less than 1e-6 of its peak.

use crate::error::{Error, Result};
use crate::post::Complex64;
use core::f64::consts::{LN_10, PI, TAU};
use core::fmt;
use std::sync::Arc;

/// Time function of a custom waveform: `t` \[s\] → amplitude.
pub type WaveformFn = Arc<dyn Fn(f64) -> f64 + Send + Sync>;

/// Excitation time signal.
///
/// Construct with [`Waveform::gaussian`], [`Waveform::sinusoid`],
/// [`Waveform::dirac`], [`Waveform::step`], [`Waveform::custom`] or
/// [`Waveform::expression`]. Constructors do not check their arguments;
/// [`Waveform::validate`] does (and is called by
/// [`Fdtd::validate`](crate::fdtd::Fdtd::validate)).
#[derive(Clone)]
#[non_exhaustive]
pub enum Waveform {
    /// Gaussian-modulated cosine centred at `f0` with 20 dB half-bandwidth
    /// `fc` (see the [module documentation](self) for the exact formula).
    Gaussian {
        /// Centre frequency \[Hz\], `≥ 0`.
        f0: f64,
        /// 20 dB half-bandwidth \[Hz\], `> 0`.
        fc: f64,
    },
    /// `sin(2π f0 t)` for `t ≥ 0`, zero before (so it starts smoothly at 0).
    Sinusoid {
        /// Frequency \[Hz\], `> 0`.
        f0: f64,
    },
    /// A single sample of amplitude 1 at the time step closest to `t = 0`;
    /// see [`Waveform::value`].
    Dirac,
    /// Unit step: 1 for `t ≥ 0`, 0 before.
    Step,
    /// Arbitrary user function of time `t` \[s\].
    Custom(WaveformFn),
}

impl fmt::Debug for Waveform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Waveform::Gaussian { f0, fc } => f
                .debug_struct("Gaussian")
                .field("f0", f0)
                .field("fc", fc)
                .finish(),
            Waveform::Sinusoid { f0 } => f.debug_struct("Sinusoid").field("f0", f0).finish(),
            Waveform::Dirac => f.write_str("Dirac"),
            Waveform::Step => f.write_str("Step"),
            Waveform::Custom(_) => f.write_str("Custom(<fn>)"),
        }
    }
}

/// Width `τ` of the Gaussian envelope for a 20 dB half-bandwidth `fc`.
fn gaussian_tau(fc: f64) -> f64 {
    LN_10.sqrt() / (PI * fc)
}

/// Delay `T` of the Gaussian centre for a 20 dB half-bandwidth `fc`.
fn gaussian_delay(fc: f64) -> f64 {
    gaussian_tau(fc) * (6.0 * LN_10).sqrt()
}

impl Waveform {
    /// Gaussian-modulated cosine centred at `f0` \[Hz\] with 20 dB
    /// half-bandwidth `fc` \[Hz\].
    pub fn gaussian(f0: f64, fc: f64) -> Self {
        Waveform::Gaussian { f0, fc }
    }

    /// Sinusoid `sin(2π f0 t)` starting at `t = 0`.
    pub fn sinusoid(f0: f64) -> Self {
        Waveform::Sinusoid { f0 }
    }

    /// Discrete unit impulse at `t = 0`.
    pub fn dirac() -> Self {
        Waveform::Dirac
    }

    /// Unit step at `t = 0`.
    pub fn step() -> Self {
        Waveform::Step
    }

    /// Arbitrary function of time `t` \[s\].
    pub fn custom(f: impl Fn(f64) -> f64 + Send + Sync + 'static) -> Self {
        Waveform::Custom(Arc::new(f))
    }

    /// Compiles a [`csx::expr`] expression in the variable `t` \[s\] (e.g.
    /// the openEMS custom excitation string `"sin(2*pi*1e9*t)"`).
    ///
    /// Names other than `t` must be the built-in constants `pi` / `e`; use
    /// [`Waveform::expression_with`] to supply parameter values.
    ///
    /// ```
    /// # use ems::waveform::Waveform;
    /// let w = Waveform::expression("2*t + 1")?;
    /// assert_eq!(w.value(0.5, 1e-3), 2.0);
    /// # Ok::<(), ems::Error>(())
    /// ```
    pub fn expression(src: &str) -> Result<Self> {
        Waveform::expression_with(src, |_| None)
    }

    /// Like [`Waveform::expression`], resolving every name other than `t`
    /// once through `lookup` (falling back to `pi` / `e`). Unresolved names
    /// are an error.
    pub fn expression_with(src: &str, lookup: impl FnMut(&str) -> Option<f64>) -> Result<Self> {
        let compiled = csx::expr::Expr::parse(src)?.compile(&["t"], lookup)?;
        Ok(Waveform::Custom(Arc::new(move |t| compiled.eval(&[t]))))
    }

    /// Checks the parameters: Gaussian needs finite `f0 ≥ 0` and `fc > 0`,
    /// Sinusoid a finite `f0 > 0`.
    pub fn validate(&self) -> Result<()> {
        match *self {
            Waveform::Gaussian { f0, fc } => {
                if !(f0.is_finite() && f0 >= 0.0) {
                    return Err(Error::Invalid(format!(
                        "Gaussian waveform: centre frequency f0 = {f0} must be finite and >= 0"
                    )));
                }
                if !(fc.is_finite() && fc > 0.0) {
                    return Err(Error::Invalid(format!(
                        "Gaussian waveform: bandwidth fc = {fc} must be finite and > 0"
                    )));
                }
                Ok(())
            }
            Waveform::Sinusoid { f0 } => {
                if f0.is_finite() && f0 > 0.0 {
                    Ok(())
                } else {
                    Err(Error::Invalid(format!(
                        "sinusoid waveform: frequency f0 = {f0} must be finite and > 0"
                    )))
                }
            }
            Waveform::Dirac | Waveform::Step | Waveform::Custom(_) => Ok(()),
        }
    }

    /// Amplitude at time `t` \[s\]; `dt` is the sampling interval (time step).
    ///
    /// `dt` only matters for [`Waveform::Dirac`], which is 1 in the window
    /// `−dt/2 < t ≤ dt/2` (shifted by `1e-6·dt` to absorb rounding) and 0
    /// elsewhere. For any uniformly spaced sampling `t0 + n·dt` exactly one
    /// sample falls into that window: `t = 0` for integer-step sampling and
    /// `t = +dt/2` for half-step sampling.
    pub fn value(&self, t: f64, dt: f64) -> f64 {
        match self {
            Waveform::Gaussian { f0, fc } => {
                let delay = gaussian_delay(*fc);
                if !(0.0..=2.0 * delay).contains(&t) {
                    return 0.0;
                }
                let x = (t - delay) / gaussian_tau(*fc);
                (TAU * f0 * (t - delay)).cos() * (-x * x).exp()
            }
            Waveform::Sinusoid { f0 } => {
                if t >= 0.0 {
                    (TAU * f0 * t).sin()
                } else {
                    0.0
                }
            }
            Waveform::Dirac => {
                let half = 0.5 * dt;
                let s = t - 1e-6 * dt;
                if s > -half && s <= half { 1.0 } else { 0.0 }
            }
            Waveform::Step => {
                if t >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Waveform::Custom(f) => f(t),
        }
    }

    /// Built-in delay \[s\]: the time of the Gaussian pulse's peak
    /// (`T = √6·ln 10/(π·fc)`), 0 for all other waveforms.
    pub fn delay(&self) -> f64 {
        match *self {
            Waveform::Gaussian { fc, .. } => gaussian_delay(fc),
            _ => 0.0,
        }
    }

    /// Time \[s\] after which the excitation is over (identically zero from
    /// then on): `2T` for the Gaussian, 0 for Dirac, `None` for waveforms
    /// that do not terminate (sinusoid, step, custom).
    pub fn duration(&self) -> Option<f64> {
        match *self {
            Waveform::Gaussian { fc, .. } => Some(2.0 * gaussian_delay(fc)),
            Waveform::Dirac => Some(0.0),
            _ => None,
        }
    }

    /// Analytic continuous Fourier transform `S(f) = ∫ s(t)·e^{−j2πft} dt`
    /// (same convention as [`mod@crate::post::dft`]), where one exists in closed
    /// form: the Gaussian, including its delay `T` (the `[0, 2T]` truncation
    /// is ignored). `None` for all other waveforms.
    pub fn spectrum(&self, f: f64) -> Option<Complex64> {
        match *self {
            Waveform::Gaussian { f0, fc } => {
                let tau = gaussian_tau(fc);
                let lobe = |df: f64| {
                    let x = PI * tau * df;
                    (-x * x).exp()
                };
                let mag = 0.5 * tau * PI.sqrt() * (lobe(f - f0) + lobe(f + f0));
                Some(Complex64::from_polar(mag, -TAU * f * gaussian_delay(fc)))
            }
            _ => None,
        }
    }

    /// Highest frequency of significant spectral content \[Hz\], used for
    /// mesh resolution checks: `f0 + fc` for the Gaussian, `f0` for the
    /// sinusoid, `None` when unknown or unbounded.
    pub fn max_frequency(&self) -> Option<f64> {
        match *self {
            Waveform::Gaussian { f0, fc } => Some(f0 + fc),
            Waveform::Sinusoid { f0 } => Some(f0),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::post::dft;

    #[test]
    fn gaussian_starts_and_ends_below_1e_6() {
        let w = Waveform::gaussian(2e9, 1e9);
        let dt = 1e-13;
        let t_end = w.duration().unwrap();
        assert!((t_end - 2.0 * w.delay()).abs() < 1e-24);
        // Envelope at both ends is 1e-6 of the peak.
        assert!(w.value(0.0, dt).abs() <= 1e-6 * (1.0 + 1e-9));
        assert!(w.value(t_end, dt).abs() <= 1e-6 * (1.0 + 1e-9));
        let x = w.delay() / gaussian_tau(1e9);
        assert!(((-x * x).exp() / 1e-6 - 1.0).abs() < 1e-9);
        // Peak 1 at the delay, zero outside [0, 2T].
        assert!((w.value(w.delay(), dt) - 1.0).abs() < 1e-15);
        assert_eq!(w.value(-1e-12, dt), 0.0);
        assert_eq!(w.value(t_end + 1e-12, dt), 0.0);
        // Closed form of the delay.
        assert!((w.delay() - 6f64.sqrt() * LN_10 / (PI * 1e9)).abs() < 1e-22);
    }

    #[test]
    fn gaussian_20db_points() {
        for (f0, fc) in [(3e9, 1e9), (10e9, 2e9), (0.0, 1e9)] {
            let w = Waveform::gaussian(f0, fc);
            let peak = w.spectrum(f0).unwrap().norm();
            for f in [f0 - fc, f0 + fc] {
                let r = w.spectrum(f).unwrap().norm() / peak;
                assert!((r - 0.1).abs() < 1e-6, "f0={f0} f={f}: {r}");
            }
        }
    }

    #[test]
    fn gaussian_sampled_dft_matches_spectrum() {
        let w = Waveform::gaussian(1e9, 0.5e9);
        let fmax = w.max_frequency().unwrap();
        assert_eq!(fmax, 1.5e9);
        let dt = 1.0 / (20.0 * fmax);
        let freqs: Vec<f64> = (0..=40).map(|i| i as f64 * 0.075e9).collect();
        let peak = w.spectrum(1e9).unwrap().norm();
        // E-like (integer step) and H-like (half step) sampling.
        for t0 in [0.0, 0.5 * dt] {
            let n = ((w.duration().unwrap() - t0) / dt).ceil() as usize + 1;
            let samples: Vec<f64> = (0..n).map(|i| w.value(t0 + i as f64 * dt, dt)).collect();
            let got = dft(t0, dt, &samples, &freqs);
            for (g, &f) in got.iter().zip(&freqs) {
                let s = w.spectrum(f).unwrap();
                let err = (*g - s).norm() / peak;
                assert!(err < 2e-6, "t0={t0} f={f}: {g} vs {s} (err {err})");
            }
        }
    }

    #[test]
    fn spectrum_phase_reflects_delay() {
        let w = Waveform::gaussian(1e9, 1e9);
        let f = 0.7e9;
        let s = w.spectrum(f).unwrap();
        let expect = (-TAU * f * w.delay()).rem_euclid(TAU);
        let got = s.arg().rem_euclid(TAU);
        assert!((got - expect).abs() < 1e-9);
    }

    #[test]
    fn dirac_hits_exactly_one_sample() {
        let w = Waveform::dirac();
        let dt = 1.3e-12;
        for t0 in [0.0, 0.5 * dt, -0.5 * dt, 0.3 * dt, -3.0 * dt] {
            let hits: Vec<f64> = (0..20)
                .map(|n| t0 + n as f64 * dt)
                .filter(|&t| w.value(t, dt) == 1.0)
                .collect();
            assert_eq!(hits.len(), 1, "t0 = {t0}");
            assert!(hits[0].abs() <= 0.5 * dt * (1.0 + 1e-6));
        }
        assert_eq!(w.value(0.0, dt), 1.0);
        assert_eq!(w.value(0.5 * dt, dt), 1.0);
        assert_eq!(w.value(-0.5 * dt, dt), 0.0);
        assert_eq!(w.duration(), Some(0.0));
        assert_eq!(w.spectrum(1.0), None);
    }

    #[test]
    fn step_and_sinusoid() {
        let s = Waveform::step();
        assert_eq!(s.value(-1e-15, 1.0), 0.0);
        assert_eq!(s.value(0.0, 1.0), 1.0);
        assert_eq!(s.value(5.0, 1.0), 1.0);
        assert_eq!(s.duration(), None);
        assert_eq!(s.max_frequency(), None);

        let w = Waveform::sinusoid(1e9);
        assert_eq!(w.value(-1e-10, 1e-12), 0.0);
        assert_eq!(w.value(0.0, 1e-12), 0.0);
        assert!((w.value(0.25e-9, 1e-12) - 1.0).abs() < 1e-12);
        assert_eq!(w.max_frequency(), Some(1e9));
        assert_eq!(w.duration(), None);
        assert_eq!(w.delay(), 0.0);
        assert_eq!(w.spectrum(1e9), None);
    }

    #[test]
    fn custom_and_expression() {
        let w = Waveform::custom(|t| 3.0 * t);
        assert_eq!(w.value(2.0, 0.1), 6.0);
        assert_eq!(format!("{w:?}"), "Custom(<fn>)");
        let c = w.clone();
        assert_eq!(c.value(1.0, 0.1), 3.0);

        let e = Waveform::expression("sin(2*pi*1e9*t)").unwrap();
        assert!((e.value(0.25e-9, 1e-12) - 1.0).abs() < 1e-12);
        assert!(Waveform::expression("t +").is_err());
        assert!(matches!(Waveform::expression("a*t"), Err(Error::Csx(_))));
        let p = Waveform::expression_with("a*t", |n| (n == "a").then_some(4.0)).unwrap();
        assert_eq!(p.value(0.5, 1.0), 2.0);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn validation() {
        assert!(Waveform::gaussian(1e9, 1e9).validate().is_ok());
        assert!(Waveform::gaussian(0.0, 1e9).validate().is_ok());
        assert!(Waveform::gaussian(1e9, 0.0).validate().is_err());
        assert!(Waveform::gaussian(1e9, -1.0).validate().is_err());
        assert!(Waveform::gaussian(-1.0, 1e9).validate().is_err());
        assert!(Waveform::gaussian(f64::NAN, 1e9).validate().is_err());
        assert!(Waveform::gaussian(1e9, f64::INFINITY).validate().is_err());
        assert!(Waveform::sinusoid(1e9).validate().is_ok());
        assert!(Waveform::sinusoid(0.0).validate().is_err());
        assert!(Waveform::sinusoid(f64::NAN).validate().is_err());
        assert!(Waveform::dirac().validate().is_ok());
        assert!(Waveform::step().validate().is_ok());
        assert_eq!(
            format!("{:?}", Waveform::gaussian(1.0, 2.0)),
            "Gaussian { f0: 1.0, fc: 2.0 }"
        );
    }
}
