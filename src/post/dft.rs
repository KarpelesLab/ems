//! Discrete Fourier transforms of uniformly sampled real time series at an
//! arbitrary set of frequencies.
//!
//! # Convention
//!
//! A series `x_n = x(t_n)` sampled at `t_n = t0 + n·dt` (`n = 0, 1, …`) is
//! transformed as
//!
//! ```text
//! X(f) = Σ_n x(t_n) · e^{-j2πf·t_n} · dt
//! ```
//!
//! which is the rectangle-rule approximation of the continuous Fourier
//! transform `∫ x(t)·e^{-j2πft} dt`. For a signal that is smooth and
//! (effectively) zero outside the sampled window, and for frequencies well
//! below the Nyquist limit `1/(2dt)`, this converges spectrally fast to the
//! continuous transform (it coincides with the trapezoidal rule when the
//! signal vanishes at both ends; see e.g. Trefethen & Weideman, "The
//! exponentially convergent trapezoidal rule", SIAM Review 56(3), 2014).
//!
//! The absolute time `t_n` (not just the index `n`) enters the kernel, so
//! quantities sampled at different instants are directly comparable. In
//! a Yee scheme electric quantities are sampled at integer steps (`t0 = 0`)
//! and magnetic ones at half steps (`t0 = dt/2`); with the absolute-time
//! kernel both transforms refer to the same time origin and no extra phase
//! correction is needed.
//!
//! # Accuracy
//!
//! [`RunningDft`] advances each frequency's phasor `e^{-j2πf·t_n}` by one
//! complex multiplication per sample: the on-the-fly ("running") DFT commonly
//! used for FDTD frequency-domain outputs (Taflove & Hagness, *Computational
//! Electrodynamics*; C. M. Furse and O. P. Gandhi, "Why the DFT is faster
//! than the FFT for FDTD time-to-frequency domain conversions", IEEE
//! Microwave and Guided Wave Letters 5(10), 1995).
//! A pure recurrence accumulates rounding error linearly in the number of
//! samples, so every [`ANCHOR_INTERVAL`] samples the phasor is recomputed
//! exactly from the absolute phase. That phase, `f·t0 + (f·dt)·n` in cycles,
//! is reduced modulo 1 with an error-free product (`mul_add` recovers the
//! rounding error of `(f·dt)·n`), so it stays accurate to ~1e-16 cycles for
//! any practical `n`. The kernel error therefore does not grow with the
//! series length: after 10⁶+ samples each phasor is still accurate to about
//! 1e-13. The accumulation itself is a plain `f64` sum, whose rounding error
//! (≲ 1e-11 relative for 10⁶ samples) is far below the `f32` field noise.

use super::complex::Complex64;
use core::f64::consts::TAU;

/// Number of samples between exact re-computations of the phasors in
/// [`RunningDft`]. Between two anchors the recurrence error is bounded by
/// about `ANCHOR_INTERVAL · ε ≈ 2e-13`.
pub const ANCHOR_INTERVAL: u64 = 1024;

/// Accumulates the DFT of a real, uniformly sampled time series at a set of
/// frequencies, one sample at a time.
///
/// Samples are taken at `t_n = t0 + n·dt`, where `n` counts the calls to
/// [`push`](RunningDft::push). See the [module documentation](self) for the
/// transform convention and the accuracy strategy.
///
/// ```
/// use ems::post::RunningDft;
/// let dt = 1e-12;
/// let mut dft = RunningDft::new(vec![1e9, 2e9], 0.0, dt);
/// for n in 0..1000 {
///     let t = dft.next_time();
///     dft.push((core::f64::consts::TAU * 1e9 * t).cos());
///     assert_eq!(dft.len(), n + 1);
/// }
/// // One full nanosecond of a 1 GHz cosine: X(1 GHz) = T/2.
/// assert!((dft.result()[0].re - 0.5e-9).abs() < 1e-18);
/// ```
#[derive(Debug, Clone)]
pub struct RunningDft {
    freqs: Vec<f64>,
    t0: f64,
    dt: f64,
    n: u64,
    acc: Vec<Complex64>,
    phasor: Vec<Complex64>,
    step: Vec<Complex64>,
}

impl RunningDft {
    /// Creates an empty accumulator for the frequencies `freqs` \[Hz\] and a
    /// series sampled at `t_n = t0 + n·dt` \[s\].
    ///
    /// `dt` should be positive and all values finite; otherwise the results
    /// are meaningless (NaN).
    pub fn new(freqs: impl Into<Vec<f64>>, t0: f64, dt: f64) -> Self {
        let freqs = freqs.into();
        let step = freqs
            .iter()
            .map(|&f| Complex64::cis(-TAU * f * dt))
            .collect();
        let mut s = RunningDft {
            acc: vec![Complex64::ZERO; freqs.len()],
            phasor: vec![Complex64::ZERO; freqs.len()],
            step,
            freqs,
            t0,
            dt,
            n: 0,
        };
        s.anchor();
        s
    }

    /// Recomputes `phasor[k] = e^{-j2π f_k t_n}` exactly for the current `n`.
    fn anchor(&mut self) {
        let n = self.n as f64;
        for (p, &f) in self.phasor.iter_mut().zip(&self.freqs) {
            *p = Complex64::cis(-TAU * phase_cycles(f, self.t0, self.dt, n));
        }
    }

    /// Adds the next sample `x(t_n)`, with `t_n` = [`next_time`](Self::next_time).
    pub fn push(&mut self, x: f64) {
        for ((a, p), s) in self.acc.iter_mut().zip(&mut self.phasor).zip(&self.step) {
            *a += *p * x;
            *p *= *s;
        }
        self.n += 1;
        if self.n.is_multiple_of(ANCHOR_INTERVAL) {
            self.anchor();
        }
    }

    /// Adds several consecutive samples.
    pub fn extend(&mut self, xs: impl IntoIterator<Item = f64>) {
        for x in xs {
            self.push(x);
        }
    }

    /// Time of the sample the next [`push`](Self::push) represents.
    pub fn next_time(&self) -> f64 {
        self.time_of(self.n)
    }

    /// Time `t0 + n·dt` of sample `n`.
    pub fn time_of(&self, n: u64) -> f64 {
        self.t0 + n as f64 * self.dt
    }

    /// Number of samples accumulated so far.
    pub fn len(&self) -> u64 {
        self.n
    }

    /// `true` if no sample has been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The analysis frequencies \[Hz\].
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// Time of the first sample \[s\].
    pub fn t0(&self) -> f64 {
        self.t0
    }

    /// Sampling interval \[s\].
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Current transform at frequency index `k`: `Σ x(t_n)·e^{-j2πf_k t_n}·dt`.
    ///
    /// # Panics
    /// If `k` is out of range.
    pub fn get(&self, k: usize) -> Complex64 {
        self.acc[k] * self.dt
    }

    /// Current transform at all frequencies (same order as
    /// [`freqs`](Self::freqs)).
    pub fn result(&self) -> Vec<Complex64> {
        self.acc.iter().map(|&a| a * self.dt).collect()
    }

    /// Discards all samples, keeping the frequencies and time axis.
    pub fn reset(&mut self) {
        self.n = 0;
        self.acc.fill(Complex64::ZERO);
        self.anchor();
    }
}

/// `f·(t0 + n·dt)` reduced modulo one cycle, computed so that the result is
/// accurate to a few ulps of one cycle regardless of the magnitude of `n`.
pub(crate) fn phase_cycles(f: f64, t0: f64, dt: f64, n: f64) -> f64 {
    let a = f * dt;
    let p = a * n;
    // Exact rounding error of the product (two-product via FMA).
    let e = a.mul_add(n, -p);
    let c = f * t0;
    (p.fract() + e + c.fract()).fract()
}

/// DFT of a recorded series `samples[n] = x(t0 + n·dt)` at `freqs` \[Hz\],
/// with the convention `X(f) = Σ x(t_n)·e^{-j2πf·t_n}·dt` (see the
/// [module documentation](self)).
///
/// This is the batch counterpart of [`RunningDft`] and gives identical
/// results.
///
/// ```
/// use ems::post::dft;
/// // A unit sample at t = 0 transforms to dt at every frequency.
/// let x = dft(0.0, 0.5, &[1.0, 0.0, 0.0], &[0.1, 0.3]);
/// assert!((x[1].re - 0.5).abs() < 1e-15 && x[1].im.abs() < 1e-15);
/// ```
pub fn dft(t0: f64, dt: f64, samples: &[f64], freqs: &[f64]) -> Vec<Complex64> {
    let mut d = RunningDft::new(freqs, t0, dt);
    d.extend(samples.iter().copied());
    d.result()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: direct summation with an independently computed kernel.
    fn naive(t0: f64, dt: f64, xs: &[f64], f: f64) -> Complex64 {
        xs.iter()
            .enumerate()
            .map(|(n, &x)| Complex64::cis(-TAU * f * (t0 + n as f64 * dt)) * x)
            .sum::<Complex64>()
            * dt
    }

    #[test]
    fn matches_naive_sum() {
        let dt = 1.7e-12;
        let t0 = 0.5 * dt;
        let xs: Vec<f64> = (0..5000)
            .map(|n| ((n * 7919) % 1000) as f64 / 1000.0 - 0.5)
            .collect();
        let freqs = [0.0, 1.3e9, 7.77e9, 41e9];
        let got = dft(t0, dt, &xs, &freqs);
        for (g, &f) in got.iter().zip(&freqs) {
            let r = naive(t0, dt, &xs, f);
            assert!(
                (*g - r).norm() < 1e-12 * xs.len() as f64 * dt,
                "{f}: {g} vs {r}"
            );
        }
    }

    #[test]
    fn dc_and_impulse() {
        let d = dft(0.0, 2.0, &[1.0, 1.0, 1.0], &[0.0]);
        assert_eq!(d[0], Complex64::new(6.0, 0.0));
        // Impulse at t = t0: X(f) = dt·e^{-j2πf t0}.
        let t0 = 0.125;
        let d = dft(t0, 1.0, &[1.0], &[1.0]);
        let expect = Complex64::cis(-TAU * t0);
        assert!((d[0] - expect).norm() < 1e-15);
    }

    #[test]
    fn running_matches_batch_and_reset() {
        let xs: Vec<f64> = (0..3000).map(|n| (n as f64 * 0.01).sin()).collect();
        let freqs = vec![1.0, 2.5, 10.0];
        let mut r = RunningDft::new(freqs.clone(), 0.0, 0.01);
        assert!(r.is_empty());
        r.extend(xs.iter().copied());
        assert_eq!(r.len(), 3000);
        assert_eq!(r.result(), dft(0.0, 0.01, &xs, &freqs));
        assert_eq!(r.get(1), r.result()[1]);
        assert_eq!(r.freqs(), &freqs[..]);
        r.reset();
        assert!(r.is_empty());
        assert_eq!(r.result(), vec![Complex64::ZERO; 3]);
        r.extend(xs.iter().copied());
        assert_eq!(r.result(), dft(0.0, 0.01, &xs, &freqs));
    }

    #[test]
    fn time_axis() {
        let r = RunningDft::new(vec![1.0], 0.5, 2.0);
        assert_eq!(r.next_time(), 0.5);
        assert_eq!(r.time_of(3), 6.5);
        assert_eq!((r.t0(), r.dt()), (0.5, 2.0));
    }

    /// A tone over an exact number of periods has a closed-form transform;
    /// check there is no drift after millions of samples, for both integer
    /// (E-like) and half-step (H-like) sampling instants.
    #[test]
    fn no_drift_over_many_samples() {
        let dt = 1e-12;
        let f = 0.125 / dt; // 8 samples per period
        // Multiple of the period, but not of the anchor interval, so the
        // final phasor comes from ~1000 recurrence steps after an anchor.
        let n_samples: u64 = 2_001_000;
        let phi = 0.3;
        for t0 in [0.0, 0.5 * dt] {
            let c0 = f * t0; // exact here
            let mut r = RunningDft::new(vec![f], t0, dt);
            for n in 0..n_samples {
                // Exactly reduced phase, so the test signal itself is exact.
                r.push((TAU * (c0 + (n % 8) as f64 / 8.0) + phi).cos());
            }
            // The running phasor must not drift.
            let exact = Complex64::cis(-TAU * (c0 + (n_samples % 8) as f64 / 8.0));
            let perr = (r.phasor[0] - exact).norm();
            assert!(perr < 1e-13, "t0={t0}: phasor error {perr}");
            // cos(ωt+φ) = ½(e^{j(ωt+φ)} + e^{-j(ωt+φ)}): the first term sums
            // to N·dt/2·e^{jφ}; the second sums to zero over full periods.
            // The tolerance covers the f64 accumulation of 2e6 terms.
            let expect = Complex64::cis(phi) * (0.5 * n_samples as f64 * dt);
            let err = (r.get(0) - expect).norm() / expect.norm();
            assert!(err < 1e-11, "t0={t0}: relative error {err}");
        }
    }

    #[test]
    fn phase_reduction_is_accurate_for_large_n() {
        // a = f·dt = 0.1 (its binary value m/2^k); compare the reduced phase
        // of a·n with an exact integer computation of (m·n mod 2^k)/2^k.
        let a = 0.1_f64;
        let n: u64 = 123_456_789_013;
        let bits = a.to_bits();
        let m = (bits & ((1 << 52) - 1)) | (1 << 52);
        let k = 1075 - ((bits >> 52) & 0x7ff); // a = m / 2^k
        let prod = m as u128 * n as u128;
        let exact = (prod % (1u128 << k)) as f64 / (1u128 << k) as f64;
        let c = phase_cycles(1.0, 0.0, a, n as f64);
        let d = (c - exact).abs();
        let d = d.min(1.0 - d);
        assert!(d < 1e-14, "{c} vs {exact}");
        // The naive product is off by orders of magnitude more.
        let naive = (a * n as f64).fract();
        assert!((naive - exact).abs() > 1e-9);
    }
}
