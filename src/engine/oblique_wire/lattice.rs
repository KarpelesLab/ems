//! Green's function of the discrete Laplacian on a uniform rectangular
//! lattice, used for the in-cell corrections of the oblique thin-wire
//! model (see the [parent module](super)).
//!
//! On a lattice with spacings `h = (hx, hy, hz)` and the 7-point
//! Laplacian `Δ_h`, `G` solves `−Δ_h G = δ_{n,0}/V` (`V = hx·hy·hz`), so
//! that `G(n) → 1/(4π·|n·h|)` far from the origin. Its Fourier
//! representation, with `1/a = ∫₀^∞ e^(−a·t) dt` and the Bessel-function
//! identity `(1/2π)∫ e^(x·cos θ)·cos(nθ) dθ = I_n(x)`, is the classic
//! one-dimensional integral (Watson; e.g. Economou, *Green's Functions in
//! Quantum Physics*, ch. 5, for the cubic case)
//!
//! ```text
//! G(n) = (1/V) ∫₀^∞ Π_d e^(−2t/h_d²) · I_{n_d}(2t/h_d²) dt.
//! ```
//!
//! The scaled Bessel functions `e^(−x)·I_n(x)` are evaluated with the
//! (spectrally accurate) trapezoidal rule of their periodic integral for
//! `x ≤ 400` and with their asymptotic series above; the `t` integral runs
//! over `ln t` with Simpson's rule up to `t = 10⁸·h²` plus the analytic
//! tail `2·(4π)^(−3/2)·t^(−1/2)`. For the cubic lattice this gives
//! Watson's `G(0) = 0.2527310098586…/h`.

use std::f64::consts::PI;

/// `e^(−x)·I_n(x)` for `n = 0..=nmax` (`x ≥ 0`).
pub(super) fn scaled_bessel_i(x: f64, nmax: usize) -> Vec<f64> {
    let mut out = vec![0.0; nmax + 1];
    if x <= 0.0 {
        out[0] = 1.0;
        return out;
    }
    if x <= 400.0 {
        // (1/M)·Σ_k e^(x(cos θ_k − 1))·cos(n θ_k), θ_k = 2πk/M: exact up
        // to the aliases e^(−x)·I_{M−n}(x), negligible for this M.
        let m = 2 * ((6.0 * x.sqrt()).ceil() as usize + nmax + 10);
        let half = m / 2;
        for k in 0..=half {
            let th = 2.0 * PI * k as f64 / m as f64;
            let c = th.cos();
            let w = if k == 0 || k == half { 1.0 } else { 2.0 };
            let f = w * (x * (c - 1.0)).exp();
            // cos(nθ) by the Chebyshev recurrence.
            let (mut c0, mut c1) = (1.0, c);
            for (n, o) in out.iter_mut().enumerate() {
                let cn = if n == 0 { 1.0 } else { c1 };
                *o += f * cn;
                if n >= 1 {
                    let c2 = 2.0 * c * c1 - c0;
                    c0 = c1;
                    c1 = c2;
                }
            }
        }
        for o in &mut out {
            *o /= m as f64;
        }
        return out;
    }
    // Asymptotic series e^(−x) I_n(x) ~ (2πx)^(−½)·Σ_k (−1)^k a_k(n)/x^k,
    // a_k = Π_{j≤k} (4n² − (2j − 1)²) / (k!·8^k).
    let pre = 1.0 / (2.0 * PI * x).sqrt();
    for (n, o) in out.iter_mut().enumerate() {
        let mu = 4.0 * (n * n) as f64;
        let mut term = 1.0;
        let mut sum = 1.0;
        for k in 1..40 {
            let j = (2 * k - 1) as f64;
            term *= -(mu - j * j) / (k as f64 * 8.0 * x);
            sum += term;
            if term.abs() < 1e-17 {
                break;
            }
        }
        *o = pre * sum;
    }
    out
}

/// Tabulated lattice Green's function for spacings `h` \[m\] (see the
/// [module docs](self)): exact values for offsets up to `n` cells per
/// axis, the continuum `1/(4π·r)` beyond.
#[derive(Debug, Clone)]
pub(super) struct LatticeGreen {
    h: [f64; 3],
    n: usize,
    table: Vec<f64>,
}

impl LatticeGreen {
    /// Tabulates `G` for `|offset_d| ≤ n`.
    pub(super) fn new(h: [f64; 3], n: usize) -> Self {
        let href = h.iter().copied().fold(f64::INFINITY, f64::min);
        let a = h.map(|v| v / href);
        // Simpson over u = ln t in [ln 1e-9, ln 1e8] (t in units of href²).
        let (u0, u1) = ((1e-9f64).ln(), (1e8f64).ln());
        let steps = 1400usize; // even
        let du = (u1 - u0) / steps as f64;
        let np = n + 1;
        // E[d][k][m]: scaled Bessel of axis d at node k, order m.
        let mut e: [Vec<Vec<f64>>; 3] = Default::default();
        let mut w = Vec::with_capacity(steps + 1);
        for k in 0..=steps {
            let t = (u0 + k as f64 * du).exp();
            let s = if k == 0 || k == steps {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            w.push(s * du / 3.0 * t);
            for d in 0..3 {
                e[d].push(scaled_bessel_i(2.0 * t / (a[d] * a[d]), n));
            }
        }
        let tail = 2.0 / (4.0 * PI).powf(1.5) / (1e8f64).sqrt();
        let vol = a[0] * a[1] * a[2];
        let mut table = vec![0.0; np * np * np];
        for i in 0..np {
            for j in 0..np {
                for l in 0..np {
                    let mut sum = 0.0;
                    for (k, wk) in w.iter().enumerate() {
                        sum += wk * e[0][k][i] * e[1][k][j] * e[2][k][l];
                    }
                    // ∫₀^{t_min} F dt = t_min for the origin (F → 1).
                    if i + j + l == 0 {
                        sum += u0.exp();
                    }
                    table[(i * np + j) * np + l] = (sum / vol + tail) / href;
                }
            }
        }
        Self { h, n, table }
    }

    /// `G` at the integer offset `d` \[1/m\].
    pub(super) fn get(&self, d: [i64; 3]) -> f64 {
        let a = d.map(|v| v.unsigned_abs() as usize);
        if a.iter().all(|&v| v <= self.n) {
            let np = self.n + 1;
            self.table[(a[0] * np + a[1]) * np + a[2]]
        } else {
            let r2: f64 = (0..3).map(|k| (d[k] as f64 * self.h[k]).powi(2)).sum();
            1.0 / (4.0 * PI * r2.sqrt())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_sum_rule_and_values() {
        for x in [0.1, 1.0, 10.0, 399.0] {
            // Σ_{n∈Z} I_n(x) = e^x.
            let e = scaled_bessel_i(x, 300);
            let s: f64 = e[0] + 2.0 * e[1..].iter().sum::<f64>();
            assert!((s - 1.0).abs() < 1e-11, "x = {x}: {s}");
        }
        // e^(−1)·I_0(1) = 0.4657596075936404.
        let e = scaled_bessel_i(1.0, 2);
        assert!((e[0] - 0.465_759_607_593_640_4).abs() < 1e-14);
        // Continuity between the two branches.
        let (a, b) = (
            scaled_bessel_i(400.0, 12),
            scaled_bessel_i(400.000_000_001, 12),
        );
        for n in 0..=12 {
            assert!((a[n] / b[n] - 1.0).abs() < 1e-9, "{n}: {} {}", a[n], b[n]);
        }
    }

    #[test]
    fn watson_integral() {
        let g = LatticeGreen::new([1.0; 3], 4);
        // Watson's integral for the simple cubic lattice.
        assert!((g.get([0, 0, 0]) - 0.252_731_009_858_663).abs() < 1e-11);
        // −Δ G = δ: 6 G(0) − 6 G(1) = 1.
        assert!((6.0 * (g.get([0, 0, 0]) - g.get([1, 0, 0])) - 1.0).abs() < 1e-9);
        // −Δ G = 0 at (1,0,0).
        let lap =
            6.0 * g.get([1, 0, 0]) - g.get([0, 0, 0]) - g.get([2, 0, 0]) - 4.0 * g.get([1, 1, 0]);
        assert!(lap.abs() < 1e-9, "{lap}");
        // Far field.
        let r = g.get([4, 3, 0]);
        assert!((r * 4.0 * PI * 5.0 - 1.0).abs() < 5e-3, "{r}");
        // Scaling with the cell size, and rectangular cells.
        let g2 = LatticeGreen::new([2.0; 3], 2);
        assert!((g2.get([1, 1, 0]) * 2.0 - g.get([1, 1, 0])).abs() < 1e-9);
        let gr = LatticeGreen::new([1.0, 1.5, 2.0], 3);
        let h2 = [1.0, 2.25, 4.0];
        let lap: f64 = (0..3)
            .map(|d| {
                let mut p = [0i64; 3];
                p[d] = 1;
                2.0 * (gr.get([0, 0, 0]) - gr.get(p)) / h2[d]
            })
            .sum();
        assert!((lap * 3.0 - 1.0).abs() < 1e-12, "{lap}");
    }
}
