//! Bessel functions of the first kind `J_n` (integer order) and the zeros
//! of `J_n` and `J_n'`, for the circular waveguide modes
//! ([`CircWaveguidePort`](super::CircWaveguidePort)); of the second kind
//! `Y_n` and the Hankel functions, for the cylindrical waves of the
//! radial line ([`RadialLinePort`](super::RadialLinePort), see [`yn`]).
//!
//! `J_n` is evaluated from Bessel's integral (Abramowitz & Stegun 9.1.21;
//! Watson, *A Treatise on the Theory of Bessel Functions*, §2.2)
//!
//! ```text
//! J_n(x) = (1/2π) ∫₀^{2π} cos(nτ − x·sin τ) dτ
//! ```
//!
//! with the trapezoidal rule. The integrand is smooth and `2π`-periodic,
//! so the rule converges exponentially (its error is the aliasing of the
//! Fourier coefficients `J_{n ± kN}(x)` for `N` points, negligible once
//! `N > |x| + n + 40`). The zeros are bracketed on a fine scan and refined
//! by bisection to the last bit.

use core::f64::consts::TAU;

/// `J_n(x)` for integer order `n` (negative orders by
/// `J_{−n} = (−1)^n J_n`).
pub(crate) fn jn(n: i32, x: f64) -> f64 {
    let pts = 2 * ((x.abs() + f64::from(n.unsigned_abs())) as usize / 2) + 64;
    let h = TAU / pts as f64;
    let nf = f64::from(n);
    let s: f64 = (0..pts)
        .map(|k| {
            let t = k as f64 * h;
            (nf * t - x * t.sin()).cos()
        })
        .sum();
    s / pts as f64
}

/// `J_n'(x) = (J_{n−1}(x) − J_{n+1}(x)) / 2`.
pub(crate) fn jn_prime(n: i32, x: f64) -> f64 {
    0.5 * (jn(n - 1, x) - jn(n + 1, x))
}

/// Nodes and weights of the `N`-point Gauss–Legendre rule on `[−1, 1]`
/// (Newton's method on the Legendre recurrence).
fn gauss_legendre<const N: usize>() -> ([f64; N], [f64; N]) {
    let (mut x, mut w) = ([0.0; N], [0.0; N]);
    let nf = N as f64;
    for i in 0..N {
        let mut t = (core::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        for _ in 0..100 {
            // P_N(t) and P_{N−1}(t).
            let (mut p0, mut p1) = (1.0, t);
            for k in 2..=N {
                let kf = k as f64;
                (p0, p1) = (p1, ((2.0 * kf - 1.0) * t * p1 - (kf - 1.0) * p0) / kf);
            }
            let dp = nf * (t * p1 - p0) / (t * t - 1.0);
            let dt = p1 / dp;
            t -= dt;
            if dt.abs() < 1e-16 {
                let (mut p0, mut p1) = (1.0, t);
                for k in 2..=N {
                    let kf = k as f64;
                    (p0, p1) = (p1, ((2.0 * kf - 1.0) * t * p1 - (kf - 1.0) * p0) / kf);
                }
                let dp = nf * (t * p1 - p0) / (t * t - 1.0);
                x[i] = t;
                w[i] = 2.0 / ((1.0 - t * t) * dp * dp);
                break;
            }
        }
    }
    (x, w)
}

/// `∫_a^b f` with the 16-point Gauss–Legendre rule on `panels` equal
/// panels.
fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64, panels: usize) -> f64 {
    let (x, w) = gauss_legendre::<16>();
    let h = (b - a) / panels as f64;
    let mut s = 0.0;
    for p in 0..panels {
        let c = a + (p as f64 + 0.5) * h;
        for (xi, wi) in x.iter().zip(&w) {
            s += wi * f(c + 0.5 * h * xi);
        }
    }
    0.5 * h * s
}

/// `Y_n(x)`, the Bessel function of the second kind, for integer
/// `n ≥ 0` and `x > 0`, from the integral representation (Abramowitz &
/// Stegun 9.1.22; Watson §6.21)
///
/// ```text
/// Y_n(x) = (1/π) ∫₀^π sin(x·sin θ − nθ) dθ
///        − (1/π) ∫₀^∞ (e^{nt} + (−1)^n e^{−nt}) e^{−x·sinh t} dt
/// ```
///
/// with composite Gauss–Legendre quadrature: panels of at most about one
/// radian of phase for the first integral, of width 1/8 up to
/// `x·sinh t − n·t ≥ 60` for the second (relative accuracy ~1e-13).
pub(crate) fn yn(n: u32, x: f64) -> f64 {
    assert!(x > 0.0, "Y_n needs x > 0");
    let nf = f64::from(n);
    let p1 = ((x + nf + 1.0) * core::f64::consts::PI / 6.0).ceil() as usize;
    let first = integrate(
        |t| (x * t.sin() - nf * t).sin(),
        0.0,
        core::f64::consts::PI,
        p1.max(2),
    );
    // Upper limit: the integrand is below e^{−60} of its scale.
    let mut t_max = 1.0f64;
    while x * t_max.sinh() - nf * t_max < 60.0 {
        t_max += 1.0;
    }
    let sign = if n.is_multiple_of(2) { 1.0 } else { -1.0 };
    let second = integrate(
        |t| ((nf * t).exp() + sign * (-nf * t).exp()) * (-x * t.sinh()).exp(),
        0.0,
        t_max,
        (8.0 * t_max).ceil() as usize,
    );
    (first - second) / core::f64::consts::PI
}

/// The Hankel functions `H_n^(1)(x) = J_n + j·Y_n` and
/// `H_n^(2)(x) = J_n − j·Y_n` (`x > 0`), as `(re, im)` pairs.
pub(crate) fn hankel(n: u32, x: f64) -> ((f64, f64), (f64, f64)) {
    let (j, y) = (jn(n as i32, x), yn(n, x));
    ((j, y), (j, -y))
}

/// The `m`-th positive zero (`m ≥ 1`) of `f`, scanning from just above 0.
fn nth_zero(f: impl Fn(f64) -> f64, m: u32) -> f64 {
    assert!(m >= 1, "zeros are counted from 1");
    let step = 0.05;
    let mut a = 1e-6;
    let mut fa = f(a);
    let mut found = 0;
    loop {
        let b = a + step;
        let fb = f(b);
        if fa == 0.0 || fa.signum() != fb.signum() {
            found += 1;
            if found == m {
                let (mut lo, mut hi, mut flo) = (a, b, fa);
                if flo == 0.0 {
                    return lo;
                }
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if mid <= lo || mid >= hi {
                        break;
                    }
                    let fm = f(mid);
                    if fm == 0.0 {
                        return mid;
                    }
                    if fm.signum() == flo.signum() {
                        lo = mid;
                        flo = fm;
                    } else {
                        hi = mid;
                    }
                }
                return 0.5 * (lo + hi);
            }
        }
        a = b;
        fa = fb;
    }
}

/// `p_nm`: the `m`-th positive zero of `J_n` (TM_nm cutoff `k_c·a`).
pub(crate) fn jn_zero(n: u32, m: u32) -> f64 {
    let n = n as i32;
    nth_zero(|x| jn(n, x), m)
}

/// `p'_nm`: the `m`-th positive zero of `J_n'` (TE_nm cutoff `k_c·a`;
/// `x = 0` is not counted for `n = 0`).
pub(crate) fn jn_prime_zero(n: u32, m: u32) -> f64 {
    let n = n as i32;
    nth_zero(|x| jn_prime(n, x), m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use csx::expr::Expr;

    #[test]
    fn tabulated_zeros() {
        // Abramowitz & Stegun, tables 9.5 and 9.7 (Pozar table 3.3/3.4).
        let tm = [
            (0, 1, 2.404_825_557_7),
            (0, 2, 5.520_078_110_3),
            (1, 1, 3.831_705_970_2),
            (1, 2, 7.015_586_669_8),
            (2, 1, 5.135_622_301_8),
        ];
        for (n, m, p) in tm {
            let z = jn_zero(n, m);
            assert!((z - p).abs() < 1e-9, "p_{n}{m} = {z}, want {p}");
        }
        let te = [
            (0, 1, 3.831_705_970_2),
            (1, 1, 1.841_183_781_3),
            (1, 2, 5.331_442_773_5),
            (2, 1, 3.054_236_928_2),
            (3, 1, 4.201_188_941_2),
        ];
        for (n, m, p) in te {
            let z = jn_prime_zero(n, m);
            assert!((z - p).abs() < 1e-9, "p'_{n}{m} = {z}, want {p}");
        }
    }

    #[test]
    fn matches_csx_jn() {
        for n in [-2, -1, 0, 1, 2, 3, 5] {
            for x in [0.0, 0.3, 1.0, 2.5, 4.0, 7.7, 15.0, -3.0] {
                let e = Expr::parse(&format!("jn({n}, {x})")).unwrap();
                let want = e.eval_with(|_| None).unwrap();
                let got = jn(n, x);
                assert!((got - want).abs() < 1e-12, "J_{n}({x}) = {got}, csx {want}");
            }
        }
        assert!((jn(0, 0.0) - 1.0).abs() < 1e-15);
        // J0' = −J1.
        assert!((jn_prime(0, 1.3) + jn(1, 1.3)).abs() < 1e-15);
    }

    #[test]
    fn second_kind() {
        // Abramowitz & Stegun, table 9.1.
        for (n, x, want) in [
            (0, 1.0, 0.088_256_964_215_677),
            (1, 1.0, -0.781_212_821_300_289),
            (0, 10.0, 0.055_671_167_283_599),
            (1, 10.0, 0.249_015_424_206_954),
            (0, 0.1, -1.534_238_651_350_367),
            (1, 0.1, -6.458_951_094_702_027),
        ] {
            let y = yn(n, x);
            assert!((y - want).abs() < 1e-12, "Y_{n}({x}) = {y}, want {want}");
        }
        // Wronskian J_{n+1}·Y_n − J_n·Y_{n+1} = 2/(πx), over a wide range.
        for x in [0.01, 0.2, 0.9, 3.3, 7.0, 15.5, 42.0, 120.0] {
            for n in 0..2u32 {
                let ni = n as i32;
                let w = jn(ni + 1, x) * yn(n, x) - jn(ni, x) * yn(n + 1, x);
                let want = 2.0 / (core::f64::consts::PI * x);
                assert!(
                    (w - want).abs() < 1e-11 * want,
                    "x = {x}, n = {n}: {w} vs {want}"
                );
            }
        }
        // Gauss–Legendre: exact for polynomials of degree ≤ 31.
        let r = integrate(|t| t.powi(31) + t.powi(30), 0.0, 1.0, 1);
        assert!((r - (1.0 / 32.0 + 1.0 / 31.0)).abs() < 1e-15);
    }
}
