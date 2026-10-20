//! Bessel functions of the first kind `J_n` (integer order) and the zeros
//! of `J_n` and `J_n'`, for the circular waveguide modes
//! ([`CircWaveguidePort`](super::CircWaveguidePort)).
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
}
