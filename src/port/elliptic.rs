//! Complete elliptic integral of the first kind, for the conformal-mapping
//! impedances of the coplanar and strip-line ports.
//!
//! `K(k) = ∫₀^{π/2} dθ / √(1 − k²·sin²θ)` (modulus `k`, complementary
//! modulus `k' = √(1 − k²)`) is computed with Gauss's arithmetic-geometric
//! mean: `K(k) = π / (2·AGM(1, k'))` (Abramowitz & Stegun 17.6). The AGM
//! converges quadratically, so a handful of iterations reach full double
//! precision for any `0 ≤ k < 1`.

use core::f64::consts::FRAC_PI_2;

/// Arithmetic-geometric mean of `a, b ≥ 0`.
fn agm(mut a: f64, mut b: f64) -> f64 {
    for _ in 0..64 {
        if (a - b).abs() <= 1e-15 * a {
            break;
        }
        (a, b) = (0.5 * (a + b), (a * b).sqrt());
    }
    0.5 * (a + b)
}

/// `K(k)` for `0 ≤ k < 1` (`+∞` at `k = 1`).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn ellip_k(k: f64) -> f64 {
    let kp = ((1.0 - k) * (1.0 + k)).max(0.0).sqrt();
    FRAC_PI_2 / agm(1.0, kp)
}

/// `K(k)/K(k')` for `0 ≤ k ≤ 1`, with `k' = √(1 − k²)`.
pub(crate) fn k_ratio(k: f64) -> f64 {
    let kp = ((1.0 - k) * (1.0 + k)).max(0.0).sqrt();
    k_ratio_with(k, kp)
}

/// `K(k)/K(k')` with both moduli given (`k² + k'² = 1`), which avoids the
/// cancellation of `√(1 − k²)` near `k = 1`: `K(k)/K(k') =
/// AGM(1, k)/AGM(1, k')`.
pub(crate) fn k_ratio_with(k: f64, kp: f64) -> f64 {
    agm(1.0, k) / agm(1.0, kp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn known_values() {
        assert!((ellip_k(0.0) - FRAC_PI_2).abs() < 1e-15);
        // K(1/√2) = Γ(1/4)² / (4√π).
        assert!((ellip_k(0.5f64.sqrt()) - 1.854_074_677_301_372).abs() < 1e-14);
        // K(sin 75°) (A&S table 17.1: m = 0.9330127).
        let k = (75.0f64).to_radians().sin();
        assert!((ellip_k(k) - 2.768_063_145_368_768).abs() < 1e-9);
        // m = k² = 0.5 has K(k)/K(k') = 1; large-modulus limit
        // K(k) ≈ ln(4/k').
        assert!((k_ratio(0.5f64.sqrt()) - 1.0).abs() < 1e-15);
        let kp = 1e-4f64;
        let k = (1.0 - kp * kp).sqrt();
        assert!((ellip_k(k) - (4.0 / kp).ln()).abs() < 1e-6);
        let kp = 1e-12f64;
        assert!((k_ratio_with(1.0, kp) - (4.0 / kp).ln() / FRAC_PI_2).abs() < 1e-9);
        // Small modulus: K(k)/K(k') → π / ln(16/k²)... i.e. π/(2 ln(4/k)).
        let k = 1e-6;
        assert!((k_ratio(k) - PI / (2.0 * (4.0 / k).ln())).abs() < 1e-9);
        // Reciprocity of the ratio.
        let k = 0.3;
        assert!((k_ratio(k) * k_ratio((1.0f64 - k * k).sqrt()) - 1.0).abs() < 1e-14);
    }
}
