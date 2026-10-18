//! Physical constants (SI units).
//!
//! Values follow the CODATA 2022 recommendation. Since the 2019 SI revision
//! the speed of light is exact while `μ0` is a measured quantity; `ε0` and
//! `Z0` are derived from those two so that `ε0·μ0·c0² = 1` holds exactly up to
//! floating-point rounding.

/// Speed of light in vacuum, `c0` \[m/s\] (exact).
pub const C0: f64 = 299_792_458.0;

/// Vacuum magnetic permeability, `μ0` \[H/m\] (CODATA 2022).
pub const MU0: f64 = 1.256_637_061_27e-6;

/// Vacuum electric permittivity, `ε0 = 1/(μ0·c0²)` \[F/m\].
pub const EPS0: f64 = 1.0 / (MU0 * C0 * C0);

/// Characteristic impedance of vacuum, `Z0 = μ0·c0` \[Ω\].
pub const Z0: f64 = MU0 * C0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_constants() {
        // CODATA 2022: ε0 = 8.8541878188(14)e-12 F/m, Z0 = 376.730313412(59) Ω.
        assert!((EPS0 / 8.854_187_818_8e-12 - 1.0).abs() < 1e-10);
        assert!((Z0 / 376.730_313_412 - 1.0).abs() < 1e-10);
        assert!((EPS0 * MU0 * C0 * C0 - 1.0).abs() < 1e-15);
    }
}
