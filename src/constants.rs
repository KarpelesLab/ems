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

/// The vacuum permittivity openEMS computes its operator and time step
/// with \[F/m\] (models read from openEMS files, design §5.16.10).
///
/// Measured on the black box, not taken from its source: every one of
/// 96 509 interior `vi = f32(dt/f32(ε0·A/l))` coefficients of 600 random
/// graded vacuum meshes (`--debug-operator`, fixed `TimeStep`) and the
/// Rennings time step of 736 meshes agree with this value, and values
/// 1e-10 (relative) above or below it already fail some; [`EPS0`] is
/// 1.3e-10 higher and rounds 10 of those coefficients differently.
pub const OPENEMS_EPS0: f64 = 8.854_187_817_62e-12;

/// The vacuum permeability openEMS computes its operator and time step
/// with \[H/m\] (see [`OPENEMS_EPS0`]): all 124 457 interior
/// `iv = f32(dt/f32(μ0·A/l))` of the same meshes agree with it, and no
/// value 4e-11 above or below it does ([`MU0`] is 5.8e-10 lower: 0.5 % of
/// the coefficients one `f32` step off). It is not `1/(ε0·c0²)` with
/// [`OPENEMS_EPS0`] (that is `4π·1e-7`, 4.5e-10 lower).
pub const OPENEMS_MU0: f64 = 1.256_637_062e-6;

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
