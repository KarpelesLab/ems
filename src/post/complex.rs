//! A minimal double-precision complex number type.
//!
//! `ems` keeps its dependency footprint to `csx` only, so instead of pulling
//! in `num-complex` it carries this small type covering what frequency-domain
//! post-processing needs.

use core::fmt;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A complex number `re + j·im` with `f64` parts.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex64 {
    /// `0 + 0j`.
    pub const ZERO: Complex64 = Complex64::new(0.0, 0.0);
    /// `1 + 0j`.
    pub const ONE: Complex64 = Complex64::new(1.0, 0.0);
    /// The imaginary unit `0 + 1j`.
    pub const I: Complex64 = Complex64::new(0.0, 1.0);

    /// Creates `re + j·im`.
    pub const fn new(re: f64, im: f64) -> Self {
        Complex64 { re, im }
    }

    /// Creates the complex number with modulus `r` and argument `theta`
    /// \[rad\]: `r·(cos θ + j sin θ)`.
    pub fn from_polar(r: f64, theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Complex64::new(r * c, r * s)
    }

    /// `e^{jθ} = cos θ + j sin θ` (unit phasor of argument `theta` \[rad\]).
    pub fn cis(theta: f64) -> Self {
        Complex64::from_polar(1.0, theta)
    }

    /// Complex conjugate `re − j·im`.
    pub const fn conj(self) -> Self {
        Complex64::new(self.re, -self.im)
    }

    /// Modulus `|z|`, computed without intermediate overflow.
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Squared modulus `|z|² = re² + im²`.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Argument (phase) in `[−π, π]` \[rad\].
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Modulus and argument `(|z|, arg z)`.
    pub fn to_polar(self) -> (f64, f64) {
        (self.norm(), self.arg())
    }

    /// Complex exponential `e^z = e^{re}·(cos im + j sin im)`.
    pub fn exp(self) -> Self {
        Complex64::from_polar(self.re.exp(), self.im)
    }

    /// Reciprocal `1/z`.
    pub fn inv(self) -> Self {
        Complex64::ONE / self
    }

    /// Multiplies by a real scalar.
    pub fn scale(self, k: f64) -> Self {
        Complex64::new(self.re * k, self.im * k)
    }

    /// `true` if both parts are finite.
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// `true` if either part is NaN.
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

impl From<f64> for Complex64 {
    fn from(re: f64) -> Self {
        Complex64::new(re, 0.0)
    }
}

impl fmt::Display for Complex64 {
    /// Formats as `a+bj` / `a-bj`. A precision applies to both parts, e.g.
    /// `format!("{z:.3}")`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.re, f)?;
        let sign = if self.im.is_sign_negative() && !self.im.is_nan() {
            '-'
        } else {
            '+'
        };
        let im = self.im.abs();
        match f.precision() {
            Some(p) => write!(f, "{sign}{im:.p$}j"),
            None => write!(f, "{sign}{im}j"),
        }
    }
}

impl Neg for Complex64 {
    type Output = Complex64;
    fn neg(self) -> Complex64 {
        Complex64::new(-self.re, -self.im)
    }
}

impl Add for Complex64 {
    type Output = Complex64;
    fn add(self, o: Complex64) -> Complex64 {
        Complex64::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for Complex64 {
    type Output = Complex64;
    fn sub(self, o: Complex64) -> Complex64 {
        Complex64::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for Complex64 {
    type Output = Complex64;
    fn mul(self, o: Complex64) -> Complex64 {
        Complex64::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl Div for Complex64 {
    type Output = Complex64;
    /// Complex division using Smith's algorithm, which scales by the larger
    /// component of the divisor to avoid premature overflow/underflow
    /// (R. L. Smith, "Algorithm 116: Complex division", CACM 5(8), 1962).
    fn div(self, o: Complex64) -> Complex64 {
        let (a, b, c, d) = (self.re, self.im, o.re, o.im);
        if c.abs() >= d.abs() {
            let r = d / c;
            let den = c + d * r;
            Complex64::new((a + b * r) / den, (b - a * r) / den)
        } else {
            let r = c / d;
            let den = c * r + d;
            Complex64::new((a * r + b) / den, (b * r - a) / den)
        }
    }
}

impl Add<f64> for Complex64 {
    type Output = Complex64;
    fn add(self, o: f64) -> Complex64 {
        Complex64::new(self.re + o, self.im)
    }
}

impl Sub<f64> for Complex64 {
    type Output = Complex64;
    fn sub(self, o: f64) -> Complex64 {
        Complex64::new(self.re - o, self.im)
    }
}

impl Mul<f64> for Complex64 {
    type Output = Complex64;
    fn mul(self, o: f64) -> Complex64 {
        self.scale(o)
    }
}

impl Div<f64> for Complex64 {
    type Output = Complex64;
    fn div(self, o: f64) -> Complex64 {
        Complex64::new(self.re / o, self.im / o)
    }
}

impl Add<Complex64> for f64 {
    type Output = Complex64;
    fn add(self, o: Complex64) -> Complex64 {
        Complex64::new(self + o.re, o.im)
    }
}

impl Sub<Complex64> for f64 {
    type Output = Complex64;
    fn sub(self, o: Complex64) -> Complex64 {
        Complex64::new(self - o.re, -o.im)
    }
}

impl Mul<Complex64> for f64 {
    type Output = Complex64;
    fn mul(self, o: Complex64) -> Complex64 {
        o.scale(self)
    }
}

impl Div<Complex64> for f64 {
    type Output = Complex64;
    fn div(self, o: Complex64) -> Complex64 {
        Complex64::from(self) / o
    }
}

macro_rules! assign_ops {
    ($($tr:ident $m:ident $op:tt),*) => {$(
        impl $tr for Complex64 {
            fn $m(&mut self, o: Complex64) {
                *self = *self $op o;
            }
        }
        impl $tr<f64> for Complex64 {
            fn $m(&mut self, o: f64) {
                *self = *self $op o;
            }
        }
    )*};
}

assign_ops!(
    AddAssign add_assign +,
    SubAssign sub_assign -,
    MulAssign mul_assign *,
    DivAssign div_assign /
);

impl Sum for Complex64 {
    fn sum<I: Iterator<Item = Complex64>>(iter: I) -> Complex64 {
        iter.fold(Complex64::ZERO, |a, b| a + b)
    }
}

impl<'a> Sum<&'a Complex64> for Complex64 {
    fn sum<I: Iterator<Item = &'a Complex64>>(iter: I) -> Complex64 {
        iter.fold(Complex64::ZERO, |a, b| a + *b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{E, FRAC_PI_2, PI};

    fn close(a: Complex64, b: Complex64) -> bool {
        (a - b).norm() <= 1e-12 * (1.0 + b.norm())
    }

    #[test]
    fn arithmetic() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(-3.0, 0.5);
        assert_eq!(a + b, Complex64::new(-2.0, 2.5));
        assert_eq!(a - b, Complex64::new(4.0, 1.5));
        assert_eq!(a * b, Complex64::new(-4.0, -5.5));
        assert!(close(a / b * b, a));
        assert!(close(b / a * a, b));
        assert!(close(
            a / Complex64::new(0.5, -4.0) * Complex64::new(0.5, -4.0),
            a
        ));
        assert_eq!(-a, Complex64::new(-1.0, -2.0));
        assert_eq!(Complex64::I * Complex64::I, -Complex64::ONE);
    }

    #[test]
    fn mixed_with_f64() {
        let a = Complex64::new(1.0, 2.0);
        assert_eq!(a + 1.0, Complex64::new(2.0, 2.0));
        assert_eq!(1.0 + a, Complex64::new(2.0, 2.0));
        assert_eq!(a - 1.0, Complex64::new(0.0, 2.0));
        assert_eq!(1.0 - a, Complex64::new(0.0, -2.0));
        assert_eq!(a * 2.0, Complex64::new(2.0, 4.0));
        assert_eq!(2.0 * a, Complex64::new(2.0, 4.0));
        assert_eq!(a / 2.0, Complex64::new(0.5, 1.0));
        assert!(close(1.0 / a, Complex64::new(0.2, -0.4)));
        assert!(close(a.inv(), Complex64::new(0.2, -0.4)));
    }

    #[test]
    fn assign_ops() {
        let mut z = Complex64::new(1.0, 1.0);
        z += Complex64::ONE;
        z -= 0.5;
        z *= Complex64::I;
        z /= 2.0;
        assert!(close(z, Complex64::new(-0.5, 0.75)));
        z *= 4.0;
        z /= Complex64::new(0.0, 1.0);
        assert!(close(z, Complex64::new(3.0, 2.0)));
        z -= Complex64::new(3.0, 2.0);
        z += 1.0;
        assert!(close(z, Complex64::ONE));
    }

    #[test]
    fn division_extreme_scale() {
        // The naive formula would overflow / underflow c² + d² here.
        let big = Complex64::new(1e300, 1e300);
        assert!(close(big / big, Complex64::ONE));
        let tiny = Complex64::new(1e-300, -1e-300);
        assert!(close(tiny / tiny, Complex64::ONE));
    }

    #[test]
    fn polar() {
        let z = Complex64::new(3.0, 4.0);
        assert_eq!(z.norm(), 5.0);
        assert_eq!(z.norm_sqr(), 25.0);
        assert_eq!(z.conj(), Complex64::new(3.0, -4.0));
        let (r, th) = z.to_polar();
        assert!(close(Complex64::from_polar(r, th), z));
        assert!((Complex64::I.arg() - FRAC_PI_2).abs() < 1e-15);
        assert!((Complex64::new(-1.0, 0.0).arg() - PI).abs() < 1e-15);
        assert!(close(Complex64::cis(PI), Complex64::new(-1.0, 0.0)));
        assert!(close(
            Complex64::new(1.0, PI).exp(),
            Complex64::new(-E, 0.0)
        ));
        assert!(close(z * z.conj(), Complex64::from(z.norm_sqr())));
    }

    #[test]
    fn display() {
        assert_eq!(Complex64::new(1.0, 2.0).to_string(), "1+2j");
        assert_eq!(Complex64::new(1.0, -2.5).to_string(), "1-2.5j");
        assert_eq!(Complex64::new(-0.5, 0.0).to_string(), "-0.5+0j");
        assert_eq!(
            format!("{:.2}", Complex64::new(1.0, -1.0 / 3.0)),
            "1.00-0.33j"
        );
    }

    #[test]
    fn sum_and_default() {
        let v = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -3.0)];
        let s: Complex64 = v.iter().sum();
        assert_eq!(s, Complex64::new(3.0, -2.0));
        let s2: Complex64 = v.into_iter().sum();
        assert_eq!(s, s2);
        assert_eq!(Complex64::default(), Complex64::ZERO);
        assert!(Complex64::ONE.is_finite());
        assert!(Complex64::new(f64::NAN, 0.0).is_nan());
    }
}
