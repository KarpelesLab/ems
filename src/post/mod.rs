//! Post-processing: complex arithmetic and discrete Fourier transforms.
//!
//! Sign convention used throughout `ems`: a time signal `x(t)` is mapped to
//! the frequency domain with the kernel `e^{-j2πft}`, i.e.
//! `X(f) = ∫ x(t)·e^{-j2πft} dt` (engineering convention, `j² = -1`). The
//! discrete transforms in [`mod@dft`] approximate this integral, and
//! [`Waveform::spectrum`](crate::waveform::Waveform::spectrum) evaluates it
//! analytically.

pub mod complex;
pub mod dft;

pub use complex::Complex64;
pub use dft::{RunningDft, dft};
