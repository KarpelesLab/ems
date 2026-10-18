//! Post-processing: complex arithmetic, discrete Fourier transforms and
//! port analysis (power waves, S-parameters).
//!
//! Sign convention used throughout `ems`: a time signal `x(t)` is mapped to
//! the frequency domain with the kernel `e^{-j2πft}`, i.e.
//! `X(f) = ∫ x(t)·e^{-j2πft} dt` (engineering convention, `j² = -1`). The
//! discrete transforms in [`mod@dft`] approximate this integral, and
//! [`Waveform::spectrum`](crate::waveform::Waveform::spectrum) evaluates it
//! analytically.

pub mod complex;
pub mod dft;
pub mod port;

pub use complex::Complex64;
pub use dft::{RunningDft, dft};
pub use port::{PortResult, s_param};
