//! Boundary operators for the outer faces of the domain (design §5.6).
//!
//! PEC and PMC faces need no operator (see [`FaceCondition`](super::FaceCondition));
//! absorbing faces are built as PEC and then taken over by one of these.

pub mod cpml;
pub mod mur;

pub use cpml::{CpmlBoundary, CpmlCoefficients};
pub use mur::MurBoundary;
