//! Boundary operators for the outer faces of the domain (design §5.6).
//!
//! PEC and PMC faces need no operator (see [`FaceCondition`](super::FaceCondition));
//! absorbing faces are built as PEC and then taken over by one of these.
//! Periodic faces are handled by the engine itself ([`periodic`]).
//! [`abc`] holds absorbing sheets on interior mesh planes (csx
//! `AbsorbingBC`).

pub mod abc;
pub mod cpml;
pub mod cyl_pml;
pub mod mur;
pub mod periodic;

pub use abc::{AbcSheet, AbsorbingSheet, SheetVelocity};
pub use cpml::{CpmlBoundary, CpmlCoefficients};
pub use mur::MurBoundary;
