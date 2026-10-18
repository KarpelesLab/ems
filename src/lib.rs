//! # ems
//!
//! Pure Rust FDTD (finite-difference time-domain) electromagnetic field
//! solver, compatible with openEMS models.
//!
//! Geometry, materials and the rectilinear grid are described with the
//! [`csx`] crate (re-exported here); `ems` turns such a model into an FDTD
//! engine and runs it. See `docs/DESIGN.md` in the repository for the
//! architecture and roadmap.

pub mod constants;
pub mod error;

pub use csx;
pub use error::{Error, Result};
