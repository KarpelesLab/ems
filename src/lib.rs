//! # ems
//!
//! Pure Rust FDTD (finite-difference time-domain) electromagnetic field
//! solver, compatible with openEMS models.
//!
//! Geometry, materials and the rectilinear grid are described with the
//! [`csx`] crate (re-exported here); `ems` turns such a model into an FDTD
//! engine and runs it. See `docs/DESIGN.md` in the repository for the
//! architecture and roadmap.

pub mod axis;
pub mod compile;
pub mod constants;
pub mod engine;
pub mod error;
pub mod fdtd;
pub mod io;
pub mod mesh;
pub mod nf2ff;
pub mod port;
pub mod post;
mod sim;
pub mod waveform;

pub use axis::Axis;
pub use compile::{CompileOptions, Compiled};
pub use csx;
pub use engine::{CellMaterials, EdgeMaterials, Engine, Grid};
pub use error::{Error, Result};
pub use fdtd::{Boundaries, Boundary, Cpml, EndCriteria, Face, Fdtd, Mur};
pub use post::Complex64;
pub use sim::{
    DumpData, DumpResult, IncidentWave, ProbeKind, ProbeResult, ProbeValues, Results, SarResult,
    Simulation,
};
pub use waveform::{ExpressionWaveform, Waveform};

/// The main user-facing types, for glob import: `use ems::prelude::*;`.
pub mod prelude {
    pub use crate::axis::Axis;
    pub use crate::constants::{C0, EPS0, MU0, Z0};
    pub use crate::engine::{CellMaterials, EdgeMaterials, Engine, Grid};
    pub use crate::error::{Error, Result};
    pub use crate::fdtd::{Boundaries, Boundary, Cpml, EndCriteria, Face, Fdtd, Mur};
    pub use crate::nf2ff::{MirrorKind, Nf2ffBox, Nf2ffResult};
    pub use crate::port::{LumpedPort, MslPort, RectWaveguidePort};
    pub use crate::post::{Complex64, PortResult};
    pub use crate::sim::{Results, Simulation};
    pub use crate::waveform::{ExpressionWaveform, Waveform};
}
