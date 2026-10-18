//! Output file writers.
//!
//! These writers are independent of the engine: they take plain slices and
//! produce files that common tools read.
//!
//! * [`vtr`]: VTK XML rectilinear-grid files (`.vtr`) for field dumps, and
//!   `.pvd` collections for time or frequency series of them (ParaView,
//!   VisIt).
//! * [`csv`]: delimited text tables for probe time series and frequency
//!   series, with comment-prefixed metadata lines.

mod base64;
pub mod csv;
pub mod vtr;

pub use csv::{CsvWriter, HeaderStyle};
pub use vtr::{
    ComplexForm, Encoding, Field, HeaderType, Layout, PvdCollection, PvdEntry, VtkFloat, VtrWriter,
};
