//! Files in the formats openEMS writes, so that post-processing written for
//! openEMS reads ems results unchanged.
//!
//! The layouts are those *observed* in openEMS v0.37 output (black-box,
//! see `CLEANROOM.md` and `scripts/openems-oracle`); the reference outputs
//! live in `tests/fixtures/openems/`.
//!
//! * [`dump`]: field dumps (`DumpBox`) as VTK (`.vtr` / `.vts`) and HDF5
//!   files (current and legacy layout), including SAR dumps, plus the
//!   openEMS dump sampling rules (box snapping, sub-sampling,
//!   `OptResolution`, sampling interval, time window).
//! * [`probes`]: the probe text files (`<name>` time domain,
//!   `<name>_FD` frequency domain) and a reader for them.

pub mod dump;
pub mod probes;

pub use probes::{
    ProbeDomain, ProbeFile, ProbeFileKind, read_probe_file, write_probe, write_probes,
};
