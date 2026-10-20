//! openEMS-compatible output files, so that post-processing written for
//! openEMS reads ems results unchanged.
//!
//! The layouts were taken from the files a black-box openEMS writes (see
//! `CLEANROOM.md` and `scripts/openems-oracle`); the reference outputs
//! live in `tests/fixtures/openems/`.
//!
//! * [`dump`]: field dumps (`DumpBox`) as VTK (`.vtr` / `.vts`) and HDF5
//!   files (current and legacy layout), including SAR dumps, plus the
//!   openEMS dump sampling rules (box snapping, sub-sampling,
//!   `OptResolution`, sampling interval, time window).

pub mod dump;
