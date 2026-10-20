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
//! * `nf2ff` (feature `hdf5`): the openEMS `nf2ff` tool (near-field box
//!   dumps described by an nf2ff XML file → far-field HDF5 file).
//! * [`probes`]: the probe text files (`<name>` time domain,
//!   `<name>_FD` frequency domain, including the waveguide "mode
//!   matching" files) and a reader for them.
//! * [`signal`]: the excitation signal files `et` / `ht` and the signal
//!   length openEMS reports.
//! * [`stats`]: the `--dump-statistics` files.
//! * [`debug`]: the `--debug-material`, `--debug-operator`, `--debug-PEC`
//!   and `--debug-boxes` files.
//!
//! The `ems` command-line program (feature `xml`) writes all of them,
//! as a drop-in replacement for the `openEMS` executable (design §7.2).

pub mod debug;
pub mod dump;
#[cfg(feature = "hdf5")]
pub mod nf2ff;
pub mod probes;
pub mod signal;
pub mod stats;

pub use probes::{
    ProbeDomain, ProbeFile, ProbeFileKind, read_probe_file, write_probe, write_probes,
};
pub use signal::{ExcitationSignal, SignalSpan};
pub use stats::{RunStats, RunStatsLog};
