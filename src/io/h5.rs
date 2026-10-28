//! HDF5 output for field dumps and probe series (feature `hdf5`).
//!
//! Files are written with the pure-Rust [`hdf5rs`] crate and read with any
//! HDF5 tool (h5py, MATLAB/Octave `h5read`, `h5dump`, ParaView via the
//! numpy route). The layout is **ems-native** (described below) and versioned
//! with root attributes; it is *not* the openEMS HDF5 layout.
//!
//! # Dump files (`format = "ems-dump"`, `version = 1`)
//!
//! One file per dump ([`write_dump`], [`DumpResult::write_h5`]).
//!
//! | object | type | contents |
//! |---|---|---|
//! | `/` attributes | | `format` (`"ems-dump"`), `version` (u32 `1`), `name`, `id` (u32 csx property id), `quantity` (`"E"` or `"H"`), `domain` (`"time"` or `"frequency"`), `position` (`"nodes"` or `"cell_centers"`), `unit` |
//! | `/mesh/x`, `/mesh/y`, `/mesh/z` | f64 `[n_a]` | sample coordinates \[m\] (attribute `unit = "m"`) |
//! | `/step` | u64 `[nt]` | time domain: completed timesteps of each snapshot |
//! | `/time` | f64 `[nt]` | time domain: time of each snapshot \[s\] (E at `m·dt`, H at `(m − ½)·dt`) |
//! | `/frequency` | f64 `[nf]` | frequency domain: analysis frequencies \[Hz\] |
//! | `/E` or `/H` | `[n, nx, ny, nz, 3]` | the field; `n` = `nt` or `nf` |
//!
//! The field dataset is in C (row-major) order: element
//! `[s, i, j, k, c]` is component `c` (0 = x, 1 = y, 2 = z) at sample point
//! `(x[i], y[j], z[k])` of snapshot / frequency `s` (so in numpy
//! `f["E"][s, i, j, k, c]`). This is the engine's own order (z fastest), so no
//! reordering happens. Time-domain values are `f32` (`unit` V/m or A/m);
//! frequency-domain values are complex `f64` stored as the compound
//! `{r: f64, i: f64}` that h5py reads as `complex128` (`unit` V·s/m or
//! A·s/m, the DFT `Σ x(t)·e^{−j2πft}·Δt` of [`FreqDump::values`]). A
//! frequency dump also has a root attribute `samples` (u64, the number of
//! accumulated states).
//!
//! The field dataset is chunked with one chunk per snapshot / frequency, so
//! single snapshots read efficiently, and is deflate-compressed when
//! requested ([`H5Options::deflate`], feature `hdf5-deflate`).
//!
//! # Probe files (`format = "ems-probes"`, `version = 1`)
//!
//! Many probes in one file ([`write_probes`]). Each probe is a group named
//! after the probe (characters other than ASCII letters, digits, `-`, `_` and
//! `.` replaced by `_`; a `_<id>` suffix resolves collisions) with
//!
//! * attributes `name` (the original name), `kind` (`"voltage"`,
//!   `"current"`, `"efield"` or `"hfield"`), `id` (u32), `number` (u32),
//!   `dt` (f64, \[s\]);
//! * `time`: f64 `[n]` sample times \[s\];
//! * `values`: f64 `[n]` (voltage \[V\], current \[A\]) or `[n, 3]` (field
//!   vectors \[V/m\] or \[A/m\]), with a `unit` attribute;
//! * `frequencies`: f64 `[nf]`, the probe's `FD_Samples` \[Hz\], if any.
//!
//! # Example
//!
//! ```
//! # fn main() -> ems::Result<()> {
//! use ems::engine::{DumpPosition, DumpQuantity, DumpRegion, FieldSampler, Snapshot};
//! use ems::{Grid, io::h5};
//!
//! let grid = Grid::uniform([4, 4, 4], [1e-3; 3])?;
//! let s = FieldSampler::new(&grid, DumpRegion::whole(&grid), DumpQuantity::E, DumpPosition::Nodes)?;
//! let snap = Snapshot::new(10, 1e-11, vec![[0.0, 0.0, 1.0]; s.len()]);
//! let bytes = h5::time_dump_bytes("Et", 0, &s, &[snap], &h5::H5Options::default())?;
//! assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");
//! # Ok(())
//! # }
//! ```

use std::path::Path;

use hdf5rs::Datatype;
use hdf5rs::format::Sizes;
use hdf5rs::write::{EncodeContext, FileWriter, GroupWriter, H5Write, WriteAt};

use crate::engine::{DumpPosition, DumpQuantity, FieldSampler, FreqDump, Snapshot};
use crate::post::Complex64;
use crate::{Axis, DumpData, DumpResult, Error, ProbeKind, ProbeResult, ProbeValues, Result};

/// Layout version written to the `version` root attribute.
pub const LAYOUT_VERSION: u32 = 1;

/// Options for the HDF5 writers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct H5Options {
    /// Deflate level (0–9) of the field datasets, `None` for uncompressed.
    pub deflate: Option<u32>,
}

impl H5Options {
    /// Compresses field datasets with deflate (gzip) at `level` (0–9,
    /// clamped), with the byte shuffle filter in front.
    #[cfg(feature = "hdf5-deflate")]
    #[must_use]
    pub fn deflate(mut self, level: u32) -> Self {
        self.deflate = Some(level.min(9));
        self
    }
}

/// Maps an `hdf5rs` error to [`Error::Io`].
fn h5(e: hdf5rs::Error) -> Error {
    Error::Io(format!("HDF5: {e}"))
}

/// A complex number as the h5py-compatible compound `{r: f64, i: f64}`.
struct C64(Complex64);

impl H5Write for C64 {
    fn datatype(sizes: Sizes) -> Datatype {
        Datatype::packed_compound(vec![
            ("r".into(), f64::datatype(sizes)),
            ("i".into(), f64::datatype(sizes)),
        ])
        .expect("two f64 members form a valid compound")
    }

    fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> hdf5rs::Result<()> {
        self.0.re.encode(out, ctx)?;
        self.0.im.encode(out, ctx)
    }
}

fn quantity_unit(q: DumpQuantity, freq: bool) -> &'static str {
    match (q, freq) {
        (DumpQuantity::E, false) => "V/m",
        (DumpQuantity::H, false) => "A/m",
        (DumpQuantity::E, true) => "V*s/m",
        (DumpQuantity::H, true) => "A*s/m",
        (DumpQuantity::J | DumpQuantity::RotH, false) => "A/m^2",
        (DumpQuantity::J | DumpQuantity::RotH, true) => "A*s/m^2",
        (DumpQuantity::D, false) => "C/m^2",
        (DumpQuantity::D, true) => "C*s/m^2",
        (DumpQuantity::B, false) => "T",
        (DumpQuantity::B, true) => "T*s",
    }
}

/// Root attributes and `/mesh` of a dump file.
fn dump_header<S: WriteAt>(
    f: &mut FileWriter<S>,
    name: &str,
    id: u32,
    s: &FieldSampler,
    freq: bool,
) -> hdf5rs::Result<()> {
    let mut root = f.root();
    root.set_attr("format", "ems-dump")?;
    root.set_attr("version", &LAYOUT_VERSION)?;
    root.set_attr("name", name)?;
    root.set_attr("id", &id)?;
    root.set_attr("quantity", s.quantity().name())?;
    root.set_attr("domain", if freq { "frequency" } else { "time" })?;
    root.set_attr(
        "position",
        match s.position() {
            DumpPosition::Nodes => "nodes",
            DumpPosition::CellCenters => "cell_centers",
            DumpPosition::Native => "native",
        },
    )?;
    root.set_attr("unit", quantity_unit(s.quantity(), freq))?;
    let mut mesh = root.create_group("mesh")?;
    for (axis, n) in Axis::ALL.into_iter().zip(["x", "y", "z"]) {
        let c = s.coords(axis);
        let mut ds = mesh
            .new_dataset::<f64>(n)
            .shape(&[c.len() as u64])
            .write(c)?;
        ds.set_attr("unit", "m")?;
    }
    Ok(())
}

/// Creates the chunked, growable `[0, nx, ny, nz, 3]` field dataset.
fn field_dataset<'w, S: WriteAt, T: H5Write>(
    root: &'w mut GroupWriter<'_, S>,
    s: &FieldSampler,
    opts: &H5Options,
) -> hdf5rs::Result<hdf5rs::write::DatasetWriter<'w, S>> {
    let [nx, ny, nz] = s.dims().map(|n| n as u64);
    let mut b = root
        .new_dataset::<T>(s.quantity().name())
        .shape(&[0, nx, ny, nz, 3])
        .max_shape(&[None, Some(nx), Some(ny), Some(nz), Some(3)])
        .chunk(&[1, nx, ny, nz, 3]);
    if let Some(level) = opts.deflate {
        b = b.shuffle().deflate(level);
    }
    b.create()
}

fn fill_time<S: WriteAt>(
    f: &mut FileWriter<S>,
    name: &str,
    id: u32,
    s: &FieldSampler,
    snapshots: &[Snapshot],
    opts: &H5Options,
) -> hdf5rs::Result<()> {
    dump_header(f, name, id, s, false)?;
    let mut root = f.root();
    let n = snapshots.len() as u64;
    let steps: Vec<u64> = snapshots.iter().map(|x| x.step).collect();
    let times: Vec<f64> = snapshots.iter().map(|x| x.time).collect();
    root.new_dataset::<u64>("step").shape(&[n]).write(&steps)?;
    root.new_dataset::<f64>("time")
        .shape(&[n])
        .write(&times)?
        .set_attr("unit", "s")?;
    let mut ds = field_dataset::<S, f32>(&mut root, s, opts)?;
    for snap in snapshots {
        if snap.data.len() != s.len() {
            return Err(hdf5rs::Error::invalid(format!(
                "snapshot of step {} has {} points, the sampler {}",
                snap.step,
                snap.data.len(),
                s.len()
            )));
        }
        ds.append(snap.data.as_flattened())?;
    }
    ds.set_attr("unit", quantity_unit(s.quantity(), false))
}

fn fill_freq<S: WriteAt>(
    f: &mut FileWriter<S>,
    name: &str,
    id: u32,
    d: &FreqDump,
    opts: &H5Options,
) -> hdf5rs::Result<()> {
    let s = d.sampler();
    dump_header(f, name, id, s, true)?;
    let mut root = f.root();
    root.set_attr("samples", &d.samples())?;
    root.new_dataset::<f64>("frequency")
        .shape(&[d.freqs().len() as u64])
        .write(d.freqs())?
        .set_attr("unit", "Hz")?;
    let mut ds = field_dataset::<S, C64>(&mut root, s, opts)?;
    for k in 0..d.freqs().len() {
        let v: Vec<C64> = d.values(k).into_iter().flatten().map(C64).collect();
        ds.append(&v)?;
    }
    ds.set_attr("unit", quantity_unit(s.quantity(), true))
}

fn finish_to(
    path: &Path,
    fill: impl FnOnce(&mut FileWriter<std::fs::File>) -> hdf5rs::Result<()>,
) -> Result<()> {
    let mut f = FileWriter::create(path).map_err(h5)?;
    fill(&mut f).map_err(h5)?;
    f.finish().map_err(h5)?;
    Ok(())
}

fn finish_bytes(
    fill: impl FnOnce(&mut FileWriter<Vec<u8>>) -> hdf5rs::Result<()>,
) -> Result<Vec<u8>> {
    let mut f = FileWriter::in_memory();
    fill(&mut f).map_err(h5)?;
    f.finish().map_err(h5)
}

/// Writes a time-domain dump (`snapshots` of `sampler`) to `path`, named
/// `name` with csx property id `id` (see the [module docs](self)).
///
/// # Errors
/// I/O and encoding errors ([`Error::Io`]), a snapshot whose size does not
/// match the sampler.
pub fn write_time_dump(
    path: impl AsRef<Path>,
    name: &str,
    id: u32,
    sampler: &FieldSampler,
    snapshots: &[Snapshot],
    opts: &H5Options,
) -> Result<()> {
    finish_to(path.as_ref(), |f| {
        fill_time(f, name, id, sampler, snapshots, opts)
    })
}

/// [`write_time_dump`] into memory: the bytes of the HDF5 file.
///
/// # Errors
/// See [`write_time_dump`].
pub fn time_dump_bytes(
    name: &str,
    id: u32,
    sampler: &FieldSampler,
    snapshots: &[Snapshot],
    opts: &H5Options,
) -> Result<Vec<u8>> {
    finish_bytes(|f| fill_time(f, name, id, sampler, snapshots, opts))
}

/// Writes a frequency-domain dump to `path` (see the [module docs](self)).
///
/// # Errors
/// I/O and encoding errors ([`Error::Io`]).
pub fn write_freq_dump(
    path: impl AsRef<Path>,
    name: &str,
    id: u32,
    dump: &FreqDump,
    opts: &H5Options,
) -> Result<()> {
    finish_to(path.as_ref(), |f| fill_freq(f, name, id, dump, opts))
}

/// [`write_freq_dump`] into memory: the bytes of the HDF5 file.
///
/// # Errors
/// See [`write_freq_dump`].
pub fn freq_dump_bytes(name: &str, id: u32, dump: &FreqDump, opts: &H5Options) -> Result<Vec<u8>> {
    finish_bytes(|f| fill_freq(f, name, id, dump, opts))
}

/// Writes a simulation dump result to `path` (time or frequency domain).
///
/// # Errors
/// See [`write_time_dump`].
pub fn write_dump(path: impl AsRef<Path>, dump: &DumpResult, opts: &H5Options) -> Result<()> {
    match dump.data() {
        DumpData::Time { sampler, snapshots } => {
            write_time_dump(path, dump.name(), dump.id(), sampler, snapshots, opts)
        }
        DumpData::Freq(d) => write_freq_dump(path, dump.name(), dump.id(), d, opts),
    }
}

/// [`write_dump`] into memory: the bytes of the HDF5 file.
///
/// # Errors
/// See [`write_time_dump`].
pub fn dump_bytes(dump: &DumpResult, opts: &H5Options) -> Result<Vec<u8>> {
    match dump.data() {
        DumpData::Time { sampler, snapshots } => {
            time_dump_bytes(dump.name(), dump.id(), sampler, snapshots, opts)
        }
        DumpData::Freq(d) => freq_dump_bytes(dump.name(), dump.id(), d, opts),
    }
}

/// The group name of a probe: the safe characters of `name`.
fn group_name(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-_.".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() || s == "." || s == ".." {
        format!("_{s}")
    } else {
        s
    }
}

fn fill_probes<'a, S: WriteAt>(
    f: &mut FileWriter<S>,
    probes: impl IntoIterator<Item = &'a ProbeResult>,
) -> hdf5rs::Result<()> {
    let mut root = f.root();
    root.set_attr("format", "ems-probes")?;
    root.set_attr("version", &LAYOUT_VERSION)?;
    let mut used = std::collections::HashSet::new();
    for p in probes {
        let mut g = group_name(p.name());
        if !used.insert(g.clone()) {
            g = format!("{g}_{}", p.id());
            let mut k = 2;
            while !used.insert(g.clone()) {
                g = format!("{}_{}_{k}", group_name(p.name()), p.id());
                k += 1;
            }
        }
        let mut grp = root.create_group(&g)?;
        grp.set_attr("name", p.name())?;
        grp.set_attr(
            "kind",
            match p.kind() {
                ProbeKind::Voltage => "voltage",
                ProbeKind::Current => "current",
                ProbeKind::EField => "efield",
                ProbeKind::HField => "hfield",
            },
        )?;
        grp.set_attr("id", &p.id())?;
        grp.set_attr("number", &p.number())?;
        grp.set_attr("dt", &p.dt())?;
        let n = p.times().len() as u64;
        grp.new_dataset::<f64>("time")
            .shape(&[n])
            .write(p.times())?
            .set_attr("unit", "s")?;
        let unit = match p.kind() {
            ProbeKind::Voltage => "V",
            ProbeKind::Current => "A",
            ProbeKind::EField => "V/m",
            ProbeKind::HField => "A/m",
        };
        let mut ds = match p.values() {
            ProbeValues::Scalar(v) => grp
                .new_dataset::<f64>("values")
                .shape(&[v.len() as u64])
                .write(v)?,
            ProbeValues::Vector(v) => grp
                .new_dataset::<f64>("values")
                .shape(&[v.len() as u64, 3])
                .write(v.as_flattened())?,
        };
        ds.set_attr("unit", unit)?;
        if !p.frequencies().is_empty() {
            grp.new_dataset::<f64>("frequencies")
                .shape(&[p.frequencies().len() as u64])
                .write(p.frequencies())?;
        }
    }
    Ok(())
}

/// Writes probe time series to `path`, one group per probe (see the
/// [module docs](self)), e.g. `results.probes()` chained with
/// `results.lumped_currents()`.
///
/// # Errors
/// I/O and encoding errors ([`Error::Io`]).
pub fn write_probes<'a>(
    path: impl AsRef<Path>,
    probes: impl IntoIterator<Item = &'a ProbeResult>,
) -> Result<()> {
    finish_to(path.as_ref(), |f| fill_probes(f, probes))
}

/// [`write_probes`] into memory: the bytes of the HDF5 file.
///
/// # Errors
/// See [`write_probes`].
pub fn probes_bytes<'a>(probes: impl IntoIterator<Item = &'a ProbeResult>) -> Result<Vec<u8>> {
    finish_bytes(|f| fill_probes(f, probes))
}

#[cfg(test)]
mod tests {
    use super::group_name;

    #[test]
    fn group_names() {
        assert_eq!(group_name("port_ut1"), "port_ut1");
        assert_eq!(group_name("a/b c"), "a_b_c");
        assert_eq!(group_name(""), "_");
        assert_eq!(group_name(".."), "_..");
    }
}
