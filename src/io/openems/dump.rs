//! openEMS-compatible field dump files (`DumpBox`) and the openEMS dump
//! sampling rules.
//!
//! Everything here reproduces what a black-box openEMS v0.37 writes for
//! the same model (fixtures in `tests/fixtures/openems/dump_*`), so that
//! scripts reading openEMS dumps (h5py / Octave `ReadHDF5Dump`, ParaView)
//! read ems dumps unchanged. [`Simulation`](crate::Simulation) samples
//! every `DumpBox` with these rules; [`write_dump`] writes a
//! [`DumpResult`] in the format its `FileType` asks for.
//!
//! # Quantities
//!
//! | `DumpType` | quantity | VTK array | HDF5 `dump_type` |
//! |---|---|---|---|
//! | 0 / 10 | E \[V/m\] | `E-Field` | 0 |
//! | 1 / 11 | H \[A/m\] | `H-Field` | 1 |
//! | 2 / 12 | conduction current density `κE` \[A/m²\] | `J-Field` | 2 |
//! | 3 / 13 | total current density `∇×H` \[A/m²\] | `RotH-Field` | 3 |
//! | 4 / 14 | `D = εE` \[C/m²\] | `D-Field` | 4 |
//! | 5 / 15 | `B = μH` \[T\] | `B-Field` | 5 |
//!
//! Types 10–15 are the frequency-domain versions (frequencies from the
//! `<FD_Samples>` child element), 20/21/22/29 SAR dumps.
//!
//! # Sample points (`DumpMode`)
//!
//! * 0 (no interpolation, [`DumpPosition::Native`](crate::engine::DumpPosition::Native)): the raw edge value
//!   of each node; E, J, D, rot H at the primary nodes, H and B at the dual
//!   nodes (cell centres, the last one `l[n−1] + Δ_{n−2}/2` beyond the
//!   mesh). HDF5 attribute `dual_mesh` = 1 for H and B.
//! * 1 (node interpolation, [`DumpPosition::Nodes`](crate::engine::DumpPosition::Nodes)): at the primary
//!   nodes.
//! * 2 (cell interpolation, [`DumpPosition::CellCenters`](crate::engine::DumpPosition::CellCenters)): at the dual
//!   nodes, including the virtual one beyond the mesh (value 0).
//!
//! **Box snapping** ([`sample_indices`]): per axis, over the candidate
//! lines (primary lines for modes 0/1, dual lines for mode 2): a box
//! entirely outside the mesh disables the dump; if both corners have the
//! same nearest line (ties to the lower one) the dump is one line thick
//! there; otherwise the range runs from the last line at or below the
//! lower corner to the first line at or above the upper one (clamped to
//! the mesh). `SubSampling` takes every `s`-th line from the first one;
//! `OptResolution` (drawing units) walks from the first line, each time
//! to the farthest line closer than the resolution (or the next line),
//! up to the last.
//!
//! # Sampling in time
//!
//! openEMS "step `n`" holds E at `n·dt` and H at `(n + ½)·dt`
//! ([`DumpTiming::OpenEms`](crate::engine::DumpTiming::OpenEms)). With `N = ⌊1/(2·f_max·dt)⌋` the Nyquist
//! step count of the excitation ([`nyquist_steps`]):
//!
//! * time-domain dumps record the steps `n ≡ 0 (mod ⌊N/OverSampling⌋)`
//!   ([`time_domain_interval`], `OverSampling` default 4);
//! * frequency-domain dumps accumulate the steps
//!   `n ≡ 0 (mod K)` with `K = min(N, ⌊1/(2·f·dt)⌋)` for the highest dump
//!   frequency `f` ([`frequency_domain_interval`]), as
//!   `F(f) = 2·Σ_n x(t_n)·e^{−j2πf·t_n}·K·dt` with `t_n` the quantity's
//!   time (so the HDF5 values are twice [`FreqDump::values`]);
//! * `StartTime` / `StopTime` ([`step_window`]) keep the steps with
//!   `(n + 1)·dt ≥ start` and `(n − 1)·dt < stop` (stop ≤ 0: no limit).
//!
//! openEMS also records the state after the last step `N_max` (H at
//! `(N_max + ½)·dt`, one H update past the run). [`Simulation::run`]
//! adds that sample from a local look-ahead H update
//! ([`FieldSampler::sample_lookahead`]; without the CPML / H-source
//! corrections, and not on periodic, multigrid or axis/closed cylindrical
//! grids).
//!
//! [`Simulation::run`]: crate::Simulation::run
//!
//! # SAR dumps
//!
//! `DumpType` 20 (local), 21 (1 g), 22 (10 g) and 29 (raw) use the
//! cell-interpolated E DFT of the frequency-domain dumps on the cells
//! `lo ..= hi` of the box snapped to the primary lines (each index the
//! cell starting there, the last possibly the virtual one), with the cell
//! conductivity `κ` and density `ρ`:
//!
//! * local SAR `κ·|E|²/(2ρ)` (0 outside tissue), absorbed power
//!   `Σ ½·κ·|E|²·V`, tissue mass `Σ ρ·V`;
//! * averaged SAR: [`SarVolume::averaged`](crate::post::sar::SarVolume::averaged)(crate::post::sar::SarVolume::averaged)
//!   over the real cells. openEMS's in-run method (`SIMPLE`) gives the
//!   same values where a valid averaging cube exists and different ones
//!   at the tissue surface;
//! * HDF5 (`FileType` 1): root `@mass` (not raw), `@proc_time`; `/Mesh`
//!   the cell centres; `/FieldData/FD` with `@frequency`,
//!   `@valid_cubes`, `@used_cubes`, `@unused_cubes`, `@air_cubes` (u64)
//!   and `f%d` f32 `[nx, ny, nz]` (`@d_order` "XYZ", `@frequency` f32,
//!   `@maxSAR` f32, `@maxSAR_idx` u32 \[3\], `@power` f32). Raw dumps
//!   instead hold `/CellData/{Conductivity, Density, Volume}` (f32
//!   `[nx, ny, nz]`), `/CellWidth/{x, y, z}` (metres, `@mesh_scaling` 1)
//!   and the complex E `f%d` `[3, nx, ny, nz]`;
//! * VTK: one `<name>_f=<f, 6 decimals>.vtr` per frequency, scalar point
//!   array `SAR-local`, `SAR_1g`, `SAR_10g` (raw: `SAR_raw`, the local
//!   SAR) on the cell centres.
//!
//! # HDF5 layout (`FileType` 1, feature `hdf5`)
//!
//! ```text
//! /                       @openEMS_HDF5_version f64 0.3, @dump_type i32,
//!                         @dual_mesh u8 (, @legacy_fmt u8 1)
//! /Mesh                   @mesh_scaling f64 (drawing unit), @mesh_type i32 (0 Cartesian, 1 cylindrical)
//! /Mesh/x, y, z           f64 [n] sample coordinates in metres
//!                         (cylindrical: rho, alpha [rad], z)
//! /FieldData/TD/%08d      f32 [3, nx, ny, nz], @d_order "NXYZ", @time f32 [1]
//! /FieldData/FD           @frequency f64 [nf]
//! /FieldData/FD/f%d       {r: f32, i: f32} [3, nx, ny, nz], @d_order, @frequency f64
//! ```
//!
//! The dataset of a step is named after `n` (8 digits); `time` is `n·dt`
//! for E, J, D and rot H, `(n + ½)·dt` for H and B. The **legacy** layout
//! (openEMS `--legacyHDF5Dumps`, [`DumpFileOptions::legacy_hdf5`]) stores
//! `[3, nz, ny, nx]` with `d_order` `"NZYX"`, a root attribute
//! `legacy_fmt` = 1, and frequency-domain data as two f32 datasets
//! `f%d_real`, `f%d_imag`. Cylindrical dumps always hold the native
//! `(ρ, α, z)` components.
//!
//! # VTK layout (`FileType` 0)
//!
//! VTK XML, little-endian, zlib-compressed binary arrays (32 KiB blocks,
//! `UInt32` headers): a `RectilinearGrid` (`.vtr`) with the sample
//! coordinates in metres, or for cylindrical grids a `StructuredGrid`
//! (`.vts`) with Cartesian `f32` points and Cartesian field components
//! (native `(ρ, α, z)` with [`DumpFileOptions::native_field_dumps`]). One
//! `f32` 3-vector point array named after the quantity, with
//! `RangeMin`/`RangeMax` and an `L2_NORM_RANGE` information key (the
//! range of the vector norms). Files:
//!
//! * time domain: `<name>_%010d.vtr` per step;
//! * frequency domain, per frequency `f`: `<name>_f=<f>_abs.vtr` (`|F|`
//!   per component), `<name>_f=<f>_arg.vtr` (phase, rad) and 21 phase
//!   snapshots `<name>_f=<f>_p=%03d.vtr` of `Re(F·e^{jφ})`,
//!   `φ = k·360°/21`.
//!
//! Without the `hdf5-deflate` feature the zlib streams use stored
//! (uncompressed) deflate blocks, which every zlib reader accepts.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::compile::sar::SarKind;
use crate::engine::{DumpQuantity, FieldSampler, FreqDump, Snapshot};
use crate::io::base64;
use crate::post::Complex64;
use crate::{Axis, DumpData, DumpResult, Error, Result, SarResult};

// ---------------------------------------------------------------------
// Sampling rules.

/// Relative tolerance of the line comparisons of [`sample_indices`].
const SNAP_TOL: f64 = 1e-9;

/// The openEMS sample indices along one axis of a dump box; see the
/// [module docs](self#sample-points-dumpmode).
///
/// `lines` are the candidate lines (strictly increasing; primary lines,
/// or the dual lines including the virtual last one for cell
/// interpolation), `domain` the extent of the mesh, `lo`/`hi` the box
/// corners (any order), `stride` the `SubSampling` (≥ 1) and `opt_res`
/// the `OptResolution` in the units of `lines`. `None` if the box lies
/// entirely outside `domain`.
pub fn sample_indices(
    lines: &[f64],
    domain: (f64, f64),
    lo: f64,
    hi: f64,
    stride: usize,
    opt_res: Option<f64>,
) -> Option<Vec<usize>> {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let n = lines.len();
    let scale = (domain.1 - domain.0).abs().max(f64::MIN_POSITIVE);
    let tol = SNAP_TOL * scale;
    if n == 0 || hi < domain.0 - tol || lo > domain.1 + tol {
        return None;
    }
    let nearest = |v: f64| -> usize {
        let i = lines.partition_point(|&x| x < v);
        if i == 0 {
            0
        } else if i == n {
            n - 1
        } else if v - lines[i - 1] <= lines[i] - v {
            i - 1
        } else {
            i
        }
    };
    let (a, b) = if nearest(lo) == nearest(hi) {
        let i = nearest(lo);
        (i, i)
    } else {
        // Last line ≤ lo, first line ≥ hi.
        let a = lines.partition_point(|&x| x <= lo + tol).saturating_sub(1);
        let b = lines.partition_point(|&x| x < hi - tol).min(n - 1);
        (a, b)
    };
    let mut out = Vec::new();
    match opt_res.filter(|r| r.is_finite() && *r > 0.0) {
        Some(r) => {
            let mut c = a;
            out.push(c);
            while c < b {
                let mut next = c + 1;
                while next < b && lines[next + 1] - lines[c] < r {
                    next += 1;
                }
                c = next;
                out.push(c);
            }
        }
        None => out.extend((a..=b).step_by(stride.max(1))),
    }
    Some(out)
}

/// The Nyquist step count `⌊1/(2·f_max·dt)⌋` (at least 1).
pub fn nyquist_steps(f_max: f64, dt: f64) -> u64 {
    let k = (1.0 / (2.0 * f_max * dt)).floor();
    if k.is_finite() && k >= 1.0 {
        k as u64
    } else {
        1
    }
}

/// The step interval of time-domain dumps: `⌊N/OverSampling⌋` (at least
/// 1) for the Nyquist step count `N`.
pub fn time_domain_interval(nyquist: u64, oversampling: u32) -> u64 {
    (nyquist / u64::from(oversampling.max(1))).max(1)
}

/// The step interval of frequency-domain dumps: `min(N, ⌊1/(2·f·dt)⌋)`
/// for the Nyquist step count `N` and the highest dump frequency `f`.
pub fn frequency_domain_interval(nyquist: u64, freqs: &[f64], dt: f64) -> u64 {
    let f = freqs.iter().copied().fold(0.0f64, f64::max);
    if f > 0.0 {
        nyquist.min(nyquist_steps(f, dt)).max(1)
    } else {
        nyquist.max(1)
    }
}

/// The steps `first ..= last` recorded by a dump with `StartTime` /
/// `StopTime` (seconds; `stop ≤ 0`: no limit): `(n + 1)·dt ≥ start` and
/// `(n − 1)·dt < stop`.
pub fn step_window(start: f64, stop: f64, dt: f64) -> (u64, Option<u64>) {
    let first = if start > 0.0 {
        // Smallest n with (n + 1)·dt ≥ start.
        let mut n = ((start / dt).ceil() - 1.0).max(0.0) as u64;
        while n > 0 && (n as f64) * dt >= start {
            n -= 1;
        }
        while ((n + 1) as f64) * dt < start {
            n += 1;
        }
        n
    } else {
        0
    };
    let last = (stop > 0.0).then(|| {
        // Largest n with (n − 1)·dt < stop.
        let mut n = (stop / dt).ceil() as u64 + 1;
        while n > 0 && ((n - 1) as f64) * dt >= stop {
            n -= 1;
        }
        n
    });
    (first, last)
}

// ---------------------------------------------------------------------
// Names and options.

/// Options of the openEMS dump writers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DumpFileOptions {
    /// The legacy HDF5 layout (openEMS `--legacyHDF5Dumps`).
    pub legacy_hdf5: bool,
    /// Native `(ρ, α, z)` components in VTK files of cylindrical dumps
    /// (openEMS `--nativeFieldDumps`); HDF5 files always have them.
    pub native_field_dumps: bool,
}

impl DumpFileOptions {
    /// Default options: current HDF5 layout, Cartesian VTK components.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets [`legacy_hdf5`](Self::legacy_hdf5).
    #[must_use]
    pub fn legacy_hdf5(mut self, on: bool) -> Self {
        self.legacy_hdf5 = on;
        self
    }

    /// Sets [`native_field_dumps`](Self::native_field_dumps).
    #[must_use]
    pub fn native_field_dumps(mut self, on: bool) -> Self {
        self.native_field_dumps = on;
        self
    }
}

/// The VTK array name of a quantity (`"E-Field"`, …).
pub fn field_name(q: DumpQuantity) -> &'static str {
    match q {
        DumpQuantity::E => "E-Field",
        DumpQuantity::H => "H-Field",
        DumpQuantity::J => "J-Field",
        DumpQuantity::RotH => "RotH-Field",
        DumpQuantity::D => "D-Field",
        DumpQuantity::B => "B-Field",
    }
}

/// The HDF5 `dump_type` of a quantity (the time-domain `DumpType`).
pub fn dump_type_code(q: DumpQuantity) -> i32 {
    match q {
        DumpQuantity::E => 0,
        DumpQuantity::H => 1,
        DumpQuantity::J => 2,
        DumpQuantity::RotH => 3,
        DumpQuantity::D => 4,
        DumpQuantity::B => 5,
    }
}

/// Whether the samples sit on the dual mesh (HDF5 `dual_mesh`).
#[cfg(feature = "hdf5")]
fn dual_mesh(s: &FieldSampler) -> bool {
    match s.position() {
        crate::engine::DumpPosition::CellCenters => true,
        crate::engine::DumpPosition::Native => s.quantity().is_dual(),
        crate::engine::DumpPosition::Nodes => false,
    }
}

/// The time attribute of step `n`: `n·dt`, `(n + ½)·dt` for H and B.
#[cfg(feature = "hdf5")]
fn label_time(q: DumpQuantity, step: u64, dt: f64) -> f64 {
    if q.is_dual() {
        (step as f64 + 0.5) * dt
    } else {
        step as f64 * dt
    }
}

/// The frequency part of FD VTK file names (`f=<value>`): the integer
/// for whole numbers of hertz, six decimals otherwise.
fn freq_label(f: f64) -> String {
    if f.fract() == 0.0 && f.abs() < 1e18 {
        format!("{f:.0}")
    } else {
        format!("{f:.6}")
    }
}

/// The 21 phase angles of the FD phase snapshots: `(label, radians)`.
fn phase_steps() -> impl Iterator<Item = (u32, f64)> {
    (0..21).map(|k| {
        // Computed in this order, the truncated labels match openEMS
        // (119 and 239, not 120 and 240).
        let rad = f64::from(k) * 2.0 * core::f64::consts::PI / 21.0;
        let deg = rad * 180.0 / core::f64::consts::PI;
        (deg as u32, rad)
    })
}

// ---------------------------------------------------------------------
// Number formatting.

/// A double as VTK writes `RangeMin`/`RangeMax`: the shortest
/// round-trip digits, positional for `1e-6 ≤ |v| < 1e21`, exponential
/// (`1.5e-9`, `1e+21`) otherwise.
fn vtk_range(v: f64) -> String {
    let a = v.abs();
    if v == 0.0 {
        "0".into()
    } else if (1e-6..1e21).contains(&a) {
        format!("{v}")
    } else {
        let s = format!("{v:e}");
        match s.split_once('e') {
            Some((m, e)) if !e.starts_with('-') => format!("{m}e+{e}"),
            _ => s,
        }
    }
}

/// `printf("%.11g", v)`, as VTK writes information key values.
fn printf_g11(v: f64) -> String {
    const P: i32 = 11;
    if v == 0.0 {
        return "0".into();
    }
    if !v.is_finite() {
        return format!("{v}");
    }
    let e = format!("{:.*e}", (P - 1) as usize, v);
    let (mant, exp) = e.split_once('e').expect("exponential format");
    let x: i32 = exp.parse().expect("exponent");
    let trim = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if !(-4..P).contains(&x) {
        let sign = if x < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mant), x.abs())
    } else {
        trim(&format!("{:.*}", (P - 1 - x) as usize, v))
    }
}

// ---------------------------------------------------------------------
// VTK writer.

/// zlib (RFC 1950) stream of `data`: deflate-compressed with the
/// `hdf5-deflate` feature, stored blocks otherwise.
fn zlib(data: &[u8]) -> Vec<u8> {
    #[cfg(feature = "hdf5-deflate")]
    if let Ok(v) = hdf5rs::filters::deflate::zlib_compress(data, 6) {
        return v;
    }
    let mut out = Vec::with_capacity(data.len() + data.len() / 65535 * 5 + 11);
    out.extend_from_slice(&[0x78, 0x01]);
    let mut chunks = data.chunks(65535).peekable();
    if chunks.peek().is_none() {
        out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    while let Some(c) = chunks.next() {
        out.push(u8::from(chunks.peek().is_none()));
        let len = c.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(c);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

/// The VTK compressed binary encoding of `raw`: base64 of the `UInt32`
/// block header, then base64 of the compressed 32 KiB blocks.
fn vtk_binary(raw: &[u8]) -> String {
    const BLOCK: usize = 32768;
    let blocks: Vec<Vec<u8>> = raw.chunks(BLOCK).map(zlib).collect();
    let last = (raw.len() % BLOCK) as u32;
    let mut head = Vec::new();
    for v in [blocks.len() as u32, BLOCK as u32, last] {
        head.extend_from_slice(&v.to_le_bytes());
    }
    for b in &blocks {
        head.extend_from_slice(&(b.len() as u32).to_le_bytes());
    }
    let mut s = base64::encode(&head);
    s.push_str(&base64::encode(&blocks.concat()));
    s
}

/// One `<DataArray>` element.
fn data_array(out: &mut String, ty: &str, name: &str, comps: usize, raw: &[u8], range: (f64, f64)) {
    let nc = if comps > 1 {
        format!(" NumberOfComponents=\"{comps}\"")
    } else {
        String::new()
    };
    let _ = writeln!(
        out,
        "      <DataArray type=\"{ty}\" Name=\"{name}\"{nc} format=\"binary\" RangeMin=\"{}\" RangeMax=\"{}\">",
        vtk_range(range.0),
        vtk_range(range.1)
    );
    let _ = writeln!(out, "        {}", vtk_binary(raw));
    if comps > 1 {
        out.push_str(
            "        <InformationKey name=\"L2_NORM_RANGE\" location=\"vtkDataArray\" length=\"2\">\n",
        );
        for (i, v) in [range.0, range.1].into_iter().enumerate() {
            let _ = write!(
                out,
                "          <Value index=\"{i}\">\n            {}\n          </Value>\n",
                printf_g11(v)
            );
        }
        out.push_str("        </InformationKey>\n");
    }
    out.push_str("      </DataArray>\n");
}

/// Range of the norms of 3-vectors (f32 components, computed in f64).
fn norm_range(v: &[[f32; 3]]) -> (f64, f64) {
    let mut r = (f64::INFINITY, f64::NEG_INFINITY);
    for x in v {
        let n = x
            .iter()
            .map(|&c| f64::from(c) * f64::from(c))
            .sum::<f64>()
            .sqrt();
        r = (r.0.min(n), r.1.max(n));
    }
    if v.is_empty() { (0.0, 0.0) } else { r }
}

fn f32_bytes(v: &[[f32; 3]]) -> Vec<u8> {
    v.iter().flatten().flat_map(|x| x.to_le_bytes()).collect()
}

/// The grid of a VTK dump file.
struct VtkGrid<'a> {
    coords: [&'a [f64]; 3],
    cylindrical: bool,
}

impl VtkGrid<'_> {
    fn dims(&self) -> [usize; 3] {
        self.coords.map(<[f64]>::len)
    }

    /// Reorders z-fastest point data into VTK (x-fastest) order, rotating
    /// cylindrical components to Cartesian unless `native`.
    fn to_vtk(&self, data: &[[f64; 3]], native: bool) -> Vec<[f32; 3]> {
        let [nx, ny, nz] = self.dims();
        let mut out = Vec::with_capacity(data.len());
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let v = data[(i * ny + j) * nz + k];
                    let v = if self.cylindrical && !native {
                        let (s, c) = self.coords[1][j].sin_cos();
                        [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]]
                    } else {
                        v
                    };
                    out.push(v.map(|x| x as f32));
                }
            }
        }
        out
    }

    /// The complete VTK file text for the point array `name`.
    fn file(&self, name: &str, data: &[[f32; 3]]) -> String {
        let [nx, ny, nz] = self.dims();
        let ext = format!("0 {} 0 {} 0 {}", nx - 1, ny - 1, nz - 1);
        let ty = if self.cylindrical {
            "StructuredGrid"
        } else {
            "RectilinearGrid"
        };
        let mut s = String::new();
        s.push_str("<?xml version=\"1.0\"?>\n");
        let _ = writeln!(
            s,
            "<VTKFile type=\"{ty}\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">"
        );
        let _ = writeln!(s, "  <{ty} WholeExtent=\"{ext}\">");
        let _ = writeln!(s, "  <Piece Extent=\"{ext}\">");
        s.push_str("    <PointData>\n");
        data_array(
            &mut s,
            "Float32",
            name,
            3,
            &f32_bytes(data),
            norm_range(data),
        );
        s.push_str("    </PointData>\n    <CellData>\n    </CellData>\n");
        if self.cylindrical {
            let mut pts = Vec::with_capacity(nx * ny * nz);
            for &z in self.coords[2] {
                for &a in self.coords[1] {
                    for &r in self.coords[0] {
                        let (sn, cs) = a.sin_cos();
                        pts.push([(r * cs) as f32, (r * sn) as f32, z as f32]);
                    }
                }
            }
            s.push_str("    <Points>\n");
            data_array(
                &mut s,
                "Float32",
                "Points",
                3,
                &f32_bytes(&pts),
                norm_range(&pts),
            );
            s.push_str("    </Points>\n");
        } else {
            s.push_str("    <Coordinates>\n");
            for (a, c) in self.coords.iter().enumerate() {
                let raw: Vec<u8> = c.iter().flat_map(|x| x.to_le_bytes()).collect();
                let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                // openEMS writes the (meaningless) address of the array.
                let name = format!("Array 0x{:x}", 0x5555_5555_0000_u64 + 0x240 * a as u64);
                data_array(&mut s, "Float64", &name, 1, &raw, (lo, hi));
            }
            s.push_str("    </Coordinates>\n");
        }
        s.push_str("  </Piece>\n");
        let _ = writeln!(s, "  </{ty}>");
        s.push_str("</VTKFile>\n");
        s
    }
}

fn vtk_grid<'a>(s: &'a FieldSampler, cylindrical: bool) -> VtkGrid<'a> {
    VtkGrid {
        coords: [s.coords(Axis::X), s.coords(Axis::Y), s.coords(Axis::Z)],
        cylindrical,
    }
}

fn write_file(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
}

/// The time-domain VTK file text of one snapshot.
pub fn vtk_time_file(
    sampler: &FieldSampler,
    snapshot: &Snapshot,
    cylindrical: bool,
    opts: &DumpFileOptions,
) -> String {
    let g = vtk_grid(sampler, cylindrical);
    let d: Vec<[f64; 3]> = snapshot.data.iter().map(|v| v.map(f64::from)).collect();
    g.file(
        field_name(sampler.quantity()),
        &g.to_vtk(&d, opts.native_field_dumps),
    )
}

/// The VTK file extension of a dump (`vtr`, `vts` for cylindrical grids).
fn vtk_ext(cylindrical: bool) -> &'static str {
    if cylindrical { "vts" } else { "vtr" }
}

/// Writes the VTK files of a time-domain dump into `dir`
/// (`<name>_%010d.vtr`); returns the paths.
///
/// # Errors
/// I/O errors.
pub fn write_vtk_time(
    dir: &Path,
    name: &str,
    sampler: &FieldSampler,
    snapshots: &[Snapshot],
    cylindrical: bool,
    opts: &DumpFileOptions,
) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for s in snapshots {
        let p = dir.join(format!("{name}_{:010}.{}", s.step, vtk_ext(cylindrical)));
        write_file(&p, &vtk_time_file(sampler, s, cylindrical, opts))?;
        out.push(p);
    }
    Ok(out)
}

/// The openEMS-normalised values of frequency `k` of a frequency dump:
/// `2·`[`FreqDump::values`].
pub fn openems_freq_values(d: &FreqDump, k: usize) -> Vec<[Complex64; 3]> {
    d.values(k)
        .into_iter()
        .map(|v| v.map(|c| c * 2.0))
        .collect()
}

/// Writes the VTK files of a frequency-domain dump into `dir` (per
/// frequency: `_abs`, `_arg` and 21 `_p=%03d` files); returns the paths.
///
/// # Errors
/// I/O errors.
pub fn write_vtk_freq(
    dir: &Path,
    name: &str,
    d: &FreqDump,
    cylindrical: bool,
    opts: &DumpFileOptions,
) -> Result<Vec<PathBuf>> {
    let s = d.sampler();
    let g = vtk_grid(s, cylindrical);
    let fname = field_name(s.quantity());
    let ext = vtk_ext(cylindrical);
    let mut out = Vec::new();
    for (k, &f) in d.freqs().iter().enumerate() {
        let v = openems_freq_values(d, k);
        // Rotate first (linear), then split.
        let rot = |part: &dyn Fn(Complex64) -> f64| -> Vec<[f32; 3]> {
            let re: Vec<[f64; 3]> = v.iter().map(|c| c.map(|z| z.re)).collect();
            let im: Vec<[f64; 3]> = v.iter().map(|c| c.map(|z| z.im)).collect();
            let re = g.to_vtk(&re, opts.native_field_dumps);
            let im = g.to_vtk(&im, opts.native_field_dumps);
            re.iter()
                .zip(&im)
                .map(|(a, b)| {
                    [0, 1, 2].map(|c| part(Complex64::new(f64::from(a[c]), f64::from(b[c]))) as f32)
                })
                .collect()
        };
        let base = format!("{name}_f={}", freq_label(f));
        let mut put = |suffix: &str, data: Vec<[f32; 3]>| -> Result<()> {
            let p = dir.join(format!("{base}_{suffix}.{ext}"));
            write_file(&p, &g.file(fname, &data))?;
            out.push(p);
            Ok(())
        };
        put("abs", rot(&|z| z.norm()))?;
        put("arg", rot(&|z| z.arg()))?;
        for (label, phi) in phase_steps() {
            let e = Complex64::cis(phi);
            put(&format!("p={label:03}"), rot(&|z| (z * e).re))?;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// SAR dumps.

/// The data of an openEMS-compatible SAR dump: the cell-interpolated E
/// DFT ([`DumpPosition::CellCenters`](crate::engine::DumpPosition::CellCenters), openEMS sampling) at the cells
/// of the dump with their conductivity and density; see the
/// [module docs](self#sar-dumps).
#[derive(Debug, Clone)]
pub struct OpenEmsSar {
    pub(crate) fd: FreqDump,
    pub(crate) sigma: Vec<f64>,
    pub(crate) density: Vec<f64>,
    /// Cell widths per axis \[m\] (the virtual last cell: the width of the
    /// last real cell).
    pub(crate) widths: [Vec<f64>; 3],
    /// Boundaries \[m\] of the real cells per axis.
    pub(crate) lines: [Vec<f64>; 3],
}

/// One frequency of an [`OpenEmsSar`]: SAR per cell and the openEMS
/// statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct SarFrequency {
    /// SAR per cell \[W/kg\] (z fastest over the dump cells).
    pub sar: Vec<f64>,
    /// Absorbed power `Σ ½·κ·|E|²·V` \[W\] (spectral density: W·s²).
    pub power: f64,
    /// Largest SAR and its cell index.
    pub max: (f64, [usize; 3]),
    /// Cube counts `[valid, used, unused, air]`.
    pub cubes: [u64; 4],
}

impl OpenEmsSar {
    pub(crate) fn new(
        fd: FreqDump,
        sigma: Vec<f64>,
        density: Vec<f64>,
        grid: &crate::Grid,
        indices: &[Vec<usize>; 3],
    ) -> Self {
        let widths = Axis::ALL.map(|a| {
            let n = grid.nodes()[a.index()];
            indices[a.index()]
                .iter()
                .map(|&j| grid.delta(a, j.min(n - 2)))
                .collect()
        });
        let lines = Axis::ALL.map(|a| {
            let n = grid.nodes()[a.index()];
            let v = &indices[a.index()];
            let lo = v[0].min(n - 2);
            let hi = v.last().copied().unwrap_or(0).min(n - 2).max(lo);
            grid.lines(a)[lo..=hi + 1].to_vec()
        });
        Self {
            fd,
            sigma,
            density,
            widths,
            lines,
        }
    }

    /// The E DFT of the cells (values: [`openems_freq_values`]).
    pub fn field(&self) -> &FreqDump {
        &self.fd
    }

    /// Conductivity per cell \[S/m\].
    pub fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    /// Mass density per cell \[kg/m³\].
    pub fn density(&self) -> &[f64] {
        &self.density
    }

    fn dims(&self) -> [usize; 3] {
        self.fd.sampler().dims()
    }

    /// Volume per cell \[m³\].
    pub fn volumes(&self) -> Vec<f64> {
        let [nx, ny, nz] = self.dims();
        let w = &self.widths;
        let mut v = Vec::with_capacity(nx * ny * nz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    v.push(w[0][i] * w[1][j] * w[2][k]);
                }
            }
        }
        v
    }

    /// Tissue mass `Σ ρ·V` \[kg\] (with `ρ` and `V` rounded to `f32` like
    /// openEMS).
    pub fn mass(&self) -> f64 {
        self.density
            .iter()
            .zip(self.volumes())
            .map(|(&r, v)| f64::from(r as f32) * f64::from(v as f32))
            .sum()
    }

    /// Local SAR `κ·|E|²/(2ρ)` (`averaging = None`) or the SAR averaged
    /// over `averaging` kg with [`SarVolume::averaged`](crate::post::sar::SarVolume::averaged) at frequency
    /// index `k`.
    ///
    /// # Errors
    /// See [`SarVolume::averaged`](crate::post::sar::SarVolume::averaged).
    pub fn frequency(&self, k: usize, averaging: Option<f64>) -> Result<SarFrequency> {
        let e = openems_freq_values(&self.fd, k);
        let vol = self.volumes();
        let pd: Vec<f64> = e
            .iter()
            .zip(&self.sigma)
            .map(|(v, s)| 0.5 * s * v.iter().map(|c| c.norm_sqr()).sum::<f64>())
            .collect();
        let power = pd.iter().zip(&vol).map(|(p, v)| p * v).sum();
        let tissue = self.density.iter().filter(|&&r| r > 0.0).count() as u64;
        let all = self.density.len() as u64;
        let (sar, cubes) = match averaging {
            None => (
                pd.iter()
                    .zip(&self.density)
                    .map(|(p, &r)| if r > 0.0 { p / r } else { 0.0 })
                    .collect::<Vec<_>>(),
                [tissue, 0, 0, all - tissue],
            ),
            Some(m) => self.averaged(&pd, m)?,
        };
        let dims = self.dims();
        // The last of equal (f32) maxima, like openEMS.
        let mut max = (0.0, [0; 3]);
        for (i, &v) in sar.iter().enumerate() {
            if v > 0.0 && v as f32 >= max.0 as f32 {
                max = (
                    v,
                    [
                        i / (dims[1] * dims[2]),
                        (i / dims[2]) % dims[1],
                        i % dims[2],
                    ],
                );
            }
        }
        Ok(SarFrequency {
            sar,
            power,
            max,
            cubes,
        })
    }

    /// Averaged SAR over the real cells, mapped back to the dump cells.
    fn averaged(&self, pd: &[f64], mass: f64) -> Result<(Vec<f64>, [u64; 4])> {
        use crate::post::sar::{SarVolume, VoxelClass};
        let dims = self.dims();
        let c = self.lines.clone().map(|l| l.len() - 1);
        let idx = |p: [usize; 3]| (p[0] * dims[1] + p[1]) * dims[2] + p[2];
        let mut power = Vec::with_capacity(c.iter().product());
        let mut density = Vec::with_capacity(power.capacity());
        for i in 0..c[0] {
            for j in 0..c[1] {
                for k in 0..c[2] {
                    power.push(pd[idx([i, j, k])]);
                    density.push(self.density[idx([i, j, k])]);
                }
            }
        }
        let v = SarVolume::new(self.lines.clone(), power, density)?;
        let f = v.averaged(mass)?;
        let mut out = vec![0.0; self.density.len()];
        let mut cubes = [0u64; 4];
        for i in 0..c[0] {
            for j in 0..c[1] {
                for k in 0..c[2] {
                    let m = (i * c[1] + j) * c[2] + k;
                    out[idx([i, j, k])] = f.values()[m];
                    // openEMS (method SIMPLE) counts every averaged voxel
                    // as valid.
                    match f.classes()[m] {
                        VoxelClass::Centered | VoxelClass::Covered | VoxelClass::Boundary => {
                            cubes[0] += 1
                        }
                        VoxelClass::Unresolved => cubes[2] += 1,
                        _ => cubes[3] += 1,
                    }
                }
            }
        }
        cubes[3] += (self.density.len() - c.iter().product::<usize>()) as u64;
        Ok((out, cubes))
    }
}

/// The VTK array name of a SAR dump (`SAR-local`, `SAR_1g`, `SAR_10g`,
/// `SAR_raw`).
fn sar_name(kind: SarKind) -> String {
    match kind {
        SarKind::Local => "SAR-local".into(),
        SarKind::Raw => "SAR_raw".into(),
        SarKind::Averaged(m) => format!("SAR_{}g", (m * 1e3).round()),
    }
}

/// The averaging mass of a SAR kind \[kg\].
fn sar_mass(kind: SarKind) -> Option<f64> {
    match kind {
        SarKind::Averaged(m) => Some(m),
        _ => None,
    }
}

/// Writes the VTK files of a SAR dump (`<name>_f=<f, %f>.vtr` per
/// frequency, point array of the local or averaged SAR on the cell
/// centres; raw dumps: the local SAR).
///
/// # Errors
/// I/O errors and those of [`OpenEmsSar::frequency`].
pub fn write_sar_vtk(dir: &Path, sar: &SarResult) -> Result<Vec<PathBuf>> {
    let Some(d) = sar.openems() else {
        return Err(Error::Unsupported(format!(
            "SAR dump {:?}: no openEMS data (cylindrical grid)",
            sar.name()
        )));
    };
    let s = d.fd.sampler();
    let coords = [s.coords(Axis::X), s.coords(Axis::Y), s.coords(Axis::Z)];
    let [nx, ny, nz] = s.dims();
    let mut out = Vec::new();
    for (k, &f) in sar.frequencies().iter().enumerate() {
        let v = d.frequency(k, sar_mass(sar.kind()))?;
        let mut data = Vec::with_capacity(v.sar.len());
        for kk in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    data.push(v.sar[(i * ny + j) * nz + kk] as f32);
                }
            }
        }
        let p = dir.join(format!("{}_f={f:.6}.vtr", sar.name()));
        let ext = format!("0 {} 0 {} 0 {}", nx - 1, ny - 1, nz - 1);
        let mut t = String::new();
        t.push_str("<?xml version=\"1.0\"?>\n<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">\n");
        let _ = writeln!(t, "  <RectilinearGrid WholeExtent=\"{ext}\">");
        let _ = writeln!(t, "  <Piece Extent=\"{ext}\">");
        t.push_str("    <PointData>\n");
        let raw: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        let lo = data.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        data_array(
            &mut t,
            "Float32",
            &sar_name(sar.kind()),
            1,
            &raw,
            (f64::from(lo), f64::from(hi)),
        );
        t.push_str("    </PointData>\n    <CellData>\n    </CellData>\n    <Coordinates>\n");
        for (a, c) in coords.iter().enumerate() {
            let raw: Vec<u8> = c.iter().flat_map(|x| x.to_le_bytes()).collect();
            let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let name = format!("Array 0x{:x}", 0x5555_5555_0000_u64 + 0x300 * a as u64);
            data_array(&mut t, "Float64", &name, 1, &raw, (lo, hi));
        }
        t.push_str("    </Coordinates>\n  </Piece>\n  </RectilinearGrid>\n</VTKFile>\n");
        write_file(&p, &t)?;
        out.push(p);
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// HDF5 writer.

#[cfg(feature = "hdf5")]
mod h5 {
    use hdf5rs::Datatype;
    use hdf5rs::format::Sizes;
    use hdf5rs::write::{EncodeContext, FileWriter, H5Write, WriteAt, WriteOptions};

    use super::*;

    /// The openEMS complex type `{r: f32, i: f32}`.
    struct C32([f32; 2]);

    impl H5Write for C32 {
        fn datatype(sizes: Sizes) -> Datatype {
            Datatype::packed_compound(vec![
                ("r".into(), f32::datatype(sizes)),
                ("i".into(), f32::datatype(sizes)),
            ])
            .expect("two f32 members form a valid compound")
        }

        fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> hdf5rs::Result<()> {
            self.0[0].encode(out, ctx)?;
            self.0[1].encode(out, ctx)
        }
    }

    fn h5err(e: hdf5rs::Error) -> Error {
        Error::Io(format!("HDF5: {e}"))
    }

    /// The header of a dump file: root attributes and `/Mesh`.
    pub(super) struct Header<'a> {
        pub sampler: &'a FieldSampler,
        pub dump_type: i32,
        pub dual_mesh: bool,
        pub mesh_scaling: f64,
        pub cylindrical: bool,
        pub legacy: bool,
    }

    pub(super) fn header<S: WriteAt>(f: &mut FileWriter<S>, h: &Header<'_>) -> hdf5rs::Result<()> {
        let mut root = f.root();
        root.set_attr("openEMS_HDF5_version", &0.3f64)?;
        root.set_attr("dump_type", &h.dump_type)?;
        root.set_attr("dual_mesh", &u8::from(h.dual_mesh))?;
        if h.legacy {
            root.set_attr("legacy_fmt", &1u8)?;
        }
        let mut mesh = root.create_group("Mesh")?;
        mesh.set_attr("mesh_scaling", &h.mesh_scaling)?;
        mesh.set_attr("mesh_type", &i32::from(h.cylindrical))?;
        let names = if h.cylindrical {
            ["rho", "alpha", "z"]
        } else {
            ["x", "y", "z"]
        };
        for (a, n) in Axis::ALL.into_iter().zip(names) {
            let c = h.sampler.coords(a);
            mesh.new_dataset::<f64>(n)
                .shape(&[c.len() as u64])
                .write(c)?;
        }
        Ok(())
    }

    /// Component-major `[3, nx, ny, nz]` (or legacy `[3, nz, ny, nx]`)
    /// order of z-fastest point data.
    pub(super) fn reorder<T: Copy>(dims: [usize; 3], data: &[[T; 3]], legacy: bool) -> Vec<T> {
        let [nx, ny, nz] = dims;
        let mut out = Vec::with_capacity(data.len() * 3);
        for c in 0..3 {
            if legacy {
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            out.push(data[(i * ny + j) * nz + k][c]);
                        }
                    }
                }
            } else {
                out.extend(data.iter().map(|v| v[c]));
            }
        }
        out
    }

    pub(super) fn shape(dims: [usize; 3], legacy: bool) -> [u64; 4] {
        let d = dims.map(|n| n as u64);
        if legacy {
            [3, d[2], d[1], d[0]]
        } else {
            [3, d[0], d[1], d[2]]
        }
    }

    pub(super) fn d_order(legacy: bool) -> &'static str {
        if legacy { "NZYX" } else { "NXYZ" }
    }

    pub(super) fn options() -> WriteOptions {
        WriteOptions {
            superblock_version: 0,
            ..WriteOptions::default()
        }
    }

    pub(super) fn time_file<S: WriteAt>(
        f: &mut FileWriter<S>,
        h: &Header<'_>,
        snapshots: &[Snapshot],
        dt: f64,
    ) -> hdf5rs::Result<()> {
        header(f, h)?;
        let dims = h.sampler.dims();
        let q = h.sampler.quantity();
        let mut root = f.root();
        let mut fd = root.create_group("FieldData")?;
        let mut td = fd.create_group("TD")?;
        for s in snapshots {
            let data = reorder(dims, &s.data, h.legacy);
            let mut ds = td
                .new_dataset::<f32>(&format!("{:08}", s.step))
                .shape(&shape(dims, h.legacy))
                .write(&data)?;
            ds.set_attr("d_order", d_order(h.legacy))?;
            ds.set_attr("time", &[label_time(q, s.step, dt) as f32])?;
        }
        Ok(())
    }

    pub(super) fn freq_file<S: WriteAt>(
        f: &mut FileWriter<S>,
        h: &Header<'_>,
        d: &FreqDump,
    ) -> hdf5rs::Result<()> {
        header(f, h)?;
        let dims = h.sampler.dims();
        let mut root = f.root();
        let mut fdg = root.create_group("FieldData")?;
        let mut g = fdg.create_group("FD")?;
        g.set_attr("frequency", d.freqs())?;
        for (k, &freq) in d.freqs().iter().enumerate() {
            let v = openems_freq_values(d, k);
            if h.legacy {
                for (suffix, part) in [("real", 0), ("imag", 1)] {
                    let p: Vec<[f32; 3]> = v
                        .iter()
                        .map(|c| c.map(|z| if part == 0 { z.re } else { z.im } as f32))
                        .collect();
                    let data = reorder(dims, &p, true);
                    let mut ds = g
                        .new_dataset::<f32>(&format!("f{k}_{suffix}"))
                        .shape(&shape(dims, true))
                        .write(&data)?;
                    ds.set_attr("d_order", d_order(true))?;
                    ds.set_attr("frequency", &freq)?;
                }
            } else {
                let mut data = Vec::with_capacity(v.len() * 3);
                for c in 0..3 {
                    data.extend(v.iter().map(|x| C32([x[c].re as f32, x[c].im as f32])));
                }
                let mut ds = g
                    .new_dataset::<C32>(&format!("f{k}"))
                    .shape(&shape(dims, false))
                    .write(&data)?;
                ds.set_attr("d_order", d_order(false))?;
                ds.set_attr("frequency", &freq)?;
            }
        }
        Ok(())
    }

    /// A SAR dump file (see the [module docs](super#sar-dumps)).
    pub(super) fn sar_file<S: WriteAt>(
        f: &mut FileWriter<S>,
        sar: &SarResult,
        d: &OpenEmsSar,
        mesh_scaling: f64,
    ) -> Result<()> {
        let e = |e: hdf5rs::Error| h5err(e);
        let s = d.fd.sampler();
        let kind = sar.kind();
        let code = match kind {
            SarKind::Local => 20,
            SarKind::Averaged(m) if m < 5e-3 => 21,
            SarKind::Averaged(_) => 22,
            SarKind::Raw => 29,
        };
        let h = Header {
            sampler: s,
            dump_type: code,
            dual_mesh: true,
            mesh_scaling,
            cylindrical: false,
            legacy: false,
        };
        header(f, &h).map_err(e)?;
        let dims = s.dims();
        let shape3 = dims.map(|n| n as u64);
        let mut root = f.root();
        if kind == SarKind::Raw {
            let mut cd = root.create_group("CellData").map_err(e)?;
            let vol = d.volumes();
            for (name, v) in [
                ("Conductivity", &d.sigma),
                ("Density", &d.density),
                ("Volume", &vol),
            ] {
                let data: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                cd.new_dataset::<f32>(name)
                    .shape(&shape3)
                    .write(&data)
                    .map_err(e)?
                    .set_attr("d_order", "XYZ")
                    .map_err(e)?;
            }
            let mut cw = root.create_group("CellWidth").map_err(e)?;
            cw.set_attr("mesh_scaling", &1.0f64).map_err(e)?;
            cw.set_attr("mesh_type", &0i32).map_err(e)?;
            for (a, n) in ["x", "y", "z"].into_iter().enumerate() {
                cw.new_dataset::<f64>(n)
                    .shape(&[d.widths[a].len() as u64])
                    .write(&d.widths[a])
                    .map_err(e)?;
            }
            let mut fdg = root.create_group("FieldData").map_err(e)?;
            let mut g = fdg.create_group("FD").map_err(e)?;
            g.set_attr("frequency", sar.frequencies()).map_err(e)?;
            for k in 0..sar.frequencies().len() {
                let v = openems_freq_values(&d.fd, k);
                let mut data = Vec::with_capacity(v.len() * 3);
                for c in 0..3 {
                    data.extend(v.iter().map(|x| C32([x[c].re as f32, x[c].im as f32])));
                }
                g.new_dataset::<C32>(&format!("f{k}"))
                    .shape(&shape(dims, false))
                    .write(&data)
                    .map_err(e)?
                    .set_attr("d_order", "NXYZ")
                    .map_err(e)?;
            }
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let per: Vec<SarFrequency> = (0..sar.frequencies().len())
            .map(|k| d.frequency(k, sar_mass(kind)))
            .collect::<Result<_>>()?;
        root.set_attr("mass", &d.mass()).map_err(e)?;
        root.set_attr("proc_time", &t0.elapsed().as_secs_f64())
            .map_err(e)?;
        let mut fdg = root.create_group("FieldData").map_err(e)?;
        let mut g = fdg.create_group("FD").map_err(e)?;
        g.set_attr("frequency", sar.frequencies()).map_err(e)?;
        let cubes = per.first().map_or([0; 4], |p| p.cubes);
        for (n, v) in ["valid_cubes", "used_cubes", "unused_cubes", "air_cubes"]
            .into_iter()
            .zip(cubes)
        {
            g.set_attr(n, &v).map_err(e)?;
        }
        for (k, (p, &freq)) in per.iter().zip(sar.frequencies()).enumerate() {
            let data: Vec<f32> = p.sar.iter().map(|&x| x as f32).collect();
            let mut ds = g
                .new_dataset::<f32>(&format!("f{k}"))
                .shape(&shape3)
                .write(&data)
                .map_err(e)?;
            ds.set_attr("d_order", "XYZ").map_err(e)?;
            ds.set_attr("frequency", &(freq as f32)).map_err(e)?;
            ds.set_attr("maxSAR", &(p.max.0 as f32)).map_err(e)?;
            ds.set_attr("maxSAR_idx", &p.max.1.map(|i| i as u32))
                .map_err(e)?;
            ds.set_attr("power", &(p.power as f32)).map_err(e)?;
        }
        Ok(())
    }

    pub(super) fn bytes(
        fill: impl FnOnce(&mut FileWriter<Vec<u8>>) -> hdf5rs::Result<()>,
    ) -> Result<Vec<u8>> {
        let mut f = FileWriter::new(Vec::new(), options()).map_err(h5err)?;
        fill(&mut f).map_err(h5err)?;
        f.finish().map_err(h5err)
    }
}

/// The HDF5 file of a field dump, in memory (feature `hdf5`; see the
/// [module docs](self#hdf5-layout-filetype-1-feature-hdf5)).
///
/// # Errors
/// Encoding errors ([`Error::Io`]).
#[cfg(feature = "hdf5")]
pub fn hdf5_bytes(dump: &DumpResult, opts: &DumpFileOptions) -> Result<Vec<u8>> {
    let s = dump.sampler();
    let header = h5::Header {
        sampler: s,
        dump_type: dump_type_code(s.quantity()),
        dual_mesh: dual_mesh(s),
        mesh_scaling: dump.delta_unit(),
        cylindrical: dump.is_cylindrical(),
        legacy: opts.legacy_hdf5,
    };
    match dump.data() {
        DumpData::Time { snapshots, .. } => {
            h5::bytes(|f| h5::time_file(f, &header, snapshots, dump.dt()))
        }
        DumpData::Freq(d) => h5::bytes(|f| h5::freq_file(f, &header, d)),
    }
}

/// The HDF5 file of a SAR dump, in memory (feature `hdf5`; see the
/// [module docs](self#sar-dumps)).
///
/// # Errors
/// Encoding errors, a SAR dump without openEMS data (cylindrical grid),
/// averaging errors.
#[cfg(feature = "hdf5")]
pub fn sar_hdf5_bytes(sar: &SarResult) -> Result<Vec<u8>> {
    let Some(d) = sar.openems() else {
        return Err(Error::Unsupported(format!(
            "SAR dump {:?}: no openEMS data (cylindrical grid)",
            sar.name()
        )));
    };
    let mut f = hdf5rs::write::FileWriter::new(Vec::new(), h5::options())
        .map_err(|e| Error::Io(format!("HDF5: {e}")))?;
    h5::sar_file(&mut f, sar, d, sar.delta_unit)?;
    f.finish().map_err(|e| Error::Io(format!("HDF5: {e}")))
}

/// Writes a SAR dump into `dir` like openEMS: `<name>.h5` (feature
/// `hdf5`) or the VTK files of [`write_sar_vtk`].
///
/// # Errors
/// See `sar_hdf5_bytes` (feature `hdf5`) and [`write_sar_vtk`].
pub fn write_sar(dir: impl AsRef<Path>, sar: &SarResult) -> Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).map_err(|e| Error::Io(format!("{}: {e}", dir.display())))?;
    match sar.file_type() {
        csx::properties::DumpFileType::Hdf5 => {
            #[cfg(feature = "hdf5")]
            {
                let p = dir.join(format!("{}.h5", sar.name()));
                std::fs::write(&p, sar_hdf5_bytes(sar)?)
                    .map_err(|e| Error::Io(format!("{}: {e}", p.display())))?;
                Ok(vec![p])
            }
            #[cfg(not(feature = "hdf5"))]
            Err(Error::Unsupported(format!(
                "SAR dump {:?}: HDF5 output needs the `hdf5` feature",
                sar.name()
            )))
        }
        _ => write_sar_vtk(dir, sar),
    }
}

/// Writes `dump` into `dir` (created if needed) the way openEMS does for
/// its `FileType`: VTK files `<name>_…` or one HDF5 file `<name>.h5`
/// (feature `hdf5`; without it an HDF5 dump is an error). Returns the
/// written paths.
///
/// # Errors
/// I/O and encoding errors; [`Error::Unsupported`] for an HDF5 dump
/// without the `hdf5` feature.
pub fn write_dump(
    dir: impl AsRef<Path>,
    dump: &DumpResult,
    opts: &DumpFileOptions,
) -> Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).map_err(|e| Error::Io(format!("{}: {e}", dir.display())))?;
    match dump.file_type() {
        csx::properties::DumpFileType::Hdf5 => {
            #[cfg(feature = "hdf5")]
            {
                let p = dir.join(format!("{}.h5", dump.name()));
                std::fs::write(&p, hdf5_bytes(dump, opts)?)
                    .map_err(|e| Error::Io(format!("{}: {e}", p.display())))?;
                Ok(vec![p])
            }
            #[cfg(not(feature = "hdf5"))]
            {
                let _ = opts;
                Err(Error::Unsupported(format!(
                    "dump {:?}: HDF5 output needs the `hdf5` feature",
                    dump.name()
                )))
            }
        }
        _ => match dump.data() {
            DumpData::Time { sampler, snapshots } => write_vtk_time(
                dir,
                dump.name(),
                sampler,
                snapshots,
                dump.is_cylindrical(),
                opts,
            ),
            DumpData::Freq(d) => write_vtk_freq(dir, dump.name(), d, dump.is_cylindrical(), opts),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_formats() {
        assert_eq!(vtk_range(0.0), "0");
        assert_eq!(vtk_range(0.03), "0.03");
        assert_eq!(vtk_range(0.018000000000000002), "0.018000000000000002");
        assert_eq!(vtk_range(1.6611293451873621e-9), "1.6611293451873621e-9");
        assert_eq!(vtk_range(3.376882650747664e-6), "0.000003376882650747664");
        assert_eq!(printf_g11(0.5335787534713745), "0.53357875347");
        assert_eq!(printf_g11(1.0347103761906146e-14), "1.0347103762e-14");
        assert_eq!(printf_g11(7.704954341529852e-7), "7.7049543415e-07");
        assert_eq!(printf_g11(0.0011686112266033888), "0.0011686112266");
        assert_eq!(printf_g11(0.0), "0");
    }

    #[test]
    fn phase_labels() {
        let l: Vec<u32> = phase_steps().map(|p| p.0).collect();
        assert_eq!(
            l,
            [
                0, 17, 34, 51, 68, 85, 102, 119, 137, 154, 171, 188, 205, 222, 239, 257, 274, 291,
                308, 325, 342
            ]
        );
    }

    #[test]
    fn snapping() {
        let x = [0.0, 4.0, 9.0, 15.0, 20.0, 26.0, 30.0];
        let d = (0.0, 30.0);
        let s = |a, b| sample_indices(&x, d, a, b, 1, None).unwrap();
        assert_eq!(s(2.0, 28.0), [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(s(28.0, 2.0), [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(s(10.0, 24.0), [2, 3, 4, 5]);
        assert_eq!(s(8.0, 16.0), [1, 2, 3, 4]);
        assert_eq!(s(10.5, 11.0), [2]);
        assert_eq!(s(12.5, 13.0), [3]);
        assert_eq!(s(6.4, 6.6), [1, 2]);
        assert_eq!(s(-5.0, 40.0), [0, 1, 2, 3, 4, 5, 6]);
        assert!(
            sample_indices(
                &[0.0, 3.0, 7.0, 12.0, 18.0],
                (0.0, 18.0),
                19.0,
                19.0,
                1,
                None
            )
            .is_none()
        );
        let z = [0.0, 3.0, 7.0, 12.0, 18.0];
        assert_eq!(
            sample_indices(&z, (0.0, 18.0), 5.0, 5.0, 1, None).unwrap(),
            [1]
        );
        assert_eq!(
            sample_indices(&z, (0.0, 18.0), 9.5, 9.5, 1, None).unwrap(),
            [2]
        );
        // Cell interpolation: dual lines incl. the virtual one.
        let xd = [2.0, 6.5, 12.0, 17.5, 23.0, 28.0, 32.0];
        let c = |a, b| sample_indices(&xd, d, a, b, 1, None).unwrap();
        assert_eq!(c(10.0, 24.0), [1, 2, 3, 4, 5]);
        assert_eq!(c(2.0, 28.0), [0, 1, 2, 3, 4, 5]);
        assert_eq!(c(1.9, 28.1), [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(c(0.5, 3.5), [0]);
        let zd = [1.5, 5.0, 9.5, 15.0, 21.0];
        assert_eq!(
            sample_indices(&zd, (0.0, 18.0), 18.0, 18.0, 1, None).unwrap(),
            [3]
        );
        assert_eq!(
            sample_indices(&zd, (0.0, 18.0), 1.6, 1.6, 1, None).unwrap(),
            [0]
        );
    }

    #[test]
    fn opt_resolution() {
        let x: Vec<f64> = (0..=20).map(f64::from).collect();
        let d = (0.0, 20.0);
        let r = |res| sample_indices(&x, d, 0.0, 20.0, 1, Some(res)).unwrap();
        assert_eq!(r(3.0), (0..=20).step_by(2).collect::<Vec<_>>());
        assert_eq!(r(2.5), (0..=20).step_by(2).collect::<Vec<_>>());
        assert_eq!(r(4.0), [0, 3, 6, 9, 12, 15, 18, 20]);
        assert_eq!(r(5.0), [0, 4, 8, 12, 16, 20]);
        let y = [0.0, 0.5, 1.5, 3.0, 5.0, 7.5, 10.5, 14.0, 18.0, 20.0];
        let ry = |res| sample_indices(&y, (0.0, 20.0), 0.0, 20.0, 1, Some(res)).unwrap();
        assert_eq!(ry(3.0), [0, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(ry(4.0), [0, 3, 4, 5, 6, 7, 8, 9]);
        let z = [0.0, 2.0, 4.0, 6.0];
        assert_eq!(
            sample_indices(&z, (0.0, 6.0), 0.0, 6.0, 1, Some(10.0)).unwrap(),
            [0, 3]
        );
        assert_eq!(
            sample_indices(&x, d, 0.0, 20.0, 3, None).unwrap(),
            [0, 3, 6, 9, 12, 15, 18]
        );
    }

    #[test]
    fn windows() {
        let dt = 6.5e-12;
        assert_eq!(step_window(5e-11, 1.2e-10, dt), (7, Some(19)));
        assert_eq!(step_window(4.875e-11, 1.17e-10, dt).0, 7);
        assert_eq!(step_window(0.0, 0.0, dt), (0, None));
        assert_eq!(step_window(7.1 * dt, 0.0, dt).0, 7);
        assert_eq!(step_window(6.9 * dt, 0.0, dt).0, 6);
        assert_eq!(step_window(0.0, 18.1 * dt, dt).1, Some(19));
        assert_eq!(step_window(0.0, 17.9 * dt, dt).1, Some(18));
        assert_eq!(step_window(0.0, 18.9 * dt, dt).1, Some(19));
        assert_eq!(step_window(0.0, 19.1 * dt, dt).1, Some(20));
        assert_eq!(nyquist_steps(5e9, 7.00629e-12), 14);
        assert_eq!(time_domain_interval(14, 4), 3);
        assert_eq!(frequency_domain_interval(25, &[1e9, 2.5e9], 6.5e-12), 25);
        assert_eq!(frequency_domain_interval(25, &[4e9], 6.5e-12), 19);
    }

    #[test]
    fn stored_zlib_round_trip_shape() {
        let z = zlib(b"hello");
        assert_eq!(z[0], 0x78);
        assert!(z.len() >= 5 + 6);
    }
}
