//! The openEMS `sar_calc` tool: mass-averaged SAR from a raw SAR dump
//! (feature `hdf5`).
//!
//! openEMS writes raw SAR dumps (`DumpType` 29, see [`super::dump`]) and
//! averages them with a separate program, `sar_calc -i raw.h5 -o out.h5
//! -m <grams> [--method SIMPLE|IEEE_C95_3|IEEE_62704] [-a <dB>] [-e]`.
//! This module reads the same raw files (openEMS's or ems's), averages
//! like openEMS ([`SarVolume::averaged_openems`], rules in
//! [`AveragingMethod`]) and writes the same output layout; [`run`] is what
//! the `ems-sar-calc` binary does. Layouts and rules were observed on a
//! black-box openEMS v0.37 (no openEMS source consulted).
//!
//! # Input (raw dump)
//!
//! ```text
//! /Mesh/{x,y,z}                   cell centres (copied to the output)
//! /CellWidth/{x,y,z}              cell widths [m]
//! /CellData/{Conductivity, Density, Volume}   f32 [nx, ny, nz] ("XYZ")
//! /FieldData/FD                   @frequency f64 [nf]
//! /FieldData/FD/f%d               {r, i} [3, nx, ny, nz] cell E field
//! ```
//!
//! Per cell the absorbed power density is `p = ½·κ·|E|²`. The averaging
//! uses the cells of the given widths, each weighted with its `Volume`
//! (as openEMS does, so the single-precision rounding of the volumes is
//! part of the result); the totals use the `Volume` dataset: mass `Σ ρ·V`,
//! absorbed power `Σ p·V`.
//!
//! Cylindrical dumps (`/Mesh` and `/CellWidth` with `rho`, `alpha`, `z`,
//! `@mesh_type` 1) are averaged like openEMS does (observed): as Cartesian
//! cells of the `CellWidth`s (`Δα` scaled by the drawing unit), each
//! weighing with its `Volume` (`ρ_i·Δα·Δρ·Δz`,
//! [`SarVolume::with_cell_volumes`]); the output mesh is written as
//! `x, y, z` with `@mesh_type` 0, like openEMS.
//!
//! # Output
//!
//! ```text
//! /                       @mass f64 (total mass), @openEMS_HDF5_version 0.3, @proc_time f64
//!                         (@legacy_fmt u8 1 with --legacyHDF5Dumps)
//! /Mesh                   @mesh_scaling f64 1, @mesh_type i32 0; x, y, z as in the input
//! /FieldData/FD           @frequency f32 [nf], @valid_cubes, @used_cubes,
//!                         @unused_cubes, @air_cubes u64 (of the first frequency);
//!                         with -a: @autorange f64 (dB), @autorange_padding f64 [m]
//! /FieldData/FD/f%d       f32 [nx, ny, nz] ("XYZ"; legacy [nz, ny, nx] "ZYX"),
//!                         @frequency f32, @maxSAR f32, @maxSAR_idx u32 [3], @power f32;
//!                         with -a: @autorange_threshold f32 [W/kg]
//! /FieldData/FD/f0_Cube{Mass, Type, Vol}   with -e and one frequency:
//!                         cube mass and volume (f32; the volume within the
//!                         domain, air included), type (u8: 7 valid centred
//!                         cube, 8 maximum of the valid cubes reaching the
//!                         voxel, 1…6 face cube extending along +x, −x, +y,
//!                         −y, +z, −z, 0 none / no tissue)
//! ```
//!
//! The counts: *valid* voxels have their own cube, *used* ones the maximum
//! of others, *unused* ones a face cube or none, *air* the rest. With `-a`
//! only the [`AutoRange`] cells are written (mesh, datasets, counts and
//! peak; the averages see the whole domain). `maxSAR_idx` is the voxel
//! holding the f32 maximum: the first centred cube among equal values,
//! else the first in data order.

use std::path::{Path, PathBuf};

use hdf5rs::{Complex, File};

use crate::post::sar::{AveragingMethod, SarField, SarVolume, VoxelClass};
use crate::{Error, Result};

fn h5err(e: hdf5rs::Error) -> Error {
    Error::Io(format!("HDF5: {e}"))
}

fn invalid(m: impl Into<String>) -> Error {
    Error::Invalid(m.into())
}

/// A raw SAR dump file (see the [module docs](self#input-raw-dump)).
#[derive(Debug, Clone, PartialEq)]
pub struct RawSar {
    /// `/Mesh/{x,y,z}` as stored (cell centres).
    pub mesh: [Vec<f64>; 3],
    /// Cell widths \[m\].
    pub widths: [Vec<f64>; 3],
    /// Conductivity per cell \[S/m\] (z fastest).
    pub conductivity: Vec<f64>,
    /// Mass density per cell \[kg/m³\].
    pub density: Vec<f64>,
    /// Volume per cell \[m³\].
    pub volume: Vec<f64>,
    /// Frequencies \[Hz\].
    pub frequencies: Vec<f64>,
    /// `|E|²` summed over the components, per frequency and cell.
    pub e_squared: Vec<Vec<f64>>,
    /// A cylindrical dump (`rho`, `alpha`, `z` axes).
    pub cylindrical: bool,
}

impl RawSar {
    /// Reads a raw SAR dump.
    ///
    /// # Errors
    /// I/O and format errors.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let f = File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        let read = |p: &str| f.dataset(p).and_then(|d| d.read::<f64>()).map_err(h5err);
        let cylindrical = f.dataset("/Mesh/rho").is_ok();
        let [nx, ny] = if cylindrical {
            ["rho", "alpha"]
        } else {
            ["x", "y"]
        };
        let mesh = [
            read(&format!("/Mesh/{nx}"))?,
            read(&format!("/Mesh/{ny}"))?,
            read("/Mesh/z")?,
        ];
        let widths = [
            read(&format!("/CellWidth/{nx}"))?,
            read(&format!("/CellWidth/{ny}"))?,
            read("/CellWidth/z")?,
        ];
        let [nx, ny, nz] = [0, 1, 2].map(|a| widths[a].len());
        let n = nx * ny * nz;
        if (0..3).any(|a| mesh[a].len() != widths[a].len()) {
            return Err(invalid(format!(
                "{}: /Mesh and /CellWidth sizes differ",
                path.display()
            )));
        }
        let cell = |name: &str| -> Result<Vec<f64>> {
            let v = read(&format!("/CellData/{name}"))?;
            if v.len() != n {
                return Err(invalid(format!("{}: {name} size mismatch", path.display())));
            }
            Ok(v)
        };
        let (conductivity, density, volume) =
            (cell("Conductivity")?, cell("Density")?, cell("Volume")?);
        let frequencies = f
            .group("/FieldData/FD")
            .and_then(|g| g.attribute("frequency"))
            .and_then(|a| a.read::<f64>())
            .map_err(h5err)?;
        let mut e_squared = Vec::with_capacity(frequencies.len());
        for k in 0..frequencies.len() {
            let e = f
                .dataset(&format!("/FieldData/FD/f{k}"))
                .and_then(|d| d.read::<Complex<f64>>())
                .map_err(h5err)?;
            if e.len() != 3 * n {
                return Err(invalid(format!("{}: f{k} size mismatch", path.display())));
            }
            e_squared.push(
                (0..n)
                    .map(|i| {
                        (0..3)
                            .map(|c| {
                                let z = e[c * n + i];
                                z.re * z.re + z.im * z.im
                            })
                            .sum()
                    })
                    .collect(),
            );
        }
        Ok(Self {
            mesh,
            widths,
            conductivity,
            density,
            volume,
            frequencies,
            e_squared,
            cylindrical,
        })
    }

    /// Cells per axis.
    pub fn dims(&self) -> [usize; 3] {
        [0, 1, 2].map(|a| self.widths[a].len())
    }

    /// Absorbed power density `½·κ·|E|²` per cell at frequency index `k`.
    pub fn power_density(&self, k: usize) -> Vec<f64> {
        self.e_squared[k]
            .iter()
            .zip(&self.conductivity)
            .map(|(e2, s)| 0.5 * s * e2)
            .collect()
    }

    /// Cell boundaries \[m\]: the widths stacked from 0 (only the widths
    /// matter for the averaging).
    pub fn lines(&self) -> [Vec<f64>; 3] {
        [0, 1, 2].map(|a| {
            let mut l = Vec::with_capacity(self.widths[a].len() + 1);
            let mut x = 0.0;
            l.push(x);
            for &d in &self.widths[a] {
                x += d;
                l.push(x);
            }
            l
        })
    }

    /// Total mass `Σ ρ·V` \[kg\].
    pub fn total_mass(&self) -> f64 {
        self.density
            .iter()
            .zip(&self.volume)
            .map(|(r, v)| r * v)
            .sum()
    }
}

/// Options of [`run`] (the `sar_calc` command line).
#[derive(Debug, Clone, PartialEq)]
pub struct SarCalcOptions {
    /// Input raw dump (`-i`).
    pub input: PathBuf,
    /// Output file (`-o`).
    pub output: PathBuf,
    /// Averaging mass \[g\] (`-m`).
    pub mass_g: f64,
    /// `--method`.
    pub method: AveragingMethod,
    /// `-n` (0: all cores).
    pub threads: usize,
    /// `-v`.
    pub verbose: bool,
    /// `-e`: write the cube statistics (single frequency only).
    pub export_cube_stats: bool,
    /// `--legacyHDF5Dumps`.
    pub legacy: bool,
    /// `-a`: write only the cells within this many dB of the peak local
    /// SAR ([`AutoRange`]).
    pub autorange: Option<f64>,
}

/// The averaged SAR of one frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct SarCalcFrequency {
    /// Frequency \[Hz\].
    pub frequency: f64,
    /// The averaged field (the [`AutoRange`] cells with `-a`).
    pub field: SarField,
    /// Absorbed power `Σ p·V` \[W\] (all cells).
    pub power: f64,
    /// Peak value and its cell (see the [module docs](self#output)).
    pub max: (f64, [usize; 3]),
    /// `[valid, used, unused, air]` voxel counts.
    pub cubes: [u64; 4],
    /// The auto-range threshold \[W/kg\] (with `-a`).
    pub threshold: Option<f64>,
}

/// The cells `sar_calc -a <dB>` writes (observed): the bounding box of the
/// tissue cells whose local SAR is at least `10^{−dB/10}` times the peak
/// local SAR, at any frequency, widened by `padding` (half the side of
/// the averaging cube of the lightest tissue, `(m/ρ_min)^{1/3}/2`) and
/// extended to the cells that contain the widened box's faces. The
/// averages themselves see the whole domain.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoRange {
    /// The `-a` value \[dB\].
    pub db: f64,
    /// Padding \[m\].
    pub padding: f64,
    /// First cell per axis.
    pub start: [usize; 3],
    /// One past the last cell per axis.
    pub stop: [usize; 3],
    /// Threshold \[W/kg\] per frequency.
    pub thresholds: Vec<f64>,
}

impl AutoRange {
    /// The range of `raw` for averaging mass `mass` \[kg\] (`None` without
    /// tissue).
    pub fn new(raw: &RawSar, mass: f64, db: f64) -> Option<Self> {
        let dims = raw.dims();
        let lines = raw.lines();
        let rho_min = raw
            .density
            .iter()
            .copied()
            .filter(|&r| r > 0.0)
            .fold(f64::INFINITY, f64::min);
        if !rho_min.is_finite() {
            return None;
        }
        let padding = 0.5 * (mass / rho_min).cbrt();
        let mut lo = [usize::MAX; 3];
        let mut hi = [0usize; 3];
        let mut thresholds = Vec::with_capacity(raw.frequencies.len());
        for k in 0..raw.frequencies.len() {
            let local: Vec<f64> = raw
                .power_density(k)
                .iter()
                .zip(&raw.density)
                .map(|(&p, &r)| if r > 0.0 { p / r } else { 0.0 })
                .collect();
            let top = local.iter().copied().fold(0.0, f64::max);
            let th = top * 10f64.powf(-db / 10.0);
            thresholds.push(th);
            for (i, &l) in local.iter().enumerate() {
                if raw.density[i] > 0.0 && l >= th {
                    let c = [
                        i / (dims[1] * dims[2]),
                        (i / dims[2]) % dims[1],
                        i % dims[2],
                    ];
                    for a in 0..3 {
                        lo[a] = lo[a].min(c[a]);
                        hi[a] = hi[a].max(c[a]);
                    }
                }
            }
        }
        if lo[0] == usize::MAX {
            return None;
        }
        let start = core::array::from_fn(|a| {
            let x = lines[a][lo[a]] - padding;
            lines[a].partition_point(|&l| l <= x).max(1) - 1
        });
        let stop = core::array::from_fn(|a| {
            let x = lines[a][hi[a] + 1] + padding;
            lines[a].partition_point(|&l| l < x).min(dims[a])
        });
        Some(Self {
            db,
            padding,
            start,
            stop,
            thresholds,
        })
    }
}

/// The openEMS cube type code of a voxel class (`-e` output): 7 centred
/// cube, 8 maximum of the valid cubes reaching the voxel, `face` (1…6,
/// [`SarField::cube_faces`]) for a face cube, 0 otherwise.
pub fn cube_type(c: VoxelClass, face: u8) -> u8 {
    match c {
        VoxelClass::Centered => 7,
        VoxelClass::Covered => 8,
        VoxelClass::Boundary => face,
        _ => 0,
    }
}

/// The cube counts and the peak of a field (see the
/// [module docs](self#output)).
fn summary(field: &SarField) -> ((f64, [usize; 3]), [u64; 4]) {
    let dims = field.cells();
    let mut cubes = [0u64; 4];
    for c in field.classes() {
        match c {
            VoxelClass::Centered => cubes[0] += 1,
            VoxelClass::Covered => cubes[1] += 1,
            VoxelClass::Boundary | VoxelClass::Unresolved => cubes[2] += 1,
            _ => cubes[3] += 1,
        }
    }
    // The f32 maximum; among equal values the first centred-cube voxel,
    // else the first voxel (data order), like openEMS.
    let top = field.values().iter().fold(0.0f32, |m, &v| m.max(v as f32));
    let pick = |centred: bool| {
        field
            .values()
            .iter()
            .zip(field.classes())
            .position(|(&v, &c)| {
                v > 0.0 && v as f32 == top && (!centred || c == VoxelClass::Centered)
            })
    };
    let max = match pick(true).or_else(|| pick(false)) {
        Some(i) => (
            field.values()[i],
            [
                i / (dims[1] * dims[2]),
                (i / dims[2]) % dims[1],
                i % dims[2],
            ],
        ),
        None => (0.0, [0; 3]),
    };
    (max, cubes)
}

/// Averages every frequency of `raw` over `mass` \[kg\] with `method`.
///
/// # Errors
/// Invalid masses or meshes ([`SarVolume::new`],
/// [`SarVolume::averaged_openems`]).
pub fn average(
    raw: &RawSar,
    mass: f64,
    method: AveragingMethod,
    threads: usize,
) -> Result<Vec<SarCalcFrequency>> {
    Ok(average_range(raw, mass, method, threads, None)?.0)
}

/// [`average`], keeping only the cells of `sar_calc -a <dB>` when
/// `autorange` is set ([`AutoRange`]).
///
/// # Errors
/// As [`average`].
pub fn average_range(
    raw: &RawSar,
    mass: f64,
    method: AveragingMethod,
    threads: usize,
    autorange: Option<f64>,
) -> Result<(Vec<SarCalcFrequency>, Option<AutoRange>)> {
    let lines = raw.lines();
    let range = autorange.and_then(|db| AutoRange::new(raw, mass, db));
    let mut out = Vec::with_capacity(raw.frequencies.len());
    for (k, &frequency) in raw.frequencies.iter().enumerate() {
        let pd = raw.power_density(k);
        let power = pd.iter().zip(&raw.volume).map(|(p, v)| p * v).sum();
        let v = SarVolume::new(lines.clone(), pd, raw.density.clone())?
            .with_cell_volumes(raw.volume.clone())?;
        let mut field = v.averaged_openems(mass, method, threads)?;
        if let Some(r) = &range {
            field = field.crop(r.start, r.stop)?;
        }
        let (max, cubes) = summary(&field);
        out.push(SarCalcFrequency {
            frequency,
            field,
            power,
            max,
            cubes,
            threshold: range.as_ref().map(|r| r.thresholds[k]),
        });
    }
    Ok((out, range))
}

mod h5 {
    use hdf5rs::write::{FileWriter, WriteAt, WriteOptions};

    use super::*;

    pub(super) fn options() -> WriteOptions {
        WriteOptions {
            superblock_version: 0,
            ..WriteOptions::default()
        }
    }

    pub(super) fn fill<S: WriteAt>(
        f: &mut FileWriter<S>,
        raw: &RawSar,
        res: &[SarCalcFrequency],
        range: Option<&AutoRange>,
        opts: &SarCalcOptions,
        proc_time: f64,
    ) -> hdf5rs::Result<()> {
        let (start, stop) = range.map_or(([0; 3], raw.dims()), |r| (r.start, r.stop));
        let dims: [usize; 3] = core::array::from_fn(|a| stop[a] - start[a]);
        let mut root = f.root();
        if opts.legacy {
            root.set_attr("legacy_fmt", &1u8)?;
        }
        root.set_attr("mass", &raw.total_mass())?;
        root.set_attr("openEMS_HDF5_version", &0.3f64)?;
        root.set_attr("proc_time", &proc_time)?;
        {
            let mut mesh = root.create_group("Mesh")?;
            mesh.set_attr("mesh_scaling", &1.0f64)?;
            mesh.set_attr("mesh_type", &0i32)?;
            for (a, n) in ["x", "y", "z"].into_iter().enumerate() {
                let m = &raw.mesh[a][start[a]..stop[a]];
                mesh.new_dataset::<f64>(n)
                    .shape(&[m.len() as u64])
                    .write(m)?;
            }
        }
        let mut fdg = root.create_group("FieldData")?;
        let mut g = fdg.create_group("FD")?;
        let fr: Vec<f32> = res.iter().map(|r| r.frequency as f32).collect();
        let cubes = res.first().map_or([0; 4], |r| r.cubes);
        g.set_attr("air_cubes", &cubes[3])?;
        if let Some(r) = range {
            g.set_attr("autorange", &r.db)?;
            g.set_attr("autorange_padding", &r.padding)?;
        }
        g.set_attr("frequency", &fr[..])?;
        g.set_attr("unused_cubes", &cubes[2])?;
        g.set_attr("used_cubes", &cubes[1])?;
        g.set_attr("valid_cubes", &cubes[0])?;
        let shape = if opts.legacy {
            [dims[2], dims[1], dims[0]].map(|n| n as u64)
        } else {
            dims.map(|n| n as u64)
        };
        let order = |v: &[f64]| -> Vec<f32> {
            if !opts.legacy {
                return v.iter().map(|&x| x as f32).collect();
            }
            let mut o = Vec::with_capacity(v.len());
            for k in 0..dims[2] {
                for j in 0..dims[1] {
                    for i in 0..dims[0] {
                        o.push(v[(i * dims[1] + j) * dims[2] + k] as f32);
                    }
                }
            }
            o
        };
        for (k, r) in res.iter().enumerate() {
            let mut ds = g
                .new_dataset::<f32>(&format!("f{k}"))
                .shape(&shape)
                .write(&order(r.field.values()))?;
            if let Some(t) = r.threshold {
                ds.set_attr("autorange_threshold", &(t as f32))?;
            }
            ds.set_attr("d_order", if opts.legacy { "ZYX" } else { "XYZ" })?;
            ds.set_attr("frequency", &(r.frequency as f32))?;
            ds.set_attr("maxSAR", &(r.max.0 as f32))?;
            ds.set_attr("maxSAR_idx", &r.max.1.map(|i| i as u32))?;
            ds.set_attr("power", &(r.power as f32))?;
        }
        if opts.export_cube_stats && res.len() == 1 {
            let r = &res[0];
            g.new_dataset::<f32>("f0_CubeMass")
                .shape(&shape)
                .write(&order(r.field.cube_masses()))?;
            let types: Vec<f64> = r
                .field
                .classes()
                .iter()
                .zip(r.field.cube_faces())
                .map(|(&c, &k)| f64::from(cube_type(c, k)))
                .collect();
            let types: Vec<u8> = order(&types).into_iter().map(|x| x as u8).collect();
            g.new_dataset::<u8>("f0_CubeType")
                .shape(&shape)
                .write(&types)?;
            g.new_dataset::<f32>("f0_CubeVol")
                .shape(&shape)
                .write(&order(r.field.cube_volumes()))?;
        }
        Ok(())
    }
}

/// The output file of an averaging in memory (see the
/// [module docs](self#output)); `range` the [`AutoRange`] of `-a`.
///
/// # Errors
/// Encoding errors ([`Error::Io`]).
pub fn output_bytes(
    raw: &RawSar,
    res: &[SarCalcFrequency],
    range: Option<&AutoRange>,
    opts: &SarCalcOptions,
    proc_time: f64,
) -> Result<Vec<u8>> {
    let mut f = hdf5rs::write::FileWriter::new(Vec::new(), h5::options()).map_err(h5err)?;
    h5::fill(&mut f, raw, res, range, opts, proc_time).map_err(h5err)?;
    f.finish().map_err(h5err)
}

/// The result of [`run`].
#[derive(Debug, Clone, PartialEq)]
pub struct SarCalcRun {
    /// The input.
    pub raw: RawSar,
    /// Per frequency.
    pub frequencies: Vec<SarCalcFrequency>,
    /// The cells written with `-a`.
    pub range: Option<AutoRange>,
}

/// Reads `opts.input`, averages and writes `opts.output` like
/// `sar_calc`.
///
/// # Errors
/// I/O, format and averaging errors; [`Error::Invalid`] for a
/// non-positive `-a`.
pub fn run(opts: &SarCalcOptions) -> Result<SarCalcRun> {
    if let Some(db) = opts.autorange
        && !(db > 0.0 && db.is_finite())
    {
        return Err(invalid(format!("the auto range must be > 0 dB, got {db}")));
    }
    let raw = RawSar::read(&opts.input)?;
    let t0 = std::time::Instant::now();
    let (frequencies, range) = average_range(
        &raw,
        opts.mass_g * 1e-3,
        opts.method,
        opts.threads,
        opts.autorange,
    )?;
    let proc_time = t0.elapsed().as_secs_f64();
    let bytes = output_bytes(&raw, &frequencies, range.as_ref(), opts, proc_time)?;
    std::fs::write(&opts.output, bytes)
        .map_err(|e| Error::Io(format!("{}: {e}", opts.output.display())))?;
    Ok(SarCalcRun {
        raw,
        frequencies,
        range,
    })
}
