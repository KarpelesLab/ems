//! Specific absorption rate (SAR): local and mass-averaged.
//!
//! # Definitions
//!
//! For a time-harmonic field with **peak** phasor `E` (the convention of
//! all ems frequency-domain data), the time-averaged power absorbed per
//! volume and the local SAR are
//!
//! ```text
//! p   = ½·σ·|E|²            [W/m³]
//! SAR = p / ρ = σ·|E|²/(2ρ) [W/kg]
//! ```
//!
//! with conductivity `σ` \[S/m\] and mass density `ρ` \[kg/m³\]. With
//! **RMS** phasors the factor ½ disappears (`SAR = σ·|E_rms|²/ρ`); scale
//! the result by 2 ([`SarVolume::scaled`]) if the field was RMS.
//!
//! When `E` comes from the DFT of a transient run (as every ems
//! frequency-domain dump does), the values are spectral densities (W/m³·s²
//! and W/kg·s²). The usual normalisation is to a given accepted power:
//! divide by the port's [`accepted_power`](super::PortResult::accepted_power)
//! at the same frequency (both have the same `s²` factor), which gives SAR
//! per watt accepted.
//!
//! # Voxels
//!
//! SAR is evaluated on the primary cells (voxels) of the mesh, which may be
//! non-uniform. A [`SarVolume`] holds, per cell, the absorbed power density
//! `p` and the mass density `ρ`; cells with `ρ = 0` are **not tissue** (air,
//! metal, lossy non-biological parts): they have no SAR and count as air
//! for the averaging rules. Power absorbed in non-tissue cells is part of
//! [`SarVolume::total_power`] but not of any SAR value.
//!
//! [`SarVolume::from_edge_fields`] builds `p` from the E field on the cell
//! edges (each cell has four edges per axis): with `σ_a` the cell
//! conductivity along axis `a` (diagonal anisotropy),
//!
//! ```text
//! p_cell = ½ · Σ_a σ_a · ¼ · Σ_{4 edges e ∥ a} |E_e|²
//! ```
//!
//! This is the mean of `|E|²` (not the square of the mean field) over the
//! cell's edges. Summed over the cells it is exactly the loss
//! `Σ_e ½·G_e·|U_e|²` of the edge conductances `G_e = κ_e·A*_e/l_e` when
//! the edge conductivity is the area-weighted average of the adjacent cell
//! conductivities, which is how
//! [`EdgeMaterials::from_cells`](crate::engine::EdgeMaterials::from_cells)
//! and csx's discretisation define it. [`SarVolume::from_edge_dft`] takes
//! the edge voltages from an [`EdgeDft`] and applies the
//! [time-averaging correction](SarVolume::from_edge_dft) of the update.
//!
//! # Mass averaging
//!
//! [`averaged_sar`] computes the SAR averaged over a cube of tissue mass
//! `m` (1 g or 10 g in the standards) around each tissue voxel, after the
//! cube procedure described in IEEE Std 1528 and IEC/IEEE 62704-1. The
//! implementation, from the published description:
//!
//! 1. **Centred cubes.** For every tissue voxel an axis-aligned cube centred
//!    on the voxel centre is grown until the tissue mass inside it equals
//!    `m`. Voxels cut by the cube count with the fraction of their volume
//!    inside it (an exact, continuous function of the side, solved to a
//!    relative mass error below 1e-10), which plays the role of the
//!    standards' fractional outer layer. The cube is **valid** when at most
//!    [`AveragingOptions::max_air_fraction`] (10 %) of its volume is not
//!    tissue; volume outside the computational domain counts as air. The
//!    averaged SAR of a valid cube is the tissue power inside it divided by
//!    its tissue mass, and is assigned to its centre voxel
//!    ([`VoxelClass::Centered`]).
//! 2. **Covered voxels.** A tissue voxel whose centred cube is not valid
//!    gets the largest averaged SAR of all valid cubes that contain its
//!    centre ([`VoxelClass::Covered`]).
//! 3. **Boundary cubes.** A tissue voxel that is in no valid cube gets the
//!    average of a cube with the voxel at the centre of one of its faces:
//!    of the six orientations (the cube extending from the voxel's lower or
//!    upper face along ±x, ±y or ±z, centred on the voxel across), the one
//!    with the smallest air fraction is used, regardless of the 10 % limit,
//!    ties broken by the larger SAR ([`VoxelClass::Boundary`]).
//!
//! Deviations from the standards, which the user should know about:
//!
//! * The cube is aligned with the mesh axes; the standards' variant that
//!   rotates or tilts cubes to follow a curved surface is not implemented.
//! * Step 3 is a simplification of the standards' treatment of surface
//!   voxels; it is used only where no valid centred cube reaches the voxel.
//! * The air fraction uses non-tissue *volume* (`ρ = 0`), including any
//!   non-biological material and metal.
//! * A voxel for which no cube of mass `m` exists (the whole tissue is
//!   lighter than `m`) is [`VoxelClass::Unresolved`] with SAR 0.
//!
//! # Example
//!
//! ```
//! use ems::post::sar::{SarVolume, VoxelClass};
//!
//! // 10 × 10 × 10 voxels of 1 cm, uniform tissue (ρ = 1000 kg/m³) with
//! // 2 W/m³ absorbed everywhere.
//! let lines = [(); 3].map(|_| (0..=10).map(|i| f64::from(i) * 0.01).collect::<Vec<_>>());
//! let v = SarVolume::new(lines, vec![2.0; 1000], vec![1000.0; 1000])?;
//! let local = v.local();
//! assert_eq!(local.values()[0], 2e-3); // W/kg
//! let avg = v.averaged(0.01)?; // 10 g
//! assert!((avg.max().unwrap().value - 2e-3).abs() < 1e-15);
//! assert_eq!(avg.classes()[555], VoxelClass::Centered);
//! # Ok::<(), ems::Error>(())
//! ```

use std::path::Path;

use crate::Axis;
use crate::engine::{EdgeDft, Grid};
use crate::io::{Layout, VtrWriter};
use crate::post::Complex64;
use crate::{Error, Result};

/// Absorbed power density and mass density per voxel; see the
/// [module docs](self).
///
/// Cells are indexed `(i·ny + j)·nz + k` (z fastest).
#[derive(Debug, Clone, PartialEq)]
pub struct SarVolume {
    lines: [Vec<f64>; 3],
    power: Vec<f64>,
    density: Vec<f64>,
    /// `(ρ, α, z)` cells ([`SarVolume::cylindrical`]).
    cylindrical: bool,
}

/// How a voxel's value in a [`SarField`] was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum VoxelClass {
    /// Not tissue (`ρ = 0`): SAR 0.
    #[default]
    NoTissue = 0,
    /// Local SAR `p/ρ` of the voxel.
    Local = 1,
    /// Averaged: valid cube centred on the voxel.
    Centered = 2,
    /// Averaged: maximum of the valid cubes containing the voxel.
    Covered = 3,
    /// Averaged: cube with the voxel at the centre of a face.
    Boundary = 4,
    /// Averaged: no cube of the requested mass exists; SAR 0.
    Unresolved = 5,
}

/// Options for [`SarVolume::averaged_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AveragingOptions {
    /// Largest non-tissue volume fraction of a valid cube (default 0.1).
    /// 1 makes every centred cube valid: each tissue voxel gets the
    /// average of the cube centred on it, whatever the air inside (the
    /// rule of openEMS's `SIMPLE` method, see
    /// [SAR dumps](crate::io::openems::dump#sar-dumps)).
    pub max_air_fraction: f64,
    /// Worker threads (0: available parallelism). The result does not
    /// depend on it.
    pub threads: usize,
}

impl Default for AveragingOptions {
    fn default() -> Self {
        Self {
            max_air_fraction: 0.1,
            threads: 0,
        }
    }
}

/// The averaging methods of openEMS's `sar_calc --method` (and of its
/// in-run SAR dumps, which use [`Simple`](Self::Simple)).
///
/// ems implements them on its own cube solver (exact cube mass, see the
/// [module docs](self#mass-averaging)); what openEMS does was observed on
/// a black-box openEMS v0.37:
///
/// * `SIMPLE`: every tissue voxel gets the average of the cube centred on
///   it, clipped by the domain, whatever the air inside; no tissue cube of
///   the mass exists → 0. openEMS solves the cube size only to ±5 % of the
///   mass (an iterative solver started from the size `(m/ρ)^{1/3}` of the
///   voxel's density), so its values equal ems's where that first size is
///   exact (a cube of uniform tissue clear of air and of the domain
///   boundary, at least two voxels wide) and differ by up to the effect of
///   a 5 % mass error elsewhere.
/// * `IEEE_C95_3`, `IEEE_62704`: valid centred cubes, then the maximum of
///   the valid cubes reaching a voxel, then cubes at a face. ems uses its
///   own IEEE 62704-1 procedure (10 % air) for both; openEMS's rules
///   differ in detail (a centred cube is invalid when a face lies in air or
///   outside the domain, even with little air inside, and valid with an
///   air pocket inside; `IEEE_C95_3` takes the maximum over the valid cubes
///   that overlap a voxel, `IEEE_62704` over those that contain it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AveragingMethod {
    /// `SIMPLE`: centred cubes only (openEMS's default and in-run method).
    #[default]
    Simple,
    /// `IEEE_C95_3`.
    IeeeC95_3,
    /// `IEEE_62704`.
    Ieee62704,
}

impl AveragingMethod {
    /// The openEMS name (`SIMPLE`, `IEEE_C95_3`, `IEEE_62704`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Simple => "SIMPLE",
            Self::IeeeC95_3 => "IEEE_C95_3",
            Self::Ieee62704 => "IEEE_62704",
        }
    }

    /// Parses an openEMS method name.
    pub fn from_name(s: &str) -> Option<Self> {
        [Self::Simple, Self::IeeeC95_3, Self::Ieee62704]
            .into_iter()
            .find(|m| m.name() == s)
    }

    /// The [`AveragingOptions`] ems uses for the method.
    pub fn options(self, threads: usize) -> AveragingOptions {
        AveragingOptions {
            max_air_fraction: if self == Self::Simple { 1.0 } else { 0.1 },
            threads,
        }
    }
}

/// The largest value of a [`SarField`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SarMax {
    /// SAR \[W/kg\].
    pub value: f64,
    /// Cell index `[i, j, k]` within the volume.
    pub cell: [usize; 3],
    /// Centre of that cell \[m\].
    pub center: [f64; 3],
}

/// Local or mass-averaged SAR per voxel.
#[derive(Debug, Clone, PartialEq)]
pub struct SarField {
    lines: [Vec<f64>; 3],
    values: Vec<f64>,
    classes: Vec<VoxelClass>,
    mass: Option<f64>,
    cube_mass: Vec<f64>,
    cube_volume: Vec<f64>,
}

fn cells_of(lines: &[Vec<f64>; 3]) -> [usize; 3] {
    [0, 1, 2].map(|a| lines[a].len().saturating_sub(1))
}

impl SarVolume {
    /// From the cell boundaries `lines` \[m\] (strictly increasing, at least
    /// two per axis), the absorbed power density per cell \[W/m³\] and the
    /// mass density per cell \[kg/m³\].
    ///
    /// # Errors
    /// [`Error::Invalid`] for bad lines, wrong lengths, or negative or
    /// non-finite values.
    pub fn new(lines: [Vec<f64>; 3], power: Vec<f64>, density: Vec<f64>) -> Result<Self> {
        for (a, l) in lines.iter().enumerate() {
            if l.len() < 2
                || l.windows(2).any(|w| w[1] <= w[0] || w[1].is_nan())
                || !l[0].is_finite()
            {
                return Err(Error::Invalid(format!(
                    "SAR volume: lines along axis {a} must be at least two increasing values"
                )));
            }
        }
        let n: usize = cells_of(&lines).iter().product();
        if power.len() != n || density.len() != n {
            return Err(Error::Invalid(format!(
                "SAR volume: {n} cells but {} power and {} density values",
                power.len(),
                density.len()
            )));
        }
        if let Some(x) = power
            .iter()
            .chain(&density)
            .find(|x| !(x.is_finite() && **x >= 0.0))
        {
            return Err(Error::Invalid(format!(
                "SAR volume: power and density must be finite and non-negative, got {x}"
            )));
        }
        Ok(Self {
            lines,
            power,
            density,
            cylindrical: false,
        })
    }

    /// As [`new`](Self::new) for the cells of a cylindrical grid: `lines`
    /// are `ρ` \[m\] (`≥ 0`), `α` \[rad\] and `z` \[m\]; the cell volumes
    /// are `Δα·Δz·(ρ_{i+1}² − ρ_i²)/2` ([`cell_volumes`](Self::cell_volumes)).
    /// Local SAR, powers and masses work as on Cartesian cells; mass
    /// averaging is not available ([`averaged_with`](Self::averaged_with)).
    ///
    /// # Errors
    /// As [`new`](Self::new), and a negative radius.
    pub fn cylindrical(lines: [Vec<f64>; 3], power: Vec<f64>, density: Vec<f64>) -> Result<Self> {
        if lines[0].first().is_some_and(|&r| r < 0.0) {
            return Err(Error::Invalid(
                "SAR volume: cylindrical radii must be >= 0".into(),
            ));
        }
        let mut v = Self::new(lines, power, density)?;
        v.cylindrical = true;
        Ok(v)
    }

    /// Whether the cells are `(ρ, α, z)` cells ([`cylindrical`](Self::cylindrical)).
    pub fn is_cylindrical(&self) -> bool {
        self.cylindrical
    }

    /// From the E field on the cell edges: `e[a]` holds the peak phasor
    /// \[V/m\] of every edge along axis `a` of the cells, in the order of
    /// [`EdgeDft::edge_dims`] (`cells[a]` along `a`, `cells[b] + 1` across,
    /// z fastest); `sigma` is the conductivity per cell and axis \[S/m\],
    /// `density` the mass density per cell. See the
    /// [module docs](self#voxels) for the formula.
    ///
    /// # Errors
    /// See [`new`](Self::new); also wrong edge array lengths or negative
    /// conductivities.
    pub fn from_edge_fields(
        lines: [Vec<f64>; 3],
        e: [&[Complex64]; 3],
        sigma: &[[f64; 3]],
        density: Vec<f64>,
    ) -> Result<Self> {
        let c = cells_of(&lines);
        let n: usize = c.iter().product();
        if sigma.len() != n {
            return Err(Error::Invalid(format!(
                "SAR volume: {n} cells but {} conductivities",
                sigma.len()
            )));
        }
        if let Some(s) = sigma
            .iter()
            .flatten()
            .find(|s| !(s.is_finite() && **s >= 0.0))
        {
            return Err(Error::Invalid(format!(
                "SAR volume: conductivity must be finite and non-negative, got {s}"
            )));
        }
        let dims =
            Axis::ALL.map(|a| [0, 1, 2].map(|b| if b == a.index() { c[b] } else { c[b] + 1 }));
        for a in 0..3 {
            let want: usize = dims[a].iter().product();
            if e[a].len() != want {
                return Err(Error::Invalid(format!(
                    "SAR volume: {} edge values along axis {a}, expected {want}",
                    e[a].len()
                )));
            }
        }
        let mut power = vec![0.0; n];
        for i in 0..c[0] {
            for j in 0..c[1] {
                for k in 0..c[2] {
                    let cell = (i * c[1] + j) * c[2] + k;
                    let mut p = 0.0;
                    for a in 0..3 {
                        let s = sigma[cell][a];
                        if s == 0.0 {
                            continue;
                        }
                        let (b, cc) = ((a + 1) % 3, (a + 2) % 3);
                        let d = dims[a];
                        let mut sum = 0.0;
                        for db in 0..2 {
                            for dc in 0..2 {
                                let mut n = [i, j, k];
                                n[b] += db;
                                n[cc] += dc;
                                sum += e[a][(n[0] * d[1] + n[1]) * d[2] + n[2]].norm_sqr();
                            }
                        }
                        p += s * 0.25 * sum;
                    }
                    power[cell] = 0.5 * p;
                }
            }
        }
        Self::new(lines, power, density)
    }

    /// From the edge voltages of `dft` at frequency index `k` on `grid`
    /// (the grid the operator ran on), with `sigma` and `density` per cell
    /// of its [`CellBox`](crate::engine::CellBox) as in
    /// [`from_edge_fields`](Self::from_edge_fields).
    ///
    /// The field of edge `e` is `E_e = U_e/l_e·cos(πf·dt)`. The cosine
    /// factor is the time-averaging of the conduction current in the update
    /// (`G·(u^{n+1} + u^n)/2`, whose transform is `U·e^{-jπf·dt}·cos(πf·dt)`),
    /// so that the power is the loss of the discrete scheme itself; it is
    /// within 1 % of 1 at 10 or more steps per period of `f`.
    ///
    /// # Errors
    /// See [`from_edge_fields`](Self::from_edge_fields), and a box that does
    /// not fit `grid`.
    ///
    /// # Panics
    /// If `k` is out of range.
    pub fn from_edge_dft(
        grid: &Grid,
        dft: &EdgeDft,
        k: usize,
        sigma: &[[f64; 3]],
        density: Vec<f64>,
    ) -> Result<Self> {
        let b = *dft.cell_box();
        let nodes = grid.nodes();
        if (0..3).any(|a| b.hi()[a] >= nodes[a]) {
            return Err(Error::Invalid(
                "SAR volume: the edge DFT box does not fit the grid".into(),
            ));
        }
        let f = dft.freqs()[k];
        let corr = (core::f64::consts::PI * f * dft.dt()).cos();
        let lo = b.lo();
        let cyl = grid.is_cylindrical();
        let e = Axis::ALL.map(|axis| {
            let a = axis.index();
            let d = dft.edge_dims(axis);
            let mut u = dft.voltages(axis, k);
            for (idx, v) in u.iter_mut().enumerate() {
                let q = [idx / (d[1] * d[2]), (idx / d[2]) % d[1], idx % d[2]];
                if cyl {
                    // Physical length; α edges on the axis do not exist.
                    let l = grid.edge_length(axis, [0, 1, 2].map(|c| lo[c] + q[c]));
                    *v = if l > 0.0 {
                        *v * (corr / l)
                    } else {
                        Complex64::ZERO
                    };
                } else {
                    *v *= corr / grid.delta(axis, lo[a] + q[a]);
                }
            }
            u
        });
        let lines = Axis::ALL.map(|a| b.lines(grid, a).to_vec());
        if cyl {
            return Self::from_cylindrical_edge_fields(
                lines,
                [&e[0], &e[1], &e[2]],
                sigma,
                density,
            );
        }
        Self::from_edge_fields(lines, [&e[0], &e[1], &e[2]], sigma, density)
    }

    /// [`from_edge_fields`](Self::from_edge_fields) on the `(ρ, α, z)`
    /// cells of a cylindrical grid ([`cylindrical`](Self::cylindrical)):
    /// each edge's `|E|²` is weighted with the share of the cell volume
    /// that the edge's dual cell has in it, as the engine's edge
    /// conductances `κ·A*/l` count it: ¼ for the ρ edges, `ρ_e/(4ρ_{i+½})`
    /// for the α edges at `ρ_e ∈ {ρ_i, ρ_{i+1}}`, and the half-annulus
    /// shares `(ρ_{i+½}² − ρ_i²)/(2(ρ_{i+1}² − ρ_i²))` (inner) and
    /// `(ρ_{i+1}² − ρ_{i+½}²)/(2(ρ_{i+1}² − ρ_i²))` (outer) for the z edges.
    /// Then `Σ p·V` is the conduction loss of the discrete scheme, as on
    /// Cartesian cells.
    fn from_cylindrical_edge_fields(
        lines: [Vec<f64>; 3],
        e: [&[Complex64]; 3],
        sigma: &[[f64; 3]],
        density: Vec<f64>,
    ) -> Result<Self> {
        let c = cells_of(&lines);
        let n: usize = c.iter().product();
        if sigma.len() != n {
            return Err(Error::Invalid(format!(
                "SAR volume: {n} cells but {} conductivities",
                sigma.len()
            )));
        }
        let dims =
            Axis::ALL.map(|a| [0, 1, 2].map(|b| if b == a.index() { c[b] } else { c[b] + 1 }));
        for a in 0..3 {
            let want: usize = dims[a].iter().product();
            if e[a].len() != want {
                return Err(Error::Invalid(format!(
                    "SAR volume: {} edge values along axis {a}, expected {want}",
                    e[a].len()
                )));
            }
        }
        let r = &lines[0];
        let mut power = vec![0.0; n];
        for i in 0..c[0] {
            let (r0, r1) = (r[i], r[i + 1]);
            let rm = 0.5 * (r0 + r1);
            let ann = r1 * r1 - r0 * r0;
            // Weights of the edges at ρ_i (index 0) and ρ_{i+1} (1).
            let w_alpha = [r0 / (4.0 * rm), r1 / (4.0 * rm)];
            let w_z = [
                (rm * rm - r0 * r0) / (2.0 * ann),
                (r1 * r1 - rm * rm) / (2.0 * ann),
            ];
            for j in 0..c[1] {
                for k in 0..c[2] {
                    let cell = (i * c[1] + j) * c[2] + k;
                    let mut p = 0.0;
                    for a in 0..3 {
                        let s = sigma[cell][a];
                        if s == 0.0 {
                            continue;
                        }
                        let (b, cc) = ((a + 1) % 3, (a + 2) % 3);
                        let d = dims[a];
                        let mut sum = 0.0;
                        for db in 0..2 {
                            for dc in 0..2 {
                                let mut q = [i, j, k];
                                q[b] += db;
                                q[cc] += dc;
                                let w = match a {
                                    0 => 0.25,
                                    1 => w_alpha[q[0] - i],
                                    _ => w_z[q[0] - i],
                                };
                                sum += w * e[a][(q[0] * d[1] + q[1]) * d[2] + q[2]].norm_sqr();
                            }
                        }
                        p += s * sum;
                    }
                    power[cell] = 0.5 * p;
                }
            }
        }
        Self::cylindrical(lines, power, density)
    }

    /// Cell boundaries along `axis` \[m\].
    pub fn lines(&self, axis: Axis) -> &[f64] {
        &self.lines[axis.index()]
    }

    /// Number of cells per axis.
    pub fn cells(&self) -> [usize; 3] {
        cells_of(&self.lines)
    }

    /// Absorbed power density per cell \[W/m³\] (time average).
    pub fn power_density(&self) -> &[f64] {
        &self.power
    }

    /// Mass density per cell \[kg/m³\].
    pub fn density(&self) -> &[f64] {
        &self.density
    }

    /// Volume of each cell \[m³\].
    pub fn cell_volumes(&self) -> Vec<f64> {
        let c = self.cells();
        let mut d: [Vec<f64>; 3] =
            core::array::from_fn(|a| self.lines[a].windows(2).map(|w| w[1] - w[0]).collect());
        if self.cylindrical {
            // ∫ρ dρ = (ρ_{i+1}² − ρ_i²)/2 over the radial cells.
            d[0] = self.lines[0]
                .windows(2)
                .map(|w| 0.5 * (w[1] * w[1] - w[0] * w[0]))
                .collect();
        }
        let mut v = Vec::with_capacity(c.iter().product());
        for i in 0..c[0] {
            for j in 0..c[1] {
                for k in 0..c[2] {
                    v.push(d[0][i] * d[1][j] * d[2][k]);
                }
            }
        }
        v
    }

    /// Total absorbed power `Σ p·V` \[W\], including non-tissue cells.
    pub fn total_power(&self) -> f64 {
        self.power
            .iter()
            .zip(self.cell_volumes())
            .map(|(p, v)| p * v)
            .sum()
    }

    /// Power absorbed in tissue cells (`ρ > 0`) \[W\].
    pub fn tissue_power(&self) -> f64 {
        self.power
            .iter()
            .zip(&self.density)
            .zip(self.cell_volumes())
            .filter(|((_, r), _)| **r > 0.0)
            .map(|((p, _), v)| p * v)
            .sum()
    }

    /// Total tissue mass `Σ ρ·V` \[kg\].
    pub fn tissue_mass(&self) -> f64 {
        self.density
            .iter()
            .zip(self.cell_volumes())
            .map(|(r, v)| r * v)
            .sum()
    }

    /// Whole-body (volume) average SAR: tissue power over tissue mass
    /// \[W/kg\] (0 without tissue).
    pub fn whole_average(&self) -> f64 {
        let m = self.tissue_mass();
        if m > 0.0 {
            self.tissue_power() / m
        } else {
            0.0
        }
    }

    /// The same volume with the power multiplied by `factor` (e.g. 2 for
    /// RMS phasors, or `1/P_acc` to normalise to 1 W accepted).
    pub fn scaled(&self, factor: f64) -> Self {
        let mut v = self.clone();
        for p in &mut v.power {
            *p *= factor;
        }
        v
    }

    /// Local SAR `p/ρ` per voxel ([`local_sar`]).
    pub fn local(&self) -> SarField {
        let (values, classes) = self
            .power
            .iter()
            .zip(&self.density)
            .map(|(&p, &r)| {
                if r > 0.0 {
                    (p / r, VoxelClass::Local)
                } else {
                    (0.0, VoxelClass::NoTissue)
                }
            })
            .unzip();
        SarField {
            lines: self.lines.clone(),
            values,
            classes,
            mass: None,
            cube_mass: Vec::new(),
            cube_volume: Vec::new(),
        }
    }

    /// SAR averaged over `mass` \[kg\] of tissue with the default
    /// [`AveragingOptions`] ([`averaged_sar`]).
    ///
    /// # Errors
    /// [`Error::Invalid`] for a mass that is not positive and finite.
    pub fn averaged(&self, mass: f64) -> Result<SarField> {
        self.averaged_with(mass, &AveragingOptions::default())
    }

    /// SAR averaged over `mass` \[kg\]; see the
    /// [module docs](self#mass-averaging).
    ///
    /// # Errors
    /// [`Error::Invalid`] for a mass that is not positive and finite or an
    /// air fraction outside `[0, 1]`; [`Error::Unsupported`] for
    /// [cylindrical](Self::cylindrical) cells.
    pub fn averaged_with(&self, mass: f64, opts: &AveragingOptions) -> Result<SarField> {
        if self.cylindrical {
            return Err(Error::Unsupported(
                "mass-averaged SAR on a cylindrical grid: the averaging cubes are defined in \
                 physical (Cartesian) coordinates, not on (rho, alpha, z) cells; use the local \
                 SAR"
                .into(),
            ));
        }
        if !(mass.is_finite() && mass > 0.0) {
            return Err(Error::Invalid(format!(
                "SAR averaging mass must be positive, got {mass}"
            )));
        }
        if !(0.0..=1.0).contains(&opts.max_air_fraction) {
            return Err(Error::Invalid(format!(
                "SAR max air fraction must be in [0, 1], got {}",
                opts.max_air_fraction
            )));
        }
        Ok(Averager::new(self, mass, opts).run())
    }
}

/// Local SAR of `v` (see [`SarVolume::local`]).
pub fn local_sar(v: &SarVolume) -> SarField {
    v.local()
}

/// SAR of `v` averaged over `mass` \[kg\] (see
/// [`SarVolume::averaged`]).
///
/// # Errors
/// See [`SarVolume::averaged`].
pub fn averaged_sar(v: &SarVolume, mass: f64) -> Result<SarField> {
    v.averaged(mass)
}

impl SarField {
    /// Cell boundaries along `axis` \[m\].
    pub fn lines(&self, axis: Axis) -> &[f64] {
        &self.lines[axis.index()]
    }

    /// Number of cells per axis.
    pub fn cells(&self) -> [usize; 3] {
        cells_of(&self.lines)
    }

    /// SAR per voxel \[W/kg\] (z fastest).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// How each value was obtained.
    pub fn classes(&self) -> &[VoxelClass] {
        &self.classes
    }

    /// The averaging mass \[kg\], `None` for local SAR.
    pub fn mass(&self) -> Option<f64> {
        self.mass
    }

    /// Per voxel, the tissue mass \[kg\] of the cube whose average it
    /// holds ([`VoxelClass::Centered`] and [`VoxelClass::Boundary`]; 0
    /// otherwise). Empty for local SAR.
    pub fn cube_masses(&self) -> &[f64] {
        &self.cube_mass
    }

    /// Per voxel, the volume \[m³\] within the domain (tissue and air) of
    /// that cube, like [`cube_masses`](Self::cube_masses).
    pub fn cube_volumes(&self) -> &[f64] {
        &self.cube_volume
    }

    /// The SAR at cell `[i, j, k]`.
    ///
    /// # Panics
    /// If the cell is out of range.
    pub fn get(&self, cell: [usize; 3]) -> f64 {
        let c = self.cells();
        assert!((0..3).all(|a| cell[a] < c[a]), "cell {cell:?} out of range");
        self.values[(cell[0] * c[1] + cell[1]) * c[2] + cell[2]]
    }

    /// The largest value and its voxel (`None` without tissue). Among
    /// equal values a [`VoxelClass::Centered`] voxel (the centre of the
    /// peak cube) is preferred, then the first voxel in data order.
    pub fn max(&self) -> Option<SarMax> {
        let c = self.cells();
        let mut best: Option<(usize, f64, bool)> = None;
        for (i, (&v, &k)) in self.values.iter().zip(&self.classes).enumerate() {
            if matches!(k, VoxelClass::NoTissue | VoxelClass::Unresolved) {
                continue;
            }
            let centred = k == VoxelClass::Centered;
            if best.is_none_or(|(_, b, bc)| v > b || (v == b && centred && !bc)) {
                best = Some((i, v, centred));
            }
        }
        best.map(|(i, value, _)| {
            let cell = [i / (c[1] * c[2]), (i / c[2]) % c[1], i % c[2]];
            SarMax {
                value,
                cell,
                center: [0, 1, 2]
                    .map(|a| 0.5 * (self.lines[a][cell[a]] + self.lines[a][cell[a] + 1])),
            }
        })
    }

    /// Writes the field as a `.vtr` file with cell arrays `SAR` \[W/kg\] and
    /// `class` (the [`VoxelClass`] number) on the cell boundaries, and a
    /// `Mass` field-data entry (kg, 0 for local SAR).
    ///
    /// # Errors
    /// I/O errors.
    pub fn write_vtr(&self, path: impl AsRef<Path>) -> Result<()> {
        let class: Vec<f32> = self.classes.iter().map(|&c| f32::from(c as u8)).collect();
        let mass = [self.mass.unwrap_or(0.0)];
        VtrWriter::new(&self.lines[0], &self.lines[1], &self.lines[2])
            .field("Mass", &mass)
            .cell_scalar("SAR", &self.values, Layout::ZFastest)
            .cell_scalar("class", &class, Layout::ZFastest)
            .write(path)
    }

    /// Writes the field as an HDF5 file (feature `hdf5`), layout
    /// `format = "ems-sar"`, `version = 1`: root attributes `format`,
    /// `version`, `mass` (kg, 0 for local SAR) and `unit` (`"W/kg"`);
    /// `/mesh/{x,y,z}` the cell boundaries \[m\]; `/SAR` f64
    /// `[nx, ny, nz]` and `/class` u8 `[nx, ny, nz]` (C order, element
    /// `[i, j, k]` is cell `(i, j, k)`).
    ///
    /// # Errors
    /// I/O and encoding errors.
    #[cfg(feature = "hdf5")]
    pub fn write_h5(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut f = hdf5rs::write::FileWriter::create(path.as_ref()).map_err(h5)?;
        self.fill_h5(&mut f).map_err(h5)?;
        f.finish().map_err(h5)?;
        Ok(())
    }

    /// The HDF5 file of [`write_h5`](Self::write_h5) in memory.
    ///
    /// # Errors
    /// Encoding errors.
    #[cfg(feature = "hdf5")]
    pub fn h5_bytes(&self) -> Result<Vec<u8>> {
        let mut f = hdf5rs::write::FileWriter::in_memory();
        self.fill_h5(&mut f).map_err(h5)?;
        f.finish().map_err(h5)
    }

    #[cfg(feature = "hdf5")]
    fn fill_h5<S: hdf5rs::write::WriteAt>(
        &self,
        f: &mut hdf5rs::write::FileWriter<S>,
    ) -> hdf5rs::Result<()> {
        let mut root = f.root();
        root.set_attr("format", "ems-sar")?;
        root.set_attr("version", &1u32)?;
        root.set_attr("mass", &self.mass.unwrap_or(0.0))?;
        root.set_attr("unit", "W/kg")?;
        let mut mesh = root.create_group("mesh")?;
        for (a, n) in ["x", "y", "z"].into_iter().enumerate() {
            let l = &self.lines[a];
            mesh.new_dataset::<f64>(n)
                .shape(&[l.len() as u64])
                .write(l)?
                .set_attr("unit", "m")?;
        }
        let shape = self.cells().map(|n| n as u64);
        root.new_dataset::<f64>("SAR")
            .shape(&shape)
            .write(&self.values)?
            .set_attr("unit", "W/kg")?;
        let class: Vec<u8> = self.classes.iter().map(|&c| c as u8).collect();
        root.new_dataset::<u8>("class")
            .shape(&shape)
            .write(&class)?;
        Ok(())
    }
}

#[cfg(feature = "hdf5")]
fn h5(e: hdf5rs::Error) -> Error {
    Error::Io(format!("HDF5: {e}"))
}

// ---------------------------------------------------------------------------
// Averaging.

/// Prefix sums of per-cell `[mass, tissue power, tissue volume]`.
struct Prefix {
    c: [usize; 3],
    data: Vec<[f64; 3]>,
}

impl Prefix {
    fn new(c: [usize; 3], cell: impl Fn(usize) -> [f64; 3]) -> Self {
        let n = [c[0] + 1, c[1] + 1, c[2] + 1];
        let idx = |i: usize, j: usize, k: usize| (i * n[1] + j) * n[2] + k;
        let mut data = vec![[0.0; 3]; n[0] * n[1] * n[2]];
        for i in 1..n[0] {
            for j in 1..n[1] {
                for k in 1..n[2] {
                    let v = cell(((i - 1) * c[1] + j - 1) * c[2] + k - 1);
                    let mut s = [0.0; 3];
                    for (q, s) in s.iter_mut().enumerate() {
                        *s = v[q]
                            + data[idx(i - 1, j, k)][q]
                            + data[idx(i, j - 1, k)][q]
                            + data[idx(i, j, k - 1)][q]
                            - data[idx(i - 1, j - 1, k)][q]
                            - data[idx(i - 1, j, k - 1)][q]
                            - data[idx(i, j - 1, k - 1)][q]
                            + data[idx(i - 1, j - 1, k - 1)][q];
                    }
                    data[idx(i, j, k)] = s;
                }
            }
        }
        Self { c, data }
    }

    /// Sum over the cells `r[0] × r[1] × r[2]` (half-open ranges).
    fn sum(&self, r: [(usize, usize); 3]) -> [f64; 3] {
        let n = [self.c[0] + 1, self.c[1] + 1, self.c[2] + 1];
        let at = |i: usize, j: usize, k: usize| &self.data[(i * n[1] + j) * n[2] + k];
        let ((i0, i1), (j0, j1), (k0, k1)) = (r[0], r[1], r[2]);
        let mut s = [0.0; 3];
        for (q, s) in s.iter_mut().enumerate() {
            *s = at(i1, j1, k1)[q] - at(i0, j1, k1)[q] - at(i1, j0, k1)[q] - at(i1, j1, k0)[q]
                + at(i0, j0, k1)[q]
                + at(i0, j1, k0)[q]
                + at(i1, j0, k0)[q]
                - at(i0, j0, k0)[q];
        }
        s
    }
}

/// Up to three `(cell range, weight)` pairs and how many are used.
type Segments = ([((usize, usize), f64); 3], usize);

/// Up to three `(cell range, weight)` segments covering `[a, b]` on one
/// axis with `lines`; the weight is the covered fraction of each cell.
fn segments(lines: &[f64], a: f64, b: f64) -> Segments {
    let mut out = [((0, 0), 0.0); 3];
    let n = lines.len() - 1;
    let (a, b) = (a.max(lines[0]), b.min(lines[n]));
    if b <= a || b.is_nan() || a.is_nan() {
        return (out, 0);
    }
    // Cell containing a (largest i with lines[i] ≤ a) and b (smallest i
    // with lines[i + 1] ≥ b).
    let i0 = lines.partition_point(|&x| x <= a).clamp(1, n) - 1;
    let i1 = lines.partition_point(|&x| x < b).clamp(1, n) - 1;
    let d = |i: usize| lines[i + 1] - lines[i];
    if i0 == i1 {
        out[0] = ((i0, i0 + 1), (b - a) / d(i0));
        return (out, 1);
    }
    let mut m = 0;
    out[m] = ((i0, i0 + 1), (lines[i0 + 1] - a) / d(i0));
    m += 1;
    if i1 > i0 + 1 {
        out[m] = ((i0 + 1, i1), 1.0);
        m += 1;
    }
    out[m] = ((i1, i1 + 1), (b - lines[i1]) / d(i1));
    (out, m + 1)
}

/// Where a cube sits relative to its reference voxel.
#[derive(Debug, Clone, Copy)]
enum Placement {
    /// Centred on the voxel centre.
    Centered,
    /// Extends from the voxel's lower (`false`) or upper (`true`) face
    /// along `axis`, centred across.
    Face { axis: usize, up: bool },
}

#[derive(Debug, Clone, Copy)]
struct Cube {
    side: f64,
    sar: f64,
    air: f64,
    /// Tissue mass \[kg\].
    mass: f64,
    /// Volume within the domain \[m³\].
    volume: f64,
}

struct Averager<'a> {
    v: &'a SarVolume,
    mass: f64,
    opts: AveragingOptions,
    c: [usize; 3],
    prefix: Prefix,
    centers: [Vec<f64>; 3],
    widths: [Vec<f64>; 3],
    total_mass: f64,
}

impl<'a> Averager<'a> {
    fn new(v: &'a SarVolume, mass: f64, opts: &AveragingOptions) -> Self {
        let c = v.cells();
        let vol = v.cell_volumes();
        let prefix = Prefix::new(c, |i| {
            if v.density[i] > 0.0 {
                [v.density[i] * vol[i], v.power[i] * vol[i], vol[i]]
            } else {
                [0.0; 3]
            }
        });
        let centers =
            core::array::from_fn(|a| v.lines[a].windows(2).map(|w| 0.5 * (w[0] + w[1])).collect());
        let widths = core::array::from_fn(|a| v.lines[a].windows(2).map(|w| w[1] - w[0]).collect());
        Self {
            v,
            mass,
            opts: *opts,
            c,
            prefix,
            centers,
            widths,
            total_mass: v.tissue_mass(),
        }
    }

    fn cell(&self, i: usize) -> [usize; 3] {
        let c = self.c;
        [i / (c[1] * c[2]), (i / c[2]) % c[1], i % c[2]]
    }

    /// `[mass, power, tissue volume]` in the box `lo..hi`.
    fn integrate(&self, lo: [f64; 3], hi: [f64; 3]) -> [f64; 3] {
        let s: [_; 3] = core::array::from_fn(|a| segments(&self.v.lines[a], lo[a], hi[a]));
        let mut out = [0.0; 3];
        for &(rx, wx) in &s[0].0[..s[0].1] {
            for &(ry, wy) in &s[1].0[..s[1].1] {
                for &(rz, wz) in &s[2].0[..s[2].1] {
                    let w = wx * wy * wz;
                    let v = self.prefix.sum([rx, ry, rz]);
                    for q in 0..3 {
                        out[q] += w * v[q];
                    }
                }
            }
        }
        out
    }

    /// Bounds of the cube of side `s` placed at `cell`.
    fn bounds(&self, cell: [usize; 3], p: Placement, s: f64) -> ([f64; 3], [f64; 3]) {
        let mut lo = [0.0; 3];
        let mut hi = [0.0; 3];
        for a in 0..3 {
            let x = self.centers[a][cell[a]];
            let h = 0.5 * self.widths[a][cell[a]];
            (lo[a], hi[a]) = match p {
                Placement::Face { axis, up } if axis == a => {
                    if up {
                        (x + h - s, x + h)
                    } else {
                        (x - h, x - h + s)
                    }
                }
                _ => (x - 0.5 * s, x + 0.5 * s),
            };
        }
        (lo, hi)
    }

    /// The cube of tissue mass `self.mass` at `cell` with placement `p`
    /// (`None` if the tissue is too light).
    fn cube(&self, cell: [usize; 3], p: Placement) -> Option<Cube> {
        let m = self.mass;
        if self.total_mass < m * (1.0 - 1e-12) {
            return None;
        }
        let eval = |s: f64| {
            let (lo, hi) = self.bounds(cell, p, s);
            self.integrate(lo, hi)
        };
        let f = |s: f64| eval(s)[0] - m;
        // Bracket: mass grows with the side, and is total_mass once the cube
        // covers the domain.
        let span: f64 = (0..3)
            .map(|a| self.v.lines[a][self.c[a]] - self.v.lines[a][0])
            .fold(0.0, f64::max);
        let s_max = 2.0 * span + 1e-300;
        let i = (cell[0] * self.c[1] + cell[1]) * self.c[2] + cell[2];
        let rho = self.v.density[i].max(1e-300);
        let mut hi = (m / rho).cbrt().min(s_max);
        let mut f_hi = f(hi);
        while f_hi < 0.0 {
            if hi >= s_max {
                return None;
            }
            hi = (2.0 * hi).min(s_max);
            f_hi = f(hi);
        }
        let (mut lo, mut f_lo) = (0.0, -m);
        // Illinois regula falsi.
        let mut side = 0;
        let mut s = hi;
        for _ in 0..200 {
            s = (lo * f_hi - hi * f_lo) / (f_hi - f_lo);
            if !(s > lo && s < hi) {
                s = 0.5 * (lo + hi);
            }
            let fs = f(s);
            if fs.abs() <= 1e-12 * m || hi - lo <= 1e-13 * hi {
                break;
            }
            if fs > 0.0 {
                hi = s;
                f_hi = fs;
                if side == 1 {
                    f_lo *= 0.5;
                }
                side = 1;
            } else {
                lo = s;
                f_lo = fs;
                if side == -1 {
                    f_hi *= 0.5;
                }
                side = -1;
            }
        }
        let [mass, power, tissue] = eval(s);
        if mass <= 0.0 {
            return None;
        }
        Some(Cube {
            side: s,
            sar: power / mass,
            air: (1.0 - tissue / (s * s * s)).max(0.0),
            mass,
            volume: {
                let (lo, hi) = self.bounds(cell, p, s);
                (0..3)
                    .map(|a| {
                        let l = &self.v.lines[a];
                        (hi[a].min(l[self.c[a]]) - lo[a].max(l[0])).max(0.0)
                    })
                    .product()
            },
        })
    }

    fn threads(&self) -> usize {
        match self.opts.threads {
            0 => std::thread::available_parallelism().map_or(1, usize::from),
            n => n,
        }
    }

    /// `f(i)` for every `i < n`, in parallel.
    fn par_map<T: Send + Clone + Default>(
        &self,
        n: usize,
        f: impl Fn(usize) -> T + Sync,
    ) -> Vec<T> {
        let t = self.threads().clamp(1, n.max(1));
        let mut out = vec![T::default(); n];
        let chunk = n.div_ceil(t).max(1);
        std::thread::scope(|sc| {
            for (c, part) in out.chunks_mut(chunk).enumerate() {
                let f = &f;
                sc.spawn(move || {
                    for (o, slot) in part.iter_mut().enumerate() {
                        *slot = f(c * chunk + o);
                    }
                });
            }
        });
        out
    }

    fn run(&self) -> SarField {
        let n: usize = self.c.iter().product();
        let tissue = |i: usize| self.v.density[i] > 0.0;
        // 1. Centred cubes.
        let centred: Vec<Option<Cube>> = self.par_map(n, |i| {
            if !tissue(i) {
                return None;
            }
            self.cube(self.cell(i), Placement::Centered)
                .filter(|q| q.air <= self.opts.max_air_fraction)
        });
        let half_max = centred
            .iter()
            .flatten()
            .map(|q| 0.5 * q.side)
            .fold(0.0, f64::max);
        // 2. Covered voxels; 3. boundary cubes.
        let rest: Vec<(f64, VoxelClass, [f64; 2])> = self.par_map(n, |i| {
            if !tissue(i) {
                return (0.0, VoxelClass::NoTissue, [0.0; 2]);
            }
            if let Some(q) = centred[i] {
                return (q.sar, VoxelClass::Centered, [q.mass, q.volume]);
            }
            let cell = self.cell(i);
            let x = [0, 1, 2].map(|a| self.centers[a][cell[a]]);
            let range = |a: usize| {
                let c = &self.centers[a];
                let lo = c.partition_point(|&y| y < x[a] - half_max);
                let hi = c.partition_point(|&y| y <= x[a] + half_max);
                lo..hi
            };
            let mut best: Option<f64> = None;
            for p in range(0) {
                for q in range(1) {
                    for r in range(2) {
                        let j = (p * self.c[1] + q) * self.c[2] + r;
                        if let Some(Cube { sar, side, .. }) = centred[j] {
                            let h = 0.5 * side;
                            let inside = (0..3).all(|a| {
                                let y = self.centers[a][[p, q, r][a]];
                                (x[a] - y).abs() <= h
                            });
                            if inside && best.is_none_or(|b| sar > b) {
                                best = Some(sar);
                            }
                        }
                    }
                }
            }
            if let Some(b) = best {
                return (b, VoxelClass::Covered, [0.0; 2]);
            }
            let mut pick: Option<Cube> = None;
            for axis in 0..3 {
                for up in [false, true] {
                    if let Some(q) = self.cube(cell, Placement::Face { axis, up }) {
                        let better = match pick {
                            None => true,
                            Some(b) => q.air < b.air || (q.air == b.air && q.sar > b.sar),
                        };
                        if better {
                            pick = Some(q);
                        }
                    }
                }
            }
            match pick {
                Some(q) => (q.sar, VoxelClass::Boundary, [q.mass, q.volume]),
                None => (0.0, VoxelClass::Unresolved, [0.0; 2]),
            }
        });
        let mut values = Vec::with_capacity(n);
        let mut classes = Vec::with_capacity(n);
        let mut cube_mass = Vec::with_capacity(n);
        let mut cube_volume = Vec::with_capacity(n);
        for (v, c, [m, vol]) in rest {
            values.push(v);
            classes.push(c);
            cube_mass.push(m);
            cube_volume.push(vol);
        }
        SarField {
            lines: self.v.lines.clone(),
            values,
            classes,
            mass: Some(self.mass),
            cube_mass,
            cube_volume,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize, d: f64) -> [Vec<f64>; 3] {
        [(); 3].map(|_| (0..=n).map(|i| i as f64 * d).collect())
    }

    #[test]
    fn segments_cover_fractions() {
        let l = [0.0, 1.0, 3.0, 4.0];
        let (s, n) = segments(&l, 0.5, 3.5);
        assert_eq!(n, 3);
        assert_eq!(s[0], ((0, 1), 0.5));
        assert_eq!(s[1], ((1, 2), 1.0));
        assert_eq!(s[2], ((2, 3), 0.5));
        let (s, n) = segments(&l, 1.5, 2.5);
        assert_eq!(n, 1);
        assert_eq!(s[0], ((1, 2), 0.5));
        // Exactly on lines, and clipped to the domain.
        let (s, n) = segments(&l, -1.0, 3.0);
        assert_eq!(n, 2);
        assert_eq!(s[0], ((0, 1), 1.0));
        assert_eq!(s[1], ((1, 2), 1.0));
        assert_eq!(segments(&l, 5.0, 6.0).1, 0);
    }

    #[test]
    fn constant_field_average_equals_local() {
        let n = 12;
        let v = SarVolume::new(
            uniform(n, 2e-3),
            vec![3.0; n * n * n],
            vec![1000.0; n * n * n],
        )
        .unwrap();
        // 1 g in 1000 kg/m³ is a 1 cm cube = 5 voxels of 2 mm.
        let a = v.averaged(1e-3).unwrap();
        for (&s, &k) in a.values().iter().zip(a.classes()) {
            assert!(matches!(
                k,
                VoxelClass::Centered | VoxelClass::Covered | VoxelClass::Boundary
            ));
            assert!((s - 3e-3).abs() < 1e-15, "{s}");
        }
        // The centre has a valid centred cube, the corner none.
        let c = |i: usize, j: usize, k: usize| a.classes()[(i * n + j) * n + k];
        assert_eq!(c(6, 6, 6), VoxelClass::Centered);
        assert_eq!(c(0, 6, 6), VoxelClass::Covered);
        assert_eq!(c(0, 0, 0), VoxelClass::Covered);
    }

    #[test]
    fn linear_gradient_average_is_centre_value() {
        // p = a + b·x: the average over a cube centred at x0 is p(x0).
        let n = 30;
        let d = 1e-3;
        let lines = uniform(n, d);
        let mut p = Vec::new();
        for i in 0..n {
            for _ in 0..n * n {
                p.push(1.0 + 1000.0 * (i as f64 + 0.5) * d);
            }
        }
        let v = SarVolume::new(lines, p.clone(), vec![1000.0; n * n * n]).unwrap();
        let a = v.averaged(1e-3).unwrap(); // 10 mm cube
        let i = (15 * n + 15) * n + 15;
        assert_eq!(a.classes()[i], VoxelClass::Centered);
        let want = p[i] / 1000.0;
        assert!(
            (a.values()[i] / want - 1.0).abs() < 1e-12,
            "{} {want}",
            a.values()[i]
        );
        // A non-cubic cube size (fractional voxels): 1.5 g → 11.45 mm.
        let a = v.averaged(1.5e-3).unwrap();
        assert!((a.values()[i] / want - 1.0).abs() < 1e-11);
        // Voxels near the x faces are covered by the hottest valid cube.
        let hot = (29 * n + 15) * n + 15;
        assert_eq!(a.classes()[hot], VoxelClass::Covered);
        assert!(a.values()[hot] > a.values()[i]);
    }

    #[test]
    fn tissue_boundary_and_air_rule() {
        // Tissue fills x < 10 voxels of a 20-voxel box; uniform power.
        let n = 20;
        let d = 1e-3;
        let mut rho = vec![0.0; n * n * n];
        let mut p = vec![0.0; n * n * n];
        rho[..10 * n * n].fill(1000.0);
        p[..10 * n * n].fill(2.0);
        // Lossy air must not count as tissue power.
        p[15 * n * n] = 1e6;
        let v = SarVolume::new(uniform(n, d), p, rho).unwrap();
        assert!((v.tissue_mass() - 1000.0 * 10.0 * 20.0 * 20.0 * 1e-9).abs() < 1e-15);
        assert!(v.total_power() > v.tissue_power());
        let a = v.averaged(0.125e-3).unwrap(); // 5 mm cube
        let at = |i: usize| a.classes()[(i * n + 10) * n + 10];
        assert_eq!(at(5), VoxelClass::Centered);
        // The cube centred on voxel 9 reaches 2 mm into air: 2.5/5 = 50 %
        // air at full tissue mass is impossible, it grows instead; with
        // 10 % allowed it is invalid.
        assert_eq!(at(9), VoxelClass::Covered);
        assert_eq!(at(15), VoxelClass::NoTissue);
        assert!(
            a.values()
                .iter()
                .zip(a.classes())
                .all(|(&s, &k)| { k == VoxelClass::NoTissue || (s - 2e-3).abs() < 1e-15 })
        );
        // A thin tissue slab (1 voxel) has no valid cube at all: boundary
        // cubes.
        let mut rho = vec![0.0; n * n * n];
        rho[..n * n].fill(1000.0);
        let v = SarVolume::new(uniform(n, d), vec![1.0; n * n * n], rho).unwrap();
        let a = v.averaged(0.125e-3).unwrap();
        let k = a.classes()[10 * n + 10];
        assert_eq!(k, VoxelClass::Boundary);
        assert!((a.values()[10 * n + 10] - 1e-3).abs() < 1e-15);
        // Heavier than all tissue: unresolved.
        let a = v.averaged(1.0).unwrap();
        assert!(
            a.classes()
                .iter()
                .all(|&k| matches!(k, VoxelClass::Unresolved | VoxelClass::NoTissue))
        );
        assert!(a.max().is_none());
    }

    #[test]
    fn non_uniform_mesh_mass_and_threads() {
        // Graded lines; constant density and power: every average equals
        // the local value, independent of the thread count.
        let l: Vec<f64> = (0..=16).map(|i| (i as f64).powf(1.3) * 1e-3).collect();
        let lines = [l.clone(), l.clone(), l];
        let n = 16 * 16 * 16;
        let v = SarVolume::new(lines, vec![5.0; n], vec![1050.0; n]).unwrap();
        let one = v
            .averaged_with(
                1e-3,
                &AveragingOptions {
                    threads: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        let many = v
            .averaged_with(
                1e-3,
                &AveragingOptions {
                    threads: 7,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(one, many);
        for &s in one.values() {
            assert!((s - 5.0 / 1050.0).abs() < 1e-14);
        }
    }

    #[test]
    fn local_and_edge_power() {
        // One cell 1×2×4 mm, σ = (1, 2, 3); E_x = 1 on all four x edges,
        // E_y = 2 on two of the four y edges, E_z = 0.
        let lines = [vec![0.0, 1e-3], vec![0.0, 2e-3], vec![0.0, 4e-3]];
        let one = Complex64::new(1.0, 0.0);
        let ex = vec![one; 4];
        let ey = vec![
            Complex64::new(0.0, 2.0),
            Complex64::ZERO,
            Complex64::new(2.0, 0.0),
            Complex64::ZERO,
        ];
        let ez = vec![Complex64::ZERO; 4];
        let v =
            SarVolume::from_edge_fields(lines, [&ex, &ey, &ez], &[[1.0, 2.0, 3.0]], vec![500.0])
                .unwrap();
        // p = ½·(1·1 + 2·¼·8) = 2.5 W/m³.
        assert!((v.power_density()[0] - 2.5).abs() < 1e-15);
        assert!((v.total_power() - 2.5 * 8e-9).abs() < 1e-22);
        let l = v.local();
        assert_eq!(l.values(), &[2.5 / 500.0]);
        let m = l.max().unwrap();
        assert_eq!(m.cell, [0, 0, 0]);
        assert_eq!(m.center, [0.5e-3, 1e-3, 2e-3]);
        assert!((v.scaled(2.0).power_density()[0] - 5.0).abs() < 1e-15);
        assert!(
            SarVolume::from_edge_fields(
                [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]],
                [&ex, &ey, &ez[..3]],
                &[[1.0; 3]],
                vec![1.0]
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_inputs() {
        let l = || [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        assert!(SarVolume::new(l(), vec![-1.0], vec![1.0]).is_err());
        assert!(SarVolume::new(l(), vec![1.0, 2.0], vec![1.0]).is_err());
        assert!(
            SarVolume::new(
                [vec![0.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0]],
                vec![1.0],
                vec![1.0]
            )
            .is_err()
        );
        let v = SarVolume::new(l(), vec![1.0], vec![1.0]).unwrap();
        assert!(v.averaged(0.0).is_err());
        assert!(
            v.averaged_with(
                1.0,
                &AveragingOptions {
                    max_air_fraction: 1.5,
                    threads: 0
                }
            )
            .is_err()
        );
    }
}
