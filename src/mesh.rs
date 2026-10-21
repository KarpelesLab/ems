//! Solver-aware mesh checks and suggestions (design §4, §5.6, §7).
//!
//! Mesh *generation* is csx's business ([`csx::mesh::MeshBuilder`],
//! [`csx::mesh::detect_edges`], [`csx::grid::smooth_lines`]). This module
//! looks at a finished mesh with the eyes of the FDTD solver and reports
//! what is likely to go wrong:
//!
//! * **Resolution.** Every cell must be smaller than `λ_min/N`, where
//!   `λ_min = c0 / (f_max·n)` with `n = sqrt(ε_r·μ_r)` the largest refractive
//!   index of the materials present in that cell slab, and `f_max` from
//!   [`Fdtd::get_max_frequency`]. The rule of thumb `N ≥ 10…20` follows
//!   from the numerical dispersion analysis of the Yee scheme (Taflove &
//!   Hagness, *Computational Electrodynamics*, 3rd ed., ch. 4: the phase
//!   velocity error is below about 1 % for 10 cells per wavelength and
//!   falls off quadratically).
//! * **Grading.** Neighbouring cells should not differ in size by more than
//!   a factor of about 1.5: on a non-uniform mesh the scheme is only
//!   first-order accurate at a size jump, and large jumps reflect waves
//!   (Taflove & Hagness, non-uniform grids in ch. 3; Monk & Süli 1994 on
//!   the supraconvergence of the Yee scheme on non-uniform meshes).
//! * **Time step.** The Courant limit (design §5.4) is set by the smallest
//!   cell on every axis; a few tiny cells can make the whole simulation
//!   many times slower. The report gives `dt`, the limiting cells and the
//!   ratio to the time step a typical (median) cell would allow.
//!   On a cylindrical grid ([`Grid::cylindrical`](crate::engine::Grid::cylindrical))
//!   the limit comes from the discrete operator and is usually set by the
//!   short arcs `ρ·Δα` next to the axis (design §5.15).
//! * **Cylindrical grids.** Messages name the axes `x`, `y`, `z` for `ρ`,
//!   `α`, `z`; α positions are in radians. The resolution check uses the
//!   largest arc `ρ_max·Δα` of each α cell.
//! * **PML** (design §5.6): the layer occupies the outermost `n` mesh cells
//!   of a face; the mesh needs enough cells for it, structure primitives
//!   should stay out of it, and the cells in it should be roughly uniform
//!   (the conductivity profile assumes slowly varying cells; Berenger 1994,
//!   Roden & Gedney 2000).
//! * **Sheets.** Zero-thickness metal or conducting sheets (and axis-aligned
//!   lines) must lie on a mesh plane (line) or csx's discretisation drops
//!   them.
//! * **Cost.** Cell count, estimated memory, memory traffic per step and the
//!   number of steps implied by the settings.
//!
//! [`check`] works on the model and its csx grid; [`check_compiled`]
//! reuses an already [compiled](crate::compile::compile) model (the actual
//! discretised materials and csx's discretisation warnings).
//!
//! All coordinates in messages and [`Location`]s are csx **drawing units**
//! (the unit of the mesh lines); lengths in [`MeshStats`] are metres.
//!
//! ```
//! use csx::primitives::BoxShape;
//! use csx::{ContinuousStructure, Primitive, Property};
//! use ems::prelude::*;
//!
//! let mut model = ContinuousStructure::new();
//! model.grid.delta_unit = 1e-3; // mm
//! for a in 0..3 {
//!     model.grid.set_lines(a, (0..=40).map(|i| i as f64))?;
//! }
//! let fdtd = Fdtd::new()
//!     .waveform(Waveform::gaussian(5e9, 5e9)) // f_max = 10 GHz, λ = 30 mm
//!     .boundaries(Boundaries::all(Boundary::pml(8)));
//! let report = ems::mesh::check(&model, &fdtd)?;
//! assert!(report.is_ok());
//!
//! // csx's mesher wants the maximum cell size in drawing units.
//! let max_res = ems::mesh::suggest_max_cell(&model, &fdtd, 20.0)?;
//! assert!((max_res - 1.499).abs() < 1e-3); // 29.98 mm / 20
//! # Ok::<(), ems::Error>(())
//! ```

use core::fmt;

use crate::axis::Axis;
use crate::compile::{Compiled, grid_from_csx};
use crate::constants::C0;
use crate::engine::{EdgeMaterials, Grid, TimeStep};
use crate::fdtd::{Boundary, Face, Fdtd};
use crate::{Error, Result};

use csx::properties::MaterialQuantity;
use csx::{BoundingBox, ContinuousStructure, CoordSystem, Primitive, PropertyKind, Vec3};

/// Severity of a [`MeshIssue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational (statistics, settings that were not checked).
    Info,
    /// Likely to reduce accuracy or performance.
    Warning,
    /// The simulation will be wrong (or fail).
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Error => "error",
        })
    }
}

/// Where a [`MeshIssue`] is located. Coordinates are drawing units.
#[derive(Debug, Clone, PartialEq)]
pub enum Location {
    /// The model or mesh as a whole.
    Global,
    /// Cell `index` along `axis`, spanning mesh lines `from..to`.
    Cell {
        /// Axis.
        axis: Axis,
        /// Cell index (between mesh lines `index` and `index + 1`).
        index: usize,
        /// Lower mesh line.
        from: f64,
        /// Upper mesh line.
        to: f64,
    },
    /// A domain face (boundary condition).
    Face(Face),
    /// Primitive `index` of the property named `property`.
    Primitive {
        /// Property name.
        property: String,
        /// Index of the primitive within the property.
        index: usize,
    },
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Location::Global => f.write_str("mesh"),
            Location::Cell {
                axis,
                index,
                from,
                to,
            } => write!(f, "{axis} cell {index} ({axis} = {from} .. {to})"),
            Location::Face(face) => write!(f, "{face} face"),
            Location::Primitive { property, index } => {
                write!(f, "property {property:?} primitive #{index}")
            }
        }
    }
}

/// One finding of a mesh check.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshIssue {
    /// Severity.
    pub severity: Severity,
    /// Human-readable description.
    pub message: String,
    /// Where it applies.
    pub location: Location,
}

impl fmt::Display for MeshIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.severity, self.message)?;
        if self.location != Location::Global {
            write!(f, " [{}]", self.location)?;
        }
        Ok(())
    }
}

/// Mesh statistics and cost estimates gathered by a check.
///
/// The memory figures are estimates for the current engine layout: per
/// node 6 field values and 12 update coefficients (`f32`, 72 bytes), plus
/// 4 `f32` CPML auxiliary values per node of every PML slab; compiling
/// additionally holds the csx and engine material arrays (about 210 bytes
/// per node) at its peak.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshStats {
    /// Mesh lines per axis.
    pub nodes: [usize; 3],
    /// Total number of primary cells.
    pub cells: usize,
    /// Smallest cell per axis \[m\].
    pub min_cell: [f64; 3],
    /// Largest cell per axis \[m\].
    pub max_cell: [f64; 3],
    /// Index of the smallest cell per axis (the cells limiting `dt`).
    pub min_cell_index: [usize; 3],
    /// Highest frequency of interest \[Hz\], if known.
    pub f_max: Option<f64>,
    /// Shortest wavelength in the model at `f_max` \[m\], if known.
    pub lambda_min: Option<f64>,
    /// Worst resolution: smallest number of cells per local wavelength
    /// found in any cell (if `f_max` is known).
    pub min_cells_per_wavelength: Option<f64>,
    /// Time step the solver will use \[s\].
    pub dt: f64,
    /// Time step the median cell sizes would allow (same Courant factor)
    /// \[s\].
    pub dt_typical: f64,
    /// Estimated engine memory \[bytes\].
    pub engine_bytes: u64,
    /// Estimated peak memory while compiling \[bytes\].
    pub compile_bytes: u64,
    /// Estimated memory traffic per time step \[bytes\] (every field and
    /// coefficient read once, every field written once).
    pub bytes_per_step: u64,
    /// Step limit from the settings ([`Fdtd::timestep_limit`]).
    pub max_steps: u64,
    /// Steps needed for the excitation to end, if it ends.
    pub excitation_steps: Option<u64>,
}

/// Result of a mesh check: issues plus [`MeshStats`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshReport {
    /// All findings, in check order.
    pub issues: Vec<MeshIssue>,
    /// Statistics and estimates.
    pub stats: MeshStats,
}

impl MeshReport {
    /// `true` if there is no [`Severity::Error`] issue.
    pub fn is_ok(&self) -> bool {
        self.errors().next().is_none()
    }

    /// Issues of exactly `severity`.
    pub fn with_severity(&self, severity: Severity) -> impl Iterator<Item = &MeshIssue> {
        self.issues.iter().filter(move |i| i.severity == severity)
    }

    /// The errors.
    pub fn errors(&self) -> impl Iterator<Item = &MeshIssue> {
        self.with_severity(Severity::Error)
    }

    /// The warnings.
    pub fn warnings(&self) -> impl Iterator<Item = &MeshIssue> {
        self.with_severity(Severity::Warning)
    }

    /// The informational items.
    pub fn infos(&self) -> impl Iterator<Item = &MeshIssue> {
        self.with_severity(Severity::Info)
    }

    /// One line per issue of at least `min` severity, prefixed with
    /// `"mesh "` (for appending to compile warnings).
    pub fn messages(&self, min: Severity) -> Vec<String> {
        self.issues
            .iter()
            .filter(|i| i.severity >= min)
            .map(|i| format!("mesh {i}"))
            .collect()
    }

    fn push(&mut self, severity: Severity, location: Location, message: String) {
        self.issues.push(MeshIssue {
            severity,
            message,
            location,
        });
    }
}

fn mib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

impl fmt::Display for MeshReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.stats;
        let (ne, nw) = (self.errors().count(), self.warnings().count());
        if ne == 0 && nw == 0 {
            writeln!(f, "mesh check: OK")?;
        } else {
            writeln!(f, "mesh check: {ne} error(s), {nw} warning(s)")?;
        }
        let c = s.nodes.map(|n| n.saturating_sub(1));
        writeln!(
            f,
            "  cells: {} x {} x {} = {} ({} nodes)",
            c[0],
            c[1],
            c[2],
            s.cells,
            s.nodes.iter().product::<usize>()
        )?;
        writeln!(
            f,
            "  cell size [m]: x {:.3e} .. {:.3e}, y {:.3e} .. {:.3e}, z {:.3e} .. {:.3e}",
            s.min_cell[0],
            s.max_cell[0],
            s.min_cell[1],
            s.max_cell[1],
            s.min_cell[2],
            s.max_cell[2]
        )?;
        match (s.f_max, s.lambda_min, s.min_cells_per_wavelength) {
            (Some(fm), Some(l), Some(n)) => writeln!(
                f,
                "  f_max {fm:.4e} Hz, lambda_min {l:.4e} m, worst resolution {n:.1} cells/lambda"
            )?,
            _ => writeln!(f, "  f_max unknown: resolution not checked")?,
        }
        writeln!(
            f,
            "  dt {:.4e} s ({:.2} x the median-cell time step)",
            s.dt,
            if s.dt_typical > 0.0 {
                s.dt / s.dt_typical
            } else {
                0.0
            }
        )?;
        write!(
            f,
            "  memory ~{:.1} MiB (compile peak ~{:.1} MiB), ~{:.1} MiB traffic/step, up to {} steps",
            mib(s.engine_bytes),
            mib(s.compile_bytes),
            mib(s.bytes_per_step),
            s.max_steps
        )?;
        if let Some(n) = s.excitation_steps {
            write!(f, " (excitation: {n} steps)")?;
        }
        writeln!(f)?;
        for i in &self.issues {
            writeln!(f, "  {i}")?;
        }
        Ok(())
    }
}

/// Thresholds of the mesh checks.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckOptions {
    /// Required cells per local wavelength at `f_max` (default 20); fewer
    /// gives a [`Severity::Warning`].
    pub cells_per_wavelength: f64,
    /// Below this many cells per wavelength (default 10) the issue is a
    /// [`Severity::Error`].
    pub min_cells_per_wavelength: f64,
    /// Largest allowed neighbour cell size ratio (default 1.5).
    pub max_grading: f64,
    /// Warn when `dt` is below this fraction of the time step the median
    /// cells would allow (default 0.2).
    pub timestep_ratio: f64,
    /// Warn when the largest and smallest cell inside a PML differ by more
    /// than this factor (default 2).
    pub pml_max_spread: f64,
    /// Relative tolerance (of the local cell size) for sheets and lines to
    /// count as lying on a mesh line (default 1e-6, as the compile step).
    pub snap_tolerance: f64,
    /// Re-evaluate parametric values on a clone of the model first
    /// (default `true`).
    pub update_model: bool,
}

impl Default for CheckOptions {
    fn default() -> Self {
        Self {
            cells_per_wavelength: 20.0,
            min_cells_per_wavelength: 10.0,
            max_grading: 1.5,
            timestep_ratio: 0.2,
            pml_max_spread: 2.0,
            snap_tolerance: 1e-6,
            update_model: true,
        }
    }
}

impl CheckOptions {
    /// Sets [`cells_per_wavelength`](Self::cells_per_wavelength).
    pub fn cells_per_wavelength(mut self, n: f64) -> Self {
        self.cells_per_wavelength = n;
        self
    }

    /// Sets [`min_cells_per_wavelength`](Self::min_cells_per_wavelength).
    pub fn min_cells_per_wavelength(mut self, n: f64) -> Self {
        self.min_cells_per_wavelength = n;
        self
    }

    /// Sets [`max_grading`](Self::max_grading).
    pub fn max_grading(mut self, r: f64) -> Self {
        self.max_grading = r;
        self
    }

    /// Sets [`timestep_ratio`](Self::timestep_ratio).
    pub fn timestep_ratio(mut self, r: f64) -> Self {
        self.timestep_ratio = r;
        self
    }

    /// Sets [`pml_max_spread`](Self::pml_max_spread).
    pub fn pml_max_spread(mut self, r: f64) -> Self {
        self.pml_max_spread = r;
        self
    }

    fn validate(&self) -> Result<()> {
        let pos = |v: f64, what: &str, min: f64| {
            if v.is_finite() && v >= min {
                Ok(())
            } else {
                Err(Error::Invalid(format!(
                    "mesh check: {what} = {v} must be finite and >= {min}"
                )))
            }
        };
        pos(self.cells_per_wavelength, "cells_per_wavelength", 0.0)?;
        pos(
            self.min_cells_per_wavelength,
            "min_cells_per_wavelength",
            0.0,
        )?;
        pos(self.max_grading, "max_grading", 1.0)?;
        pos(self.timestep_ratio, "timestep_ratio", 0.0)?;
        pos(self.pml_max_spread, "pml_max_spread", 1.0)?;
        pos(self.snap_tolerance, "snap_tolerance", 0.0)
    }
}

/// Checks the mesh of `model` for the solver settings `fdtd` with the
/// default [`CheckOptions`]; see the [module docs](self).
///
/// # Errors
/// Only for unusable input: invalid settings, a grid with fewer than two
/// lines on an axis, cylindrical grids ([`Error::Unsupported`]) or model
/// parameter evaluation errors. Everything else is reported in the
/// [`MeshReport`].
pub fn check(model: &ContinuousStructure, fdtd: &Fdtd) -> Result<MeshReport> {
    check_with(model, fdtd, &CheckOptions::default())
}

/// [`check`] with explicit options.
///
/// Material regions are taken from the primitives' bounding boxes
/// (conservative for non-box shapes). Weighted materials are sampled at
/// the corners and the centre of their bounding box; dispersive materials
/// use their static part.
///
/// # Errors
/// See [`check`].
pub fn check_with(
    model: &ContinuousStructure,
    fdtd: &Fdtd,
    opts: &CheckOptions,
) -> Result<MeshReport> {
    opts.validate()?;
    fdtd.validate()?;
    let grid = grid_from_csx(&model.grid)?;
    let grid = crate::compile::cylindrical::multigrid(
        grid,
        &model.grid,
        fdtd.get_multigrid(),
        &mut Vec::new(),
    )?;
    let model = updated(model, opts)?;
    let model = model.as_ref();
    let ctx = Ctx::new(model, &grid, fdtd, opts);
    let index = ctx.slab_index_from_model();
    let mut r = ctx.run(&index, timestep_of(fdtd));
    ctx.check_sheets(&mut r);
    Ok(r)
}

/// Checks an already [compiled](crate::compile::compile) model: the
/// refractive index per cell slab comes from the discretised
/// [`EdgeMaterials`] (what the solver actually sees, including sub-cell
/// averaging), sheet and line snapping problems are taken from csx's
/// discretisation warnings in [`Compiled::warnings`], and the grid, time
/// step and PML faces from `compiled`. `model` is still needed for the PML
/// intrusion check.
///
/// # Errors
/// See [`check`].
pub fn check_compiled(
    model: &ContinuousStructure,
    fdtd: &Fdtd,
    compiled: &Compiled,
    opts: &CheckOptions,
) -> Result<MeshReport> {
    opts.validate()?;
    let model = updated(model, opts)?;
    let model = model.as_ref();
    let mut ctx = Ctx::new(model, &compiled.grid, fdtd, opts);
    ctx.unit = compiled.delta_unit;
    ctx.pml = compiled.pml.iter().map(|&(f, p)| (f, p.cells)).collect();
    // The thin-wire correction scales materials next to the wires; it is
    // not a medium (the wave speed is unchanged), so look past it.
    // Nor are openEMS's sheet capacitances (compile/openems_sheet.rs).
    let index = if compiled.thin_wires.is_empty() && compiled.openems_sheets.is_empty() {
        slab_index_from_materials(&compiled.grid, &compiled.materials)
    } else {
        let mut m = compiled.materials.clone();
        for w in &compiled.thin_wires {
            w.remove(&compiled.grid, &mut m)?;
        }
        for &(axis, node, [eps, _]) in &compiled.openems_sheets {
            m.eps_r_mut(axis)[compiled.grid.index(node)] = eps;
        }
        slab_index_from_materials(&compiled.grid, &m)
    };
    let mut r = ctx.run(&index, compiled.timestep);
    for w in &compiled.warnings {
        let sev = if w.contains("is not on a mesh plane") || w.contains("line not on mesh lines") {
            Severity::Error
        } else if w.contains("staircased") {
            Severity::Info
        } else {
            continue;
        };
        r.push(
            sev,
            Location::Global,
            format!("{CSX_DISCRETISATION_PREFIX}{w}"),
        );
    }
    Ok(r)
}

/// Prefix of the [`MeshIssue::message`]s with which [`check_compiled`]
/// re-reports csx discretisation warnings (the original warnings are in
/// [`Compiled::warnings`]; filter on this prefix to avoid duplicates).
pub const CSX_DISCRETISATION_PREFIX: &str = "csx discretisation: ";

/// Largest cell size (in drawing units, as expected by
/// [`csx::mesh::MeshOptions::max_res`]) that resolves the shortest
/// wavelength in `model` with `n` cells: `c0 / (f_max·n_max·n) /
/// delta_unit`, where `n_max` is the largest refractive index
/// `sqrt(ε_r·μ_r)` of the background and all material properties.
///
/// # Errors
/// [`Error::Invalid`] if `fdtd` has no maximum frequency
/// ([`Fdtd::get_max_frequency`]), `n` is not positive, or the delta unit is
/// not positive.
pub fn suggest_max_cell(model: &ContinuousStructure, fdtd: &Fdtd, n: f64) -> Result<f64> {
    let f = fdtd.get_max_frequency().ok_or_else(|| {
        Error::Invalid("no maximum frequency (set a waveform or Fdtd::max_frequency)".into())
    })?;
    if !(n.is_finite() && n > 0.0) {
        return Err(Error::Invalid(format!(
            "cells per wavelength {n} must be finite and > 0"
        )));
    }
    let unit = model.grid.delta_unit;
    if !(unit.is_finite() && unit > 0.0) {
        return Err(Error::Invalid(format!(
            "grid delta unit {unit} must be finite and > 0"
        )));
    }
    let mut idx = background_index(model);
    for (p, prim) in model.primitives() {
        if let Some(m) = p.kind.material() {
            let b = prim.bounding_box_in(CoordSystem::Cartesian, model.coord_system);
            idx = idx.max(material_index(m, &b));
        }
    }
    Ok(C0 / (f * idx * n) / unit)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn updated<'a>(
    model: &'a ContinuousStructure,
    opts: &CheckOptions,
) -> Result<std::borrow::Cow<'a, ContinuousStructure>> {
    if opts.update_model {
        let mut m = model.clone();
        m.update()?;
        Ok(std::borrow::Cow::Owned(m))
    } else {
        Ok(std::borrow::Cow::Borrowed(model))
    }
}

fn timestep_of(fdtd: &Fdtd) -> TimeStep {
    fdtd.engine_timestep()
}

/// Refractive index of `eps·mu` (1 for non-physical products).
fn index_of(eps: f64, mu: f64) -> f64 {
    let p = eps * mu;
    if p.is_finite() && p > 0.0 {
        p.sqrt()
    } else {
        1.0
    }
}

fn background_index(model: &ContinuousStructure) -> f64 {
    let b = &model.background;
    index_of(b.epsilon.value(), b.mue.value())
}

/// Largest refractive index of a material over its region `b`: the
/// largest ε over the axes times the largest μ (conservative for
/// anisotropic materials); weighted materials are sampled at the corners
/// and centre of `b`.
fn material_index(m: &csx::Material, b: &BoundingBox) -> f64 {
    let max3 = |v: [f64; 3]| v.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let weighted = m.quantity(MaterialQuantity::Epsilon).is_weighted()
        || m.quantity(MaterialQuantity::Mue).is_weighted();
    if !weighted || b.is_empty() || !(b.min.is_finite() && b.max.is_finite()) {
        let e = max3(core::array::from_fn(|d| {
            m.base_value(MaterialQuantity::Epsilon, d)
        }));
        let u = max3(core::array::from_fn(|d| {
            m.base_value(MaterialQuantity::Mue, d)
        }));
        return index_of(e, u);
    }
    let mut pts: Vec<Vec3> = b.corners().to_vec();
    pts.push(b.center());
    pts.into_iter()
        .map(|p| {
            index_of(
                max3(m.values_at(MaterialQuantity::Epsilon, p)),
                max3(m.values_at(MaterialQuantity::Mue, p)),
            )
        })
        .fold(1.0, f64::max)
}

/// Max refractive index per cell slab: `[axis][cell]`.
type SlabIndex = [Vec<f64>; 3];

/// Per-slab refractive index from discretised edge materials: every node
/// contributes `max ε · max μ` of its non-PEC edges / faces to the cells on
/// both sides of it along each axis.
fn slab_index_from_materials(grid: &Grid, m: &EdgeMaterials) -> SlabIndex {
    let n = grid.nodes();
    let mut out: SlabIndex = core::array::from_fn(|a| vec![1.0; n[a] - 1]);
    let mut idx = 0;
    for i in 0..n[0] {
        for j in 0..n[1] {
            for k in 0..n[2] {
                let mut e = f64::NEG_INFINITY;
                let mut u = f64::NEG_INFINITY;
                for ax in Axis::ALL {
                    if !m.pec_mask(ax)[idx] {
                        e = e.max(m.eps_r(ax)[idx]);
                    }
                    u = u.max(m.mu_r(ax)[idx]);
                }
                if e.is_finite() {
                    let v = index_of(e, u);
                    for (a, c) in [i, j, k].into_iter().enumerate() {
                        let s = &mut out[a];
                        if c < s.len() {
                            s[c] = s[c].max(v);
                        }
                        if c > 0 {
                            s[c - 1] = s[c - 1].max(v);
                        }
                    }
                }
                idx += 1;
            }
        }
    }
    out
}

/// Shared state of one check.
struct Ctx<'a> {
    model: &'a ContinuousStructure,
    /// Mesh in metres.
    grid: &'a Grid,
    /// Drawing unit \[m\].
    unit: f64,
    fdtd: &'a Fdtd,
    opts: &'a CheckOptions,
    /// PML faces and their cell counts.
    pml: Vec<(Face, usize)>,
}

impl<'a> Ctx<'a> {
    fn new(
        model: &'a ContinuousStructure,
        grid: &'a Grid,
        fdtd: &'a Fdtd,
        opts: &'a CheckOptions,
    ) -> Self {
        let pml = fdtd
            .get_boundaries()
            .iter()
            .filter_map(|(f, b)| match b {
                Boundary::Pml(p) => Some((f, p.cells)),
                _ => None,
            })
            .collect();
        Self {
            model,
            grid,
            unit: model.grid.delta_unit,
            fdtd,
            opts,
            pml,
        }
    }

    /// Drawing unit of `axis` in grid units: the delta unit, or 1 for the α
    /// axis (radians) of a cylindrical grid.
    fn unit_of(&self, axis: Axis) -> f64 {
        if self.grid.is_cylindrical() && axis == Axis::Y {
            1.0
        } else {
            self.unit
        }
    }

    /// Mesh lines of `axis` in drawing units (radians for α).
    fn du_lines(&self, axis: Axis) -> Vec<f64> {
        let u = self.unit_of(axis);
        self.grid.lines(axis).iter().map(|x| x / u).collect()
    }

    fn cell_loc(&self, axis: Axis, index: usize) -> Location {
        let l = self.grid.lines(axis);
        let u = self.unit_of(axis);
        Location::Cell {
            axis,
            index,
            from: l[index] / u,
            to: l[index + 1] / u,
        }
    }

    /// Physical cell sizes \[m\] along `axis`: [`Grid::deltas`], except
    /// for the α axis of a cylindrical grid, where the largest arc
    /// `ρ_max·Δα` of each cell is used (the wavelength resolution is worst
    /// at the outer radius).
    fn phys_deltas(&self, axis: Axis) -> Vec<f64> {
        let d = self.grid.deltas(axis);
        if self.grid.is_cylindrical() && axis == Axis::Y {
            let r = self.grid.lines(Axis::X);
            let rmax = r[r.len() - 1];
            d.into_iter().map(|x| x * rmax).collect()
        } else {
            d
        }
    }

    /// Bounding box of a primitive in drawing units, in the grid's
    /// coordinate system.
    fn bbox(&self, prim: &Primitive) -> BoundingBox {
        let cs = if self.grid.is_cylindrical() {
            CoordSystem::Cylindrical
        } else {
            CoordSystem::Cartesian
        };
        prim.bounding_box_in(cs, self.model.coord_system)
    }

    /// `true` if `b` overlaps the mesh on every axis
    /// (closed intervals).
    fn overlaps_domain(&self, b: &BoundingBox) -> bool {
        (0..3).all(|a| {
            let l = self.grid.lines(Axis::ALL[a]);
            let u = self.unit_of(Axis::ALL[a]);
            let (lo, hi) = (l[0] / u, l[l.len() - 1] / u);
            let tol = 1e-9 * (hi - lo).abs();
            b.max[a] >= lo - tol && b.min[a] <= hi + tol
        })
    }

    /// Per-slab refractive index from the material primitives' bounding
    /// boxes (plus the background everywhere).
    fn slab_index_from_model(&self) -> SlabIndex {
        let bg = background_index(self.model);
        let mut out: SlabIndex = core::array::from_fn(|a| vec![bg; self.grid.nodes()[a] - 1]);
        for (p, prim) in self.model.primitives() {
            let Some(m) = p.kind.material() else {
                continue;
            };
            let b = self.bbox(prim);
            if b.is_empty() || !self.overlaps_domain(&b) {
                continue;
            }
            let v = material_index(m, &b);
            for axis in Axis::ALL {
                let a = axis.index();
                let l = self.du_lines(axis);
                for (c, s) in out[a].iter_mut().enumerate() {
                    let tol = 1e-9 * (l[c + 1] - l[c]);
                    if l[c + 1] > b.min[a] + tol && l[c] < b.max[a] - tol {
                        *s = s.max(v);
                    }
                }
            }
        }
        out
    }

    fn run(&self, index: &SlabIndex, timestep: TimeStep) -> MeshReport {
        let mut r = MeshReport::default();
        self.stats(&mut r, timestep);
        self.check_resolution(&mut r, index);
        self.check_grading(&mut r);
        self.check_timestep(&mut r);
        self.check_pml(&mut r);
        r
    }

    fn stats(&self, r: &mut MeshReport, timestep: TimeStep) {
        let g = self.grid;
        let s = &mut r.stats;
        s.nodes = g.nodes();
        s.cells = g.num_cells();
        for axis in Axis::ALL {
            let a = axis.index();
            let d = self.phys_deltas(axis);
            let (mut imin, mut dmin, mut dmax) = (0, f64::INFINITY, 0.0f64);
            for (i, &x) in d.iter().enumerate() {
                if x < dmin {
                    (imin, dmin) = (i, x);
                }
                dmax = dmax.max(x);
            }
            s.min_cell[a] = dmin;
            s.max_cell[a] = dmax;
            s.min_cell_index[a] = imin;
        }
        let limit = crate::engine::courant_limit(g);
        let factor = match timestep {
            TimeStep::Courant(f) => f,
            TimeStep::Fixed(dt) => dt / limit,
            TimeStep::OpenEms(o) => o
                .resolve(g, &crate::engine::EdgeMaterials::vacuum(g))
                .map_or(1.0, |dt| dt / limit),
        };
        s.dt = match timestep.resolve(g) {
            Ok(dt) => dt,
            Err(e) => {
                let dt = self.fdtd.get_timestep_factor() * limit;
                r.issues.push(MeshIssue {
                    severity: Severity::Error,
                    message: format!("{e}; estimates use {dt:.4e} s"),
                    location: Location::Global,
                });
                dt
            }
        };
        let median: f64 = Axis::ALL
            .iter()
            .map(|&ax| {
                let mut d = self.phys_deltas(ax);
                d.sort_by(f64::total_cmp);
                let m = d[d.len() / 2];
                1.0 / (m * m)
            })
            .sum();
        s.dt_typical = factor.clamp(0.0, 1.0) / (C0 * median.sqrt());
        if s.dt_typical <= 0.0 {
            s.dt_typical = s.dt;
        }
        // Memory estimates (see MeshStats docs).
        let nodes = g.num_nodes() as u64;
        let mut pml_nodes = 0u64;
        for &(face, n) in &self.pml {
            let a = face.axis().index();
            let others: u64 = (0..3)
                .filter(|&b| b != a)
                .map(|b| s.nodes[b] as u64)
                .product();
            pml_nodes += (n.min(s.nodes[a] - 1) as u64 + 1) * others;
        }
        s.engine_bytes = nodes * 72 + pml_nodes * 16;
        s.compile_bytes = s.engine_bytes + nodes * 210;
        s.bytes_per_step = nodes * (72 + 24) + pml_nodes * 32;
        s.max_steps = self.fdtd.timestep_limit(s.dt);
        s.excitation_steps = self
            .fdtd
            .get_waveform()
            .and_then(|w| w.duration())
            .filter(|d| d.is_finite())
            .map(|d| (d / s.dt).ceil() as u64);
        s.f_max = self.fdtd.get_max_frequency();
        if let Some(n) = s.excitation_steps
            && n > s.max_steps
        {
            let max_steps = s.max_steps;
            r.push(
                Severity::Warning,
                Location::Global,
                format!(
                    "the excitation needs {n} steps but the run is limited to {max_steps} steps"
                ),
            );
        }
    }

    fn check_resolution(&self, r: &mut MeshReport, index: &SlabIndex) {
        let Some(f) = self.fdtd.get_max_frequency() else {
            r.push(
                Severity::Info,
                Location::Global,
                "no maximum frequency known (waveform without bounded spectrum and no \
                 Fdtd::max_frequency): cell sizes not checked against the wavelength"
                    .into(),
            );
            return;
        };
        let (n_ok, n_err) = (
            self.opts.cells_per_wavelength,
            self.opts.min_cells_per_wavelength,
        );
        let mut n_max = 1.0f64;
        let mut worst_all = f64::INFINITY;
        for axis in Axis::ALL {
            let a = axis.index();
            let d = self.phys_deltas(axis);
            let mut worst: Option<(usize, f64, f64)> = None; // (cell, cells/λ, n)
            let (mut count, mut count_err) = (0usize, 0usize);
            for (i, &dx) in d.iter().enumerate() {
                let n = index[a][i];
                n_max = n_max.max(n);
                let per = C0 / (f * n) / dx;
                worst_all = worst_all.min(per);
                if per < n_ok {
                    count += 1;
                    if per < n_err {
                        count_err += 1;
                    }
                    if worst.is_none_or(|w| per < w.1) {
                        worst = Some((i, per, n));
                    }
                }
            }
            if let Some((i, per, n)) = worst {
                let sev = if count_err > 0 {
                    Severity::Error
                } else {
                    Severity::Warning
                };
                let lam = C0 / (f * n);
                r.push(
                    sev,
                    self.cell_loc(axis, i),
                    format!(
                        "{count} {axis} cell(s) coarser than lambda/{n_ok} at f_max = {f:.4e} Hz \
                         ({count_err} below lambda/{n_err}); worst: {:.4} (drawing units) = \
                         lambda/{per:.1} with n = {n:.3}, needs <= {:.4}",
                        d[i] / self.unit,
                        lam / n_ok / self.unit
                    ),
                );
            }
        }
        r.stats.lambda_min = Some(C0 / (f * n_max));
        r.stats.min_cells_per_wavelength = Some(worst_all);
    }

    fn check_grading(&self, r: &mut MeshReport) {
        let lim = self.opts.max_grading * (1.0 + 1e-9);
        for axis in Axis::ALL {
            let mut d = self.grid.deltas(axis);
            let n = d.len();
            // Across a periodic boundary the last cell neighbours the first.
            if self.fdtd.get_boundaries().is_periodic(axis) && n > 1 {
                d.push(d[0]);
            }
            let mut count = 0usize;
            let mut worst: Option<(usize, f64)> = None;
            for (i, w) in d.windows(2).enumerate() {
                let q = (w[1] / w[0]).max(w[0] / w[1]);
                if q > lim {
                    count += 1;
                    if worst.is_none_or(|x| q > x.1) {
                        // Report the larger of the two cells.
                        worst = Some((if w[1] > w[0] { (i + 1) % n } else { i }, q));
                    }
                }
            }
            if let Some((i, q)) = worst {
                r.push(
                    Severity::Warning,
                    self.cell_loc(axis, i),
                    format!(
                        "{count} {axis} neighbour cell size ratio(s) above {}; worst {q:.2}",
                        self.opts.max_grading
                    ),
                );
            }
        }
    }

    fn check_timestep(&self, r: &mut MeshReport) {
        if self.grid.is_cylindrical() {
            return self.check_timestep_cylindrical(r);
        }
        let s = &r.stats;
        // The axis contributing most to Σ 1/Δ_min².
        let a = (0..3)
            .min_by(|&x, &y| s.min_cell[x].total_cmp(&s.min_cell[y]))
            .unwrap_or(0);
        let axis = Axis::ALL[a];
        let ratio = s.dt / s.dt_typical;
        let msg = format!(
            "time step {:.4e} s, limited by the smallest cells x[{}] = {:.3e} m, y[{}] = {:.3e} m, \
             z[{}] = {:.3e} m; {:.2} x the step of the median cells",
            s.dt,
            s.min_cell_index[0],
            s.min_cell[0],
            s.min_cell_index[1],
            s.min_cell[1],
            s.min_cell_index[2],
            s.min_cell[2],
            ratio
        );
        let loc = self.cell_loc(axis, s.min_cell_index[a]);
        if ratio < self.opts.timestep_ratio {
            r.push(
                Severity::Warning,
                loc,
                format!("{msg}: tiny cells dominate the time step"),
            );
        } else {
            r.push(Severity::Info, loc, msg);
        }
    }

    /// The time step of a cylindrical grid comes from its discrete operator
    /// and is usually set by the short arcs `ρ·Δα` next to `ρ_min` (or the
    /// axis), unless multigrid ([`Fdtd::multigrid`]) coarsens them.
    fn check_timestep_cylindrical(&self, r: &mut MeshReport) {
        let s = &r.stats;
        let rho = self.grid.lines(Axis::X);
        let inner = if self.grid.has_axis() {
            0.5 * rho[1]
        } else {
            rho[0]
        };
        let arc = inner * self.grid.min_delta(Axis::Y) * self.grid.alpha_factor(0) as f64;
        let ratio = s.dt / s.dt_typical;
        let msg = format!(
            "time step {:.4e} s (stability limit of the cylindrical operator); the shortest \
             arc next to {} is {arc:.3e} m; {ratio:.2} x the step of the median cells",
            s.dt,
            if self.grid.has_axis() {
                "the axis"
            } else {
                "rho_min"
            }
        );
        let loc = Location::Global;
        if ratio < self.opts.timestep_ratio {
            r.push(
                Severity::Warning,
                loc,
                format!(
                    "{msg}: the arcs near the axis dominate the time step (fewer alpha lines, \
                     a larger rho_min or multigrid near the axis help)"
                ),
            );
        } else {
            r.push(Severity::Info, loc, msg);
        }
    }

    fn check_pml(&self, r: &mut MeshReport) {
        let cells = self.grid.cells();
        for &(face, n) in &self.pml {
            let axis = face.axis();
            let a = axis.index();
            let nc = cells[a];
            if n >= nc {
                r.push(
                    Severity::Error,
                    Location::Face(face),
                    format!("PML needs {n} cells but the {axis} axis has only {nc} cells"),
                );
                continue;
            }
            if !face.is_max()
                && let Some(&(_, m)) = self.pml.iter().find(|(f, _)| *f == face.opposite())
                && n + m >= nc
            {
                r.push(
                    Severity::Error,
                    Location::Face(face),
                    format!(
                        "PMLs on both {axis} faces ({n} + {m} cells) leave no interior cells \
                         ({nc} cells on the axis)"
                    ),
                );
            }
            // Cell spread inside the layer (plus the adjacent interior cell).
            let d = self.grid.deltas(axis);
            let range = if face.is_max() {
                nc - n - 1..nc
            } else {
                0..n + 1
            };
            let (lo, hi) = d[range.clone()]
                .iter()
                .fold((f64::INFINITY, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
            if hi / lo > self.opts.pml_max_spread {
                r.push(
                    Severity::Warning,
                    Location::Face(face),
                    format!(
                        "PML cells are strongly graded (sizes {lo:.3e} .. {hi:.3e} m, ratio {:.2}); \
                         use roughly uniform cells in the absorbing layer",
                        hi / lo
                    ),
                );
            }
        }
        self.check_pml_intrusion(r);
    }

    fn check_pml_intrusion(&self, r: &mut MeshReport) {
        if self.pml.is_empty() {
            return;
        }
        let bg = &self.model.background;
        for p in self.model.properties() {
            let relevant = match &p.kind {
                PropertyKind::Metal
                | PropertyKind::ConductingSheet(_)
                | PropertyKind::LumpedElement(_)
                | PropertyKind::Excitation(_) => true,
                PropertyKind::Material(_)
                | PropertyKind::LorentzMaterial(_)
                | PropertyKind::DebyeMaterial(_)
                | PropertyKind::DiscMaterial(_) => {
                    let m = p.kind.material().expect("material kind");
                    let same = |q: MaterialQuantity, v: f64| {
                        !m.quantity(q).is_weighted() && (0..3).all(|d| m.base_value(q, d) == v)
                    };
                    let plain = matches!(p.kind, PropertyKind::Material(_));
                    !(plain
                        && same(MaterialQuantity::Epsilon, bg.epsilon.value())
                        && same(MaterialQuantity::Mue, bg.mue.value())
                        && same(MaterialQuantity::Kappa, bg.kappa.value())
                        && same(MaterialQuantity::Sigma, bg.sigma.value()))
                }
                _ => false,
            };
            if !relevant {
                continue;
            }
            for (k, prim) in p.primitives.iter().enumerate() {
                let b = self.bbox(prim);
                if b.is_empty() || !self.overlaps_domain(&b) {
                    continue;
                }
                let mut faces = Vec::new();
                for &(face, n) in &self.pml {
                    let axis = face.axis();
                    let a = axis.index();
                    let l = self.du_lines(axis);
                    let nc = l.len() - 1;
                    if n >= nc {
                        continue;
                    }
                    let hit = if face.is_max() {
                        let inner = l[nc - n];
                        let tol = 1e-6 * (l[nc - n] - l[nc - n - 1]);
                        b.max[a] > inner + tol
                    } else {
                        let inner = l[n];
                        let tol = 1e-6 * (l[n + 1] - l[n]);
                        b.min[a] < inner - tol
                    };
                    if hit {
                        faces.push(face.name());
                    }
                }
                if !faces.is_empty() {
                    r.push(
                        Severity::Warning,
                        Location::Primitive {
                            property: p.name.clone(),
                            index: k,
                        },
                        format!(
                            "{} {:?} extends into the PML of face(s) {}",
                            p.kind.type_name(),
                            p.name,
                            faces.join(", ")
                        ),
                    );
                }
            }
        }
    }

    /// Distance check of a flat coordinate `v` on axis `a` against the mesh:
    /// `Ok(())` if on a line, `Err(nearest)` otherwise; `None` if outside.
    fn snapped(&self, a: usize, v: f64) -> Option<core::result::Result<(), f64>> {
        let l = self.du_lines(Axis::ALL[a]);
        let n = l.len();
        let (lo, hi) = (l[0], l[n - 1]);
        let i = l.partition_point(|&x| x < v);
        let mut best = (f64::INFINITY, 0.0, 0.0); // (dist, line, local cell)
        for j in [i.saturating_sub(1), i.min(n - 1)] {
            let local = if j + 1 < n {
                l[j + 1] - l[j]
            } else {
                l[j] - l[j - 1]
            };
            let local = if j > 0 {
                local.min(l[j] - l[j - 1])
            } else {
                local
            };
            let dist = (l[j] - v).abs();
            if dist < best.0 {
                best = (dist, l[j], local);
            }
        }
        let (dist, line, local) = best;
        if dist <= self.opts.snap_tolerance * local {
            return Some(Ok(()));
        }
        if v < lo || v > hi {
            return None;
        }
        Some(Err(line))
    }

    /// Zero-thickness metal / conducting sheets and axis-aligned metal
    /// lines must lie on mesh planes / lines (otherwise csx's
    /// discretisation drops them).
    fn check_sheets(&self, r: &mut MeshReport) {
        for p in self.model.properties() {
            if !matches!(
                p.kind,
                PropertyKind::Metal | PropertyKind::ConductingSheet(_)
            ) {
                continue;
            }
            for (k, prim) in p.primitives.iter().enumerate() {
                let b = self.bbox(prim);
                if b.is_empty() || !(b.min.is_finite() && b.max.is_finite()) {
                    continue;
                }
                let flat: Vec<usize> = (0..3)
                    .filter(|&a| {
                        let l = self.du_lines(Axis::ALL[a]);
                        let scale = l[l.len() - 1] - l[0];
                        b.max[a] - b.min[a] <= self.opts.snap_tolerance * scale.abs()
                    })
                    .collect();
                let what = match (flat.len(), prim.dimension()) {
                    (2, _) => "line",
                    (1, 2) => "sheet",
                    _ => continue,
                };
                let loc = || Location::Primitive {
                    property: p.name.clone(),
                    index: k,
                };
                let mut outside = false;
                for &a in &flat {
                    let axis = Axis::ALL[a];
                    let v = b.min[a];
                    match self.snapped(a, v) {
                        Some(Ok(())) => {}
                        Some(Err(line)) => {
                            r.push(
                                Severity::Error,
                                loc(),
                                format!(
                                    "{} {what} {:?} at {axis} = {v} is not on a mesh line \
                                     (nearest {line}) and will be ignored; add a mesh line there",
                                    p.kind.type_name(),
                                    p.name
                                ),
                            );
                        }
                        None => outside = true,
                    }
                }
                if outside {
                    r.push(
                        Severity::Warning,
                        loc(),
                        format!(
                            "{} {what} {:?} lies outside the mesh",
                            p.kind.type_name(),
                            p.name
                        ),
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{CompileOptions, compile};
    use crate::fdtd::Boundaries;
    use crate::waveform::Waveform;
    use csx::primitives::BoxShape;
    use csx::{Material, Property};

    /// Uniform mesh of `cells` cells of `h` drawing units (mm) per axis.
    fn model(cells: usize, h: f64) -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        for a in 0..3 {
            m.grid
                .set_lines(a, (0..=cells).map(|i| i as f64 * h))
                .unwrap();
        }
        m
    }

    /// f_max = 10 GHz → λ0 = 29.98 mm.
    fn fdtd() -> Fdtd {
        Fdtd::new().waveform(Waveform::gaussian(5e9, 5e9))
    }

    fn add_box(m: &mut ContinuousStructure, p: Property, lo: [f64; 3], hi: [f64; 3]) {
        let id = m.add_property(p);
        m.add_primitive(id, Primitive::new(BoxShape::new(lo, hi)))
            .unwrap();
    }

    #[test]
    fn good_mesh_passes() {
        let m = model(30, 1.0); // λ/30
        let r = check(&m, &fdtd()).unwrap();
        assert!(r.is_ok(), "{r}");
        assert_eq!(r.warnings().count(), 0, "{r}");
        let n = r.stats.min_cells_per_wavelength.unwrap();
        assert!((n - 29.979).abs() < 1e-2, "{n}");
        assert_eq!(r.stats.cells, 27_000);
        assert_eq!(r.stats.nodes, [31; 3]);
        assert_eq!(r.stats.engine_bytes, 31u64.pow(3) * 72);
        assert!((r.stats.dt / r.stats.dt_typical - 1.0).abs() < 1e-12);
    }

    #[test]
    fn coarse_mesh_flagged() {
        // 2 mm = λ/15: warning only.
        let r = check(&model(10, 2.0), &fdtd()).unwrap();
        assert!(r.is_ok(), "{r}");
        assert_eq!(r.warnings().count(), 3, "{r}");
        assert!(
            r.warnings()
                .all(|i| matches!(i.location, Location::Cell { .. }))
        );
        // 4 mm = λ/7.5: error.
        let r = check(&model(10, 4.0), &fdtd()).unwrap();
        assert!(!r.is_ok());
        assert_eq!(r.errors().count(), 3, "{r}");
        // A custom N relaxes it.
        let opts = CheckOptions::default()
            .cells_per_wavelength(7.0)
            .min_cells_per_wavelength(5.0);
        let r = check_with(&model(10, 4.0), &fdtd(), &opts).unwrap();
        assert!(r.is_ok() && r.warnings().count() == 0, "{r}");
    }

    #[test]
    fn no_frequency_is_info() {
        let r = check(&model(10, 4.0), &Fdtd::new()).unwrap();
        assert!(r.is_ok());
        assert!(
            r.infos()
                .any(|i| i.message.contains("no maximum frequency"))
        );
        assert!(r.stats.f_max.is_none());
        let e = suggest_max_cell(&model(10, 1.0), &Fdtd::new(), 20.0);
        assert!(e.is_err());
    }

    #[test]
    fn high_permittivity_needs_finer_cells() {
        // 1 mm is λ0/30 but only λ/15 in ε_r = 4.
        let mut m = model(30, 1.0);
        add_box(
            &mut m,
            Property::material("sub", Material::new(4.0, 0.0)),
            [0.0, 0.0, 0.0],
            [30.0, 30.0, 5.0],
        );
        let r = check(&m, &fdtd()).unwrap();
        assert!(r.is_ok(), "{r}");
        assert_eq!(r.warnings().count(), 3, "{r}");
        // z: only the slabs inside the substrate are flagged.
        let z = r
            .warnings()
            .find(|i| matches!(i.location, Location::Cell { axis: Axis::Z, .. }))
            .unwrap();
        assert!(z.message.starts_with("5 z cell(s)"), "{}", z.message);
        assert!((r.stats.lambda_min.unwrap() - C0 / 1e10 / 2.0).abs() < 1e-9);

        // The compiled variant sees the same.
        let c = compile(&m, &fdtd(), &CompileOptions::default()).unwrap();
        let rc = check_compiled(&m, &fdtd(), &c, &CheckOptions::default()).unwrap();
        assert_eq!(rc.warnings().count(), 3, "{rc}");

        // The suggestion resolves the substrate.
        let h = suggest_max_cell(&m, &fdtd(), 20.0).unwrap();
        assert!((h - C0 / 1e10 / 2.0 / 20.0 / 1e-3).abs() < 1e-9, "{h}");
    }

    #[test]
    fn grading_flagged() {
        let mut m = model(30, 1.0);
        let mut x: Vec<f64> = (0..=10).map(f64::from).collect();
        x.extend([10.5, 11.0]); // 1 → 0.5: ratio 2
        x.extend((12..=30).map(f64::from));
        m.grid.set_lines(0, x).unwrap();
        let r = check(&m, &fdtd()).unwrap();
        let w: Vec<_> = r.warnings().collect();
        assert_eq!(w.len(), 1, "{r}");
        assert!(w[0].message.contains("ratio(s) above 1.5; worst 2.00"));
        assert!(matches!(
            w[0].location,
            Location::Cell {
                axis: Axis::X,
                index: 9,
                ..
            } | Location::Cell {
                axis: Axis::X,
                index: 12,
                ..
            }
        ));
    }

    #[test]
    fn grading_wraps_across_periodic_faces() {
        let mut m = model(30, 1.0);
        let mut x = vec![0.0, 0.5]; // first cell 0.5, last cell 1: ratio 2
        x.extend((1..=30).map(f64::from));
        m.grid.set_lines(0, x).unwrap();
        // 0.5 → 1 inside the domain too; count the ratios.
        let count = |f: &Fdtd| {
            let r = check(&m, f).unwrap();
            r.warnings()
                .find(|i| i.message.contains("x neighbour"))
                .map(|i| i.message.clone())
        };
        let open = count(&fdtd()).unwrap();
        assert!(open.starts_with("1 x neighbour"), "{open}");
        let periodic = fdtd()
            .boundary(crate::Face::XMin, Boundary::Periodic)
            .boundary(crate::Face::XMax, Boundary::Periodic);
        let wrapped = count(&periodic).unwrap();
        assert!(wrapped.starts_with("2 x neighbour"), "{wrapped}");
    }

    #[test]
    fn csx_smoothed_mesh_passes_grading() {
        let mut lines = csx::grid::smooth_lines(&[0.0, 0.2, 30.0], 1.0, 1.4).unwrap();
        lines.dedup();
        let mut m = model(30, 1.0);
        m.grid.set_lines(0, lines).unwrap();
        let r = check(&m, &fdtd()).unwrap();
        assert!(r.warnings().all(|i| !i.message.contains("ratio")), "{r}");
    }

    #[test]
    fn tiny_cells_dominate_timestep() {
        let mut m = model(30, 1.0);
        m.grid.add_line(2, 10.01).unwrap();
        let r = check(&m, &fdtd()).unwrap();
        let w = r
            .warnings()
            .find(|i| i.message.contains("dominate"))
            .expect("time step warning");
        assert!(matches!(
            w.location,
            Location::Cell {
                axis: Axis::Z,
                index: 10,
                ..
            }
        ));
        assert!(r.stats.dt < 0.2 * r.stats.dt_typical);
        assert_eq!(r.stats.min_cell_index[2], 10);
    }

    #[test]
    fn pml_checks() {
        let pml = fdtd().boundaries(Boundaries::all(Boundary::pml(8)));
        let mut m = model(30, 1.0);
        add_box(&mut m, Property::metal("inside"), [10.0; 3], [20.0; 3]);
        let r = check(&m, &pml).unwrap();
        assert!(r.is_ok() && r.warnings().count() == 0, "{r}");
        assert!(r.stats.engine_bytes > 31u64.pow(3) * 72);

        // Intruding into the xmin and zmax layers (x < 8, z > 22).
        add_box(
            &mut m,
            Property::material("diel", Material::new(2.0, 0.0)),
            [5.0, 10.0, 10.0],
            [20.0, 20.0, 25.0],
        );
        // Background-equivalent material and probes are ignored.
        add_box(
            &mut m,
            Property::material("air", Material::new(1.0, 0.0)),
            [0.0; 3],
            [30.0; 3],
        );
        let r = check(&m, &pml).unwrap();
        let w: Vec<_> = r.warnings().collect();
        assert_eq!(w.len(), 1, "{r}");
        assert!(w[0].message.contains("PML of face(s) xmin, zmax"), "{r}");
        assert_eq!(
            w[0].location,
            Location::Primitive {
                property: "diel".into(),
                index: 0
            }
        );

        // Not enough cells.
        let r = check(&model(10, 1.0), &pml).unwrap();
        assert_eq!(r.errors().count(), 3, "{r}"); // one per axis: no interior left
        let r = check(
            &model(10, 1.0),
            &fdtd().boundary(Face::XMin, Boundary::pml(20)),
        )
        .unwrap();
        assert!(r.errors().any(|i| i.location == Location::Face(Face::XMin)));

        // Graded PML.
        let mut m = model(30, 1.0);
        let mut x = vec![-12.0, -6.0, -3.0, -1.5];
        x.extend((0..=30).map(f64::from));
        m.grid.set_lines(0, x).unwrap();
        let r = check(&m, &fdtd().boundary(Face::XMin, Boundary::pml(4))).unwrap();
        assert!(
            r.warnings()
                .any(|i| i.location == Location::Face(Face::XMin)
                    && i.message.contains("strongly graded")),
            "{r}"
        );
    }

    #[test]
    fn sheets_off_mesh() {
        let mut m = model(30, 1.0);
        add_box(
            &mut m,
            Property::metal("ok"),
            [5.0, 5.0, 10.0],
            [9.0, 9.0, 10.0],
        );
        let r = check(&m, &fdtd()).unwrap();
        assert!(r.is_ok() && r.warnings().count() == 0, "{r}");
        add_box(
            &mut m,
            Property::metal("off"),
            [5.0, 5.0, 10.4],
            [9.0, 9.0, 10.4],
        );
        add_box(
            &mut m,
            Property::metal("wire"),
            [5.5, 5.0, 3.0],
            [5.5, 9.0, 3.0],
        );
        let r = check(&m, &fdtd()).unwrap();
        let e: Vec<_> = r.errors().collect();
        assert_eq!(e.len(), 2, "{r}");
        assert!(e[0].message.contains("z = 10.4") && e[0].message.contains("nearest 10"));
        assert!(e[1].message.contains("line") && e[1].message.contains("x = 5.5"));

        // The compiled variant reuses csx's discretisation warnings.
        let c = compile(&m, &fdtd(), &CompileOptions::default()).unwrap();
        let rc = check_compiled(&m, &fdtd(), &c, &CheckOptions::default()).unwrap();
        assert_eq!(rc.errors().count(), 2, "{rc}");
    }

    #[test]
    fn steps_and_limits() {
        let f = fdtd().max_timesteps(10);
        let r = check(&model(30, 1.0), &f).unwrap();
        assert_eq!(r.stats.max_steps, 10);
        let n = r.stats.excitation_steps.unwrap();
        assert!(n > 10);
        assert!(r.warnings().any(|i| i.message.contains("excitation needs")));
        // An invalid dt override is an error item, not a failure.
        let r = check(&model(30, 1.0), &fdtd().timestep(1.0)).unwrap();
        assert!(!r.is_ok());
    }

    #[test]
    fn report_formatting() {
        let mut m = model(10, 4.0);
        m.grid.add_line(0, 0.2).unwrap();
        let r = check(&m, &fdtd()).unwrap();
        let s = r.to_string();
        assert!(
            s.starts_with("mesh check: 3 error(s), 2 warning(s)\n"),
            "{s}"
        );
        assert!(
            s.contains("  cells: 11 x 10 x 10 = 1100 (1452 nodes)"),
            "{s}"
        );
        assert!(s.contains("f_max 1.0000e10 Hz"), "{s}");
        assert!(
            s.contains("error: 10 x cell(s) coarser than lambda/20"),
            "{s}"
        );
        assert!(s.contains("[x cell "), "{s}");
        let msgs = r.messages(Severity::Warning);
        assert_eq!(msgs.len(), 5);
        assert!(msgs.iter().all(|m| m.starts_with("mesh ")));

        let ok = check(&model(30, 1.0), &fdtd()).unwrap().to_string();
        assert!(ok.starts_with("mesh check: OK\n"), "{ok}");
        assert!(ok.contains("info: time step"), "{ok}");

        let i = MeshIssue {
            severity: Severity::Warning,
            message: "m".into(),
            location: Location::Face(Face::YMax),
        };
        assert_eq!(i.to_string(), "warning: m [ymax face]");
    }
}
