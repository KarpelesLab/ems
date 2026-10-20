//! Solver settings: the equivalent of the openEMS `<FDTD>` section.
//!
//! [`Fdtd`] collects everything that is not part of the model itself: the
//! excitation [`Waveform`], the [`Boundaries`] of the six domain faces, the
//! [`EndCriteria`], the timestep count and the time-step choice. This module
//! only holds data and validates it; the engine interprets it.
//!
//! The file-level openEMS `<FDTD>` element is typed by csx
//! ([`csx::fdtd::FdtdSettings`]); [`Fdtd::from_csx`] and [`Fdtd::to_csx`]
//! convert between the two (see [`Fdtd::from_csx`] for the mapping).

use crate::axis::Axis;
use crate::engine::{OpenEmsTimestep, TimeStep, TimestepMethod};
use crate::error::{Error, Result};
use crate::waveform::Waveform;
use core::fmt;
use core::ops::{Index, IndexMut};

mod convert;

/// One of the six faces of the computational domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Face {
    /// Face at the lowest x mesh line.
    XMin,
    /// Face at the highest x mesh line.
    XMax,
    /// Face at the lowest y mesh line.
    YMin,
    /// Face at the highest y mesh line.
    YMax,
    /// Face at the lowest z mesh line.
    ZMin,
    /// Face at the highest z mesh line.
    ZMax,
}

impl Face {
    /// All faces in index order (the order of the openEMS `<BoundaryCond>`
    /// attributes: xmin, xmax, ymin, ymax, zmin, zmax).
    pub const ALL: [Face; 6] = [
        Face::XMin,
        Face::XMax,
        Face::YMin,
        Face::YMax,
        Face::ZMin,
        Face::ZMax,
    ];

    /// Index in `0..6`: `2·axis + (1 if max side)`.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The face with index `i` (see [`Face::index`]).
    pub const fn from_index(i: usize) -> Option<Face> {
        if i < 6 { Some(Face::ALL[i]) } else { None }
    }

    /// The face of `axis` on the max (`true`) or min (`false`) side.
    pub const fn new(axis: Axis, max: bool) -> Face {
        Face::ALL[2 * axis.index() + max as usize]
    }

    /// The axis normal to the face.
    pub const fn axis(self) -> Axis {
        match self {
            Face::XMin | Face::XMax => Axis::X,
            Face::YMin | Face::YMax => Axis::Y,
            Face::ZMin | Face::ZMax => Axis::Z,
        }
    }

    /// `true` for the faces at the highest mesh line of their axis.
    pub const fn is_max(self) -> bool {
        self.index() % 2 == 1
    }

    /// The face on the other side of the domain.
    pub const fn opposite(self) -> Face {
        Face::new(self.axis(), !self.is_max())
    }

    /// Lower-case name, as in openEMS XML attributes: `"xmin"`, `"xmax"`, ….
    pub const fn name(self) -> &'static str {
        match self {
            Face::XMin => "xmin",
            Face::XMax => "xmax",
            Face::YMin => "ymin",
            Face::YMax => "ymax",
            Face::ZMin => "zmin",
            Face::ZMax => "zmax",
        }
    }
}

impl fmt::Display for Face {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// First-order Mur absorbing boundary (G. Mur, "Absorbing boundary
/// conditions for the finite-difference approximation of the time-domain
/// electromagnetic-field equations", IEEE Trans. EMC 23(4), 1981).
///
/// The one-way wave equation is applied to the tangential E field of the
/// face, tuned for waves leaving at normal incidence with the given phase
/// velocity.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Mur {
    /// Phase velocity the boundary is tuned for \[m/s\]; `None` means the
    /// local medium speed `c0/√(ε_r·μ_r)` of each boundary edge when the
    /// model is compiled (as openEMS does), see
    /// [`MurBoundary::with_faces_in`](crate::engine::boundary::mur::MurBoundary::with_faces_in).
    pub phase_velocity: Option<f64>,
}

impl Mur {
    /// Mur boundary tuned for the local medium (`c0` in vacuum).
    pub const fn new() -> Self {
        Mur {
            phase_velocity: None,
        }
    }

    /// Mur boundary tuned for phase velocity `v` \[m/s\].
    pub const fn with_phase_velocity(v: f64) -> Self {
        Mur {
            phase_velocity: Some(v),
        }
    }

    /// The phase velocity \[m/s\], the vacuum speed of light
    /// [`C0`](crate::constants::C0) if none is set (the value for a vacuum
    /// boundary; compiled models use the local medium instead).
    pub fn velocity(&self) -> f64 {
        self.phase_velocity.unwrap_or(crate::constants::C0)
    }

    /// Checks that the phase velocity (if set) is finite and positive.
    pub fn validate(&self) -> Result<()> {
        match self.phase_velocity {
            Some(v) if !(v.is_finite() && v > 0.0) => Err(Error::Invalid(format!(
                "Mur boundary: phase velocity {v} must be finite and > 0"
            ))),
            _ => Ok(()),
        }
    }
}

/// Convolutional perfectly matched layer (CPML) parameters.
///
/// Formulation of J. A. Roden and S. D. Gedney, "Convolutional PML (CPML):
/// An efficient FDTD implementation of the CFS-PML for arbitrary media",
/// Microwave and Optical Technology Letters 27(5), 2000; see also Taflove &
/// Hagness, *Computational Electrodynamics*, 3rd ed., ch. 7.
///
/// The layer occupies the outermost `cells` cells of the mesh
/// at the face (openEMS `PML_n` semantics). With `ρ ∈ [0, 1]` the normalised
/// depth into the layer, the stretching parameters are graded as
///
/// ```text
/// σ(ρ) = σ_max·ρ^m,   κ(ρ) = 1 + (κ_max − 1)·ρ^m,   α(ρ) = α_max·(1 − ρ)
/// ```
///
/// with `m` = `grading_order` and `σ_max = sigma_factor · 0.8·(m + 1)/(η·Δ)`,
/// the near-optimal value for polynomial grading given in Taflove & Hagness,
/// ch. 7, with `Δ` the local cell size normal to the layer and
/// `η = n·Z0`, `n` the mean refractive index of the layer's medium (the
/// engine measures it per face, see
/// [`CpmlBoundary`](crate::engine::boundary::cpml::CpmlBoundary)), so the
/// default is near-optimal in dielectrics too; `sigma_factor` is relative
/// to that optimum. The defaults (`m = 3`, `κ_max = 1`, `α_max = 0`) give a
/// classic graded PML; `κ_max > 1` and `α_max > 0` enable the CFS terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cpml {
    /// Number of mesh cells the layer occupies (default 8).
    pub cells: usize,
    /// Polynomial grading order `m` (default 3).
    pub grading_order: f64,
    /// Multiplier on the optimal `σ_max` (default 1).
    pub sigma_factor: f64,
    /// Maximum coordinate stretching `κ_max ≥ 1` (default 1).
    pub kappa_max: f64,
    /// Maximum complex-frequency shift `α_max ≥ 0` \[S/m\] (default 0).
    pub alpha_max: f64,
}

impl Default for Cpml {
    fn default() -> Self {
        Cpml {
            cells: 8,
            grading_order: 3.0,
            sigma_factor: 1.0,
            kappa_max: 1.0,
            alpha_max: 0.0,
        }
    }
}

impl Cpml {
    /// Default CPML with `n` cells.
    pub fn cells(n: usize) -> Self {
        Cpml {
            cells: n,
            ..Cpml::default()
        }
    }

    /// Sets the polynomial grading order `m`.
    pub fn grading_order(mut self, m: f64) -> Self {
        self.grading_order = m;
        self
    }

    /// Sets the multiplier on the optimal `σ_max`.
    pub fn sigma_factor(mut self, k: f64) -> Self {
        self.sigma_factor = k;
        self
    }

    /// Sets `κ_max`.
    pub fn kappa_max(mut self, k: f64) -> Self {
        self.kappa_max = k;
        self
    }

    /// Sets `α_max` \[S/m\].
    pub fn alpha_max(mut self, a: f64) -> Self {
        self.alpha_max = a;
        self
    }

    /// `σ_max = sigma_factor · 0.8·(m + 1)/(η·Δ)` \[S/m\] for `eta` \[Ω\]
    /// (`n·Z0` for a layer of refractive index `n`) and cell size `delta`
    /// \[m\] normal to the layer.
    pub fn sigma_max(&self, eta: f64, delta: f64) -> f64 {
        self.sigma_factor * 0.8 * (self.grading_order + 1.0) / (eta * delta)
    }

    /// Checks the parameters: `cells ≥ 1`, finite `m ≥ 0`, `sigma_factor ≥ 0`,
    /// `κ_max ≥ 1`, `α_max ≥ 0`.
    pub fn validate(&self) -> Result<()> {
        let bad = |what: &str, v: f64, req: &str| {
            Err(Error::Invalid(format!("CPML: {what} = {v} must be {req}")))
        };
        if self.cells == 0 {
            return Err(Error::Invalid("CPML: needs at least one cell".into()));
        }
        if !(self.grading_order.is_finite() && self.grading_order >= 0.0) {
            return bad("grading order", self.grading_order, "finite and >= 0");
        }
        if !(self.sigma_factor.is_finite() && self.sigma_factor >= 0.0) {
            return bad("sigma factor", self.sigma_factor, "finite and >= 0");
        }
        if !(self.kappa_max.is_finite() && self.kappa_max >= 1.0) {
            return bad("kappa_max", self.kappa_max, "finite and >= 1");
        }
        if !(self.alpha_max.is_finite() && self.alpha_max >= 0.0) {
            return bad("alpha_max", self.alpha_max, "finite and >= 0");
        }
        Ok(())
    }
}

/// Boundary condition of one domain face.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Boundary {
    /// Perfect electric conductor: tangential E is zero on the face.
    #[default]
    Pec,
    /// Perfect magnetic conductor: tangential H is zero on the face.
    Pmc,
    /// First-order Mur absorbing boundary.
    Mur(Mur),
    /// Convolutional PML.
    Pml(Cpml),
    /// Periodic (zero phase shift): the face is identified with the
    /// opposite face, which must be periodic too. The period is the domain
    /// length along the axis (first to last mesh line). ems-only: the
    /// openEMS file format has no periodic boundary (see
    /// [`Fdtd::to_csx`]).
    Periodic,
}

impl Boundary {
    /// Mur boundary tuned for the local medium (`c0` in vacuum).
    pub const fn mur() -> Self {
        Boundary::Mur(Mur::new())
    }

    /// Default CPML with `n` cells.
    pub fn pml(n: usize) -> Self {
        Boundary::Pml(Cpml::cells(n))
    }

    /// Number of mesh cells the boundary occupies inside the domain
    /// (the PML thickness; 0 for the other kinds).
    pub fn cells(&self) -> usize {
        match self {
            Boundary::Pml(p) => p.cells,
            _ => 0,
        }
    }

    /// Checks the boundary's parameters.
    pub fn validate(&self) -> Result<()> {
        match self {
            Boundary::Pec | Boundary::Pmc | Boundary::Periodic => Ok(()),
            Boundary::Mur(m) => m.validate(),
            Boundary::Pml(p) => p.validate(),
        }
    }
}

/// Boundary conditions of all six faces (default: all PEC).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Boundaries([Boundary; 6]);

impl Boundaries {
    /// The same boundary on every face.
    pub fn all(b: Boundary) -> Self {
        Boundaries([b; 6])
    }

    /// Builder: sets the boundary of `face`.
    pub fn with(mut self, face: Face, b: Boundary) -> Self {
        self.0[face.index()] = b;
        self
    }

    /// Sets the boundary of `face`.
    pub fn set(&mut self, face: Face, b: Boundary) {
        self.0[face.index()] = b;
    }

    /// The boundary of `face`.
    pub fn get(&self, face: Face) -> &Boundary {
        &self.0[face.index()]
    }

    /// The boundaries in [`Face::ALL`] order.
    pub fn as_array(&self) -> &[Boundary; 6] {
        &self.0
    }

    /// Iterates over `(face, boundary)` pairs in [`Face::ALL`] order.
    pub fn iter(&self) -> impl Iterator<Item = (Face, &Boundary)> {
        Face::ALL.into_iter().zip(self.0.iter())
    }

    /// Checks every face's parameters, and that periodic faces come in
    /// opposite pairs.
    pub fn validate(&self) -> Result<()> {
        for (face, b) in self.iter() {
            b.validate()
                .map_err(|e| Error::Invalid(format!("boundary {face}: {e}")))?;
            if *b == Boundary::Periodic && self[face.opposite()] != Boundary::Periodic {
                return Err(Error::Invalid(format!(
                    "boundary {face}: periodic needs the opposite face {} periodic too",
                    face.opposite()
                )));
            }
        }
        Ok(())
    }

    /// Whether the faces of `axis` are periodic.
    pub fn is_periodic(&self, axis: Axis) -> bool {
        self[Face::new(axis, false)] == Boundary::Periodic
            && self[Face::new(axis, true)] == Boundary::Periodic
    }
}

impl From<[Boundary; 6]> for Boundaries {
    fn from(b: [Boundary; 6]) -> Self {
        Boundaries(b)
    }
}

impl Index<Face> for Boundaries {
    type Output = Boundary;
    fn index(&self, face: Face) -> &Boundary {
        &self.0[face.index()]
    }
}

impl IndexMut<Face> for Boundaries {
    fn index_mut(&mut self, face: Face) -> &mut Boundary {
        &mut self.0[face.index()]
    }
}

/// When to stop a simulation before the maximum timestep count: once the
/// excitation is over and the total field energy has decayed below
/// `energy_ratio` times its maximum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndCriteria {
    /// Threshold as a linear energy ratio `W/W_max` in `(0, 1]`.
    pub energy_ratio: f64,
    /// Number of timesteps between energy evaluations (default 50).
    pub check_interval: u64,
}

impl EndCriteria {
    /// Default `check_interval`.
    pub const DEFAULT_CHECK_INTERVAL: u64 = 50;

    /// Stop when the energy is `db` decibels below its maximum, e.g.
    /// `-40.0` for a ratio of 1e-4 (energy is a power quantity:
    /// `ratio = 10^(db/10)`).
    pub fn energy_decay_db(db: f64) -> Self {
        EndCriteria::energy_ratio(10f64.powf(db / 10.0))
    }

    /// Stop when `W/W_max < ratio`.
    pub fn energy_ratio(ratio: f64) -> Self {
        EndCriteria {
            energy_ratio: ratio,
            check_interval: Self::DEFAULT_CHECK_INTERVAL,
        }
    }

    /// Builder: sets the number of timesteps between energy evaluations.
    pub fn check_interval(mut self, steps: u64) -> Self {
        self.check_interval = steps;
        self
    }

    /// The threshold in decibels, `10·log10(energy_ratio)`.
    pub fn decay_db(&self) -> f64 {
        10.0 * self.energy_ratio.log10()
    }

    /// Checks that the ratio is in `(0, 1]` and the interval is at least 1.
    pub fn validate(&self) -> Result<()> {
        let r = self.energy_ratio;
        if !(r > 0.0 && r <= 1.0) {
            return Err(Error::Invalid(format!(
                "end criteria: energy ratio {r} must be in (0, 1]"
            )));
        }
        if self.check_interval == 0 {
            return Err(Error::Invalid(
                "end criteria: check interval must be at least 1 step".into(),
            ));
        }
        Ok(())
    }
}

/// FDTD solver settings (the openEMS `<FDTD>` section).
///
/// Built with chained setters; read back with the `get_*` getters (the plain
/// names are taken by the setters, as in [`std::process::Command`]).
///
/// ```
/// use ems::prelude::*;
/// let fdtd = Fdtd::new()
///     .waveform(Waveform::gaussian(2e9, 1e9))
///     .boundaries(Boundaries::all(Boundary::Pml(Cpml::cells(8))))
///     .end_criteria(EndCriteria::energy_decay_db(-40.0))
///     .max_timesteps(200_000);
/// fdtd.validate()?;
/// assert_eq!(fdtd.get_max_timesteps(), 200_000);
/// # Ok::<(), ems::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Fdtd {
    waveform: Option<Waveform>,
    boundaries: Boundaries,
    end_criteria: Option<EndCriteria>,
    max_timesteps: u64,
    timestep_factor: f64,
    timestep: Option<f64>,
    timestep_method: Option<TimestepMethod>,
    openems_run: bool,
    cell_constant_material: bool,
    threads: Option<usize>,
    max_frequency: Option<f64>,
    max_time: Option<f64>,
    oversampling: Option<u32>,
    bloch_wavevector: Option<[f64; 3]>,
    bloch_frequency: Option<f64>,
    multigrid: Vec<f64>,
}

impl Default for Fdtd {
    fn default() -> Self {
        Fdtd {
            waveform: None,
            boundaries: Boundaries::default(),
            end_criteria: None,
            max_timesteps: Self::DEFAULT_MAX_TIMESTEPS,
            timestep_factor: Self::DEFAULT_TIMESTEP_FACTOR,
            timestep: None,
            timestep_method: None,
            openems_run: false,
            cell_constant_material: false,
            threads: None,
            max_frequency: None,
            max_time: None,
            oversampling: None,
            bloch_wavevector: None,
            bloch_frequency: None,
            multigrid: Vec::new(),
        }
    }
}

impl Fdtd {
    /// Default [`max_timesteps`](Fdtd::max_timesteps).
    pub const DEFAULT_MAX_TIMESTEPS: u64 = 1_000_000;
    /// Default [`timestep_factor`](Fdtd::timestep_factor) (Courant number
    /// safety factor).
    pub const DEFAULT_TIMESTEP_FACTOR: f64 = 0.99;

    /// Default settings: no waveform, all-PEC boundaries, no end criteria,
    /// [`DEFAULT_MAX_TIMESTEPS`](Fdtd::DEFAULT_MAX_TIMESTEPS) steps, time
    /// step factor [`DEFAULT_TIMESTEP_FACTOR`](Fdtd::DEFAULT_TIMESTEP_FACTOR),
    /// automatic time step and thread count.
    pub fn new() -> Self {
        Fdtd::default()
    }

    /// Sets the excitation waveform shared by all excitations of the model.
    pub fn waveform(mut self, w: Waveform) -> Self {
        self.waveform = Some(w);
        self
    }

    /// Sets the boundary conditions of all faces.
    pub fn boundaries(mut self, b: Boundaries) -> Self {
        self.boundaries = b;
        self
    }

    /// Sets the boundary condition of one face.
    pub fn boundary(mut self, face: Face, b: Boundary) -> Self {
        self.boundaries.set(face, b);
        self
    }

    /// Enables early termination by energy decay. Without end criteria the
    /// simulation runs for [`max_timesteps`](Fdtd::max_timesteps).
    pub fn end_criteria(mut self, e: EndCriteria) -> Self {
        self.end_criteria = Some(e);
        self
    }

    /// Removes the end criteria (run to the maximum timestep count).
    pub fn no_end_criteria(mut self) -> Self {
        self.end_criteria = None;
        self
    }

    /// Sets the maximum number of timesteps (≥ 1).
    pub fn max_timesteps(mut self, n: u64) -> Self {
        self.max_timesteps = n;
        self
    }

    /// Sets the Courant safety factor `S ∈ (0, 1]`: the automatic time step
    /// is `S` times the Courant–Friedrichs–Lewy limit of the mesh.
    pub fn timestep_factor(mut self, s: f64) -> Self {
        self.timestep_factor = s;
        self
    }

    /// Overrides the automatic time step with `dt` \[s\]. It must still
    /// satisfy the stability limit of the mesh (checked when compiling).
    pub fn timestep(mut self, dt: f64) -> Self {
        self.timestep = Some(dt);
        self
    }

    /// Chooses the time step with an openEMS criterion
    /// ([`TimestepMethod`], see [`crate::engine::timestep`]) on the
    /// compiled operator (materials included) instead of the ems default
    /// (the global Courant limit of the smallest cells,
    /// [`courant_limit`](crate::engine::courant_limit)), times the
    /// [time step factor](Self::timestep_factor). An explicit
    /// [`timestep`](Self::timestep) is then also multiplied by the factor,
    /// and a [sinusoidal](Waveform::Sinusoid) waveform shortens the step so
    /// that its period is an integer number of steps (openEMS behaviour).
    /// [`Fdtd::from_csx`] sets this (openEMS files always use one).
    pub fn timestep_method(mut self, m: impl Into<Option<TimestepMethod>>) -> Self {
        self.timestep_method = m.into();
        self
    }

    /// openEMS run semantics (set by [`Fdtd::from_csx`]; established by
    /// black-box comparison with openEMS v0.37):
    ///
    /// * the [end criteria](Self::end_criteria) are evaluated every
    ///   Nyquist interval of the waveform ([`Waveform::nyquist_interval`],
    ///   1 step without a waveform) instead of the criteria's own check
    ///   interval, on the state reached after each interval
    ///   ([`RunLimits::check_after_interval`](crate::engine::RunLimits::check_after_interval)),
    ///   from the start of the run instead of after the excitation has
    ///   ended;
    /// * a [`max_time`](Self::max_time) allows `⌊max_time/dt⌋` steps
    ///   (instead of `⌈max_time/dt⌉`);
    /// * [`max_timesteps`](Self::max_timesteps) may be 0 (openEMS's value
    ///   when `NumberOfTimesteps` is missing: nothing is run).
    /// * a [`Boundary::Pmc`] face holds the tangential H of the outermost
    ///   dual layer at zero (the magnetic wall half a cell inside the
    ///   domain, see [`Compiled::openems_pmc`](crate::compile::Compiled::openems_pmc));
    /// * lumped elements follow openEMS (design §5.16.1): parallel elements
    ///   replace their edges' medium capacitance (`C`) and conductance
    ///   (`R`); a parallel R/C element without inductor is spread over its
    ///   edges like a homogeneous material (per-edge `G`, `C` proportional
    ///   to dual area over length), others in equal parts; parallel
    ///   inductors are updated explicitly
    ///   ([`LumpedElement::explicit_inductor`](crate::engine::LumpedElement::explicit_inductor),
    ///   unless that would be unstable); an excitation over any element is
    ///   a plain soft source, and the element's branch is driven by the
    ///   edge voltage without it (see [`Simulation`](crate::Simulation),
    ///   lumped elements and excitations).
    pub fn openems_run(mut self, on: bool) -> Self {
        self.openems_run = on;
        self
    }

    /// Samples materials once per cell, at the cell centre, instead of
    /// averaging sub-cell samples onto each edge (openEMS
    /// `CellConstantMaterial`; see `compile/cellconst.rs`). On cylindrical
    /// grids the centre is the `(ρ, α, z)` midpoint of the cell, like openEMS.
    pub fn cell_constant_material(mut self, on: bool) -> Self {
        self.cell_constant_material = on;
        self
    }

    /// These settings with the semantics they get when written to an
    /// openEMS file and read back ([`Fdtd::to_csx`], [`Fdtd::from_csx`]):
    /// a [`Waveform::Gaussian`] becomes openEMS's
    /// [`Waveform::OpenEmsGaussian`], the time step openEMS's criterion
    /// ([`TimestepMethod::Rennings`] unless one is set), and
    /// [`openems_run`](Self::openems_run) is on. A model run with these
    /// settings gives the same results as after a round trip through XML.
    pub fn file_equivalent(&self) -> Fdtd {
        let mut f = self.clone();
        if let Some(Waveform::Gaussian { f0, fc }) = f.waveform {
            f.waveform = Some(Waveform::OpenEmsGaussian { f0, fc });
        }
        f.timestep_method.get_or_insert(TimestepMethod::Rennings);
        f.openems_run = true;
        f
    }

    /// Sets the number of worker threads (≥ 1).
    pub fn threads(mut self, n: usize) -> Self {
        self.threads = Some(n);
        self
    }

    /// Sets the highest frequency of interest `f` \[Hz\] (openEMS `f_max`).
    /// Needed for waveforms without a bounded spectrum (Dirac, step) when
    /// the mesh resolution is checked; see [`Fdtd::get_max_frequency`].
    pub fn max_frequency(mut self, f: f64) -> Self {
        self.max_frequency = Some(f);
        self
    }

    /// Limits the simulated (physical) time to `t` \[s\] (openEMS
    /// `MaxTime`); the run stops after `ceil(t/dt)` steps if that is fewer
    /// than [`max_timesteps`](Fdtd::max_timesteps).
    pub fn max_time(mut self, t: f64) -> Self {
        self.max_time = Some(t);
        self
    }

    /// Sets the probe / time-domain dump oversampling factor relative to
    /// the Nyquist rate of the maximum frequency (openEMS `OverSampling`,
    /// default 4). Stored for interoperability; not yet used by the engine.
    pub fn oversampling(mut self, n: u32) -> Self {
        self.oversampling = Some(n);
        self
    }

    /// Bloch (Floquet) boundaries with the fixed transverse wavevector `k`
    /// \[rad/m\] (design §5.6.2): every axis with a non-zero component
    /// must have [`Boundary::Periodic`] on both faces, and the phase shift
    /// per period becomes `ψ = k·L`. The simulation then runs the
    /// sine–cosine method (two engines, see
    /// [`BlochPair`](crate::engine::BlochPair)) and reports complex results
    /// (see the Bloch mode of [`Simulation`](crate::Simulation)). With a
    /// fixed `k` each frequency `f` sees its own incidence angle,
    /// `sin θ = |k|·c/(2πf)`.
    ///
    /// Without this setting, an oblique plane-wave excitation spanning a
    /// periodic cell selects the Bloch mode, with `k` derived from its
    /// direction at the [`bloch_frequency`](Fdtd::bloch_frequency).
    /// `k = [0; 3]` forces the Bloch mode at normal incidence.
    pub fn bloch_wavevector(mut self, k: [f64; 3]) -> Self {
        self.bloch_wavevector = Some(k);
        self
    }

    /// The design frequency \[Hz\] at which an oblique plane wave in a
    /// periodic cell travels along its propagation direction (default: the
    /// plane wave's dispersion-matching frequency, i.e. the centre of the
    /// waveform); see [`bloch_wavevector`](Fdtd::bloch_wavevector).
    pub fn bloch_frequency(mut self, f: f64) -> Self {
        self.bloch_frequency = Some(f);
        self
    }

    /// Azimuthal multigrid near the axis of a cylindrical grid (openEMS
    /// `MultiGrid`, design §5.15.1): radii in **drawing units** (like the
    /// mesh lines). Inside each radius the number of α lines is halved
    /// once more, so inside the smallest of `L` radii only every `2^L`-th
    /// α line is kept and the short arcs next to the axis no longer set
    /// the time step. Each radius snaps to the nearest `ρ` mesh line, which
    /// must have a cell on both sides; the number of α cells must be a
    /// multiple of `2^L`. Ignored (with a warning) on Cartesian grids. An
    /// empty list (the default) switches multigrid off.
    ///
    /// The zone boundaries couple the zones to second order for the z
    /// field and first order (in the arc `ρ·Δα`) for the azimuthal one, so
    /// keep the radii small: the time step is set by the innermost cells
    /// anyway (see [`Grid::with_multigrid`](crate::engine::Grid::with_multigrid)).
    pub fn multigrid(mut self, radii: impl Into<Vec<f64>>) -> Self {
        self.multigrid = radii.into();
        self
    }

    /// The multigrid radii (drawing units); empty when off.
    pub fn get_multigrid(&self) -> &[f64] {
        &self.multigrid
    }

    /// The explicit Bloch wavevector \[rad/m\], if set.
    pub fn get_bloch_wavevector(&self) -> Option<[f64; 3]> {
        self.bloch_wavevector
    }

    /// The Bloch design frequency \[Hz\], if set.
    pub fn get_bloch_frequency(&self) -> Option<f64> {
        self.bloch_frequency
    }

    /// The highest frequency of interest \[Hz\]: the explicit
    /// [`max_frequency`](Fdtd::max_frequency), else the waveform's
    /// [`Waveform::max_frequency`].
    pub fn get_max_frequency(&self) -> Option<f64> {
        self.max_frequency
            .or_else(|| self.waveform.as_ref().and_then(Waveform::max_frequency))
    }

    /// The explicitly set highest frequency of interest \[Hz\], if any.
    pub fn get_explicit_max_frequency(&self) -> Option<f64> {
        self.max_frequency
    }

    /// The simulated time limit \[s\], if any.
    pub fn get_max_time(&self) -> Option<f64> {
        self.max_time
    }

    /// The oversampling factor, if set.
    pub fn get_oversampling(&self) -> Option<u32> {
        self.oversampling
    }

    /// The step limit for time step `dt` \[s\]: [`max_timesteps`](Fdtd::max_timesteps),
    /// reduced to `ceil(max_time/dt)` (at least 1; `floor(max_time/dt)` with
    /// [`openems_run`](Self::openems_run)) when a
    /// [`max_time`](Fdtd::max_time) is set.
    pub fn timestep_limit(&self, dt: f64) -> u64 {
        match self.max_time {
            Some(t) if dt > 0.0 => {
                // Round first so that e.g. 1e-9 / 1e-11 = 100.000…01 gives 100.
                let r = t / dt;
                let n = if (r - r.round()).abs() <= 1e-9 * r {
                    r.round()
                } else if self.openems_run {
                    r.floor()
                } else {
                    r.ceil()
                };
                if n < self.max_timesteps as f64 {
                    (n as u64).max(u64::from(!self.openems_run))
                } else {
                    self.max_timesteps
                }
            }
            _ => self.max_timesteps,
        }
    }

    /// The excitation waveform, if set.
    pub fn get_waveform(&self) -> Option<&Waveform> {
        self.waveform.as_ref()
    }

    /// The boundary conditions.
    pub fn get_boundaries(&self) -> &Boundaries {
        &self.boundaries
    }

    /// Mutable access to the boundary conditions.
    pub fn boundaries_mut(&mut self) -> &mut Boundaries {
        &mut self.boundaries
    }

    /// The end criteria, if any.
    pub fn get_end_criteria(&self) -> Option<&EndCriteria> {
        self.end_criteria.as_ref()
    }

    /// The maximum number of timesteps.
    pub fn get_max_timesteps(&self) -> u64 {
        self.max_timesteps
    }

    /// The Courant safety factor `S`.
    pub fn get_timestep_factor(&self) -> f64 {
        self.timestep_factor
    }

    /// The time step override \[s\], if any.
    pub fn get_timestep(&self) -> Option<f64> {
        self.timestep
    }

    /// Whether materials are sampled at cell centres (see
    /// [`cell_constant_material`](Self::cell_constant_material)).
    pub fn get_cell_constant_material(&self) -> bool {
        self.cell_constant_material
    }

    /// Whether openEMS run semantics are used (see
    /// [`openems_run`](Self::openems_run)).
    pub fn get_openems_run(&self) -> bool {
        self.openems_run
    }

    /// The openEMS Nyquist interval in steps for time step `dt`
    /// ([`Waveform::nyquist_interval`] of the waveform, 1 without one).
    /// openEMS evaluates its end criterion, and samples frequency-domain
    /// probes, at this interval; time-domain probes every
    /// `max(1, ⌊interval/OverSampling⌋)` steps (OverSampling default 4).
    pub fn openems_nyquist_interval(&self, dt: f64) -> u64 {
        self.waveform.as_ref().map_or(1, |w| w.nyquist_interval(dt))
    }

    /// The openEMS time-domain probe sampling interval in steps for time
    /// step `dt`: `max(1, ⌊N/OS⌋)` with `N` the
    /// [Nyquist interval](Self::openems_nyquist_interval) and `OS` the
    /// [oversampling](Self::oversampling) (default 4; values below 2 are
    /// replaced by 4, as openEMS does).
    pub fn openems_probe_interval(&self, dt: f64) -> u64 {
        let os = self.oversampling.filter(|&o| o >= 2).unwrap_or(4) as u64;
        (self.openems_nyquist_interval(dt) / os).max(1)
    }

    /// The openEMS time-step criterion, if any (see
    /// [`timestep_method`](Self::timestep_method)).
    pub fn get_timestep_method(&self) -> Option<TimestepMethod> {
        self.timestep_method
    }

    /// The engine time-step request for these settings: with a
    /// [`timestep_method`](Self::timestep_method),
    /// [`TimeStep::OpenEms`]; else a fixed `dt` override or the Courant
    /// factor.
    pub fn engine_timestep(&self) -> TimeStep {
        match self.timestep_method {
            Some(method) => TimeStep::OpenEms(OpenEmsTimestep {
                method,
                factor: self.timestep_factor,
                fixed: self.timestep,
                period: match self.waveform {
                    Some(Waveform::Sinusoid { f0 }) if f0 > 0.0 => Some(1.0 / f0),
                    _ => None,
                },
            }),
            None => match self.timestep {
                Some(dt) => TimeStep::Fixed(dt),
                None => TimeStep::Courant(self.timestep_factor),
            },
        }
    }

    /// The configured thread count; `None` lets the engine choose
    /// ([`engine::default_threads`](crate::engine::default_threads)).
    pub fn get_threads(&self) -> Option<usize> {
        self.threads
    }

    /// The thread count for the compile step (discretisation): the
    /// configured one, else [`std::thread::available_parallelism`] (1 if
    /// unknown). The engine's default is
    /// [`engine::default_threads`](crate::engine::default_threads).
    pub fn resolved_threads(&self) -> usize {
        self.threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }

    /// Checks all settings that can be checked without the mesh.
    ///
    /// A missing waveform is not an error here: whether one is needed
    /// depends on the model's excitations, which the compile step checks.
    pub fn validate(&self) -> Result<()> {
        if let Some(w) = &self.waveform {
            w.validate()?;
        }
        self.boundaries.validate()?;
        if let Some(e) = &self.end_criteria {
            e.validate()?;
        }
        if self.max_timesteps == 0 && !self.openems_run {
            return Err(Error::Invalid("max_timesteps must be at least 1".into()));
        }
        let s = self.timestep_factor;
        if !(s > 0.0 && s <= 1.0) {
            return Err(Error::Invalid(format!(
                "timestep factor {s} must be in (0, 1]"
            )));
        }
        if let Some(r) = self
            .multigrid
            .iter()
            .find(|r| !(r.is_finite() && **r > 0.0))
        {
            return Err(Error::Invalid(format!(
                "multigrid radius {r} must be finite and > 0"
            )));
        }
        if let Some(dt) = self.timestep
            && !(dt.is_finite() && dt > 0.0)
        {
            return Err(Error::Invalid(format!(
                "timestep {dt} must be finite and > 0"
            )));
        }
        if self.threads == Some(0) {
            return Err(Error::Invalid("thread count must be at least 1".into()));
        }
        if let Some(f) = self.max_frequency
            && !(f.is_finite() && f > 0.0)
        {
            return Err(Error::Invalid(format!(
                "maximum frequency {f} must be finite and > 0"
            )));
        }
        if let Some(t) = self.max_time
            && !(t.is_finite() && t > 0.0)
        {
            return Err(Error::Invalid(format!(
                "maximum time {t} must be finite and > 0"
            )));
        }
        if self.oversampling == Some(0) {
            return Err(Error::Invalid("oversampling must be at least 1".into()));
        }
        if let Some(k) = self.bloch_wavevector {
            for a in Axis::ALL {
                let ka = k[a.index()];
                if !ka.is_finite() {
                    return Err(Error::Invalid(format!(
                        "Bloch wavevector component {ka} along {a} must be finite"
                    )));
                }
                if ka != 0.0 && !self.boundaries.is_periodic(a) {
                    return Err(Error::Invalid(format!(
                        "Bloch wavevector has a component along {a}, whose faces are not \
                         periodic"
                    )));
                }
            }
        }
        if let Some(f) = self.bloch_frequency
            && !(f.is_finite() && f > 0.0)
        {
            return Err(Error::Invalid(format!(
                "Bloch design frequency {f} must be finite and > 0"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{C0, Z0};

    #[test]
    fn face_helpers() {
        for (i, f) in Face::ALL.into_iter().enumerate() {
            assert_eq!(f.index(), i);
            assert_eq!(Face::from_index(i), Some(f));
            assert_eq!(Face::new(f.axis(), f.is_max()), f);
            assert_eq!(f.opposite().opposite(), f);
            assert_ne!(f.opposite(), f);
            assert_eq!(f.opposite().axis(), f.axis());
        }
        assert_eq!(Face::from_index(6), None);
        assert_eq!(Face::new(Axis::Y, true), Face::YMax);
        assert_eq!(Face::ZMin.axis(), Axis::Z);
        assert!(!Face::XMin.is_max());
        assert_eq!(Face::XMax.to_string(), "xmax");
    }

    #[test]
    fn boundaries() {
        let b = Boundaries::default();
        assert!(b.iter().all(|(_, b)| *b == Boundary::Pec));
        let b = Boundaries::all(Boundary::pml(8))
            .with(Face::ZMin, Boundary::Pec)
            .with(Face::XMax, Boundary::mur());
        assert_eq!(*b.get(Face::ZMin), Boundary::Pec);
        assert_eq!(b[Face::XMax], Boundary::Mur(Mur::new()));
        assert_eq!(b[Face::YMin], Boundary::Pml(Cpml::default()));
        assert_eq!(b[Face::YMin].cells(), 8);
        assert_eq!(b[Face::ZMin].cells(), 0);
        let mut b2 = b;
        b2[Face::ZMin] = Boundary::Pmc;
        b2.set(Face::ZMax, Boundary::Pmc);
        assert_eq!(b2.as_array()[4], Boundary::Pmc);
        assert_eq!(b2.as_array()[5], Boundary::Pmc);
        assert!(b.validate().is_ok());
        let bad = b.with(Face::YMax, Boundary::pml(0));
        let e = bad.validate().unwrap_err().to_string();
        assert!(e.contains("ymax"), "{e}");
        assert_eq!(
            Boundaries::from([Boundary::Pmc; 6]),
            Boundaries::all(Boundary::Pmc)
        );
    }

    #[test]
    fn periodic_pairs() {
        let b = Boundaries::all(Boundary::pml(8))
            .with(Face::XMin, Boundary::Periodic)
            .with(Face::XMax, Boundary::Periodic);
        assert!(b.validate().is_ok());
        assert!(b.is_periodic(Axis::X));
        assert!(!b.is_periodic(Axis::Y));
        let e = b
            .with(Face::YMax, Boundary::Periodic)
            .validate()
            .unwrap_err()
            .to_string();
        assert!(e.contains("ymax") && e.contains("ymin"), "{e}");
        assert_eq!(Boundary::Periodic.cells(), 0);
    }

    #[test]
    fn cpml() {
        let p = Cpml::cells(10)
            .grading_order(4.0)
            .sigma_factor(0.5)
            .kappa_max(5.0)
            .alpha_max(0.05);
        assert_eq!(p.cells, 10);
        assert_eq!(p.grading_order, 4.0);
        assert!(p.validate().is_ok());
        let d = Cpml::default();
        assert_eq!((d.cells, d.grading_order, d.sigma_factor), (8, 3.0, 1.0));
        assert_eq!((d.kappa_max, d.alpha_max), (1.0, 0.0));
        let s = d.sigma_max(Z0, 1e-3);
        assert!((s - 0.8 * 4.0 / (Z0 * 1e-3)).abs() < 1e-9 * s);
        assert!((p.sigma_max(Z0, 1e-3) - 0.5 * 0.8 * 5.0 / (Z0 * 1e-3)).abs() < 1e-9);
        assert!(Cpml::cells(0).validate().is_err());
        assert!(d.kappa_max(0.5).validate().is_err());
        assert!(d.alpha_max(-1.0).validate().is_err());
        assert!(d.sigma_factor(f64::NAN).validate().is_err());
        assert!(d.grading_order(-1.0).validate().is_err());
    }

    #[test]
    fn mur() {
        assert_eq!(Mur::new().velocity(), C0);
        assert_eq!(Mur::with_phase_velocity(1e8).velocity(), 1e8);
        assert!(Mur::new().validate().is_ok());
        assert!(Mur::with_phase_velocity(0.0).validate().is_err());
        assert!(
            Boundary::Mur(Mur::with_phase_velocity(f64::INFINITY))
                .validate()
                .is_err()
        );
    }

    #[test]
    fn end_criteria() {
        let e = EndCriteria::energy_decay_db(-40.0);
        assert!((e.energy_ratio - 1e-4).abs() < 1e-18);
        assert!((e.decay_db() + 40.0).abs() < 1e-12);
        assert_eq!(e.check_interval, EndCriteria::DEFAULT_CHECK_INTERVAL);
        let e = EndCriteria::energy_ratio(1e-5).check_interval(10);
        assert_eq!(e.check_interval, 10);
        assert!(e.validate().is_ok());
        assert!(e.check_interval(0).validate().is_err());
        assert!(EndCriteria::energy_ratio(0.0).validate().is_err());
        assert!(EndCriteria::energy_ratio(1.5).validate().is_err());
        assert!(EndCriteria::energy_ratio(f64::NAN).validate().is_err());
        assert!(EndCriteria::energy_decay_db(0.0).validate().is_ok());
    }

    #[test]
    fn fdtd_builder_and_getters() {
        let f = Fdtd::new();
        assert!(f.get_waveform().is_none());
        assert_eq!(*f.get_boundaries(), Boundaries::default());
        assert!(f.get_end_criteria().is_none());
        assert_eq!(f.get_max_timesteps(), Fdtd::DEFAULT_MAX_TIMESTEPS);
        assert_eq!(f.get_timestep_factor(), 0.99);
        assert_eq!(f.get_timestep(), None);
        assert_eq!(f.get_threads(), None);
        assert!(f.resolved_threads() >= 1);
        assert!(f.validate().is_ok());

        let mut f = Fdtd::new()
            .waveform(Waveform::gaussian(1e9, 0.5e9))
            .boundaries(Boundaries::all(Boundary::pml(8)))
            .boundary(Face::ZMin, Boundary::Pec)
            .end_criteria(EndCriteria::energy_decay_db(-50.0))
            .max_timesteps(1234)
            .timestep_factor(0.5)
            .timestep(1e-12)
            .threads(3);
        assert!(f.validate().is_ok());
        assert!(matches!(
            f.get_waveform(),
            Some(Waveform::Gaussian { f0: 1e9, .. })
        ));
        assert_eq!(f.get_boundaries()[Face::ZMin], Boundary::Pec);
        assert_eq!(f.get_boundaries()[Face::ZMax], Boundary::pml(8));
        assert!(f.get_end_criteria().is_some());
        assert_eq!(f.get_max_timesteps(), 1234);
        assert_eq!(f.get_timestep_factor(), 0.5);
        assert_eq!(f.get_timestep(), Some(1e-12));
        assert_eq!((f.get_threads(), f.resolved_threads()), (Some(3), 3));
        f.boundaries_mut()[Face::XMin] = Boundary::Pmc;
        assert_eq!(f.get_boundaries()[Face::XMin], Boundary::Pmc);
        assert!(f.clone().no_end_criteria().get_end_criteria().is_none());
    }

    #[test]
    fn fdtd_validation() {
        let ok = Fdtd::new();
        assert!(
            ok.clone()
                .waveform(Waveform::gaussian(1e9, 0.0))
                .validate()
                .is_err()
        );
        assert!(
            ok.clone()
                .boundary(Face::XMin, Boundary::pml(0))
                .validate()
                .is_err()
        );
        assert!(
            ok.clone()
                .end_criteria(EndCriteria::energy_ratio(-1.0))
                .validate()
                .is_err()
        );
        assert!(ok.clone().max_timesteps(0).validate().is_err());
        assert!(ok.clone().timestep_factor(0.0).validate().is_err());
        assert!(ok.clone().timestep_factor(1.01).validate().is_err());
        assert!(ok.clone().timestep_factor(f64::NAN).validate().is_err());
        assert!(ok.clone().timestep_factor(1.0).validate().is_ok());
        assert!(ok.clone().timestep(0.0).validate().is_err());
        assert!(ok.clone().timestep(f64::INFINITY).validate().is_err());
        assert!(ok.clone().threads(0).validate().is_err());
        assert!(ok.clone().max_frequency(0.0).validate().is_err());
        assert!(ok.clone().max_time(f64::NAN).validate().is_err());
        assert!(ok.clone().oversampling(0).validate().is_err());
    }

    #[test]
    fn max_frequency_time_and_oversampling() {
        let f = Fdtd::new();
        assert_eq!(f.get_max_frequency(), None);
        assert_eq!(f.get_max_time(), None);
        assert_eq!(f.get_oversampling(), None);
        assert_eq!(f.timestep_limit(1e-12), Fdtd::DEFAULT_MAX_TIMESTEPS);
        let f = f.waveform(Waveform::gaussian(1e9, 5e8));
        assert_eq!(f.get_max_frequency(), Some(1.5e9));
        assert_eq!(f.get_explicit_max_frequency(), None);
        let f = f
            .max_frequency(3e9)
            .max_time(1e-9)
            .oversampling(8)
            .max_timesteps(500);
        assert!(f.validate().is_ok());
        assert_eq!(f.get_max_frequency(), Some(3e9));
        assert_eq!(
            (f.get_max_time(), f.get_oversampling()),
            (Some(1e-9), Some(8))
        );
        assert_eq!(f.timestep_limit(1e-11), 100);
        assert_eq!(f.timestep_limit(3e-12), 334);
        assert_eq!(f.timestep_limit(1e-13), 500);
        assert_eq!(f.timestep_limit(1.0), 1);
    }
}
