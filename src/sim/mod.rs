//! High-level simulation: a csx model plus [`Fdtd`] settings, compiled
//! into a ready-to-run [`Engine`] with the model's sources, lumped
//! elements, probes and dumps mapped onto engine operators (design §4,
//! §5.5).

mod bloch;
mod map;
mod results;
#[cfg(feature = "xml")]
mod xml;

pub use results::{
    DumpData, DumpResult, IncidentWave, ProbeKind, ProbeLocation, ProbeResult, ProbeValues,
    Results, SarResult,
};

use crate::Result;
use crate::compile::{BoundaryHandles, CompileOptions, Compiled, ModelSources, NodeMap, compile};
use crate::engine::boundary::periodic::{BlochCoupling, BlochStepper};
use crate::engine::run::drive;
use crate::engine::{
    BlochPair, EdgeSource, Engine, FreqDump, Grid, LumpedElement, OperatorHandle,
    PeriodicPlaneWave, PlaneWave, Progress, RunLimits, RunOutcome, TimeDump, WavePart,
};
use crate::fdtd::Fdtd;
use crate::nf2ff::{Nf2ffBox, Nf2ffHandle, Nf2ffResult};
use crate::post::sar::SarVolume;
use map::{BlochMap, DumpOp, Mapped, ProbeOp};

/// A compiled simulation: the [`Engine`] with the model's excitations,
/// lumped elements, probes and dumps installed, plus the metadata needed to
/// relate engine results back to the model.
///
/// ```
/// use csx::primitives::BoxShape;
/// use csx::{ContinuousStructure, Primitive, Property};
/// use ems::{Fdtd, Simulation};
///
/// let mut model = ContinuousStructure::new();
/// model.grid.delta_unit = 1e-3; // mm
/// for a in 0..3 {
///     model.grid.set_lines(a, (0..=10).map(|i| i as f64))?;
/// }
/// let diel = model.add_property(Property::material("diel", csx::Material::new(4.0, 0.0)));
/// model.add_primitive(diel, Primitive::new(BoxShape::new([0.0, 0.0, 0.0], [10.0, 10.0, 5.0])))?;
///
/// let mut sim = Simulation::new(&model, &Fdtd::new().max_timesteps(10).threads(1))?;
/// assert_eq!(sim.grid().lines(ems::Axis::X)[10], 0.01); // metres
/// let res = sim.run()?;
/// assert_eq!(res.outcome().unwrap().steps, 10);
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Mapping of the model
///
/// After the materials and boundaries ([`compile`]), the model's
/// remaining properties become engine operators, registered in this order
/// (which is also their order within each hook):
///
/// 1. **Lumped elements** (csx `LumpedElement`, values per edge as
///    distributed by csx: `R·p/s`, `L·p/s`, `C·s/p` for `s` series and `p`
///    parallel edges) → one edge-list [`LumpedElement`] per property, with
///    `LEtype` parallel (default) or series. Unset (NaN) values are absent;
///    non-positive ones are dropped with a warning. Edges that are PEC are
///    dropped with a warning. With `Caps` set (the csx default), the
///    transverse E edges of the two end planes of the element (within its
///    transverse extent) are made PEC, i.e. metal end plates. Every
///    element records its terminal current
///    ([`Results::lumped_current`]). With [`Fdtd::openems_run`] (openEMS
///    files) the elements behave like openEMS's (black-box comparison,
///    design §5.16.1): a parallel element's `C` *replaces* the capacitance
///    of its edges' medium and its `R` the medium's conductance (otherwise
///    the element adds to the medium); a parallel R/C element without
///    inductor is spread like a homogeneous material (per-edge `G`, `C`
///    proportional to the edge's dual area over its length, same totals)
///    instead of in equal parts; parallel inductors use the explicit
///    update ([`LumpedElement::explicit_inductor`]) unless it would be
///    unstable (warning).
/// 2. **Excitations** (csx `Excitation`, only enabled ones), driven by the
///    settings' [`Waveform`](crate::Waveform) with openEMS's timing
///    (established by black-box comparison with openEMS v0.37, see
///    `tests/openems_core.rs`): a field value at time `T` (E at
///    `(n + 1)·dt`, H at `(n + ½)·dt`) receives the signal sample at
///    `T − dt − D·dt`, where `D = ⌊Delay/dt⌋` (openEMS delays by whole
///    steps, rounding down, computed in floating point: 2e-10 s at
///    `dt = 1e-9/525` s is 104 steps), and nothing before the signal's
///    first sample. The samples follow the waveform's
///    [sampling convention](crate::waveform#sampling-convention). csx gives the field
///    amplitude per edge (V/m for E types, A/m for H types, including the
///    weighting function); the integral-form weight is that amplitude
///    times the edge length (primary edge for E, dual edge for H) in
///    metres. Kinds and sample times are those of
///    [`SourceKind`](crate::engine::SourceKind):
///    * `EFieldSoft` adds `w·a(t)` to the edge voltage. **On an edge of a
///      lumped element with a resistance (parallel) or of a series element**
///      the excitation instead becomes that element's Thevenin source with
///      open-circuit voltage `w` per edge (design §5.5): the resistor current
///      is `(u − w·a)/R_e`, so a lumped resistor plus an excitation over
///      the same box is a resistive voltage source of `Σw = E·L` volts per
///      column — exactly the engine's [`LumpedPort`](crate::engine::LumpedPort)
///      model, so XML ports behave like native ones. With
///      [`Fdtd::openems_run`] the excitation is instead what openEMS does
///      on every element: a plain soft source added after the element's
///      update, with the element's branch driven by the edge voltage
///      without it (the stored voltage minus the previous step's source:
///      a [`LumpedElement::branch_drive_source`] for an inductor, a
///      half-voltage source one step later for a series element; a
///      folded parallel R/C sees the whole voltage). Agreement ~1e-6.
///    * `EFieldHard` overwrites the edge voltage (after the lumped
///      updates, so on a lumped element's edge the element has no effect:
///      warning), `HFieldSoft` / `HFieldHard` add to / overwrite the H
///      dual-edge currents.
///    * `PlaneWave` (type 10) excitations become total-field /
///      scattered-field [`PlaneWave`](crate::engine::PlaneWave) operators,
///      one per primitive box, with incident E `Excite·a(t − Delay)` at the
///      box corner the wave enters first (mapping rules in
///      [`crate::compile::planewave`]); they are registered after all other
///      sources.
///
///    Without a waveform in the settings, excitations are ignored with a
///    warning.
/// 3. **Probes** (csx `ProbeBox`, corners snapped to mesh nodes by the
///    compile step, see [`crate::compile`]), all multiplied by the probe
///    `Weight`:
///    The corners are snapped like openEMS does (black-box comparison):
///    to the nearest mesh line, a tie going to the lower line, and for H
///    and current probes to the dual lines (cell centres) where noted.
///    Every probe records every step (openEMS writes its files every
///    [`Results::openems_probe_interval`] steps, see
///    [`crate::io::openems`]); [`ProbeResult::location`] gives the
///    snapped position as openEMS reports it.
///    * `Voltage`: `∫E·dl` from `start` to `stop` along the staircase path
///      x, then y, then z (edges traversed against their direction count
///      negative, so swapping start and stop flips the sign). Sampled at
///      the E times `(n + 1)·dt`. (openEMS accepts lines only.)
///    * `Current`: the total current through the box cross-section normal
///      to `NormDir` (or to the box's only flat axis), counted positive
///      from `start` towards `stop` along the normal (along `+normal` for a
///      flat box), measured as `∮H·dl` around the E edges whose transverse
///      node coordinates are inside the box. A flat box integrates in the
///      dual plane nearest to its (unsnapped) normal coordinate, ties to
///      the lower one (openEMS: a box on the node plane `k` uses the plane
///      between `k − 1` and `k`). A box with an extent along the normal
///      (which openEMS disables) uses the dual plane through its middle
///      edge layer. Sampled at the H times `(n + ½)·dt`.
///    * `EField` / `HField`: the **native** field components at the
///      (snapped) start corner of the box, at the E / H times, like
///      openEMS: for E at node `(i, j, k)` the three primary edges leaving
///      it (`E_x` on the edge from `i` to `i + 1`, …; a component without
///      an edge there reads 0); for H at the dual node `(i + ½, j + ½,
///      k + ½)` (start corner snapped to the dual lines) `H_x` of the dual
///      edge at `(i, j + ½, k + ½)`, `H_y` at `(i + ½, j, k + ½)` and `H_z`
///      at `(i + ½, j + ½, k)`. The values are the edge voltages / currents
///      divided by the (dual) edge length. A box reads its start corner
///      only.
///    * `WaveguideVoltage` / `WaveguideCurrent`: the mode-weighted
///      integral of tangential E / H over the probe plane, with the mode
///      function of the property attributes `ModeFunctionX/Y/Z` (1/m, over
///      drawing-unit coordinates; see the
///      [compile docs](crate::compile#waveguide-probes)), at the E / H
///      times. They are reported as [`ProbeKind::Voltage`] /
///      [`ProbeKind::Current`].
///      [`RectWaveguidePort`](crate::port::RectWaveguidePort) writes such
///      probes.
///
///    `FD_Samples` frequencies are available through
///    [`ProbeResult::spectrum`]; any other frequency through
///    [`ProbeResult::dft`].
/// 4. **Dumps** (csx `DumpBox`): `DumpType` 0–5 (E, H, J, rot H, D, B)
///    → in-memory [`TimeDump`], 10–15 → [`FreqDump`] at the property's
///    frequencies, sampled with the openEMS rules of
///    [`io::openems::dump`](crate::io::openems::dump): box snapping,
///    `DumpMode` (0 raw edge values, 1 nodes, 2 cell centres),
///    `SubSampling`, `OptResolution`, openEMS step timing
///    ([`DumpTiming::OpenEms`](crate::engine::DumpTiming)), the time-domain
///    interval `⌊N/OverSampling⌋` and frequency-domain interval
///    `min(N, ⌊1/(2·f·dt)⌋)` for the Nyquist step count `N` of the
///    excitation, and the `StartTime` / `StopTime` step window. [`run`](Self::run)
///    adds the step after the last one like openEMS. Write openEMS files
///    with [`DumpResult::write_openems`] (frequency-domain values there are
///    twice [`FreqDump::values`], the openEMS normalisation), ems VTR files
///    with [`DumpResult::write_vtr`].
/// 5. **SAR dumps** (csx `DumpBox` of type `SARLocal`, `SAR1g`, `SAR10g`,
///    `SARRaw`): an [`EdgeDft`](crate::engine::EdgeDft) of the E edges of
///    the box's cells at the property's frequencies (interval `K` of
///    [`Fdtd::get_max_frequency`] and the oversampling factor), with the
///    cell conductivity and density sampled from the model
///    ([`compile::sar`](crate::compile::sar)), plus the openEMS-style
///    cell-centre E DFT for the openEMS SAR files
///    ([`SarResult::write_openems`]). They are reported in
///    [`Results::sar_dumps`] (not in [`Results::dumps`]) as
///    [`SarResult`]s; see [`post::sar`](crate::post::sar) for the
///    evaluation. Other dump types are ignored (warning).
///
/// # Bloch mode
///
/// Periodic structures at oblique incidence (design §5.6.2) run the
/// sine–cosine method: two engines carry the real and imaginary parts of
/// a Bloch-periodic complex field, coupled at the periodic faces with the
/// phase shift `ψ = k_t·L` per period (see
/// [`BlochPair`](crate::engine::BlochPair)). The mode is selected
///
/// * explicitly by [`Fdtd::bloch_wavevector`] (any transverse wavevector on
///   the periodic axes, `[0; 3]` included), or
/// * by an **oblique plane-wave excitation spanning a periodic cell**: its
///   box covers every periodic axis and all but one other axis (the
///   normal) completely, and its direction has a transverse part; then
///   `k_t = (2π f_d/c)·k̂_t` with the design frequency `f_d`
///   ([`Fdtd::bloch_frequency`], else the plane wave's dispersion-matching
///   frequency, the centre of the waveform). With `k_t` fixed, each
///   frequency `f` travels at its own angle, `sin θ = |k_t|·c/(2πf)`;
///   frequencies below `|k_t|·c/(2π)` are evanescent.
///
/// The model is mapped onto both engines alike (materials, boundaries,
/// lumped elements, probes, dumps); excitations (soft / hard sources and
/// Thevenin sources) drive only the real engine. Plane waves spanning the
/// cell become [`PeriodicPlaneWave`] sources injected through the box
/// faces normal to the open axis (a face on the domain boundary gets no
/// corrections, so the total-field region then reaches through the
/// absorbing layer), driven on both engines with their part of the
/// Bloch-phased incident field; other plane waves are ignored (warning).
/// SAR dumps are not supported (warning), nor NF2FF boxes (error).
///
/// The results are complex: [`ProbeResult::imag`] holds the imaginary
/// engine's samples and the probe DFTs combine both,
/// `X(f) = DFT[x_re] + j·DFT[x_im]`; frequency-domain dumps hold the
/// complex field, time-domain dumps the real part plus
/// [`DumpResult::imag_snapshots`]. Each periodic plane wave's incident E on
/// its entry plane ([`Results::incident`]) is the reference for
/// reflection and transmission coefficients. The energy end criterion uses
/// the energy of both engines.
#[derive(Debug)]
pub struct Simulation {
    engine: Engine,
    fdtd: Fdtd,
    delta_unit: f64,
    node_map: NodeMap,
    boundaries: BoundaryHandles,
    sources: ModelSources,
    mapped: Mapped,
    warnings: Vec<String>,
    bloch: Option<Box<BlochState>>,
}

/// The second engine of a Bloch run and its coupling to the first.
#[derive(Debug)]
struct BlochState {
    imag: Engine,
    coupling: BlochCoupling,
    mapped: Mapped,
    k: [f64; 3],
}

impl Simulation {
    /// Compiles `model` with `fdtd` using the default [`CompileOptions`].
    ///
    /// # Errors
    /// See [`compile`](crate::compile::compile) and
    /// [`Compiled::build`](crate::compile::Compiled::build); also invalid
    /// lumped-element values.
    pub fn new(model: &csx::ContinuousStructure, fdtd: &Fdtd) -> Result<Self> {
        Self::with_options(model, fdtd, &CompileOptions::default())
    }

    /// Compiles `model` with `fdtd` and explicit compile options.
    ///
    /// With [`CompileOptions::mesh_check`] (the default) the compiled mesh
    /// is checked by [`crate::mesh::check_compiled`] (default
    /// [`CheckOptions`](crate::mesh::CheckOptions)) and every issue of
    /// [`Severity::Warning`](crate::mesh::Severity::Warning) or worse is
    /// appended to [`warnings`](Self::warnings), as `"mesh <severity>:
    /// …"` lines. The csx discretisation warnings the check re-reports are
    /// not repeated (they are already in the list).
    ///
    /// # Errors
    /// See [`new`](Self::new).
    pub fn with_options(
        model: &csx::ContinuousStructure,
        fdtd: &Fdtd,
        opts: &CompileOptions,
    ) -> Result<Self> {
        let compiled = compile(model, fdtd, opts)?;
        let k = bloch::wavevector(fdtd, &compiled.sources.plane_waves, compiled.grid.nodes())?;
        let (mut engine, boundaries, imag) = match k {
            None => {
                let (e, h) = compiled.build()?;
                (e, h, None)
            }
            Some(k) => {
                let psi = BlochPair::phase_for(&compiled.grid, k);
                let (coupling, b) = BlochCoupling::new(compiled.builder(), psi)?;
                let mut re = b.clone().build()?;
                let mut im = b.build()?;
                let h = compiled.add_boundary_operators(&mut re)?;
                compiled.add_boundary_operators(&mut im)?;
                (re, h, Some((im, coupling, k)))
            }
        };
        let mut warnings = compiled.warnings.clone();
        if let Some(w) = boundaries.conformal.as_ref().and_then(|r| r.warning()) {
            warnings.push(w);
        }
        if opts.mesh_check {
            warnings.extend(mesh_warnings(model, fdtd, &compiled, opts)?);
        }
        let part = |part| k.map(|k| BlochMap { part, k });
        let mut mapped = map::map_sources(
            &mut engine,
            &compiled.sources,
            &compiled.node_map,
            fdtd,
            &mut warnings,
            part(WavePart::Real),
        )?;
        mapped.lumped_ids = mapped
            .lumped
            .iter()
            .map(|(n, _)| model.property_id(n).map_or(u32::MAX, |p| p.get()))
            .collect();
        let bloch = match imag {
            None => None,
            Some((mut imag, coupling, k)) => {
                // Same model, same order: the warnings are those of the
                // real engine.
                let mut ignored = Vec::new();
                let mapped = map::map_sources(
                    &mut imag,
                    &compiled.sources,
                    &compiled.node_map,
                    fdtd,
                    &mut ignored,
                    part(WavePart::Imag),
                )?;
                Some(Box::new(BlochState {
                    imag,
                    coupling,
                    mapped,
                    k,
                }))
            }
        };
        Ok(Self {
            engine,
            fdtd: fdtd.clone(),
            delta_unit: compiled.delta_unit,
            node_map: compiled.node_map,
            boundaries,
            sources: compiled.sources,
            mapped,
            warnings,
            bloch,
        })
    }

    /// The engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Mutable access to the engine (stepping, extra operators).
    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Consumes the simulation, returning the engine.
    pub fn into_engine(self) -> Engine {
        self.engine
    }

    /// The mesh in metres.
    pub fn grid(&self) -> &Grid {
        self.engine.grid()
    }

    /// The csx drawing unit in metres.
    pub fn delta_unit(&self) -> f64 {
        self.delta_unit
    }

    /// The solver settings this simulation was compiled with.
    pub fn fdtd(&self) -> &Fdtd {
        &self.fdtd
    }

    /// csx ↔ engine node index conversion.
    pub fn node_map(&self) -> &NodeMap {
        &self.node_map
    }

    /// Handles of the Mur / CPML boundary operators, if any.
    pub fn boundaries(&self) -> &BoundaryHandles {
        &self.boundaries
    }

    /// The discretised model parts that were mapped onto operators.
    pub fn sources(&self) -> &ModelSources {
        &self.sources
    }

    /// Compile and mapping warnings (csx model checks, discretisation,
    /// approximations, unsupported properties).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The lumped-element operator of the csx property `name`.
    pub fn lumped_element(&self, name: &str) -> Option<&LumpedElement> {
        let h = find(&self.mapped.lumped, name)?;
        Some(self.engine.operator(&h))
    }

    /// The source operator of the csx excitation `name` (none if the
    /// excitation became a lumped element's Thevenin source only).
    pub fn source(&self, name: &str) -> Option<&EdgeSource> {
        let h = find(&self.mapped.sources, name)?;
        Some(self.engine.operator(&h))
    }

    /// The plane-wave operator of the csx excitation `name` (the first one
    /// if the property has several primitives).
    pub fn plane_wave(&self, name: &str) -> Option<&PlaneWave> {
        let h = find(&self.mapped.plane_waves, name)?;
        Some(self.engine.operator(&h))
    }

    /// The periodic-cell plane-wave operator (real part) of the csx
    /// excitation `name` in a Bloch run.
    pub fn periodic_plane_wave(&self, name: &str) -> Option<&PeriodicPlaneWave> {
        let h = find(&self.mapped.periodic_waves, name)?;
        Some(self.engine.operator(&h))
    }

    /// Whether this is a Bloch run (two engines, complex results; see the
    /// [type docs](Self#bloch-mode)).
    pub fn is_bloch(&self) -> bool {
        self.bloch.is_some()
    }

    /// The Bloch wavevector \[rad/m\] of a Bloch run.
    pub fn bloch_wavevector(&self) -> Option<[f64; 3]> {
        self.bloch.as_ref().map(|b| b.k)
    }

    /// The Bloch phase shifts per period \[rad\] of a Bloch run.
    pub fn bloch_phase(&self) -> Option<[f64; 3]> {
        self.bloch.as_ref().map(|b| b.coupling.phase())
    }

    /// The engine of the imaginary part in a Bloch run ([`engine`](Self::engine)
    /// carries the real part).
    pub fn imag_engine(&self) -> Option<&Engine> {
        self.bloch.as_ref().map(|b| &b.imag)
    }

    /// The time-domain dump operator of the csx property `name`, for
    /// zero-copy access (e.g. [`TimeDump::write_vtr`]).
    pub fn time_dump(&self, name: &str) -> Option<&TimeDump> {
        match self.mapped.dumps.iter().find(|d| d.name == name)?.op {
            DumpOp::Time(h) => Some(self.engine.operator(&h)),
            DumpOp::Freq(_) => None,
        }
    }

    /// The frequency-domain dump operator of the csx property `name`.
    pub fn freq_dump(&self, name: &str) -> Option<&FreqDump> {
        match self.mapped.dumps.iter().find(|d| d.name == name)?.op {
            DumpOp::Freq(h) => Some(self.engine.operator(&h)),
            DumpOp::Time(_) => None,
        }
    }

    /// Adds a near-field to far-field box (see [`crate::nf2ff`]), with its
    /// corners in drawing units snapped to the nearest mesh lines. Call it
    /// before running; the box records from the next step on. A box
    /// reaching into a PML adds a [warning](Self::warnings). Unless
    /// [`Nf2ffBox::background`] sets it, the background medium is detected
    /// from the materials on the recorded faces (see
    /// [`crate::nf2ff#requirements`]); an inhomogeneous surface adds a
    /// warning and falls back to vacuum.
    ///
    /// # Errors
    /// See [`Nf2ff::with_faces`](crate::nf2ff::Nf2ff::with_faces);
    /// [`Error::Unsupported`](crate::Error::Unsupported) in a lossy
    /// background, or for a mirror not normal to z on a cylindrical grid
    /// (cylindrical boxes: [`crate::nf2ff#cylindrical-grids`]).
    pub fn add_nf2ff(&mut self, b: &Nf2ffBox) -> Result<Nf2ffHandle> {
        if self.bloch.is_some() {
            return Err(crate::Error::Unsupported(
                "NF2FF boxes in a Bloch run (periodic structures radiate Floquet modes, \
                 not a far field)"
                    .into(),
            ));
        }
        let k = map::dump_interval(&self.fdtd, self.engine.dt());
        let (op, mirrors) = b.build(self.engine.grid(), self.delta_unit, k)?;
        let n = self.engine.grid().nodes();
        for (face, bc) in self.fdtd.get_boundaries().iter() {
            let cells = bc.cells();
            if cells == 0 {
                continue;
            }
            let a = face.axis().index();
            let inside = if face.is_max() {
                op.hi()[a] + cells >= n[a] - 1
            } else {
                op.lo()[a] <= cells
            };
            if inside {
                self.warnings.push(format!(
                    "NF2FF box reaches into the PML of {face:?}; keep it inside the \
                     interior"
                ));
            }
        }
        let mut op = op;
        if b.get_background().is_none() {
            let mut faces = [false; 6];
            for f in op.faces() {
                faces[f.index()] = true;
            }
            // (Faces on a mirror plane are not recorded.)
            let co = self.engine.coefficients();
            let media = crate::engine::medium::CoefficientMedia {
                grid: self.engine.grid(),
                co: &co,
                dt: self.engine.dt(),
            };
            let found = crate::engine::medium::surface_medium(
                self.engine.grid(),
                &media,
                op.lo(),
                op.hi(),
                faces,
            );
            drop(co);
            match found {
                Ok(Some(m)) if !m.is_lossless() => {
                    return Err(crate::Error::Unsupported(format!(
                        "NF2FF box in a lossy background (kappa = {:.4e} S/m, sigma_m = {:.4e} \
                         Ohm/m): there is no far field in a lossy medium; set \
                         Nf2ffBox::background to use the lossless part",
                        m.kappa, m.sigma_m
                    )));
                }
                // (Values within f32 rounding of vacuum stay exactly vacuum.)
                Ok(Some(m)) if (m.eps_r - 1.0).abs() > 1e-6 || (m.mu_r - 1.0).abs() > 1e-6 => {
                    op = op.with_background(m.eps_r, m.mu_r)?;
                }
                Ok(_) => {}
                Err(e) => self.warnings.push(format!(
                    "NF2FF box: {e}; the transform assumes vacuum (keep the faces in one \
                     homogeneous medium, or set Nf2ffBox::background)"
                )),
            }
        }
        let op = self.engine.add_operator(op);
        Ok(Nf2ffHandle { op, mirrors })
    }

    /// The far-field result of an NF2FF box added with
    /// [`add_nf2ff`](Self::add_nf2ff), from the fields recorded so far,
    /// with the box's mirror planes applied.
    pub fn nf2ff(&self, h: &Nf2ffHandle) -> Nf2ffResult {
        let mut r = self.engine.operator(&h.op).result();
        for &m in &h.mirrors {
            r = r
                .with_mirror(m)
                .expect("mirror planes were validated when the box was added");
        }
        r
    }

    /// The run limits derived from the settings: at most
    /// [`Fdtd::timestep_limit`] steps for the engine time step (i.e.
    /// [`Fdtd::get_max_timesteps`], reduced by [`Fdtd::get_max_time`] if
    /// set; counted from the engine's current step, as
    /// [`RunLimits::max_timesteps`]), the settings' end criteria, and
    /// the waveform's [`duration`](crate::Waveform::duration) as the
    /// excitation end (`∞` for a waveform that never ends, none without a
    /// waveform). With plane-wave sources the excitation end is pushed
    /// back to when the incident pulse has left every total-field box
    /// (`Delay + duration +`
    /// [`PlaneWave::transit_time`](crate::engine::PlaneWave::transit_time)),
    /// so the energy criterion judges the scattered field only (see
    /// [`crate::engine::planewave`] on energy).
    ///
    /// With [`Fdtd::openems_run`] (openEMS files), the criteria are
    /// instead checked every [Nyquist interval](Fdtd::openems_nyquist_interval)
    /// from the first step on, without an excitation end.
    pub fn run_limits(&self) -> RunLimits {
        let end = self.fdtd.get_waveform().map(|w| {
            let d = w.duration().unwrap_or(f64::INFINITY);
            self.mapped.plane_waves.iter().fold(d, |end, (name, h)| {
                let delay = self
                    .sources
                    .plane_waves
                    .iter()
                    .filter(|p| &p.name == name)
                    .fold(0.0f64, |m, p| m.max(p.delay));
                end.max(delay + d + self.engine.operator(h).transit_time())
            })
        });
        let dt = self.engine.dt();
        if self.fdtd.get_openems_run() {
            let k = self.fdtd.openems_nyquist_interval(dt);
            return RunLimits::new(self.fdtd.timestep_limit(dt))
                .end_criteria(self.fdtd.get_end_criteria().map(|c| c.check_interval(k)))
                .check_after_interval(true);
        }
        RunLimits::new(self.fdtd.timestep_limit(dt))
            .end_criteria(self.fdtd.get_end_criteria().copied())
            .excitation_end(end)
    }

    /// Runs with [`run_limits`](Self::run_limits) via
    /// [`Engine::run_until`] and returns the [`results`](Self::results)
    /// with the run outcome.
    ///
    /// # Errors
    /// See [`Engine::run_until`].
    pub fn run(&mut self) -> Result<Results> {
        self.run_with(|_| core::ops::ControlFlow::Continue(()))
    }

    /// Runs like [`run`](Self::run), calling `progress` after every energy
    /// check (see [`Engine::run_until_with`]); returning
    /// [`ControlFlow::Break`](core::ops::ControlFlow::Break) stops the run.
    ///
    /// # Errors
    /// See [`Engine::run_until_with`].
    pub fn run_with<F>(&mut self, progress: F) -> Result<Results>
    where
        F: FnMut(&Progress) -> core::ops::ControlFlow<()>,
    {
        let limits = self.run_limits();
        self.clear_final_probe_samples();
        let outcome = self.run_engines(&limits, progress)?;
        self.finish_dumps();
        self.finish_probes();
        let mut r = self.results();
        r.outcome = Some(outcome);
        Ok(r)
    }

    /// Records the last openEMS dump step of a run (the state after the
    /// last step, with the H of one more H update evaluated locally; see
    /// [`FreqDump::final_sample`]), as openEMS does. Not in Bloch runs.
    fn finish_dumps(&mut self) {
        if self.bloch.is_some() {
            return;
        }
        for s in &self.mapped.sar {
            if let Some(h) = s.oe
                && let Some(f) = self.engine.operator(&h).final_sample(&self.engine)
            {
                self.engine.operator_mut(&h).apply_final(f);
            }
        }
        for d in &self.mapped.dumps {
            match d.op {
                DumpOp::Time(h) => {
                    if let Some(s) = self.engine.operator(&h).final_sample(&self.engine) {
                        self.engine.operator_mut(&h).apply_final(s);
                    }
                }
                DumpOp::Freq(h) => {
                    if let Some(s) = self.engine.operator(&h).final_sample(&self.engine) {
                        self.engine.operator_mut(&h).apply_final(s);
                    }
                }
            }
        }
    }

    /// Records, for the H-type probes (current, H field), the sample at
    /// `(N + ½)·dt` after the last step `N` that openEMS writes to its
    /// probe files (the H update of step `N` evaluated locally, see
    /// [`Probe::lookahead`](crate::engine::Probe::lookahead)); it is not
    /// part of [`ProbeResult::values`], only of the openEMS probe files
    /// ([`io::openems::probes`](crate::io::openems::probes)). Not in Bloch
    /// runs.
    fn finish_probes(&mut self) {
        if self.bloch.is_some() {
            return;
        }
        let engine = &self.engine;
        let no_step = engine.timestep() == 0;
        for p in &mut self.mapped.probes {
            // A run of 0 steps: openEMS still writes the initial (zero)
            // state of the E-type probes, at t = 0 (fixture
            // `sweep_fdtd_ts_and_steps0`, design §5.16.3).
            if no_step && !p.waveguide && matches!(p.kind, ProbeKind::Voltage | ProbeKind::EField) {
                let n = match p.op {
                    ProbeOp::Scalar(_) => 1,
                    ProbeOp::Native(hs) => hs.len(),
                };
                p.final_sample = Some((0.0, vec![0.0; n], None));
                continue;
            }
            p.final_sample = match p.op {
                ProbeOp::Scalar(h) => engine
                    .operator(&h)
                    .lookahead(engine)
                    .map(|(t, v, q)| (t, vec![v], q)),
                ProbeOp::Native(hs) => {
                    let s: Option<Vec<_>> = hs
                        .iter()
                        .map(|h| engine.operator(h).lookahead(engine))
                        .collect();
                    s.map(|s| (s[0].0, s.iter().map(|x| x.1).collect(), None))
                }
            };
        }
    }

    fn clear_final_probe_samples(&mut self) {
        for p in &mut self.mapped.probes {
            p.final_sample = None;
        }
    }

    /// The run loop: the engine's, or that of both engines of a Bloch run
    /// (energy summed over both).
    fn run_engines<F>(&mut self, limits: &RunLimits, progress: F) -> Result<RunOutcome>
    where
        F: FnMut(&Progress) -> core::ops::ControlFlow<()>,
    {
        match &mut self.bloch {
            None => self.engine.run_until_with(limits, progress),
            Some(b) => {
                let mut s = BlochStepper {
                    re: &mut self.engine,
                    im: &mut b.imag,
                    coupling: &b.coupling,
                };
                drive(&mut s, limits, progress)
            }
        }
    }

    /// Advances `n` steps (both engines in a Bloch run), without end
    /// criteria.
    pub fn step(&mut self, n: u64) {
        self.clear_final_probe_samples();
        match &mut self.bloch {
            None => self.engine.run(n),
            Some(b) => {
                for _ in 0..n {
                    b.coupling.step(&mut self.engine, &mut b.imag, false);
                }
            }
        }
    }

    /// A copy of everything recorded so far by the model's probes and
    /// dumps (no run outcome).
    pub fn results(&self) -> Results {
        let dt = self.engine.dt();
        let imag = self.bloch.as_deref().map(|b| (&b.imag, &b.mapped));
        let probes = self
            .mapped
            .probes
            .iter()
            .enumerate()
            .map(|(q, p)| {
                let (times, values) = probe_series(&self.engine, p);
                // Recording window (stop ≤ 0: no limit), with a small
                // tolerance against rounding of the sample times. openEMS
                // keeps one more sample at each end on the E side
                // (`start − dt ≤ t ≤ stop + dt`) and on the H side the
                // first after the stop (`start ≤ t ≤ stop + dt`), see
                // `ProbeResult`.
                let eps = 1e-9 * dt;
                let magnetic = matches!(p.kind, ProbeKind::Current | ProbeKind::HField);
                let lo_margin = if magnetic { 0.0 } else { dt };
                let lo = times.partition_point(|&t| t < p.start_time - lo_margin - eps);
                let hi = if p.stop_time > 0.0 {
                    times.partition_point(|&t| t <= p.stop_time + dt + eps)
                } else {
                    times.len()
                }
                .max(lo);
                let window = |v: results::ProbeValues| match v {
                    results::ProbeValues::Scalar(v) => {
                        results::ProbeValues::Scalar(v[lo..hi].to_vec())
                    }
                    results::ProbeValues::Vector(v) => {
                        results::ProbeValues::Vector(v[lo..hi].to_vec())
                    }
                };
                let imag = imag.map(|(e, m)| window(probe_series(e, &m.probes[q]).1));
                let mode_purity = match p.op {
                    ProbeOp::Scalar(h) if p.waveguide => {
                        Some(match self.engine.operator(&h).purity() {
                            Some(v) => v[lo..hi].to_vec(),
                            // Not identified (cylindrical grids).
                            None => vec![f64::NAN; hi - lo],
                        })
                    }
                    _ => None,
                };
                // The openEMS final sample, inside the recording window.
                let final_sample = p.final_sample.clone().filter(|(t, ..)| {
                    *t >= p.start_time - eps && (p.stop_time <= 0.0 || *t <= p.stop_time + dt + eps)
                });
                ProbeResult {
                    name: p.name.clone(),
                    id: p.id,
                    number: p.number,
                    kind: p.kind,
                    frequencies: p.frequencies.clone(),
                    dt,
                    times: times[lo..hi].to_vec(),
                    values: window(values),
                    imag,
                    location: p.location,
                    mode_purity,
                    final_sample,
                }
            })
            .collect();
        let lumped = self
            .mapped
            .lumped
            .iter()
            .zip(&self.mapped.lumped_ids)
            .enumerate()
            .map(|(q, ((name, h), &id))| {
                let op = self.engine.operator(h);
                let imag = imag.map(|(e, m)| {
                    results::ProbeValues::Scalar(e.operator(&m.lumped[q].1).current().to_vec())
                });
                ProbeResult {
                    name: name.clone(),
                    id,
                    number: 0,
                    kind: ProbeKind::Current,
                    frequencies: Vec::new(),
                    dt,
                    times: op.current_times(),
                    values: results::ProbeValues::Scalar(op.current().to_vec()),
                    imag,
                    location: None,
                    mode_purity: None,
                    final_sample: None,
                }
            })
            .collect();
        let dumps = self
            .mapped
            .dumps
            .iter()
            .enumerate()
            .map(|(q, d)| {
                let other = imag.map(|(e, m)| (e, m.dumps[q].op));
                let (data, imag) = match d.op {
                    DumpOp::Time(h) => {
                        let op = self.engine.operator(&h);
                        let imag = match other {
                            Some((e, DumpOp::Time(hi))) => {
                                Some(e.operator(&hi).sink().snapshots().to_vec())
                            }
                            _ => None,
                        };
                        let data = DumpData::Time {
                            sampler: op.sampler().clone(),
                            snapshots: op.sink().snapshots().to_vec(),
                        };
                        (data, imag)
                    }
                    DumpOp::Freq(h) => {
                        let mut f = self.engine.operator(&h).clone();
                        if let Some((e, DumpOp::Freq(hi))) = other {
                            f.add_imaginary(e.operator(&hi));
                        }
                        (DumpData::Freq(f), None)
                    }
                };
                DumpResult {
                    name: d.name.clone(),
                    id: d.id,
                    file_type: d.file_type,
                    dump_type: d.dump_type,
                    data,
                    imag,
                    delta_unit: self.delta_unit(),
                    cylindrical: self.engine.grid().is_cylindrical(),
                    dt,
                }
            })
            .collect();
        let sar = self
            .mapped
            .sar
            .iter()
            .map(|e| {
                let spec = &self.sources.sar[e.spec];
                let op = self.engine.operator(&e.op);
                let volumes = (0..spec.frequencies.len())
                    .map(|k| {
                        SarVolume::from_edge_dft(
                            self.engine.grid(),
                            op,
                            k,
                            &spec.sigma,
                            spec.density.clone(),
                        )
                        .expect("SAR specs are validated at compile time")
                    })
                    .collect();
                SarResult {
                    name: spec.name.clone(),
                    id: spec.id,
                    kind: spec.kind,
                    file_type: spec.file_type,
                    frequencies: spec.frequencies.clone(),
                    volumes,
                    openems: e.oe.zip(spec.openems.as_ref()).map(|(h, c)| {
                        crate::io::openems::dump::OpenEmsSar::new(
                            self.engine.operator(&h).clone(),
                            c.sigma.clone(),
                            c.density.clone(),
                            self.engine.grid(),
                            &c.indices,
                        )
                    }),
                    delta_unit: self.delta_unit(),
                }
            })
            .collect();
        let incident = self
            .mapped
            .periodic_waves
            .iter()
            .map(|(name, h)| {
                let op = self.engine.operator(h);
                IncidentWave {
                    name: name.clone(),
                    dt,
                    k_t: op.spec().k_t,
                    e: op.incident().to_vec(),
                }
            })
            .collect();
        Results {
            outcome: None,
            dt,
            openems_intervals: (
                self.fdtd.openems_probe_interval(dt),
                self.fdtd.openems_nyquist_interval(dt),
            ),
            probes,
            lumped,
            dumps,
            sar,
            incident,
        }
    }
}

/// Sample times and values of a mapped probe on `engine`.
fn probe_series(engine: &Engine, p: &map::ProbeEntry) -> (Vec<f64>, results::ProbeValues) {
    match p.op {
        ProbeOp::Scalar(h) => {
            let op = engine.operator(&h);
            (
                op.times().to_vec(),
                results::ProbeValues::Scalar(op.values().to_vec()),
            )
        }
        ProbeOp::Native(h) => {
            // Weights (1/length × probe weight) are in the entries.
            let ops = h.map(|h| engine.operator(&h));
            let v = (0..ops[0].values().len())
                .map(|i| ops.map(|o| o.values()[i]))
                .collect();
            (ops[0].times().to_vec(), results::ProbeValues::Vector(v))
        }
    }
}

/// The mesh-check issues of at least warning severity as warning lines,
/// without the csx discretisation warnings (already in the compile
/// warnings).
fn mesh_warnings(
    model: &csx::ContinuousStructure,
    fdtd: &Fdtd,
    compiled: &Compiled,
    opts: &CompileOptions,
) -> Result<Vec<String>> {
    use crate::mesh::{CheckOptions, Severity, check_compiled};
    let check = CheckOptions {
        snap_tolerance: opts.snap_tolerance,
        update_model: opts.update_model,
        ..CheckOptions::default()
    };
    let report = check_compiled(model, fdtd, compiled, &check)?;
    Ok(report
        .issues
        .iter()
        .filter(|i| i.severity >= Severity::Warning)
        .filter(|i| {
            !i.message
                .starts_with(crate::mesh::CSX_DISCRETISATION_PREFIX)
        })
        .map(|i| format!("mesh {i}"))
        .collect())
}

fn find<T>(list: &[(String, OperatorHandle<T>)], name: &str) -> Option<OperatorHandle<T>> {
    list.iter().find(|(n, _)| n == name).map(|(_, h)| *h)
}
