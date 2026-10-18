//! High-level simulation: a csx model plus [`Fdtd`] settings, compiled
//! into a ready-to-run [`Engine`] with the model's sources, lumped
//! elements, probes and dumps mapped onto engine operators (design §4,
//! §5.5).

mod map;
mod results;
#[cfg(feature = "xml")]
mod xml;

pub use results::{DumpData, DumpResult, ProbeKind, ProbeResult, ProbeValues, Results};

use crate::Result;
use crate::compile::{BoundaryHandles, CompileOptions, Compiled, ModelSources, NodeMap, compile};
use crate::engine::{
    EdgeSource, Engine, FreqDump, Grid, LumpedElement, OperatorHandle, Progress, RunLimits,
    TimeDump,
};
use crate::fdtd::Fdtd;
use map::{DumpOp, Mapped, ProbeOp};

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
///    ([`Results::lumped_current`]).
/// 2. **Excitations** (csx `Excitation`, only enabled ones), driven by the
///    settings' [`Waveform`](crate::Waveform) delayed by the property's
///    `Delay`: `a(t) = waveform.value(t − delay)`. csx gives the field
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
///      model, so XML ports behave like native ones.
///    * `EFieldHard` overwrites the edge voltage (after the lumped
///      updates, so on a lumped element's edge the element has no effect:
///      warning), `HFieldSoft` / `HFieldHard` add to / overwrite the H
///      dual-edge currents.
///    * Plane waves are not supported yet (warning).
///
///    Without a waveform in the settings, excitations are ignored with a
///    warning.
/// 3. **Probes** (csx `ProbeBox`, corners snapped to mesh nodes by the
///    compile step, see [`crate::compile`]), all multiplied by the probe
///    `Weight`:
///    * `Voltage`: `∫E·dl` from `start` to `stop` along the staircase path
///      x, then y, then z (edges traversed against their direction count
///      negative, so swapping start and stop flips the sign). Sampled at
///      the E times `(n + 1)·dt`.
///    * `Current`: the total current through the box cross-section normal
///      to `NormDir` (or to the box's only flat axis), counted positive
///      from `start` towards `stop` along the normal (along `+normal` for a
///      flat box), measured as `∮H·dl` around the E edges whose transverse
///      node coordinates are inside the box, in the dual plane through the
///      middle edge layer of the box (for a flat box at node `k`, the layer
///      from `k` to `k + 1`). Sampled at the H times `(n + ½)·dt`.
///    * `EField` / `HField`: the field vector at the **centre** of the
///      (snapped) box, at the E / H times: for a point probe
///      (`start == stop`) the node itself, interpolated like the dumps (see
///      [`crate::engine::dump`]); for a box, the node values trilinearly
///      interpolated at its centre ([`FieldProbe::at`](crate::engine::FieldProbe::at)).
///      The openEMS documentation describes field probes at a point; using
///      the centre of a box is this crate's choice (no averaging over the
///      box).
///    * Waveguide probes: not supported yet (warning).
///
///    `FD_Samples` frequencies are available through
///    [`ProbeResult::spectrum`]; any other frequency through
///    [`ProbeResult::dft`].
/// 4. **Dumps** (csx `DumpBox`): `EField`/`HField` → in-memory
///    [`TimeDump`], `EFieldFD`/`HFieldFD` → [`FreqDump`] at the property's
///    frequencies; `DumpMode` `Cell` → cell centres, otherwise nodes;
///    `SubSampling` → stride. Both sample every `K` steps, with `K` the
///    Nyquist interval of [`Fdtd::get_max_frequency`] divided by the
///    oversampling factor ([`Fdtd::get_oversampling`], default 4); `K = 1`
///    without a maximum frequency. The probe settings' `StartTime` /
///    `StopTime` restrict the recorded states (by the quantity's own time;
///    `StartTime ≤ 0`: from the start, `StopTime ≤ 0`: no limit; see
///    [`TimeDump::window`]). Other dump types are not supported
///    (warning). Write VTR files with [`DumpResult::write_vtr`].
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
        let (mut engine, boundaries) = compiled.build()?;
        let mut warnings = compiled.warnings.clone();
        if opts.mesh_check {
            warnings.extend(mesh_warnings(model, fdtd, &compiled, opts)?);
        }
        let mut mapped = map::map_sources(
            &mut engine,
            &compiled.sources,
            &compiled.node_map,
            fdtd,
            &mut warnings,
        )?;
        mapped.lumped_ids = mapped
            .lumped
            .iter()
            .map(|(n, _)| model.property_id(n).map_or(u32::MAX, |p| p.0))
            .collect();
        Ok(Self {
            engine,
            fdtd: fdtd.clone(),
            delta_unit: compiled.delta_unit,
            node_map: compiled.node_map,
            boundaries,
            sources: compiled.sources,
            mapped,
            warnings,
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

    /// The run limits derived from the settings: at most
    /// [`Fdtd::timestep_limit`] steps for the engine time step (i.e.
    /// [`Fdtd::get_max_timesteps`], reduced by [`Fdtd::get_max_time`] if
    /// set; counted from the engine's current step, as
    /// [`RunLimits::max_timesteps`]), the settings' end criteria, and
    /// the waveform's [`duration`](crate::Waveform::duration) as the
    /// excitation end (`∞` for a waveform that never ends, none without a
    /// waveform).
    pub fn run_limits(&self) -> RunLimits {
        let end = self
            .fdtd
            .get_waveform()
            .map(|w| w.duration().unwrap_or(f64::INFINITY));
        RunLimits::new(self.fdtd.timestep_limit(self.engine.dt()))
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
        let limits = self.run_limits();
        let outcome = self.engine.run_until(&limits)?;
        let mut r = self.results();
        r.outcome = Some(outcome);
        Ok(r)
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
        let outcome = self.engine.run_until_with(&limits, progress)?;
        let mut r = self.results();
        r.outcome = Some(outcome);
        Ok(r)
    }

    /// A copy of everything recorded so far by the model's probes and
    /// dumps (no run outcome).
    pub fn results(&self) -> Results {
        let dt = self.engine.dt();
        let probes = self
            .mapped
            .probes
            .iter()
            .map(|p| {
                let (times, values) = match p.op {
                    ProbeOp::Scalar(h) => {
                        let op = self.engine.operator(&h);
                        (
                            op.times(),
                            results::ProbeValues::Scalar(op.values().to_vec()),
                        )
                    }
                    ProbeOp::Field(h) => {
                        let op = self.engine.operator(&h);
                        let w = p.weight;
                        let v = op.values().iter().map(|v| v.map(|x| x * w)).collect();
                        (op.times(), results::ProbeValues::Vector(v))
                    }
                };
                // Recording window (stop ≤ 0: no limit), with a small
                // tolerance against rounding of the sample times.
                let eps = 1e-9 * dt;
                let lo = times.partition_point(|&t| t < p.start_time - eps);
                let hi = if p.stop_time > 0.0 {
                    times.partition_point(|&t| t <= p.stop_time + eps)
                } else {
                    times.len()
                }
                .max(lo);
                let values = match values {
                    results::ProbeValues::Scalar(v) => {
                        results::ProbeValues::Scalar(v[lo..hi].to_vec())
                    }
                    results::ProbeValues::Vector(v) => {
                        results::ProbeValues::Vector(v[lo..hi].to_vec())
                    }
                };
                ProbeResult {
                    name: p.name.clone(),
                    id: p.id,
                    number: p.number,
                    kind: p.kind,
                    frequencies: p.frequencies.clone(),
                    dt,
                    times: times[lo..hi].to_vec(),
                    values,
                }
            })
            .collect();
        let lumped = self
            .mapped
            .lumped
            .iter()
            .zip(&self.mapped.lumped_ids)
            .map(|((name, h), &id)| {
                let op = self.engine.operator(h);
                ProbeResult {
                    name: name.clone(),
                    id,
                    number: 0,
                    kind: ProbeKind::Current,
                    frequencies: Vec::new(),
                    dt,
                    times: op.current_times(),
                    values: results::ProbeValues::Scalar(op.current().to_vec()),
                }
            })
            .collect();
        let dumps = self
            .mapped
            .dumps
            .iter()
            .map(|d| DumpResult {
                name: d.name.clone(),
                id: d.id,
                data: match d.op {
                    DumpOp::Time(h) => {
                        let op = self.engine.operator(&h);
                        DumpData::Time {
                            sampler: op.sampler().clone(),
                            snapshots: op.sink().snapshots().to_vec(),
                        }
                    }
                    DumpOp::Freq(h) => DumpData::Freq(self.engine.operator(&h).clone()),
                },
            })
            .collect();
        Results {
            outcome: None,
            dt,
            probes,
            lumped,
            dumps,
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
