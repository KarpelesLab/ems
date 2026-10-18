//! High-level simulation: a csx model plus [`Fdtd`] settings, compiled
//! into a ready-to-run [`Engine`] (design §4).

use crate::Result;
use crate::compile::{BoundaryHandles, CompileOptions, ModelSources, NodeMap, compile};
use crate::engine::{Engine, Grid, RunLimits, RunOutcome};
use crate::fdtd::Fdtd;

/// A compiled simulation: the [`Engine`] plus the metadata needed to relate
/// engine results back to the model.
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
/// let out = sim.run()?;
/// assert_eq!(out.steps, 10);
/// # Ok::<(), ems::Error>(())
/// ```
#[derive(Debug)]
pub struct Simulation {
    engine: Engine,
    fdtd: Fdtd,
    delta_unit: f64,
    node_map: NodeMap,
    boundaries: BoundaryHandles,
    sources: ModelSources,
    warnings: Vec<String>,
}

impl Simulation {
    /// Compiles `model` with `fdtd` using the default [`CompileOptions`].
    ///
    /// # Errors
    /// See [`compile`](crate::compile::compile) and
    /// [`Compiled::build`](crate::compile::Compiled::build).
    pub fn new(model: &csx::ContinuousStructure, fdtd: &Fdtd) -> Result<Self> {
        Self::with_options(model, fdtd, &CompileOptions::default())
    }

    /// Compiles `model` with `fdtd` and explicit compile options.
    ///
    /// # Errors
    /// See [`new`](Self::new).
    pub fn with_options(
        model: &csx::ContinuousStructure,
        fdtd: &Fdtd,
        opts: &CompileOptions,
    ) -> Result<Self> {
        let compiled = compile(model, fdtd, opts)?;
        let (engine, boundaries) = compiled.build()?;
        // ---------------------------------------------------------------
        // FOLLOW-UP (design §10 step 6): map `compiled.sources` onto engine
        // operators here, after the boundary operators:
        //   * `sources.excitations` (E types: field amplitude per edge, so
        //     u-source weight = amplitude × edge length in metres; the time
        //     function is `fdtd.get_waveform()` delayed by `delay`) →
        //     `EdgeSource`s;
        //   * `sources.lumped` (per-edge R/L/C already distributed) →
        //     `LumpedElement`s (and Thevenin sources where an excitation
        //     overlaps), possibly via an edge-list constructor;
        //   * ProbeBox / DumpBox properties → `Probe` / dump operators.
        // Edges are in csx indexing: use `compiled.node_map.edge(e)`.
        // Remove the corresponding "not mapped yet" warnings in
        // `compile::property_warnings` at the same time.
        // ---------------------------------------------------------------
        Ok(Self {
            engine,
            fdtd: fdtd.clone(),
            delta_unit: compiled.delta_unit,
            node_map: compiled.node_map,
            boundaries,
            sources: compiled.sources,
            warnings: compiled.warnings,
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

    /// Discretised model parts not yet mapped to operators.
    pub fn sources(&self) -> &ModelSources {
        &self.sources
    }

    /// Compile warnings (csx model checks, discretisation, approximations).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The run limits derived from the settings: at most
    /// [`Fdtd::get_max_timesteps`] steps, the settings' end criteria, and
    /// the waveform's [`duration`](crate::Waveform::duration) as the
    /// excitation end (`∞` for a waveform that never ends, none without a
    /// waveform).
    pub fn run_limits(&self) -> RunLimits {
        let end = self
            .fdtd
            .get_waveform()
            .map(|w| w.duration().unwrap_or(f64::INFINITY));
        RunLimits::new(self.fdtd.get_max_timesteps())
            .end_criteria(self.fdtd.get_end_criteria().copied())
            .excitation_end(end)
    }

    /// Runs with [`run_limits`](Self::run_limits) via
    /// [`Engine::run_until`].
    ///
    /// # Errors
    /// See [`Engine::run_until`].
    pub fn run(&mut self) -> Result<RunOutcome> {
        let limits = self.run_limits();
        self.engine.run_until(&limits)
    }
}
