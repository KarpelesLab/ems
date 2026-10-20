//! Compile step: [`csx::ContinuousStructure`] + [`Fdtd`] settings → engine
//! inputs (design §6, §10 step 5).
//!
//! [`compile`] does, in this order:
//!
//! 1. **Model update.** If [`CompileOptions::update_model`] is set (the
//!    default), parametric values are re-evaluated on a *clone* of the model
//!    ([`csx::ContinuousStructure::update`]); the caller's model is never
//!    mutated.
//! 2. **Grid.** The csx mesh lines (drawing units) are multiplied by
//!    [`csx::RectilinearGrid::delta_unit`] to give the engine [`Grid`] in
//!    metres. A cylindrical csx grid becomes a [`Grid::cylindrical`] (`ρ`,
//!    `z` scaled, `α` in radians); the cylindrical rules (features that
//!    still need a Cartesian grid, ignored boundaries, copies of edges) are
//!    in design §5.15 and the `compile::cylindrical` module.
//! 3. **Materials.** [`csx::discretize::discretize`] samples the model onto
//!    the grid (sub-cell averaging and priorities are csx's business, see
//!    its module docs). Its arrays use the csx node order
//!    `i + nx·(j + ny·k)` (x fastest); they are reordered into the engine
//!    layout `(i·ny + j)·nz + k` (z fastest) by [`NodeMap`] and stored as
//!    [`EdgeMaterials`]. The value conventions agree one to one: csx's
//!    per-E-edge `epsilon`/`kappa` are the dual-face (quarter-cell, volume
//!    weighted) averages and its per-H-face `mue` is the harmonic mean over
//!    the two half cells, which is exactly what [`EdgeMaterials`] expects.
//!    The only difference to the per-cell path ([`CellMaterials`](crate::engine::CellMaterials))
//!    is the magnetic loss: csx averages `sigma` arithmetically, the
//!    per-cell path combines the half cells as series `R`–`L` branches
//!    (`R = L²·Σ R_c/L_c²`). Both agree on homogeneous dual edges.
//! 4. **Metal.** The csx PEC mask becomes the engine PEC mask (with
//!    [`CompileOptions::conformal`], solid metal is instead resolved below
//!    the cell size, see [Conformal PEC](#conformal-pec)).
//!    Conducting sheets become [`Compiled::sheets`], installed by
//!    [`Compiled::build`] as one [`ConductingSheet`] operator (thin lossy
//!    conductor with skin effect, see [`crate::engine::sheet`]); the
//!    mapping rules (sheet normal, face shares, fallbacks) are in
//!    [`sheet`]. Metal `Wire` primitives thinner than the staircase get
//!    the thin-wire correction of [`crate::engine::wire`] on their edge
//!    materials ([`Compiled::thin_wires`]; rules in `compile/wire.rs`).
//!    Metal `Wire`s with an oblique segment are taken out of the model
//!    before step 3 and become the wire network of
//!    [`crate::engine::oblique_wire`] ([`Compiled::oblique_wires`]).
//! 5. **Boundaries** from [`Fdtd::get_boundaries`]: PEC and PMC map to
//!    [`FaceCondition`]s; Mur and PML faces are built as PEC faces and
//!    listed in [`Compiled::mur`] / [`Compiled::pml`], from which
//!    [`Compiled::build`] creates **one** [`MurBoundary`] covering all Mur
//!    faces and **one** [`CpmlBoundary`] covering all PML faces.
//! 6. **Absorbing sheets.** csx `AbsorbingBC` primitives become
//!    [`AbcSpec`]s ([`Compiled::abc`]), installed by [`Compiled::build`] as
//!    one [`AbsorbingSheet`] each (see [Absorbing sheets](#absorbing-sheets)).
//! 7. **Plane waves.** csx `PlaneWave` excitations (type 10, which csx
//!    does not discretise) become [`PlaneWaveSource`]s in
//!    [`ModelSources::plane_waves`] (total-field / scattered-field boxes;
//!    rules in [`planewave`]).
//! 8. **Time step and threads** from [`Fdtd`]: a time step override becomes
//!    [`TimeStep::Fixed`], otherwise [`TimeStep::Courant`] with the time
//!    step factor; the thread count is used for both the discretisation and
//!    the engine.
//!
//! # Conformal PEC
//!
//! With [`CompileOptions::conformal`], the open (non-metal) fraction of
//! every E edge and cell face cut by a *solid* metal primitive is computed
//! ([`Compiled::conformal`]; geometry rules in `src/compile/conformal.rs`):
//! edges are bisected to the metal surface, faces get the polygon of their
//! open part. Cut edges are removed from the PEC mask, fully covered ones
//! added. [`Compiled::build`] then scales the coefficients
//! (Dey–Mittra in integral form, with a stability floor and check; see
//! [`crate::engine::conformal`]). Zero-thickness metal sheets off the
//! mesh planes (oblique plates, cylindrical and spherical shells) are
//! split into pieces and regions on both sides (thin-sheet scheme of the
//! same module); sheets on mesh planes, curves and wires keep the
//! staircase mask. A cut edge's voltage `u` is then the voltage over its
//! open part, `E·l_open`.
//!
//! # Dispersive materials
//!
//! csx `LorentzMaterial` (Drude/Lorentz) and `DebyeMaterial` properties
//! contribute their static part (`Epsilon`, `Mue`, … = `ε∞`, `μ∞`)
//! through `discretize` like any material, and their poles through
//! [`Compiled::dispersive`]: per E edge (and H dual edge for magnetic
//! Lorentz poles), the full poles of the dispersive property at the edge
//! midpoint (openEMS semantics). [`Compiled::build`] installs them as one
//! [`Dispersive`] operator, with openEMS's explicit leapfrog scheme by
//! default ([`CompileOptions::dispersive_scheme`]). Parameter conventions
//! and the sampling rules (both confirmed against openEMS output) are
//! documented in [`dispersive`]; in short: `ω_p = 2π·f_plasma·√ε_e` with
//! the edge's static `ε_e`, `ω_0 = 2π·f_Lor`, `γ = 1/τ` (0 for `τ = 0`),
//! Debye `Δε`, `τ` as given.
//!
//! # Absorbing sheets
//!
//! Every primitive of an `AbsorbingBC` property is snapped to the mesh like
//! a probe box (see below) and must then be flat along exactly one axis,
//! the sheet normal; anything else (a volume, a line, a point) is ignored
//! with a warning, as is a box that was not flat before snapping (it is
//! still used, flattened by the snap, with a warning). `NormalSignPositive`
//! is the sign of the normal pointing *into* the region the sheet
//! terminates, i.e. against the waves it absorbs: `1` absorbs waves
//! travelling along `−a`, `0` along `+a` ([`AbcSheet::positive`] is the
//! opposite sign; as openEMS, whose probes ems then reproduces to 1e-6,
//! fixture `sweep_abc_line`). `PhaseVelocity` > 0 is used as given;
//! 0 (csx's "speed of light") selects the local medium velocity
//! `c0/√(ε_r·μ_r)` per edge ([`SheetVelocity::Local`]), which is `c0` in
//! vacuum; negative or non-finite values fall back to it with a warning.
//! `AbsorbingBoundaryType` `MUR_1ST` is implemented; `MUR_1ST_SA` (the
//! super-absorption variant) and unknown types use plain first-order Mur,
//! with a warning. Behaviour and limitations of the sheet (one-way, hard
//! wall for waves from behind) are in [`abc`](crate::engine::boundary::abc).
//!
//! Excitations and lumped elements of the model are discretised by csx as
//! well, and ProbeBox / DumpBox primitives are snapped to the mesh
//! ([`ProbeSpec`]); all of these are kept in [`ModelSources`].
//! [`Compiled::build`] creates only the bare engine with its boundaries;
//! [`crate::Simulation`] maps the sources onto engine operators.
//!
//! **Snapping.** Each coordinate of a probe or dump box corner (drawing
//! units, after the primitive's transformation) is moved to the nearest
//! mesh line on its axis ([`csx::RectilinearGrid::nearest_line`]); a corner
//! that is not on a line (within csx's [`snap`](csx::RectilinearGrid::snap)
//! tolerance) or outside the mesh adds a warning. Box primitives keep their
//! corner order (`start`, `stop`), which carries the probe direction; for
//! other shapes the bounding box is used (with a warning) and point
//! primitives give `start = stop`. Only the first primitive of a probe or
//! dump property is used.
//!
//! # Waveguide probes
//!
//! `WaveguideVoltage` / `WaveguideCurrent` probes integrate the tangential
//! E / H field over the probe plane, weighted by the mode function of the
//! property attributes `ModeFunctionX/Y/Z`. Their weights are computed
//! here ([`ProbeSpec::mode_entries`]); the conventions and the
//! discretisation are documented in `src/compile/mode.rs`: by default
//! openEMS's sampling (drawing-unit coordinates, the mode function
//! normalised by the probe itself, `V = w·Σ E_t·f·A/√(Σ f²·A)` over the
//! plane's nodes, the current over its cells with `f` at the cell centre),
//! or with the attribute `emsModeSampling="exact"` the plain overlap
//! integrals `V = w·∫E_t·f dA`, `I = w·∫H_t·f dA`. The mode purity
//! openEMS writes next to the value (`w·(Σ E_t·f·A)²/(Σ|E_t|²A·Σ|f|²A)`,
//! with openEMS's samples) comes from [`ProbeSpec::mode_samples`] (not
//! on cylindrical grids).
//!
//! # SAR dumps
//!
//! `SARLocal` / `SAR1g` / `SAR10g` / `SARRaw` dumps with frequencies are
//! collected in [`ModelSources::sar`], with the cells of their box and the
//! per-cell conductivity and mass density sampled from the model (see
//! [`sar`]).

mod cellconst;
mod conformal;
pub(crate) mod cylindrical;
pub mod dispersive;
mod mode;
mod openems;
pub mod planewave;
pub mod sar;
pub mod sheet;
mod wire;

pub(crate) use mode::SAMPLING_ATTR as MODE_SAMPLING_ATTR;
pub use planewave::PlaneWaveSource;

use crate::axis::Axis;
use crate::engine::boundary::abc::{AbcSheet, AbsorbingSheet, SheetVelocity};
use crate::engine::boundary::cpml::CpmlBoundary;
use crate::engine::boundary::mur::MurBoundary;
use crate::engine::dispersive::Scheme as DispersiveScheme;
use crate::engine::{
    ConductingSheet, ConformalOptions, ConformalPec, ConformalReport, Dispersive, EdgeMaterials,
    EdgePole, Engine, FaceCondition, FaceConditions, Grid, ObliqueWires, OperatorHandle, SheetEdge,
    ThinWire, TimeStep,
};
use crate::fdtd::{Boundary, Cpml, Face, Fdtd};
use crate::{Error, Result};

use csx::discretize::{
    DiscretizeOptions, ExcitationField, LumpedEdge, MaterialGrid, SheetInfo, discretize,
};
use csx::properties::{AbcType, DumpType, ExcitationType, ProbeType};
use csx::{CoordSystem, PropertyKind};

/// Options of [`compile`].
#[derive(Debug, Clone, PartialEq)]
pub struct CompileOptions {
    /// Sub-samples per axis in each quarter / half cell used by csx to
    /// average materials (`N`, `N³` samples; default 1).
    pub subsampling: usize,
    /// Relative tolerance for snapping sheets and lines to mesh lines
    /// (default 1e-6 of the local cell size).
    pub snap_tolerance: f64,
    /// Re-evaluate parametric values on a clone of the model before
    /// discretising (default `true`). Set to `false` if the model is known
    /// to be up to date, to avoid the clone.
    pub update_model: bool,
    /// Run the solver-aware mesh checks ([`crate::mesh::check_compiled`])
    /// when building a [`Simulation`](crate::Simulation) and append their
    /// warnings and errors to its warnings (default `true`). Not used by
    /// [`compile`] itself.
    pub mesh_check: bool,
    /// Conformal (sub-cell) PEC for solid metal (default `false`: the
    /// staircase PEC mask of csx). See [Conformal PEC](self#conformal-pec).
    pub conformal: bool,
    /// Options of the conformal scheme (stability floor and check), used
    /// when [`conformal`](Self::conformal) is set.
    pub conformal_options: ConformalOptions,
    /// Time integration of Lorentz/Drude poles (default
    /// [`Scheme::Leapfrog`](crate::engine::dispersive::Scheme::Leapfrog),
    /// the scheme openEMS uses, so XML models give the same signals; see
    /// [`dispersive`]).
    pub dispersive_scheme: DispersiveScheme,
}

impl CompileOptions {
    /// Returns `self` with [`conformal`](Self::conformal) set to `on`.
    ///
    /// ```
    /// let o = ems::CompileOptions::default().with_conformal(true);
    /// assert!(o.conformal);
    /// ```
    #[must_use]
    pub fn with_conformal(mut self, on: bool) -> Self {
        self.conformal = on;
        self
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        let d = DiscretizeOptions::default();
        Self {
            subsampling: d.subsampling,
            snap_tolerance: d.snap_tolerance,
            update_model: true,
            mesh_check: true,
            conformal: false,
            conformal_options: ConformalOptions::default(),
            dispersive_scheme: DispersiveScheme::Leapfrog,
        }
    }
}

/// Conversion between csx and engine node/edge indexing for one grid.
///
/// csx flattens node `[i, j, k]` as `i + nx·(j + ny·k)` (x fastest); the
/// engine uses `(i·ny + j)·nz + k` (z fastest). Axis indices agree
/// (0 = x, 1 = y, 2 = z), and an edge/face of axis `a` "at node `n`" means
/// the same object in both crates (E edge from `n` along `a`; H face normal
/// to `a` with lowest corner `n`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeMap {
    nodes: [usize; 3],
}

impl NodeMap {
    /// Map for a grid with `nodes` lines per axis.
    pub fn new(nodes: [usize; 3]) -> Self {
        Self { nodes }
    }

    /// Lines per axis.
    pub fn nodes(&self) -> [usize; 3] {
        self.nodes
    }

    /// Node of a csx flat index.
    #[inline]
    pub fn csx_node(&self, idx: usize) -> [usize; 3] {
        let n = self.nodes;
        let i = idx % n[0];
        let r = idx / n[0];
        [i, r % n[1], r / n[1]]
    }

    /// csx flat index of a node.
    #[inline]
    pub fn csx_index(&self, [i, j, k]: [usize; 3]) -> usize {
        let n = self.nodes;
        i + n[0] * (j + n[1] * k)
    }

    /// Engine flat index of a node.
    #[inline]
    pub fn engine_index(&self, [i, j, k]: [usize; 3]) -> usize {
        let n = self.nodes;
        (i * n[1] + j) * n[2] + k
    }

    /// Engine `(axis, node)` of a csx edge / face.
    ///
    /// # Panics
    /// If the direction is not 0, 1 or 2.
    pub fn edge(&self, e: csx::discretize::Edge) -> (Axis, [usize; 3]) {
        let axis = Axis::from_index(e.dir).expect("csx edge direction in 0..3");
        (axis, self.csx_node(e.index))
    }

    /// Reorders a csx node array into engine layout.
    pub fn reorder<T: Copy + Default>(&self, csx: &[T]) -> Vec<T> {
        let n = self.nodes;
        let mut out = vec![T::default(); n[0] * n[1] * n[2]];
        let mut e = 0;
        for i in 0..n[0] {
            for j in 0..n[1] {
                let base = i + n[0] * j;
                let stride = n[0] * n[1];
                for (k, o) in out[e..e + n[2]].iter_mut().enumerate() {
                    *o = csx[base + stride * k];
                }
                e += n[2];
            }
        }
        out
    }
}

/// Model parts that are mapped onto engine operators by
/// [`crate::Simulation`] (not by [`Compiled::build`]). Edges/faces are in
/// csx indexing; convert with [`Compiled::node_map`] / [`NodeMap::edge`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelSources {
    /// Enabled excitations, with amplitudes per E edge (E types, field
    /// value along the edge in V/m) or per H face (H types, A/m).
    pub excitations: Vec<ExcitationField>,
    /// Lumped-element edges with values already distributed per edge.
    pub lumped: Vec<LumpedEdge>,
    /// The conducting sheets of the model (their edges are in
    /// [`Compiled::sheets`]).
    pub sheets: Vec<SheetInfo>,
    /// ProbeBox properties, snapped to the mesh.
    pub probes: Vec<ProbeSpec<csx::properties::ProbeBox>>,
    /// DumpBox properties, snapped to the mesh.
    pub dumps: Vec<ProbeSpec<csx::properties::DumpBox>>,
    /// Plane-wave excitations (total-field / scattered-field boxes, see
    /// [`planewave`]).
    pub plane_waves: Vec<PlaneWaveSource>,
    /// The SAR dumps among [`dumps`](Self::dumps), with their cells and
    /// per-cell materials ([`sar`]).
    pub sar: Vec<sar::SarSpec>,
    /// The drawing unit of the model \[m\] (lengths of property settings,
    /// e.g. a dump's `OptResolution`, are in drawing units).
    pub delta_unit: f64,
}

/// A ProbeBox or DumpBox property (`T`) with its box snapped to mesh
/// nodes (see the [module docs](self) on snapping).
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeSpec<T> {
    /// Property name.
    pub name: String,
    /// csx property id ([`csx::PropertyId`] value).
    pub id: u32,
    /// The property settings (parameters evaluated).
    pub settings: T,
    /// Node indices of the first box corner.
    pub start: [usize; 3],
    /// Node indices of the opposite corner (may be below `start`).
    pub stop: [usize; 3],
    /// The unsnapped box corners `[first, opposite]` in the engine grid's
    /// units: metres (drawing units times the drawing unit; radians along
    /// `α` on a cylindrical grid, see [`grid_from_csx`]), after the
    /// primitive's transformation; the bounding box for non-box shapes.
    /// Dumps snap them with the openEMS rules of
    /// [`io::openems::dump`](crate::io::openems::dump), probes with
    /// openEMS's probe rules (dual-grid positions of H and current probes,
    /// see [`Simulation`](crate::Simulation)).
    pub corners: [[f64; 3]; 2],
    /// Waveguide probes (`WaveguideVoltage` / `WaveguideCurrent`): the
    /// mode-weighted field entries `(axis, engine node, weight)` (see
    /// [Waveguide probes](self#waveguide-probes)); empty otherwise.
    pub mode_entries: Vec<(Axis, [usize; 3], f64)>,
    /// Waveguide probes on Cartesian grids: the sample points of the mode
    /// purity ([`Probe::mode_purity`](crate::engine::Probe::mode_purity),
    /// openEMS's sampling); `None` otherwise.
    pub mode_samples: Option<Vec<crate::engine::ModeSample>>,
    /// Waveguide probes on Cartesian grids: the first and last node of
    /// openEMS's samples (the box without the outermost mesh lines; for
    /// the current, the lower corners of the box's cells), the position
    /// openEMS reports in its probe files; `None` otherwise.
    pub mode_range: Option<([usize; 3], [usize; 3])>,
}

/// An absorbing sheet from a csx `AbsorbingBC` primitive (see
/// [Absorbing sheets](self#absorbing-sheets)).
#[derive(Debug, Clone, PartialEq)]
pub struct AbcSpec {
    /// Property name.
    pub name: String,
    /// The sheet, in engine node indices.
    pub sheet: AbcSheet,
    /// Phase velocity \[m/s\]; `None`: local medium velocity.
    pub phase_velocity: Option<f64>,
}

/// Engine inputs produced by [`compile`]; [`build`](Self::build) turns
/// them into an [`Engine`].
#[derive(Debug, Clone)]
pub struct Compiled {
    /// Mesh in metres.
    pub grid: Grid,
    /// The csx drawing unit in metres (the grid lines were multiplied by
    /// it).
    pub delta_unit: f64,
    /// Effective per-edge materials and PEC mask, engine layout.
    pub materials: EdgeMaterials,
    /// Outer face conditions (Mur and PML faces are PEC).
    pub faces: FaceConditions,
    /// openEMS PMC faces ([`Fdtd::openems_run`]): the magnetic wall lies
    /// on the outermost dual plane, half a cell inside the domain (the
    /// tangential H there is held at zero), instead of on the face itself.
    /// Established by black-box comparison with openEMS v0.37 (agreement
    /// ~1e-6, against ~3–30 % for the on-face wall). The same switch
    /// makes Mur faces count the energy of their edges in the end
    /// criterion, as openEMS does
    /// ([`MurBoundary::count_energy`](crate::engine::boundary::mur::MurBoundary::count_energy)).
    pub openems_pmc: bool,
    /// Mur faces and their phase velocities \[m/s\].
    pub mur: Vec<(Face, Option<f64>)>,
    /// PML faces and their parameters.
    pub pml: Vec<(Face, Cpml)>,
    /// Time step choice.
    pub timestep: TimeStep,
    /// Engine threads (`0`: available parallelism).
    pub threads: usize,
    /// csx ↔ engine index conversion.
    pub node_map: NodeMap,
    /// Absorbing sheets from csx `AbsorbingBC` properties (see
    /// [Absorbing sheets](self#absorbing-sheets)); installed by
    /// [`build`](Self::build), one [`AbsorbingSheet`] each.
    pub abc: Vec<AbcSpec>,
    /// Dispersive poles per edge from csx `LorentzMaterial` /
    /// `DebyeMaterial` properties (see [`dispersive`](self#dispersive-materials));
    /// installed by [`build`](Self::build) as one [`Dispersive`] operator.
    pub dispersive: Vec<EdgePole>,
    /// Time integration of the Lorentz/Drude poles of
    /// [`dispersive`](Self::dispersive) ([`CompileOptions::dispersive_scheme`]).
    pub dispersive_scheme: DispersiveScheme,
    /// Thin wires (csx `Wire` primitives thinner than the staircase; see
    /// [`crate::engine::wire`]), already applied to
    /// [`materials`](Self::materials).
    pub thin_wires: Vec<ThinWire>,
    /// Oblique thin wires (csx `Wire` primitives with a segment that is
    /// not along a mesh axis; see [`crate::engine::oblique_wire`] and
    /// `compile/wire.rs`), removed from the model before discretisation;
    /// installed by [`build`](Self::build).
    pub oblique_wires: Option<ObliqueWires>,
    /// Conducting-sheet edges from csx `ConductingSheet` properties (see
    /// [`sheet`]); installed by [`build`](Self::build) as one
    /// [`ConductingSheet`] operator.
    pub sheets: Vec<SheetEdge>,
    /// Highest frequency of interest \[Hz\] ([`Fdtd::get_max_frequency`]),
    /// used to grade the conducting-sheet layers.
    pub max_frequency: Option<f64>,
    /// Conformal PEC geometry ([`CompileOptions::conformal`]); applied to
    /// the coefficients by [`build`](Self::build) before any operator.
    pub conformal: Option<ConformalPec>,
    /// Not yet mapped model parts.
    pub sources: ModelSources,
    /// Diagnostics: csx model and discretisation warnings plus the
    /// approximations made here.
    pub warnings: Vec<String>,
}

/// The boundary operators created by [`Compiled::build`].
#[derive(Debug, Clone, Default)]
pub struct BoundaryHandles {
    /// The Mur operator covering every Mur face, if any.
    pub mur: Option<OperatorHandle<MurBoundary>>,
    /// The dispersive-material operator, if the model has dispersive
    /// poles.
    pub dispersive: Option<OperatorHandle<Dispersive>>,
    /// The CPML operator covering every PML face, if any.
    pub cpml: Option<OperatorHandle<CpmlBoundary>>,
    /// The conducting-sheet operator, if the model has conducting sheets.
    pub sheets: Option<OperatorHandle<ConductingSheet>>,
    /// One operator per [`Compiled::abc`] sheet, in the same order.
    pub abc: Vec<OperatorHandle<AbsorbingSheet>>,
    /// The oblique thin-wire operator, if [`Compiled::oblique_wires`] is
    /// set.
    pub oblique_wires: Option<OperatorHandle<ObliqueWires>>,
    /// What the conformal PEC step did, if [`Compiled::conformal`] is set.
    pub conformal: Option<ConformalReport>,
}

impl Compiled {
    /// Builds the engine (coefficients, threads), applies the conformal PEC
    /// geometry if any ([`ConformalPec::apply`]), and registers the boundary
    /// operators, first the [`CpmlBoundary`], then the [`MurBoundary`],
    /// then the [`Dispersive`] medium (after the CPML, so that a medium
    /// reaching into the PML is coupled implicitly with it), then the
    /// [`ConductingSheet`]s, then the [`ObliqueWires`], then the
    /// [`AbsorbingSheet`]s (last, so that they see the final interior E).
    ///
    /// # Errors
    /// Invalid time step or materials, or boundary operator setup errors
    /// (e.g. a PML thicker than the mesh).
    pub fn build(&self) -> Result<(Engine, BoundaryHandles)> {
        let mut engine = self.builder().build()?;
        let h = self.add_boundary_operators(&mut engine)?;
        Ok((engine, h))
    }

    /// The engine builder of [`build`](Self::build), before any operator.
    pub(crate) fn builder(&self) -> crate::engine::EngineBuilder {
        Engine::builder(self.grid.clone())
            .edge_materials(self.materials.clone())
            .faces(self.faces)
            .timestep(self.timestep)
            .threads(self.threads)
    }

    /// Registers the boundary operators of [`build`](Self::build) on
    /// `engine`.
    pub(crate) fn add_boundary_operators(&self, engine: &mut Engine) -> Result<BoundaryHandles> {
        let mut h = BoundaryHandles::default();
        if self.openems_pmc {
            pmc_half_cell(engine, &self.faces);
        }
        if let Some(c) = &self.conformal {
            h.conformal = Some(c.apply(engine)?);
        }
        if !self.pml.is_empty() {
            let op = CpmlBoundary::new(engine, self.pml.iter().copied())?;
            h.cpml = Some(engine.add_operator(op));
        }
        if !self.mur.is_empty() {
            let op = MurBoundary::with_faces_in(engine, &self.mur, &self.materials)?
                .count_energy(self.openems_pmc);
            h.mur = Some(engine.add_operator(op));
        }
        if !self.dispersive.is_empty() {
            let mut d = Dispersive::new().with_scheme(self.dispersive_scheme);
            for &e in &self.dispersive {
                d.push_entry(e);
            }
            h.dispersive = Some(d.add_to(engine)?);
        }
        if !self.sheets.is_empty() {
            let mut s = ConductingSheet::new(self.max_frequency);
            for &e in &self.sheets {
                s.push(e);
            }
            h.sheets = Some(s.add_to(engine)?);
        }
        if let Some(w) = &self.oblique_wires {
            h.oblique_wires = Some(engine.add_operator(w.clone()));
        }
        for a in &self.abc {
            let v = match a.phase_velocity {
                Some(v) => SheetVelocity::Fixed(v),
                None => SheetVelocity::Local(&self.materials),
            };
            let op = AbsorbingSheet::new(engine, &a.sheet, v)
                .map_err(|e| Error::Invalid(format!("AbsorbingBC {:?}: {e}", a.name)))?;
            h.abc.push(engine.add_operator(op));
        }
        Ok(h)
    }
}

/// Converts the csx grid (drawing units) into an engine [`Grid`] in metres.
///
/// A cylindrical csx grid becomes a [`Grid::cylindrical`]: `ρ` and `z`
/// are scaled by the delta unit, `α` stays in radians.
///
/// # Errors
/// [`Error::Invalid`] for a non-finite or non-positive delta unit, a grid
/// with fewer than two lines on an axis, or an invalid cylindrical grid
/// (negative `ρ`, azimuth span over 2π).
pub fn grid_from_csx(g: &csx::RectilinearGrid) -> Result<Grid> {
    let unit = g.delta_unit;
    if !(unit.is_finite() && unit > 0.0) {
        return Err(Error::Invalid(format!(
            "grid delta unit {unit} must be finite and > 0"
        )));
    }
    if g.coord_system == CoordSystem::Cylindrical {
        let scale = |a: usize, s: f64| g.lines(a).iter().map(|&x| x * s).collect();
        return Grid::cylindrical([scale(0, unit), scale(1, 1.0), scale(2, unit)]);
    }
    Grid::new(core::array::from_fn(|a| {
        g.lines(a).iter().map(|&x| x * unit).collect()
    }))
}

/// Mur faces with phase velocities, and PML faces with parameters.
type AbsorbingFaces = (Vec<(Face, Option<f64>)>, Vec<(Face, Cpml)>);

/// Maps the boundary settings: returns the face conditions plus the Mur
/// and PML faces (built as PEC).
fn map_boundaries(fdtd: &Fdtd) -> (FaceConditions, AbsorbingFaces) {
    let mut faces = FaceConditions::default();
    let (mut mur, mut pml) = (Vec::new(), Vec::new());
    for (face, b) in fdtd.get_boundaries().iter() {
        faces[face] = match b {
            Boundary::Pec => FaceCondition::Pec,
            Boundary::Pmc => FaceCondition::Pmc,
            Boundary::Periodic => FaceCondition::Periodic,
            Boundary::Mur(m) => {
                mur.push((face, m.phase_velocity));
                FaceCondition::Pec
            }
            Boundary::Pml(p) => {
                pml.push((face, *p));
                FaceCondition::Pec
            }
        };
    }
    (faces, (mur, pml))
}

/// openEMS's PMC: zero the H dual edges tangential to every PMC face in
/// the dual layer next to it (see [`Compiled::openems_pmc`]).
fn pmc_half_cell(engine: &mut Engine, faces: &FaceConditions) {
    let n = engine.grid().nodes();
    let mut co = engine.coefficients_mut();
    for face in Face::ALL {
        if faces[face] != FaceCondition::Pmc {
            continue;
        }
        let t = face.axis();
        let ti = t.index();
        if n[ti] < 2 {
            continue;
        }
        let layer = if face.is_max() { n[ti] - 2 } else { 0 };
        for a in Axis::ALL {
            if a == t {
                continue;
            }
            let (da, db) = co.h_mut(a);
            for i in 0..n[0] {
                for j in 0..n[1] {
                    for k in 0..n[2] {
                        if [i, j, k][ti] != layer {
                            continue;
                        }
                        let idx = (i * n[1] + j) * n[2] + k;
                        da[idx] = 0.0;
                        db[idx] = 0.0;
                    }
                }
            }
        }
    }
}

/// Warnings for properties the compile step does not (fully) handle.
fn property_warnings(model: &csx::ContinuousStructure, out: &mut Vec<String>) {
    for p in model.properties() {
        let what = match &p.kind {
            PropertyKind::Excitation(e) if !e.enabled => "disabled; ignored",
            PropertyKind::Excitation(e)
                if !e.excitation_type.is_electric()
                    && !e.excitation_type.is_magnetic()
                    && e.excitation_type != ExcitationType::PlaneWave =>
            {
                "unknown excitation type; ignored"
            }
            PropertyKind::ProbeBox(pb) => match pb.probe_type {
                ProbeType::Voltage
                | ProbeType::Current
                | ProbeType::EField
                | ProbeType::HField
                | ProbeType::WaveguideVoltage
                | ProbeType::WaveguideCurrent => continue,
                _ => "unknown probe type; ignored",
            },
            PropertyKind::DumpBox(d) => match d.dump_type {
                DumpType::EField
                | DumpType::HField
                | DumpType::Current
                | DumpType::TotalCurrentDensity
                | DumpType::DField
                | DumpType::BField => continue,
                t if t.is_frequency_domain() && d.probe.frequencies.is_empty() => {
                    "frequency-domain dump without frequencies; ignored"
                }
                DumpType::EFieldFd
                | DumpType::HFieldFd
                | DumpType::CurrentFd
                | DumpType::TotalCurrentDensityFd
                | DumpType::DFieldFd
                | DumpType::BFieldFd
                | DumpType::SarLocal
                | DumpType::Sar1g
                | DumpType::Sar10g
                | DumpType::SarRaw => continue,
                _ => "unknown dump type; ignored",
            },
            _ => continue,
        };
        out.push(format!(
            "property {:?} ({}): {what}",
            p.name,
            p.kind.type_name()
        ));
    }
}

/// Probe and dump specs.
type Specs = (
    Vec<ProbeSpec<csx::properties::ProbeBox>>,
    Vec<ProbeSpec<csx::properties::DumpBox>>,
);

/// Snaps the ProbeBox / DumpBox properties of `model` to its grid
/// (`grid`: the same mesh in metres).
fn probe_specs(model: &csx::ContinuousStructure, grid: &Grid, warnings: &mut Vec<String>) -> Specs {
    let (mut probes, mut dumps) = (Vec::new(), Vec::new());
    for p in model.properties() {
        if !matches!(p.kind, PropertyKind::ProbeBox(_) | PropertyKind::DumpBox(_)) {
            continue;
        }
        let Some(prim) = p.primitives.first() else {
            warnings.push(format!(
                "property {:?} ({}): no primitive; ignored",
                p.name,
                p.kind.type_name()
            ));
            continue;
        };
        if p.primitives.len() > 1 {
            warnings.push(format!(
                "property {:?} ({}): only the first of {} primitives is used",
                p.name,
                p.kind.type_name(),
                p.primitives.len()
            ));
        }
        let (start, stop, corners) =
            snap_primitive_corners(&model.grid, prim, "probe/dump", &p.name, warnings);
        let du = model.grid.delta_unit;
        // Cylindrical grids keep α in radians.
        let cyl = grid.is_cylindrical();
        let corners =
            corners.map(|c| [0, 1, 2].map(|a| if cyl && a == 1 { c[a] } else { c[a] * du }));
        match &p.kind {
            PropertyKind::ProbeBox(pb) => {
                let waveguide = matches!(
                    pb.probe_type,
                    ProbeType::WaveguideVoltage | ProbeType::WaveguideCurrent
                );
                let mode = if waveguide {
                    match mode::mode_entries(
                        model,
                        p,
                        pb,
                        grid,
                        model.grid.delta_unit,
                        start,
                        stop,
                        warnings,
                    ) {
                        Some(e) => e,
                        None => continue,
                    }
                } else {
                    mode::ModeProbe {
                        entries: Vec::new(),
                        samples: None,
                        range: None,
                    }
                };
                probes.push(ProbeSpec {
                    name: p.name.clone(),
                    id: p.id(),
                    settings: pb.clone(),
                    start,
                    stop,
                    corners,
                    mode_entries: mode.entries,
                    mode_samples: mode.samples,
                    mode_range: mode.range,
                })
            }
            PropertyKind::DumpBox(d) => dumps.push(ProbeSpec {
                name: p.name.clone(),
                id: p.id(),
                settings: d.clone(),
                start,
                stop,
                corners,
                mode_entries: Vec::new(),
                mode_samples: None,
                mode_range: None,
            }),
            _ => {}
        }
    }
    (probes, dumps)
}

/// Corner node indices of a probe / dump primitive; see the module docs.
fn snap_primitive(
    grid: &csx::RectilinearGrid,
    prim: &csx::Primitive,
    what: &str,
    name: &str,
    warnings: &mut Vec<String>,
) -> ([usize; 3], [usize; 3]) {
    let (s, e, _) = snap_primitive_corners(grid, prim, what, name, warnings);
    (s, e)
}

/// [`snap_primitive`] plus the unsnapped corners (drawing units, after
/// the primitive's transformation).
fn snap_primitive_corners(
    grid: &csx::RectilinearGrid,
    prim: &csx::Primitive,
    what: &str,
    name: &str,
    warnings: &mut Vec<String>,
) -> ([usize; 3], [usize; 3], [[f64; 3]; 2]) {
    let (a, b) = primitive_corners(grid, prim, what, name, warnings);
    let mut off = false;
    let mut outside = false;
    let mut snap = |v: f64, d: usize| -> usize {
        let lines = grid.lines(d);
        if v < lines[0] || v > lines[lines.len() - 1] {
            outside = true;
        }
        if grid.snap(d, v).is_none() {
            off = true;
        }
        grid.nearest_line(d, v).unwrap_or(0)
    };
    let start = [0, 1, 2].map(|d| snap(a.0[d], d));
    let stop = [0, 1, 2].map(|d| snap(b.0[d], d));
    if outside {
        warnings.push(format!(
            "{what} {name:?}: box extends outside the mesh; clamped to the mesh"
        ));
    } else if off {
        warnings.push(format!(
            "{what} {name:?}: box corners snapped to the nearest mesh lines"
        ));
    }
    (start, stop, [a.0, b.0])
}

/// The corners of a probe / dump primitive in drawing units (after its
/// transformation), before snapping: a box's corners in order, a point
/// twice, the bounding box of other shapes (with a warning).
fn primitive_corners(
    grid: &csx::RectilinearGrid,
    prim: &csx::Primitive,
    what: &str,
    name: &str,
    warnings: &mut Vec<String>,
) -> (csx::Vec3, csx::Vec3) {
    // Box and point values are used as given when they are in the grid's
    // coordinate system (the default for primitives without their own).
    let gcs = grid.coord_system;
    let cart = prim.effective_cs(gcs) == gcs;
    let tf = |v: csx::Vec3| {
        if prim.transform.is_empty() {
            v
        } else {
            prim.transform.apply(v)
        }
    };
    match &prim.shape {
        csx::Shape::Box(bx) if cart => (tf(bx.start.value()), tf(bx.stop.value())),
        csx::Shape::Point(pt) if cart => {
            let v = tf(pt.point.value());
            (v, v)
        }
        other => {
            warnings.push(format!(
                "{what} {name:?}: {} primitive replaced by its bounding box",
                other.type_name()
            ));
            let bb = prim.bounding_box_in(gcs, gcs);
            (bb.min, bb.max)
        }
    }
}

/// Snaps the AbsorbingBC primitives of `model` to its grid (see
/// [Absorbing sheets](self#absorbing-sheets)).
fn abc_specs(model: &csx::ContinuousStructure, warnings: &mut Vec<String>) -> Vec<AbcSpec> {
    let mut out = Vec::new();
    for p in model.properties() {
        let PropertyKind::AbsorbingBC(abc) = &p.kind else {
            continue;
        };
        if p.primitives.is_empty() {
            warnings.push(format!("AbsorbingBC {:?}: no primitive; ignored", p.name));
            continue;
        }
        match abc.abc_type {
            AbcType::Mur1st => {}
            t => warnings.push(format!(
                "AbsorbingBC {:?}: AbsorbingBoundaryType {} not implemented; \
                 first-order Mur (MUR_1ST) used",
                p.name,
                t.name().map_or_else(|| t.id().to_string(), str::to_string)
            )),
        }
        let v = abc.phase_velocity.value();
        let phase_velocity = if v.is_finite() && v > 0.0 {
            Some(v)
        } else {
            if v != 0.0 {
                warnings.push(format!(
                    "AbsorbingBC {:?}: PhaseVelocity {v} invalid; local medium velocity used",
                    p.name
                ));
            }
            None
        };
        for prim in &p.primitives {
            let (a, b) = snap_primitive(&model.grid, prim, "AbsorbingBC", &p.name, warnings);
            let start = [0, 1, 2].map(|d| a[d].min(b[d]));
            let stop = [0, 1, 2].map(|d| a[d].max(b[d]));
            let flat: Vec<usize> = (0..3).filter(|&d| start[d] == stop[d]).collect();
            let &[d] = flat.as_slice() else {
                let what = if flat.is_empty() {
                    "a volume"
                } else {
                    "a line or point"
                };
                warnings.push(format!(
                    "AbsorbingBC {:?}: primitive is {what} on the mesh, not a sheet; ignored",
                    p.name
                ));
                continue;
            };
            let gcs = model.grid.coord_system;
            let bb = prim.bounding_box_in(gcs, gcs);
            if bb.max.0[d] - bb.min.0[d] > 0.0 {
                warnings.push(format!(
                    "AbsorbingBC {:?}: primitive has a thickness along {}; flattened onto \
                     mesh plane {}",
                    p.name,
                    ["x", "y", "z"][d],
                    start[d]
                ));
            }
            let normal = Axis::from_index(d).expect("axis in 0..3");
            out.push(AbcSpec {
                name: p.name.clone(),
                sheet: AbcSheet {
                    normal,
                    plane: start[d],
                    positive: !abc.normal_sign_positive,
                    start,
                    stop,
                },
                phase_velocity,
            });
        }
    }
    out
}

/// Compiles `model` with solver settings `fdtd` into engine inputs; see
/// the [module docs](self) for the steps and conventions.
///
/// # Errors
/// [`Error::Invalid`] for invalid settings or meshes (including invalid
/// cylindrical grids), [`Error::Csx`] for model errors (parameter
/// evaluation, discretisation).
pub fn compile(
    model: &csx::ContinuousStructure,
    fdtd: &Fdtd,
    opts: &CompileOptions,
) -> Result<Compiled> {
    fdtd.validate()?;
    let grid = grid_from_csx(&model.grid)?;
    let updated;
    let model = if opts.update_model {
        let mut m = model.clone();
        m.update()?;
        updated = m;
        &updated
    } else {
        model
    };
    let mut warnings: Vec<String> = model.validate().iter().map(ToString::to_string).collect();
    let grid = cylindrical::multigrid(grid, &model.grid, fdtd.get_multigrid(), &mut warnings)?;
    // Oblique thin wires are not discretised by csx (see compile/wire.rs).
    let (oblique, stripped) = wire::oblique_wires(model, &grid, &mut warnings);
    let full_model = model;
    let model = stripped.as_ref().unwrap_or(model);
    let mut mg: MaterialGrid = discretize(
        model,
        &DiscretizeOptions::default()
            .with_subsampling(opts.subsampling)
            .with_threads(fdtd.resolved_threads())
            .with_snap_tolerance(opts.snap_tolerance),
    )?;
    oblique.renumber_warnings(&mut mg.warnings);
    if fdtd.get_openems_run() {
        openems::strict_surfaces(model, mg.dims, &mut mg.pec);
    }
    let node_map = NodeMap::new(grid.nodes());
    if mg.dims != grid.nodes() {
        return Err(Error::Invalid(format!(
            "discretised grid has {:?} nodes, expected {:?}",
            mg.dims,
            grid.nodes()
        )));
    }
    let pw_names: Vec<&str> = model
        .properties()
        .iter()
        .filter(|p| {
            matches!(&p.kind, PropertyKind::Excitation(e)
                if e.excitation_type == ExcitationType::PlaneWave)
        })
        .map(|p| p.name.as_str())
        .collect();
    warnings.extend(
        mg.warnings
            .iter()
            .filter(|w| !planewave::is_csx_skip_warning(w, &pw_names))
            .cloned(),
    );
    property_warnings(model, &mut warnings);
    let cyl = grid.is_cylindrical();
    let mut plane_waves = planewave::plane_wave_sources(model, fdtd, &mut warnings);
    if cyl && !plane_waves.is_empty() {
        warnings.push(format!(
            "{} plane-wave excitation(s) ignored: plane waves need a Cartesian grid \
             (openEMS ignores them on cylindrical grids too)",
            plane_waves.len()
        ));
        plane_waves.clear();
    }

    let sheet_specs = if cyl {
        cylindrical::wrapped_sheets(
            model,
            &mut mg,
            &node_map,
            opts.snap_tolerance,
            &mut warnings,
        )
    } else {
        sheet::SheetSpec::of(&mg)
    };
    let mut sheet_map = sheet::sheet_edges(
        model,
        &mg,
        &mg.sheet,
        &sheet_specs,
        &node_map,
        opts.snap_tolerance,
        &mut warnings,
    );
    if cyl {
        cylindrical::sheet_edges(&grid, &mut sheet_map.edges, &mut warnings);
    }
    let mut sheet_pec: [Vec<bool>; 3] = core::array::from_fn(|a| vec![false; mg.pec[a].len()]);
    for &(d, idx) in &sheet_map.pec {
        sheet_pec[d][idx] = true;
    }
    let mut materials = EdgeMaterials::vacuum(&grid);
    for axis in Axis::ALL {
        let a = axis.index();
        materials
            .eps_r_mut(axis)
            .copy_from_slice(&node_map.reorder(&mg.epsilon[a]));
        materials
            .kappa_mut(axis)
            .copy_from_slice(&node_map.reorder(&mg.kappa[a]));
        materials
            .mu_r_mut(axis)
            .copy_from_slice(&node_map.reorder(&mg.mue[a]));
        materials
            .sigma_m_mut(axis)
            .copy_from_slice(&node_map.reorder(&mg.sigma[a]));
        let pec: Vec<bool> = mg.pec[a]
            .iter()
            .zip(&sheet_pec[a])
            .map(|(&p, &s)| p || s)
            .collect();
        materials
            .pec_mask_mut(axis)
            .copy_from_slice(&node_map.reorder(&pec));
    }
    if fdtd.get_cell_constant_material() {
        if cyl {
            warnings.push(
                "CellConstantMaterial is not available on cylindrical grids; materials are \
                 averaged onto the edges"
                    .into(),
            );
        } else {
            cellconst::apply(model, &grid, &mut materials)?;
        }
    }
    // (Cylindrical grids: solid metal only, see compile::conformal.)
    let conformal = if opts.conformal {
        let pec: [&[bool]; 3] = [
            materials.pec_mask(Axis::X),
            materials.pec_mask(Axis::Y),
            materials.pec_mask(Axis::Z),
        ];
        let threads = fdtd.resolved_threads();
        let geo = conformal::geometry(model, &pec, opts.snap_tolerance, threads, &mut warnings);
        for &(axis, k, v) in &geo.pec {
            materials.pec_mask_mut(axis)[k] = v;
        }
        Some(geo.conformal.with_options(opts.conformal_options))
    } else {
        None
    };

    let dispersive = dispersive::dispersive_poles(model, &node_map, &materials, &mut warnings);
    let thin_wires = wire::thin_wires(
        full_model,
        &grid,
        &mg.lumped,
        opts.snap_tolerance,
        &mut materials,
        &mut warnings,
    );

    planewave::set_backgrounds(&mut plane_waves, &grid, &materials)?;
    let (faces, (mut mur, mut pml)) = map_boundaries(fdtd);
    if cyl {
        cylindrical::drop_ignored_boundaries(&grid, &mut mur, &mut pml, &mut warnings);
    }
    sheet::pml_warning(&sheet_map.edges, grid.nodes(), &pml, &mut warnings);
    let oblique_wires = wire::oblique_operator(
        &oblique,
        &grid,
        &materials,
        &mg.lumped,
        &faces,
        &mut warnings,
    );
    let timestep = fdtd.engine_timestep();
    let MaterialGrid {
        mut excitations,
        lumped,
        sheets,
        ..
    } = mg;
    excitations.retain(|f| {
        model
            .property_by_name(&f.name)
            .is_none_or(|p| !matches!(&p.kind, PropertyKind::Excitation(e) if !e.enabled))
    });
    cylindrical::multigrid_edge_warnings(&grid, &node_map, &excitations, &lumped, &mut warnings);
    let (probes, dumps) = probe_specs(model, &grid, &mut warnings);
    let sar = sar::sar_specs(model, &dumps, opts.subsampling, &mut warnings);
    let abc = abc_specs(model, &mut warnings);
    Ok(Compiled {
        grid,
        delta_unit: model.grid.delta_unit,
        materials,
        faces,
        openems_pmc: fdtd.get_openems_run(),
        mur,
        pml,
        timestep,
        threads: fdtd.get_threads().unwrap_or(0),
        node_map,
        abc,
        dispersive,
        dispersive_scheme: opts.dispersive_scheme,
        thin_wires,
        oblique_wires,
        sheets: sheet_map.edges,
        max_frequency: fdtd.get_max_frequency(),
        conformal,
        sources: ModelSources {
            excitations,
            lumped,
            sheets,
            probes,
            dumps,
            plane_waves,
            sar,
            delta_unit: model.grid.delta_unit,
        },
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_map_round_trip() {
        let m = NodeMap::new([3, 4, 5]);
        let csx: Vec<usize> = (0..60).collect();
        let e = m.reorder(&csx);
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    let n = [i, j, k];
                    assert_eq!(m.csx_node(m.csx_index(n)), n);
                    assert_eq!(e[m.engine_index(n)], m.csx_index(n));
                }
            }
        }
        let (a, n) = m.edge(csx::discretize::Edge {
            dir: 2,
            index: m.csx_index([1, 2, 3]),
        });
        assert_eq!((a, n), (Axis::Z, [1, 2, 3]));
    }
}
