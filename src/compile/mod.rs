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
//!    metres. Cylindrical grids are rejected with [`Error::Unsupported`]
//!    (milestone M4).
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
//! 4. **Metal.** The csx PEC mask becomes the engine PEC mask. Conducting
//!    sheets are **approximated as PEC** for now (thin-conductor model is
//!    M2); every such sheet adds a warning.
//! 5. **Boundaries** from [`Fdtd::get_boundaries`]: PEC and PMC map to
//!    [`FaceCondition`]s; Mur and PML faces are built as PEC faces and
//!    listed in [`Compiled::mur`] / [`Compiled::pml`], from which
//!    [`Compiled::build`] creates **one** [`MurBoundary`] covering all Mur
//!    faces and **one** [`CpmlBoundary`] covering all PML faces.
//! 6. **Absorbing sheets.** csx `AbsorbingBC` primitives become
//!    [`AbcSpec`]s ([`Compiled::abc`]), installed by [`Compiled::build`] as
//!    one [`AbsorbingSheet`] each (see [Absorbing sheets](#absorbing-sheets)).
//! 7. **Time step and threads** from [`Fdtd`]: a time step override becomes
//!    [`TimeStep::Fixed`], otherwise [`TimeStep::Courant`] with the time
//!    step factor; the thread count is used for both the discretisation and
//!    the engine.
//!
//! # Dispersive materials
//!
//! csx `LorentzMaterial` (Drude/Lorentz) and `DebyeMaterial` properties
//! contribute their static part (`Epsilon`, `Mue`, … = `ε∞`, `μ∞`)
//! through `discretize` like any material, and their poles through
//! [`Compiled::dispersive`]: per E edge (and H dual edge for magnetic
//! Lorentz poles), sampled with csx's priority query over the same
//! quarter / half cells as the static average. [`Compiled::build`]
//! installs them as one [`Dispersive`] operator. Parameter conventions
//! (openEMS documentation) and the sampling rules are documented in
//! [`dispersive`]; in short: `ω_p = 2π·f_plasma·√ε∞`,
//! `ω_0 = 2π·f_Lor`, `γ = 1/τ` (0 for `τ = 0`), Debye `Δε`, `τ` as given.
//!
//! # Absorbing sheets
//!
//! Every primitive of an `AbsorbingBC` property is snapped to the mesh like
//! a probe box (see below) and must then be flat along exactly one axis,
//! the sheet normal; anything else (a volume, a line, a point) is ignored
//! with a warning, as is a box that was not flat before snapping (it is
//! still used, flattened by the snap, with a warning). `NormalSignPositive`
//! gives the outward normal, i.e. the direction of the waves the sheet
//! absorbs ([`AbcSheet::positive`]). `PhaseVelocity` > 0 is used as given;
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

pub mod dispersive;

use crate::axis::Axis;
use crate::engine::boundary::abc::{AbcSheet, AbsorbingSheet, SheetVelocity};
use crate::engine::boundary::cpml::CpmlBoundary;
use crate::engine::boundary::mur::MurBoundary;
use crate::engine::{
    Dispersive, EdgeMaterials, EdgePole, Engine, FaceCondition, FaceConditions, Grid,
    OperatorHandle, TimeStep,
};
use crate::fdtd::{Boundary, Cpml, Face, Fdtd};
use crate::{Error, Result};

use csx::discretize::{
    DiscretizeOptions, ExcitationField, LumpedEdge, MaterialGrid, SheetInfo, discretize,
};
use csx::properties::{AbcType, DumpMode, DumpType, ProbeType};
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
}

impl Default for CompileOptions {
    fn default() -> Self {
        let d = DiscretizeOptions::default();
        Self {
            subsampling: d.subsampling,
            snap_tolerance: d.snap_tolerance,
            update_model: true,
            mesh_check: true,
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
    /// Conducting sheets (currently approximated as PEC).
    pub sheets: Vec<SheetInfo>,
    /// ProbeBox properties, snapped to the mesh.
    pub probes: Vec<ProbeSpec<csx::properties::ProbeBox>>,
    /// DumpBox properties, snapped to the mesh.
    pub dumps: Vec<ProbeSpec<csx::properties::DumpBox>>,
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
    /// Mur faces and their phase velocities \[m/s\].
    pub mur: Vec<(Face, f64)>,
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
    /// One operator per [`Compiled::abc`] sheet, in the same order.
    pub abc: Vec<OperatorHandle<AbsorbingSheet>>,
}

impl Compiled {
    /// Builds the engine (coefficients, threads) and registers the boundary
    /// operators, first the [`CpmlBoundary`], then the [`MurBoundary`],
    /// then the [`Dispersive`] medium (after the CPML, so that a medium
    /// reaching into the PML is coupled implicitly with it), then the
    /// [`AbsorbingSheet`]s (last, so that they see the final interior E).
    ///
    /// # Errors
    /// Invalid time step or materials, or boundary operator setup errors
    /// (e.g. a PML thicker than the mesh).
    pub fn build(&self) -> Result<(Engine, BoundaryHandles)> {
        let mut engine = Engine::builder(self.grid.clone())
            .edge_materials(self.materials.clone())
            .faces(self.faces)
            .timestep(self.timestep)
            .threads(self.threads)
            .build()?;
        let mut h = BoundaryHandles::default();
        if !self.pml.is_empty() {
            let op = CpmlBoundary::new(&engine, self.pml.iter().copied())?;
            h.cpml = Some(engine.add_operator(op));
        }
        if !self.mur.is_empty() {
            let op = MurBoundary::with_faces(&engine, &self.mur)?;
            h.mur = Some(engine.add_operator(op));
        }
        if !self.dispersive.is_empty() {
            let mut d = Dispersive::new();
            for &e in &self.dispersive {
                d.push_entry(e);
            }
            h.dispersive = Some(d.add_to(&mut engine)?);
        }
        for a in &self.abc {
            let v = match a.phase_velocity {
                Some(v) => SheetVelocity::Fixed(v),
                None => SheetVelocity::Local(&self.materials),
            };
            let op = AbsorbingSheet::new(&engine, &a.sheet, v)
                .map_err(|e| Error::Invalid(format!("AbsorbingBC {:?}: {e}", a.name)))?;
            h.abc.push(engine.add_operator(op));
        }
        Ok((engine, h))
    }
}

/// Converts the csx grid (drawing units) into an engine [`Grid`] in metres.
///
/// # Errors
/// [`Error::Unsupported`] for a cylindrical grid; [`Error::Invalid`] for a
/// non-finite or non-positive delta unit or a grid with fewer than two
/// lines on an axis.
pub fn grid_from_csx(g: &csx::RectilinearGrid) -> Result<Grid> {
    if g.coord_system != CoordSystem::Cartesian {
        return Err(Error::Unsupported(
            "cylindrical grids are not supported yet (milestone M4)".into(),
        ));
    }
    let unit = g.delta_unit;
    if !(unit.is_finite() && unit > 0.0) {
        return Err(Error::Invalid(format!(
            "grid delta unit {unit} must be finite and > 0"
        )));
    }
    Grid::new(core::array::from_fn(|a| {
        g.lines(a).iter().map(|&x| x * unit).collect()
    }))
}

/// Mur faces with phase velocities, and PML faces with parameters.
type AbsorbingFaces = (Vec<(Face, f64)>, Vec<(Face, Cpml)>);

/// Maps the boundary settings: returns the face conditions plus the Mur
/// and PML faces (built as PEC).
fn map_boundaries(fdtd: &Fdtd) -> (FaceConditions, AbsorbingFaces) {
    let mut faces = FaceConditions::default();
    let (mut mur, mut pml) = (Vec::new(), Vec::new());
    for (face, b) in fdtd.get_boundaries().iter() {
        faces[face] = match b {
            Boundary::Pec => FaceCondition::Pec,
            Boundary::Pmc => FaceCondition::Pmc,
            Boundary::Mur(m) => {
                mur.push((face, m.velocity()));
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

/// Warnings for properties the compile step does not (fully) handle.
fn property_warnings(model: &csx::ContinuousStructure, out: &mut Vec<String>) {
    for p in model.properties() {
        let what = match &p.kind {
            PropertyKind::Excitation(e) if !e.enabled => "disabled; ignored",
            PropertyKind::Excitation(e)
                if !e.excitation_type.is_electric() && !e.excitation_type.is_magnetic() =>
            {
                "only E- and H-field excitations are supported (plane waves: later); ignored"
            }
            PropertyKind::ProbeBox(pb) => match pb.probe_type {
                ProbeType::Voltage | ProbeType::Current | ProbeType::EField | ProbeType::HField => {
                    continue;
                }
                _ => "waveguide probes are not supported yet (M2); ignored",
            },
            PropertyKind::DumpBox(d) => match d.dump_type {
                DumpType::EField | DumpType::HField => continue,
                DumpType::EFieldFd | DumpType::HFieldFd if d.probe.frequencies.is_empty() => {
                    "frequency-domain dump without frequencies; ignored"
                }
                DumpType::EFieldFd | DumpType::HFieldFd => continue,
                _ => "only E/H field dumps (time or frequency domain) are supported; ignored",
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

/// Snaps the ProbeBox / DumpBox properties of `model` to its grid.
fn probe_specs(model: &csx::ContinuousStructure, warnings: &mut Vec<String>) -> Specs {
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
        let (start, stop) = snap_primitive(&model.grid, prim, "probe/dump", &p.name, warnings);
        match &p.kind {
            PropertyKind::ProbeBox(pb) => probes.push(ProbeSpec {
                name: p.name.clone(),
                id: p.id(),
                settings: pb.clone(),
                start,
                stop,
            }),
            PropertyKind::DumpBox(d) => {
                if d.dump_mode == DumpMode::NoInterpolation {
                    warnings.push(format!(
                        "dump {:?}: NoInterpolation is dumped at the nodes (Node mode)",
                        p.name
                    ));
                }
                if d.opt_resolution.is_some() {
                    warnings.push(format!(
                        "dump {:?}: OptResolution is not supported; full resolution used",
                        p.name
                    ));
                }
                dumps.push(ProbeSpec {
                    name: p.name.clone(),
                    id: p.id(),
                    settings: d.clone(),
                    start,
                    stop,
                })
            }
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
    let cart = prim
        .coord_system
        .is_none_or(|c| c == CoordSystem::Cartesian);
    let tf = |v: csx::Vec3| {
        if prim.transform.is_empty() {
            v
        } else {
            prim.transform.apply(v)
        }
    };
    let (a, b) = match &prim.shape {
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
            let bb = prim.bounding_box(CoordSystem::Cartesian);
            (bb.min, bb.max)
        }
    };
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
    (start, stop)
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
            let bb = prim.bounding_box(CoordSystem::Cartesian);
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
                    positive: abc.normal_sign_positive,
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
/// [`Error::Unsupported`] for cylindrical grids, [`Error::Invalid`] for
/// invalid settings or meshes, [`Error::Csx`] for model errors (parameter
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
    let mg: MaterialGrid = discretize(
        model,
        &DiscretizeOptions::default()
            .with_subsampling(opts.subsampling)
            .with_threads(fdtd.resolved_threads())
            .with_snap_tolerance(opts.snap_tolerance),
    )?;
    let node_map = NodeMap::new(grid.nodes());
    if mg.dims != grid.nodes() {
        return Err(Error::Invalid(format!(
            "discretised grid has {:?} nodes, expected {:?}",
            mg.dims,
            grid.nodes()
        )));
    }
    warnings.extend(mg.warnings.iter().cloned());
    property_warnings(model, &mut warnings);

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
        let mut pec: Vec<bool> = mg.pec[a]
            .iter()
            .zip(&mg.sheet[a])
            .map(|(&p, &s)| p || s != 0)
            .collect();
        pec = node_map.reorder(&pec);
        materials.pec_mask_mut(axis).copy_from_slice(&pec);
    }
    for s in &mg.sheets {
        warnings.push(format!(
            "ConductingSheet {:?} (conductivity {} S/m, thickness {} m) approximated as PEC (M2)",
            s.name, s.conductivity, s.thickness
        ));
    }

    let dispersive =
        dispersive::dispersive_poles(model, &node_map, opts.subsampling, &mut warnings);

    let (faces, (mur, pml)) = map_boundaries(fdtd);
    let timestep = match fdtd.get_timestep() {
        Some(dt) => TimeStep::Fixed(dt),
        None => TimeStep::Courant(fdtd.get_timestep_factor()),
    };
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
    let (probes, dumps) = probe_specs(model, &mut warnings);
    let abc = abc_specs(model, &mut warnings);
    Ok(Compiled {
        grid,
        delta_unit: model.grid.delta_unit,
        materials,
        faces,
        mur,
        pml,
        timestep,
        threads: fdtd.get_threads().unwrap_or(0),
        node_map,
        abc,
        dispersive,
        sources: ModelSources {
            excitations,
            lumped,
            sheets,
            probes,
            dumps,
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
