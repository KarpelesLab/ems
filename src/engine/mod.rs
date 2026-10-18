//! FDTD engine core: mesh, fields, coefficients, kernels, thread pool and
//! operator hooks.
//!
//! The engine is independent of [`csx`]: it is built from a [`Grid`], a
//! material map (per cell, [`CellMaterials`], or per edge,
//! [`EdgeMaterials`]), face conditions and a [`TimeStep`]. The csx → engine
//! compile step ([`crate::compile`]) produces exactly these inputs.
//!
//! # Formulation
//!
//! Standard Yee scheme (Yee 1966) in integral ("voltage/current") form, as
//! in Taflove & Hagness, *Computational Electrodynamics*, 3rd ed., ch. 3:
//!
//! ```text
//! C·du/dt + G·u = ∮ i   (around the dual face pierced by the primary edge)
//! L·di/dt + R·i = −∮ u  (around the primary face pierced by the dual edge)
//! ```
//!
//! with `u = ∫E·dl` along primary edges and `i = ∫H·dl` along dual edges.
//! See [`Coefficients`] for the circuit quantities and update coefficients.
//!
//! # Array layout
//!
//! Field components and edge directions are selected by [`Axis`]; in the
//! table below `u[0]` is the [`Axis::X`] component of `u`, and so on
//! ([`Axis::index`]). Outer faces are named by [`Face`]. Every field and
//! coefficient array has one entry per **primary node** `[i, j, k]`
//! (`i < nx`, `j < ny`, `k < nz`) at linear index `(i·ny + j)·nz + k`
//! (z contiguous). Write `x_i`, `y_j`, `z_k` for the grid lines and
//! `x_{i+½}` for the midpoint of cell `i`.
//!
//! | array | entry `[i, j, k]` is | valid for | padding (always 0) |
//! |---|---|---|---|
//! | `u[0]` | E_x edge `(x_i..x_{i+1}, y_j, z_k)` | `i < nx−1` | `i = nx−1` |
//! | `u[1]` | E_y edge `(x_i, y_j..y_{j+1}, z_k)` | `j < ny−1` | `j = ny−1` |
//! | `u[2]` | E_z edge `(x_i, y_j, z_k..z_{k+1})` | `k < nz−1` | `k = nz−1` |
//! | `i[0]` | H_x dual edge `(x_{i−½}..x_{i+½}, y_{j+½}, z_{k+½})` | `j < ny−1`, `k < nz−1` | `j = ny−1` or `k = nz−1` |
//! | `i[1]` | H_y dual edge `(x_{i+½}, y_{j−½}..y_{j+½}, z_{k+½})` | `i < nx−1`, `k < nz−1` | `i = nx−1` or `k = nz−1` |
//! | `i[2]` | H_z dual edge `(x_{i+½}, y_{j+½}, z_{k−½}..z_{k+½})` | `i < nx−1`, `j < ny−1` | `i = nx−1` or `j = ny−1` |
//!
//! So `i[a]` at `[i, j, k]` pierces the primary face spanned by the two
//! transverse edges starting at node `[i, j, k]`. Dual edges at the outer
//! nodes (`i = 0` or `nx−1` for `i[0]`, etc.) are half length
//! ([`Grid::dual_delta`]). The kernels read out-of-domain neighbours
//! (`j−1 < 0`, …) as zero.
//!
//! The kernels also compute the **tangential E edges on the six outer
//! faces** (e.g. `u[1]` at `i = 0`), using their half (quarter at domain
//! edges) dual face. Whether they stay at zero is decided purely by their
//! coefficients; see [`FaceCondition`].
//!
//! # Time staggering (leapfrog)
//!
//! After `n` completed steps ([`Engine::timestep`] = n), `u` holds
//! `t = n·dt` ([`Engine::time`]) and `i` holds `t = (n − ½)·dt`. One
//! [`Engine::step`] runs `update_h` (`i`: `n−½ → n+½`, using `u^n`) then
//! `update_e` (`u`: `n → n+1`, using `i^{n+½}`):
//!
//! ```text
//! i^{n+½} = da·i^{n−½} − db·(∮u)^n
//! u^{n+1} = ca·u^n     + cb·(∮i)^{n+½}
//! ```
//!
//! # Threads and determinism
//!
//! The x-planes are split into contiguous slabs, one per thread. The H
//! update only writes `i` and only reads `u` (own slab plus the first plane
//! of the next slab); the E update only writes `u` and reads `i` (own slab
//! plus the last plane of the previous slab). Each slab's E and H blocks sit
//! behind separate `RwLock`s, so this is expressed in safe Rust. A step
//! costs one barrier per half-step. Bulk operator work (CPML) runs inside
//! the half-steps on every slab ([`SlabHooks`]); only serial operator hooks
//! ([`Operator::hooks`]) add a barrier, once per step for each of the two
//! gaps that has work (`post_h`/`pre_e`, and `post_e`/`pre_h`), while the
//! hooks run on the calling thread. Every value is computed by the same
//! expression regardless of the partition and there are no cross-thread
//! reductions, so results are bit-identical for any thread count. Energy
//! sums are formed per x-plane in a fixed order and then combined in plane
//! order, so they are bit-identical too.
//!
//! # Energy
//!
//! Write the lossless update in matrix form, with `K` the primary-face
//! circulation of `u` (so that `Kᵀ` is the dual-face circulation of `i`,
//! the integral-form Yee curls being transposes of each other):
//!
//! ```text
//! L·(i^{n+½} − i^{n−½}) = −dt·K·u^n
//! C·(u^{n+1} − u^n)     =  dt·Kᵀ·i^{n+½}
//! ```
//!
//! Define `W^n = ½·uⁿᵀ·C·uⁿ + ½·i^{n−½}ᵀ·L·i^{n+½}`. Then
//!
//! ```text
//! W^{n+1} − W^n = ½(u^{n+1} + u^n)ᵀ·C·(u^{n+1} − u^n)
//!               + ½ i^{n+½}ᵀ·L·(i^{n+3/2} − i^{n−½})
//!               = (dt/2)·(u^{n+1} + u^n)ᵀ·Kᵀ·i^{n+½}
//!               − (dt/2)·i^{n+½}ᵀ·K·(u^{n+1} + u^n)  = 0,
//! ```
//!
//! using the E update in the first term and the two H updates around
//! `n+½` in the second. So `W^n` is exactly conserved by the scheme (it is
//! the energy-like invariant of leapfrog; positive definite below the
//! Courant limit), whereas `½·Σ C·u² + ½·Σ L·i²` with `u` and `i` half a
//! step apart is not ([`Engine::energy_staggered`]). The dual form
//! `½·uⁿᵀC·u^{n+1} + ½·|i^{n+½}|²_L` is conserved as well; we use `W^n`
//! because it is a quantity at `t = n·dt` that the H update can produce
//! (it reads `u^n` and has both `i^{n−½}` and `i^{n+½}` at hand).
//!
//! `C` and `L` are those *of the `f32` coefficients*, which makes the
//! invariant exact for the scheme actually run: from
//! `u^{n+1} = ca·u^n + cb·Kᵀi`, with `C = dt·(1 + ca)/(2·cb)` and
//! `G = (1 − ca)/cb` the update is exactly
//! `C·(u^{n+1} − u^n)/dt + G·(u^{n+1} + u^n)/2 = Kᵀi` (likewise `L`, `R`
//! from `da`, `db`).
//! With these losses the same computation gives
//!
//! ```text
//! W^{n+1} − W^n = −(dt/4)·[ (u^{n+1} + u^n)ᵀ·G·(u^{n+1} + u^n)
//!                         + i^{n+½}ᵀ·R·(i^{n+3/2} + 2·i^{n+½} + i^{n−½}) ],
//! ```
//!
//! so electric loss makes `W` non-increasing step by step; the magnetic
//! term is a positive dissipation on average (exactly so for a slowly
//! varying `i`). Edges with `cb = 0` (PEC, padding) or `db = 0` are
//! skipped. Operators (sources, boundaries, lumped elements) may add
//! or remove energy; the invariant then is only the field energy, plus
//! what operators report as stored in their own state
//! ([`Operator::energy`], e.g. lumped L and C branches): for a lossless
//! closed region with lossless lumped elements that sum is conserved.
//!
//! Measured on lossless closed cavities (tests/engine.rs): after the source
//! has stopped, `W` stays constant to about `1e-6` relative over thousands
//! of steps (`f32` rounding of the fields), on uniform and graded meshes
//! with dielectric fill.

pub mod bloch_wave;
pub mod boundary;
mod coeffs;
pub mod conformal;
pub mod cylindrical;
pub mod dispersive;
pub mod dump;
pub mod edge_dft;
mod edge_material;
mod excite;
mod fields;
mod grid;
mod kernel;
pub mod lumped;
mod material;
mod operator;
pub mod planewave;
mod pool;
pub mod port;
mod probe;
mod rows;
pub(crate) mod run;
pub mod sheet;
mod slab;
pub mod wire;

use core::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub use bloch_wave::{PeriodicPlaneWave, PeriodicPlaneWaveSpec, WavePart};
pub use boundary::periodic::BlochPair;
pub use coeffs::{
    Coefficients, FaceCondition, FaceConditions, TimeStep, courant_limit, dual_edge_lr,
    primary_edge_cg,
};
pub use conformal::{
    ConformalEdge, ConformalFace, ConformalOptions, ConformalPec, ConformalReport, MergedFaces,
};
pub use dispersive::{Dispersive, EdgePole, Pole};
pub use dump::{
    DumpPosition, DumpQuantity, DumpRegion, DumpSink, FieldSampler, FreqDump, MemorySink, Schedule,
    Snapshot, TimeDump, VtrSeries,
};
pub use edge_dft::{CellBox, EdgeDft};
pub use edge_material::EdgeMaterials;
pub use excite::{Amplitude, EdgeSource, SourceKind};
pub use fields::{FieldArrays, Fields};
pub use grid::{Coordinates, Grid};
pub use lumped::{EdgeValues, LumpedBox, LumpedElement, Topology, fold_parallel};
pub use material::{CellMaterials, Material};
pub use operator::{Hooks, Operator, OperatorHandle, StepContext};
pub use planewave::{PlaneWave, PlaneWaveSpec};
pub use port::LumpedPort;
pub use probe::{FieldProbe, Probe, ProbeQuantity};
pub use run::{Progress, RunLimits, RunOutcome, StopReason};
pub use sheet::{ConductingSheet, SheetEdge};
pub use slab::{SlabContext, SlabHooks, SlabMut, SlabPartition, SlabRef};
pub use wire::ThinWire;

use crate::axis::Axis;
use crate::fdtd::Face;
use crate::{Error, Result};
use boundary::periodic::Periodic;
use fields::{FieldStore, Layout, read, write};
use pool::{Gap, Gaps, Plan, Pool, Shared};

/// Configures and builds an [`Engine`]; see [`Engine::builder`].
#[derive(Debug, Clone)]
pub struct EngineBuilder {
    grid: Grid,
    materials: Option<Materials>,
    faces: FaceConditions,
    timestep: TimeStep,
    threads: usize,
}

/// The material input of an [`EngineBuilder`].
#[derive(Debug, Clone)]
enum Materials {
    Cells(CellMaterials),
    Edges(Box<EdgeMaterials>),
}

impl EngineBuilder {
    /// Per-cell material map (default: vacuum, no PEC edges). Replaces a
    /// map set with [`edge_materials`](Self::edge_materials).
    pub fn materials(mut self, m: CellMaterials) -> Self {
        self.materials = Some(Materials::Cells(m));
        self
    }

    /// Per-edge material map (see [`EdgeMaterials`]), as produced by the
    /// compile step. Replaces a map set with
    /// [`materials`](Self::materials).
    pub fn edge_materials(mut self, m: EdgeMaterials) -> Self {
        self.materials = Some(Materials::Edges(Box::new(m)));
        self
    }

    /// Condition of the outer face `face`.
    pub fn face(mut self, face: Face, c: FaceCondition) -> Self {
        self.faces[face] = c;
        self
    }

    /// Conditions of all six outer faces (default: all PEC). Accepts a
    /// [`FaceConditions`] or a `[FaceCondition; 6]` in [`Face::ALL`] order.
    pub fn faces(mut self, faces: impl Into<FaceConditions>) -> Self {
        self.faces = faces.into();
        self
    }

    /// Time step choice (default: Courant factor 0.99).
    pub fn timestep(mut self, t: TimeStep) -> Self {
        self.timestep = t;
        self
    }

    /// Number of threads, including the caller; `1` runs everything
    /// inline. Capped at `nx` (one x-plane per thread at least).
    ///
    /// `0` (default) picks [`default_threads`] for the grid: one thread per
    /// physical core, fewer on small grids.
    pub fn threads(mut self, n: usize) -> Self {
        self.threads = n;
        self
    }

    /// Computes the time step and coefficients, allocates the fields and
    /// starts the worker threads.
    ///
    /// # Errors
    /// Invalid time step or materials, or a [`FaceCondition::Periodic`]
    /// face whose opposite face is not periodic.
    pub fn build(self) -> Result<Engine> {
        let grid = self.grid;
        let faces = cylindrical::effective_faces(&grid, &self.faces);
        let periodic = Periodic::from_faces(&faces, grid.nodes())?;
        let axis = cylindrical::AxisMerge::new(&grid);
        let dt = self.timestep.resolve(&grid)?;
        let coeffs = match &self.materials {
            Some(Materials::Cells(m)) => Coefficients::compute(&grid, m, &self.faces, dt)?,
            Some(Materials::Edges(m)) => Coefficients::from_edges(&grid, m, &self.faces, dt)?,
            None => {
                Coefficients::from_edges(&grid, &EdgeMaterials::vacuum(&grid), &self.faces, dt)?
            }
        };
        let threads = match self.threads {
            0 => default_threads(grid.num_cells()),
            n => n,
        };
        let layout = Layout::new(grid.nodes(), threads);
        let threads = layout.slabs();
        let shared = Arc::new(Shared::new(FieldStore::new(layout), coeffs, threads));
        let pool = if threads > 1 {
            Some(Pool::spawn(&shared).map_err(|e| Error::Io(format!("spawning workers: {e}")))?)
        } else {
            None
        };
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Ok(Engine {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            grid,
            dt,
            step: 0,
            shared,
            ops: Vec::new(),
            periodic,
            axis,
            pool,
        })
    }
}

/// Grid cells per thread below which the default thread count is reduced:
/// on small grids the per-step synchronisation of more threads costs more
/// than their share of the update.
const MIN_CELLS_PER_THREAD: usize = 8192;

/// The thread count an engine uses by default for a grid of `cells` cells:
/// one thread per physical core available to the process, but at most one
/// per 8192 grid cells (and at least 1).
///
/// The kernels are memory-bound, so a second hardware thread on a core
/// (SMT) adds no throughput, while every extra thread adds to the per-step
/// synchronisation and makes the run more sensitive to other load on the
/// machine. Physical cores are counted from
/// `/sys/devices/system/cpu/*/topology` on Linux: the
/// [available parallelism](std::thread::available_parallelism) divided by
/// the number of hardware threads per core. Elsewhere, or if that
/// information is unavailable, the available parallelism is used.
pub fn default_threads(cells: usize) -> usize {
    let avail = std::thread::available_parallelism().map_or(1, |n| n.get());
    let cores = (avail / smt_width()).max(1);
    cores.min(cells / MIN_CELLS_PER_THREAD).max(1)
}

/// Hardware threads per core (1 if unknown): logical CPUs divided by
/// distinct cores, from Linux sysfs.
fn smt_width() -> usize {
    use std::sync::OnceLock;
    static W: OnceLock<usize> = OnceLock::new();
    *W.get_or_init(|| {
        let mut cores = std::collections::HashSet::new();
        let mut cpus = 0usize;
        for cpu in 0.. {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
            match std::fs::read_to_string(path) {
                Ok(s) => {
                    cpus += 1;
                    cores.insert(s.trim().to_owned());
                }
                Err(_) => break,
            }
        }
        if cores.is_empty() {
            1
        } else {
            (cpus / cores.len()).max(1)
        }
    })
}

/// A ready-to-run FDTD engine.
///
/// Stepping is infallible. Dropping the engine stops its worker threads.
pub struct Engine {
    id: u64,
    grid: Grid,
    dt: f64,
    step: u64,
    shared: Arc<Shared>,
    ops: Vec<Registered>,
    /// Periodic face merge, run in the `post_e` gap.
    periodic: Option<Periodic>,
    /// Axis merge of a cylindrical grid, run after the periodic merge.
    axis: Option<cylindrical::AxisMerge>,
    pool: Option<Pool>,
}

/// A registered operator and the serial hooks it declared.
struct Registered {
    op: Box<dyn Operator>,
    hooks: Hooks,
}

impl core::fmt::Debug for Engine {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Engine")
            .field("nodes", &self.grid.nodes())
            .field("dt", &self.dt)
            .field("step", &self.step)
            .field("threads", &self.threads())
            .field("operators", &self.ops.len())
            .finish_non_exhaustive()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(p) = &mut self.pool {
            p.shutdown(&self.shared);
        }
    }
}

impl Engine {
    /// Starts configuring an engine on `grid`.
    pub fn builder(grid: Grid) -> EngineBuilder {
        EngineBuilder {
            grid,
            materials: None,
            faces: FaceConditions::default(),
            timestep: TimeStep::default(),
            threads: 0,
        }
    }

    /// The mesh.
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Time step \[s\].
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Number of completed steps `n`.
    pub fn timestep(&self) -> u64 {
        self.step
    }

    /// Time of the E voltages, `n·dt` \[s\] (H currents are at
    /// `(n − ½)·dt`).
    pub fn time(&self) -> f64 {
        self.step as f64 * self.dt
    }

    /// Number of threads (slabs) actually used.
    pub fn threads(&self) -> usize {
        self.shared.store.layout.slabs()
    }

    /// Registers an operator; it runs after all previously registered ones
    /// in each hook. The handle gives typed access back.
    ///
    /// [`Operator::hooks`] and [`Operator::slab_hooks`] are queried here,
    /// once.
    pub fn add_operator<T: Operator>(&mut self, mut op: T) -> OperatorHandle<T> {
        let hooks = op.hooks();
        let part = SlabPartition {
            grid: &self.grid,
            dt: self.dt,
            starts: &self.shared.store.layout.slab_start,
        };
        if let Some(s) = op.slab_hooks(&part) {
            write(&self.shared.slab_ops).0.push(s);
        }
        self.ops.push(Registered {
            op: Box::new(op),
            hooks,
        });
        OperatorHandle {
            engine: self.id,
            index: self.ops.len() - 1,
            _t: PhantomData,
        }
    }

    fn op_index<T>(&self, h: &OperatorHandle<T>) -> usize {
        assert_eq!(h.engine, self.id, "operator handle from another engine");
        h.index
    }

    /// The operator behind `h`.
    ///
    /// # Panics
    /// If `h` was returned by a different engine.
    pub fn operator<T: Operator>(&self, h: &OperatorHandle<T>) -> &T {
        let op: &dyn core::any::Any = &*self.ops[self.op_index(h)].op;
        op.downcast_ref().expect("operator type matches its handle")
    }

    /// Mutable access to the operator behind `h`.
    ///
    /// # Panics
    /// If `h` was returned by a different engine.
    pub fn operator_mut<T: Operator>(&mut self, h: &OperatorHandle<T>) -> &mut T {
        let i = self.op_index(h);
        let op: &mut dyn core::any::Any = &mut *self.ops[i].op;
        op.downcast_mut().expect("operator type matches its handle")
    }

    /// Read access to the update coefficients.
    pub fn coefficients(&self) -> impl core::ops::Deref<Target = Coefficients> + '_ {
        read(&self.shared.coeffs)
    }

    /// Mutable access to the update coefficients, for boundary and
    /// lumped-element setup between steps.
    pub fn coefficients_mut(&mut self) -> impl core::ops::DerefMut<Target = Coefficients> + '_ {
        self.shared.coeffs_changed();
        write(&self.shared.coeffs)
    }

    /// Exclusive access to all field values (e.g. to set initial
    /// conditions).
    pub fn fields_mut(&mut self) -> Fields<'_> {
        self.shared.store.lock()
    }

    /// E edge voltage of component `axis` at `node` \[V\].
    pub fn u(&self, axis: Axis, node: [usize; 3]) -> f32 {
        let (s, o) = self.shared.store.layout.locate(node);
        read(&self.shared.store.u[s])[axis.index()][o]
    }

    /// H dual-edge current of component `axis` at `node` \[A\].
    pub fn i(&self, axis: Axis, node: [usize; 3]) -> f32 {
        let (s, o) = self.shared.store.layout.locate(node);
        read(&self.shared.store.i[s])[axis.index()][o]
    }

    /// Copy of the E voltages of component `axis` in global layout.
    pub fn u_component(&self, axis: Axis) -> Vec<f32> {
        self.shared
            .store
            .u
            .iter()
            .flat_map(|b| read(b)[axis.index()].clone())
            .collect()
    }

    /// Copy of the H currents of component `axis` in global layout.
    pub fn i_component(&self, axis: Axis) -> Vec<f32> {
        self.shared
            .store
            .i
            .iter()
            .flat_map(|b| read(b)[axis.index()].clone())
            .collect()
    }

    /// Electromagnetic energy at `t = n·dt` \[J\], as the discrete invariant
    /// of the leapfrog scheme (design §5.8):
    ///
    /// ```text
    /// W^n = ½·Σ C·(u^n)² + ½·Σ L·i^{n−½}·i^{n+½}  +  Σ_ops W_op^n
    /// ```
    ///
    /// In a lossless closed region with no operators, `W^n` is exactly
    /// constant (up to `f32` rounding of the fields); with losses it never
    /// increases. See the [module docs](self#energy) for the derivation.
    /// The last term is the energy stored in operator state
    /// ([`Operator::energy`], summed in registration order after the field
    /// part, see [`operator_energy`](Self::operator_energy)); with it, a
    /// cavity loaded by lossless lumped L/C elements conserves `W^n` too.
    ///
    /// `i^{n+½}` is not stored, so this evaluates the H update once more
    /// without committing it (a full read-only pass over the grid on the
    /// calling thread, roughly the cost of one H half-step plus a division
    /// per edge). Operators are not run for that look-ahead, so an operator
    /// acting on `i` in `pre_h`, `post_h` or a slab hook (e.g. CPML) is not
    /// reflected in the `i^{n+½}` factor. [`step_with_energy`](Self::step_with_energy) and
    /// [`run_until`](Self::run_until) obtain the same value as a by-product
    /// of a real step, in parallel.
    ///
    /// The result is bit-identical for any thread count.
    pub fn energy(&self) -> f64 {
        self.shared.current_energy(self.dt) + self.operator_energy()
    }

    /// Sum of [`Operator::energy`] over the registered operators, in
    /// registration order \[J\] (`t = n·dt`).
    pub fn operator_energy(&self) -> f64 {
        self.ops.iter().map(|r| r.op.energy()).sum()
    }

    /// The simple staggered energy `½·Σ C·u² + ½·Σ L·i²` \[J\], with `u` at
    /// `n·dt` and `i` at `(n − ½)·dt`.
    ///
    /// Always non-negative and cheap (no curl), but because `u` and `i` are
    /// half a step apart it is not conserved: in a resonating cavity it
    /// ripples at twice the mode frequencies, by up to `O(ω·dt)` relative
    /// (tens of percent on coarse meshes). Prefer [`energy`](Self::energy);
    /// this is kept as a diagnostic. `C` and `L` are recovered from the
    /// coefficients (`C = dt·(1 + ca)/(2·cb)`, likewise `L`; edges with
    /// `cb = 0` / `db = 0` are skipped). The sum runs in global index order,
    /// independent of the thread count.
    pub fn energy_staggered(&self) -> f64 {
        let co = read(&self.shared.coeffs);
        let st = &self.shared.store;
        let p = st.layout.plane;
        let mut w = 0.0f64;
        for (blocks, lane) in [(&st.u, 0), (&st.i, 6)] {
            for axis in 0..3 {
                let (a, b) = (&co.data[lane + axis], &co.data[lane + 3 + axis]);
                for (s, blk) in blocks.iter().enumerate() {
                    let off = st.layout.planes(s).0 * p;
                    let blk = read(blk);
                    let v = &blk[axis];
                    let a = &a[off..off + v.len()];
                    let b = &b[off..off + v.len()];
                    for ((&x, &ca), &cb) in v.iter().zip(a).zip(b) {
                        let c = Coefficients::capacitance(ca, cb, self.dt);
                        w += 0.5 * c * (x as f64) * (x as f64);
                    }
                }
            }
        }
        w
    }

    /// Advances one full step (H then E, with all operator hooks).
    pub fn step(&mut self) {
        self.run(1);
    }

    /// Advances one full step like [`step`](Self::step) and returns the
    /// energy invariant `W^n` \[J\] of the state *before* the step
    /// (`t = n·dt` with `n` the [`timestep`](Self::timestep) before the
    /// call), computed by the worker threads as part of the H update.
    ///
    /// Without operators this equals [`energy`](Self::energy) called just
    /// before the step, bit for bit (operator energies are taken before
    /// the step, as there). With operators, `i^{n+½}` is the H kernel result
    /// after the `pre_h` hooks and before the slab and `post_h` hooks.
    /// The extra cost is about one division per edge in that half-step;
    /// the result is bit-identical for any thread count.
    pub fn step_with_energy(&mut self) -> f64 {
        let w_ops = self.operator_energy();
        self.advance(1, true);
        self.shared.recorded_energy(self.dt) + w_ops
    }

    /// Advances `n` full steps.
    pub fn run(&mut self, n: u64) {
        self.advance(n, false);
    }

    /// `n` steps through the pool (the H update of the first one recording
    /// the energy if `energy_first`), with the serial hooks in the gaps
    /// that some operator declared.
    fn advance(&mut self, n: u64, energy_first: bool) {
        let mask = self.ops.iter().fold(Hooks::NONE, |m, r| m | r.hooks);
        let plan = Plan {
            steps: n,
            step0: self.step,
            dt: self.dt,
            energy_first,
            gaps: Gaps {
                a: mask.intersects(Hooks::PRE_H | Hooks::POST_E)
                    || self.periodic.is_some()
                    || self.axis.is_some(),
                b: mask.intersects(Hooks::POST_H | Hooks::PRE_E),
            },
        };
        let Self {
            grid,
            dt,
            shared,
            ops,
            periodic,
            axis,
            pool,
            ..
        } = self;
        let post_e_ops = mask.contains(Hooks::POST_E);
        let mut hook = |g: Gap| {
            let (t, which): (u64, &[Hooks]) = match g {
                Gap::PreH(t) => (t, &[Hooks::PRE_H]),
                Gap::Mid(t) => (t, &[Hooks::POST_H, Hooks::PRE_E]),
                Gap::PostE(t) => (t, &[Hooks::POST_E]),
            };
            let co = read(&shared.coeffs);
            let ctx = StepContext {
                step: plan.step0 + t,
                dt: *dt,
                grid,
                coefficients: &co,
            };
            let mut f = shared.store.lock();
            // Periodic faces: merge the kernel result before the operators
            // see it, and fold in their corrections afterwards.
            let post_e = matches!(g, Gap::PostE(_));
            let periodic = periodic.as_ref().filter(|_| post_e);
            let axis = axis.as_ref().filter(|_| post_e);
            if let Some(p) = periodic {
                p.merge(&mut f, &co);
            }
            if let Some(a) = axis {
                a.merge(&mut f, &co);
            }
            for &w in which {
                for r in ops.iter_mut().filter(|r| r.hooks.contains(w)) {
                    match w {
                        Hooks::PRE_H => r.op.pre_h(&mut f, &ctx),
                        Hooks::POST_H => r.op.post_h(&mut f, &ctx),
                        Hooks::PRE_E => r.op.pre_e(&mut f, &ctx),
                        _ => r.op.post_e(&mut f, &ctx),
                    }
                }
            }
            if post_e_ops {
                if let Some(p) = periodic {
                    p.merge(&mut f, &co);
                }
                if let Some(a) = axis {
                    a.merge(&mut f, &co);
                }
            }
        };
        pool::drive(pool.as_mut(), shared, &plan, &mut hook);
        self.step += n;
    }
}
