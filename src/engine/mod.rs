//! FDTD engine core: mesh, fields, coefficients, kernels, thread pool and
//! operator hooks.
//!
//! The engine is independent of [`csx`]: it is built from a [`Grid`], a
//! [`CellMaterials`] map, face conditions and a [`TimeStep`]. The csx →
//! engine compile step produces exactly these inputs.
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
//! behind separate `RwLock`s, so this is expressed in safe Rust. Without
//! operators a step costs one barrier per half-step; with operators, two
//! (the operators run on the calling thread in between). Every value is
//! computed by the same expression regardless of the partition and there
//! are no cross-thread reductions, so results are bit-identical for any
//! thread count. [`Engine::energy`] sums in global index order.

pub mod boundary;
mod coeffs;
mod excite;
mod fields;
mod grid;
mod kernel;
pub mod lumped;
mod material;
mod operator;
mod pool;
pub mod port;
mod probe;

use core::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub use coeffs::{
    Coefficients, FaceCondition, FaceConditions, TimeStep, courant_limit, dual_edge_lr,
    primary_edge_cg,
};
pub use excite::{Amplitude, EdgeSource};
pub use fields::{FieldArrays, Fields};
pub use grid::Grid;
pub use lumped::{LumpedBox, LumpedElement, Topology, fold_parallel};
pub use material::{CellMaterials, Material};
pub use operator::{Operator, OperatorHandle, StepContext};
pub use port::LumpedPort;
pub use probe::{Probe, ProbeQuantity};

use crate::axis::Axis;
use crate::fdtd::Face;
use crate::{Error, Result};
use fields::{FieldStore, Layout, read, write};
use pool::{Pool, Shared};

/// Configures and builds an [`Engine`]; see [`Engine::builder`].
#[derive(Debug, Clone)]
pub struct EngineBuilder {
    grid: Grid,
    materials: Option<CellMaterials>,
    faces: FaceConditions,
    timestep: TimeStep,
    threads: usize,
}

impl EngineBuilder {
    /// Material map (default: vacuum, no PEC edges).
    pub fn materials(mut self, m: CellMaterials) -> Self {
        self.materials = Some(m);
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

    /// Number of threads, including the caller; `0` (default) means
    /// [`std::thread::available_parallelism`]. `1` runs everything inline.
    /// Capped at `nx` (one x-plane per thread at least).
    pub fn threads(mut self, n: usize) -> Self {
        self.threads = n;
        self
    }

    /// Computes the time step and coefficients, allocates the fields and
    /// starts the worker threads.
    pub fn build(self) -> Result<Engine> {
        let grid = self.grid;
        let mats = self
            .materials
            .unwrap_or_else(|| CellMaterials::vacuum(&grid));
        let dt = self.timestep.resolve(&grid)?;
        let coeffs = Coefficients::compute(&grid, &mats, &self.faces, dt)?;
        let threads = match self.threads {
            0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
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
            pool,
        })
    }
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
    ops: Vec<Box<dyn Operator>>,
    pool: Option<Pool>,
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
    pub fn add_operator<T: Operator>(&mut self, op: T) -> OperatorHandle<T> {
        self.ops.push(Box::new(op));
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
        let op: &dyn core::any::Any = &*self.ops[self.op_index(h)];
        op.downcast_ref().expect("operator type matches its handle")
    }

    /// Mutable access to the operator behind `h`.
    ///
    /// # Panics
    /// If `h` was returned by a different engine.
    pub fn operator_mut<T: Operator>(&mut self, h: &OperatorHandle<T>) -> &mut T {
        let i = self.op_index(h);
        let op: &mut dyn core::any::Any = &mut *self.ops[i];
        op.downcast_mut().expect("operator type matches its handle")
    }

    /// Read access to the update coefficients.
    pub fn coefficients(&self) -> impl core::ops::Deref<Target = Coefficients> + '_ {
        read(&self.shared.coeffs)
    }

    /// Mutable access to the update coefficients, for boundary and
    /// lumped-element setup between steps.
    pub fn coefficients_mut(&mut self) -> impl core::ops::DerefMut<Target = Coefficients> + '_ {
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

    /// Electromagnetic energy `W = ½·Σ C·u² + ½·Σ L·i²` \[J\], in `f64`.
    ///
    /// `C` and `L` are recovered from the coefficients
    /// (`C = dt·(1 + ca)/(2·cb)`, likewise `L`). `u` and `i` are half a step
    /// apart, so for a lossless closed cavity `W` is not exactly constant
    /// but bounded with a small `O((ω·dt)²)` ripple. The sum runs in global
    /// index order and is therefore independent of the thread count.
    pub fn energy(&self) -> f64 {
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

    fn hooks(&mut self, which: u8) {
        let co = read(&self.shared.coeffs);
        let ctx = StepContext {
            step: self.step,
            dt: self.dt,
            grid: &self.grid,
            coefficients: &co,
        };
        let mut f = self.shared.store.lock();
        for op in &mut self.ops {
            match which {
                0 => op.pre_h(&mut f, &ctx),
                1 => op.post_h(&mut f, &ctx),
                2 => op.pre_e(&mut f, &ctx),
                _ => op.post_e(&mut f, &ctx),
            }
        }
    }

    fn half(&mut self, e: bool) {
        match &mut self.pool {
            Some(p) => p.half(&self.shared, e),
            None if e => self.shared.e(0),
            None => self.shared.h(0),
        }
    }

    /// Advances one full step (H then E, with all operator hooks).
    pub fn step(&mut self) {
        self.run(1);
    }

    /// Advances `n` full steps.
    pub fn run(&mut self, n: u64) {
        if self.ops.is_empty() {
            match &mut self.pool {
                Some(p) => p.run(&self.shared, n),
                None => {
                    for _ in 0..n {
                        self.shared.h(0);
                        self.shared.e(0);
                    }
                }
            }
            self.step += n;
            return;
        }
        for _ in 0..n {
            self.hooks(0);
            self.half(false);
            self.hooks(1);
            self.hooks(2);
            self.half(true);
            self.hooks(3);
            self.step += 1;
        }
    }
}
