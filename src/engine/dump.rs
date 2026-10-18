//! Field dump operators: time-domain snapshots and frequency-domain (running
//! DFT) fields over a box of nodes (design §3, "Dumps").
//!
//! # Overview
//!
//! * A [`DumpRegion`] is an inclusive box of primary-node indices (it may be
//!   degenerate along one to three axes: a plane, a line or a point) with
//!   an optional sub-sampling stride per axis.
//! * A [`FieldSampler`] turns the engine's integral-form unknowns of one
//!   [`DumpQuantity`] (E or H) into field values in V/m or A/m at the
//!   region's sample points (see [interpolation](#interpolation)), with
//!   the sample coordinates in metres.
//! * A [`TimeDump`] operator records snapshots on a [`Schedule`] and hands
//!   each one to a [`DumpSink`]: in memory ([`MemorySink`]), streamed to
//!   `.vtr` files plus a `.pvd` index ([`VtrSeries`]), or to any closure.
//! * A [`FreqDump`] operator accumulates a running DFT of every sample point
//!   and component at a list of frequencies.
//!
//! ```
//! use ems::engine::{DumpPosition, DumpQuantity, DumpRegion, Engine, FieldSampler, Grid,
//!     Schedule, TimeDump};
//!
//! let grid = Grid::uniform([8, 8, 8], [1e-3; 3])?;
//! let mut engine = Engine::builder(grid).threads(1).build()?;
//! // The z = 4 plane, every other node in x and y.
//! let region = DumpRegion::new([0, 0, 4], [8, 8, 4]).stride([2, 2, 1]);
//! let sampler = FieldSampler::new(engine.grid(), region, DumpQuantity::E, DumpPosition::Nodes)?;
//! assert_eq!(sampler.dims(), [5, 5, 1]);
//! let h = engine.add_operator(TimeDump::new(sampler).schedule(Schedule::Every(10)));
//! engine.run(20);
//! let steps: Vec<u64> = engine.operator(&h).sink().snapshots().iter().map(|s| s.step).collect();
//! assert_eq!(steps, [0, 10, 20]);
//! # Ok::<(), ems::Error>(())
//! ```
//!
//! # Units
//!
//! The engine stores `u = ∫E·dl` along primary edges \[V\] and `i = ∫H·dl`
//! along dual edges \[A\] ([module docs](super)). A dump divides each by its
//! edge length: `E = u/Δ` with `Δ` the primary edge length
//! ([`Grid::delta`]) and `H = i/Δ*` with `Δ*` the dual edge length
//! ([`Grid::dual_delta`], half a cell on the outer nodes). This is the
//! average of the field along the edge, which is second-order accurate at
//! the edge midpoint. On a cylindrical grid ([`Grid::cylindrical`]) the
//! components are `(E_ρ, E_α, E_z)` (likewise H), the α edge lengths are
//! `ρ·Δα` ([`Grid::edge_length`]), and the sample coordinates are
//! `(ρ, α, z)` (metres, radians, metres); a VTK file then shows the
//! `(ρ, α, z)` parameter box, not the physical cylinder.
//!
//! # Interpolation
//!
//! Every value is a separable (tensor-product) interpolation of one field
//! component, built from 1-D rules per axis. Write `Δ_j` for the size of
//! cell `j` along the axis in question. For component `a` and axis `b`:
//!
//! | quantity | `b` | [`DumpPosition::Nodes`] (node `j`) | [`DumpPosition::CellCenters`] (cell `j`) |
//! |---|---|---|---|
//! | E | `b = a` | linear interpolation of the two edges `j−1`, `j` (midpoints): weights `Δ_j/(Δ_{j−1}+Δ_j)` and `Δ_{j−1}/(Δ_{j−1}+Δ_j)` | edge `j` itself |
//! | E | `b ≠ a` | edge on line `j` itself | mean of lines `j` and `j+1` |
//! | H | `b = a` | dual edge through node `j` itself | mean of dual edges `j` and `j+1` |
//! | H | `b ≠ a` | linear interpolation of cells `j−1`, `j` (midpoints): weights `Δ_j/(Δ_{j−1}+Δ_j)` and `Δ_{j−1}/(Δ_{j−1}+Δ_j)` | cell `j` itself |
//!
//! So at the nodes, E is interpolated along its own axis and H across its
//! two transverse axes (the four surrounding dual edges, bilinearly); at
//! cell centres E is the mean of the four parallel edges of the cell and H
//! the mean of the two dual edges through the opposite faces. All rules
//! are exact for fields that are linear along each axis, on any
//! non-uniform grid (the midpoint rules are exact because cell centres are
//! exactly half-way between node lines).
//!
//! **Boundaries.** At an outer node (index 0 or `n−1`) the linear rules
//! have only one neighbour inside the domain and use it alone
//! (constant extrapolation from half a cell away): E along its axis takes
//! the single adjacent edge, H across a transverse axis the single
//! adjacent cell. No value outside the domain is ever read.
//!
//! With [`DumpPosition::CellCenters`], an axis on which the region is
//! degenerate (`lo = hi`) has no cell; it is sampled at the node `lo`
//! with the node rule. Along other axes the samples are the cells
//! `lo, lo+s, … < hi` (so `hi − lo` cells at stride 1).
//!
//! # Sample layout
//!
//! A sampler has [`dims`](FieldSampler::dims) `[dx, dy, dz]` points with
//! coordinates [`coords`](FieldSampler::coords). Data vectors hold one
//! `[x, y, z]` tuple per point at index `(p·dy + q)·dz + r` (z fastest,
//! [`Layout::ZFastest`]), like the engine arrays. VTR output always uses
//! **point data** on a rectilinear grid whose lines are the sample
//! coordinates (node lines, or cell midpoints for
//! [`DumpPosition::CellCenters`]).
//!
//! # Time stamps
//!
//! A dump samples the state *after `m` completed steps* (engine
//! [`timestep`](super::Engine::timestep) `= m`), in which E is at `m·dt` and
//! H at `(m − ½)·dt` (leapfrog; [module docs](super#time-staggering-leapfrog)).
//! Each [`Snapshot`] records `step = m` and the time of *its* quantity
//! ([`FieldSampler::time`]); frequency-domain dumps use the same instants in
//! the DFT kernel.
//!
//! The state `m + 1` is sampled in the `post_e` hook of step `m`, i.e. after
//! every operator registered earlier has finished the step (register dumps
//! **after** sources, boundaries and lumped elements). The very first state
//! an operator sees (usually `m = 0`, the initial conditions) is sampled in
//! the `pre_h` hook of the first step; [`FieldSampler::snapshot`] samples
//! any state on demand (e.g. the final one).
//!
//! # Threads
//!
//! Sampling runs on the stepping thread in a fixed order (f64 arithmetic),
//! so dumps are bit-identical for any engine thread count.

use core::f64::consts::TAU;
use std::path::{Path, PathBuf};

use super::Engine;
use super::fields::{FieldArrays, Fields};
use super::grid::Grid;
use super::operator::{Operator, StepContext};
use crate::axis::Axis;
use crate::io::{ComplexForm, Field, Layout, PvdCollection, VtrWriter};
use crate::post::Complex64;
use crate::post::dft::phase_cycles;
use crate::{Error, Result};

/// An inclusive box of primary-node indices with a sub-sampling stride.
///
/// Along axis `a` the sampled node indices are `lo[a], lo[a] + s[a], …`
/// up to and including `hi[a]` (`hi[a]` itself is sampled only if
/// `hi[a] − lo[a]` is a multiple of `s[a]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DumpRegion {
    lo: [usize; 3],
    hi: [usize; 3],
    stride: [usize; 3],
}

impl DumpRegion {
    /// Box between the nodes `a` and `b` (inclusive; corners in any order),
    /// stride 1.
    pub fn new(a: [usize; 3], b: [usize; 3]) -> Self {
        Self {
            lo: [0, 1, 2].map(|i| a[i].min(b[i])),
            hi: [0, 1, 2].map(|i| a[i].max(b[i])),
            stride: [1; 3],
        }
    }

    /// The whole grid.
    pub fn whole(grid: &Grid) -> Self {
        Self::new([0; 3], grid.nodes().map(|n| n - 1))
    }

    /// Sub-sampling stride per axis (each ≥ 1; 0 is treated as 1).
    pub fn stride(mut self, stride: [usize; 3]) -> Self {
        self.stride = stride.map(|s| s.max(1));
        self
    }

    /// Lower corner (node indices).
    pub fn lo(&self) -> [usize; 3] {
        self.lo
    }

    /// Upper corner (node indices, inclusive).
    pub fn hi(&self) -> [usize; 3] {
        self.hi
    }

    /// The stride per axis.
    pub fn strides(&self) -> [usize; 3] {
        self.stride
    }

    /// Fails with [`Error::Invalid`] unless the box lies inside a grid of
    /// `nodes` nodes.
    pub fn check_fits(&self, nodes: [usize; 3]) -> Result<()> {
        for a in Axis::ALL {
            let i = a.index();
            if self.hi[i] >= nodes[i] {
                return Err(Error::Invalid(format!(
                    "dump box {:?}..={:?} exceeds the grid ({} nodes along {a})",
                    self.lo, self.hi, nodes[i]
                )));
            }
        }
        Ok(())
    }
}

/// The field a dump records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DumpQuantity {
    /// Electric field E \[V/m\], at `m·dt`.
    E,
    /// Magnetic field H \[A/m\], at `(m − ½)·dt`.
    H,
}

impl DumpQuantity {
    /// Array name used in VTR files: `"E"` or `"H"`.
    pub fn name(self) -> &'static str {
        match self {
            DumpQuantity::E => "E",
            DumpQuantity::H => "H",
        }
    }

    /// Time offset of the quantity relative to `m·dt`, in steps (0 or −½).
    fn offset(self) -> f64 {
        match self {
            DumpQuantity::E => 0.0,
            DumpQuantity::H => -0.5,
        }
    }
}

/// Where a dump places its sample points; see the
/// [module docs](self#interpolation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DumpPosition {
    /// At the primary nodes of the region (default).
    #[default]
    Nodes,
    /// At the centres of the primary cells inside the region (at the node
    /// along degenerate axes).
    CellCenters,
}

/// One 1-D interpolation rule: up to two `(index, weight)` taps (unused
/// taps have weight 0).
type Taps = [(usize, f64); 2];

/// A sample position along one axis.
#[derive(Debug, Clone, Copy)]
enum Pos {
    Node(usize),
    Mid(usize),
}

fn taps(grid: &Grid, q: DumpQuantity, comp: Axis, axis: Axis, pos: Pos) -> Taps {
    let n = grid.nodes()[axis.index()];
    let d = |i: usize| grid.delta(axis, i);
    let one = |i: usize, w: f64| [(i, w), (i, 0.0)];
    // Linear interpolation from the midpoints of cells j−1, j to node j,
    // clamped at the outer nodes.
    let lin = |j: usize| -> [(usize, f64); 2] {
        if j == 0 {
            one(0, 1.0)
        } else if j == n - 1 {
            one(n - 2, 1.0)
        } else {
            let (dl, dh) = (d(j - 1), d(j));
            [(j - 1, dh / (dl + dh)), (j, dl / (dl + dh))]
        }
    };
    let mut t = match (q, comp == axis, pos) {
        (DumpQuantity::E, true, Pos::Node(j)) => {
            let [(a, wa), (b, wb)] = lin(j);
            [(a, wa / d(a)), (b, wb / d(b))]
        }
        (DumpQuantity::E, true, Pos::Mid(j)) => one(j, 1.0 / d(j)),
        (DumpQuantity::E, false, Pos::Node(j)) => one(j, 1.0),
        (DumpQuantity::E, false, Pos::Mid(j)) => [(j, 0.5), (j + 1, 0.5)],
        (DumpQuantity::H, true, Pos::Node(j)) => one(j, 1.0 / grid.dual_delta(axis, j)),
        (DumpQuantity::H, true, Pos::Mid(j)) => [
            (j, 0.5 / grid.dual_delta(axis, j)),
            (j + 1, 0.5 / grid.dual_delta(axis, j + 1)),
        ],
        (DumpQuantity::H, false, Pos::Node(j)) => lin(j),
        (DumpQuantity::H, false, Pos::Mid(j)) => one(j, 1.0),
    };
    // Cylindrical grids: the α edges have length ρ·Δα (E on the ρ lines,
    // H on the dual lines ρ_{i+½}); the α rule above divides by Δα only, so
    // the ρ taps divide by ρ. On the axis α edges do not exist: their
    // weight goes to the other tap (0 if there is none).
    if grid.is_cylindrical() && comp == Axis::Y && axis == Axis::X {
        let r = grid.lines(Axis::X);
        let rho = |i: usize| match q {
            DumpQuantity::E => r[i],
            DumpQuantity::H => grid.dual_line(Axis::X, i.min(n - 2)),
        };
        let total: f64 = t.iter().map(|x| x.1).sum();
        let valid: f64 = t.iter().filter(|x| rho(x.0) > 0.0).map(|x| x.1).sum();
        for x in &mut t {
            let p = rho(x.0);
            x.1 = if p > 0.0 && valid != 0.0 {
                x.1 * (total / valid) / p
            } else {
                0.0
            };
        }
    }
    t
}

/// Converts and interpolates one field over a [`DumpRegion`]; see the
/// [module documentation](self) for units, interpolation and layout.
///
/// A sampler is tied to the grid it was built for.
#[derive(Debug, Clone)]
pub struct FieldSampler {
    region: DumpRegion,
    quantity: DumpQuantity,
    position: DumpPosition,
    coords: [Vec<f64>; 3],
    /// `taps[component][axis][sample index along axis]`.
    taps: [[Vec<Taps>; 3]; 3],
}

impl FieldSampler {
    /// Sampler of `quantity` over `region` of `grid`, at `position`.
    /// Fails with [`Error::Invalid`] if the region does not fit the grid.
    pub fn new(
        grid: &Grid,
        region: DumpRegion,
        quantity: DumpQuantity,
        position: DumpPosition,
    ) -> Result<Self> {
        region.check_fits(grid.nodes())?;
        let pos = Axis::ALL.map(|a| {
            let i = a.index();
            let (lo, hi, s) = (region.lo[i], region.hi[i], region.stride[i]);
            match position {
                DumpPosition::CellCenters if hi > lo => {
                    (lo..hi).step_by(s).map(Pos::Mid).collect::<Vec<_>>()
                }
                _ => (lo..=hi).step_by(s).map(Pos::Node).collect(),
            }
        });
        let coords = Axis::ALL.map(|a| {
            pos[a.index()]
                .iter()
                .map(|&p| match p {
                    Pos::Node(j) => grid.lines(a)[j],
                    Pos::Mid(j) => grid.dual_line(a, j),
                })
                .collect()
        });
        let taps = Axis::ALL.map(|c| {
            Axis::ALL.map(|a| {
                pos[a.index()]
                    .iter()
                    .map(|&p| taps(grid, quantity, c, a, p))
                    .collect()
            })
        });
        Ok(Self {
            region,
            quantity,
            position,
            coords,
            taps,
        })
    }

    /// The region.
    pub fn region(&self) -> &DumpRegion {
        &self.region
    }

    /// The sampled quantity.
    pub fn quantity(&self) -> DumpQuantity {
        self.quantity
    }

    /// The sample placement.
    pub fn position(&self) -> DumpPosition {
        self.position
    }

    /// Number of sample points per axis.
    pub fn dims(&self) -> [usize; 3] {
        [0, 1, 2].map(|i| self.coords[i].len())
    }

    /// Total number of sample points.
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Always `false` (a region holds at least one point).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Coordinates \[m\] of the sample points along `axis`.
    pub fn coords(&self, axis: Axis) -> &[f64] {
        &self.coords[axis.index()]
    }

    /// Time \[s\] of this quantity in the state after `step` completed steps:
    /// `step·dt` for E, `(step − ½)·dt` for H.
    pub fn time(&self, step: u64, dt: f64) -> f64 {
        (step as f64 + self.quantity.offset()) * dt
    }

    /// Calls `f(index, [vx, vy, vz])` for every sample point, in data order
    /// (z fastest), with the values in f64.
    pub fn sample_with(&self, fields: &Fields<'_>, mut f: impl FnMut(usize, [f64; 3])) {
        let arr: &FieldArrays<'_> = match self.quantity {
            DumpQuantity::E => &fields.u,
            DumpQuantity::H => &fields.i,
        };
        let [dx, dy, dz] = self.dims();
        let mut idx = 0;
        for p in 0..dx {
            for q in 0..dy {
                for r in 0..dz {
                    let mut v = [0.0; 3];
                    for c in Axis::ALL {
                        let t = &self.taps[c.index()];
                        let mut s = 0.0;
                        for &(i, wx) in &t[0][p] {
                            if wx == 0.0 {
                                continue;
                            }
                            for &(j, wy) in &t[1][q] {
                                if wy == 0.0 {
                                    continue;
                                }
                                for &(k, wz) in &t[2][r] {
                                    if wz != 0.0 {
                                        s += wx * wy * wz * f64::from(arr.get(c, [i, j, k]));
                                    }
                                }
                            }
                        }
                        v[c.index()] = s;
                    }
                    f(idx, v);
                    idx += 1;
                }
            }
        }
    }

    /// Samples `fields` (see [`sample_with`](Self::sample_with)) into a
    /// vector of `f32` tuples.
    pub fn sample(&self, fields: &Fields<'_>) -> Vec<[f32; 3]> {
        let mut out = vec![[0.0f32; 3]; self.len()];
        self.sample_with(fields, |i, v| out[i] = v.map(|x| x as f32));
        out
    }

    /// Samples the current state of `engine` (step
    /// [`Engine::timestep`]); `self` must have been built for its grid.
    ///
    /// # Panics
    /// If the region does not fit the engine's grid.
    pub fn snapshot(&self, engine: &Engine) -> Snapshot {
        let fields = engine.shared.store.lock();
        let step = engine.timestep();
        Snapshot {
            step,
            time: self.time(step, engine.dt()),
            data: self.sample(&fields),
        }
    }

    /// The VTR writer for `data` (one tuple per sample point) at `time`,
    /// with the quantity's name as point-vector array.
    fn vtr<'a>(&'a self, data: &'a [[f32; 3]], time: f64) -> VtrWriter<'a> {
        VtrWriter::new(&self.coords[0], &self.coords[1], &self.coords[2])
            .time(time)
            .point_vector(self.quantity.name(), data, Layout::ZFastest)
    }

    /// Writes `snapshot` as a `.vtr` file at `path`: point data named
    /// `"E"` or `"H"` on the sample coordinates, `TimeValue` = its time.
    pub fn write_vtr(&self, path: impl AsRef<Path>, snapshot: &Snapshot) -> Result<()> {
        self.vtr(&snapshot.data, snapshot.time).write(path)
    }
}

/// One time-domain sample of a dump.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    /// Number of completed engine steps `m` of the sampled state.
    pub step: u64,
    /// Time of the sampled quantity \[s\]: `m·dt` (E) or `(m − ½)·dt` (H).
    pub time: f64,
    /// One `[x, y, z]` value per sample point, z fastest (V/m or A/m).
    pub data: Vec<[f32; 3]>,
}

/// When a [`TimeDump`] records a snapshot, in terms of the state index `m`
/// (completed steps).
#[derive(Debug, Clone, PartialEq)]
pub enum Schedule {
    /// Every state with `m % k == 0` (`k ≥ 1`; 0 is treated as 1).
    Every(u64),
    /// The listed states only (any order, duplicates ignored).
    Steps(Vec<u64>),
    /// The first state at or after each multiple of the interval \[s\] of
    /// the E time `m·dt` (the same steps for E and H dumps). Non-positive
    /// or non-finite intervals sample every step.
    Interval(f64),
}

/// A recording window on the sampled quantity's own time `t`
/// ([`FieldSampler::time`]): `start ≤ t ≤ stop`, with `stop ≤ 0` meaning
/// "no upper limit" (the csx `StartTime` / `StopTime` convention), both
/// with a tolerance of `1e-9·dt` against rounding of the sample times.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Window {
    start: f64,
    stop: f64,
}

impl Default for Window {
    fn default() -> Self {
        Self {
            start: f64::NEG_INFINITY,
            stop: 0.0,
        }
    }
}

impl Window {
    fn contains(&self, t: f64, dt: f64) -> bool {
        let eps = 1e-9 * dt;
        t >= self.start - eps && (self.stop <= 0.0 || t <= self.stop + eps)
    }
}

/// Schedule with its running state.
#[derive(Debug, Clone)]
struct Scheduler {
    schedule: Schedule,
    next: f64,
}

impl Scheduler {
    fn new(mut schedule: Schedule) -> Self {
        if let Schedule::Steps(v) = &mut schedule {
            v.sort_unstable();
            v.dedup();
        }
        Self {
            schedule,
            next: f64::NEG_INFINITY,
        }
    }

    fn due(&mut self, m: u64, dt: f64) -> bool {
        match &self.schedule {
            Schedule::Every(k) => m.is_multiple_of((*k).max(1)),
            Schedule::Steps(v) => v.binary_search(&m).is_ok(),
            Schedule::Interval(t) => {
                let t = *t;
                if !(t > 0.0 && t.is_finite()) {
                    return true;
                }
                let now = m as f64 * dt;
                // Tolerate rounding of m·dt just below a multiple of t.
                if now < self.next - 1e-9 * dt {
                    return false;
                }
                self.next = ((now / t + 1e-9).floor() + 1.0) * t;
                true
            }
        }
    }
}

/// Receives the snapshots of a [`TimeDump`] as they are taken.
///
/// Implemented by [`MemorySink`], [`VtrSeries`] and by closures
/// `FnMut(&FieldSampler, Snapshot) -> Result<()>`. An error stops the dump
/// (no further snapshots are taken); it is kept in [`TimeDump::error`].
pub trait DumpSink: Send + 'static {
    /// Handles one snapshot.
    fn snapshot(&mut self, sampler: &FieldSampler, snapshot: Snapshot) -> Result<()>;
}

impl<F> DumpSink for F
where
    F: FnMut(&FieldSampler, Snapshot) -> Result<()> + Send + 'static,
{
    fn snapshot(&mut self, sampler: &FieldSampler, snapshot: Snapshot) -> Result<()> {
        self(sampler, snapshot)
    }
}

/// Keeps every snapshot in memory (`4·3·points` bytes per snapshot).
#[derive(Debug, Clone, Default)]
pub struct MemorySink {
    snapshots: Vec<Snapshot>,
}

impl MemorySink {
    /// An empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// The snapshots taken so far, in time order.
    pub fn snapshots(&self) -> &[Snapshot] {
        &self.snapshots
    }

    /// Removes and returns the snapshots taken so far.
    pub fn take(&mut self) -> Vec<Snapshot> {
        core::mem::take(&mut self.snapshots)
    }
}

impl DumpSink for MemorySink {
    fn snapshot(&mut self, _: &FieldSampler, snapshot: Snapshot) -> Result<()> {
        self.snapshots.push(snapshot);
        Ok(())
    }
}

/// File name of the snapshot at `step`: `<basename>_<step, 8 digits>.vtr`.
fn snapshot_file(basename: &str, step: u64) -> String {
    format!("{basename}_{step:08}.vtr")
}

/// Streams snapshots to `<dir>/<basename>_<step>.vtr` (step zero-padded to
/// 8 digits) and keeps `<dir>/<basename>.pvd` up to date (rewritten after
/// every snapshot, so it is valid even if the run stops early). The
/// directory is created on the first snapshot.
#[derive(Debug, Clone)]
pub struct VtrSeries {
    dir: PathBuf,
    basename: String,
    pvd: PvdCollection,
}

impl VtrSeries {
    /// Series in `dir` with file names starting with `basename`.
    pub fn new(dir: impl Into<PathBuf>, basename: impl Into<String>) -> Self {
        Self {
            dir: dir.into(),
            basename: basename.into(),
            pvd: PvdCollection::new(),
        }
    }

    /// The output directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Path of the `.pvd` index.
    pub fn pvd_path(&self) -> PathBuf {
        self.dir.join(format!("{}.pvd", self.basename))
    }

    /// The files written so far (relative to [`dir`](Self::dir)) with their
    /// times.
    pub fn collection(&self) -> &PvdCollection {
        &self.pvd
    }
}

impl DumpSink for VtrSeries {
    fn snapshot(&mut self, sampler: &FieldSampler, snapshot: Snapshot) -> Result<()> {
        if self.pvd.entries().is_empty() {
            std::fs::create_dir_all(&self.dir)?;
        }
        let file = snapshot_file(&self.basename, snapshot.step);
        sampler.write_vtr(self.dir.join(&file), &snapshot)?;
        self.pvd.push(snapshot.time, file);
        self.pvd.write(self.pvd_path())
    }
}

/// Tracks which state an operator samples next (shared by both dumps).
#[derive(Debug, Clone, Copy, Default)]
struct Cursor {
    started: bool,
}

impl Cursor {
    /// State index to sample in `pre_h`, if this is the first hook call.
    fn pre_h(&mut self, ctx: &StepContext<'_>) -> Option<u64> {
        (!core::mem::replace(&mut self.started, true)).then_some(ctx.step)
    }

    /// State index to sample in `post_e`.
    fn post_e(&mut self, ctx: &StepContext<'_>) -> u64 {
        self.started = true;
        ctx.step + 1
    }
}

/// Time-domain field dump operator: samples a [`FieldSampler`] on a
/// [`Schedule`] and passes each [`Snapshot`] to a [`DumpSink`]. See the
/// [module docs](self#time-stamps) for when states are sampled.
///
/// The default sink is a [`MemorySink`]; read it back with
/// [`Engine::operator`]`(&handle).`[`sink`](Self::sink)`()`.
#[derive(Debug)]
pub struct TimeDump<S = MemorySink> {
    sampler: FieldSampler,
    window: Window,
    scheduler: Scheduler,
    sink: S,
    cursor: Cursor,
    error: Option<Error>,
}

impl TimeDump<MemorySink> {
    /// In-memory dump of every state (change with
    /// [`schedule`](Self::schedule)).
    pub fn new(sampler: FieldSampler) -> Self {
        Self::with_sink(sampler, MemorySink::new())
    }

    /// Writes all recorded snapshots as `<dir>/<basename>_<step>.vtr` plus
    /// `<dir>/<basename>.pvd` (creating `dir`), like [`VtrSeries`].
    pub fn write_vtr(&self, dir: impl AsRef<Path>, basename: &str) -> Result<()> {
        let mut series = VtrSeries::new(dir.as_ref(), basename);
        for s in self.sink.snapshots() {
            series.snapshot(&self.sampler, s.clone())?;
        }
        Ok(())
    }
}

impl<S: DumpSink> TimeDump<S> {
    /// Dump of every state into `sink`.
    pub fn with_sink(sampler: FieldSampler, sink: S) -> Self {
        Self {
            sampler,
            window: Window::default(),
            scheduler: Scheduler::new(Schedule::Every(1)),
            sink,
            cursor: Cursor::default(),
            error: None,
        }
    }

    /// Sets the schedule (default: every step).
    pub fn schedule(mut self, schedule: Schedule) -> Self {
        self.scheduler = Scheduler::new(schedule);
        self
    }

    /// Records only states whose quantity time `t` ([`FieldSampler::time`])
    /// satisfies `start ≤ t ≤ stop` \[s\]; `stop ≤ 0` means no upper limit
    /// (the csx `StartTime` / `StopTime` convention). The schedule is
    /// evaluated only inside the window. Default: no window.
    pub fn window(mut self, start: f64, stop: f64) -> Self {
        self.window = Window { start, stop };
        self
    }

    /// The sampler.
    pub fn sampler(&self) -> &FieldSampler {
        &self.sampler
    }

    /// The sink.
    pub fn sink(&self) -> &S {
        &self.sink
    }

    /// Mutable access to the sink (e.g. [`MemorySink::take`]).
    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    /// The first sink error, after which the dump stopped.
    pub fn error(&self) -> Option<&Error> {
        self.error.as_ref()
    }

    fn sample(&mut self, fields: &Fields<'_>, m: u64, dt: f64) {
        if self.error.is_some()
            || !self.window.contains(self.sampler.time(m, dt), dt)
            || !self.scheduler.due(m, dt)
        {
            return;
        }
        let snap = Snapshot {
            step: m,
            time: self.sampler.time(m, dt),
            data: self.sampler.sample(fields),
        };
        if let Err(e) = self.sink.snapshot(&self.sampler, snap) {
            self.error = Some(e);
        }
    }
}

impl<S: DumpSink> Operator for TimeDump<S> {
    fn pre_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(m) = self.cursor.pre_h(ctx) {
            self.sample(fields, m, ctx.dt);
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        let m = self.cursor.post_e(ctx);
        self.sample(fields, m, ctx.dt);
    }
}

/// Frequency-domain field dump operator: a running DFT of every sample
/// point and component at a list of frequencies.
///
/// With states `m` sampled every `K` steps ([`interval`](Self::interval),
/// `m % K == 0`), the result at frequency `f` is
///
/// ```text
/// X(f) = Σ_m x(t_m) · e^{-j2πf·t_m} · K·dt
/// ```
///
/// with `t_m` the quantity's own time (`m·dt` for E, `(m − ½)·dt` for H),
/// the convention of [`mod@crate::post::dft`]. E and H transforms therefore
/// share one time origin and need no half-step phase correction. Sampling
/// every `K` steps aliases frequencies above `1/(2K·dt)`; keep `K` small
/// enough. Like [`RunningDft`](crate::post::RunningDft), each sample's
/// phasors are computed from the exactly reduced absolute phase, so there
/// is no drift with the run length.
///
/// No normalisation is applied: to get a transfer function, divide by the
/// excitation spectrum (e.g. [`Waveform::spectrum`](crate::Waveform::spectrum))
/// at each frequency.
///
/// # Memory
///
/// Accumulators are `f64` complex: `48` bytes per sample point per
/// frequency (three components × 16 bytes), e.g. 48 MB for a
/// 100 × 100 × 100 node box at 1 frequency. Use a plane or a stride for
/// many frequencies. The cost per sample is one interpolation per point
/// plus three complex multiply-adds per point and frequency.
#[derive(Debug, Clone)]
pub struct FreqDump {
    sampler: FieldSampler,
    freqs: Vec<f64>,
    interval: u64,
    window: Window,
    acc: Vec<[Complex64; 3]>,
    samples: u64,
    dt: f64,
    cursor: Cursor,
    phasors: Vec<Complex64>,
}

impl FreqDump {
    /// DFT dump of `sampler`'s quantity at `freqs` \[Hz\], sampling every
    /// step.
    pub fn new(sampler: FieldSampler, freqs: impl Into<Vec<f64>>) -> Self {
        let freqs = freqs.into();
        let n = sampler.len() * freqs.len();
        Self {
            sampler,
            phasors: vec![Complex64::ZERO; freqs.len()],
            freqs,
            interval: 1,
            window: Window::default(),
            acc: vec![[Complex64::ZERO; 3]; n],
            samples: 0,
            dt: 0.0,
            cursor: Cursor::default(),
        }
    }

    /// Samples only the states with `m % k == 0` (`k ≥ 1`; 0 is treated as
    /// 1). Default 1.
    pub fn interval(mut self, k: u64) -> Self {
        self.interval = k.max(1);
        self
    }

    /// Accumulates only states whose quantity time `t`
    /// ([`FieldSampler::time`]) satisfies `start ≤ t ≤ stop` \[s\];
    /// `stop ≤ 0` means no upper limit (see [`TimeDump::window`]).
    /// Default: no window.
    pub fn window(mut self, start: f64, stop: f64) -> Self {
        self.window = Window { start, stop };
        self
    }

    /// The sampler.
    pub fn sampler(&self) -> &FieldSampler {
        &self.sampler
    }

    /// The analysis frequencies \[Hz\].
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// Number of states accumulated so far.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Sampling interval `K·dt` \[s\] (0 before the first step).
    pub fn sample_interval(&self) -> f64 {
        self.interval as f64 * self.dt
    }

    /// The transform at frequency index `k`: one complex `[x, y, z]` per
    /// sample point, z fastest (V/m·s or A/m·s).
    ///
    /// # Panics
    /// If `k` is out of range.
    pub fn values(&self, k: usize) -> Vec<[Complex64; 3]> {
        assert!(k < self.freqs.len(), "frequency index {k} out of range");
        let n = self.sampler.len();
        let s = self.sample_interval();
        self.acc[k * n..(k + 1) * n]
            .iter()
            .map(|v| v.map(|c| c * s))
            .collect()
    }

    /// Adds `j·` the transform of `imag`, the same dump of the imaginary
    /// engine of a Bloch pair, so that this dump holds the transform of
    /// the complex field (`F_re + j·F_im`).
    pub(crate) fn add_imaginary(&mut self, imag: &FreqDump) {
        debug_assert_eq!(self.acc.len(), imag.acc.len());
        for (a, b) in self.acc.iter_mut().zip(&imag.acc) {
            for c in 0..3 {
                a[c] += Complex64::I * b[c];
            }
        }
    }

    /// Clears the accumulators (the next sampled state starts a new
    /// transform).
    pub fn reset(&mut self) {
        self.acc.fill([Complex64::ZERO; 3]);
        self.samples = 0;
    }

    /// Writes one `<dir>/<basename>_f<k>.vtr` per frequency index `k`
    /// (creating `dir`) with point arrays `<Q>_re` and `<Q>_im` (`Q` = `E` or
    /// `H`), `TimeValue` and a `Frequency` field-data entry set to the
    /// frequency, plus `<dir>/<basename>.pvd` indexed by frequency.
    pub fn write_vtr(&self, dir: impl AsRef<Path>, basename: &str) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let s = &self.sampler;
        let mut pvd = PvdCollection::new();
        for (k, &f) in self.freqs.iter().enumerate() {
            let v = self.values(k);
            let re: Vec<[f64; 3]> = v.iter().map(|c| c.map(|z| z.re)).collect();
            let im: Vec<[f64; 3]> = v.iter().map(|c| c.map(|z| z.im)).collect();
            let fa = [f];
            let file = format!("{basename}_f{k}.vtr");
            VtrWriter::new(&s.coords[0], &s.coords[1], &s.coords[2])
                .time(f)
                .field("Frequency", &fa)
                .point_complex(
                    s.quantity.name(),
                    Field::Vector(&re),
                    Field::Vector(&im),
                    Layout::ZFastest,
                    ComplexForm::ReIm,
                )
                .write(dir.join(&file))?;
            pvd.push(f, file);
        }
        pvd.write(dir.join(format!("{basename}.pvd")))
    }

    fn sample(&mut self, fields: &Fields<'_>, m: u64, dt: f64) {
        self.dt = dt;
        if !m.is_multiple_of(self.interval) || !self.window.contains(self.sampler.time(m, dt), dt) {
            return;
        }
        let t0 = self.sampler.quantity.offset() * dt;
        for (p, &f) in self.phasors.iter_mut().zip(&self.freqs) {
            *p = Complex64::cis(-TAU * phase_cycles(f, t0, dt, m as f64));
        }
        let n = self.sampler.len();
        let (acc, phasors) = (&mut self.acc, &self.phasors);
        self.sampler.sample_with(fields, |i, v| {
            for (k, &p) in phasors.iter().enumerate() {
                let a = &mut acc[k * n + i];
                for (a, &x) in a.iter_mut().zip(&v) {
                    *a += p * x;
                }
            }
        });
        self.samples += 1;
    }
}

impl Operator for FreqDump {
    fn pre_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(m) = self.cursor.pre_h(ctx) {
            self.sample(fields, m, ctx.dt);
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        let m = self.cursor.post_e(ctx);
        self.sample(fields, m, ctx.dt);
    }
}
