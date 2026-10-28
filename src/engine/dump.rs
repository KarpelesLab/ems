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
//! # Quantities
//!
//! Besides E and H a sampler derives, per edge, before interpolating:
//!
//! * `J = κ·E = G·u/A*` and `D = ε·E = C·u/A*`, with the edge capacitance
//!   `C` and conductance `G` recovered from the update coefficients
//!   (`C = dt·(1 + ca)/(2·cb)`, `G = (1 − ca)/cb`) and `A*` the dual face
//!   of the edge, so `ε` and `κ` are the effective (averaged) edge
//!   values the engine uses ([`FieldSampler::prepare`]);
//! * `B = μ·H = L·i/A` (`L` from `da`, `db`, `A` the primary face);
//! * `rot H = ∇×H` on the primary edges: the circulation of `i` around the
//!   edge divided by its dual face area (Stokes), i.e. the discrete curl
//!   the E update uses.
//!
//! J, D and rot H are interpolated with the E rules, B with the H rules.
//!
//! # openEMS rules
//!
//! [`FieldSampler::openems`] builds a sampler with the conventions of
//! openEMS dumps (see [`io::openems::dump`](crate::io::openems::dump)):
//! explicit sample indices; at the nodes H is the plain mean of the
//! surrounding dual edges (0 on the outer nodes); at cell centres H along
//! its own axis is interpolated between the midpoints of the two dual
//! edges, and rot H divides by dual faces whose outer edges count a full
//! cell; cells include the virtual one beyond the last node (value 0).
//! [`DumpPosition::Native`] reports raw edge values (`DumpMode` 0).
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
//! That is [`DumpTiming::State`], the default. With
//! [`DumpTiming::OpenEms`] (what [`Simulation`](crate::Simulation) uses
//! for model dumps) "step `n`" is sampled in the `post_h` hook of step
//! `n`: E at `n·dt`, H and B at `(n + ½)·dt`, like openEMS.
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
use super::operator::{Hooks, Operator, StepContext};
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
#[non_exhaustive]
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
///
/// E, J, D and rot H live on the primary (E) edges, H and B on the dual
/// (H) edges; see the [module docs](self#quantities) for their
/// definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DumpQuantity {
    /// Electric field E \[V/m\].
    E,
    /// Magnetic field H \[A/m\].
    H,
    /// Conduction current density `J = κ·E` \[A/m²\].
    J,
    /// Total current density `rot H = ∇×H` \[A/m²\].
    RotH,
    /// Electric flux density `D = ε·E` \[C/m²\].
    D,
    /// Magnetic flux density `B = μ·H` \[T\].
    B,
}

impl DumpQuantity {
    /// Array name used in ems VTR files: `"E"`, `"H"`, `"J"`, `"RotH"`,
    /// `"D"` or `"B"`.
    pub fn name(self) -> &'static str {
        match self {
            DumpQuantity::E => "E",
            DumpQuantity::H => "H",
            DumpQuantity::J => "J",
            DumpQuantity::RotH => "RotH",
            DumpQuantity::D => "D",
            DumpQuantity::B => "B",
        }
    }

    /// Whether the quantity lives on the dual (H) edges (H, B).
    pub fn is_dual(self) -> bool {
        matches!(self, DumpQuantity::H | DumpQuantity::B)
    }

    /// Whether the quantity is computed from the H currents (H, B, rot H)
    /// rather than the E voltages (E, J, D).
    pub fn from_currents(self) -> bool {
        matches!(self, DumpQuantity::H | DumpQuantity::B | DumpQuantity::RotH)
    }

    /// Whether sampling needs the update coefficients (J, D, B).
    fn needs_coefficients(self) -> bool {
        matches!(self, DumpQuantity::J | DumpQuantity::D | DumpQuantity::B)
    }
}

/// Where a dump places its sample points; see the
/// [module docs](self#interpolation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum DumpPosition {
    /// At the primary nodes of the region (default).
    #[default]
    Nodes,
    /// At the centres of the primary cells inside the region (at the node
    /// along degenerate axes).
    CellCenters,
    /// No interpolation (openEMS `DumpMode` 0): the raw value of the edge
    /// belonging to each node, i.e. the primary edge starting at the node
    /// for E, J, D and rot H (reported at the node) and the dual edge
    /// through the node for H and B (reported at the dual node, the
    /// centre of the cell starting at the node).
    Native,
}

/// Which state a dump samples at "step `n`"; see the
/// [module docs](self#time-stamps).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum DumpTiming {
    /// The state after `n` completed steps: E at `n·dt`, H at
    /// `(n − ½)·dt` (default).
    #[default]
    State,
    /// The openEMS convention: E at `n·dt` and H at `(n + ½)·dt`, i.e.
    /// the fields right after the H update of step `n` (state `n` for E,
    /// `n + 1` for H). The operators sample `n = 0 ..= N − 1` in a run of
    /// `N` steps; `n = N` needs a look-ahead H update
    /// ([`TimeDump::final_sample`], [`FreqDump::final_sample`]).
    OpenEms,
}

/// One 1-D interpolation rule: up to two `(index, weight)` taps (unused
/// taps have weight 0).
type Taps = [(usize, f64); 2];

/// A sample position along one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pos {
    /// Primary node `j`.
    Node(usize),
    /// Centre of cell `j`; `j = n − 1` is the virtual cell beyond the
    /// last node (zero field, see [`FieldSampler::from_indices`]).
    Mid(usize),
    /// Raw (not interpolated) value of node `j` ([`DumpPosition::Native`]).
    Raw(usize),
}

/// The coordinate of dual node `j` along `axis`, including the virtual
/// one beyond the last node (`l[n−1] + Δ_{n−2}/2`).
fn dual_coord(grid: &Grid, axis: Axis, j: usize) -> f64 {
    let l = grid.lines(axis);
    let n = l.len();
    if j + 1 < n {
        grid.dual_line(axis, j)
    } else if n >= 2 {
        l[n - 1] + 0.5 * (l[n - 1] - l[n - 2])
    } else {
        l[n - 1]
    }
}

/// The dual edge length of node `j` with full cells at the outer nodes
/// (openEMS interpolation and curl weights).
fn oe_dual(grid: &Grid, axis: Axis, j: usize) -> f64 {
    let n = grid.nodes()[axis.index()];
    let d = grid.dual_delta(axis, j);
    if n >= 2 && (j == 0 || j == n - 1) {
        2.0 * d
    } else {
        d
    }
}

fn taps(grid: &Grid, q: DumpQuantity, comp: Axis, axis: Axis, pos: Pos, oe: bool) -> Taps {
    let n = grid.nodes()[axis.index()];
    let d = |i: usize| grid.delta(axis, i);
    let one = |i: usize, w: f64| [(i, w), (i, 0.0)];
    let zero = [(0, 0.0), (0, 0.0)];
    // A grid with a single node along `axis`: only the node itself.
    if n < 2 {
        return match (q.is_dual(), comp == axis) {
            (false, true) => zero,
            _ => one(0, 1.0),
        };
    }
    // A closed azimuth: the α axis is periodic, its first and last nodes
    // the same half-plane between cells n − 2 and 0.
    let periodic = axis == Axis::Y && grid.is_closed();
    // Linear interpolation from the midpoints of cells j−1, j to node j,
    // clamped at the outer nodes (periodic on a closed azimuth).
    let lin = |j: usize| -> [(usize, f64); 2] {
        if periodic && (j == 0 || j == n - 1) {
            let (dl, dh) = (d(n - 2), d(0));
            [(n - 2, dh / (dl + dh)), (0, dl / (dl + dh))]
        } else if j == 0 {
            one(0, 1.0)
        } else if j == n - 1 {
            one(n - 2, 1.0)
        } else {
            let (dl, dh) = (d(j - 1), d(j));
            [(j - 1, dh / (dl + dh)), (j, dl / (dl + dh))]
        }
    };
    let virtual_cell = matches!(pos, Pos::Mid(j) if j + 1 >= n);
    if virtual_cell {
        return zero;
    }
    let mut t = match (q.is_dual(), comp == axis, pos) {
        (false, true, Pos::Node(j)) => {
            let [(a, wa), (b, wb)] = lin(j);
            [(a, wa / d(a)), (b, wb / d(b))]
        }
        (false, true, Pos::Mid(j)) => one(j, 1.0 / d(j)),
        (false, true, Pos::Raw(j)) => {
            if j + 1 < n {
                one(j, 1.0 / d(j))
            } else {
                zero
            }
        }
        (false, false, Pos::Node(j) | Pos::Raw(j)) => one(j, 1.0),
        (false, false, Pos::Mid(j)) => [(j, 0.5), (j + 1, 0.5)],
        (true, true, Pos::Node(j) | Pos::Raw(j)) => one(j, 1.0 / grid.dual_delta(axis, j)),
        (true, true, Pos::Mid(j)) if oe => {
            // Linear between the midpoints of the dual edges j, j+1, with
            // full cells as the outer dual edges.
            let (a, b) = (oe_dual(grid, axis, j), oe_dual(grid, axis, j + 1));
            [
                (j, b / (a + b) / grid.dual_delta(axis, j)),
                (j + 1, a / (a + b) / grid.dual_delta(axis, j + 1)),
            ]
        }
        (true, true, Pos::Mid(j)) => [
            (j, 0.5 / grid.dual_delta(axis, j)),
            (j + 1, 0.5 / grid.dual_delta(axis, j + 1)),
        ],
        (true, false, Pos::Node(j)) if oe => {
            // openEMS writes 0 on the outer nodes (on a closed azimuth it
            // interpolates across the seam).
            if periodic && (j == 0 || j == n - 1) {
                [(n - 2, 0.5), (0, 0.5)]
            } else if j == 0 || j == n - 1 {
                zero
            } else {
                [(j - 1, 0.5), (j, 0.5)]
            }
        }
        (true, false, Pos::Node(j)) => lin(j),
        (true, false, Pos::Mid(j) | Pos::Raw(j)) => one(j, 1.0),
    };
    // Cylindrical grids: the α edges have length ρ·Δα (E on the ρ lines,
    // H on the dual lines ρ_{i+½}); the α rule above divides by Δα only, so
    // the ρ taps divide by ρ. On the axis α edges do not exist: their
    // weight goes to the other tap (0 if there is none); openEMS averages
    // them as 0 (a cell centre in the first ring reads half the outer
    // edge, black box `dump_cyl_closed`).
    if grid.is_cylindrical() && comp == Axis::Y && axis == Axis::X {
        let r = grid.lines(Axis::X);
        let rho = |i: usize| {
            if q.is_dual() {
                grid.dual_line(Axis::X, i.min(n - 2))
            } else {
                r[i]
            }
        };
        let total: f64 = t.iter().map(|x| x.1).sum();
        let valid: f64 = t.iter().filter(|x| rho(x.0) > 0.0).map(|x| x.1).sum();
        for x in &mut t {
            let p = rho(x.0);
            let spread = if oe { 1.0 } else { total / valid };
            x.1 = if p > 0.0 && valid != 0.0 {
                x.1 * spread / p
            } else {
                0.0
            };
        }
    }
    t
}

/// Per-edge factors of the derived quantities over a box of nodes: the
/// value fed to the interpolation rules of E (or H) is `factor · u` (or
/// `factor · i`, or `factor · circulation` for rot H).
#[derive(Debug, Clone, PartialEq)]
struct Factors {
    lo: [usize; 3],
    dims: [usize; 3],
    /// One array per component, z fastest over `dims`.
    data: [Vec<f64>; 3],
}

impl Factors {
    #[inline]
    fn get(&self, c: usize, p: [usize; 3]) -> f64 {
        let [i, j, k] = [0, 1, 2].map(|a| p[a] - self.lo[a]);
        self.data[c][(i * self.dims[1] + j) * self.dims[2] + k]
    }
}

/// Converts and interpolates one field over a set of sample points; see
/// the [module documentation](self) for units, interpolation and layout.
///
/// A sampler is tied to the grid it was built for.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FieldSampler {
    region: DumpRegion,
    quantity: DumpQuantity,
    position: DumpPosition,
    timing: DumpTiming,
    coords: [Vec<f64>; 3],
    /// `taps[component][axis][sample index along axis]`.
    taps: [[Vec<Taps>; 3]; 3],
    /// Bounding box `[lo, hi]` (inclusive) of all tap nodes.
    tap_box: ([usize; 3], [usize; 3]),
    /// Factors of J, D, B (from the coefficients) and rot H (geometry).
    factors: Option<Factors>,
}

impl FieldSampler {
    /// Sampler of `quantity` over `region` of `grid`, at `position`.
    /// Fails with [`Error::Invalid`] if the region does not fit the grid.
    ///
    /// With [`DumpPosition::CellCenters`] the samples are the cells
    /// `lo, lo+s, … < hi` (the node `lo` along a degenerate axis); with
    /// [`DumpPosition::Nodes`] and [`DumpPosition::Native`] the nodes
    /// `lo, lo+s, … ≤ hi`.
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
                DumpPosition::Native => (lo..=hi).step_by(s).map(Pos::Raw).collect(),
                _ => (lo..=hi).step_by(s).map(Pos::Node).collect(),
            }
        });
        Self::build(grid, region, quantity, position, pos, false)
    }

    /// Sampler of `quantity` at explicit sample indices per axis (each
    /// list non-empty, strictly increasing), as openEMS dumps with
    /// sub-sampling or `OptResolution` select them:
    ///
    /// * [`DumpPosition::Nodes`], [`DumpPosition::Native`]: node indices
    ///   `0 ≤ j < n`;
    /// * [`DumpPosition::CellCenters`]: cell indices `0 ≤ j < n`, where
    ///   `j = n − 1` is the *virtual* cell beyond the last node (centre
    ///   `l[n−1] + Δ_{n−2}/2`, value 0), as openEMS writes it; there is
    ///   no special rule for degenerate axes.
    ///
    /// # Errors
    /// [`Error::Invalid`] for empty, unsorted or out-of-range lists.
    pub fn openems(
        grid: &Grid,
        indices: [Vec<usize>; 3],
        quantity: DumpQuantity,
        position: DumpPosition,
    ) -> Result<Self> {
        let n = grid.nodes();
        for a in Axis::ALL {
            let v = &indices[a.index()];
            if v.is_empty()
                || v.windows(2).any(|w| w[0] >= w[1])
                || v.last().is_some_and(|&j| j >= n[a.index()])
            {
                return Err(Error::Invalid(format!(
                    "dump sample indices along {a} must be increasing and below {}: {v:?}",
                    n[a.index()]
                )));
            }
        }
        let region = DumpRegion::new(
            [0, 1, 2].map(|i| indices[i][0]),
            [0, 1, 2].map(|i| *indices[i].last().expect("non-empty")),
        );
        let pos = [0, 1, 2].map(|i| {
            indices[i]
                .iter()
                .map(|&j| match position {
                    DumpPosition::Nodes => Pos::Node(j),
                    DumpPosition::CellCenters => Pos::Mid(j),
                    DumpPosition::Native => Pos::Raw(j),
                })
                .collect::<Vec<_>>()
        });
        Self::build(grid, region, quantity, position, pos, true)
    }

    fn build(
        grid: &Grid,
        region: DumpRegion,
        quantity: DumpQuantity,
        position: DumpPosition,
        pos: [Vec<Pos>; 3],
        openems: bool,
    ) -> Result<Self> {
        let coords = Axis::ALL.map(|a| {
            pos[a.index()]
                .iter()
                .map(|&p| match p {
                    Pos::Node(j) => grid.lines(a)[j],
                    Pos::Mid(j) => dual_coord(grid, a, j),
                    Pos::Raw(j) if quantity.is_dual() => dual_coord(grid, a, j),
                    Pos::Raw(j) => grid.lines(a)[j],
                })
                .collect()
        });
        let taps: [[Vec<Taps>; 3]; 3] = Axis::ALL.map(|c| {
            Axis::ALL.map(|a| {
                pos[a.index()]
                    .iter()
                    .map(|&p| taps(grid, quantity, c, a, p, openems))
                    .collect()
            })
        });
        let mut lo = [usize::MAX; 3];
        let mut hi = [0; 3];
        for comp in &taps {
            for (a, list) in comp.iter().enumerate() {
                for t in list {
                    for &(i, w) in t {
                        if w != 0.0 {
                            lo[a] = lo[a].min(i);
                            hi[a] = hi[a].max(i);
                        }
                    }
                }
            }
        }
        for a in 0..3 {
            if lo[a] > hi[a] {
                (lo[a], hi[a]) = (0, 0);
            }
        }
        let mut s = Self {
            region,
            quantity,
            position,
            timing: DumpTiming::State,
            coords,
            taps,
            tap_box: (lo, hi),
            factors: None,
        };
        if quantity == DumpQuantity::RotH {
            let n = grid.nodes();
            s.factors = Some(s.factors_with(|c, p| {
                if p[c.index()] + 1 >= n[c.index()] {
                    return 0.0;
                }
                let mut a = grid.dual_face_area(c, p);
                if openems {
                    // openEMS: full cells as the outer dual edges.
                    let (b, d) = c.others();
                    for t in [b, d] {
                        let (j, m) = (p[t.index()], n[t.index()]);
                        if m >= 2 && (j == 0 || j == m - 1) {
                            a *= 2.0;
                        }
                    }
                }
                if a > 0.0 {
                    grid.edge_length(c, p) / a
                } else {
                    0.0
                }
            }));
        }
        Ok(s)
    }

    fn factors_with(&self, f: impl Fn(Axis, [usize; 3]) -> f64) -> Factors {
        let (lo, hi) = self.tap_box;
        let dims = [0, 1, 2].map(|a| hi[a] - lo[a] + 1);
        let data = Axis::ALL.map(|c| {
            let mut v = Vec::with_capacity(dims.iter().product());
            for i in lo[0]..=hi[0] {
                for j in lo[1]..=hi[1] {
                    for k in lo[2]..=hi[2] {
                        v.push(f(c, [i, j, k]));
                    }
                }
            }
            v
        });
        Factors { lo, dims, data }
    }

    /// Sets which state "step `n`" samples (default
    /// [`DumpTiming::State`]); the dump operators follow it.
    #[must_use]
    pub fn timing(mut self, timing: DumpTiming) -> Self {
        self.timing = timing;
        self
    }

    /// The sampling convention.
    pub fn get_timing(&self) -> DumpTiming {
        self.timing
    }

    /// Takes the material factors of J (`κ`), D (`ε`) and B (`μ`) of each
    /// edge from the update coefficients: with `C`, `G` (`L`) recovered
    /// from `ca`, `cb` (`da`, `db`), `D = C·u/A*`, `J = G·u/A*` and
    /// `B = L·i/A` (`A*` the dual face of the edge, `A` the primary face
    /// of the dual edge; see [`Coefficients`](super::Coefficients)). PEC
    /// edges give 0. The dump operators call this on their first hook;
    /// call it yourself before [`sample_with`](Self::sample_with) on a
    /// J, D or B sampler. Nothing to do for the other quantities.
    pub fn prepare(&mut self, grid: &Grid, co: &super::Coefficients, dt: f64) {
        if !self.quantity.needs_coefficients() {
            return;
        }
        let n = grid.nodes();
        let q = self.quantity;
        let f = self.factors_with(|c, p| {
            let idx = grid.index(p);
            let a = c.index();
            match q {
                DumpQuantity::B => {
                    let (ob, oc) = c.others();
                    if p[ob.index()] + 1 >= n[ob.index()] || p[oc.index()] + 1 >= n[oc.index()] {
                        return 0.0;
                    }
                    let (da, db) = (co.da(c)[idx], co.db(c)[idx]);
                    let area = grid.face_area(c, p);
                    if db == 0.0 || area <= 0.0 {
                        return 0.0;
                    }
                    let l = dt * (1.0 + f64::from(da)) / (2.0 * f64::from(db));
                    l * grid.dual_edge_length(c, p) / area
                }
                _ => {
                    if p[a] + 1 >= n[a] {
                        return 0.0;
                    }
                    let (ca, cb) = (co.ca(c)[idx], co.cb(c)[idx]);
                    let area = grid.dual_face_area(c, p);
                    if cb == 0.0 || area <= 0.0 {
                        return 0.0;
                    }
                    let (ca, cb) = (f64::from(ca), f64::from(cb));
                    let v = if q == DumpQuantity::D {
                        dt * (1.0 + ca) / (2.0 * cb)
                    } else {
                        (1.0 - ca) / cb
                    };
                    v * grid.edge_length(c, p) / area
                }
            }
        });
        self.factors = Some(f);
    }

    fn is_prepared(&self) -> bool {
        !self.quantity.needs_coefficients() || self.factors.is_some()
    }

    /// The bounding region of the samples (for samplers built with
    /// [`openems`](Self::openems): the first and last index per
    /// axis, stride 1).
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

    /// Coordinates \[m\] of the sample points along `axis` (radians along
    /// α on a cylindrical grid).
    pub fn coords(&self, axis: Axis) -> &[f64] {
        &self.coords[axis.index()]
    }

    /// Time offset of the sampled quantity relative to `n·dt`, in steps.
    fn offset(&self) -> f64 {
        match (self.timing, self.quantity.from_currents()) {
            (_, false) => 0.0,
            (DumpTiming::State, true) => -0.5,
            // openEMS labels rot H (the curl of H at (n + ½)·dt) with the E
            // time n·dt, in the files and in the DFT phase.
            (DumpTiming::OpenEms, true) if self.quantity == DumpQuantity::RotH => 0.0,
            (DumpTiming::OpenEms, true) => 0.5,
        }
    }

    /// Time \[s\] of this quantity at "step `step`" of its
    /// [timing](DumpTiming): `step·dt` for E, J and D; `(step − ½)·dt`
    /// ([`DumpTiming::State`]) or `(step + ½)·dt`
    /// ([`DumpTiming::OpenEms`]) for H and B. Rot H is the curl of the H
    /// of that state, at `(step − ½)·dt` ([`DumpTiming::State`]); with
    /// [`DumpTiming::OpenEms`] it is labelled `step·dt` like openEMS
    /// does (the curl of H at `(step + ½)·dt`), which is also the phase
    /// reference of its frequency-domain dumps.
    pub fn time(&self, step: u64, dt: f64) -> f64 {
        (step as f64 + self.offset()) * dt
    }

    /// The circulation of `i` around the primary edge `c` at `p`
    /// (missing dual edges outside the domain count as 0).
    #[inline]
    fn circulation(get: &impl Fn(Axis, [usize; 3]) -> f64, c: Axis, p: [usize; 3]) -> f64 {
        let (a, b) = c.others();
        let (ai, bi) = (a.index(), b.index());
        let mut s = get(b, p) - get(a, p);
        if p[ai] > 0 {
            let mut q = p;
            q[ai] -= 1;
            s -= get(b, q);
        }
        if p[bi] > 0 {
            let mut q = p;
            q[bi] -= 1;
            s += get(a, q);
        }
        s
    }

    /// Calls `f(index, [vx, vy, vz])` for every sample point, in data order
    /// (z fastest), with the values in f64.
    ///
    /// # Panics
    /// For a J, D or B sampler that was not [prepared](Self::prepare).
    pub fn sample_with(&self, fields: &Fields<'_>, f: impl FnMut(usize, [f64; 3])) {
        let arr: &FieldArrays<'_> = if self.quantity.from_currents() {
            &fields.i
        } else {
            &fields.u
        };
        self.sample_from(|c, p| f64::from(arr.get(c, p)), f);
    }

    /// The sampling core over the unknowns `get(component, node)` (`u`
    /// or `i`, by quantity).
    fn sample_from(
        &self,
        get: impl Fn(Axis, [usize; 3]) -> f64,
        mut f: impl FnMut(usize, [f64; 3]),
    ) {
        assert!(
            self.is_prepared(),
            "FieldSampler::prepare must be called before sampling {}",
            self.quantity.name()
        );
        let roth = self.quantity == DumpQuantity::RotH;
        let fac = self.factors.as_ref();
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
                                    if wz == 0.0 {
                                        continue;
                                    }
                                    let node = [i, j, k];
                                    let x = if roth {
                                        Self::circulation(&get, c, node)
                                    } else {
                                        get(c, node)
                                    };
                                    let m = fac.map_or(1.0, |f| f.get(c.index(), node));
                                    s += wx * wy * wz * m * x;
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

    /// Samples the openEMS "step `N`" of an engine that has completed `N`
    /// steps ([`DumpTiming::OpenEms`]): E, J and D from the current state,
    /// H, B and rot H from `i^{N+½}`, the H update of step `N` evaluated
    /// locally from the current fields and coefficients (`i' = da·i −
    /// db·curl u`) without running the step. Operators acting on H
    /// (CPML, H sources) are not applied to this look-ahead; `None` on
    /// periodic and multigrid grids and on cylindrical grids with the
    /// axis or a closed azimuth, where the H update has extra terms.
    pub fn sample_lookahead(&self, engine: &Engine) -> Option<Vec<[f64; 3]>> {
        let mut s;
        let this = if self.is_prepared() {
            self
        } else {
            s = self.clone();
            s.prepare(engine.grid(), &engine.coefficients(), engine.dt());
            &s
        };
        let mut out = vec![[0.0f64; 3]; this.len()];
        if !this.quantity.from_currents() {
            let fields = engine.shared.store.lock();
            this.sample_with(&fields, |i, v| out[i] = v);
            return Some(out);
        }
        with_lookahead_currents(engine, |get| this.sample_from(get, |k, v| out[k] = v))?;
        Some(out)
    }

    /// Samples `fields` (see [`sample_with`](Self::sample_with)) into a
    /// vector of `f32` tuples.
    pub fn sample(&self, fields: &Fields<'_>) -> Vec<[f32; 3]> {
        let mut out = vec![[0.0f32; 3]; self.len()];
        self.sample_with(fields, |i, v| out[i] = v.map(|x| x as f32));
        out
    }

    /// Samples the current state of `engine` (step
    /// [`Engine::timestep`], time per [`DumpTiming::State`]); `self` must
    /// have been built for its grid. J, D and B samplers are prepared
    /// from the engine's coefficients on the fly.
    ///
    /// # Panics
    /// If the region does not fit the engine's grid.
    pub fn snapshot(&self, engine: &Engine) -> Snapshot {
        let mut s;
        let this = if self.is_prepared() {
            self
        } else {
            s = self.clone();
            s.prepare(engine.grid(), &engine.coefficients(), engine.dt());
            &s
        };
        let fields = engine.shared.store.lock();
        let step = engine.timestep();
        Snapshot {
            step,
            time: (step as f64
                + if this.quantity.from_currents() {
                    -0.5
                } else {
                    0.0
                })
                * engine.dt(),
            data: this.sample(&fields),
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
    /// after the quantity on the sample coordinates, `TimeValue` = its
    /// time.
    pub fn write_vtr(&self, path: impl AsRef<Path>, snapshot: &Snapshot) -> Result<()> {
        self.vtr(&snapshot.data, snapshot.time).write(path)
    }
}

/// One time-domain sample of a dump.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Snapshot {
    /// Number of completed engine steps `m` of the sampled state.
    pub step: u64,
    /// Time of the sampled quantity \[s\]: `m·dt` (E) or `(m − ½)·dt` (H).
    pub time: f64,
    /// One `[x, y, z]` value per sample point, z fastest (V/m or A/m).
    pub data: Vec<[f32; 3]>,
}

impl Snapshot {
    /// The values `data` sampled after `step` steps, at `time` \[s\].
    pub fn new(step: u64, time: f64, data: Vec<[f32; 3]>) -> Self {
        Self { step, time, data }
    }
}

/// When a [`TimeDump`] records a snapshot, in terms of the state index `m`
/// (completed steps).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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

/// A recording window: on the sampled quantity's own time `t`
/// ([`FieldSampler::time`]), `start ≤ t ≤ stop`, with `stop ≤ 0` meaning
/// "no upper limit" (the csx `StartTime` / `StopTime` convention), both
/// with a tolerance of `1e-9·dt` against rounding of the sample times; or
/// an inclusive range of steps.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Window {
    Time { start: f64, stop: f64 },
    Steps { first: u64, last: Option<u64> },
}

impl Default for Window {
    fn default() -> Self {
        Window::Time {
            start: f64::NEG_INFINITY,
            stop: 0.0,
        }
    }
}

impl Window {
    fn contains(&self, step: u64, t: f64, dt: f64) -> bool {
        match *self {
            Window::Time { start, stop } => {
                let eps = 1e-9 * dt;
                t >= start - eps && (stop <= 0.0 || t <= stop + eps)
            }
            Window::Steps { first, last } => step >= first && last.is_none_or(|l| step <= l),
        }
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
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// State index to sample in `pre_h`, if this is the first hook call
    /// ([`DumpTiming::State`] only).
    fn pre_h(&mut self, timing: DumpTiming, ctx: &StepContext<'_>) -> Option<u64> {
        if timing != DumpTiming::State {
            return None;
        }
        (!core::mem::replace(&mut self.started, true)).then_some(ctx.step)
    }

    /// Step index to sample in `post_h` ([`DumpTiming::OpenEms`] only).
    fn post_h(&mut self, timing: DumpTiming, ctx: &StepContext<'_>) -> Option<u64> {
        (timing == DumpTiming::OpenEms).then_some(ctx.step)
    }

    /// State index to sample in `post_e` ([`DumpTiming::State`] only).
    fn post_e(&mut self, timing: DumpTiming, ctx: &StepContext<'_>) -> Option<u64> {
        if timing != DumpTiming::State {
            return None;
        }
        self.started = true;
        Some(ctx.step + 1)
    }
}

/// Time-domain field dump operator: samples a [`FieldSampler`] on a
/// [`Schedule`] and passes each [`Snapshot`] to a [`DumpSink`]. See the
/// [module docs](self#time-stamps) for when states are sampled.
///
/// The default sink is a [`MemorySink`]; read it back with
/// [`Engine::operator`]`(&handle).`[`sink`](Self::sink)`()`.
#[derive(Debug)]
#[non_exhaustive]
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
        self.window = Window::Time { start, stop };
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

    fn sample(&mut self, fields: &Fields<'_>, m: u64, ctx: &StepContext<'_>) {
        let dt = ctx.dt;
        if self.error.is_some()
            || !self.window.contains(m, self.sampler.time(m, dt), dt)
            || !self.scheduler.due(m, dt)
        {
            return;
        }
        if !self.sampler.is_prepared() {
            self.sampler.prepare(ctx.grid, ctx.coefficients, dt);
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

    /// Records only the steps `first ..= last` (`None`: no upper limit),
    /// in the step numbering of the sampler's [timing](DumpTiming),
    /// instead of a time window.
    pub fn step_window(mut self, first: u64, last: Option<u64>) -> Self {
        self.window = Window::Steps { first, last };
        self
    }

    /// The last openEMS step of a run, like [`FreqDump::final_sample`];
    /// pass it to [`apply_final`](Self::apply_final).
    pub fn final_sample(&self, engine: &Engine) -> Option<FinalSample> {
        let m = engine.timestep();
        let dt = engine.dt();
        if self.error.is_some()
            || self.sampler.timing != DumpTiming::OpenEms
            || !self.window.contains(m, self.sampler.time(m, dt), dt)
            || !self.scheduler.clone().due(m, dt)
        {
            return None;
        }
        Some(FinalSample {
            step: m,
            dt,
            data: self.sampler.sample_lookahead(engine)?,
        })
    }

    /// Records a [`final_sample`](Self::final_sample) as a snapshot.
    pub fn apply_final(&mut self, s: FinalSample) {
        if self.error.is_some() {
            return;
        }
        let _ = self.scheduler.due(s.step, s.dt);
        let snap = Snapshot {
            step: s.step,
            time: self.sampler.time(s.step, s.dt),
            data: s.data.iter().map(|v| v.map(|x| x as f32)).collect(),
        };
        if let Err(e) = self.sink.snapshot(&self.sampler, snap) {
            self.error = Some(e);
        }
    }
}

impl<S: DumpSink> Operator for TimeDump<S> {
    fn pre_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(m) = self.cursor.pre_h(self.sampler.timing, ctx) {
            self.sample(fields, m, ctx);
        }
    }

    fn post_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(m) = self.cursor.post_h(self.sampler.timing, ctx) {
            self.sample(fields, m, ctx);
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(m) = self.cursor.post_e(self.sampler.timing, ctx) {
            self.sample(fields, m, ctx);
        }
    }

    fn hooks(&self) -> Hooks {
        match self.sampler.timing {
            DumpTiming::State => Hooks::PRE_H | Hooks::POST_E,
            DumpTiming::OpenEms => Hooks::POST_H,
        }
    }

    /// openEMS-timed dumps of the currents only read them in `post_h`.
    fn movable_hooks(&self) -> Hooks {
        match self.sampler.timing {
            DumpTiming::OpenEms if self.sampler.quantity().from_currents() => Hooks::POST_H,
            _ => Hooks::NONE,
        }
    }

    /// openEMS timing with a fixed step interval: only at the due steps.
    fn hooks_at(&self, ctx: &StepContext<'_>) -> Hooks {
        match (self.sampler.timing, &self.scheduler.schedule) {
            (DumpTiming::OpenEms, Schedule::Every(k)) if !ctx.step.is_multiple_of((*k).max(1)) => {
                Hooks::NONE
            }
            _ => self.hooks(),
        }
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
#[non_exhaustive]
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
        self.window = Window::Time { start, stop };
        self
    }

    /// Accumulates only the steps `first ..= last` (`None`: no upper
    /// limit) instead of a time window (see [`TimeDump::step_window`]).
    pub fn step_window(mut self, first: u64, last: Option<u64>) -> Self {
        self.window = Window::Steps { first, last };
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

    fn sample(&mut self, fields: &Fields<'_>, m: u64, ctx: &StepContext<'_>) {
        let dt = ctx.dt;
        self.dt = dt;
        if !m.is_multiple_of(self.interval)
            || !self.window.contains(m, self.sampler.time(m, dt), dt)
        {
            return;
        }
        if !self.sampler.is_prepared() {
            self.sampler.prepare(ctx.grid, ctx.coefficients, dt);
        }
        let t0 = self.sampler.offset() * dt;
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

    /// The last openEMS step of a run ([`DumpTiming::OpenEms`] only): the
    /// state after `N = engine.timestep()` completed steps with the H of
    /// step `N` from [`FieldSampler::sample_lookahead`], if the step is
    /// due; pass it to [`apply_final`](Self::apply_final). openEMS records
    /// this step too (it runs the H update of the last step); `None` if not
    /// due or not computable.
    pub fn final_sample(&self, engine: &Engine) -> Option<FinalSample> {
        let m = engine.timestep();
        let dt = engine.dt();
        if self.sampler.timing != DumpTiming::OpenEms
            || !m.is_multiple_of(self.interval)
            || !self.window.contains(m, self.sampler.time(m, dt), dt)
        {
            return None;
        }
        Some(FinalSample {
            step: m,
            dt,
            data: self.sampler.sample_lookahead(engine)?,
        })
    }

    /// Accumulates a [`final_sample`](Self::final_sample).
    pub fn apply_final(&mut self, s: FinalSample) {
        let (m, dt) = (s.step, s.dt);
        self.dt = dt;
        let t0 = self.sampler.offset() * dt;
        for (p, &f) in self.phasors.iter_mut().zip(&self.freqs) {
            *p = Complex64::cis(-TAU * phase_cycles(f, t0, dt, m as f64));
        }
        let n = self.sampler.len();
        for (i, v) in s.data.iter().enumerate() {
            for (k, &p) in self.phasors.iter().enumerate() {
                let a = &mut self.acc[k * n + i];
                for (a, &x) in a.iter_mut().zip(v) {
                    *a += p * x;
                }
            }
        }
        self.samples += 1;
    }
}

/// The look-ahead sample of the last openEMS step of a run, see
/// [`FreqDump::final_sample`] and [`TimeDump::final_sample`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FinalSample {
    /// The openEMS step `N` (completed engine steps).
    pub step: u64,
    /// The time step \[s\].
    pub dt: f64,
    /// One value per sample point (z fastest).
    pub data: Vec<[f64; 3]>,
}

impl Operator for FreqDump {
    fn pre_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(m) = self.cursor.pre_h(self.sampler.timing, ctx) {
            self.sample(fields, m, ctx);
        }
    }

    fn post_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(m) = self.cursor.post_h(self.sampler.timing, ctx) {
            self.sample(fields, m, ctx);
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(m) = self.cursor.post_e(self.sampler.timing, ctx) {
            self.sample(fields, m, ctx);
        }
    }

    fn hooks(&self) -> Hooks {
        match self.sampler.timing {
            DumpTiming::State => Hooks::PRE_H | Hooks::POST_E,
            DumpTiming::OpenEms => Hooks::POST_H,
        }
    }

    /// openEMS-timed dumps of the currents only read them in `post_h`.
    fn movable_hooks(&self) -> Hooks {
        match self.sampler.timing {
            DumpTiming::OpenEms if self.sampler.quantity().from_currents() => Hooks::POST_H,
            _ => Hooks::NONE,
        }
    }

    /// openEMS timing: only at the sampled steps.
    fn hooks_at(&self, ctx: &StepContext<'_>) -> Hooks {
        if self.sampler.timing == DumpTiming::OpenEms && !ctx.step.is_multiple_of(self.interval) {
            Hooks::NONE
        } else {
            self.hooks()
        }
    }
}

/// Runs `f` with the dual-edge currents `i^{N+½}` of the H update of
/// step `N` of an engine that has completed `N` steps, evaluated locally
/// from the current fields and coefficients (`i' = da·i − db·curl u`)
/// without running the step (see [`FieldSampler::sample_lookahead`]);
/// `None` on multigrid grids (zone projection) and on periodic Cartesian
/// grids. (The axis and the closed azimuth of a cylindrical grid only
/// merge the E copies, so their H update is the plain one.)
/// Operators that complete the H update of their edges themselves
/// ([`Operator::lookahead_current`](super::Operator::lookahead_current):
/// openEMS's PML) are applied, then the H sources
/// ([`Operator::lookahead_sources`](super::Operator::lookahead_sources));
/// other operators acting on H (CPML) are not.
pub(crate) fn with_lookahead_currents<R>(
    engine: &Engine,
    f: impl FnOnce(&dyn Fn(Axis, [usize; 3]) -> f64) -> R,
) -> Option<R> {
    let g = engine.grid();
    // The axis and the seam of a cylindrical grid merge E copies only;
    // their H update is the plain one.
    if g.has_multigrid() || (engine.periodic.is_some() && !g.is_cylindrical()) {
        return None;
    }
    let oe_curl = engine.curl_order() == super::CurlOrder::OpenEms;
    let co = engine.coefficients();
    let fields = engine.shared.store.lock();
    let n = g.nodes();
    let (u, i) = (&fields.u, &fields.i);
    let owners: Vec<&dyn super::Operator> = engine
        .ops
        .iter()
        .map(|r| &*r.op)
        .filter(|o| o.has_lookahead_currents())
        .collect();
    // u_c at p + â (0 outside the domain).
    let up = |c: Axis, mut p: [usize; 3], a: Axis| -> f32 {
        p[a.index()] += 1;
        if p[a.index()] < n[a.index()] {
            u.get(c, p)
        } else {
            0.0
        }
    };
    // H sources of the next step, applied after the update as the hooks
    // do (registration order).
    let ctx = super::StepContext {
        step: engine.timestep(),
        dt: engine.dt(),
        grid: g,
        coefficients: &co,
    };
    type SourceMap = std::collections::HashMap<(Axis, [usize; 3]), Vec<(f32, bool)>>;
    let mut sources = SourceMap::new();
    for r in &engine.ops {
        for (axis, node, v, hard) in r.op.lookahead_sources(&ctx) {
            sources.entry((axis, node)).or_default().push((v, hard));
        }
    }
    let get = |c: Axis, p: [usize; 3]| -> f64 {
        let (a, b) = c.others();
        let idx = g.index(p);
        // The kernel's expression and summation order, bit for bit.
        let (da, db) = (co.da(c)[idx], co.db(c)[idx]);
        let (t0, t1) = (up(b, p, a), u.get(b, p));
        let (t2, t3) = (up(a, p, b), u.get(a, p));
        let k = if oe_curl {
            super::kernel::oe_h(da, i.get(c, p), db, t0, t1, t2, t3)
        } else {
            super::kernel::flush(da * i.get(c, p) - db * ((t0 - t1) - (t2 - t3)))
        };
        let mut v = owners
            .iter()
            .find_map(|o| o.lookahead_current(c, p, k))
            .unwrap_or(k);
        if let Some(list) = sources.get(&(c, p)) {
            for &(s, hard) in list {
                v = if hard { s } else { v + s };
            }
        }
        f64::from(v)
    };
    Some(f(&get))
}
