//! The time step from the cells of the grid (design §5.4.1).
//!
//! # The bound
//!
//! The lossless scheme `C·du/dt = Kᵀi`, `L·di/dt = −K·u`, stepped by
//! leapfrog, is stable iff `dt²·λ_max(C⁻¹·Kᵀ·L⁻¹·K) ≤ 4`. Both the
//! curl-curl operator and the capacitance are sums over the cells (an
//! edge gives each cell around it the part of its capacitance that lies
//! in the cell, a face gives each of its two cells half of its dual
//! edge), so the largest eigenvalue of the grid is at most the largest of
//! its cells' own (the element bound of Irons and Treharne, 1971; see
//! [`conformal`](super::conformal#small-faces-the-time-step-allows)):
//!
//! ```text
//! λ_max ≤ max_c λ_c,     dt ≤ min_c 2/√λ_c
//! ```
//!
//! A cell of sizes `Δx, Δy, Δz` whose twelve edges have the permittivity
//! `ε` and whose six faces the permeability `μ` has
//! `λ_c/4 = (1/Δx² + 1/Δy² + 1/Δz²)/(ε·μ)` exactly, the Courant limit of
//! that cell in that medium. With different values on its edges and faces
//! `λ_c` is at most that of the least `ε` and the least `μ` among them
//! (the quotient `uᵀAu / uᵀCu` only falls when a capacitance or an
//! inductance grows), and an edge held at zero (PEC) only takes vectors
//! away. So
//!
//! ```text
//! dt_cells = min over the cells of  √(ε_min·μ_min) / √(1/Δx² + 1/Δy² + 1/Δz²)
//! ```
//!
//! is a time step at which the lossless scheme is stable, by proof. It is
//! never shorter than [`courant_limit`](super::courant_limit) for
//! `ε_r, μ_r ≥ 1` (which takes the smallest cell size of each axis
//! wherever it is, at the vacuum speed), equal to it on a uniform mesh in
//! vacuum, and longer where the smallest sizes of the three axes are not
//! in one cell or lie in a dielectric: on a board, whose thinnest cells
//! are in the laminate.
//!
//! The closed form is an upper bound of `λ_c`, and what is taken is
//! `λ_c` itself where it matters: the largest eigenvalue of the cell's
//! own problem of six faces and twelve edges, each edge with its own
//! permittivity, the edges on a PEC wall of the domain and the PEC edges
//! closed ([`cell_limit`] and the compile step solve the cells whose
//! closed form is above the largest eigenvalue found, each distinct cell
//! once: 6 to 12 ms on board regions of 67 000 to 131 000 cells). A cell
//! against a wall or a ground plane, or between air and a laminate, has
//! a lower `λ_c` than its closed form: on cadlab's board regions, whose
//! smallest cells are at the walls, the step of the cells is 1.00 to
//! 1.31 times the Courant step, of which the closed form alone finds
//! 1.00 to 1.06.
//!
//! The bound is not the limit. A mode of the grid cannot live in one
//! cell, and the power iteration on the same operators
//! ([`stability_factor`](super::conformal::stability_factor), an estimate
//! from below, converged in 1000 products) reads 1.29 to 1.45 times the
//! Courant limit on those regions. A step from that estimate would be a
//! step from a measurement, so it is not taken; a bound over patches of
//! cells in the place of single ones (the same theorem on any partition
//! of the cells) would close part of the difference and is not made.
//!
//! # What the other parts of the scheme add
//!
//! The bound is on the lossless core. Every other part is either inside
//! it, or passive whatever the step, or outside this bound (and then the
//! compile step keeps the Courant step,
//! [`StepReport`](crate::compile::StepReport)):
//!
//! * *Conductivity* (`κ`, `σ_m`) is integrated semi-implicitly: the
//!   energy `W = ½·Σ C·u² + ½·Σ L·i^{n−½}·i^{n+½}` falls by the loss, and
//!   is positive under the same condition.
//! * *Debye poles, and Lorentz and Drude poles on the trapezoidal rule*
//!   ([`dispersive`](super::dispersive)): each pole is a series R–L–C
//!   branch across its edge, coupled implicitly; `W` plus the branches'
//!   energy does not grow, so the condition is the one of the medium of
//!   `ε∞`, which is what the edge's capacitance holds. (The leapfrog
//!   scheme of Lorentz and Drude poles has a limit of its own: not here.)
//! * *Lumped R, L, C and ports* ([`lumped`](super::lumped)): R and C are
//!   folded into the edge (more capacitance, loss); L and series branches
//!   are trapezoidal branches as the poles. No limit.
//! * *Conducting sheets* ([`sheet`](super::sheet),
//!   [`surface`](super::surface)): the edge is split into its two face
//!   voltages `u±` with the capacitances `C±` (the shares
//!   [`SheetEdge::upper_share`](super::SheetEdge::upper_share) of the
//!   edge's), joined by a ladder or a rational impedance on the
//!   trapezoidal rule (passive at any step). The faces above read `u+`,
//!   those below `u−`, and a face in the sheet's plane reads
//!   `f+·u+ + f−·u−` with the cells' shares `f±` of its dual edge; since
//!   `(f+·a + f−·b)² ≤ f+·a² + f−·b²`, the split system is bounded by the
//!   cells above with `C+` and the cells below with `C−`. The bound takes
//!   a sheet edge with `C+/f+` in the cells above it and `C−/f−` in the
//!   cells below (the medium of that side). An edge the sheet's outline
//!   cuts is its open part in series with the covered part's faces: at
//!   least the capacitance of the whole edge.
//! * *Lossy metal* ([`surface`](super::surface)): an edge in the
//!   surface is an ordinary edge with the capacitance of the part of its
//!   dual face outside the metal, at least vacuum's on the quarters
//!   beside a face with field, and a passive branch. The bound takes such
//!   an edge (PEC in the materials) with that capacitance over the
//!   quarters in cells that are not all metal, at least `ε_r = 1`; a cell
//!   all of whose edges are PEC has no field and is left out.
//! * *Conformal PEC* ([`conformal`](super::conformal)): a cut cell enters
//!   the same bound with its open lengths and areas; the compile step
//!   takes the cut cells into the step (below).
//! * *CPML* ([`boundary::cpml`](super::boundary::cpml)): no energy proof.
//!   With frozen coefficients the layer's stretch
//!   `s(z) = 1/κ + a/(1 − b/z)` maps `|z| ≥ 1` into the disc on
//!   `[α/(σ + κα), (1 − σ/(σ + κα)·tanh(x/2))/κ] ⊂ [0, 1]`, and the
//!   characteristic polynomial of a plane wave,
//!   `(z − 1)²(z − b)² + 4z·(A·(c₁z − b/κ)² + B·(z − b)²)`, has no root
//!   outside the unit circle for `A + B ≤ 1` (the Courant number squared
//!   of the unstretched cell) over `κ` 1 to 15, `σ·dt/ε0` 0 to 100 and
//!   `α·dt/ε0` 0 to 5 (a unit test scans it). The graded layer is tested
//!   by long runs.
//!
//! The f32 fields round every step; a rounding error is a source, not a
//! growth rate, so it does not move the limit, and the factor below 1
//! (`Fdtd::timestep_factor`) is the margin for it.

use std::collections::HashMap;

use super::coeffs::{FaceCondition, FaceConditions};
use super::{EdgeMaterials, Grid};
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};
use crate::fdtd::Face;

/// What an edge the materials do not describe by themselves counts as in
/// the bound ([`cell_limit_with`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LiveEdge {
    /// An edge in the surface of lossy metal: PEC in the materials, an
    /// ordinary edge in the scheme, whose capacitance in a cell outside
    /// the metal is at least that of this relative permittivity (1:
    /// vacuum, the least the scheme gives it).
    Solid {
        /// The relative permittivity.
        eps: f64,
    },
    /// A conducting-sheet edge with normal `normal`: its relative
    /// permittivity is multiplied by `above` in the cells above the sheet
    /// (higher index along the normal) and by `below` in those below.
    Sheet {
        /// The sheet's normal.
        normal: Axis,
        /// `C+/(f+·C)`.
        above: f64,
        /// `C−/(f−·C)`.
        below: f64,
    },
}

/// The edges of [`LiveEdge`], by axis and node index.
#[derive(Debug, Clone, Default)]
pub(crate) struct LiveEdges {
    /// Whether the edge along the axis at the node index has an entry.
    flagged: [Vec<bool>; 3],
    edges: HashMap<(usize, usize), LiveEdge>,
}

impl LiveEdges {
    /// No such edge on a grid of `nodes` nodes.
    pub(crate) fn new(nodes: usize) -> Self {
        Self {
            flagged: core::array::from_fn(|_| vec![false; nodes]),
            edges: HashMap::new(),
        }
    }

    /// Declares the edge along `axis` at node index `k`. A sheet on an
    /// edge of lossy metal stays the metal's.
    pub(crate) fn set(&mut self, axis: Axis, k: usize, e: LiveEdge) {
        let a = axis.index();
        if self.flagged[a][k]
            && matches!(self.edges.get(&(a, k)), Some(LiveEdge::Solid { .. }))
            && !matches!(e, LiveEdge::Solid { .. })
        {
            return;
        }
        self.flagged[a][k] = true;
        self.edges.insert((a, k), e);
    }

    /// Whether there is none.
    pub(crate) fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    #[inline]
    fn get(&self, a: usize, k: usize) -> Option<LiveEdge> {
        if self.flagged[a].get(k).copied().unwrap_or(false) {
            self.edges.get(&(a, k)).copied()
        } else {
            None
        }
    }
}

/// The grid and materials as the cells see them.
pub(crate) struct Cells<'a> {
    grid: &'a Grid,
    mats: &'a EdgeMaterials,
    live: Option<&'a LiveEdges>,
    deltas: [Vec<f64>; 3],
    n: [usize; 3],
}

impl<'a> Cells<'a> {
    /// `None` on a cylindrical grid.
    pub(crate) fn new(
        grid: &'a Grid,
        mats: &'a EdgeMaterials,
        live: Option<&'a LiveEdges>,
    ) -> Option<Self> {
        if grid.is_cylindrical() || mats.nodes() != grid.nodes() {
            return None;
        }
        Some(Self {
            grid,
            mats,
            live: live.filter(|l| !l.is_empty()),
            deltas: Axis::ALL.map(|a| grid.deltas(a)),
            n: grid.nodes(),
        })
    }

    /// The relative permittivity of the edge along `d` at `p` as the cell
    /// `q` has it; `None` for an edge held at zero.
    #[inline]
    fn edge_eps(&self, d: usize, p: [usize; 3], q: [usize; 3]) -> Option<f64> {
        let k = self.grid.index(p);
        let axis = Axis::ALL[d];
        let live = self.live.and_then(|l| l.get(d, k));
        match live {
            Some(LiveEdge::Solid { eps }) => Some(eps),
            _ if self.mats.pec_mask(axis)[k] => None,
            Some(LiveEdge::Sheet {
                normal,
                above,
                below,
            }) => {
                let f = if q[normal.index()] == p[normal.index()] {
                    above
                } else {
                    below
                };
                Some(self.mats.eps_r(axis)[k] * f)
            }
            None => Some(self.mats.eps_r(axis)[k]),
        }
    }

    /// The nodes of the four edges along `d` of the cell `q`.
    #[inline]
    fn edge_nodes(d: usize, q: [usize; 3]) -> [[usize; 3]; 4] {
        let (b, c) = ((d + 1) % 3, (d + 2) % 3);
        let mut out = [q; 4];
        out[1][b] += 1;
        out[2][c] += 1;
        out[3][b] += 1;
        out[3][c] += 1;
        out
    }

    /// `Σ 1/Δ²` of the cell `q`.
    #[inline]
    fn inverse_squares(&self, q: [usize; 3]) -> f64 {
        (0..3)
            .map(|a| {
                let d = self.deltas[a][q[a]];
                1.0 / (d * d)
            })
            .sum()
    }

    /// Whether the cell `q` has an edge that is not PEC in the materials:
    /// a cell without one lies inside metal (or is closed by it), has no
    /// field, and nothing of an edge in the metal's surface lies in it.
    #[inline]
    pub(crate) fn is_open(&self, q: [usize; 3]) -> bool {
        (0..3).any(|d| {
            let mask = self.mats.pec_mask(Axis::ALL[d]);
            Self::edge_nodes(d, q)
                .iter()
                .any(|&p| !mask[self.grid.index(p)])
        })
    }

    /// The least relative permittivity among the cell's edges that are
    /// not held at zero and the least relative permeability among its
    /// faces; `None` for a cell with no such edge (nothing to bound).
    fn least(&self, q: [usize; 3]) -> Option<(f64, f64)> {
        if self.live.is_some() && !self.is_open(q) {
            return None;
        }
        let mut eps = f64::INFINITY;
        for d in 0..3 {
            for p in Self::edge_nodes(d, q) {
                if let Some(e) = self.edge_eps(d, p, q) {
                    eps = eps.min(e);
                }
            }
        }
        if eps == f64::INFINITY {
            return None;
        }
        let mut mu = f64::INFINITY;
        for d in 0..3 {
            let lane = self.mats.mu_r(Axis::ALL[d]);
            for o in 0..2 {
                let mut p = q;
                p[d] += o;
                mu = mu.min(lane[self.grid.index(p)]);
            }
        }
        Some((eps, mu))
    }

    /// `λ_c/4` \[1/s²\] of the cell with lowest node `q`, at most: the
    /// closed form of the [module docs](self) (exact where the cell's
    /// edges have one permittivity and its faces one permeability). The
    /// cell is stable at `dt` if `dt²` times this is at most 1.
    pub(crate) fn value(&self, q: [usize; 3]) -> f64 {
        match self.least(q) {
            Some((eps, mu)) if eps > 0.0 && mu > 0.0 => {
                self.inverse_squares(q) / (EPS0 * MU0 * eps * mu)
            }
            Some(_) => f64::INFINITY,
            None => 0.0,
        }
    }

    /// `λ_c/4` \[1/s²\] of the cell with lowest node `q`, by its own
    /// eigenvalue problem: every edge with its capacitance in the cell,
    /// every face with its half of the dual edge. `edge` gives the factor
    /// of an edge's capacitance (the open length of a cut edge: `1/f`;
    /// `None`: the edge is closed), `face` the factor of a face's
    /// inductance (its open area fraction; `None`: no current),
    /// both after the materials and [`LiveEdges`]. `walls` are the outer
    /// faces that hold their tangential edges at zero.
    pub(crate) fn eigenvalue(
        &self,
        q: [usize; 3],
        walls: &[[bool; 2]; 3],
        edge: &dyn Fn(Axis, [usize; 3]) -> Option<f64>,
        face: &dyn Fn(Axis, [usize; 3]) -> Option<f64>,
    ) -> f64 {
        let n = self.n;
        if self.live.is_some() && !self.is_open(q) {
            return 0.0;
        }
        let dl = |a: usize| self.deltas[a][q[a]];
        // Inverse capacitance in the cell of the edge along `d` at the
        // offsets `(ob, oc)` along the two other axes; 0: closed.
        let mut inv_c = [[[0.0f64; 2]; 2]; 3];
        for (d, at_d) in inv_c.iter_mut().enumerate() {
            let (b, c) = ((d + 1) % 3, (d + 2) % 3);
            for (ob, at_b) in at_d.iter_mut().enumerate() {
                for (oc, inv) in at_b.iter_mut().enumerate() {
                    let mut p = q;
                    p[b] += ob;
                    p[c] += oc;
                    let on_wall = [b, c]
                        .iter()
                        .any(|&t| (p[t] == 0 && walls[t][0]) || (p[t] == n[t] - 1 && walls[t][1]));
                    if on_wall {
                        continue;
                    }
                    let (Some(eps), Some(f)) = (self.edge_eps(d, p, q), edge(Axis::ALL[d], p))
                    else {
                        continue;
                    };
                    let cap = EPS0 * eps * f * dl(b) * dl(c) / (4.0 * dl(d));
                    if cap > 0.0 && cap.is_finite() {
                        *inv = 1.0 / cap;
                    }
                }
            }
        }
        // Faces: (normal, side, half the dual edge over L).
        let mut faces: Vec<(usize, usize, f64)> = Vec::with_capacity(6);
        for d in 0..3 {
            let (b, c) = ((d + 1) % 3, (d + 2) % 3);
            let lane = self.mats.mu_r(Axis::ALL[d]);
            for o in 0..2 {
                let mut p = q;
                p[d] += o;
                let Some(f) = face(Axis::ALL[d], p) else {
                    continue;
                };
                let l = MU0 * lane[self.grid.index(p)] * f * dl(b) * dl(c) * 2.0 / dl(d);
                if l > 0.0 && l.is_finite() {
                    faces.push((d, o, 1.0 / l));
                }
            }
        }
        // The edges of the face (d, o), each with its sign in the
        // circulation: (axis, ob, oc) in the edge's own (b, c) order.
        let edges_of = |d: usize, o: usize| -> [(usize, usize, usize, f64); 4] {
            let (b, c) = ((d + 1) % 3, (d + 2) % 3);
            // Edge along b: its other axes are (c, d) in cyclic order
            // (b + 1 = c, b + 2 = d); along c: (d, b).
            [
                (b, 0, o, 1.0),
                (c, o, 1, 1.0),
                (b, 1, o, -1.0),
                (c, o, 0, -1.0),
            ]
        };
        let m = faces.len();
        let mut a = vec![vec![0.0; m]; m];
        for i in 0..m {
            let ei = edges_of(faces[i].0, faces[i].1);
            for j in 0..m {
                let ej = edges_of(faces[j].0, faces[j].1);
                let mut sum = 0.0;
                for &(xa, xb, xc, xs) in &ei {
                    for &(ya, yb, yc, ys) in &ej {
                        if (xa, xb, xc) == (ya, yb, yc) {
                            sum += xs * ys * inv_c[xa][xb][xc];
                        }
                    }
                }
                a[i][j] = sum * (faces[i].2 * faces[j].2).sqrt();
            }
        }
        0.25 * super::conformal::largest_eigenvalue(a)
    }

    /// The largest [`value`](Self::value) over the cells and its cell.
    pub(crate) fn worst(&self) -> (f64, [usize; 3]) {
        let n = self.n;
        let (mut worst, mut at) = (0.0f64, [0; 3]);
        for i in 0..n[0].saturating_sub(1) {
            for j in 0..n[1].saturating_sub(1) {
                for k in 0..n[2].saturating_sub(1) {
                    let v = self.value([i, j, k]);
                    if v > worst {
                        (worst, at) = (v, [i, j, k]);
                    }
                }
            }
        }
        (worst, at)
    }

    /// The largest [`eigenvalue`](Self::eigenvalue) over the whole cells
    /// (every edge at its length, every face at its area) and its cell,
    /// with the outer faces `walls` holding their edges at zero.
    ///
    /// The closed form is an upper bound of a cell's eigenvalue, so only
    /// the cells whose closed form is above the largest eigenvalue found
    /// so far are solved, starting from the cell of the largest closed
    /// form; cells of the same sizes, permittivities, permeabilities and
    /// closed edges are solved once.
    pub(crate) fn worst_exact(&self, walls: &[[bool; 2]; 3]) -> (f64, [usize; 3]) {
        let n = self.n;
        let whole = |_: Axis, _: [usize; 3]| Some(1.0);
        let (closed, first) = self.worst();
        if !(closed > 0.0 && closed.is_finite()) {
            return (closed, first);
        }
        let mut solved: HashMap<[u64; 21], f64> = HashMap::new();
        let mut best = self.eigenvalue(first, walls, &whole, &whole);
        let mut at = first;
        for i in 0..n[0].saturating_sub(1) {
            for j in 0..n[1].saturating_sub(1) {
                for k in 0..n[2].saturating_sub(1) {
                    let q = [i, j, k];
                    if self.value(q) <= best {
                        continue;
                    }
                    let key = self.signature(q, walls);
                    let v = *solved
                        .entry(key)
                        .or_insert_with(|| self.eigenvalue(q, walls, &whole, &whole));
                    if v > best {
                        (best, at) = (v, q);
                    }
                }
            }
        }
        (best, at)
    }

    /// What the eigenvalue of the whole cell `q` depends on, as bits: its
    /// sizes, the permittivity of each edge (0: closed) and the
    /// permeability of each face.
    fn signature(&self, q: [usize; 3], walls: &[[bool; 2]; 3]) -> [u64; 21] {
        let n = self.n;
        let mut key = [0u64; 21];
        for a in 0..3 {
            key[a] = self.deltas[a][q[a]].to_bits();
        }
        let mut at = 3;
        for d in 0..3 {
            let (b, c) = ((d + 1) % 3, (d + 2) % 3);
            // (In the order of `eigenvalue`'s offsets: (0,0), (1,0),
            // (0,1), (1,1) along the two other axes.)
            for p in Self::edge_nodes(d, q) {
                let on_wall = [b, c]
                    .iter()
                    .any(|&t| (p[t] == 0 && walls[t][0]) || (p[t] == n[t] - 1 && walls[t][1]));
                let eps = if on_wall {
                    None
                } else {
                    self.edge_eps(d, p, q)
                };
                key[at] = eps.map_or(0, f64::to_bits);
                at += 1;
            }
        }
        for d in 0..3 {
            let lane = self.mats.mu_r(Axis::ALL[d]);
            for o in 0..2 {
                let mut p = q;
                p[d] += o;
                key[at] = lane[self.grid.index(p)].to_bits();
                at += 1;
            }
        }
        key
    }
}

/// The outer faces that hold their tangential edges at zero.
pub(crate) fn walls(faces: &FaceConditions) -> [[bool; 2]; 3] {
    Axis::ALL.map(|t| [false, true].map(|max| faces[Face::new(t, max)] == FaceCondition::Pec))
}

/// The time step limit of the lossless Yee scheme on `grid` with the
/// materials `mats` by its cells: `2/√λ_c` of the cell with the largest
/// eigenvalue `λ_c` of its own curl-curl problem, each edge with its
/// permittivity and its part of the capacitance, PEC edges closed (the
/// outer faces are taken as open: a PEC wall only lowers a cell's
/// eigenvalue). See the [module docs](self) for the proof and for what it
/// leaves out; at least [`courant_limit`](super::courant_limit) for
/// `ε_r, μ_r ≥ 1`, and equal to it on a uniform mesh in vacuum.
///
/// `None` on a cylindrical grid, for materials of another grid, and for
/// a grid without a cell that has an edge.
///
/// ```
/// use ems::engine::{EdgeMaterials, Grid, Material, cell_limit, courant_limit};
/// let g = Grid::uniform([4, 4, 4], [1e-3; 3]).unwrap();
/// let vacuum = cell_limit(&g, &EdgeMaterials::vacuum(&g)).unwrap();
/// assert!((vacuum / courant_limit(&g) - 1.0).abs() < 1e-12);
/// let filled = cell_limit(&g, &EdgeMaterials::new(&g, Material::dielectric(4.0))).unwrap();
/// assert!((filled / vacuum - 2.0).abs() < 1e-12);
/// ```
pub fn cell_limit(grid: &Grid, mats: &EdgeMaterials) -> Option<f64> {
    let (worst, _) = Cells::new(grid, mats, None)?.worst_exact(&[[false; 2]; 3]);
    (worst > 0.0 && worst.is_finite()).then(|| 1.0 / worst.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Material, courant_limit};

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn graded(seed: &mut u64, n: usize) -> Vec<f64> {
        let mut x = vec![0.0];
        for _ in 0..n {
            let d = 1e-4 * (0.2 + 3.0 * lcg(seed));
            x.push(x.last().unwrap() + d);
        }
        x
    }

    #[test]
    fn uniform_vacuum_is_the_courant_limit() {
        let g = Grid::uniform([3, 4, 5], [1e-3, 2e-3, 0.5e-3]).unwrap();
        let m = EdgeMaterials::vacuum(&g);
        let dt = cell_limit(&g, &m).unwrap();
        assert!((dt / courant_limit(&g) - 1.0).abs() < 1e-12);
        // The closed form is the cell's own eigenvalue.
        let cells = Cells::new(&g, &m, None).unwrap();
        let none = [[false; 2]; 3];
        let e = cells.eigenvalue([1, 1, 1], &none, &|_, _| Some(1.0), &|_, _| Some(1.0));
        assert!((e / cells.value([1, 1, 1]) - 1.0).abs() < 1e-10, "{e}");
    }

    #[test]
    fn graded_mesh_is_never_below_the_courant_limit() {
        let mut seed = 7;
        for _ in 0..20 {
            let g = Grid::new([
                graded(&mut seed, 6),
                graded(&mut seed, 5),
                graded(&mut seed, 7),
            ])
            .unwrap();
            let m = EdgeMaterials::vacuum(&g);
            let dt = cell_limit(&g, &m).unwrap();
            assert!(dt >= courant_limit(&g) * (1.0 - 1e-12));
        }
    }

    /// The closed form is an upper bound of every cell's eigenvalue, with
    /// mixed materials and PEC edges.
    #[test]
    fn the_closed_form_bounds_the_cell() {
        let mut seed = 11;
        for round in 0..10 {
            let g = Grid::new([
                graded(&mut seed, 4),
                graded(&mut seed, 4),
                graded(&mut seed, 4),
            ])
            .unwrap();
            let mut m = EdgeMaterials::new(&g, Material::dielectric(1.0));
            let n = g.nodes();
            for a in Axis::ALL {
                for k in 0..g.num_nodes() {
                    m.eps_r_mut(a)[k] = 1.0 + 4.0 * lcg(&mut seed);
                    m.mu_r_mut(a)[k] = 1.0 + 2.0 * lcg(&mut seed);
                }
            }
            if round % 2 == 1 {
                m.set_pec_edge(Axis::X, [1, 1, 1]);
                m.set_pec_edge(Axis::Z, [2, 1, 1]);
            }
            let cells = Cells::new(&g, &m, None).unwrap();
            let none = [[false; 2]; 3];
            for i in 0..n[0] - 1 {
                for j in 0..n[1] - 1 {
                    for k in 0..n[2] - 1 {
                        let q = [i, j, k];
                        let e = cells.eigenvalue(q, &none, &|_, _| Some(1.0), &|_, _| Some(1.0));
                        assert!(e <= cells.value(q) * (1.0 + 1e-9), "{q:?}");
                    }
                }
            }
        }
    }

    /// A sheet edge has its side's permittivity in the cells of each
    /// side, an edge of lossy metal vacuum's.
    #[test]
    fn live_edges_count_by_side() {
        let g = Grid::uniform([2, 2, 2], [1e-3; 3]).unwrap();
        let mut m = EdgeMaterials::new(&g, Material::dielectric(4.0));
        m.set_pec_edge(Axis::X, [0, 0, 0]);
        let mut live = LiveEdges::new(g.num_nodes());
        // In the plane z = 1: air's share above, the laminate's below.
        live.set(
            Axis::X,
            g.index([0, 1, 1]),
            LiveEdge::Sheet {
                normal: Axis::Z,
                above: 0.25,
                below: 1.75,
            },
        );
        let cells = Cells::new(&g, &m, Some(&live)).unwrap();
        let base = cells.value([1, 0, 0]);
        assert!((cells.value([0, 1, 1]) / base - 4.0).abs() < 1e-12);
        assert!((cells.value([0, 1, 0]) / base - 1.0).abs() < 1e-12);
        // PEC: left out. In the surface of lossy metal: vacuum.
        assert!((cells.value([0, 0, 0]) / base - 1.0).abs() < 1e-12);
        live.set(Axis::X, g.index([0, 0, 0]), LiveEdge::Solid { eps: 1.0 });
        let cells = Cells::new(&g, &m, Some(&live)).unwrap();
        assert!((cells.value([0, 0, 0]) / base - 4.0).abs() < 1e-12);
    }

    /// The CPML with frozen coefficients: the characteristic polynomial of
    /// a plane wave has no root outside the unit circle up to the Courant
    /// limit of the unstretched cell (module docs).
    #[test]
    fn frozen_cpml_has_no_root_outside_the_unit_circle() {
        // Roots of a monic quartic by Durand–Kerner.
        fn roots(c: [f64; 4]) -> [(f64, f64); 4] {
            let mul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
            let div = |a: (f64, f64), b: (f64, f64)| {
                let d = b.0 * b.0 + b.1 * b.1;
                ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
            };
            let mut z = [(0.4, 0.9), (-0.65, 0.72), (-0.9, -0.31), (0.37, -0.83)];
            for _ in 0..400 {
                for i in 0..4 {
                    let x = z[i];
                    let mut p = (1.0, 0.0);
                    for &ck in &c {
                        p = mul(p, x);
                        p.0 += ck;
                    }
                    let mut q = (1.0, 0.0);
                    for (j, &y) in z.iter().enumerate() {
                        if j != i {
                            q = mul(q, (x.0 - y.0, x.1 - y.1));
                        }
                    }
                    let s = div(p, q);
                    z[i] = (x.0 - s.0, x.1 - s.1);
                }
            }
            z
        }
        let mut worst = 0.0f64;
        for kappa in [1.0, 1.5, 3.0, 8.0, 15.0] {
            for xs in [0.0, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0] {
                for xa in [0.0, 1e-4, 1e-2, 0.1, 1.0, 5.0] {
                    let x: f64 = xs / kappa + xa;
                    let b = (-x).exp();
                    let a = if xs == 0.0 {
                        0.0
                    } else {
                        xs / (kappa * (xs + kappa * xa)) * (b - 1.0)
                    };
                    let c1 = 1.0 / kappa + a;
                    for total in [0.25, 0.5, 0.9, 0.99, 1.0] {
                        for part in 0..=20 {
                            let fa = total * part as f64 / 20.0;
                            let fb = total - fa;
                            // (z − 1)²(z − b)² + 4z·(A·(c1·z − b/κ)² + B·(z − b)²)
                            let p1 = [
                                -2.0 - 2.0 * b,
                                1.0 + 4.0 * b + b * b,
                                -2.0 * b - 2.0 * b * b,
                                b * b,
                            ];
                            let bk = b / kappa;
                            let p2 = [
                                4.0 * (fa * c1 * c1 + fb),
                                4.0 * (-2.0 * fa * c1 * bk - 2.0 * fb * b),
                                4.0 * (fa * bk * bk + fb * b * b),
                                0.0,
                            ];
                            let c = [p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], p1[3] + p2[3]];
                            for r in roots(c) {
                                worst = worst.max(r.0.hypot(r.1));
                            }
                        }
                    }
                }
            }
        }
        // (Double roots on the circle are found to the root of the
        // rounding.)
        assert!(worst < 1.0 + 1e-5, "{worst}");
    }
}
