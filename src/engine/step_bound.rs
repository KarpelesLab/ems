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
//! smallest cells are at the walls, the single cells give 1.00 to 1.31
//! times the Courant step, of which the closed form alone finds 1.00 to
//! 1.06.
//!
//! # The shares of the cells
//!
//! The theorem asks only that the cells' parts add up to the grid's: an
//! edge's capacitance may be divided among the cells around it in any
//! parts that are not negative, and so may a face's dual edge between
//! its two cells. With the quarters of the dual face and the halves of
//! the dual edge (the geometric shares) a thin cell has little of the
//! edges it shares with thick neighbours, and its eigenvalue is the
//! Courant limit of its own sizes; with more of them it has less, and
//! its neighbours more. `Cells::balance` looks for shares with which
//! the largest eigenvalue over the cells is small, and whatever it
//! finds, the bound holds with: `λ_max ≤ max_c λ_c(shares)`.
//!
//! The shares are taken in one family (`Shares`). Along each axis the
//! dual length at a node (half the cell below plus half the cell above)
//! is divided between those two cells, `lo` to the upper cell and `hi`
//! to the lower, the same for every cell of the slab. A cell then has
//! of an edge along `d` the product of its shares of the two dual
//! lengths across, `ε·m_b·m_c/Δ_d`, and of a face normal to `d` its
//! share `m_d` of the dual edge, `m_d/(μ·Δ_b·Δ_c)`: the cell of a grid
//! whose dual lengths at its two ends are its shares. For one `ε` and
//! `μ` its eigenvalue is at most
//!
//! ```text
//! λ_c/4 ≤ (1/4εμ)·Σ_a (1/lo_a + 1/hi_a)/Δ_a
//! ```
//!
//! (`Σ 1/Δ_a²` for `lo = hi = Δ/2`): the cell is the product of three
//! elements of two nodes, on which `KᵀL⁻¹K` plus the gradient of the
//! divergence is the sum of the three second differences, each with the
//! one eigenvalue `(1/lo + 1/hi)/Δ` beside zero, and the gradient part
//! is not negative. An end all of whose edges are closed (a wall, a
//! ground plane) is the limit of an infinite share and drops its term.
//! With several permittivities the least bounds it, as before.
//!
//! On a grid in one medium between walls this family holds the limit
//! itself. The operator is then the sum of three chains, one per axis,
//! whose largest eigenvalues add up, and a chain's largest eigenvalue
//! is that of each of its elements for the shares its eigenvector gives
//! (each node's dual length divided in the ratio of the two elements'
//! pulls on the node: a tridiagonal matrix is the sum of its elements'
//! in exactly this way). A unit test holds the
//! balanced bound to the power iteration on such a grid within a part
//! in 10⁴, where the geometric shares leave the Courant limit, 1.65
//! times shorter there. With a laminate, a ground plane and a strip the
//! shares of one family are not all there is, and the bound stays above
//! the limit: by 0.0 to 0.04 % on the boards of `tests/step_limit.rs`
//! with perfect copper, by 2 % on that unit test's.
//!
//! **Finding them.** One axis at a time, the two others held
//! (`Cells::balance`): each cell asks
//! `χ_lo/lo + χ_hi/hi ≤ Δ·(4·t·εμ − the two other axes' terms)`, which
//! is convex in its two shares, so the least level `t` at which a walk
//! along the axis can give every cell what it asks (the least it needs
//! at its far end, the rest to the next) is found by bisection, and the
//! shares are the middle of the walks from the two ends. A cell whose
//! own eigenvalue is below its closed form (several permittivities, a
//! conductor closing some of its edges) counts as slower by that ratio,
//! taken at the shares of the round before; two to four rounds, 35 to
//! 70 ms on board regions of 67 000 to 131 000 cells. The result is
//! the largest of the cells' eigenvalues with the shares found, each
//! cell solved or bounded by its closed form: never above that of the
//! geometric shares, which is where the search starts.
//!
//! **What does not follow the shares.** A sheet's edge has the
//! capacitance of its side whatever the cells of the two sides share
//! at its plane, and a face in the sheet's plane with a sheet edge
//! reads `f+·u+ + f−·u−` with the halves `f±`:
//! `(f+·a + f−·b)² ≤ f+²/g+·a² + f−²/g−·b²` for shares `g±`, which is
//! worse than `f±` on either side, so the planes of sheets keep the
//! halves. An edge of lossy metal has a capacitance outside the metal,
//! and the cells with field around it divide that by their shares.
//!
//! The bound is still not the limit where the family is not enough, and
//! where a model's own parts cost more than their share (a sheet's two
//! sides, the floor of lossy metal's edges): the power iteration on the
//! kernel's coefficients ([`stability_factor`](super::conformal::stability_factor),
//! an estimate from below, of the scheme without those parts) reads
//! 1.45, 1.38, 1.40 and 1.29 times the Courant limit on cadlab's four
//! regions, where the bound is 1.41, 1.35, 1.40 and 1.28 for the whole
//! cells.
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
//!   an edge (PEC in the materials) with that capacitance among the
//!   cells that are not all metal, at least `ε_r = 1`; a cell all of
//!   whose edges are PEC has no field and is left out. Beside a cut
//!   cell (a face around it has an edge the metal cuts) the edge has a
//!   floor of capacitance instead, `C ≥ dt²·Σ n/L/(4θ)` over its faces
//!   with `n` such edges each and `θ` = 1/16, and answers for itself
//!   with it: `(s + Σ_n u)² ≤ s²/(1 − θ) + n·Σ_n u²/θ` for a face's
//!   circulation, `s` over its other edges, so the edge is closed in
//!   the cells and its faces count with `1/(1 − θ)` of their inverse
//!   inductance (`LiveEdge::Floored`), in every cell they are in.
//! * *Conformal PEC* ([`conformal`](super::conformal)): a cut cell enters
//!   the same bound with its open lengths and areas and the shares of
//!   its slab; the compile step takes the cut cells into the step.
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
//! growth rate, so it does not move the limit. The coefficients are
//! `f32` too, which moves it by parts in 10⁷, and where the bound is the
//! limit itself (a board of perfect copper in a closed box) a step of
//! factor 1 is on it: the highest mode then grows with the step count,
//! as at the Courant limit of a uniform mesh. The factor below 1
//! (`Fdtd::timestep_factor`, 0.99 by default) is the margin for both.

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
    /// ordinary edge in the scheme, whose capacitance outside the metal
    /// is at least `ε0·total/l`; the cells with field around it divide
    /// that by their [`Shares`].
    Solid {
        /// The relative permittivity times the area \[m²\] of the part of
        /// the dual face that lies in the cells with field: what those
        /// cells have of the edge's capacitance between them, over
        /// `ε0/l`.
        total: f64,
    },
    /// An edge in the surface of lossy metal beside a cut cell, whose
    /// capacitance is at least `dt²·Σ n/L/(4θ)` over the faces around it
    /// (`θ` = [`FLOOR_SHARE`]; [`surface`](super::surface), the floor of
    /// its edges beside cut cells): with it the edge answers for its
    /// part of those faces' circulations by itself. It is closed in the
    /// cells, and a face with such an edge has `1/(1 − θ)` of its
    /// inverse inductance there.
    Floored,
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

/// The part `θ` of a face's circulation squared that the edges of
/// [`LiveEdge::Floored`] in it answer for: `(s + Σ_n u)² ≤ s²/(1 − θ) +
/// n·Σ_n u²/θ`.
pub(crate) const FLOOR_SHARE: f64 = 1.0 / 16.0;

/// The edges of [`LiveEdge`], by axis and node index.
#[derive(Debug, Clone, Default)]
pub(crate) struct LiveEdges {
    /// Whether the edge along the axis at the node index has an entry.
    flagged: [Vec<bool>; 3],
    edges: HashMap<(usize, usize), LiveEdge>,
    /// Whether an edge is [`LiveEdge::Floored`].
    floored: bool,
}

impl LiveEdges {
    /// No such edge on a grid of `nodes` nodes.
    pub(crate) fn new(nodes: usize) -> Self {
        Self {
            flagged: core::array::from_fn(|_| vec![false; nodes]),
            edges: HashMap::new(),
            floored: false,
        }
    }

    /// Declares the edge along `axis` at node index `k`. A sheet on an
    /// edge of lossy metal stays the metal's.
    pub(crate) fn set(&mut self, axis: Axis, k: usize, e: LiveEdge) {
        let a = axis.index();
        let metal = |e: &LiveEdge| matches!(e, LiveEdge::Solid { .. } | LiveEdge::Floored);
        if self.flagged[a][k] && self.edges.get(&(a, k)).is_some_and(metal) && !metal(&e) {
            return;
        }
        self.floored |= e == LiveEdge::Floored;
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
    shares: Shares,
    /// Whether the node plane of each axis has a sheet edge with that
    /// normal.
    sheet_planes: [Vec<bool>; 3],
}

/// What each cell takes of the dual lengths at its two ends, along each
/// axis (module docs, *The shares of the cells*): `lo[a][i]` is the part
/// of the dual length at node `i` that cell `i` of axis `a` has, `hi[a][i]`
/// its part of the dual length at node `i + 1`. Half the cell each: the
/// geometric shares.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Shares {
    lo: [Vec<f64>; 3],
    hi: [Vec<f64>; 3],
}

impl Shares {
    /// Half of each cell at each of its ends.
    fn halves(deltas: &[Vec<f64>; 3]) -> Self {
        let half = |a: usize| deltas[a].iter().map(|d| 0.5 * d).collect::<Vec<_>>();
        Self {
            lo: core::array::from_fn(half),
            hi: core::array::from_fn(half),
        }
    }
}

/// The eigenvalue, `λ/4` \[1/s²\], of a cell that is not the whole cell
/// the materials describe (one conformal PEC cuts), with the shares the
/// cells have; `None` for every other cell.
pub(crate) type Own<'o> = &'o dyn Fn(&Cells<'_>, [usize; 3]) -> Option<f64>;

/// The least part of a dual length a cell is left with by
/// [`Cells::balance`].
const LEAST_SHARE: f64 = 0.02;

/// What the closed form of a cell is made of ([`Cells::parts`]).
#[derive(Debug, Clone, Copy)]
struct Parts {
    /// `ε0·μ0` times the least relative permittivity and permeability.
    z: f64,
    /// Whether the cell has an edge that is not closed on its face at
    /// the low and the high end of each axis.
    open: [[bool; 2]; 3],
    /// Whether the closed form is the cell's eigenvalue: one
    /// permittivity on its edges, one permeability on its faces, and
    /// every edge between two faces with an open edge open.
    plain: bool,
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
        let deltas = Axis::ALL.map(|a| grid.deltas(a));
        let n = grid.nodes();
        let live = live.filter(|l| !l.is_empty());
        let mut sheet_planes: [Vec<bool>; 3] = core::array::from_fn(|a| vec![false; n[a]]);
        if let Some(l) = live {
            // (Sorted: nothing here depends on the map's order, a flag
            // per plane.)
            for (&(_, k), e) in &l.edges {
                if let LiveEdge::Sheet { normal, .. } = e {
                    let a = normal.index();
                    let p = [k / (n[1] * n[2]), (k / n[2]) % n[1], k % n[2]];
                    debug_assert_eq!(grid.index(p), k);
                    sheet_planes[a][p[a]] = true;
                }
            }
        }
        Some(Self {
            grid,
            mats,
            live,
            shares: Shares::halves(&deltas),
            deltas,
            n,
            sheet_planes,
        })
    }

    /// The cells' shares.
    #[cfg(test)]
    pub(crate) fn shares(&self) -> &Shares {
        &self.shares
    }

    /// Sets the cells' shares.
    #[cfg(test)]
    pub(crate) fn set_shares(&mut self, shares: Shares) {
        self.shares = shares;
    }

    /// Half of every cell at each of its ends again.
    pub(crate) fn reset_shares(&mut self) {
        self.shares = Shares::halves(&self.deltas);
    }

    /// The cell `q`'s part of the dual length at its end `o` of axis `a`.
    #[inline]
    fn share(&self, a: usize, q: [usize; 3], o: usize) -> f64 {
        if o == 0 {
            self.shares.lo[a][q[a]]
        } else {
            self.shares.hi[a][q[a]]
        }
    }

    /// The capacitance, over `ε0/l`, that the cell `q` has of the edge
    /// along `d` at `p`, which lies at the cell's ends `ob` and `oc` of
    /// the two other axes (in cyclic order): the relative permittivity
    /// times the cell's shares of the two dual lengths. `None` for an
    /// edge held at zero.
    #[inline]
    fn edge_area(
        &self,
        d: usize,
        p: [usize; 3],
        q: [usize; 3],
        ob: usize,
        oc: usize,
    ) -> Option<f64> {
        let k = self.grid.index(p);
        let axis = Axis::ALL[d];
        let (b, c) = ((d + 1) % 3, (d + 2) % 3);
        let (mb, mc) = (self.share(b, q, ob), self.share(c, q, oc));
        let live = self.live.and_then(|l| l.get(d, k));
        match live {
            Some(LiveEdge::Solid { total }) => {
                let among = self.solid_shares(d, p);
                (among > 0.0).then(|| total * mb * mc / among)
            }
            Some(LiveEdge::Floored) => None,
            _ if self.mats.pec_mask(axis)[k] => None,
            Some(LiveEdge::Sheet {
                normal,
                above,
                below,
            }) => {
                // The side's capacitance is the side's, whatever the
                // cells of the two sides share at the plane: half the
                // cell along the normal.
                let nrm = normal.index();
                let f = if q[nrm] == p[nrm] { above } else { below };
                let half = 0.5 * self.deltas[nrm][q[nrm]];
                let other = if nrm == b { mc } else { mb };
                Some(self.mats.eps_r(axis)[k] * f * half * other)
            }
            None => Some(self.mats.eps_r(axis)[k] * mb * mc),
        }
    }

    /// The sum over the cells with field around the edge along `d` at `p`
    /// of the products of their shares of the two dual lengths: what an
    /// edge of lossy metal's capacitance outside the metal is divided by.
    fn solid_shares(&self, d: usize, p: [usize; 3]) -> f64 {
        let (b, c) = ((d + 1) % 3, (d + 2) % 3);
        let n = self.n;
        let mut sum = 0.0;
        for ob in 0..2 {
            for oc in 0..2 {
                // The cell that has the edge at its ends (ob, oc).
                let mut q = p;
                if (ob == 1 && q[b] == 0) || (oc == 1 && q[c] == 0) {
                    continue;
                }
                q[b] -= ob;
                q[c] -= oc;
                if q[b] + 1 >= n[b] || q[c] + 1 >= n[c] || q[d] + 1 >= n[d] {
                    continue;
                }
                if self.is_open(q) {
                    sum += self.share(b, q, ob) * self.share(c, q, oc);
                }
            }
        }
        sum
    }

    /// The cell `q`'s part of the dual edge of the face normal to `d` at
    /// `p`, at the cell's end `o`: its share of the dual length, over
    /// `1 − θ` for a face with an edge of [`LiveEdge::Floored`], but for
    /// a face in the plane of a sheet with a sheet edge in its contour.
    /// That face reads `f+·u+ + f−·u−` on the sheet's edges, `f±` the
    /// halves of the two cells over the dual length, and
    /// `(f+·a + f−·b)² ≤ f+²/g+·a² + f−²/g−·b²` for the cells' shares
    /// `g±` of the dual length (Cauchy–Schwarz): `f²/g` of the dual
    /// length, which is `f` of it for the geometric shares.
    #[inline]
    fn face_share(&self, d: usize, p: [usize; 3], q: [usize; 3], o: usize) -> f64 {
        let m = self.share(d, q, o);
        let Some(live) = self.live else {
            return m;
        };
        let sheets = self.sheet_planes[d][p[d]];
        if !sheets && !live.floored {
            return m;
        }
        let (b, c) = ((d + 1) % 3, (d + 2) % 3);
        let (mut pb, mut pc) = (p, p);
        pb[b] += 1;
        pc[c] += 1;
        let around =
            [(b, p), (c, p), (b, pc), (c, pb)].map(|(ax, at)| live.get(ax, self.grid.index(at)));
        let sheet = sheets
            && around
                .iter()
                .any(|e| matches!(e, Some(LiveEdge::Sheet { normal, .. }) if normal.index() == d));
        let m = if sheet {
            let half = 0.5 * self.deltas[d][q[d]];
            half * half / m
        } else {
            m
        };
        if around.contains(&Some(LiveEdge::Floored)) {
            m / (1.0 - FLOOR_SHARE)
        } else {
            m
        }
    }

    /// Whether the node `p` is on an outer face that holds the edges
    /// along `d` there at zero.
    #[inline]
    fn on_wall(&self, d: usize, p: [usize; 3], walls: &[[bool; 2]; 3]) -> bool {
        [(d + 1) % 3, (d + 2) % 3]
            .iter()
            .any(|&t| (p[t] == 0 && walls[t][0]) || (p[t] == self.n[t] - 1 && walls[t][1]))
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

    /// What the closed form of the cell `q` is made of: the least
    /// permittivity among its edges that are not held at zero, each as
    /// the cell has it (its capacitance in the cell over that of vacuum
    /// on the cell's shares), the least permeability among its faces,
    /// and which of its faces have such an edge. `None` for a cell
    /// without one (nothing to bound).
    fn parts(&self, q: [usize; 3], walls: &[[bool; 2]; 3]) -> Option<Parts> {
        if self.live.is_some() && !self.is_open(q) {
            return None;
        }
        let (mut eps, mut most) = (f64::INFINITY, 0.0f64);
        let mut open = [[false; 2]; 3];
        let mut edges = 0;
        for d in 0..3 {
            let (b, c) = ((d + 1) % 3, (d + 2) % 3);
            for ob in 0..2 {
                for oc in 0..2 {
                    let mut p = q;
                    p[b] += ob;
                    p[c] += oc;
                    if self.on_wall(d, p, walls) {
                        continue;
                    }
                    if let Some(area) = self.edge_area(d, p, q, ob, oc) {
                        let e = area / (self.share(b, q, ob) * self.share(c, q, oc));
                        eps = eps.min(e);
                        most = most.max(e);
                        open[b][ob] = true;
                        open[c][oc] = true;
                        edges += 1;
                    }
                }
            }
        }
        if eps == f64::INFINITY {
            return None;
        }
        let (mut mu, mut most_mu) = (f64::INFINITY, 0.0f64);
        for d in 0..3 {
            let lane = self.mats.mu_r(Axis::ALL[d]);
            for o in 0..2 {
                let mut p = q;
                p[d] += o;
                // (A face with less of the dual edge than the cell's
                // share counts as that much more permeable.)
                let m = self.share(d, q, o);
                let v = lane[self.grid.index(p)] * m / self.face_share(d, p, q, o);
                mu = mu.min(v);
                most_mu = most_mu.max(v);
            }
        }
        let ends = |a: usize| usize::from(open[a][0]) + usize::from(open[a][1]);
        let between = ends(1) * ends(2) + ends(2) * ends(0) + ends(0) * ends(1);
        let same = |lo: f64, hi: f64| hi <= lo * (1.0 + 1e-12);
        Some(Parts {
            z: EPS0 * MU0 * eps * mu,
            open,
            plain: edges == between && same(eps, most) && same(mu, most_mu),
        })
    }

    /// `Σ_a (1/lo_a + 1/hi_a)/Δ_a` of the cell `q` over four, an end
    /// without an open edge left out: `Σ 1/Δ²` for the geometric shares.
    #[inline]
    fn inverse_squares(&self, q: [usize; 3], open: &[[bool; 2]; 3]) -> f64 {
        let mut sum = 0.0;
        for a in 0..3 {
            let t: f64 = (0..2)
                .filter(|&o| open[a][o])
                .map(|o| 1.0 / self.share(a, q, o))
                .sum();
            sum += t / self.deltas[a][q[a]];
        }
        0.25 * sum
    }

    /// `λ_c/4` \[1/s²\] of the cell with lowest node `q`, at most: the
    /// closed form of the [module docs](self) (exact where the cell's
    /// edges have one permittivity and its faces one permeability). The
    /// cell is stable at `dt` if `dt²` times this is at most 1.
    #[cfg(test)]
    pub(crate) fn value(&self, q: [usize; 3], walls: &[[bool; 2]; 3]) -> f64 {
        match self.parts(q, walls) {
            Some(p) if p.z > 0.0 => self.inverse_squares(q, &p.open) / p.z,
            Some(_) => f64::INFINITY,
            None => 0.0,
        }
    }

    /// `λ_c/4` \[1/s²\] of the cell with lowest node `q`, by its own
    /// eigenvalue problem: every edge with the capacitance the cell has
    /// of it, every face with the cell's part of its dual edge. `edge`
    /// gives the factor of an edge's capacitance (the open length of a
    /// cut edge: `1/f`; `None`: the edge is closed), `face` the factor of
    /// a face's inductance (its open area fraction; `None`: no current),
    /// both after the materials and [`LiveEdges`]. `walls` are the outer
    /// faces that hold their tangential edges at zero.
    pub(crate) fn eigenvalue(
        &self,
        q: [usize; 3],
        walls: &[[bool; 2]; 3],
        edge: &dyn Fn(Axis, [usize; 3]) -> Option<f64>,
        face: &dyn Fn(Axis, [usize; 3]) -> Option<f64>,
    ) -> f64 {
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
                    if self.on_wall(d, p, walls) {
                        continue;
                    }
                    let (Some(area), Some(f)) =
                        (self.edge_area(d, p, q, ob, oc), edge(Axis::ALL[d], p))
                    else {
                        continue;
                    };
                    let cap = EPS0 * area * f / dl(d);
                    if cap > 0.0 && cap.is_finite() {
                        *inv = 1.0 / cap;
                    }
                }
            }
        }
        // Faces: (normal, side, the cell's part of the dual edge over L).
        let mut faces = [(0usize, 0usize, 0.0f64); 6];
        let mut m = 0;
        for d in 0..3 {
            let (b, c) = ((d + 1) % 3, (d + 2) % 3);
            let lane = self.mats.mu_r(Axis::ALL[d]);
            for o in 0..2 {
                let mut p = q;
                p[d] += o;
                let Some(f) = face(Axis::ALL[d], p) else {
                    continue;
                };
                let l = MU0 * lane[self.grid.index(p)] * f * dl(b) * dl(c)
                    / self.face_share(d, p, q, o);
                if l > 0.0 && l.is_finite() {
                    faces[m] = (d, o, 1.0 / l);
                    m += 1;
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
        let mut a = [[0.0f64; 6]; 6];
        for i in 0..m {
            let ei = edges_of(faces[i].0, faces[i].1);
            for j in i..m {
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
                a[j][i] = a[i][j];
            }
        }
        0.25 * largest_of_six(a, m)
    }

    /// One pass over the whole cells (every edge at its length, every
    /// face at its area) with the shares as they are: the largest
    /// eigenvalue among them, `λ/4` \[1/s²\], and its cell, with the
    /// outer faces `walls` holding their edges at zero.
    ///
    /// The closed form is an upper bound of a cell's eigenvalue, and is
    /// that eigenvalue for a cell of one medium with nothing closed
    /// inside it. Of the others, the cells whose closed form is above
    /// the largest eigenvalue found so far are solved, and so are those
    /// whose closed form is above `level`. A cell `own` has a value for
    /// (a cell that conformal PEC cuts, as the scheme has it) counts with
    /// that. `notes`, one entry per cell
    /// in the order of the pass, takes what [`balance`](Self::balance)
    /// places the shares by: `ε0·μ0·ε_r·μ_r` of the cell's closed form,
    /// times what a solved cell's eigenvalue is below its closed form,
    /// and its ends with an open edge as bits.
    fn survey(
        &self,
        walls: &[[bool; 2]; 3],
        own: Option<Own<'_>>,
        level: f64,
        mut notes: Option<(&mut [f32], &mut [u8])>,
    ) -> (f64, [usize; 3]) {
        let n = self.n;
        let whole = |_: Axis, _: [usize; 3]| Some(1.0);
        let (mut best, mut at) = (0.0f64, [0; 3]);
        let mut cell = 0;
        for i in 0..n[0].saturating_sub(1) {
            for j in 0..n[1].saturating_sub(1) {
                for k in 0..n[2].saturating_sub(1) {
                    let q = [i, j, k];
                    let parts = self.parts(q, walls);
                    let mut note = (0.0f32, 0u8);
                    if let Some(p) = parts {
                        let closed = if p.z > 0.0 {
                            self.inverse_squares(q, &p.open) / p.z
                        } else {
                            f64::INFINITY
                        };
                        let mut z = p.z;
                        let v = if let Some(v) = own.and_then(|f| f(self, q)) {
                            // A cell with an eigenvalue of its own: as
                            // much slower or faster than its closed form.
                            if v > 0.0 && closed.is_finite() {
                                z *= closed / v;
                            }
                            v
                        } else if p.plain && closed.is_finite() {
                            closed
                        } else if closed > best.min(level) || !closed.is_finite() {
                            let own = self.eigenvalue(q, walls, &whole, &whole);
                            if own > 0.0 && own < closed && closed.is_finite() {
                                z *= closed / own;
                            }
                            own
                        } else {
                            // At most the closed form, which is at most
                            // the largest found.
                            0.0
                        };
                        if v > best {
                            (best, at) = (v, q);
                        }
                        // (Rounded down: the cell counts as a little
                        // faster than it is.)
                        note = (
                            (z as f32).next_down().max(0.0),
                            (0..6).fold(0u8, |m, b| m | (u8::from(p.open[b / 2][b % 2]) << b)),
                        );
                    }
                    if let Some((z, open)) = notes.as_mut() {
                        (z[cell], open[cell]) = note;
                    }
                    cell += 1;
                }
            }
        }
        (best, at)
    }

    /// The largest eigenvalue over the whole cells with the shares as
    /// they are, `λ/4` \[1/s²\], and its cell ([`survey`](Self::survey)).
    #[cfg(test)]
    pub(crate) fn worst_exact(
        &self,
        walls: &[[bool; 2]; 3],
        own: Option<Own<'_>>,
    ) -> (f64, [usize; 3]) {
        self.survey(walls, own, f64::INFINITY, None)
    }

    /// [`worst_exact`](Self::worst_exact) with the shares
    /// [`balance`](Self::balance) finds from the geometric ones, which
    /// the cells are left with.
    pub(crate) fn worst_balanced(
        &mut self,
        walls: &[[bool; 2]; 3],
        own: Option<Own<'_>>,
    ) -> (f64, [usize; 3]) {
        self.reset_shares();
        self.balance(walls, own)
    }

    /// Chooses the cells' shares so that the largest eigenvalue over the
    /// cells ([`worst_exact`](Self::worst_exact)) is as small as this
    /// finds it (module docs, *The shares of the cells*), and returns it,
    /// `λ/4` \[1/s²\], with its cell: never above that of the shares it
    /// starts from. Whatever shares it ends on, the bound holds with
    /// them: this only looks for good ones.
    ///
    /// A round takes one axis at a time, the two others held: the least
    /// `t` for which every cell's closed form is at most `t` is found by
    /// bisection, each trial by one walk along the axis that gives every
    /// cell the least it needs at its far end and the rest to the next.
    /// A cell whose eigenvalue is below its closed form counts by its
    /// eigenvalue at the shares of the round before, and so does a cell
    /// `own` has a value for (above its closed form or below).
    pub(crate) fn balance(
        &mut self,
        walls: &[[bool; 2]; 3],
        own: Option<Own<'_>>,
    ) -> (f64, [usize; 3]) {
        let n = self.n;
        let cells = [
            n[0].saturating_sub(1),
            n[1].saturating_sub(1),
            n[2].saturating_sub(1),
        ];
        let count = cells[0] * cells[1] * cells[2];
        if count == 0 {
            return (0.0, [0; 3]);
        }
        // The dual lengths.
        let dual: [Vec<f64>; 3] = core::array::from_fn(|a| {
            let d = &self.deltas[a];
            (0..=d.len())
                .map(|k| {
                    let below = if k > 0 { 0.5 * d[k - 1] } else { 0.0 };
                    let above = if k < d.len() { 0.5 * d[k] } else { 0.0 };
                    below + above
                })
                .collect()
        });
        let pinned: [Vec<bool>; 3] = core::array::from_fn(|a| self.sheet_planes[a].clone());
        let mut z = vec![0.0f32; count];
        let mut open = vec![0u8; count];
        let mut level = f64::INFINITY;
        let mut best: Option<((f64, [usize; 3]), Shares)> = None;
        for round in 0..=BALANCE_ROUNDS {
            let found = self.survey(walls, own, level, Some((&mut z, &mut open)));
            let gain = match &best {
                Some((b, _)) if found.0 < b.0 => 1.0 - found.0 / b.0,
                Some(_) => break,
                None => 1.0,
            };
            best = Some((found, self.shares.clone()));
            if !(found.0 > 0.0 && found.0.is_finite())
                || round == BALANCE_ROUNDS
                || (round >= 2 && gain < BALANCE_GAIN)
            {
                break;
            }
            level = FEEDBACK_FROM * found.0;
            for a in 0..3 {
                self.balance_axis(a, &dual[a], &pinned[a], &z, &open, cells);
            }
        }
        let (found, shares) = best.expect("one round");
        self.shares = shares;
        found
    }

    /// One axis of [`balance`](Self::balance).
    fn balance_axis(
        &mut self,
        a: usize,
        dual: &[f64],
        pinned: &[bool],
        z: &[f32],
        open: &[u8],
        cells: [usize; 3],
    ) {
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        let term = |x: usize, i: usize, bits: u8, s: &Shares| {
            let mut t = 0.0;
            if bits & (1 << (2 * x)) != 0 {
                t += 1.0 / s.lo[x][i];
            }
            if bits & (2 << (2 * x)) != 0 {
                t += 1.0 / s.hi[x][i];
            }
            t / self.deltas[x][i]
        };
        // Per slab of the axis and per pair of open ends: the cells by
        // their medium, each with the largest sum of the two other axes'
        // terms. A cell asks `χ_lo/lo + χ_hi/hi ≤ Δ·(4·t·z − rest)`.
        let mut slabs: Vec<[Vec<(f32, f64)>; 4]> = vec![Default::default(); cells[a]];
        let mut at = 0;
        let mut top = 0.0f64;
        for i in 0..cells[0] {
            for j in 0..cells[1] {
                for k in 0..cells[2] {
                    let (zc, bits) = (z[at], open[at]);
                    at += 1;
                    if zc <= 0.0 {
                        continue;
                    }
                    let q = [i, j, k];
                    let rest =
                        term(b, q[b], bits, &self.shares) + term(c, q[c], bits, &self.shares);
                    let own = term(a, q[a], bits, &self.shares);
                    top = top.max(0.25 * (rest + own) / f64::from(zc));
                    let ends = usize::from((bits >> (2 * a)) & 3);
                    if ends == 0 {
                        continue;
                    }
                    let group = &mut slabs[q[a]][ends];
                    match group.iter_mut().find(|g| g.0 == zc) {
                        Some(g) => g.1 = g.1.max(rest),
                        None => group.push((zc, rest)),
                    }
                }
            }
        }
        if !(top > 0.0 && top.is_finite()) {
            return;
        }
        // One walk along the axis at the level `t`, upwards or downwards:
        // every cell takes the least it needs at its far end and leaves
        // the rest to the next. The shares, or `None` where a cell cannot
        // have what it asks.
        let m = cells[a];
        let walk = |t: f64, up: bool| -> Option<(Vec<f64>, Vec<f64>)> {
            // (`near`: the end the walk comes from.)
            let (mut near, mut far) = (vec![0.0; m], vec![0.0; m]);
            let mut left = if up { dual[0] } else { dual[m] };
            for step in 0..m {
                let i = if up { step } else { m - 1 - step };
                near[i] = left;
                let d = self.deltas[a][i];
                let room = |ends: usize| {
                    slabs[i][ends]
                        .iter()
                        .map(|&(zc, rest)| d * (4.0 * t * f64::from(zc) - rest))
                        .fold(f64::INFINITY, f64::min)
                };
                // (Ends: 1 the low one alone, 2 the high one alone.)
                let (only_near, only_far) = if up {
                    (room(1), room(2))
                } else {
                    (room(2), room(1))
                };
                let both = room(3);
                if only_near <= 0.0 || only_far <= 0.0 || both <= 0.0 || 1.0 / left > only_near {
                    return None;
                }
                let mut need = 0.0f64;
                if only_far.is_finite() {
                    need = need.max(1.0 / only_far);
                }
                if both.is_finite() {
                    if both <= 1.0 / left {
                        return None;
                    }
                    need = need.max(1.0 / (both - 1.0 / left));
                }
                // The node at the cell's far end, and the cell beyond.
                let node = if up { i + 1 } else { i };
                let whole = dual[node];
                if step + 1 == m {
                    // The last node is the last cell's alone.
                    if need > whole {
                        return None;
                    }
                    far[i] = whole;
                } else if pinned[node] {
                    // The plane of a sheet: half of each cell.
                    let half = 0.5 * d;
                    if need > half {
                        return None;
                    }
                    far[i] = half;
                    left = whole - half;
                } else {
                    let need = need.max(LEAST_SHARE * whole);
                    if need > (1.0 - LEAST_SHARE) * whole {
                        return None;
                    }
                    far[i] = need;
                    left = whole - need;
                }
            }
            Some(if up { (near, far) } else { (far, near) })
        };
        // `top` is the largest closed form with the shares as they are.
        let (mut below, mut above) = (0.0, top);
        let mut found = None;
        for _ in 0..60 {
            let t = 0.5 * (below + above);
            match walk(t, true) {
                Some(s) => {
                    above = t;
                    found = Some(s);
                }
                None => below = t,
            }
            if above - below <= 1e-9 * above {
                break;
            }
        }
        if let Some((mut lo, mut hi)) = found {
            // The shares that leave every cell the most at its upper end
            // hold at the same level, and so does anything between the
            // two (what a cell asks is convex in its two shares): the
            // middle, which leaves the cells that are not at the level
            // near the halves they had.
            if let Some((lo_down, hi_down)) = walk(above, false) {
                for (x, y) in lo.iter_mut().zip(&lo_down) {
                    *x = 0.5 * (*x + y);
                }
                for (x, y) in hi.iter_mut().zip(&hi_down) {
                    *x = 0.5 * (*x + y);
                }
            }
            self.shares.lo[a] = lo;
            self.shares.hi[a] = hi;
        }
    }
}

/// Rounds of [`Cells::balance`] over the three axes, at most.
const BALANCE_ROUNDS: usize = 4;

/// The part of the largest eigenvalue that a round of
/// [`Cells::balance`] after the second has to take off for another.
const BALANCE_GAIN: f64 = 0.01;

/// The part of the largest eigenvalue found from which a cell's own
/// eigenvalue is asked for in [`Cells::balance`].
const FEEDBACK_FROM: f64 = 0.7;

/// The largest eigenvalue of the leading `m × m` part of the symmetric
/// `a` (cyclic Jacobi rotations).
fn largest_of_six(mut a: [[f64; 6]; 6], m: usize) -> f64 {
    for _ in 0..60 {
        let (mut off, mut diag) = (0.0, 0.0);
        for (i, row) in a.iter().enumerate().take(m) {
            for (j, v) in row.iter().enumerate().take(m) {
                if i == j {
                    diag += v * v;
                } else {
                    off += v * v;
                }
            }
        }
        if off <= 1e-28 * diag {
            break;
        }
        for p in 0..m {
            for q in p + 1..m {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut().take(m) {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                for k in 0..m {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..m).map(|i| a[i][i]).fold(0.0, f64::max)
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
    let mut cells = Cells::new(grid, mats, None)?;
    let (worst, _) = cells.worst_balanced(&[[false; 2]; 3], None);
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
        assert!(
            (e / cells.value([1, 1, 1], &none) - 1.0).abs() < 1e-10,
            "{e}"
        );
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
                        assert!(e <= cells.value(q, &none) * (1.0 + 1e-9), "{q:?}");
                    }
                }
            }
        }
    }

    /// Random shares of the dual lengths: every cell between a tenth and
    /// nine tenths of each inner one.
    fn random_shares(seed: &mut u64, cells: &Cells) -> Shares {
        let mut s = cells.shares().clone();
        for a in 0..3 {
            let d = &cells.deltas[a];
            for k in 1..d.len() {
                let whole = 0.5 * (d[k - 1] + d[k]);
                let f = 0.1 + 0.8 * lcg(seed);
                s.hi[a][k - 1] = f * whole;
                s.lo[a][k] = whole - s.hi[a][k - 1];
            }
        }
        s
    }

    /// The closed form bounds the cell's eigenvalue whatever the cells'
    /// shares, with walls, mixed materials and PEC edges.
    #[test]
    fn the_closed_form_bounds_the_cell_with_any_shares() {
        let mut seed = 23;
        for round in 0..12 {
            let g = Grid::new([
                graded(&mut seed, 4),
                graded(&mut seed, 4),
                graded(&mut seed, 4),
            ])
            .unwrap();
            let mut m = EdgeMaterials::new(&g, Material::dielectric(1.0));
            let n = g.nodes();
            if round % 3 != 0 {
                for a in Axis::ALL {
                    for k in 0..g.num_nodes() {
                        m.eps_r_mut(a)[k] = 1.0 + 4.0 * lcg(&mut seed);
                        m.mu_r_mut(a)[k] = 1.0 + 2.0 * lcg(&mut seed);
                    }
                }
            }
            if round % 2 == 1 {
                m.set_pec_edge(Axis::X, [1, 1, 1]);
                m.set_pec_edge(Axis::Z, [2, 1, 1]);
                // A whole face.
                m.set_pec_edge(Axis::X, [2, 2, 2]);
                m.set_pec_edge(Axis::X, [2, 3, 2]);
                m.set_pec_edge(Axis::Y, [2, 2, 2]);
                m.set_pec_edge(Axis::Y, [3, 2, 2]);
            }
            let mut cells = Cells::new(&g, &m, None).unwrap();
            let shares = random_shares(&mut seed, &cells);
            cells.set_shares(shares);
            let walls = [
                [round % 4 < 2, true],
                [false, round % 4 == 1],
                [true, false],
            ];
            let whole = |_: Axis, _: [usize; 3]| Some(1.0);
            for i in 0..n[0] - 1 {
                for j in 0..n[1] - 1 {
                    for k in 0..n[2] - 1 {
                        let q = [i, j, k];
                        let e = cells.eigenvalue(q, &walls, &whole, &whole);
                        let v = cells.value(q, &walls);
                        assert!(e <= v * (1.0 + 1e-9), "{q:?} {e} {v}");
                        // One medium, nothing closed inside: exact.
                        if round % 3 == 0 && round % 2 == 0 {
                            assert!(e >= v * (1.0 - 1e-9), "{q:?} {e} {v}");
                        }
                    }
                }
            }
        }
    }

    /// `λ_max/4` of the grid's own operator from below (power iteration
    /// on `C⁻¹KᵀL⁻¹K` assembled from the grid's lengths and areas, the
    /// outer faces PEC), for the bound to be held against.
    fn grid_eigenvalue(g: &Grid, m: &EdgeMaterials, iterations: usize) -> f64 {
        let n = g.nodes();
        let live = |a: Axis, p: [usize; 3]| {
            let (b, c) = a.others();
            p[a.index()] + 1 < n[a.index()]
                && [b, c]
                    .iter()
                    .all(|t| p[t.index()] > 0 && p[t.index()] + 1 < n[t.index()])
                && !m.pec_mask(a)[g.index(p)]
        };
        let nodes: Vec<[usize; 3]> = (0..n[0])
            .flat_map(|i| (0..n[1]).flat_map(move |j| (0..n[2]).map(move |k| [i, j, k])))
            .collect();
        let mut seed = 99;
        let mut u: [Vec<f64>; 3] = core::array::from_fn(|a| {
            nodes
                .iter()
                .map(|&p| {
                    if live(Axis::ALL[a], p) {
                        lcg(&mut seed) - 0.5
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        let mut lambda = 0.0;
        for _ in 0..iterations {
            // i = L⁻¹·K·u on the faces.
            let mut cur: [Vec<f64>; 3] = core::array::from_fn(|_| vec![0.0; nodes.len()]);
            for a in Axis::ALL {
                let (b, c) = a.others();
                for &p in &nodes {
                    if p[b.index()] + 1 >= n[b.index()] || p[c.index()] + 1 >= n[c.index()] {
                        continue;
                    }
                    let (mut pb, mut pc) = (p, p);
                    pb[b.index()] += 1;
                    pc[c.index()] += 1;
                    let circ = u[b.index()][g.index(p)] + u[c.index()][g.index(pb)]
                        - u[b.index()][g.index(pc)]
                        - u[c.index()][g.index(p)];
                    let l =
                        MU0 * m.mu_r(a)[g.index(p)] * g.face_area(a, p) / g.dual_edge_length(a, p);
                    cur[a.index()][g.index(p)] = circ / l;
                }
            }
            // y = C⁻¹·Kᵀ·i on the edges.
            let mut y: [Vec<f64>; 3] = core::array::from_fn(|_| vec![0.0; nodes.len()]);
            for a in Axis::ALL {
                let (b, c) = a.others();
                for &p in &nodes {
                    if !live(a, p) {
                        continue;
                    }
                    let (mut mb, mut mc) = (p, p);
                    mb[b.index()] -= 1;
                    mc[c.index()] -= 1;
                    // The edge along a is in the faces normal to c at p
                    // and p − b (signs +, −) and normal to b at p and
                    // p − c (signs −, +).
                    let sum = cur[c.index()][g.index(p)]
                        - cur[c.index()][g.index(mb)]
                        - cur[b.index()][g.index(p)]
                        + cur[b.index()][g.index(mc)];
                    let cap = EPS0 * m.eps_r(a)[g.index(p)] * g.dual_face_area(a, p)
                        / g.edge_length(a, p);
                    y[a.index()][g.index(p)] = sum / cap;
                }
            }
            // The Rayleigh quotient in the capacitance's norm.
            let (mut num, mut den) = (0.0, 0.0);
            for a in Axis::ALL {
                for &p in &nodes {
                    if !live(a, p) {
                        continue;
                    }
                    let k = g.index(p);
                    let cap = EPS0 * m.eps_r(a)[k] * g.dual_face_area(a, p) / g.edge_length(a, p);
                    num += cap * u[a.index()][k] * y[a.index()][k];
                    den += cap * u[a.index()][k] * u[a.index()][k];
                }
            }
            lambda = num / den;
            let norm = y.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
            for (ua, ya) in u.iter_mut().zip(&y) {
                for (x, v) in ua.iter_mut().zip(ya) {
                    *x = v / norm;
                }
            }
        }
        0.25 * lambda
    }

    /// With the shares balanced the bound is the limit of a graded mesh
    /// in one medium, where the single cells with half of themselves
    /// leave a part of it; with a laminate and a ground plane it is
    /// above the limit and near it.
    #[test]
    fn balanced_shares_reach_the_limit_of_a_graded_mesh() {
        let um = 1e-6;
        let lines = |sizes: &[(f64, usize)]| {
            let mut x = vec![0.0];
            for &(d, count) in sizes {
                for _ in 0..count {
                    x.push(x.last().unwrap() + d * um);
                }
            }
            x
        };
        let g = Grid::new([
            lines(&[(83.0, 3), (250.0, 5), (83.0, 3)]),
            lines(&[(83.0, 3), (250.0, 2), (100.0, 4), (250.0, 2), (83.0, 3)]),
            lines(&[(125.0, 3), (35.0, 1), (174.0, 2), (300.0, 2)]),
        ])
        .unwrap();
        let walls = [[true; 2]; 3];
        for laminate in [false, true] {
            let mut m = EdgeMaterials::vacuum(&g);
            let n = g.nodes();
            if laminate {
                for a in Axis::ALL {
                    for i in 0..n[0] {
                        for j in 0..n[1] {
                            for k in 0..=3 {
                                // (The edges in the laminate's upper face
                                // have the mean of the two media.)
                                let eps = if k < 3 || a == Axis::Z { 4.3 } else { 2.65 };
                                if !(a == Axis::Z && k == 3) {
                                    m.eps_r_mut(a)[g.index([i, j, k])] = eps;
                                }
                            }
                        }
                    }
                }
                // A strip on the laminate.
                for i in 0..n[0] - 1 {
                    for j in 5..9 {
                        m.set_pec_edge(Axis::X, [i, j, 3]);
                        if j < 8 {
                            m.set_pec_edge(Axis::Y, [i, j, 3]);
                        }
                    }
                }
            }
            let truth = grid_eigenvalue(&g, &m, 6000);
            let mut cells = Cells::new(&g, &m, None).unwrap();
            let (plain, _) = cells.worst_exact(&walls, None);
            let (balanced, _) = cells.worst_balanced(&walls, None);
            let courant = 1.0 / courant_limit(&g).powi(2);
            // A bound: not below the operator's largest eigenvalue.
            assert!(balanced >= truth * (1.0 - 1e-6), "{balanced} {truth}");
            assert!(balanced <= plain);
            let gain = |v: f64| (courant / v).sqrt();
            let (of_truth, of_plain) = (gain(truth), gain(plain));
            let of_balanced = gain(balanced);
            if laminate {
                assert!(of_balanced > 0.97 * of_truth, "{of_balanced} {of_truth}");
            } else {
                assert!(of_balanced > 0.9999 * of_truth, "{of_balanced} {of_truth}");
            }
            assert!(of_balanced > 1.1 * of_plain, "{of_balanced} {of_plain}");
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
        let none = [[false; 2]; 3];
        let cells = Cells::new(&g, &m, Some(&live)).unwrap();
        let base = cells.value([1, 0, 0], &none);
        assert!((cells.value([0, 1, 1], &none) / base - 4.0).abs() < 1e-12);
        assert!((cells.value([0, 1, 0], &none) / base - 1.0).abs() < 1e-12);
        // PEC: left out. In the surface of lossy metal: vacuum on the
        // quarter of its dual face in the one cell around it.
        assert!((cells.value([0, 0, 0], &none) / base - 1.0).abs() < 1e-12);
        live.set(
            Axis::X,
            g.index([0, 0, 0]),
            LiveEdge::Solid { total: 0.25e-6 },
        );
        let cells = Cells::new(&g, &m, Some(&live)).unwrap();
        assert!((cells.value([0, 0, 0], &none) / base - 4.0).abs() < 1e-12);
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
