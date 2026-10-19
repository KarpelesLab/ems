//! Thin-wire model: straight, axis-aligned wires whose radius is smaller
//! than the cell size (design §5.14).
//!
//! # The problem
//!
//! A line of PEC edges (the staircase of a csx `Curve` or a thin `Wire`)
//! behaves like a round wire of a fixed radius `r_eq ≈ 0.2·Δ`, whatever
//! the radius of the real wire: the transverse field around the wire is
//! the field of a single node of the Yee lattice. Antenna impedances
//! depend on the wire radius through `ln(Δ/r0)`, so a dipole of radius
//! `Δ/100` and one of `Δ/5` come out the same.
//!
//! # The model
//!
//! Close to a thin wire the transverse fields of the wire current and
//! charge vary as `1/r` (the quasi-static field of a line current and a
//! line charge). The model corrects the lattice elements that connect the
//! wire to its first ring of neighbours so that the per-unit-length
//! capacitance and inductance of the wire are those of a round wire of
//! radius `r0`:
//!
//! * the four **radial E edges** leaving every interior wire node
//!   (transverse to the wire) get `ε → ε/g` (and `κ → κ/g`);
//! * the four **wire faces** of every wire segment (the primary faces
//!   that have the wire edge on their boundary, normal to the two
//!   transverse axes) get `μ → μ·g` (and `σ_m → σ_m·g`);
//! * the four **axial faces** around every interior wire node (normal to
//!   the wire, with a corner on it) also get `μ → μ·g`.
//!
//! **Why this works.** Per unit length, the transverse E field of the
//! lattice is a resistor network (edge conductance `ε·l*/l`), and the
//! transverse magnetic field is the same network for the flux function
//! `A_z` (link conductance `l*/(μ·l)` across each wire face). Both
//! networks are the 2-D lattice around the wire node. The potential of an
//! isolated node of the infinite lattice, relative to a point at distance
//! `ρ ≫ Δ`, is that of a round wire of radius `r_eq`:
//!
//! ```text
//! R(ρ) = (1/2π)·ln(ρ/r_eq)     (per unit conductivity),
//! ```
//!
//! where `r_eq` is the *lattice radius* of the node ([`lattice_radius`]:
//! `e^(−γ)/(2√2)·Δ = 0.1985·Δ` for square cells). Dividing the four
//! conductances from the wire node to its neighbours by `g` adds
//! `(g − 1)/Σc` in series (exact for square cells, where the four
//! neighbours are at one potential by symmetry; `Σc` is the sum of the
//! four conductances). Choosing
//!
//! ```text
//! g = 1 + Σc/(2π) · ln(r_eq/r0)
//! ```
//!
//! makes the node look like a wire of radius `r0`: `C' = 2πε/ln(ρ/r0)` and
//! `L' = μ/(2π)·ln(ρ/r0)`. Because the electric and magnetic networks get
//! the same factor, `L'·C' = μ·ε` is unchanged and the wave along the wire
//! still travels at the speed of the medium (on the lattice: with the
//! dispersion of the Yee line). For square cells `g = 1 + (2/π)·ln(r_eq/r0)`;
//! the classic contour-path thin-wire update (Umashankar and Taflove,
//! 1987), which assumes the `1/r` profile across the first cell, gives
//! `g = ln(Δ/r0)/2` instead, i.e. `r_eq = Δ·e⁻² = 0.135·Δ`, which ignores
//! the lattice and leaves the capacitance uncorrected.
//!
//! The model only changes material values, so the scheme stays the
//! symmetric Yee scheme: the energy invariant holds exactly and every
//! operator (lumped elements, CPML, dumps, ...) works unchanged. The
//! axial faces are not part of the per-unit-length analysis (for wire
//! currents the axial H near the wire is zero); scaling them keeps every
//! local loop of the modified elements at most as stiff as the unmodified
//! lattice (a radial edge with `C/g` sees only faces with `L·g` or
//! heavier, every other loop gets softer), so the usual Courant limit
//! still applies. Without them a radial edge would form a loop of
//! `C/g` with two unmodified faces, which is unstable at the Courant
//! limit for `g` of 2 and more.
//!
//! # Wire ends
//!
//! A discrete wire whose end node carries the full cell of wire behaves
//! as if it ended half a cell beyond that node (this is also what a plain
//! staircase does). Moreover the E edge continuing the wire beyond its end
//! (the *cap edge*) has the lattice capacitance `ε·A*/l`, which loads the
//! end like a top hat: for a staircase it is the end capacitance of a
//! wire of radius `r_eq`, for a thinner wire it is far too large. An
//! **open end** (the cap edge exists and is not PEC) therefore gets:
//!
//! * half a cell of wire at the end node: its radial edges `ε/(2g)`, its
//!   axial faces and the faces of the end segment `μ·2g` (the doubled
//!   inductance of the end segment matters little: the current vanishes
//!   at an open end);
//! * the cap edge scaled like a lattice whose staircase radius is `r0`:
//!   `ε → ε·r0/r_eq`, its four faces `μ·max(2g, r_eq/r0)` (for
//!   stability, as above).
//!
//! The open end of the model is then at the end node, as drawn. Ends on
//! the domain boundary or continuing into metal (a monopole on a ground
//! plane) keep the end node's radial edges and axial faces.
//!
//! **Validation** (`tests/thin_wire*.rs`): wire in a square tube, `Z0`
//! within 6·10⁻⁵ of `Z0(staircase) + η/(2π)·ln(r_eq/r0)` for
//! `r_eq/r0` = 5…1000 (and within 0.1 % of the closed form for a round
//! wire in a square tube), the wave speed that of the Yee line; dipoles on
//! 2–40× coarser meshes than a staircase reference of the same radius
//! within 0.7 % in resonant frequency and input resistance.
//!
//! # Scope and validity
//!
//! * Straight wires along a mesh axis, on a mesh line (`r0` given per
//!   wire); wires of other orientations or positions:
//!   [`super::oblique_wire`]. The transverse cell sizes next to the wire may differ (graded
//!   mesh, rectangular cells); the lattice radius then uses the harmonic
//!   mean of the two cell sizes per axis, and `(g − 1)/Σc` is an
//!   approximation (0.4 % of `Z0` in a tube of 1 × 1.5 mm cells).
//!   The surrounding medium may be any (isotropic or diagonal) material:
//!   the factors scale its values.
//! * `0 < r0 < r_eq`: thinner than the staircase (`g > 1`). Thicker
//!   wires are better modelled as volumes.
//! * Radial edges that are PEC are skipped. Wires closer than about three
//!   cells to each other, or to another conductor, are outside the model's
//!   assumptions (the `1/r` field of an isolated wire), and so are wire
//!   bends and junctions (each straight run is corrected on its own).
//! * The model corrects the coupling of the wire current and charge;
//!   fields polarised across the wire (E perpendicular, H along the wire)
//!   see a small artificial rod of `ε/g`, `μ·g` of about one cell cross
//!   section, which a real thin wire does not present.
//!
//! The wire edges themselves must be PEC (or carry a lumped element, e.g.
//! the feed gap of a dipole: include the gap in the wire so that its faces
//! are corrected too and the nodes next to it are interior nodes);
//! [`ThinWire::apply`] does not change PEC flags, see
//! [`ThinWire::set_pec`].

use std::collections::BTreeMap;

use super::edge_material::EdgeMaterials;
use super::grid::Grid;
use crate::axis::Axis;
use crate::{Error, Result};

/// Lattice radius of a node of the infinite 2-D rectangular lattice with
/// cell sizes `hx`, `hy` \[m\]: the radius of the round wire whose
/// potential (relative to points far away) equals that of a single node
/// driven with the same charge. For square cells it is
/// `e^(−γ)/(2√2)·h ≈ 0.19851·h`.
///
/// Computed from the lattice Green's function,
///
/// ```text
/// G(0) − G(m, 0) = (1/2π) ∫₀^{2π} (1 − ρ(θ)^m) / √(A² − B²) dθ,
/// A = 2cx + 2cy(1 − cos θ),  B = 2cx,  ρ = (A − √(A² − B²))/B,
/// ```
///
/// with conductances `cx = hy/hx`, `cy = hx/hy`, at `m = 64` and `128`
/// cells (Richardson-extrapolated in `1/m²`), and
/// `G(0) − G(m, 0) → (1/2π)·ln(m·hx/r_eq)`.
///
/// # Panics
/// If `hx` or `hy` is not finite and positive.
pub fn lattice_radius(hx: f64, hy: f64) -> f64 {
    assert!(
        hx.is_finite() && hx > 0.0 && hy.is_finite() && hy > 0.0,
        "cell sizes must be finite and positive"
    );
    let (cx, cy) = (hy / hx, hx / hy);
    let r = |m: f64| {
        // ∫₀^π (θ symmetric), with θ = π·e^(−s): dθ = θ ds. The integrand
        // times θ vanishes like θ as θ → 0.
        let f = |th: f64| {
            let a = 2.0 * cx + 2.0 * cy * (1.0 - th.cos());
            let b = 2.0 * cx;
            // A − B = 2cy(1 − cos θ) = 4cy·sin²(θ/2), exactly.
            let amb = 4.0 * cy * (0.5 * th).sin().powi(2);
            let root = (amb * (a + b)).sqrt();
            let rho = (a - root) / b;
            (1.0 - rho.powf(m)) / root * th
        };
        let s_max = (std::f64::consts::PI * m * 1e9).ln();
        let n = 8000; // even
        let h = s_max / n as f64;
        let mut sum = f(std::f64::consts::PI) + f(std::f64::consts::PI * (-s_max).exp());
        for i in 1..n {
            let th = std::f64::consts::PI * (-(i as f64) * h).exp();
            sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f(th);
        }
        // (1/2π)·2·∫₀^π.
        sum * h / 3.0 / std::f64::consts::PI
    };
    let ln_ratio = |m: f64| (m * hx).ln() - 2.0 * std::f64::consts::PI * r(m);
    let (l1, l2) = (ln_ratio(64.0), ln_ratio(128.0));
    // ln(r_eq) + c/m²: extrapolate.
    ((4.0 * l2 - l1) / 3.0).exp()
}

/// A straight thin wire along a mesh axis; see the [module docs](self).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThinWire {
    axis: Axis,
    start: [usize; 3],
    segments: usize,
    radius: f64,
}

impl ThinWire {
    /// The wire along `axis` from node `start` over `segments` edges (to
    /// node `start + segments·axis`), with radius `radius` \[m\].
    pub fn new(axis: Axis, start: [usize; 3], segments: usize, radius: f64) -> Self {
        Self {
            axis,
            start,
            segments,
            radius,
        }
    }

    /// The wire axis.
    pub fn axis(&self) -> Axis {
        self.axis
    }

    /// First node.
    pub fn start(&self) -> [usize; 3] {
        self.start
    }

    /// Number of wire edges.
    pub fn segments(&self) -> usize {
        self.segments
    }

    /// Last node.
    pub fn end(&self) -> [usize; 3] {
        let mut e = self.start;
        e[self.axis.index()] += self.segments;
        e
    }

    /// Wire radius \[m\].
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// The transverse cell sizes `(h−, h+)` around the wire along the
    /// transverse axis `t`, a missing side mirrored from the other (a
    /// wire on a symmetry plane of the domain).
    fn sides(&self, grid: &Grid, t: Axis) -> Result<(f64, f64)> {
        let i = self.start[t.index()];
        let n = grid.nodes()[t.index()];
        let lo = (i > 0).then(|| grid.delta(t, i - 1));
        let hi = (i + 1 < n).then(|| grid.delta(t, i));
        match (lo, hi) {
            (Some(a), Some(b)) => Ok((a, b)),
            (Some(a), None) => Ok((a, a)),
            (None, Some(b)) => Ok((b, b)),
            (None, None) => Err(Error::Invalid(format!(
                "thin wire: the grid has a single line along {t}"
            ))),
        }
    }

    fn check(&self, grid: &Grid) -> Result<()> {
        let n = grid.nodes();
        let a = self.axis.index();
        if self.segments == 0 || (0..3).any(|d| self.start[d] >= n[d]) {
            return Err(Error::Invalid(format!(
                "thin wire at {:?} with {} segments is empty or outside the grid",
                self.start, self.segments
            )));
        }
        if self.start[a] + self.segments >= n[a] {
            return Err(Error::Invalid(format!(
                "thin wire from {:?} with {} segments along {} leaves the grid",
                self.start, self.segments, self.axis
            )));
        }
        if !(self.radius.is_finite() && self.radius > 0.0) {
            return Err(Error::Invalid(format!(
                "thin wire radius {} must be finite and positive",
                self.radius
            )));
        }
        Ok(())
    }

    /// The lattice radius `r_eq` \[m\] of the wire's transverse position
    /// (the radius a plain staircase wire there models), see
    /// [`lattice_radius`].
    ///
    /// # Errors
    /// [`Error::Invalid`] if the wire is outside the grid.
    pub fn staircase_radius(&self, grid: &Grid) -> Result<f64> {
        self.check(grid)?;
        let (b, c) = self.axis.others();
        let (b0, b1) = self.sides(grid, b)?;
        let (c0, c1) = self.sides(grid, c)?;
        let harm = |x: f64, y: f64| 2.0 / (1.0 / x + 1.0 / y);
        Ok(lattice_radius(harm(b0, b1), harm(c0, c1)))
    }

    /// The correction factor `g` (see the [module docs](self)); `g > 1`
    /// for wires thinner than the [staircase radius](Self::staircase_radius).
    ///
    /// # Errors
    /// [`Error::Invalid`] if the wire is outside the grid or its radius is
    /// not positive.
    pub fn factor(&self, grid: &Grid) -> Result<f64> {
        let r_eq = self.staircase_radius(grid)?;
        let (b, c) = self.axis.others();
        let (b0, b1) = self.sides(grid, b)?;
        let (c0, c1) = self.sides(grid, c)?;
        // Conductances (per unit length and conductivity) of the four
        // radial edges: transverse dual length over edge length.
        let (lb, lc) = (0.5 * (b0 + b1), 0.5 * (c0 + c1));
        let sum_c = lc * (1.0 / b0 + 1.0 / b1) + lb * (1.0 / c0 + 1.0 / c1);
        Ok(1.0 + sum_c / (2.0 * std::f64::consts::PI) * (r_eq / self.radius).ln())
    }

    /// Applies the model to `mats` (see the [module docs](self)) and
    /// returns the factor `g`.
    ///
    /// An end of the wire is *open* when the edge continuing the wire
    /// beyond it exists and is not PEC; it then gets the half-cell end
    /// treatment of the [module docs](self#wire-ends). Other ends (on the
    /// domain boundary, or continuing into metal) keep the end node's
    /// radial edges and axial faces.
    ///
    /// # Errors
    /// [`Error::Invalid`] if the wire is outside the grid, `mats` does not
    /// match it, or the radius is not below the
    /// [staircase radius](Self::staircase_radius) (`g ≤ 1`).
    pub fn apply(&self, grid: &Grid, mats: &mut EdgeMaterials) -> Result<f64> {
        if mats.nodes() != grid.nodes() {
            return Err(Error::Invalid(format!(
                "edge material map has {:?} nodes, grid has {:?}",
                mats.nodes(),
                grid.nodes()
            )));
        }
        let g = self.factor(grid)?;
        let r_eq = self.staircase_radius(grid)?;
        if g <= 1.0 {
            return Err(Error::Invalid(format!(
                "thin wire radius {:e} m is not below the staircase radius {r_eq:e} m",
                self.radius
            )));
        }
        self.scale(grid, mats, g, r_eq, false);
        Ok(g)
    }

    /// Undoes [`apply`](Self::apply) on `mats` (same PEC flags), e.g. to
    /// look at the materials without the thin-wire correction.
    ///
    /// # Errors
    /// As [`apply`](Self::apply).
    pub fn remove(&self, grid: &Grid, mats: &mut EdgeMaterials) -> Result<()> {
        let g = self.factor(grid)?;
        let r_eq = self.staircase_radius(grid)?;
        if g <= 1.0 || mats.nodes() != grid.nodes() {
            return Err(Error::Invalid("thin wire: nothing to remove".into()));
        }
        self.scale(grid, mats, g, r_eq, true);
        Ok(())
    }

    /// Multiplies (or divides) the elements of the model.
    fn scale(&self, grid: &Grid, mats: &mut EdgeMaterials, g: f64, r_eq: f64, inverse: bool) {
        let (edges, faces) = self.plan(grid, mats, g, self.radius / r_eq);
        let inv = |f: f64| if inverse { 1.0 / f } else { f };
        for ((a, node), f) in edges {
            let f = inv(f);
            let t = Axis::ALL[a];
            if !mats.is_pec_edge(t, node) {
                let i = grid.index(node);
                let (e, k) = (mats.eps_r(t)[i], mats.kappa(t)[i]);
                mats.set_edge(t, node, e * f, k * f);
            }
        }
        for ((a, node), f) in faces {
            let f = inv(f);
            let t = Axis::ALL[a];
            let i = grid.index(node);
            let (m, s) = (mats.mu_r(t)[i], mats.sigma_m(t)[i]);
            mats.set_face(t, node, m * f, s * f);
        }
    }

    /// Whether the start and the end of the wire are open (see
    /// [`apply`](Self::apply)) with the PEC flags of `mats`.
    ///
    /// # Panics
    /// If the wire is outside the grid of `mats`.
    pub fn open_ends(&self, mats: &EdgeMaterials) -> [bool; 2] {
        let a = self.axis.index();
        let n = mats.nodes();
        let start = self.start[a] > 0 && {
            let mut q = self.start;
            q[a] -= 1;
            !mats.is_pec_edge(self.axis, q)
        };
        let end =
            self.start[a] + self.segments + 1 < n[a] && !mats.is_pec_edge(self.axis, self.end());
        [start, end]
    }

    /// Marks every wire edge PEC except the edges at the given offsets
    /// along the wire (`0..segments`), e.g. a feed gap.
    ///
    /// # Panics
    /// If the wire is outside the grid of `mats`.
    pub fn set_pec(&self, mats: &mut EdgeMaterials, except: &[usize]) {
        for s in (0..self.segments).filter(|s| !except.contains(s)) {
            mats.set_pec_edge(self.axis, self.node(s as isize));
        }
    }

    /// The node `s` steps from the start along the wire (may be before
    /// the start or after the end; the caller checks the grid).
    fn node(&self, s: isize) -> [usize; 3] {
        let mut p = self.start;
        let a = self.axis.index();
        p[a] = (p[a] as isize + s) as usize;
        p
    }

    /// The scale factors of the E edges (`ε`, `κ`) and faces (`μ`, `σ_m`)
    /// of the model, keyed by `(axis index, node)`, with `c = r0/r_eq`.
    #[allow(clippy::type_complexity)]
    fn plan(
        &self,
        grid: &Grid,
        mats: &EdgeMaterials,
        g: f64,
        c: f64,
    ) -> (
        BTreeMap<(usize, [usize; 3]), f64>,
        BTreeMap<(usize, [usize; 3]), f64>,
    ) {
        let n = grid.nodes();
        let a = self.axis.index();
        let (b, cc) = self.axis.others();
        let mut edges: BTreeMap<(usize, [usize; 3]), f64> = BTreeMap::new();
        let mut faces: BTreeMap<(usize, [usize; 3]), f64> = BTreeMap::new();
        // Edges take the smallest factor, faces the largest, when an
        // element belongs to several parts (e.g. a one-segment wire).
        let mut edge = |t: usize, p: [usize; 3], f: f64| {
            let e = edges.entry((t, p)).or_insert(f);
            *e = e.min(f);
        };
        let mut face = |t: usize, p: [usize; 3], f: f64| {
            let e = faces.entry((t, p)).or_insert(f);
            *e = e.max(f);
        };
        // The two transverse edges or faces at offset 0 and −1 along `u`
        // of node `p`, where they exist.
        let pair = |p: [usize; 3], u: usize| {
            let fwd = (p[u] + 1 < n[u]).then_some(p);
            let back = (p[u] > 0).then(|| {
                let mut q = p;
                q[u] -= 1;
                q
            });
            fwd.into_iter().chain(back)
        };
        let radial = |p: [usize; 3]| {
            [b.index(), cc.index()]
                .into_iter()
                .flat_map(move |t| pair(p, t).map(move |q| (t, q)))
        };
        // Faces normal to `t` along segment `p`: corner on the wire or one
        // cell back along the other transverse axis `u`.
        let seg_faces = |p: [usize; 3]| {
            [(b.index(), cc.index()), (cc.index(), b.index())]
                .into_iter()
                .flat_map(move |(t, u)| pair(p, u).map(move |q| (t, q)))
        };
        let axial = |p: [usize; 3]| {
            pair(p, b.index())
                .flat_map(move |q| pair(q, cc.index()))
                .filter(move |q| {
                    q[b.index()] + 1 < n[b.index()] && q[cc.index()] + 1 < n[cc.index()]
                })
                .map(move |q| (a, q))
        };
        let [open_start, open_end] = self.open_ends(mats);
        let last = self.segments as isize - 1;
        for s in 1..self.segments as isize {
            for (t, q) in radial(self.node(s)) {
                edge(t, q, 1.0 / g);
            }
            for (t, q) in axial(self.node(s)) {
                face(t, q, g);
            }
        }
        for s in 0..=last {
            let half = (s == 0 && open_start) || (s == last && open_end);
            for (t, q) in seg_faces(self.node(s)) {
                face(t, q, if half { 2.0 * g } else { g });
            }
        }
        // Open ends: half a cell of wire at the end node, the cap edge
        // beyond it scaled like a lattice of radius r0, and its faces
        // made at least as heavy as needed for stability.
        let caps = [
            (open_start, 0, -1),
            (open_end, self.segments as isize, self.segments as isize),
        ];
        for (open, end, beyond) in caps {
            if !open {
                continue;
            }
            for (t, q) in radial(self.node(end)) {
                edge(t, q, 0.5 / g);
            }
            for (t, q) in axial(self.node(end)) {
                face(t, q, 2.0 * g);
            }
            edge(a, self.node(beyond), c);
            for (t, q) in seg_faces(self.node(beyond)) {
                face(t, q, (2.0 * g).max(1.0 / c));
            }
        }
        (edges, faces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_lattice_radius() {
        // e^(−γ)/(2√2).
        let want = (-core::f64::consts::EULER_GAMMA).exp() / (2.0 * 2f64.sqrt());
        let r = lattice_radius(1.0, 1.0);
        assert!((r - want).abs() < 1e-6 * want, "{r} vs {want}");
        // Scales with the cell size.
        assert!((lattice_radius(2e-3, 2e-3) / 2e-3 - want).abs() < 1e-6 * want);
    }

    #[test]
    fn rectangular_lattice_radius_is_symmetric() {
        let (a, b) = (lattice_radius(1.0, 2.0), lattice_radius(2.0, 1.0));
        assert!((a - b).abs() < 1e-6 * a, "{a} {b}");
        // Between the two square lattices.
        assert!(a > lattice_radius(1.0, 1.0) && a < lattice_radius(2.0, 2.0));
    }

    #[test]
    fn factor_and_elements() {
        let grid = Grid::uniform([10, 10, 10], [1e-3; 3]).unwrap();
        let r_eq = lattice_radius(1e-3, 1e-3);
        let w = ThinWire::new(Axis::Z, [5, 5, 2], 4, r_eq / 10.0);
        let g = w.factor(&grid).unwrap();
        assert!((g - (1.0 + 2.0 / std::f64::consts::PI * 10f64.ln())).abs() < 1e-12);
        let mut m = EdgeMaterials::vacuum(&grid);
        w.set_pec(&mut m, &[2]);
        assert!(m.is_pec_edge(Axis::Z, [5, 5, 2]) && !m.is_pec_edge(Axis::Z, [5, 5, 4]));
        assert_eq!(w.open_ends(&m), [true, true]);
        let (edges, faces) = w.plan(&grid, &m, g, 0.1);
        // 3 interior + 2 end nodes x 4 radial edges, plus 2 cap edges.
        assert_eq!(edges.len(), 5 * 4 + 2);
        // 4 + 2 segments x 4 faces, plus 5 nodes x 4 axial faces.
        assert_eq!(faces.len(), 6 * 4 + 5 * 4);
        assert_eq!(w.apply(&grid, &mut m).unwrap(), g);
        let eps = |t: Axis, p: [usize; 3]| m.eps_r(t)[grid.index(p)];
        let mu = |t: Axis, p: [usize; 3]| m.mu_r(t)[grid.index(p)];
        assert!((eps(Axis::X, [4, 5, 3]) - 1.0 / g).abs() < 1e-15);
        // Open end nodes: half a cell.
        assert!((eps(Axis::X, [5, 5, 2]) - 0.5 / g).abs() < 1e-15);
        assert!((eps(Axis::Y, [5, 5, 6]) - 0.5 / g).abs() < 1e-15);
        // Cap edges beyond the ends, and their faces.
        assert!((eps(Axis::Z, [5, 5, 1]) - 0.1).abs() < 1e-15);
        assert!((eps(Axis::Z, [5, 5, 6]) - 0.1).abs() < 1e-15);
        assert_eq!(mu(Axis::Y, [5, 5, 1]), 10.0);
        // Wire faces: end segments doubled.
        assert_eq!(mu(Axis::Y, [5, 5, 2]), 2.0 * g);
        assert_eq!(mu(Axis::X, [5, 4, 3]), g);
        assert_eq!(mu(Axis::X, [5, 4, 5]), 2.0 * g);
        // Axial faces.
        assert_eq!(mu(Axis::Z, [4, 4, 3]), g);
        assert_eq!(mu(Axis::Z, [4, 4, 2]), 2.0 * g);
        assert_eq!(mu(Axis::Z, [4, 4, 1]), 1.0);
        // The gap edge is not PEC, the others are; a wire ending in metal
        // has closed ends.
        let mut m2 = EdgeMaterials::vacuum(&grid);
        m2.set_pec_edge(Axis::Z, [5, 5, 1]);
        assert_eq!(w.open_ends(&m2), [false, true]);
        // Too thick, outside.
        let thick = ThinWire::new(Axis::Z, [5, 5, 2], 4, 0.3e-3);
        assert!(thick.apply(&grid, &mut m).is_err());
        assert!(
            ThinWire::new(Axis::Z, [5, 5, 2], 9, 1e-5)
                .factor(&grid)
                .is_err()
        );
    }
}
