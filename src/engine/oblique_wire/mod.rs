//! Oblique thin wires: wires of any orientation and position on a
//! Cartesian grid, as a one-dimensional wire network coupled to the Yee
//! lattice (design §5.14.1).
//!
//! The axis-aligned model of [`super::wire`] changes edge materials along
//! a line of mesh edges; it cannot follow a wire that leaves the mesh
//! lines. Here the wire carries its own unknowns instead, in the spirit of
//! the thin-wire models of Holland and Simpson (1981) and of the hybrid
//! wire models of Edelvik et al. (2003): a current `I_s` per wire segment
//! and a correction potential `V_j` per wire node, on a polyline that is
//! independent of the mesh.
//!
//! # Coupling
//!
//! Each segment `s` (straight, from node `a` to node `b`) is coupled to the
//! E edges through the lowest-order **Whitney edge functions**
//! (`deposit.rs`): `P_{e,s} = ∫_s w_e·dl`, with `w_e` constant along edge
//! `e` and bilinear across it (the standard trilinear interpolation of the
//! Yee E components). The *same* matrix is used both ways:
//!
//! * the wire sees the grid voltage `(Pᵀu)_s = ∫_s E_interp·dl`;
//! * the grid sees the wire current as the source `−P·I` in Ampère's law,
//!   `C·du/dt = Kᵀi − P·I`.
//!
//! Because `∇λ_n` (trilinear node hat) is the sum of the Whitney functions
//! of the edges at node `n`, the deposited current is **charge
//! conserving**: the charge it leaves on the grid nodes is exactly the
//! wire's node charges spread with the trilinear hats.
//!
//! # Wire equations
//!
//! With `L_s` the correction inductance of segment `s`, `C_j` the
//! correction capacitance of node `j`, and `G` the wire's node-to-segment
//! incidence (`(G·V)_s = V_b − V_a`),
//!
//! ```text
//! L_s·dI_s/dt + R_s·I_s = (Pᵀu)_s − (G·V)_s + V_src,s
//! C_j·dV_j/dt          = (Gᵀ·I)_j  (current into node j)
//! ```
//!
//! (`R_s`, `V_src,s` only on a [feed](WireFeed) segment.) Nodes where a
//! wire ends on a conductor (PEC edge, lumped element, …) have `V = 0`;
//! open ends and junctions of several wires are ordinary nodes.
//!
//! # Time stepping and stability
//!
//! `I` and `V` live at integer times and are advanced with the
//! **trapezoidal rule, implicitly coupled with the E update** (the
//! semi-implicit lumped-branch scheme of [`super::lumped`], for a whole
//! network): with `u* = ca·uⁿ + cb·Kᵀi^{n+½}` the kernel result and
//! `Ī = (Iⁿ + I^{n+1})/2`,
//!
//! ```text
//! u^{n+1} = u* − cb·P·Ī
//! [2L/dt + R + ½·PᵀD_cb·P + (dt/2)·G·C⁻¹·Gᵀ]·Ī = (2L/dt)·Iⁿ + Pᵀ(u* + uⁿ)/2 − G·Vⁿ + V_src
//! I^{n+1} = 2Ī − Iⁿ,   V^{n+1} = Vⁿ + dt·C⁻¹·Gᵀ·Ī
//! ```
//!
//! The matrix is symmetric positive definite, constant, and banded along
//! the wire; it is factored once (envelope Cholesky) at the first step.
//! Multiplying the E update by `ū = (uⁿ + u^{n+1})/2` and the wire
//! equations by `Ī`, `V̄` shows that the coupling terms cancel exactly:
//!
//! ```text
//! (Wⁿ⁺¹ + W_wⁿ⁺¹) − (Wⁿ + W_wⁿ) = dt·Ī·V_src − dt·R·Ī²  (lossless grid),
//! W_w = ½·Σ L_s·I_s² + ½·Σ C_j·V_j²
//! ```
//!
//! with `W` the leapfrog field invariant of [`Engine::energy`](super::Engine::energy).
//! `W` is positive definite under the usual Courant condition and `W_w`
//! always, so the coupled scheme is stable with the **unchanged Yee time
//! step**, for any wire radius, segment length and orientation. The
//! operator reports `W_w` through [`Operator::energy`].
//!
//! # In-cell corrections
//!
//! The grid already carries part of the wire's field: the field of the
//! deposited current and charge, i.e. of a wire smeared over about one
//! cell. `L_s` and `C_j` supply the rest, the field between the real
//! radius `r0` and the lattice's own representation, so that the total is
//! the thin-wire (reduced-kernel) inductance and elastance of a round wire
//! of radius `r0`. They are computed per segment and per node from the
//! quasi-static response of the lattice (`correction.rs`), for the wire
//! carrying a uniform current and charge density within a window of five
//! cell sizes:
//!
//! ```text
//! k(x) = φ_cont(x) − φ_lat(x),   C_j = ε·∫ ds/k(s)  (over the half segments at j)
//! L_s  = μ·[Λ_cont,s − Λ_lat,s]
//! ```
//!
//! * `φ_cont(x) = (1/4π)·∫ ds/√(|x(s) − x|² + r0²)` over the window
//!   (each node's charge `ℓ_j` spread over its half segments), and
//!   `Λ_cont,s = (1/4π)·∫_s∫ t̂·t̂'/√(|x − x'|² + r0²)`, the reduced
//!   thin-wire kernel;
//! * `φ_lat`, `Λ_lat` are the same quantities seen through the lattice:
//!   the node charges spread with the trilinear hats (resp. the Whitney
//!   current deposit), propagated with the lattice Green's function of
//!   the discrete Laplacian (`lattice.rs`; for static fields the Yee
//!   lattice is exactly `−Δ_h φ = ρ/ε` for the charge and `−Δ_h A = μ·J`
//!   per component in the Coulomb gauge, on a uniform lattice), and
//!   sampled back with the same weights.
//!
//! Far from `x_j` both agree, so the difference converges quickly with
//! the window. The lattice is taken uniform with the spacings of the cell
//! that contains the node (segment midpoint), which is exact on uniform
//! meshes and an approximation on graded ones. For a straight wire along
//! a mesh line this gives the lattice radius `r_eq = 0.1985·Δ` of
//! [`super::wire::lattice_radius`] (the corrections are then
//! `(1/2π)·ln(r_eq/r0)` per unit length, as in the material model); at a
//! cell centre the smeared wire looks like a wire of `0.72·Δ`, and an
//! oblique wire passes through all positions in between, which is why the
//! correction is computed per element.
//!
//! The node capacitance integrates `1/k` over the node's half segments
//! (three Gauss points each) instead of sampling `k` at the node: at an
//! open end `k` falls to about half its interior value (the near field is
//! one-sided), and the end-point value would give the end node the
//! capacitance of a whole interior node, i.e. half a segment of extra
//! wire (a resonance error first order in the segment length).
//!
//! The model needs both corrections positive: the wire must be thinner
//! than the lattice's representation everywhere (roughly `r0 < 0.15·Δ`).
//!
//! **Validation** (design §5.14.1, `tests/oblique_wire*.rs`): a 170 mm
//! dipole on 10 mm cells along z (on a mesh line and at a cell centre),
//! tilted 30° and 45° in a plane and along (1, 2, 3) off the mesh lines
//! matches the axis-aligned material model of the same wire within
//! 1.05 % in resonance, 1.6 % in input resistance and 1.7 % in reactance
//! slope (spread over the orientations 0.5 %); the energy of a closed
//! cavity with bent, junctioned and grounded wires is conserved to
//! 3·10⁻⁸ over 15 000 steps at Courant 0.999.
//!
//! # Cylindrical grids
//!
//! On a `(ρ, α, z)` grid the wire points are Cartesian and the coupling,
//! the copies of seam and axis edges and the in-cell corrections follow
//! the rules of the `cyl` submodule (src/engine/oblique_wire/cyl.rs):
//! Whitney forms in parameter space along each segment's image, the
//! canonical copy of an edge with the copies' merged `cb`, and a locally
//! uniform lattice of the physical spacings `(Δρ, ρ·Δα, Δz)` for the
//! corrections. Wires must stay out of the first radial cell around the
//! axis.
//!
//! # Scope
//!
//! * Cartesian and cylindrical grids. The wire may cross any cells; it
//!   should stay a few cells away from absorbing boundaries (the
//!   correction line is not absorbed by the PML).
//! * A homogeneous medium around the wire (the local `ε_r`, `μ_r` are
//!   averaged over the coupled edges).
//! * Wires closer than a few cells to each other or to other conductors
//!   are outside the isolated-wire assumption of the corrections (the
//!   grid still couples them).
//! * Ends: an end on a conductor edge (PEC, a lumped element given as a
//!   contact) is connected (`V = 0`); every other end is open, with the
//!   open end at the end node. Coincident end points of different wires
//!   form a junction (Kirchhoff's current law at the node).

mod correction;
mod cyl;
mod deposit;
mod lattice;

use std::collections::BTreeMap;

use super::edge_material::EdgeMaterials;
use super::fields::Fields;
use super::grid::Grid;
use super::operator::{Hooks, Operator, StepContext};
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};
use crate::post::Complex64;
use crate::waveform::Waveform;
use crate::{Error, Result};

use correction::{Geometry, norm};

/// Radius of the correction window, in local cell sizes.
const WINDOW: f64 = 5.0;

/// A feed in one segment of an [`ObliqueWire`]: a series resistance and an
/// optional voltage source (a delta-gap source with internal resistance).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WireFeed {
    segment: usize,
    resistance: f64,
    source: Option<(Waveform, f64)>,
}

/// A thin wire: a polyline with a radius, optionally with a feed (see the
/// [module docs](self)).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ObliqueWire {
    points: Vec<[f64; 3]>,
    radius: f64,
    feed: Option<WireFeed>,
    max_segment: Option<f64>,
}

impl ObliqueWire {
    /// The wire through `points` \[m\] with radius `radius` \[m\].
    pub fn new(points: impl IntoIterator<Item = [f64; 3]>, radius: f64) -> Self {
        Self {
            points: points.into_iter().collect(),
            radius,
            feed: None,
            max_segment: None,
        }
    }

    /// Puts a feed of internal resistance `resistance` \[Ω\] into the
    /// polyline segment `segment` (from `points[segment]` to
    /// `points[segment + 1]`), which becomes a single wire segment (the
    /// feed gap). Without [`excite`](Self::excite) it is a lumped
    /// resistor.
    pub fn feed(mut self, segment: usize, resistance: f64) -> Self {
        self.feed = Some(WireFeed {
            segment,
            resistance,
            source: None,
        });
        self
    }

    /// Drives the feed with `amplitude·waveform(t)` \[V\] (an EMF along the
    /// wire direction, in series with the feed resistance).
    ///
    /// # Panics
    /// If no [`feed`](Self::feed) was set.
    pub fn excite(mut self, waveform: Waveform, amplitude: f64) -> Self {
        let f = self.feed.as_mut().expect("excite needs a feed");
        f.source = Some((waveform, amplitude));
        self
    }

    /// Longest wire segment \[m\] (default: half the smallest cell size
    /// at the polyline segment).
    pub fn max_segment(mut self, length: f64) -> Self {
        self.max_segment = Some(length);
        self
    }

    /// The polyline \[m\].
    pub fn points(&self) -> &[[f64; 3]] {
        &self.points
    }

    /// The radius \[m\].
    pub fn radius(&self) -> f64 {
        self.radius
    }

    fn validate(&self) -> Result<()> {
        if self.points.len() < 2 || self.points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::Invalid(
                "oblique wire needs at least two finite points".into(),
            ));
        }
        if !(self.radius.is_finite() && self.radius > 0.0) {
            return Err(Error::Invalid(format!(
                "oblique wire radius {} must be finite and positive",
                self.radius
            )));
        }
        if let Some(f) = &self.feed {
            if f.segment + 1 >= self.points.len() {
                return Err(Error::Invalid(format!(
                    "wire feed segment {} out of range",
                    f.segment
                )));
            }
            if !(f.resistance.is_finite() && f.resistance >= 0.0) {
                return Err(Error::Invalid(format!(
                    "wire feed resistance {} must be finite and non-negative",
                    f.resistance
                )));
            }
            if let Some((w, a)) = &f.source {
                w.validate()?;
                if !a.is_finite() {
                    return Err(Error::Invalid("wire feed amplitude is not finite".into()));
                }
            }
        }
        if let Some(l) = self.max_segment
            && !(l.is_finite() && l > 0.0)
        {
            return Err(Error::Invalid(format!(
                "wire max segment length {l} must be finite and positive"
            )));
        }
        Ok(())
    }
}

/// State and records of one feed.
#[derive(Debug, Clone)]
struct FeedState {
    segment: usize,
    resistance: f64,
    source: Option<(Waveform, f64)>,
    voltage: Vec<f64>,
    current: Vec<f64>,
}

/// Envelope (profile) Cholesky factor of a symmetric positive definite
/// matrix: row `i` stores columns `first[i]..=i`.
#[derive(Debug, Clone)]
struct Envelope {
    first: Vec<usize>,
    start: Vec<usize>,
    data: Vec<f64>,
}

impl Envelope {
    /// Factors the matrix given by its lower-triangle entries `(i, j)`,
    /// `j ≤ i`.
    fn factor(n: usize, lower: &BTreeMap<(usize, usize), f64>) -> Result<Self> {
        let mut first: Vec<usize> = (0..n).collect();
        for &(i, j) in lower.keys() {
            first[i] = first[i].min(j);
        }
        let mut start = Vec::with_capacity(n + 1);
        let mut len = 0;
        for (i, &f) in first.iter().enumerate() {
            start.push(len);
            len += i - f + 1;
        }
        let mut data = vec![0.0; len];
        for (&(i, j), &v) in lower {
            data[start[i] + j - first[i]] += v;
        }
        let mut m = Self { first, start, data };
        for i in 0..n {
            for j in m.first[i]..=i {
                let k0 = m.first[i].max(m.first[j]);
                let mut s = m.at(i, j);
                for k in k0..j {
                    s -= m.at(i, k) * m.at(j, k);
                }
                if j < i {
                    let d = m.at(j, j);
                    m.set(i, j, s / d);
                } else {
                    if !(s > 0.0 && s.is_finite()) {
                        return Err(Error::Invalid(
                            "oblique wire: the coupled wire system is not positive definite".into(),
                        ));
                    }
                    m.set(i, i, s.sqrt());
                }
            }
        }
        Ok(m)
    }

    fn at(&self, i: usize, j: usize) -> f64 {
        self.data[self.start[i] + j - self.first[i]]
    }

    fn set(&mut self, i: usize, j: usize, v: f64) {
        let k = self.start[i] + j - self.first[i];
        self.data[k] = v;
    }

    /// Solves `A·x = b` in place.
    #[allow(clippy::needless_range_loop)] // triangular index ranges
    fn solve(&self, x: &mut [f64]) {
        let n = x.len();
        for i in 0..n {
            let mut s = x[i];
            for k in self.first[i]..i {
                s -= self.at(i, k) * x[k];
            }
            x[i] = s / self.at(i, i);
        }
        for i in (0..n).rev() {
            x[i] /= self.at(i, i);
            let xi = x[i];
            for k in self.first[i]..i {
                x[k] -= self.at(i, k) * xi;
            }
        }
    }
}

/// The factored system and the edge coefficients it was built with.
#[derive(Debug, Clone)]
struct Solver {
    dt: f64,
    cb: Vec<f64>,
    chol: Envelope,
}

/// The oblique thin-wire operator: every [`ObliqueWire`] of a model as one
/// coupled wire network (see the [module docs](self)).
///
/// Create it with [`new`](Self::new) and register it with
/// [`Engine::add_operator`](super::Engine::add_operator); it uses the
/// `pre_e` and `post_e` hooks. Register it before operators that must see
/// the final E field of a step (e.g. absorbing sheets).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ObliqueWires {
    nodes: Vec<[f64; 3]>,
    segs: Vec<(usize, usize)>,
    /// Wire index per segment.
    seg_wire: Vec<usize>,
    /// Correction inductance per segment \[H\].
    l: Vec<f64>,
    /// Series resistance per segment \[Ω\] (feeds).
    r: Vec<f64>,
    /// Inverse correction capacitance per node \[1/F\]; 0 for connected
    /// ends.
    c_inv: Vec<f64>,
    connected: Vec<bool>,
    /// Coupled E edges (the canonical copy on a cylindrical grid).
    edges: Vec<(Axis, [usize; 3])>,
    /// All stored copies of each coupled edge (itself on a Cartesian
    /// grid).
    copies: Vec<Vec<[usize; 3]>>,
    /// Per segment: `(edge slot, P_{e,s})`.
    p: Vec<Vec<(usize, f64)>>,
    /// Per edge slot: `(segment, P_{e,s})`.
    pt: Vec<Vec<(usize, f64)>>,
    feeds: Vec<FeedState>,
    /// Segment currents `Iⁿ` \[A\].
    current: Vec<f64>,
    /// Node correction potentials `Vⁿ` \[V\].
    potential: Vec<f64>,
    /// `uⁿ` and `u*` of the coupled edges.
    u_old: Vec<f64>,
    ustar: Vec<f64>,
    /// Right-hand side, then `Ī`.
    avg: Vec<f64>,
    /// Feed source voltages of the step.
    src: Vec<f64>,
    solver: Option<Solver>,
    first_step: Option<u64>,
    dt: f64,
}

impl ObliqueWires {
    /// Builds the wire network of `wires` on `grid`, with the medium and
    /// the PEC edges of `materials`.
    ///
    /// A wire end is connected (see the [module docs](self#scope)) if it
    /// lies on a PEC edge of `materials` or an edge for which
    /// `contact(axis, node)` is true (e.g. lumped elements, edges of PEC
    /// outer faces): on a node, any of its edges; on an edge, that edge;
    /// inside a cell face, all four edges of the face.
    ///
    /// On a cylindrical grid the wire points are Cartesian \[m\] (see
    /// [cylindrical grids](self#cylindrical-grids)).
    ///
    /// # Errors
    /// [`Error::Invalid`] for invalid wires, points outside the grid (or
    /// within the first radial cell of a cylindrical grid's axis), or a
    /// wire too thick for the model (a correction is not positive).
    pub fn new(
        grid: &Grid,
        materials: &EdgeMaterials,
        wires: &[ObliqueWire],
        contact: impl Fn(Axis, [usize; 3]) -> bool,
    ) -> Result<Self> {
        if materials.nodes() != grid.nodes() {
            return Err(Error::Invalid(
                "oblique wires: material map does not match the grid".into(),
            ));
        }
        let lines: [&[f64]; 3] = [
            grid.lines(Axis::X),
            grid.lines(Axis::Y),
            grid.lines(Axis::Z),
        ];
        let frame = grid.is_cylindrical().then(|| cyl::CylFrame::new(lines));
        let extent = match &frame {
            Some(_) => {
                let r = lines[0][lines[0].len() - 1];
                (2.0 * r).max(lines[2][lines[2].len() - 1] - lines[2][0])
            }
            None => (0..3)
                .map(|d| lines[d][lines[d].len() - 1] - lines[d][0])
                .fold(0.0, f64::max),
        };
        let tol = 1e-9 * extent;
        for (wi, w) in wires.iter().enumerate() {
            w.validate()?;
            for p in &w.points {
                let inside = match &frame {
                    Some(f) => f.contains(*p, tol),
                    None => (0..3).all(|d| {
                        let l = lines[d];
                        p[d] >= l[0] - tol && p[d] <= l[l.len() - 1] + tol
                    }),
                };
                if !inside {
                    return Err(Error::Invalid(format!(
                        "oblique wire #{wi}: point {p:?} is outside the grid{}",
                        if frame.is_some_and(|f| f.axis) {
                            " (or within the first radial cell of the axis)"
                        } else {
                            ""
                        }
                    )));
                }
            }
            if let Some(f) = frame.filter(|f| f.axis) {
                for s in w.points.windows(2) {
                    if cyl::CylFrame::axis_distance(s[0], s[1]) < lines[0][1] - tol {
                        return Err(Error::Invalid(format!(
                            "oblique wire #{wi}: a segment passes within the first radial \
                             cell of the axis ({} m)",
                            f.lines[0][1]
                        )));
                    }
                }
            }
        }
        // Nodes: polyline vertices (merged when they coincide) and the
        // subdivision points.
        let mut nodes: Vec<[f64; 3]> = Vec::new();
        let mut vertex_nodes: Vec<usize> = Vec::new();
        let mut vertex = |p: [f64; 3], nodes: &mut Vec<[f64; 3]>| -> usize {
            for &k in &vertex_nodes {
                let q = nodes[k];
                if norm(core::array::from_fn(|d| q[d] - p[d])) <= tol {
                    return k;
                }
            }
            nodes.push(p);
            vertex_nodes.push(nodes.len() - 1);
            nodes.len() - 1
        };
        let mut segs = Vec::new();
        let mut seg_wire = Vec::new();
        let mut radius = Vec::new();
        let mut r = Vec::new();
        let mut feeds = Vec::new();
        for (wi, w) in wires.iter().enumerate() {
            let mut prev = vertex(w.points[0], &mut nodes);
            for k in 0..w.points.len() - 1 {
                let (a, b) = (w.points[k], w.points[k + 1]);
                let len = norm(core::array::from_fn(|d| b[d] - a[d]));
                if len <= tol {
                    continue;
                }
                let feed = w.feed.as_ref().filter(|f| f.segment == k);
                let n = if feed.is_some() {
                    1
                } else {
                    let lmax = w.max_segment.unwrap_or_else(|| {
                        let mid: [f64; 3] = core::array::from_fn(|d| 0.5 * (a[d] + b[d]));
                        0.5 * [a, mid, b]
                            .iter()
                            .flat_map(|p| match &frame {
                                Some(f) => f.cell_size(*p),
                                None => core::array::from_fn(|d| {
                                    let c = deposit::cell(lines[d], p[d]);
                                    lines[d][c + 1] - lines[d][c]
                                }),
                            })
                            .fold(f64::INFINITY, f64::min)
                    });
                    ((len / lmax).ceil() as usize).max(1)
                };
                for i in 1..=n {
                    let next = if i == n {
                        vertex(b, &mut nodes)
                    } else {
                        let t = i as f64 / n as f64;
                        nodes.push(core::array::from_fn(|d| a[d] + t * (b[d] - a[d])));
                        nodes.len() - 1
                    };
                    if let Some(f) = feed {
                        feeds.push(FeedState {
                            segment: segs.len(),
                            resistance: f.resistance,
                            source: f.source.clone(),
                            voltage: Vec::new(),
                            current: Vec::new(),
                        });
                        r.push(f.resistance);
                    } else {
                        r.push(0.0);
                    }
                    segs.push((prev, next));
                    seg_wire.push(wi);
                    radius.push(w.radius);
                    prev = next;
                }
            }
        }
        if segs.is_empty() {
            return Err(Error::Invalid("oblique wires: no segment".into()));
        }
        let nn = nodes.len();
        // Connected components (union–find).
        let mut parent: Vec<usize> = (0..nn).collect();
        fn root(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for &(a, b) in &segs {
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
        let component: Vec<usize> = (0..nn).map(|i| root(&mut parent, i)).collect();
        let mut dual = vec![0.0; nn];
        let mut degree = vec![0usize; nn];
        let seg_len = |s: (usize, usize)| {
            let (a, b) = (nodes[s.0], nodes[s.1]);
            norm(core::array::from_fn(|d| b[d] - a[d]))
        };
        for &s in &segs {
            let l = seg_len(s);
            dual[s.0] += 0.5 * l;
            dual[s.1] += 0.5 * l;
            degree[s.0] += 1;
            degree[s.1] += 1;
        }
        let geo = Geometry {
            nodes: &nodes,
            segs: &segs,
            radius: &radius,
            component: &component,
        };
        let (cap_len, flux) = correction::corrections(lines, &geo, &dual, WINDOW, frame.as_ref());
        // Coupling matrix on the real grid.
        let mut slot: BTreeMap<(usize, [usize; 3]), usize> = BTreeMap::new();
        let mut edges = Vec::new();
        let mut copies = Vec::new();
        let mut p = Vec::with_capacity(segs.len());
        for &(a, b) in &segs {
            let mut w = BTreeMap::new();
            match &frame {
                Some(f) => f.segment_weights(nodes[a], nodes[b], &mut w)?,
                None => deposit::segment_weights(lines, nodes[a], nodes[b], &mut w),
            }
            let mut row = Vec::with_capacity(w.len());
            for ((d, node), v) in w {
                let axis = Axis::ALL[d];
                let k = *slot.entry((d, node)).or_insert_with(|| {
                    edges.push((axis, node));
                    copies.push(frame.map_or_else(|| vec![node], |f| f.copies(d, node)));
                    edges.len() - 1
                });
                // Parameter length (the edge's physical length on a
                // Cartesian grid).
                let len = match &frame {
                    Some(_) => lines[d][node[d] + 1] - lines[d][node[d]],
                    None => grid.edge_length(axis, node),
                };
                row.push((k, v / len));
            }
            p.push(row);
        }
        let mut pt = vec![Vec::new(); edges.len()];
        for (s, row) in p.iter().enumerate() {
            for &(k, v) in row {
                pt[k].push((s, v));
            }
        }
        // Local medium: weighted means over the coupled edges.
        let gi = |n: [usize; 3]| grid.index(n);
        let seg_medium = |s: usize| {
            let (mut we, mut e, mut m) = (0.0, 0.0, 0.0);
            for &(k, v) in &p[s] {
                let (axis, node) = edges[k];
                let w = v.abs();
                let i = gi(node);
                we += w;
                e += w * materials.eps_r(axis)[i];
                let (o1, o2) = axis.others();
                m += w * 0.5 * (materials.mu_r(o1)[i] + materials.mu_r(o2)[i]);
            }
            if we > 0.0 {
                (e / we, m / we)
            } else {
                (1.0, 1.0)
            }
        };
        let medium: Vec<(f64, f64)> = (0..segs.len()).map(seg_medium).collect();
        let mut l = Vec::with_capacity(segs.len());
        for (s, &f) in flux.iter().enumerate() {
            if !(f.is_finite() && f > 0.0) {
                return Err(Error::Invalid(format!(
                    "oblique wire #{}: radius {:e} m is too large for the thin-wire model \
                     (no positive inductance correction at segment {s})",
                    seg_wire[s], radius[s]
                )));
            }
            l.push(MU0 * medium[s].1 * f);
        }
        // Connected ends.
        let on_conductor = |q: [f64; 3]| -> bool {
            let q = frame.map_or(q, |f| f.param(q));
            let pec = |axis: Axis, node: [usize; 3]| {
                let n = grid.nodes();
                let a = axis.index();
                node.iter().zip(&n).all(|(i, m)| i < m)
                    && node[a] + 1 < n[a]
                    && (materials.is_pec_edge(axis, node) || contact(axis, node))
            };
            let snap: [Option<usize>; 3] = core::array::from_fn(|d| {
                let ls = lines[d];
                let c = deposit::cell(ls, q[d]);
                let h = ls[c + 1] - ls[c];
                [c, c + 1]
                    .into_iter()
                    .find(|&i| (ls[i] - q[d]).abs() <= 1e-6 * h)
            });
            let base: [usize; 3] =
                core::array::from_fn(|d| snap[d].unwrap_or_else(|| deposit::cell(lines[d], q[d])));
            let free: Vec<usize> = (0..3).filter(|&d| snap[d].is_none()).collect();
            match free[..] {
                [] => Axis::ALL.iter().any(|&axis| {
                    let a = axis.index();
                    let mut back = base;
                    pec(axis, base)
                        || (base[a] > 0 && {
                            back[a] -= 1;
                            pec(axis, back)
                        })
                }),
                [d] => pec(Axis::ALL[d], base),
                [d1, d2] => {
                    let (a1, a2) = (Axis::ALL[d1], Axis::ALL[d2]);
                    let mut up1 = base;
                    up1[d1] += 1;
                    let mut up2 = base;
                    up2[d2] += 1;
                    pec(a1, base) && pec(a1, up2) && pec(a2, base) && pec(a2, up1)
                }
                _ => false,
            }
        };
        let connected: Vec<bool> = (0..nn)
            .map(|j| degree[j] == 1 && on_conductor(nodes[j]))
            .collect();
        let mut c_inv = vec![0.0; nn];
        for j in 0..nn {
            if connected[j] {
                continue;
            }
            if !(cap_len[j].is_finite() && cap_len[j] > 0.0) {
                return Err(Error::Invalid(format!(
                    "oblique wire: radius too large for the thin-wire model (no positive \
                     capacitance correction at node {:?})",
                    nodes[j]
                )));
            }
            // ε of the node: mean over its segments.
            let (mut e, mut k) = (0.0, 0.0);
            for (s, &(a, b)) in segs.iter().enumerate() {
                if a == j || b == j {
                    e += medium[s].0;
                    k += 1.0;
                }
            }
            let eps = EPS0 * e / k;
            c_inv[j] = 1.0 / (eps * cap_len[j]);
        }
        let ns = segs.len();
        let ne = edges.len();
        let nfeeds = feeds.len();
        Ok(Self {
            nodes,
            segs,
            seg_wire,
            l,
            r,
            c_inv,
            connected,
            edges,
            copies,
            p,
            pt,
            feeds,
            current: vec![0.0; ns],
            potential: vec![0.0; nn],
            u_old: vec![0.0; ne],
            ustar: vec![0.0; ne],
            avg: vec![0.0; ns],
            src: vec![0.0; nfeeds],
            solver: None,
            first_step: None,
            dt: 0.0,
        })
    }

    /// Number of wire segments.
    pub fn segments(&self) -> usize {
        self.segs.len()
    }

    /// Wire node positions \[m\].
    pub fn nodes(&self) -> &[[f64; 3]] {
        &self.nodes
    }

    /// Segments as `(start node, end node)`, wire by wire along the
    /// polylines.
    pub fn segment_nodes(&self) -> &[(usize, usize)] {
        &self.segs
    }

    /// Index of the [`ObliqueWire`] each segment belongs to.
    pub fn segment_wires(&self) -> &[usize] {
        &self.seg_wire
    }

    /// Whether each node is a connected end (`V = 0`).
    pub fn connected_nodes(&self) -> &[bool] {
        &self.connected
    }

    /// Correction inductance per segment \[H\].
    pub fn inductances(&self) -> &[f64] {
        &self.l
    }

    /// Correction capacitance per node \[F\] (infinite for connected
    /// ends).
    pub fn capacitances(&self) -> Vec<f64> {
        self.c_inv
            .iter()
            .map(|&c| if c > 0.0 { 1.0 / c } else { f64::INFINITY })
            .collect()
    }

    /// Number of coupled E edges.
    pub fn coupled_edges(&self) -> usize {
        self.edges.len()
    }

    /// Segment currents `Iⁿ` \[A\] at `t = n·dt`, along each segment's
    /// direction.
    pub fn currents(&self) -> &[f64] {
        &self.current
    }

    /// Node correction potentials `Vⁿ` \[V\].
    pub fn potentials(&self) -> &[f64] {
        &self.potential
    }

    /// Number of feeds, in wire order.
    pub fn feeds(&self) -> usize {
        self.feeds.len()
    }

    /// Recorded terminal voltages of feed `k` \[V\]: `V_src − R·Ī`, the
    /// voltage across the feed gap along the wire direction, at
    /// `t = (n + ½)·dt` (see [`feed_t0`](Self::feed_t0)).
    ///
    /// # Panics
    /// If `k` is out of range.
    pub fn feed_voltage(&self, k: usize) -> &[f64] {
        &self.feeds[k].voltage
    }

    /// Recorded currents `Ī` of feed `k` \[A\] along the wire direction,
    /// at the times of [`feed_voltage`](Self::feed_voltage).
    ///
    /// # Panics
    /// If `k` is out of range.
    pub fn feed_current(&self, k: usize) -> &[f64] {
        &self.feeds[k].current
    }

    /// Time of the first feed sample, `(n0 + ½)·dt` \[s\].
    pub fn feed_t0(&self) -> f64 {
        (self.first_step.unwrap_or(0) as f64 + 0.5) * self.dt
    }

    /// Input impedance `V/I` of feed `k` at `freqs` \[Ω\] (DFTs of the
    /// recorded series).
    ///
    /// # Panics
    /// If `k` is out of range.
    pub fn feed_impedance(&self, k: usize, freqs: &[f64]) -> Vec<Complex64> {
        let t0 = self.feed_t0();
        let v = crate::post::dft(t0, self.dt, self.feed_voltage(k), freqs);
        let i = crate::post::dft(t0, self.dt, self.feed_current(k), freqs);
        v.iter().zip(&i).map(|(v, i)| *v / *i).collect()
    }

    /// Energy in the wire state, `½·Σ L·I² + ½·Σ C·V²` \[J\], at `n·dt`.
    pub fn wire_energy(&self) -> f64 {
        let mut w = 0.0;
        for (l, i) in self.l.iter().zip(&self.current) {
            w += 0.5 * l * i * i;
        }
        for (c, v) in self.c_inv.iter().zip(&self.potential) {
            if *c > 0.0 {
                w += 0.5 * v * v / c;
            }
        }
        w
    }

    fn setup(&mut self, ctx: &StepContext<'_>) -> Result<()> {
        let dt = ctx.dt;
        let co = ctx.coefficients;
        // The merged cb of the stored copies (1/cb = Σ 1/cb_j; a PEC copy
        // makes the edge PEC).
        let cb: Vec<f64> = self
            .edges
            .iter()
            .zip(&self.copies)
            .map(|(&(axis, node), cs)| {
                if cs.len() <= 1 {
                    return co.cb(axis)[ctx.grid.index(node)] as f64;
                }
                let mut w = 0.0;
                for c in cs {
                    let v = co.cb(axis)[ctx.grid.index(*c)] as f64;
                    if v <= 0.0 {
                        return 0.0;
                    }
                    w += 1.0 / v;
                }
                1.0 / w
            })
            .collect();
        let mut lower: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut add = |i: usize, j: usize, v: f64| {
            let k = if j <= i { (i, j) } else { (j, i) };
            *lower.entry(k).or_insert(0.0) += v;
        };
        for s in 0..self.segs.len() {
            add(s, s, 2.0 * self.l[s] / dt + self.r[s]);
        }
        for (k, col) in self.pt.iter().enumerate() {
            for (x, &(s1, p1)) in col.iter().enumerate() {
                for &(s2, p2) in &col[x..] {
                    let v = 0.5 * cb[k] * p1 * p2;
                    // Off-diagonal pairs appear once in the triangle.
                    add(s1, s2, v);
                }
            }
        }
        let mut incident: Vec<Vec<(usize, f64)>> = vec![Vec::new(); self.nodes.len()];
        for (s, &(a, b)) in self.segs.iter().enumerate() {
            incident[a].push((s, -1.0));
            incident[b].push((s, 1.0));
        }
        for (j, inc) in incident.iter().enumerate() {
            let c = self.c_inv[j];
            if c == 0.0 {
                continue;
            }
            for (x, &(s1, g1)) in inc.iter().enumerate() {
                for &(s2, g2) in &inc[x..] {
                    add(s1, s2, 0.5 * dt * c * g1 * g2);
                }
            }
        }
        let chol = Envelope::factor(self.segs.len(), &lower)?;
        self.dt = dt;
        self.solver = Some(Solver { dt, cb, chol });
        Ok(())
    }
}

impl Operator for ObliqueWires {
    fn hooks(&self) -> Hooks {
        Hooks::PRE_E | Hooks::POST_E
    }

    /// [`wire_energy`](Self::wire_energy).
    fn energy(&self) -> f64 {
        self.wire_energy()
    }

    fn pre_e(&mut self, fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        for (u, &(axis, node)) in self.u_old.iter_mut().zip(&self.edges) {
            *u = fields.u.get(axis, node) as f64;
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if self.solver.is_none() {
            // The matrix is positive definite by construction; a failure
            // means invalid coefficients, which the engine rejects.
            self.setup(ctx)
                .expect("oblique wire system is positive definite");
            self.first_step = Some(ctx.step);
        }
        let Some(sv) = &self.solver else { return };
        let dt = sv.dt;
        for (u, &(axis, node)) in self.ustar.iter_mut().zip(&self.edges) {
            *u = fields.u.get(axis, node) as f64;
        }
        let rhs = &mut self.avg;
        for (s, row) in self.p.iter().enumerate() {
            let mut v = 2.0 * self.l[s] / dt * self.current[s];
            for &(k, pv) in row {
                v += 0.5 * pv * (self.ustar[k] + self.u_old[k]);
            }
            let (a, b) = self.segs[s];
            v -= self.potential[b] - self.potential[a];
            rhs[s] = v;
        }
        let t = ctx.h_time();
        for (f, v) in self.feeds.iter().zip(&mut self.src) {
            *v = f.source.as_ref().map_or(0.0, |(w, a)| a * w.value(t, dt));
            rhs[f.segment] += *v;
        }
        sv.chol.solve(rhs);
        let avg = &self.avg;
        for (k, col) in self.pt.iter().enumerate() {
            let mut j = 0.0;
            for &(s, pv) in col {
                j += pv * avg[s];
            }
            if j != 0.0 && sv.cb[k] != 0.0 {
                let (axis, _) = self.edges[k];
                let v = (self.ustar[k] - sv.cb[k] * j) as f32;
                for &node in &self.copies[k] {
                    fields.u.set(axis, node, v);
                }
            }
        }
        for (s, &(a, b)) in self.segs.iter().enumerate() {
            let jb = avg[s];
            self.current[s] = 2.0 * jb - self.current[s];
            self.potential[a] -= dt * self.c_inv[a] * jb;
            self.potential[b] += dt * self.c_inv[b] * jb;
        }
        for (f, &v) in self.feeds.iter_mut().zip(&self.src) {
            let jb = avg[f.segment];
            f.voltage.push(v - f.resistance * jb);
            f.current.push(jb);
        }
    }
}
