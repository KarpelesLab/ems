//! Conformal (sub-cell) PEC: curved metal surfaces on the Cartesian grid
//! without staircasing (design §5.13).
//!
//! # Method
//!
//! The Dey–Mittra conformal PEC scheme (S. Dey, R. Mittra, "A locally
//! conformal finite-difference time-domain (FDTD) algorithm for modeling
//! three-dimensional perfectly conducting objects", IEEE Microwave Guided
//! Wave Lett. 7(9), 1997) evaluates Faraday's law on the part of each cell
//! face that lies outside the metal,
//!
//! ```text
//! μ·A_open·dH/dt = −∮_open E·dl = −Σ_e E_e·l_e,open ,
//! ```
//!
//! while Ampère's law is left as it is. In the integral form of the engine
//! (design §5.2) this is a pure change of coefficients once the stored
//! edge voltage of a cut edge is read as the voltage over its *open part*,
//! `u = E·l_open = f_e·E·l` (`f_e`: open length fraction of the edge):
//!
//! * the E update `E ← E + (dt/ε)·curl H` becomes
//!   `u ← u + f_e·cb·Kᵀi`: **`cb → f_e·cb`** (edge capacitance `C/f_e`;
//!   `ca` is unchanged because `C` and `G` scale alike);
//! * the H update becomes `i ← i − (db/f_A)·K·u` with the ordinary curl
//!   `K`: **`db → db/f_A`** (`f_A`: open area fraction of the face; the
//!   inductance becomes `f_A·L`, `da` is unchanged).
//!
//! The scheme keeps the symmetric form `C·du/dt = Kᵀi`, `L·di/dt = −K·u`
//! of the uncut grid (it is the finite-integration reading of
//! Dey–Mittra), so the engine's energy invariant (module docs of
//! [`engine`](super)) stays exact. Edges with `f_e = 0` are PEC.
//!
//! # Small cut faces: merging
//!
//! A face that keeps a long open edge but little area has a small `f_A·L`
//! and a local resonance far above the Courant limit of the uncut grid:
//! plain Dey–Mittra needs a time step `∝ √f_A,min`. Raising `f_A` to a
//! minimum cures that only for minima around 0.3–0.5, and then costs most
//! of the accuracy gain (first-order convergence again; measured with the
//! cylinder cavity of tests/conformal.rs). Instead, every face with
//! `f_A <` [`ConformalOptions::merge_below`] is **merged** with an in-plane
//! neighbour, in the spirit of the enlarged-cell technique (cell
//! borrowing; e.g. T. Xiao, Q. H. Liu, IEEE MWCL 14(12), 2004) and of the
//! uniformly stable conformal scheme (I. Zagorodnov, R. Schuhmann,
//! T. Weiland, IEEE T-MTT 51(4), 2003):
//!
//! * the neighbour is the one across the small face's most-open edge
//!   (largest `f_e`, ties to the larger neighbour area); a group still
//!   below the threshold merges again with a neighbour of any member;
//! * a group carries **one** current `i_g` on all its faces, with
//!   `L_g = Σ f_A·L`, `R_g = Σ f_A·R` and the circulation of the merged
//!   contour `Σ_f K_f·u` (edges shared inside the group cancel);
//! * Ampère reads `i_g` on every face of the group, which is the transpose
//!   of that circulation, so the scheme stays symmetric: in matrix form
//!   `K → P·K` with the 0/1 merge matrix `P`.
//!
//! A merged contour has the open perimeter of its faces but at least the
//! area of the host face, so its local resonance stays near that of an
//! uncut face. The group update runs in the serial `post_h` hook of the
//! [`MergedFaces`] operator; the merged faces get `da = 1`, `db = 0`, so the
//! kernel leaves them alone, and the operator reports the group energy
//! `½·L_g·i_g^{n−½}·i_g^{n+½}` ([`Operator::energy`], computed in
//! `post_e`), so the invariant remains exact. A small face without an
//! open in-plane neighbour keeps its own current with `f_A` raised to the
//! threshold ([`ConformalReport::floored`]).
//!
//! # Stability check
//!
//! The lossless scheme is stable iff `λ_max(S) < 4` for the symmetric
//! `S = √cb·(PK)ᵀ·(dt/L)·(PK)·√cb` (`cb·db = dt²/(L·C)`).
//! [`ConformalPec::apply`] estimates `s = √λ_max(S)/2` by a deterministic
//! power iteration ([`stability_factor`]) started on the cut cells, where
//! an unstable mode would be localised (`s < 1`: stable; an uncut grid at
//! Courant factor `S_c` gives `s ≈ S_c`). If `s >`
//! [`ConformalOptions::max_stability`], the threshold is raised (×1.25, up
//! to 1) and the construction repeated, at most
//! [`ConformalOptions::max_rounds`] times; [`ConformalReport::verified`]
//! tells whether it passed. Power iteration approaches `λ_max` from below,
//! so `s` is an estimate, not a bound; the long-run tests check
//! boundedness directly.

use std::collections::HashMap;

use super::Engine;
use super::coeffs::Coefficients;
use super::fields::Fields;
use super::kernel::flush;
use super::operator::{Hooks, Operator, StepContext};
use crate::axis::Axis;
use crate::{Error, Result};

/// An E edge partly inside metal: open length fraction `fraction` in
/// `[0, 1)` (0: PEC).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalEdge {
    /// Edge direction.
    pub axis: Axis,
    /// Start node.
    pub node: [usize; 3],
    /// Open (non-metal) fraction of the edge length.
    pub fraction: f64,
}

/// A cell face (H dual edge) partly inside metal: open area fraction
/// `fraction` in `[0, 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalFace {
    /// Face normal (the H component).
    pub axis: Axis,
    /// Lowest corner node.
    pub node: [usize; 3],
    /// Open (non-metal) fraction of the face area.
    pub fraction: f64,
}

/// Options of [`ConformalPec::apply`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalOptions {
    /// Faces with an open area fraction below this are merged with a
    /// neighbour (default 0.5; see the [module docs](self)).
    pub merge_below: f64,
    /// Accepted stability factor `s = √λ_max/2` (default 0.999).
    pub max_stability: f64,
    /// Power iterations per check (default 300; 0 skips the check).
    pub iterations: usize,
    /// Rounds of threshold raising (default 4).
    pub max_rounds: usize,
}

impl Default for ConformalOptions {
    fn default() -> Self {
        Self {
            merge_below: 0.5,
            max_stability: 0.999,
            iterations: 300,
            max_rounds: 4,
        }
    }
}

/// What [`ConformalPec::apply`] did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalReport {
    /// Cut edges scaled (including fully covered ones made PEC).
    pub edges: usize,
    /// Cut faces with their own current (`db → db/f_A`).
    pub faces: usize,
    /// Faces merged into groups (cut faces and their hosts).
    pub merged: usize,
    /// Merge groups (one current each).
    pub groups: usize,
    /// Small faces without an open neighbour, `f_A` raised to the
    /// threshold instead.
    pub floored: usize,
    /// Final merge threshold.
    pub merge_below: f64,
    /// Estimated stability factor `s` (NaN if not checked).
    pub stability: f64,
    /// Whether `s ≤ max_stability` (true if not checked).
    pub verified: bool,
}

impl ConformalReport {
    /// A warning text if the stability check failed.
    pub fn warning(&self) -> Option<String> {
        (!self.verified).then(|| {
            format!(
                "conformal PEC: estimated stability factor {:.4} (> 1 is unstable) with \
                 merge threshold {:.3}; reduce the time step factor",
                self.stability, self.merge_below
            )
        })
    }
}

/// Conformal PEC geometry (cut edges and faces) to be applied to an
/// engine; see the [module docs](self).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConformalPec {
    edges: Vec<ConformalEdge>,
    faces: Vec<ConformalFace>,
    options: ConformalOptions,
}

/// Face key: `(normal axis index, engine node index)`.
type Key = (usize, usize);

impl ConformalPec {
    /// Empty geometry with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the options.
    #[must_use]
    pub fn with_options(mut self, options: ConformalOptions) -> Self {
        self.options = options;
        self
    }

    /// The options.
    pub fn options(&self) -> &ConformalOptions {
        &self.options
    }

    /// Adds a cut edge.
    pub fn push_edge(&mut self, e: ConformalEdge) {
        self.edges.push(e);
    }

    /// Adds a cut face.
    pub fn push_face(&mut self, f: ConformalFace) {
        self.faces.push(f);
    }

    /// The cut edges.
    pub fn edges(&self) -> &[ConformalEdge] {
        &self.edges
    }

    /// The cut faces.
    pub fn faces(&self) -> &[ConformalFace] {
        &self.faces
    }

    /// Whether there is nothing to apply.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty() && self.faces.is_empty()
    }

    /// Applies the scheme to `engine`: scales the coefficients of cut edges
    /// and faces, builds the merge groups and registers their
    /// [`MergedFaces`] operator (if any), after the stability check of the
    /// [module docs](self#stability-check). Call once, before operators
    /// that read or modify coefficients are added and before the first
    /// step.
    ///
    /// # Errors
    /// [`Error::Invalid`] for fractions outside `[0, 1]` or edges / faces
    /// that do not exist on the grid.
    pub fn apply(&self, engine: &mut Engine) -> Result<ConformalReport> {
        let n = engine.grid().nodes();
        self.validate(n)?;
        let dt = engine.dt();
        let mut co = engine.coefficients().clone();
        for e in &self.edges {
            let (_, cb) = co.e_mut(e.axis);
            let k = index(n, e.node);
            cb[k] = (cb[k] as f64 * e.fraction) as f32;
        }
        let mut seeds: Vec<(usize, usize)> = Vec::new();
        for f in &self.faces {
            for (a, p) in face_edges(f.axis, f.node) {
                seeds.push((a.index(), index(n, p)));
            }
        }
        let mut tau = self.options.merge_below.clamp(0.0, 1.0);
        let rounds = self.options.max_rounds.max(1);
        for round in 0..rounds {
            let (coeffs, groups, mut report) = self.construct(&co, n, dt, tau);
            let last = round + 1 == rounds || tau >= 1.0;
            if self.options.iterations > 0 {
                let s = stability(&coeffs, n, &groups, &seeds, self.options.iterations);
                report.stability = s;
                report.verified = s <= self.options.max_stability;
            }
            if report.verified || last {
                *engine.coefficients_mut() = coeffs;
                if !groups.is_empty() {
                    engine.add_operator(MergedFaces::new(groups));
                }
                return Ok(report);
            }
            tau = (tau * 1.25).min(1.0);
        }
        unreachable!("the last round returns")
    }

    fn validate(&self, n: [usize; 3]) -> Result<()> {
        for e in &self.edges {
            let a = e.axis.index();
            if !(0.0..=1.0).contains(&e.fraction)
                || (0..3).any(|d| e.node[d] >= n[d])
                || e.node[a] + 1 >= n[a]
            {
                return Err(Error::Invalid(format!("invalid conformal edge {e:?}")));
            }
        }
        for f in &self.faces {
            if !(0.0..=1.0).contains(&f.fraction) || !face_exists(f.axis, f.node, n) {
                return Err(Error::Invalid(format!("invalid conformal face {f:?}")));
            }
        }
        Ok(())
    }

    /// Coefficients (edge scaling already in `co`), merge groups and
    /// counts for merge threshold `tau`.
    fn construct(
        &self,
        co: &Coefficients,
        n: [usize; 3],
        dt: f64,
        tau: f64,
    ) -> (Coefficients, Vec<Group>, ConformalReport) {
        let idx = |p: [usize; 3]| index(n, p);
        let mut out = co.clone();
        let mut report = ConformalReport {
            edges: self.edges.len(),
            faces: 0,
            merged: 0,
            groups: 0,
            floored: 0,
            merge_below: tau,
            stability: f64::NAN,
            verified: true,
        };
        let face_frac: HashMap<Key, f64> = self
            .faces
            .iter()
            .map(|f| ((f.axis.index(), idx(f.node)), f.fraction))
            .collect();
        let edge_frac: HashMap<Key, f64> = self
            .edges
            .iter()
            .map(|e| ((e.axis.index(), idx(e.node)), e.fraction))
            .collect();
        let fa_of = |a: Axis, p: [usize; 3]| *face_frac.get(&(a.index(), idx(p))).unwrap_or(&1.0);
        let open = |a: Axis, p: [usize; 3]| co.cb(a)[idx(p)] > 0.0;
        let fe_of = |a: Axis, p: [usize; 3]| {
            if open(a, p) {
                *edge_frac.get(&(a.index(), idx(p))).unwrap_or(&1.0)
            } else {
                0.0
            }
        };
        let live = |a: Axis, p: [usize; 3]| face_edges(a, p).iter().any(|&(e, q)| open(e, q));

        // Union-find over faces; each root keeps its member list (sorted)
        // and total open area.
        let mut uf = Groups::default();
        let small: Vec<&ConformalFace> = self
            .faces
            .iter()
            .filter(|f| f.fraction < tau && live(f.axis, f.node))
            .collect();
        for f in &small {
            uf.add((f.axis.index(), idx(f.node)), f.node, f.fraction);
        }
        for _pass in 0..small.len().max(1) {
            let mut changed = false;
            for f in &small {
                let k0 = (f.axis.index(), idx(f.node));
                let root = uf.find(k0);
                // Once per group and pass: by its first member.
                if uf.area[&root] >= tau || uf.members[&root][0] != k0 {
                    continue;
                }
                let mut best: Option<((f64, f64), Key, [usize; 3])> = None;
                for m in uf.members[&root].clone() {
                    let p = uf.node[&m];
                    for (e, q, nb) in in_plane_neighbours(f.axis, p, n) {
                        let fe = fe_of(e, q);
                        if fe <= 0.0 || !live(f.axis, nb) {
                            continue;
                        }
                        let kn = (f.axis.index(), idx(nb));
                        if uf.contains(kn) && uf.find(kn) == root {
                            continue;
                        }
                        let score = (fe, fa_of(f.axis, nb));
                        if best.is_none_or(|b| score > b.0) {
                            best = Some((score, kn, nb));
                        }
                    }
                }
                let Some(((_, fa_nb), kn, nb)) = best else {
                    continue;
                };
                if !uf.contains(kn) {
                    uf.add(kn, nb, fa_nb);
                }
                uf.union(root, kn);
                changed = true;
            }
            if !changed {
                break;
            }
        }

        let mut grouped: HashMap<Key, ()> = HashMap::new();
        let mut groups = Vec::new();
        for (_, members) in uf.groups() {
            let (mut l, mut r) = (0.0, 0.0);
            let mut terms: HashMap<Key, (Axis, [usize; 3], i32)> = HashMap::new();
            let mut faces = Vec::with_capacity(members.len());
            for &k in &members {
                let a = Axis::from_index(k.0).expect("axis in 0..3");
                let p = uf.node[&k];
                let (da, db) = (co.da(a)[k.1] as f64, co.db(a)[k.1] as f64);
                if db > 0.0 {
                    // L, R of the full face from (da, db), scaled by f_A.
                    let fa = fa_of(a, p);
                    l += fa * dt * (1.0 + da) / (2.0 * db);
                    r += fa * (1.0 - da) / db;
                }
                for ((e, q), sign) in face_edges(a, p).into_iter().zip(SIGNS) {
                    terms
                        .entry((e.index(), idx(q)))
                        .and_modify(|t| t.2 += sign)
                        .or_insert((e, q, sign));
                }
                faces.push((a, p));
                grouped.insert(k, ());
                let (da_m, db_m) = out.h_mut(a);
                da_m[k.1] = 1.0;
                db_m[k.1] = 0.0;
            }
            report.merged += members.len();
            let mut terms: Vec<(Axis, [usize; 3], f64)> = terms
                .into_values()
                .filter(|t| t.2 != 0)
                .map(|t| (t.0, t.1, t.2 as f64))
                .collect();
            terms.sort_by_key(|t| (t.0.index(), idx(t.1)));
            if l <= 0.0 {
                continue;
            }
            let den = l + 0.5 * r * dt;
            groups.push(Group {
                faces,
                terms,
                da: (l - 0.5 * r * dt) / den,
                db: dt / den,
                l,
                dt_over_l: dt / l,
                i: 0.0,
            });
        }
        report.groups = groups.len();
        // Remaining cut faces keep their own current with db/f_A (raised
        // to the threshold if small and not merged).
        for f in &self.faces {
            let k = idx(f.node);
            if grouped.contains_key(&(f.axis.index(), k)) || !live(f.axis, f.node) {
                continue;
            }
            let mut fa = f.fraction;
            if fa < tau {
                fa = tau.max(1e-6);
                report.floored += 1;
            }
            if fa < 1.0 {
                let (_, db) = out.h_mut(f.axis);
                db[k] = (db[k] as f64 / fa) as f32;
            }
            report.faces += 1;
        }
        (out, groups, report)
    }
}

/// Engine flat index of node `p`.
fn index(n: [usize; 3], p: [usize; 3]) -> usize {
    (p[0] * n[1] + p[1]) * n[2] + p[2]
}

/// Signs of the [`face_edges`] in the kernel's curl
/// (`+u_b(p) + u_c(p+e_b) − u_b(p+e_c) − u_c(p)`).
const SIGNS: [i32; 4] = [1, 1, -1, -1];

fn face_exists(axis: Axis, p: [usize; 3], n: [usize; 3]) -> bool {
    let (b, c) = axis.others();
    (0..3).all(|d| p[d] < n[d])
        && p[b.index()] + 1 < n[b.index()]
        && p[c.index()] + 1 < n[c.index()]
}

/// The four E edges `(axis, start node)` bounding the face normal to
/// `axis` with lowest corner `node`, in perimeter order: `b` at `p`, `c`
/// at `p + e_b`, `b` at `p + e_c`, `c` at `p` (`(b, c) = axis.others()`).
fn face_edges(axis: Axis, node: [usize; 3]) -> [(Axis, [usize; 3]); 4] {
    let (b, c) = axis.others();
    let mut pb = node;
    pb[b.index()] += 1;
    let mut pc = node;
    pc[c.index()] += 1;
    [(b, node), (c, pb), (b, pc), (c, node)]
}

/// In-plane neighbours of the face `(axis, p)` that exist: `(shared edge
/// axis, shared edge node, neighbour face node)`.
fn in_plane_neighbours(
    axis: Axis,
    p: [usize; 3],
    n: [usize; 3],
) -> Vec<(Axis, [usize; 3], [usize; 3])> {
    let (b, c) = axis.others();
    let mut v = Vec::with_capacity(4);
    // (shared edge axis, offset axis): the b edges border the neighbours
    // along c and vice versa.
    for (edge, off) in [(b, c), (c, b)] {
        let o = off.index();
        if p[o] > 0 {
            let mut q = p;
            q[o] -= 1;
            v.push((edge, p, q));
        }
        let mut q = p;
        q[o] += 1;
        if face_exists(axis, q, n) {
            v.push((edge, q, q));
        }
    }
    v
}

/// Union-find over face keys with member lists and areas per root.
#[derive(Default)]
struct Groups {
    parent: HashMap<Key, Key>,
    node: HashMap<Key, [usize; 3]>,
    members: HashMap<Key, Vec<Key>>,
    area: HashMap<Key, f64>,
}

impl Groups {
    fn add(&mut self, k: Key, node: [usize; 3], area: f64) {
        self.parent.insert(k, k);
        self.node.insert(k, node);
        self.members.insert(k, vec![k]);
        self.area.insert(k, area);
    }

    fn contains(&self, k: Key) -> bool {
        self.parent.contains_key(&k)
    }

    fn find(&mut self, k: Key) -> Key {
        let mut r = k;
        while self.parent[&r] != r {
            r = self.parent[&r];
        }
        let mut c = k;
        while self.parent[&c] != r {
            let next = self.parent[&c];
            self.parent.insert(c, r);
            c = next;
        }
        r
    }

    /// Joins the groups of `a` and `b`; the smaller key stays the root.
    fn union(&mut self, a: Key, b: Key) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let (keep, gone) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.parent.insert(gone, keep);
        let a = self.area.remove(&gone).unwrap_or(0.0);
        *self.area.get_mut(&keep).expect("root area") += a;
        let mv = self.members.remove(&gone).unwrap_or_default();
        let list = self.members.get_mut(&keep).expect("root members");
        list.extend(mv);
        list.sort_unstable();
    }

    /// Groups with more than one face, sorted by root.
    fn groups(&self) -> Vec<(Key, Vec<Key>)> {
        let mut g: Vec<(Key, Vec<Key>)> = self
            .members
            .iter()
            .filter(|(_, m)| m.len() > 1)
            .map(|(&r, m)| (r, m.clone()))
            .collect();
        g.sort_unstable();
        g
    }
}

/// One merge group: a single current on several faces.
#[derive(Debug, Clone, PartialEq)]
struct Group {
    faces: Vec<(Axis, [usize; 3])>,
    /// Circulation terms `(edge axis, edge node, sign)` of the merged
    /// contour.
    terms: Vec<(Axis, [usize; 3], f64)>,
    da: f64,
    db: f64,
    /// Group inductance \[H\].
    l: f64,
    /// `dt/L` (lossless update factor, for the stability check).
    dt_over_l: f64,
    /// Current `i_g` (after the last H update).
    i: f64,
}

/// The merge groups of a [`ConformalPec`], registered by
/// [`ConformalPec::apply`]: updates each group's single current after the
/// H kernel and reports the group energy; see the
/// [module docs](self#small-cut-faces-merging).
#[derive(Debug, Clone, PartialEq)]
pub struct MergedFaces {
    groups: Vec<Group>,
    energy: f64,
}

impl MergedFaces {
    fn new(groups: Vec<Group>) -> Self {
        Self {
            groups,
            energy: 0.0,
        }
    }

    /// Number of groups.
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Whether there are no groups.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

fn circulation(g: &Group, f: &Fields<'_>) -> f64 {
    g.terms
        .iter()
        .map(|&(a, p, s)| s * f.u.get(a, p) as f64)
        .sum()
}

impl Operator for MergedFaces {
    fn hooks(&self) -> Hooks {
        Hooks::POST_H | Hooks::POST_E
    }

    fn post_h(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        for g in &mut self.groups {
            let c = circulation(g, f);
            g.i = flush((g.da * g.i - g.db * c) as f32) as f64;
            for &(a, p) in &g.faces {
                f.i.set(a, p, g.i as f32);
            }
        }
    }

    fn post_e(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        // ½·L·i^{n+½}·i^{n+3/2} for the energy at t = (n + 1)·dt, with the
        // expression the next post_h will evaluate.
        let mut w = 0.0;
        for g in &self.groups {
            let next = flush((g.da * g.i - g.db * circulation(g, f)) as f32) as f64;
            w += 0.5 * g.l * g.i * next;
        }
        self.energy = w;
    }

    fn energy(&self) -> f64 {
        self.energy
    }
}

/// Estimated stability factor `s = √λ_max(S)/2` of the lossless leapfrog
/// scheme with coefficients `co` on a grid with `nodes` nodes (no merge
/// groups), `S = √cb·Kᵀ·db·K·√cb` (`s < 1`: stable; see the
/// [module docs](self#stability-check)).
///
/// Deterministic power iteration (`iterations` steps, `f64`, single
/// thread) from a start vector concentrated on the E edges listed in
/// `seeds` (`(axis index, engine node index)`), plus a small fixed
/// pseudo-random component on every edge; returns the final Rayleigh
/// quotient, which approaches `λ_max` from below.
pub fn stability_factor(
    co: &Coefficients,
    nodes: [usize; 3],
    seeds: &[(usize, usize)],
    iterations: usize,
) -> f64 {
    stability(co, nodes, &[], seeds, iterations)
}

/// A merge group for [`apply_s`]: `dt/L_g` and the circulation terms
/// `(edge axis index, edge node index, sign)`.
type GroupTerms = (f64, Vec<(usize, usize, f64)>);

fn stability(
    co: &Coefficients,
    n: [usize; 3],
    groups: &[Group],
    seeds: &[(usize, usize)],
    iterations: usize,
) -> f64 {
    let len = n[0] * n[1] * n[2];
    let sq: [Vec<f64>; 3] =
        core::array::from_fn(|a| co.data[3 + a].iter().map(|&c| (c as f64).sqrt()).collect());
    let db: [&[f32]; 3] = core::array::from_fn(|a| &co.data[9 + a][..]);
    let gterms: Vec<GroupTerms> = groups
        .iter()
        .map(|g| {
            (
                g.dt_over_l,
                g.terms
                    .iter()
                    .map(|&(a, p, s)| (a.index(), index(n, p), s))
                    .collect(),
            )
        })
        .collect();
    let mut v: [Vec<f64>; 3] = core::array::from_fn(|a| {
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ (a as u64 + 1);
        (0..len)
            .map(|k| {
                // xorshift: a fixed, thread-independent sequence.
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let r = (state >> 11) as f64 / (1u64 << 53) as f64;
                if sq[a][k] > 0.0 {
                    1e-3 * (r - 0.5)
                } else {
                    0.0
                }
            })
            .collect()
    });
    for &(a, k) in seeds {
        if sq[a][k] > 0.0 {
            v[a][k] += 1.0;
        }
    }
    let mut h: [Vec<f64>; 3] = core::array::from_fn(|_| vec![0.0; len]);
    let mut w: [Vec<f64>; 3] = core::array::from_fn(|_| vec![0.0; len]);
    let mut lambda = 0.0;
    for _ in 0..iterations.max(1) {
        let norm = v.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        for x in v.iter_mut().flatten() {
            *x /= norm;
        }
        apply_s(n, &sq, &db, &gterms, &v, &mut h, &mut w);
        lambda = v
            .iter()
            .flatten()
            .zip(w.iter().flatten())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        core::mem::swap(&mut v, &mut w);
    }
    lambda.max(0.0).sqrt() / 2.0
}

/// `w = S·v` with `S = √cb·(PK)ᵀ·(dt/L)·(PK)·√cb`, using the kernel's curl
/// convention (face `a` at `p`: `u_c(p+e_b) − u_c(p) − u_b(p+e_c) + u_b(p)`
/// with `(b, c) = a.others()`); merged faces have `db = 0` and are covered
/// by `groups` (`(dt/L_g, terms)`); `h` is scratch.
fn apply_s(
    n: [usize; 3],
    sq: &[Vec<f64>; 3],
    db: &[&[f32]; 3],
    groups: &[GroupTerms],
    v: &[Vec<f64>; 3],
    h: &mut [Vec<f64>; 3],
    w: &mut [Vec<f64>; 3],
) {
    let stride = [n[1] * n[2], n[2], 1];
    let x: [Vec<f64>; 3] =
        core::array::from_fn(|a| v[a].iter().zip(&sq[a]).map(|(v, s)| v * s).collect());
    for wa in w.iter_mut() {
        wa.fill(0.0);
    }
    for axis in Axis::ALL {
        let a = axis.index();
        let (bb, cc) = axis.others();
        let (b, c) = (bb.index(), cc.index());
        let ha = &mut h[a];
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let p = [i, j, k];
                    let q = (i * n[1] + j) * n[2] + k;
                    let d = db[a][q] as f64;
                    if d == 0.0 || p[b] + 1 >= n[b] || p[c] + 1 >= n[c] {
                        ha[q] = 0.0;
                        continue;
                    }
                    let curl = x[c][q + stride[b]] - x[c][q] - x[b][q + stride[c]] + x[b][q];
                    ha[q] = d * curl;
                }
            }
        }
        // Kᵀ: scatter each face value to its four edges.
        for q in 0..ha.len() {
            let f = ha[q];
            if f == 0.0 {
                continue;
            }
            w[c][q + stride[b]] += f;
            w[c][q] -= f;
            w[b][q + stride[c]] -= f;
            w[b][q] += f;
        }
    }
    for (d, terms) in groups {
        let curl: f64 = terms.iter().map(|&(a, k, s)| s * x[a][k]).sum();
        let f = d * curl;
        for &(a, k, s) in terms {
            w[a][k] += s * f;
        }
    }
    for a in 0..3 {
        for (wv, s) in w[a].iter_mut().zip(&sq[a]) {
            *wv *= s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{CellMaterials, Grid, TimeStep};

    fn engine(s: f64) -> Engine {
        let g = Grid::uniform([12, 12, 12], [1e-3; 3]).unwrap();
        Engine::builder(g.clone())
            .materials(CellMaterials::vacuum(&g))
            .timestep(TimeStep::Courant(s))
            .threads(1)
            .build()
            .unwrap()
    }

    #[test]
    fn uncut_grid_stability_is_courant_factor() {
        let e = engine(0.9);
        let n = e.grid().nodes();
        let s = stability_factor(&e.coefficients(), n, &[], 400);
        // PEC walls: the highest mode is slightly below the infinite-grid
        // limit.
        assert!(s < 0.9 && s > 0.85, "{s}");
    }

    /// A z-normal face keeping its full x edge at y = 5 but only 2 % of
    /// its area (the y edges 2 % open, the far x edge closed).
    fn sliver() -> ConformalPec {
        let mut c = ConformalPec::new();
        c.push_face(ConformalFace {
            axis: Axis::Z,
            node: [5, 5, 5],
            fraction: 0.02,
        });
        for q in [[5, 5, 5], [6, 5, 5]] {
            c.push_edge(ConformalEdge {
                axis: Axis::Y,
                node: q,
                fraction: 0.02,
            });
        }
        c.push_edge(ConformalEdge {
            axis: Axis::X,
            node: [5, 6, 5],
            fraction: 0.0,
        });
        c
    }

    #[test]
    fn sliver_face_is_merged_and_stable() {
        // Plain Dey–Mittra (threshold 0: no merging) is unstable.
        let raw = sliver().with_options(ConformalOptions {
            merge_below: 0.0,
            max_rounds: 1,
            ..ConformalOptions::default()
        });
        let mut e = engine(0.99);
        let r = raw.apply(&mut e).unwrap();
        assert!(!r.verified && r.stability > 1.5, "{r:?}");
        assert!(r.warning().is_some());
        assert_eq!(r.merged, 0);
        let mut e = engine(0.99);
        let r = sliver().apply(&mut e).unwrap();
        assert!(r.verified, "{r:?}");
        assert_eq!((r.merged, r.groups, r.floored), (2, 1, 0), "{r:?}");
        assert!(r.stability < 0.999 && r.stability > 0.9, "{r:?}");
        // Edge scaling cb·f_e; the merged faces are left to the operator.
        let n = e.grid().nodes();
        let base = engine(0.99);
        let k = index(n, [5, 5, 5]);
        let want = base.coefficients().cb(Axis::Y)[k] as f64 * 0.02;
        assert!((e.coefficients().cb(Axis::Y)[k] as f64 / want - 1.0).abs() < 1e-6);
        assert_eq!(e.coefficients().cb(Axis::X)[index(n, [5, 6, 5])], 0.0);
        assert_eq!(e.coefficients().db(Axis::Z)[k], 0.0);
        // Merged into the face across the full x edge (y − 1).
        assert_eq!(e.coefficients().db(Axis::Z)[index(n, [5, 4, 5])], 0.0);
    }

    #[test]
    fn merged_scheme_conserves_energy() {
        let mut e = engine(0.99);
        let r = sliver().apply(&mut e).unwrap();
        assert!(r.verified);
        {
            let mut f = e.fields_mut();
            f.u.set(Axis::X, [5, 5, 5], 1.0);
            f.u.set(Axis::Y, [3, 7, 4], -0.5);
        }
        e.run(1);
        let w0 = e.energy();
        let mut dev: f64 = 0.0;
        for _ in 0..50 {
            e.run(100);
            dev = dev.max((e.energy() / w0 - 1.0).abs());
        }
        assert!(dev < 1e-5, "{dev}");
        // The in-step energy agrees with the look-ahead.
        let a = e.energy();
        let b = e.step_with_energy();
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_bad_input() {
        let mut e = engine(0.9);
        let mut c = ConformalPec::new();
        c.push_edge(ConformalEdge {
            axis: Axis::X,
            node: [12, 0, 0],
            fraction: 0.5,
        });
        assert!(c.apply(&mut e).is_err());
        let mut c = ConformalPec::new();
        c.push_face(ConformalFace {
            axis: Axis::X,
            node: [0, 0, 0],
            fraction: 1.5,
        });
        assert!(c.apply(&mut e).is_err());
    }
}
