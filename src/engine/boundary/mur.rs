//! First-order Mur absorbing boundary (design §5.6).
//!
//! G. Mur, "Absorbing boundary conditions for the finite-difference
//! approximation of the time-domain electromagnetic-field equations",
//! *IEEE Trans. Electromagn. Compat.* EMC-23(4), 377–382, 1981; see also
//! Taflove & Hagness, *Computational Electrodynamics*, 3rd ed., ch. 6
//! (analytical absorbing boundary conditions, Mur's first-order
//! discretisation of the one-way wave equation).
//!
//! # Scheme
//!
//! On a face normal to axis `a`, every tangential E component `E_t`
//! satisfies, to first order, the one-way wave equation for waves leaving
//! the domain at phase velocity `v`. On the `min` face (outward direction
//! `−a`):
//!
//! ```text
//! ∂E_t/∂a − (1/v)·∂E_t/∂t = 0
//! ```
//!
//! Mur discretises it at the midpoint between the boundary node `b` and the
//! first interior node `1` (distance `Δ` along `a`) and at the half time
//! step `(n + ½)·dt`, averaging both spatial and both temporal samples:
//!
//! ```text
//! E_b^{n+1} = E_1^n + ((v·dt − Δ)/(v·dt + Δ)) · (E_1^{n+1} − E_b^n)
//! ```
//!
//! On the `max` face the sign of the spatial derivative flips, and the
//! same formula results with `b`/`1` mirrored. On a non-uniform mesh `Δ`
//! is the size of the boundary cell along `a` (`Δ_a(0)` or `Δ_a(n_a−2)`).
//!
//! **Integral form.** The engine stores `u = E_t·l` with `l` the edge length
//! along `t`. The boundary edge and its interior neighbour differ only in
//! their coordinate along the normal `a`; both span the same interval along
//! `t` (same tangential index), so they have the same length `l`
//! (`Grid::delta(t, ·)` at the same index). The update is linear and
//! homogeneous, so multiplying by `l` gives exactly the same formula for
//! `u`; it is applied to `u` directly.
//!
//! # Engine integration
//!
//! The face must be built with [`FaceCondition::Pec`](crate::engine::FaceCondition::Pec)
//! (checked at construction: all tangential edges on the face must have
//! `ca = cb = 0`). In `pre_e` the operator saves `u_b^n` and `u_1^n`; the E
//! kernel then zeroes the face edges and advances `u_1`; in `post_e` the
//! operator writes `u_b^{n+1}`. The H kernel of the next step sees the
//! Mur values on the face.
//!
//! # Edges where faces meet
//!
//! A primary edge along `t` lies on at most two outer faces (those normal
//! to the two axes other than `t`). For a tangential edge of a Mur face
//! that also lies on a second face:
//!
//! * If the second face is **not** handled by the same [`MurBoundary`]
//!   (PEC, PMC, …), the edge gets the plain 1-D update along the Mur face's
//!   normal. Its interior neighbour lies on the second face: on a PEC face
//!   it stays zero, so the boundary edge stays exactly zero as well; on a
//!   PMC face it is a regular kernel-updated edge, so a TEM wave guided by
//!   PMC walls is absorbed consistently up to the wall.
//! * If both faces are handled by the same [`MurBoundary`] (an edge of the
//!   domain box), the edge is updated last, as the **average of the two
//!   1-D Mur updates** along the two face normals. Each of them uses an
//!   interior neighbour that is itself a face edge of the other Mur face,
//!   which has already been updated in the same `post_e`. This simple
//!   corner treatment is first order like the rest of the scheme.
//!
//! Configure all Mur faces of a model in **one** [`MurBoundary`]
//! ([`MurBoundary::with_faces`]); two separate operators on adjacent faces
//! would read each other's not-yet-updated (zeroed) edges at the common
//! edge, making the result depend on registration order.
//!
//! Edges inside metal (`ca = cb = 0` from the material map) that touch the
//! face are not special-cased: if the interior neighbour is metal too, it
//! stays zero and the boundary edge stays zero.
//!
//! # Threads
//!
//! Where every edge's interior neighbours lie in the edge's own x-slab
//! (each slab holds at least two x-planes) and the engine merges no face
//! copies (no periodic faces, Cartesian grid), the operator runs as
//! [slab hooks](crate::engine::SlabHooks): each worker saves `u_b^n`,
//! `u_1^n` of its slab's edges after its H update and writes the Mur
//! values after its E update (face edges, then the corners), with no
//! serial hook and no extra barrier. Otherwise it runs serially on the
//! calling thread (`pre_e` / `post_e`). Either way every edge is updated
//! by the same expression from the same values, so results are
//! bit-identical for any engine thread count and in both modes.

use std::sync::{Arc, Mutex};

use crate::axis::Axis;
use crate::constants::C0;
use crate::engine::coeffs::Coefficients;
use crate::engine::{
    EdgeMaterials, Engine, Fields, Fusion, Hooks, Operator, SlabContext, SlabHooks, SlabMut,
    SlabPartition, SlabRef, StepContext,
};
use crate::fdtd::Face;
use crate::{Error, Result};

/// A tangential face edge updated by one 1-D Mur update.
#[derive(Debug, Clone)]
struct Edge {
    /// Position in the operator's global edge order (energy sums).
    order: usize,
    axis: Axis,
    node: [usize; 3],
    inner: [usize; 3],
    k: f32,
    /// `u_b^n`, `u_1^n` saved in `pre_e`.
    saved: [f32; 2],
    /// Capacitance counted by [`MurBoundary::boundary_energy`] \[F\].
    c: f64,
    /// The edge voltage after the last update.
    last: f32,
    /// No update before this step ([`MurBoundary::hold_faces`]).
    hold: u64,
}

/// An edge on two Mur faces: average of two 1-D updates.
#[derive(Debug, Clone)]
struct Corner {
    /// Position in the operator's global order, after all edges.
    order: usize,
    axis: Axis,
    node: [usize; 3],
    inner: [[usize; 3]; 2],
    k: [f32; 2],
    /// `u_b^n`, then `u_1^n` of each direction.
    saved: [f32; 3],
    /// Capacitance counted by [`MurBoundary::boundary_energy`] \[F\].
    c: f64,
    /// The edge voltage after the last update.
    last: f32,
    /// No update before this step ([`MurBoundary::hold_faces`]).
    hold: u64,
}

/// Per-edge phase velocities for the face edges of Mur faces without a
/// fixed velocity, overriding the edge-averaged medium of
/// [`MurBoundary::with_faces_in`] (see [`MurBoundary::with_faces_media`]).
///
/// Keyed by face, edge axis and node (engine indices); edges without an
/// entry fall back to the averaged medium.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MurMedia {
    v: std::collections::BTreeMap<(usize, usize, [usize; 3]), f64>,
}

impl MurMedia {
    /// No entries.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the phase velocity \[m/s\] of the edge along `axis` at `node`
    /// on `face`.
    pub fn insert(&mut self, face: Face, axis: Axis, node: [usize; 3], v: f64) {
        self.v.insert((face.index(), axis.index(), node), v);
    }

    /// The phase velocity of the edge along `axis` at `node` on `face`,
    /// if set.
    pub fn get(&self, face: Face, axis: Axis, node: [usize; 3]) -> Option<f64> {
        self.v.get(&(face.index(), axis.index(), node)).copied()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.v.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

/// First-order Mur absorbing boundary on one or more outer faces, as an
/// engine [`Operator`]. See the [module docs](self) for the scheme, the
/// treatment of domain edges and the requirements.
///
/// ```
/// use ems::engine::boundary::mur::MurBoundary;
/// use ems::engine::{Engine, Grid};
/// use ems::{Face, Mur};
///
/// let grid = Grid::uniform([20, 4, 4], [1e-3; 3])?;
/// // All faces PEC by default, as Mur requires.
/// let mut engine = Engine::builder(grid).threads(1).build()?;
/// let v = Mur::new().velocity();
/// let mur = MurBoundary::with_faces(&engine, &[(Face::XMin, v), (Face::XMax, v)])?;
/// engine.add_operator(mur);
/// engine.run(10);
/// # Ok::<(), ems::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct MurBoundary {
    faces: Vec<(Face, Option<f64>)>,
    /// The edges and corners (empty once handed to `slabs`).
    edges: Vec<Edge>,
    corners: Vec<Corner>,
    /// Number of edges plus corners.
    count: usize,
    count_energy: bool,
    openems_corners: bool,
    openems_order: bool,
    /// Whether slab mode is allowed (no face merges in the engine).
    slab_ok: bool,
    /// Slab mode: the edges and corners of each x-slab.
    slabs: Option<Arc<MurSlabs>>,
    /// Serial mode: the storage locations of every edge's node and
    /// interior neighbours (`[node, inner]`) and corner's
    /// (`[node, inner0, inner1]`), from the first serial call.
    locs: Option<SerialLocs>,
}

/// A field storage location `(slab, offset)` ([`FieldArrays::at`]).
type Loc = (usize, usize);

/// The serial-mode locations of the edges and of the corners.
type SerialLocs = (Vec<[Loc; 2]>, Vec<[Loc; 3]>);

/// The edges and corners of one x-slab (slab mode).
#[derive(Debug, Default)]
struct MurSlab {
    edges: Vec<Edge>,
    corners: Vec<Corner>,
}

/// Slab-mode state of a [`MurBoundary`].
#[derive(Debug)]
struct MurSlabs {
    slabs: Vec<Mutex<MurSlab>>,
    openems_corners: bool,
    openems_order: bool,
}

/// Locks a slab's state (uncontended: only its own worker locks it during
/// a step).
fn lock(m: &Mutex<MurSlab>) -> std::sync::MutexGuard<'_, MurSlab> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The 1-D Mur update `u_b^{n+1}` from `u_1^n` (`i`), `u_b^n` (`b`) and
/// `u_1^{n+1}` (`x`), in ems's order `u_1^n + k·(u_1^{n+1} − u_b^n)` or in
/// openEMS's (`openems`, [`MurBoundary::openems_order`]).
#[inline]
fn mur_1d(i: f32, k: f32, x: f32, b: f32, openems: bool) -> f32 {
    if openems {
        (i - k * b) + k * x
    } else {
        i + k * (x - b)
    }
}

/// The 1-D updates of `edges`, then the corner updates of `corners`, on
/// a field accessed through `get`/`set` (shared by both modes).
fn update(
    edges: &mut [Edge],
    corners: &mut [Corner],
    openems_corners: bool,
    openems_order: bool,
    step: u64,
    get: &dyn Fn(Axis, [usize; 3]) -> f32,
    set: &mut dyn FnMut(Axis, [usize; 3], f32),
) {
    for e in edges.iter_mut() {
        if step < e.hold {
            continue;
        }
        let [b, i] = e.saved;
        let v = mur_1d(i, e.k, get(e.axis, e.inner), b, openems_order);
        set(e.axis, e.node, v);
        e.last = v;
    }
    // After the face edges: corner neighbours are face edges.
    for c in corners.iter_mut() {
        if step < c.hold {
            continue;
        }
        let [b, i0, i1] = c.saved;
        let v = if openems_corners {
            // The face of the lower axis alone, its neighbour at n + 1
            // taken as 0.
            i0 - c.k[0] * b
        } else {
            let v0 = mur_1d(i0, c.k[0], get(c.axis, c.inner[0]), b, openems_order);
            let v1 = mur_1d(i1, c.k[1], get(c.axis, c.inner[1]), b, openems_order);
            0.5 * (v0 + v1)
        };
        set(c.axis, c.node, v);
        c.last = v;
    }
}

impl SlabHooks for MurSlabs {
    fn post_h_slab(&self, _i: &mut SlabMut<'_>, u: &SlabRef<'_>, ctx: &SlabContext<'_>) {
        let mut s = lock(&self.slabs[ctx.slab]);
        let nz = u.nodes()[2];
        let get = |a: Axis, n: [usize; 3]| u.plane(a, n[0])[n[1] * nz + n[2]];
        for e in &mut s.edges {
            e.saved = [get(e.axis, e.node), get(e.axis, e.inner)];
        }
        for c in &mut s.corners {
            c.saved = [
                get(c.axis, c.node),
                get(c.axis, c.inner[0]),
                get(c.axis, c.inner[1]),
            ];
        }
    }

    fn post_e_slab(&self, u: &mut SlabMut<'_>, _i: &SlabRef<'_>, ctx: &SlabContext<'_>) {
        let mut s = lock(&self.slabs[ctx.slab]);
        let MurSlab { edges, corners } = &mut *s;
        let nz = u.nodes()[2];
        let cell = core::cell::RefCell::new(u);
        let get = |a: Axis, n: [usize; 3]| cell.borrow().plane(a, n[0])[n[1] * nz + n[2]];
        let mut set = |a: Axis, n: [usize; 3], v: f32| {
            cell.borrow_mut().plane_mut(a, n[0])[n[1] * nz + n[2]] = v
        };
        update(
            edges,
            corners,
            self.openems_corners,
            self.openems_order,
            ctx.step,
            &get,
            &mut set,
        );
    }

    /// Both hooks act on the voltages only.
    fn fusion(&self) -> Fusion {
        Fusion::Voltages
    }
}

impl MurBoundary {
    /// Mur boundary on the single face `face`, tuned for
    /// `phase_velocity` \[m/s\] (e.g. [`crate::constants::C0`] or
    /// [`crate::Mur::velocity`]).
    ///
    /// # Errors
    /// See [`with_faces`](Self::with_faces).
    pub fn new(engine: &Engine, face: Face, phase_velocity: f64) -> Result<Self> {
        Self::with_faces(engine, &[(face, phase_velocity)])
    }

    /// Mur boundary on several faces, each `(face, phase_velocity)`, with
    /// joint treatment of the domain edges where two of them meet.
    ///
    /// # Errors
    /// [`Error::Invalid`] if a face is listed twice, a phase velocity is not
    /// finite and positive, the mesh has fewer than 3 nodes along a face
    /// normal, or a face was not built as
    /// [`FaceCondition::Pec`](crate::engine::FaceCondition::Pec).
    pub fn with_faces(engine: &Engine, faces: &[(Face, f64)]) -> Result<Self> {
        let faces: Vec<(Face, Option<f64>)> = faces.iter().map(|&(f, v)| (f, Some(v))).collect();
        Self::build(engine, &faces, None, None)
    }

    /// Mur boundary on several faces like [`with_faces`](Self::with_faces),
    /// where a face without a phase velocity (`None`) uses the local
    /// medium speed `c0/√(ε_r·μ_r)` of each boundary edge, from `materials`
    /// (`ε_r` of the interior neighbour edge, `μ_r` of the H face between
    /// them). This is what openEMS does when `MUR_PhaseVelocity` is not
    /// given (design §5.16): in an `ε_r = 4` medium a vacuum-tuned Mur face
    /// reflects 34–49 % of a pulse, a locally tuned one 0.5 %.
    ///
    /// # Errors
    /// See [`with_faces`](Self::with_faces); also if a local velocity is not
    /// finite and positive.
    pub fn with_faces_in(
        engine: &Engine,
        faces: &[(Face, Option<f64>)],
        materials: &EdgeMaterials,
    ) -> Result<Self> {
        Self::build(engine, faces, Some(materials), None)
    }

    /// [`with_faces_in`](Self::with_faces_in) where the face edges listed
    /// in `media` take its phase velocity instead of the averaged medium
    /// (used for openEMS files, which sample the medium at one point per
    /// edge, design §5.16.1).
    ///
    /// # Errors
    /// See [`with_faces_in`](Self::with_faces_in).
    pub fn with_faces_media(
        engine: &Engine,
        faces: &[(Face, Option<f64>)],
        materials: &EdgeMaterials,
        media: &MurMedia,
    ) -> Result<Self> {
        Self::build(engine, faces, Some(materials), Some(media))
    }

    fn build(
        engine: &Engine,
        faces: &[(Face, Option<f64>)],
        materials: Option<&EdgeMaterials>,
        media: Option<&MurMedia>,
    ) -> Result<Self> {
        let grid = engine.grid();
        let n = grid.nodes();
        let dt = engine.dt();
        let face_delta = |face: Face| {
            let a = face.axis();
            if face.is_max() {
                grid.delta(a, n[a.index()] - 2)
            } else {
                grid.delta(a, 0)
            }
        };
        let mut k_face = [None::<f32>; 6];
        for &(face, v) in faces {
            let v = match v {
                Some(v) => v,
                None if materials.is_some() => C0,
                None => {
                    return Err(Error::Invalid(format!(
                        "Mur boundary on {face}: no phase velocity and no materials"
                    )));
                }
            };
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::Invalid(format!(
                    "Mur boundary on {face}: phase velocity {v} must be finite and > 0"
                )));
            }
            if k_face[face.index()].is_some() {
                return Err(Error::Invalid(format!(
                    "Mur boundary: face {face} listed twice"
                )));
            }
            let a = face.axis();
            if n[a.index()] < 3 {
                return Err(Error::Invalid(format!(
                    "Mur boundary on {face}: needs at least 3 mesh lines along {a}, got {}",
                    n[a.index()]
                )));
            }
            k_face[face.index()] = Some(Self::coefficient(v, dt, face_delta(face)) as f32);
        }

        // Boundary plane index and interior neighbour of `node` for `face`.
        let plane = |f: Face| {
            if f.is_max() {
                n[f.axis().index()] - 1
            } else {
                0
            }
        };
        let inner = |f: Face, mut node: [usize; 3]| {
            let a = f.axis().index();
            node[a] = if f.is_max() { node[a] - 1 } else { node[a] + 1 };
            node
        };

        let co = engine.coefficients();
        // Capacitance of a boundary edge with full dual cells on the outer
        // planes: the interior neighbour's (its medium), with the dual
        // extent along each face normal taken as the whole first cell.
        let full_c = |t: Axis, node: [usize; 3], inner: [usize; 3]| -> f64 {
            let i = grid.index(inner);
            let c = Coefficients::capacitance(co.ca(t)[i], co.cb(t)[i], dt);
            let mut f = 1.0;
            for b in Axis::ALL.into_iter().filter(|&b| b != t) {
                let (k, ki) = (node[b.index()], inner[b.index()]);
                let nb = n[b.index()];
                if k == 0 || k + 1 == nb {
                    let cell = grid.delta(b, if k == 0 { 0 } else { nb - 2 });
                    f *= cell / grid.dual_delta(b, ki);
                }
            }
            if c.is_finite() { c * f } else { 0.0 }
        };
        let mut edges = Vec::new();
        let mut corners = Vec::new();
        // Coefficient of the edge along `t` at `node` for `face`: fixed per
        // face, or from the local medium.
        let local = |face: Face, t: Axis, o: Axis, node: [usize; 3]| -> Result<f32> {
            let m = materials.expect("local velocity needs materials");
            let a = face.axis();
            let inner = inner(face, node);
            // H face between the boundary edge and its neighbour: normal to
            // `o`, lowest corner at the smaller of the two planes.
            let mut hf = node;
            hf[a.index()] = node[a.index()].min(inner[a.index()]);
            let v = match media.and_then(|md| md.get(face, t, node)) {
                Some(v) => v,
                None => {
                    let eps = m.eps_r(t)[grid.index(inner)];
                    let mu = m.mu_r(o)[grid.index(hf)];
                    C0 / (eps * mu).sqrt()
                }
            };
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::Invalid(format!(
                    "Mur boundary on {face}: local phase velocity {v} at {node:?} must be \
                     finite and > 0"
                )));
            }
            Ok(Self::coefficient(v, dt, face_delta(face)) as f32)
        };
        let fixed: [bool; 6] =
            core::array::from_fn(|i| faces.iter().any(|&(f, v)| f.index() == i && v.is_some()));
        let coef = |face: Face, t: Axis, o: Axis, node: [usize; 3]| -> Result<f32> {
            if fixed[face.index()] {
                Ok(k_face[face.index()].expect("set above"))
            } else {
                local(face, t, o, node)
            }
        };
        for &(face, _) in faces {
            let a = face.axis();
            for t in Axis::ALL.into_iter().filter(|&t| t != a) {
                // The third axis: the edge may also lie on one of its faces.
                let o = Axis::ALL
                    .into_iter()
                    .find(|&x| x != a && x != t)
                    .expect("three axes");
                let (ca, cb) = (co.ca(t), co.cb(t));
                for p in 0..n[t.index()] - 1 {
                    for q in 0..n[o.index()] {
                        let mut node = [0; 3];
                        node[a.index()] = plane(face);
                        node[t.index()] = p;
                        node[o.index()] = q;
                        let idx = grid.index(node);
                        if ca[idx] != 0.0 || cb[idx] != 0.0 {
                            return Err(Error::Invalid(format!(
                                "Mur boundary on {face}: the face must be built with \
                                 FaceCondition::Pec (edge {t} at {node:?} is not)"
                            )));
                        }
                        // Second face this edge lies on, if it is Mur too.
                        let other = [false, true]
                            .map(|max| Face::new(o, max))
                            .into_iter()
                            .find(|&g| plane(g) == q && k_face[g.index()].is_some());
                        match other {
                            None => edges.push(Edge {
                                order: 0,
                                axis: t,
                                node,
                                inner: inner(face, node),
                                k: coef(face, t, o, node)?,
                                saved: [0.0; 2],
                                hold: 0,
                                c: full_c(t, node, inner(face, node)),
                                last: 0.0,
                            }),
                            // Each domain edge is visited from both faces;
                            // keep it once (from the lower-index face).
                            Some(g) if face.index() < g.index() => corners.push(Corner {
                                order: 0,
                                axis: t,
                                node,
                                inner: [inner(face, node), inner(g, node)],
                                k: [coef(face, t, o, node)?, coef(g, t, a, node)?],
                                saved: [0.0; 3],
                                hold: 0,
                                c: full_c(t, node, inner(g, inner(face, node))),
                                last: 0.0,
                            }),
                            Some(_) => {}
                        }
                    }
                }
            }
        }
        for (o, e) in edges.iter_mut().enumerate() {
            e.order = o;
        }
        for (o, c) in corners.iter_mut().enumerate() {
            c.order = edges.len() + o;
        }
        Ok(Self {
            faces: faces.to_vec(),
            count: edges.len() + corners.len(),
            edges,
            corners,
            count_energy: false,
            openems_corners: false,
            openems_order: false,
            // Slab hooks run before every serial hook: only when no operator
            // registered before this one touches E serially around the E
            // update (the serial order would change).
            slab_ok: !grid.is_cylindrical()
                && engine.periodic.is_none()
                && !engine
                    .ops
                    .iter()
                    .any(|r| r.hooks.intersects(Hooks::PRE_E | Hooks::POST_E)),
            slabs: None,
            locs: None,
        })
    }

    /// Holds the boundary edges of `faces` (and the domain edges on them)
    /// before step `until`: they are not updated then and keep what the
    /// other operators write (0 from their PEC coefficients, plus sources).
    /// openEMS does this to a Mur face that carries an E excitation while
    /// the excitation runs (black box: such a face stays 0 except for the
    /// source, which holds just the signal, up to step `L + 11` for an
    /// excitation of `L` steps, measured for four signal lengths and two
    /// time steps; the other Mur faces are not affected; design §5.16.3).
    /// Call it before registering.
    pub fn hold_faces(mut self, faces: &[Face], until: u64) -> Self {
        let on = |node: [usize; 3], inner: [usize; 3]| {
            faces.iter().any(|f| {
                let a = f.axis().index();
                node[a] != inner[a] && (node[a] == 0) != f.is_max()
            })
        };
        for e in &mut self.edges {
            if on(e.node, e.inner) {
                e.hold = until;
            }
        }
        for c in &mut self.corners {
            if on(c.node, c.inner[0]) || on(c.node, c.inner[1]) {
                c.hold = until;
            }
        }
        self
    }

    /// openEMS's update of the edges where two Mur faces meet (default
    /// off): only the face of the lower axis (x before y before z) updates
    /// the edge, with its interior neighbour at `n + 1` taken as 0,
    /// `u_c^{n+1} = u_1^n − k_1·u_c^n`, as if that face read the other's
    /// edge before it was updated. Identified by least-squares fits of the
    /// twelve domain edges of an all-Mur box with a different last cell on
    /// every face, from openEMS's per-step field dumps (black box, residual
    /// 1e-7; design §5.16.1; the uniform box first measured could not tell
    /// this from the average of both faces); ems's own corner average uses
    /// the updated neighbours.
    pub fn openems_corners(mut self, on: bool) -> Self {
        self.openems_corners = on;
        self
    }

    /// openEMS's operation order of the 1-D update (default off):
    /// `u_b^{n+1} = (u_1^n − k·u_b^n) + k·u_1^{n+1}` in `f32` instead of
    /// ems's `u_1^n + k·(u_1^{n+1} − u_b^n)`. The same value up to
    /// round-off; this order reproduces openEMS's face edges bit for bit
    /// (fitted from openEMS's per-step field dumps of a box with Mur
    /// faces and 1 m cells, black box; design §5.16.8). With
    /// [`openems_corners`](Self::openems_corners) it is the part before
    /// the interior neighbour's new value of the same expression.
    pub fn openems_order(mut self, on: bool) -> Self {
        self.openems_order = on;
        self
    }

    /// Counts the energy of the boundary edges in
    /// [`Operator::energy`] (default off): `Σ ½·C_e·u_e²` over the face
    /// edges, whose own update coefficients are those of PEC, with `C_e`
    /// the capacitance of the interior neighbour's medium over a full
    /// first cell along each face normal. This is the energy openEMS's end
    /// criterion includes (its operator gives boundary edges full dual
    /// cells; black-box comparison, design §5.16.1). With it the energy
    /// of a Mur-terminated box at openEMS's stop step agrees to 1e-3 dB
    /// at −32 dB (3.5e-3 dB without).
    pub fn count_energy(mut self, on: bool) -> Self {
        self.count_energy = on;
        self
    }

    /// The energy of the boundary edges counted with
    /// [`count_energy`](Self::count_energy) \[J\] (`t = n·dt`).
    pub fn boundary_energy(&self) -> f64 {
        // In the operator's global order, whatever the mode.
        let mut v = vec![(0.0f64, 0.0f32); self.count];
        let mut put = |edges: &[Edge], corners: &[Corner]| {
            for e in edges {
                v[e.order] = (e.c, e.last);
            }
            for c in corners {
                v[c.order] = (c.c, c.last);
            }
        };
        put(&self.edges, &self.corners);
        if let Some(sl) = &self.slabs {
            for m in &sl.slabs {
                let s = lock(m);
                put(&s.edges, &s.corners);
            }
        }
        v.iter()
            .map(|&(c, u)| 0.5 * c * f64::from(u) * f64::from(u))
            .sum()
    }

    /// The faces handled by this operator with their phase velocities
    /// (`None`: the local medium speed, [`with_faces_in`](Self::with_faces_in)).
    pub fn faces(&self) -> &[(Face, Option<f64>)] {
        &self.faces
    }

    /// Number of boundary edges updated each step.
    pub fn edge_count(&self) -> usize {
        self.count
    }

    /// Whether the operator runs as slab hooks (see the
    /// [module docs](self#threads)) rather than serially.
    pub fn is_slab_parallel(&self) -> bool {
        self.slabs.is_some()
    }

    /// The Mur coefficient `(v·dt − Δ)/(v·dt + Δ)` for phase velocity `v`,
    /// time step `dt` and node distance `delta`.
    pub fn coefficient(v: f64, dt: f64, delta: f64) -> f64 {
        (v * dt - delta) / (v * dt + delta)
    }
}

impl Operator for MurBoundary {
    fn hooks(&self) -> Hooks {
        if self.slabs.is_some() {
            Hooks::NONE
        } else {
            Hooks::PRE_E | Hooks::POST_E
        }
    }

    /// Slab mode where possible (see the [module docs](self#threads)).
    fn slab_hooks(&mut self, part: &SlabPartition<'_>) -> Option<Arc<dyn SlabHooks>> {
        // A clone of a registered operator: take its edges back first.
        self.locs = None;
        if let Some(sl) = self.slabs.take() {
            for m in &sl.slabs {
                let s = lock(m);
                self.edges.extend(s.edges.iter().cloned());
                self.corners.extend(s.corners.iter().cloned());
            }
            self.edges.sort_by_key(|e| e.order);
            self.corners.sort_by_key(|c| c.order);
        }
        if !self.slab_ok {
            return None;
        }
        let slab_of = |i: usize| (0..part.slabs()).find(|&s| part.planes(s).contains(&i));
        let same = |node: [usize; 3], others: &[[usize; 3]]| -> Option<usize> {
            let s = slab_of(node[0])?;
            others
                .iter()
                .all(|o| part.planes(s).contains(&o[0]))
                .then_some(s)
        };
        let mut per: Vec<MurSlab> = (0..part.slabs()).map(|_| MurSlab::default()).collect();
        for e in &self.edges {
            per[same(e.node, &[e.inner])?].edges.push(e.clone());
        }
        for c in &self.corners {
            per[same(c.node, &c.inner)?].corners.push(c.clone());
        }
        self.edges.clear();
        self.corners.clear();
        let sl = Arc::new(MurSlabs {
            slabs: per.into_iter().map(Mutex::new).collect(),
            openems_corners: self.openems_corners,
            openems_order: self.openems_order,
        });
        self.slabs = Some(Arc::clone(&sl));
        Some(sl)
    }

    fn pre_e(&mut self, fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        let u = &fields.u;
        let (le, lc) = self.locs.get_or_insert_with(|| {
            let lay = u.layout();
            (
                self.edges
                    .iter()
                    .map(|e| [lay.locate(e.node), lay.locate(e.inner)])
                    .collect(),
                self.corners
                    .iter()
                    .map(|c| {
                        [
                            lay.locate(c.node),
                            lay.locate(c.inner[0]),
                            lay.locate(c.inner[1]),
                        ]
                    })
                    .collect(),
            )
        });
        for (e, l) in self.edges.iter_mut().zip(le.iter()) {
            e.saved = [u.at(e.axis, l[0]), u.at(e.axis, l[1])];
        }
        for (c, l) in self.corners.iter_mut().zip(lc.iter()) {
            c.saved = [u.at(c.axis, l[0]), u.at(c.axis, l[1]), u.at(c.axis, l[2])];
        }
    }

    /// The same updates as the slab mode's (`update`), on the locations
    /// precomputed in `pre_e`.
    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        let u = &mut fields.u;
        let Some((le, lc)) = &self.locs else {
            return;
        };
        for (e, l) in self.edges.iter_mut().zip(le.iter()) {
            if ctx.step < e.hold {
                continue;
            }
            let [b, i] = e.saved;
            let v = mur_1d(i, e.k, u.at(e.axis, l[1]), b, self.openems_order);
            u.set_at(e.axis, l[0], v);
            e.last = v;
        }
        // After the face edges: corner neighbours are face edges.
        for (c, l) in self.corners.iter_mut().zip(lc.iter()) {
            if ctx.step < c.hold {
                continue;
            }
            let [b, i0, i1] = c.saved;
            let v = if self.openems_corners {
                // As the slab update: the lower axis's face alone.
                i0 - c.k[0] * b
            } else {
                let v0 = mur_1d(i0, c.k[0], u.at(c.axis, l[1]), b, self.openems_order);
                let v1 = mur_1d(i1, c.k[1], u.at(c.axis, l[2]), b, self.openems_order);
                0.5 * (v0 + v1)
            };
            u.set_at(c.axis, l[0], v);
            c.last = v;
        }
    }

    /// [`boundary_energy`](Self::boundary_energy) with
    /// [`count_energy`](Self::count_energy), else 0.
    fn energy(&self) -> f64 {
        if self.count_energy {
            self.boundary_energy()
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{FaceCondition, Grid};

    fn engine(faces: crate::engine::FaceConditions) -> Engine {
        let g = Grid::uniform([4, 3, 5], [1e-3; 3]).unwrap();
        Engine::builder(g).faces(faces).threads(1).build().unwrap()
    }

    #[test]
    fn counts_and_validation() {
        let e = engine(Default::default());
        // x_min face: y edges 3·6 (j<3, k<6), z edges 4·5.
        let m = MurBoundary::new(&e, Face::XMin, 3e8).unwrap();
        assert_eq!(m.edge_count(), 3 * 6 + 4 * 5);
        assert_eq!(m.corners.len(), 0);
        // x_min + z_min share the y edges at x = 0, z = 0 (3 of them).
        let m = MurBoundary::with_faces(&e, &[(Face::XMin, 3e8), (Face::ZMin, 3e8)]).unwrap();
        assert_eq!(m.corners.len(), 3);
        assert_eq!(m.edge_count(), (3 * 6 + 4 * 5) + (4 * 4 + 5 * 3) - 3);
        // All six faces: 12 box edges = 4·(4 + 3 + 5) corner edges.
        let all: Vec<_> = Face::ALL.iter().map(|&f| (f, 3e8)).collect();
        let m = MurBoundary::with_faces(&e, &all).unwrap();
        assert_eq!(m.corners.len(), 4 * (4 + 3 + 5));
        assert!(MurBoundary::new(&e, Face::XMin, 0.0).is_err());
        assert!(MurBoundary::new(&e, Face::XMin, f64::NAN).is_err());
        assert!(MurBoundary::with_faces(&e, &[(Face::YMax, 1.0), (Face::YMax, 1.0)]).is_err());
        let e =
            engine(crate::engine::FaceConditions::default().with(Face::ZMax, FaceCondition::Pmc));
        assert!(MurBoundary::new(&e, Face::ZMax, 3e8).is_err());
        assert!(MurBoundary::new(&e, Face::ZMin, 3e8).is_ok());
        let thin = Engine::builder(Grid::uniform([1, 3, 3], [1e-3; 3]).unwrap())
            .threads(1)
            .build()
            .unwrap();
        assert!(MurBoundary::new(&thin, Face::XMin, 3e8).is_err());
    }

    #[test]
    fn local_cell_size() {
        let g = Grid::new([
            vec![0.0, 1e-3, 3e-3, 3.5e-3],
            vec![0.0, 1e-3],
            vec![0.0, 1e-3],
        ])
        .unwrap();
        let e = Engine::builder(g).threads(1).build().unwrap();
        let m = MurBoundary::with_faces(&e, &[(Face::XMin, 2e8), (Face::XMax, 2e8)]).unwrap();
        for ed in &m.edges {
            let (delta, inner) = if ed.node[0] == 0 {
                (1e-3, 1)
            } else {
                (0.5e-3, 2)
            };
            assert_eq!(ed.inner[0], inner);
            assert_eq!(ed.k, MurBoundary::coefficient(2e8, e.dt(), delta) as f32);
        }
        assert_eq!(m.edges.len(), 2 * (2 + 2));
    }

    #[test]
    fn coefficient() {
        // Magic time step: v·dt = Δ → plain shift u_b^{n+1} = u_1^n.
        assert_eq!(MurBoundary::coefficient(1.0, 2.0, 2.0), 0.0);
        let k = MurBoundary::coefficient(3e8, 1e-12, 1e-3);
        assert!(k < 0.0 && k > -1.0);
    }
}
