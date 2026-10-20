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
//! The operator runs on the calling thread with exclusive field access and
//! processes its edges in a fixed order, so results are bit-identical for
//! any engine thread count.

use crate::axis::Axis;
use crate::constants::C0;
use crate::engine::{EdgeMaterials, Engine, Fields, Hooks, Operator, StepContext};
use crate::fdtd::Face;
use crate::{Error, Result};

/// A tangential face edge updated by one 1-D Mur update.
#[derive(Debug, Clone)]
struct Edge {
    axis: Axis,
    node: [usize; 3],
    inner: [usize; 3],
    k: f32,
    /// `u_b^n`, `u_1^n` saved in `pre_e`.
    saved: [f32; 2],
}

/// An edge on two Mur faces: average of two 1-D updates.
#[derive(Debug, Clone)]
struct Corner {
    axis: Axis,
    node: [usize; 3],
    inner: [[usize; 3]; 2],
    k: [f32; 2],
    /// `u_b^n`, then `u_1^n` of each direction.
    saved: [f32; 3],
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
    edges: Vec<Edge>,
    corners: Vec<Corner>,
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
        Self::build(engine, &faces, None)
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
        Self::build(engine, faces, Some(materials))
    }

    fn build(
        engine: &Engine,
        faces: &[(Face, Option<f64>)],
        materials: Option<&EdgeMaterials>,
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
            let eps = m.eps_r(t)[grid.index(inner)];
            let mu = m.mu_r(o)[grid.index(hf)];
            let v = C0 / (eps * mu).sqrt();
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
                                axis: t,
                                node,
                                inner: inner(face, node),
                                k: coef(face, t, o, node)?,
                                saved: [0.0; 2],
                            }),
                            // Each domain edge is visited from both faces;
                            // keep it once (from the lower-index face).
                            Some(g) if face.index() < g.index() => corners.push(Corner {
                                axis: t,
                                node,
                                inner: [inner(face, node), inner(g, node)],
                                k: [coef(face, t, o, node)?, coef(g, t, a, node)?],
                                saved: [0.0; 3],
                            }),
                            Some(_) => {}
                        }
                    }
                }
            }
        }
        Ok(Self {
            faces: faces.to_vec(),
            edges,
            corners,
        })
    }

    /// The faces handled by this operator with their phase velocities
    /// (`None`: the local medium speed, [`with_faces_in`](Self::with_faces_in)).
    pub fn faces(&self) -> &[(Face, Option<f64>)] {
        &self.faces
    }

    /// Number of boundary edges updated each step.
    pub fn edge_count(&self) -> usize {
        self.edges.len() + self.corners.len()
    }

    /// The Mur coefficient `(v·dt − Δ)/(v·dt + Δ)` for phase velocity `v`,
    /// time step `dt` and node distance `delta`.
    pub fn coefficient(v: f64, dt: f64, delta: f64) -> f64 {
        (v * dt - delta) / (v * dt + delta)
    }
}

impl Operator for MurBoundary {
    fn hooks(&self) -> Hooks {
        Hooks::PRE_E | Hooks::POST_E
    }

    fn pre_e(&mut self, fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        let u = &fields.u;
        for e in &mut self.edges {
            e.saved = [u.get(e.axis, e.node), u.get(e.axis, e.inner)];
        }
        for c in &mut self.corners {
            c.saved = [
                u.get(c.axis, c.node),
                u.get(c.axis, c.inner[0]),
                u.get(c.axis, c.inner[1]),
            ];
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        let u = &mut fields.u;
        for e in &self.edges {
            let [b, i] = e.saved;
            let v = i + e.k * (u.get(e.axis, e.inner) - b);
            u.set(e.axis, e.node, v);
        }
        // After the face edges: corner neighbours are face edges.
        for c in &self.corners {
            let [b, i0, i1] = c.saved;
            let v0 = i0 + c.k[0] * (u.get(c.axis, c.inner[0]) - b);
            let v1 = i1 + c.k[1] * (u.get(c.axis, c.inner[1]) - b);
            u.set(c.axis, c.node, 0.5 * (v0 + v1));
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
