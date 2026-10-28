//! Absorbing sheets inside the domain: first-order Mur on a mesh plane
//! (csx `AbsorbingBC`, design §5.6).
//!
//! An [`AbsorbingSheet`] is a rectangle on a mesh plane normal to axis `a`
//! whose tangential E edges obey Mur's first-order one-way wave equation
//! for waves travelling along `a` in the direction of the sheet's
//! **outward normal** (`+a` if [`AbcSheet::positive`], else `−a`). The
//! discretisation is the one of the outer-face Mur boundary (see
//! [`mur`](super::mur)): with `b` the sheet edge, `1` its neighbour one
//! node *against* the outward normal (on the absorbing side) and `Δ` their
//! distance,
//!
//! ```text
//! u_b^{n+1} = u_1^n + ((v·dt − Δ)/(v·dt + Δ)) · (u_1^{n+1} − u_b^n)
//! ```
//!
//! **Behaviour.** The sheet value depends only on the fields on the
//! absorbing side, so a wave arriving from that side (travelling along the
//! outward normal) sees a non-reflecting termination, up to the usual Mur
//! error. Unlike an outer face, the region behind the sheet still exists:
//! the sheet acts on it like a hard source carrying the transmitted
//! one-way wave, so a pulse absorbed by the sheet continues past it
//! (approximately unchanged at normal incidence). A wave arriving from
//! the *other* side is not absorbed: the sheet edges ignore it, which makes
//! the sheet a hard (prescribed-E) wall for it, and it is reflected
//! (inverted, like at PEC, where the absorbing side is quiet). Put the
//! sheet where nothing comes from behind it, typically at the end of a
//! feed line or a waveguide in front of a PEC/PML face.
//!
//! **Phase velocity.** [`SheetVelocity::Fixed`] uses one velocity for
//! every edge; [`SheetVelocity::Local`] uses `c0/√(ε_r·μ_r)` per edge from
//! the effective materials of the cell between the sheet and the
//! absorbing-side neighbour (the neighbour edge's `ε_r` and the `μ_r` of
//! the H face spanning both edges), which is exact for a TEM wave in a
//! locally homogeneous medium.
//!
//! **Edges.** Every tangential edge of the rectangle is updated, except
//! edges the engine keeps at zero (`ca = cb = 0`: metal, or PEC outer
//! faces) when the sheet is inside the domain; such edges stay zero. On an
//! outer plane the sheet behaves like a Mur face restricted to the
//! rectangle (the face edges are written regardless of the PEC condition).
//!
//! # Engine integration and threads
//!
//! `pre_e` saves `u_b^n` and `u_1^n`, the E kernel advances both normally,
//! `post_e` overwrites `u_b^{n+1}`. The operator runs on the calling thread
//! with exclusive field access in a fixed order; results are bit-identical
//! for any thread count. Register it after operators that modify E in
//! `post_e` (dispersive media, Mur faces), so that it sees their final
//! interior values; [`Compiled::build`](crate::compile::Compiled::build)
//! does.

use crate::axis::Axis;
use crate::constants::C0;
use crate::engine::{EdgeMaterials, Engine, Fields, Hooks, Operator, StepContext};
use crate::{Error, Result};

/// Geometry of an absorbing sheet in node indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbcSheet {
    /// Sheet normal.
    pub normal: Axis,
    /// Mesh plane index along the normal.
    pub plane: usize,
    /// Outward normal (direction of the absorbed waves) is `+normal`.
    pub positive: bool,
    /// Lowest node of the rectangle (the `normal` entry is ignored).
    pub start: [usize; 3],
    /// Highest node of the rectangle, inclusive (the `normal` entry is
    /// ignored).
    pub stop: [usize; 3],
}

/// Phase velocity choice of an [`AbsorbingSheet`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum SheetVelocity<'a> {
    /// The same velocity \[m/s\] for every edge.
    Fixed(f64),
    /// `c0/√(ε_r·μ_r)` per edge from these effective materials (the ones
    /// the engine was built with); see the [module docs](self).
    Local(&'a EdgeMaterials),
}

#[derive(Debug, Clone)]
struct Edge {
    axis: Axis,
    node: [usize; 3],
    inner: [usize; 3],
    k: f32,
    /// `k` with openEMS's `f32` `v·dt` ([`AbsorbingSheet::openems_order`]).
    k_openems: f32,
    /// `u_b^n`, `u_1^n` saved in `pre_e`.
    saved: [f32; 2],
}

/// First-order Mur absorbing sheet on an interior (or outer) mesh plane,
/// as an engine [`Operator`]; see the [module docs](self).
///
/// ```
/// use ems::engine::boundary::abc::{AbcSheet, AbsorbingSheet, SheetVelocity};
/// use ems::engine::{Engine, Grid};
/// use ems::Axis;
///
/// let grid = Grid::uniform([20, 4, 4], [1e-3; 3])?;
/// let mut engine = Engine::builder(grid).threads(1).build()?;
/// // Absorb waves travelling in +x at plane x = 15, over the whole cross-section.
/// let sheet = AbcSheet { normal: Axis::X, plane: 15, positive: true, start: [0; 3], stop: [0, 4, 4] };
/// let op = AbsorbingSheet::new(&engine, &sheet, SheetVelocity::Fixed(ems::constants::C0))?;
/// engine.add_operator(op);
/// engine.run(10);
/// # Ok::<(), ems::Error>(())
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AbsorbingSheet {
    sheet: AbcSheet,
    edges: Vec<Edge>,
    openems_order: bool,
}

impl AbsorbingSheet {
    /// Builds the sheet operator for `engine`.
    ///
    /// # Errors
    /// [`Error::Invalid`] if the plane or rectangle is outside the mesh, the
    /// sheet has no neighbour plane on its absorbing side (`plane = 0` with
    /// a positive, or the last plane with a negative outward normal), a
    /// velocity is not finite and positive, or the materials do not match
    /// the grid.
    pub fn new(engine: &Engine, sheet: &AbcSheet, velocity: SheetVelocity<'_>) -> Result<Self> {
        let grid = engine.grid();
        let n = grid.nodes();
        let a = sheet.normal;
        let ai = a.index();
        let s = *sheet;
        if s.plane >= n[ai] {
            return Err(Error::Invalid(format!(
                "absorbing sheet: plane {} outside the mesh ({} lines along {a})",
                s.plane, n[ai]
            )));
        }
        let inner_plane = if s.positive {
            s.plane.checked_sub(1)
        } else {
            Some(s.plane + 1).filter(|&p| p < n[ai])
        }
        .ok_or_else(|| {
            Error::Invalid(format!(
                "absorbing sheet at {a} plane {}: no mesh plane on its absorbing side",
                s.plane
            ))
        })?;
        for d in Axis::ALL.into_iter().filter(|&d| d != a) {
            let di = d.index();
            if s.start[di] > s.stop[di] || s.stop[di] >= n[di] {
                return Err(Error::Invalid(format!(
                    "absorbing sheet: range {}..={} along {d} invalid for {} lines",
                    s.start[di], s.stop[di], n[di]
                )));
            }
        }
        if let SheetVelocity::Local(m) = velocity
            && m.nodes() != n
        {
            return Err(Error::Invalid(
                "absorbing sheet: materials do not match the grid".into(),
            ));
        }
        let lo = s.plane.min(inner_plane);
        let delta = grid.delta(a, lo);
        let dt = engine.dt();
        let outer = s.plane == 0 || s.plane == n[ai] - 1;
        let co = engine.coefficients();
        let mut edges = Vec::new();
        for t in Axis::ALL.into_iter().filter(|&t| t != a) {
            let o = Axis::ALL
                .into_iter()
                .find(|&x| x != a && x != t)
                .expect("three axes");
            let (ti, oi) = (t.index(), o.index());
            let (ca, cb) = (co.ca(t), co.cb(t));
            for p in s.start[ti]..s.stop[ti] {
                for q in s.start[oi]..=s.stop[oi] {
                    let mut node = [0; 3];
                    node[ai] = s.plane;
                    node[ti] = p;
                    node[oi] = q;
                    let idx = grid.index(node);
                    if !outer && ca[idx] == 0.0 && cb[idx] == 0.0 {
                        continue;
                    }
                    let mut inner = node;
                    inner[ai] = inner_plane;
                    let v = match velocity {
                        SheetVelocity::Fixed(v) => v,
                        SheetVelocity::Local(m) => {
                            let mut face = node;
                            face[ai] = lo;
                            let eps = m.eps_r(t)[grid.index(inner)];
                            let mu = m.mu_r(o)[grid.index(face)];
                            C0 / (eps * mu).sqrt()
                        }
                    };
                    if !(v.is_finite() && v > 0.0) {
                        return Err(Error::Invalid(format!(
                            "absorbing sheet: phase velocity {v} must be finite and > 0"
                        )));
                    }
                    edges.push(Edge {
                        axis: t,
                        node,
                        inner,
                        k: Self::coefficient(v, dt, delta),
                        k_openems: Self::coefficient(f64::from((v * dt) as f32), 1.0, delta),
                        saved: [0.0; 2],
                    });
                }
            }
        }
        Ok(Self {
            sheet: s,
            edges,
            openems_order: false,
        })
    }

    /// openEMS's operation order of the update (default off), as
    /// [`MurBoundary::openems_order`](super::mur::MurBoundary::openems_order):
    /// `(u_1^n − k·u_b^n) + k·u_1^{n+1}` in `f32`, with `v·dt` rounded to
    /// `f32` inside `k = (v·dt − Δ)/(v·dt + Δ)` (unlike the Mur faces,
    /// whose `k` is the correctly rounded value). Black box: per-step
    /// dumps of `AbsorbingBC` sheets with 1 m, 1 mm and 0.7 mm cells,
    /// every update bit for bit (design §5.16.8).
    pub fn openems_order(mut self, on: bool) -> Self {
        self.openems_order = on;
        self
    }

    /// The coefficient `(v·dt − Δ)/(v·dt + Δ)`, rounded to `f32`.
    fn coefficient(v: f64, dt: f64, delta: f64) -> f32 {
        ((v * dt - delta) / (v * dt + delta)) as f32
    }

    /// The sheet geometry.
    pub fn sheet(&self) -> &AbcSheet {
        &self.sheet
    }

    /// Number of edges updated each step.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Operator for AbsorbingSheet {
    fn hooks(&self) -> Hooks {
        Hooks::PRE_E | Hooks::POST_E
    }

    fn pre_e(&mut self, fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        let u = &fields.u;
        for e in &mut self.edges {
            e.saved = [u.get(e.axis, e.node), u.get(e.axis, e.inner)];
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        let u = &mut fields.u;
        for e in &self.edges {
            let [b, i] = e.saved;
            let x = u.get(e.axis, e.inner);
            let v = if self.openems_order {
                (i - e.k_openems * b) + e.k_openems * x
            } else {
                i + e.k * (x - b)
            };
            u.set(e.axis, e.node, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Grid;

    fn engine() -> Engine {
        let g = Grid::uniform([6, 3, 4], [1e-3; 3]).unwrap();
        Engine::builder(g).threads(1).build().unwrap()
    }

    #[test]
    fn counts_and_validation() {
        let e = engine();
        let full = AbcSheet {
            normal: Axis::X,
            plane: 3,
            positive: true,
            start: [0; 3],
            stop: [0, 3, 4],
        };
        let op = AbsorbingSheet::new(&e, &full, SheetVelocity::Fixed(C0)).unwrap();
        // Interior y edges j<3, k=1..3 (k=0,4 are on the PEC z faces), z
        // edges k<4, j=1..2.
        assert_eq!(op.edge_count(), 3 * 3 + 4 * 2);
        assert!(op.edges.iter().all(|ed| ed.inner[0] == 2));
        let neg = AbcSheet {
            positive: false,
            ..full
        };
        let op = AbsorbingSheet::new(&e, &neg, SheetVelocity::Fixed(C0)).unwrap();
        assert!(op.edges.iter().all(|ed| ed.inner[0] == 4));
        // Outer plane: all face edges, inner neighbour inside.
        let face = AbcSheet {
            plane: 0,
            positive: false,
            ..full
        };
        let op = AbsorbingSheet::new(&e, &face, SheetVelocity::Fixed(C0)).unwrap();
        assert_eq!(op.edge_count(), 3 * 5 + 4 * 4);
        let bad = |s: AbcSheet| AbsorbingSheet::new(&e, &s, SheetVelocity::Fixed(C0)).is_err();
        assert!(bad(AbcSheet {
            plane: 0,
            positive: true,
            ..full
        }));
        assert!(bad(AbcSheet {
            plane: 6,
            positive: false,
            ..full
        }));
        assert!(bad(AbcSheet {
            stop: [0, 4, 4],
            ..full
        }));
        assert!(AbsorbingSheet::new(&e, &full, SheetVelocity::Fixed(-1.0)).is_err());
    }

    #[test]
    fn local_velocity() {
        let e = engine();
        let mut m = EdgeMaterials::vacuum(e.grid());
        m.eps_r_mut(Axis::Y).fill(4.0);
        let s = AbcSheet {
            normal: Axis::X,
            plane: 3,
            positive: true,
            start: [0, 1, 1],
            stop: [0, 2, 2],
        };
        let op = AbsorbingSheet::new(&e, &s, SheetVelocity::Local(&m)).unwrap();
        let k = |v: f64| ((v * e.dt() - 1e-3) / (v * e.dt() + 1e-3)) as f32;
        for ed in &op.edges {
            let v = if ed.axis == Axis::Y { C0 / 2.0 } else { C0 };
            assert_eq!(ed.k, k(v), "{ed:?}");
        }
        assert_eq!(op.edge_count(), 2 + 2);
    }
}
