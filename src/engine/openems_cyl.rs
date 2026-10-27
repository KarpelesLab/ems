//! openEMS's arithmetic on the seam of a closed cylindrical grid (models
//! read from openEMS files, design §5.16.8).
//!
//! ems stores the seam (the `α = 0` half-plane of a closed azimuth) twice,
//! as the first and the last α plane, and joins the copies with the
//! periodic merge (§5.6.1): each copy of a tangential E edge is updated
//! over its half dual face and both are replaced by the `1/cb`-weighted
//! mean; each copy of an `H_α` dual edge holds the current of its half dual
//! length. That is the exact Yee update of the seam edge, but in `f32` it
//! rounds differently from openEMS, which updates every seam edge once
//! with the coefficients of its whole dual face and the same summation
//! order as everywhere else (black box, per-step field dumps of closed
//! grids with 1 m cells). [`OpenemsSeam`] recomputes the seam that way:
//!
//! - `H_α` (post-H): `i' = ii·i − iv·(((a − b) + d) − c)` with `i` the
//!   whole dual edge's current (the sum of the two halves) and `ii`, `iv`
//!   of the whole dual edge; each copy gets `i'/2` (exact);
//! - `E_ρ`, `E_z` (post-E, before the other operators):
//!   `u' = vv·u + vi·(((a − b) − c) + d)` with `vv`, `vi` of the whole dual
//!   face and the neighbours across the seam (`H_z`, `H_ρ` of the last α
//!   cell, `H_α` the whole dual edge).
//!
//! Edges with a modified coefficient (lumped elements, PEC, absorbing
//! layers) keep the periodic merge.

use super::coeffs::Coefficients;
use super::edge_material::EdgeMaterials;
use super::fields::Fields;
use super::grid::Grid;
use super::kernel::{flush, oe_e, oe_h};
use super::operator::{Hooks, Operator, StepContext};
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};

/// One seam E edge: component, the low (`α_0`) node; the other copy is at
/// `α_N`. `vv`, `vi` of the whole dual face, and the copies' `cb` at setup
/// (the edge is skipped if they change).
#[derive(Debug, Clone)]
struct SeamE {
    axis: Axis,
    node: [usize; 3],
    vv: f32,
    vi: f32,
    cb: (f32, f32),
    old: f32,
}

/// One seam `H_α` dual edge (low node), `ii`, `iv` of the whole dual edge,
/// the copies' `db` at setup.
#[derive(Debug, Clone)]
struct SeamH {
    node: [usize; 3],
    ii: f32,
    iv: f32,
    db: (f32, f32),
    old: f32,
}

/// openEMS's seam update of a closed cylindrical grid; see the
/// [module docs](self).
#[derive(Debug, Clone)]
pub(crate) struct OpenemsSeam {
    e: Vec<SeamE>,
    h: Vec<SeamH>,
    /// The axis z edges (node `[0, 0, k]`, all α copies), if the grid has
    /// the axis.
    axis: Vec<SeamE>,
    last: usize,
    nz: usize,
    /// Per `H_α` seam dual edge (`i·nz + k`): the half written to both
    /// copies and the whole edge's current it came from (a half in the
    /// subnormal range would lose the whole value).
    whole: Vec<Option<(f32, f32)>>,
}

/// `(vv, vi)` (or `(ii, iv)`) from the circuit quantities, rounded as
/// openEMS does (`CoeffRounding::OpenEms`).
fn coeffs(c: f64, g: f64, dt: f64) -> (f32, f32) {
    coeffs_f64(f64::from(c as f32), f64::from(g as f32), dt)
}

/// `(vv, vi)` from `f64` circuit quantities (openEMS's axis z edge: its
/// `C` is not rounded to `f32` first; black box, `vi` 96.84546 where the
/// rounded `C` gives 96.84545).
fn coeffs_f64(c: f64, g: f64, dt: f64) -> (f32, f32) {
    let den = c + 0.5 * g * dt;
    (((c - 0.5 * g * dt) / den) as f32, (dt / den) as f32)
}

/// The mean of the copies' values `lo`, `hi` with weights `wl`, `wh`
/// (exactly `lo` if both are equal).
fn mean(lo: f64, hi: f64, wl: f64, wh: f64) -> f64 {
    if lo == hi {
        lo
    } else {
        (lo * wl + hi * wh) / (wl + wh)
    }
}

impl OpenemsSeam {
    /// The seam of `grid` (closed, without multigrid) with the edge
    /// materials `mats`, time step `dt` and the engine's coefficients `co`;
    /// `skip(axis, node, h)` leaves an edge (E, or `H_α` with `h`) to the
    /// periodic merge (e.g. inside an absorbing layer).
    pub(crate) fn new(
        grid: &Grid,
        mats: &EdgeMaterials,
        co: &Coefficients,
        dt: f64,
        skip: impl Fn(Axis, [usize; 3], bool) -> bool,
    ) -> Self {
        let n = grid.nodes();
        let last = n[1] - 1;
        let r = grid.lines(Axis::X);
        let da0 = grid.delta(Axis::Y, 0);
        let da1 = grid.delta(Axis::Y, last - 1);
        // The whole dual α width of the seam.
        let dstar = 0.5 * (da0 + da1);
        let mut e = Vec::new();
        let mut h = Vec::new();
        for axis in [Axis::X, Axis::Z] {
            let a = axis.index();
            let (eps, kap) = (mats.eps_r(axis), mats.kappa(axis));
            let cb = co.cb(axis);
            for i in 0..n[0] {
                for k in 0..n[2] {
                    let lo = [i, 0, k];
                    let hi = [i, last, k];
                    if lo[a] + 1 >= n[a] || (axis == Axis::Z && i == 0 && grid.has_axis()) {
                        continue;
                    }
                    let (il, ih) = (grid.index(lo), grid.index(hi));
                    if cb[il] == 0.0 || cb[ih] == 0.0 || skip(axis, lo, false) {
                        continue;
                    }
                    let (area, len) = if axis == Axis::X {
                        let rho = grid.dual_line(Axis::X, i);
                        (
                            rho * dstar * grid.dual_delta(Axis::Z, k),
                            grid.delta(Axis::X, i),
                        )
                    } else {
                        let lo_r = if i > 0 { 0.5 * (r[i - 1] + r[i]) } else { r[i] };
                        let hi_r = if i + 1 < n[0] {
                            grid.dual_line(Axis::X, i)
                        } else {
                            r[i]
                        };
                        (
                            dstar * 0.5 * (hi_r * hi_r - lo_r * lo_r),
                            grid.delta(Axis::Z, k),
                        )
                    };
                    let geo = area / len;
                    let ep = mean(eps[il], eps[ih], da0, da1);
                    let kp = mean(kap[il], kap[ih], da0, da1);
                    let (vv, vi) = coeffs(EPS0 * ep * geo, kp * geo, dt);
                    e.push(SeamE {
                        axis,
                        node: lo,
                        vv,
                        vi,
                        cb: (cb[il], cb[ih]),
                        old: 0.0,
                    });
                }
            }
        }
        let (mu, sig) = (mats.mu_r(Axis::Y), mats.sigma_m(Axis::Y));
        let db = co.db(Axis::Y);
        for i in 0..n[0].saturating_sub(1) {
            for k in 0..n[2].saturating_sub(1) {
                let lo = [i, 0, k];
                let hi = [i, last, k];
                let (il, ih) = (grid.index(lo), grid.index(hi));
                if db[il] == 0.0 || db[ih] == 0.0 || skip(Axis::Y, lo, true) {
                    continue;
                }
                let area = grid.delta(Axis::Z, k) * grid.delta(Axis::X, i);
                let len = grid.dual_line(Axis::X, i) * dstar;
                let geo = area / len;
                let m = if mu[il] == mu[ih] {
                    mu[il]
                } else {
                    (da0 + da1) / (da0 / mu[il] + da1 / mu[ih])
                };
                let s = mean(sig[il], sig[ih], da0, da1);
                let (ii, iv) = coeffs(MU0 * m * geo, s * geo, dt);
                h.push(SeamH {
                    node: lo,
                    ii,
                    iv,
                    db: (db[il], db[ih]),
                    old: 0.0,
                });
            }
        }
        let mut axis = Vec::new();
        if grid.has_axis() {
            let (eps, kap) = (mats.eps_r(Axis::Z), mats.kappa(Axis::Z));
            let cb = co.cb(Axis::Z);
            for k in 0..n[2].saturating_sub(1) {
                let copies: Vec<usize> = (0..n[1]).map(|j| grid.index([0, j, k])).collect();
                if copies.iter().any(|&c| cb[c] == 0.0)
                    || (0..n[1]).any(|j| skip(Axis::Z, [0, j, k], false))
                {
                    continue;
                }
                // The disc ρ ≤ ρ_½: the copies' sectors.
                let parts: Vec<f64> = (0..n[1])
                    .map(|j| grid.dual_face_area(Axis::Z, [0, j, k]))
                    .collect();
                let area: f64 = parts.iter().sum();
                let avg = |v: &[f64]| -> f64 {
                    if copies.iter().all(|&c| v[c] == v[copies[0]]) {
                        v[copies[0]]
                    } else {
                        copies
                            .iter()
                            .zip(&parts)
                            .map(|(&c, &w)| v[c] * w)
                            .sum::<f64>()
                            / area
                    }
                };
                let geo = area / grid.delta(Axis::Z, k);
                let (vv, vi) = coeffs_f64(EPS0 * avg(eps) * geo, avg(kap) * geo, dt);
                axis.push(SeamE {
                    axis: Axis::Z,
                    node: [0, 0, k],
                    vv,
                    vi,
                    cb: (cb[copies[0]], cb[copies[n[1] - 1]]),
                    old: 0.0,
                });
            }
        }
        Self {
            e,
            h,
            axis,
            last,
            nz: n[2],
            whole: vec![None; n[0] * n[2]],
        }
    }

    fn hi(&self, p: [usize; 3]) -> [usize; 3] {
        [p[0], self.last, p[2]]
    }

    /// The whole current of the seam `H_α` dual edge at `(i, k)`: the
    /// value this operator wrote if the copies still hold its half, else
    /// the sum of the copies.
    fn whole(&self, f: &Fields<'_>, i: usize, k: usize) -> f32 {
        let (lo, hi) = (
            f.i.get(Axis::Y, [i, 0, k]),
            f.i.get(Axis::Y, [i, self.last, k]),
        );
        match self.whole[i * self.nz + k] {
            Some((half, v)) if lo == half && hi == half => v,
            _ => flush(lo + hi),
        }
    }
}

impl Operator for OpenemsSeam {
    fn hooks(&self) -> Hooks {
        Hooks::PRE_H | Hooks::POST_H | Hooks::POST_E
    }

    fn pre_h(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        for n in 0..self.h.len() {
            let [i, _, k] = self.h[n].node;
            self.h[n].old = self.whole(f, i, k);
        }
    }

    fn post_h(&mut self, f: &mut Fields<'_>, ctx: &StepContext<'_>) {
        let db = ctx.coefficients.db(Axis::Y);
        let last = self.last;
        for s in &self.h {
            let [i, _, k] = s.node;
            self.whole[i * self.nz + k] = None;
            let hi = [i, last, k];
            if (db[ctx.grid.index(s.node)], db[ctx.grid.index(hi)]) != s.db {
                continue;
            }
            let u = &f.u;
            let a = u.get(Axis::X, [i, 0, k + 1]);
            let b = u.get(Axis::X, s.node);
            let c = u.get(Axis::Z, [i + 1, 0, k]);
            let d = u.get(Axis::Z, s.node);
            let v = oe_h(s.ii, s.old, s.iv, a, b, c, d);
            let half = 0.5 * v;
            f.i.set(Axis::Y, s.node, half);
            f.i.set(Axis::Y, hi, half);
            self.whole[i * self.nz + k] = Some((half, v));
        }
        for s in self.e.iter_mut().chain(&mut self.axis) {
            s.old = f.u.get(s.axis, s.node);
        }
    }

    fn post_e(&mut self, f: &mut Fields<'_>, ctx: &StepContext<'_>) {
        let n = ctx.grid.nodes();
        let last = self.last;
        for s in &self.e {
            let cb = ctx.coefficients.cb(s.axis);
            let hi = self.hi(s.node);
            if (cb[ctx.grid.index(s.node)], cb[ctx.grid.index(hi)]) != s.cb {
                continue;
            }
            let [i, _, k] = s.node;
            let cur = &f.i;
            // The whole seam H_α dual edge, 0 outside.
            let ha = |i: usize, k: usize| -> f32 {
                if i + 1 >= n[0] || k + 1 >= n[2] {
                    0.0
                } else {
                    self.whole(f, i, k)
                }
            };
            let (a, b, c, d) = if s.axis == Axis::X {
                (
                    cur.get(Axis::Z, s.node),
                    cur.get(Axis::Z, [i, last - 1, k]),
                    ha(i, k),
                    if k > 0 { ha(i, k - 1) } else { 0.0 },
                )
            } else {
                (
                    ha(i, k),
                    if i > 0 { ha(i - 1, k) } else { 0.0 },
                    cur.get(Axis::X, s.node),
                    cur.get(Axis::X, [i, last - 1, k]),
                )
            };
            let v = oe_e(s.vv, s.old, s.vi, a, b, c, d);
            f.u.set(s.axis, s.node, v);
            f.u.set(s.axis, hi, v);
        }
        // The axis: openEMS adds the ring of H_α at ρ_½ term by term over
        // all N + 1 α lines, the seam's at both ends.
        for s in &self.axis {
            let cb = ctx.coefficients.cb(Axis::Z);
            let k = s.node[2];
            let (lo, hi) = ([0, 0, k], [0, last, k]);
            if (cb[ctx.grid.index(lo)], cb[ctx.grid.index(hi)]) != s.cb {
                continue;
            }
            let cur = &f.i;
            let seam = self.whole(f, 0, k);
            let mut v = flush(s.vv * s.old);
            v = flush(v + flush(s.vi * seam));
            for j in 1..last {
                v = flush(v + flush(s.vi * cur.get(Axis::Y, [0, j, k])));
            }
            v = flush(v + flush(s.vi * seam));
            for j in 0..=last {
                f.u.set(Axis::Z, [0, j, k], v);
            }
        }
    }
}
