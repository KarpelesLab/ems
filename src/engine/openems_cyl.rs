//! openEMS's arithmetic on the seam of a closed cylindrical grid (models
//! read from openEMS files, design §5.16.11).
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
//!
//! **Dispersive poles on the seam** (design §5.16.12). openEMS keeps the
//! seam's state on the `α = 2π` line and copies it to `α = 0` right after
//! its main update, before the poles act: the `α = 0` copy (the one dumps,
//! probes and the H cells `α_0 … α_1` read) holds the uncorrected `u*`,
//! the `α = 2π` copy (read by the last H cells and by its own update) the
//! corrected `u* − ΣV`, the poles in openEMS's `f32` arithmetic
//! ([`OpenEmsPole`]) with the whole dual face's `vi` and `C`. The seam
//! `H_α` next to such edges is then updated twice, once from each copy's
//! E (each with its own whole current); the `α = 2π` E update reads the
//! `α = 2π` current. Black box: a closed grid filled with a lossy Drude
//! medium, every seam `E_ρ`, `E_z` at every step reproduced by an `f32`
//! model of this. A Debye pole alone on a seam edge is run the same way,
//! with openEMS's Debye update
//! ([`Dispersive::with_openems_debye`](super::dispersive::Dispersive::with_openems_debye))
//! and the whole dual face's `vi` and `ε0·A*/l` (black box: a closed grid
//! with a Debye ring reproduced bit for bit).

use super::boundary::upml::{self, PmlProfile};
use super::coeffs::Coefficients;
use super::dispersive::{OpenEmsPole, openems_debye_coefficients};
use super::edge_material::EdgeMaterials;
use super::fields::Fields;
use super::grid::Grid;
use super::kernel::{flush, oe_e, oe_h};
use super::operator::{Hooks, Operator, StepContext};
use crate::axis::Axis;
use crate::constants::{OPENEMS_EPS0 as EPS0, OPENEMS_MU0 as MU0};

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
    /// Inside an openEMS PML layer: the layer update (flux increment
    /// coefficient and update coefficients) and the edge's flux.
    layer: Option<(f32, [f32; 4])>,
    flux: f32,
    /// openEMS's poles of the edge: `[A, B, K]` and the state `[V, W]`
    /// (module docs); with poles `old` is the `α = 2π` copy's value.
    poles: Vec<([f32; 3], [f32; 2])>,
    /// openEMS's Debye pole of the edge instead: `[F, b, 1 − g]` and the
    /// polarisation `Q` (module docs).
    debye: Option<([f32; 3], f32)>,
}

impl SeamE {
    /// Whether the edge carries openEMS poles (run on the `α = 2π` copy).
    fn has_poles(&self) -> bool {
        !self.poles.is_empty() || self.debye.is_some()
    }
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
    /// As [`SeamE::layer`].
    layer: Option<(f32, [f32; 4])>,
    flux: f32,
    /// Next to a seam E edge with poles: the two copies are updated apart
    /// (`old_hi` the `α = 2π` copy's whole current).
    split: bool,
    old_hi: f32,
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
    /// The same for the `α = 2π` copy of a split edge ([`SeamH::split`]).
    whole_hi: Vec<Option<(f32, f32)>>,
    /// Whether the seam `H_α` at `(i, k)` is split.
    split: Vec<bool>,
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
    /// periodic merge (e.g. inside an absorbing layer); `poles`, `debye`
    /// and `sheets` give a seam E edge (low node) its openEMS poles: its
    /// Lorentz and Drude poles, its single Debye pole `(Δε, τ)`, or the
    /// conducting-sheet branches `(L, R)` of the whole edge (design
    /// §5.16.13).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        grid: &Grid,
        mats: &EdgeMaterials,
        co: &Coefficients,
        dt: f64,
        pml: Option<&PmlProfile>,
        skip: impl Fn(Axis, [usize; 3], bool) -> bool,
        poles: impl Fn(Axis, [usize; 3]) -> Vec<OpenEmsPole>,
        debye: impl Fn(Axis, [usize; 3]) -> Option<(f64, f64)>,
        sheets: impl Fn(Axis, [usize; 3]) -> Option<[(f64, f64); 2]>,
    ) -> Self {
        let layer = |e_side: bool, axis: Axis, node: [usize; 3], geo: f64, rel: f64| {
            pml.and_then(|p| upml::layer_update(p, dt, e_side, axis, node, geo, rel))
        };
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
                    let layer = layer(true, axis, hi, geo, ep);
                    // openEMS's layer replaces the update: no poles there.
                    let circuit = f64::from((EPS0 * ep * geo) as f32);
                    let poles = if layer.is_none() {
                        match sheets(axis, lo) {
                            // A conducting-sheet edge: its two branches as
                            // poles without restoring term (`K = 0`), the
                            // coefficients of `OpenEmsSheetBranches` with
                            // the whole dual face's `vi`.
                            Some(branches) => branches
                                .iter()
                                .map(|&(l, r)| {
                                    let d = l / dt + 0.5 * r;
                                    let a = ((l / dt - 0.5 * r) / d) as f32;
                                    ([a, (f64::from(vi) / d) as f32, 0.0], [0.0; 2])
                                })
                                .collect(),
                            None => poles(axis, lo)
                                .into_iter()
                                .map(|p| {
                                    let p = OpenEmsPole { circuit, ..p };
                                    (p.coefficients(f64::from(vi), dt), [0.0; 2])
                                })
                                .collect(),
                        }
                    } else {
                        Vec::new()
                    };
                    let debye = debye(axis, lo)
                        .filter(|_| layer.is_none())
                        .map(|(de, tau)| {
                            (
                                openems_debye_coefficients(f64::from(vi), EPS0 * geo, de, tau, dt),
                                0.0,
                            )
                        });
                    e.push(SeamE {
                        axis,
                        node: lo,
                        vv,
                        vi,
                        cb: (cb[il], cb[ih]),
                        old: 0.0,
                        layer,
                        flux: 0.0,
                        poles,
                        debye,
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
                    layer: layer(false, Axis::Y, hi, geo, m),
                    flux: 0.0,
                    split: false,
                    old_hi: 0.0,
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
                    || layer(true, Axis::Z, [0, 0, k], 1.0, 1.0).is_some()
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
                // The pole takes the whole disc's `vi` but the first α
                // sector's `A*/l` (black box: `β` one N-th of the whole
                // disc's).
                let sector = 2.0 * parts[0] / grid.delta(Axis::Z, k);
                let debye = debye(Axis::Z, [0, 0, k]).map(|(de, tau)| {
                    (
                        openems_debye_coefficients(f64::from(vi), EPS0 * sector, de, tau, dt),
                        0.0,
                    )
                });
                axis.push(SeamE {
                    axis: Axis::Z,
                    node: [0, 0, k],
                    vv,
                    vi,
                    cb: (cb[copies[0]], cb[copies[n[1] - 1]]),
                    old: 0.0,
                    layer: None,
                    flux: 0.0,
                    poles: Vec::new(),
                    debye,
                });
            }
        }
        // The H_α dual edges whose E neighbours on the seam have poles.
        let mut split = vec![false; n[0] * n[2]];
        for s in e.iter().filter(|s| s.has_poles()) {
            let [i, _, k] = s.node;
            let mut mark = |i: usize, k: usize| {
                if i + 1 < n[0] && k + 1 < n[2] {
                    split[i * n[2] + k] = true;
                }
            };
            if s.axis == Axis::X {
                mark(i, k);
                if k > 0 {
                    mark(i, k - 1);
                }
            } else {
                mark(i, k);
                if i > 0 {
                    mark(i - 1, k);
                }
            }
        }
        for s in &mut h {
            s.split = s.layer.is_none() && split[s.node[0] * n[2] + s.node[2]];
        }
        Self {
            e,
            h,
            axis,
            last,
            nz: n[2],
            whole: vec![None; n[0] * n[2]],
            whole_hi: vec![None; n[0] * n[2]],
            split,
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

    /// The whole current of one copy (`hi`: the `α = 2π` one) of a split
    /// seam `H_α` dual edge ([`SeamH::split`]); [`whole`](Self::whole)
    /// for the others.
    fn whole_copy(&self, f: &Fields<'_>, i: usize, k: usize, hi: bool) -> f32 {
        if !self.split[i * self.nz + k] {
            return self.whole(f, i, k);
        }
        let x = f.i.get(Axis::Y, [i, if hi { self.last } else { 0 }, k]);
        let w = if hi { &self.whole_hi } else { &self.whole };
        match w[i * self.nz + k] {
            Some((half, v)) if x == half => v,
            _ => flush(x + x),
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
            self.h[n].old = self.whole_copy(f, i, k, false);
            self.h[n].old_hi = self.whole_copy(f, i, k, true);
        }
    }

    fn post_h(&mut self, f: &mut Fields<'_>, ctx: &StepContext<'_>) {
        let db = ctx.coefficients.db(Axis::Y);
        let last = self.last;
        for s in &mut self.h {
            let [i, _, k] = s.node;
            self.whole[i * self.nz + k] = None;
            self.whole_hi[i * self.nz + k] = None;
            let hi = [i, last, k];
            if s.layer.is_none() && (db[ctx.grid.index(s.node)], db[ctx.grid.index(hi)]) != s.db {
                continue;
            }
            let u = &f.u;
            let a = u.get(Axis::X, [i, 0, k + 1]);
            let b = u.get(Axis::X, s.node);
            let c = u.get(Axis::Z, [i + 1, 0, k]);
            let d = u.get(Axis::Z, s.node);
            let v = match s.layer {
                None => oe_h(s.ii, s.old, s.iv, a, b, c, d),
                Some((cbf, co)) => {
                    // The kernel's flux increment with `da = 0`, then the
                    // layer update (engine::boundary::upml).
                    let t = flush(0.0 - flush(cbf * flush(flush(flush(a - b) + d) - c)));
                    upml::layer_step(co, &mut s.flux, s.old, t)
                }
            };
            let half = 0.5 * v;
            f.i.set(Axis::Y, s.node, half);
            self.whole[i * self.nz + k] = Some((half, v));
            if s.split {
                // The α = 2π copy from its own E (module docs).
                let a = u.get(Axis::X, [i, last, k + 1]);
                let b = u.get(Axis::X, hi);
                let c = u.get(Axis::Z, [i + 1, last, k]);
                let d = u.get(Axis::Z, hi);
                let v = oe_h(s.ii, s.old_hi, s.iv, a, b, c, d);
                let half = 0.5 * v;
                f.i.set(Axis::Y, hi, half);
                self.whole_hi[i * self.nz + k] = Some((half, v));
            } else {
                f.i.set(Axis::Y, hi, half);
            }
        }
        for s in self.e.iter_mut() {
            // With poles the α = 2π copy holds the corrected state.
            let j = if s.has_poles() { last } else { 0 };
            s.old = f.u.get(s.axis, [s.node[0], j, s.node[2]]);
        }
        for s in &mut self.axis {
            s.old = f.u.get(s.axis, s.node);
        }
    }

    fn post_e(&mut self, f: &mut Fields<'_>, ctx: &StepContext<'_>) {
        let n = ctx.grid.nodes();
        let last = self.last;
        let mut fluxes = Vec::new();
        let mut states = Vec::new();
        let mut debye_states = Vec::new();
        for (q, s) in self.e.iter().enumerate() {
            let cb = ctx.coefficients.cb(s.axis);
            let hi = self.hi(s.node);
            if s.layer.is_none() && (cb[ctx.grid.index(s.node)], cb[ctx.grid.index(hi)]) != s.cb {
                continue;
            }
            let [i, _, k] = s.node;
            let cur = &f.i;
            // The whole seam H_α dual edge (its α = 2π copy), 0 outside.
            let ha = |i: usize, k: usize| -> f32 {
                if i + 1 >= n[0] || k + 1 >= n[2] {
                    0.0
                } else {
                    self.whole_copy(f, i, k, true)
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
            let v = match s.layer {
                None => oe_e(s.vv, s.old, s.vi, a, b, c, d),
                Some((cbf, co)) => {
                    // The kernel's flux increment with `ca = 0`, then the
                    // layer update (engine::boundary::upml).
                    let t = flush(cbf * flush(flush(flush(a - b) - c) + d));
                    let mut flux = s.flux;
                    let v = upml::layer_step(co, &mut flux, s.old, t);
                    fluxes.push((q, flux));
                    v
                }
            };
            f.u.set(s.axis, s.node, v);
            if let Some(([fc, bc, om], pol)) = s.debye {
                // openEMS's Debye update on the α = 2π copy only.
                let p = flush(flush(om * pol) + flush(bc * s.old));
                let un = flush(fc * flush(v - flush(p - pol)));
                f.u.set(s.axis, hi, flush(v - flush(v - un)));
                debye_states.push((q, flush(p + flush(bc * un))));
            } else if s.poles.is_empty() {
                f.u.set(s.axis, hi, v);
            } else {
                // openEMS's poles on the α = 2π copy only, driven by its
                // previous value (module docs, `OpenEmsPole`).
                let mut st: Vec<[f32; 2]> = Vec::with_capacity(s.poles.len());
                let mut c = v;
                for &([a, b, kk], [pv, pw]) in &s.poles {
                    let pv = flush(flush(a * pv) + flush(b * flush(s.old - pw)));
                    let pw = flush(pw + flush(kk * pv));
                    c = flush(c - pv);
                    st.push([pv, pw]);
                }
                f.u.set(s.axis, hi, c);
                states.push((q, st));
            }
        }
        for (q, flux) in fluxes {
            self.e[q].flux = flux;
        }
        for (q, st) in states {
            for (p, x) in self.e[q].poles.iter_mut().zip(st) {
                p.1 = x;
            }
        }
        for (q, pol) in debye_states {
            if let Some(d) = &mut self.e[q].debye {
                d.1 = pol;
            }
        }
        // The axis: openEMS adds the ring of H_α at ρ_½ term by term over
        // all N + 1 α lines, the seam's at both ends.
        let mut axis_states = Vec::new();
        for (q, s) in self.axis.iter().enumerate() {
            let cb = ctx.coefficients.cb(Axis::Z);
            let k = s.node[2];
            let (lo, hi) = ([0, 0, k], [0, last, k]);
            if (cb[ctx.grid.index(lo)], cb[ctx.grid.index(hi)]) != s.cb {
                continue;
            }
            let cur = &f.i;
            let (seam, seam_hi) = (
                self.whole_copy(f, 0, k, false),
                self.whole_copy(f, 0, k, true),
            );
            let mut v = flush(s.vv * s.old);
            v = flush(v + flush(s.vi * seam));
            for j in 1..last {
                v = flush(v + flush(s.vi * cur.get(Axis::Y, [0, j, k])));
            }
            v = flush(v + flush(s.vi * seam_hi));
            if let Some(([fc, bc, om], pol)) = s.debye {
                // openEMS's Debye update of the (single) axis edge.
                let p = flush(flush(om * pol) + flush(bc * s.old));
                let un = flush(fc * flush(v - flush(p - pol)));
                axis_states.push((q, flush(p + flush(bc * un))));
                v = flush(v - flush(v - un));
            }
            for j in 0..=last {
                f.u.set(Axis::Z, [0, j, k], v);
            }
        }
        for (q, pol) in axis_states {
            if let Some(d) = &mut self.axis[q].debye {
                d.1 = pol;
            }
        }
    }
}

/// openEMS's azimuthal multigrid on a closed grid with the axis and one
/// zone boundary (models read from openEMS files, design §5.16.11).
///
/// openEMS runs a coarse operator (every second α line) on the rings
/// inside the zone boundary `ρ_I` next to the fine operator, which
/// updates the edges on and outside `ρ_I` and reads the inner fields from
/// the fine arrays, where openEMS writes the coarse values interpolated
/// onto the fine lines. Black box (per-step dumps of a 1 m-cell model,
/// reproduced bit for bit over 38 steps by an `f32` model of):
///
/// - the coarse H update (cells and lines inside `ρ_I`) from the coarse
///   E and, on `ρ_I`, the fine E: `E_z` of the even lines, `E_α` of a
///   coarse arc the sum of its two fine edges;
/// - the coarse E update inside `ρ_I`, the axis by openEMS's ring sum of
///   the coarse `H_α` (§5.15);
/// - coarse coefficients of the coarse cell geometry (`C`, `L` rounded to
///   `f32` first);
/// - fine values inside `ρ_I`: quantities on the α lines (`E_ρ`, `E_z`,
///   `H_α`) the coarse value on even lines and the mean of the two
///   neighbours on odd ones, quantities on α cells (`E_α`, `H_ρ`, `H_z`)
///   the linear interpolation between the centres of the coarse cell and
///   its neighbour towards the fine cell (3/4 and 1/4), arc voltages and
///   `H_α` currents scaled to the fine lengths (½); the first fine `E_α`
///   cell extrapolates from the last coarse cell's unwrapped centre
///   (weights 13/12 and −1/12), unlike `H_ρ` and `H_z`.
///
/// The engine's grid is the plain fine grid; the operator overwrites
/// every field inside `ρ_I` after the kernels (and after
/// [`OpenemsSeam`]).
#[derive(Debug, Clone)]
pub(crate) struct OpenemsMultigrid {
    /// Zone boundary line.
    ring: usize,
    /// Coarse α cells.
    nc: usize,
    nz: usize,
    /// Coarse E and H, `[c][(i·nc + jc)·nz + k]`, `i ≤ ring`.
    e: [Vec<f32>; 3],
    h: [Vec<f32>; 3],
    /// Coarse coefficients (`ca`/`cb`, `da`/`db`; `cb = 0`: no edge).
    ce: [Vec<(f32, f32)>; 3],
    ch: [Vec<(f32, f32)>; 3],
    /// Axis z edges: `(vv, vi)` per k (`vi = 0`: PEC).
    axis: Vec<(f32, f32)>,
    /// Weights `(own, neighbour)` of the fine α-cell interpolation of a
    /// coarse cell's low and high fine cell, and of the first `E_α` cell.
    w_lo: (f32, f32),
    w_hi: (f32, f32),
    w_first_ea: (f32, f32),
}

impl OpenemsMultigrid {
    /// The emulation for `grid` (closed, with the axis, uniform α, `ring`
    /// the zone boundary line) on vacuum inside the zone, with the fine
    /// engine coefficients `co` (for the faces and PEC edges) and time
    /// step `dt`.
    pub(crate) fn new(grid: &Grid, co: &Coefficients, dt: f64, ring: usize) -> Self {
        let n = grid.nodes();
        let (nr, nz) = (n[0], n[2]);
        let nc = (n[1] - 1) / 2;
        let r = grid.lines(Axis::X);
        let z = grid.lines(Axis::Z);
        let al: Vec<f64> = grid.lines(Axis::Y).iter().step_by(2).copied().collect();
        let da = |j: usize| al[j + 1] - al[j];
        let dda = |j: usize| 0.5 * (da(j) + da(if j > 0 { j - 1 } else { nc - 1 }));
        let dl = |i: usize| if i + 1 < nr { r[i + 1] - r[i] } else { 0.0 };
        let ddr = |i: usize| 0.5 * (if i > 0 { r[i] - r[i - 1] } else { 0.0 } + dl(i));
        let dz = |k: usize| if k + 1 < nz { z[k + 1] - z[k] } else { 0.0 };
        let ddz = |k: usize| 0.5 * (if k > 0 { z[k] - z[k - 1] } else { 0.0 } + dz(k));
        let rh = |i: usize| {
            if i + 1 < nr {
                0.5 * (r[i] + r[i + 1])
            } else {
                r[i]
            }
        };
        let len = (ring + 1) * nc * nz;
        let idx = |i: usize, jc: usize, k: usize| (i * nc + jc) * nz + k;
        let mut ce: [Vec<(f32, f32)>; 3] = core::array::from_fn(|_| vec![(0.0, 0.0); len]);
        let mut ch: [Vec<(f32, f32)>; 3] = core::array::from_fn(|_| vec![(0.0, 0.0); len]);
        for c in 0..3 {
            let axis = Axis::ALL[c];
            let (cb, db) = (co.cb(axis), co.db(axis));
            for i in 0..ring {
                for jc in 0..nc {
                    for k in 0..nz {
                        let fine = grid.index([i, 2 * jc, k]);
                        if cb[fine] != 0.0 && !(c == 2 && i == 0) {
                            let (area, l) = match c {
                                0 => (rh(i) * dda(jc) * ddz(k), dl(i)),
                                1 => (ddr(i) * ddz(k), r[i] * da(jc)),
                                _ => {
                                    let lo = if i > 0 { 0.5 * (r[i - 1] + r[i]) } else { r[i] };
                                    (dda(jc) * 0.5 * (rh(i) * rh(i) - lo * lo), dz(k))
                                }
                            };
                            if area > 0.0 && l > 0.0 {
                                ce[c][idx(i, jc, k)] = coeffs(EPS0 * area / l, 0.0, dt);
                            }
                        }
                        if db[fine] != 0.0 {
                            let (area, l) = match c {
                                0 => (r[i] * da(jc) * dz(k), ddr(i)),
                                1 => (dl(i) * dz(k), rh(i) * dda(jc)),
                                _ => (da(jc) * 0.5 * (r[i + 1] * r[i + 1] - r[i] * r[i]), ddz(k)),
                            };
                            if area > 0.0 && l > 0.0 {
                                ch[c][idx(i, jc, k)] = coeffs(MU0 * area / l, 0.0, dt);
                            }
                        }
                    }
                }
            }
        }
        let cbz = co.cb(Axis::Z);
        let axis = (0..nz)
            .map(|k| {
                if k + 1 >= nz || cbz[grid.index([0, 0, k])] == 0.0 {
                    return (0.0, 0.0);
                }
                let area: f64 = (0..nc).map(|jc| dda(jc) * 0.5 * rh(0) * rh(0)).sum();
                coeffs_f64(EPS0 * area / dz(k), 0.0, dt)
            })
            .collect();
        // Interpolation weights between coarse cell centres (uniform α).
        let fl = grid.lines(Axis::Y);
        let cen = |jc: usize| 0.5 * (al[jc] + al[jc + 1]);
        let fcen = |j: usize| 0.5 * (fl[j] + fl[j + 1]);
        let w = |x: f64, own: f64, nb: f64| {
            let t = (x - nb) / (own - nb);
            (t as f32, (1.0 - t) as f32)
        };
        let step = cen(1) - cen(0);
        let w_lo = w(fcen(2), cen(1), cen(0));
        let w_hi = w(fcen(1), cen(0), cen(0) + step);
        let w_first_ea = w(fcen(0), cen(0), cen(nc - 1));
        Self {
            ring,
            nc,
            nz,
            e: core::array::from_fn(|_| vec![0.0; len]),
            h: core::array::from_fn(|_| vec![0.0; len]),
            ce,
            ch,
            axis,
            w_lo,
            w_hi,
            w_first_ea,
        }
    }

    fn idx(&self, i: usize, jc: usize, k: usize) -> usize {
        (i * self.nc + jc) * self.nz + k
    }

    /// Coarse value of component `c` at `p = (i, jc, k)` with periodic α
    /// and 0 outside `0 ≤ i ≤ ring`, `0 ≤ k < nz`.
    fn get(&self, v: &[Vec<f32>; 3], c: usize, p: [isize; 3]) -> f32 {
        let [i, jc, k] = p;
        if i < 0 || k < 0 || i as usize > self.ring || k as usize >= self.nz {
            return 0.0;
        }
        let jc = jc.rem_euclid(self.nc as isize) as usize;
        v[c][self.idx(i as usize, jc, k as usize)]
    }

    /// `own·a + nb·b`, flushed per operation.
    fn lerp((own, nb): (f32, f32), a: f32, b: f32) -> f32 {
        flush(flush(own * a) + flush(nb * b))
    }
}

impl Operator for OpenemsMultigrid {
    fn hooks(&self) -> Hooks {
        Hooks::POST_H | Hooks::POST_E
    }

    fn post_h(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        let (ring, nc, nz) = (self.ring, self.nc, self.nz);
        // The fine E on ρ_I onto the coarse grid's line I.
        for jc in 0..nc {
            for k in 0..nz {
                let q = self.idx(ring, jc, k);
                self.e[2][q] = f.u.get(Axis::Z, [ring, 2 * jc, k]);
                self.e[1][q] = flush(
                    f.u.get(Axis::Y, [ring, 2 * jc, k]) + f.u.get(Axis::Y, [ring, 2 * jc + 1, k]),
                );
            }
        }
        // The coarse H update inside the zone.
        let mut next = self.h.clone();
        for (c, out) in next.iter_mut().enumerate() {
            let (ca, cb) = ((c + 1) % 3, (c + 2) % 3);
            for i in 0..ring {
                for jc in 0..nc {
                    for k in 0..nz {
                        let q = self.idx(i, jc, k);
                        let (ii, iv) = self.ch[c][q];
                        if iv == 0.0 {
                            continue;
                        }
                        let p = [i as isize, jc as isize, k as isize];
                        let (mut pa, mut pb) = (p, p);
                        pa[ca] += 1;
                        pb[cb] += 1;
                        let g = |c: usize, p: [isize; 3]| self.get(&self.e, c, p);
                        out[q] = oe_h(
                            ii,
                            self.h[c][q],
                            iv,
                            g(cb, pa),
                            g(cb, p),
                            g(ca, pb),
                            g(ca, p),
                        );
                    }
                }
            }
        }
        self.h = next;
        // Into the fine arrays.
        let last = 2 * nc;
        for i in 0..ring {
            for jc in 0..nc {
                let (nx, pv) = ((jc + 1) % nc, (jc + nc - 1) % nc);
                let (j0, j1) = (2 * jc, 2 * jc + 1);
                for k in 0..nz {
                    let h = |c: usize, j: usize| self.h[c][self.idx(i, j, k)];
                    for c in [0, 2] {
                        let axis = Axis::ALL[c];
                        f.i.set(axis, [i, j0, k], Self::lerp(self.w_lo, h(c, jc), h(c, pv)));
                        f.i.set(axis, [i, j1, k], Self::lerp(self.w_hi, h(c, jc), h(c, nx)));
                    }
                    let even = flush(0.5 * h(1, jc));
                    let odd = flush(0.25 * flush(h(1, jc) + h(1, nx)));
                    if jc == 0 {
                        // The seam's two copies hold half the fine edge each.
                        let q = 0.5 * even;
                        f.i.set(Axis::Y, [i, 0, k], q);
                        f.i.set(Axis::Y, [i, last, k], q);
                    } else {
                        f.i.set(Axis::Y, [i, j0, k], even);
                    }
                    f.i.set(Axis::Y, [i, j1, k], odd);
                }
            }
        }
    }

    fn post_e(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        let (ring, nc, nz) = (self.ring, self.nc, self.nz);
        let mut next = self.e.clone();
        for (c, out) in next.iter_mut().enumerate() {
            let (ca, cb) = ((c + 1) % 3, (c + 2) % 3);
            for i in 0..ring {
                for jc in 0..nc {
                    for k in 0..nz {
                        let q = self.idx(i, jc, k);
                        let (vv, vi) = self.ce[c][q];
                        if vi == 0.0 {
                            continue;
                        }
                        let p = [i as isize, jc as isize, k as isize];
                        let (mut pa, mut pb) = (p, p);
                        pa[ca] -= 1;
                        pb[cb] -= 1;
                        let g = |c: usize, p: [isize; 3]| self.get(&self.h, c, p);
                        out[q] = oe_e(
                            vv,
                            self.e[c][q],
                            vi,
                            g(cb, p),
                            g(cb, pa),
                            g(ca, p),
                            g(ca, pb),
                        );
                    }
                }
            }
        }
        // The axis: openEMS's ring sum of the coarse H_α, seam twice.
        for k in 0..nz {
            let (vv, vi) = self.axis[k];
            if vi == 0.0 {
                continue;
            }
            let ha = |jc: usize| self.h[1][self.idx(0, jc, k)];
            let mut v = flush(vv * self.e[2][self.idx(0, 0, k)]);
            for jc in (0..nc).chain([0]) {
                v = flush(v + flush(vi * ha(jc)));
            }
            for jc in 0..nc {
                next[2][self.idx(0, jc, k)] = v;
            }
        }
        self.e = next;
        // Into the fine arrays (line I stays the fine operator's).
        let last = 2 * nc;
        for i in 0..ring {
            for jc in 0..nc {
                let (nx, pv) = ((jc + 1) % nc, (jc + nc - 1) % nc);
                let (j0, j1) = (2 * jc, 2 * jc + 1);
                for k in 0..nz {
                    let e = |c: usize, j: usize| self.e[c][self.idx(i, j, k)];
                    for c in [0, 2] {
                        let axis = Axis::ALL[c];
                        let even = e(c, jc);
                        f.u.set(axis, [i, j0, k], even);
                        f.u.set(axis, [i, j1, k], flush(0.5 * flush(e(c, jc) + e(c, nx))));
                        if jc == 0 {
                            f.u.set(axis, [i, last, k], even);
                        }
                    }
                    let first = if jc == 0 { self.w_first_ea } else { self.w_lo };
                    let lo = flush(0.5 * Self::lerp(first, e(1, jc), e(1, pv)));
                    let hi = flush(0.5 * Self::lerp(self.w_hi, e(1, jc), e(1, nx)));
                    f.u.set(Axis::Y, [i, j0, k], lo);
                    f.u.set(Axis::Y, [i, j1, k], hi);
                }
            }
        }
    }
}
