//! Oblique thin wires on cylindrical grids (design §5.14.1, §5.15).
//!
//! The wire stays a polyline of straight Cartesian segments; the grid is
//! the `(ρ, α, z)` lattice. Everything of the Cartesian model carries
//! over in parameter space:
//!
//! * **Coupling.** The lowest-order Whitney 1-forms of the `(ρ, α, z)`
//!   cells (constant along the edge's parameter, bilinear hats across)
//!   are integrated along the parameter-space image of each segment:
//!   `P_{e,s} = ∫_s W_e dq_d / Δq_e`, `q_d ∈ {ρ, α, z}`, `Δq_e` the edge's
//!   parameter length. The voltages are circulations `u_e = ∫_e E·dl`, so
//!   `(Pᵀu)_s = ∫_s E·dl` exactly as on a Cartesian grid, and the deposit
//!   is charge conserving (`∇λ_n` is the sum of the edge forms of `n`,
//!   a topological identity). The segment is split where it crosses a
//!   `ρ` cylinder, an `α` half-plane or a `z` plane; each piece is
//!   integrated with 5-point Gauss–Legendre (the integrand is smooth but
//!   not polynomial).
//! * **Stored copies.** On a closed azimuth the edges of the seam, and on
//!   a grid with an axis the axis z edge, are stored more than once; the
//!   wire couples to the physical edge: its canonical copy with the
//!   merged `cb` (`1/cb = Σ 1/cb_j`), and the update is written into every
//!   copy (so the periodic and axis merges keep it).
//! * **In-cell corrections.** The lattice is taken locally uniform with
//!   the physical spacings `(Δρ, ρ_c·Δα, Δz)` of the reference cell
//!   (`ρ_c` its mid radius) in the unrolled frame `(ρ, ρ_c·α, z)`, whose
//!   hats are exactly the parameter hats of the real cells; the thin-wire
//!   kernels use the true Cartesian geometry. The lattice Green's function
//!   ignores the variation of the metric across the correction window
//!   (relative error of order `window/ρ`).
//!
//! Wires must keep at least the first radial cell between them and the
//! axis (the parameter map is singular there).

use std::collections::BTreeMap;

use super::deposit::cell;
use crate::{Error, Result};

/// The `(ρ, α, z)` lattice of a cylindrical grid (lines in metres and
/// radians).
#[derive(Debug, Clone, Copy)]
pub(super) struct CylFrame<'a> {
    pub lines: [&'a [f64]; 3],
    pub closed: bool,
    pub axis: bool,
}

impl<'a> CylFrame<'a> {
    pub(super) fn new(lines: [&'a [f64]; 3]) -> Self {
        let a = lines[1];
        let tau = core::f64::consts::TAU;
        Self {
            lines,
            closed: ((a[a.len() - 1] - a[0]) - tau).abs() <= 1e-9 * tau,
            axis: lines[0][0] == 0.0,
        }
    }

    /// `α` of the Cartesian point `(x, y)` in the grid's α range (closed:
    /// `[α_0, α_0 + 2π)`; a sector: the representative nearest to it).
    fn alpha(&self, x: f64, y: f64) -> f64 {
        let tau = core::f64::consts::TAU;
        let a = self.lines[1];
        let (lo, hi) = (a[0], a[a.len() - 1]);
        let t = y.atan2(x);
        let mut best = t;
        let mut dist = f64::INFINITY;
        for k in -2..=2 {
            let v = t + f64::from(k) * tau;
            let d = if v < lo {
                lo - v
            } else if v > hi {
                v - hi
            } else {
                0.0
            };
            let d = if self.closed && v >= hi {
                f64::INFINITY
            } else {
                d
            };
            if d < dist {
                dist = d;
                best = v;
            }
        }
        best
    }

    /// Parameter point `(ρ, α, z)` of the Cartesian point `p` \[m\].
    pub(super) fn param(&self, p: [f64; 3]) -> [f64; 3] {
        [p[0].hypot(p[1]), self.alpha(p[0], p[1]), p[2]]
    }

    /// Physical cell sizes `(Δρ, ρ_c·Δα, Δz)` \[m\] of the cell containing
    /// the Cartesian point `p`.
    pub(super) fn cell_size(&self, p: [f64; 3]) -> [f64; 3] {
        let q = self.param(p);
        let c: [usize; 3] = core::array::from_fn(|d| cell(self.lines[d], q[d]));
        let l = self.lines;
        let rc = 0.5 * (l[0][c[0]] + l[0][c[0] + 1]);
        [
            l[0][c[0] + 1] - l[0][c[0]],
            rc * (l[1][c[1] + 1] - l[1][c[1]]),
            l[2][c[2] + 1] - l[2][c[2]],
        ]
    }

    /// Whether the Cartesian point lies inside the grid (with tolerance
    /// `tol` m), clear of the first radial cell on a grid with an axis.
    pub(super) fn contains(&self, p: [f64; 3], tol: f64) -> bool {
        let q = self.param(p);
        let l = self.lines;
        let rmin = if self.axis { l[0][1] } else { l[0][0] };
        let ok_r = q[0] >= rmin - tol && q[0] <= l[0][l[0].len() - 1] + tol;
        let ok_z = q[2] >= l[2][0] - tol && q[2] <= l[2][l[2].len() - 1] + tol;
        let ok_a = self.closed || {
            let (lo, hi) = (l[1][0], l[1][l[1].len() - 1]);
            let at = tol / q[0].max(1e-30);
            q[1] >= lo - at && q[1] <= hi + at
        };
        ok_r && ok_z && ok_a
    }

    /// Smallest distance of the straight segment `a → b` from the z axis.
    pub(super) fn axis_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 {
            (-(a[0] * dx + a[1] * dy) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (a[0] + t * dx).hypot(a[1] + t * dy)
    }

    /// The stored copies of the E edge `(d, node)`, canonical first.
    pub(super) fn copies(&self, d: usize, node: [usize; 3]) -> Vec<[usize; 3]> {
        let na = self.lines[1].len();
        if self.axis && d == 2 && node[0] == 0 {
            return (0..na)
                .map(|j| {
                    let mut q = node;
                    q[1] = j;
                    q
                })
                .collect();
        }
        if self.closed && d != 1 && (node[1] == 0 || node[1] == na - 1) {
            let (mut lo, mut hi) = (node, node);
            lo[1] = 0;
            hi[1] = na - 1;
            return vec![lo, hi];
        }
        vec![node]
    }

    /// Whitney edge weights `∫_s W_e dq_d` (parameter units) of the
    /// straight Cartesian segment `a → b` \[m\], added to `out` under the
    /// canonical copy of each edge.
    pub(super) fn segment_weights(
        &self,
        a: [f64; 3],
        b: [f64; 3],
        out: &mut BTreeMap<(usize, [usize; 3]), f64>,
    ) -> Result<()> {
        let l = self.lines;
        let d: [f64; 3] = core::array::from_fn(|k| b[k] - a[k]);
        let at = |t: f64| -> [f64; 3] { core::array::from_fn(|k| a[k] + t * d[k]) };
        let mut ts = vec![0.0, 1.0];
        // ρ cylinders: |a + t d|_xy = ρ_i.
        let (qa, qb) = (d[0] * d[0] + d[1] * d[1], a[0] * d[0] + a[1] * d[1]);
        let qc0 = a[0] * a[0] + a[1] * a[1];
        if qa > 0.0 {
            for &r in l[0] {
                let disc = qb * qb - qa * (qc0 - r * r);
                if disc > 0.0 {
                    let s = disc.sqrt();
                    ts.push((-qb - s) / qa);
                    ts.push((-qb + s) / qa);
                }
            }
        }
        // α half-planes through the axis.
        for &al in l[1] {
            let (s, c) = al.sin_cos();
            let den = d[0] * s - d[1] * c;
            if den != 0.0 {
                let t = -(a[0] * s - a[1] * c) / den;
                let p = at(t);
                if p[0] * c + p[1] * s > 0.0 {
                    ts.push(t);
                }
            }
        }
        // z planes.
        if d[2] != 0.0 {
            for &z in l[2] {
                ts.push((z - a[2]) / d[2]);
            }
        }
        ts.retain(|t| (0.0..=1.0).contains(t));
        ts.sort_by(f64::total_cmp);
        ts.dedup_by(|x, y| (*x - *y).abs() <= 1e-14);
        // 5-point Gauss–Legendre on [0, 1].
        let g5: [(f64, f64); 5] = {
            let x1 = (5.0 - 2.0 * (10.0f64 / 7.0).sqrt()).sqrt() / 3.0;
            let x2 = (5.0 + 2.0 * (10.0f64 / 7.0).sqrt()).sqrt() / 3.0;
            let w1 = (322.0 + 13.0 * 70f64.sqrt()) / 900.0;
            let w2 = (322.0 - 13.0 * 70f64.sqrt()) / 900.0;
            [
                (0.5 - 0.5 * x2, 0.5 * w2),
                (0.5 - 0.5 * x1, 0.5 * w1),
                (0.5, 0.5 * 128.0 / 225.0),
                (0.5 + 0.5 * x1, 0.5 * w1),
                (0.5 + 0.5 * x2, 0.5 * w2),
            ]
        };
        for w in ts.windows(2) {
            let (t0, t1) = (w[0], w[1]);
            let len = t1 - t0;
            if len <= 0.0 {
                continue;
            }
            let qm = self.param(at(0.5 * (t0 + t1)));
            if qm[0] <= 0.0 {
                return Err(Error::Invalid(
                    "oblique wire: a segment passes through the axis".into(),
                ));
            }
            let cells: [usize; 3] = core::array::from_fn(|k| cell(l[k], qm[k]));
            for (g, wg) in g5 {
                let t = t0 + g * len;
                let p = at(t);
                let rho = p[0].hypot(p[1]);
                // α continuous with the piece's midpoint.
                let mut al = p[1].atan2(p[0]);
                let tau = core::f64::consts::TAU;
                al += tau * ((qm[1] - al) / tau).round();
                let q = [rho, al, p[2]];
                let dq = [
                    (p[0] * d[0] + p[1] * d[1]) / rho,
                    (p[0] * d[1] - p[1] * d[0]) / (rho * rho),
                    d[2],
                ];
                let f: [f64; 3] = core::array::from_fn(|k| {
                    let c = cells[k];
                    ((q[k] - l[k][c]) / (l[k][c + 1] - l[k][c])).clamp(0.0, 1.0)
                });
                for (dd, &v) in dq.iter().enumerate() {
                    if v == 0.0 {
                        continue;
                    }
                    let (o1, o2) = ((dd + 1) % 3, (dd + 2) % 3);
                    for k in 0..4 {
                        let (u1, u2) = (k & 1 == 1, k & 2 == 2);
                        let wt = (if u1 { f[o1] } else { 1.0 - f[o1] })
                            * (if u2 { f[o2] } else { 1.0 - f[o2] });
                        if wt == 0.0 {
                            continue;
                        }
                        let mut node = cells;
                        node[o1] += usize::from(u1);
                        node[o2] += usize::from(u2);
                        let node = self.copies(dd, node)[0];
                        *out.entry((dd, node)).or_insert(0.0) += wg * len * v * wt;
                    }
                }
            }
        }
        Ok(())
    }

    /// The unrolled local frame at the Cartesian point `p`: virtual lattice
    /// lines (uniform, the physical spacings of `p`'s cell, anchored at its
    /// lower lines, `reach` m around it) and the map of Cartesian points
    /// into it, `(ρ, ρ_c·α, z)` with `α` continuous around `p`.
    pub(super) fn local(&self, p: [f64; 3], reach: f64) -> Local {
        let l = self.lines;
        let q = self.param(p);
        let c: [usize; 3] = core::array::from_fn(|d| cell(l[d], q[d]));
        let rc = 0.5 * (l[0][c[0]] + l[0][c[0] + 1]);
        let h = [
            l[0][c[0] + 1] - l[0][c[0]],
            rc * (l[1][c[1] + 1] - l[1][c[1]]),
            l[2][c[2] + 1] - l[2][c[2]],
        ];
        let x0 = [l[0][c[0]], rc * l[1][c[1]], l[2][c[2]]];
        let lines = core::array::from_fn(|d| {
            let m = (reach / h[d]).ceil() as i64 + 3;
            (-m..=m + 1).map(|i| x0[d] + i as f64 * h[d]).collect()
        });
        Local {
            lines,
            h,
            rc,
            alpha: q[1],
        }
    }
}

/// See [`CylFrame::local`].
pub(super) struct Local {
    pub lines: [Vec<f64>; 3],
    pub h: [f64; 3],
    rc: f64,
    alpha: f64,
}

impl Local {
    /// The unrolled coordinates of the Cartesian point `x`.
    pub(super) fn map(&self, x: [f64; 3]) -> [f64; 3] {
        let tau = core::f64::consts::TAU;
        let mut al = x[1].atan2(x[0]);
        al += tau * ((self.alpha - al) / tau).round();
        [x[0].hypot(x[1]), self.rc * al, x[2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_lines() -> [Vec<f64>; 3] {
        let r: Vec<f64> = (0..=10).map(|i| 0.1 + 0.01 * f64::from(i)).collect();
        let a: Vec<f64> = (0..=24)
            .map(|j| core::f64::consts::TAU * f64::from(j) / 24.0)
            .collect();
        let z: Vec<f64> = (0..=6).map(|k| 0.01 * f64::from(k)).collect();
        [r, a, z]
    }

    /// The edge weights are the parameter-space Whitney forms: their
    /// discrete divergence at every node is `λ_n(b) − λ_n(a)` (charge
    /// conservation), for a segment that crosses the seam.
    #[test]
    fn segment_weights_conserve_charge_across_the_seam() {
        let l = frame_lines();
        let f = CylFrame::new([&l[0], &l[1], &l[2]]);
        assert!(f.closed && !f.axis);
        let a = [0.15, -0.03, 0.012];
        let b = [0.13, 0.04, 0.037];
        let mut w = BTreeMap::new();
        f.segment_weights(a, b, &mut w).unwrap();
        // Node hats in parameter space (canonical copies).
        let hats = |p: [f64; 3]| -> BTreeMap<[usize; 3], f64> {
            let q = f.param(p);
            let mut m = BTreeMap::new();
            let c: [usize; 3] = core::array::from_fn(|d| cell(&l[d], q[d]));
            let fr: [f64; 3] =
                core::array::from_fn(|d| (q[d] - l[d][c[d]]) / (l[d][c[d] + 1] - l[d][c[d]]));
            for k in 0..8 {
                let mut n = c;
                let mut v = 1.0;
                for d in 0..3 {
                    let up = (k >> d) & 1 == 1;
                    n[d] += usize::from(up);
                    v *= if up { fr[d] } else { 1.0 - fr[d] };
                }
                if n[1] == l[1].len() - 1 {
                    n[1] = 0;
                }
                *m.entry(n).or_insert(0.0) += v;
            }
            m
        };
        let (ha, hb) = (hats(a), hats(b));
        let mut div: BTreeMap<[usize; 3], f64> = BTreeMap::new();
        for (&(d, n), &v) in &w {
            let dq = l[d][n[d] + 1] - l[d][n[d]];
            let mut m = n;
            m[d] += 1;
            if d == 1 && m[1] == l[1].len() - 1 {
                m[1] = 0;
            }
            *div.entry(m).or_insert(0.0) += v / dq;
            *div.entry(n).or_insert(0.0) -= v / dq;
        }
        let mut keys: Vec<_> = div
            .keys()
            .chain(ha.keys())
            .chain(hb.keys())
            .copied()
            .collect();
        keys.sort();
        keys.dedup();
        for n in keys {
            let want = hb.get(&n).unwrap_or(&0.0) - ha.get(&n).unwrap_or(&0.0);
            let got = div.get(&n).unwrap_or(&0.0);
            assert!((got - want).abs() < 1e-12, "{n:?}: {got} vs {want}");
        }
        // And ∫ E·dl of a uniform field E = x̂: Σ P_e u_e ≈ Δx.
        let mut s = 0.0;
        for (&(d, n), &v) in &w {
            let (q0, mut q1) = (n, n);
            q1[d] += 1;
            let cart = |n: [usize; 3]| {
                let (r, al, z) = (l[0][n[0]], l[1][n[1]], l[2][n[2]]);
                [r * al.cos(), r * al.sin(), z]
            };
            // u_e of E = x̂ is the x extent of the edge (exact for any path).
            let u = cart(q1)[0] - cart(q0)[0];
            s += v / (l[d][n[d] + 1] - l[d][n[d]]) * u;
        }
        // (Second order in the cell size: 15° cells interpolate it to 2 %.)
        assert!(
            (s - (b[0] - a[0])).abs() < 3e-2 * (b[0] - a[0]).abs(),
            "{s}"
        );
    }
}
