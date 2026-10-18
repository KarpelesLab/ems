//! Cylindrical-coordinate support: the axis, the closed azimuth and the
//! stability limit (design §5.15).
//!
//! A cylindrical [`Grid`] has the axes `(ρ, α, z)` ([`Axis::X`], `Y`, `Z`).
//! The update kernels are the Cartesian ones: in integral form they only
//! see circulations of `u` and `i`, and all of the geometry sits in the
//! coefficients, built from the physical edge lengths and face areas of
//! the grid ([`Grid::edge_length`], [`Grid::dual_face_area`], …). Three
//! things need extra care.
//!
//! # Closed azimuth
//!
//! When the α lines span 2π ([`Grid::is_closed`]) the first and last α
//! planes are the same half-plane. The α faces are then always treated as
//! [`FaceCondition::Periodic`] (whatever was configured), so the periodic
//! face merge of [`boundary::periodic`](super::boundary::periodic) joins the
//! two copies.
//!
//! # The axis
//!
//! When the first ρ line is 0 ([`Grid::has_axis`]) the `ρ_min` face is the
//! z axis, not a boundary; its face condition is ignored (the face is
//! built like a [`FaceCondition::Pmc`] face, i.e. its edges keep their
//! coefficients). On the axis:
//!
//! * **α edges** have zero length (`ρ·Δα = 0`); they do not exist
//!   (`ca = cb = 0`, `u = 0`).
//! * **ρ faces** have zero area, so the H_ρ dual edges through them carry
//!   no flux and do not exist (`da = db = 0`). (Their circulation, two α
//!   edges and two copies of the axis z edge, vanishes anyway.)
//! * **The z edge** on the axis is a single edge, stored once per α index
//!   `j`. Copy `j` is updated by the kernel over its sector of the dual
//!   disc, `ρ ≤ ρ_½`, `α*_{j−½} … α*_{j+½}`, whose contour is the arc of
//!   H_α at `ρ_½` (plus radial pieces carried by the non-existing H_ρ). By
//!   Ampère's law over the whole disc,
//!
//!   ```text
//!   C·du/dt + G·u = Σ_j i_α(ρ_½, α_j),   C = Σ_j C_j,  G = Σ_j G_j,
//!   ```
//!
//!   so — exactly as for the two copies of a periodic face edge — the true
//!   update is the `1/cb`-weighted mean of the copies' kernel results,
//!   `u' = Σ w_j·u'_j / Σ w_j` with `w_j = 1/cb_j = C_j/dt + G_j/2`
//!   (see the derivation in [`boundary::periodic`](super::boundary::periodic)).
//!   After every E update (in the serial `post_e` gap, after the periodic
//!   merge and again after operator `post_e` hooks) all copies are replaced
//!   by that mean. A copy with `cb = 0` (e.g. on a PEC α
//!   face of a sector) makes the axis edge PEC; if all copies have
//!   `cb = 0` (an axis end on a Mur face) the operator's values are kept
//!   (averaged). The field energy sums all
//!   copies, `Σ_j C_j u² = C·u²`, so the energy invariant holds.
//!
//! # Time step
//!
//! The α edges near the axis are short (`ρ·Δα`), and no simple per-axis
//! formula bounds the Yee operator there. [`courant_limit`](super::courant_limit) computes the
//! limit of the actual discrete operator instead: the leapfrog scheme is
//! stable for `dt < 2/√λ_max` with `λ_max` the largest eigenvalue of
//! `A = C⁻¹·Kᵀ·L⁻¹·K` (`K` the face circulation, `C`, `L` the edge
//! capacitances and inductances). It is estimated with the Lanczos method
//! in the `C` inner product (in which `A` is self-adjoint) on the vacuum
//! operator with all non-periodic faces open (PMC, the most unstable
//! choice: PEC faces and materials with `ε_r, μ_r ≥ 1` only lower
//! `λ_max`), including the periodic and axis merges, and enlarged by
//! 0.1 % for safety. The result is cached in the grid.

use super::coeffs::{Coefficients, FaceCondition, FaceConditions};
use super::edge_material::EdgeMaterials;
use super::fields::Fields;
use super::grid::Grid;
use crate::axis::Axis;
use crate::fdtd::Face;

/// The face conditions the coefficient builder and the engine actually
/// use: on a closed cylindrical grid the α faces are periodic, on a grid
/// with an axis the `ρ_min` face is open (built like PMC). Cartesian grids
/// are returned unchanged.
pub(crate) fn effective_faces(grid: &Grid, faces: &FaceConditions) -> FaceConditions {
    let mut f = *faces;
    if grid.is_closed() {
        f[Face::YMin] = FaceCondition::Periodic;
        f[Face::YMax] = FaceCondition::Periodic;
    }
    if grid.has_axis() {
        f[Face::XMin] = FaceCondition::Pmc;
    }
    f
}

/// The stored copies of the physical E edge along `axis` at `node`, the
/// canonical one (lowest index) first: all α copies of an axis z edge,
/// both copies of an edge on the α faces of a closed grid, else the edge
/// itself (always on Cartesian grids).
///
/// Sources and lumped elements act on the physical edge: the simulation
/// mapping applies an additive source to every copy (the merges keep the
/// copies equal) and a lumped element to the canonical copy only (the
/// `1/cb`-weighted merge then gives the full edge exactly that element).
pub fn e_edge_copies(grid: &Grid, axis: Axis, node: [usize; 3]) -> Vec<[usize; 3]> {
    let n = grid.nodes();
    if grid.has_axis() && axis == Axis::Z && node[0] == 0 {
        return (0..n[1])
            .map(|j| {
                let mut p = node;
                p[1] = j;
                p
            })
            .collect();
    }
    if grid.is_closed() && axis != Axis::Y && (node[1] == 0 || node[1] == n[1] - 1) {
        let (mut lo, mut hi) = (node, node);
        lo[1] = 0;
        hi[1] = n[1] - 1;
        return vec![lo, hi];
    }
    vec![node]
}

/// The axis merge of a cylindrical grid with an axis; see the
/// [module docs](self#the-axis).
#[derive(Debug, Clone)]
pub(crate) struct AxisMerge {
    n: [usize; 3],
}

impl AxisMerge {
    /// The merge for `grid`, if it has an axis.
    pub(crate) fn new(grid: &Grid) -> Option<Self> {
        grid.has_axis().then(|| Self { n: grid.nodes() })
    }

    /// Replaces every copy of each axis z edge by the `1/cb`-weighted mean
    /// of the copies (0 if any copy is PEC). Fixed order, deterministic.
    pub(crate) fn merge(&self, f: &mut Fields<'_>, co: &Coefficients) {
        let [_, na, nz] = self.n;
        let cb = &co.cb(Axis::Z)[..na * nz];
        let plane = f.u.plane_mut(Axis::Z, 0);
        for k in 0..nz - 1 {
            let (mut num, mut den, mut pec) = (0.0f64, 0.0f64, 0usize);
            let mut mean = 0.0f64;
            for j in 0..na {
                let x = plane[j * nz + k] as f64;
                mean += x;
                let b = cb[j * nz + k];
                if b == 0.0 {
                    pec += 1;
                    continue;
                }
                let w = 1.0 / b as f64;
                num += w * x;
                den += w;
            }
            // All copies without kernel update (e.g. on a Mur face): keep
            // the operator's values, averaged; some PEC: the edge is PEC.
            let v = if pec == na {
                (mean / na as f64) as f32
            } else if pec > 0 {
                0.0
            } else {
                (num / den) as f32
            };
            for j in 0..na {
                plane[j * nz + k] = v;
            }
        }
    }
}

/// Stability limit `dt_max` \[s\] of a cylindrical grid, from the largest
/// eigenvalue of its vacuum Yee operator (see the
/// [module docs](self#time-step)).
pub(crate) fn courant_limit(grid: &Grid) -> f64 {
    let faces = FaceConditions::all(FaceCondition::Pmc);
    2.0 / (operator_lambda_max(grid, &faces) * 1.001).sqrt()
}

/// The operator `A = P·C⁻¹·Kᵀ·L⁻¹·K` on vacuum coefficients (built with
/// `dt = 1`, so `cb = 1/C` and `db = 1/L`), with `P` the periodic and
/// axis merges.
struct Op {
    n: [usize; 3],
    co: Coefficients,
    periodic: Option<super::boundary::periodic::Periodic>,
    axis: Option<AxisMerge>,
}

impl Op {
    fn new(grid: &Grid, faces: &FaceConditions) -> Self {
        let faces = effective_faces(grid, faces);
        let co = Coefficients::from_edges(grid, &EdgeMaterials::vacuum(grid), &faces, 1.0)
            .expect("vacuum coefficients of a valid grid");
        let periodic = super::boundary::periodic::Periodic::from_faces(&faces, grid.nodes())
            .expect("effective faces pair up");
        Self {
            n: grid.nodes(),
            co,
            periodic,
            axis: AxisMerge::new(grid),
        }
    }

    fn idx(&self, [i, j, k]: [usize; 3]) -> usize {
        (i * self.n[1] + j) * self.n[2] + k
    }

    /// `C`-weight of an E entry (`1/cb`, 0 for non-existing edges).
    fn weight(&self, a: usize, idx: usize) -> f64 {
        let b = self.co.cb(Axis::ALL[a])[idx];
        if b == 0.0 { 0.0 } else { 1.0 / b as f64 }
    }

    fn dot(&self, x: &[Vec<f64>; 3], y: &[Vec<f64>; 3]) -> f64 {
        let mut s = 0.0;
        for a in 0..3 {
            for (idx, (&p, &q)) in x[a].iter().zip(&y[a]).enumerate() {
                if p != 0.0 && q != 0.0 {
                    s += self.weight(a, idx) * p * q;
                }
            }
        }
        s
    }

    /// The `C`-orthogonal projection onto merged fields (periodic copies
    /// and axis copies equal), done through the engine's own merges on a
    /// scratch `f32`-free path: weighted means in `f64`.
    fn project(&self, u: &mut [Vec<f64>; 3]) {
        // Zero non-existing edges.
        for (a, comp) in u.iter_mut().enumerate() {
            let cb = self.co.cb(Axis::ALL[a]);
            for (v, &b) in comp.iter_mut().zip(cb) {
                if b == 0.0 {
                    *v = 0.0;
                }
            }
        }
        if let Some(p) = &self.periodic {
            p.merge_f64(u, &self.co);
        }
        if self.axis.is_some() {
            let [_, na, nz] = self.n;
            let cb = self.co.cb(Axis::Z);
            for k in 0..nz - 1 {
                let (mut num, mut den) = (0.0, 0.0);
                for j in 0..na {
                    let idx = j * nz + k;
                    let b = cb[idx];
                    if b != 0.0 {
                        num += u[2][idx] / b as f64;
                        den += 1.0 / b as f64;
                    }
                }
                let v = if den > 0.0 { num / den } else { 0.0 };
                for j in 0..na {
                    if cb[j * nz + k] != 0.0 {
                        u[2][j * nz + k] = v;
                    }
                }
            }
        }
    }

    /// `w = A·u`.
    fn apply(&self, u: &[Vec<f64>; 3]) -> [Vec<f64>; 3] {
        let n = self.n;
        let len = n[0] * n[1] * n[2];
        let mut out: [Vec<f64>; 3] = core::array::from_fn(|_| vec![0.0; len]);
        // Faces normal to `a` at node p, (a, b, c) cyclic:
        // ∮u = u_b(p) + u_c(p + e_b) − u_b(p + e_c) − u_c(p).
        for a in 0..3 {
            let (b, c) = ((a + 1) % 3, (a + 2) % 3);
            let db = self.co.db(Axis::ALL[a]);
            for i in 0..n[0] {
                for j in 0..n[1] {
                    for k in 0..n[2] {
                        let p = [i, j, k];
                        let idx = self.idx(p);
                        let d = db[idx];
                        if d == 0.0 || p[b] + 1 >= n[b] || p[c] + 1 >= n[c] {
                            continue;
                        }
                        let mut pb = p;
                        pb[b] += 1;
                        let mut pc = p;
                        pc[c] += 1;
                        let (ib, ic) = (self.idx(pb), self.idx(pc));
                        let circ = u[b][idx] + u[c][ib] - u[b][ic] - u[c][idx];
                        let h = d as f64 * circ;
                        // Kᵀ: the same four edges with the same signs.
                        out[b][idx] += h;
                        out[c][ib] += h;
                        out[b][ic] -= h;
                        out[c][idx] -= h;
                    }
                }
            }
        }
        for (a, comp) in out.iter_mut().enumerate() {
            let cb = self.co.cb(Axis::ALL[a]);
            for (v, &b) in comp.iter_mut().zip(cb) {
                *v *= b as f64;
            }
        }
        self.project(&mut out);
        out
    }
}

/// Largest eigenvalue \[1/s²\] of the vacuum Yee operator of `grid` with the
/// face conditions `faces` (made effective, see [`effective_faces`]), by
/// the Lanczos method in the `C` inner product.
pub(crate) fn operator_lambda_max(grid: &Grid, faces: &FaceConditions) -> f64 {
    let op = Op::new(grid, faces);
    let len = grid.num_nodes();
    // Deterministic pseudo-random start vector.
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut q: [Vec<f64>; 3] = core::array::from_fn(|_| {
        (0..len)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    });
    op.project(&mut q);
    let norm = op.dot(&q, &q).sqrt();
    if norm == 0.0 {
        return 0.0;
    }
    scale(&mut q, 1.0 / norm);
    let mut prev: [Vec<f64>; 3] = core::array::from_fn(|_| vec![0.0; len]);
    let (mut alpha, mut beta): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut last = 0.0f64;
    let mut stable = 0;
    let max_iter = 400.min(3 * len);
    for _ in 0..max_iter {
        let mut w = op.apply(&q);
        if let Some(&b) = beta.last() {
            axpy(&mut w, -b, &prev);
        }
        let a = op.dot(&w, &q);
        axpy(&mut w, -a, &q);
        alpha.push(a);
        let theta = tridiagonal_max(&alpha, &beta);
        if (theta - last).abs() <= 1e-9 * theta {
            stable += 1;
            if stable >= 10 {
                return theta;
            }
        } else {
            stable = 0;
        }
        last = theta;
        let b = op.dot(&w, &w).sqrt();
        if b <= 1e-12 * theta.abs().max(f64::MIN_POSITIVE) {
            return theta;
        }
        beta.push(b);
        scale(&mut w, 1.0 / b);
        prev = core::mem::replace(&mut q, w);
    }
    last
}

fn scale(x: &mut [Vec<f64>; 3], s: f64) {
    for v in x.iter_mut().flatten() {
        *v *= s;
    }
}

fn axpy(y: &mut [Vec<f64>; 3], a: f64, x: &[Vec<f64>; 3]) {
    for (yc, xc) in y.iter_mut().zip(x) {
        for (p, &q) in yc.iter_mut().zip(xc) {
            *p += a * q;
        }
    }
}

/// Largest eigenvalue of the symmetric tridiagonal matrix with diagonal
/// `d` and off-diagonal `e` (`e.len() ≥ d.len() − 1`), by Sturm-sequence
/// bisection.
fn tridiagonal_max(d: &[f64], e: &[f64]) -> f64 {
    let m = d.len();
    let off = |i: usize| if i < e.len() { e[i].abs() } else { 0.0 };
    // Gershgorin interval.
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (i, &di) in d.iter().enumerate() {
        let r = if i + 1 < m { off(i) } else { 0.0 } + if i > 0 { off(i - 1) } else { 0.0 };
        lo = lo.min(di - r);
        hi = hi.max(di + r);
    }
    // Number of eigenvalues below x.
    let count = |x: f64| {
        let mut c = 0;
        let mut q = 1.0f64;
        for (i, &di) in d.iter().enumerate() {
            let e2 = if i > 0 { off(i - 1).powi(2) } else { 0.0 };
            q = di - x - if i > 0 { e2 / q } else { 0.0 };
            if q == 0.0 {
                q = -1e-300;
            }
            if q < 0.0 {
                c += 1;
            }
        }
        c
    };
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if count(mid) >= m {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::C0;
    use crate::engine::coeffs;
    use core::f64::consts::TAU;

    #[test]
    fn tridiagonal() {
        // [[2, 1], [1, 2]] → 3.
        assert!((tridiagonal_max(&[2.0, 2.0], &[1.0]) - 3.0).abs() < 1e-12);
        assert!((tridiagonal_max(&[5.0], &[]) - 5.0).abs() < 1e-12);
        // 1-D Laplacian, n = 6: 2 − 2cos(6π/7).
        let d = vec![2.0; 6];
        let e = vec![-1.0; 5];
        let want = 2.0 - 2.0 * (6.0 * core::f64::consts::PI / 7.0).cos();
        assert!((tridiagonal_max(&d, &e) - want).abs() < 1e-12);
    }

    #[test]
    fn lanczos_matches_cartesian_limit() {
        // On a uniform Cartesian box with PMC faces the alternating
        // (checkerboard) mode reaches the CFL formula: the estimate must
        // reproduce it (up to the f32 coefficients).
        let g = Grid::uniform([12, 10, 8], [1e-3, 1.5e-3, 2e-3]).unwrap();
        let lam = operator_lambda_max(&g, &FaceConditions::all(FaceCondition::Pmc));
        let dt = 2.0 / lam.sqrt();
        let cfl = coeffs::courant_limit(&g);
        assert!((dt / cfl - 1.0).abs() < 1e-6, "dt {dt:e} vs cfl {cfl:e}");
    }

    #[test]
    fn cylinder_limit_far_from_axis_is_cartesian_like() {
        // A thin shell at large radius is almost a Cartesian box with
        // Δy = ρ·Δα.
        let r0 = 1.0;
        let rho: Vec<f64> = (0..=8).map(|i| r0 + i as f64 * 1e-3).collect();
        let alpha: Vec<f64> = (0..=8).map(|j| j as f64 * 1e-3 / r0).collect();
        let z: Vec<f64> = (0..=8).map(|k| k as f64 * 1e-3).collect();
        let g = Grid::cylindrical([rho, alpha, z]).unwrap();
        let cart = Grid::uniform([8, 8, 8], [1e-3; 3]).unwrap();
        let (a, b) = (coeffs::courant_limit(&g), coeffs::courant_limit(&cart));
        assert!((a / b - 1.0).abs() < 0.03, "{a:e} vs {b:e}");
        // Cached.
        assert_eq!(coeffs::courant_limit(&g), a);
    }

    #[test]
    fn axis_limit_scales_with_the_inner_arc() {
        // Full disc with an axis: the limit is set near the axis, of the
        // order of the arc ρ_½·Δα, and halves when the α resolution doubles.
        let rho: Vec<f64> = (0..=6).map(|i| i as f64 * 1e-3).collect();
        let z = vec![0.0, 1e-3, 2e-3];
        let lim = |na: usize| {
            let alpha = (0..=na).map(|j| j as f64 * TAU / na as f64).collect();
            Grid::cylindrical([rho.clone(), alpha, z.clone()]).unwrap()
        };
        let (d16, d32) = (
            coeffs::courant_limit(&lim(16)),
            coeffs::courant_limit(&lim(32)),
        );
        let arc = 0.5e-3 * TAU / 16.0;
        assert!(d16 > 0.3 * arc / C0 && d16 < 3.0 * arc / C0, "{d16:e}");
        assert!((d16 / d32 - 2.0).abs() < 0.2, "{}", d16 / d32);
    }
}
