//! Azimuthal multigrid near the axis of a cylindrical grid (design
//! §5.15.1).
//!
//! Inside a zone boundary `ρ_I` ([`Grid::with_multigrid`]) only every
//! `f`-th α line is kept (`f = 2^ℓ` inside `ℓ` boundaries), so the arcs
//! `ρ·Δα` near the axis are `f` times longer and the stability limit
//! grows accordingly. The coarse zones are not stored separately: the
//! fields keep the fine layout and the coarse Yee grid is embedded in it
//! as a subspace,
//!
//! * the E edges and H dual edges off the kept lines do not exist
//!   (coefficients 0, see [`Grid::mg_removed_e`]),
//! * the kept ρ and z edges and α dual edges carry the coarse dual
//!   geometry ([`Grid::dual_face_area`], [`Grid::dual_edge_length`]),
//! * a coarse α edge is the chain of the `f` fine α edges it covers, with
//!   `u_e = (Δα_e/Δα_coarse)·U` (E_α constant along the arc), and a coarse
//!   ρ or z face is the union of `f` fine faces, all carrying the same
//!   `i` (copies),
//! * on a zone boundary `ρ_I` (coarse inside, `f/2` outside) the z edges
//!   kept only by the outer zone are slaves: `u = (1 − t)·U_a + t·U_b`
//!   from the two neighbouring kept ("master") z edges on that line.
//!
//! Write `S` (E) and `T` (H) for these embeddings. The scheme run is the
//! Galerkin restriction of the fine Yee scheme to `range S × range T`:
//! after each kernel pass the fields are replaced by their `C`- (E) and
//! `L`-orthogonal (H) projections, `u ← S·(SᵀWS)⁻¹·SᵀW·u*` with
//! `W = diag(1/cb)` (and likewise `1/db` for H), which is exactly the
//! leapfrog update of the coarse unknowns with `C̃ = SᵀCS`, `L̃ = TᵀLT`
//! and the curls `TᵀKS`, `SᵀKᵀT` (lossy terms included, as in the periodic
//! merge). So the coupled scheme conserves the engine's energy invariant
//! (`½uᵀCu + ½iᵀLi'` of the fine arrays is the coarse one) and is stable
//! below its own operator limit, which [`courant_limit`](super::courant_limit)
//! computes with these projections.
//!
//! The projections are block-diagonal except on the zone boundaries,
//! where the master/slave coupling gives a (cyclic, on a closed grid)
//! tridiagonal system per z index, solved directly in `f64`. Everything
//! runs serially in the engine's operator gaps, in a fixed order, so the
//! results do not depend on the thread count.

use super::coeffs::Coefficients;
use super::fields::FieldArrays;
use super::grid::Grid;
use crate::axis::Axis;

/// Field values the projections work on (`f32` engine fields, `f64` in
/// the stability estimate).
pub(crate) trait Num: Copy {
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
}

impl Num for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl Num for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// Mutable x-planes (`ny·nz` values, index `j·nz + k`) of one field type.
pub(crate) trait Planes {
    type T: Num;
    fn plane(&mut self, axis: Axis, i: usize) -> &mut [Self::T];
}

impl Planes for FieldArrays<'_> {
    type T = f32;
    fn plane(&mut self, axis: Axis, i: usize) -> &mut [f32] {
        self.plane_mut(axis, i)
    }
}

/// `f64` arrays in field layout.
pub(crate) struct F64Planes<'a> {
    pub(crate) plane: usize,
    pub(crate) data: &'a mut [Vec<f64>; 3],
}

impl Planes for F64Planes<'_> {
    type T = f64;
    fn plane(&mut self, axis: Axis, i: usize) -> &mut [f64] {
        let p = self.plane;
        &mut self.data[axis.index()][i * p..(i + 1) * p]
    }
}

/// A zone boundary `ρ_I`: masters and slaves of its z edges.
#[derive(Debug, Clone)]
struct Interface {
    /// ρ line index.
    ring: usize,
    /// α indices of each master (one, or two for the periodic copy of
    /// master 0 on a closed grid).
    masters: Vec<Vec<usize>>,
    /// `(j, a, b, t)`: slave at α index `j`, `u = (1 − t)·U_a + t·U_b`.
    slaves: Vec<(usize, usize, usize, f64)>,
    /// Whether the master chain is cyclic (closed grid).
    cyclic: bool,
}

/// The multigrid projections of a cylindrical grid; see the
/// [module docs](self).
#[derive(Debug, Clone)]
pub(crate) struct Multigrid {
    n: [usize; 3],
    /// Coarse α edges: `(ring, first α index, weights Δα_e/Δα_coarse)`.
    alpha: Vec<(usize, usize, Vec<f64>)>,
    /// Coarse H faces: `(axis X or Z, ρ index, first α index, f)`.
    faces: Vec<(Axis, usize, usize, usize)>,
    interfaces: Vec<Interface>,
}

impl Multigrid {
    /// The projections for `grid`, if it has multigrid.
    pub(crate) fn new(grid: &Grid) -> Option<Self> {
        if !grid.has_multigrid() {
            return None;
        }
        let n = grid.nodes();
        let [nr, na, _] = n;
        let lines = grid.lines(Axis::Y);
        let mut alpha = Vec::new();
        let mut faces = Vec::new();
        for i in 0..nr {
            let f = grid.alpha_factor(i);
            if f == 1 {
                continue;
            }
            let on_axis = i == 0 && grid.has_axis();
            for j0 in (0..na - 1).step_by(f) {
                if !on_axis {
                    let span = lines[j0 + f] - lines[j0];
                    let w = (j0..j0 + f)
                        .map(|j| (lines[j + 1] - lines[j]) / span)
                        .collect();
                    alpha.push((i, j0, w));
                    faces.push((Axis::X, i, j0, f));
                }
                if i + 1 < nr {
                    faces.push((Axis::Z, i, j0, f));
                }
            }
        }
        let closed = grid.is_closed();
        let mut interfaces = Vec::new();
        for &ring in grid.multigrid() {
            let (inn, out) = grid.ring_factors(ring);
            debug_assert_eq!(inn, 2 * out);
            let p = (na - 1) / out;
            let m = if closed { p / 2 } else { p / 2 + 1 };
            let mut masters: Vec<Vec<usize>> = (0..m).map(|q| vec![2 * q * out]).collect();
            if closed {
                masters[0].push(na - 1);
            }
            let slaves = (0..p / 2)
                .map(|q| {
                    let (ja, js, jb) = (2 * q * out, (2 * q + 1) * out, (2 * q + 2) * out);
                    let t = (lines[js] - lines[ja]) / (lines[jb] - lines[ja]);
                    (js, q, (q + 1) % m, t)
                })
                .collect();
            interfaces.push(Interface {
                ring,
                masters,
                slaves,
                cyclic: closed,
            });
        }
        Some(Self {
            n,
            alpha,
            faces,
            interfaces,
        })
    }

    /// Replaces the E voltages by their `C`-orthogonal projection onto
    /// the multigrid subspace: coarse α edges and the master/slave z edges
    /// of the zone boundaries. A coarse α edge with some PEC fine edge is
    /// PEC; one whose fine edges all have `cb = 0` (e.g. on a Mur face)
    /// keeps the operator's values.
    pub(crate) fn merge_e<P: Planes>(&self, u: &mut P, co: &Coefficients) {
        let [_, na, nz] = self.n;
        let pl = na * nz;
        let cb_y = co.cb(Axis::Y);
        for (i, j0, w) in &self.alpha {
            let cb = &cb_y[i * pl..(i + 1) * pl];
            let v = u.plane(Axis::Y, *i);
            for k in 0..nz {
                let (mut num, mut den, mut pec) = (0.0f64, 0.0f64, 0usize);
                for (e, &s) in w.iter().enumerate() {
                    let idx = (j0 + e) * nz + k;
                    let b = cb[idx];
                    if b == 0.0 {
                        pec += 1;
                        continue;
                    }
                    let b = b as f64;
                    num += s * v[idx].to_f64() / b;
                    den += s * s / b;
                }
                if pec == w.len() {
                    continue;
                }
                let x = if pec > 0 { 0.0 } else { num / den };
                for (e, &s) in w.iter().enumerate() {
                    v[(j0 + e) * nz + k] = P::T::from_f64(s * x);
                }
            }
        }
        let cb_z = co.cb(Axis::Z);
        let mut sys = System::default();
        for itf in &self.interfaces {
            let cb = &cb_z[itf.ring * pl..(itf.ring + 1) * pl];
            let v = u.plane(Axis::Z, itf.ring);
            for k in 0..nz - 1 {
                if itf.solve(&mut sys, cb, v, nz, k) {
                    itf.store(&sys.x, v, cb, nz, k);
                }
            }
        }
    }

    /// Replaces the H currents of each coarse ρ and z face by the
    /// `1/db`-weighted mean of its fine copies (the `L`-orthogonal
    /// projection; copies with `db = 0` are left alone).
    pub(crate) fn merge_h<P: Planes>(&self, h: &mut P, co: &Coefficients) {
        let [_, na, nz] = self.n;
        let pl = na * nz;
        for &(axis, i, j0, f) in &self.faces {
            let db = &co.db(axis)[i * pl..(i + 1) * pl];
            let v = h.plane(axis, i);
            for k in 0..nz {
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for j in j0..j0 + f {
                    let idx = j * nz + k;
                    if db[idx] != 0.0 {
                        let w = 1.0 / db[idx] as f64;
                        num += w * v[idx].to_f64();
                        den += w;
                    }
                }
                if den == 0.0 {
                    continue;
                }
                let x = P::T::from_f64(num / den);
                for j in j0..j0 + f {
                    let idx = j * nz + k;
                    if db[idx] != 0.0 {
                        v[idx] = x;
                    }
                }
            }
        }
    }
}

/// Scratch space of a zone-boundary solve: the symmetric (cyclic)
/// tridiagonal system `d`, `e` (e[m] couples m and m + 1), `r`, its
/// solution `x` and a second work vector.
#[derive(Debug, Default)]
struct System {
    d: Vec<f64>,
    e: Vec<f64>,
    r: Vec<f64>,
    x: Vec<f64>,
    z: Vec<f64>,
    c: Vec<f64>,
}

impl Interface {
    /// Assembles and solves the normal equations `(SᵀWS)·U = SᵀW·u*` of
    /// plane `k`; `false` if no edge of the line is updated by the kernel
    /// (all `cb = 0`: keep the operator's values).
    fn solve<T: Num>(&self, s: &mut System, cb: &[f32], v: &[T], nz: usize, k: usize) -> bool {
        let m = self.masters.len();
        for buf in [&mut s.d, &mut s.e, &mut s.r] {
            buf.clear();
            buf.resize(m, 0.0);
        }
        let mut live = false;
        let mut pec = vec![false; m];
        for (q, js) in self.masters.iter().enumerate() {
            for &j in js {
                let b = cb[j * nz + k];
                if b == 0.0 {
                    pec[q] = true;
                } else {
                    live = true;
                    let w = 1.0 / b as f64;
                    s.d[q] += w;
                    s.r[q] += w * v[j * nz + k].to_f64();
                }
            }
        }
        for &(j, a, b, t) in &self.slaves {
            let c = cb[j * nz + k];
            if c == 0.0 {
                continue;
            }
            live = true;
            let w = 1.0 / c as f64;
            let x = v[j * nz + k].to_f64();
            s.d[a] += w * (1.0 - t) * (1.0 - t);
            s.d[b] += w * t * t;
            // `b = a + 1` (mod m): the coupling of a and its successor.
            s.e[a] += w * t * (1.0 - t);
            s.r[a] += w * (1.0 - t) * x;
            s.r[b] += w * t * x;
        }
        if !live {
            return false;
        }
        // PEC masters are fixed at 0: drop their couplings.
        for q in (0..m).filter(|&q| pec[q]) {
            s.d[q] = 1.0;
            s.r[q] = 0.0;
            s.e[q] = 0.0;
            let p = if q > 0 { q - 1 } else { m - 1 };
            if q > 0 || self.cyclic {
                s.e[p] = 0.0;
            }
        }
        if !self.cyclic {
            s.e[m - 1] = 0.0;
        }
        s.solve(self.cyclic);
        true
    }

    /// Writes the solution `x` (masters) and the interpolated slaves of
    /// plane `k`; PEC slaves stay 0.
    fn store<T: Num>(&self, x: &[f64], v: &mut [T], cb: &[f32], nz: usize, k: usize) {
        for (q, js) in self.masters.iter().enumerate() {
            for &j in js {
                v[j * nz + k] = T::from_f64(x[q]);
            }
        }
        for &(j, a, b, t) in &self.slaves {
            let val = if cb[j * nz + k] == 0.0 {
                0.0
            } else {
                (1.0 - t) * x[a] + t * x[b]
            };
            v[j * nz + k] = T::from_f64(val);
        }
    }
}

impl System {
    /// Solves the system into `x`: tridiagonal (Thomas), or cyclic by the
    /// Sherman–Morrison correction (the system is diagonally dominant).
    fn solve(&mut self, cyclic: bool) {
        let m = self.d.len();
        if !cyclic || m < 3 {
            let mut x = core::mem::take(&mut self.x);
            tridiagonal(&self.d, &self.e, &self.r, &mut x, &mut self.c);
            self.x = x;
            return;
        }
        let corner = self.e[m - 1];
        let gamma = -self.d[0];
        let mut d = core::mem::take(&mut self.d);
        d[0] -= gamma;
        d[m - 1] -= corner * corner / gamma;
        let (mut x, mut z) = (core::mem::take(&mut self.x), core::mem::take(&mut self.z));
        tridiagonal(&d, &self.e, &self.r, &mut x, &mut self.c);
        let mut rhs = vec![0.0; m];
        rhs[0] = gamma;
        rhs[m - 1] = corner;
        tridiagonal(&d, &self.e, &rhs, &mut z, &mut self.c);
        let fact = (x[0] + corner * x[m - 1] / gamma) / (1.0 + z[0] + corner * z[m - 1] / gamma);
        for (xi, zi) in x.iter_mut().zip(&z) {
            *xi -= fact * zi;
        }
        self.d = d;
        self.x = x;
        self.z = z;
    }
}

/// Thomas algorithm for the symmetric tridiagonal system with diagonal
/// `d` and off-diagonal `e` (`e[m]` between m and m + 1; `e[last]` is
/// ignored); `c` is scratch.
fn tridiagonal(d: &[f64], e: &[f64], r: &[f64], x: &mut Vec<f64>, c: &mut Vec<f64>) {
    let m = d.len();
    x.clear();
    x.resize(m, 0.0);
    c.clear();
    c.resize(m, 0.0);
    let mut den = d[0];
    c[0] = if m > 1 { e[0] / den } else { 0.0 };
    x[0] = r[0] / den;
    for q in 1..m {
        den = d[q] - e[q - 1] * c[q - 1];
        c[q] = if q + 1 < m { e[q] / den } else { 0.0 };
        x[q] = (r[q] - e[q - 1] * x[q - 1]) / den;
    }
    for q in (0..m - 1).rev() {
        x[q] -= c[q] * x[q + 1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(d: &[f64], e: &[f64], cyclic: bool, x: &[f64]) -> Vec<f64> {
        let m = d.len();
        (0..m)
            .map(|q| {
                let mut s = d[q] * x[q];
                if q + 1 < m || cyclic {
                    s += e[q] * x[(q + 1) % m];
                }
                if q > 0 {
                    s += e[q - 1] * x[q - 1];
                } else if cyclic {
                    s += e[m - 1] * x[m - 1];
                }
                s
            })
            .collect()
    }

    #[test]
    fn solvers() {
        for (m, cyclic) in [(1, false), (2, false), (5, false), (3, true), (6, true)] {
            let d: Vec<f64> = (0..m).map(|q| 3.0 + q as f64 * 0.25).collect();
            let mut e: Vec<f64> = (0..m).map(|q| 0.5 + 0.1 * q as f64).collect();
            if !cyclic {
                e[m - 1] = 0.0;
            }
            let want: Vec<f64> = (0..m).map(|q| (q as f64 * 1.3).sin() + 0.2).collect();
            let r = dense(&d, &e, cyclic, &want);
            let mut s = System {
                d: d.clone(),
                e: e.clone(),
                r,
                ..Default::default()
            };
            s.solve(cyclic);
            for (a, b) in s.x.iter().zip(&want) {
                assert!((a - b).abs() < 1e-12, "m {m} cyclic {cyclic}: {a} vs {b}");
            }
        }
    }
}
