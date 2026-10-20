//! openEMS-compatible time-step criteria (the `TimeStepMethod` of an
//! openEMS `<FDTD>` element).
//!
//! openEMS chooses its time step from the operator with one of two local
//! criteria. Their definitions are not part of the openEMS documentation;
//! the formulas below were **established by black-box comparison** with
//! openEMS v0.37 (the time step and the position of the limiting edge it
//! reports for some 150 random rectilinear meshes, see
//! `tests/openems_core.rs` and DESIGN §5.4), not taken from its source.
//!
//! Both are written with the circuit quantities of the integral-form Yee
//! scheme (design §5.1): an E edge along `a` at node `n` has
//! `1/C = l/(ε0·εr·A*)` (`l` its length, `A*` its dual face) and an H
//! dual edge `1/L = l*/(μ0·μr·A)` (`l*` the dual edge length, `A` the
//! primary face), with the edge materials of [`EdgeMaterials`].
//!
//! # Method 1 ("CFL"): Gershgorin bound
//!
//! For every E edge `e`,
//!
//! ```text
//! ω_e² = (1/C_e) · Σ_{faces f ∋ e} 1/L_f          dt = 1/max_e ω_e
//! ```
//!
//! i.e. `dt = 2/√ρ` with `ρ` the Gershgorin row bound of `C⁻¹·K·L⁻¹·Kᵀ`
//! (each face couples 4 edges, `ρ ≤ 4·max ω²`). This is a rigorous
//! stability bound; on a uniform cubic mesh `dt = Δ/(2c)`, a factor
//! `√3/2` below the Courant limit. Dual lengths at the domain boundary are
//! half cells.
//!
//! # Method 3 (Rennings, the openEMS default)
//!
//! For every E edge along `a` at node `n` and its two transverse axes
//! `b`, `c` (with `n ± e_b` the neighbouring nodes),
//!
//! ```text
//! T_b   = ¼·[ 1/L_c(n−e_b)·(1/C_a(n−e_b) + 1/C_a(n)) + 1/L_c(n)·(1/C_a(n) + 1/C_a(n+e_b)) ]
//! O_b   = ½·(1/L_c(n−e_b) + 1/L_c(n)) · ½·(1/C_b(n) + 1/C_b(n+e_a))
//! O⁰_b  =      1/L_c(n)               · ½·(1/C_b(n) + 1/C_b(n+e_a))
//! ω²    = T_b + T_c + O_b + O_c − O⁰_b'          (b' the lower of b, c)
//! ```
//!
//! and `dt = 1/max ω`. On a uniform mesh this is exactly the Courant
//! limit `1/(c·√(Σ 1/Δ²))`. The criterion is not symmetric: a cell that
//! grows *upwards* (`Δ_{n} > Δ_{n−1}`) lowers the time step, the mirrored
//! mesh does not. openEMS evaluates it on every node index including the
//! last one of each axis, where the missing cells and dual lengths are
//! those of the last cell (and the dual length at the first node is the
//! full first cell, not a half cell); ems does the same. With this rule
//! the formula reproduces openEMS to its printed 6 digits on meshes whose
//! finest cells are not at the upper end of two axes at once; there,
//! openEMS's time step can be up to a few percent smaller (the boundary
//! terms of openEMS are not fully reproduced, see DESIGN §5.4).
//!
//! Materials enter through `εr` of the E edges and `μr` of the H dual
//! edges (positions outside the mesh use the nearest edge's). PEC edges
//! are treated like their material (openEMS may exclude them).

use super::edge_material::EdgeMaterials;
use super::grid::Grid;
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};

/// The openEMS time-step criterion (`TimeStepMethod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TimestepMethod {
    /// Method 1 ("CFL"): Gershgorin bound per E edge.
    Cfl,
    /// Method 3 (Rennings), the openEMS default.
    #[default]
    Rennings,
}

impl TimestepMethod {
    /// The openEMS `TimeStepMethod` id (1 or 3).
    pub fn id(self) -> u32 {
        match self {
            TimestepMethod::Cfl => 1,
            TimestepMethod::Rennings => 3,
        }
    }
}

/// Mesh geometry with openEMS's boundary rules.
struct Geo {
    n: [usize; 3],
    /// Primary deltas per axis.
    p: [Vec<f64>; 3],
    /// Dual lengths per node.
    d: [Vec<f64>; 3],
}

impl Geo {
    fn new(grid: &Grid, full_boundary_duals: bool) -> Self {
        let n = grid.nodes();
        let p: [Vec<f64>; 3] = Axis::ALL.map(|a| {
            (0..n[a.index()].saturating_sub(1))
                .map(|i| grid.delta(a, i))
                .collect()
        });
        let d = core::array::from_fn(|a| {
            let p = &p[a];
            (0..n[a])
                .map(|i| {
                    let lo = i.checked_sub(1).and_then(|j| p.get(j)).copied();
                    let hi = p.get(i).copied();
                    match (lo, hi) {
                        (Some(l), Some(h)) => 0.5 * (l + h),
                        (Some(x), None) | (None, Some(x)) => {
                            if full_boundary_duals {
                                x
                            } else {
                                0.5 * x
                            }
                        }
                        (None, None) => 1.0,
                    }
                })
                .collect()
        });
        Self { n, p, d }
    }

    /// Primary delta of cell `i` on axis `a`, clamped into the mesh.
    fn p(&self, a: usize, i: isize) -> f64 {
        let v = &self.p[a];
        if v.is_empty() {
            return 1.0;
        }
        v[i.clamp(0, v.len() as isize - 1) as usize]
    }

    /// Dual length at node `i` on axis `a`, clamped into the mesh.
    fn d(&self, a: usize, i: isize) -> f64 {
        let v = &self.d[a];
        v[i.clamp(0, v.len() as isize - 1) as usize]
    }

    /// Engine array index of a node clamped into the mesh (`edge_axis`:
    /// also keep the index along that axis on a real edge / face).
    fn idx(&self, n: [isize; 3], edge_axes: [bool; 3]) -> usize {
        let c: [usize; 3] = core::array::from_fn(|d| {
            let max = if edge_axes[d] {
                self.n[d].saturating_sub(2)
            } else {
                self.n[d] - 1
            };
            n[d].clamp(0, max as isize) as usize
        });
        (c[0] * self.n[1] + c[1]) * self.n[2] + c[2]
    }
}

/// `1/C` \[1/F\] of the E edge along `a` at `n` (clamped).
fn inv_c(g: &Geo, m: &EdgeMaterials, a: usize, n: [isize; 3]) -> f64 {
    let (b, c) = ((a + 1) % 3, (a + 2) % 3);
    let mut along = [false; 3];
    along[a] = true;
    let eps = m.eps_r(Axis::ALL[a])[g.idx(n, along)];
    g.p(a, n[a]) / (EPS0 * eps * g.d(b, n[b]) * g.d(c, n[c]))
}

/// `1/L` \[1/H\] of the H dual edge along `a` at `n` (clamped).
fn inv_l(g: &Geo, m: &EdgeMaterials, a: usize, n: [isize; 3]) -> f64 {
    let (b, c) = ((a + 1) % 3, (a + 2) % 3);
    let mut across = [true; 3];
    across[a] = false;
    let mu = m.mu_r(Axis::ALL[a])[g.idx(n, across)];
    g.d(a, n[a]) / (MU0 * mu * g.p(b, n[b]) * g.p(c, n[c]))
}

/// [`inv_c`] and [`inv_l`] of every edge, precomputed: both depend on
/// the node only through its clamped indices, so a lookup clamps the same
/// way and returns the same value bit for bit (the criteria evaluate each
/// edge's value up to 15 times).
struct Tables {
    /// Per axis: table dimensions, then values (index `(i·d1 + j)·d2 + k`).
    ic: [([usize; 3], Vec<f64>); 3],
    il: [([usize; 3], Vec<f64>); 3],
}

impl Tables {
    fn new(g: &Geo, m: &EdgeMaterials) -> Self {
        let dims = |a: usize, edge: bool| -> [usize; 3] {
            core::array::from_fn(|d| {
                let along = d == a;
                // inv_c: along a cells, across nodes; inv_l: the reverse.
                if along == edge {
                    g.n[d].saturating_sub(1).max(1)
                } else {
                    g.n[d]
                }
            })
        };
        let fill = |a: usize, edge: bool| -> ([usize; 3], Vec<f64>) {
            let d = dims(a, edge);
            let mut v = Vec::with_capacity(d[0] * d[1] * d[2]);
            for i in 0..d[0] {
                for j in 0..d[1] {
                    for k in 0..d[2] {
                        let n = [i as isize, j as isize, k as isize];
                        v.push(if edge {
                            inv_c(g, m, a, n)
                        } else {
                            inv_l(g, m, a, n)
                        });
                    }
                }
            }
            (d, v)
        };
        Self {
            ic: core::array::from_fn(|a| fill(a, true)),
            il: core::array::from_fn(|a| fill(a, false)),
        }
    }

    fn get(t: &([usize; 3], Vec<f64>), n: [isize; 3]) -> f64 {
        let (d, v) = t;
        let c: [usize; 3] = core::array::from_fn(|q| n[q].clamp(0, d[q] as isize - 1) as usize);
        v[(c[0] * d[1] + c[1]) * d[2] + c[2]]
    }

    /// [`inv_c`] of the E edge along `a` at `n`.
    fn c(&self, a: usize, n: [isize; 3]) -> f64 {
        Self::get(&self.ic[a], n)
    }

    /// [`inv_l`] of the H dual edge along `a` at `n`.
    fn l(&self, a: usize, n: [isize; 3]) -> f64 {
        Self::get(&self.il[a], n)
    }
}

fn shift(n: [isize; 3], a: usize, k: isize) -> [isize; 3] {
    let mut m = n;
    m[a] += k;
    m
}

/// The openEMS time step \[s\] of `grid` with `materials` by `method`
/// (before any time-step factor); see the [module docs](self).
///
/// Cartesian grids only (a cylindrical grid gets the same formula on its
/// coordinate lines, which is not what openEMS does there).
pub fn openems_timestep(grid: &Grid, materials: &EdgeMaterials, method: TimestepMethod) -> f64 {
    match method {
        TimestepMethod::Cfl => gershgorin(grid, materials),
        TimestepMethod::Rennings => rennings(grid, materials),
    }
}

fn nodes_iter(n: [usize; 3]) -> impl Iterator<Item = [isize; 3]> {
    (0..n[0]).flat_map(move |i| {
        (0..n[1]).flat_map(move |j| (0..n[2]).map(move |k| [i as isize, j as isize, k as isize]))
    })
}

fn gershgorin(grid: &Grid, m: &EdgeMaterials) -> f64 {
    let g = Geo::new(grid, false);
    let tb = Tables::new(&g, m);
    let mut w2max = 0.0f64;
    for a in 0..3 {
        for n in nodes_iter(g.n) {
            if n[a] as usize + 1 >= g.n[a] {
                continue;
            }
            let ic = tb.c(a, n);
            let mut s = 0.0;
            for t in [(a + 1) % 3, (a + 2) % 3] {
                let o = 3 - a - t;
                for ct in [n[t] - 1, n[t]] {
                    if ct < 0 || ct as usize + 1 >= g.n[t] {
                        continue;
                    }
                    // Face normal to o spanning a × t-cell ct.
                    let mut f = n;
                    f[t] = ct;
                    s += tb.l(o, f);
                }
            }
            w2max = w2max.max(ic * s);
        }
    }
    1.0 / w2max.sqrt()
}

fn rennings(grid: &Grid, m: &EdgeMaterials) -> f64 {
    let g = Geo::new(grid, true);
    let tb = Tables::new(&g, m);
    let mut w2max = 0.0f64;
    for a in 0..3 {
        for n in nodes_iter(g.n) {
            let mut w2 = 0.0;
            let mut own0 = None;
            for b in 0..3 {
                if b == a {
                    continue;
                }
                let c = 3 - a - b;
                let lm = tb.l(c, shift(n, b, -1));
                let l0 = tb.l(c, n);
                let t = 0.25
                    * (lm * (tb.c(a, shift(n, b, -1)) + tb.c(a, n))
                        + l0 * (tb.c(a, n) + tb.c(a, shift(n, b, 1))));
                let q = 0.5 * (tb.c(b, n) + tb.c(b, shift(n, a, 1)));
                w2 += t + 0.5 * (lm + l0) * q;
                if own0.is_none() {
                    own0 = Some(l0 * q);
                }
            }
            w2 -= own0.unwrap_or(0.0);
            w2max = w2max.max(w2);
        }
    }
    1.0 / w2max.sqrt()
}

/// openEMS's adjustment of `dt` to a periodic excitation of period `t`
/// \[s\]: the period is split into `⌊t/dt⌋ + 1` steps.
pub fn periodic_timestep(dt: f64, period: f64) -> f64 {
    let n = (period / dt).floor() + 1.0;
    period / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::C0;

    fn lines(cells: &[f64]) -> Vec<f64> {
        let mut v = vec![0.0];
        for c in cells {
            v.push(v.last().unwrap() + c * 1e-3);
        }
        v
    }

    fn dt(x: &[f64], method: TimestepMethod) -> f64 {
        let u = [1.0; 10];
        let g = Grid::new([lines(x), lines(&u), lines(&u)]).unwrap();
        openems_timestep(&g, &EdgeMaterials::vacuum(&g), method)
    }

    #[test]
    fn uniform_mesh() {
        let u = [1.0; 10];
        // Rennings = Courant limit, CFL (Gershgorin) = Δ/(2c).
        let r = dt(&u, TimestepMethod::Rennings);
        assert!((r / (1e-3 / (C0 * 3f64.sqrt())) - 1.0).abs() < 1e-12);
        let c = dt(&u, TimestepMethod::Cfl);
        assert!((c / (1e-3 / (2.0 * C0)) - 1.0).abs() < 1e-12);
        // openEMS prints 1.92583e-12 and 1.66782e-12.
        assert!((r / 1.92583e-12 - 1.0).abs() < 3e-6);
        assert!((c / 1.66782e-12 - 1.0).abs() < 3e-6);
        // ε_r = 4 doubles both (openEMS: 3.85167e-12).
        let g = Grid::new([lines(&u), lines(&u), lines(&u)]).unwrap();
        let m = EdgeMaterials::new(&g, crate::engine::Material::dielectric(4.0));
        let r4 = openems_timestep(&g, &m, TimestepMethod::Rennings);
        assert!((r4 / 3.85167e-12 - 1.0).abs() < 3e-6);
    }

    #[test]
    fn nonuniform_meshes_match_openems() {
        // (x cells, openEMS dt for methods 3 and 1), y = z = 10 × 1 mm.
        let cases: [(&[f64], f64, f64); 5] = [
            (
                &[1., 1., 1., 1., 1., 0.3, 1., 1., 1., 1.],
                1.24936e-12,
                1.13306e-12,
            ),
            (
                &[1., 1., 1., 1., 1., 0.5, 0.5, 1., 1., 1.],
                1.44437e-12,
                1.05482e-12,
            ),
            (
                &[1., 1., 1., 1., 1., 2., 2., 2., 2., 2.],
                1.90608e-12,
                1.66782e-12,
            ),
            (
                &[1., 1., 1., 1., 1., 0.3, 0.3, 1., 1., 1.],
                1.04857e-12,
                6.77754e-13,
            ),
            (&[1., 1., 1., 1., 1., 0.3], 9.21212e-13, 6.77754e-13),
        ];
        for (x, r, c) in cases {
            let got = dt(x, TimestepMethod::Rennings);
            assert!((got / r - 1.0).abs() < 5e-6, "{x:?}: {got:e} vs {r:e}");
            let got = dt(x, TimestepMethod::Cfl);
            assert!((got / c - 1.0).abs() < 5e-6, "{x:?}: {got:e} vs {c:e}");
        }
        // Not symmetric: the mirrored 1|2 mesh has the uniform time step.
        let mirrored = dt(
            &[2., 2., 2., 2., 2., 1., 1., 1., 1., 1.],
            TimestepMethod::Rennings,
        );
        assert!((mirrored / 1.92583e-12 - 1.0).abs() < 3e-6);
    }

    #[test]
    fn periodic_adjustment() {
        // openEMS: 1.92583e-12 → 1.92308e-12 for 1 GHz (520 steps), and
        // an exact 1000 steps becomes 1001.
        assert!((periodic_timestep(1.92583e-12, 1e-9) / 1.92308e-12 - 1.0).abs() < 3e-6);
        assert!((periodic_timestep(1e-12, 1e-9) - 1e-9 / 1001.0).abs() < 1e-24);
    }
}
