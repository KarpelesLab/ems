//! openEMS-compatible time-step criteria (the `TimeStepMethod` of an
//! openEMS `<FDTD>` element).
//!
//! openEMS chooses its time step from the operator with one of two local
//! criteria. Their definitions are not part of the openEMS documentation;
//! the formulas below were **established by black-box comparison** with
//! openEMS v0.37 (the time step and the position of the limiting edge it
//! reports for some 1 600 random rectilinear meshes, see
//! `tests/openems_core.rs` and DESIGN §5.4, §5.16.1, §5.16.10), not taken
//! from its source.
//!
//! Both are written with the circuit quantities of the integral-form Yee
//! scheme (design §5.1): an E edge along `a` at node `n` has the
//! capacitance `C = ε0·εr·A*/l` (`l` its length, `A*` its dual face) and an
//! H dual edge the inductance `L = μ0·μr·A/l*` (`l*` the dual edge length,
//! `A` the primary face), with the edge materials of [`EdgeMaterials`].
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
//! `√3/2` below the Courant limit.
//!
//! # Method 3 (Rennings, the openEMS default)
//!
//! For every E edge along `a` at node `n` and each of its two transverse
//! axes `b` (with `c` the third axis and `n ± e_b` the neighbouring
//! nodes),
//!
//! ```text
//! T_b = ¼·[ 1/L'_c(n−e_b)·(1/C'_a(n−e_b) + 1/C'_a(n)) + 1/L_c(n)·(1/C_a(n) + 1/C_a(n+e_b)) ]
//! Q_b = ½·(1/C_b(n) + 1/C_b(n+e_a))
//! O_b = ½·(1/L_c(n−e_b) + 1/L_c(n))·Q_b      D_b = ½·(1/L_c(n−e_b) − 1/L_c(n))·Q_b
//! ω²  = T_b + T_c + max(O_c + |D_b|, O_b + |D_c|)
//! ```
//!
//! and `dt = 1/max ω`. `L'`, `C'` are `L`, `C` of the (dual) edges one
//! node lower along `c` when `(a, b)` runs against the cyclic order
//! (`b = a − 1`: (x, z), (y, x), (z, y)); otherwise `L' = L`, `C' = C`. On
//! a uniform mesh `T_b = c²/Δ_b²` and the `max` term is `c²/Δ_a²`, the
//! exact Courant limit `1/(c·√(Σ 1/Δ²))`. `D_b` measures how the cells
//! along `b` change at the node; the criterion is not symmetric: a cell
//! that grows *upwards* (`Δ_{n} > Δ_{n−1}`) lowers the time step more than
//! the mirrored mesh. openEMS evaluates it on every node index including
//! the last one of each axis, where the missing cells and dual lengths are
//! those of the last cell (and the dual length at the first node is the
//! full first cell, not a half cell); ems does the same.
//!
//! The `|D|` terms and the shifted materials were fitted from black-box
//! runs to openEMS's printed 6 digits (~1 600 random meshes graded along
//! one, two or three axes, cell ratios up to 30; vacuum, grid-aligned and
//! off-grid boxes and layers of ε, μ, κ and anisotropic media; DESIGN
//! §5.16.1): the earlier form `T_b + T_c + O_b + O_c − L_c(n)·Q_b` (lower
//! `b`, unshifted) agrees where the cells grow along the transverse axes
//! and was up to 8 % off elsewhere.
//!
//! # openEMS's arithmetic
//!
//! Both steps are reproduced **bit for bit** (DESIGN §5.16.10; read back
//! from the probe files' 12-digit time column of 3 000-step runs, which
//! pins openEMS's `dt` to a few `f64` ulps). openEMS evaluates them with
//! its own constants ([`OPENEMS_EPS0`], [`OPENEMS_MU0`]) and with the
//! circuit values `C`, `L` rounded to `f32`; each coupling `1/(C·L)` is a
//! single-precision product and quotient, `X(C, L) = f32(1/f32(C·L))`:
//!
//! - Rennings: each bracket of `T_b` (the two couplings of one face) is
//!   summed in `f32`, `T_b` from the two in `f64`; the own-axis term is
//!   evaluated at both ends `m ∈ {n, n + e_a}` of the edge from the
//!   couplings `X(C_b(m), L_c)` (`O_b ± D_b` as `f64` half sums and half
//!   differences) and averaged over the ends; the face below for `(a, b)`
//!   against the cyclic order takes its geometry, not only its materials,
//!   one node lower along `c` (the same circuit values at a node one lower:
//!   the coupling is unchanged in exact arithmetic, the rounding is not);
//!   `ω²`, the maximum and `dt = 1/√ω²` are `f64`.
//! - Gershgorin: the dual lengths at the boundary are full cells and the
//!   faces beyond the boundary count, mirrored (the half-cell rule in
//!   exact arithmetic); `Σ 1/L` is an `f32` sum of `f32(1/L)` in the order
//!   (c upper, c lower, b upper, b lower), `ω² = f32(Σ/C)` and
//!   `dt = 1/√ω²` in `f32`.
//!
//! On 736 random vacuum meshes (graded along up to three axes, uniform
//! cubes of 0.1 mm…1 m) the Rennings step and on 300 the Gershgorin step
//! are openEMS's to the last bit (before: Rennings within 4e-8, e.g.
//! 3.5e-9 below on a uniform 1 mm cube, Gershgorin 1 `f32` step off on 15 %
//! of the meshes).
//!
//! Materials enter through `εr` of the E edges and `μr` of the H dual
//! edges (positions outside the mesh use the nearest edge's). PEC edges
//! count like any other edge, with their material: openEMS gives an edge
//! inside metal the material under the metal (the file path's materials
//! do the same, `compile::openems`), and with those every random case
//! with metal boxes, also sharing faces with higher-priority materials,
//! agrees (DESIGN §5.16.1).

use super::edge_material::EdgeMaterials;
use super::grid::Grid;
use crate::axis::Axis;
use crate::constants::{OPENEMS_EPS0, OPENEMS_MU0};

/// The openEMS time-step criterion (`TimeStepMethod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
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

/// `C` (E edges) and `L` (H dual edges) of every edge in `f32`, with full
/// dual lengths at the boundary. A lookup clamps every index into the
/// mesh: geometry and material of a position outside are those of the
/// nearest edge (openEMS's lines extrapolated with the outermost cell give
/// the same cells).
struct Circuit {
    /// Per axis: table dimensions, then values (index `(i·d1 + j)·d2 + k`).
    c: [([usize; 3], Vec<f32>); 3],
    l: [([usize; 3], Vec<f32>); 3],
}

impl Circuit {
    fn new(grid: &Grid, m: &EdgeMaterials) -> Self {
        let n = grid.nodes();
        let p: [Vec<f64>; 3] = Axis::ALL.map(|a| {
            let k = n[a.index()].saturating_sub(1);
            if k == 0 {
                vec![1.0]
            } else {
                (0..k).map(|i| grid.delta(a, i)).collect()
            }
        });
        // Dual lengths per node: the full cell at either end.
        let d: [Vec<f64>; 3] = core::array::from_fn(|a| {
            let p = &p[a];
            (0..n[a])
                .map(
                    |i| match (i.checked_sub(1).and_then(|j| p.get(j)), p.get(i)) {
                        (Some(&l), Some(&h)) if i + 1 < n[a] => 0.5 * (l + h),
                        (Some(&x), _) | (None, Some(&x)) => x,
                        (None, None) => 1.0,
                    },
                )
                .collect()
        });
        let fill = |a: usize, edge: bool| -> ([usize; 3], Vec<f32>) {
            let dims: [usize; 3] = core::array::from_fn(|q| {
                // C: cells along a, nodes across; L: the reverse.
                if (q == a) == edge {
                    n[q].saturating_sub(1).max(1)
                } else {
                    n[q]
                }
            });
            let (b, c) = ((a + 1) % 3, (a + 2) % 3);
            let ax = Axis::ALL[a];
            let mut v = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    for k in 0..dims[2] {
                        let q = [i, j, k];
                        let idx = (i * n[1] + j) * n[2] + k;
                        v.push(if edge {
                            let geo = d[b][q[b]] * d[c][q[c]] / p[a][q[a]];
                            (OPENEMS_EPS0 * m.eps_r(ax)[idx] * geo) as f32
                        } else {
                            let geo = p[b][q[b]] * p[c][q[c]] / d[a][q[a]];
                            (OPENEMS_MU0 * m.mu_r(ax)[idx] * geo) as f32
                        });
                    }
                }
            }
            (dims, v)
        };
        Self {
            c: core::array::from_fn(|a| fill(a, true)),
            l: core::array::from_fn(|a| fill(a, false)),
        }
    }

    fn get(t: &([usize; 3], Vec<f32>), n: [isize; 3]) -> f32 {
        let (d, v) = t;
        let c: [usize; 3] = core::array::from_fn(|q| n[q].clamp(0, d[q] as isize - 1) as usize);
        v[(c[0] * d[1] + c[1]) * d[2] + c[2]]
    }

    /// `C` of the E edge along `a` at `n`.
    fn c(&self, a: usize, n: [isize; 3]) -> f32 {
        Self::get(&self.c[a], n)
    }

    /// `L` of the H dual edge along `a` at `n`.
    fn l(&self, a: usize, n: [isize; 3]) -> f32 {
        Self::get(&self.l[a], n)
    }
}

/// openEMS's coupling `1/(C·L)` of an edge and a face, in single
/// precision.
fn x(c: f32, l: f32) -> f32 {
    1.0 / (c * l)
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
    let cir = Circuit::new(grid, materials);
    match method {
        TimestepMethod::Cfl => gershgorin(grid.nodes(), &cir),
        TimestepMethod::Rennings => rennings(grid.nodes(), &cir),
    }
}

fn nodes_iter(n: [usize; 3]) -> impl Iterator<Item = [isize; 3]> {
    (0..n[0]).flat_map(move |i| {
        (0..n[1]).flat_map(move |j| (0..n[2]).map(move |k| [i as isize, j as isize, k as isize]))
    })
}

fn gershgorin(n: [usize; 3], cir: &Circuit) -> f64 {
    let mut w2max = 0.0f64;
    for a in 0..3 {
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        for q in nodes_iter(n) {
            if q[a] as usize + 1 >= n[a] {
                continue;
            }
            // The faces normal to b (spanning a × c) and to c (a × b),
            // upper then lower; beyond the boundary the nearest one.
            let mut s = 0.0f32;
            for t in [c, b] {
                let o = 3 - a - t;
                s += 1.0 / cir.l(o, q);
                s += 1.0 / cir.l(o, shift(q, t, -1));
            }
            w2max = w2max.max(f64::from(s) / f64::from(cir.c(a, q)));
        }
    }
    // openEMS's single-precision step (module docs).
    f64::from(1.0 / (w2max as f32).sqrt())
}

fn rennings(n: [usize; 3], cir: &Circuit) -> f64 {
    let mut w2max = 0.0f64;
    for a in 0..3 {
        let bs = [(a + 1) % 3, (a + 2) % 3];
        for q in nodes_iter(n) {
            // Per transverse axis b: T_b and the two dual edges along c.
            let mut tl = [(0.0f64, 0.0f32, 0.0f32); 2];
            for (slot, b) in bs.into_iter().enumerate() {
                let c = 3 - a - b;
                let lm = cir.l(c, shift(q, b, -1));
                let l0 = cir.l(c, q);
                // The face below along b: for (a, b) against the cyclic
                // order the circuit values one node lower along c.
                let below = if (b + 3 - a) % 3 == 2 {
                    let f = shift(shift(q, b, -1), c, -1);
                    let lb = cir.l(c, f);
                    x(cir.c(a, f), lb) + x(cir.c(a, shift(q, c, -1)), lb)
                } else {
                    x(cir.c(a, shift(q, b, -1)), lm) + x(cir.c(a, q), lm)
                };
                let above = x(cir.c(a, q), l0) + x(cir.c(a, shift(q, b, 1)), l0);
                tl[slot] = (0.25 * (f64::from(below) + f64::from(above)), lm, l0);
            }
            let [(t0, lm0, l00), (t1, lm1, l01)] = tl;
            // The own-axis term at both ends of the edge.
            let mut own = 0.0f64;
            for m in [q, shift(q, a, 1)] {
                let (c0, c1) = (cir.c(bs[0], m), cir.c(bs[1], m));
                let (x0m, x00) = (f64::from(x(c0, lm0)), f64::from(x(c0, l00)));
                let (x1m, x10) = (f64::from(x(c1, lm1)), f64::from(x(c1, l01)));
                let (lc0, dc0) = (0.5 * (x0m + x00), 0.5 * (x0m - x00));
                let (lc1, dc1) = (0.5 * (x1m + x10), 0.5 * (x1m - x10));
                own += (lc1 + dc0.abs()).max(lc0 + dc1.abs());
            }
            w2max = w2max.max(t0 + t1 + 0.5 * own);
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
        // Rennings = Courant limit, CFL (Gershgorin) = Δ/(2c), up to
        // openEMS's single-precision circuit values.
        let r = dt(&u, TimestepMethod::Rennings);
        assert!((r / (1e-3 / (C0 * 3f64.sqrt())) - 1.0).abs() < 6e-8);
        let c = dt(&u, TimestepMethod::Cfl);
        assert!((c / (1e-3 / (2.0 * C0)) - 1.0).abs() < 6e-8);
        assert_eq!(c, c as f32 as f64);
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
    fn doubly_graded_meshes_match_openems() {
        let dt3 = |c: [&[f64]; 3]| {
            let g = Grid::new(c.map(lines)).unwrap();
            openems_timestep(&g, &EdgeMaterials::vacuum(&g), TimestepMethod::Rennings)
        };
        // A fine x cell next to cells shrinking along y (the first y cell
        // is larger): |D_y| (the earlier rule gave 1.3361e-12, 4.9 % more).
        let got = dt3([&[1., 0.35, 1.], &[2., 1., 1., 1.], &[1., 1., 1., 1.]]);
        assert!((got / 1.27416e-12 - 1.0).abs() < 5e-6, "{got:e}");
        // Cells shrinking along both transverse axes at once.
        let got = dt3([
            &[1.11, 0.84, 0.65, 0.31, 0.8],
            &[1.96, 0.46, 0.55, 0.83, 1.42],
            &[1.68, 1.6, 1.41, 1.37, 1.47],
        ]);
        assert!((got / 1.00192e-12 - 1.0).abs() < 5e-6, "{got:e}");
    }

    #[test]
    fn layered_media_match_openems() {
        // y edges at (1, 0, 2), one node above an ε / μ layer in z cell
        // 0: the face below along x takes its materials one node lower.
        let g = Grid::new([
            lines(&[1., 2., 1.]),
            lines(&[0.3, 1., 1.]),
            lines(&[1., 1., 2., 1.]),
        ])
        .unwrap();
        let rennings = |layer: crate::engine::Material| {
            let mut cells = crate::engine::CellMaterials::vacuum(&g);
            cells.fill_box([0..3, 0..3, 0..1], layer);
            let m = EdgeMaterials::from_cells(&g, &cells).unwrap();
            openems_timestep(&g, &m, TimestepMethod::Rennings)
        };
        let vacuum = rennings(crate::engine::Material::VACUUM);
        assert!((vacuum / 7.99521e-13 - 1.0).abs() < 5e-6, "{vacuum:e}");
        // openEMS: 8.08869e-13 with ε_r = 4, 8.03134e-13 with μ_r = 4.
        let eps = rennings(crate::engine::Material::dielectric(4.0));
        assert!((eps / 8.08869e-13 - 1.0).abs() < 5e-6, "{eps:e}");
        let mu = rennings(crate::engine::Material::VACUUM.with_mu_r(4.0));
        assert!((mu / 8.03134e-13 - 1.0).abs() < 5e-6, "{mu:e}");
    }

    #[test]
    fn periodic_adjustment() {
        // openEMS: 1.92583e-12 → 1.92308e-12 for 1 GHz (520 steps), and
        // an exact 1000 steps becomes 1001.
        assert!((periodic_timestep(1.92583e-12, 1e-9) / 1.92308e-12 - 1.0).abs() < 3e-6);
        assert!((periodic_timestep(1e-12, 1e-9) - 1e-9 / 1001.0).abs() < 1e-24);
    }

    /// The time step of `mm` lines (as written in a file with
    /// `DeltaUnit` 1e-3), vacuum.
    fn dt_mm(mm: [&[f64]; 3], method: TimestepMethod) -> f64 {
        let g = Grid::new(mm.map(|l| l.iter().map(|v| v * 1e-3).collect())).unwrap();
        openems_timestep(&g, &EdgeMaterials::vacuum(&g), method)
    }

    #[test]
    fn bit_exact_steps() {
        // openEMS's dt read back from the probe time column of 3 000-step
        // runs (the interval that all rows allow; design §5.16.10).
        let u: &[f64] = &[0., 1., 2., 3., 4.];
        let cases: [([&[f64]; 3], TimestepMethod, f64, f64); 5] = [
            // Uniform 1 mm cube: 3.5e-9 below Δ/(c·√3).
            (
                [u, u, u],
                TimestepMethod::Rennings,
                1.925833194728315e-12,
                1.9258331947283165e-12,
            ),
            // One graded x cell.
            (
                [&[0., 1., 2., 3.7, 4.7, 5.7], u, u],
                TimestepMethod::Rennings,
                1.9248549583386012e-12,
                1.9248549583386077e-12,
            ),
            // Graded along all three axes; the limit at the last x node.
            (
                [
                    &[0., 1.39, 3.03, 5.43, 7.43, 8.31],
                    &[0., 0.65, 1.7, 4.1],
                    &[0., 0.67, 1.78, 2.86, 5.17, 6.57],
                ],
                TimestepMethod::Rennings,
                1.40843904836242e-12,
                1.4084390483624214e-12,
            ),
            (
                [
                    &[0., 0.37, 1.8, 2.84],
                    &[0., 2.29, 2.98, 3.38],
                    &[0., 0.6, 1.2, 1.8, 2.4, 3., 3.6],
                ],
                TimestepMethod::Cfl,
                6.406501306017993e-13,
                6.406501306017998e-13,
            ),
            (
                [
                    &[0., 0.54, 1.99, 2.25, 3.66, 6.11],
                    &[0., 1.8, 2.6, 3.64],
                    &[0., 1.02, 1.73, 3.18, 4.54],
                ],
                TimestepMethod::Cfl,
                1.075432603213355e-12,
                1.075432603213357e-12,
            ),
        ];
        for (mm, method, lo, hi) in cases {
            let dt = dt_mm(mm, method);
            assert!(
                lo <= dt && dt <= hi,
                "{mm:?} {method:?}: {dt:e} not in [{lo:e}, {hi:e}]"
            );
        }
    }
}
