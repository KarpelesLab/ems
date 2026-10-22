//! openEMS's time step on cylindrical grids (models read from openEMS
//! files, design §5.16.1).
//!
//! The rules were **established by black-box comparison** with openEMS
//! v0.37 (`--no-simulation -vvv` prints the step and the limiting edge of
//! every grid; some 560 random meshes, see `tests/openems_cyl.rs` and
//! DESIGN §5.16.1), not taken from its source. They are the two criteria
//! of [`timestep`](super::timestep) evaluated on the physical circuit
//! values of the cylindrical cells, with openEMS's conventions:
//!
//! - **Boundaries.** Every quantity outside the mesh is that of the
//!   nearest index (as on Cartesian grids), but its *geometry* comes from
//!   lines extrapolated with the outermost cell: the cell beyond `ρ_max` is
//!   `[ρ_max, ρ_max + Δρ_last]` (its radius enters), the dual line below
//!   `ρ_min` lies at `ρ_min − Δρ_0/2` (radii below 0 count as 0). For the
//!   Gershgorin criterion (method 1) the dual lengths at a boundary are
//!   the full cell and the faces beyond the boundary are counted (this is
//!   the half-cell rule of the Cartesian criterion when cells are mirror
//!   images).
//! - **z components.** The dual face of an E_z edge at `α_j` is
//!   `Δα_j·(ρ_hi² − ρ_lo²)/2` with the *primary* width `Δα_j = α_{j+1} − α_j`,
//!   and the face of an H_z dual edge in `α` cell `j` takes the *dual*
//!   width `(α_{j+1} − α_{j−1})/2` (openEMS's operator dump shows the same
//!   coefficients; they coincide on uniform `α` meshes).
//! - **Rennings.** The face below along `b` for `(a, b)` against the
//!   cyclic order takes its geometry, not only its materials, one node
//!   lower along `c`, and the term of the edge's own axis is evaluated at
//!   each end of the edge and averaged (see `rennings` below; both are the
//!   Cartesian criterion on Cartesian meshes, where the shift cancels and
//!   both ends pick the same branch). The second was fitted on E_ρ edges
//!   of uniform meshes, whose own-axis term is
//!   `½(ρ_{i+1}/ρ_{i+½} + ρ_{i+½}/ρ_i)/Δρ²` in openEMS (each end takes the
//!   larger of its α and z couplings; a mean of the ends' couplings, then
//!   the larger, gives `½(ρ_{i+½}/ρ_i + ρ_{i+½}/ρ_{i+1})/Δρ²`, up to 2 % off).
//! - **Closed azimuth.** openEMS appends the line `α_1 + 2π` and treats
//!   the result as a sector.
//! - **Axis.** With `ρ_min = 0` openEMS uses the Gershgorin criterion
//!   whatever `TimeStepMethod` says, leaves out the axis z edge (it is
//!   updated outside the operator) and the zero-area H_ρ faces on the axis.
//! - **Multigrid** (`MultiGrid`, [`Grid::multigrid`]): openEMS runs one
//!   grid per zone. A zone boundary on line `I` splits the mesh into an
//!   inner grid with the lines `ρ_0 … ρ_I` and every second `α` line, and
//!   the outer grid, whose edges with `ρ` index `> I` (on the full mesh)
//!   set its step. The step is the smallest of all grids.
//!
//! Agreement with openEMS's printed 6 digits on every measured mesh, both
//! methods (sectors and closed grids, with and without the axis, graded
//! `ρ`, `α`, `z`, multigrid, materials; DESIGN §5.16.1 has the counts).

use super::edge_material::EdgeMaterials;
use super::grid::Grid;
use super::timestep::TimestepMethod;
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};

/// Ghost lines added on each side of every axis.
const GHOST: usize = 3;

/// One of openEMS's (sub-)grids.
struct Sub<'a> {
    m: &'a EdgeMaterials,
    /// Materials of a closed grid's seam at this grid's `α` stride.
    seam: Option<&'a EdgeMaterials>,
    /// Engine node counts (the full grid's).
    eng: [usize; 3],
    closed: bool,
    axis: bool,
    /// Line counts of this grid (closed: with the extra `α_1 + 2π` line).
    n: [usize; 3],
    /// Lines with `GHOST` extrapolated lines on each side.
    x: [Vec<f64>; 3],
    /// Fine `α` lines per line of this grid (multigrid inner grids).
    stride: usize,
    /// Smallest `ρ` index of the edges that count.
    imin: usize,
}

impl<'a> Sub<'a> {
    fn new(
        grid: &Grid,
        m: &'a EdgeMaterials,
        seam: Option<&'a EdgeMaterials>,
        rho_lines: usize,
        stride: usize,
        imin: usize,
    ) -> Self {
        let closed = grid.is_closed();
        let mut lines = [
            grid.lines(Axis::X)[..rho_lines].to_vec(),
            grid.lines(Axis::Y)
                .iter()
                .step_by(stride)
                .copied()
                .collect::<Vec<_>>(),
            grid.lines(Axis::Z).to_vec(),
        ];
        if closed {
            let a = &mut lines[1];
            a.push(a[a.len() - 1] + (a[1] - a[0]));
        }
        let n = core::array::from_fn(|q| lines[q].len());
        let x = lines.map(|l| {
            let (lo, hi) = (l[1] - l[0], l[l.len() - 1] - l[l.len() - 2]);
            let mut v = Vec::with_capacity(l.len() + 2 * GHOST);
            v.extend((1..=GHOST).rev().map(|k| l[0] - k as f64 * lo));
            v.extend_from_slice(&l);
            v.extend((1..=GHOST).map(|k| l[l.len() - 1] + k as f64 * hi));
            v
        });
        Self {
            m,
            seam,
            eng: grid.nodes(),
            closed,
            axis: grid.has_axis(),
            n,
            x,
            stride,
            imin,
        }
    }

    /// Line `i` of axis `a` (`i` may lie up to `GHOST` outside).
    fn line(&self, a: usize, i: isize) -> f64 {
        self.x[a][(i + GHOST as isize) as usize]
    }

    fn p(&self, a: usize, i: isize) -> f64 {
        self.line(a, i + 1) - self.line(a, i)
    }

    fn d(&self, a: usize, i: isize) -> f64 {
        0.5 * (self.line(a, i + 1) - self.line(a, i - 1))
    }

    /// Radius of node `i` (non-negative).
    fn rn(&self, i: isize) -> f64 {
        self.line(0, i).max(0.0)
    }

    /// Radius of the middle of cell `i` (non-negative).
    fn rm(&self, i: isize) -> f64 {
        (0.5 * (self.line(0, i) + self.line(0, i + 1))).max(0.0)
    }

    /// `n` clamped into this grid's lines.
    fn clamp(&self, n: [isize; 3]) -> [isize; 3] {
        core::array::from_fn(|q| n[q].clamp(0, self.n[q] as isize - 1))
    }

    /// The materials and array index of the edge (`along`: its axis) or
    /// dual edge at `n` of this grid. On a closed grid the `α` lines from
    /// `2π` on come from the seam overlay when there is one (openEMS's
    /// literal angles), else from their periodic images.
    fn mat_index(&self, n: [isize; 3], along: [bool; 3]) -> (&EdgeMaterials, usize) {
        let n = self.clamp(n);
        let (i, k) = (n[0] as usize, n[2] as usize);
        let mut j = n[1] as usize * self.stride;
        let mut m = self.m;
        let mut dims = self.eng;
        if self.closed && j >= self.eng[1] - 1 {
            match self.seam {
                Some(s) => {
                    // Seam lines: α_{N−s}, α_N, α_N + (α_s − α_0).
                    m = s;
                    dims = s.nodes();
                    j = (j - (self.eng[1] - 1)) / self.stride + 1;
                }
                None => j -= self.eng[1] - 1,
            }
        }
        let c = [i, j, k];
        let c: [usize; 3] = core::array::from_fn(|d| {
            let max = if along[d] {
                dims[d].saturating_sub(2)
            } else {
                dims[d] - 1
            };
            c[d].min(max)
        });
        (m, (c[0] * dims[1] + c[1]) * dims[2] + c[2])
    }

    /// `1/C` of the E edge along `a` at `n` with the permittivity of the
    /// edge at `mat`.
    fn inv_c(&self, a: usize, n: [isize; 3], mat: [isize; 3]) -> f64 {
        let [i, j, k] = self.clamp(n);
        let (l, area) = match a {
            0 => (self.p(0, i), self.rm(i) * self.d(1, j) * self.d(2, k)),
            1 => (self.rn(i) * self.p(1, j), self.d(0, i) * self.d(2, k)),
            _ => {
                let (lo, hi) = (self.rm(i - 1), self.rm(i));
                (self.p(2, k), self.p(1, j) * 0.5 * (hi * hi - lo * lo))
            }
        };
        if l == 0.0 {
            return 0.0;
        }
        let mut along = [false; 3];
        along[a] = true;
        let (m, q) = self.mat_index(mat, along);
        let eps = m.eps_r(Axis::ALL[a])[q];
        l / (EPS0 * eps * area)
    }

    /// `1/L` of the H dual edge along `a` at `n` with the permeability of
    /// the dual edge at `mat` (0 for a face of zero area, on the axis).
    fn inv_l(&self, a: usize, n: [isize; 3], mat: [isize; 3]) -> f64 {
        let [i, j, k] = self.clamp(n);
        let (l, area) = match a {
            0 => (self.d(0, i), self.rn(i) * self.p(1, j) * self.p(2, k)),
            1 => (self.rm(i) * self.d(1, j), self.p(0, i) * self.p(2, k)),
            _ => {
                let (lo, hi) = (self.rn(i), self.rn(i + 1));
                (self.d(2, k), self.d(1, j) * 0.5 * (hi * hi - lo * lo))
            }
        };
        if area == 0.0 {
            return 0.0;
        }
        let mut across = [true; 3];
        across[a] = false;
        let (m, q) = self.mat_index(mat, across);
        let mu = m.mu_r(Axis::ALL[a])[q];
        l / (MU0 * mu * area)
    }

    fn c(&self, a: usize, n: [isize; 3]) -> f64 {
        self.inv_c(a, n, n)
    }

    fn l(&self, a: usize, n: [isize; 3]) -> f64 {
        self.inv_l(a, n, n)
    }

    fn nodes(&self) -> impl Iterator<Item = [isize; 3]> + '_ {
        let n = self.n;
        (self.imin..n[0]).flat_map(move |i| {
            (0..n[1])
                .flat_map(move |j| (0..n[2]).map(move |k| [i as isize, j as isize, k as isize]))
        })
    }

    /// Largest `ω²` of the Gershgorin criterion.
    fn gershgorin(&self) -> f64 {
        let mut w2max = 0.0f64;
        for a in 0..3 {
            for n in self.nodes() {
                if n[a] as usize + 1 >= self.n[a] || (self.axis && a == 2 && n[0] == 0) {
                    continue;
                }
                let mut s = 0.0;
                for t in [(a + 1) % 3, (a + 2) % 3] {
                    let o = 3 - a - t;
                    for ct in [n[t] - 1, n[t]] {
                        let mut f = n;
                        f[t] = ct;
                        s += self.l(o, f);
                    }
                }
                w2max = w2max.max(self.c(a, n) * s);
            }
        }
        w2max
    }

    /// Largest `ω²` of the Rennings criterion: `T_b + T_c` plus the mean
    /// over the two ends `m` of the edge of
    /// `max(P_c(m) + |D_b(m)|, P_b(m) + |D_c(m)|)`, `P_b(m) = L̄_c·(1/C_b(m))`,
    /// `D_b(m) = ½(1/L_c(n − e_b) − 1/L_c(n))·(1/C_b(m))` (see the module
    /// docs; on Cartesian meshes the same branch wins at both ends and this
    /// is the `max(O_c + |D_b|, O_b + |D_c|)` of
    /// [`timestep`](super::timestep)).
    fn rennings(&self) -> f64 {
        let mut w2max = 0.0f64;
        for a in 0..3 {
            for n in self.nodes() {
                let bs = [(a + 1) % 3, (a + 2) % 3];
                // Per transverse axis b: T_b, the mean and half the
                // difference of the two dual edges along c.
                let mut tld = [(0.0, 0.0, 0.0); 2];
                for (slot, b) in bs.into_iter().enumerate() {
                    let c = 3 - a - b;
                    let lm = self.l(c, shift(n, b, -1));
                    let l0 = self.l(c, n);
                    let t_below = if (b + 3 - a) % 3 == 2 {
                        let f = shift(shift(n, b, -1), c, -1);
                        self.l(c, f) * (self.c(a, f) + self.c(a, shift(n, c, -1)))
                    } else {
                        lm * (self.c(a, shift(n, b, -1)) + self.c(a, n))
                    };
                    let t = 0.25 * (t_below + l0 * (self.c(a, n) + self.c(a, shift(n, b, 1))));
                    tld[slot] = (t, 0.5 * (lm + l0), 0.5 * (lm - l0));
                }
                let [(t0, l0, d0), (t1, l1, d1)] = tld;
                let mut own = 0.0;
                for m in [n, shift(n, a, 1)] {
                    let (c0, c1) = (self.c(bs[0], m), self.c(bs[1], m));
                    own += (l1 * c1 + (d0 * c0).abs()).max(l0 * c0 + (d1 * c1).abs());
                }
                w2max = w2max.max(t0 + t1 + 0.5 * own);
            }
        }
        w2max
    }

    fn omega2(&self, method: TimestepMethod) -> f64 {
        if self.axis || method == TimestepMethod::Cfl {
            self.gershgorin()
        } else {
            self.rennings()
        }
    }
}

fn shift(n: [isize; 3], a: usize, k: isize) -> [isize; 3] {
    let mut m = n;
    m[a] += k;
    m
}

/// Materials of a closed cylindrical grid around its seam as openEMS
/// sees them ([`openems_timestep_cyl_seam`]).
///
/// openEMS evaluates the materials of its lines at `α ≥ 2π` (the `2π` line
/// and the extra line `α_1 + 2π`) at the literal angles: a box over
/// `[0, 2π]` covers only half the dual cells of the `2π` line and none of
/// the extra line (black box, `--debug-material`, design §5.15). Level `ℓ`
/// (`α` stride `s = 2^ℓ`, one per multigrid grid, outermost first) holds
/// the edge materials of the grid with the full `ρ` and `z` lines and the
/// `α` lines of [`alpha_lines`](Self::alpha_lines), averaged as openEMS
/// does at those literal angles.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SeamMaterials {
    /// One per `α` stride 1, 2, 4, ….
    pub levels: Vec<EdgeMaterials>,
}

impl SeamMaterials {
    /// The `α` lines of seam level `level` of the closed `grid`:
    /// `α_N − (α_N − α_{N−s})`, `α_N`, `α_N + (α_s − α_0)` (`α_N = α_0 + 2π`,
    /// `s = 2^level`): the `2π` line between its cells and openEMS's extra
    /// line, which is its last.
    pub fn alpha_lines(grid: &Grid, level: usize) -> [f64; 3] {
        let a = grid.lines(Axis::Y);
        let (n, s) = (a.len() - 1, 1usize << level);
        [a[n - s], a[n], a[n] + (a[s] - a[0])]
    }
}

/// openEMS's time step \[s\] on the cylindrical `grid` (before any
/// time-step factor), the seam of a closed grid taken as periodic; see the
/// [module docs](self).
pub fn openems_timestep_cyl(grid: &Grid, materials: &EdgeMaterials, method: TimestepMethod) -> f64 {
    openems_timestep_cyl_seam(grid, materials, method, None)
}

/// [`openems_timestep_cyl`] with openEMS's literal materials at the seam
/// of a closed grid (models read from openEMS files; levels missing from
/// `seam` are periodic).
pub fn openems_timestep_cyl_seam(
    grid: &Grid,
    materials: &EdgeMaterials,
    method: TimestepMethod,
    seam: Option<&SeamMaterials>,
) -> f64 {
    let level = |l: usize| seam.and_then(|s| s.levels.get(l));
    let mut w2 = 0.0f64;
    let mut lines = grid.nodes()[0];
    let mut stride = 1;
    for (l, &zone) in grid.multigrid().iter().enumerate() {
        w2 = w2.max(Sub::new(grid, materials, level(l), lines, stride, zone + 1).omega2(method));
        lines = zone + 1;
        stride *= 2;
    }
    let l = grid.multigrid().len();
    w2 = w2.max(Sub::new(grid, materials, level(l), lines, stride, 0).omega2(method));
    1.0 / w2.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn mm(v: &[f64]) -> Vec<f64> {
        v.iter().map(|x| x * 1e-3).collect()
    }

    fn dt(rho: &[f64], alpha: Vec<f64>, z: &[f64], method: TimestepMethod) -> f64 {
        let g = Grid::cylindrical([mm(rho), alpha, mm(z)]).unwrap();
        openems_timestep_cyl(&g, &EdgeMaterials::vacuum(&g), method)
    }

    fn uniform(n: usize, d: f64) -> Vec<f64> {
        (0..=n).map(|j| j as f64 * d).collect()
    }

    #[test]
    fn sector_matches_openems() {
        // (rho, dalpha x 4, z) -> openEMS dt for methods 1 and 3.
        let z = [0.0, 1.0, 2.0, 3.0, 4.0];
        let r = [10.0, 11.0, 12.0, 13.0, 14.0];
        let m1 = dt(&r, uniform(6, 0.1), &z, TimestepMethod::Cfl);
        assert!((m1 / 1.64736e-12 - 1.0).abs() < 5e-6, "{m1:e}");
        let m3 = dt(&r, uniform(6, 0.1), &z, TimestepMethod::Rennings);
        assert!((m3 / 1.9175e-12 - 1.0).abs() < 5e-6, "{m3:e}");
        // A thin last cell: the cell beyond rho_max enters with its radius.
        let r = [10.0, 11.0, 12.0, 12.2];
        let m1 = dt(&r, uniform(4, 0.1), &z, TimestepMethod::Cfl);
        assert!((m1 / 4.62555e-13 - 1.0).abs() < 5e-6, "{m1:e}");
        // rho_min below half the first cell: the dual line below is at 0.
        let r = [0.1, 1.1, 2.1, 3.1, 4.1];
        let m1 = dt(&r, uniform(4, 0.1), &z, TimestepMethod::Cfl);
        assert!((m1 / 3.16352e-14 - 1.0).abs() < 5e-6, "{m1:e}");
    }

    /// `(ρ, α, z)` lines and openEMS's step by methods 3 and 1.
    type Case<'a> = (&'a [f64], Vec<f64>, &'a [f64], f64, f64);

    #[test]
    // 0.6366 is a recorded mesh line, not 2/π.
    #[allow(clippy::approx_constant)]
    fn rho_edges_match_openems() {
        // Meshes limited by an E_rho edge (the own-axis term per end).
        let cases: [Case; 3] = [
            (
                &[1.0, 2.2479, 3.4958, 4.7437, 5.9916, 7.2395, 8.4873],
                uniform(3, TAU / 3.0),
                &[0.0, 0.4875, 2.0098],
                1.53561e-12,
                1.06515e-12,
            ),
            (
                &[3.0, 3.73, 4.46, 5.1901, 5.9201],
                uniform(11, TAU / 11.0),
                &[0.0, 0.6366, 1.9249, 3.4407, 4.1733],
                1.53861e-12,
                1.1293e-12,
            ),
            (
                &[3.0, 3.7112, 4.4224, 5.1336],
                uniform(3, 0.6188213548033067),
                &[0.0, 1.3598, 2.4018],
                1.83028e-12,
                1.38116e-12,
            ),
        ];
        for (rho, alpha, z, m3, m1) in cases {
            let d = dt(rho, alpha.clone(), z, TimestepMethod::Rennings);
            assert!((d / m3 - 1.0).abs() < 5e-6, "{rho:?}: {d:e} vs {m3:e}");
            let d = dt(rho, alpha, z, TimestepMethod::Cfl);
            assert!((d / m1 - 1.0).abs() < 5e-6, "{rho:?}: {d:e} vs {m1:e}");
        }
    }

    #[test]
    fn axis_and_multigrid_match_openems() {
        // The adv_cyl_multigrid mesh: 0.5 mm x 15 deg x 1 mm, closed, with
        // the axis; openEMS: 1.54044e-13 s on the single grid (any
        // method), 3.06136e-13 s with MultiGrid="4" (inner grid; the
        // outer one alone would allow 1.05352e-12 s).
        let rho = uniform(20, 0.5);
        let alpha = uniform(24, TAU / 24.0);
        let z = uniform(15, 1.0);
        let g = Grid::cylindrical([mm(&rho), alpha, mm(&z)]).unwrap();
        let m = EdgeMaterials::vacuum(&g);
        for method in [TimestepMethod::Cfl, TimestepMethod::Rennings] {
            let d = openems_timestep_cyl(&g, &m, method);
            assert!((d / 1.54044e-13 - 1.0).abs() < 5e-6, "{d:e}");
        }
        let g = g.with_multigrid(&[8]).unwrap();
        let d = openems_timestep_cyl(&g, &m, TimestepMethod::Rennings);
        assert!((d / 3.06136e-13 - 1.0).abs() < 5e-6, "{d:e}");
        let outer = 1.0
            / Sub::new(&g, &m, None, 21, 1, 9)
                .omega2(TimestepMethod::Rennings)
                .sqrt();
        assert!((outer / 1.05352e-12 - 1.0).abs() < 5e-6, "{outer:e}");
    }
}
