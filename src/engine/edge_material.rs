//! Per-edge / per-face material description fed into the coefficient
//! builder.
//!
//! [`EdgeMaterials`] is the second engine-side input format, next to the
//! per-cell [`CellMaterials`]: it holds the *effective* material of every
//! primary (E) edge and every dual (H) edge directly, which is what a
//! sub-cell rasteriser such as `csx::discretize` produces. The coefficient
//! builder ([`Coefficients::from_edges`](super::Coefficients::from_edges))
//! only consumes this format; [`CellMaterials`] is converted with
//! [`EdgeMaterials::from_cells`].

use super::coeffs::{dual_edge_lr, primary_edge_cg};
use super::grid::Grid;
use super::material::{CellMaterials, Material};
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};
use crate::{Error, Result};

/// Effective materials per edge, plus a PEC mask on primary (E) edges.
///
/// Every array has one entry per **primary node** in field-array layout
/// (index `(i·ny + j)·nz + k`, z contiguous; see the
/// [engine module docs](super)), selected by [`Axis`]:
///
/// | array | entry `[i, j, k]` of axis `a` belongs to |
/// |---|---|
/// | [`eps_r`](Self::eps_r), [`kappa`](Self::kappa), [`pec_mask`](Self::pec_mask) | the E edge along `a` starting at the node (same as `u[a]`) |
/// | [`mu_r`](Self::mu_r), [`sigma_m`](Self::sigma_m) | the H dual edge along `a` through the node, i.e. the primary face normal to `a` whose lowest corner is the node (same as `i[a]`) |
///
/// Entries of non-existing (padding) edges are ignored.
///
/// # Circuit quantities
///
/// With `l` the primary edge length, `A*` the (in-domain part of the) dual
/// face it pierces, `A` the primary face and `l*` the (in-domain part of
/// the) dual edge through it ([`Grid::dual_delta`]):
///
/// ```text
/// C = ε0·eps_r·A*/l     G = kappa·A*/l       (E edge)
/// L = μ0·mu_r·A/l*      R = sigma_m·A/l*     (H dual edge)
/// ```
///
/// So `eps_r`/`kappa` are the averages of the cell values over the dual
/// face (area-weighted arithmetic mean, the parallel combination), and
/// `mu_r` the average over the dual edge (length-weighted *harmonic* mean,
/// the series combination). See [`Coefficients`](super::Coefficients).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EdgeMaterials {
    nodes: [usize; 3],
    eps_r: [Vec<f64>; 3],
    kappa: [Vec<f64>; 3],
    mu_r: [Vec<f64>; 3],
    sigma_m: [Vec<f64>; 3],
    pec: [Vec<bool>; 3],
}

impl EdgeMaterials {
    /// All edges vacuum, no PEC edges.
    pub fn vacuum(grid: &Grid) -> Self {
        Self::new(grid, Material::VACUUM)
    }

    /// All edges filled with the (diagonally anisotropic) `background`: E
    /// edges of axis `a` get `eps_r[a]`, `kappa[a]`, H edges `mu_r[a]`,
    /// `sigma_m[a]`.
    pub fn new(grid: &Grid, background: Material) -> Self {
        let n = grid.num_nodes();
        let fill = |v: [f64; 3]| v.map(|x| vec![x; n]);
        Self {
            nodes: grid.nodes(),
            eps_r: fill(background.eps_r),
            kappa: fill(background.kappa),
            mu_r: fill(background.mu_r),
            sigma_m: fill(background.sigma_m),
            pec: core::array::from_fn(|_| vec![false; n]),
        }
    }

    /// Converts a per-cell map into effective edge values, so that
    /// [`Coefficients::from_edges`](super::Coefficients::from_edges)
    /// reproduces [`primary_edge_cg`] / [`dual_edge_lr`] (up to rounding):
    /// `eps_r = C·l/(ε0·A*)`, `kappa = G·l/A*`, `mu_r = L·l*/(μ0·A)`,
    /// `sigma_m = R·l*/A`. The PEC mask is copied.
    ///
    /// # Errors
    /// [`Error::Invalid`] if the map does not match the grid.
    pub fn from_cells(grid: &Grid, cells: &CellMaterials) -> Result<Self> {
        if cells.cells() != grid.cells() {
            return Err(Error::Invalid(format!(
                "material map has {:?} cells, grid has {:?}",
                cells.cells(),
                grid.cells()
            )));
        }
        if grid.is_cylindrical() {
            return Ok(Self::from_cells_cylindrical(grid, cells, false));
        }
        let mut out = Self::vacuum(grid);
        let n = grid.nodes();
        for axis in Axis::ALL {
            let (ab, ac) = axis.others();
            let (a, b, c) = (axis.index(), ab.index(), ac.index());
            out.pec[a].copy_from_slice(cells.pec_mask(axis));
            for i in 0..n[0] {
                for j in 0..n[1] {
                    for k in 0..n[2] {
                        let p = [i, j, k];
                        let idx = grid.index(p);
                        if p[a] + 1 < n[a] {
                            let (cap, g) = primary_edge_cg(grid, cells, axis, p);
                            let k_ = grid.delta(axis, p[a])
                                / (grid.dual_delta(ab, p[b]) * grid.dual_delta(ac, p[c]));
                            out.eps_r[a][idx] = cap * k_ / EPS0;
                            out.kappa[a][idx] = g * k_;
                        }
                        if p[b] + 1 < n[b] && p[c] + 1 < n[c] {
                            let (l, r) = dual_edge_lr(grid, cells, axis, p);
                            let k_ = grid.dual_delta(axis, p[a])
                                / (grid.delta(ab, p[b]) * grid.delta(ac, p[c]));
                            out.mu_r[a][idx] = l * k_ / MU0;
                            out.sigma_m[a][idx] = r * k_;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// [`from_cells`](Self::from_cells) with openEMS's weights on a
    /// cylindrical grid (models read from openEMS files): the quarter
    /// cells around a z edge on line `ρ_i` weigh `(Δα/2)·|ρ_i − ρ_b|·ρ_b`,
    /// `ρ_b` the dual line bounding the quarter, instead of their annular
    /// areas (black box, design §5.15). The same as
    /// [`from_cells`](Self::from_cells) on Cartesian grids.
    ///
    /// # Errors
    /// As [`from_cells`](Self::from_cells).
    pub fn from_cells_openems(grid: &Grid, cells: &CellMaterials) -> Result<Self> {
        if grid.is_cylindrical() && cells.cells() == grid.cells() {
            return Ok(Self::from_cells_cylindrical(grid, cells, true));
        }
        Self::from_cells(grid, cells)
    }

    /// [`from_cells`](Self::from_cells) on a cylindrical grid: the same
    /// combination rules with the cylindrical weights, i.e. the
    /// quarter-cell areas `(Δα/2)·(ρ_out² − ρ_in²)/2` around a z edge (the
    /// other quarter areas and the half-cell lengths along a dual edge only
    /// enter as ratios, in which the Jacobian `ρ` cancels); `openems`:
    /// see [`from_cells_openems`](Self::from_cells_openems).
    fn from_cells_cylindrical(grid: &Grid, cells: &CellMaterials, openems: bool) -> Self {
        let mut out = Self::vacuum(grid);
        let n = grid.nodes();
        let r = grid.lines(Axis::X);
        for axis in Axis::ALL {
            let (ab, ac) = axis.others();
            let (a, b, c) = (axis.index(), ab.index(), ac.index());
            out.pec[a].copy_from_slice(cells.pec_mask(axis));
            for i in 0..n[0] {
                for j in 0..n[1] {
                    for k in 0..n[2] {
                        let p = [i, j, k];
                        let idx = grid.index(p);
                        if p[a] + 1 < n[a] {
                            let (mut w, mut eps, mut kap) = (0.0, 0.0, 0.0);
                            for cb in [p[b].wrapping_sub(1), p[b]] {
                                if cb >= n[b] - 1 {
                                    continue;
                                }
                                for cc in [p[c].wrapping_sub(1), p[c]] {
                                    if cc >= n[c] - 1 {
                                        continue;
                                    }
                                    let area = if axis == Axis::Z {
                                        // b = ρ, c = α.
                                        let r_b = 0.5 * (r[cb] + r[cb + 1]);
                                        let (r_in, r_out) = if cb < p[0] {
                                            (r_b, r[p[0]])
                                        } else {
                                            (r[p[0]], r_b)
                                        };
                                        let radial = if openems {
                                            (r_out - r_in) * r_b
                                        } else {
                                            0.5 * (r_out * r_out - r_in * r_in)
                                        };
                                        0.5 * grid.delta(ac, cc) * radial
                                    } else {
                                        0.25 * grid.delta(ab, cb) * grid.delta(ac, cc)
                                    };
                                    let mut cell = [0; 3];
                                    cell[a] = p[a];
                                    cell[b] = cb;
                                    cell[c] = cc;
                                    let m = cells.get(cell);
                                    w += area;
                                    eps += area * m.eps_r[a];
                                    kap += area * m.kappa[a];
                                }
                            }
                            if w > 0.0 {
                                out.eps_r[a][idx] = eps / w;
                                out.kappa[a][idx] = kap / w;
                            }
                        }
                        if p[b] + 1 < n[b] && p[c] + 1 < n[c] {
                            // Series over the half cells along the dual edge.
                            let (mut l, mut inv, mut sig) = (0.0, 0.0, 0.0);
                            for ca in [p[a].wrapping_sub(1), p[a]] {
                                if ca >= n[a] - 1 {
                                    continue;
                                }
                                let mut cell = p;
                                cell[a] = ca;
                                let m = cells.get(cell);
                                let half = 0.5 * grid.delta(axis, ca);
                                l += half;
                                inv += half / m.mu_r[a];
                                sig += half * m.sigma_m[a] / (m.mu_r[a] * m.mu_r[a]);
                            }
                            let mu = l / inv;
                            out.mu_r[a][idx] = mu;
                            out.sigma_m[a][idx] = mu * mu * sig / l;
                        }
                    }
                }
            }
        }
        out
    }

    /// Node counts of the grid these values belong to.
    pub fn nodes(&self) -> [usize; 3] {
        self.nodes
    }

    /// Relative permittivity of the E edges along `axis`.
    pub fn eps_r(&self, axis: Axis) -> &[f64] {
        &self.eps_r[axis.index()]
    }
    /// Electric conductivity \[S/m\] of the E edges along `axis`.
    pub fn kappa(&self, axis: Axis) -> &[f64] {
        &self.kappa[axis.index()]
    }
    /// Relative permeability of the H dual edges along `axis`.
    pub fn mu_r(&self, axis: Axis) -> &[f64] {
        &self.mu_r[axis.index()]
    }
    /// Magnetic loss \[Ω/m\] of the H dual edges along `axis`.
    pub fn sigma_m(&self, axis: Axis) -> &[f64] {
        &self.sigma_m[axis.index()]
    }
    /// PEC flags of the E edges along `axis` (padding entries are ignored).
    pub fn pec_mask(&self, axis: Axis) -> &[bool] {
        &self.pec[axis.index()]
    }

    /// Mutable relative permittivity of the E edges along `axis`.
    pub fn eps_r_mut(&mut self, axis: Axis) -> &mut [f64] {
        &mut self.eps_r[axis.index()]
    }
    /// Mutable electric conductivity of the E edges along `axis`.
    pub fn kappa_mut(&mut self, axis: Axis) -> &mut [f64] {
        &mut self.kappa[axis.index()]
    }
    /// Mutable relative permeability of the H dual edges along `axis`.
    pub fn mu_r_mut(&mut self, axis: Axis) -> &mut [f64] {
        &mut self.mu_r[axis.index()]
    }
    /// Mutable magnetic loss of the H dual edges along `axis`.
    pub fn sigma_m_mut(&mut self, axis: Axis) -> &mut [f64] {
        &mut self.sigma_m[axis.index()]
    }
    /// Mutable PEC flags of the E edges along `axis`.
    pub fn pec_mask_mut(&mut self, axis: Axis) -> &mut [bool] {
        &mut self.pec[axis.index()]
    }

    fn flat(&self, node: [usize; 3]) -> usize {
        let n = self.nodes;
        assert!(
            (0..3).all(|a| node[a] < n[a]),
            "node {node:?} out of range {n:?}"
        );
        (node[0] * n[1] + node[1]) * n[2] + node[2]
    }

    /// Sets `eps_r`, `kappa` of the E edge along `axis` at `node`.
    ///
    /// # Panics
    /// If the node is out of range.
    pub fn set_edge(&mut self, axis: Axis, node: [usize; 3], eps_r: f64, kappa: f64) {
        let i = self.flat(node);
        self.eps_r[axis.index()][i] = eps_r;
        self.kappa[axis.index()][i] = kappa;
    }

    /// Sets `mu_r`, `sigma_m` of the H dual edge along `axis` at `node`.
    ///
    /// # Panics
    /// If the node is out of range.
    pub fn set_face(&mut self, axis: Axis, node: [usize; 3], mu_r: f64, sigma_m: f64) {
        let i = self.flat(node);
        self.mu_r[axis.index()][i] = mu_r;
        self.sigma_m[axis.index()][i] = sigma_m;
    }

    /// Marks the E edge along `axis` at `node` as PEC.
    ///
    /// # Panics
    /// If the node is out of range.
    pub fn set_pec_edge(&mut self, axis: Axis, node: [usize; 3]) {
        let i = self.flat(node);
        self.pec[axis.index()][i] = true;
    }

    /// Whether the E edge along `axis` at `node` is marked PEC.
    ///
    /// # Panics
    /// If the node is out of range.
    pub fn is_pec_edge(&self, axis: Axis, node: [usize; 3]) -> bool {
        self.pec[axis.index()][self.flat(node)]
    }

    /// Checks the array sizes against the node counts and every value of
    /// an existing edge: finite, `eps_r`, `mu_r` > 0, `kappa`, `sigma_m` ≥ 0.
    /// Padding entries are not checked.
    pub fn validate(&self) -> Result<()> {
        let n = self.nodes;
        let len = n[0] * n[1] * n[2];
        let arrays = [&self.eps_r, &self.kappa, &self.mu_r, &self.sigma_m];
        if arrays.iter().any(|a| a.iter().any(|v| v.len() != len))
            || self.pec.iter().any(|v| v.len() != len)
        {
            return Err(Error::Invalid(format!(
                "edge material arrays do not match {n:?} nodes"
            )));
        }
        for axis in Axis::ALL {
            let (b, c) = axis.others();
            let (a, b, c) = (axis.index(), b.index(), c.index());
            for i in 0..n[0] {
                for j in 0..n[1] {
                    for k in 0..n[2] {
                        let p = [i, j, k];
                        let idx = (i * n[1] + j) * n[2] + k;
                        if p[a] + 1 < n[a] && !self.pec[a][idx] {
                            let (e, g) = (self.eps_r[a][idx], self.kappa[a][idx]);
                            if !(e.is_finite() && e > 0.0 && g.is_finite() && g >= 0.0) {
                                return Err(Error::Invalid(format!(
                                    "E edge {axis}:{p:?}: invalid eps_r {e} / kappa {g}"
                                )));
                            }
                        }
                        if p[b] + 1 < n[b] && p[c] + 1 < n[c] {
                            let (m, s) = (self.mu_r[a][idx], self.sigma_m[a][idx]);
                            if !(m.is_finite() && m > 0.0 && s.is_finite() && s >= 0.0) {
                                return Err(Error::Invalid(format!(
                                    "H edge {axis}:{p:?}: invalid mu_r {m} / sigma_m {s}"
                                )));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Coefficients, FaceCondition, FaceConditions, TimeStep};
    use crate::fdtd::Face;

    #[test]
    fn cells_and_edges_paths_agree() {
        // Graded grid, inhomogeneous lossy anisotropic fill, a PEC edge,
        // one PMC face so face edges are exercised as well.
        let g = Grid::new([
            vec![0.0, 1.0, 2.5, 3.0, 4.5],
            vec![0.0, 0.5, 1.5, 3.0],
            vec![0.0, 2.0, 3.0, 3.5, 5.0],
        ])
        .unwrap();
        let mut m = CellMaterials::vacuum(&g);
        let mut aniso = Material::VACUUM.with_kappa(0.3).with_sigma_m(40.0);
        aniso.eps_r = [2.0, 3.0, 5.0];
        aniso.mu_r = [1.5, 2.0, 4.0];
        m.fill_box([1..3, 0..2, 1..4], aniso);
        m.set([0, 2, 0], Material::dielectric(7.0).with_mu_r(3.0));
        m.set_pec_edge(Axis::Y, [2, 1, 2]);
        let faces = FaceConditions::default().with(Face::XMin, FaceCondition::Pmc);
        let dt = TimeStep::default().resolve(&g).unwrap();
        let a = Coefficients::compute(&g, &m, &faces, dt).unwrap();
        let e = EdgeMaterials::from_cells(&g, &m).unwrap();
        e.validate().unwrap();
        assert!(e.is_pec_edge(Axis::Y, [2, 1, 2]));
        let b = Coefficients::from_edges(&g, &e, &faces, dt).unwrap();
        for lane in 0..12 {
            for (x, y) in a.data[lane].iter().zip(&b.data[lane]) {
                let tol = 4.0 * f32::EPSILON * x.abs().max(y.abs());
                assert!((x - y).abs() <= tol, "lane {lane}: {x} vs {y}");
            }
        }
        // A homogeneous edge has exactly the cell value.
        assert!((e.eps_r(Axis::Z)[g.index([2, 1, 1])] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn validation_rejects_bad_values() {
        let g = Grid::uniform([2, 2, 2], [1.0; 3]).unwrap();
        let mut e = EdgeMaterials::vacuum(&g);
        assert!(e.validate().is_ok());
        // Padding entries are ignored.
        e.eps_r_mut(Axis::X)[g.index([2, 0, 0])] = -1.0;
        assert!(e.validate().is_ok());
        // PEC edges need no material.
        e.set_edge(Axis::X, [0, 0, 0], f64::NAN, 0.0);
        e.set_pec_edge(Axis::X, [0, 0, 0]);
        assert!(e.validate().is_ok());
        e.set_face(Axis::Z, [1, 1, 2], 0.0, 0.0);
        assert!(e.validate().is_err());
    }
}
