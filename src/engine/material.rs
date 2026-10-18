//! Per-cell material description fed into the coefficient builder.
//!
//! This is the engine-side input format: one (diagonally anisotropic)
//! [`Material`] per primary cell plus a PEC mask on primary edges. The
//! csx → engine compile step rasterises a model into this structure; tests
//! fill it directly.

use core::ops::Range;

use super::grid::Grid;
use crate::{Error, Result};

/// Diagonal-anisotropic linear material.
///
/// Every quantity has one value per axis (`[x, y, z]`). An edge along axis
/// `a` only uses the `a` component, so e.g. x-directed E edges see
/// `eps_r[0]` and `kappa[0]`, x-directed H edges see `mu_r[0]` and
/// `sigma_m[0]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Relative permittivity ε_r (> 0).
    pub eps_r: [f64; 3],
    /// Relative permeability μ_r (> 0).
    pub mu_r: [f64; 3],
    /// Electric conductivity κ \[S/m\] (≥ 0).
    pub kappa: [f64; 3],
    /// Magnetic loss σ_m \[Ω/m\] (≥ 0), the magnetic dual of κ.
    pub sigma_m: [f64; 3],
}

impl Default for Material {
    fn default() -> Self {
        Self::VACUUM
    }
}

impl Material {
    /// Vacuum: ε_r = μ_r = 1, no losses.
    pub const VACUUM: Material = Material {
        eps_r: [1.0; 3],
        mu_r: [1.0; 3],
        kappa: [0.0; 3],
        sigma_m: [0.0; 3],
    };

    /// Lossless isotropic dielectric with relative permittivity `eps_r`.
    pub fn dielectric(eps_r: f64) -> Self {
        Self {
            eps_r: [eps_r; 3],
            ..Self::VACUUM
        }
    }

    /// Sets an isotropic relative permittivity.
    pub fn with_eps_r(mut self, eps_r: f64) -> Self {
        self.eps_r = [eps_r; 3];
        self
    }

    /// Sets an isotropic relative permeability.
    pub fn with_mu_r(mut self, mu_r: f64) -> Self {
        self.mu_r = [mu_r; 3];
        self
    }

    /// Sets an isotropic electric conductivity \[S/m\].
    pub fn with_kappa(mut self, kappa: f64) -> Self {
        self.kappa = [kappa; 3];
        self
    }

    /// Sets an isotropic magnetic loss \[Ω/m\].
    pub fn with_sigma_m(mut self, sigma_m: f64) -> Self {
        self.sigma_m = [sigma_m; 3];
        self
    }

    /// Checks that all values are finite, ε_r, μ_r > 0 and κ, σ_m ≥ 0.
    pub fn validate(&self) -> Result<()> {
        let ok = (0..3).all(|a| {
            self.eps_r[a].is_finite()
                && self.eps_r[a] > 0.0
                && self.mu_r[a].is_finite()
                && self.mu_r[a] > 0.0
                && self.kappa[a].is_finite()
                && self.kappa[a] >= 0.0
                && self.sigma_m[a].is_finite()
                && self.sigma_m[a] >= 0.0
        });
        if ok {
            Ok(())
        } else {
            Err(Error::Invalid(format!("invalid material {self:?}")))
        }
    }
}

/// Material per primary cell plus a PEC mask on primary (E) edges.
///
/// Cells are addressed by `[i, j, k]` with `i < nx − 1` etc.; cell
/// `[i, j, k]` spans `[x_i, x_{i+1}] × [y_j, y_{j+1}] × [z_k, z_{k+1}]`.
///
/// Storage is a material table plus one `u32` table index per cell, so
/// piecewise-constant models cost 4 bytes per cell. Identical materials are
/// de-duplicated on insertion (recent and small tables only; a model with
/// continuously varying materials simply grows the table).
#[derive(Debug, Clone)]
pub struct CellMaterials {
    cells: [usize; 3],
    nodes: [usize; 3],
    table: Vec<Material>,
    index: Vec<u32>,
    pec: [Vec<bool>; 3],
}

/// Table entries searched for an identical material before a new one is
/// appended.
const DEDUP_SEARCH: usize = 64;

impl CellMaterials {
    /// All cells vacuum, no PEC edges.
    pub fn vacuum(grid: &Grid) -> Self {
        Self::new(grid, Material::VACUUM)
    }

    /// All cells filled with `background`, no PEC edges.
    pub fn new(grid: &Grid, background: Material) -> Self {
        let n = grid.num_nodes();
        Self {
            cells: grid.cells(),
            nodes: grid.nodes(),
            table: vec![background],
            index: vec![0; grid.num_cells()],
            pec: [vec![false; n], vec![false; n], vec![false; n]],
        }
    }

    /// Number of cells per axis.
    pub fn cells(&self) -> [usize; 3] {
        self.cells
    }

    fn cell_index(&self, [i, j, k]: [usize; 3]) -> usize {
        let c = self.cells;
        assert!(
            i < c[0] && j < c[1] && k < c[2],
            "cell [{i}, {j}, {k}] out of range {c:?}"
        );
        (i * c[1] + j) * c[2] + k
    }

    fn intern(&mut self, m: Material) -> u32 {
        let len = self.table.len();
        let from = len.saturating_sub(DEDUP_SEARCH);
        if let Some(p) = self.table[from..].iter().rposition(|t| *t == m) {
            return (from + p) as u32;
        }
        if self.table[0] == m {
            return 0;
        }
        assert!(len < u32::MAX as usize, "material table overflow");
        self.table.push(m);
        len as u32
    }

    /// Material of cell `[i, j, k]`.
    ///
    /// # Panics
    /// If the cell is out of range.
    pub fn get(&self, cell: [usize; 3]) -> &Material {
        &self.table[self.index[self.cell_index(cell)] as usize]
    }

    /// Sets the material of one cell.
    ///
    /// # Panics
    /// If the cell is out of range.
    pub fn set(&mut self, cell: [usize; 3], m: Material) {
        let c = self.cell_index(cell);
        let t = self.intern(m);
        self.index[c] = t;
    }

    /// Fills the box of cells `ranges[0] × ranges[1] × ranges[2]` (cell
    /// indices, end exclusive) with `m`.
    ///
    /// # Panics
    /// If a range exceeds the cell count of its axis.
    pub fn fill_box(&mut self, ranges: [Range<usize>; 3], m: Material) {
        for (a, r) in ranges.iter().enumerate() {
            assert!(
                r.end <= self.cells[a],
                "cell range {r:?} exceeds {} cells on axis {a}",
                self.cells[a]
            );
        }
        let t = self.intern(m);
        let c = self.cells;
        for i in ranges[0].clone() {
            for j in ranges[1].clone() {
                let row = (i * c[1] + j) * c[2];
                self.index[row + ranges[2].start..row + ranges[2].end].fill(t);
            }
        }
    }

    /// The distinct materials referenced by the cells (and possibly some
    /// that were overwritten).
    pub fn table(&self) -> &[Material] {
        &self.table
    }

    /// Validates every material in the table.
    pub fn validate(&self) -> Result<()> {
        self.table.iter().try_for_each(Material::validate)
    }

    fn edge_index(&self, axis: usize, [i, j, k]: [usize; 3]) -> usize {
        let n = self.nodes;
        assert!(axis < 3, "axis {axis} out of range");
        assert!(
            i < n[0] && j < n[1] && k < n[2] && [i, j, k][axis] + 1 < n[axis],
            "edge {axis}:[{i}, {j}, {k}] is not a primary edge of a {n:?}-node grid"
        );
        (i * n[1] + j) * n[2] + k
    }

    /// Marks the primary edge along `axis` starting at node `[i, j, k]` as
    /// perfect electric conductor (`ca = cb = 0`).
    ///
    /// # Panics
    /// If the edge does not exist (e.g. an x edge at `i = nx − 1`).
    pub fn set_pec_edge(&mut self, axis: usize, node: [usize; 3]) {
        let e = self.edge_index(axis, node);
        self.pec[axis][e] = true;
    }

    /// Whether the primary edge along `axis` at `node` is marked PEC.
    ///
    /// # Panics
    /// If the edge does not exist.
    pub fn is_pec_edge(&self, axis: usize, node: [usize; 3]) -> bool {
        self.pec[axis][self.edge_index(axis, node)]
    }

    /// The PEC mask of `axis`, one flag per node in field-array order
    /// (padding entries are always `false`).
    pub fn pec_mask(&self, axis: usize) -> &[bool] {
        &self.pec[axis]
    }

    /// Marks every primary edge that lies inside the closed node box
    /// `ranges` (node indices, end exclusive) as PEC. A box that is flat
    /// along one axis (`ranges[a].len() == 1`) marks the in-plane edges of a
    /// zero-thickness sheet.
    ///
    /// # Panics
    /// If a range exceeds the node count of its axis.
    pub fn fill_pec_box(&mut self, ranges: [Range<usize>; 3]) {
        for (a, r) in ranges.iter().enumerate() {
            assert!(
                r.end <= self.nodes[a],
                "node range {r:?} out of range on axis {a}"
            );
        }
        for axis in 0..3 {
            for i in ranges[0].clone() {
                for j in ranges[1].clone() {
                    for k in ranges[2].clone() {
                        let p = [i, j, k];
                        if p[axis] + 1 < ranges[axis].end {
                            self.set_pec_edge(axis, p);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_and_dedup() {
        let g = Grid::uniform([4, 3, 2], [1.0; 3]).unwrap();
        let mut m = CellMaterials::vacuum(&g);
        let d = Material::dielectric(4.0);
        m.fill_box([1..3, 0..3, 1..2], d);
        m.set([0, 0, 0], d);
        assert_eq!(m.table().len(), 2);
        assert_eq!(*m.get([1, 2, 1]), d);
        assert_eq!(*m.get([1, 2, 0]), Material::VACUUM);
        assert_eq!(*m.get([0, 0, 0]), d);
        m.set([0, 0, 0], Material::VACUUM);
        assert_eq!(m.table().len(), 2);
        assert!(m.validate().is_ok());
        m.set([3, 0, 0], Material::dielectric(-1.0));
        assert!(m.validate().is_err());
    }

    #[test]
    fn pec_box() {
        let g = Grid::uniform([4, 4, 4], [1.0; 3]).unwrap();
        let mut m = CellMaterials::vacuum(&g);
        // Sheet in the z = 2 plane covering x in [1, 3], y in [0, 1].
        m.fill_pec_box([1..4, 0..2, 2..3]);
        assert!(m.is_pec_edge(0, [1, 0, 2]));
        assert!(m.is_pec_edge(0, [2, 1, 2]));
        assert!(!m.is_pec_edge(0, [3, 0, 2]));
        assert!(m.is_pec_edge(1, [3, 0, 2]));
        assert!(!m.is_pec_edge(1, [3, 1, 2]));
        assert!(!m.is_pec_edge(2, [1, 0, 2]));
        assert_eq!(m.pec_mask(2).iter().filter(|&&b| b).count(), 0);
    }

    #[test]
    #[should_panic]
    fn pec_padding_edge_rejected() {
        let g = Grid::uniform([2, 2, 2], [1.0; 3]).unwrap();
        CellMaterials::vacuum(&g).set_pec_edge(0, [2, 0, 0]);
    }
}
