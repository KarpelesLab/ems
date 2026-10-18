//! Rectilinear (tensor-product) FDTD mesh.
//!
//! The primary mesh is given by its node coordinates along each axis. Primary
//! cells span two consecutive lines on every axis; the dual mesh has its nodes
//! at the primary-cell midpoints (Yee 1966; Taflove & Hagness,
//! *Computational Electrodynamics*, 3rd ed., ch. 3 and the non-uniform-grid
//! discussion there). At the outer boundary the dual cells are half cells.

use crate::axis::Axis;
use crate::{Error, Result};

/// A validated rectilinear mesh in metres.
///
/// Per-axis arrays such as [`nodes`](Self::nodes) are indexed by
/// [`Axis::index`] (`[x, y, z]`). The grid has `n[a]` node lines along axis
/// `a` (at least 2) and therefore `n[a] − 1` cells.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    lines: [Vec<f64>; 3],
}

impl Grid {
    /// Builds a grid from the node coordinates (metres) of each axis.
    ///
    /// Every axis needs at least two lines; the coordinates must be finite
    /// and strictly increasing. The total node count must fit in `usize`.
    pub fn new(lines: [Vec<f64>; 3]) -> Result<Self> {
        for (a, l) in Axis::ALL.into_iter().zip(&lines) {
            if l.len() < 2 {
                return Err(Error::Invalid(format!(
                    "grid axis {a} needs at least 2 lines, got {}",
                    l.len()
                )));
            }
            if let Some(v) = l.iter().find(|v| !v.is_finite()) {
                return Err(Error::Invalid(format!(
                    "grid axis {a} has a non-finite line {v}"
                )));
            }
            if let Some(w) = l.windows(2).position(|w| w[1] <= w[0]) {
                return Err(Error::Invalid(format!(
                    "grid axis {a} lines are not strictly increasing at index {w} ({} >= {})",
                    l[w],
                    l[w + 1]
                )));
            }
        }
        let n = [lines[0].len(), lines[1].len(), lines[2].len()];
        n[0].checked_mul(n[1])
            .and_then(|v| v.checked_mul(n[2]))
            .ok_or_else(|| Error::Invalid("grid node count overflows usize".into()))?;
        Ok(Self { lines })
    }

    /// Uniform grid with `cells[a]` cells of size `cell_size[a]` (metres)
    /// along each axis, starting at the origin.
    pub fn uniform(cells: [usize; 3], cell_size: [f64; 3]) -> Result<Self> {
        let axis =
            |a: usize| -> Vec<f64> { (0..=cells[a]).map(|i| i as f64 * cell_size[a]).collect() };
        Self::new([axis(0), axis(1), axis(2)])
    }

    /// Node coordinates along `axis`.
    pub fn lines(&self, axis: Axis) -> &[f64] {
        &self.lines[axis.index()]
    }

    /// Number of nodes per axis, `[nx, ny, nz]`.
    pub fn nodes(&self) -> [usize; 3] {
        [
            self.lines[0].len(),
            self.lines[1].len(),
            self.lines[2].len(),
        ]
    }

    /// Total number of nodes `nx·ny·nz` (the length of every field array).
    pub fn num_nodes(&self) -> usize {
        let n = self.nodes();
        n[0] * n[1] * n[2]
    }

    /// Number of primary cells per axis, `[nx − 1, ny − 1, nz − 1]`.
    pub fn cells(&self) -> [usize; 3] {
        let n = self.nodes();
        [n[0] - 1, n[1] - 1, n[2] - 1]
    }

    /// Total number of primary cells.
    pub fn num_cells(&self) -> usize {
        let c = self.cells();
        c[0] * c[1] * c[2]
    }

    /// Linear node index `(i·ny + j)·nz + k` (z contiguous).
    #[inline]
    pub fn index(&self, [i, j, k]: [usize; 3]) -> usize {
        let n = self.nodes();
        (i * n[1] + j) * n[2] + k
    }

    /// Primary edge length `Δ_a(i) = line[i+1] − line[i]` along `axis`,
    /// for `i < n[axis] − 1` (also the size of cell `i` along that axis).
    #[inline]
    pub fn delta(&self, axis: Axis, i: usize) -> f64 {
        let l = self.lines(axis);
        l[i + 1] - l[i]
    }

    /// All primary edge lengths of `axis` (`n[axis] − 1` values).
    pub fn deltas(&self, axis: Axis) -> Vec<f64> {
        self.lines(axis).windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Dual node position along `axis`: midpoint of cell `i`
    /// (`i < n[axis] − 1`).
    pub fn dual_line(&self, axis: Axis, i: usize) -> f64 {
        let l = self.lines(axis);
        0.5 * (l[i] + l[i + 1])
    }

    /// All dual node positions of `axis` (`n[axis] − 1` values).
    pub fn dual_lines(&self, axis: Axis) -> Vec<f64> {
        self.lines(axis)
            .windows(2)
            .map(|w| 0.5 * (w[0] + w[1]))
            .collect()
    }

    /// Dual edge length through primary node `i` along `axis`: from the
    /// midpoint of cell `i − 1` to the midpoint of cell `i`, i.e.
    /// `(Δ(i−1) + Δ(i)) / 2`. At the outer nodes (`i = 0`, `i = n − 1`) the
    /// dual cell is a half cell and the length is `Δ/2` of the single
    /// adjacent cell.
    pub fn dual_delta(&self, axis: Axis, i: usize) -> f64 {
        let n = self.lines(axis).len();
        let lo = if i > 0 { self.delta(axis, i - 1) } else { 0.0 };
        let hi = if i + 1 < n { self.delta(axis, i) } else { 0.0 };
        0.5 * (lo + hi)
    }

    /// Smallest cell size along `axis`.
    pub fn min_delta(&self, axis: Axis) -> f64 {
        self.lines(axis)
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::INFINITY, f64::min)
    }

    /// Smallest cell size per axis.
    pub fn min_deltas(&self) -> [f64; 3] {
        Axis::ALL.map(|a| self.min_delta(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation() {
        let ok = vec![0.0, 1.0];
        assert!(Grid::new([ok.clone(), ok.clone(), ok.clone()]).is_ok());
        assert!(Grid::new([vec![0.0], ok.clone(), ok.clone()]).is_err());
        assert!(Grid::new([ok.clone(), vec![0.0, 0.0], ok.clone()]).is_err());
        assert!(Grid::new([ok.clone(), ok.clone(), vec![1.0, 0.0]]).is_err());
        assert!(Grid::new([ok.clone(), vec![0.0, f64::NAN], ok.clone()]).is_err());
        assert!(Grid::new([ok.clone(), ok.clone(), vec![0.0, f64::INFINITY]]).is_err());
    }

    #[test]
    fn helpers() {
        let g = Grid::new([
            vec![0.0, 1.0, 3.0, 6.0],
            vec![0.0, 2.0],
            vec![-1.0, 0.0, 0.5],
        ])
        .unwrap();
        assert_eq!(g.nodes(), [4, 2, 3]);
        assert_eq!(g.num_nodes(), 24);
        assert_eq!(g.cells(), [3, 1, 2]);
        assert_eq!(g.num_cells(), 6);
        assert_eq!(g.deltas(Axis::X), vec![1.0, 2.0, 3.0]);
        assert_eq!(g.dual_lines(Axis::X), vec![0.5, 2.0, 4.5]);
        assert_eq!(g.dual_delta(Axis::X, 0), 0.5);
        assert_eq!(g.dual_delta(Axis::X, 1), 1.5);
        assert_eq!(g.dual_delta(Axis::X, 3), 1.5);
        assert_eq!(g.min_deltas(), [1.0, 2.0, 0.5]);
        assert_eq!(g.index([1, 1, 2]), (2 + 1) * 3 + 2);
        // Dual edges tile the whole axis.
        let s: f64 = (0..4).map(|i| g.dual_delta(Axis::X, i)).sum();
        assert_eq!(s, 6.0);
    }
}
