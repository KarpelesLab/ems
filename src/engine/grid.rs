//! Rectilinear (tensor-product) FDTD mesh.
//!
//! The primary mesh is given by its node coordinates along each axis. Primary
//! cells span two consecutive lines on every axis; the dual mesh has its nodes
//! at the primary-cell midpoints (Yee 1966; Taflove & Hagness,
//! *Computational Electrodynamics*, 3rd ed., ch. 3 and the non-uniform-grid
//! discussion there). At the outer boundary the dual cells are half cells.
//!
//! A grid is either Cartesian `(x, y, z)` or cylindrical `(ρ, α, z)`
//! ([`Grid::cylindrical`], design §5.15). The cylindrical geometry
//! (edge lengths, face areas) is in [`Grid::edge_length`],
//! [`Grid::dual_edge_length`], [`Grid::dual_face_area`] and
//! [`Grid::face_area`]; the [`Axis`] names `X`, `Y`, `Z` then stand for
//! `ρ`, `α`, `z`.

use std::sync::OnceLock;

use crate::axis::Axis;
use crate::{Error, Result};

/// The coordinate system of a [`Grid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Coordinates {
    /// `(x, y, z)` in metres.
    #[default]
    Cartesian,
    /// `(ρ, α, z)`: radius and height in metres, azimuth in radians
    /// ([`Axis::X`] = ρ, [`Axis::Y`] = α, [`Axis::Z`] = z).
    Cylindrical,
}

/// Relative tolerance on `α_max − α_min = 2π` for a closed azimuth.
const CLOSED_TOL: f64 = 1e-6;

/// A validated rectilinear mesh, Cartesian (metres) or cylindrical
/// (`ρ`, `z` in metres, `α` in radians).
///
/// Per-axis arrays such as [`nodes`](Self::nodes) are indexed by
/// [`Axis::index`] (`[x, y, z]`, or `[ρ, α, z]`). The grid has `n[a]` node
/// lines along axis `a` (at least 2) and therefore `n[a] − 1` cells.
#[derive(Debug, Clone)]
pub struct Grid {
    lines: [Vec<f64>; 3],
    coords: Coordinates,
    /// Cached stability limit of a cylindrical grid (power iteration).
    courant: OnceLock<f64>,
}

impl PartialEq for Grid {
    fn eq(&self, other: &Self) -> bool {
        self.lines == other.lines && self.coords == other.coords
    }
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
        Ok(Self {
            lines,
            coords: Coordinates::Cartesian,
            courant: OnceLock::new(),
        })
    }

    /// Builds a cylindrical grid from the lines `[ρ, α, z]`: `ρ` and `z` in
    /// metres, `α` in radians.
    ///
    /// Besides the checks of [`new`](Self::new), `ρ` must be non-negative
    /// and the azimuth span `α_max − α_min` at most 2π. A first `ρ` line at
    /// 0 is the axis ([`has_axis`](Self::has_axis)); an azimuth span of 2π
    /// (to a relative 1e-6) closes the circle
    /// ([`is_closed`](Self::is_closed)): the first and last `α` lines are
    /// then the same half-plane, and the engine treats the α faces as
    /// periodic.
    ///
    /// ```
    /// use ems::engine::Grid;
    /// use std::f64::consts::TAU;
    ///
    /// let g = Grid::cylindrical([
    ///     vec![0.0, 1e-3, 2e-3],
    ///     (0..=8).map(|j| j as f64 * TAU / 8.0).collect(),
    ///     vec![0.0, 1e-3],
    /// ])?;
    /// assert!(g.has_axis() && g.is_closed());
    /// # Ok::<(), ems::Error>(())
    /// ```
    pub fn cylindrical(lines: [Vec<f64>; 3]) -> Result<Self> {
        let mut g = Self::new(lines)?;
        if g.lines[0][0] < 0.0 {
            return Err(Error::Invalid(format!(
                "cylindrical grid: rho lines must be >= 0, got {}",
                g.lines[0][0]
            )));
        }
        let span = g.alpha_span();
        if span > core::f64::consts::TAU * (1.0 + CLOSED_TOL) {
            return Err(Error::Invalid(format!(
                "cylindrical grid: alpha span {span} rad exceeds 2π"
            )));
        }
        g.coords = Coordinates::Cylindrical;
        Ok(g)
    }

    /// The coordinate system.
    pub fn coordinates(&self) -> Coordinates {
        self.coords
    }

    /// Whether the grid is cylindrical.
    pub fn is_cylindrical(&self) -> bool {
        self.coords == Coordinates::Cylindrical
    }

    /// Whether the grid is cylindrical with its first `ρ` line at 0 (the
    /// z axis lies on the `ρ_min` face, see design §5.15).
    pub fn has_axis(&self) -> bool {
        self.is_cylindrical() && self.lines[0][0] == 0.0
    }

    /// Whether the grid is cylindrical and spans the full circle in `α`.
    pub fn is_closed(&self) -> bool {
        self.is_cylindrical()
            && (self.alpha_span() - core::f64::consts::TAU).abs()
                <= CLOSED_TOL * core::f64::consts::TAU
    }

    fn alpha_span(&self) -> f64 {
        let a = &self.lines[1];
        a[a.len() - 1] - a[0]
    }

    /// The cached cylindrical stability limit, computed by `f` once.
    pub(crate) fn cached_courant(&self, f: impl FnOnce(&Self) -> f64) -> f64 {
        *self.courant.get_or_init(|| f(self))
    }

    /// Radius of the dual line through cell `i` (`ρ_{i+½}`), or of node
    /// `i` itself for the last node (cylindrical helper).
    fn rho_dual(&self, i: usize) -> f64 {
        let r = &self.lines[0];
        if i + 1 < r.len() {
            0.5 * (r[i] + r[i + 1])
        } else {
            r[i]
        }
    }

    /// Physical length \[m\] of the primary edge along `axis` starting at
    /// `node`: [`delta`](Self::delta), or `ρ_i·Δα_j` for an α edge of a
    /// cylindrical grid (0 on the axis).
    pub fn edge_length(&self, axis: Axis, node: [usize; 3]) -> f64 {
        let d = self.delta(axis, node[axis.index()]);
        if self.is_cylindrical() && axis == Axis::Y {
            self.lines[0][node[0]] * d
        } else {
            d
        }
    }

    /// Physical length \[m\] of the (in-domain part of the) dual edge along
    /// `axis` through `node`: [`dual_delta`](Self::dual_delta), or
    /// `ρ_{i+½}·Δα*_j` for an α dual edge of a cylindrical grid.
    pub fn dual_edge_length(&self, axis: Axis, node: [usize; 3]) -> f64 {
        let d = self.dual_delta(axis, node[axis.index()]);
        if self.is_cylindrical() && axis == Axis::Y {
            self.rho_dual(node[0]) * d
        } else {
            d
        }
    }

    /// Area \[m²\] of the (in-domain part of the) dual face pierced by the
    /// primary edge along `axis` starting at `node`: the product of the two
    /// transverse [`dual_delta`](Self::dual_delta)s in Cartesian
    /// coordinates; in cylindrical coordinates `ρ_{i+½}·Δα*·Δz*` (ρ edge),
    /// `Δρ*·Δz*` (α edge) and the annular sector `Δα*·(ρ_hi² − ρ_lo²)/2`
    /// (z edge, `ρ_lo`, `ρ_hi` the dual lines around node `i`, clamped to
    /// the domain).
    pub fn dual_face_area(&self, axis: Axis, node: [usize; 3]) -> f64 {
        let (ab, ac) = axis.others();
        let (db, dc) = (
            self.dual_delta(ab, node[ab.index()]),
            self.dual_delta(ac, node[ac.index()]),
        );
        if !self.is_cylindrical() {
            return db * dc;
        }
        let i = node[0];
        match axis {
            // others() = (Y, Z): Δα*, Δz*.
            Axis::X => self.rho_dual(i) * db * dc,
            // others() = (Z, X): Δz*, Δρ*.
            Axis::Y => db * dc,
            // others() = (X, Y).
            Axis::Z => {
                let r = &self.lines[0];
                let lo = if i > 0 { 0.5 * (r[i - 1] + r[i]) } else { r[i] };
                let hi = self.rho_dual(i);
                dc * 0.5 * (hi * hi - lo * lo)
            }
        }
    }

    /// Area \[m²\] of the primary face normal to `axis` whose lowest corner
    /// is `node`: the product of the two transverse
    /// [`delta`](Self::delta)s in Cartesian coordinates; in cylindrical
    /// coordinates `ρ_i·Δα·Δz` (ρ face, 0 on the axis), `Δρ·Δz` (α face)
    /// and the annular sector `Δα·(ρ_{i+1}² − ρ_i²)/2` (z face).
    pub fn face_area(&self, axis: Axis, node: [usize; 3]) -> f64 {
        let (ab, ac) = axis.others();
        let (db, dc) = (
            self.delta(ab, node[ab.index()]),
            self.delta(ac, node[ac.index()]),
        );
        if !self.is_cylindrical() {
            return db * dc;
        }
        let r = &self.lines[0];
        let i = node[0];
        match axis {
            Axis::X => r[i] * db * dc,
            Axis::Y => db * dc,
            Axis::Z => dc * 0.5 * (r[i + 1] * r[i + 1] - r[i] * r[i]),
        }
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
