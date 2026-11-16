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
///
/// # Storage
///
/// On most models few values are distinct (a handful of materials, the
/// averages on their interfaces), so an array may be held as a table of
/// its distinct values with one small index per node
/// ([`compact`](Self::compact): about 2 bytes per node and array instead
/// of 8). The read accessors return a [`MaterialValues`] view that is indexed like
/// a slice either way; the `_mut` accessors return a slice and expand the
/// array they are called on. The values are the same `f64` bit patterns in
/// both forms, so nothing computed from them changes.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EdgeMaterials {
    nodes: [usize; 3],
    eps_r: [Lane; 3],
    kappa: [Lane; 3],
    mu_r: [Lane; 3],
    sigma_m: [Lane; 3],
    pec: [Vec<bool>; 3],
}

impl PartialEq for EdgeMaterials {
    /// Equal values (whatever the storage of either side).
    fn eq(&self, o: &Self) -> bool {
        let same = |a: &[Lane; 3], b: &[Lane; 3]| {
            a.iter()
                .zip(b)
                .all(|(x, y)| x.len() == y.len() && x.iter().eq(y.iter()))
        };
        self.nodes == o.nodes
            && self.pec == o.pec
            && same(&self.eps_r, &o.eps_r)
            && same(&self.kappa, &o.kappa)
            && same(&self.mu_r, &o.mu_r)
            && same(&self.sigma_m, &o.sigma_m)
    }
}

/// The indices of a pooled [`Lane`].
#[derive(Debug, Clone)]
enum PoolIndex {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

/// One material array: one `f64` per node, or the distinct values and an
/// index per node.
#[derive(Debug, Clone)]
enum Lane {
    Dense(Vec<f64>),
    Pooled { values: Vec<f64>, index: PoolIndex },
}

impl Default for Lane {
    fn default() -> Self {
        Self::Dense(Vec::new())
    }
}

impl Lane {
    fn len(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::Pooled {
                index: PoolIndex::U8(i),
                ..
            } => i.len(),
            Self::Pooled {
                index: PoolIndex::U16(i),
                ..
            } => i.len(),
        }
    }

    #[inline]
    fn at(&self, k: usize) -> &f64 {
        match self {
            Self::Dense(v) => &v[k],
            Self::Pooled {
                values,
                index: PoolIndex::U8(i),
            } => &values[i[k] as usize],
            Self::Pooled {
                values,
                index: PoolIndex::U16(i),
            } => &values[i[k] as usize],
        }
    }

    fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        (0..self.len()).map(|k| *self.at(k))
    }

    /// The values as a slice, expanding a pooled array.
    fn dense_mut(&mut self) -> &mut Vec<f64> {
        if let Self::Pooled { .. } = self {
            *self = Self::Dense(self.iter().collect());
        }
        match self {
            Self::Dense(v) => v,
            Self::Pooled { .. } => unreachable!("expanded above"),
        }
    }

    /// Pools `v` if it has at most 65 536 distinct bit patterns (and is
    /// then smaller that way).
    fn pooled(v: Vec<f64>) -> Self {
        const MOST: usize = 1 << 16;
        let mut values: Vec<f64> = Vec::new();
        let mut ids: std::collections::HashMap<u64, u16> = std::collections::HashMap::new();
        let mut index: Vec<u16> = Vec::with_capacity(v.len());
        // (Runs of one value are the rule: the last one is not looked up.)
        let mut last: Option<(u64, u16)> = None;
        for x in &v {
            let bits = x.to_bits();
            let id = match last {
                Some((b, id)) if b == bits => id,
                _ => match ids.get(&bits) {
                    Some(&id) => id,
                    None => {
                        if values.len() == MOST {
                            return Self::Dense(v);
                        }
                        let id = values.len() as u16;
                        values.push(*x);
                        ids.insert(bits, id);
                        id
                    }
                },
            };
            last = Some((bits, id));
            index.push(id);
        }
        drop(v);
        values.shrink_to_fit();
        let index = if values.len() <= 1 << 8 {
            PoolIndex::U8(index.iter().map(|&i| i as u8).collect())
        } else {
            PoolIndex::U16(index)
        };
        Self::Pooled { values, index }
    }

    fn compact(&mut self) {
        if let Self::Dense(v) = self {
            *self = Self::pooled(std::mem::take(v));
        }
    }
}

/// A read-only material array of [`EdgeMaterials`]: one value per node,
/// indexed like a slice ([storage](EdgeMaterials#storage)).
///
/// ```
/// use ems::engine::{EdgeMaterials, Grid, Material};
/// use ems::Axis;
/// let g = Grid::uniform([2, 2, 2], [1e-3; 3]).unwrap();
/// let m = EdgeMaterials::new(&g, Material::dielectric(4.0));
/// let eps = m.eps_r(Axis::X);
/// assert_eq!((eps.len(), eps[5]), (27, 4.0));
/// assert!(eps.iter().all(|e| e == 4.0));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MaterialValues<'a>(&'a Lane);

impl MaterialValues<'_> {
    /// Number of values (one per node).
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    /// The value at `index`, if in range.
    pub fn get(&self, index: usize) -> Option<f64> {
        (index < self.0.len()).then(|| *self.0.at(index))
    }

    /// The values in order.
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.0.iter()
    }

    /// The values as a vector.
    pub fn to_vec(&self) -> Vec<f64> {
        self.0.iter().collect()
    }
}

impl core::ops::Index<usize> for MaterialValues<'_> {
    type Output = f64;

    #[inline]
    fn index(&self, index: usize) -> &f64 {
        self.0.at(index)
    }
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
        // (One value: a table of one entry.)
        let fill = |v: [f64; 3]| {
            v.map(|x| Lane::Pooled {
                values: vec![x],
                index: PoolIndex::U8(vec![0; n]),
            })
        };
        Self {
            nodes: grid.nodes(),
            eps_r: fill(background.eps_r),
            kappa: fill(background.kappa),
            mu_r: fill(background.mu_r),
            sigma_m: fill(background.sigma_m),
            pec: core::array::from_fn(|_| vec![false; n]),
        }
    }

    /// A map for `grid` whose arrays are still empty; every axis must be
    /// given with [`set_axis`](Self::set_axis) before use. (The compile
    /// step moves its arrays in, instead of copying them over a vacuum
    /// map.)
    pub(crate) fn unfilled(grid: &Grid) -> Self {
        Self {
            nodes: grid.nodes(),
            eps_r: Default::default(),
            kappa: Default::default(),
            mu_r: Default::default(),
            sigma_m: Default::default(),
            pec: Default::default(),
        }
    }

    /// Sets the arrays of the edges along `axis` (one value per node,
    /// engine layout).
    ///
    /// # Panics
    /// If an array does not have one value per node.
    pub(crate) fn set_axis(
        &mut self,
        axis: Axis,
        eps_r: Vec<f64>,
        kappa: Vec<f64>,
        mu_r: Vec<f64>,
        sigma_m: Vec<f64>,
        pec: Vec<bool>,
    ) {
        let n: usize = self.nodes.iter().product();
        assert!(
            [
                eps_r.len(),
                kappa.len(),
                mu_r.len(),
                sigma_m.len(),
                pec.len()
            ] == [n; 5],
            "edge material arrays must have one value per node"
        );
        let a = axis.index();
        self.eps_r[a] = Lane::pooled(eps_r);
        self.kappa[a] = Lane::pooled(kappa);
        self.mu_r[a] = Lane::pooled(mu_r);
        self.sigma_m[a] = Lane::pooled(sigma_m);
        self.pec[a] = pec;
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
                            out.eps_r[a].dense_mut()[idx] = cap * k_ / EPS0;
                            out.kappa[a].dense_mut()[idx] = g * k_;
                        }
                        if p[b] + 1 < n[b] && p[c] + 1 < n[c] {
                            let (l, r) = dual_edge_lr(grid, cells, axis, p);
                            let k_ = grid.dual_delta(axis, p[a])
                                / (grid.delta(ab, p[b]) * grid.delta(ac, p[c]));
                            out.mu_r[a].dense_mut()[idx] = l * k_ / MU0;
                            out.sigma_m[a].dense_mut()[idx] = r * k_;
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
                                out.eps_r[a].dense_mut()[idx] = eps / w;
                                out.kappa[a].dense_mut()[idx] = kap / w;
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
                            out.mu_r[a].dense_mut()[idx] = mu;
                            out.sigma_m[a].dense_mut()[idx] = mu * mu * sig / l;
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
    pub fn eps_r(&self, axis: Axis) -> MaterialValues<'_> {
        MaterialValues(&self.eps_r[axis.index()])
    }
    /// Electric conductivity \[S/m\] of the E edges along `axis`.
    pub fn kappa(&self, axis: Axis) -> MaterialValues<'_> {
        MaterialValues(&self.kappa[axis.index()])
    }
    /// Relative permeability of the H dual edges along `axis`.
    pub fn mu_r(&self, axis: Axis) -> MaterialValues<'_> {
        MaterialValues(&self.mu_r[axis.index()])
    }
    /// Magnetic loss \[Ω/m\] of the H dual edges along `axis`.
    pub fn sigma_m(&self, axis: Axis) -> MaterialValues<'_> {
        MaterialValues(&self.sigma_m[axis.index()])
    }
    /// PEC flags of the E edges along `axis` (padding entries are ignored).
    pub fn pec_mask(&self, axis: Axis) -> &[bool] {
        &self.pec[axis.index()]
    }

    /// Mutable relative permittivity of the E edges along `axis`.
    pub fn eps_r_mut(&mut self, axis: Axis) -> &mut [f64] {
        self.eps_r[axis.index()].dense_mut()
    }
    /// Mutable electric conductivity of the E edges along `axis`.
    pub fn kappa_mut(&mut self, axis: Axis) -> &mut [f64] {
        self.kappa[axis.index()].dense_mut()
    }
    /// Mutable relative permeability of the H dual edges along `axis`.
    pub fn mu_r_mut(&mut self, axis: Axis) -> &mut [f64] {
        self.mu_r[axis.index()].dense_mut()
    }
    /// Mutable magnetic loss of the H dual edges along `axis`.
    pub fn sigma_m_mut(&mut self, axis: Axis) -> &mut [f64] {
        self.sigma_m[axis.index()].dense_mut()
    }
    /// Mutable PEC flags of the E edges along `axis`.
    pub fn pec_mask_mut(&mut self, axis: Axis) -> &mut [bool] {
        &mut self.pec[axis.index()]
    }

    /// Takes the material values of `other` (same nodes), keeping the PEC
    /// mask.
    pub(crate) fn take_values(&mut self, other: Self) {
        assert_eq!(self.nodes, other.nodes, "edge materials of another grid");
        self.eps_r = other.eps_r;
        self.kappa = other.kappa;
        self.mu_r = other.mu_r;
        self.sigma_m = other.sigma_m;
    }

    /// Stores every array with at most 65 536 distinct values as a table
    /// of those values and an index per node ([storage](Self#storage)).
    /// The values do not change. A later `_mut` accessor expands the
    /// array it is called on again.
    pub fn compact(&mut self) {
        for lanes in [
            &mut self.eps_r,
            &mut self.kappa,
            &mut self.mu_r,
            &mut self.sigma_m,
        ] {
            for l in lanes {
                l.compact();
            }
        }
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
        self.eps_r[axis.index()].dense_mut()[i] = eps_r;
        self.kappa[axis.index()].dense_mut()[i] = kappa;
    }

    /// Sets `mu_r`, `sigma_m` of the H dual edge along `axis` at `node`.
    ///
    /// # Panics
    /// If the node is out of range.
    pub fn set_face(&mut self, axis: Axis, node: [usize; 3], mu_r: f64, sigma_m: f64) {
        let i = self.flat(node);
        self.mu_r[axis.index()].dense_mut()[i] = mu_r;
        self.sigma_m[axis.index()].dense_mut()[i] = sigma_m;
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
                            let (e, g) = (*self.eps_r[a].at(idx), *self.kappa[a].at(idx));
                            if !(e.is_finite() && e > 0.0 && g.is_finite() && g >= 0.0) {
                                return Err(Error::Invalid(format!(
                                    "E edge {axis}:{p:?}: invalid eps_r {e} / kappa {g}"
                                )));
                            }
                        }
                        if p[b] + 1 < n[b] && p[c] + 1 < n[c] {
                            let (m, s) = (*self.mu_r[a].at(idx), *self.sigma_m[a].at(idx));
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
            for (x, y) in a.values(lane).iter().zip(b.values(lane).iter()) {
                let tol = 4.0 * f32::EPSILON * x.abs().max(y.abs());
                assert!((x - y).abs() <= tol, "lane {lane}: {x} vs {y}");
            }
        }
        // A homogeneous edge has exactly the cell value.
        assert!((e.eps_r(Axis::Z)[g.index([2, 1, 1])] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn compact_keeps_every_value() {
        let g = Grid::new([
            vec![0.0, 1.0, 2.5, 3.0, 4.5],
            vec![0.0, 0.5, 1.5, 3.0],
            (0..=3300).map(|k| k as f64).collect(),
        ])
        .unwrap();
        let mut dense = EdgeMaterials::new(&g, Material::dielectric(2.0).with_kappa(0.1));
        let n = g.num_nodes();
        // One value, a few values (one byte each), a few hundred (two
        // bytes), more than 65 536 (kept as they are); ±0 and NaN patterns
        // are values of their own.
        for (k, v) in dense.kappa_mut(Axis::Y).iter_mut().enumerate() {
            *v = (k % 7) as f64;
        }
        for (k, v) in dense.mu_r_mut(Axis::Z).iter_mut().enumerate() {
            *v = 1.0 + (k % 300) as f64;
        }
        for (k, v) in dense.sigma_m_mut(Axis::X).iter_mut().enumerate() {
            *v = k as f64;
        }
        assert!(n > 1 << 16);
        dense.eps_r_mut(Axis::Z)[3] = -0.0;
        dense.eps_r_mut(Axis::Z)[4] = 0.0;
        dense.eps_r_mut(Axis::Z)[5] = f64::NAN;
        let mut m = dense.clone();
        m.compact();
        let lanes = |m: &EdgeMaterials| {
            [&m.eps_r, &m.kappa, &m.mu_r, &m.sigma_m].map(|l| {
                l.each_ref().map(|l| match l {
                    Lane::Dense(_) => 8,
                    Lane::Pooled {
                        index: PoolIndex::U8(_),
                        ..
                    } => 1,
                    Lane::Pooled {
                        index: PoolIndex::U16(_),
                        ..
                    } => 2,
                })
            })
        };
        assert_eq!(
            lanes(&m),
            [[1, 1, 1], [1, 1, 1], [1, 1, 2], [8, 1, 1]],
            "bytes per node"
        );
        for axis in Axis::ALL {
            for (a, b) in [
                (m.eps_r(axis), dense.eps_r(axis)),
                (m.kappa(axis), dense.kappa(axis)),
                (m.mu_r(axis), dense.mu_r(axis)),
                (m.sigma_m(axis), dense.sigma_m(axis)),
            ] {
                assert_eq!(a.len(), n);
                assert!((0..n).all(|k| a[k].to_bits() == b[k].to_bits()));
                assert!(
                    a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x.to_bits() == y.to_bits())
                );
                assert_eq!(a.get(n), None);
            }
        }
        // Writing expands the array written to, and only that one.
        m.set_edge(Axis::Y, [1, 1, 1], 9.0, 0.5);
        dense.set_edge(Axis::Y, [1, 1, 1], 9.0, 0.5);
        assert_eq!(lanes(&m)[1], [1, 8, 1]);
        assert_eq!(m.kappa(Axis::Y)[g.index([1, 1, 1])], 0.5);
        // (NaN is not equal to itself: compare without it.)
        m.eps_r_mut(Axis::Z)[5] = 1.0;
        dense.eps_r_mut(Axis::Z)[5] = 1.0;
        assert_eq!(m, dense);
        m.compact();
        assert_eq!(m, dense);
        assert_eq!(lanes(&m)[1], [1, 1, 1]);
        m.set_face(Axis::X, [0, 0, 0], 3.0, 0.0);
        assert_ne!(m, dense);
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
