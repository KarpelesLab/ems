//! Update coefficients (materials → `ca, cb, da, db`) and the time step.

use super::grid::Grid;
use super::material::CellMaterials;
use crate::constants::{C0, EPS0, MU0};
use crate::{Error, Result};

/// What the coefficient builder does with the tangential E edges lying on
/// one of the six outer faces of the domain.
///
/// Faces are indexed `2·axis + side` with `side = 0` for the low face
/// (`x_min`, …) and `1` for the high face; see [`face_index`].
///
/// Higher-level boundaries map onto this as follows (for the operator
/// authors):
///
/// * **PEC** → [`FaceCondition::Pec`]: tangential `ca = cb = 0`.
/// * **PMC** → [`FaceCondition::Pmc`]: the face edges keep their regular
///   coefficients computed from the half/quarter dual face. The dual-face
///   contour then has its outer side on the face itself, where the
///   out-of-domain H currents read as zero: this is exactly the
///   `n × H = 0` image condition in integral form, so no operator is needed.
/// * **Mur / CPML** → build with [`FaceCondition::Pec`] (the outer face of
///   a CPML is PEC; for Mur the operator overwrites the face values in
///   `post_e`, so the kernel result for those edges is irrelevant) and add
///   the corresponding [`Operator`](super::Operator). An operator may also
///   tweak coefficients through [`Engine::coefficients_mut`](super::Engine::coefficients_mut)
///   before the first step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FaceCondition {
    /// Perfect electric conductor: tangential E forced to zero.
    #[default]
    Pec,
    /// Perfect magnetic conductor: tangential H on the face is zero.
    Pmc,
}

/// Index of the outer face normal to `axis` (`0..3`); `high = false` is the
/// face at the first grid line, `true` the face at the last one.
pub const fn face_index(axis: usize, high: bool) -> usize {
    2 * axis + high as usize
}

/// How the time step is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeStep {
    /// `dt = S · dt_max` with Courant number `S` in `(0, 1]`, see
    /// [`courant_limit`].
    Courant(f64),
    /// A fixed `dt` \[s\]; rejected if it exceeds [`courant_limit`].
    Fixed(f64),
}

impl Default for TimeStep {
    /// `S = 0.99` (design §5.4).
    fn default() -> Self {
        TimeStep::Courant(0.99)
    }
}

impl TimeStep {
    /// Resolves the actual `dt` for `grid`, validating the request.
    pub fn resolve(self, grid: &Grid) -> Result<f64> {
        let max = courant_limit(grid);
        match self {
            TimeStep::Courant(s) if s.is_finite() && s > 0.0 && s <= 1.0 => Ok(s * max),
            TimeStep::Courant(s) => Err(Error::Invalid(format!(
                "Courant factor {s} must be in (0, 1]"
            ))),
            TimeStep::Fixed(dt) if dt.is_finite() && dt > 0.0 && dt <= max => Ok(dt),
            TimeStep::Fixed(dt) => Err(Error::Invalid(format!(
                "time step {dt:e} s must be positive and at most the stability limit {max:e} s"
            ))),
        }
    }
}

/// Stability bound of the Yee scheme on a non-uniform mesh,
/// `dt_max = 1 / (c0·sqrt(Σ_a 1/Δ_a,min²))` (Courant–Friedrichs–Lewy, see
/// Taflove & Hagness ch. 4). Using the smallest cell per axis and the vacuum
/// speed of light is conservative for every material with ε_r, μ_r ≥ 1.
pub fn courant_limit(grid: &Grid) -> f64 {
    let s: f64 = grid.min_deltas().iter().map(|d| 1.0 / (d * d)).sum();
    1.0 / (C0 * s.sqrt())
}

/// Capacitance `C` \[F\] and conductance `G` \[S\] of the primary edge along
/// `axis` starting at node `node`, by area-weighted parallel combination of
/// the up to four cells sharing its dual face (see the module docs). PEC
/// flags and face conditions are not applied here.
///
/// # Panics
/// If the edge does not exist.
pub fn primary_edge_cg(
    grid: &Grid,
    mats: &CellMaterials,
    axis: usize,
    node: [usize; 3],
) -> (f64, f64) {
    let n = grid.nodes();
    assert!(node[axis] + 1 < n[axis], "not a primary edge");
    let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
    let l = grid.delta(axis, node[axis]);
    let (mut cap, mut cond) = (0.0, 0.0);
    for cb in [node[b].wrapping_sub(1), node[b]] {
        if cb >= n[b] - 1 {
            continue;
        }
        for cc in [node[c].wrapping_sub(1), node[c]] {
            if cc >= n[c] - 1 {
                continue;
            }
            let mut cell = [0; 3];
            cell[axis] = node[axis];
            cell[b] = cb;
            cell[c] = cc;
            let area = 0.25 * grid.delta(b, cb) * grid.delta(c, cc);
            let m = mats.get(cell);
            cap += EPS0 * m.eps_r[axis] * area;
            cond += m.kappa[axis] * area;
        }
    }
    (cap / l, cond / l)
}

/// Inductance `L` \[H\] and resistance `R` \[Ω\] of the dual edge along `axis`
/// through node `node` (piercing the primary face spanned at that node), by
/// series combination of its one or two half cells (see the module docs).
///
/// # Panics
/// If the dual edge does not exist (a transverse index at its last node).
pub fn dual_edge_lr(
    grid: &Grid,
    mats: &CellMaterials,
    axis: usize,
    node: [usize; 3],
) -> (f64, f64) {
    let n = grid.nodes();
    let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
    assert!(node[b] + 1 < n[b] && node[c] + 1 < n[c], "not a dual edge");
    let area = grid.delta(b, node[b]) * grid.delta(c, node[c]);
    let (mut inv_l, mut r_over_l2) = (0.0, 0.0);
    for ca in [node[axis].wrapping_sub(1), node[axis]] {
        if ca >= n[axis] - 1 {
            continue;
        }
        let mut cell = node;
        cell[axis] = ca;
        let half = 0.5 * grid.delta(axis, ca);
        let m = mats.get(cell);
        let lc = MU0 * m.mu_r[axis] * area / half;
        let rc = m.sigma_m[axis] * area / half;
        inv_l += 1.0 / lc;
        r_over_l2 += rc / (lc * lc);
    }
    let l = 1.0 / inv_l;
    (l, l * l * r_over_l2)
}

/// Per-component update coefficients, `f32`, in field-array layout (see the
/// [engine module docs](super) for the index convention).
///
/// `ca[a]`, `cb[a]` belong to the E edges of axis `a`, `da[a]`, `db[a]` to the
/// H dual edges of axis `a`. Padding entries and PEC edges are zero, so the
/// kernels keep the corresponding field values at zero.
///
/// # Circuit quantities
///
/// In the integral form (design §5.2) every primary edge carries a
/// capacitance `C = ε·A*/l` and conductance `G = κ·A*/l`, every dual edge an
/// inductance `L = μ·A/l*` and resistance `R = σ_m·A/l*` (`A*`: dual face,
/// `l`: primary edge, `A`: primary face, `l*`: dual edge). This is the
/// contour-path reading of the Yee scheme (Yee 1966; Taflove & Hagness,
/// *Computational Electrodynamics*, 3rd ed., ch. 3, integral-form
/// interpretation and non-uniform grids).
///
/// **Inhomogeneous dual face (E edges).** Tangential E is continuous across
/// the cell interfaces that contain the edge, so the up to four cells
/// around a primary edge act in parallel: `C = Σ ε_c·A*_c / l` and
/// `G = Σ κ_c·A*_c / l`, where `A*_c` is the quarter of the dual face lying
/// in cell `c` (area-weighted averaging, Taflove & Hagness ch. 3 on material
/// interfaces). Cells outside the domain contribute nothing, so boundary
/// edges see half (face) or quarter (domain edge) dual faces.
///
/// **Inhomogeneous dual edge (H edges).** Normal B is continuous across the
/// interface between the two half cells along a dual edge, so the half cells
/// act in series: with `L_c = μ_c·A/l_c` and `R_c = σ_c·A/l_c` for the half
/// cell `c` of length `l_c`, `1/L = Σ 1/L_c` (exact for lossless media). The
/// exact lossy combination `1/(sL+R) = Σ 1/(sL_c+R_c)` is not of `L + R`
/// form; we keep its first-order low-loss expansion, `R = L²·Σ R_c/L_c²`,
/// which reproduces `R` exactly for a homogeneous dual edge and the single
/// half cell at the boundary.
///
/// # Semi-implicit loss
///
/// With the loss term averaged over `n` and `n+1` (Taflove & Hagness ch. 3,
/// lossy-medium update equations):
///
/// ```text
/// ca = (C − G·dt/2)/(C + G·dt/2),  cb = dt/(C + G·dt/2)
/// da = (L − R·dt/2)/(L + R·dt/2),  db = dt/(L + R·dt/2)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Coefficients {
    /// ca (0–2), cb (3–5), da (6–8), db (9–11).
    pub(crate) data: [Vec<f32>; 12],
    nodes: [usize; 3],
}

impl Coefficients {
    /// Builds the coefficients for `grid`, `mats` and time step `dt`, with
    /// the outer faces handled according to `faces` (indexed by
    /// [`face_index`]).
    pub fn compute(
        grid: &Grid,
        mats: &CellMaterials,
        faces: &[FaceCondition; 6],
        dt: f64,
    ) -> Result<Self> {
        if mats.cells() != grid.cells() {
            return Err(Error::Invalid(format!(
                "material map has {:?} cells, grid has {:?}",
                mats.cells(),
                grid.cells()
            )));
        }
        mats.validate()?;
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::Invalid(format!("invalid time step {dt}")));
        }
        let n = grid.nodes();
        let len = grid.num_nodes();
        let mut data: [Vec<f32>; 12] = core::array::from_fn(|_| vec![0.0; len]);
        let co = &mut data;
        for axis in 0..3 {
            let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
            let pec = mats.pec_mask(axis);
            for i in 0..n[0] {
                for j in 0..n[1] {
                    for k in 0..n[2] {
                        let p = [i, j, k];
                        let idx = grid.index(p);
                        // E edge along `axis` starting at p.
                        if p[axis] + 1 < n[axis] {
                            let on_pec_face = [b, c].iter().any(|&t| {
                                (p[t] == 0 && faces[face_index(t, false)] == FaceCondition::Pec)
                                    || (p[t] == n[t] - 1
                                        && faces[face_index(t, true)] == FaceCondition::Pec)
                            });
                            if !(on_pec_face || pec[idx]) {
                                let (cap, g) = primary_edge_cg(grid, mats, axis, p);
                                let den = cap + 0.5 * g * dt;
                                co[axis][idx] = ((cap - 0.5 * g * dt) / den) as f32;
                                co[3 + axis][idx] = (dt / den) as f32;
                            }
                        }
                        // H dual edge along `axis` through p.
                        if p[b] + 1 < n[b] && p[c] + 1 < n[c] {
                            let (l, r) = dual_edge_lr(grid, mats, axis, p);
                            let den = l + 0.5 * r * dt;
                            co[6 + axis][idx] = ((l - 0.5 * r * dt) / den) as f32;
                            co[9 + axis][idx] = (dt / den) as f32;
                        }
                    }
                }
            }
        }
        Ok(Self { data, nodes: n })
    }

    /// Node counts of the grid these coefficients belong to.
    pub fn nodes(&self) -> [usize; 3] {
        self.nodes
    }

    /// `ca` of the E edges along `axis`.
    pub fn ca(&self, axis: usize) -> &[f32] {
        &self.data[axis]
    }
    /// `cb` of the E edges along `axis`.
    pub fn cb(&self, axis: usize) -> &[f32] {
        &self.data[3 + axis]
    }
    /// `da` of the H dual edges along `axis`.
    pub fn da(&self, axis: usize) -> &[f32] {
        &self.data[6 + axis]
    }
    /// `db` of the H dual edges along `axis`.
    pub fn db(&self, axis: usize) -> &[f32] {
        &self.data[9 + axis]
    }

    /// Mutable `ca`, `cb` of the E edges along `axis` (for boundary and
    /// lumped-element operators at setup time).
    pub fn e_mut(&mut self, axis: usize) -> (&mut [f32], &mut [f32]) {
        self.pair_mut(axis)
    }

    /// Mutable `da`, `db` of the H dual edges along `axis`.
    pub fn h_mut(&mut self, axis: usize) -> (&mut [f32], &mut [f32]) {
        self.pair_mut(6 + axis)
    }

    fn pair_mut(&mut self, first: usize) -> (&mut [f32], &mut [f32]) {
        let (lo, hi) = self.data.split_at_mut(first + 3);
        (&mut lo[first], &mut hi[0])
    }

    /// Edge capacitance recovered from `ca`, `cb`:
    /// `C = dt·(1 + ca)/(2·cb)` (0 for PEC/padding edges).
    #[inline]
    pub(crate) fn capacitance(ca: f32, cb: f32, dt: f64) -> f64 {
        if cb == 0.0 {
            0.0
        } else {
            dt * (1.0 + ca as f64) / (2.0 * cb as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::material::Material;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    #[test]
    fn courant() {
        let g = Grid::uniform([4, 4, 4], [1e-3, 2e-3, 1e-3]).unwrap();
        let lim = courant_limit(&g);
        let want = 1.0 / (C0 * (1e6f64 + 0.25e6 + 1e6).sqrt());
        assert!(rel(lim, want) < 1e-14);
        assert!(rel(TimeStep::default().resolve(&g).unwrap(), 0.99 * want) < 1e-14);
        assert_eq!(TimeStep::Fixed(0.5 * lim).resolve(&g).unwrap(), 0.5 * lim);
        assert!(TimeStep::Fixed(lim).resolve(&g).is_ok());
        assert!(TimeStep::Fixed(1.0001 * lim).resolve(&g).is_err());
        assert!(TimeStep::Fixed(-1.0).resolve(&g).is_err());
        assert!(TimeStep::Fixed(f64::NAN).resolve(&g).is_err());
        assert!(TimeStep::Courant(1.01).resolve(&g).is_err());
        assert!(TimeStep::Courant(0.0).resolve(&g).is_err());
        // Graded grid uses the smallest cell.
        let g2 = Grid::new([vec![0.0, 1.0, 1.5], vec![0.0, 1.0], vec![0.0, 1.0]]).unwrap();
        let want = 1.0 / (C0 * (4.0f64 + 1.0 + 1.0).sqrt());
        assert!(rel(courant_limit(&g2), want) < 1e-14);
    }

    #[test]
    fn edge_between_dielectric_and_vacuum() {
        // 2 x 2 x 2 cells, non-uniform in y and z.
        let g = Grid::new([
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 3.0],
            vec![0.0, 2.0, 3.0],
        ])
        .unwrap();
        let mut m = CellMaterials::vacuum(&g);
        // Fill the y < 1 half (cell j = 0) with eps_r = 4, kappa = 2.
        m.fill_box(
            [0..2, 0..1, 0..2],
            Material::dielectric(4.0).with_kappa(2.0),
        );
        // Interior x edge at node (0, 1, 1): dual face y in [0.5, 2], z in [1, 2.5].
        let (c, gg) = primary_edge_cg(&g, &m, 0, [0, 1, 1]);
        // Areas: cells j=0 -> dy/2 = 0.5, j=1 -> 1.0; k=0 -> 1.0, k=1 -> 0.5.
        let a_diel = 0.5 * (1.0 + 0.5);
        let a_vac = 1.0 * (1.0 + 0.5);
        let want_c = EPS0 * (4.0 * a_diel + a_vac) / 1.0;
        assert!(rel(c, want_c) < 1e-14);
        assert!(rel(gg, 2.0 * a_diel) < 1e-14);
        // Effective permittivity is the area-weighted mean.
        let eps_eff = c * 1.0 / (EPS0 * (a_diel + a_vac));
        assert!(rel(eps_eff, (4.0 * a_diel + a_vac) / (a_diel + a_vac)) < 1e-14);
        // Corner edge at the domain edge: a single quarter cell.
        let (c, _) = primary_edge_cg(&g, &m, 0, [1, 0, 0]);
        assert!(rel(c, EPS0 * 4.0 * 0.5 * 1.0) < 1e-14);
    }

    #[test]
    fn dual_edge_series() {
        let g = Grid::new([vec![0.0, 1.0, 3.0], vec![0.0, 1.0], vec![0.0, 2.0]]).unwrap();
        let mut m = CellMaterials::vacuum(&g);
        m.set([1, 0, 0], Material::VACUUM.with_mu_r(3.0).with_sigma_m(5.0));
        // x dual edge through node 1: half cells of 0.5 (vacuum) and 1.0 (mu_r 3).
        let (l, r) = dual_edge_lr(&g, &m, 0, [1, 0, 0]);
        let area = 2.0;
        let l1 = MU0 * area / 0.5;
        let l2 = MU0 * 3.0 * area / 1.0;
        let want = 1.0 / (1.0 / l1 + 1.0 / l2);
        assert!(rel(l, want) < 1e-14);
        assert!(rel(r, want * want * (5.0 * area / 1.0) / (l2 * l2)) < 1e-14);
        // Homogeneous dual edge reproduces L = mu A / l*, R = sigma A / l*.
        let mut m2 = CellMaterials::new(&g, Material::VACUUM.with_mu_r(2.0).with_sigma_m(7.0));
        m2.set([0, 0, 0], *m2.get([1, 0, 0]));
        let (l, r) = dual_edge_lr(&g, &m2, 0, [1, 0, 0]);
        assert!(rel(l, MU0 * 2.0 * area / 1.5) < 1e-14);
        assert!(rel(r, 7.0 * area / 1.5) < 1e-14);
        // Boundary dual edge: single half cell.
        let (l, _) = dual_edge_lr(&g, &m2, 0, [2, 0, 0]);
        assert!(rel(l, MU0 * 2.0 * area / 1.0) < 1e-14);
    }

    #[test]
    fn faces_pec_and_loss() {
        let g = Grid::uniform([3, 3, 3], [1e-3; 3]).unwrap();
        let mut m = CellMaterials::new(&g, Material::VACUUM.with_kappa(1.0));
        m.set_pec_edge(2, [1, 1, 1]);
        let dt = TimeStep::default().resolve(&g).unwrap();
        let mut faces = [FaceCondition::Pec; 6];
        let co = Coefficients::compute(&g, &m, &faces, dt).unwrap();
        // Tangential edge on the x_min face is PEC; interior edge is not.
        assert_eq!(co.cb(1)[g.index([0, 1, 1])], 0.0);
        assert!(co.cb(1)[g.index([1, 1, 1])] > 0.0);
        // PEC mask.
        assert_eq!(co.cb(2)[g.index([1, 1, 1])], 0.0);
        assert_eq!(co.ca(2)[g.index([1, 1, 1])], 0.0);
        // Normal edge touching the face is not tangential.
        assert!(co.cb(0)[g.index([0, 1, 1])] > 0.0);
        // Padding.
        assert_eq!(co.cb(0)[g.index([3, 1, 1])], 0.0);
        assert_eq!(co.db(0)[g.index([1, 3, 1])], 0.0);
        // Loss coefficients.
        let (c, gg) = primary_edge_cg(&g, &m, 0, [1, 1, 1]);
        let idx = g.index([1, 1, 1]);
        let ca = (c - gg * dt / 2.0) / (c + gg * dt / 2.0);
        assert!(rel(co.ca(0)[idx] as f64, ca) < 1e-6);
        assert!(ca < 1.0);
        assert!(
            rel(
                Coefficients::capacitance(co.ca(0)[idx], co.cb(0)[idx], dt),
                c
            ) < 1e-6
        );
        // PMC on x_min keeps the face edges alive (half dual face).
        faces[face_index(0, false)] = FaceCondition::Pmc;
        let co = Coefficients::compute(&g, &m, &faces, dt).unwrap();
        let (c_face, _) = primary_edge_cg(&g, &m, 1, [0, 1, 1]);
        assert!(rel(c_face, 0.5 * c) < 1e-12);
        assert!(co.cb(1)[g.index([0, 1, 1])] > 0.0);
        // But not where the x_min face meets the (PEC) y_min... z_min face.
        assert_eq!(co.cb(1)[g.index([0, 1, 0])], 0.0);
    }
}
