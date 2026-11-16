//! Update coefficients (materials → `ca, cb, da, db`) and the time step.

use core::ops::{Index, IndexMut};

use super::edge_material::EdgeMaterials;
use super::grid::Grid;
use super::material::CellMaterials;
use crate::axis::Axis;
use crate::constants::{C0, EPS0, MU0};
use crate::fdtd::Face;
use crate::{Error, Result};

/// What the coefficient builder does with the tangential E edges lying on
/// one of the six outer faces of the domain; one per [`Face`] in
/// [`FaceConditions`].
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
/// * **Periodic** → [`FaceCondition::Periodic`] on both faces of an axis:
///   coefficients as for PMC (each face plane is one half of the periodic
///   plane); the engine merges the two copies after every E update (see
///   [`boundary::periodic`](super::boundary::periodic)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum FaceCondition {
    /// Perfect electric conductor: tangential E forced to zero.
    #[default]
    Pec,
    /// Perfect magnetic conductor: tangential H on the face is zero.
    Pmc,
    /// Periodic: the face is identified with the opposite face, which must
    /// be periodic too ([`EngineBuilder::build`](super::EngineBuilder::build)
    /// fails otherwise).
    Periodic,
}

/// The [`FaceCondition`] of each of the six outer faces, indexed by
/// [`Face`] (default: all PEC).
///
/// ```
/// use ems::engine::{FaceCondition, FaceConditions};
/// use ems::Face;
///
/// let mut f = FaceConditions::default().with(Face::ZMin, FaceCondition::Pmc);
/// f[Face::XMax] = FaceCondition::Pmc;
/// assert_eq!(f[Face::ZMin], FaceCondition::Pmc);
/// assert_eq!(f[Face::YMin], FaceCondition::Pec);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct FaceConditions([FaceCondition; 6]);

impl FaceConditions {
    /// The same condition `c` on all six faces.
    pub const fn all(c: FaceCondition) -> Self {
        Self([c; 6])
    }

    /// From one condition per face in [`Face::ALL`] order
    /// (x_min, x_max, y_min, y_max, z_min, z_max).
    pub const fn from_array(faces: [FaceCondition; 6]) -> Self {
        Self(faces)
    }

    /// The conditions in [`Face::ALL`] order.
    pub const fn to_array(self) -> [FaceCondition; 6] {
        self.0
    }

    /// Returns `self` with the condition of `face` set to `c`.
    pub const fn with(mut self, face: Face, c: FaceCondition) -> Self {
        self.0[face.index()] = c;
        self
    }

    /// Sets the condition of `face`.
    pub fn set(&mut self, face: Face, c: FaceCondition) {
        self.0[face.index()] = c;
    }

    /// The condition of `face`.
    pub const fn get(&self, face: Face) -> FaceCondition {
        self.0[face.index()]
    }

    /// Iterates over `(face, condition)` pairs in [`Face::ALL`] order.
    pub fn iter(&self) -> impl Iterator<Item = (Face, FaceCondition)> + '_ {
        Face::ALL.into_iter().zip(self.0.iter().copied())
    }
}

impl From<[FaceCondition; 6]> for FaceConditions {
    /// See [`FaceConditions::from_array`].
    fn from(faces: [FaceCondition; 6]) -> Self {
        Self(faces)
    }
}

impl Index<Face> for FaceConditions {
    type Output = FaceCondition;
    fn index(&self, face: Face) -> &FaceCondition {
        &self.0[face.index()]
    }
}

impl IndexMut<Face> for FaceConditions {
    fn index_mut(&mut self, face: Face) -> &mut FaceCondition {
        &mut self.0[face.index()]
    }
}

/// How the time step is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum TimeStep {
    /// `dt = S · dt_max` with Courant number `S` in `(0, 1]`, see
    /// [`courant_limit`].
    Courant(f64),
    /// A fixed `dt` \[s\]; rejected if it exceeds [`courant_limit`].
    Fixed(f64),
    /// openEMS's time step: its `TimeStepMethod` criterion on the actual
    /// operator (materials included), times a factor, optionally shortened
    /// so that a period fits an integer number of steps (see
    /// [`OpenEmsTimestep`]). Not checked against [`courant_limit`], which
    /// is conservative on non-uniform meshes and in dielectrics.
    OpenEms(OpenEmsTimestep),
}

/// openEMS time-step settings ([`TimeStep::OpenEms`]).
///
/// `dt = factor · dt_method` ([`openems_timestep`](super::timestep::openems_timestep)),
/// or `factor · fixed` when an explicit time step is given (openEMS
/// applies its `TimeStepFactor` to an explicit `TimeStep` too); then, for
/// a periodic excitation of period `T`, `dt = T/(⌊T/dt⌋ + 1)`
/// ([`periodic_timestep`](super::timestep::periodic_timestep)).
///
/// On a cylindrical grid the criteria take openEMS's cylindrical form
/// ([`openems_timestep_cyl`](super::timestep_cyl::openems_timestep_cyl):
/// physical cell geometry, Gershgorin with the axis, one grid per
/// multigrid zone).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct OpenEmsTimestep {
    /// The criterion.
    pub method: super::timestep::TimestepMethod,
    /// Factor in `(0, 1]`.
    pub factor: f64,
    /// Explicit time step \[s\] instead of the criterion.
    pub fixed: Option<f64>,
    /// Period \[s\] of a periodic (sinusoidal) excitation.
    pub period: Option<f64>,
}

impl OpenEmsTimestep {
    /// `dt` for `grid` and `materials`.
    ///
    /// # Errors
    /// [`Error::Invalid`] for a factor outside `(0, 1]` or a non-positive
    /// explicit time step or period.
    pub fn resolve(&self, grid: &Grid, materials: &EdgeMaterials) -> Result<f64> {
        let f = self.factor;
        if !(f.is_finite() && f > 0.0 && f <= 1.0) {
            return Err(Error::Invalid(format!(
                "time step factor {f} must be in (0, 1]"
            )));
        }
        let base = match self.fixed {
            Some(dt) if dt.is_finite() && dt > 0.0 => dt,
            Some(dt) => {
                return Err(Error::Invalid(format!(
                    "time step {dt:e} s must be positive and finite"
                )));
            }
            None if grid.is_cylindrical() => {
                super::timestep_cyl::openems_timestep_cyl(grid, materials, self.method)
            }
            None => super::timestep::openems_timestep(grid, materials, self.method),
        };
        let mut dt = f * base;
        if let Some(t) = self.period {
            if !(t.is_finite() && t > 0.0) {
                return Err(Error::Invalid(format!(
                    "excitation period {t:e} s must be > 0"
                )));
            }
            dt = super::timestep::periodic_timestep(dt, t);
        }
        Ok(dt)
    }
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
            // Without materials: vacuum.
            TimeStep::OpenEms(o) => o.resolve(grid, &EdgeMaterials::vacuum(grid)),
        }
    }
}

/// Stability bound of the Yee scheme on a non-uniform mesh,
/// `dt_max = 1 / (c0·sqrt(Σ_a 1/Δ_a,min²))` (Courant–Friedrichs–Lewy, see
/// Taflove & Hagness ch. 4). Using the smallest cell per axis and the vacuum
/// speed of light is conservative for every material with ε_r, μ_r ≥ 1.
///
/// For a cylindrical grid ([`Grid::cylindrical`]) the limit is computed
/// from the discrete operator itself (power iteration, see
/// [`cylindrical`](super::cylindrical)), once per grid.
pub fn courant_limit(grid: &Grid) -> f64 {
    if grid.is_cylindrical() {
        return grid.cached_courant(super::cylindrical::courant_limit);
    }
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
    axis: Axis,
    node: [usize; 3],
) -> (f64, f64) {
    let n = grid.nodes();
    let (ab, ac) = axis.others();
    let (a, b, c) = (axis.index(), ab.index(), ac.index());
    assert!(node[a] + 1 < n[a], "not a primary edge");
    let l = grid.delta(axis, node[a]);
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
            cell[a] = node[a];
            cell[b] = cb;
            cell[c] = cc;
            let area = 0.25 * grid.delta(ab, cb) * grid.delta(ac, cc);
            let m = mats.get(cell);
            cap += EPS0 * m.eps_r[a] * area;
            cond += m.kappa[a] * area;
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
pub fn dual_edge_lr(grid: &Grid, mats: &CellMaterials, axis: Axis, node: [usize; 3]) -> (f64, f64) {
    let n = grid.nodes();
    let (ab, ac) = axis.others();
    let (a, b, c) = (axis.index(), ab.index(), ac.index());
    assert!(node[b] + 1 < n[b] && node[c] + 1 < n[c], "not a dual edge");
    let area = grid.delta(ab, node[b]) * grid.delta(ac, node[c]);
    let (mut inv_l, mut r_over_l2) = (0.0, 0.0);
    for ca in [node[a].wrapping_sub(1), node[a]] {
        if ca >= n[a] - 1 {
            continue;
        }
        let mut cell = node;
        cell[a] = ca;
        let half = 0.5 * grid.delta(axis, ca);
        let m = mats.get(cell);
        let lc = MU0 * m.mu_r[a] * area / half;
        let rc = m.sigma_m[a] * area / half;
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
///
/// # Storage
///
/// An array has one value per node, `nz` of them a z-row. On a grid whose
/// cells repeat (a board on a uniform mesh: a handful of materials, the
/// same cell sizes over most of the plane) most rows of an array are
/// repeats of a few: `ca` and `da` are 1 in every lossless medium, `cb`
/// and `db` follow the material and the cell sizes. An array with few
/// distinct rows is held as those rows and an index per row
/// ([`compact`](Self::compact), which the engine calls when it is built
/// and before a run that follows a change), 4 bytes per row for 4 bytes
/// per node; the kernels read the rows where they are. The accessors
/// return a [`CoeffValues`], indexed like the slice they used to return;
/// [`e_mut`](Self::e_mut) and [`h_mut`](Self::h_mut) expand the arrays
/// they are called on.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Coefficients {
    /// ca (0–2), cb (3–5), da (6–8), db (9–11).
    lanes: [Lane; 12],
    nodes: [usize; 3],
}

/// One coefficient array ([storage](Coefficients#storage)).
#[derive(Debug, Clone)]
enum Lane {
    /// A value per node.
    Dense(Vec<f32>),
    /// The distinct rows (`nz` values each), the row of each `(i, j)`
    /// (index `i·ny + j`) among them, how many rows use each, and
    /// whether a write has given a row a copy of its own since the rows
    /// were pooled (two of them may be the same again).
    Pooled {
        rows: Vec<f32>,
        index: Vec<u32>,
        uses: Vec<u32>,
        written: bool,
    },
}

/// `⌈2⁶⁴/nz⌉`, with which `i / nz` is the high half of a product for
/// every `i` of 32 bits ([`Lane::at`]); 0 where there is none (`nz < 2`).
fn inverse(nz: usize) -> u64 {
    if nz < 2 { 0 } else { u64::MAX / nz as u64 + 1 }
}

/// An array is pooled when at most `1/POOL_RATIO` of its rows are distinct
/// (the kernels then read it from a pool that stays in the caches).
const POOL_RATIO: usize = 4;

/// Grids of fewer nodes are pooled on the calling thread
/// ([`Coefficients::compact_on`]).
const PARALLEL_NODES: usize = 1 << 22;

/// FNV-1a over the bit patterns.
fn hash(row: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in row {
        h = (h ^ u64::from(x.to_bits())).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Lane {
    fn len(&self, nz: usize) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::Pooled { index, .. } => index.len() * nz,
        }
    }

    /// The `nz` values of row `r`.
    #[inline(always)]
    fn row(&self, r: usize, nz: usize) -> &[f32] {
        match self {
            Self::Dense(v) => &v[r * nz..(r + 1) * nz],
            Self::Pooled { rows, index, .. } => {
                let o = index[r] as usize * nz;
                &rows[o..o + nz]
            }
        }
    }

    /// The value at `i`; `inverse` is [`inverse`] of `nz`.
    #[inline(always)]
    fn at(&self, i: usize, nz: usize, inverse: u64) -> &f32 {
        match self {
            Self::Dense(v) => &v[i],
            Self::Pooled { rows, index, .. } => {
                // `i / nz` without a division (operators read single
                // values every step): exact for an index of 32 bits
                // (Lemire, Kaser and Kurz 2019, "Faster remainder by
                // direct computation").
                let r = if inverse != 0 && i <= u32::MAX as usize {
                    ((i as u128 * u128::from(inverse)) >> 64) as usize
                } else {
                    i / nz
                };
                &rows[index[r] as usize * nz + (i - r * nz)]
            }
        }
    }

    /// The array as a value per node.
    fn dense_mut(&mut self, nz: usize) -> &mut Vec<f32> {
        if let Self::Pooled { rows, index, .. } = self {
            let mut v = Vec::with_capacity(index.len() * nz);
            for &id in index.iter() {
                let o = id as usize * nz;
                v.extend_from_slice(&rows[o..o + nz]);
            }
            *self = Self::Dense(v);
        }
        match self {
            Self::Dense(v) => v,
            Self::Pooled { .. } => unreachable!("expanded above"),
        }
    }

    /// Sets the value at `i`. In a pool the row becomes one of its own
    /// (the other rows that shared it keep theirs).
    fn set(&mut self, i: usize, value: f32, nz: usize) {
        let (rows, index, uses) = match self {
            Self::Dense(v) => {
                v[i] = value;
                return;
            }
            Self::Pooled {
                rows,
                index,
                uses,
                written,
            } => {
                *written = true;
                (rows, index, uses)
            }
        };
        let (r, k) = (i / nz, i % nz);
        let id = index[r] as usize;
        if rows[id * nz + k].to_bits() == value.to_bits() {
            return;
        }
        if uses[id] == 1 {
            rows[id * nz + k] = value;
            return;
        }
        // (No longer worth a pool once half the rows are their own.)
        if rows.len() / nz >= index.len() / 2 {
            self.dense_mut(nz)[i] = value;
            return;
        }
        uses[id] -= 1;
        index[r] = (rows.len() / nz) as u32;
        rows.extend_from_within(id * nz..(id + 1) * nz);
        let last = rows.len() - nz;
        rows[last + k] = value;
        uses.push(1);
    }

    /// Pools the rows if few are distinct ([`POOL_RATIO`]); a pool that
    /// writes have left with rows of its own is pooled again.
    fn compact(&mut self, nz: usize) {
        let pooled = match self {
            Self::Dense(v) => Self::pool(v.chunks_exact(nz), v.len() / nz, nz),
            Self::Pooled {
                rows,
                index,
                written,
                ..
            } => {
                if !*written {
                    return;
                }
                let it = index.iter().map(|&id| {
                    let o = id as usize * nz;
                    &rows[o..o + nz]
                });
                Self::pool(it, index.len(), nz)
            }
        };
        match pooled {
            Some(p) => *self = p,
            None => {
                self.dense_mut(nz);
            }
        }
    }

    /// The rows `rows` (`count` of them) as a pool, if at most
    /// `count / POOL_RATIO` are distinct.
    fn pool<'a>(rows: impl Iterator<Item = &'a [f32]>, count: usize, nz: usize) -> Option<Self> {
        let limit = (count / POOL_RATIO).max(1);
        let mut pool: Vec<f32> = Vec::new();
        let mut uses: Vec<u32> = Vec::new();
        let mut index = Vec::with_capacity(count);
        // Hash of the row bits → pool rows with that hash. (Looked up
        // only: the order of the map does not matter.)
        let mut seen: std::collections::HashMap<u64, Vec<u32>> = std::collections::HashMap::new();
        for row in rows {
            let cands = seen.entry(hash(row)).or_default();
            let hit = cands.iter().copied().find(|&c| {
                let p = &pool[c as usize * nz..(c as usize + 1) * nz];
                p.iter().zip(row).all(|(a, b)| a.to_bits() == b.to_bits())
            });
            let id = match hit {
                Some(c) => c,
                None => {
                    let c = uses.len() as u32;
                    if c as usize >= limit {
                        return None;
                    }
                    pool.extend_from_slice(row);
                    uses.push(0);
                    cands.push(c);
                    c
                }
            };
            uses[id as usize] += 1;
            index.push(id);
        }
        pool.shrink_to_fit();
        uses.shrink_to_fit();
        Some(Self::Pooled {
            rows: pool,
            index,
            uses,
            written: false,
        })
    }
}

/// A read-only coefficient array of [`Coefficients`]: one value per node,
/// indexed like a slice ([storage](Coefficients#storage)).
///
/// ```
/// use ems::engine::{Coefficients, EdgeMaterials, FaceConditions, Grid};
/// use ems::Axis;
/// let g = Grid::uniform([4, 4, 4], [1e-3; 3]).unwrap();
/// let co = Coefficients::from_edges(&g, &EdgeMaterials::vacuum(&g), &FaceConditions::default(), 1e-12).unwrap();
/// let cb = co.cb(Axis::Z);
/// assert_eq!(cb.len(), 125);
/// assert!(cb[g.index([2, 2, 1])] > 0.0);
/// assert_eq!(cb.row(2 * 5 + 2).len(), 5);
/// assert_eq!(cb.iter().count(), 125);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CoeffValues<'a> {
    lane: &'a Lane,
    nz: usize,
    inverse: u64,
    /// Index of the first value of the view.
    start: usize,
}

impl<'a> CoeffValues<'a> {
    /// Number of values (one per node).
    pub fn len(&self) -> usize {
        self.lane.len(self.nz) - self.start
    }

    /// The values from `start` on (an x-plane: `start = i·ny·nz`),
    /// indexed from 0.
    ///
    /// # Panics
    /// If `start` is past the end.
    pub fn starting_at(self, start: usize) -> Self {
        assert!(start <= self.len(), "start {start} past the end");
        Self {
            start: self.start + start,
            ..self
        }
    }

    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The value at `index`, if in range.
    pub fn get(&self, index: usize) -> Option<f32> {
        (index < self.len()).then(|| *self.lane.at(self.start + index, self.nz, self.inverse))
    }

    /// The z-row `row` of the array (`nz` values; row `i·ny + j` of node
    /// `(i, j, ·)`), whatever the view starts at.
    ///
    /// # Panics
    /// If the row is out of range.
    #[inline(always)]
    pub fn row(&self, row: usize) -> &'a [f32] {
        self.lane.row(row, self.nz)
    }

    /// The values in order.
    pub fn iter(&self) -> impl Iterator<Item = f32> + 'a {
        let (lane, nz, start) = (self.lane, self.nz.max(1), self.start);
        let rows = lane.len(nz) / nz;
        (start / nz..rows)
            .flat_map(move |r| lane.row(r, nz).iter().copied())
            .skip(start % nz)
    }

    /// The values as a vector.
    pub fn to_vec(&self) -> Vec<f32> {
        self.iter().collect()
    }
}

impl core::ops::Index<usize> for CoeffValues<'_> {
    type Output = f32;
    #[inline(always)]
    fn index(&self, index: usize) -> &f32 {
        self.lane.at(self.start + index, self.nz, self.inverse)
    }
}

impl PartialEq for CoeffValues<'_> {
    /// The same values.
    fn eq(&self, o: &Self) -> bool {
        self.len() == o.len() && self.iter().zip(o.iter()).all(|(a, b)| a == b)
    }
}

impl PartialEq<[f32]> for CoeffValues<'_> {
    fn eq(&self, o: &[f32]) -> bool {
        self.len() == o.len() && self.iter().zip(o).all(|(a, b)| a == *b)
    }
}

impl PartialEq for Coefficients {
    /// The same values on the same grid, however they are stored.
    fn eq(&self, o: &Self) -> bool {
        self.nodes == o.nodes && (0..12).all(|l| self.values(l) == o.values(l))
    }
}

/// How [`Coefficients::from_edges_rounded`] rounds the circuit
/// quantities of an edge (capacitance `C = ε·A/l`, conductance
/// `G = κ·A/l`, inductance `L = μ·A/l`, resistance `R = σ_m·A/l`) before
/// forming `ca = (C − G·dt/2)/(C + G·dt/2)`, `cb = dt/(C + G·dt/2)` and
/// the H analogues in `f64`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CoeffRounding {
    /// No intermediate rounding: each coefficient is the `f64` value
    /// rounded once to `f32` (default).
    #[default]
    Exact,
    /// openEMS's: `C`, `G`, `L` and `R` are rounded to `f32` first
    /// (design §5.16.3). Black box, `--debug-operator`: with it every
    /// `vv`, `vi`, `ii`, `iv` of the vacuum and material edges of the
    /// feature sweep's Cartesian models is openEMS's bit for bit, among
    /// them the vacuum `dt/C` at 1 m and 1 mm cells and dt 1.0…1.9 (ns
    /// resp. ps), where the `f64` quotient is one `f32` step off in 15 of
    /// 40 cases, and the material-boundary edges (`ε_r` 1.75: `f64`
    /// 96.80649, openEMS 96.806496). The capacitances and inductances
    /// use openEMS's vacuum constants ([`OPENEMS_EPS0`], [`OPENEMS_MU0`])
    /// instead of [`EPS0`](crate::constants::EPS0),
    /// [`MU0`](crate::constants::MU0) (design §5.16.9).
    OpenEms,
}

// openEMS's vacuum constants (design §5.16.9, §5.16.10).
pub use crate::constants::{OPENEMS_EPS0, OPENEMS_MU0};

impl CoeffRounding {
    /// A circuit quantity as stored before the coefficients are formed:
    /// unchanged ([`Exact`](Self::Exact)) or rounded to `f32`
    /// ([`OpenEms`](Self::OpenEms)).
    pub(crate) fn store(self, x: f64) -> f64 {
        match self {
            Self::Exact => x,
            Self::OpenEms => f64::from(x as f32),
        }
    }

    /// The vacuum permittivity the capacitances are formed with.
    pub(crate) fn eps0(self) -> f64 {
        match self {
            Self::Exact => EPS0,
            Self::OpenEms => OPENEMS_EPS0,
        }
    }

    /// The vacuum permeability the inductances are formed with.
    pub(crate) fn mu0(self) -> f64 {
        match self {
            Self::Exact => MU0,
            Self::OpenEms => OPENEMS_MU0,
        }
    }
}

/// `A/l` of the E edge along `axis` starting at node `p` (dual face area
/// over edge length), the factor of `ε` in its capacitance `C = ε·A/l`
/// and of `κ` in its conductance.
pub(crate) fn e_edge_geometry(grid: &Grid, axis: Axis, p: [usize; 3]) -> f64 {
    let (ab, ac) = axis.others();
    if grid.is_cylindrical() {
        grid.dual_face_area(axis, p) / grid.edge_length(axis, p)
    } else {
        grid.dual_delta(ab, p[ab.index()]) * grid.dual_delta(ac, p[ac.index()])
            / grid.delta(axis, p[axis.index()])
    }
}

/// The medium capacitance and conductance `(C, G)` of the E edge along
/// `axis` at `p` as [`Coefficients::from_edges_rounded`] forms them (with
/// `rounding` applied).
pub(crate) fn e_edge_circuit(
    grid: &Grid,
    mats: &EdgeMaterials,
    axis: Axis,
    p: [usize; 3],
    rounding: CoeffRounding,
) -> (f64, f64) {
    let idx = grid.index(p);
    let geo = e_edge_geometry(grid, axis, p);
    (
        rounding.store(rounding.eps0() * mats.eps_r(axis)[idx] * geo),
        rounding.store(mats.kappa(axis)[idx] * geo),
    )
}

/// The medium inductance of the H dual edge along `axis` at `p` as
/// [`Coefficients::from_edges_rounded`] forms it (with `rounding`
/// applied).
pub(crate) fn h_edge_inductance(
    grid: &Grid,
    mats: &EdgeMaterials,
    axis: Axis,
    p: [usize; 3],
    rounding: CoeffRounding,
) -> f64 {
    let (ab, ac) = axis.others();
    let geo = if grid.is_cylindrical() {
        grid.face_area(axis, p) / grid.dual_edge_length(axis, p)
    } else {
        grid.delta(ab, p[ab.index()]) * grid.delta(ac, p[ac.index()])
            / grid.dual_delta(axis, p[axis.index()])
    };
    rounding.store(rounding.mu0() * mats.mu_r(axis)[grid.index(p)] * geo)
}

/// `(ca, cb)` of an E edge with capacitance `cap` and conductance `g`:
/// `ca = (C − G·dt/2)/(C + G·dt/2)`, `cb = dt/(C + G·dt/2)` in `f64`,
/// rounded once to `f32`.
pub(crate) fn e_coefficients(cap: f64, g: f64, dt: f64) -> (f32, f32) {
    let den = cap + 0.5 * g * dt;
    (((cap - 0.5 * g * dt) / den) as f32, (dt / den) as f32)
}

impl Coefficients {
    /// Builds the coefficients for `grid`, the per-cell map `mats` and time
    /// step `dt`, with the outer faces handled according to `faces`.
    ///
    /// The cell map is first turned into effective edge values
    /// ([`EdgeMaterials::from_cells`]), then [`from_edges`](Self::from_edges)
    /// does the rest, so both input formats share one builder.
    pub fn compute(
        grid: &Grid,
        mats: &CellMaterials,
        faces: &FaceConditions,
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
        Self::check_dt(dt)?;
        Self::from_edges(grid, &EdgeMaterials::from_cells(grid, mats)?, faces, dt)
    }

    fn check_dt(dt: f64) -> Result<()> {
        if dt.is_finite() && dt > 0.0 {
            Ok(())
        } else {
            Err(Error::Invalid(format!("invalid time step {dt}")))
        }
    }

    /// Builds the coefficients from effective per-edge materials (see
    /// [`EdgeMaterials`] for the circuit quantities), time step `dt` and
    /// outer face conditions `faces`.
    ///
    /// E edges that are PEC (mask) or tangential to a
    /// [`FaceCondition::Pec`] face get `ca = cb = 0`.
    ///
    /// # Errors
    /// [`Error::Invalid`] if `mats` does not match the grid or fails
    /// [`EdgeMaterials::validate`], or `dt` is not finite and positive.
    pub fn from_edges(
        grid: &Grid,
        mats: &EdgeMaterials,
        faces: &FaceConditions,
        dt: f64,
    ) -> Result<Self> {
        Self::from_edges_rounded(grid, mats, faces, dt, CoeffRounding::Exact)
    }

    /// [`from_edges`](Self::from_edges) with the rounding `rounding` of
    /// the circuit quantities (see [`CoeffRounding`]).
    ///
    /// # Errors
    /// As [`from_edges`](Self::from_edges).
    pub fn from_edges_rounded(
        grid: &Grid,
        mats: &EdgeMaterials,
        faces: &FaceConditions,
        dt: f64,
        rounding: CoeffRounding,
    ) -> Result<Self> {
        let store = |x: f64| rounding.store(x);
        let (eps0, mu0) = (rounding.eps0(), rounding.mu0());
        if mats.nodes() != grid.nodes() {
            return Err(Error::Invalid(format!(
                "edge material map has {:?} nodes, grid has {:?}",
                mats.nodes(),
                grid.nodes()
            )));
        }
        mats.validate()?;
        Self::check_dt(dt)?;
        let n = grid.nodes();
        let len = grid.num_nodes();
        let mut data: [Vec<f32>; 12] = core::array::from_fn(|_| vec![0.0; len]);
        let co = &mut data;
        let cyl = grid.is_cylindrical();
        let faces = super::cylindrical::effective_faces(grid, faces);
        // Per axis `t`: whether its [low, high] face is PEC.
        let pec_face = Axis::ALL
            .map(|t| [false, true].map(|max| faces[Face::new(t, max)] == FaceCondition::Pec));
        for axis in Axis::ALL {
            let (eps, kap) = (mats.eps_r(axis), mats.kappa(axis));
            let (mu, sig) = (mats.mu_r(axis), mats.sigma_m(axis));
            let pec = mats.pec_mask(axis);
            let (ab, ac) = axis.others();
            let (a, b, c) = (axis.index(), ab.index(), ac.index());
            for i in 0..n[0] {
                for j in 0..n[1] {
                    for k in 0..n[2] {
                        let p = [i, j, k];
                        let idx = grid.index(p);
                        // E edge along `axis` starting at p.
                        if p[a] + 1 < n[a] {
                            let on_pec_face = [b, c].iter().any(|&t| {
                                (p[t] == 0 && pec_face[t][0])
                                    || (p[t] == n[t] - 1 && pec_face[t][1])
                            });
                            // Zero-length edges (α edges on the axis of a
                            // cylindrical grid) do not exist.
                            // Multigrid removes the edges off the kept α
                            // lines (§5.15.1).
                            let degenerate = cyl
                                && (grid.edge_length(axis, p) == 0.0 || grid.mg_removed_e(axis, p));
                            if !(on_pec_face || pec[idx] || degenerate) {
                                let geo = e_edge_geometry(grid, axis, p);
                                let cap = store(eps0 * eps[idx] * geo);
                                let g = store(kap[idx] * geo);
                                (co[a][idx], co[3 + a][idx]) = e_coefficients(cap, g, dt);
                            }
                        }
                        // H dual edge along `axis` through p.
                        // Zero-area faces (ρ faces on the axis) carry no
                        // flux: their dual edge does not exist.
                        if p[b] + 1 < n[b]
                            && p[c] + 1 < n[c]
                            && !(cyl
                                && (grid.face_area(axis, p) == 0.0 || grid.mg_removed_h(axis, p)))
                        {
                            let geo = if cyl {
                                grid.face_area(axis, p) / grid.dual_edge_length(axis, p)
                            } else {
                                grid.delta(ab, p[b]) * grid.delta(ac, p[c])
                                    / grid.dual_delta(axis, p[a])
                            };
                            let l = store(mu0 * mu[idx] * geo);
                            let r = store(sig[idx] * geo);
                            let den = l + 0.5 * r * dt;
                            co[6 + a][idx] = ((l - 0.5 * r * dt) / den) as f32;
                            co[9 + a][idx] = (dt / den) as f32;
                        }
                    }
                }
            }
        }
        let mut out = Self {
            lanes: data.map(Lane::Dense),
            nodes: n,
        };
        out.compact();
        Ok(out)
    }

    /// Stores every array with few distinct rows as those rows
    /// ([storage](Self#storage)). The values do not change.
    pub fn compact(&mut self) {
        self.compact_on(1);
    }

    /// [`compact`](Self::compact), an array per thread where `threads > 1`
    /// and the grid is large. (What a thread allocates stays with its
    /// allocator arena after it ends, and a pass over a board region is
    /// a few milliseconds: small grids are pooled on the calling thread.)
    pub(crate) fn compact_on(&mut self, threads: usize) {
        let nz = self.nodes[2];
        if nz == 0 {
            return;
        }
        if threads > 1 && self.nodes.iter().product::<usize>() >= PARALLEL_NODES {
            std::thread::scope(|s| {
                for lane in &mut self.lanes {
                    s.spawn(move || lane.compact(nz));
                }
            });
        } else {
            for lane in &mut self.lanes {
                lane.compact(nz);
            }
        }
    }

    /// Number of arrays held as their distinct rows.
    pub fn pooled(&self) -> usize {
        self.lanes
            .iter()
            .filter(|l| matches!(l, Lane::Pooled { .. }))
            .count()
    }

    /// Array `lane` (ca 0–2, cb 3–5, da 6–8, db 9–11).
    #[inline(always)]
    pub(crate) fn values(&self, lane: usize) -> CoeffValues<'_> {
        CoeffValues {
            lane: &self.lanes[lane],
            nz: self.nodes[2],
            inverse: inverse(self.nodes[2]),
            start: 0,
        }
    }

    /// Row `r` (`i·ny + j`; `nz` values) of array `lane`.
    #[inline(always)]
    pub(crate) fn row(&self, lane: usize, r: usize) -> &[f32] {
        self.lanes[lane].row(r, self.nodes[2])
    }

    /// Sets the value at `index` of array `lane` without expanding an
    /// array held as its distinct rows (the row gets one of its own).
    pub(crate) fn set(&mut self, lane: usize, index: usize, value: f32) {
        self.lanes[lane].set(index, value, self.nodes[2]);
    }

    /// Node counts of the grid these coefficients belong to.
    pub fn nodes(&self) -> [usize; 3] {
        self.nodes
    }

    /// `ca` of the E edges along `axis`.
    pub fn ca(&self, axis: Axis) -> CoeffValues<'_> {
        self.values(axis.index())
    }
    /// `cb` of the E edges along `axis`.
    pub fn cb(&self, axis: Axis) -> CoeffValues<'_> {
        self.values(3 + axis.index())
    }
    /// `da` of the H dual edges along `axis`.
    pub fn da(&self, axis: Axis) -> CoeffValues<'_> {
        self.values(6 + axis.index())
    }
    /// `db` of the H dual edges along `axis`.
    pub fn db(&self, axis: Axis) -> CoeffValues<'_> {
        self.values(9 + axis.index())
    }

    /// Mutable `ca`, `cb` of the E edges along `axis` (for boundary and
    /// lumped-element operators at setup time), each as a value per node
    /// ([storage](Self#storage)).
    pub fn e_mut(&mut self, axis: Axis) -> (&mut [f32], &mut [f32]) {
        self.pair_mut(axis.index())
    }

    /// Mutable `da`, `db` of the H dual edges along `axis`.
    pub fn h_mut(&mut self, axis: Axis) -> (&mut [f32], &mut [f32]) {
        self.pair_mut(6 + axis.index())
    }

    fn pair_mut(&mut self, first: usize) -> (&mut [f32], &mut [f32]) {
        let nz = self.nodes[2];
        let (lo, hi) = self.lanes.split_at_mut(first + 3);
        (lo[first].dense_mut(nz), hi[0].dense_mut(nz))
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
        let (c, gg) = primary_edge_cg(&g, &m, Axis::X, [0, 1, 1]);
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
        let (c, _) = primary_edge_cg(&g, &m, Axis::X, [1, 0, 0]);
        assert!(rel(c, EPS0 * 4.0 * 0.5 * 1.0) < 1e-14);
    }

    #[test]
    fn dual_edge_series() {
        let g = Grid::new([vec![0.0, 1.0, 3.0], vec![0.0, 1.0], vec![0.0, 2.0]]).unwrap();
        let mut m = CellMaterials::vacuum(&g);
        m.set([1, 0, 0], Material::VACUUM.with_mu_r(3.0).with_sigma_m(5.0));
        // x dual edge through node 1: half cells of 0.5 (vacuum) and 1.0 (mu_r 3).
        let (l, r) = dual_edge_lr(&g, &m, Axis::X, [1, 0, 0]);
        let area = 2.0;
        let l1 = MU0 * area / 0.5;
        let l2 = MU0 * 3.0 * area / 1.0;
        let want = 1.0 / (1.0 / l1 + 1.0 / l2);
        assert!(rel(l, want) < 1e-14);
        assert!(rel(r, want * want * (5.0 * area / 1.0) / (l2 * l2)) < 1e-14);
        // Homogeneous dual edge reproduces L = mu A / l*, R = sigma A / l*.
        let mut m2 = CellMaterials::new(&g, Material::VACUUM.with_mu_r(2.0).with_sigma_m(7.0));
        m2.set([0, 0, 0], *m2.get([1, 0, 0]));
        let (l, r) = dual_edge_lr(&g, &m2, Axis::X, [1, 0, 0]);
        assert!(rel(l, MU0 * 2.0 * area / 1.5) < 1e-14);
        assert!(rel(r, 7.0 * area / 1.5) < 1e-14);
        // Boundary dual edge: single half cell.
        let (l, _) = dual_edge_lr(&g, &m2, Axis::X, [2, 0, 0]);
        assert!(rel(l, MU0 * 2.0 * area / 1.0) < 1e-14);
    }

    #[test]
    fn faces_pec_and_loss() {
        let g = Grid::uniform([3, 3, 3], [1e-3; 3]).unwrap();
        let mut m = CellMaterials::new(&g, Material::VACUUM.with_kappa(1.0));
        m.set_pec_edge(Axis::Z, [1, 1, 1]);
        let dt = TimeStep::default().resolve(&g).unwrap();
        let mut faces = FaceConditions::default();
        let co = Coefficients::compute(&g, &m, &faces, dt).unwrap();
        // Tangential edge on the x_min face is PEC; interior edge is not.
        assert_eq!(co.cb(Axis::Y)[g.index([0, 1, 1])], 0.0);
        assert!(co.cb(Axis::Y)[g.index([1, 1, 1])] > 0.0);
        // PEC mask.
        assert_eq!(co.cb(Axis::Z)[g.index([1, 1, 1])], 0.0);
        assert_eq!(co.ca(Axis::Z)[g.index([1, 1, 1])], 0.0);
        // Normal edge touching the face is not tangential.
        assert!(co.cb(Axis::X)[g.index([0, 1, 1])] > 0.0);
        // Padding.
        assert_eq!(co.cb(Axis::X)[g.index([3, 1, 1])], 0.0);
        assert_eq!(co.db(Axis::X)[g.index([1, 3, 1])], 0.0);
        // Loss coefficients.
        let (c, gg) = primary_edge_cg(&g, &m, Axis::X, [1, 1, 1]);
        let idx = g.index([1, 1, 1]);
        let ca = (c - gg * dt / 2.0) / (c + gg * dt / 2.0);
        assert!(rel(co.ca(Axis::X)[idx] as f64, ca) < 1e-6);
        assert!(ca < 1.0);
        assert!(
            rel(
                Coefficients::capacitance(co.ca(Axis::X)[idx], co.cb(Axis::X)[idx], dt),
                c
            ) < 1e-6
        );
        // PMC on x_min keeps the face edges alive (half dual face).
        faces[Face::XMin] = FaceCondition::Pmc;
        let co = Coefficients::compute(&g, &m, &faces, dt).unwrap();
        let (c_face, _) = primary_edge_cg(&g, &m, Axis::Y, [0, 1, 1]);
        assert!(rel(c_face, 0.5 * c) < 1e-12);
        assert!(co.cb(Axis::Y)[g.index([0, 1, 1])] > 0.0);
        // But not where the x_min face meets the (PEC) y_min... z_min face.
        assert_eq!(co.cb(Axis::Y)[g.index([0, 1, 0])], 0.0);
    }
}
