//! openEMS's radial material averaging on cylindrical grids (files only,
//! [`Fdtd::openems_run`](crate::Fdtd::openems_run)).
//!
//! csx averages the materials around an edge over its quarter cells with
//! volume weights that include the Jacobian `ρ` (design §5.3, §5.15): for
//! a z edge this is the exact annular area of each quarter, for an α edge
//! and a ρ dual edge the ρ of the quarter's (half cell's) centre. openEMS
//! v0.37 samples the same points (the centre of each quarter, a single
//! sample; black box, `--debug-material`, a box face swept through a cell)
//! but weighs the two radial halves differently (fixtures
//! `cli_debug_cyl`, `cli_debug_cyl_sector`, `cyl_cellconst`):
//!
//! * ε and κ of a **z edge** on line `ρ_i`: the half between the dual line
//!   `ρ_b` and `ρ_i` weighs `|ρ_i − ρ_b|·ρ_b`, i.e. its width times the
//!   radius of the dual line that bounds it (instead of the annular area
//!   `|ρ_i − ρ_b|·(ρ_i + ρ_b)/2`), times an α weight that is not the
//!   half's own width: the dual width of line `j` for the upper half and
//!   of line `j − 1` for the lower one (a whole cell at the first line);
//!   black box, a box face swept through the lower quarter on random
//!   sectors, design §5.16.11;
//! * ε and κ of an **α edge**: the radial halves weigh their width `Δρ_h`
//!   (no `ρ`), times their z width;
//! * μ of a **ρ dual edge** (the H_ρ face): the harmonic mean of the two
//!   radial half cells weighted by their widths `Δρ_h` (no `ρ`).
//!
//! Every other average has no radial weight in either code (the `ρ`
//! factors cancel) and is unchanged. The exact capacitance of the two
//! halves of an α edge's dual face in parallel would weigh them by
//! `Δρ_h/ρ_h`, so neither code is exact there; ems's native path keeps
//! csx's averages (exact for z edges). [`apply`] recomputes the three
//! averages with openEMS's weights; σ_m is left to csx.
//!
//! Like csx, samples in metal are left out (an edge whose samples are all
//! metal keeps csx's value), identical samples average to themselves
//! exactly, and weighting functions are evaluated at the Cartesian
//! position of the sample.
//!
//! # α index ranges of boxes
//!
//! Both codes test a sample against a cylindrical box with its angle
//! taken modulo 2π, but openEMS also limits each (untransformed,
//! cylindrical) box to a range of α *indices*: from the last α line
//! below the box's lower angle `a_0` (strictly, `α_j < a_0`; line 0 if
//! there is none) to the first line above its upper angle `a_1`
//! (`α_j > a_1`; the last line if none), nothing if the box lies wholly
//! below the first line or above the last one (on a closed grid openEMS
//! adds the line `α_1 + 2π`, which counts here). An edge or face counts a
//! sample in the box only if its own α index (line or cell; the seam of
//! a closed grid has the index of the `α = 2π` line, where openEMS
//! updates it) lies in that range (black box, `--debug-material` and
//! `--debug-operator` of closed grids, uniform and finely graded α
//! meshes, `[0, 2π]` and `[−π, π]` lines, bounds on, an ulp and 1e-9
//! off the lines). The modulo test alone gives the same result for a box
//! within the grid's α lines; the range shows where the box's α range
//! leaves them: a box over `α ∈ [−0.5, 2]` on a `[0, 2π]` grid fills
//! only `[0, 2]` (not the seam, nor the cells below `2π`); one over
//! `[π, 3π]` (or any span ≥ 2π) covers from the cell below its lower
//! angle up to `2π`, not the cells between 0 and `a_0`; one over `[0, a]`
//! leaves out the seam's upper half unless `a ≥ α_{N−1}`. ems's native
//! path keeps the periodic reading (a box means the same region whatever
//! multiple of 2π its corners are given in). [`AlphaRanges`] holds the
//! ranges (taken from the undilated model, compile::openems), and
//! [`apply`] re-samples every edge and face that has a sample where the
//! ranges change the winning primitive, csx's own averages included
//! (with csx's weights, single sample per quarter cell).

use csx::primitives::{Primitive, Shape};
use csx::properties::{MaterialQuantity, PropertyType, TypeMask};
use csx::{CoordSystem, Hit, Vec3};

use crate::Axis;
use crate::engine::{EdgeMaterials, Grid};

/// `[ε, μ, κ, σ_m]` per component at a sample point; `None` in metal.
type Sample = [[f64; 3]; 4];

/// openEMS's α index ranges of the boxes of a model (see the
/// [module docs](self#α-index-ranges-of-boxes)): `(property index, box
/// ordinal within the property, first index, last index)` of the boxes
/// whose range leaves out some α index of the grid.
#[derive(Debug, Clone, Default)]
pub(crate) struct AlphaRanges(Vec<(usize, usize, usize, usize)>);

impl AlphaRanges {
    /// The ranges of the untransformed cylindrical boxes of `model`'s
    /// materials and metal on its α lines (none on Cartesian grids);
    /// `closed`: the grid spans 2π.
    pub(crate) fn of(model: &csx::ContinuousStructure, closed: bool) -> Self {
        let al = model.grid.lines(1);
        if model.grid.coord_system != CoordSystem::Cylindrical || al.len() < 2 {
            return Self::default();
        }
        let last = al.len() - 1;
        // openEMS's α lines end with `α_1 + 2π` on a closed grid.
        let end = if closed {
            al[1] + core::f64::consts::TAU
        } else {
            al[last]
        };
        let mut out = Vec::new();
        for (pi, p) in model.properties().iter().enumerate() {
            if p.kind.material().is_none() && p.property_type() != PropertyType::Metal {
                continue;
            }
            for (bi, prim) in boxes(p) {
                let Shape::Box(b) = &prim.shape else {
                    continue;
                };
                if prim.shape_cs(model.coord_system) != CoordSystem::Cylindrical
                    || !prim.transform.is_empty()
                {
                    continue;
                }
                let (s, e) = (b.start.value()[1], b.stop.value()[1]);
                let (a0, a1) = (s.min(e), s.max(e));
                // A box wholly outside the lines covers nothing.
                let (lo, hi) = if a1 < al[0] || a0 > end {
                    (1, 0)
                } else {
                    (
                        al.iter().rposition(|&a| a < a0).unwrap_or(0),
                        al.iter().position(|&a| a > a1).unwrap_or(last),
                    )
                };
                if lo > 0 || hi < last {
                    out.push((pi, bi, lo, hi));
                }
            }
        }
        Self(out)
    }
}

/// The α index of a sample that no range limits (the axis; a box wholly
/// outside the α lines still has no effect).
const ANY: usize = usize::MAX;

/// The box primitives of `p` with their ordinal among its boxes (stable
/// under csx's dilation and the removal of oblique wires).
fn boxes(p: &csx::Property) -> impl Iterator<Item = (usize, &Primitive)> {
    p.primitives
        .iter()
        .filter(|q| matches!(q.shape, Shape::Box(_)))
        .enumerate()
}

/// Weighted (`harmonic` = false: arithmetic, true: harmonic) mean of
/// samples, exact for identical samples; `None` without samples.
fn mean(samples: &[(f64, f64)], harmonic: bool) -> Option<f64> {
    let first = samples.first()?.1;
    if samples.iter().all(|s| s.1 == first) {
        return Some(first);
    }
    let w: f64 = samples.iter().map(|s| s.0).sum();
    if harmonic {
        Some(w / samples.iter().map(|s| s.0 / s.1).sum::<f64>())
    } else {
        Some(samples.iter().map(|s| s.0 * s.1).sum::<f64>() / w)
    }
}

/// Replaces the ε and κ of the α and z edges and the μ of the ρ dual
/// edges of `materials` (on the cylindrical `grid` of `model`) by
/// openEMS's radially weighted averages (see the module docs). Edges with
/// a single radial half (axis, `ρ_min`, `ρ_max`) are unchanged, unless the
/// α index `ranges` of the boxes change one of their samples; with them,
/// every other edge and face with such a sample is re-sampled too (csx's
/// weights).
///
/// Returns, on a closed grid, the `[μ_r, σ_m]` of openEMS's own `α = 0`
/// line of each seam `H_α` dual edge (`i·n_z + k`; NaN in metal): the two
/// halves of the seam's dual edge sampled with α index 0 (black box: the
/// material dump of that line), μ in series, σ_m averaged
/// (`engine::openems_cyl`); empty otherwise.
pub(crate) fn apply(
    model: &csx::ContinuousStructure,
    ranges: &AlphaRanges,
    grid: &Grid,
    materials: &mut EdgeMaterials,
) -> Vec<[f64; 2]> {
    let lines: [&[f64]; 3] = core::array::from_fn(|a| model.grid.lines(a));
    let n = grid.nodes();
    let closed = grid.is_closed();
    let mask = TypeMask::MATERIALS | TypeMask::of(PropertyType::Metal);
    let index = model.spatial_index(CoordSystem::Cylindrical, mask);
    let bg = &model.background;
    let bg: Sample = [
        [bg.epsilon.value(); 3],
        [bg.mue.value(); 3],
        [bg.kappa.value(); 3],
        [bg.sigma.value(); 3],
    ];
    // The limited boxes of `model` with their α index ranges.
    let limited: Vec<(&Primitive, usize, usize)> = model
        .properties()
        .iter()
        .enumerate()
        .flat_map(|(pi, p)| {
            boxes(p).filter_map(move |(bi, prim)| {
                ranges
                    .0
                    .iter()
                    .find(|r| r.0 == pi && r.1 == bi)
                    .map(|r| (prim, r.2, r.3))
            })
        })
        .collect();
    let allowed = |h: &Hit<'_>, j: usize| {
        limited
            .iter()
            .find(|r| core::ptr::eq(r.0, h.primitive))
            .is_none_or(|r| r.1 <= r.2 && (j == ANY || (r.1 <= j && j <= r.2)))
    };
    let material = |hit: Option<Hit<'_>>, p: [f64; 3]| -> Option<Sample> {
        match hit {
            None => Some(bg),
            Some(h) => {
                let m = h.property.kind.material()?;
                let [r, a, z] = p;
                let c = Vec3::new(r * a.cos(), r * a.sin(), z);
                Some([
                    m.values_at(MaterialQuantity::Epsilon, c),
                    m.values_at(MaterialQuantity::Mue, c),
                    m.values_at(MaterialQuantity::Kappa, c),
                    m.values_at(MaterialQuantity::Sigma, c),
                ])
            }
        }
    };
    // openEMS's sample at `p` for an edge or face of α index `j`, and
    // whether the α ranges changed the winning primitive there.
    let sample = |p: [f64; 3], j: usize| -> (Option<Sample>, bool) {
        match index.query(Vec3(p)) {
            Some(h) if !allowed(&h, j) => {
                let hit = model
                    .properties_at(Vec3(p), CoordSystem::Cylindrical, mask)
                    .into_iter()
                    .find(|h| allowed(h, j));
                (material(hit, p), true)
            }
            hit => (material(hit, p), false),
        }
    };
    // openEMS's α index of the edges and dual edges on line `j`: the seam
    // of a closed grid is the `α = 2π` line.
    let line_index = |j: usize| if closed && j == 0 { n[1] - 1 } else { j };
    // Signed half cell next to line `i` of axis `a` on side `s`.
    let half = |a: usize, i: usize, s: i32| -> Option<f64> {
        let l = lines[a];
        if s < 0 {
            (i >= 1).then(|| -(l[i] - l[i - 1]) / 2.0)
        } else {
            (i + 1 < l.len()).then(|| (l[i + 1] - l[i]) / 2.0)
        }
    };
    let halves = |a: usize, i: usize| [half(a, i, -1), half(a, i, 1)].into_iter().flatten();
    let mid = |a: usize, i: usize| 0.5 * (lines[a][i] + lines[a][i + 1]);
    let delta = |a: usize, i: usize| lines[a][i + 1] - lines[a][i];
    // openEMS's α weight of a z edge's half (black box, a box face swept
    // through the lower quarter on random sectors): the upper half weighs
    // the dual width of line `j`, the lower one that of line `j − 1` (a
    // whole cell at the first line, also on a closed grid: graded closed
    // meshes), not their own widths. (The copies of a seam edge have one
    // α half each; `Grid::with_openems_alpha_widths` weighs them.)
    let al = lines[1];
    let na = al.len();
    let dual = |j: isize| -> f64 {
        if j <= 0 {
            al[1] - al[0]
        } else if j as usize >= na - 1 {
            al[na - 1] - al[na - 2]
        } else {
            let j = j as usize;
            0.5 * (al[j + 1] - al[j - 1])
        }
    };
    let alpha_weight = |j: usize, lower: bool| -> f64 {
        if lower {
            dual(j as isize - 1)
        } else {
            dual(j as isize)
        }
    };
    let limits = !limited.is_empty();
    let graded = grid.openems_alpha_widths();
    let (mut eps, mut kap, mut mu, mut sig) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n[0] {
        let r = lines[0][i];
        // Both radial halves exist (α edges, H_ρ faces).
        let inner = i >= 1 && i + 1 < n[0];
        for j in 0..n[1] {
            let jl = line_index(j);
            for k in 0..n[2] {
                let idx = grid.index([i, j, k]);
                // α edge: radial halves × z halves.
                if (inner || limits) && j + 1 < n[1] {
                    eps.clear();
                    kap.clear();
                    let mut changed = false;
                    for hr in halves(0, i) {
                        for hz in halves(2, k) {
                            let p = [r + hr / 2.0, mid(1, j), lines[2][k] + hz / 2.0];
                            let (s, c) = sample(p, j);
                            changed |= c;
                            if let Some(s) = s {
                                let w = hr.abs() * hz.abs();
                                eps.push((w, s[0][1]));
                                kap.push((w, s[2][1]));
                            }
                        }
                    }
                    if (inner || changed)
                        && let (Some(e), Some(c)) = (mean(&eps, false), mean(&kap, false))
                    {
                        materials.eps_r_mut(Axis::Y)[idx] = e;
                        materials.kappa_mut(Axis::Y)[idx] = c;
                    }
                }
                // z edge: radial halves × α halves, `Δρ_h·ρ_bound` times
                // the α weight of `alpha_weight` (also with a single
                // radial half; on the axis only on a graded α mesh, where
                // openEMS sums the lines' capacitances, each line averaged
                // this way: black box, the axis E_z next to single samples).
                if k + 1 < n[2] && (r > 0.0 || graded) {
                    eps.clear();
                    kap.clear();
                    for hr in halves(0, i) {
                        for ha in halves(1, j) {
                            let p = [r + hr / 2.0, lines[1][j] + ha / 2.0, mid(2, k)];
                            // The axis is not limited by the ranges, only
                            // a box wholly outside the lines has no effect
                            // (black box: samples below 2π of a box over
                            // `[−0.4, 0.9]` count there).
                            let jz = if r > 0.0 { jl } else { ANY };
                            if let (Some(s), _) = sample(p, jz) {
                                let w = hr.abs() * (r + hr).abs() * alpha_weight(j, ha < 0.0);
                                eps.push((w, s[0][2]));
                                kap.push((w, s[2][2]));
                            }
                        }
                    }
                    if let (Some(e), Some(c)) = (mean(&eps, false), mean(&kap, false)) {
                        materials.eps_r_mut(Axis::Z)[idx] = e;
                        materials.kappa_mut(Axis::Z)[idx] = c;
                    }
                }
                // ρ dual edge (H_ρ face): radial half cells in series (σ_m:
                // csx's arithmetic mean, re-sampled only for the ranges).
                if (inner || limits) && j + 1 < n[1] && k + 1 < n[2] {
                    mu.clear();
                    sig.clear();
                    let mut changed = false;
                    for hr in halves(0, i) {
                        let p = [r + hr / 2.0, mid(1, j), mid(2, k)];
                        let (s, c) = sample(p, j);
                        changed |= c;
                        if let Some(s) = s {
                            mu.push((hr.abs(), s[1][0]));
                            sig.push((hr.abs() * delta(1, j) * p[0].abs(), s[3][0]));
                        }
                    }
                    if (inner || changed)
                        && let Some(m) = mean(&mu, true)
                    {
                        materials.mu_r_mut(Axis::X)[idx] = m;
                    }
                    if changed && let Some(s) = mean(&sig, false) {
                        materials.sigma_m_mut(Axis::X)[idx] = s;
                    }
                }
                if !limits || i + 1 >= n[0] {
                    continue;
                }
                // csx's own averages, re-sampled where the ranges change a
                // sample (its weights, one sample per quarter or half cell).
                let (rm, dr) = (mid(0, i), delta(0, i));
                // ρ edge: α halves × z halves.
                eps.clear();
                kap.clear();
                let mut changed = false;
                for ha in halves(1, j) {
                    for hz in halves(2, k) {
                        let p = [rm, lines[1][j] + ha / 2.0, lines[2][k] + hz / 2.0];
                        let (s, c) = sample(p, jl);
                        changed |= c;
                        if let Some(s) = s {
                            let w = dr * (ha.abs() * rm.abs()) * hz.abs();
                            eps.push((w, s[0][0]));
                            kap.push((w, s[2][0]));
                        }
                    }
                }
                if changed && let (Some(e), Some(c)) = (mean(&eps, false), mean(&kap, false)) {
                    materials.eps_r_mut(Axis::X)[idx] = e;
                    materials.kappa_mut(Axis::X)[idx] = c;
                }
                // α dual edge (H_α face): α halves.
                if k + 1 < n[2] {
                    mu.clear();
                    sig.clear();
                    let mut changed = false;
                    for ha in halves(1, j) {
                        let p = [rm, lines[1][j] + ha / 2.0, mid(2, k)];
                        let (s, c) = sample(p, jl);
                        changed |= c;
                        if let Some(s) = s {
                            let w = dr * (ha.abs() * rm.abs()) * delta(2, k);
                            mu.push((w, s[1][1]));
                            sig.push((w, s[3][1]));
                        }
                    }
                    if changed && let (Some(m), Some(s)) = (mean(&mu, true), mean(&sig, false)) {
                        materials.mu_r_mut(Axis::Y)[idx] = m;
                        materials.sigma_m_mut(Axis::Y)[idx] = s;
                    }
                }
                // z dual edge (H_z face): z halves.
                if j + 1 < n[1] {
                    mu.clear();
                    sig.clear();
                    let mut changed = false;
                    for hz in halves(2, k) {
                        let p = [rm, mid(1, j), lines[2][k] + hz / 2.0];
                        let (s, c) = sample(p, j);
                        changed |= c;
                        if let Some(s) = s {
                            let w = dr * (delta(1, j) * rm.abs()) * hz.abs();
                            mu.push((w, s[1][2]));
                            sig.push((w, s[3][2]));
                        }
                    }
                    if changed && let (Some(m), Some(s)) = (mean(&mu, true), mean(&sig, false)) {
                        materials.mu_r_mut(Axis::Z)[idx] = m;
                        materials.sigma_m_mut(Axis::Z)[idx] = s;
                    }
                }
            }
        }
    }
    if !closed {
        return Vec::new();
    }
    // openEMS's α = 0 line of the seam H_α: the halves of the seam's dual
    // edge (the last and the first cell), sampled with α index 0.
    let (dl, dh) = (delta(1, n[1] - 2), delta(1, 0));
    let mut alpha0 = vec![[f64::NAN; 2]; n[0] * n[2]];
    for i in 0..n[0] - 1 {
        for k in 0..n[2] - 1 {
            mu.clear();
            sig.clear();
            for (w, a) in [(dl, lines[1][0] - 0.25 * dl), (dh, lines[1][0] + 0.25 * dh)] {
                if let (Some(s), _) = sample([mid(0, i), a, mid(2, k)], 0) {
                    mu.push((w, s[1][1]));
                    sig.push((w, s[3][1]));
                }
            }
            if let (Some(m), Some(s)) = (mean(&mu, true), mean(&sig, false)) {
                alpha0[i * n[2] + k] = [m, s];
            }
        }
    }
    alpha0
}

#[cfg(test)]
mod tests {
    use super::*;
    use csx::primitives::{BoxShape, Primitive};
    use csx::properties::Material;
    use csx::{ContinuousStructure, Property};

    /// A dielectric outside `ρ = 2` on lines 1, 2, 4: the z edge on
    /// `ρ = 2` weighs its halves `0.5·1.5` (inner) and `1·3` (outer).
    #[test]
    fn radial_weights() {
        let mut s = ContinuousStructure::new();
        s.coord_system = CoordSystem::Cylindrical;
        s.grid.coord_system = CoordSystem::Cylindrical;
        s.grid.set_lines(0, [1.0, 2.0, 4.0]).unwrap();
        s.grid.set_lines(1, [0.0, 0.5, 1.0]).unwrap();
        s.grid.set_lines(2, [0.0, 1.0, 2.0]).unwrap();
        let mut m = Material::new(4.0, 0.0);
        m.mue.set(2.0);
        let d = s.add_property(Property::material("d", m));
        s.add_primitive(
            d,
            Primitive::new(BoxShape::new([2.0, -1.0, -1.0], [5.0, 2.0, 3.0]))
                .with_coord_system(CoordSystem::Cylindrical),
        )
        .unwrap();
        let grid = Grid::cylindrical([
            vec![1.0, 2.0, 4.0],
            vec![0.0, 0.5, 1.0],
            vec![0.0, 1.0, 2.0],
        ])
        .unwrap();
        let mut mats = EdgeMaterials::vacuum(&grid);
        apply(&s, &AlphaRanges::default(), &grid, &mut mats);
        let idx = grid.index([1, 1, 1]);
        let (wi, wo) = (0.5 * 1.5, 1.0 * 3.0);
        let want = (wi + 4.0 * wo) / (wi + wo);
        assert!((mats.eps_r(Axis::Z)[idx] - want).abs() < 1e-12);
        // α edge: widths 0.5 and 1.
        assert!((mats.eps_r(Axis::Y)[idx] - (0.5 + 4.0) / 1.5).abs() < 1e-12);
        // H_ρ: harmonic over the widths 0.5 and 1.
        assert!((mats.mu_r(Axis::X)[idx] - 1.5 / (0.5 + 0.5)).abs() < 1e-12);
    }
}
