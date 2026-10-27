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

use csx::properties::{MaterialQuantity, PropertyType, TypeMask};
use csx::{CoordSystem, Vec3};

use crate::Axis;
use crate::engine::{EdgeMaterials, Grid};

/// `[ε, μ, κ]` per component at a sample point; `None` in metal.
type Sample = [[f64; 3]; 3];

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
/// a single radial half (axis, `ρ_min`, `ρ_max`) are unchanged.
pub(crate) fn apply(model: &csx::ContinuousStructure, grid: &Grid, materials: &mut EdgeMaterials) {
    let lines: [&[f64]; 3] = core::array::from_fn(|a| model.grid.lines(a));
    let n = grid.nodes();
    let closed = grid.is_closed();
    let index = model.spatial_index(
        CoordSystem::Cylindrical,
        TypeMask::MATERIALS | TypeMask::of(PropertyType::Metal),
    );
    let bg = &model.background;
    let bg: Sample = [
        [bg.epsilon.value(); 3],
        [bg.mue.value(); 3],
        [bg.kappa.value(); 3],
    ];
    let sample = |p: [f64; 3]| -> Option<Sample> {
        match index.query(Vec3(p)) {
            None => Some(bg),
            Some(h) => {
                let m = h.property.kind.material()?;
                let [r, a, z] = p;
                let c = Vec3::new(r * a.cos(), r * a.sin(), z);
                Some([
                    m.values_at(MaterialQuantity::Epsilon, c),
                    m.values_at(MaterialQuantity::Mue, c),
                    m.values_at(MaterialQuantity::Kappa, c),
                ])
            }
        }
    };
    // Signed half cell next to line `i` of axis `a` on side `s`.
    let half = |a: usize, i: usize, s: i32| -> Option<f64> {
        let l = lines[a];
        if s < 0 {
            (i >= 1).then(|| -(l[i] - l[i - 1]) / 2.0)
        } else {
            (i + 1 < l.len()).then(|| (l[i + 1] - l[i]) / 2.0)
        }
    };
    let mid = |a: usize, i: usize| 0.5 * (lines[a][i] + lines[a][i + 1]);
    // openEMS's α weight of a z edge's half (black box, a box face swept
    // through the lower quarter on random sectors): the upper half weighs
    // the dual width of line `j`, the lower one that of line `j − 1` (a
    // whole cell at the first line), not their own widths; on a closed
    // grid the lines are periodic.
    let al = lines[1];
    let na = al.len();
    let dual = |j: isize| -> f64 {
        if closed {
            let cells = (na - 1) as isize;
            let jj = j.rem_euclid(cells) as usize;
            let lo = if jj > 0 {
                al[jj] - al[jj - 1]
            } else {
                al[na - 1] - al[na - 2]
            };
            0.5 * (lo + (al[jj + 1] - al[jj]))
        } else if j <= 0 {
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
    let (mut eps, mut kap, mut mu) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n[0] {
        let r = lines[0][i];
        // Both radial halves exist (α edges, H_ρ faces).
        let inner = i >= 1 && i + 1 < n[0];
        for j in 0..n[1] {
            for k in 0..n[2] {
                let idx = grid.index([i, j, k]);
                // α edge: radial halves × z halves.
                if inner && j + 1 < n[1] {
                    eps.clear();
                    kap.clear();
                    for hr in [half(0, i, -1), half(0, i, 1)].into_iter().flatten() {
                        for hz in [half(2, k, -1), half(2, k, 1)].into_iter().flatten() {
                            let p = [r + hr / 2.0, mid(1, j), lines[2][k] + hz / 2.0];
                            if let Some(s) = sample(p) {
                                let w = hr.abs() * hz.abs();
                                eps.push((w, s[0][1]));
                                kap.push((w, s[2][1]));
                            }
                        }
                    }
                    if let (Some(e), Some(c)) = (mean(&eps, false), mean(&kap, false)) {
                        materials.eps_r_mut(Axis::Y)[idx] = e;
                        materials.kappa_mut(Axis::Y)[idx] = c;
                    }
                }
                // z edge: radial halves × α halves, `Δρ_h·ρ_bound` times
                // the α weight of `alpha_weight` (also with a single
                // radial half, not on the axis).
                if k + 1 < n[2] && r > 0.0 {
                    eps.clear();
                    kap.clear();
                    for hr in [half(0, i, -1), half(0, i, 1)].into_iter().flatten() {
                        for ha in [half(1, j, -1), half(1, j, 1)].into_iter().flatten() {
                            let p = [r + hr / 2.0, lines[1][j] + ha / 2.0, mid(2, k)];
                            if let Some(s) = sample(p) {
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
                // ρ dual edge (H_ρ face): radial half cells in series.
                if inner && j + 1 < n[1] && k + 1 < n[2] {
                    mu.clear();
                    for hr in [half(0, i, -1), half(0, i, 1)].into_iter().flatten() {
                        if let Some(s) = sample([r + hr / 2.0, mid(1, j), mid(2, k)]) {
                            mu.push((hr.abs(), s[1][0]));
                        }
                    }
                    if let Some(m) = mean(&mu, true) {
                        materials.mu_r_mut(Axis::X)[idx] = m;
                    }
                }
            }
        }
    }
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
        apply(&s, &grid, &mut mats);
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
