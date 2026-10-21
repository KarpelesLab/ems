//! openEMS's metal surface rule for curved primitives and its magnetic
//! conductivity average (files only,
//! [`Fdtd::openems_run`](crate::Fdtd::openems_run)).
//!
//! # Metal surfaces
//!
//! csx marks an edge as PEC when its midpoint lies in a metal primitive,
//! with closed bounds (a small tolerance). openEMS v0.37 agrees for boxes,
//! cylinders and cylindrical shells, but for spheres, spherical shells and
//! wires an edge whose midpoint lies exactly *on* the surface is not metal
//! (black box, `--debug-PEC` dumps of the fixtures
//! `tests/fixtures/openems/sweep_tie_*`: a sphere of radius 4.5 mm on a
//! 1 mm mesh loses the 102 edges with `|m − c| = 4.5`, a wire of radius
//! 3 mm the 40 edges at distance 3; every other edge is identical, design
//! §5.16.3). [`strict_surfaces`] removes those edges from the PEC mask
//! unless another metal primitive of the stack claims them.
//!
//! Only primitives without a transformation and in Cartesian coordinates
//! are handled: an exact tie needs exact coordinates, which transformed
//! primitives do not have (openEMS's and csx's rounding differ there
//! anyway).
//!
//! # Magnetic conductivity
//!
//! The `σ_m` of an H face (normal `d`, lowest corner at node `n`, csx
//! layout) comes from the same two samples in both codes: the material
//! at the face centre in the transverse axes and at the middle of each
//! half of the dual edge along `d` (`x_n ∓ Δ/4`), the two half lengths
//! `l_−`, `l_+` as weights. csx (and ems's native path) takes the
//! arithmetic mean; openEMS v0.37 (black box, `--debug-material` dumps on
//! hand-written slabs, uniform and graded meshes, all three axes, weights
//! and anisotropic values, design §5.16.3) takes the length-weighted
//! *harmonic* mean `(l_− + l_+)/(l_−/σ_− + l_+/σ_+)`, like `μ`, except that
//! a zero half is not a short circuit on both sides: a zero upper half
//! (`σ_+ = 0`) gives 0, a zero lower half is left out of the denominator
//! (`(l_− + l_+)·σ_+/l_+`, i.e. `2σ` on a uniform mesh). A face on the
//! first or last line has one half only and takes its value. Metal does
//! not enter: the samples see the highest-priority material under it
//! (or the background). [`openems_sigma_m`] rewrites csx's averages
//! with that rule.

use csx::discretize::MaterialGrid;
use csx::primitives::{Primitive, Shape};
use csx::properties::{MaterialQuantity, PropertyKind, TypeMask};
use csx::{ContinuousStructure, CoordSystem, Vec3};

/// Relative tolerance of a surface tie.
const TIE_TOL: f64 = 1e-9;

fn point(p: &csx::param::ScalarPoint) -> Vec3 {
    p.value()
}

fn dist(a: Vec3, b: Vec3) -> f64 {
    (0..3)
        .map(|i| (a.0[i] - b.0[i]).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Distance from `p` to the segment `a`–`b`.
fn seg_dist(p: Vec3, a: Vec3, b: Vec3) -> f64 {
    let d: [f64; 3] = core::array::from_fn(|i| b.0[i] - a.0[i]);
    let l2: f64 = d.iter().map(|x| x * x).sum();
    let t = if l2 > 0.0 {
        ((0..3).map(|i| (p.0[i] - a.0[i]) * d[i]).sum::<f64>() / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    dist(p, Vec3(core::array::from_fn(|i| a.0[i] + t * d[i])))
}

/// True if `p` lies on the surface of a sphere, spherical shell or wire
/// (within [`TIE_TOL`] of the size), `None` for other shapes.
fn on_surface(shape: &Shape, p: Vec3) -> Option<bool> {
    let tie = |d: f64, scale: f64| d.abs() <= TIE_TOL * scale.abs().max(f64::MIN_POSITIVE);
    match shape {
        Shape::Sphere(s) => {
            let r = s.radius.value();
            Some(tie(dist(p, point(&s.center)) - r, r))
        }
        Shape::SphericalShell(s) => {
            let (r, w) = (s.radius.value(), s.shell_width.value());
            let d = dist(p, point(&s.center));
            Some(tie((d - r).abs() - 0.5 * w, r + w))
        }
        Shape::Wire(w) => {
            let r = w.radius.value();
            let pts: Vec<Vec3> = w.points.iter().map(point).collect();
            let d = pts
                .windows(2)
                .map(|s| seg_dist(p, s[0], s[1]))
                .fold(f64::INFINITY, f64::min);
            let d = if pts.len() == 1 { dist(p, pts[0]) } else { d };
            Some(r > 0.0 && d.is_finite() && tie(d - r, r))
        }
        _ => None,
    }
}

fn applies(p: &Primitive, default_cs: CoordSystem) -> bool {
    p.transform.is_empty()
        && p.effective_cs(default_cs) == CoordSystem::Cartesian
        && matches!(
            p.shape,
            Shape::Sphere(_) | Shape::SphericalShell(_) | Shape::Wire(_)
        )
}

/// Clears the PEC flag (csx node layout, `pec[a][mg_index(node)]`) of the
/// edges whose midpoint lies on the surface of the winning metal sphere,
/// spherical shell or wire, unless a metal primitive below it in the
/// stack contains the midpoint strictly. Returns the number of edges
/// cleared.
pub(crate) fn strict_surfaces(
    model: &ContinuousStructure,
    dims: [usize; 3],
    pec: &mut [Vec<bool>; 3],
) -> usize {
    if model.grid.coord_system != CoordSystem::Cartesian {
        return 0;
    }
    let cs = model.coord_system;
    let any = model.properties().iter().any(|p| {
        matches!(p.kind, PropertyKind::Metal) && p.primitives.iter().any(|q| applies(q, cs))
    });
    if !any {
        return 0;
    }
    let lines: [&[f64]; 3] = core::array::from_fn(|a| model.grid.lines(a));
    let index = |n: [usize; 3]| n[0] + dims[0] * (n[1] + dims[1] * n[2]);
    let mut cleared = 0;
    for prop in model.properties() {
        if !matches!(prop.kind, PropertyKind::Metal) {
            continue;
        }
        for prim in prop.primitives.iter().filter(|q| applies(q, cs)) {
            let bb = prim.bounding_box(cs);
            if bb.is_empty() {
                continue;
            }
            let range = |a: usize| {
                let l = lines[a];
                let lo = l.partition_point(|&x| x < bb.min.0[a] - 1e-9 * (1.0 + bb.min.0[a].abs()));
                let hi =
                    l.partition_point(|&x| x <= bb.max.0[a] + 1e-9 * (1.0 + bb.max.0[a].abs()));
                (lo.saturating_sub(1), hi.min(l.len()))
            };
            let r = [range(0), range(1), range(2)];
            for (a, pec_a) in pec.iter_mut().enumerate() {
                for i in r[0].0..r[0].1 {
                    for j in r[1].0..r[1].1 {
                        for k in r[2].0..r[2].1 {
                            let n = [i, j, k];
                            if n[a] + 1 >= dims[a] || !pec_a[index(n)] {
                                continue;
                            }
                            let m = Vec3(core::array::from_fn(|b| {
                                if b == a {
                                    0.5 * (lines[b][n[b]] + lines[b][n[b] + 1])
                                } else {
                                    lines[b][n[b]]
                                }
                            }));
                            if on_surface(&prim.shape, m) != Some(true) {
                                continue;
                            }
                            // The stack at the midpoint: this primitive must
                            // win, and nothing metal below it may contain
                            // the point strictly.
                            let hits =
                                model.properties_at(m, CoordSystem::Cartesian, TypeMask::PHYSICAL);
                            let mut it = hits.iter().filter(|h| {
                                matches!(
                                    h.property.kind,
                                    PropertyKind::Metal
                                        | PropertyKind::Material(_)
                                        | PropertyKind::LorentzMaterial(_)
                                        | PropertyKind::DebyeMaterial(_)
                                        | PropertyKind::DiscMaterial(_)
                                )
                            });
                            let Some(top) = it.next() else { continue };
                            if !core::ptr::eq(top.primitive, prim) {
                                continue;
                            }
                            let below_metal = it.find(|h| {
                                !(applies(h.primitive, cs)
                                    && on_surface(&h.primitive.shape, m) == Some(true))
                            });
                            if below_metal
                                .is_some_and(|h| matches!(h.property.kind, PropertyKind::Metal))
                            {
                                continue;
                            }
                            pec_a[index(n)] = false;
                            cleared += 1;
                        }
                    }
                }
            }
        }
    }
    cleared
}

/// openEMS's dual-edge average of two half-edge magnetic conductivities
/// (see the [module docs](self#magnetic-conductivity)): `lo`/`hi` are the
/// lower and upper halves `(length, σ_m)` when they exist.
fn sigma_m_average(lo: Option<(f64, f64)>, hi: Option<(f64, f64)>) -> f64 {
    if hi.is_some_and(|(_, s)| s == 0.0) {
        return 0.0;
    }
    let len: f64 = lo.iter().chain(&hi).map(|h| h.0).sum();
    let inv: f64 = lo
        .iter()
        .chain(&hi)
        .filter(|h| h.1 != 0.0)
        .map(|h| h.0 / h.1)
        .sum();
    if inv == 0.0 { 0.0 } else { len / inv }
}

/// Replaces csx's arithmetic `σ_m` face averages by openEMS's rule (see
/// the [module docs](self#magnetic-conductivity)). Cartesian grids only;
/// a model without any magnetic conductivity is left untouched. Returns
/// the number of faces whose value changed.
pub(crate) fn openems_sigma_m(model: &ContinuousStructure, mg: &mut MaterialGrid) -> usize {
    if model.grid.coord_system != CoordSystem::Cartesian {
        return 0;
    }
    let lossy = |m: &csx::properties::Material| {
        (0..3).any(|d| m.base_value(MaterialQuantity::Sigma, d) != 0.0)
    };
    let any = model.background.sigma.value() != 0.0
        || model.properties().iter().any(|p| {
            p.kind
                .material()
                .is_some_and(|m| lossy(m) && !p.primitives.is_empty())
        });
    if !any {
        return 0;
    }
    let index = model.spatial_index(CoordSystem::Cartesian, TypeMask::MATERIALS);
    let bg = model.background.sigma.value();
    let sample = |p: Vec3, d: usize| -> f64 {
        match index.query(p).and_then(|h| h.property.kind.material()) {
            Some(m) => m.value_at(MaterialQuantity::Sigma, d, p),
            None => bg,
        }
    };
    let dims = mg.dims;
    let lines: [&[f64]; 3] = core::array::from_fn(|a| model.grid.lines(a));
    let mut changed = 0;
    for d in 0..3 {
        let (t1, t2) = ((d + 1) % 3, (d + 2) % 3);
        for idx in 0..mg.sigma[d].len() {
            let n = mg.node(idx);
            if n[t1] + 1 >= dims[t1] || n[t2] + 1 >= dims[t2] {
                continue;
            }
            let mut p = Vec3::ZERO;
            for t in [t1, t2] {
                p.0[t] = 0.5 * (lines[t][n[t]] + lines[t][n[t] + 1]);
            }
            let x = lines[d][n[d]];
            let mut half = |other: Option<f64>| {
                other.map(|y| {
                    let h = 0.5 * (y - x);
                    p.0[d] = x + 0.5 * h;
                    (h.abs(), sample(p, d))
                })
            };
            let lo = half((n[d] >= 1).then(|| lines[d][n[d] - 1]));
            let hi = half((n[d] + 1 < dims[d]).then(|| lines[d][n[d] + 1]));
            let v = sigma_m_average(lo, hi);
            if v != mg.sigma[d][idx] {
                mg.sigma[d][idx] = v;
                changed += 1;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use csx::primitives::{Sphere, Wire};
    use csx::properties::Property;

    fn model(shape: impl Into<Primitive>) -> ContinuousStructure {
        let mut s = ContinuousStructure::new();
        let lines: Vec<f64> = (0..=10).map(f64::from).collect();
        for a in 0..3 {
            s.grid.set_lines(a, lines.clone()).unwrap();
        }
        let id = s.add_property(Property::metal("m"));
        s.add_primitive(id, shape).unwrap();
        s
    }

    fn count(pec: &[Vec<bool>; 3]) -> usize {
        pec.iter().map(|p| p.iter().filter(|&&b| b).count()).sum()
    }

    #[test]
    fn sigma_m_rule() {
        // Measured with openEMS's --debug-material (design §5.16.3).
        let h = |s: f64| Some((0.5, s));
        assert_eq!(sigma_m_average(h(3000.0), h(1000.0)), 1500.0);
        assert!((sigma_m_average(h(500.0), h(3000.0)) - 857.142_857).abs() < 1e-3);
        assert_eq!(sigma_m_average(h(0.0), h(2000.0)), 4000.0);
        assert_eq!(sigma_m_average(h(2000.0), h(0.0)), 0.0);
        assert_eq!(
            sigma_m_average(Some((0.5, 0.0)), Some((0.25, 2000.0))),
            6000.0
        );
        assert_eq!(sigma_m_average(None, h(1000.0)), 1000.0);
        assert_eq!(sigma_m_average(h(3000.0), None), 3000.0);
        assert_eq!(sigma_m_average(h(0.0), None), 0.0);
    }

    #[test]
    fn sphere_ties() {
        let s = model(Sphere::new([5.0, 5.0, 5.0], 2.5));
        let n = 11 * 11 * 11;
        // The x edge from (2, 5, 5) to (3, 5, 5): midpoint at r = 2.5.
        let mut pec: [Vec<bool>; 3] = core::array::from_fn(|_| vec![false; n]);
        let idx = |i: usize, j: usize, k: usize| i + 11 * (j + 11 * k);
        pec[0][idx(2, 5, 5)] = true;
        pec[0][idx(3, 5, 5)] = true; // r = 1.5: inside, kept
        assert_eq!(strict_surfaces(&s, [11; 3], &mut pec), 1);
        assert!(!pec[0][idx(2, 5, 5)] && pec[0][idx(3, 5, 5)]);
        assert_eq!(count(&pec), 1);
    }

    #[test]
    fn wire_ties() {
        let s = model(Wire::new([[5.0, 5.0, 1.0], [5.0, 5.0, 9.0]], 2.0));
        let n = 11 * 11 * 11;
        let idx = |i: usize, j: usize, k: usize| i + 11 * (j + 11 * k);
        let mut pec: [Vec<bool>; 3] = core::array::from_fn(|_| vec![false; n]);
        pec[2][idx(3, 5, 4)] = true; // z edge at distance 2: on the surface
        pec[2][idx(4, 5, 4)] = true; // distance 1: inside
        assert_eq!(strict_surfaces(&s, [11; 3], &mut pec), 1);
        assert!(!pec[2][idx(3, 5, 4)] && pec[2][idx(4, 5, 4)]);
    }
}
