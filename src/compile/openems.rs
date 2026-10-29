//! openEMS's inside tolerance of primitives and its magnetic conductivity
//! average (files only,
//! [`Fdtd::openems_run`](crate::Fdtd::openems_run)).
//!
//! # Inside tolerance
//!
//! csx's inside tests are closed with a tiny relative tolerance
//! (`1e-12`). openEMS since openEMS-Project be30220 (2026-10-21; CSXCAD
//! v0.7.0-8) is closed too, for every primitive, and accepts every point
//! within a distance `t = 1e-3 · h_min` of the primitive, `h_min` the
//! smallest cell width of the mesh over all three axes in drawing units
//! (on cylindrical grids the α widths count in radians); for a
//! transformed primitive the tolerance is `t / σ_min` in the primitive's
//! own coordinates, `σ_min` the smallest singular value of the
//! transformation (black box, bisection of `--debug-PEC`,
//! `--debug-material` and `--debug-operator` dumps: box faces, edges and
//! corners, sphere, cylinder and shell surfaces in several directions,
//! wire, polygon and extruded-polygon sides and ends, polyhedron faces,
//! uniform and graded meshes, `DeltaUnit` 1 and 1e-6, offsets, `Scale3`
//! of spheres and boxes; metal, materials, excitations and conducting
//! sheets; lumped elements, probes and dumps snap to the mesh and are not
//! affected; design §5.16.3). [`dilated`] gives the model whose csx
//! inside tests have that tolerance: volume boxes, spheres, cylinders,
//! their shells, wires and extruded polygons grown by it (boxes in
//! cylindrical coordinates by `t/ρ_max` in α). Not grown: rotational
//! polygons, polyhedra, thin wires (their radius sets the in-cell
//! correction), and sheets and lines (csx selects their edges by overlap
//! and snapping, `snap_tolerance` of the cell, not by inside tests);
//! exact ties are closed in both codes, only points `0 < d ≤ t` outside
//! them differ.
//!
//! openEMS-Project e89286f (2026-10-18) had no tolerance, left the
//! surface of spheres, spherical shells and wires and some polygon
//! vertices open, and decided points on polyhedron faces by the
//! primitive's position (design §5.16.3); ems emulated the first two
//! until the fixtures moved to be30220.
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
//!
//! # Thin metal under materials
//!
//! csx makes the edges of a metal sheet or line PEC whatever lies there;
//! for solid metal it takes the highest priority at the edge midpoint.
//! openEMS v0.37 applies the priority to sheets and lines too: a material
//! of higher priority containing the midpoint (closed bounds, its faces
//! included) removes the edge; at equal priority the property defined
//! first in the file wins (black box, `--debug-PEC`: a sheet under a
//! dielectric box below, above and through its plane at priorities 1, 2,
//! 3 against the sheet's 2, defined after the sheet; the tie the other
//! way round from the random sweep; a metal line under a box; lumped
//! elements and conducting sheets do not count). [`thin_metal_priority`]
//! clears those edges.
//!
//! # Lumped elements in metal
//!
//! An edge of a lumped element is the element's whatever metal lies
//! there: openEMS v0.37 installs the element on its edges after the metal,
//! at any priority (black box, `--debug-PEC` and probe files: R and C
//! elements with and without caps crossing a metal sheet, a metal line
//! and a solid metal box, element priority above, equal to and below the
//! metal's; random sweep seed 157). Only the element's own edges (along
//! its direction) are affected; the metal's other edges stay PEC, and the
//! caps span the element's whole box. [`lumped_over_metal`] clears the PEC
//! flag of those edges, which then take the medium the metal hides (see
//! below) before the element replaces it.
//!
//! # Material samples in metal
//!
//! csx leaves the material samples that fall into metal out of an edge's
//! average (and gives an edge whose samples are all metal the
//! background). openEMS v0.37 does not: a sample in metal sees the
//! highest-priority material under the metal, or the background, as if
//! the metal were not there (black box, `--debug-material`: a metal box
//! sharing a face with a higher-priority dielectric gives the face edges
//! `(3·1 + 4)/4 = 1.75` on vacuum, `(3·9 + 4)/4 = 7.75` inside an
//! `ε_r = 9` block below the metal, and every PEC edge inside the metal
//! the `ε`, `μ`, `κ` of that block; design §5.16.1). The PEC edges do not
//! need a material, but openEMS's time step evaluates their circuit
//! values like any other edge's, so their materials matter there.
//! [`without_metal`] gives the model whose discretisation has those
//! averages.

use csx::discretize::MaterialGrid;
use csx::param::ScalarPoint;
use csx::primitives::{PolyVertices, Primitive, Shape};
use csx::properties::{ExcitationType, MaterialQuantity, PropertyKind, TypeMask};
use csx::{ContinuousStructure, CoordSystem, Vec3};

/// Clears the PEC flag (csx node layout) of the edges of thin metal
/// (sheets, lines: primitives of dimension below 3) whose midpoint lies in
/// a material that wins the priority there over every metal primitive
/// containing it (see the [module docs](self#thin-metal-under-materials)).
/// Cartesian grids only; returns the number of edges cleared.
pub(crate) fn thin_metal_priority(
    model: &ContinuousStructure,
    dims: [usize; 3],
    pec: &mut [Vec<bool>; 3],
) -> usize {
    if model.grid.coord_system != CoordSystem::Cartesian {
        return 0;
    }
    let cs = model.coord_system;
    let thin: Vec<&Primitive> = model
        .properties()
        .iter()
        .filter(|p| matches!(p.kind, PropertyKind::Metal))
        .flat_map(|p| p.primitives.iter())
        .filter(|q| q.dimension() < 3)
        .collect();
    let any_material = model
        .properties()
        .iter()
        .any(|p| p.kind.material().is_some() && !p.primitives.is_empty());
    if thin.is_empty() || !any_material {
        return 0;
    }
    let lines: [&[f64]; 3] = core::array::from_fn(|a| model.grid.lines(a));
    let index = |n: [usize; 3]| n[0] + dims[0] * (n[1] + dims[1] * n[2]);
    let mask = TypeMask::MATERIALS | TypeMask::of(csx::properties::PropertyType::Metal);
    let mut cleared = 0;
    for prim in thin {
        let bb = prim.bounding_box(cs);
        if bb.is_empty() {
            continue;
        }
        let range = |a: usize| {
            let l = lines[a];
            let lo = l.partition_point(|&x| x < bb.min.0[a] - 1e-9 * (1.0 + bb.min.0[a].abs()));
            let hi = l.partition_point(|&x| x <= bb.max.0[a] + 1e-9 * (1.0 + bb.max.0[a].abs()));
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
                        // Highest priority first, ties in definition
                        // order: a material on top of every metal there.
                        let hits = model.properties_at(m, CoordSystem::Cartesian, mask);
                        let metal = hits
                            .iter()
                            .any(|h| matches!(h.property.kind, PropertyKind::Metal));
                        if metal && hits[0].property.kind.material().is_some() {
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

/// Clears the PEC flag of every lumped-element edge (see the
/// [module docs](self#lumped-elements-in-metal)); returns the number of
/// edges cleared.
pub(crate) fn lumped_over_metal(mg: &mut MaterialGrid) -> usize {
    let mut cleared = 0;
    for le in &mg.lumped {
        let pec = &mut mg.pec[le.edge.dir][le.edge.index];
        if *pec {
            *pec = false;
            cleared += 1;
        }
    }
    cleared
}

/// The model without its metal primitives, for openEMS's material
/// averages (see the [module docs](self#material-samples-in-metal)), or
/// `None` when the metal cannot change them: no metal primitive, or no
/// material primitive and a vacuum background.
pub(crate) fn without_metal(model: &ContinuousStructure) -> Option<ContinuousStructure> {
    let props = model.properties();
    let metal = props
        .iter()
        .any(|p| matches!(p.kind, PropertyKind::Metal) && !p.primitives.is_empty());
    let bg = &model.background;
    let material = props
        .iter()
        .any(|p| p.kind.material().is_some() && !p.primitives.is_empty())
        || bg.epsilon.value() != 1.0
        || bg.mue.value() != 1.0
        || bg.kappa.value() != 0.0
        || bg.sigma.value() != 0.0;
    if !(metal && material) {
        return None;
    }
    let mut m = model.clone();
    for p in m.properties_mut() {
        if matches!(p.kind, PropertyKind::Metal) {
            p.primitives.clear();
        }
    }
    Some(m)
}

/// openEMS's inside tolerance `1e-3 · h_min` in drawing units (see the
/// [module docs](self#inside-tolerance)); 0 without cells.
pub(crate) fn inside_tolerance(model: &ContinuousStructure) -> f64 {
    let h = (0..3)
        .flat_map(|a| model.grid.lines(a).windows(2).map(|w| (w[1] - w[0]).abs()))
        .filter(|d| *d > 0.0)
        .fold(f64::INFINITY, f64::min);
    if h.is_finite() { 1e-3 * h } else { 0.0 }
}

/// Smallest singular value of the linear part of `m`: the square root
/// of the smallest eigenvalue of `AᵀA` (cyclic Jacobi rotations).
fn sigma_min(m: &csx::Mat4) -> f64 {
    let mut s = [[0.0f64; 3]; 3];
    for (i, row) in s.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| m.0[k][i] * m.0[k][j]).sum();
        }
    }
    for _ in 0..32 {
        let off = s[0][1].abs() + s[0][2].abs() + s[1][2].abs();
        if off <= 1e-30 * (s[0][0].abs() + s[1][1].abs() + s[2][2].abs()) {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if s[p][q] == 0.0 {
                continue;
            }
            let th = (s[q][q] - s[p][p]) / (2.0 * s[p][q]);
            let t = if th == 0.0 {
                1.0
            } else {
                th.signum() / (th.abs() + (th * th + 1.0).sqrt())
            };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let sn = t * c;
            for row in &mut s {
                let (kp, kq) = (row[p], row[q]);
                row[p] = c * kp - sn * kq;
                row[q] = sn * kp + c * kq;
            }
            let (rp, rq) = (s[p], s[q]);
            for k in 0..3 {
                s[p][k] = c * rp[k] - sn * rq[k];
                s[q][k] = sn * rp[k] + c * rq[k];
            }
        }
    }
    (0..3)
        .map(|i| s[i][i])
        .fold(f64::INFINITY, f64::min)
        .max(0.0)
        .sqrt()
}

/// Grows the box with corners `start`, `stop` (any order) by `t[a]` on
/// every axis `a` where it has an extent.
fn grow_box(start: &mut ScalarPoint, stop: &mut ScalarPoint, t: [f64; 3]) {
    for (a, &ta) in t.iter().enumerate() {
        let (x0, x1) = (start.0[a].value(), stop.0[a].value());
        if x0 == x1 {
            continue;
        }
        let d = if x0 < x1 { ta } else { -ta };
        start.0[a].set(x0 - d);
        stop.0[a].set(x1 + d);
    }
}

/// Moves the end points `a`, `b` of an axis apart by `t` each.
fn extend_axis(a: &mut ScalarPoint, b: &mut ScalarPoint, t: f64) {
    let (pa, pb) = (a.value(), b.value());
    let d: [f64; 3] = core::array::from_fn(|i| pb.0[i] - pa.0[i]);
    let l = d.iter().map(|x| x * x).sum::<f64>().sqrt();
    if l > 0.0 {
        for (i, di) in d.iter().enumerate() {
            a.0[i].set(pa.0[i] - t * di / l);
            b.0[i].set(pb.0[i] + t * di / l);
        }
    }
}

/// The vertices of the polygon `v` offset outwards by `t` (each side
/// moved by `t` along its outward normal, consecutive sides intersected).
fn grow_polygon(v: &[(f64, f64)], t: f64) -> Vec<(f64, f64)> {
    let n = v.len();
    let area: f64 = (0..n)
        .map(|i| {
            let (a, b) = (v[i], v[(i + 1) % n]);
            a.0 * b.1 - b.0 * a.1
        })
        .sum();
    // Outward normal of the side i → i+1 (counter-clockwise: its right).
    let sign = if area >= 0.0 { 1.0 } else { -1.0 };
    let normal = |i: usize| -> Option<(f64, f64)> {
        let (a, b) = (v[i], v[(i + 1) % n]);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let l = dx.hypot(dy);
        (l > 0.0).then(|| (sign * dy / l, -sign * dx / l))
    };
    (0..n)
        .map(|i| {
            let prev = (1..n).map(|k| (i + n - k) % n).find_map(normal);
            let m = match (prev, normal(i)) {
                (Some(p), Some(q)) => {
                    let d = 1.0 + p.0 * q.0 + p.1 * q.1;
                    if d > 1e-6 {
                        ((p.0 + q.0) / d, (p.1 + q.1) / d)
                    } else {
                        q
                    }
                }
                (Some(p), None) => p,
                (None, Some(q)) => q,
                (None, None) => (0.0, 0.0),
            };
            (v[i].0 + t * m.0, v[i].1 + t * m.1)
        })
        .collect()
}

fn grow_vertices(v: &PolyVertices, t: f64) -> PolyVertices {
    let mut out = PolyVertices::new();
    for (u, w) in grow_polygon(v.values(), t) {
        out.push(u, w);
    }
    out
}

/// Grows `prim` by the tolerance `t` (drawing units, see the
/// [module docs](self#inside-tolerance)); false if its shape is not
/// handled (left as it is). Wires thinner than half the smallest cell
/// are not grown.
fn grow(prim: &mut Primitive, t: f64, default_cs: CoordSystem) -> bool {
    let t = if prim.transform.is_empty() {
        t
    } else {
        let s = sigma_min(prim.transform.matrix());
        if s > 0.0 {
            t / s
        } else {
            return false;
        }
    };
    let cyl = prim.effective_cs(default_cs) == CoordSystem::Cylindrical;
    match &mut prim.shape {
        Shape::Box(b) => {
            let ta = if cyl {
                let r = b.start.0[0].value().abs().max(b.stop.0[0].value().abs());
                if r > 0.0 { t / r } else { 0.0 }
            } else {
                t
            };
            grow_box(&mut b.start, &mut b.stop, [t, ta, t]);
        }
        Shape::MultiBox(m) if !cyl => {
            for (a, b) in &mut m.boxes {
                grow_box(a, b, [t; 3]);
            }
        }
        Shape::Sphere(s) => {
            let r = s.radius.value();
            s.radius.set(r + t);
        }
        Shape::SphericalShell(s) => {
            let w = s.shell_width.value();
            s.shell_width.set(w + 2.0 * t);
        }
        Shape::Cylinder(c) if !cyl => {
            let r = c.radius.value();
            c.radius.set(r + t);
            extend_axis(&mut c.start, &mut c.stop, t);
        }
        Shape::CylindricalShell(c) if !cyl => {
            let w = c.shell_width.value();
            c.shell_width.set(w + 2.0 * t);
            extend_axis(&mut c.start, &mut c.stop, t);
        }
        // Thin wires (below half the smallest cell) keep their radius: it
        // sets their in-cell correction (compile::wire).
        Shape::Wire(w) if !cyl && w.radius.value() >= 500.0 * t => {
            let r = w.radius.value();
            w.radius.set(r + t);
        }
        Shape::LinPoly(g) if !cyl => {
            g.vertices = grow_vertices(&g.vertices, t);
            let (e, l) = (g.elevation.value(), g.length.value());
            if l != 0.0 {
                g.elevation.set(e - t * l.signum());
                g.length.set(l + 2.0 * t * l.signum());
            }
        }
        _ => return false,
    }
    true
}

/// The model with openEMS's inside tolerance applied to the volume
/// primitives of metal, materials, excitations (not plane waves) and
/// conducting sheets (see the [module docs](self#inside-tolerance)), or
/// `None` if there is nothing to grow.
pub(crate) fn dilated(model: &ContinuousStructure) -> Option<ContinuousStructure> {
    let t = inside_tolerance(model);
    let applies = |k: &PropertyKind| match k {
        PropertyKind::Metal | PropertyKind::ConductingSheet(_) => true,
        PropertyKind::Excitation(e) => e.excitation_type != ExcitationType::PlaneWave,
        k => k.material().is_some(),
    };
    if t <= 0.0
        || !model
            .properties()
            .iter()
            .any(|p| applies(&p.kind) && !p.primitives.is_empty())
    {
        return None;
    }
    let cs = model.coord_system;
    let mut m = model.clone();
    for p in m.properties_mut() {
        if applies(&p.kind) {
            // Sheets and lines: csx selects their edges by overlap and
            // snapping, not by inside tests.
            for q in p.primitives.iter_mut().filter(|q| q.dimension() == 3) {
                grow(q, t, cs);
            }
        }
    }
    Some(m)
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
    fn tolerance_and_growth() {
        let s = model(Sphere::new([5.0, 5.0, 5.0], 2.5));
        assert_eq!(inside_tolerance(&s), 1e-3);
        let d = dilated(&s).unwrap();
        let Shape::Sphere(g) = &d.properties()[0].primitives[0].shape else {
            panic!("a sphere")
        };
        assert_eq!(g.radius.value(), 2.501);
        // Scale3 (4, 2, 2): the local tolerance is t / 2; rotations keep t.
        let mut t = csx::Transform::new();
        t.push(csx::transform::TransformOp::Scale3([
            4.0.into(),
            2.0.into(),
            2.0.into(),
        ]));
        assert!((sigma_min(t.matrix()) - 2.0).abs() < 1e-12);
        let mut r = csx::Transform::new();
        r.rotate(Vec3::new(1.0, 1.0, 0.0), 0.6);
        assert!((sigma_min(r.matrix()) - 1.0).abs() < 1e-12);
        // A wire of radius 0 is a line, a thin wire keeps its radius.
        for r in [0.0, 0.2] {
            let mut w = Primitive::new(Wire::new([[5.0, 5.0, 1.0], [5.0, 5.0, 9.0]], r));
            assert!(!grow(&mut w, 1e-3, CoordSystem::Cartesian));
        }
    }

    #[test]
    fn polygon_growth() {
        for v in [
            vec![(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)],
            vec![(0.0, 0.0), (0.0, 1.0), (2.0, 1.0), (2.0, 0.0)],
        ] {
            let g = grow_polygon(&v, 0.1);
            let lo = g
                .iter()
                .fold((f64::MAX, f64::MAX), |m, p| (m.0.min(p.0), m.1.min(p.1)));
            let hi = g
                .iter()
                .fold((f64::MIN, f64::MIN), |m, p| (m.0.max(p.0), m.1.max(p.1)));
            assert!(
                (lo.0 + 0.1).abs() < 1e-12 && (lo.1 + 0.1).abs() < 1e-12,
                "{g:?}"
            );
            assert!(
                (hi.0 - 2.1).abs() < 1e-12 && (hi.1 - 1.1).abs() < 1e-12,
                "{g:?}"
            );
        }
    }
}
