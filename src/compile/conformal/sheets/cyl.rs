//! Conformal thin PEC sheets on cylindrical grids (design §5.13.1,
//! §5.15).
//!
//! The sheets are the Cartesian surfaces of the [parent module](super)
//! (planes, cylinder mantles, spheres in any position); the grid's edges
//! and faces are those of the `(ρ, α, z)` lines, mapped to Cartesian
//! points `(ρ cos α, ρ sin α, z)`:
//!
//! * ρ and z edges are straight segments, crossed as on Cartesian grids;
//!   α edges are arcs, whose crossings are the sign changes of the side
//!   function `φ` at 16 samples along the arc, bisected (60 steps). Cut
//!   parameters are fractions of the parameter (= physical) length.
//! * Faces are split in their `(u, v)` parameter rectangle with straight
//!   chords between the crossings, as on Cartesian grids (second order for
//!   traces that are curved in parameter space; exact for a coaxial shell
//!   on a z face, whose trace is `ρ = r`). ρ and α faces have a constant
//!   metric (`ρ·dα·dz`, `dρ·dz`), so parameter area fractions are
//!   physical; a z face has the area element `ρ·dρ·dα`: its polygons are
//!   measured by `∮ ρ²/2 dα` (exact along straight parameter segments).
//! * Stored copies of a physical edge (the seam of a closed azimuth, the
//!   axis z edge) are cut alike; only the canonical copy is listed (the
//!   engine shares its pieces between the copies), and the faces around
//!   every copy are split.
//!
//! Sheets on a mesh surface of the cylindrical grid (`ρ`, `α` or `z` =
//! const within csx's snapping tolerance) keep csx's PEC mask (ems
//! re-selects them periodically, `compile::cylindrical`). Boxes given in
//! cylindrical coordinates are not handled (csx's handling and warnings
//! apply).

use super::*;

/// Cartesian point of the parameter point `q = (ρ, α, z)`.
fn cart(q: [f64; 3]) -> Vec3 {
    let (s, c) = q[1].sin_cos();
    Vec3::new(q[0] * c, q[0] * s, q[2])
}

/// Samples per α edge for the crossing search.
const ARC_SAMPLES: usize = 16;

/// Crossings `(t, on sheet)` of the α arc from parameter point `q0` with
/// span `da` (radians).
fn arc_crossings(sh: &Sheet<'_>, q0: [f64; 3], da: f64) -> Vec<(f64, bool)> {
    let at = |t: f64| {
        let mut q = q0;
        q[1] += t * da;
        sh.local(cart(q))
    };
    let phi = |t: f64| sh.phi(at(t));
    let mut out = Vec::new();
    let mut t0 = 0.0;
    let mut f0 = phi(0.0);
    for k in 1..=ARC_SAMPLES {
        let t1 = k as f64 / ARC_SAMPLES as f64;
        let f1 = phi(t1);
        if (f0 >= 0.0) != (f1 >= 0.0) {
            let (mut lo, mut hi) = (t0, t1);
            for _ in 0..60 {
                let m = 0.5 * (lo + hi);
                if (phi(m) >= 0.0) == (f0 >= 0.0) {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            let t = 0.5 * (lo + hi);
            out.push((t, sh.on_metal(sh.project(at(t)))));
        }
        t0 = t1;
        f0 = f1;
    }
    out
}

/// Whether csx maps the primitive with cylindrical bounding box `b` onto
/// its PEC mask (flat in `ρ`, `α` or `z` on a mesh line).
fn on_cyl_mesh_surface(b: &BoundingBox, lines: [&[f64]; 3], snap: f64) -> bool {
    let tau = core::f64::consts::TAU;
    (0..3).any(|a| {
        let l = lines[a];
        let cands: &[f64] = if a == 1 { &[0.0, tau, -tau] } else { &[0.0] };
        cands.iter().any(|&shift| {
            let v = b.min[a] + shift;
            let i = l.partition_point(|&x| x < v).min(l.len() - 1);
            let near = [i.saturating_sub(1), i]
                .into_iter()
                .min_by(|&p, &q| (l[p] - v).abs().total_cmp(&(l[q] - v).abs()))
                .unwrap_or(0);
            let cell = |i: usize| {
                let lo = if i > 0 {
                    l[i] - l[i - 1]
                } else {
                    f64::INFINITY
                };
                let hi = if i + 1 < l.len() {
                    l[i + 1] - l[i]
                } else {
                    f64::INFINITY
                };
                let h = lo.min(hi);
                if h.is_finite() { h } else { 1.0 }
            };
            let tol = snap * cell(near) + 1e-12 * v.abs();
            b.max[a] - b.min[a] <= tol && (l[near] - v).abs() <= tol
        })
    })
}

/// `∫∫ ρ dρ dα` of the polygon `poly` of `(ρ − ρ0, α − α0)` points with
/// straight sides.
fn rho_area(poly: &[(f64, f64)], rho0: f64) -> f64 {
    let mut s = 0.0;
    for q in 0..poly.len() {
        let (u0, v0) = poly[q];
        let (u1, v1) = poly[(q + 1) % poly.len()];
        let (r0, r1) = (rho0 + u0, rho0 + u1);
        // ∮ ρ²/2 dα along a segment linear in (ρ, α).
        s += (v1 - v0) * (r0 * r0 + r0 * r1 + r1 * r1) / 6.0;
    }
    s.abs()
}

/// Cut edges and faces of the conformal sheets of `model` on its
/// cylindrical grid (engine layout); see the [module docs](self).
pub(in crate::compile::conformal) fn geometry(
    model: &csx::ContinuousStructure,
    snap: f64,
    threads: usize,
    warnings: &mut Vec<String>,
) -> SheetGeometry {
    let lines = [0, 1, 2].map(|a| model.grid.lines(a));
    let dims = lines.map(<[f64]>::len);
    if dims.iter().any(|&d| d < 2) {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let tau = core::f64::consts::TAU;
    let al = lines[1];
    let closed = ((al[dims[1] - 1] - al[0]) - tau).abs() <= 1e-9 * tau;
    let axis = lines[0][0] == 0.0;
    let mut sheets = Vec::new();
    let mut cbox = BoundingBox::EMPTY;
    for p in model.properties_of(TypeMask::of(PropertyType::Metal)) {
        for (pi, prim) in p.primitives.iter().enumerate() {
            if prim.dimension() != 2 {
                continue;
            }
            let b = prim.bounding_box_in(CoordSystem::Cylindrical, model.coord_system);
            if b.is_empty() || !(b.min.is_finite() && b.max.is_finite()) {
                continue;
            }
            if on_cyl_mesh_surface(&b, lines, snap) {
                continue;
            }
            let Some(mut s) = surface(prim, model.coord_system) else {
                continue;
            };
            s.bbox = prim.bounding_box_in(CoordSystem::Cartesian, model.coord_system);
            if s.bbox.is_empty() || !(s.bbox.min.is_finite() && s.bbox.max.is_finite()) {
                continue;
            }
            s.set_tolerance();
            cbox = cbox.union(&s.bbox);
            let label = format!("property {:?} primitive #{pi} (", p.name);
            warnings.retain(|w| !(w.starts_with(&label) && w.contains("sheet")));
            sheets.push(s);
        }
    }
    if sheets.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    // ρ range of the Cartesian box (distance of its xy rectangle from the
    // axis), all α lines, the box's z range.
    let (x0, x1, y0, y1) = (cbox.min[0], cbox.max[0], cbox.min[1], cbox.max[1]);
    let rmin = 0.0f64.clamp(x0, x1).hypot(0.0f64.clamp(y0, y1));
    let rmax = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        .iter()
        .map(|&(x, y)| f64::hypot(x, y))
        .fold(0.0, f64::max);
    let r: [core::ops::Range<usize>; 3] = [
        range(lines[0], rmin, rmax),
        0..dims[1],
        range(lines[2], cbox.min[2], cbox.max[2]),
    ];
    let param = |n: [usize; 3]| [lines[0][n[0]], lines[1][n[1]], lines[2][n[2]]];
    let eidx = |n: [usize; 3]| (n[0] * dims[1] + n[1]) * dims[2] + n[2];
    let (ry, rz) = (r[1].clone(), r[2].clone());
    type EdgeOut = ((usize, [usize; 3]), Vec<Cross>, bool);
    let found: Vec<Vec<EdgeOut>> = par(threads, r[0].clone(), |i| {
        let mut v = Vec::new();
        for j in ry.clone() {
            for k in rz.clone() {
                let n = [i, j, k];
                for a in 0..3 {
                    if n[a] + 1 >= dims[a] {
                        continue;
                    }
                    // α edges on the axis have no length.
                    if a == 1 && axis && i == 0 {
                        continue;
                    }
                    let mut m = n;
                    m[a] += 1;
                    let (q0, q1) = (param(n), param(m));
                    let mut qm = q0;
                    qm[a] = 0.5 * (q0[a] + q1[a]);
                    let mut cs = Vec::new();
                    let mut inside = false;
                    for (s, sh) in sheets.iter().enumerate() {
                        let xs = if a == 1 {
                            // Bounding box of the arc: its chord box grown
                            // by the sagitta.
                            let (p0, p1) = (cart(q0), cart(q1));
                            let sag = q0[0] * (1.0 - (0.5 * (q1[1] - q0[1])).cos());
                            let mut mid = q0;
                            mid[1] = 0.5 * (q0[1] + q1[1]);
                            let eb = BoundingBox::from_corners(p0, p1)
                                .union(&BoundingBox::from_corners(cart(mid), cart(mid)))
                                .expanded(sag + 1e-9 * q0[0].max(1e-30));
                            if !sh.bbox.intersects(&eb) {
                                continue;
                            }
                            arc_crossings(sh, q0, q1[1] - q0[1])
                        } else {
                            let (p0, p1) = (cart(q0), cart(q1));
                            let pad = 1e-9 * (q1[a] - q0[a]);
                            let eb = BoundingBox::from_corners(p0, p1);
                            if !sh.bbox.expanded(pad).intersects(&eb) {
                                continue;
                            }
                            sh.crossings(p0, p1)
                        };
                        for (t, metal) in xs {
                            cs.push(Cross { t, sheet: s, metal });
                        }
                        inside |= sh.contains_edge(cart(q0), cart(qm), cart(q1));
                    }
                    if !cs.is_empty() || inside {
                        cs.sort_by(|x, y| x.t.total_cmp(&y.t));
                        v.push(((a, n), cs, inside));
                    }
                }
            }
        }
        v
    });
    let mut in_sheet = Vec::new();
    let crossings: HashMap<(usize, [usize; 3]), Vec<Cross>> = found
        .into_iter()
        .flatten()
        .filter_map(|(k, cs, inside)| {
            if inside {
                in_sheet.push(k);
            }
            (!cs.is_empty()).then_some((k, cs))
        })
        .collect();
    // The canonical copy of an edge (seam: α index 0; axis: α index 0).
    let canonical = |a: usize, n: [usize; 3]| -> bool {
        let seam = closed && a != 1 && n[1] == dims[1] - 1;
        let on_axis = axis && a == 2 && n[0] == 0 && n[1] != 0;
        !(seam || on_axis)
    };
    // The stored copies of an edge.
    let copies = |a: usize, n: [usize; 3]| -> Vec<[usize; 3]> {
        if axis && a == 2 && n[0] == 0 {
            return (0..dims[1])
                .map(|j| {
                    let mut q = n;
                    q[1] = j;
                    q
                })
                .collect();
        }
        if closed && a != 1 && (n[1] == 0 || n[1] == dims[1] - 1) {
            let (mut lo, mut hi) = (n, n);
            lo[1] = 0;
            hi[1] = dims[1] - 1;
            return vec![lo, hi];
        }
        vec![n]
    };
    let mut cut_edges: Vec<SheetCutEdge> = Vec::new();
    let mut cuts: HashMap<(usize, [usize; 3]), Vec<f64>> = HashMap::new();
    let mut keys: Vec<&(usize, [usize; 3])> = crossings.keys().collect();
    keys.sort_by_key(|&&(a, n)| (a, eidx(n)));
    for &key in &keys {
        let mut c: Vec<f64> = Vec::new();
        for x in &crossings[key] {
            if !x.metal {
                continue;
            }
            let t = x.t.clamp(GAP, 1.0 - GAP);
            if c.last().is_none_or(|&l| t - l >= GAP) && t <= 1.0 - GAP {
                c.push(t);
            }
        }
        if !c.is_empty() {
            cuts.insert(*key, c.clone());
            if canonical(key.0, key.1) {
                cut_edges.push(SheetCutEdge {
                    axis: Axis::from_index(key.0).expect("axis"),
                    node: key.1,
                    cuts: c,
                });
            }
        }
    }
    // Copies cut on one side only (rounding at the seam): use the
    // canonical copy's cuts for all of them.
    for e in &cut_edges {
        let a = e.axis.index();
        for c in copies(a, e.node) {
            cuts.insert((a, c), e.cuts.clone());
        }
    }
    let face_ok = |f: usize, q: [usize; 3]| {
        let (b, c) = ((f + 1) % 3, (f + 2) % 3);
        q[b] + 1 < dims[b] && q[c] + 1 < dims[c]
    };
    let mut faces: BTreeSet<(usize, usize, [usize; 3])> = BTreeSet::new();
    for e in &cut_edges {
        let a = e.axis.index();
        for node in copies(a, e.node) {
            for f in (0..3).filter(|&f| f != a) {
                let o = 3 - f - a;
                if face_ok(f, node) {
                    faces.insert((f, eidx(node), node));
                }
                if node[o] > 0 {
                    let mut q = node;
                    q[o] -= 1;
                    if face_ok(f, q) {
                        faces.insert((f, eidx(q), q));
                    }
                }
            }
        }
    }
    let mut multi = 0usize;
    let mut out_faces = Vec::with_capacity(faces.len());
    for &(f, _, n) in &faces {
        let fa = Axis::from_index(f).expect("axis");
        let (bb_, cc_) = fa.others();
        let (b, c) = (bb_.index(), cc_.index());
        let (mut nb, mut nc) = (n, n);
        nb[b] += 1;
        nc[c] += 1;
        let fe = [(b, n), (c, nb), (b, nc), (c, n)];
        let none: Vec<Cross> = Vec::new();
        let nocut: Vec<f64> = Vec::new();
        let xs: [&[Cross]; 4] = fe.map(|k| crossings.get(&k).unwrap_or(&none).as_slice());
        let cs: [&[f64]; 4] = fe.map(|k| cuts.get(&k).unwrap_or(&nocut).as_slice());
        let mut count: HashMap<usize, usize> = HashMap::new();
        for x in xs.iter().flat_map(|v| v.iter()).filter(|x| x.metal) {
            *count.entry(x.sheet).or_insert(0) += 1;
        }
        if count.len() > 1 {
            multi += 1;
        }
        let Some(s) = count
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
            .map(|(&s, _)| s)
        else {
            // The face's cut edges are cut on their other copies only.
            continue;
        };
        let sh = &sheets[s];
        let lb = lines[b][n[b] + 1] - lines[b][n[b]];
        let lc = lines[c][n[c] + 1] - lines[c][n[c]];
        let origin = param(n);
        let world = |(u, v): (f64, f64)| {
            let mut q = origin;
            q[b] += u;
            q[c] += v;
            cart(q)
        };
        let trace_metal = |p0: (f64, f64), p1: (f64, f64)| {
            [0.25, 0.5, 0.75].iter().all(|&w| {
                let q = sh.local(world((p0.0 + w * (p1.0 - p0.0), p0.1 + w * (p1.1 - p0.1))));
                sh.on_metal(sh.project(q))
            })
        };
        let events: [Vec<(f64, bool)>; 4] = core::array::from_fn(|q| {
            xs[q]
                .iter()
                .filter(|x| x.sheet == s)
                .map(|x| (x.t, x.metal))
                .collect()
        });
        let start = sh.side(cart(origin));
        let regions = if fa == Axis::Z {
            // (b, c) = (ρ, α): area element ρ dρ dα.
            let r0 = origin[0];
            let total = 0.5 * lc * ((r0 + lb) * (r0 + lb) - r0 * r0);
            face_regions_with(
                &events,
                cs,
                start,
                lb,
                lc,
                &trace_metal,
                &|p| rho_area(p, r0),
                total,
            )
        } else {
            face_regions(&events, cs, start, lb, lc, &trace_metal)
        };
        out_faces.push(SheetCutFace {
            axis: fa,
            node: n,
            regions,
        });
    }
    if multi > 0 {
        warnings.push(format!(
            "conformal PEC sheets: {multi} face(s) cut by more than one sheet use the \
             partition of one of them (small leaks at sheet junctions)"
        ));
    }
    // PEC mask overrides (see the Cartesian `geometry`), on every copy.
    let others = other_primitives(model, &sheets);
    let mut pec = Vec::new();
    for e in &cut_edges {
        let a = e.axis.index();
        let (q0, mut q1) = (param(e.node), param(e.node));
        q1[a] = lines[a][e.node[a] + 1];
        let mut qm = q0;
        qm[a] = 0.5 * (q0[a] + q1[a]);
        if !covered(model, &others, cart(q0), cart(q1), Some(cart(qm))) {
            for c in copies(a, e.node) {
                pec.push((e.axis, eidx(c), false));
            }
        }
    }
    for &(a, n) in &in_sheet {
        for c in copies(a, n) {
            pec.push((Axis::from_index(a).expect("axis"), eidx(c), true));
        }
    }
    (cut_edges, out_faces, pec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rho_weighted_area() {
        // The whole annular sector ρ0..ρ0+1 × 0..0.5.
        let poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)];
        let a = rho_area(&poly, 2.0);
        assert!((a - 0.5 * 0.5 * (9.0 - 4.0)).abs() < 1e-14, "{a}");
        // A triangle: ∫∫ ρ = centroid ρ × area for a linear weight.
        let tri = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let a = rho_area(&tri, 2.0);
        assert!((a - 0.5 * (2.0 + 1.0 / 3.0)).abs() < 1e-14, "{a}");
    }
}
