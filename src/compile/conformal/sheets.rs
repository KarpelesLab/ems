//! Geometry of conformal zero-thickness PEC sheets (the scheme is in
//! [`crate::engine::conformal`], design §5.13 "Thin sheets").
//!
//! # Which sheets
//!
//! Metal primitives of dimension 2 that csx cannot put on the PEC mask:
//! planar sheets off the mesh planes or oblique to them (zero-thickness
//! boxes, polygons and zero-length extrusions, any transformation) and
//! curved sheets (zero-width cylindrical and spherical shells). Sheets on
//! a mesh plane keep csx's exact edge mask. Other surfaces (open
//! polyhedra, boxes in cylindrical coordinates) are left to csx (which
//! warns that it cannot discretise them).
//!
//! # Edges
//!
//! Each sheet has a signed side function `φ` (plane: normal distance;
//! cylinder and sphere: squared distance from axis or centre minus `r²`),
//! evaluated at the nodes, so every node is on one side of every sheet
//! and the cut parity of all edges is consistent. The crossings of an
//! edge with the (unbounded) surface are the roots of `φ` along it
//! (linear or quadratic); a crossing is a **cut** if the point lies on the
//! sheet's actual extent (polygon, plate rectangle, cylinder length). Cuts
//! are kept at least `1e-6` of the edge from its ends and from each other.
//!
//! # Faces
//!
//! Every face around a cut edge is split by the sheet trace. For the
//! sheet with the most cuts on its edges, the perimeter is divided into
//! arcs by that sheet's crossings (cut or not); arcs alternate between the
//! two sides. The negative side (half-space, cylinder or sphere interior)
//! is convex, so its arcs form a single region, and each positive arc is a
//! region bounded by the trace between its two end crossings. A trace that
//! is not entirely on the sheet (a plate edge or hole crosses the face, a
//! crossing outside the extent) does not separate: its arc joins the
//! negative region (a slit). Areas are those of the polygons of crossing
//! points and corners with straight chords for the traces (exact for
//! planes, second order for curved sheets), as for solid metal. Pieces
//! (the parts of the four edges between cuts) belong to the region of the
//! arc containing their midpoint. Faces whose edges are cut by two or more
//! sheets use the partition of one of them (warning).

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};

use super::{par, range};

mod cyl;
use crate::axis::Axis;
use crate::engine::{SheetCutEdge, SheetCutFace, SheetRegion};
use csx::primitives::{Primitive, Shape};
use csx::{BoundingBox, CoordSystem, PropertyType, Transform, TypeMask, Vec3};
pub(super) use cyl::geometry as cylindrical_geometry;

/// Minimum distance of a cut from the edge ends and from other cuts
/// (fraction of the edge).
const GAP: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
enum Surf {
    /// `x[nrm] = val` in the local frame.
    Plane { nrm: usize, val: f64 },
    /// Cylinder mantle from `a` along unit `ax`, length `len`, radius `r`.
    Cyl { a: Vec3, ax: Vec3, len: f64, r: f64 },
    /// Sphere of centre `c`, radius `r`.
    Sph { c: Vec3, r: f64 },
}

struct Sheet<'a> {
    shape: Cow<'a, Shape>,
    transform: Option<&'a Transform>,
    surf: Surf,
    bbox: BoundingBox,
}

impl Sheet<'_> {
    fn local(&self, p: Vec3) -> Vec3 {
        self.transform.map_or(p, |t| t.apply_inverse(p))
    }

    fn phi(&self, q: Vec3) -> f64 {
        match self.surf {
            Surf::Plane { nrm, val } => q[nrm] - val,
            Surf::Cyl { a, ax, r, .. } => {
                let w = q - a;
                let perp = w - ax * w.dot(ax);
                perp.norm_sq() - r * r
            }
            Surf::Sph { c, r } => (q - c).norm_sq() - r * r,
        }
    }

    /// Whether the surface point `q` (local) is on the sheet.
    fn on_metal(&self, q: Vec3) -> bool {
        match self.surf {
            Surf::Plane { nrm, val } => {
                let mut q = q;
                q[nrm] = val;
                self.shape.contains_local(q, CoordSystem::Cartesian)
            }
            Surf::Cyl { a, ax, len, .. } => {
                let t = (q - a).dot(ax) / len;
                (-1e-9..=1.0 + 1e-9).contains(&t)
            }
            Surf::Sph { .. } => true,
        }
    }

    /// Nearest surface point to `q` (local).
    fn project(&self, q: Vec3) -> Vec3 {
        match self.surf {
            Surf::Plane { nrm, val } => {
                let mut q = q;
                q[nrm] = val;
                q
            }
            Surf::Cyl { a, ax, r, .. } => {
                let w = q - a;
                let along = w.dot(ax);
                let perp = w - ax * along;
                let d = perp.norm();
                if d > 0.0 {
                    a + ax * along + perp * (r / d)
                } else {
                    q
                }
            }
            Surf::Sph { c, r } => {
                let d = (q - c).norm();
                if d > 0.0 { c + (q - c) * (r / d) } else { q }
            }
        }
    }

    /// Side of the world point `p` (true: `φ ≥ 0`).
    fn side(&self, p: Vec3) -> bool {
        self.phi(self.local(p)) >= 0.0
    }

    /// Crossing parameters of the segment `p0 → p1` (world) with the
    /// unbounded surface, consistent with the end sides, and whether each
    /// crossing is on the sheet.
    fn crossings(&self, p0: Vec3, p1: Vec3) -> Vec<(f64, bool)> {
        let (q0, q1) = (self.local(p0), self.local(p1));
        let (f0, f1) = (self.phi(q0), self.phi(q1));
        let odd = (f0 >= 0.0) != (f1 >= 0.0);
        let d = q1 - q0;
        let mut ts: Vec<f64> = match self.surf {
            Surf::Plane { .. } => {
                if odd {
                    vec![f0 / (f0 - f1)]
                } else {
                    Vec::new()
                }
            }
            Surf::Cyl { a, ax, r, .. } => {
                let w = q0 - a;
                let wp = w - ax * w.dot(ax);
                let dp = d - ax * d.dot(ax);
                roots(dp.norm_sq(), 2.0 * wp.dot(dp), wp.norm_sq() - r * r)
            }
            Surf::Sph { c, r } => {
                let w = q0 - c;
                roots(d.norm_sq(), 2.0 * w.dot(d), w.norm_sq() - r * r)
            }
        };
        ts.retain(|t| (0.0..=1.0).contains(t));
        if (ts.len() % 2 == 1) != odd {
            // Rounding at a grazing or end-point root: bisect the side
            // function instead (one root if the ends differ, none else).
            ts.clear();
            if odd {
                let (mut lo, mut hi) = (0.0, 1.0);
                for _ in 0..60 {
                    let m = 0.5 * (lo + hi);
                    if (self.phi(q0 + d * m) >= 0.0) == (f0 >= 0.0) {
                        lo = m;
                    } else {
                        hi = m;
                    }
                }
                ts.push(0.5 * (lo + hi));
            }
        }
        ts.sort_by(f64::total_cmp);
        ts.into_iter()
            .map(|t| (t, self.on_metal(self.project(q0 + d * t))))
            .collect()
    }
}

/// Real roots of `a·t² + b·t + c` (none if degenerate).
fn roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a <= 0.0 {
        return Vec::new();
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return Vec::new();
    }
    let s = disc.sqrt();
    let q = -0.5 * (b + if b >= 0.0 { s } else { -s });
    if q == 0.0 {
        return vec![0.0];
    }
    let (r1, r2) = (q / a, c / q);
    vec![r1.min(r2), r1.max(r2)]
}

/// The surface model of a dimension-2 metal primitive, if supported.
fn surface<'a>(prim: &'a Primitive, default_cs: CoordSystem) -> Option<Sheet<'a>> {
    let scs = prim.shape_cs(default_cs);
    let shape = prim.cartesian_shape(default_cs);
    let surf = match &*shape {
        Shape::Box(b) if scs == CoordSystem::Cartesian => {
            let (a, c) = (b.start.value(), b.stop.value());
            let nrm = (0..3).find(|&d| a[d] == c[d])?;
            Surf::Plane { nrm, val: a[nrm] }
        }
        Shape::Polygon(p) => Surf::Plane {
            nrm: p.norm_dir,
            val: p.elevation.value(),
        },
        Shape::LinPoly(p) if p.length.value() == 0.0 => Surf::Plane {
            nrm: p.norm_dir,
            val: p.elevation.value(),
        },
        Shape::CylindricalShell(s) if s.shell_width.value() == 0.0 => {
            let (a, b) = (s.start.value(), s.stop.value());
            let len = (b - a).norm();
            if len <= 0.0 {
                return None;
            }
            Surf::Cyl {
                a,
                ax: (b - a) / len,
                len,
                r: s.radius.value(),
            }
        }
        Shape::SphericalShell(s) if s.shell_width.value() == 0.0 => Surf::Sph {
            c: s.center.value(),
            r: s.radius.value(),
        },
        _ => return None,
    };
    if let Surf::Plane { nrm, .. } = surf
        && nrm > 2
    {
        return None;
    }
    let transform = if prim.transform.is_empty() {
        None
    } else if prim.transform.is_invertible() {
        Some(&prim.transform)
    } else {
        return None;
    };
    Some(Sheet {
        shape,
        transform,
        surf,
        bbox: prim.bounding_box(default_cs),
    })
}

/// Whether csx maps the sheet with bounding box `b` onto the PEC mask
/// (flat in one axis and on a mesh plane).
fn on_mesh_plane(b: &BoundingBox, lines: [&[f64]; 3], snap: f64) -> bool {
    (0..3).any(|a| {
        let l = lines[a];
        let i = l.partition_point(|&x| x < b.min[a]).min(l.len() - 1);
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
            lo.min(hi)
        };
        let near = [i.saturating_sub(1), i]
            .into_iter()
            .min_by(|&p, &q| (l[p] - b.min[a]).abs().total_cmp(&(l[q] - b.min[a]).abs()))
            .unwrap_or(0);
        let tol = snap * cell(near) + 1e-12 * b.min[a].abs();
        b.max[a] - b.min[a] <= tol && (l[near] - b.min[a]).abs() <= tol
    })
}

/// A crossing of an edge: parameter, sheet, on the sheet.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cross {
    t: f64,
    sheet: usize,
    metal: bool,
}

/// Cut edges and faces of the conformal sheets of `model` (engine
/// layout); `snap` is csx's relative snapping tolerance. Warnings of csx
/// about the sheets handled here are removed from `warnings`.
pub(super) fn geometry(
    model: &csx::ContinuousStructure,
    snap: f64,
    threads: usize,
    warnings: &mut Vec<String>,
) -> (Vec<SheetCutEdge>, Vec<SheetCutFace>) {
    let lines = [0, 1, 2].map(|a| model.grid.lines(a));
    let dims = lines.map(<[f64]>::len);
    let mut sheets = Vec::new();
    for p in model.properties_of(TypeMask::of(PropertyType::Metal)) {
        for (pi, prim) in p.primitives.iter().enumerate() {
            if prim.dimension() != 2 {
                continue;
            }
            let b = prim.bounding_box(model.coord_system);
            if b.is_empty() || !(b.min.is_finite() && b.max.is_finite()) {
                continue;
            }
            if on_mesh_plane(&b, lines, snap) {
                continue;
            }
            let Some(s) = surface(prim, model.coord_system) else {
                continue;
            };
            let label = format!("property {:?} primitive #{pi} (", p.name);
            warnings.retain(|w| !(w.starts_with(&label) && w.contains("sheet")));
            sheets.push(s);
        }
    }
    if sheets.is_empty() || dims.iter().any(|&d| d < 2) {
        return (Vec::new(), Vec::new());
    }
    let bb = sheets
        .iter()
        .fold(BoundingBox::EMPTY, |acc, s| acc.union(&s.bbox));
    let r: [core::ops::Range<usize>; 3] =
        core::array::from_fn(|a| range(lines[a], bb.min[a], bb.max[a]));
    let node = |n: [usize; 3]| Vec3::new(lines[0][n[0]], lines[1][n[1]], lines[2][n[2]]);
    let eidx = |n: [usize; 3]| (n[0] * dims[1] + n[1]) * dims[2] + n[2];
    let (ry, rz) = (r[1].clone(), r[2].clone());
    // Crossings of every edge in range.
    type EdgeOut = ((usize, [usize; 3]), Vec<Cross>);
    let found: Vec<Vec<EdgeOut>> = par(threads, r[0].clone(), |i| {
        let mut v = Vec::new();
        for j in ry.clone() {
            for k in rz.clone() {
                let n = [i, j, k];
                for a in 0..3 {
                    if n[a] + 1 >= dims[a] {
                        continue;
                    }
                    let mut m = n;
                    m[a] += 1;
                    let (p0, p1) = (node(n), node(m));
                    let eb = BoundingBox::from_corners(p0, p1);
                    let mut cs = Vec::new();
                    for (s, sh) in sheets.iter().enumerate() {
                        let pad = 1e-9 * (p1[a] - p0[a]);
                        if !sh.bbox.expanded(pad).intersects(&eb) {
                            continue;
                        }
                        for (t, metal) in sh.crossings(p0, p1) {
                            cs.push(Cross { t, sheet: s, metal });
                        }
                    }
                    if !cs.is_empty() {
                        cs.sort_by(|x, y| x.t.total_cmp(&y.t));
                        v.push(((a, n), cs));
                    }
                }
            }
        }
        v
    });
    let crossings: HashMap<(usize, [usize; 3]), Vec<Cross>> = found.into_iter().flatten().collect();
    // Cuts.
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
            cut_edges.push(SheetCutEdge {
                axis: Axis::from_index(key.0).expect("axis"),
                node: key.1,
                cuts: c,
            });
        }
    }
    // Faces around cut edges.
    let face_ok = |f: usize, q: [usize; 3]| {
        let (b, c) = ((f + 1) % 3, (f + 2) % 3);
        q[b] + 1 < dims[b] && q[c] + 1 < dims[c]
    };
    let mut faces: BTreeSet<(usize, usize, [usize; 3])> = BTreeSet::new();
    for e in &cut_edges {
        let a = e.axis.index();
        for f in (0..3).filter(|&f| f != a) {
            let o = 3 - f - a;
            if face_ok(f, e.node) {
                faces.insert((f, eidx(e.node), e.node));
            }
            if e.node[o] > 0 {
                let mut q = e.node;
                q[o] -= 1;
                if face_ok(f, q) {
                    faces.insert((f, eidx(q), q));
                }
            }
        }
    }
    let mut multi = 0usize;
    let mut out_faces = Vec::with_capacity(faces.len());
    for &(f, _, n) in &faces {
        let axis = Axis::from_index(f).expect("axis");
        let (bb_, cc_) = axis.others();
        let (b, c) = (bb_.index(), cc_.index());
        let (mut nb, mut nc) = (n, n);
        nb[b] += 1;
        nc[c] += 1;
        let fe = [(b, n), (c, nb), (b, nc), (c, n)];
        let none: Vec<Cross> = Vec::new();
        let nocut: Vec<f64> = Vec::new();
        let xs: [&[Cross]; 4] = fe.map(|k| crossings.get(&k).unwrap_or(&none).as_slice());
        let cs: [&[f64]; 4] = fe.map(|k| cuts.get(&k).unwrap_or(&nocut).as_slice());
        // The sheet with the most cuts here.
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
            continue;
        };
        let sh = &sheets[s];
        let lb = lines[b][n[b] + 1] - lines[b][n[b]];
        let lc = lines[c][n[c] + 1] - lines[c][n[c]];
        let origin = node(n);
        let world = |(u, v): (f64, f64)| {
            let mut p = origin;
            p[b] += u;
            p[c] += v;
            p
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
        let regions = face_regions(&events, cs, sh.side(origin), lb, lc, &trace_metal);
        out_faces.push(SheetCutFace {
            axis,
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
    (cut_edges, out_faces)
}

/// Perimeter coordinate in `[0, 4)` of edge `q` at its own parameter `t`
/// (edges 2 and 3 are traversed backwards).
fn perim(q: usize, t: f64) -> f64 {
    q as f64 + if q < 2 { t } else { 1.0 - t }
}

/// Face coordinates of perimeter coordinate `p`.
fn uv(p: f64, lb: f64, lc: f64) -> (f64, f64) {
    let q = (p.floor() as usize).min(3);
    let s = p - q as f64;
    match q {
        0 => (s * lb, 0.0),
        1 => (lb, s * lc),
        2 => ((1.0 - s) * lb, lc),
        _ => (0.0, (1.0 - s) * lc),
    }
}

fn shoelace(poly: &[(f64, f64)]) -> f64 {
    let mut a2 = 0.0;
    for q in 0..poly.len() {
        let (x0, y0) = poly[q];
        let (x1, y1) = poly[(q + 1) % poly.len()];
        a2 += x0 * y1 - x1 * y0;
    }
    0.5 * a2.abs()
}

/// Regions of a face (see the [module docs](self#faces)): `events[q]` are
/// the crossings `(t, on sheet)` of the partitioning sheet on edge `q`
/// (perimeter order, edge parameters), `cuts[q]` the cuts of edge `q` (all
/// sheets), `start_positive` the side of the face's corner node,
/// `trace_metal(a, b)` whether the trace between the face points `a` and
/// `b` is on the sheet.
fn face_regions(
    events: &[Vec<(f64, bool)>; 4],
    cuts: [&[f64]; 4],
    start_positive: bool,
    lb: f64,
    lc: f64,
    trace_metal: &dyn Fn((f64, f64), (f64, f64)) -> bool,
) -> Vec<SheetRegion> {
    face_regions_with(
        events,
        cuts,
        start_positive,
        lb,
        lc,
        trace_metal,
        &shoelace,
        lb * lc,
    )
}

/// The area of a face polygon (face coordinates).
type AreaFn<'a> = dyn Fn(&[(f64, f64)]) -> f64 + 'a;

/// [`face_regions`] with the polygon area `area_of` (face coordinates) and
/// the face's total area `area` (a weighted area on cylindrical z faces).
#[allow(clippy::too_many_arguments)]
fn face_regions_with(
    events: &[Vec<(f64, bool)>; 4],
    cuts: [&[f64]; 4],
    start_positive: bool,
    lb: f64,
    lc: f64,
    trace_metal: &dyn Fn((f64, f64), (f64, f64)) -> bool,
    area_of: &AreaFn<'_>,
    area: f64,
) -> Vec<SheetRegion> {
    let mut ev: Vec<(f64, bool)> = Vec::new();
    for (q, e) in events.iter().enumerate() {
        for &(t, m) in e {
            ev.push((perim(q, t), m));
        }
    }
    ev.sort_by(|a, b| a.0.total_cmp(&b.0));
    let m = ev.len();
    // Arc index of perimeter coordinate p: arc j runs from ev[j] to
    // ev[j + 1] (arc m − 1 wraps through 0).
    let arc_of = |p: f64| {
        let j = ev.partition_point(|e| e.0 < p);
        if j == 0 { m - 1 } else { j - 1 }
    };
    let pieces_of = |q: usize| {
        let c = cuts[q];
        (0..=c.len()).map(move |k| {
            let lo = if k == 0 { 0.0 } else { c[k - 1] };
            let hi = if k == c.len() { 1.0 } else { c[k] };
            (k, perim(q, 0.5 * (lo + hi)))
        })
    };
    if m == 0 || m % 2 == 1 {
        // No partition (or an inconsistent one): a single region.
        let pieces = (0..4)
            .flat_map(|q| pieces_of(q).map(move |(k, _)| (q as u8, k as u8)))
            .collect();
        return vec![SheetRegion {
            fraction: 1.0,
            pieces,
            centroid: Some([0.5, 0.5]),
        }];
    }
    // Arc sides: arc m − 1 holds the start corner; sides alternate.
    let positive = |j: usize| (j % 2 == 1) == start_positive;
    // Region of each arc: 0 = the negative (convex) side, else own.
    let mut region = vec![0usize; m];
    let mut fractions = vec![0.0];
    // Unweighted polygon area and first moments in face-side units, for
    // the centroids (the negative region's is the complement's).
    let mut moments = vec![[0.0; 3]];
    for j in 0..m {
        let (p0, p1) = (ev[j].0, ev[(j + 1) % m].0);
        let p1w = if j + 1 == m { p1 + 4.0 } else { p1 };
        let corners: Vec<f64> = (0..8)
            .map(|k| k as f64)
            .filter(|&k| k > p0 && k < p1w)
            .collect();
        if positive(j) {
            let mut poly = vec![uv(p0, lb, lc)];
            poly.extend(corners.iter().map(|&k| uv(k % 4.0, lb, lc)));
            poly.push(uv(p1, lb, lc));
            let a = area_of(&poly) / area;
            let sep = ev[j].1 && ev[(j + 1) % m].1 && trace_metal(uv(p0, lb, lc), uv(p1, lb, lc));
            if sep {
                region[j] = fractions.len();
                fractions.push(a);
                let unit: Vec<(f64, f64)> = poly.iter().map(|&(u, v)| (u / lb, v / lc)).collect();
                moments.push(polygon_moments(&unit));
            }
        }
    }
    // The negative region: everything not in a separate positive region.
    let sep_total: f64 = fractions[1..].iter().sum();
    fractions[0] = (1.0 - sep_total).max(0.0);
    let mut rest = [1.0, 0.5, 0.5];
    for m in &moments[1..] {
        for (r, m) in rest.iter_mut().zip(m) {
            *r -= m;
        }
    }
    moments[0] = rest;
    let mut pieces: Vec<Vec<(u8, u8)>> = vec![Vec::new(); fractions.len()];
    for q in 0..4 {
        for (k, p) in pieces_of(q) {
            pieces[region[arc_of(p)]].push((q as u8, k as u8));
        }
    }
    fractions
        .into_iter()
        .zip(pieces)
        .zip(moments)
        .filter(|((_, p), _)| !p.is_empty())
        .map(|((fraction, pieces), m)| SheetRegion {
            fraction: fraction.clamp(0.0, 1.0),
            pieces,
            centroid: (m[0] > 1e-12)
                .then(|| [(m[1] / m[0]).clamp(0.0, 1.0), (m[2] / m[0]).clamp(0.0, 1.0)]),
        })
        .collect()
}

/// `[area, ∫x, ∫y]` of a simple polygon (either orientation).
fn polygon_moments(poly: &[(f64, f64)]) -> [f64; 3] {
    let (mut a2, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for q in 0..poly.len() {
        let (x0, y0) = poly[q];
        let (x1, y1) = poly[(q + 1) % poly.len()];
        let cr = x0 * y1 - x1 * y0;
        a2 += cr;
        cx += (x0 + x1) * cr;
        cy += (y0 + y1) * cr;
    }
    let sign = if a2 < 0.0 { -1.0 } else { 1.0 };
    [0.5 * a2 * sign, cx / 6.0 * sign, cy / 6.0 * sign]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn never(_: (f64, f64), _: (f64, f64)) -> bool {
        false
    }
    fn always(_: (f64, f64), _: (f64, f64)) -> bool {
        true
    }

    #[test]
    fn straight_cut_two_regions() {
        // Trace from (0.25, 0) on edge 0 to (0, 0.75) on edge 3 (edge 3 at
        // t = 0.75): the corner triangle 0.5·0.25·0.75 is on the start
        // side.
        let ev = [vec![(0.25, true)], vec![], vec![], vec![(0.75, true)]];
        let cuts: [&[f64]; 4] = [&[0.25], &[], &[], &[0.75]];
        let r = face_regions(&ev, cuts, true, 1.0, 1.0, &always);
        assert_eq!(r.len(), 2);
        let tri = 0.5 * 0.25 * 0.75;
        // Start corner positive: its arc (the last) is a positive region.
        let (neg, pos) = (&r[0], &r[1]);
        assert!((pos.fraction - tri).abs() < 1e-12, "{r:?}");
        assert!((neg.fraction - (1.0 - tri)).abs() < 1e-12);
        assert_eq!(pos.pieces, vec![(0, 0), (3, 0)]);
        // Centroids: the triangle's, and the complement's.
        let c = pos.centroid.unwrap();
        assert!(
            (c[0] - 0.25 / 3.0).abs() < 1e-12 && (c[1] - 0.25).abs() < 1e-12,
            "{c:?}"
        );
        let c = neg.centroid.unwrap();
        let want = [
            (0.5 - tri * 0.25 / 3.0) / (1.0 - tri),
            (0.5 - tri * 0.25) / (1.0 - tri),
        ];
        assert!((c[0] - want[0]).abs() < 1e-12 && (c[1] - want[1]).abs() < 1e-12);
        assert_eq!(neg.pieces, vec![(0, 1), (1, 0), (2, 0), (3, 1)]);
    }

    #[test]
    fn plate_edge_inside_face_is_a_slit() {
        let ev = [vec![(0.25, true)], vec![], vec![], vec![(0.75, false)]];
        let cuts: [&[f64]; 4] = [&[0.25], &[], &[], &[]];
        let r = face_regions(&ev, cuts, true, 1.0, 1.0, &always);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].fraction, 1.0);
        assert_eq!(r[0].pieces.len(), 5);
        let ev = [vec![(0.25, true)], vec![], vec![], vec![(0.75, true)]];
        let cuts: [&[f64]; 4] = [&[0.25], &[], &[], &[0.75]];
        let r = face_regions(&ev, cuts, true, 1.0, 1.0, &never);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn opposite_edges_trapezoid() {
        // Trace from edge 1 at v = 0.2 to edge 3 at v = 0.6 (edge 3 at
        // t = 0.6), lb = 2: the part below is a trapezoid of mean height
        // 0.4.
        let ev = [vec![], vec![(0.2, true)], vec![], vec![(0.6, true)]];
        let cuts: [&[f64]; 4] = [&[], &[0.2], &[], &[0.6]];
        let r = face_regions(&ev, cuts, false, 2.0, 1.0, &always);
        assert_eq!(r.len(), 2);
        // Start corner negative: arc 1 (edge 1 above the cut … edge 3
        // above the cut) is positive.
        assert!((r[0].fraction - 0.4).abs() < 1e-12, "{r:?}");
        assert!((r[1].fraction - 0.6).abs() < 1e-12, "{r:?}");
        assert_eq!(r[1].pieces, vec![(1, 1), (2, 0), (3, 1)]);
    }

    #[test]
    fn quadratic_roots() {
        let r = roots(1.0, -3.0, 2.0);
        assert!((r[0] - 1.0).abs() < 1e-15 && (r[1] - 2.0).abs() < 1e-15);
        assert!(roots(1.0, 0.0, 1.0).is_empty());
        assert!(roots(0.0, 1.0, 1.0).is_empty());
    }
}
