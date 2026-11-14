//! Outlines of sheets lying on mesh planes: open lengths of the in-plane
//! E edges and open areas of the in-plane cell faces that the outline of
//! a planar metal sheet or conducting sheet crosses (design §5.13.2; the
//! scheme is the Dey–Mittra one of [`crate::engine::conformal`]).
//!
//! # Which sheets
//!
//! Metal and `ConductingSheet` primitives of dimension 2 that csx puts on
//! a mesh plane (flat along one axis, within the snapping tolerance of a
//! mesh line), on Cartesian grids. csx takes an in-plane edge as covered
//! when its midpoint is on the sheet, so a slanted or curved outline is a
//! staircase. Sheets off the mesh planes are split instead (`sheets`).
//!
//! # Why the solid-metal scheme applies
//!
//! A sheet on a mesh plane does not need the split unknowns of an oblique
//! sheet: the in-plane edges and the faces of that plane lie *in* the
//! sheet's plane, where the field outside the sheet is single valued.
//! Faraday's law on the open part of such a face has the contour of the
//! open parts of its edges and of the outline, on which the tangential
//! `E` vanishes, so the Dey–Mittra form holds with the open lengths and
//! the open area (S. Dey, R. Mittra, IEEE Microwave Guided Wave Lett.
//! 7(9), 1997, for a surface in the plane of the face). The faces normal
//! to the plane keep their area: the sheet has no thickness.
//!
//! # Geometry
//!
//! All sheets of one plane are taken together (a point is covered when
//! any of them contains it; the point is tested at each primitive's own
//! coordinate along the normal, as csx does). Every in-plane edge near
//! them is sampled at 9 points; state changes are bisected and
//! transitions at the ends snapped as for solid metal. The open area of
//! an in-plane face is the polygon of its open corners and transition
//! points, with straight chords for the outline.
//!
//! By default ([`ConformalOutlines::Slanted`]) only transitions where the
//! outline is oblique to the mesh (more than about 0.6° from the normal
//! of the edge, tested by the cover on both sides of the transition half
//! a percent of the edge to either side of it) cut an edge. An edge
//! crossed by an axis-aligned outline only keeps csx's midpoint rule, so
//! axis-aligned sheets are discretised as without the option, whether
//! their outline is on mesh lines or between them (meshes made for the
//! staircase, with the thirds rule at strip edges, keep their meaning).
//! [`ConformalOutlines::All`] cuts those too.
//!
//! # Result
//!
//! * a cut edge of a metal sheet is taken off the PEC mask (unless
//!   another primitive may have put it there) and becomes a
//!   [`ConformalEdge`];
//! * a cut edge of a conducting sheet becomes a [`ConformalEdge`] as well
//!   and is a sheet edge (added if csx did not select it): the
//!   conducting-sheet operator puts its ladder on the covered part, in
//!   series with the open part ([`crate::engine::sheet`]);
//! * in-plane faces with a cut edge and an open area below the whole are
//!   [`ConformalFace`]s; the open parts of the edges of a face without
//!   open area under a metal sheet are PEC (as for solid metal).
//!
//! Edges and faces already cut by solid metal or by an oblique sheet are
//! left to those schemes. Where a metal sheet and a conducting sheet
//! cover parts of the same edge, the covered part is taken as the kind
//! found in the middle of its first covered interval.

use std::collections::{BTreeMap, HashMap, HashSet};

use csx::primitives::Primitive;
use csx::{BoundingBox, CoordSystem, PropertyKind, Vec3};

use super::super::ConformalOutlines;
use super::super::sheet::{local_cell, plane_line, shares};
use super::sheets::{covered, on_mesh_plane};
use super::{Profile, SAMPLES, face_fraction, open_centroid, par, range, refine};
use crate::axis::Axis;
use crate::engine::{ConformalEdge, ConformalFace, SheetEdge};

/// Cover bit of a metal sheet.
const METAL: u8 = 1;
/// Cover bit of a conducting sheet.
const SHEET: u8 = 2;
/// Offsets of the slant test, in edge lengths: along the edge and across
/// it.
pub(super) const SLANT_ALONG: f64 = 5e-5;
pub(super) const SLANT_ACROSS: f64 = 5e-3;

/// An edge `(axis index, node)` or a face `(normal index, node)`.
pub(super) type Key = (usize, [usize; 3]);

/// What the outlines add to the conformal geometry.
#[derive(Default)]
pub(super) struct Outlines {
    /// Cut edges.
    pub edges: Vec<ConformalEdge>,
    /// Cut faces with the centroids of their open areas.
    pub faces: Vec<(ConformalFace, [f64; 2])>,
    /// `(axis, engine index, pec)` overrides of the PEC mask.
    pub pec: Vec<(Axis, usize, bool)>,
    /// Conducting-sheet edges csx did not select (cut edges whose midpoint
    /// is open), with their property names.
    pub sheet_edges: Vec<(SheetEdge, String)>,
    /// `(axis, node, open fraction)` of the cut conducting-sheet edges.
    pub sheet_cuts: Vec<(Axis, [usize; 3], f64)>,
    /// `(normal, node)` of the cut faces in the planes with such edges.
    pub sheet_cut_faces: Vec<(Axis, [usize; 3])>,
    /// Edges a conducting sheet covers whole, next to its cut edges:
    /// closed for the small faces' interpolation and merging
    /// ([`ConformalPec::push_wall`](crate::engine::ConformalPec::push_wall)).
    pub walls: Vec<(Axis, [usize; 3])>,
}

/// One sheet primitive on a mesh plane.
struct Plate<'a> {
    prim: &'a Primitive,
    bbox: BoundingBox,
    pad: f64,
    /// The primitive's own coordinate along the normal.
    at: f64,
    /// Conducting sheet: name, conductivity, thickness.
    sheet: Option<(&'a str, f64, f64)>,
}

/// The sheets of one mesh plane.
struct Plane<'a> {
    nrm: usize,
    cs: CoordSystem,
    plates: Vec<Plate<'a>>,
}

impl<'a> Plane<'a> {
    fn plates_at(&self, p: Vec3) -> impl Iterator<Item = &Plate<'a>> {
        let nrm = self.nrm;
        let cs = self.cs;
        self.plates.iter().filter(move |q| {
            let inside = (0..3).all(|a| {
                a == nrm || (p[a] >= q.bbox.min[a] - q.pad && p[a] <= q.bbox.max[a] + q.pad)
            });
            inside && {
                let mut x = p;
                x.0[nrm] = q.at;
                q.prim.contains_in(x, CoordSystem::Cartesian, cs)
            }
        })
    }

    /// Cover bits at the in-plane point `p`.
    fn cover(&self, p: Vec3) -> u8 {
        let mut bits = 0;
        for q in self.plates_at(p) {
            bits |= if q.sheet.is_some() { SHEET } else { METAL };
            if bits == METAL | SHEET {
                break;
            }
        }
        bits
    }

    /// The conducting sheet at `p`.
    fn sheet_at(&self, p: Vec3) -> Option<(&'a str, f64, f64)> {
        self.plates_at(p).find_map(|q| q.sheet)
    }
}

/// The mesh plane `(normal, line)` csx puts a planar primitive with
/// bounding box `b` on (its rule: flat along exactly one axis, on a line).
fn mesh_plane(
    model: &csx::ContinuousStructure,
    b: &BoundingBox,
    snap: f64,
) -> Option<(usize, usize)> {
    let tol = |a: usize, v: f64| {
        let l = model.grid.lines(a);
        model
            .grid
            .nearest_line(a, v)
            .map_or(0.0, |i| snap * local_cell(l, i))
            + 1e-12 * v.abs()
    };
    let flat: Vec<usize> = (0..3)
        .filter(|&a| b.max[a] - b.min[a] <= tol(a, b.min[a]))
        .collect();
    let [nrm] = flat[..] else {
        return None;
    };
    plane_line(model, nrm, b.min[nrm], snap).map(|k| (nrm, k))
}

/// The planes with sheets, by `(normal, line)`.
fn planes<'a>(
    model: &'a csx::ContinuousStructure,
    snap: f64,
    conducting: bool,
) -> BTreeMap<(usize, usize), Plane<'a>> {
    let lines = [0, 1, 2].map(|a| model.grid.lines(a));
    let mut out: BTreeMap<(usize, usize), Plane<'a>> = BTreeMap::new();
    for p in model.properties() {
        let sheet = match &p.kind {
            PropertyKind::Metal => None,
            PropertyKind::ConductingSheet(c) if conducting => {
                let (s, t) = (c.conductivity.value(), c.thickness.value());
                if !(s.is_finite() && s > 0.0 && t.is_finite() && t > 0.0) {
                    continue;
                }
                Some((p.name.as_str(), s, t))
            }
            _ => continue,
        };
        for prim in &p.primitives {
            if prim.dimension() != 2 {
                continue;
            }
            let bbox = prim.bounding_box(model.coord_system);
            if bbox.is_empty() || !(bbox.min.is_finite() && bbox.max.is_finite()) {
                continue;
            }
            if !on_mesh_plane(&bbox, lines, snap) {
                continue;
            }
            let Some((nrm, k)) = mesh_plane(model, &bbox, snap) else {
                continue;
            };
            out.entry((nrm, k))
                .or_insert_with(|| Plane {
                    nrm,
                    cs: model.coord_system,
                    plates: Vec::new(),
                })
                .plates
                .push(Plate {
                    prim,
                    bbox,
                    pad: 1e-9 * (bbox.max - bbox.min).norm(),
                    at: bbox.min[nrm],
                    sheet,
                });
        }
    }
    out
}

/// Middle of the first covered interval of `p`.
fn first_covered(p: &Profile) -> f64 {
    let mut metal = p.start_metal;
    let mut last = 0.0;
    for &t in p.transitions.iter().chain(core::iter::once(&1.0)) {
        if metal {
            return 0.5 * (last + t);
        }
        last = t;
        metal = !metal;
    }
    0.5
}

/// An edge touched by the sheets of a plane.
struct Touched {
    key: Key,
    /// Cover along the edge.
    profile: Profile,
    /// Whether the edge is cut (false: it keeps csx's midpoint rule).
    cut: bool,
    /// Cover bits of its covered part.
    kind: u8,
    /// The conducting sheet of its covered part.
    sheet: Option<(String, f64, f64)>,
}

/// The primitives the caller's schemes already cut, and the conducting
/// sheet edges that exist.
pub(super) struct Taken<'a> {
    /// Edges cut by solid metal or oblique sheets.
    pub edges: &'a HashSet<Key>,
    /// Faces cut by solid metal or oblique sheets.
    pub faces: &'a HashSet<Key>,
    /// Edges csx selected for a conducting sheet.
    pub sheet_edges: &'a HashSet<Key>,
}

/// Cut edges and faces of the outlines of the sheets on mesh planes of
/// `model` (Cartesian grid, engine layout). `pec` is the PEC mask so far;
/// `conducting`: conducting sheets too (not for openEMS's sheet model).
pub(super) fn geometry(
    model: &csx::ContinuousStructure,
    pec: &[Vec<bool>; 3],
    snap: f64,
    threads: usize,
    mode: ConformalOutlines,
    conducting: bool,
    taken: &Taken<'_>,
) -> Outlines {
    let mut out = Outlines::default();
    if mode == ConformalOutlines::Off || model.grid.coord_system != CoordSystem::Cartesian {
        return out;
    }
    let lines = [0, 1, 2].map(|a| model.grid.lines(a));
    let dims = lines.map(<[f64]>::len);
    if dims.iter().any(|&d| d < 2) {
        return out;
    }
    let planes = planes(model, snap, conducting);
    if planes.is_empty() {
        return out;
    }
    // Primitives other than these sheets that can put an edge on the PEC
    // mask.
    let others: Vec<(&Primitive, bool)> = model
        .properties()
        .iter()
        .filter(|p| {
            matches!(
                p.kind,
                PropertyKind::Metal | PropertyKind::ConductingSheet(_)
            )
        })
        .flat_map(|p| {
            let metal = matches!(p.kind, PropertyKind::Metal);
            p.primitives
                .iter()
                .map(move |q| (q, metal && q.dimension() == 3))
        })
        .filter(|(q, _)| {
            !planes
                .values()
                .any(|pl| pl.plates.iter().any(|s| core::ptr::eq(s.prim, *q)))
        })
        .collect();
    let point = |n: [usize; 3]| Vec3::new(lines[0][n[0]], lines[1][n[1]], lines[2][n[2]]);
    let eidx = |n: [usize; 3]| (n[0] * dims[1] + n[1]) * dims[2] + n[2];
    for (&(nrm, k), plane) in &planes {
        let axis = Axis::from_index(nrm).expect("axis in 0..3");
        let (ab, ac) = axis.others();
        let (b, c) = (ab.index(), ac.index());
        let bb = plane
            .plates
            .iter()
            .fold(BoundingBox::EMPTY, |acc, s| acc.union(&s.bbox));
        let (rb, rc) = (
            range(lines[b], bb.min[b], bb.max[b]),
            range(lines[c], bb.min[c], bb.max[c]),
        );
        let node = |i: usize, j: usize| {
            let mut n = [0; 3];
            n[nrm] = k;
            n[b] = i;
            n[c] = j;
            n
        };
        let states: Vec<Vec<bool>> = par(threads, rb.clone(), |i| {
            rc.clone()
                .map(|j| plane.cover(point(node(i, j))) != 0)
                .collect()
        });
        let state = |n: [usize; 3]| {
            rb.contains(&n[b]) && rc.contains(&n[c]) && states[n[b] - rb.start][n[c] - rc.start]
        };
        let touched: Vec<Vec<Touched>> = par(threads, rb.clone(), |i| {
            let mut v = Vec::new();
            for j in rc.clone() {
                let n = node(i, j);
                for (d, o) in [(b, c), (c, b)] {
                    if n[d] + 1 >= dims[d] {
                        continue;
                    }
                    let mut m = n;
                    m[d] += 1;
                    let p0 = point(n);
                    let len = lines[d][n[d] + 1] - lines[d][n[d]];
                    let at = |t: f64, across: f64| {
                        let mut p = p0;
                        p.0[d] += t * len;
                        p.0[o] += across * len;
                        p
                    };
                    let cov = |t: f64| plane.cover(at(t, 0.0)) != 0;
                    let mut s = [false; SAMPLES + 1];
                    for (q, s) in s.iter_mut().enumerate() {
                        *s = match q {
                            0 => state(n),
                            SAMPLES => state(m),
                            q => cov(q as f64 / SAMPLES as f64),
                        };
                    }
                    if !s.iter().any(|&x| x) {
                        continue;
                    }
                    let mid = s[SAMPLES / 2];
                    // An outline running along the edge: the cover on the
                    // edge itself is a matter of rounding; csx's midpoint
                    // decides.
                    let along = !s.iter().all(|&x| x)
                        && [0.25, 0.5, 0.75].iter().all(|&t| {
                            (plane.cover(at(t, 1e-6)) != 0) != (plane.cover(at(t, -1e-6)) != 0)
                        });
                    let profile = if along {
                        Profile::uniform(mid)
                    } else {
                        refine(&s, cov)
                    };
                    if !profile.start_metal && profile.transitions.is_empty() {
                        continue;
                    }
                    // A transition cuts the edge where the outline is
                    // oblique to the mesh (module docs).
                    let slanted = |t: f64| {
                        let side = |dt: f64, across: f64| plane.cover(at(t + dt, across)) != 0;
                        [SLANT_ACROSS, -SLANT_ACROSS].iter().any(|&w| {
                            side(-SLANT_ALONG, w) != side(-SLANT_ALONG, 0.0)
                                || side(SLANT_ALONG, w) != side(SLANT_ALONG, 0.0)
                        })
                    };
                    let cut = !profile.transitions.is_empty()
                        && (mode == ConformalOutlines::All
                            || profile.transitions.iter().any(|&t| slanted(t)));
                    let t0 = first_covered(&profile);
                    let kind = plane.cover(at(t0, 0.0));
                    let sheet = (kind & METAL == 0)
                        .then(|| plane.sheet_at(at(t0, 0.0)))
                        .flatten()
                        .map(|(name, s, t)| (name.to_string(), s, t));
                    v.push(Touched {
                        key: (d, n),
                        profile,
                        cut,
                        kind,
                        sheet,
                    });
                }
            }
            v
        });
        let touched: Vec<Touched> = touched.into_iter().flatten().collect();
        let by_key: HashMap<Key, usize> = touched
            .iter()
            .enumerate()
            .map(|(q, t)| (t.key, q))
            .collect();
        // Edges to cut, in order: index into `touched`.
        let mut cuts: Vec<usize> = Vec::new();
        for (q, t) in touched.iter().enumerate() {
            let (d, n) = t.key;
            if !t.cut || taken.edges.contains(&t.key) {
                continue;
            }
            let f = t.profile.open_fraction();
            if f <= 0.0 || f >= 1.0 {
                continue;
            }
            let idx = eidx(n);
            if t.kind & METAL != 0 {
                let mut m = n;
                m[d] += 1;
                if pec[d][idx] && covered(model, &others, point(n), point(m), None) {
                    continue;
                }
            } else if pec[d][idx] || t.sheet.is_none() {
                continue;
            }
            cuts.push(q);
        }
        let is_cut: HashSet<Key> = cuts.iter().map(|&q| touched[q].key).collect();
        let conducting = cuts.iter().any(|&q| touched[q].kind & METAL == 0);
        if conducting {
            for t in &touched {
                if t.kind & METAL == 0 && t.profile.start_metal && t.profile.transitions.is_empty()
                {
                    let (d, n) = t.key;
                    out.walls
                        .push((Axis::from_index(d).expect("axis in 0..3"), n));
                }
            }
        }
        // Faces of the plane with a cut edge, and the cut edges of faces
        // without open area under metal (PEC, as for solid metal).
        let mut surface: HashSet<Key> = HashSet::new();
        let profile = |key: &Key| {
            by_key
                .get(key)
                .map_or_else(|| Profile::uniform(false), |&q| touched[q].profile.clone())
        };
        for i in rb.clone() {
            for j in rc.clone() {
                let n = node(i, j);
                if n[b] + 1 >= dims[b] || n[c] + 1 >= dims[c] {
                    continue;
                }
                let (mut nb, mut nc) = (n, n);
                nb[b] += 1;
                nc[c] += 1;
                let keys = [(b, n), (c, nb), (b, nc), (c, n)];
                if !keys.iter().any(|key| by_key.contains_key(key)) {
                    continue;
                }
                let lb = lines[b][n[b] + 1] - lines[b][n[b]];
                let lc = lines[c][n[c] + 1] - lines[c][n[c]];
                let e = [
                    profile(&keys[0]),
                    profile(&keys[1]),
                    profile(&keys[2]),
                    profile(&keys[3]),
                ];
                let fa = face_fraction(e.clone(), lb, lc);
                if fa <= 0.0 {
                    let mut centre = point(n);
                    centre.0[b] += 0.5 * lb;
                    centre.0[c] += 0.5 * lc;
                    if plane.cover(centre) & METAL != 0 {
                        surface.extend(keys);
                    }
                    continue;
                }
                if fa < 1.0
                    && keys.iter().any(|key| is_cut.contains(key))
                    && !taken.faces.contains(&(nrm, n))
                {
                    if conducting {
                        out.sheet_cut_faces.push((axis, n));
                    }
                    out.faces.push((
                        ConformalFace {
                            axis,
                            node: n,
                            fraction: fa,
                        },
                        open_centroid(&e, lb, lc),
                    ));
                }
            }
        }
        for q in cuts {
            let t = &touched[q];
            let (d, n) = t.key;
            let ax = Axis::from_index(d).expect("axis in 0..3");
            if surface.contains(&t.key) {
                // Its open part lies in the metal.
                if t.kind & METAL != 0 {
                    out.pec.push((ax, eidx(n), true));
                }
                continue;
            }
            if t.kind & METAL != 0 {
                out.pec.push((ax, eidx(n), false));
            } else if let Some((name, s, th)) = &t.sheet {
                out.sheet_cuts.push((ax, n, t.profile.open_fraction()));
                if taken.sheet_edges.contains(&t.key) {
                    // csx selected it.
                } else {
                    out.sheet_edges.push((
                        SheetEdge {
                            axis: ax,
                            node: n,
                            normal: axis,
                            conductivity: *s,
                            thickness: *th,
                            upper_share: shares(model, d, n, nrm),
                        },
                        name.clone(),
                    ));
                }
            }
            out.edges.push(ConformalEdge {
                axis: ax,
                node: n,
                fraction: t.profile.open_fraction(),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use csx::primitives::Polygon;
    use csx::{ContinuousStructure, Property};

    /// 9 × 9 × 3 lines of unit cells with one polygon (metal or conducting
    /// sheet) on the plane z = 1.
    fn model(verts: &[(f64, f64)], conducting: bool) -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        let l: Vec<f64> = (0..9).map(f64::from).collect();
        m.grid.set_lines(0, l.clone()).unwrap();
        m.grid.set_lines(1, l).unwrap();
        m.grid.set_lines(2, vec![0.0, 1.0, 2.0]).unwrap();
        let p = m.add_property(if conducting {
            Property::conducting_sheet("s", 5.8e7, 35e-6)
        } else {
            Property::metal("s")
        });
        m.add_primitive(p, Polygon::new(2, 1.0, verts.to_vec()))
            .unwrap();
        m
    }

    fn run(m: &ContinuousStructure, mode: ConformalOutlines) -> Outlines {
        let n = 9 * 9 * 3;
        let pec = [vec![false; n], vec![false; n], vec![false; n]];
        let none = HashSet::new();
        geometry(
            m,
            &pec,
            1e-6,
            2,
            mode,
            true,
            &Taken {
                edges: &none,
                faces: &none,
                sheet_edges: &none,
            },
        )
    }

    /// Covered: `y ≤ x − d` (a 45° outline through the mesh).
    fn diagonal(d: f64) -> Vec<(f64, f64)> {
        vec![(d - 20.0, -20.0), (20.0, -20.0), (20.0, 20.0 - d)]
    }

    #[test]
    fn axis_aligned_outlines_are_cut_on_request_only() {
        let m = model(&[(1.3, 2.2), (5.6, 2.2), (5.6, 4.7), (1.3, 4.7)], false);
        let o = run(&m, ConformalOutlines::Slanted);
        assert!(o.edges.is_empty() && o.faces.is_empty() && o.pec.is_empty());
        let o = run(&m, ConformalOutlines::Off);
        assert!(o.edges.is_empty() && o.faces.is_empty() && o.pec.is_empty());
        let o = run(&m, ConformalOutlines::All);
        // x edges across x = 1.3 and 5.6 on the rows y = 3, 4; y edges
        // across y = 2.2 and 4.7 on the columns x = 2…5.
        assert_eq!(o.edges.len(), 2 * 2 + 2 * 4);
        let f = |axis: Axis, node: [usize; 3]| {
            o.edges
                .iter()
                .find(|e| e.axis == axis && e.node == node)
                .map(|e| e.fraction)
        };
        assert!((f(Axis::X, [1, 3, 1]).unwrap() - 0.3).abs() < 1e-6);
        assert!((f(Axis::X, [5, 4, 1]).unwrap() - 0.4).abs() < 1e-6);
        assert!((f(Axis::Y, [3, 2, 1]).unwrap() - 0.2).abs() < 1e-6);
        assert!((f(Axis::Y, [3, 4, 1]).unwrap() - 0.3).abs() < 1e-6);
        // Faces along a side: the strip of the cell outside it.
        let fa = |node: [usize; 3]| {
            o.faces
                .iter()
                .find(|(f, _)| f.axis == Axis::Z && f.node == node)
                .map(|(f, _)| f.fraction)
        };
        assert!((fa([3, 2, 1]).unwrap() - 0.2).abs() < 1e-6);
        assert!((fa([1, 3, 1]).unwrap() - 0.3).abs() < 1e-6);
        assert!(o.sheet_cuts.is_empty() && o.walls.is_empty());
    }

    #[test]
    fn diagonal_outline_fractions() {
        let m = model(&diagonal(0.4), false);
        let o = run(&m, ConformalOutlines::Slanted);
        // x edges on row j are covered from x = j + 0.4, y edges on
        // column i up to y = i − 0.4: open 0.4 of each cut edge.
        assert!(!o.edges.is_empty());
        for e in &o.edges {
            assert!((e.fraction - 0.4).abs() < 1e-6, "{e:?}");
            assert_eq!(e.node[2], 1);
            match e.axis {
                Axis::X => assert_eq!(e.node[0], e.node[1]),
                Axis::Y => assert_eq!(e.node[0], e.node[1] + 1),
                Axis::Z => panic!("{e:?}"),
            }
            assert!(
                o.pec
                    .contains(&(e.axis, (e.node[0] * 9 + e.node[1]) * 3 + 1, false))
            );
        }
        // Cells on the diagonal lose a corner with legs 0.6, the cells
        // below it keep one with legs 0.4.
        for (f, c) in &o.faces {
            let want = if f.node[0] == f.node[1] {
                1.0 - 0.18
            } else {
                assert_eq!(f.node[0], f.node[1] + 1);
                0.08
            };
            assert!((f.fraction - want).abs() < 1e-6, "{f:?}");
            assert!(c.iter().all(|t| (0.0..=1.0).contains(t)));
        }
        assert_eq!(o.faces.len(), 8 + 7);
    }

    #[test]
    fn conducting_sheet_edges() {
        // Covered from x = j + 0.7 on the x edges: their midpoints are
        // open, so csx does not select them.
        let m = model(&diagonal(0.7), true);
        let o = run(&m, ConformalOutlines::Slanted);
        assert!(!o.sheet_cuts.is_empty());
        assert_eq!(o.sheet_cuts.len(), o.edges.len());
        assert_eq!(o.sheet_edges.len(), o.edges.len());
        assert!(o.pec.is_empty());
        for ((e, name), c) in o.sheet_edges.iter().zip(&o.edges) {
            assert_eq!((e.axis, e.node, e.normal), (c.axis, c.node, Axis::Z));
            assert_eq!((e.conductivity, e.thickness), (5.8e7, 35e-6));
            assert_eq!(name, "s");
            assert!((c.fraction - 0.7).abs() < 1e-6);
        }
        assert_eq!(o.sheet_cut_faces.len(), o.faces.len());
        // Whole edges of the sheet are walls: e.g. the x edge from
        // (5, 2) and the y edge from (5, 2).
        assert!(o.walls.contains(&(Axis::X, [5, 2, 1])));
        assert!(o.walls.contains(&(Axis::Y, [5, 2, 1])));
        assert!(
            !o.walls
                .iter()
                .any(|w| o.edges.iter().any(|e| (e.axis, e.node) == *w))
        );
        // Without conducting sheets (openEMS's sheet model) nothing.
        let n = 9 * 9 * 3;
        let pec = [vec![false; n], vec![false; n], vec![false; n]];
        let none = HashSet::new();
        let taken = Taken {
            edges: &none,
            faces: &none,
            sheet_edges: &none,
        };
        let o = geometry(&m, &pec, 1e-6, 1, ConformalOutlines::Slanted, false, &taken);
        assert!(o.edges.is_empty() && o.sheet_cuts.is_empty());
    }
}
