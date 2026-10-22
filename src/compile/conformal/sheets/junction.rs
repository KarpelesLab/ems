//! Faces cut by several sheets, or by sheets and solid metal (design
//! §5.13.1, "Junctions").
//!
//! The partition of [`face_regions`](super::face_regions) follows one
//! sheet: where a second sheet crosses the face, or a sheet ends on
//! another sheet or on a solid-metal surface inside the face (T and L
//! junctions, a septum touching a wall), the trace of the ending sheet is
//! not entirely on it, so it did not separate and the two sides shared the
//! face's current: a gap of under a cell through which the field leaked.
//!
//! Here the face is divided by the planar arrangement of all its traces:
//! the four sides (split at the cut points and at the solid-metal
//! transitions), every sheet's chord between paired crossings restricted
//! to the part on the sheet's extent (sampled and bisected, so a plate
//! ending inside the face gives a segment ending there), and the chords of
//! the solid metal's open polygon. A segment end within `1e-6` of the face
//! size from another segment is moved onto it (a septum ending on a wall,
//! two plates meeting at a corner); a sheet whose extent ends on another
//! sheet's surface or on the solid metal (tested on the surfaces, not the
//! chords) is extended along its trace to that surface's chord, at most a
//! quarter of the face away (the chords of curved traces are off the
//! surface by their sagitta). Segments are then split at their
//! intersections, ends left dangling are pruned (plate edges inside the
//! face, which do not separate), and the bounded faces of the remaining
//! planar graph are the candidate regions. Those inside the solid metal
//! (classified by a point next to their longest side against the open
//! polygon) are dropped. Each piece belongs to the region holding the
//! longest open part of it; areas are polygon areas (`area_of`), centroids
//! those of the polygons in face-side units.

use super::{SheetRegion, polygon_moments};
use crate::compile::conformal::Profile;
use crate::compile::conformal::open_polygon;

/// A sheet crossing a face, for [`partition`].
pub(super) struct FaceTrace<'a> {
    /// Its crossings `(t, on sheet)` on the four edges (perimeter order,
    /// edge parameters).
    pub events: [Vec<(f64, bool)>; 4],
    /// Side of the face's corner node.
    pub start_positive: bool,
    /// Whether the face point `(u, v)` lies on the sheet (its trace there
    /// is on the sheet's extent).
    pub on_sheet: &'a dyn Fn((f64, f64)) -> bool,
}

/// Chord samples for the extent search (then bisected).
const CHORD_SAMPLES: usize = 16;

/// A face of the planar graph: its polygon and its half-edges.
type Cycle = (Vec<(f64, f64)>, Vec<usize>);

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    /// Side `q` of the face (perimeter order).
    Side(usize),
    /// A chord of trace `k` ([`METAL`]: of the solid metal).
    Inner(usize),
}

/// The [`Kind::Inner`] tag of the solid metal's chords.
const METAL: usize = usize::MAX;

/// Farthest a sheet ending on another surface is extended to reach that
/// surface's chord (fraction of the face size): the chords of curved
/// traces are up to their sagitta (`h²/8R`) off the surface.
const REACH: f64 = 0.25;

#[derive(Debug, Clone)]
struct Seg {
    a: (f64, f64),
    b: (f64, f64),
    kind: Kind,
    /// Split parameters in `[0, 1]`.
    splits: Vec<f64>,
}

impl Seg {
    fn at(&self, t: f64) -> (f64, f64) {
        if t <= 0.0 {
            self.a
        } else if t >= 1.0 {
            self.b
        } else {
            (
                self.a.0 + t * (self.b.0 - self.a.0),
                self.a.1 + t * (self.b.1 - self.a.1),
            )
        }
    }

    /// Distance from `p` and the parameter of the nearest point.
    fn nearest(&self, p: (f64, f64)) -> (f64, f64) {
        let d = (self.b.0 - self.a.0, self.b.1 - self.a.1);
        let l2 = d.0 * d.0 + d.1 * d.1;
        let t = if l2 > 0.0 {
            (((p.0 - self.a.0) * d.0 + (p.1 - self.a.1) * d.1) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q = self.at(t);
        ((p.0 - q.0).hypot(p.1 - q.1), t)
    }
}

/// Edge parameter of face point `p` on side `q` (as in
/// [`face_regions`](super::face_regions): sides 0 and 2 along `b`, 1 and 3
/// along `c`).
fn side_param(q: usize, p: (f64, f64), lb: f64, lc: f64) -> f64 {
    match q {
        0 | 2 => p.0 / lb,
        _ => p.1 / lc,
    }
}

/// Face point of edge parameter `t` on side `q`.
fn side_point(q: usize, t: f64, lb: f64, lc: f64) -> (f64, f64) {
    match q {
        0 => (t * lb, 0.0),
        1 => (lb, t * lc),
        2 => (t * lb, lc),
        _ => (0.0, t * lc),
    }
}

/// Whether the open polygon `poly` (even–odd rule) contains `p`.
fn contains(poly: &[(f64, f64)], p: (f64, f64)) -> bool {
    let mut inside = false;
    for q in 0..poly.len() {
        let (a, b) = (poly[q], poly[(q + 1) % poly.len()]);
        if (a.1 > p.1) != (b.1 > p.1) {
            let x = a.0 + (p.1 - a.1) / (b.1 - a.1) * (b.0 - a.0);
            if p.0 < x {
                inside = !inside;
            }
        }
    }
    inside
}

/// The regions of a face cut by `traces` (and solid metal with the edge
/// `profiles`, if any); see the [module docs](self). `cuts[q]` are the cut
/// parameters of edge `q` (all sheets), `lb`, `lc` the side lengths,
/// `area_of` the area of a face polygon and `area` the face's;
/// `ends_on(k, p, d)` tells whether trace `k`'s extent ending at face point
/// `p` (outward direction `d`) ends on another sheet or on the solid metal.
#[allow(clippy::too_many_arguments)]
pub(super) fn partition(
    traces: &[FaceTrace<'_>],
    cuts: [&[f64]; 4],
    profiles: Option<&[Profile; 4]>,
    lb: f64,
    lc: f64,
    area_of: &super::AreaFn<'_>,
    area: f64,
    ends_on: &dyn Fn(usize, (f64, f64), (f64, f64)) -> bool,
) -> Vec<SheetRegion> {
    let size = lb.max(lc);
    let (tol_v, tol_snap) = (1e-9 * size, 1e-6 * size);
    let open_poly = profiles.map(|e| open_polygon(e, lb, lc));
    let mut segs: Vec<Seg> = (0..4)
        .map(|q| {
            let (a, b) = (side_point(q, 0.0, lb, lc), side_point(q, 1.0, lb, lc));
            let mut splits: Vec<f64> = cuts[q].to_vec();
            if let Some(e) = profiles {
                splits.extend(&e[q].transitions);
            }
            Seg {
                a,
                b,
                kind: Kind::Side(q),
                splits,
            }
        })
        .collect();
    let on_side = |p: (f64, f64), q: usize| match q {
        0 => p.1.abs() <= tol_v,
        1 => (p.0 - lb).abs() <= tol_v,
        2 => (p.1 - lc).abs() <= tol_v,
        _ => p.0.abs() <= tol_v,
    };
    let inner = |a: (f64, f64), b: (f64, f64), tag: usize, segs: &mut Vec<Seg>| {
        let along = (0..4).any(|q| on_side(a, q) && on_side(b, q));
        if !along && (a.0 - b.0).hypot(a.1 - b.1) > tol_v {
            segs.push(Seg {
                a,
                b,
                kind: Kind::Inner(tag),
                splits: Vec::new(),
            });
        }
    };
    // Sheet chords, restricted to the sheets' extent.
    for (tag, tr) in traces.iter().enumerate() {
        let mut ev: Vec<(f64, bool, usize, f64)> = Vec::new();
        for (q, e) in tr.events.iter().enumerate() {
            for &(t, m) in e {
                ev.push((super::perim(q, t), m, q, t));
            }
        }
        ev.sort_by(|a, b| a.0.total_cmp(&b.0));
        let m = ev.len();
        if m == 0 || m % 2 == 1 {
            continue;
        }
        // The cut point of a crossing on the sheet (cuts are moved off the
        // edge ends and merged within 1e-6).
        let end = |x: &(f64, bool, usize, f64)| -> (f64, f64) {
            let c = cuts[x.2];
            let t = c
                .iter()
                .copied()
                .min_by(|a, b| (a - x.3).abs().total_cmp(&(b - x.3).abs()))
                .filter(|_| x.1)
                .unwrap_or(x.3);
            side_point(x.2, t, lb, lc)
        };
        let positive = |j: usize| (j % 2 == 1) == tr.start_positive;
        for j in (0..m).filter(|&j| positive(j)) {
            let (x0, x1) = (&ev[j], &ev[(j + 1) % m]);
            let (p0, p1) = (end(x0), end(x1));
            let at = |s: f64| (p0.0 + s * (p1.0 - p0.0), p0.1 + s * (p1.1 - p0.1));
            let flag = |k: usize| match k {
                0 => x0.1,
                k if k == CHORD_SAMPLES => x1.1,
                k => (tr.on_sheet)(at(k as f64 / CHORD_SAMPLES as f64)),
            };
            let mut start: Option<f64> = None;
            let mut prev = flag(0);
            if prev {
                start = Some(0.0);
            }
            for k in 1..=CHORD_SAMPLES {
                let f = flag(k);
                if f != prev {
                    let (mut lo, mut hi) = (
                        (k - 1) as f64 / CHORD_SAMPLES as f64,
                        k as f64 / CHORD_SAMPLES as f64,
                    );
                    for _ in 0..40 {
                        let mid = 0.5 * (lo + hi);
                        if (tr.on_sheet)(at(mid)) == prev {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    let s = 0.5 * (lo + hi);
                    if f {
                        start = Some(s);
                    } else if let Some(s0) = start.take() {
                        inner(at(s0), at(s), tag, &mut segs);
                    }
                }
                prev = f;
            }
            if let Some(s0) = start {
                inner(if s0 == 0.0 { p0 } else { at(s0) }, p1, tag, &mut segs);
            }
        }
    }
    // Chords of the solid metal (open polygon sides off the face sides).
    if let Some(poly) = &open_poly {
        for q in 0..poly.len() {
            inner(poly[q], poly[(q + 1) % poly.len()], METAL, &mut segs);
        }
    }
    // Ends of inner segments near another segment are moved onto it; a
    // sheet ending on another surface is extended to that surface's chord.
    for i in 0..segs.len() {
        let Kind::Inner(tag) = segs[i].kind else {
            continue;
        };
        for which in 0..2 {
            let p = if which == 0 { segs[i].a } else { segs[i].b };
            if (0..4).any(|q| on_side(p, q)) {
                // On a side: split it there.
                for (q, side) in segs.iter_mut().enumerate().take(4) {
                    if on_side(p, q) {
                        side.splits.push(side_param(q, p, lb, lc).clamp(0.0, 1.0));
                    }
                }
                continue;
            }
            let mut best: Option<(f64, usize, f64)> = None;
            for (j, s) in segs.iter().enumerate() {
                if j == i {
                    continue;
                }
                let (d, t) = s.nearest(p);
                if d < tol_snap && best.is_none_or(|b| d < b.0) {
                    best = Some((d, j, t));
                }
            }
            if best.is_none() && tag != METAL {
                let (o, e) = if which == 0 {
                    (segs[i].b, segs[i].a)
                } else {
                    (segs[i].a, segs[i].b)
                };
                let len = (e.0 - o.0).hypot(e.1 - o.1);
                let d = ((e.0 - o.0) / len, (e.1 - o.1) / len);
                if ends_on(tag, p, d) {
                    // The nearest chord of another trace ahead.
                    for (j, s) in segs.iter().enumerate() {
                        let Kind::Inner(other) = s.kind else {
                            continue;
                        };
                        if other == tag {
                            continue;
                        }
                        let r = (s.b.0 - s.a.0, s.b.1 - s.a.1);
                        let den = d.0 * r.1 - d.1 * r.0;
                        if den.abs() <= 1e-300 {
                            continue;
                        }
                        let w = (s.a.0 - p.0, s.a.1 - p.1);
                        let dist = (w.0 * r.1 - w.1 * r.0) / den;
                        let t = (w.0 * d.1 - w.1 * d.0) / den;
                        if (0.0..=REACH * size).contains(&dist)
                            && (0.0..=1.0).contains(&t)
                            && best.is_none_or(|b| dist < b.0)
                        {
                            best = Some((dist, j, t));
                        }
                    }
                }
            }
            if let Some((_, j, t)) = best {
                let q = segs[j].at(t);
                segs[j].splits.push(t);
                if which == 0 {
                    segs[i].a = q;
                } else {
                    segs[i].b = q;
                }
            }
        }
    }
    // Intersections.
    for i in 0..segs.len() {
        for j in i + 1..segs.len() {
            if !matches!(segs[i].kind, Kind::Inner(_)) && !matches!(segs[j].kind, Kind::Inner(_)) {
                continue;
            }
            let (a, b) = (&segs[i], &segs[j]);
            let r = (a.b.0 - a.a.0, a.b.1 - a.a.1);
            let s = (b.b.0 - b.a.0, b.b.1 - b.a.1);
            let den = r.0 * s.1 - r.1 * s.0;
            if den.abs() <= 1e-300 {
                continue;
            }
            let w = (b.a.0 - a.a.0, b.a.1 - a.a.1);
            let t = (w.0 * s.1 - w.1 * s.0) / den;
            let u = (w.0 * r.1 - w.1 * r.0) / den;
            let (ea, eb) = (
                tol_v / r.0.hypot(r.1).max(1e-300),
                tol_v / s.0.hypot(s.1).max(1e-300),
            );
            if t >= -ea && t <= 1.0 + ea && u >= -eb && u <= 1.0 + eb {
                let (t, u) = (t.clamp(0.0, 1.0), u.clamp(0.0, 1.0));
                segs[i].splits.push(t);
                segs[j].splits.push(u);
            }
        }
    }
    // Vertices and graph edges.
    let mut verts: Vec<(f64, f64)> = Vec::new();
    let mut vid = |p: (f64, f64)| -> usize {
        if let Some(k) = verts
            .iter()
            .position(|v| (v.0 - p.0).hypot(v.1 - p.1) <= tol_v)
        {
            return k;
        }
        verts.push(p);
        verts.len() - 1
    };
    let mut edges: Vec<(usize, usize, Kind)> = Vec::new();
    for s in &segs {
        let mut ts: Vec<f64> = s.splits.clone();
        ts.push(0.0);
        ts.push(1.0);
        ts.sort_by(f64::total_cmp);
        let ids: Vec<usize> = ts.iter().map(|&t| vid(s.at(t))).collect();
        for w in ids.windows(2) {
            let (x, y) = (w[0], w[1]);
            if x != y
                && !edges
                    .iter()
                    .any(|e| (e.0 == x && e.1 == y) || (e.0 == y && e.1 == x))
            {
                edges.push((x, y, s.kind));
            }
        }
    }
    // Prune dangling inner edges.
    loop {
        let mut deg = vec![0usize; verts.len()];
        for e in &edges {
            deg[e.0] += 1;
            deg[e.1] += 1;
        }
        let before = edges.len();
        edges.retain(|e| !matches!(e.2, Kind::Inner(_)) || (deg[e.0] > 1 && deg[e.1] > 1));
        if edges.len() == before {
            break;
        }
    }
    // Faces: half-edge h = 2e (e.0 → e.1) or 2e + 1 (back); at each vertex
    // the outgoing half-edges by angle.
    let from = |h: usize| {
        if h.is_multiple_of(2) {
            edges[h / 2].0
        } else {
            edges[h / 2].1
        }
    };
    let to = |h: usize| {
        if h.is_multiple_of(2) {
            edges[h / 2].1
        } else {
            edges[h / 2].0
        }
    };
    let angle = |h: usize| {
        let (p, q) = (verts[from(h)], verts[to(h)]);
        (q.1 - p.1).atan2(q.0 - p.0)
    };
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); verts.len()];
    for h in 0..2 * edges.len() {
        out[from(h)].push(h);
    }
    for o in &mut out {
        o.sort_by(|&a, &b| angle(a).total_cmp(&angle(b)).then(a.cmp(&b)));
    }
    // Next half-edge of the face on the left: at the head, the outgoing
    // half-edge just clockwise of the twin.
    let next = |h: usize| -> usize {
        let v = to(h);
        let twin = h ^ 1;
        let o = &out[v];
        let k = o.iter().position(|&x| x == twin).expect("twin");
        o[(k + o.len() - 1) % o.len()]
    };
    let mut seen = vec![false; 2 * edges.len()];
    // (Polygon, half-edges) of the open bounded faces.
    let mut regions: Vec<Cycle> = Vec::new();
    for h0 in 0..2 * edges.len() {
        if seen[h0] {
            continue;
        }
        let mut cyc = Vec::new();
        let mut h = h0;
        while !seen[h] {
            seen[h] = true;
            cyc.push(h);
            h = next(h);
        }
        let poly: Vec<(f64, f64)> = cyc.iter().map(|&h| verts[from(h)]).collect();
        let mut a2 = 0.0;
        for q in 0..poly.len() {
            let (p0, p1) = (poly[q], poly[(q + 1) % poly.len()]);
            a2 += p0.0 * p1.1 - p1.0 * p0.1;
        }
        if a2 <= 1e-14 * lb * lc {
            continue; // the outer face (or degenerate)
        }
        if let Some(op) = &open_poly {
            // A point just inside, next to the longest side.
            let &hl = cyc
                .iter()
                .max_by(|&&x, &&y| {
                    let l = |h: usize| {
                        let (p, q) = (verts[from(h)], verts[to(h)]);
                        (q.0 - p.0).hypot(q.1 - p.1)
                    };
                    l(x).total_cmp(&l(y))
                })
                .expect("a side");
            let (p, q) = (verts[from(hl)], verts[to(hl)]);
            let (dx, dy) = (q.0 - p.0, q.1 - p.1);
            let len = dx.hypot(dy);
            let eps = 1e-7 * size;
            let probe = (
                0.5 * (p.0 + q.0) - dy / len * eps,
                0.5 * (p.1 + q.1) + dx / len * eps,
            );
            if !contains(op, probe) {
                continue;
            }
        }
        regions.push((poly, cyc));
    }
    // Pieces: the region with the longest open part of each.
    let pieces_on = |q: usize| cuts[q].len() + 1;
    let mut best: Vec<Vec<Option<(f64, usize)>>> =
        (0..4).map(|q| vec![None; pieces_on(q)]).collect();
    for (r, (_, cyc)) in regions.iter().enumerate() {
        for &h in cyc {
            let Kind::Side(q) = edges[h / 2].2 else {
                continue;
            };
            let (p, w) = (verts[from(h)], verts[to(h)]);
            let mid = side_param(q, (0.5 * (p.0 + w.0), 0.5 * (p.1 + w.1)), lb, lc);
            let k = cuts[q].iter().filter(|&&c| c < mid).count();
            let len = (w.0 - p.0).hypot(w.1 - p.1);
            if best[q][k].is_none_or(|b| len > b.0) {
                best[q][k] = Some((len, r));
            }
        }
    }
    let mut pieces: Vec<Vec<(u8, u8)>> = vec![Vec::new(); regions.len().max(1)];
    for (q, b) in best.iter().enumerate() {
        for (k, x) in b.iter().enumerate() {
            // Pieces inside the metal (no open region) go to the first.
            let r = x.map_or(0, |x| x.1);
            pieces[r].push((q as u8, k as u8));
        }
    }
    if regions.is_empty() {
        return vec![SheetRegion {
            fraction: 0.0,
            pieces: pieces.swap_remove(0),
            centroid: None,
        }];
    }
    regions
        .into_iter()
        .zip(pieces)
        .filter(|(_, p)| !p.is_empty())
        .map(|((poly, _), mut pieces)| {
            pieces.sort_unstable();
            let unit: Vec<(f64, f64)> = poly.iter().map(|&(u, v)| (u / lb, v / lc)).collect();
            let m = polygon_moments(&unit);
            SheetRegion {
                fraction: (area_of(&poly) / area).clamp(0.0, 1.0),
                pieces,
                centroid: (m[0] > 1e-12)
                    .then(|| [(m[1] / m[0]).clamp(0.0, 1.0), (m[2] / m[0]).clamp(0.0, 1.0)]),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::{face_regions, shoelace};
    use super::*;

    fn always(_: (f64, f64)) -> bool {
        true
    }

    fn never(_: usize, _: (f64, f64), _: (f64, f64)) -> bool {
        false
    }

    fn same(a: &[SheetRegion], b: &[SheetRegion]) -> bool {
        let key = |r: &SheetRegion| r.pieces.clone();
        let mut a: Vec<&SheetRegion> = a.iter().collect();
        let mut b: Vec<&SheetRegion> = b.iter().collect();
        a.sort_by_key(|r| key(r));
        b.sort_by_key(|r| key(r));
        a.len() == b.len()
            && a.iter().zip(&b).all(|(x, y)| {
                x.pieces == y.pieces
                    && (x.fraction - y.fraction).abs() < 1e-12
                    && match (x.centroid, y.centroid) {
                        (Some(p), Some(q)) => {
                            (p[0] - q[0]).abs() < 1e-9 && (p[1] - q[1]).abs() < 1e-9
                        }
                        (None, None) => true,
                        _ => false,
                    }
            })
    }

    #[test]
    fn one_sheet_matches_the_arc_partition() {
        type Case = ([Vec<(f64, bool)>; 4], bool);
        let cases: Vec<Case> = vec![
            (
                [vec![(0.25, true)], vec![], vec![], vec![(0.75, true)]],
                true,
            ),
            (
                [vec![], vec![(0.2, true)], vec![], vec![(0.6, true)]],
                false,
            ),
            ([vec![(0.3, true)], vec![], vec![(0.8, true)], vec![]], true),
            (
                [vec![], vec![(0.4, true)], vec![(0.1, true)], vec![]],
                false,
            ),
        ];
        for (ev, start) in cases {
            let cuts: [Vec<f64>; 4] = core::array::from_fn(|q| ev[q].iter().map(|e| e.0).collect());
            let cuts: [&[f64]; 4] = core::array::from_fn(|q| cuts[q].as_slice());
            let old = face_regions(&ev, cuts, start, 2.0, 1.0, &|_, _| true);
            let tr = FaceTrace {
                events: ev.clone(),
                start_positive: start,
                on_sheet: &always,
            };
            let new = partition(&[tr], cuts, None, 2.0, 1.0, &shoelace, 2.0, &never);
            assert!(same(&old, &new), "{old:?}\n{new:?}");
        }
    }

    #[test]
    fn t_junction_separates() {
        // Wall: the diagonal from (0, 0.2) on side 3 to (1, 0.7) on side 1.
        // Septum: from (0.5, 1) on side 2 down to the wall at (0.5, 0.45),
        // ending there (its plane continues to (0.5, 0) on side 0).
        let wall_y = |u: f64| 0.2 + 0.5 * u;
        let septum = |p: (f64, f64)| p.1 >= wall_y(p.0) - 1e-12;
        let wall = FaceTrace {
            events: [vec![], vec![(0.7, true)], vec![], vec![(0.2, true)]],
            start_positive: false,
            on_sheet: &always,
        };
        let sep = FaceTrace {
            events: [vec![(0.5, false)], vec![], vec![(0.5, true)], vec![]],
            start_positive: false,
            on_sheet: &septum,
        };
        let cuts: [&[f64]; 4] = [&[], &[0.7], &[0.5], &[0.2]];
        let r = partition(&[wall, sep], cuts, None, 1.0, 1.0, &shoelace, 1.0, &never);
        assert_eq!(r.len(), 3, "{r:?}");
        let total: f64 = r.iter().map(|x| x.fraction).sum();
        assert!((total - 1.0).abs() < 1e-12, "{r:?}");
        // Below the wall: 0.45; above it, left and right of the septum.
        let below = r.iter().find(|x| x.pieces.contains(&(0, 0))).unwrap();
        assert!((below.fraction - 0.45).abs() < 1e-9, "{r:?}");
        let left = r.iter().find(|x| x.pieces.contains(&(2, 0))).unwrap();
        let right = r.iter().find(|x| x.pieces.contains(&(2, 1))).unwrap();
        assert!(!core::ptr::eq(left, right));
        // Left: trapezoid u in [0, 0.5], from y = 0.2..0.45 up to 1.
        assert!(
            (left.fraction - 0.5 * (0.8 + 0.55) * 0.5).abs() < 1e-9,
            "{r:?}"
        );
        assert!(
            (right.fraction - 0.5 * (0.55 + 0.3) * 0.5).abs() < 1e-9,
            "{r:?}"
        );
        // The septum alone (no wall) is a slit: one region.
        let sep = FaceTrace {
            events: [vec![(0.5, false)], vec![], vec![(0.5, true)], vec![]],
            start_positive: false,
            on_sheet: &septum,
        };
        let cuts: [&[f64]; 4] = [&[], &[], &[0.5], &[]];
        let r = partition(&[sep], cuts, None, 1.0, 1.0, &shoelace, 1.0, &never);
        assert_eq!(r.len(), 1, "{r:?}");
    }

    #[test]
    fn sheet_ending_on_a_curved_trace_is_extended() {
        // The wall's chord runs 0.05 below where the septum ends (its
        // sagitta, say): reached only because the septum ends on the wall.
        let wall_y = |u: f64| 0.2 + 0.5 * u + 0.05;
        let septum = |p: (f64, f64)| p.1 >= wall_y(p.0) - 1e-12;
        let traces = || {
            [
                FaceTrace {
                    events: [vec![], vec![(0.7, true)], vec![], vec![(0.2, true)]],
                    start_positive: false,
                    on_sheet: &always,
                },
                FaceTrace {
                    events: [vec![(0.5, false)], vec![], vec![(0.5, true)], vec![]],
                    start_positive: false,
                    on_sheet: &septum,
                },
            ]
        };
        let cuts: [&[f64]; 4] = [&[], &[0.7], &[0.5], &[0.2]];
        let r = partition(&traces(), cuts, None, 1.0, 1.0, &shoelace, 1.0, &never);
        assert_eq!(r.len(), 2, "{r:?}");
        let ends = |k: usize, _: (f64, f64), _: (f64, f64)| k == 1;
        let r = partition(&traces(), cuts, None, 1.0, 1.0, &shoelace, 1.0, &ends);
        assert_eq!(r.len(), 3, "{r:?}");
    }

    #[test]
    fn crossing_sheets_make_four_regions() {
        let a = FaceTrace {
            events: [vec![(0.3, true)], vec![], vec![(0.6, true)], vec![]],
            start_positive: false,
            on_sheet: &always,
        };
        let b = FaceTrace {
            events: [vec![], vec![(0.4, true)], vec![], vec![(0.55, true)]],
            start_positive: true,
            on_sheet: &always,
        };
        let cuts: [&[f64]; 4] = [&[0.3], &[0.4], &[0.6], &[0.55]];
        let r = partition(&[a, b], cuts, None, 1.0, 1.0, &shoelace, 1.0, &never);
        assert_eq!(r.len(), 4, "{r:?}");
        let total: f64 = r.iter().map(|x| x.fraction).sum();
        assert!((total - 1.0).abs() < 1e-12);
        for x in &r {
            assert_eq!(x.pieces.len(), 2, "{r:?}");
        }
    }

    #[test]
    fn sheet_ending_on_solid_metal_separates() {
        // Solid metal below v = 0.3 (edges 1 and 3 metal up to 0.3, edge 0
        // metal); a septum from (0.4, 1) down to the metal surface.
        let p = |m: bool, t: &[f64]| Profile {
            start_metal: m,
            transitions: t.to_vec(),
        };
        let prof = [
            p(true, &[]),
            p(true, &[0.3]),
            p(false, &[]),
            p(true, &[0.3]),
        ];
        let septum = |q: (f64, f64)| q.1 >= 0.3 - 1e-12;
        let sep = FaceTrace {
            events: [vec![(0.4, false)], vec![], vec![(0.4, true)], vec![]],
            start_positive: false,
            on_sheet: &septum,
        };
        let cuts: [&[f64]; 4] = [&[], &[], &[0.4], &[]];
        let r = partition(&[sep], cuts, Some(&prof), 1.0, 1.0, &shoelace, 1.0, &never);
        assert_eq!(r.len(), 2, "{r:?}");
        let total: f64 = r.iter().map(|x| x.fraction).sum();
        assert!((total - 0.7).abs() < 1e-12, "{r:?}");
        let left = r.iter().find(|x| x.pieces.contains(&(2, 0))).unwrap();
        assert!((left.fraction - 0.28).abs() < 1e-9, "{r:?}");
        // Every piece listed once (edge 0, in metal, with the first).
        let n: usize = r.iter().map(|x| x.pieces.len()).sum();
        assert_eq!(n, 5);
    }
}
