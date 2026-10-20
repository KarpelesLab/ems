//! Conformal zero-thickness PEC sheets (design §5.13, "Thin sheets").
//!
//! A PEC sheet that is not on a mesh plane cuts E edges at isolated points
//! and divides the cell faces it crosses into parts that only touch along
//! the sheet. The tangential field vanishes on the sheet but the two sides
//! are independent, so a single voltage per cut edge or a single current
//! per cut face cannot represent the field (and the staircase, which
//! closes whole edges, moves the wall by up to a cell). The scheme here is
//! the finite-integration form of the conformal thin-PEC treatment
//! (contour-path / Dey–Mittra-type modelling of infinitely thin plates):
//!
//! * a cut edge is split at its cut points into **pieces**, each with its
//!   own voltage `u_p = E·l_p` (`f_p = l_p/l`, `cb_p = f_p·cb`, i.e. the
//!   capacitance `C/f_p` as for a Dey–Mittra edge);
//! * a cut face is split into **regions** (the connected parts of the face
//!   outside the sheet); each region has its own current `i_r`, inductance
//!   `f_r·L` (`f_r`: area fraction) and its own Faraday contour: the
//!   pieces and uncut edges on its boundary (the sheet part of the contour
//!   contributes nothing, `E_tan = 0`);
//! * Ampère's law for a piece reads the currents of the regions that
//!   contain it, the transpose of the region contours.
//!
//! So the system remains `C·du/dt = K'ᵀ·i`, `L·di/dt = −K'·u` with an
//! incidence `K'` of the split grid, which keeps the energy invariant of
//! the engine exact. Small regions (`f_r <` the merge threshold) are merged
//! with the region on the same side of the in-plane neighbour across their
//! most-open piece, exactly as small volume-cut faces are
//! ([`super::MergedFaces`]); the stability check covers the whole system.
//!
//! # Coupling with the kernel
//!
//! [`ConformalSheets`] owns the pieces of cut edges and one current per
//! (merged) region. The kernel leaves cut edges at zero (`ca = cb = 0`) and
//! the owned faces unchanged (`da = 1`, `db = 0`). After the H kernel the
//! operator updates the region currents and writes into each owned face
//! the current of its largest region; uncut edges of a cut face are updated
//! by the kernel with that current, and the operator adds
//! `cb·s·(i_r − i_primary)` in `post_e` for edges on another region. Cut
//! edges are updated by the operator only (every face around a cut edge is
//! owned); their stored voltage is the sum of the piece voltages.
//!
//! # Stored copies (cylindrical grids)
//!
//! On a cylindrical grid an E edge on the seam of a closed azimuth or on
//! the axis is stored more than once
//! ([`e_edge_copies`](crate::engine::cylindrical::e_edge_copies)). A cut
//! edge is one physical edge: its pieces are shared by all copies (listed
//! once, under any copy), with the merged coefficients of the copies
//! (`1/cb = Σ 1/cb_j`, `ca/cb = Σ ca_j/cb_j`, the periodic-merge rule);
//! faces refer to it through any copy, the kernel leaves every copy at
//! zero and the operator writes the piece sum into all of them, so the
//! periodic and axis merges keep it. H faces stay per copy, as for the
//! uncut grid. On a Cartesian grid every edge is its only copy.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::{Key, SIGNS, face_edges, face_exists, in_plane_neighbours, index};
use crate::axis::Axis;
use crate::engine::coeffs::Coefficients;
use crate::engine::fields::Fields;
use crate::engine::kernel::flush;
use crate::engine::operator::{Hooks, Operator, StepContext};
use crate::{Error, Result};

/// An E edge cut by thin PEC sheets at the edge parameters `cuts`
/// (fractions of the edge length from its start node, strictly increasing
/// in `(0, 1)`): the edge consists of `cuts.len() + 1` pieces, numbered
/// from the start node.
#[derive(Debug, Clone, PartialEq)]
pub struct SheetCutEdge {
    /// Edge direction.
    pub axis: Axis,
    /// Start node.
    pub node: [usize; 3],
    /// Cut parameters.
    pub cuts: Vec<f64>,
}

/// One part of a cut face outside the sheet(s).
#[derive(Debug, Clone, PartialEq)]
pub struct SheetRegion {
    /// Area fraction of the face.
    pub fraction: f64,
    /// Pieces on its boundary: `(side, piece)` with `side` in `0..4` the
    /// face edge in perimeter order (`b` at the node, `c` at `node + e_b`,
    /// `b` at `node + e_c`, `c` at the node, `(b, c) = axis.others()`) and
    /// `piece` the piece index along that edge (0 for an uncut edge).
    pub pieces: Vec<(u8, u8)>,
}

/// A cell face cut by thin PEC sheets: its regions, which together list
/// every piece of its four edges exactly once.
#[derive(Debug, Clone, PartialEq)]
pub struct SheetCutFace {
    /// Face normal (the H component).
    pub axis: Axis,
    /// Lowest corner node.
    pub node: [usize; 3],
    /// The regions (at least one).
    pub regions: Vec<SheetRegion>,
}

/// A voltage of the split system: an uncut edge's own entry in the field
/// array or a piece of a cut edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Ref {
    Main(usize, [usize; 3]),
    Piece(usize),
}

#[derive(Debug, Clone)]
struct PieceInfo {
    ca: f64,
    cb: f64,
    /// `cb` of the whole (merged) edge.
    full: f64,
}

/// The stored copies of an E edge, the canonical one first (see the
/// [module docs](self#stored-copies-cylindrical-grids)).
pub(super) type Copies<'a> = dyn Fn(Axis, [usize; 3]) -> Vec<[usize; 3]> + 'a;

#[derive(Debug, Clone)]
struct PRegion {
    face: usize,
    fraction: f64,
    /// `(voltage, sign, edge key, piece index)`.
    terms: Vec<(Ref, f64, Key, usize)>,
}

#[derive(Debug, Clone)]
struct PFace {
    axis: Axis,
    node: [usize; 3],
    l: f64,
    r: f64,
    regions: Vec<usize>,
}

/// The tau-independent part: pieces, faces and regions.
#[derive(Debug, Clone)]
pub(super) struct Prepared {
    pieces: Vec<PieceInfo>,
    /// Cut edges (all stored copies, canonical first) and their piece ids.
    cut: Vec<(Axis, Vec<[usize; 3]>, Vec<usize>)>,
    faces: Vec<PFace>,
    regions: Vec<PRegion>,
    /// Region of `(face key, edge key, piece)`.
    region_at: HashMap<(Key, Key, usize), usize>,
    face_index: HashMap<Key, usize>,
}

/// `(L, R)` of a face from its coefficients (0 if `db = 0`).
fn face_lr(co: &Coefficients, a: Axis, k: usize, dt: f64) -> (f64, f64) {
    let (da, db) = (co.da(a)[k] as f64, co.db(a)[k] as f64);
    if db > 0.0 {
        (dt * (1.0 + da) / (2.0 * db), (1.0 - da) / db)
    } else {
        (0.0, 0.0)
    }
}

/// Validates the sheet input on a grid with `n` nodes whose E edges have
/// the stored `copies`.
pub(super) fn validate(
    edges: &[SheetCutEdge],
    faces: &[SheetCutFace],
    n: [usize; 3],
    copies: &Copies<'_>,
) -> Result<()> {
    let bad = |what: String| Err(Error::Invalid(format!("invalid conformal sheet {what}")));
    let canon = |a: Axis, p: [usize; 3]| (a.index(), index(n, copies(a, p)[0]));
    let mut cuts: HashMap<Key, usize> = HashMap::new();
    for e in edges {
        let a = e.axis.index();
        let ok = (0..3).all(|d| e.node[d] < n[d])
            && e.node[a] + 1 < n[a]
            && !e.cuts.is_empty()
            && e.cuts.len() < 255
            && e.cuts.iter().all(|t| *t > 0.0 && *t < 1.0)
            && e.cuts.windows(2).all(|w| w[0] < w[1]);
        if !ok || cuts.insert(canon(e.axis, e.node), e.cuts.len()).is_some() {
            return bad(format!("edge {e:?}"));
        }
    }
    let mut listed: HashSet<Key> = HashSet::new();
    for f in faces {
        if !face_exists(f.axis, f.node, n)
            || f.regions.is_empty()
            || !listed.insert((f.axis.index(), index(n, f.node)))
        {
            return bad(format!("face {f:?}"));
        }
        let fe = face_edges(f.axis, f.node);
        let mut seen: [Vec<bool>; 4] = core::array::from_fn(|q| {
            let (a, p) = fe[q];
            vec![false; cuts.get(&canon(a, p)).map_or(1, |c| c + 1)]
        });
        for r in &f.regions {
            if !(0.0..=1.0).contains(&r.fraction) {
                return bad(format!("face {f:?}"));
            }
            for &(q, k) in &r.pieces {
                let s = seen.get_mut(q as usize).and_then(|v| v.get_mut(k as usize));
                match s {
                    Some(s) if !*s => *s = true,
                    _ => return bad(format!("face {f:?}")),
                }
            }
        }
        if seen.iter().flatten().any(|s| !s) {
            return bad(format!("face {f:?} (pieces not covered)"));
        }
    }
    // Every face around (every copy of) a cut edge must be listed.
    for e in edges {
        for c in copies(e.axis, e.node) {
            for (fa, q) in faces_of_edge(e.axis, c, n) {
                if !listed.contains(&(fa.index(), index(n, q))) {
                    return bad(format!("edge {e:?}: face {fa:?} at {q:?} not listed"));
                }
            }
        }
    }
    Ok(())
}

/// The existing faces `(normal, node)` that have the edge `(a, p)` on their
/// boundary.
fn faces_of_edge(a: Axis, p: [usize; 3], n: [usize; 3]) -> Vec<(Axis, [usize; 3])> {
    let mut v = Vec::with_capacity(4);
    for f in Axis::ALL {
        if f == a {
            continue;
        }
        let o = 3 - f.index() - a.index();
        if face_exists(f, p, n) {
            v.push((f, p));
        }
        if p[o] > 0 {
            let mut q = p;
            q[o] -= 1;
            if face_exists(f, q, n) {
                v.push((f, q));
            }
        }
    }
    v
}

impl Prepared {
    /// Indexes the sheet input against the coefficients `co` (after the
    /// volume edge scaling); `copies` gives the stored copies of an edge.
    pub(super) fn new(
        edges: &[SheetCutEdge],
        faces: &[SheetCutFace],
        co: &Coefficients,
        n: [usize; 3],
        dt: f64,
        copies: &Copies<'_>,
    ) -> Self {
        let idx = |p: [usize; 3]| index(n, p);
        let canon = |a: Axis, p: [usize; 3]| (a.index(), idx(copies(a, p)[0]));
        let mut pieces = Vec::new();
        let mut cut = Vec::new();
        let mut cut_of: HashMap<Key, usize> = HashMap::new();
        for e in edges {
            let all = copies(e.axis, e.node);
            let (ca, cb) = if all.len() == 1 {
                let k = idx(e.node);
                (co.ca(e.axis)[k] as f64, co.cb(e.axis)[k] as f64)
            } else {
                // The merged edge: 1/cb = Σ 1/cb_j, ca/cb = Σ ca_j/cb_j.
                let (mut w, mut wm) = (0.0, 0.0);
                for c in &all {
                    let (ca, cb) = (co.ca(e.axis)[idx(*c)] as f64, co.cb(e.axis)[idx(*c)] as f64);
                    if cb > 0.0 {
                        w += 1.0 / cb;
                        wm += ca / cb;
                    }
                }
                if w > 0.0 {
                    (wm / w, 1.0 / w)
                } else {
                    (0.0, 0.0)
                }
            };
            let mut ids = Vec::with_capacity(e.cuts.len() + 1);
            let mut last = 0.0;
            for t in e.cuts.iter().copied().chain(core::iter::once(1.0)) {
                ids.push(pieces.len());
                pieces.push(PieceInfo {
                    ca,
                    cb: cb * (t - last),
                    full: cb,
                });
                last = t;
            }
            cut_of.insert(canon(e.axis, e.node), cut.len());
            cut.push((e.axis, all, ids));
        }
        let mut out = Self {
            pieces,
            cut,
            faces: Vec::new(),
            regions: Vec::new(),
            region_at: HashMap::new(),
            face_index: HashMap::new(),
        };
        for f in faces {
            let k = idx(f.node);
            let (l, r) = face_lr(co, f.axis, k, dt);
            let fi = out.faces.len();
            let fe = face_edges(f.axis, f.node);
            let mut ids = Vec::with_capacity(f.regions.len());
            for reg in &f.regions {
                let ri = out.regions.len();
                let mut terms = Vec::with_capacity(reg.pieces.len());
                for &(q, pk) in &reg.pieces {
                    let (ea, ep) = fe[q as usize];
                    let ek = canon(ea, ep);
                    let v = match cut_of.get(&ek) {
                        Some(&c) => Ref::Piece(out.cut[c].2[pk as usize]),
                        None => Ref::Main(ea.index(), ep),
                    };
                    terms.push((v, SIGNS[q as usize] as f64, ek, pk as usize));
                    out.region_at
                        .insert(((f.axis.index(), k), ek, pk as usize), ri);
                }
                out.regions.push(PRegion {
                    face: fi,
                    fraction: reg.fraction,
                    terms,
                });
                ids.push(ri);
            }
            out.face_index.insert((f.axis.index(), k), fi);
            out.faces.push(PFace {
                axis: f.axis,
                node: f.node,
                l,
                r,
                regions: ids,
            });
        }
        out
    }

    /// Seeds of the stability iteration: the pieces (array 3) and the
    /// uncut edges of the cut faces.
    pub(super) fn seeds(&self) -> Vec<(usize, usize)> {
        let mut s: Vec<(usize, usize)> = (0..self.pieces.len()).map(|p| (3, p)).collect();
        for r in &self.regions {
            for &(v, _, ek, _) in &r.terms {
                if let Ref::Main(..) = v {
                    s.push(ek);
                }
            }
        }
        s
    }

    /// Builds the operator for merge threshold `tau`; `co` holds the
    /// coefficients after the volume edge scaling, `exclude` the faces the
    /// volume scheme modifies (never used as merge hosts).
    pub(super) fn construct(
        &self,
        co: &Coefficients,
        n: [usize; 3],
        dt: f64,
        tau: f64,
        exclude: &HashSet<Key>,
        copies: &Copies<'_>,
    ) -> Build {
        let idx = |p: [usize; 3]| index(n, p);
        let canon = |a: Axis, p: [usize; 3]| (a.index(), idx(copies(a, p)[0]));
        let mut faces = self.faces.clone();
        let mut regions = self.regions.clone();
        let mut region_at = self.region_at.clone();
        let mut face_index = self.face_index.clone();
        let open_ref = |v: Ref| match v {
            Ref::Main(a, p) => co.cb(Axis::from_index(a).expect("axis"))[idx(p)] > 0.0,
            Ref::Piece(j) => self.pieces[j].cb > 0.0,
        };
        let frac_ref = |v: Ref| match v {
            Ref::Main(..) => 1.0,
            Ref::Piece(j) => {
                let full = self.pieces[j].full;
                if full > 0.0 {
                    self.pieces[j].cb / full
                } else {
                    0.0
                }
            }
        };
        let live = |r: &PRegion| r.terms.iter().any(|t| open_ref(t.0));
        // Union-find over region ids.
        let mut parent: Vec<usize> = (0..regions.len()).collect();
        fn find(parent: &mut [usize], k: usize) -> usize {
            let mut r = k;
            while parent[r] != r {
                r = parent[r];
            }
            let mut c = k;
            while parent[c] != r {
                let next = parent[c];
                parent[c] = r;
                c = next;
            }
            r
        }
        let small: Vec<usize> = (0..regions.len())
            .filter(|&r| regions[r].fraction < tau && live(&regions[r]))
            .collect();
        let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut area: HashMap<usize, f64> = HashMap::new();
        for &r in &small {
            members.insert(r, vec![r]);
            area.insert(r, regions[r].fraction);
        }
        for _pass in 0..small.len().max(1) {
            let mut changed = false;
            let mut done: HashSet<usize> = HashSet::new();
            for &r0 in &small {
                let root = find(&mut parent, r0);
                let a = area.get(&root).copied().unwrap_or(1.0);
                if a >= tau || !done.insert(root) {
                    continue;
                }
                // (score, target): target is a region id or a plain face.
                let mut best: Option<((f64, f64), Target)> = None;
                let mem = members.get(&root).cloned().unwrap_or_else(|| vec![root]);
                for &m in &mem {
                    let f = &faces[regions[m].face];
                    let (fa, fnode) = (f.axis, f.node);
                    for &(v, _, ek, pk) in &regions[m].terms {
                        let fp = frac_ref(v);
                        if fp <= 0.0 || !open_ref(v) {
                            continue;
                        }
                        for (e, q, nb) in in_plane_neighbours(fa, fnode, n) {
                            if canon(e, q) != ek {
                                continue;
                            }
                            let nk = (fa.index(), idx(nb));
                            let cand = if let Some(&ri) = region_at.get(&(nk, ek, pk)) {
                                if find(&mut parent, ri) == root || !live(&regions[ri]) {
                                    continue;
                                }
                                (regions[ri].fraction, Target::Region(ri))
                            } else if face_index.contains_key(&nk) || exclude.contains(&nk) {
                                continue;
                            } else {
                                let (l, _) = face_lr(co, fa, idx(nb), dt);
                                let any_open = face_edges(fa, nb)
                                    .iter()
                                    .any(|&(a, p)| co.cb(a)[idx(p)] > 0.0);
                                if l <= 0.0 || !any_open {
                                    continue;
                                }
                                (1.0, Target::Plain(fa, nb))
                            };
                            let score = (fp, cand.0);
                            if best.as_ref().is_none_or(|b| score > b.0) {
                                best = Some((score, cand.1));
                            }
                        }
                    }
                }
                let Some((_, target)) = best else {
                    continue;
                };
                let tr = match target {
                    Target::Region(ri) => ri,
                    Target::Plain(fa, nb) => {
                        // Absorb a plain face as a one-region face.
                        let k = idx(nb);
                        let (l, r) = face_lr(co, fa, k, dt);
                        let fi = faces.len();
                        let ri = regions.len();
                        let mut terms = Vec::with_capacity(4);
                        for (q, (ea, ep)) in face_edges(fa, nb).into_iter().enumerate() {
                            let ek = canon(ea, ep);
                            terms.push((Ref::Main(ea.index(), ep), SIGNS[q] as f64, ek, 0));
                            region_at.insert(((fa.index(), k), ek, 0), ri);
                        }
                        regions.push(PRegion {
                            face: fi,
                            fraction: 1.0,
                            terms,
                        });
                        parent.push(ri);
                        face_index.insert((fa.index(), k), fi);
                        faces.push(PFace {
                            axis: fa,
                            node: nb,
                            l,
                            r,
                            regions: vec![ri],
                        });
                        ri
                    }
                };
                let rt = find(&mut parent, tr);
                let (keep, gone) = if root < rt { (root, rt) } else { (rt, root) };
                parent[gone] = keep;
                let ag = area.remove(&gone).unwrap_or_else(|| regions[gone].fraction);
                let ak = area.remove(&keep).unwrap_or_else(|| regions[keep].fraction);
                area.insert(keep, ag + ak);
                let mg = members.remove(&gone).unwrap_or_else(|| vec![gone]);
                let mut mk = members.remove(&keep).unwrap_or_else(|| vec![keep]);
                mk.extend(mg);
                mk.sort_unstable();
                members.insert(keep, mk);
                changed = true;
            }
            if !changed {
                break;
            }
        }

        // Groups: one per union-find class, in order of the smallest member.
        let nreg = regions.len();
        let mut gid: Vec<usize> = vec![usize::MAX; nreg];
        let mut classes: Vec<Vec<usize>> = Vec::new();
        for r in 0..nreg {
            let root = find(&mut parent, r);
            if gid[root] == usize::MAX {
                gid[root] = classes.len();
                classes.push(Vec::new());
            }
            let g = gid[root];
            gid[r] = g;
            classes[g].push(r);
        }
        let mut report = SheetCounts {
            edges: self.cut.len(),
            faces: self.faces.len(),
            ..SheetCounts::default()
        };
        let mut groups = Vec::with_capacity(classes.len());
        for members in &classes {
            let (mut l, mut r) = (0.0, 0.0);
            let mut terms: BTreeMap<Ref, f64> = BTreeMap::new();
            for &m in members {
                let reg = &regions[m];
                let f = &faces[reg.face];
                let mut fr = reg.fraction;
                if members.len() == 1 && fr < tau && live(reg) {
                    fr = tau.max(1e-6);
                    report.floored += 1;
                }
                l += fr * f.l;
                r += fr * f.r;
                for &(v, s, _, _) in &reg.terms {
                    *terms.entry(v).or_insert(0.0) += s;
                }
            }
            if members.len() > 1 {
                report.merged += members.len();
            }
            let terms: Vec<(Ref, f64)> = terms.into_iter().filter(|t| t.1 != 0.0).collect();
            let (da, db, dt_over_l) = if l > 0.0 {
                let den = l + 0.5 * r * dt;
                ((l - 0.5 * r * dt) / den, dt / den, dt / l)
            } else {
                (0.0, 0.0, 0.0)
            };
            groups.push(SGroup {
                terms,
                writes: Vec::new(),
                da,
                db,
                l,
                dt_over_l,
                i: 0.0,
            });
        }
        report.groups = groups.len();
        // Primary region per face (largest fraction), writes and
        // corrections.
        let mut corrections = Vec::new();
        let mut owned = HashSet::new();
        for f in &faces {
            owned.insert((f.axis.index(), idx(f.node)));
            let primary = *f
                .regions
                .iter()
                .min_by(|&&a, &&b| {
                    regions[b]
                        .fraction
                        .total_cmp(&regions[a].fraction)
                        .then(a.cmp(&b))
                })
                .expect("a region");
            let gp = gid[primary];
            groups[gp].writes.push((f.axis, f.node));
            for &r in &f.regions {
                let gr = gid[r];
                if gr == gp {
                    continue;
                }
                for &(v, s, _, _) in &regions[r].terms {
                    if let Ref::Main(a, p) = v {
                        let axis = Axis::from_index(a).expect("axis");
                        let cb = co.cb(axis)[idx(p)] as f64;
                        if cb > 0.0 {
                            corrections.push(Correction {
                                axis,
                                node: p,
                                cbs: cb * s,
                                region: gr,
                                primary: gp,
                            });
                        }
                    }
                }
            }
        }
        // Piece incidences.
        let mut inc: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); self.pieces.len()];
        for (r, reg) in regions.iter().enumerate() {
            for &(v, s, _, _) in &reg.terms {
                if let Ref::Piece(j) = v {
                    *inc[j].entry(gid[r]).or_insert(0.0) += s;
                }
            }
        }
        let pieces = self
            .pieces
            .iter()
            .zip(inc)
            .map(|(p, inc)| Piece {
                ca: p.ca,
                cb: p.cb,
                cap: if p.cb > 0.0 {
                    dt * (1.0 + p.ca) / (2.0 * p.cb)
                } else {
                    0.0
                },
                u: 0.0,
                inc: inc.into_iter().filter(|t| t.1 != 0.0).collect(),
            })
            .collect();
        Build {
            op: ConformalSheets {
                pieces,
                cut: self.cut.clone(),
                groups,
                corrections,
                energy: 0.0,
            },
            owned,
            report,
        }
    }
}

enum Target {
    Region(usize),
    Plain(Axis, [usize; 3]),
}

/// Counts for the report.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct SheetCounts {
    pub edges: usize,
    pub faces: usize,
    pub groups: usize,
    pub merged: usize,
    pub floored: usize,
}

/// Result of [`Prepared::construct`].
pub(super) struct Build {
    pub op: ConformalSheets,
    /// Faces driven by the operator (kernel `da = 1`, `db = 0`).
    pub owned: HashSet<Key>,
    pub report: SheetCounts,
}

impl Build {
    /// Kernel coefficients: cut edges zero, owned faces frozen.
    pub(super) fn apply_coefficients(&self, co: &mut Coefficients, n: [usize; 3]) {
        for (a, ps, _) in &self.op.cut {
            let (ca, cb) = co.e_mut(*a);
            for p in ps {
                let k = index(n, *p);
                ca[k] = 0.0;
                cb[k] = 0.0;
            }
        }
        for &(a, k) in &self.owned {
            let (da, db) = co.h_mut(Axis::from_index(a).expect("axis"));
            da[k] = 1.0;
            db[k] = 0.0;
        }
    }

    /// Stability terms: `(dt/L, [(array, index, sign)])` per group (array 3:
    /// pieces) and `√cb` of the pieces.
    pub(super) fn stability_terms(&self, n: [usize; 3]) -> (Vec<super::GroupTerms>, Vec<f64>) {
        let g = self
            .op
            .groups
            .iter()
            .filter(|g| g.l > 0.0)
            .map(|g| {
                (
                    g.dt_over_l,
                    g.terms
                        .iter()
                        .map(|&(v, s)| match v {
                            Ref::Main(a, p) => (a, index(n, p), s),
                            Ref::Piece(j) => (3, j, s),
                        })
                        .collect(),
                )
            })
            .collect();
        let sq = self
            .op
            .pieces
            .iter()
            .map(|p| p.cb.max(0.0).sqrt())
            .collect();
        (g, sq)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Piece {
    ca: f64,
    cb: f64,
    /// Capacitance for the energy.
    cap: f64,
    u: f64,
    /// `(group, sign)` of the regions containing the piece.
    inc: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
struct SGroup {
    terms: Vec<(Ref, f64)>,
    /// Faces whose field entry holds this group's current.
    writes: Vec<(Axis, [usize; 3])>,
    da: f64,
    db: f64,
    l: f64,
    dt_over_l: f64,
    i: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct Correction {
    axis: Axis,
    node: [usize; 3],
    cbs: f64,
    region: usize,
    primary: usize,
}

/// The split-cell system of conformal thin PEC sheets, registered by
/// [`ConformalPec::apply`](super::ConformalPec::apply); see the
/// [conformal module docs](crate::engine::conformal#thin-pec-sheets) and design
/// §5.13.1. Cut edges are left at zero by the kernel and updated here;
/// owned faces hold the current of their largest region, and uncut edges
/// on another region are corrected after the E kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformalSheets {
    pieces: Vec<Piece>,
    cut: Vec<(Axis, Vec<[usize; 3]>, Vec<usize>)>,
    groups: Vec<SGroup>,
    corrections: Vec<Correction>,
    energy: f64,
}

impl ConformalSheets {
    /// Number of edge pieces (voltages of cut edges).
    pub fn pieces(&self) -> usize {
        self.pieces.len()
    }

    /// Number of region currents (after merging).
    pub fn currents(&self) -> usize {
        self.groups.len()
    }

    fn voltage(&self, v: Ref, f: &Fields<'_>) -> f64 {
        match v {
            Ref::Main(a, p) => f.u.get(Axis::from_index(a).expect("axis"), p) as f64,
            Ref::Piece(j) => self.pieces[j].u,
        }
    }

    fn circulation(&self, g: &SGroup, f: &Fields<'_>) -> f64 {
        g.terms.iter().map(|&(v, s)| s * self.voltage(v, f)).sum()
    }
}

impl Operator for ConformalSheets {
    fn hooks(&self) -> Hooks {
        Hooks::POST_H | Hooks::POST_E
    }

    fn post_h(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        for q in 0..self.groups.len() {
            let c = self.circulation(&self.groups[q], f);
            let g = &mut self.groups[q];
            g.i = flush((g.da * g.i - g.db * c) as f32) as f64;
            for &(a, p) in &g.writes {
                f.i.set(a, p, g.i as f32);
            }
        }
    }

    fn post_e(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        for c in &self.corrections {
            let d = c.cbs * (self.groups[c.region].i - self.groups[c.primary].i);
            let u = f.u.get_mut(c.axis, c.node);
            *u = flush((*u as f64 + d) as f32);
        }
        for p in &mut self.pieces {
            let s: f64 = p.inc.iter().map(|&(g, s)| s * self.groups[g].i).sum();
            p.u = flush((p.ca * p.u + p.cb * s) as f32) as f64;
        }
        for (a, nodes, ids) in &self.cut {
            let s: f64 = ids.iter().map(|&j| self.pieces[j].u).sum();
            for node in nodes {
                f.u.set(*a, *node, s as f32);
            }
        }
        // Energy at the new time: pieces plus ½·L·i^{n+½}·i^{n+3/2} with
        // the value the next post_h will compute.
        let mut w: f64 = self.pieces.iter().map(|p| 0.5 * p.cap * p.u * p.u).sum();
        for g in &self.groups {
            if g.l > 0.0 {
                let next = flush((g.da * g.i - g.db * self.circulation(g, f)) as f32) as f64;
                w += 0.5 * g.l * g.i * next;
            }
        }
        self.energy = w;
    }

    fn energy(&self) -> f64 {
        self.energy
    }
}
