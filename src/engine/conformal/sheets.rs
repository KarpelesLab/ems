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
//! the engine exact. Small regions (`f_r <` the merge threshold) lose their
//! own current exactly as small volume-cut faces do
//! ([`super::MergedFaces`], [module docs](super#small-cut-faces-interpolation)):
//! their `H` is the affine interpolation, at their centroid, of three
//! regions or uncut faces **on the same side of the sheet** with their own
//! current (the host across the most open piece and one host neighbour per
//! in-plane axis, reached across pieces and uncut edges, so never through
//! the sheet), and the hosts sharing small regions form blocks with the
//! consistent inductance `PᵀLP`. That is second order for `H` tangential
//! to the sheet (TE fields), where merging (constant extrapolation, the
//! fallback for regions without a valid stencil and inside absorbing
//! layers) is first order. Each interpolated region also keeps a
//! deviation current (inductance `(f_r + a)·L`, see the
//! [module docs](super#deviation-currents-no-spurious-static-fields)), so
//! no spurious static field is left on its cut pieces; merged regions do
//! not have one yet. The stability check covers the whole system.
//!
//! # Coupling with the kernel
//!
//! [`ConformalSheets`] owns the pieces of cut edges and one reduced current
//! per merged group, unmerged region or host (uncut faces used as merge
//! partners or hosts are taken over as one-region faces). The kernel
//! leaves cut edges at zero (`ca = cb = 0`) and
//! the owned faces unchanged (`da = 1`, `db = 0`). After the H kernel the
//! operator updates the reduced currents and writes into each owned face
//! the current of its largest region (for an interpolated region, the
//! weighted host currents); uncut edges of a cut face are updated
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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::{
    Block, Key, SIGNS, centroid_dual_length, face_edges, face_exists, in_plane_neighbours, index,
    interpolation_weights,
};
use crate::axis::Axis;
use crate::engine::Grid;
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
    /// Centroid of the region, `[t_b, t_c]` in units of the face sides
    /// (as [`ConformalPec::push_face_with_centroid`](super::ConformalPec::push_face_with_centroid));
    /// a small region without one cannot be interpolated and is merged.
    pub centroid: Option<[f64; 2]>,
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
    centroid: Option<[f64; 2]>,
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
                    centroid: reg.centroid,
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

    /// Builds the operator for merge threshold `tau` on `grid`; `co` holds
    /// the coefficients after the volume edge scaling, `exclude` the faces
    /// the volume scheme modifies (never used as hosts), `absorbing`
    /// whether a face touches an absorbing layer (no interpolation there).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn construct(
        &self,
        co: &Coefficients,
        grid: &Grid,
        dt: f64,
        tau: f64,
        exclude: &HashSet<Key>,
        absorbing: &dyn Fn(Axis, [usize; 3]) -> bool,
        copies: &Copies<'_>,
        deviation: f64,
    ) -> Build {
        let n = grid.nodes();
        let idx = |p: [usize; 3]| index(n, p);
        let canon = |a: Axis, p: [usize; 3]| (a.index(), idx(copies(a, p)[0]));
        let mut split = Split {
            faces: self.faces.clone(),
            regions: self.regions.clone(),
            region_at: self.region_at.clone(),
            face_index: self.face_index.clone(),
        };
        let open_ref = |v: Ref| self.open_ref(co, n, v);
        let live = |r: &PRegion| r.terms.iter().any(|t| open_ref(t.0));
        // Union-find over region ids.
        let mut parent: Vec<usize> = (0..split.regions.len()).collect();
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
        let small: Vec<usize> = (0..split.regions.len())
            .filter(|&r| split.regions[r].fraction < tau && live(&split.regions[r]))
            .collect();
        let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut area: HashMap<usize, f64> = HashMap::new();
        // Regions in merge groups (no stencil faces), and the small regions
        // merged because they have no valid stencil.
        let mut merged: HashSet<usize> = HashSet::new();
        let mut seeds: Vec<usize> = Vec::new();
        let mut stencils: Vec<(usize, Vec<(Target, f64)>)>;
        loop {
            stencils = Vec::new();
            let mut failing = Vec::new();
            {
                let ctx = Ctx {
                    prepared: self,
                    split: &split,
                    co,
                    grid,
                    tau,
                    exclude,
                    absorbing,
                    copies,
                    merged: &merged,
                };
                for &r in &small {
                    if merged.contains(&r) {
                        continue;
                    }
                    match ctx.stencil(r) {
                        Some(s) => stencils.push((r, s)),
                        None => failing.push(r),
                    }
                }
            }
            if failing.is_empty() {
                break;
            }
            for &r in &failing {
                merged.insert(r);
                members.insert(r, vec![r]);
                area.insert(r, split.regions[r].fraction);
            }
            seeds.extend(failing);
            self.merge(
                &mut split,
                &mut parent,
                &mut members,
                &mut area,
                &mut merged,
                &seeds,
                (co, n, dt, tau),
                exclude,
                &canon,
                find,
            );
        }
        // Uncut faces serving as hosts become one-region faces.
        let interp: Vec<(usize, Vec<(usize, f64)>)> = stencils
            .into_iter()
            .map(|(r, st)| {
                let st = st
                    .into_iter()
                    .map(|(t, w)| match t {
                        Target::Region(ri) => (ri, w),
                        Target::Plain(fa, nb) => {
                            let ri = split.absorb(co, fa, nb, n, dt, &canon);
                            if ri == parent.len() {
                                parent.push(ri);
                            }
                            (ri, w)
                        }
                    })
                    .collect();
                (r, st)
            })
            .collect();
        let is_interp: HashSet<usize> = interp.iter().map(|s| s.0).collect();
        let host_set: BTreeSet<usize> = interp
            .iter()
            .flat_map(|s| s.1.iter().map(|h| h.0))
            .collect();
        let regions = &split.regions;
        let faces = &split.faces;

        // Reduced currents: one per union-find class (interpolated regions
        // excepted), in order of the smallest member; the classes of hosts
        // (single regions) after the others, for the blocks.
        let nreg = regions.len();
        let mut class_of: Vec<usize> = vec![usize::MAX; nreg];
        let mut classes: Vec<Vec<usize>> = Vec::new();
        for r in 0..nreg {
            if is_interp.contains(&r) {
                continue;
            }
            let root = find(&mut parent, r);
            if class_of[root] == usize::MAX {
                class_of[root] = classes.len();
                classes.push(Vec::new());
            }
            let c = class_of[root];
            class_of[r] = c;
            classes[c].push(r);
        }
        let is_host = |c: &Vec<usize>| c.len() == 1 && host_set.contains(&c[0]);
        let order: Vec<usize> = (0..classes.len())
            .filter(|&c| !is_host(&classes[c]))
            .chain((0..classes.len()).filter(|&c| is_host(&classes[c])))
            .collect();
        let scalar = order.iter().filter(|&&c| !is_host(&classes[c])).count();
        let mut gid: Vec<usize> = vec![usize::MAX; nreg];
        for (g, &c) in order.iter().enumerate() {
            for &r in &classes[c] {
                gid[r] = g;
            }
        }
        // Current of each region as a combination of reduced currents.
        let mut combo: Vec<Vec<(usize, f64)>> = (0..nreg).map(|r| vec![(gid[r], 1.0)]).collect();
        for (r, st) in &interp {
            combo[*r] = st.iter().map(|&(h, w)| (gid[h], w)).collect();
        }
        let mut report = SheetCounts {
            edges: self.cut.len(),
            faces: self.faces.len(),
            interpolated: interp.len(),
            hosts: host_set.len(),
            ..SheetCounts::default()
        };
        let mut groups = Vec::with_capacity(order.len());
        for &c in &order {
            let members = &classes[c];
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
            groups.push(SGroup::new(terms, l, r, dt));
        }
        // Hosts: plus the weighted contours of the regions reading them
        // (`PᵀK`), and the consistent inductance `PᵀLP` of the blocks.
        let mut ml: Vec<HashMap<usize, (f64, f64)>> = (scalar..groups.len())
            .map(|g| HashMap::from([(g - scalar, (groups[g].l, groups[g].r))]))
            .collect();
        let mut extra: Vec<BTreeMap<Ref, f64>> = vec![BTreeMap::new(); groups.len() - scalar];
        let mut link: Vec<usize> = (0..ml.len()).collect();
        for (s, st) in &interp {
            let reg = &regions[*s];
            let f = &faces[reg.face];
            // The inductance of the small region with its current measured
            // at its centroid (see `centroid_dual_length`).
            let m = grid.dual_edge_length(f.axis, f.node)
                / centroid_dual_length(grid, f.axis, f.node, reg.centroid.unwrap_or([0.5; 2]));
            let (lf, rf) = (reg.fraction * f.l * m, reg.fraction * f.r * m);
            for &(hj, wj) in st {
                let j = gid[hj] - scalar;
                for &(v, sg, _, _) in &reg.terms {
                    *extra[j].entry(v).or_insert(0.0) += wj * sg;
                }
                for &(hk, wk) in st {
                    let k = gid[hk] - scalar;
                    let e = ml[j].entry(k).or_insert((0.0, 0.0));
                    e.0 += wj * wk * lf;
                    e.1 += wj * wk * rf;
                }
                let (a, b) = (find(&mut link, j), find(&mut link, gid[st[0].0] - scalar));
                link[a.max(b)] = a.min(b);
            }
            if deviation > 0.0 {
                // The region's deviation current (module docs of
                // `conformal`): `i_s = Σ w·g + d_s`, mass `f_r·L + dev·L`.
                let qd = ml.len();
                let mut row = HashMap::from([(qd, (lf + deviation * f.l, rf))]);
                for &(hj, wj) in st {
                    let j = gid[hj] - scalar;
                    let cross = (wj * lf, wj * rf);
                    let e = row.entry(j).or_insert((0.0, 0.0));
                    e.0 += cross.0;
                    e.1 += cross.1;
                    let e = ml[j].entry(qd).or_insert((0.0, 0.0));
                    e.0 += cross.0;
                    e.1 += cross.1;
                }
                ml.push(row);
                link.push(qd);
                let (a, b) = (find(&mut link, qd), find(&mut link, gid[st[0].0] - scalar));
                link[a.max(b)] = a.min(b);
                let mut terms: BTreeMap<Ref, f64> = BTreeMap::new();
                for &(v, sg, _, _) in &reg.terms {
                    *terms.entry(v).or_insert(0.0) += sg;
                }
                let (l, r) = ml[qd][&qd];
                groups.push(SGroup::new(terms, l, r, dt));
                combo[*s].push((scalar + qd, 1.0));
            }
        }
        for (q, ex) in extra.into_iter().enumerate() {
            let g = &mut groups[scalar + q];
            for (v, w) in ex {
                g.terms.push((v, w));
            }
            g.terms.sort_by_key(|t| t.0);
            let mut merged_terms: Vec<(Ref, f64)> = Vec::with_capacity(g.terms.len());
            for &(v, w) in &g.terms {
                match merged_terms.last_mut() {
                    Some(t) if t.0 == v => t.1 += w,
                    _ => merged_terms.push((v, w)),
                }
            }
            merged_terms.retain(|t| t.1.abs() > 1e-12);
            g.terms = merged_terms;
        }
        let mut comps: Vec<Vec<usize>> = Vec::new();
        let mut comp_of: HashMap<usize, usize> = HashMap::new();
        for q in 0..ml.len() {
            let r = find(&mut link, q);
            let c = *comp_of.entry(r).or_insert_with(|| {
                comps.push(Vec::new());
                comps.len() - 1
            });
            comps[c].push(q);
        }
        let blocks: Vec<Block> = comps
            .iter()
            .map(|comp| Block::build(comp, &ml, scalar, dt))
            .collect();
        for g in &mut groups[scalar..] {
            // Block members are advanced by their block.
            g.da = 0.0;
            g.db = 0.0;
        }
        report.groups = groups.len();
        // Slots: the reduced currents, then the interpolated regions.
        let mut slots: Vec<Vec<(usize, f64)>> = (0..groups.len()).map(|g| vec![(g, 1.0)]).collect();
        let mut slot: Vec<usize> = gid.clone();
        for (r, _) in &interp {
            slot[*r] = slots.len();
            slots.push(combo[*r].clone());
        }
        // Primary region per face (largest fraction), writes and
        // corrections.
        let mut corrections = Vec::new();
        let mut writes = Vec::with_capacity(faces.len());
        let mut owned = HashSet::new();
        for f in faces {
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
            let sp = slot[primary];
            writes.push((f.axis, f.node, sp));
            for &r in &f.regions {
                let sr = slot[r];
                if sr == sp {
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
                                region: sr,
                                primary: sp,
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
                    for &(g, w) in &combo[r] {
                        *inc[j].entry(g).or_insert(0.0) += s * w;
                    }
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
                inc: inc.into_iter().filter(|t| t.1.abs() > 1e-12).collect(),
            })
            .collect();
        let ng = groups.len();
        let ns = slots.len();
        Build {
            op: ConformalSheets {
                pieces,
                cut: self.cut.clone(),
                groups,
                scalar,
                blocks,
                slots,
                slot_i: vec![0.0; ns],
                writes,
                corrections,
                circ: vec![0.0; ng],
                next: vec![0.0; ng],
                scratch: Vec::new(),
            },
            owned,
            report,
        }
    }

    fn open_ref(&self, co: &Coefficients, n: [usize; 3], v: Ref) -> bool {
        match v {
            Ref::Main(a, p) => co.cb(Axis::from_index(a).expect("axis"))[index(n, p)] > 0.0,
            Ref::Piece(j) => self.pieces[j].cb > 0.0,
        }
    }

    /// Open length fraction of a voltage's edge part (1 for uncut edges).
    fn frac_ref(&self, v: Ref) -> f64 {
        match v {
            Ref::Main(..) => 1.0,
            Ref::Piece(j) => {
                let full = self.pieces[j].full;
                if full > 0.0 {
                    self.pieces[j].cb / full
                } else {
                    0.0
                }
            }
        }
    }

    /// Merges the small regions `seeds` whose groups are still below
    /// `tau` with the region on the same side of the in-plane neighbour
    /// across their most open piece (absorbing uncut faces), until no group
    /// changes; every region of a group of several is added to `merged`.
    #[allow(clippy::too_many_arguments)]
    fn merge(
        &self,
        split: &mut Split,
        parent: &mut Vec<usize>,
        members: &mut BTreeMap<usize, Vec<usize>>,
        area: &mut HashMap<usize, f64>,
        merged: &mut HashSet<usize>,
        seeds: &[usize],
        (co, n, dt, tau): (&Coefficients, [usize; 3], f64, f64),
        exclude: &HashSet<Key>,
        canon: &dyn Fn(Axis, [usize; 3]) -> Key,
        find: fn(&mut [usize], usize) -> usize,
    ) {
        let idx = |p: [usize; 3]| index(n, p);
        let open_ref = |v: Ref| self.open_ref(co, n, v);
        let live = |r: &PRegion| r.terms.iter().any(|t| open_ref(t.0));
        for _pass in 0..seeds.len().max(1) {
            let mut changed = false;
            let mut done: HashSet<usize> = HashSet::new();
            for &r0 in seeds {
                let root = find(parent, r0);
                let a = area.get(&root).copied().unwrap_or(1.0);
                if a >= tau || !done.insert(root) {
                    continue;
                }
                // (score, target): target is a region id or a plain face.
                let mut best: Option<((f64, f64), Target)> = None;
                let mem = members.get(&root).cloned().unwrap_or_else(|| vec![root]);
                for &m in &mem {
                    let f = &split.faces[split.regions[m].face];
                    let (fa, fnode) = (f.axis, f.node);
                    for &(v, _, ek, pk) in &split.regions[m].terms {
                        let fp = self.frac_ref(v);
                        if fp <= 0.0 || !open_ref(v) {
                            continue;
                        }
                        for (e, q, nb) in in_plane_neighbours(fa, fnode, n) {
                            if canon(e, q) != ek {
                                continue;
                            }
                            let nk = (fa.index(), idx(nb));
                            let cand = if let Some(&ri) = split.region_at.get(&(nk, ek, pk)) {
                                if find(parent, ri) == root || !live(&split.regions[ri]) {
                                    continue;
                                }
                                (split.regions[ri].fraction, Target::Region(ri))
                            } else if split.face_index.contains_key(&nk) || exclude.contains(&nk) {
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
                        let ri = split.absorb(co, fa, nb, n, dt, canon);
                        parent.push(ri);
                        ri
                    }
                };
                let rt = find(parent, tr);
                let (keep, gone) = if root < rt { (root, rt) } else { (rt, root) };
                parent[gone] = keep;
                let ag = area
                    .remove(&gone)
                    .unwrap_or_else(|| split.regions[gone].fraction);
                let ak = area
                    .remove(&keep)
                    .unwrap_or_else(|| split.regions[keep].fraction);
                area.insert(keep, ag + ak);
                let mg = members.remove(&gone).unwrap_or_else(|| vec![gone]);
                let mut mk = members.remove(&keep).unwrap_or_else(|| vec![keep]);
                mk.extend(mg);
                mk.sort_unstable();
                merged.extend(mk.iter().copied());
                members.insert(keep, mk);
                changed = true;
            }
            if !changed {
                break;
            }
        }
    }
}

/// The faces and regions of the split system while it is built (merging
/// and interpolation absorb uncut faces as one-region faces).
struct Split {
    faces: Vec<PFace>,
    regions: Vec<PRegion>,
    /// Region of `(face key, edge key, piece)`.
    region_at: HashMap<(Key, Key, usize), usize>,
    face_index: HashMap<Key, usize>,
}

impl Split {
    /// The region of the uncut face `(fa, nb)`, absorbed as a one-region
    /// face if it is not listed yet (then its id is `regions.len()` before
    /// the call).
    fn absorb(
        &mut self,
        co: &Coefficients,
        fa: Axis,
        nb: [usize; 3],
        n: [usize; 3],
        dt: f64,
        canon: &dyn Fn(Axis, [usize; 3]) -> Key,
    ) -> usize {
        let k = index(n, nb);
        if let Some(&fi) = self.face_index.get(&(fa.index(), k)) {
            return self.faces[fi].regions[0];
        }
        let (l, r) = face_lr(co, fa, k, dt);
        let fi = self.faces.len();
        let ri = self.regions.len();
        let mut terms = Vec::with_capacity(4);
        for (q, (ea, ep)) in face_edges(fa, nb).into_iter().enumerate() {
            let ek = canon(ea, ep);
            terms.push((Ref::Main(ea.index(), ep), SIGNS[q] as f64, ek, 0));
            self.region_at.insert(((fa.index(), k), ek, 0), ri);
        }
        self.regions.push(PRegion {
            face: fi,
            fraction: 1.0,
            centroid: Some([0.5, 0.5]),
            terms,
        });
        self.face_index.insert((fa.index(), k), fi);
        self.faces.push(PFace {
            axis: fa,
            node: nb,
            l,
            r,
            regions: vec![ri],
        });
        ri
    }
}

/// A reduced-current candidate: a region, or an uncut face not listed yet
/// (absorbed as a one-region face when used).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Target {
    Region(usize),
    Plain(Axis, [usize; 3]),
}

/// What the interpolation stencils are built from.
struct Ctx<'a> {
    prepared: &'a Prepared,
    split: &'a Split,
    co: &'a Coefficients,
    grid: &'a Grid,
    tau: f64,
    exclude: &'a HashSet<Key>,
    absorbing: &'a dyn Fn(Axis, [usize; 3]) -> bool,
    copies: &'a Copies<'a>,
    merged: &'a HashSet<usize>,
}

impl Ctx<'_> {
    fn n(&self) -> [usize; 3] {
        self.grid.nodes()
    }

    fn canon(&self, a: Axis, p: [usize; 3]) -> Key {
        (a.index(), index(self.n(), (self.copies)(a, p)[0]))
    }

    fn open(&self, v: Ref) -> bool {
        self.prepared.open_ref(self.co, self.n(), v) && self.prepared.frac_ref(v) > 0.0
    }

    fn node(&self, t: Target) -> [usize; 3] {
        match t {
            Target::Region(r) => self.split.faces[self.split.regions[r].face].node,
            Target::Plain(_, p) => p,
        }
    }

    fn fraction(&self, t: Target) -> f64 {
        match t {
            Target::Region(r) => self.split.regions[r].fraction,
            Target::Plain(..) => 1.0,
        }
    }

    fn centroid(&self, t: Target) -> Option<[f64; 2]> {
        match t {
            Target::Region(r) => self.split.regions[r].centroid,
            Target::Plain(..) => Some([0.5, 0.5]),
        }
    }

    /// The contour of a candidate: `(voltage, edge key, piece)`.
    fn terms(&self, t: Target) -> Vec<(Ref, Key, usize)> {
        match t {
            Target::Region(r) => self.split.regions[r]
                .terms
                .iter()
                .map(|&(v, _, ek, pk)| (v, ek, pk))
                .collect(),
            Target::Plain(fa, p) => face_edges(fa, p)
                .into_iter()
                .map(|(ea, ep)| (Ref::Main(ea.index(), ep), self.canon(ea, ep), 0))
                .collect(),
        }
    }

    /// The candidates on the other side of piece `pk` of edge `ek` of the
    /// face `(fa, p)`: the region of the in-plane neighbour holding that
    /// piece (same side of the sheet), or the neighbour itself if it is an
    /// uncut face the sheet system may take over.
    fn across(&self, fa: Axis, p: [usize; 3], ek: Key, pk: usize) -> Vec<Target> {
        let n = self.n();
        let mut v = Vec::new();
        for (e, q, nb) in in_plane_neighbours(fa, p, n) {
            if self.canon(e, q) != ek {
                continue;
            }
            let nk = (fa.index(), index(n, nb));
            if let Some(&ri) = self.split.region_at.get(&(nk, ek, pk)) {
                v.push(Target::Region(ri));
            } else if !self.split.face_index.contains_key(&nk) && !self.exclude.contains(&nk) {
                let k = index(n, nb);
                let any_open = face_edges(fa, nb)
                    .iter()
                    .any(|&(a, q)| self.co.cb(a)[index(n, q)] > 0.0);
                if self.co.db(fa)[k] > 0.0 && any_open {
                    v.push(Target::Plain(fa, nb));
                }
            }
        }
        v
    }

    /// Whether a candidate keeps its own current and may serve a stencil.
    fn free(&self, t: Target, fa: Axis) -> bool {
        let p = self.node(t);
        if (self.absorbing)(fa, p) || self.centroid(t).is_none() {
            return false;
        }
        match t {
            Target::Region(r) => {
                let reg = &self.split.regions[r];
                reg.fraction >= self.tau
                    && !self.merged.contains(&r)
                    && self.split.faces[reg.face].l > 0.0
                    && reg.terms.iter().any(|t| self.open(t.0))
            }
            Target::Plain(..) => true,
        }
    }

    /// Position of a candidate's centroid in grid coordinates along the
    /// in-plane axes of `fa`.
    fn pos(&self, fa: Axis, t: Target) -> [f64; 2] {
        let p = self.node(t);
        let c = self.centroid(t).unwrap_or([0.5, 0.5]);
        let (b, cc) = fa.others();
        let at = |d: Axis, s: f64| {
            let l = self.grid.lines(d);
            let i = p[d.index()];
            l[i] + s * (l[i + 1] - l[i])
        };
        [at(b, c[0]), at(cc, c[1])]
    }

    /// The interpolation stencil of the small region `r`, as for small
    /// volume-cut faces ([`super::ConformalPec`]): its host (the free
    /// candidate across its most open piece) and one free neighbour of the
    /// host along each in-plane axis (preferably away from `r`), all on
    /// `r`'s side of the sheet; barycentric weights at `r`'s centroid,
    /// scaled to currents. `None` if there is no valid stencil.
    fn stencil(&self, r: usize) -> Option<Vec<(Target, f64)>> {
        let reg = &self.split.regions[r];
        reg.centroid?;
        let f = &self.split.faces[reg.face];
        let (a, p) = (f.axis, f.node);
        if (self.absorbing)(a, p) {
            return None;
        }
        let mut host: Option<((f64, f64), Target)> = None;
        for &(v, _, ek, pk) in &reg.terms {
            if !self.open(v) {
                continue;
            }
            let fp = self.prepared.frac_ref(v);
            for t in self.across(a, p, ek, pk) {
                if !self.free(t, a) {
                    continue;
                }
                let score = (fp, self.fraction(t));
                if host.is_none_or(|h| score > h.0) {
                    host = Some((score, t));
                }
            }
        }
        let host = host?.1;
        let hn = self.node(host);
        let (b, c) = a.others();
        let scale = [
            self.grid.delta(b, p[b.index()]),
            self.grid.delta(c, p[c.index()]),
        ];
        let xs = self.pos(a, Target::Region(r));
        let rel = |t: Target| {
            let x = self.pos(a, t);
            [(x[0] - xs[0]) / scale[0], (x[1] - xs[1]) / scale[1]]
        };
        let dh = rel(host);
        let mut pts = vec![(host, dh)];
        for (q, d) in [b, c].into_iter().enumerate() {
            let mut pick: Option<(bool, Target, [f64; 2])> = None;
            for (v, ek, pk) in self.terms(host) {
                if !self.open(v) {
                    continue;
                }
                for t in self.across(a, hn, ek, pk) {
                    let tn = self.node(t);
                    if tn[d.index()] == hn[d.index()] || tn == p || !self.free(t, a) {
                        continue;
                    }
                    let rr = rel(t);
                    let toward = rr[q].abs() < dh[q].abs();
                    if pick.is_none_or(|k| k.0 && !toward) {
                        pick = Some((toward, t, rr));
                    }
                }
            }
            match pick {
                Some((_, t, rr)) => pts.push((t, rr)),
                None if dh[q].abs() > 1e-9 => return None,
                None => {}
            }
        }
        let d: Vec<[f64; 2]> = pts.iter().map(|x| x.1).collect();
        let w = interpolation_weights(&d)?;
        let ls = centroid_dual_length(self.grid, a, p, reg.centroid?);
        Some(
            pts.iter()
                .zip(w)
                .map(|(x, w)| (x.0, w * ls / self.grid.dual_edge_length(a, self.node(x.0))))
                .collect(),
        )
    }
}

/// Counts for the report.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct SheetCounts {
    pub edges: usize,
    pub faces: usize,
    pub groups: usize,
    pub merged: usize,
    pub floored: usize,
    pub interpolated: usize,
    pub hosts: usize,
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

    /// Stability terms: `(dt/L, [(array, index, sign)])` per scalar group
    /// (array 3: pieces), the blocks of interpolation hosts, and `√cb` of
    /// the pieces.
    pub(super) fn stability_terms(
        &self,
        n: [usize; 3],
    ) -> (Vec<super::GroupTerms>, Vec<super::BlockTerms<'_>>, Vec<f64>) {
        let flat = |g: &SGroup| -> Vec<(usize, usize, f64)> {
            g.terms
                .iter()
                .map(|&(v, s)| match v {
                    Ref::Main(a, p) => (a, index(n, p), s),
                    Ref::Piece(j) => (3, j, s),
                })
                .collect()
        };
        let op = &self.op;
        let g = op.groups[..op.scalar]
            .iter()
            .filter(|g| g.l > 0.0)
            .map(|g| (g.dt_over_l, flat(g)))
            .collect();
        let b = op
            .blocks
            .iter()
            .map(|b| {
                (
                    b.members.iter().map(|&q| flat(&op.groups[q])).collect(),
                    b.mfactor.as_ref().unwrap_or(&b.factor),
                    b.dt,
                )
            })
            .collect();
        let sq = op.pieces.iter().map(|p| p.cb.max(0.0).sqrt()).collect();
        (g, b, sq)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Piece {
    ca: f64,
    cb: f64,
    /// Capacitance for the energy.
    cap: f64,
    u: f64,
    /// `(group, weight)` of the reduced currents of the regions containing
    /// the piece (signed).
    inc: Vec<(usize, f64)>,
}

/// One reduced current: a merge group, a single region, or an
/// interpolation host (advanced by its [`Block`]).
#[derive(Debug, Clone, PartialEq)]
struct SGroup {
    terms: Vec<(Ref, f64)>,
    da: f64,
    db: f64,
    l: f64,
    /// Resistance (for the blocks' `PᵀRP`).
    r: f64,
    dt_over_l: f64,
    i: f64,
}

impl SGroup {
    fn new(terms: BTreeMap<Ref, f64>, l: f64, r: f64, dt: f64) -> Self {
        let terms: Vec<(Ref, f64)> = terms.into_iter().filter(|t| t.1 != 0.0).collect();
        let (da, db, dt_over_l) = if l > 0.0 {
            let den = l + 0.5 * r * dt;
            ((l - 0.5 * r * dt) / den, dt / den, dt / l)
        } else {
            (0.0, 0.0, 0.0)
        };
        Self {
            terms,
            da,
            db,
            l,
            r,
            dt_over_l,
            i: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Correction {
    axis: Axis,
    node: [usize; 3],
    cbs: f64,
    /// Slots of the region and of the face's primary region.
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
    /// Reduced currents: `..scalar` advanced on their own, the rest
    /// (interpolation hosts) by `blocks`.
    groups: Vec<SGroup>,
    scalar: usize,
    blocks: Vec<Block>,
    /// Region currents read by the kernel faces and corrections: the
    /// reduced currents, then the interpolated regions (`(group, weight)`).
    slots: Vec<Vec<(usize, f64)>>,
    slot_i: Vec<f64>,
    /// Owned faces and the slot they hold.
    writes: Vec<(Axis, [usize; 3], usize)>,
    corrections: Vec<Correction>,
    /// Circulations of the groups (after `post_e`: of `u^{n+1}`, for the
    /// energy look-ahead), next currents and scratch.
    circ: Vec<f64>,
    next: Vec<f64>,
    scratch: Vec<f64>,
}

impl ConformalSheets {
    /// Number of edge pieces (voltages of cut edges).
    pub fn pieces(&self) -> usize {
        self.pieces.len()
    }

    /// Number of reduced region currents (after merging and
    /// interpolation).
    pub fn currents(&self) -> usize {
        self.groups.len()
    }

    fn voltage(&self, v: Ref, f: &Fields<'_>) -> f64 {
        match v {
            Ref::Main(a, p) => f.u.get(Axis::from_index(a).expect("axis"), p) as f64,
            Ref::Piece(j) => self.pieces[j].u,
        }
    }

    fn circulations(&mut self, f: &Fields<'_>) {
        let mut circ = core::mem::take(&mut self.circ);
        for (c, g) in circ.iter_mut().zip(&self.groups) {
            *c = g.terms.iter().map(|&(v, s)| s * self.voltage(v, f)).sum();
        }
        self.circ = circ;
    }

    /// `g^{n+½}` of every reduced current from the currents and the
    /// circulations `c`, into `out`.
    fn advance(&self, c: &[f64], x: &mut Vec<f64>, out: &mut [f64]) {
        for (q, g) in self.groups[..self.scalar].iter().enumerate() {
            out[q] = flush((g.da * g.i - g.db * c[q]) as f32) as f64;
        }
        for b in &self.blocks {
            b.advance(&|q| self.groups[q].i, c, x, out);
        }
    }
}

impl Operator for ConformalSheets {
    fn hooks(&self) -> Hooks {
        Hooks::POST_H | Hooks::POST_E
    }

    fn post_h(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        self.circulations(f);
        let (mut next, mut x) = (
            core::mem::take(&mut self.next),
            core::mem::take(&mut self.scratch),
        );
        self.advance(&self.circ, &mut x, &mut next);
        for (g, &i) in self.groups.iter_mut().zip(&next) {
            g.i = i;
        }
        (self.next, self.scratch) = (next, x);
        for (s, w) in self.slot_i.iter_mut().zip(&self.slots) {
            *s = w.iter().map(|&(g, w)| w * self.groups[g].i).sum();
        }
        for &(a, p, s) in &self.writes {
            f.i.set(a, p, self.slot_i[s] as f32);
        }
    }

    fn post_e(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        for c in &self.corrections {
            let d = c.cbs * (self.slot_i[c.region] - self.slot_i[c.primary]);
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
        // For the energy at the new time (`energy`).
        self.circulations(f);
    }

    /// Pieces plus `½·g^{n+½}ᵀ·M·g^{n+3/2}`, with `g^{n+3/2}` the value the
    /// next `post_h` computes (from the circulations of `post_e`).
    fn energy(&self) -> f64 {
        let mut next = vec![0.0; self.groups.len()];
        self.advance(&self.circ, &mut Vec::new(), &mut next);
        let mut w: f64 = self.pieces.iter().map(|p| 0.5 * p.cap * p.u * p.u).sum();
        for (g, nx) in self.groups[..self.scalar].iter().zip(&next) {
            if g.l > 0.0 {
                w += 0.5 * g.l * g.i * nx;
            }
        }
        for b in &self.blocks {
            for (row, &q) in b.m.iter().zip(&b.members) {
                let mn: f64 = row.iter().map(|&(k, v)| v * next[b.members[k]]).sum();
                w += 0.5 * self.groups[q].i * mn;
            }
        }
        w
    }
}
