//! Conformal PEC geometry: open fractions of the E edges and cell faces
//! cut by solid metal (enabled by [`CompileOptions::conformal`](super::CompileOptions::conformal);
//! the update scheme is in [`crate::engine::conformal`]).
//!
//! # Metal test
//!
//! A point is *in metal* when the highest-priority material or metal
//! property containing it (csx priority query over materials and metals)
//! is a `Metal` property and the containing primitive is a solid
//! (dimension 3). Curves and wires stay with csx's PEC mask; metal sheets
//! off the mesh planes (oblique or curved) are split cell by cell instead,
//! see `sheets` (src/compile/conformal/sheets.rs); metal sheets and
//! conducting sheets on mesh planes keep csx's edges except along
//! outlines oblique to the mesh, whose in-plane edges and faces are cut
//! like those of solid metal, see `planar`
//! (src/compile/conformal/planar.rs).
//!
//! # Edges
//!
//! Only nodes inside the bounding box of the solid metal primitives (plus
//! one line) are queried. Each E edge is first classified by its two end
//! nodes and its midpoint; if all three agree the edge is taken as
//! uniformly open or metal (features thinner than half a cell between the
//! samples are missed). Otherwise the edge is sampled at 9 points and every
//! state change is located by bisection to `2⁻³⁰` of the edge length; the
//! open fraction `f_e` is the total length of the open intervals.
//! Transitions within `1e-6` of an end are moved onto it (a surface
//! through a node otherwise leaves a `2⁻³⁰` sliver on one side).
//!
//! # Surfaces along the mesh
//!
//! With [`ConformalOutlines::Slanted`](super::ConformalOutlines) for
//! solid metal ([`CompileOptions::conformal_solids`](super::CompileOptions::conformal_solids);
//! Cartesian grids) an edge is cut only if the surface is oblique to the
//! mesh at one of its transitions, or bends away from the plane normal
//! to the edge within a quarter of the edge (a round post crossed along
//! a diameter; `Ctx::oblique`); an edge that only
//! surfaces along the mesh cross keeps csx's staircase, and a face is
//! cut if one of its edges is. A box, or the straight side of a prism,
//! is then discretised as without conformal PEC whether its faces are on
//! mesh lines or between them, as with the outlines of sheets on mesh
//! planes (`planar`): a mesh laid out for the staircase (the thirds rule
//! at the edges of a strip) keeps its meaning. `Off` leaves all solid
//! metal to csx.
//!
//! # Faces
//!
//! The open area of a cell face is the polygon formed, in order around the
//! face perimeter, by its open corners and the metal/open transition
//! points on its four edges; between an exit and the next entry point the
//! metal boundary is approximated by a straight chord (the standard
//! conformal face construction, exact for planar surfaces and
//! second-order accurate for smooth ones). Its area by the shoelace
//! formula gives `f_A`, its centroid the position the engine interpolates
//! small faces at. A corner is open if either edge meeting there is open
//! at it.
//!
//! The polygon pairs every transition with the next one around the
//! perimeter. That is wrong for metal that crosses the face between two
//! opposite sides while both are open at their ends (a post, a barrel, a
//! strip narrower than the face: the polygon is the whole face, and the
//! flux would go through the metal), and its chords cut off a corner
//! inside the face and miss a fifth of the face on a pad of one cell in
//! radius. Where a side has two transitions or more, or the surface is
//! more than `SAGITTA` of the shorter side away from the middle of a
//! chord (`polygon_holds`), the face is integrated instead
//! (`integrated_open`: rows of point queries with every change of state
//! bisected, design §5.13.3; Cartesian grids). Metal intruding into a
//! face none of whose edges it crosses is not seen.
//!
//! # Cylindrical grids
//!
//! On a cylindrical grid (design §5.15) the edges and faces are those of
//! the `(ρ, α, z)` lines, and the metal test is made at their `(ρ, α, z)`
//! points. The open fraction of an edge is the same in parameter and in
//! physical length (ρ and z edges are straight, α edges are arcs of
//! constant radius). So is the open area of ρ and α faces (constant
//! metric); a z face has the area element `ρ·dρ·dα`, so its open polygon
//! (straight in `(ρ, α)`) is measured by `∮ ρ²/2 dα` instead of the
//! shoelace formula. Thin conformal sheets (`sheets`) are cut against the
//! `(ρ, α, z)` edges mapped to Cartesian points, with the same metric
//! (src/compile/conformal/sheets/cyl.rs).
//!
//! # PEC mask
//!
//! An edge with `f_e = 0` is PEC. A cut edge (`0 < f_e < 1`) is made
//! non-PEC and listed as a [`ConformalEdge`], unless its PEC flag came
//! from something other than solid metal (csx marks it PEC but its midpoint
//! is not in solid metal: a sheet or wire), in which case it stays PEC.
//! Uncut open edges keep csx's flag. The edges of a face with no open
//! area are PEC: their open part lies on the metal surface, where the
//! tangential E vanishes (left open, such an edge is bordered by
//! zero-area faces and held an undamped local resonance; it happens
//! whenever the surface passes through mesh nodes, e.g. a centred
//! circle of radius `5·h` through `(±3h, ±4h)`).
//!
//! # Narrow slanted metal
//!
//! A cut edge is its open part in series with its covered part, so metal
//! without a row of whole edges along it would be a chain of pieces. With
//! [`CompileOptions::conformal_bridges`](super::CompileOptions::conformal_bridges)
//! (the default) the edges that carry it from one of its nodes to the
//! diagonal neighbour stay whole (`bridges`, design §5.13.3).
//!
//! # Sources
//!
//! E-field excitations on a cut edge are scaled by `f_e`
//! ([`scale_excitations`]): the engine stores `u = E·l_open` there.

use crate::axis::Axis;
use crate::engine::{ConformalEdge, ConformalFace, ConformalPec};
use csx::{BoundingBox, CoordSystem, PropertyKind, PropertyType, TypeMask, Vec3};
use std::collections::HashMap;

mod planar;
mod sheets;

/// Bisection steps per transition.
const BISECT: usize = 30;
/// Samples per edge (intervals) when refining.
const SAMPLES: usize = 8;
/// Transitions closer than this (relative) to an edge end are moved onto
/// the end.
const SNAP: f64 = 1e-6;

/// The conformal geometry and the PEC mask changes (engine layout).
pub(super) struct Geometry {
    /// Cut edges and faces.
    pub conformal: ConformalPec,
    /// `(axis, engine index, pec)` overrides of the PEC mask.
    pub pec: Vec<(Axis, usize, bool)>,
    /// Conducting-sheet edges to add (cut edges csx did not select), with
    /// their property names (`planar`).
    pub sheet_edges: Vec<(crate::engine::SheetEdge, String)>,
    /// `(axis, node, open fraction)` of the conducting-sheet edges their
    /// sheet's outline cuts (`planar`).
    pub sheet_cuts: Vec<(Axis, [usize; 3], f64)>,
    /// `(normal, node)` of the cut faces in the planes of those sheets.
    pub sheet_cut_faces: Vec<(Axis, [usize; 3])>,
    /// Edges kept whole for the narrow slanted conductors they carry
    /// (`bridges`).
    pub bridges: usize,
}

/// Open intervals of one edge: transition parameters (sorted, in (0, 1))
/// and the state at `t = 0`.
#[derive(Debug, Clone)]
struct Profile {
    start_metal: bool,
    transitions: Vec<f64>,
}

impl Profile {
    pub(super) fn uniform(metal: bool) -> Self {
        Self {
            start_metal: metal,
            transitions: Vec::new(),
        }
    }

    /// Drops transitions within [`SNAP`] of the ends (the end takes the
    /// state beyond the transition): a surface through a node is located
    /// to the bisection precision, `2⁻³⁰` of the edge, on one side or the
    /// other; snapping puts it on the node, so the faces on the metal side
    /// have no open area at all (see [`solid_geometry`] on such faces).
    fn snapped(mut self) -> Self {
        while self.transitions.first().is_some_and(|&t| t < SNAP) {
            self.transitions.remove(0);
            self.start_metal = !self.start_metal;
        }
        while self.transitions.last().is_some_and(|&t| t > 1.0 - SNAP) {
            self.transitions.pop();
        }
        self
    }

    /// Open length of the part `[lo, hi]` of the edge (fraction of the
    /// edge).
    pub(super) fn open_between(&self, lo: f64, hi: f64) -> f64 {
        let mut metal = self.start_metal;
        let mut last = 0.0f64;
        let mut open = 0.0;
        for &t in self.transitions.iter().chain(core::iter::once(&1.0)) {
            if !metal {
                open += (t.min(hi) - last.max(lo)).max(0.0);
            }
            last = t;
            metal = !metal;
        }
        open
    }

    /// Whether the edge is in metal at the parameter `t`.
    fn metal_at(&self, t: f64) -> bool {
        self.start_metal ^ (self.transitions.iter().filter(|&&x| x <= t).count() % 2 == 1)
    }

    /// Open length fraction.
    fn open_fraction(&self) -> f64 {
        let mut metal = self.start_metal;
        let mut last = 0.0;
        let mut open = 0.0;
        for &t in self.transitions.iter().chain(core::iter::once(&1.0)) {
            if !metal {
                open += t - last;
            }
            last = t;
            metal = !metal;
        }
        open.clamp(0.0, 1.0)
    }
}

struct Ctx<'a> {
    index: csx::structure::SpatialIndex<'a>,
    lines: [&'a [f64]; 3],
}

impl Ctx<'_> {
    fn metal(&self, p: Vec3) -> bool {
        matches!(self.index.query(p),
            Some(h) if matches!(h.property.kind, PropertyKind::Metal) && h.primitive.dimension() == 3)
    }

    /// Whether the metal surface crossing the edge along `a` from node
    /// `n` at the parameter `t` is oblique to the mesh there: on a line
    /// beside the edge (`SLANT_ACROSS` of its length away, along either
    /// other axis) the metal reaches past the crossing, by `SLANT_ALONG`
    /// of the edge or more (about 0.6° from the edge's normal). A line
    /// beside the edge that is outside the metal altogether (the edge
    /// lies in a face of the metal) says nothing.
    fn oblique(&self, a: usize, n: [usize; 3], t: f64) -> bool {
        use planar::{SLANT_ACROSS, SLANT_ALONG};
        let p0 = self.node(n);
        let len = self.lines[a][n[a] + 1] - self.lines[a][n[a]];
        let at = |dt: f64, o: usize, w: f64| {
            let mut p = p0;
            p.0[a] += (t + dt) * len;
            p.0[o] += w * len;
            self.metal(p)
        };
        let flat = [(a + 1) % 3, (a + 2) % 3].into_iter().any(|o| {
            [SLANT_ACROSS, -SLANT_ACROSS]
                .into_iter()
                .any(|w| at(-SLANT_ALONG, o, w) && at(SLANT_ALONG, o, w))
        });
        flat || {
            let (before, after) = (at(-SLANT_ALONG, a, 0.0), at(SLANT_ALONG, a, 0.0));
            before != after
                && [(a + 1) % 3, (a + 2) % 3]
                    .into_iter()
                    .any(|o| planar::curved(if before { -1.0 } else { 1.0 }, |dt, w| at(dt, o, w)))
        }
    }

    fn node(&self, n: [usize; 3]) -> Vec3 {
        Vec3::new(
            self.lines[0][n[0]],
            self.lines[1][n[1]],
            self.lines[2][n[2]],
        )
    }

    /// Profile of the edge along `a` from node `n` with end states `s0`,
    /// `s1` and midpoint state `sm`.
    fn profile(&self, a: usize, n: [usize; 3], s0: bool, sm: bool, s1: bool) -> Profile {
        if s0 == sm && sm == s1 {
            return Profile::uniform(s0);
        }
        let p0 = self.node(n);
        let len = self.lines[a][n[a] + 1] - self.lines[a][n[a]];
        let at = |t: f64| {
            let mut p = p0;
            p.0[a] += t * len;
            p
        };
        let mut states = [false; SAMPLES + 1];
        for (q, s) in states.iter_mut().enumerate() {
            *s = match q {
                0 => s0,
                q if 2 * q == SAMPLES => sm,
                SAMPLES => s1,
                q => self.metal(at(q as f64 / SAMPLES as f64)),
            };
        }
        refine(&states, |t| self.metal(at(t)))
    }
}

/// The profile of an edge from its states at the `SAMPLES + 1` equally
/// spaced samples and the state `metal(t)` at any parameter: every state
/// change between two samples is bisected, transitions at the ends are
/// snapped onto them.
fn refine(states: &[bool; SAMPLES + 1], metal: impl Fn(f64) -> bool) -> Profile {
    let mut transitions = Vec::new();
    for q in 0..SAMPLES {
        if states[q] == states[q + 1] {
            continue;
        }
        let (mut lo, mut hi) = (q as f64 / SAMPLES as f64, (q + 1) as f64 / SAMPLES as f64);
        for _ in 0..BISECT {
            let mid = 0.5 * (lo + hi);
            if metal(mid) == states[q] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        transitions.push(0.5 * (lo + hi));
    }
    Profile {
        start_metal: states[0],
        transitions,
    }
    .snapped()
}

/// The edge between two neighbouring nodes: `(axis index, lower node)`.
fn edge_between(u: [usize; 3], v: [usize; 3]) -> (usize, [usize; 3]) {
    let a = (0..3).find(|&a| u[a] != v[a]).unwrap_or(0);
    (a, if u[a] < v[a] { u } else { v })
}

/// What [`bridges`] asks about the conductor.
struct Conductor<'a> {
    /// Whether the node is in the conductor.
    covered: &'a dyn Fn([usize; 3]) -> bool,
    /// Whether the straight line between two nodes is in the conductor.
    joined: &'a dyn Fn([usize; 3], [usize; 3]) -> bool,
    /// Whether the edge conducts along its whole length as discretised (a
    /// PEC or sheet edge that is not cut).
    conducts: &'a dyn Fn((usize, [usize; 3])) -> bool,
    /// Covered fraction of the edge.
    coverage: &'a dyn Fn((usize, [usize; 3])) -> f64,
}

/// The edges that keep a narrow slanted conductor in one piece (design
/// §5.13.3).
///
/// A cut edge is its open part in series with its covered part: current
/// along the conductor has to pass the open part as displacement current.
/// Where the conductor is wide, whole edges inside it carry the current
/// and the cut edges only place its rim. A strip under about a cell and a
/// half wide at 45° has no row of whole edges, so every path along it
/// would cross open parts and the strip would be a chain of pieces
/// coupled by capacitors of a few femtofarads.
///
/// So wherever two nodes in the conductor are diagonal neighbours in a
/// mesh plane (`planes`: pairs of axis indices), the conductor contains
/// the line between them, and neither of the two paths of two edges
/// around the cell between them conducts, the path whose less covered
/// edge is covered most is returned: its edges are to be whole conductor
/// edges, as the staircase has them. A conductor with whole edges on one
/// of the paths is not touched, so a resolved outline keeps every cut;
/// two conductors a fraction of a cell apart are not joined (the line
/// between their nodes crosses the gap), where the staircase can short
/// them.
fn bridges(
    nodes: &[[usize; 3]],
    planes: &[(usize, usize)],
    dims: [usize; 3],
    c: &Conductor<'_>,
) -> Vec<(usize, [usize; 3])> {
    let mut kept: Vec<(usize, [usize; 3])> = Vec::new();
    let mut set: std::collections::HashSet<(usize, [usize; 3])> = Default::default();
    for &u in nodes {
        for &(b, cc) in planes {
            for up in [true, false] {
                // w = u + e_b ± e_c.
                if u[b] + 1 >= dims[b] || (up && u[cc] + 1 >= dims[cc]) || (!up && u[cc] == 0) {
                    continue;
                }
                let mut w = u;
                w[b] += 1;
                w[cc] = if up { w[cc] + 1 } else { w[cc] - 1 };
                if !(c.covered)(w) {
                    continue;
                }
                let (mut v1, mut v2) = (u, u);
                v1[b] = w[b];
                v2[cc] = w[cc];
                let paths = [v1, v2].map(|v| [edge_between(u, v), edge_between(v, w)]);
                let on = |k: (usize, [usize; 3])| set.contains(&k) || (c.conducts)(k);
                if paths.iter().any(|p| p.iter().all(|&k| on(k))) || !(c.joined)(u, w) {
                    continue;
                }
                let score =
                    |p: &[(usize, [usize; 3]); 2]| (c.coverage)(p[0]).min((c.coverage)(p[1]));
                let best = if score(&paths[1]) > score(&paths[0]) {
                    paths[1]
                } else {
                    paths[0]
                };
                for k in best {
                    if !(c.conducts)(k) && set.insert(k) {
                        kept.push(k);
                    }
                }
            }
        }
    }
    kept
}

/// Union of the bounding boxes of solid metal primitives, in the grid's
/// coordinates (cylindrical: `(ρ, α, z)`, conservative; the caller ignores
/// the α range, which csx normalises to `[−π, π)`).
fn metal_bbox(model: &csx::ContinuousStructure) -> BoundingBox {
    let cyl = model.grid.coord_system == CoordSystem::Cylindrical;
    let mut bb = BoundingBox::EMPTY;
    for p in model.properties_of(TypeMask::of(PropertyType::Metal)) {
        for q in &p.primitives {
            if q.dimension() == 3 {
                let b = if cyl {
                    q.bounding_box_in(CoordSystem::Cylindrical, model.coord_system)
                } else {
                    q.bounding_box(model.coord_system)
                };
                bb = bb.union(&b);
            }
        }
    }
    bb
}

/// Node index range covering `[lo, hi]` plus one line on each side.
fn range(lines: &[f64], lo: f64, hi: f64) -> core::ops::Range<usize> {
    let a = lines.partition_point(|&x| x < lo).saturating_sub(1);
    let b = (lines.partition_point(|&x| x <= hi) + 1).min(lines.len());
    a..b.max(a)
}

/// Runs `f(i)` for `i` in `r` on up to `threads` threads, results in
/// order.
fn par<T: Send>(
    threads: usize,
    r: core::ops::Range<usize>,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let n = r.len();
    let t = threads.clamp(1, n.max(1));
    if t == 1 {
        return r.map(f).collect();
    }
    let chunk = n.div_ceil(t);
    let f = &f;
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..t)
            .map(|q| {
                let lo = r.start + q * chunk;
                let hi = (lo + chunk).min(r.end);
                s.spawn(move || (lo..hi).map(f).collect::<Vec<T>>())
            })
            .collect();
        hs.into_iter()
            .flat_map(|h| h.join().expect("conformal worker panicked"))
            .collect()
    })
}

/// What [`geometry`] needs besides the model.
pub(super) struct Request<'a> {
    /// The engine-layout PEC mask after csx and sheets.
    pub pec: [&'a [bool]; 3],
    /// csx's relative snapping tolerance.
    pub snap: f64,
    /// Threads of the point queries (the result does not depend on it).
    pub threads: usize,
    /// Outlines of sheets on mesh planes (`planar`).
    pub outlines: super::ConformalOutlines,
    /// Which surfaces of solid metal cut the edges they cross.
    pub solids: super::ConformalOutlines,
    /// Whether narrow slanted conductors keep whole edges (`bridges`).
    pub bridges: bool,
    /// The edges `(axis index, node)` that carry a lumped element.
    pub lumped: &'a std::collections::HashSet<(usize, [usize; 3])>,
    /// The conducting-sheet edges csx selected, or `None` where the
    /// outlines of conducting sheets are not cut (openEMS's sheet model).
    pub sheet_edges: Option<&'a [crate::engine::SheetEdge]>,
}

/// Computes the conformal geometry of `model` on its grid: solid metal,
/// thin PEC sheets off the mesh planes (`sheets`) and the outlines of
/// sheets on mesh planes (`planar`, Cartesian grids). csx warnings about
/// the sheets handled here are removed from `warnings`.
pub(super) fn geometry(
    model: &csx::ContinuousStructure,
    req: &Request<'_>,
    warnings: &mut Vec<String>,
) -> Geometry {
    let (pec, snap, threads) = (&req.pec, req.snap, req.threads);
    let bridge = req.bridges;
    // The edges that continue a lumped element's column beyond its ends.
    let dims = [0, 1, 2].map(|a| model.grid.lines(a).len());
    let mut terminals: std::collections::HashSet<(usize, [usize; 3])> = Default::default();
    for &(a, n) in req.lumped {
        if n[a] > 0 {
            let mut m = n;
            m[a] -= 1;
            terminals.insert((a, m));
        }
        if n[a] + 2 < dims[a] {
            let mut m = n;
            m[a] += 1;
            terminals.insert((a, m));
        }
    }
    terminals.retain(|k| !req.lumped.contains(k));
    let sheets = planar::Sheets::new(model, snap);
    let (mut out, solid) =
        solid_geometry(model, pec, threads, req.solids, bridge, &terminals, &sheets);
    // The metal test of the solid geometry (grid coordinates), for sheets
    // ending on a metal surface.
    let index = model.spatial_index(
        model.grid.coord_system,
        TypeMask::MATERIALS | TypeMask::of(PropertyType::Metal),
    );
    let ctx = Ctx {
        index,
        lines: [0, 1, 2].map(|a| model.grid.lines(a)),
    };
    let solid_at = |p: Vec3| !solid.is_empty() && ctx.metal(p);
    let (edges, faces, sheet_pec) = if model.grid.coord_system == CoordSystem::Cylindrical {
        sheets::cylindrical_geometry(model, snap, threads, &solid, &solid_at, warnings)
    } else {
        sheets::geometry(model, snap, threads, &solid, &solid_at, warnings)
    };
    out.pec.extend(sheet_pec);
    // Outlines of the sheets on mesh planes, around what is cut already.
    let mut mask: [Vec<bool>; 3] = pec.map(<[bool]>::to_vec);
    for &(axis, k, v) in &out.pec {
        mask[axis.index()][k] = v;
    }
    let mut taken_edges: std::collections::HashSet<planar::Key> = solid.keys().copied().collect();
    taken_edges.extend(edges.iter().map(|e| (e.axis.index(), e.node)));
    let taken_faces: std::collections::HashSet<planar::Key> = out
        .conformal
        .faces()
        .iter()
        .map(|f| (f.axis.index(), f.node))
        .chain(faces.iter().map(|f| (f.axis.index(), f.node)))
        .collect();
    let sheet_edges: std::collections::HashSet<planar::Key> = req
        .sheet_edges
        .unwrap_or(&[])
        .iter()
        .map(|e| (e.axis.index(), e.node))
        .collect();
    let outlines = planar::geometry(
        model,
        &mask,
        snap,
        threads,
        planar::Rules {
            mode: req.outlines,
            conducting: req.sheet_edges.is_some(),
            bridge,
        },
        &planar::Taken {
            edges: &taken_edges,
            faces: &taken_faces,
            sheet_edges: &sheet_edges,
            lumped: req.lumped,
            terminals: &terminals,
        },
    );
    out.pec.extend(outlines.pec);
    for e in outlines.edges {
        out.conformal.push_edge(e);
    }
    for (f, centroid) in outlines.faces {
        out.conformal.push_face_with_centroid(f, centroid);
    }
    for (axis, node) in outlines.walls {
        out.conformal.push_wall(axis, node);
    }
    out.sheet_edges = outlines.sheet_edges;
    out.sheet_cuts = outlines.sheet_cuts;
    out.sheet_cut_faces = outlines.sheet_cut_faces;
    out.bridges += outlines.bridges;
    for e in edges {
        out.conformal.push_sheet_edge(e);
    }
    for f in faces {
        out.conformal.push_sheet_face(f);
    }
    out
}

/// Scales the amplitudes of E-field excitations on cut edges by their open
/// fraction `f_e`: the engine stores `u = E·l_open` there (design §5.13),
/// so a source of field `E` adds `E·l·f_e`, not `E·l`. Without it a soft
/// source on a sliver edge (`f_e ≪ 1`, capacitance `C/f_e`) charges it with
/// `1/f_e` times the intended field and energy, which its tiny `cb` then
/// barely relaxes: a quasi-static charge that keeps the energy end
/// criterion from ever being met.
pub(super) fn scale_excitations(
    c: &ConformalPec,
    node_map: &super::NodeMap,
    excitations: &mut [csx::discretize::ExcitationField],
) {
    if c.edges().is_empty() {
        return;
    }
    let cut: HashMap<(Axis, [usize; 3]), f64> = c
        .edges()
        .iter()
        .map(|e| ((e.axis, e.node), e.fraction))
        .collect();
    for ex in excitations {
        if !ex.excitation_type.is_electric() {
            continue;
        }
        for (edge, a) in &mut ex.amplitudes {
            if let Some(f) = cut.get(&node_map.edge(*edge)) {
                *a *= f;
            }
        }
    }
}

/// Open intervals of the E edges touched by solid metal, by `(axis index,
/// node)` (edges not listed are open).
type SolidProfiles = HashMap<(usize, [usize; 3]), Profile>;

/// The solid-metal part of [`geometry`], and the edge profiles (for the
/// sheets touching the metal).
fn solid_geometry(
    model: &csx::ContinuousStructure,
    pec: &[&[bool]; 3],
    threads: usize,
    mode: super::ConformalOutlines,
    bridge: bool,
    terminals: &std::collections::HashSet<(usize, [usize; 3])>,
    sheets: &planar::Sheets<'_>,
) -> (Geometry, SolidProfiles) {
    let lines = [0, 1, 2].map(|a| model.grid.lines(a));
    let dims = lines.map(<[f64]>::len);
    let mut out = Geometry {
        conformal: ConformalPec::new(),
        pec: Vec::new(),
        sheet_edges: Vec::new(),
        sheet_cuts: Vec::new(),
        sheet_cut_faces: Vec::new(),
        bridges: 0,
    };
    let bb = metal_bbox(model);
    if bb.is_empty()
        || !(bb.min.is_finite() && bb.max.is_finite())
        || mode == super::ConformalOutlines::Off
    {
        return (out, HashMap::new());
    }
    let cyl = model.grid.coord_system == CoordSystem::Cylindrical;
    // (The test of an oblique surface is Cartesian.)
    let all = mode == super::ConformalOutlines::All || cyl;
    let ctx = Ctx {
        index: model.spatial_index(
            model.grid.coord_system,
            TypeMask::MATERIALS | TypeMask::of(PropertyType::Metal),
        ),
        lines,
    };
    let r: [core::ops::Range<usize>; 3] = core::array::from_fn(|a| {
        if cyl && a == 1 {
            0..dims[1]
        } else {
            range(lines[a], bb.min[a], bb.max[a])
        }
    });
    let (rx, ry, rz) = (r[0].clone(), r[1].clone(), r[2].clone());
    let eidx = |n: [usize; 3]| (n[0] * dims[1] + n[1]) * dims[2] + n[2];
    // Node states inside the range (outside: open).
    let planes = par(threads, rx.clone(), |i| {
        let mut v = Vec::with_capacity(ry.len() * rz.len());
        for j in ry.clone() {
            for k in rz.clone() {
                v.push(ctx.metal(ctx.node([i, j, k])));
            }
        }
        v
    });
    let state = |n: [usize; 3]| -> bool {
        if (0..3).any(|a| !r[a].contains(&n[a])) {
            return false;
        }
        let (i, j, k) = (n[0] - r[0].start, n[1] - r[1].start, n[2] - r[2].start);
        planes[i][j * rz.len() + k]
    };
    // Edge profiles: (axis, node) → profile, for edges with any state
    // change; plus the midpoint state of every edge in range.
    type EdgeOut = (usize, [usize; 3], bool, Profile, bool);
    let edges: Vec<Vec<EdgeOut>> = par(threads, rx.clone(), |i| {
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
                    let mut mid = ctx.node(n);
                    mid.0[a] = 0.5 * (lines[a][n[a]] + lines[a][n[a] + 1]);
                    let sm = ctx.metal(mid);
                    let (s0, s1) = (state(n), state(m));
                    if s0 || sm || s1 {
                        let prof = ctx.profile(a, n, s0, sm, s1);
                        // Cut where a surface crosses the edge obliquely
                        // (module docs, *Surfaces along the mesh*).
                        let cut = all || prof.transitions.iter().any(|&t| ctx.oblique(a, n, t));
                        v.push((a, n, sm, prof, cut));
                    }
                }
            }
        }
        v
    });
    let mut profiles: HashMap<(usize, [usize; 3]), Profile> = HashMap::new();
    let mut cut = Vec::new();
    for (a, n, sm, prof, oblique) in edges.into_iter().flatten() {
        let axis = Axis::from_index(a).expect("axis in 0..3");
        let f = prof.open_fraction();
        let k = eidx(n);
        if f <= 0.0 {
            out.pec.push((axis, k, true));
        } else if f < 1.0 {
            if (pec[a][k] && !sm)
                || !oblique
                || (sm && terminals.contains(&(a, n)))
                || sheets.whole(a, n)
            {
                // PEC from a sheet or wire: keep it. A surface along the
                // mesh: csx's staircase decides. The edge that continues
                // a lumped element's column into the metal stays whole,
                // or the column would hang on its open part. An edge in
                // a sheet on its mesh plane (a barrel through a pad or a
                // plane) is the sheet's, a conductor from end to end: it
                // has no open part, and a conducting sheet's edge is its
                // operator's.
            } else {
                out.pec.push((axis, k, false));
                cut.push(ConformalEdge {
                    axis,
                    node: n,
                    fraction: f,
                });
            }
        }
        profiles.insert((a, n), prof);
    }
    // Narrow slanted metal stays in one piece: the edges that carry it
    // stay whole (`bridges`; Cartesian grids).
    if !cyl && bridge {
        let nodes: Vec<[usize; 3]> = rx
            .clone()
            .flat_map(|i| {
                let (ry, rz) = (ry.clone(), rz.clone());
                ry.flat_map(move |j| rz.clone().map(move |k| [i, j, k]))
            })
            .filter(|&n| state(n))
            .collect();
        let is_cut: std::collections::HashSet<(usize, [usize; 3])> =
            cut.iter().map(|e| (e.axis.index(), e.node)).collect();
        let whole = bridges(
            &nodes,
            &[(0, 1), (1, 2), (2, 0)],
            dims,
            &Conductor {
                covered: &state,
                joined: &|u, w| {
                    let (p, q) = (ctx.node(u), ctx.node(w));
                    (1..8).all(|m| ctx.metal(p + (q - p) * (m as f64 / 8.0)))
                },
                conducts: &|key| {
                    !is_cut.contains(&key)
                        && (pec[key.0][eidx(key.1)]
                            || profiles.get(&key).is_some_and(|p| p.open_fraction() <= 0.0))
                },
                coverage: &|key| profiles.get(&key).map_or(0.0, |p| 1.0 - p.open_fraction()),
            },
        );
        for (a, n) in whole {
            let axis = Axis::from_index(a).expect("axis in 0..3");
            out.pec.push((axis, eidx(n), true));
            cut.retain(|e| (e.axis, e.node) != (axis, n));
            out.bridges += 1;
        }
    }
    // Edges of faces without open area: the face is metal up to its
    // boundary, so an open part of such an edge lies on the metal surface,
    // where the tangential E vanishes; they are made PEC. (Left open, an
    // edge on the surface is bordered by faces of zero area that cannot
    // carry its circulation: measured on a circular guide meshed through
    // its extreme points, it held a local resonance that never decayed.)
    let mut surface: std::collections::HashSet<(usize, [usize; 3])> = Default::default();
    let is_cut: std::collections::HashSet<(usize, [usize; 3])> =
        cut.iter().map(|e| (e.axis.index(), e.node)).collect();
    // Faces.
    let profile = |a: usize, n: [usize; 3]| -> Profile {
        profiles
            .get(&(a, n))
            .cloned()
            .unwrap_or_else(|| Profile::uniform(state(n)))
    };
    for i in rx {
        for j in ry.clone() {
            for k in rz.clone() {
                let n = [i, j, k];
                for axis in Axis::ALL {
                    let (bb, cc) = axis.others();
                    let (b, c) = (bb.index(), cc.index());
                    if n[b] + 1 >= dims[b] || n[c] + 1 >= dims[c] {
                        continue;
                    }
                    let (mut nb, mut nc) = (n, n);
                    nb[b] += 1;
                    nc[c] += 1;
                    let keys = [(b, n), (c, nb), (b, nc), (c, n)];
                    let corners = [
                        n,
                        nb,
                        {
                            let mut q = nb;
                            q[c] += 1;
                            q
                        },
                        nc,
                    ];
                    let any = keys.iter().any(|key| profiles.contains_key(key))
                        || corners.iter().any(|&q| state(q));
                    if !any {
                        continue;
                    }
                    let lb = lines[b][n[b] + 1] - lines[b][n[b]];
                    let lc = lines[c][n[c] + 1] - lines[c][n[c]];
                    let e = [profile(b, n), profile(c, nb), profile(b, nc), profile(c, n)];
                    let at = |u: f64, v: f64| {
                        let mut p = ctx.node(n);
                        p.0[b] += u * lb;
                        p.0[c] += v * lc;
                        ctx.metal(p)
                    };
                    let whole = !cyl
                        && e.iter().any(|p| !p.transitions.is_empty())
                        && !polygon_holds(&e, lb, lc, at);
                    // z faces of a cylindrical grid: b = ρ, c = α.
                    let (fa, centroid) = if whole {
                        integrated_open(at)
                    } else if cyl && axis == Axis::Z {
                        (
                            face_fraction_annular(e.clone(), lines[0][n[0]], lb, lc),
                            open_centroid(&e, lb, lc),
                        )
                    } else {
                        (face_fraction(e.clone(), lb, lc), open_centroid(&e, lb, lc))
                    };
                    if fa <= 0.0 {
                        surface.extend(keys);
                        continue;
                    }
                    // (A face is cut if one of its edges is.)
                    if fa < 1.0 && (all || keys.iter().any(|key| is_cut.contains(key))) {
                        out.conformal.push_face_with_centroid(
                            ConformalFace {
                                axis,
                                node: n,
                                fraction: fa,
                            },
                            centroid,
                        );
                    }
                }
            }
        }
    }
    for &(a, n) in &surface {
        let open = profiles
            .get(&(a, n))
            .map_or(!state(n), |p| p.open_fraction() > 0.0);
        if open {
            let axis = Axis::from_index(a).expect("axis in 0..3");
            out.pec.push((axis, eidx(n), true));
        }
    }
    for e in cut {
        if !surface.contains(&(e.axis.index(), e.node)) {
            out.conformal.push_edge(e);
        }
    }
    profiles.retain(|_, p| p.start_metal || !p.transitions.is_empty());
    (out, profiles)
}

/// The open polygon of a face (see [`face_fraction`]) in local `(u, v)`
/// coordinates.
fn open_polygon(e: &[Profile; 4], lb: f64, lc: f64) -> Vec<(f64, f64)> {
    tagged_polygon(e, lb, lc).into_iter().map(|p| p.0).collect()
}

/// [`open_polygon`] with, for each point, the face side (0…3) it is a
/// transition on (`None`: a corner).
fn tagged_polygon(e: &[Profile; 4], lb: f64, lc: f64) -> Vec<((f64, f64), Option<usize>)> {
    let pos = |q: usize, t: f64| match q {
        0 => (t * lb, 0.0),
        1 => (lb, t * lc),
        2 => (t * lb, lc),
        _ => (0.0, t * lc),
    };
    // Per edge in perimeter order: (metal at its start, transitions, metal
    // at its end).
    let walk: Vec<(bool, Vec<f64>, bool)> = e
        .iter()
        .enumerate()
        .map(|(q, p)| {
            let end_metal = p.start_metal ^ (p.transitions.len() % 2 == 1);
            if q < 2 {
                (p.start_metal, p.transitions.clone(), end_metal)
            } else {
                let ts = p.transitions.iter().rev().copied().collect();
                (end_metal, ts, p.start_metal)
            }
        })
        .collect();
    let mut poly = Vec::new();
    for (q, (start_metal, ts, _)) in walk.iter().enumerate() {
        let t0 = if q < 2 { 0.0 } else { 1.0 };
        // A corner is open if either edge meeting there is open at it
        // (they can disagree when a surface through the corner was snapped
        // onto it on one edge only).
        if !start_metal || !walk[(q + 3) % 4].2 {
            poly.push((pos(q, t0), None));
        }
        for &t in ts {
            poly.push((pos(q, t), Some(q)));
        }
    }
    poly
}

/// Whether the open polygon of a face ([`open_polygon`]) can be trusted
/// with its area; `metal(u, v)` is the state at a point of the face in
/// units of its sides. It cannot where one side of the face has two
/// transitions or more (metal crossing the face between two opposite
/// sides, a post or a strip narrower than the face, is paired wrongly by
/// the walk around the perimeter: the polygon keeps the whole face), nor
/// where the surface is more than [`SAGITTA`] of the shorter side away
/// from one of the chords at its middle (a corner in the face, or a
/// radius of about a cell). Such faces are integrated
/// ([`integrated_open`]).
fn polygon_holds(e: &[Profile; 4], lb: f64, lc: f64, metal: impl Fn(f64, f64) -> bool) -> bool {
    if e.iter().any(|p| p.transitions.len() >= 2) {
        return false;
    }
    let poly = tagged_polygon(e, lb, lc);
    let d = SAGITTA * lb.min(lc);
    for q in 0..poly.len() {
        let ((x0, y0), s0) = poly[q];
        let ((x1, y1), s1) = poly[(q + 1) % poly.len()];
        let (Some(s0), Some(s1)) = (s0, s1) else {
            continue;
        };
        if s0 == s1 {
            continue;
        }
        let len = (x1 - x0).hypot(y1 - y0);
        if len <= 0.0 {
            continue;
        }
        let (nx, ny) = ((y1 - y0) / len * d, -(x1 - x0) / len * d);
        let (mx, my) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
        let at = |sx: f64, sy: f64| metal((mx + sx) / lb, (my + sy) / lc);
        if at(nx, ny) == at(-nx, -ny) {
            return false;
        }
    }
    true
}

/// Largest distance of the surface from a chord of the open polygon, in
/// units of the shorter side of the face, for which the polygon's area is
/// used ([`polygon_holds`]): the sagitta of a chord across a cell of side
/// `h` on a circle of radius `R` is about `h²/(4R)`, so circles of four
/// cells in radius and more keep the polygon.
const SAGITTA: f64 = 0.06;

/// Centroid of the open polygon of a face (see [`face_fraction`]) in
/// units of the sides `lb`, `lc`, `[0.5, 0.5]` if it is degenerate. On a
/// cylindrical grid this is the centroid in the `(ρ, α)` parameters, not
/// under the `ρ·dρ·dα` measure: the two differ by `O(h²/ρ)`, which keeps
/// the interpolation of the engine second-order accurate.
fn open_centroid(e: &[Profile; 4], lb: f64, lc: f64) -> [f64; 2] {
    let poly = open_polygon(e, lb, lc);
    let (mut a2, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for q in 0..poly.len() {
        let (x0, y0) = poly[q];
        let (x1, y1) = poly[(q + 1) % poly.len()];
        let cr = x0 * y1 - x1 * y0;
        a2 += cr;
        cx += (x0 + x1) * cr;
        cy += (y0 + y1) * cr;
    }
    if poly.len() < 3 || a2.abs() <= 1e-12 * lb * lc {
        return [0.5, 0.5];
    }
    [
        (cx / (3.0 * a2 * lb)).clamp(0.0, 1.0),
        (cy / (3.0 * a2 * lc)).clamp(0.0, 1.0),
    ]
}

/// Rows of [`integrated_open`].
const ROWS: usize = 16;

/// Open length, its first moment and the number of state changes along
/// the line `v` of the unit face: the state is sampled at the
/// `SAMPLES + 1` points of an edge and every change bisected, as on the
/// edges.
fn open_row(metal: &impl Fn(f64, f64) -> bool, v: f64) -> (f64, f64, usize) {
    const STEPS: usize = 24;
    let (mut len, mut moment, mut changes) = (0.0, 0.0, 0);
    let mut state = metal(0.0, v);
    let mut last = 0.0;
    for q in 1..=SAMPLES {
        let u1 = q as f64 / SAMPLES as f64;
        let s1 = metal(u1, v);
        if s1 == state {
            continue;
        }
        let (mut lo, mut hi) = ((q - 1) as f64 / SAMPLES as f64, u1);
        for _ in 0..STEPS {
            let mid = 0.5 * (lo + hi);
            if metal(mid, v) == state {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = 0.5 * (lo + hi);
        if !state {
            len += t - last;
            moment += 0.5 * (t * t - last * last);
        }
        last = t;
        state = s1;
        changes += 1;
    }
    if !state {
        len += 1.0 - last;
        moment += 0.5 * (1.0 - last * last);
    }
    (len, moment, changes)
}

/// Open area fraction and open-area centroid (in units of the sides) of a
/// face by direct integration: `metal(u, v)` is the state at the point
/// `(u, v)` of the unit face. The face is cut into [`ROWS`] rows along
/// `v`, each with the open length on its centre line ([`open_row`]).
/// Where two neighbouring rows differ in their number of state changes, a
/// surface along the rows may lie between their centre lines (the side of
/// a box, a corner): the height at which the number changes is bisected,
/// and the strip between the two lines is integrated in two parts, below
/// that height and above it. Unlike the polygon of
/// [`open_polygon`] this sees metal that crosses the face between two
/// opposite edges (a post or a strip thinner than the face, both of whose
/// edges are open at their ends), metal that bulges into the face without
/// crossing an edge, corners, and the curvature of an outline whose
/// radius is about a cell.
fn integrated_open(metal: impl Fn(f64, f64) -> bool) -> (f64, [f64; 2]) {
    const STEPS: usize = 20;
    /// Rows of each part of a strip in which the state changes differ.
    const SUB: usize = 8;
    let h = 1.0 / ROWS as f64;
    let rows: Vec<(f64, f64, usize)> = (0..ROWS)
        .map(|r| open_row(&metal, (r as f64 + 0.5) * h))
        .collect();
    let (mut area, mut mu, mut mv) = (0.0, 0.0, 0.0);
    for (r, &(len, moment, _)) in rows.iter().enumerate() {
        area += len * h;
        mu += moment * h;
        mv += len * h * (r as f64 + 0.5) * h;
    }
    for r in 0..ROWS - 1 {
        let (a, b) = (rows[r], rows[r + 1]);
        if a.2 == b.2 {
            continue;
        }
        // The row `a` holds up to `v`, the row `b` from there.
        let (mut lo, mut hi) = ((r as f64 + 0.5) * h, (r as f64 + 1.5) * h);
        for _ in 0..STEPS {
            let mid = 0.5 * (lo + hi);
            if open_row(&metal, mid).2 == a.2 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // Between the two centre lines the midpoint rule gave each row
        // half of the strip; each part is integrated by itself instead
        // (a surface that touches a row, a circle's lowest point, makes
        // the open length a square root of the height there).
        let (va, vb) = ((r as f64 + 0.5) * h, (r as f64 + 1.5) * h);
        area -= 0.5 * h * (a.0 + b.0);
        mu -= 0.5 * h * (a.1 + b.1);
        mv -= 0.5 * h * (a.0 * va + b.0 * vb);
        for (from, to) in [(va, lo), (hi, vb)] {
            let step = (to - from) / SUB as f64;
            for q in 0..SUB {
                let at = from + (q as f64 + 0.5) * step;
                let (len, moment, _) = open_row(&metal, at);
                area += len * step;
                mu += moment * step;
                mv += len * step * at;
            }
        }
    }
    if area <= 1e-9 {
        return (0.0, [0.5, 0.5]);
    }
    if area >= 1.0 - 1e-9 {
        return (1.0, [0.5, 0.5]);
    }
    (
        area,
        [(mu / area).clamp(0.0, 1.0), (mv / area).clamp(0.0, 1.0)],
    )
}

/// [`face_fraction`] for a z face of a cylindrical grid, `u = ρ − ρ0`
/// along the `lb` side and `v = α` along `lc`: the open area
/// `∬ ρ dρ dα = ∮ ρ²/2 dα` over the polygon (straight sides in `(ρ, α)`:
/// `Δα·(ρ_0² + ρ_0ρ_1 + ρ_1²)/6` per side) over the face area
/// `lc·((ρ0 + lb)² − ρ0²)/2`.
fn face_fraction_annular(e: [Profile; 4], rho0: f64, lb: f64, lc: f64) -> f64 {
    let poly = open_polygon(&e, lb, lc);
    if poly.len() < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for q in 0..poly.len() {
        let (u0, v0) = poly[q];
        let (u1, v1) = poly[(q + 1) % poly.len()];
        let (r0, r1) = (rho0 + u0, rho0 + u1);
        a += (v1 - v0) * (r0 * r0 + r0 * r1 + r1 * r1) / 6.0;
    }
    let total = lc * 0.5 * ((rho0 + lb) * (rho0 + lb) - rho0 * rho0);
    if total <= 0.0 {
        return 1.0;
    }
    let f = a.abs() / total;
    // (Rounding of the uncut face.)
    if f > 1.0 - 1e-12 { 1.0 } else { f.max(0.0) }
}

/// Open area fraction of a face from the profiles of its edges, in
/// perimeter order: `[b at n, c at n+e_b, b at n+e_c, c at n]` (the last
/// two are traversed backwards); `lb`, `lc` are the side lengths.
fn face_fraction(e: [Profile; 4], lb: f64, lc: f64) -> f64 {
    let poly = open_polygon(&e, lb, lc);
    if poly.len() < 3 {
        return 0.0;
    }
    let mut a2 = 0.0;
    for q in 0..poly.len() {
        let (x0, y0) = poly[q];
        let (x1, y1) = poly[(q + 1) % poly.len()];
        a2 += x0 * y1 - x1 * y0;
    }
    (0.5 * a2.abs() / (lb * lc)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(start_metal: bool, t: &[f64]) -> Profile {
        Profile {
            start_metal,
            transitions: t.to_vec(),
        }
    }

    #[test]
    fn edge_fractions() {
        assert_eq!(p(false, &[]).open_fraction(), 1.0);
        assert_eq!(p(true, &[]).open_fraction(), 0.0);
        assert!((p(false, &[0.25]).open_fraction() - 0.25).abs() < 1e-15);
        assert!((p(true, &[0.25]).open_fraction() - 0.75).abs() < 1e-15);
        assert!((p(false, &[0.25, 0.5]).open_fraction() - 0.75).abs() < 1e-15);
    }

    #[test]
    fn face_polygons() {
        let open = || p(false, &[]);
        let metal = || p(true, &[]);
        // Uncut and fully metal.
        assert_eq!(
            face_fraction([open(), open(), open(), open()], 1.0, 2.0),
            1.0
        );
        assert_eq!(
            face_fraction([metal(), metal(), metal(), metal()], 1.0, 2.0),
            0.0
        );
        // Corner (0,0) cut off by the chord from (0.5, 0) to (0, 0.25·lc):
        // edge 0 metal up to t = 0.5, edge 3 (c at n) metal below t = 0.25.
        let f = face_fraction(
            [p(true, &[0.5]), open(), open(), p(true, &[0.25])],
            1.0,
            2.0,
        );
        let want = 1.0 - 0.5 * 0.5 * 0.5 / 2.0;
        assert!((f - want).abs() < 1e-12, "{f} {want}");
        // Straight cut parallel to b at v = 0.3·lc: metal below.
        let f = face_fraction(
            [metal(), p(true, &[0.3]), open(), p(true, &[0.3])],
            2.0,
            1.0,
        );
        assert!((f - 0.7).abs() < 1e-12, "{f}");
        // Trapezoid: v from 0.2 (u = 0) to 0.6 (u = lb).
        let f = face_fraction(
            [metal(), p(true, &[0.6]), open(), p(true, &[0.2])],
            1.0,
            1.0,
        );
        assert!((f - 0.6).abs() < 1e-12, "{f}");
    }

    #[test]
    fn snapped_surface_through_a_corner() {
        let open = || p(false, &[]);
        let metal = || p(true, &[]);
        // Surface through the corner (lb, 0): edge 1 metal right after it,
        // edge 0 open up to it.
        let e1 = p(false, &[1e-9]).snapped();
        assert!(e1.start_metal && e1.transitions.is_empty());
        // The open corner is kept (edge 0 is open at it): the face is the
        // triangle (0, 0), (lb, 0), (0, lc) with edge 2 metal and edge 3
        // open, not the degenerate segment.
        let f = face_fraction(
            [
                open(),
                e1.clone(),
                metal(),
                p(false, &[1.0 - 1e-9]).snapped(),
            ],
            1.0,
            1.0,
        );
        assert!((f - 0.5).abs() < 1e-12, "{f}");
        // Only one open edge and no area: zero.
        let f = face_fraction([e1, metal(), metal(), open()], 1.0, 1.0);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn open_centroids() {
        let open = || p(false, &[]);
        let metal = || p(true, &[]);
        let c = open_centroid(&[open(), open(), open(), open()], 1.0, 2.0);
        assert!((c[0] - 0.5).abs() < 1e-12 && (c[1] - 0.5).abs() < 1e-12);
        // Metal below v = 0.3·lc: open strip 0.3..1 along c.
        let c = open_centroid(
            &[metal(), p(true, &[0.3]), open(), p(true, &[0.3])],
            2.0,
            1.0,
        );
        assert!(
            (c[0] - 0.5).abs() < 1e-12 && (c[1] - 0.65).abs() < 1e-12,
            "{c:?}"
        );
        // Open corner triangle (0,0), (0.5 lb, 0), (0, 0.5 lc).
        let c = open_centroid(
            &[p(false, &[0.5]), metal(), metal(), p(false, &[0.5])],
            1.0,
            2.0,
        );
        assert!(
            (c[0] - 1.0 / 6.0).abs() < 1e-12 && (c[1] - 1.0 / 6.0).abs() < 1e-12,
            "{c:?}"
        );
    }

    #[test]
    fn faces_the_polygon_misses_are_integrated() {
        let open = || p(false, &[]);
        // A post through the face between two opposite sides, both open at
        // their ends: the walk around the perimeter joins the two
        // transitions of a side and keeps the whole face.
        let band = |u: f64, _v: f64| (0.3..0.7).contains(&u);
        let e = [p(false, &[0.3, 0.7]), open(), p(false, &[0.3, 0.7]), open()];
        assert_eq!(face_fraction(e.clone(), 1.0, 1.0), 1.0);
        assert!(!polygon_holds(&e, 1.0, 1.0, band));
        let (f, c) = integrated_open(band);
        assert!((f - 0.6).abs() < 1e-6, "{f}");
        assert!(
            (c[0] - 0.5).abs() < 1e-6 && (c[1] - 0.5).abs() < 1e-6,
            "{c:?}"
        );
        // A straight cut: the polygon is exact and is kept.
        let half = |u: f64, v: f64| u + v < 0.8;
        let e = [p(true, &[0.8]), open(), open(), p(true, &[0.8])];
        assert!(polygon_holds(&e, 1.0, 1.0, half));
        let (f, c) = integrated_open(half);
        assert!((f - 0.68).abs() < 2e-3, "{f}");
        let want = open_centroid(&e, 1.0, 1.0);
        assert!((c[0] - want[0]).abs() < 5e-3 && (c[1] - want[1]).abs() < 5e-3);
        // The same transitions with a square corner between them: the
        // chord is off by a quarter of the face.
        let corner = |u: f64, v: f64| u < 0.8 && v < 0.8;
        assert!(!polygon_holds(&e, 1.0, 1.0, corner));
        let (f, _) = integrated_open(corner);
        assert!((f - 0.36).abs() < 1e-6, "{f}");
        // A quarter disc of radius 0.8 of the face (a pad of a cell): the
        // chord misses 0.18 of the face.
        let disc = |u: f64, v: f64| u.hypot(v) < 0.8;
        assert!(!polygon_holds(&e, 1.0, 1.0, disc));
        let (f, _) = integrated_open(disc);
        let want = 1.0 - 0.25 * core::f64::consts::PI * 0.64;
        assert!((f - want).abs() < 2e-3, "{f} {want}");
        // … and of ten faces in radius about a far centre: kept.
        let big = |u: f64, v: f64| (u + 6.6).hypot(v + 6.6) < 10.0;
        let t = (100.0f64 - 6.6 * 6.6).sqrt() - 6.6;
        let e = [p(true, &[t]), open(), open(), p(true, &[t])];
        assert!(polygon_holds(&e, 1.0, 1.0, big));
        // A disc inside the face, crossing no side.
        let (f, _) = integrated_open(|u, v| (u - 0.5).hypot(v - 0.5) < 0.4);
        assert!(
            (f - (1.0 - core::f64::consts::PI * 0.16)).abs() < 3e-3,
            "{f}"
        );
        // Nothing and everything.
        assert_eq!(integrated_open(|_, _| false).0, 1.0);
        assert_eq!(integrated_open(|_, _| true).0, 0.0);
    }

    #[test]
    fn bridges_keep_a_diagonal_of_nodes_connected() {
        // Nodes (i, i, 0) in a conductor along the diagonal of the x–y
        // plane; the x edge from each is covered 0.7, the y edge 0.4.
        let nodes: Vec<[usize; 3]> = (0..4).map(|i| [i, i, 0]).collect();
        let covered = |n: [usize; 3]| n[0] == n[1] && n[0] < 4 && n[2] == 0;
        let coverage = |k: (usize, [usize; 3])| match k.0 {
            0 => 0.7,
            1 => 0.4,
            _ => 0.0,
        };
        let run = |joined: bool, conducts: &dyn Fn((usize, [usize; 3])) -> bool| {
            bridges(
                &nodes,
                &[(0, 1), (1, 2), (2, 0)],
                [5, 5, 2],
                &Conductor {
                    covered: &covered,
                    joined: &|_, _| joined,
                    conducts,
                    coverage: &coverage,
                },
            )
        };
        // The path over (i + 1, i): the x edge from (i, i) and the y edge
        // from (i + 1, i). Its edges are covered 0.7 and 0.4; the other
        // path's (y from (i, i), x from (i, i + 1)) 0.4 and 0.7: a tie,
        // the first path.
        let got = run(true, &|_| false);
        let want: Vec<(usize, [usize; 3])> = (0..3)
            .flat_map(|i| [(0, [i, i, 0]), (1, [i + 1, i, 0])])
            .collect();
        assert_eq!(got, want);
        // The conductor does not contain the line between the nodes (two
        // conductors): nothing.
        assert!(run(false, &|_| false).is_empty());
        // One of the two paths conducts already: nothing.
        assert!(run(true, &|k| k.0 == 1 || k.1[0] == k.1[1]).is_empty());
        // One of its edges does: only the other is returned.
        let got = run(true, &|k| k.0 == 0 && k.1[0] == k.1[1]);
        assert_eq!(got, (0..3).map(|i| (1, [i + 1, i, 0])).collect::<Vec<_>>());
    }

    #[test]
    fn annular_face_polygons() {
        let open = || p(false, &[]);
        let metal = || p(true, &[]);
        let (r0, lb, lc) = (2.0, 1.0, 0.3);
        assert_eq!(
            face_fraction_annular([open(), open(), open(), open()], r0, lb, lc),
            1.0
        );
        // Metal for ρ < r0 + 0.3·lb (a cut along α): annulus share.
        let f = face_fraction_annular(
            [p(true, &[0.3]), open(), p(true, &[0.3]), metal()],
            r0,
            lb,
            lc,
        );
        let (r1, r2) = (r0 + 0.3 * lb, r0 + lb);
        let want = (r2 * r2 - r1 * r1) / (r2 * r2 - r0 * r0);
        assert!((f - want).abs() < 1e-12, "{f} {want}");
        // Metal for α < 0.4·lc (a radial cut): the α share, any radius.
        let f = face_fraction_annular(
            [metal(), p(true, &[0.4]), open(), p(true, &[0.4])],
            r0,
            lb,
            lc,
        );
        assert!((f - 0.6).abs() < 1e-12, "{f}");
        // On the axis (r0 = 0) a corner triangle in (ρ, α): ∫∫ρ.
        let f = face_fraction_annular(
            [p(false, &[0.5]), metal(), metal(), p(false, &[0.5])],
            0.0,
            lb,
            lc,
        );
        // Open region u ≤ 0.5·lb·(1 − v/(0.5·lc)), v ≤ 0.5·lc:
        // ∫_0^{0.5 lc} (0.5 lb (1 − 2v/lc))²/2 dv = lb² lc/48.
        let want = (lb * lb * lc / 48.0) / (0.5 * lc * lb * lb);
        assert!((f - want).abs() < 1e-12, "{f} {want}");
    }
}
