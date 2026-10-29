//! Compile-step rules for cylindrical grids (design §5.15).
//!
//! Most of the model maps as on a Cartesian grid: csx discretises in the
//! grid's `(ρ, α, z)` coordinates (with the Jacobian in its volume
//! weights), and the engine takes the geometry from the grid. Plane-wave
//! excitations are dropped with a warning (openEMS ignores them on
//! cylindrical grids too). Conducting sheets run their slab model with the
//! cylindrical metric ([`sheet_edges`]); conformal PEC handles solid metal
//! and thin sheets (`compile::conformal`, with the `ρ` metric in the cut z
//! faces), oblique thin wires the `(ρ, α, z)` Whitney coupling
//! (`engine::oblique_wire`), SAR dumps local and mass-averaged SAR
//! (`post::sar`), `CellConstantMaterial` the cell midpoints
//! (`compile/cellconst.rs`). Dispersive poles are mapped (midpoint rule in
//! `(ρ, α, z)`, physical edge areas; within 1e-3 of openEMS on a
//! Drude-filled coax). Waveguide mode probes work with a `z` normal (exact
//! sampling, see `compile/mode.rs`).
//!
//! Angles are periodic: sheets ([`wrapped_sheets`]), excitations and lumped
//! elements ([`wrapped_sources`]) are selected on every α line with the
//! primitive tested at `α` and `α ± 2π`, since csx's bounding boxes
//! normalise α to `[−π, π)`. For models read from openEMS files,
//! [`openems_alpha_warnings`] names the boxes whose α range openEMS reads
//! differently (design §5.15, azimuth conventions).

use super::sheet::SheetSpec;
use crate::engine::Grid;
use crate::fdtd::{Cpml, Face};

/// Completes csx's sheet edges on a cylindrical grid.
///
/// csx selects the edges of a planar metal or conducting-sheet primitive
/// within its bounding box, whose α range it normalises to `[−π, π)`: on
/// a grid with α lines in `[0, 2π]`, a sheet over the full circle then only
/// gets the edges with `α ≤ π`, and a sheet at `α > π` none (with a "not
/// on a mesh plane" warning). Here every planar (2-D, flat in one of
/// `ρ`, `α`, `z`) `Metal` or `ConductingSheet` primitive is selected again
/// on its mesh plane: the edges of the plane whose midpoints the primitive
/// contains (tested at `α`, `α ± 2π`) are added to the PEC mask or the
/// sheet ids (PEC and csx's own assignment win), and csx's warnings about
/// such a primitive are removed when it got edges here.
///
/// Returns the sheets the ids of `mg.sheet` refer to (csx's, plus one per
/// property found here only).
pub(super) fn wrapped_sheets(
    model: &csx::ContinuousStructure,
    mg: &mut csx::discretize::MaterialGrid,
    node_map: &super::NodeMap,
    snap_tol: f64,
    warnings: &mut Vec<String>,
) -> Vec<SheetSpec> {
    use csx::{CoordSystem, PropertyKind, Vec3};
    use std::f64::consts::TAU;
    let mut specs = SheetSpec::of(mg);
    let lines = [0, 1, 2].map(|a| model.grid.lines(a).to_vec());
    let n = lines.clone().map(|l| l.len());
    for p in model.properties() {
        let sheet = match &p.kind {
            PropertyKind::Metal => None,
            PropertyKind::ConductingSheet(c) => Some((c.conductivity.value(), c.thickness.value())),
            _ => continue,
        };
        for (pi, prim) in p.primitives.iter().enumerate() {
            if prim.dimension() != 2 {
                continue;
            }
            let b = prim.bounding_box_in(CoordSystem::Cylindrical, model.coord_system);
            if b.is_empty() {
                continue;
            }
            let flat: Vec<usize> = (0..3)
                .filter(|&a| {
                    let k = model.grid.nearest_line(a, b.min[a]).unwrap_or(0);
                    let l = &lines[a];
                    let cell = if l.len() > 1 {
                        (l[k.min(l.len() - 2) + 1] - l[k.min(l.len() - 2)]).abs()
                    } else {
                        0.0
                    };
                    b.max[a] - b.min[a] <= snap_tol * cell + 1e-12 * b.min[a].abs()
                })
                .collect();
            let [nrm] = flat[..] else { continue };
            let Some(k) = super::sheet::plane_line(model, nrm, b.min[nrm], snap_tol) else {
                continue;
            };
            let id = sheet.map(|(s, t)| {
                let found = specs
                    .iter()
                    .position(|x| x.name == p.name && x.conductivity == s && x.thickness == t);
                found.unwrap_or_else(|| {
                    specs.push(SheetSpec {
                        name: p.name.clone(),
                        conductivity: s,
                        thickness: t,
                    });
                    specs.len() - 1
                }) as u32
                    + 1
            });
            let mut hits = 0usize;
            for d in (0..3).filter(|&d| d != nrm) {
                let t = 3 - d - nrm;
                for i in 0..n[d] - 1 {
                    for j in 0..n[t] {
                        let mut node = [0; 3];
                        node[d] = i;
                        node[t] = j;
                        node[nrm] = k;
                        let mut q = [0.0; 3];
                        q[d] = 0.5 * (lines[d][i] + lines[d][i + 1]);
                        q[t] = lines[t][j];
                        q[nrm] = lines[nrm][k];
                        let inside = [0.0, TAU, -TAU].iter().any(|&s| {
                            prim.contains_in(
                                Vec3::new(q[0], q[1] + s, q[2]),
                                CoordSystem::Cylindrical,
                                model.coord_system,
                            )
                        });
                        if !inside {
                            continue;
                        }
                        hits += 1;
                        let idx = node_map.csx_index(node);
                        match id {
                            None => mg.pec[d][idx] = true,
                            Some(id) if mg.sheet[d][idx] == 0 => mg.sheet[d][idx] = id,
                            Some(_) => {}
                        }
                    }
                }
            }
            if hits > 0 {
                let tag = format!("property {:?} primitive #{pi} ", p.name);
                warnings.retain(|w| {
                    !(w.starts_with(&tag)
                        && (w.contains("is not on a mesh plane")
                            || w.contains("covers no grid edge")))
                });
            }
        }
    }
    specs
}

/// Cylindrical rules for conducting-sheet edges (design §5.9, §5.15): the
/// sheet model runs on `ρ = const`, `α = const` and `z = const` sheets with
/// the physical edge lengths and strip widths of the grid
/// ([`ConductingSheet`](crate::engine::ConductingSheet)). Removed here
/// (with a warning): z edges on the axis, whose dual face is the whole disc
/// around the axis (they stay ordinary edges). Sheet edges inside the
/// multigrid zones (§5.15.1) are kept with a warning: the zone projection
/// acts on the face voltage only.
pub(super) fn sheet_edges(
    grid: &Grid,
    edges: &mut Vec<crate::engine::SheetEdge>,
    warnings: &mut Vec<String>,
) {
    use crate::Axis;
    let before = edges.len();
    edges.retain(|e| !(grid.has_axis() && e.axis == Axis::Z && e.node[0] == 0));
    if edges.len() < before {
        warnings.push(format!(
            "{} conducting-sheet edge(s) on the axis are left as ordinary edges",
            before - edges.len()
        ));
    }
    let zoned = edges
        .iter()
        .filter(|e| grid.alpha_factor(e.node[0]) > 1)
        .count();
    if zoned > 0 {
        warnings.push(format!(
            "{zoned} conducting-sheet edge(s) inside the multigrid zones: the zones couple only \
             the face voltage of the sheet (approximate)"
        ));
    }
}

/// Edge and face selection with periodic azimuth (design §5.15, §8.1).
///
/// csx selects the edges of a primitive within its bounding box, whose α
/// range it normalises to `[−π, π)`: on a grid with α lines in `[0, 2π]`
/// a primitive at `α > π` (or wrapping past 2π) only gets part of its
/// edges, or none. This selector takes every α line as a candidate and
/// tests the primitive at `α`, `α ± 2π` (the ρ and z ranges, the snapping
/// tolerance and the edge-midpoint / face-centre rules are csx's).
struct Periodic<'a> {
    model: &'a csx::ContinuousStructure,
    lines: [&'a [f64]; 3],
    snap_tol: f64,
}

impl<'a> Periodic<'a> {
    fn new(model: &'a csx::ContinuousStructure, snap_tol: f64) -> Self {
        Self {
            model,
            lines: [0, 1, 2].map(|a| model.grid.lines(a)),
            snap_tol,
        }
    }

    fn tol(&self, a: usize, i: usize, v: f64) -> f64 {
        let l = self.lines[a];
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
        self.snap_tol * if h.is_finite() { h } else { 1.0 } + 1e-12 * v.abs()
    }

    /// Line of axis `a` at `v` (α: also `v ± 2π`).
    fn snap(&self, a: usize, v: f64) -> Option<usize> {
        let tau = core::f64::consts::TAU;
        let c: &[f64] = if a == 1 { &[v, v + tau, v - tau] } else { &[v] };
        c.iter().find_map(|&v| {
            let i = self.model.grid.nearest_line(a, v)?;
            ((self.lines[a][i] - v).abs() <= self.tol(a, i, v)).then_some(i)
        })
    }

    /// Lines of axis `a` within `[min, max]` (α: all lines).
    fn line_range(&self, a: usize, min: f64, max: f64) -> Option<(usize, usize)> {
        let l = self.lines[a];
        if a == 1 {
            return Some((0, l.len() - 1));
        }
        let lo = self
            .snap(a, min)
            .unwrap_or_else(|| l.partition_point(|&x| x < min));
        let hi = match self.snap(a, max) {
            Some(i) => i,
            None => l.partition_point(|&x| x <= max).checked_sub(1)?,
        };
        (lo < l.len() && lo <= hi).then_some((lo, hi))
    }

    /// Edge start indices of axis `a` overlapping `[min, max]` (α: all).
    fn edge_range(&self, a: usize, min: f64, max: f64) -> Option<(usize, usize)> {
        let l = self.lines[a];
        if l.len() < 2 {
            return None;
        }
        if a == 1 {
            return Some((0, l.len() - 2));
        }
        if max < l[0] || min > l[l.len() - 1] {
            return None;
        }
        let lo = l.partition_point(|&x| x <= min).saturating_sub(1);
        let hi = l
            .partition_point(|&x| x < max)
            .saturating_sub(1)
            .min(l.len() - 2);
        (lo <= hi).then_some((lo, hi))
    }

    fn contains(&self, prim: &csx::Primitive, p: [f64; 3]) -> bool {
        use csx::{CoordSystem, Vec3};
        let tau = core::f64::consts::TAU;
        [0.0, tau, -tau].iter().any(|&s| {
            prim.contains_in(
                Vec3::new(p[0], p[1] + s, p[2]),
                CoordSystem::Cylindrical,
                self.model.coord_system,
            )
        })
    }

    fn mid(&self, d: usize, n: [usize; 3]) -> [f64; 3] {
        let mut p = [0, 1, 2].map(|a| self.lines[a][n[a]]);
        p[d] = 0.5 * (self.lines[d][n[d]] + self.lines[d][n[d] + 1]);
        p
    }

    /// The E edges of `prim` (csx's rules, periodic in α), or `None` for
    /// shapes this selector leaves to csx (curves, wires, non-axis-aligned
    /// sheets).
    fn edges(&self, prim: &csx::Primitive) -> Option<Vec<csx::discretize::Edge>> {
        use csx::CoordSystem;
        use csx::discretize::Edge;
        use csx::primitives::Shape;
        if matches!(prim.shape, Shape::Wire(_) | Shape::Curve(_)) {
            return None;
        }
        let b = prim.bounding_box_in(CoordSystem::Cylindrical, self.model.coord_system);
        if b.is_empty() {
            return Some(Vec::new());
        }
        let map = super::NodeMap::new(self.lines.map(<[f64]>::len));
        let flat: Vec<usize> = (0..3)
            .filter(|&a| {
                let i = self.model.grid.nearest_line(a, b.min[a]).unwrap_or(0);
                b.max[a] - b.min[a] <= self.tol(a, i, b.min[a])
            })
            .collect();
        let dim = prim.dimension();
        let mut out = Vec::new();
        let mut push = |d: usize, n: [usize; 3]| {
            out.push(Edge {
                dir: d,
                index: map.csx_index(n),
            });
        };
        match (dim, flat.len()) {
            (0, _) | (_, 3) => {}
            (_, 2) => {
                let d = (0..3).find(|a| !flat.contains(a))?;
                let (a1, a2) = (flat[0], flat[1]);
                let (i1, i2) = (self.snap(a1, b.min[a1])?, self.snap(a2, b.min[a2])?);
                let (lo, hi) = self.edge_range(d, b.min[d], b.max[d])?;
                for i in lo..=hi {
                    let mut n = [0; 3];
                    n[d] = i;
                    n[a1] = i1;
                    n[a2] = i2;
                    if self.contains(prim, self.mid(d, n)) {
                        push(d, n);
                    }
                }
            }
            (1, _) => return None,
            (2, 1) => {
                let nrm = flat[0];
                let k = self.snap(nrm, b.min[nrm])?;
                for d in (0..3).filter(|&d| d != nrm) {
                    let t = 3 - d - nrm;
                    let (Some((lo, hi)), Some((tlo, thi))) = (
                        self.edge_range(d, b.min[d], b.max[d]),
                        self.line_range(t, b.min[t], b.max[t]),
                    ) else {
                        continue;
                    };
                    for i in lo..=hi {
                        for j in tlo..=thi {
                            let mut n = [0; 3];
                            n[d] = i;
                            n[t] = j;
                            n[nrm] = k;
                            if self.contains(prim, self.mid(d, n)) {
                                push(d, n);
                            }
                        }
                    }
                }
            }
            (2, _) => return None,
            _ => {
                for d in 0..3 {
                    let (t1, t2) = ((d + 1) % 3, (d + 2) % 3);
                    let (Some((lo, hi)), Some((l1, h1)), Some((l2, h2))) = (
                        self.edge_range(d, b.min[d], b.max[d]),
                        self.line_range(t1, b.min[t1], b.max[t1]),
                        self.line_range(t2, b.min[t2], b.max[t2]),
                    ) else {
                        continue;
                    };
                    for i in lo..=hi {
                        for j in l1..=h1 {
                            for k in l2..=h2 {
                                let mut n = [0; 3];
                                n[d] = i;
                                n[t1] = j;
                                n[t2] = k;
                                if self.contains(prim, self.mid(d, n)) {
                                    push(d, n);
                                }
                            }
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        Some(out)
    }

    /// The H faces of `prim` whose centres it contains (periodic in α).
    fn faces(&self, prim: &csx::Primitive) -> Vec<csx::discretize::Edge> {
        use csx::CoordSystem;
        use csx::discretize::Edge;
        let b = prim.bounding_box_in(CoordSystem::Cylindrical, self.model.coord_system);
        let mut out = Vec::new();
        if b.is_empty() {
            return out;
        }
        let map = super::NodeMap::new(self.lines.map(<[f64]>::len));
        for d in 0..3 {
            let (t1, t2) = ((d + 1) % 3, (d + 2) % 3);
            let (Some((lo, hi)), Some((l1, h1)), Some((l2, h2))) = (
                self.line_range(d, b.min[d], b.max[d]),
                self.edge_range(t1, b.min[t1], b.max[t1]),
                self.edge_range(t2, b.min[t2], b.max[t2]),
            ) else {
                continue;
            };
            for i in lo..=hi {
                for j in l1..=h1 {
                    for k in l2..=h2 {
                        let mut n = [0; 3];
                        n[d] = i;
                        n[t1] = j;
                        n[t2] = k;
                        let mut p = self.mid(t1, n);
                        p[t2] = 0.5 * (self.lines[t2][k] + self.lines[t2][k + 1]);
                        if self.contains(prim, p) {
                            out.push(Edge {
                                dir: d,
                                index: map.csx_index(n),
                            });
                        }
                    }
                }
            }
        }
        out
    }
}

/// A [`LumpedEdge`](csx::discretize::LumpedEdge) to fill in (csx's type
/// has no public constructor): one of `existing`, or the edge csx makes of
/// a one-edge model.
fn lumped_template(
    existing: &[csx::discretize::LumpedEdge],
    prop: &csx::Property,
) -> Option<csx::discretize::LumpedEdge> {
    if let Some(l) = existing.first() {
        return Some(l.clone());
    }
    let mut m = csx::ContinuousStructure::new();
    for a in 0..3 {
        m.grid.set_lines(a, [0.0, 1.0]).ok()?;
    }
    let mut p = prop.clone();
    let csx::PropertyKind::LumpedElement(le) = &p.kind else {
        return None;
    };
    let mut hi = [0.0; 3];
    hi[le.direction.index()] = 1.0;
    p.primitives.clear();
    let id = m.add_property(p);
    m.add_primitive(
        id,
        csx::Primitive::new(csx::primitives::BoxShape::new([0.0; 3], hi)),
    )
    .ok()?;
    let mg =
        csx::discretize::discretize(&m, &csx::discretize::DiscretizeOptions::default()).ok()?;
    mg.lumped.into_iter().next()
}

/// Re-selects the edges of excitations and lumped elements on a
/// cylindrical grid with [`Periodic`] (csx's α bounding-box range misses
/// primitives at `α > π` and boxes wrapping past 2π, e.g. a lumped ring
/// over `[0, 2π]` got only `α ≤ π`, a source at `α = 3π/2` none).
/// Amplitudes (`Excite` × weight at the edge midpoint) and the lumped
/// values per edge (`R·p/s`, `L·p/s`, `C·s/p` over the `s` series and `p`
/// parallel positions of each primitive) follow csx; properties with
/// shapes the selector does not handle keep csx's edges. csx's "covers no
/// edge" warnings of re-selected properties are dropped when edges were
/// found.
pub(super) fn wrapped_sources(
    model: &csx::ContinuousStructure,
    mg: &mut csx::discretize::MaterialGrid,
    snap_tol: f64,
    warnings: &mut Vec<String>,
) {
    use csx::PropertyKind;
    let sel = Periodic::new(model, snap_tol);
    let map = super::NodeMap::new(sel.lines.map(<[f64]>::len));
    let mut lumped: Vec<csx::discretize::LumpedEdge> = Vec::with_capacity(mg.lumped.len());
    let mut found: Vec<&str> = Vec::new();
    for p in model.properties() {
        match &p.kind {
            PropertyKind::Excitation(ex) => {
                let h = ex.excitation_type.is_magnetic();
                if !h && !ex.excitation_type.is_electric() {
                    continue;
                }
                let mut amps = Vec::new();
                let mut handled = true;
                for prim in &p.primitives {
                    let edges = if h {
                        sel.faces(prim)
                    } else if let Some(e) = sel.edges(prim) {
                        e
                    } else {
                        handled = false;
                        break;
                    };
                    for e in edges {
                        let n = map.csx_node(e.index);
                        let q = if h {
                            let (t1, t2) = ((e.dir + 1) % 3, (e.dir + 2) % 3);
                            let mut q = sel.mid(t1, n);
                            q[t2] = 0.5 * (sel.lines[t2][n[t2]] + sel.lines[t2][n[t2] + 1]);
                            q
                        } else {
                            sel.mid(e.dir, n)
                        };
                        let c = csx::CoordSystem::Cylindrical.to_cartesian(csx::Vec3(q));
                        let a = ex.amplitude_at(e.dir, c);
                        if a != 0.0 {
                            amps.push((e, a));
                        }
                    }
                }
                if !handled {
                    continue;
                }
                if let Some(f) = mg.excitations.iter_mut().find(|f| f.name == p.name) {
                    let mut old = f.amplitudes.clone();
                    let mut new = amps.clone();
                    old.sort_by_key(|a| a.0);
                    new.sort_by_key(|a| a.0);
                    if old != new {
                        f.amplitudes = amps;
                        if !f.amplitudes.is_empty() {
                            found.push(&p.name);
                        }
                    }
                }
            }
            PropertyKind::LumpedElement(le) => {
                let dir = le.direction.index();
                let mine: Vec<_> = mg.lumped.iter().filter(|l| l.name == p.name).collect();
                let mut new = Vec::new();
                let mut handled = true;
                for prim in &p.primitives {
                    let Some(edges) = sel.edges(prim) else {
                        handled = false;
                        break;
                    };
                    let edges: Vec<_> = edges.into_iter().filter(|e| e.dir == dir).collect();
                    if edges.is_empty() {
                        continue;
                    }
                    let mut along: Vec<usize> =
                        edges.iter().map(|e| map.csx_node(e.index)[dir]).collect();
                    along.sort_unstable();
                    along.dedup();
                    let mut across: Vec<[usize; 2]> = edges
                        .iter()
                        .map(|e| {
                            let n = map.csx_node(e.index);
                            [n[(dir + 1) % 3], n[(dir + 2) % 3]]
                        })
                        .collect();
                    across.sort_unstable();
                    across.dedup();
                    let (ns, np) = (along.len() as f64, across.len() as f64);
                    let Some(t) = lumped_template(&mg.lumped, p) else {
                        handled = false;
                        break;
                    };
                    for e in edges {
                        let mut l = t.clone();
                        l.name.clone_from(&p.name);
                        l.edge = e;
                        l.r = le.r.value() * np / ns;
                        l.c = le.c.value() * ns / np;
                        l.l = le.l.value() * np / ns;
                        l.le_type = le.le_type;
                        l.caps = le.caps;
                        new.push(l);
                    }
                }
                let same = handled && {
                    let mut a: Vec<_> = mine.iter().map(|l| l.edge).collect();
                    let mut b: Vec<_> = new.iter().map(|l| l.edge).collect();
                    a.sort_unstable();
                    b.sort_unstable();
                    a == b
                };
                if handled && !same {
                    if !new.is_empty() {
                        found.push(&p.name);
                    }
                    lumped.extend(new);
                } else {
                    lumped.extend(mine.into_iter().cloned());
                }
            }
            _ => {}
        }
    }
    // Lumped edges of properties not in the model's list (none expected).
    for l in &mg.lumped {
        if model.property_by_name(&l.name).is_none() {
            lumped.push(l.clone());
        }
    }
    mg.lumped = lumped;
    for name in found {
        let tag = format!("property {name:?} primitive #");
        warnings.retain(|w| {
            !(w.starts_with(&tag)
                && (w.contains("covers no") || w.contains("is not on a mesh plane")))
        });
    }
}

/// Whether the α mesh of the cylindrical `grid` is graded (cell widths
/// differing by more than a relative 1e-9).
pub(super) fn graded_alpha(grid: &Grid) -> bool {
    use crate::Axis;
    let a = grid.lines(Axis::Y);
    let d0 = a[1] - a[0];
    grid.is_cylindrical() && a.windows(2).any(|w| ((w[1] - w[0]) - d0).abs() > 1e-9 * d0)
}

/// Warns about boxes whose α range openEMS (CSXCAD) interprets
/// differently from ems (black box, design §5.15): ems maps angles
/// periodically, so a box means the same region whatever multiple of 2π
/// its α corners are given in. openEMS
///
/// * takes the α range of materials, metal, sheets and excitations
///   literally against the grid's α lines (nothing outside the grid's α
///   range: a box over `[−π, 0]` on a `[0, 2π]` grid is empty, one over
///   `[π, 3π]` covers about `[π, 2π]` only; for materials the exact rule
///   is the α index range of `compile::cylavg`, which files follow);
/// * reduces each α bound of a lumped element modulo 2π into `[0, 2π)`
///   separately, so `[0, 2π]` becomes the α = 0 line, `[π, 2π]` and
///   `[3π/2, 5π/2]` become empty and `[−π, π]` the α = π line;
/// * ignores what lies on the α = 0 line of a closed grid only (sources,
///   lumped elements, sheets: it updates the α = 2π copy of that line),
///   e.g. a lumped ring over `[0, 2π]` has no effect at all.
///
/// Only identity-transformed `Box` primitives in cylindrical coordinates
/// are checked. Called for models read from openEMS files; with
/// `material_ranges` the materials follow openEMS's reading
/// (`compile::cylavg`, design §5.16.11) and are not reported.
pub(super) fn openems_alpha_warnings(
    model: &csx::ContinuousStructure,
    grid: &Grid,
    material_ranges: bool,
    warnings: &mut Vec<String>,
) {
    use csx::primitives::Shape;
    use csx::{CoordSystem, PropertyKind};
    let tau = core::f64::consts::TAU;
    let a = model.grid.lines(1);
    let (Some(&g0), Some(&g1)) = (a.first(), a.last()) else {
        return;
    };
    let eps = 1e-9 * tau;
    let closed = grid.is_closed();
    for p in model.properties() {
        let what = match &p.kind {
            PropertyKind::LumpedElement(_) => "lumped",
            PropertyKind::Material(_)
            | PropertyKind::Metal
            | PropertyKind::ConductingSheet(_)
            | PropertyKind::Excitation(_) => "other",
            _ => continue,
        };
        for (pi, prim) in p.primitives.iter().enumerate() {
            let Shape::Box(b) = &prim.shape else {
                continue;
            };
            if prim.shape_cs(model.coord_system) != CoordSystem::Cylindrical
                || !prim.transform.is_empty()
            {
                continue;
            }
            let (s, e) = (b.start.value()[1], b.stop.value()[1]);
            let (lo, hi) = (s.min(e), s.max(e));
            let tag = format!("property {:?} primitive #{pi}", p.name);
            if what == "lumped" {
                let r = |x: f64| x.rem_euclid(tau);
                let (rl, rh) = (r(lo), r(hi));
                if (rl - lo).abs() > eps || (rh - hi).abs() > eps {
                    let fate = if rh < rl - eps {
                        "an empty range (the element is dropped)".to_string()
                    } else if closed && rh < g0 + eps {
                        "the alpha = 0 line, where it has no effect".to_string()
                    } else {
                        format!("[{rl}, {rh}]")
                    };
                    warnings.push(format!(
                        "{tag}: lumped element over alpha [{lo}, {hi}]; openEMS reduces each \
                         alpha bound modulo 2*pi, which gives {fate}; ems keeps the element as \
                         drawn (periodic azimuth)"
                    ));
                    continue;
                }
            } else if (lo < g0 - eps || hi > g1 + eps)
                && !(material_ranges && matches!(p.kind, PropertyKind::Material(_)))
            {
                warnings.push(format!(
                    "{tag}: alpha range [{lo}, {hi}] leaves the grid's [{g0}, {g1}]; ems wraps \
                     it periodically, openEMS takes it literally (nothing outside the grid's \
                     alpha lines)"
                ));
                continue;
            }
            if closed
                && hi < g0 + eps
                && lo > g0 - eps
                && !matches!(p.kind, PropertyKind::Material(_))
            {
                warnings.push(format!(
                    "{tag}: lies on the alpha = {g0} line of a closed grid only; openEMS \
                         ignores sources, lumped elements and sheets there (it updates the \
                         alpha = {g1} copy of the line), ems applies them"
                ));
            }
        }
    }
}

/// Drops the thin conformal sheets (design §5.13.1) when any of their cut
/// edges lies inside the multigrid zones (§5.15.1), with a warning: the
/// zone projection does not know the split edges. Solid conformal metal
/// is kept.
pub(super) fn conformal_sheets_in_zones(
    grid: &Grid,
    pec: crate::engine::ConformalPec,
    warnings: &mut Vec<String>,
) -> crate::engine::ConformalPec {
    let zoned = grid.has_multigrid()
        && pec.sheet_edges().iter().any(|e| {
            let i = e.node[0];
            grid.alpha_factor(i) > 1 || (i > 0 && grid.alpha_factor(i - 1) > 1)
        });
    if !zoned {
        return pec;
    }
    warnings.push(format!(
        "conformal PEC sheets: {} cut edge(s) inside the multigrid zones; the sheets are not \
         modelled (keep sheets outside the zones)",
        pec.sheet_edges().len()
    ));
    let mut out = crate::engine::ConformalPec::new();
    for e in pec.edges() {
        out.push_edge(*e);
    }
    for f in pec.faces() {
        out.push_face(*f);
    }
    out
}

/// Drops (with a warning) the absorbing boundaries that a cylindrical
/// grid ignores: on the α faces of a closed grid (periodic) and on the
/// `ρ_min` face of a grid with an axis.
pub(super) fn drop_ignored_boundaries(
    grid: &Grid,
    mur: &mut Vec<(Face, Option<f64>)>,
    pml: &mut Vec<(Face, Cpml)>,
    warnings: &mut Vec<String>,
) {
    let ignored = |f: Face| {
        (grid.is_closed() && matches!(f, Face::YMin | Face::YMax))
            || (grid.has_axis() && f == Face::XMin)
    };
    let mut warn = |f: Face| {
        warnings.push(format!(
            "absorbing boundary on {f} ignored: on this cylindrical grid it is {}",
            if f == Face::XMin {
                "the axis"
            } else {
                "periodic (the azimuth spans 2π)"
            }
        ));
    };
    mur.retain(|m| {
        let keep = !ignored(m.0);
        if !keep {
            warn(m.0);
        }
        keep
    });
    pml.retain(|p| {
        let keep = !ignored(p.0);
        if !keep {
            warn(p.0);
        }
        keep
    });
}

/// Applies the [`Fdtd::multigrid`](crate::Fdtd::multigrid) radii (drawing
/// units) to `grid`: each radius snaps to the nearest `ρ` line of the csx
/// grid (with a warning if it is not on one). Cartesian grids ignore them
/// with a warning.
///
/// # Errors
/// [`Error::Invalid`](crate::Error::Invalid) from
/// [`Grid::with_multigrid`] (zone boundary at the first or last line, two
/// radii on the same line, α cells not a multiple of `2^L`, …).
pub(crate) fn multigrid(
    grid: Grid,
    csx_grid: &csx::RectilinearGrid,
    radii: &[f64],
    warnings: &mut Vec<String>,
) -> crate::Result<Grid> {
    if radii.is_empty() {
        return Ok(grid);
    }
    if !grid.is_cylindrical() {
        warnings.push(format!(
            "multigrid radii {radii:?} ignored: multigrid needs a cylindrical grid"
        ));
        return Ok(grid);
    }
    let rho = csx_grid.lines(0);
    let rings: Vec<usize> = radii
        .iter()
        .map(|&r| {
            let i = (0..rho.len())
                .min_by(|&a, &b| (rho[a] - r).abs().total_cmp(&(rho[b] - r).abs()))
                .unwrap_or(0);
            if (rho[i] - r).abs() > 1e-9 * r.abs().max(rho[i].abs()) {
                warnings.push(format!(
                    "multigrid radius {r} snapped to the rho line {} (index {i})",
                    rho[i]
                ));
            }
            i
        })
        .collect();
    grid.with_multigrid(&rings)
}

/// Warns about excitation and lumped-element edges that the multigrid
/// zones of `grid` remove (they have no effect, or make the element fail)
/// or couple (α edges of a coarse arc, slave z edges on a zone boundary:
/// the element then acts through the zone projection).
pub(super) fn multigrid_edge_warnings(
    grid: &Grid,
    node_map: &super::NodeMap,
    excitations: &[csx::discretize::ExcitationField],
    lumped: &[csx::discretize::LumpedEdge],
    warnings: &mut Vec<String>,
) {
    use crate::Axis;
    if !grid.has_multigrid() {
        return;
    }
    let coupled = |axis: Axis, [i, j, _]: [usize; 3]| match axis {
        Axis::Y => !(i == 0 && grid.has_axis()) && grid.alpha_factor(i) > 1,
        Axis::Z => {
            let (inn, out) = grid.ring_factors(i);
            !(i == 0 && grid.has_axis()) && inn > out && j % inn != 0
        }
        Axis::X => false,
    };
    for ex in excitations {
        let magnetic = ex.excitation_type.is_magnetic();
        let removed = ex
            .amplitudes
            .iter()
            .filter(|(e, _)| {
                let (axis, node) = node_map.edge(*e);
                if magnetic {
                    grid.mg_removed_h(axis, node)
                } else {
                    grid.mg_removed_e(axis, node)
                }
            })
            .count();
        if removed > 0 {
            warnings.push(format!(
                "excitation {:?}: {removed} of its {} edges lie off the alpha lines kept by the \
                 multigrid zones and have no effect",
                ex.name,
                ex.amplitudes.len()
            ));
        }
    }
    let mut names: Vec<(&str, usize, usize)> = Vec::new();
    for le in lumped {
        let (axis, node) = node_map.edge(le.edge);
        let (r, c) = (grid.mg_removed_e(axis, node), coupled(axis, node));
        if r || c {
            match names.iter_mut().find(|n| n.0 == le.name) {
                Some(n) => {
                    n.1 += usize::from(r);
                    n.2 += usize::from(c);
                }
                None => names.push((&le.name, usize::from(r), usize::from(c))),
            }
        }
    }
    for (name, r, c) in names {
        warnings.push(format!(
            "lumped element {name:?}: {r} edge(s) removed and {c} coupled by the multigrid \
             zones (place lumped elements on the rho / z edges of kept alpha lines, or outside \
             the zones)"
        ));
    }
}
