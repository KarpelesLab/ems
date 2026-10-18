//! csx `ConductingSheet` → engine sheet edges ([`crate::engine::sheet`]).
//!
//! `csx::discretize` marks the edges of every conducting sheet primitive in
//! [`MaterialGrid::sheet`]; this module turns them into [`SheetEdge`]s:
//!
//! * **Normal.** The sheet normal of an edge comes from the planar
//!   (two-dimensional, axis-aligned) primitives of its property: the
//!   primitive whose mesh plane contains the edge. Sheet edges that belong
//!   to no planar primitive (lines, curves, volumes drawn as conducting
//!   sheets) have no defined normal; they are made PEC, with a warning
//!   that counts them.
//! * **Shares.** The capacitance and conductance shares of the two faces
//!   are sampled with the csx priority query
//!   ([`medium_at`](csx::ContinuousStructure::medium_at)) at the centres of
//!   the four quarter cells of the edge's dual face (two on each side of
//!   the sheet), weighted by their areas; metal samples are left out. A
//!   strip on a substrate thus puts the substrate's capacitance on the
//!   substrate side.
//! * **Priorities.** A sheet edge that is also PEC (solid or sheet metal)
//!   stays PEC. Overlapping sheets: csx keeps the last one.
//! * A sheet with conductivity or thickness `≤ 0` has no effect (the
//!   edges stay ordinary), with a warning.

use std::collections::HashMap;

use csx::discretize::MaterialGrid;
use csx::{CoordSystem, PropertyKind, Vec3};

use super::NodeMap;
use crate::axis::Axis;
use crate::engine::SheetEdge;
use crate::fdtd::{Cpml, Face};

/// The sheet edges of a model and the sheet edges made PEC instead.
#[derive(Debug, Default)]
pub(super) struct SheetMap {
    /// Edges for the engine operator (engine nodes).
    pub edges: Vec<SheetEdge>,
    /// `(dir, csx index)` of sheet edges without a normal (PEC).
    pub pec: Vec<(usize, usize)>,
}

/// Local cell size around line `i` of `lines`.
fn local_cell(lines: &[f64], i: usize) -> f64 {
    let lo = if i > 0 { lines[i] - lines[i - 1] } else { 0.0 };
    let hi = if i + 1 < lines.len() {
        lines[i + 1] - lines[i]
    } else {
        0.0
    };
    lo.max(hi)
}

/// `(normal, plane index)` of every planar primitive of the conducting
/// sheets, by property name.
fn sheet_planes(
    model: &csx::ContinuousStructure,
    snap_tol: f64,
) -> HashMap<&str, Vec<(usize, usize)>> {
    let mut out: HashMap<&str, Vec<(usize, usize)>> = HashMap::new();
    for p in model.properties() {
        if !matches!(p.kind, PropertyKind::ConductingSheet(_)) {
            continue;
        }
        let planes = out.entry(p.name.as_str()).or_default();
        for prim in &p.primitives {
            if prim.dimension() != 2 {
                continue;
            }
            let b = prim.bounding_box_in(CoordSystem::Cartesian, model.coord_system);
            if b.is_empty() {
                continue;
            }
            let tol = |a: usize, v: f64| {
                let l = model.grid.lines(a);
                model
                    .grid
                    .nearest_line(a, v)
                    .map_or(0.0, |i| snap_tol * local_cell(l, i))
                    + 1e-12 * v.abs()
            };
            let flat: Vec<usize> = (0..3)
                .filter(|&a| b.max[a] - b.min[a] <= tol(a, b.min[a]))
                .collect();
            if let [nrm] = flat[..]
                && let Some(k) = model.grid.nearest_line(nrm, b.min[nrm])
                && (model.grid.lines(nrm)[k] - b.min[nrm]).abs() <= tol(nrm, b.min[nrm])
            {
                planes.push((nrm, k));
            }
        }
    }
    out
}

/// Capacitance and conductance shares of the `+` side of the sheet edge
/// along `d` at `node` with normal `nrm` (module docs).
fn shares(
    model: &csx::ContinuousStructure,
    d: usize,
    node: [usize; 3],
    nrm: usize,
) -> Option<(f64, f64)> {
    let t = 3 - d - nrm;
    let (ld, lt, ln) = (
        model.grid.lines(d),
        model.grid.lines(t),
        model.grid.lines(nrm),
    );
    // (position, half-cell length) on each side along an axis.
    let sides = |l: &[f64], i: usize| {
        let lo = (i > 0).then(|| (l[i] - 0.25 * (l[i] - l[i - 1]), 0.5 * (l[i] - l[i - 1])));
        let hi =
            (i + 1 < l.len()).then(|| (l[i] + 0.25 * (l[i + 1] - l[i]), 0.5 * (l[i + 1] - l[i])));
        [lo, hi]
    };
    let mid = 0.5 * (ld[node[d]] + ld[node[d] + 1]);
    let [dn, up] = sides(ln, node[nrm]);
    let mut acc = [[0.0; 2]; 2]; // [side][C, G]
    for (side, s) in [dn, up].into_iter().enumerate() {
        let Some((pn, wn)) = s else { continue };
        for (pt, wt) in sides(lt, node[t]).into_iter().flatten() {
            let mut p = Vec3::new(0.0, 0.0, 0.0);
            p[d] = mid;
            p[t] = pt;
            p[nrm] = pn;
            if let Some(m) = model.medium_at(p, CoordSystem::Cartesian) {
                acc[side][0] += m.epsilon[d] * wn * wt;
                acc[side][1] += m.kappa[d] * wn * wt;
            }
        }
    }
    let c = acc[0][0] + acc[1][0];
    if c.is_nan() || c <= 0.0 {
        return None;
    }
    let sc = acc[1][0] / c;
    let g = acc[0][1] + acc[1][1];
    let sg = if g > 0.0 { acc[1][1] / g } else { sc };
    Some((sc, sg))
}

/// Maps the conducting-sheet edges of `mg` (see the module docs).
pub(super) fn sheet_edges(
    model: &csx::ContinuousStructure,
    mg: &MaterialGrid,
    node_map: &NodeMap,
    snap_tol: f64,
    warnings: &mut Vec<String>,
) -> SheetMap {
    let mut out = SheetMap::default();
    if mg.sheets.is_empty() {
        return out;
    }
    let planes = sheet_planes(model, snap_tol);
    // Per sheet index: edges without effect, edges made PEC.
    let mut idle = vec![0usize; mg.sheets.len()];
    let mut pec = vec![0usize; mg.sheets.len()];
    for d in 0..3 {
        for (idx, &sid) in mg.sheet[d].iter().enumerate() {
            if sid == 0 || mg.pec[d][idx] {
                continue;
            }
            let s = sid as usize - 1;
            let info = &mg.sheets[s];
            let (sigma, t) = (info.conductivity, info.thickness);
            if !(sigma.is_finite() && sigma > 0.0 && t.is_finite() && t > 0.0) {
                idle[s] += 1;
                continue;
            }
            let node = node_map.csx_node(idx);
            let normal = planes.get(info.name.as_str()).and_then(|p| {
                p.iter()
                    .find(|&&(nrm, k)| nrm != d && node[nrm] == k)
                    .map(|&(nrm, _)| nrm)
            });
            let Some(nrm) = normal else {
                pec[s] += 1;
                out.pec.push((d, idx));
                continue;
            };
            out.edges.push(SheetEdge {
                axis: Axis::ALL[d],
                node,
                normal: Axis::ALL[nrm],
                conductivity: sigma,
                thickness: t,
                upper_share: shares(model, d, node, nrm),
            });
        }
    }
    for (s, info) in mg.sheets.iter().enumerate() {
        if idle[s] > 0 {
            warnings.push(format!(
                "ConductingSheet {:?}: conductivity {} S/m and thickness {} m must be > 0; \
                 the sheet has no effect",
                info.name, info.conductivity, info.thickness
            ));
        }
        if pec[s] > 0 {
            warnings.push(format!(
                "ConductingSheet {:?}: {} edge(s) not on a planar sheet (line, curve or volume \
                 primitive) approximated as PEC",
                info.name, pec[s]
            ));
        }
    }
    out
}

/// Warns about sheet edges inside a CPML (they do not get the CPML
/// correction).
pub(super) fn pml_warning(
    edges: &[SheetEdge],
    nodes: [usize; 3],
    pml: &[(Face, Cpml)],
    warnings: &mut Vec<String>,
) {
    let inside = edges
        .iter()
        .filter(|e| {
            pml.iter().any(|(face, p)| {
                let a = face.axis().index();
                let along = usize::from(e.axis.index() == a);
                if face.is_max() {
                    e.node[a] + along > nodes[a] - 1 - p.cells.min(nodes[a] - 1)
                } else {
                    e.node[a] < p.cells
                }
            })
        })
        .count();
    if inside > 0 {
        warnings.push(format!(
            "{inside} conducting-sheet edge(s) inside the PML do not get the PML correction; \
             keep sheets out of the absorbing layers"
        ));
    }
}
