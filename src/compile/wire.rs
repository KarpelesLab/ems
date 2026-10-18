//! csx `Wire` primitives on metal → thin wires ([`crate::engine::wire`]).
//!
//! `csx::discretize` makes the edges of a metal wire's centre line PEC (a
//! staircase through the nearest mesh nodes) and, for a wire radius of at
//! least half a cell, also the edges inside the wire volume. A wire
//! thinner than the staircase itself (radius below the lattice radius
//! `r_eq ≈ 0.2·Δ` of its position, [`ThinWire::staircase_radius`]) gets
//! the thin-wire correction of [`ThinWire::apply`] on top:
//!
//! * Every polyline segment of the wire that runs along a mesh axis, on
//!   mesh lines in the two transverse directions (within the compile snap
//!   tolerance), becomes a run of wire edges between the mesh lines
//!   nearest to its end points. Oblique segments, and segments off the
//!   mesh lines, keep the plain staircase, with a warning.
//! * Lumped-element edges along the same line that touch a wire run (the
//!   feed gap of a dipole, the base of a monopole) join the run, so the
//!   current through the element sees the wire's inductance and the nodes
//!   next to the gap are interior wire nodes.
//! * Contiguous edges of the same line and radius form one [`ThinWire`];
//!   the correction is applied to the edge materials.
//! * Wires with a radius at or above the staircase radius keep the plain
//!   staircase (or the volume rasterisation of csx), with a warning when
//!   the radius is below half a cell (where the staircase is all csx
//!   produces). Wires in a cylindrical structure are not handled (warning).
//! * Two thin wires sharing a node (a junction or crossing) are both
//!   corrected; the model assumes isolated wires, so a warning is given.
//! * Curves (zero radius) are plain staircases.

use std::collections::{BTreeMap, BTreeSet};

use csx::discretize::LumpedEdge;
use csx::primitives::Shape;
use csx::{CoordSystem, PropertyKind, Vec3};

use crate::axis::Axis;
use crate::engine::wire::ThinWire;
use crate::engine::{EdgeMaterials, Grid};

/// `(axis, transverse node indices, radius bits)`: one mesh line carrying
/// wire edges of one radius.
type LineKey = (usize, [usize; 2], u64);

/// Index of the mesh line of `lines` within `tol` (relative to the local
/// cell) of `v`.
fn snap(lines: &[f64], v: f64, tol: f64) -> Option<usize> {
    let i = nearest(lines, v)?;
    let cell = [
        i.checked_sub(1).map(|j| lines[i] - lines[j]),
        lines.get(i + 1).map(|x| x - lines[i]),
    ]
    .into_iter()
    .flatten()
    .fold(0.0, f64::max);
    ((lines[i] - v).abs() <= tol * cell + 1e-12 * v.abs()).then_some(i)
}

/// Index of the mesh line nearest to `v`.
fn nearest(lines: &[f64], v: f64) -> Option<usize> {
    (0..lines.len()).min_by(|&a, &b| (lines[a] - v).abs().total_cmp(&(lines[b] - v).abs()))
}

/// Finds the thin wires of `model`, applies them to `materials` and
/// returns them (see the module docs).
pub(super) fn thin_wires(
    model: &csx::ContinuousStructure,
    grid: &Grid,
    lumped: &[LumpedEdge],
    snap_tol: f64,
    materials: &mut EdgeMaterials,
    warnings: &mut Vec<String>,
) -> Vec<ThinWire> {
    let unit = model.grid.delta_unit;
    let cs = model.coord_system;
    let n = grid.nodes();
    // Covered edge positions along each line, per key.
    let mut lines: BTreeMap<LineKey, BTreeSet<usize>> = BTreeMap::new();
    for p in model.properties() {
        if !matches!(p.kind, PropertyKind::Metal) {
            continue;
        }
        for (pi, prim) in p.primitives.iter().enumerate() {
            let Shape::Wire(w) = &prim.shape else {
                continue;
            };
            let radius = w.radius.value() * unit;
            if radius.is_nan() || radius <= 0.0 {
                continue;
            }
            let label = || format!("wire {:?} primitive #{pi}", p.name);
            if cs != CoordSystem::Cartesian || model.grid.coord_system != CoordSystem::Cartesian {
                warnings.push(format!(
                    "{}: thin-wire model not available in cylindrical structures; staircase",
                    label()
                ));
                continue;
            }
            let shape = prim.cartesian_shape(cs);
            let Shape::Wire(w) = &*shape else {
                continue;
            };
            let pts: Vec<Vec3> = w
                .points
                .iter()
                .map(|q| {
                    let c = q.value();
                    if prim.transform.is_empty() {
                        c
                    } else {
                        prim.transform.apply(c)
                    }
                })
                .collect();
            let mut thick = false;
            for (si, seg) in pts.windows(2).enumerate() {
                let (a, b) = (seg[0], seg[1]);
                let scale = (0..3)
                    .map(|d| a[d].abs().max(b[d].abs()))
                    .fold(0.0, f64::max);
                let moving: Vec<usize> = (0..3)
                    .filter(|&d| (a[d] - b[d]).abs() > 1e-9 * scale.max(f64::MIN_POSITIVE))
                    .collect();
                let [d] = moving[..] else {
                    if !moving.is_empty() {
                        warnings.push(format!(
                            "{} segment {si}: not along a mesh axis; staircase without \
                             thin-wire correction",
                            label()
                        ));
                    }
                    continue;
                };
                let (t1, t2) = Axis::ALL[d].others();
                let (t1, t2) = (t1.index(), t2.index());
                let (Some(i1), Some(i2)) = (
                    snap(model.grid.lines(t1), a[t1], snap_tol),
                    snap(model.grid.lines(t2), a[t2], snap_tol),
                ) else {
                    warnings.push(format!(
                        "{} segment {si}: not on mesh lines; staircase without thin-wire \
                         correction",
                        label()
                    ));
                    continue;
                };
                let (Some(k0), Some(k1)) = (
                    nearest(model.grid.lines(d), a[d]),
                    nearest(model.grid.lines(d), b[d]),
                ) else {
                    continue;
                };
                let (k0, k1) = (k0.min(k1), k0.max(k1));
                if k0 == k1 {
                    continue;
                }
                let mut start = [0; 3];
                start[t1] = i1;
                start[t2] = i2;
                start[d] = k0;
                let tw = ThinWire::new(Axis::ALL[d], start, k1 - k0, radius);
                match tw.factor(grid) {
                    Ok(g) if g > 1.0 => {
                        lines
                            .entry((d, [i1, i2], radius.to_bits()))
                            .or_default()
                            .extend(k0..k1);
                    }
                    Ok(_) => thick = true,
                    Err(e) => warnings.push(format!("{} segment {si}: {e}", label())),
                }
            }
            if thick {
                let h = (0..3)
                    .map(|d| grid.min_delta(Axis::ALL[d]))
                    .fold(f64::INFINITY, f64::min);
                if radius < 0.5 * h {
                    warnings.push(format!(
                        "{}: radius {radius:e} m is not below the staircase radius (≈ 0.2 \
                         cells); modelled as a plain staircase",
                        label()
                    ));
                }
            }
        }
    }
    if lines.is_empty() {
        return Vec::new();
    }
    // Lumped edges touching a wire run join it.
    for le in lumped {
        let d = le.edge.dir;
        let idx = le.edge.index;
        let node = [idx % n[0], (idx / n[0]) % n[1], idx / (n[0] * n[1])];
        let (t1, t2) = Axis::ALL[d].others();
        let tr = [node[t1.index()], node[t2.index()]];
        let k = node[d];
        for ((ld, lt, _), set) in lines.iter_mut() {
            if *ld == d
                && *lt == tr
                && !set.contains(&k)
                && ((k > 0 && set.contains(&(k - 1))) || set.contains(&(k + 1)))
            {
                set.insert(k);
            }
        }
    }
    // Runs.
    let mut wires = Vec::new();
    for (&(d, tr, bits), set) in &lines {
        let (t1, t2) = Axis::ALL[d].others();
        let mut ks = set.iter().copied().peekable();
        while let Some(k0) = ks.next() {
            let mut k1 = k0 + 1;
            while ks.peek() == Some(&k1) {
                ks.next();
                k1 += 1;
            }
            let mut start = [0; 3];
            start[t1.index()] = tr[0];
            start[t2.index()] = tr[1];
            start[d] = k0;
            wires.push(ThinWire::new(
                Axis::ALL[d],
                start,
                k1 - k0,
                f64::from_bits(bits),
            ));
        }
    }
    // Junctions: wires sharing a node.
    let nodes_of = |w: &ThinWire| {
        let (a, start) = (w.axis().index(), w.start());
        (0..=w.segments()).map(move |s| {
            let mut p = start;
            p[a] += s;
            p
        })
    };
    let mut seen: BTreeMap<[usize; 3], usize> = BTreeMap::new();
    let mut junction = false;
    for (wi, w) in wires.iter().enumerate() {
        for p in nodes_of(w) {
            if let Some(&o) = seen.get(&p) {
                junction |= o != wi;
            } else {
                seen.insert(p, wi);
            }
        }
    }
    if junction {
        warnings.push(
            "thin wires meet at a node (junction or radius change); the thin-wire model \
             assumes isolated straight wires there"
                .into(),
        );
    }
    wires.retain(|w| match w.apply(grid, materials) {
        Ok(_) => true,
        Err(e) => {
            warnings.push(format!("thin wire at {:?}: {e}", w.start()));
            false
        }
    });
    wires
}
