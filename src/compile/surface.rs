//! Lossy metal: csx `Metal` properties given a conductivity by
//! [`CompileOptions::with_lossy_metal`](super::CompileOptions::with_lossy_metal)
//! → edges of [`crate::engine::surface`].
//!
//! * **Edges.** The model is discretised a second time without the
//!   primitives of the lossy metals; the edges that are PEC with them and
//!   not without are theirs (so an edge that another, perfect metal covers
//!   as well stays perfect: PEC wins). With several different conductors
//!   an edge takes the one of the highest-priority lossy metal containing
//!   its midpoint.
//! * **Outside.** The capacitance and conductance of an edge are sampled
//!   with the csx priority query at the centres of the four quarter cells
//!   of its dual face, metal samples left out: the part of the dual face
//!   outside the metal, with the dielectric that lies there.
//! * Which of these edges lie on the surface, and the surface width they
//!   carry, is decided by the operator from the engine's PEC edges
//!   ([`SurfaceImpedance`](crate::engine::SurfaceImpedance)): the walls of
//!   the domain and other metal count as metal there.
//! * Cartesian grids only; on a cylindrical grid the metal stays perfect,
//!   with a warning.

use csx::discretize::{DiscretizeOptions, MaterialGrid, discretize};
use csx::properties::TypeMask;
use csx::{PropertyKind, PropertyType, Vec3};

use super::NodeMap;
use crate::Result;
use crate::axis::Axis;
use crate::constants::EPS0;
use crate::engine::surface::{Conductor, SolidEdge};

/// Capacitance and conductance of the edge along `d` at `node` for the
/// quarters of its dual face that are not metal (module docs).
fn outside(model: &csx::ContinuousStructure, d: usize, node: [usize; 3]) -> (f64, f64) {
    let (b, c) = ((d + 1) % 3, (d + 2) % 3);
    let lines = [
        model.grid.lines(0),
        model.grid.lines(1),
        model.grid.lines(2),
    ];
    // (position, half-cell length) on each side along an axis.
    let sides = |a: usize| {
        let (l, i) = (lines[a], node[a]);
        let lo = (i > 0).then(|| (l[i] - 0.25 * (l[i] - l[i - 1]), 0.5 * (l[i] - l[i - 1])));
        let hi =
            (i + 1 < l.len()).then(|| (l[i] + 0.25 * (l[i + 1] - l[i]), 0.5 * (l[i + 1] - l[i])));
        [lo, hi]
    };
    let len = lines[d][node[d] + 1] - lines[d][node[d]];
    let mid = 0.5 * (lines[d][node[d]] + lines[d][node[d] + 1]);
    let (mut cap, mut cond) = (0.0, 0.0);
    for (pb, wb) in sides(b).into_iter().flatten() {
        for (pc, wc) in sides(c).into_iter().flatten() {
            let mut p = Vec3::new(0.0, 0.0, 0.0);
            p[d] = mid;
            p[b] = pb;
            p[c] = pc;
            if let Some(m) = model.medium_at(p, model.grid.coord_system) {
                cap += EPS0 * m.epsilon[d] * wb * wc;
                cond += m.kappa[d] * wb * wc;
            }
        }
    }
    // Lengths are in drawing units: area/length is one length.
    let unit = model.grid.delta_unit / len;
    let ok = |x: f64| if x.is_finite() && x > 0.0 { x } else { 0.0 };
    (ok(cap * unit), ok(cond * unit))
}

/// The PEC edges of `model` without the primitives of its lossy metals
/// `metals` (module docs), or `None` if it has none of them (or the grid
/// is cylindrical). Call it before the model itself is discretised: only
/// the PEC masks of this second discretisation are kept, so the two sets
/// of material arrays never exist side by side.
pub(super) fn bare_pec(
    model: &csx::ContinuousStructure,
    metals: &[(String, Conductor)],
    discretize_opts: &DiscretizeOptions,
) -> Result<Option<[Vec<bool>; 3]>> {
    let listed: Vec<&str> = metals
        .iter()
        .map(|m| m.0.as_str())
        .filter(|n| {
            matches!(
                model.property_by_name(n).map(|p| &p.kind),
                Some(PropertyKind::Metal)
            )
        })
        .collect();
    if listed.is_empty() || model.grid.coord_system != csx::CoordSystem::Cartesian {
        return Ok(None);
    }
    let mut bare = model.clone();
    for name in listed {
        if let Some(p) = bare.property_by_name_mut(name) {
            p.primitives.clear();
        }
    }
    Ok(Some(discretize(&bare, discretize_opts)?.pec))
}

/// The edges of the lossy metals `metals` of `model`, whose discretisation
/// is `mg` and whose PEC edges without those metals are `bare`
/// ([`bare_pec`]; module docs).
pub(super) fn solid_edges(
    model: &csx::ContinuousStructure,
    mg: &MaterialGrid,
    metals: &[(String, Conductor)],
    bare: Option<&[Vec<bool>; 3]>,
    node_map: &NodeMap,
    cylindrical: bool,
    warnings: &mut Vec<String>,
) -> Result<Vec<SolidEdge>> {
    // The last entry of a name counts.
    let mut listed: Vec<&(String, Conductor)> = Vec::new();
    for (i, m) in metals.iter().enumerate() {
        if metals[i + 1..].iter().any(|o| o.0 == m.0) {
            continue;
        }
        match model.property_by_name(&m.0).map(|p| &p.kind) {
            Some(PropertyKind::Metal) => listed.push(m),
            Some(_) => warnings.push(format!(
                "lossy metal {:?}: the property is not a Metal; ignored",
                m.0
            )),
            None => warnings.push(format!(
                "lossy metal {:?}: no property of that name; ignored",
                m.0
            )),
        }
    }
    if listed.is_empty() {
        return Ok(Vec::new());
    }
    if cylindrical {
        warnings.push(format!(
            "{} lossy metal(s) on a cylindrical grid stay perfect conductors (the surface \
             impedance needs a Cartesian grid)",
            listed.len()
        ));
        return Ok(Vec::new());
    }
    let Some(bare) = bare else {
        return Ok(Vec::new());
    };
    let uniform = listed.iter().all(|m| m.1 == listed[0].1);
    let mut out = Vec::new();
    for d in 0..3 {
        for (idx, (&with, &without)) in mg.pec[d].iter().zip(&bare[d]).enumerate() {
            if !with || without {
                continue;
            }
            let node = node_map.csx_node(idx);
            let conductor = if uniform {
                listed[0].1
            } else {
                let mut p = Vec3::new(
                    model.grid.lines(0)[node[0]],
                    model.grid.lines(1)[node[1]],
                    model.grid.lines(2)[node[2]],
                );
                p[d] = 0.5 * (model.grid.lines(d)[node[d]] + model.grid.lines(d)[node[d] + 1]);
                model
                    .properties_at(
                        p,
                        model.grid.coord_system,
                        TypeMask::of(PropertyType::Metal),
                    )
                    .iter()
                    .find_map(|h| listed.iter().find(|m| m.0 == h.property.name))
                    .unwrap_or(&listed[0])
                    .1
            };
            out.push(SolidEdge {
                axis: Axis::ALL[d],
                node,
                conductor,
                outside: Some(outside(model, d, node)),
            });
        }
    }
    Ok(out)
}
