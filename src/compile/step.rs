//! The time step of a compiled model from its cells
//! ([`StepLimit::Cells`], design §5.4.1): the bound of
//! [`crate::engine::step_bound`] on the model's materials, sheets and
//! lossy metal, with the cut cells of conformal PEC.

use std::collections::{HashMap, HashSet};

use super::Compiled;
use crate::axis::Axis;
use crate::constants::EPS0;
use crate::engine::dispersive::Scheme as DispersiveScheme;
use crate::engine::sheet::CUT_FACE_LOADING;
use crate::engine::step_bound::{Cells, LiveEdge, LiveEdges, walls};
use crate::engine::surface::SolidEdge;
use crate::engine::{EdgeMaterials, FaceCondition, Grid, Pole, SheetEdge, TimeStep, courant_limit};
use crate::fdtd::{Face, Fdtd, StepLimit};

/// What the time step of a compiled model was taken from
/// ([`Compiled::step`]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct StepReport {
    /// The time step \[s\].
    pub dt: f64,
    /// The Courant step: [`Fdtd::timestep_factor`] times
    /// [`courant_limit`] \[s\].
    pub courant: f64,
    /// The factor times the bound of the cells \[s\]
    /// ([`cell_limit`](crate::engine::cell_limit) with the sheets and the
    /// lossy metal of the model); `None` where it was not evaluated.
    pub cells: Option<f64>,
    /// The cell that bound is the limit of (lowest node).
    pub cell: Option<[usize; 3]>,
    /// The longest step at which every cell with a cut face of conformal
    /// PEC keeps its faces and stays within the factor of its own limit
    /// \[s\]; `None` without such cells, or where it was not evaluated.
    pub cut_cells: Option<f64>,
    /// What decided.
    pub limit: StepChoice,
}

/// What decided the time step ([`StepReport::limit`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepChoice {
    /// The Courant step, as asked ([`StepLimit::Courant`], a fixed step,
    /// an openEMS criterion).
    Courant,
    /// The bound of the cells.
    Cells,
    /// The cells that conformal PEC cuts.
    CutCells,
    /// The Courant step, because the cells give no longer one.
    CourantLonger,
    /// The Courant step, because the model has a part the bound of the
    /// cells does not cover.
    CourantKept(StepObstacle),
}

/// The part of a model that keeps it on the Courant step
/// ([`StepChoice::CourantKept`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepObstacle {
    /// A cylindrical grid (its limit is computed from its operator).
    Cylindrical,
    /// A periodic or Bloch boundary.
    Periodic,
    /// A Mur boundary or an absorbing sheet.
    Mur,
    /// A plane wave (total-field / scattered-field box).
    PlaneWave,
    /// An oblique thin wire.
    ObliqueWire,
    /// Lorentz or Drude poles on the leapfrog scheme.
    LeapfrogPoles,
    /// Conformal PEC whose cut faces the cells do not decide: thin sheets
    /// off the mesh planes, a cut face inside an absorbing layer, or
    /// [`ConformalOptions::keep_stable`](crate::engine::ConformalOptions::keep_stable)
    /// off.
    Conformal,
}

impl StepReport {
    /// The report of a step that was not taken from the cells.
    pub(super) fn courant(dt: f64, courant: f64) -> Self {
        Self {
            dt,
            courant,
            cells: None,
            cell: None,
            cut_cells: None,
            limit: StepChoice::Courant,
        }
    }
}

/// The stability factor squared a cut cell beside lossy metal may have:
/// the floor of the capacitance of the metal's edges there holds for
/// cells within `1 − θ` of their limit, `θ = 1/16`
/// ([`crate::engine::surface`], `FLOOR_MARGIN`).
const SOLID_CUT_LIMIT: f64 = 1.0 - 1.0 / 16.0;

/// Half the cell on each side of node `k` along `ax`: `(below, above)`.
fn halves(grid: &Grid, ax: Axis, k: usize) -> (f64, f64) {
    let n = grid.nodes()[ax.index()];
    let up = if k + 1 < n {
        0.5 * grid.delta(ax, k)
    } else {
        0.0
    };
    let dn = if k > 0 {
        0.5 * grid.delta(ax, k - 1)
    } else {
        0.0
    };
    (dn, up)
}

/// The sheet edge `e` as the cells on its two sides have it.
fn sheet_live(grid: &Grid, e: &SheetEdge) -> LiveEdge {
    let (dn, up) = halves(grid, e.normal, e.node[e.normal.index()]);
    let f_up = up / (up + dn);
    let sc = e.upper_share.map_or(f_up, |s| s.0);
    let ratio = |share: f64, f: f64| {
        if f > 0.0 && share.is_finite() {
            (share / f).max(0.0)
        } else {
            1.0
        }
    };
    LiveEdge::Sheet {
        normal: e.normal,
        above: ratio(sc, f_up),
        below: ratio(1.0 - sc, 1.0 - f_up),
    }
}

/// The least relative permittivity the edge `e` of lossy metal has in a
/// cell outside the metal: its capacitance outside the metal
/// ([`SolidEdge::outside`]) over the quarters of its dual face that lie
/// in cells with an edge that is not PEC, and at least vacuum's (the
/// operator gives every quarter beside a face with field that much, and
/// those quarters are among these).
fn solid_eps(grid: &Grid, mats: &EdgeMaterials, e: &SolidEdge) -> f64 {
    let Some((cap, _)) = e.outside else {
        return 1.0;
    };
    let d = e.axis.index();
    let (b, c) = ((d + 1) % 3, (d + 2) % 3);
    let n = grid.nodes();
    let open = |q: [usize; 3]| {
        (0..3).any(|a| {
            let (x, y) = ((a + 1) % 3, (a + 2) % 3);
            let mask = mats.pec_mask(Axis::ALL[a]);
            [(0, 0), (1, 0), (0, 1), (1, 1)].iter().any(|&(ox, oy)| {
                let mut p = q;
                p[x] += ox;
                p[y] += oy;
                !mask[grid.index(p)]
            })
        })
    };
    let mut area = 0.0;
    for ob in 0..2 {
        for oc in 0..2 {
            let mut q = e.node;
            if (ob == 0 && q[b] == 0) || (oc == 0 && q[c] == 0) {
                continue;
            }
            q[b] -= 1 - ob;
            q[c] -= 1 - oc;
            if q[b] + 1 >= n[b] || q[c] + 1 >= n[c] || q[d] + 1 >= n[d] {
                continue;
            }
            if open(q) {
                area += 0.25 * grid.delta(Axis::ALL[b], q[b]) * grid.delta(Axis::ALL[c], q[c]);
            }
        }
    }
    if area > 0.0 {
        (cap * grid.delta(e.axis, e.node[d]) / (EPS0 * area)).max(1.0)
    } else {
        1.0
    }
}

/// Sets the time step of `c` by the cells of its grid where `fdtd` asks
/// for it and the model allows it, and the report either way.
pub(super) fn choose(c: &mut Compiled, fdtd: &Fdtd, plane_waves: bool) {
    let courant = match c.timestep {
        TimeStep::Courant(s) if !c.grid.is_cylindrical() => s * courant_limit(&c.grid),
        _ => f64::NAN,
    };
    let asked = fdtd.get_step_limit() == StepLimit::Cells
        && matches!(c.timestep, TimeStep::Courant(_))
        && !fdtd.get_openems_run();
    let TimeStep::Courant(s) = c.timestep else {
        c.step = None;
        return;
    };
    if !asked {
        c.step = (!c.grid.is_cylindrical()).then(|| StepReport::courant(courant, courant));
        return;
    }
    let kept = |c: &mut Compiled, why: StepObstacle| {
        let mut r = StepReport::courant(courant, courant);
        r.limit = StepChoice::CourantKept(why);
        c.step = Some(r);
    };
    if c.grid.is_cylindrical() {
        c.step = None;
        return;
    }
    if Face::ALL
        .into_iter()
        .any(|f| c.faces[f] == FaceCondition::Periodic)
    {
        return kept(c, StepObstacle::Periodic);
    }
    if !c.mur.is_empty() || !c.abc.is_empty() {
        return kept(c, StepObstacle::Mur);
    }
    if plane_waves {
        return kept(c, StepObstacle::PlaneWave);
    }
    if c.oblique_wires.is_some() {
        return kept(c, StepObstacle::ObliqueWire);
    }
    if c.dispersive_scheme != DispersiveScheme::Trapezoidal
        && c.dispersive
            .iter()
            .any(|p| !matches!(p.pole, Pole::Debye { .. }))
    {
        return kept(c, StepObstacle::LeapfrogPoles);
    }
    let grid = &c.grid;
    // The edges the materials do not describe.
    let mut live = LiveEdges::new(grid.num_nodes());
    let mut sheet_normal: HashMap<(usize, [usize; 3]), Axis> = HashMap::new();
    for e in c.sheets.iter().chain(c.surface_sheets.iter().map(|s| &s.0)) {
        live.set(e.axis, grid.index(e.node), sheet_live(grid, e));
        sheet_normal.insert((e.axis.index(), e.node), e.normal);
    }
    for e in &c.solid_edges {
        let eps = solid_eps(grid, &c.materials, e);
        live.set(e.axis, grid.index(e.node), LiveEdge::Solid { eps });
    }
    let Some(cells) = Cells::new(grid, &c.materials, Some(&live)) else {
        c.step = None;
        return;
    };
    let (worst, at) = cells.worst_exact(&walls(&c.faces));
    if !(worst > 0.0 && worst.is_finite()) {
        c.step = Some(StepReport::courant(courant, courant));
        return;
    }
    let by_cells = s / worst.sqrt();
    // The cut cells of conformal PEC.
    let mut by_cut = None;
    if let Some(conformal) = &c.conformal {
        let layers: Vec<(Face, usize)> = c.pml.iter().map(|&(f, p)| (f, p.cells)).collect();
        let Some(cut) = conformal.cut_cells(grid, &layers) else {
            return kept(c, StepObstacle::Conformal);
        };
        if !cut.is_empty() {
            let edge_open: HashMap<(usize, [usize; 3]), f64> = conformal
                .edges()
                .iter()
                .map(|e| ((e.axis.index(), e.node), e.fraction))
                .collect();
            let face_open: HashMap<(usize, [usize; 3]), f64> = conformal
                .faces()
                .iter()
                .map(|f| ((f.axis.index(), f.node), f.fraction))
                .collect();
            let cut_faces: HashSet<(usize, [usize; 3])> = c
                .sheet_cut_faces
                .iter()
                .map(|&(a, p)| (a.index(), p))
                .collect();
            let face = |a: Axis, p: [usize; 3]| -> Option<f64> {
                match face_open.get(&(a.index(), p)) {
                    Some(&f) if f > 0.0 => Some(f),
                    Some(_) => None,
                    None => Some(1.0),
                }
            };
            // As the scheme decides which faces keep their current: the
            // open part of a cut edge, its covered part closed.
            let as_kernel = |a: Axis, p: [usize; 3]| -> Option<f64> {
                match edge_open.get(&(a.index(), p)) {
                    Some(&f) if f > 0.0 => Some(1.0 / f),
                    Some(_) => None,
                    None => Some(1.0),
                }
            };
            // As the scheme is: the covered part of an edge that the
            // outline of a lossy sheet cuts is the sheet's two faces in
            // series with the open part, with `load` times the
            // capacitance of that part of the dual face.
            let as_is = |a: Axis, p: [usize; 3]| -> Option<f64> {
                let Some(&f) = edge_open.get(&(a.index(), p)) else {
                    return Some(1.0);
                };
                let Some(&nrm) = sheet_normal.get(&(a.index(), p)) else {
                    return (f > 0.0).then(|| 1.0 / f);
                };
                let t = Axis::ALL[3 - a.index() - nrm.index()];
                let mut lo = p;
                let below = lo[t.index()] > 0;
                lo[t.index()] = lo[t.index()].saturating_sub(1);
                let by_cut_face = cut_faces.contains(&(nrm.index(), p))
                    || (below && cut_faces.contains(&(nrm.index(), lo)));
                let load = if by_cut_face { CUT_FACE_LOADING } else { 1.0 };
                Some(1.0 / (f.max(0.0) + (1.0 - f.max(0.0)) / load))
            };
            let plain = Cells::new(grid, &c.materials, None).expect("checked above");
            let w = walls(&c.faces);
            let (mut kernel, mut real) = (0.0f64, 0.0f64);
            for &q in &cut {
                kernel = kernel.max(plain.eigenvalue(q, &w, &as_kernel, &face));
                real = real.max(cells.eigenvalue(q, &w, &as_is, &face));
            }
            // Within the factor of their limit as they are (a part less
            // beside lossy metal, whose edges there have a floor that
            // asks for it), and kept by the scheme's own test (a part in
            // 10³ inside its limit: that test reads `f32` coefficients).
            let own = if c.solid_edges.is_empty() {
                s
            } else {
                s.min(SOLID_CUT_LIMIT.sqrt())
            };
            let keep = conformal.options().max_stability * (1.0 - 1e-3);
            let mut dt = f64::INFINITY;
            if real > 0.0 {
                dt = dt.min(own / real.sqrt());
            }
            if kernel > 0.0 {
                dt = dt.min(keep / kernel.sqrt());
            }
            if dt.is_finite() {
                by_cut = Some(dt);
            }
        }
    }
    let (dt, limit) = match by_cut {
        Some(cut) if cut < by_cells => (cut, StepChoice::CutCells),
        _ => (by_cells, StepChoice::Cells),
    };
    let (dt, limit) = if dt > courant {
        (dt, limit)
    } else {
        (courant, StepChoice::CourantLonger)
    };
    if limit != StepChoice::CourantLonger {
        c.timestep = TimeStep::Bound(dt);
    }
    c.step = Some(StepReport {
        dt,
        courant,
        cells: Some(by_cells),
        cell: Some(at),
        cut_cells: by_cut,
        limit,
    });
}
