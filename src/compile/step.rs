//! The time step of a compiled model from its cells
//! ([`StepLimit::Cells`], design §5.4.1): the bound of
//! [`crate::engine::step_bound`] on the model's materials, sheets and
//! lossy metal, the cells' shares balanced, with the cut cells of
//! conformal PEC.

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
    /// lossy metal of the model, and the cells conformal PEC cuts as the
    /// scheme has them); `None` where it was not evaluated.
    pub cells: Option<f64>,
    /// The cell that bound is the limit of (lowest node).
    pub cell: Option<[usize; 3]>,
    /// The longest step at which every cell with a cut face of conformal
    /// PEC stays within the factor of its own limit, with the shares of
    /// the bound \[s\]; `None` without such cells, or where it was not
    /// evaluated.
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

/// What the cells outside the metal have between them of the edge `e`
/// of lossy metal, as relative permittivity times area \[m²\]: its
/// capacitance outside the metal ([`SolidEdge::outside`]) over `ε0/l`,
/// and at least the quarters of its dual face that lie in cells with an
/// edge that is not PEC (the operator gives every quarter beside a face
/// with field vacuum's capacitance, and those quarters are among
/// these).
fn solid_total(grid: &Grid, mats: &EdgeMaterials, e: &SolidEdge) -> f64 {
    let cap = e.outside.map_or(0.0, |o| o.0);
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
    let eps = if area > 0.0 {
        (cap * grid.delta(e.axis, e.node[d]) / (EPS0 * area)).max(1.0)
    } else {
        1.0
    };
    eps * area
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
    // An edge of lossy metal is one in the scheme where it is a perfect
    // conductor in the materials (the others are left as they are: the
    // edges its outline cuts, with their open parts). Beside a cut cell
    // (a face around it has an edge the metal cuts) its capacitance has
    // the floor that lets it answer for itself.
    let metal_cuts: HashSet<(usize, [usize; 3])> = c
        .conformal
        .iter()
        .flat_map(|g| g.edges())
        .filter(|e| e.fraction > 0.0 && e.fraction < 1.0)
        .map(|e| (e.axis.index(), e.node))
        .collect();
    let nodes = grid.nodes();
    let beside_cut = |e: &SolidEdge| {
        let d = e.axis.index();
        [(d + 1) % 3, (d + 2) % 3].into_iter().any(|t| {
            // The faces with the edge in them: normal to `t`, at the
            // edge's node and one before it along the third axis.
            let o = 3 - d - t;
            [0, 1].into_iter().any(|back| {
                let mut m = e.node;
                if back == 1 {
                    if m[o] == 0 {
                        return false;
                    }
                    m[o] -= 1;
                }
                if m[d] + 1 >= nodes[d] || m[o] + 1 >= nodes[o] {
                    return false;
                }
                let (mut md, mut mo) = (m, m);
                md[d] += 1;
                mo[o] += 1;
                [(d, m), (d, mo), (o, m), (o, md)]
                    .iter()
                    .any(|k| metal_cuts.contains(k))
            })
        })
    };
    for e in &c.solid_edges {
        let k = grid.index(e.node);
        if !c.materials.pec_mask(e.axis)[k] {
            continue;
        }
        let edge = if beside_cut(e) {
            LiveEdge::Floored
        } else {
            LiveEdge::Solid {
                total: solid_total(grid, &c.materials, e),
            }
        };
        live.set(e.axis, k, edge);
    }
    let Some(mut cells) = Cells::new(grid, &c.materials, Some(&live)) else {
        c.step = None;
        return;
    };
    let w = walls(&c.faces);
    // The cut cells of conformal PEC.
    let cut = match &c.conformal {
        Some(conformal) => {
            let layers: Vec<(Face, usize)> = c.pml.iter().map(|&(f, p)| (f, p.cells)).collect();
            let Some(cut) = conformal.cut_cells(grid, &layers) else {
                return kept(c, StepObstacle::Conformal);
            };
            cut
        }
        None => Vec::new(),
    };
    let (worst, at, by_cut, keep_step) = if cut.is_empty() {
        let (worst, at) = cells.worst_balanced(&w, None);
        (worst, at, None, None)
    } else {
        let conformal = c.conformal.as_ref().expect("it has cut cells");
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
            let open = edge_open.get(&(a.index(), p)).copied();
            let Some(&nrm) = sheet_normal.get(&(a.index(), p)) else {
                return match open {
                    Some(f) => (f > 0.0).then(|| 1.0 / f),
                    None => Some(1.0),
                };
            };
            // (A sheet edge the outline does not cut is all covered.)
            let f = open.unwrap_or(0.0).max(0.0);
            let t = Axis::ALL[3 - a.index() - nrm.index()];
            let mut lo = p;
            let below = lo[t.index()] > 0;
            lo[t.index()] = lo[t.index()].saturating_sub(1);
            let by_cut_face = cut_faces.contains(&(nrm.index(), p))
                || (below && cut_faces.contains(&(nrm.index(), lo)));
            let load = if by_cut_face { CUT_FACE_LOADING } else { 1.0 };
            Some(1.0 / (f + (1.0 - f) / load))
        };
        // The scheme's own test of which small faces keep their current
        // is made at the Courant step (`keep_step` below), where every
        // cut cell has to pass it for the cells to decide the step (a
        // part in 10³ inside its limit: that test reads `f32`
        // coefficients). A face that fails it is interpolated at any
        // step, and its cells bound nothing.
        let plain = Cells::new(grid, &c.materials, None).expect("checked above");
        let kernel = cut
            .iter()
            .map(|&q| plain.eigenvalue(q, &w, &as_kernel, &face))
            .fold(0.0f64, f64::max);
        let keep = conformal.options().max_stability * (1.0 - 1e-3);
        let kept_at_courant = kernel > 0.0 && courant <= keep / kernel.sqrt();
        // The cut cells as they are, in the bound with the whole cells:
        // within the factor of their limit with the shares of the grid.
        let among: HashSet<[usize; 3]> = cut.iter().copied().collect();
        let own = |cells: &Cells, q: [usize; 3]| {
            among
                .contains(&q)
                .then(|| cells.eigenvalue(q, &w, &as_is, &face))
        };
        let (worst, at) = cells.worst_balanced(&w, Some(&own));
        let real = cut
            .iter()
            .map(|&q| cells.eigenvalue(q, &w, &as_is, &face))
            .fold(0.0f64, f64::max);
        let by_cut = (real > 0.0).then(|| s / real.sqrt());
        if !kept_at_courant {
            // As before the cells decided anything.
            let mut r = StepReport::courant(courant, courant);
            r.cells = (worst > 0.0 && worst.is_finite()).then(|| s / worst.sqrt());
            r.cell = Some(at);
            r.cut_cells = by_cut;
            r.limit = StepChoice::CourantLonger;
            c.step = Some(r);
            return;
        }
        (worst, at, by_cut, Some(courant))
    };
    if !(worst > 0.0 && worst.is_finite()) {
        c.step = Some(StepReport::courant(courant, courant));
        return;
    }
    // (The cut cells are among the cells of `worst`.)
    let by_cells = s / worst.sqrt();
    let limit = match by_cut {
        Some(cut) if cut <= by_cells * (1.0 + 1e-12) => StepChoice::CutCells,
        _ => StepChoice::Cells,
    };
    let (dt, limit) = if by_cells > courant {
        (by_cells, limit)
    } else {
        (courant, StepChoice::CourantLonger)
    };
    if limit != StepChoice::CourantLonger {
        c.timestep = TimeStep::Bound(dt);
        // The faces conformal PEC keeps are those of the Courant step,
        // and the check it makes of the step it is applied at (the bound
        // of the geometric shares over the grid, which a longer step
        // does not pass, and then a power iteration) is what the bound
        // here has settled, the cut cells in it.
        if let Some(conformal) = c.conformal.take() {
            let mut options = *conformal.options();
            options.keep_step = keep_step;
            options.iterations = 0;
            c.conformal = Some(conformal.with_options(options));
        }
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
