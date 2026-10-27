//! openEMS's `ConductingSheet` model (files only,
//! [`Fdtd::openems_run`](crate::Fdtd::openems_run), Cartesian and
//! cylindrical grids).
//!
//! ems's own sheet ([`crate::engine::sheet`]) is a thin lossy slab with
//! skin effect, accurate to about 1 % against the exact slab (design
//! §5.9). openEMS v0.37 uses a "behavioural" model instead, which
//! differs from the slab by up to a factor two in loss and by tens of dB
//! in transmission (design §5.16.2). For models read from openEMS files
//! ems reproduces openEMS's model, as measured on the black box (operator
//! dumps `--debug-operator` and per-step field dumps on hand-written
//! models; system identification of the sheet-edge update, design
//! §5.16.2; no openEMS source consulted):
//!
//! * **Edge circuit.** A sheet edge (integral form, voltage `u = E·l`,
//!   curl current `I = ∮i`) is a capacitor `C` with a conductance `G` in
//!   parallel with two series R–L branches, `C·du/dt + G·u + Σ_b i_b = I`,
//!   `L_b·di_b/dt + R_b·i_b = u`, integrated like openEMS's Drude poles
//!   (branch currents at half steps driven by `u^n`, the leapfrog scheme
//!   of [`Scheme::Leapfrog`](crate::engine::dispersive::Scheme::Leapfrog)).
//!   On Cartesian grids the branches keep openEMS's single-precision state
//!   and operation order ([`crate::engine::openems_sheet`], design
//!   §5.16.13); on cylindrical grids they are poles of the general
//!   dispersive operator (`f64` state). The medium around the sheet does
//!   not enter (the vacuum capacitance of the cell is replaced, not
//!   added).
//! * **Scaling.** Per square of sheet, `G = g·σt`, `R_b = r_b/(σt)`,
//!   `L_b = l_b·μ0·t` and `C = dt²/(c·μ0·t)` (the capacitance depends on
//!   the time step: it is no physical sheet capacitance); an edge of
//!   length `l` with sheet width `w` gets `w/l` squares. The table
//!   [`TABLE`] lists `(g, c, l_1, r_1, l_2, r_2)` for 30 bins `k` of the
//!   normalised thickness `ν = (t/δ)²/80` at the excitation's highest
//!   frequency: `k = 0` for `ν ≤ 1`, `k = ⌈log_1.5 ν⌉` above, `k = 29`
//!   from `ν > 1.5^28` on (all bins measured; the model of a bin is
//!   independent of `σ`, `t`, `dt` and the mesh to 7 digits). At DC every
//!   bin conducts `≈ σt` per square (`g + 1/r_1 + 1/r_2` = 0.9988 for
//!   `k = 0`).
//! * **Frequency.** `f0 + fc` of a Gaussian excitation, `f0` of a sine,
//!   the `<FDTD f_max>` attribute for the other signals (0 without it).
//! * **Width.** An edge whose midpoint lies in the sheet (a primitive of
//!   the sheet that wins the priority there) is a sheet edge. Its width
//!   `w` is the dual edge across it in the sheet plane (the half widths of
//!   its two neighbouring cells along the sheet, across the edge) if it is
//!   an *interior* edge, half of that on a *border* edge. The edge is
//!   interior if the sheet contains the two points at its start node and
//!   the centres of the two neighbouring cells (where the dual edges
//!   through the start node cross them), else it is a border edge: one
//!   rule for sheet borders on and off the mesh lines, axis-aligned or
//!   oblique (polygon sides). A neighbouring cell whose lower corner lies
//!   on the first mesh line of the axis across the edge never counts
//!   (openEMS's quirk: a sheet across a two-cell TEM line carries half its
//!   current), and an edge in the first cell of its own axis is PEC.
//!   Measured on operator dumps of 140 hand-written and random models
//!   (graded meshes, off-mesh borders, polygons, every normal), all edges
//!   identical.
//! * **Cylindrical grids.** The same model with the physical lengths: an
//!   α edge `ρ·Δα`, a dual width across along α `ρ·Δα*` at the edge's
//!   start node. On a closed azimuth openEMS updates the ρ and z edges of
//!   the seam at `α = 2π` (the `α = 0` copy is not used): the neighbouring
//!   cells there are cells `N − 1` and `N`, the copy of cell 0 (which
//!   counts); the edge is in a box that contains 2π as written, or 0 and
//!   `α_{N−1}`. openEMS compares box angles as written (no wrap), except
//!   that a point past 2π (cell `N`) also matches as `α − 2π` in a box of
//!   nonzero α extent: a fin drawn at `α = 0` has no effect (it is not on
//!   the `2π` copy), boxes past 2π do not reach the first α cells. ems
//!   stores the seam edge twice (`j = 0` and `j = N`, joined by the
//!   periodic merge) and splits the edge's `C`, `G` and branches between
//!   the copies by their halves of the dual face.
//! * **Boundaries.** The "first mesh line" above is the last line an
//!   absorbing boundary claims on a lower face: line 0 for PEC and PMC,
//!   lines 0…1 for Mur, 0…n for `PML_n`; on an upper face the last 2
//!   (Mur) or `n + 1` (`PML_n`) lines. A neighbouring cell whose lower
//!   corner lies on such a line does not count, and a sheet edge whose
//!   start node lies on one (along either in-plane axis) is PEC (black
//!   box, wave 14: `--debug-PEC` and `--debug-operator` dumps of sheets
//!   starting and ending at every line near Mur, `PML_3` and PMC faces;
//!   design §5.16.3; Cartesian grids).
//! * **Priorities.** A material or metal of higher priority at the edge
//!   midpoint removes the sheet edge (it keeps its material); the sheet
//!   wins ties. Edges on the outer mesh planes are left to the boundary.
//!
//! The time step openEMS uses ignores the sheets; ems resolves it before
//! the sheet capacitances are set and pins it.

use csx::ContinuousStructure;
use csx::primitives::{Primitive, Shape};
use csx::properties::{PropertyKind, TypeMask};
use csx::{CoordSystem, Vec3};

use crate::Result;
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};
use crate::engine::dispersive::{EdgePole, Field, Pole};
use crate::engine::{EdgeMaterials, Grid, OPENEMS_EPS0, SheetEdge, TimeStep};
use crate::fdtd::Fdtd;
use crate::waveform::Waveform;

/// openEMS's sheet model per bin `k` (see the [module docs](self)):
/// `(g, c, l_1, r_1, l_2, r_2)` per square, `G = g·σt`,
/// `C = dt²/(c·μ0·t)`, `L_b = l_b·μ0·t`, `R_b = r_b/(σt)`. Measured with
/// openEMS v0.37 (σ = 1e6 S/m, 1 mm cells, `dt` = 1.8 ps, 1500 steps;
/// reproduced with other `σ`, `t` and `dt` to 7 digits). `c` of bin 0 is
/// refined to the value that reproduces the most `f32` coefficients of
/// openEMS (operator dumps of 1 523 random models; design §5.16.13):
/// 0.02222457325 instead of the measured 0.022224575.
#[rustfmt::skip]
pub(crate) const TABLE: [[f64; 6]; 30] = [
    [0.13708436, 0.02222457325, 0.11928209, 1.3027544, 0.099069049, 10.62245],
    [0.12443278, 0.019525936, 0.11765065, 1.2992612, 0.086577439, 10.219529],
    [0.10874258, 0.016475925, 0.11249568, 1.3200334, 0.072683973, 10.247559],
    [0.09215452, 0.013604945, 0.10237379, 1.4182922, 0.059808052, 10.991476],
    [0.076392499, 0.011153572, 0.087753569, 1.6743265, 0.048958344, 12.704509],
    [0.062392063, 0.0091246982, 0.07148522, 2.1387697, 0.040058271, 15.598306],
    [0.050687011, 0.0074560144, 0.057215781, 2.7691268, 0.032755173, 19.55715],
    [0.041241612, 0.0060890102, 0.046112844, 3.4988017, 0.026761739, 24.325811],
    [0.03364926, 0.004971778, 0.037551334, 4.311448, 0.021853402, 29.887339],
    [0.027482566, 0.0040593582, 0.030693937, 5.2662049, 0.017842195, 36.556805],
    [0.022441566, 0.0033144285, 0.025070413, 6.4439781, 0.014567805, 44.753744],
    [0.018323172, 0.0027062229, 0.020468708, 7.8934045, 0.011894602, 54.815747],
    [0.014960806, 0.0022096219, 0.016712625, 9.6674121, 0.0097119032, 67.135337],
    [0.01221545, 0.0018041488, 0.01364581, 11.8401, 0.0079297354, 82.223603],
    [0.0099738759, 0.0014730814, 0.011141759, 14.501103, 0.0064746017, 100.7029],
    [0.008143637, 0.0012027661, 0.0090972116, 17.760148, 0.0052864913, 123.33532],
    [0.006649251, 0.0009820545, 0.0074278444, 21.751641, 0.0043164022, 151.05424],
    [0.0054290922, 0.00080184417, 0.0060648135, 26.640202, 0.0035243278, 185.00275],
    [0.0044328367, 0.0006547032, 0.004951903, 32.627421, 0.0028776024, 226.58098],
    [0.0036193982, 0.00053456309, 0.0040432162, 39.960208, 0.002349553, 277.50354],
    [0.0029552275, 0.00043646901, 0.003301277, 48.941, 0.0019184023, 339.87044],
    [0.0024129352, 0.0003563757, 0.0026954871, 59.940125, 0.0015663698, 416.2535],
    [0.0019701562, 0.00029097973, 0.0022008634, 73.411114, 0.0012789365, 509.8022],
    [0.0016086298, 0.00023758431, 0.0017970087, 89.909314, 0.0010442485, 624.37284],
    [0.0013134234, 0.00019398532, 0.0014672034, 110.11961, 0.00085261976, 764.72806],
    [0.0010724224, 0.00015838974, 0.001198013, 134.86311, 0.00069616658, 936.55255],
    [0.00090313827, 0.00012996173, 0.0010938562, 149.27697, 0.00056924546, 1000.0],
    [0.0007674973, 0.00010602016, 0.0010684632, 153.52594, 0.00046204756, 1000.0],
    [0.00064825789, 8.5842202e-05, 0.0010676593, 148.93717, 0.00037231309, 1000.0],
    [0.00054507, 6.9225859e-05, 0.0011001749, 119.35233, 0.00029891198, 1000.0],
];

/// The highest frequency openEMS's sheet model is chosen for (module
/// docs): `f0 + fc` of a Gaussian, `f0` of a sine, else the explicit
/// `f_max` (0 without one).
pub(crate) fn sheet_frequency(fdtd: &Fdtd) -> f64 {
    match fdtd.get_waveform() {
        Some(Waveform::Gaussian { f0, fc } | Waveform::OpenEmsGaussian { f0, fc }) => f0 + fc,
        Some(Waveform::Sinusoid { f0 }) => *f0,
        _ => fdtd.get_explicit_max_frequency().unwrap_or(0.0),
    }
}

/// The table bin of a sheet of conductivity `sigma`, thickness `t` at
/// frequency `f` (module docs).
pub(crate) fn bin(sigma: f64, t: f64, f: f64) -> usize {
    let nu = core::f64::consts::PI * f * MU0 * sigma * t * t / 80.0;
    if nu.is_nan() || nu <= 1.0 {
        return 0;
    }
    let k = (nu.ln() / 1.5f64.ln()).ceil();
    if k >= 29.0 { 29 } else { k.max(1.0) as usize }
}

/// The emulated sheet edges: edge materials, poles and PEC edges.
#[derive(Debug, Default)]
pub(super) struct Emulation {
    /// Native sheet edges that are not emulated (kept for the slab
    /// operator): those of cylindrical grids with an azimuthal multigrid.
    pub rest: Vec<SheetEdge>,
    /// Branch poles of the emulated edges on cylindrical grids (the
    /// general dispersive operator).
    pub poles: Vec<EdgePole>,
    /// Branches `(L, R)` of the emulated edges on Cartesian grids, with
    /// openEMS's single-precision update
    /// ([`OpenEmsSheetBranches`](crate::engine::openems_sheet)), and the
    /// time step they are formed with.
    pub branches: Vec<super::OpenEmsSheetBranch>,
    /// The emulated edges with the `[ε_r, κ]` they had.
    pub edges: Vec<(Axis, [usize; 3], [f64; 2])>,
    /// Number of edges made PEC (in the first cell of their axis).
    pub pec: usize,
}

/// Whether the sheet `name` contains `p` (drawing units, in the grid's
/// coordinates `cs`) and wins the priority there (ties go to the sheet).
/// `wrap`: on cylindrical grids, whether an `α` past 2π may also match
/// as `α − 2π` ([`alpha_in`]).
fn sheet_at(model: &ContinuousStructure, cs: CoordSystem, name: &str, p: Vec3, wrap: bool) -> bool {
    let hits = model.properties_at(p, cs, TypeMask::PHYSICAL);
    let Some(top) = hits.first() else {
        return false;
    };
    let prio = top.primitive.priority;
    // The sheet wins ties.
    hits.iter()
        .take_while(|h| h.primitive.priority == prio)
        .any(|h| {
            h.property.name == name
                && matches!(h.property.kind, PropertyKind::ConductingSheet(_))
                && (cs == CoordSystem::Cartesian || alpha_in(model, h.primitive, p[1], wrap))
        })
}

/// openEMS's azimuth test of a cylindrical box (module docs): `α` within
/// the box's α range as written, or (`wrap`) `α − 2π` for points past 2π
/// (the copies of the first α cells) in a box of nonzero α extent. csx
/// compares angles modulo 2π instead, which also puts a fin drawn at
/// `α = 0` on the `α = 2π` copy of the seam (openEMS: not) and wraps
/// boxes past 2π onto the first α cells. Other primitives keep csx's
/// test.
fn alpha_in(model: &ContinuousStructure, prim: &Primitive, alpha: f64, wrap: bool) -> bool {
    let Shape::Box(b) = &prim.shape else {
        return true;
    };
    if !prim.transform.is_empty() || prim.shape_cs(model.coord_system) != CoordSystem::Cylindrical {
        return true;
    }
    let (a0, a1) = (b.start.value()[1], b.stop.value()[1]);
    let (lo, hi) = (a0.min(a1), a0.max(a1));
    let tol = 1e-12 * lo.abs().max(hi.abs()).max(1.0);
    let tau = core::f64::consts::TAU;
    let within = |a: f64| a >= lo - tol && a <= hi + tol;
    within(alpha) || (wrap && hi - lo > tol && alpha >= tau - tol && within(alpha - tau))
}

/// Replaces the native sheet edges `edges` by openEMS's model (module
/// docs): sets `ε`, `κ` (and PEC) of the edges in `materials`, returns
/// the branch poles. `timestep` is resolved here with `materials` and
/// pinned to that value.
pub(super) fn emulate(
    model: &ContinuousStructure,
    grid: &Grid,
    fdtd: &Fdtd,
    edges: Vec<SheetEdge>,
    names: Vec<String>,
    materials: &mut EdgeMaterials,
    timestep: &mut TimeStep,
) -> Result<Emulation> {
    let mut out = Emulation::default();
    if edges.is_empty() {
        return Ok(out);
    }
    if grid.has_multigrid() {
        out.rest = edges;
        return Ok(out);
    }
    let dt = match *timestep {
        TimeStep::OpenEms(o) => {
            let dt = o.resolve(grid, materials)?;
            *timestep = TimeStep::OpenEms(crate::engine::OpenEmsTimestep {
                fixed: Some(dt),
                factor: 1.0,
                period: None,
                ..o
            });
            dt
        }
        t => t.resolve(grid)?,
    };
    let f = sheet_frequency(fdtd);
    let n = grid.nodes();
    let cyl = grid.is_cylindrical();
    let cs = if cyl {
        CoordSystem::Cylindrical
    } else {
        CoordSystem::Cartesian
    };
    let closed = grid.is_closed();
    // The seam: α line 0 and its copy N on a closed azimuth.
    let seam_n = n[1] - 1;
    let lines: [&[f64]; 3] = core::array::from_fn(|a| model.grid.lines(a));
    let mut done = std::collections::HashSet::new();
    // The mesh lines each boundary claims (module docs): the first
    // `r_lo + 1` and the last `r_hi + 1` lines of every axis.
    let reach = |face: crate::Face| match fdtd.get_boundaries().get(face) {
        crate::Boundary::Mur(_) => 1,
        crate::Boundary::Pml(p) => p.cells,
        _ => 0,
    };
    let region: [(usize, usize); 3] = core::array::from_fn(|a| {
        let ax = Axis::ALL[a];
        (
            reach(crate::Face::new(ax, false)),
            n[a].saturating_sub(1 + reach(crate::Face::new(ax, true))),
        )
    });
    let in_region = |a: usize, i: usize| !cyl && i > 0 && (i <= region[a].0 || i >= region[a].1);
    for (e, name) in edges.into_iter().zip(names) {
        let d = e.axis.index();
        let nrm = e.normal.index();
        let t = 3 - d - nrm;
        let mut node = e.node;
        // ρ and z edges on the seam of a closed azimuth: one physical edge
        // stored twice (j = 0 and j = N, joined by the periodic merge);
        // openEMS evaluates it at α_N.
        let seam = closed && d != 1 && (node[1] == 0 || node[1] == seam_n);
        if seam {
            node[1] = 0;
        }
        if !done.insert((d, node)) {
            continue;
        }
        // Outer mesh planes: the boundary decides (not the seam).
        if (0..3).any(|a| a != d && !(closed && a == 1) && (node[a] == 0 || node[a] + 1 == n[a])) {
            continue;
        }
        // Line `i` of axis `a` as openEMS evaluates it (the seam at α_N).
        let at = |a: usize, i: usize| {
            if seam && a == 1 {
                lines[1][seam_n]
            } else {
                lines[a][i]
            }
        };
        let mut mid = Vec3::new(at(0, node[0]), at(1, node[1]), at(2, node[2]));
        mid.0[d] = 0.5 * (lines[d][node[d]] + lines[d][node[d] + 1]);
        // The seam edge (at α_N = 2π) is in a box that contains 2π as
        // written, or 0 and α_{N−1} (measured on tubes [0, a]: the seam
        // edge is a sheet edge from a ≥ α_{N−1} on).
        let on_sheet = sheet_at(model, cs, &name, mid, false)
            || (seam
                && sheet_at(model, cs, &name, mid, true)
                && sheet_at(
                    model,
                    cs,
                    &name,
                    Vec3::new(mid[0], lines[1][seam_n - 1], mid[2]),
                    false,
                ));
        if !on_sheet {
            continue;
        }
        // An interior sheet edge carries the full width, a border edge
        // half of it. The edge is interior if the sheet contains the two
        // points at its start node along d and the centres of its two
        // neighbouring cells along t (where the dual edges through the
        // start node cross the cells); a cell with index 0 along t (lower
        // corner on the first mesh line) never counts. On the seam the
        // cells along α are N − 1 and N (the copy of cell 0, which
        // counts). An edge in the first cell along d (index 0) is PEC.
        let jt = if seam && t == 1 { seam_n } else { node[t] };
        let centre = |c: usize| {
            if seam && t == 1 && c == seam_n {
                lines[1][seam_n] + 0.5 * (lines[1][1] - lines[1][0])
            } else {
                0.5 * (lines[t][c] + lines[t][c + 1])
            }
        };
        let inside = |p: Vec3| sheet_at(model, cs, &name, p, true);
        let interior = [jt - 1, jt].into_iter().all(|c| {
            let mut p = mid;
            p.0[d] = lines[d][node[d]];
            p.0[t] = centre(c);
            c != 0 && !in_region(t, c) && inside(p)
        });
        // The stored copies of the edge with their share of it.
        let copies: Vec<([usize; 3], f64)> = if seam {
            let (lo, hi) = (grid.delta(Axis::Y, 0), grid.delta(Axis::Y, seam_n - 1));
            let mut top = node;
            top[1] = seam_n;
            vec![(node, lo / (lo + hi)), (top, hi / (lo + hi))]
        } else {
            vec![(node, 1.0)]
        };
        if node[d] == 0 || in_region(d, node[d]) || in_region(t, node[t]) {
            for &(p, _) in &copies {
                materials.pec_mask_mut(e.axis)[grid.index(p)] = true;
                out.pec += 1;
            }
            continue;
        }
        // Sheet width across the edge: the dual edge along t (physical,
        // at the edge's start node), half of it on a border edge.
        let dual = if seam && t == 1 {
            0.5 * (grid.delta(Axis::Y, 0) + grid.delta(Axis::Y, seam_n - 1))
        } else {
            grid.dual_delta(Axis::ALL[t], node[t])
        };
        let dual = if cyl && t == 1 {
            grid.lines(Axis::X)[node[0]] * dual
        } else {
            dual
        };
        let w = if interior { dual } else { 0.5 * dual };
        let (sigma, th) = (e.conductivity, e.thickness);
        let row = TABLE[bin(sigma, th, f)];
        let sq = w / grid.edge_length(e.axis, node);
        let (st, lt) = (sigma * th, MU0 * th);
        for (p, share) in copies {
            let idx = grid.index(p);
            let s = sq * share;
            let c = dt * dt * s / (row[1] * lt);
            let g_e = row[0] * st * s;
            let geo = grid.dual_face_area(e.axis, p) / grid.edge_length(e.axis, p);
            out.edges.push((
                e.axis,
                p,
                [materials.eps_r(e.axis)[idx], materials.kappa(e.axis)[idx]],
            ));
            // The coefficients of openEMS files are formed with openEMS's
            // ε0 (`CoeffRounding::OpenEms`): the edge gets `C = c`.
            materials.eps_r_mut(e.axis)[idx] = c / (OPENEMS_EPS0 * geo);
            materials.kappa_mut(e.axis)[idx] = g_e / geo;
            let lr =
                [(row[2], row[3]), (row[4], row[5])].map(|(lb, rb)| (lb * lt / s, rb / (st * s)));
            if !cyl {
                out.branches.push((e.axis, p, lr));
                continue;
            }
            let m = EPS0 * geo;
            for (lb, rb) in lr {
                out.poles.push(EdgePole {
                    field: Field::Electric,
                    axis: e.axis,
                    node: p,
                    pole: Pole::drude((1.0 / (m * lb)).sqrt(), rb / lb),
                });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins() {
        // Copper, 35 µm: the bin edges measured on the black box (f0 + fc
        // of a Gaussian): 7.3094 GHz and 10.965 GHz bound bin 9.
        let (s, t) = (5.8e7, 3.5e-5);
        assert_eq!(bin(s, t, 7.30e9), 8);
        assert_eq!(bin(s, t, 7.32e9), 9);
        assert_eq!(bin(s, t, 10.95e9), 9);
        assert_eq!(bin(s, t, 10.98e9), 10);
        assert_eq!(bin(s, t, 0.0), 0);
        assert_eq!(bin(1e3, 1e-5, 1e12), 0);
        assert_eq!(bin(s, 1.0, 1e12), 29);
        // t/δ = √80 is the edge of bin 0.
        let f = 80.0 / (core::f64::consts::PI * MU0 * 1e6 * 4e-10);
        assert_eq!(bin(1e6, 2e-5, f * 0.9999), 0);
        assert_eq!(bin(1e6, 2e-5, f * 1.0001), 1);
    }

    #[test]
    fn thin_sheets_conduct_sigma_t() {
        // Bin 0 (t ≤ √80·δ): the DC sheet conductance is σt within 0.12 %.
        let row = TABLE[0];
        let g = row[0] + 1.0 / row[3] + 1.0 / row[5];
        assert!((g - 1.0).abs() < 2e-3, "{g}");
    }
}
