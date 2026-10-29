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
//!   The branches keep openEMS's single-precision state and operation
//!   order ([`crate::engine::openems_sheet`]; on the seam of a closed
//!   azimuth openEMS's seam update, design §5.16.13). The medium around
//!   the sheet does not enter (the vacuum capacitance of the cell is
//!   replaced, not added).
//! * **Scaling.** Per square of sheet, `G = g·σt`, `R_b = r_b/(σt)`,
//!   `L_b = l_b·μ0·t` and `C = dt²/(c·μ0·t)` (the capacitance depends on
//!   the time step: it is no physical sheet capacitance; in every bin
//!   `1/c = (1/l_1 + 17/l_2)/4` to 2e-7, i.e. `C = dt²·(1/(4L_1) +
//!   17/(4L_2))`); an edge of length `l` with sheet width `w` gets `w/l`
//!   squares. All in openEMS's single-precision arithmetic
//!   ([`edge_circuit`], design §5.16.13). The table
//!   [`TABLE`] lists `(g, c, l_1, r_1, l_2, r_2)` for 30 bins `k` of the
//!   normalised thickness `ν = (t/δ)²/80` at the excitation's highest
//!   frequency: `k = 0` for `ν < 1`, `k = ⌊log_1.5 ν⌋ + 1` above, `k = 29`
//!   from `ν ≥ 1.5^28` on, `ν` in openEMS's single precision (`bin`)
//!   (all bins measured; the model of a bin is
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
//!   stores the seam edge twice (`j = 0` and `j = N`) and splits the
//!   edge's `C` and `G` between the copies by their halves of the dual
//!   face; openEMS's seam update (`OpenemsSeam`) recombines them and runs
//!   the branches of the whole edge on the `α = 2π` copy, the `α = 0` copy
//!   holding the uncorrected value (as for dispersive poles, design
//!   §5.16.12). Without that update (other curl orders, multigrid, …) each
//!   copy carries its share of the branches and the periodic merge joins
//!   them.
//! * **Boundaries.** The "first mesh line" above is the last line an
//!   absorbing boundary claims on a lower face: line 0 for PEC and PMC,
//!   lines 0…1 for Mur, 0…n for `PML_n`; on an upper face the last 2
//!   (Mur) or `n + 1` (`PML_n`) lines. A neighbouring cell whose lower
//!   corner lies on such a line does not count, and a sheet edge whose
//!   start node lies on one (along any axis: a sheet parallel to the face
//!   on one of those lines is PEC as a whole, wave 19) is PEC (black
//!   box, wave 14: `--debug-PEC` and `--debug-operator` dumps of sheets
//!   starting and ending at every line near Mur, `PML_3` and PMC faces;
//!   design §5.16.3; Cartesian grids). Next to an upper PMC face openEMS
//!   keeps the H edge of the same direction and node as a sheet edge in
//!   the dual layer it otherwise zeroes (`pmc_half_cell`, design
//!   §5.16.13).
//! * **Priorities.** A material or metal of higher priority at the edge
//!   midpoint removes the sheet edge (it keeps its material); at equal
//!   priority the property defined first in the file wins (wave 19,
//!   `--debug-operator` dumps of a sheet against a dielectric, a Drude
//!   medium and a metal box defined before and after it; it had been
//!   measured with the sheet defined first only). A sheet that wins over
//!   metal is a sheet edge, not PEC
//!   ([`super::openems::sheet_over_metal`]). Lumped elements do not take
//!   part in the priority: a parallel R, C or R‖C element leaves a sheet
//!   edge alone (no effect on it; the element is still spread over all
//!   its edges, and its caps leave sheet edges alone too, `sim::map`),
//!   while an element with an inductor or a
//!   series element is installed over the sheet's circuit, at any
//!   priority: the sheet's capacitance is the medium the element replaces
//!   or limits (§5.16.1), the sheet's conductance is dropped, the sheet's
//!   branches stay (operator dumps identical; the probes still differ by
//!   2e-4…1e-1, the order of the element's and the branches' updates is
//!   not reproduced, design §5.16.3).
//!   Dispersive poles at a sheet edge's midpoint have no effect (the
//!   poles of the media the sheet wins over are dropped). Edges on the
//!   outer mesh planes are left to the boundary.
//!
//! The time step openEMS uses ignores the sheets; ems resolves it before
//! the sheet capacitances are set and pins it.

use csx::ContinuousStructure;
use csx::primitives::{Primitive, Shape};
use csx::properties::{PropertyKind, TypeMask};
use csx::{CoordSystem, Vec3};

use crate::Result;
use crate::axis::Axis;
use crate::engine::{EdgeMaterials, Grid, OPENEMS_EPS0, OPENEMS_MU0, SheetEdge, TimeStep};
use crate::fdtd::Fdtd;
use crate::waveform::Waveform;

/// openEMS's sheet model per bin `k` (see the [module docs](self)):
/// `(g, c, l_1, r_1, l_2, r_2)` per square, `G = g·σt`,
/// `C = dt²/(c·μ0·t)`, `L_b = l_b·μ0·t`, `R_b = r_b/(σt)`. Measured with
/// openEMS v0.37 (σ = 1e6 S/m, 1 mm cells, `dt` = 1.8 ps, 1500 steps;
/// reproduced with other `σ`, `t` and `dt` to 7 digits). The capacitance
/// is formed from `l_1`, `l_2` ([`edge_circuit`]); `c` is the measured
/// `(1/l_1 + 17/l_2)/4` (to 2e-7, unused). `g`, `l_1` and `l_2` of every
/// bin and `r_1`, `r_2` of bin 0 are refined to 11–13 digits, to the
/// values that reproduce openEMS's `f32` rounding steps (operator dumps
/// and per-step dumps of single-edge models, design §5.16.13; e.g. bin 0
/// `l_2` 0.0990690603 instead of the measured 0.099069049). `r_b` of the
/// other bins is the measured value: there the branches are inductive at
/// any frequency an excitation can carry (`R·dt/2L` < 0.05).
#[rustfmt::skip]
pub(crate) const TABLE: [[f64; 6]; 30] = [
    [0.137084340054, 0.02222457325, 0.119282088951, 1.302754546, 0.0990690603255, 10.62245057],
    [0.124432770979, 0.019525936, 0.117650643055, 1.2992612, 0.0865774432243, 10.219529],
    [0.10874259256, 0.016475925, 0.112495689579, 1.3200334, 0.0726839782207, 10.247559],
    [0.0921545029514, 0.013604945, 0.102373790897, 1.4182922, 0.0598080542444, 10.991476],
    [0.0763924952568, 0.011153572, 0.0877535678803, 1.6743265, 0.0489583437563, 12.704509],
    [0.0623920685841, 0.0091246982, 0.0714852202949, 2.1387697, 0.0400582737111, 15.598306],
    [0.0506870193634, 0.0074560144, 0.0572157790471, 2.7691268, 0.0327551749758, 19.55715],
    [0.0412416168046, 0.0060890102, 0.0461128464864, 3.4988017, 0.026761739013, 24.325811],
    [0.0336492594616, 0.004971778, 0.0375513332623, 4.311448, 0.0218534019281, 29.887339],
    [0.0274825647083, 0.0040593582, 0.030693935758, 5.2662049, 0.017842192351, 36.556805],
    [0.0224415656409, 0.0033144285, 0.0250704153117, 6.4439781, 0.0145678049496, 44.753744],
    [0.0183231737407, 0.0027062229, 0.0204687079875, 7.8934045, 0.0118946020796, 54.815747],
    [0.0149608078551, 0.0022096219, 0.0167126250576, 9.6674121, 0.00971190254458, 67.135337],
    [0.0122154505375, 0.0018041488, 0.0136458082892, 11.8401, 0.00792973544699, 82.223603],
    [0.00997387466324, 0.0014730814, 0.0111417587353, 14.501103, 0.00647460224422, 100.7029],
    [0.00814363571331, 0.0012027661, 0.00909721105062, 17.760148, 0.00528649098115, 123.33532],
    [0.0066492519176, 0.0009820545, 0.00742784502358, 21.751641, 0.00431640220756, 151.05424],
    [0.00542909273748, 0.00080184417, 0.00606481357423, 26.640202, 0.00352432807109, 185.00275],
    [0.00443283701916, 0.0006547032, 0.00495190327264, 32.627421, 0.00287760227497, 226.58098],
    [0.00361939779282, 0.00053456309, 0.0040432162758, 39.960208, 0.00234955292005, 277.50354],
    [0.00295522765367, 0.00043646901, 0.00330127701783, 48.941, 0.00191840249878, 339.87044],
    [0.00241293536167, 0.0003563757, 0.00269548714004, 59.940125, 0.00156636976862, 416.2535],
    [0.00197015616651, 0.00029097973, 0.00220086344996, 73.411114, 0.00127893644741, 509.8022],
    [0.00160862981206, 0.00023758431, 0.00179700862248, 89.909314, 0.00104424855814, 624.37284],
    [0.0013134233954, 0.00019398532, 0.00146720336246, 110.11961, 0.000852619746587, 764.72806],
    [0.00107242246864, 0.00015838974, 0.0011980129395, 134.86311, 0.000696166544835, 936.55255],
    [0.000903138344057, 0.00012996173, 0.00109385610511, 149.27697, 0.000569245491525, 1000.0],
    [0.000767497363702, 0.00010602016, 0.00106846334836, 153.52594, 0.000462047579022, 1000.0],
    [0.000648257956771, 8.5842202e-05, 0.00106765935995, 148.93717, 0.000372313090849, 1000.0],
    [0.000545069956667, 6.9225859e-05, 0.00110017492041, 119.35233, 0.000298911971364, 1000.0],
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
/// frequency `f` (module docs), with openEMS's single-precision
/// arithmetic (black box, design §5.16.13): `σ` and `t` are read as `f32`,
/// the per-square conductance `G = f32(σt)`, the normalised frequency
/// `X = f32(8/(G·μ0·t))` (openEMS's `μ0`), `(t/δ)² = f32(4·f32(2πf)/X)`
/// and `ν = (t/δ)²/80`; bin `⌊log_1.5 ν⌋ + 1` from `ν = 1` on (`ν = 1.5^k`
/// is bin `k + 1`). This reproduces 2965 of 2975 bin boundaries measured to
/// one `f32` step of `f0` (the formula in `f64` misses 60); the others are
/// `ν` exactly `1.5^11`, which openEMS puts in bin 11.
pub(crate) fn bin(sigma: f64, t: f64, f: f64) -> usize {
    let r32 = |x: f64| x as f32 as f64;
    let t = r32(t);
    let g = r32(r32(sigma) * t);
    let x = r32(8.0 / (g * OPENEMS_MU0 * t));
    let w = r32(2.0 * core::f64::consts::PI * f);
    let nu = r32(4.0 * w / x) / 80.0;
    if nu.is_nan() || nu < 1.0 {
        return 0;
    }
    let k = (nu.log10() / 1.5f64.log10()).floor() + 1.0;
    if k >= 29.0 { 29 } else { k as usize }
}

/// The circuit of one sheet edge in openEMS's model (module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EdgeCircuit {
    /// The two branches `(L, R)` \[H, Ω\].
    pub branches: [(f64, f64); 2],
    /// The capacitance per `dt²` \[F/s²\].
    pub c_per_dt2: f64,
    /// The conductance \[S\].
    pub g: f64,
}

/// The circuit of a sheet edge with `l/w` (edge length over sheet width:
/// the inverse number of squares) in bin `row`, with openEMS's
/// single-precision arithmetic (black box, design §5.16.13): per square
/// `L_b = f32(l_b·8/(G·X))`, `R_b = f32(r_b/G)` and `G_s = f32(g·G)`
/// (`G = f32(σt)`, `X` as in [`bin`]); per edge, with `k = f32(l/w)`,
/// `L_b,e = f32(L_b·k)`, `R_b,e = f32(R_b·k)`, `G_e = f32(G_s/k)` and
/// `C/dt² = f32(1/L_1,e)/4 + f32(1/L_2,e)/4 + 4/L_2,e` (mathematically
/// `dt²/(c·μ0·t)` per square with `1/c = (1/l_1 + 17/l_2)/4`). With the
/// table's refined constants this reproduces every `vi` and `vv` of
/// several hundred single-edge models and random meshes per bin and every
/// step of the branch currents of 200 single-edge models.
pub(crate) fn edge_circuit(sigma: f64, t: f64, l_over_w: f64, row: &[f64; 6]) -> EdgeCircuit {
    let r32 = |x: f64| x as f32 as f64;
    let t = r32(t);
    let g = r32(r32(sigma) * t);
    let x = r32(8.0 / (g * OPENEMS_MU0 * t));
    let q = 8.0 / (g * x);
    let k = r32(l_over_w);
    let branches = [(row[2], row[3]), (row[4], row[5])]
        .map(|(lb, rb)| (r32(r32(lb * q) * k), r32(r32(rb / g) * k)));
    let (l1, l2) = (branches[0].0, branches[1].0);
    EdgeCircuit {
        branches,
        c_per_dt2: r32(1.0 / l1) / 4.0 + r32(1.0 / l2) / 4.0 + 4.0 / l2,
        g: r32(r32(row[0] * g) / k),
    }
}

/// The emulated sheet edges: edge materials, branches and PEC edges.
#[derive(Debug, Default)]
pub(super) struct Emulation {
    /// Native sheet edges that are not emulated (kept for the slab
    /// operator): those of cylindrical grids with an azimuthal multigrid.
    pub rest: Vec<SheetEdge>,
    /// Branches `(L, R)` of the emulated edges (all but the seam edges of
    /// a closed azimuth), with openEMS's single-precision update
    /// ([`OpenEmsSheetBranches`](crate::engine::openems_sheet)).
    pub branches: Vec<super::OpenEmsSheetBranch>,
    /// The branches `(L, R)` of the whole edge of the emulated seam edges
    /// of a closed azimuth (by their `α = 0` node), run by openEMS's seam
    /// update (`OpenemsSeam`).
    pub seam_branches: Vec<super::OpenEmsSheetBranch>,
    /// The emulated edges with the `[ε_r, κ]` they had.
    pub edges: Vec<(Axis, [usize; 3], [f64; 2])>,
    /// Number of edges made PEC (in the first cell of their axis).
    pub pec: usize,
}

/// Whether the sheet `name` contains `p` (drawing units, in the grid's
/// coordinates `cs`) and wins the priority there over every material,
/// metal and other sheet containing it (ties go to the property defined
/// first; lumped elements do not count). `wrap`: on cylindrical grids,
/// whether an `α` past 2π may also match as `α − 2π` ([`alpha_in`]).
fn sheet_at(model: &ContinuousStructure, cs: CoordSystem, name: &str, p: Vec3, wrap: bool) -> bool {
    let is_sheet = |h: &csx::Hit<'_>| matches!(h.property.kind, PropertyKind::ConductingSheet(_));
    model
        .properties_at(p, cs, TypeMask::PHYSICAL)
        .iter()
        .filter(|h| !matches!(h.property.kind, PropertyKind::LumpedElement(_)))
        // A sheet primitive whose α range misses `p` does not contain it.
        .find(|h| {
            !is_sheet(h) || cs == CoordSystem::Cartesian || alpha_in(model, h.primitive, p[1], wrap)
        })
        .is_some_and(|h| is_sheet(h) && h.property.name == name)
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
        if node[d] == 0
            || in_region(d, node[d])
            || in_region(t, node[t])
            || in_region(nrm, node[nrm])
        {
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
        let circuit = edge_circuit(sigma, th, grid.edge_length(e.axis, node) / w, &row);
        for (p, share) in copies {
            let idx = grid.index(p);
            let c = circuit.c_per_dt2 * share * dt * dt;
            let g_e = circuit.g * share;
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
            // A seam copy carries its share of the edge: `share` of the
            // conductance and capacitance, `L/share` and `R/share`.
            let lr = circuit.branches.map(|(l, r)| (l / share, r / share));
            if !seam {
                out.branches.push((e.axis, p, lr));
            } else if p[1] == 0 {
                // openEMS updates the seam edge once, with its whole dual
                // face (`OpenemsSeam`): the branches of the whole edge.
                out.seam_branches.push((e.axis, p, circuit.branches));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::MU0;

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
        // Bin edges measured to one f32 step of f0 = fc (f = f0 + fc), each
        // pair one step apart (design §5.16.13); with the formula in f64
        // one side of every pair is one bin off.
        for (s, t, f_lo, f_hi, k) in [
            (
                4613344.0,
                1.526974870103004e-06,
                1883868758016.0,
                1883868889088.0,
                1,
            ),
            (
                22566374.0,
                2.323430271644611e-05,
                1663447680.0,
                1663447808.0,
                1,
            ),
            (
                27018.404296875,
                7.05397324054502e-05,
                3863073521664.0,
                3863073783808.0,
                9,
            ),
            (
                10959826.0,
                4.6812994696665555e-05,
                72978948096.0,
                72978956288.0,
                12,
            ),
            (
                10243358.0,
                3.557360469130799e-05,
                3465501474816.0,
                3465501736960.0,
                20,
            ),
        ] {
            assert_eq!(bin(s, t, f_lo), k - 1, "{s} {t} {f_lo}");
            assert_eq!(bin(s, t, f_hi), k, "{s} {t} {f_hi}");
        }
    }

    #[test]
    fn capacitance_is_openems() {
        // openEMS's `vi = f32(dt/f32(C))` of a 1 m-cell sheet edge (bin 0,
        // `G·dt` negligible), operator dumps; the single rounding of
        // `dt²/(c·μ0·t)` gives the neighbouring `f32` value in all three.
        let (s, t) = (9.536741259755848e-07, 6.090221230296086e-05);
        for (dt, vi) in [
            (1.8108346533866837e-09, 0x3a763a62),
            (3.5948730289534073e-10, 0x3b9b0a27),
            (1.807569004847039e-09, 0x3a76ac43),
        ] {
            let c = edge_circuit(s, t, 1.0, &TABLE[bin(s, t, 1e6)]).c_per_dt2 * dt * dt;
            assert_eq!(
                (dt / f64::from(c as f32)) as f32,
                f32::from_bits(vi),
                "{dt}"
            );
        }
    }

    #[test]
    fn thin_sheets_conduct_sigma_t() {
        // Bin 0 (t ≤ √80·δ): the DC sheet conductance is σt within 0.12 %.
        let row = TABLE[0];
        let g = row[0] + 1.0 / row[3] + 1.0 / row[5];
        assert!((g - 1.0).abs() < 2e-3, "{g}");
    }
}
