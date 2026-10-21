//! openEMS's `ConductingSheet` model (files only,
//! [`Fdtd::openems_run`](crate::Fdtd::openems_run), Cartesian grids).
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
//!   The medium around the sheet does not enter (the vacuum capacitance of
//!   the cell is replaced, not added).
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
//! * **Width.** `w` sums half the widths (along the sheet, across the
//!   edge) of the edge's two neighbouring sheet cells that lie in the
//!   sheet (cell centre in a planar primitive of the sheet that wins the
//!   priority there). A cell whose lower corner lies on the first mesh
//!   line of either in-plane axis does not count (openEMS's own quirk: the
//!   first row and column of a sheet that reaches the lower mesh boundary
//!   are left out, so a sheet across a two-cell TEM line carries half its
//!   current); an edge without counted cells is PEC.
//! * **Priorities.** A material or metal of higher priority at the edge
//!   midpoint removes the sheet edge (it keeps its material); the sheet
//!   wins ties. Edges on the outer mesh planes are left to the boundary.
//!
//! The time step openEMS uses ignores the sheets; ems resolves it before
//! the sheet capacitances are set and pins it.

use csx::ContinuousStructure;
use csx::properties::{PropertyKind, TypeMask};
use csx::{CoordSystem, Vec3};

use crate::Result;
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};
use crate::engine::dispersive::{EdgePole, Field, Pole};
use crate::engine::{EdgeMaterials, Grid, SheetEdge, TimeStep};
use crate::fdtd::Fdtd;
use crate::waveform::Waveform;

/// openEMS's sheet model per bin `k` (see the [module docs](self)):
/// `(g, c, l_1, r_1, l_2, r_2)` per square, `G = g·σt`,
/// `C = dt²/(c·μ0·t)`, `L_b = l_b·μ0·t`, `R_b = r_b/(σt)`. Measured with
/// openEMS v0.37 (σ = 1e6 S/m, 1 mm cells, `dt` = 1.8 ps, 1500 steps;
/// reproduced with other `σ`, `t` and `dt` to 7 digits).
#[rustfmt::skip]
pub(crate) const TABLE: [[f64; 6]; 30] = [
    [0.13708436, 0.022224575, 0.11928209, 1.3027544, 0.099069049, 10.62245],
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
    /// operator): those of cylindrical grids.
    pub rest: Vec<SheetEdge>,
    /// Branch poles of the emulated edges.
    pub poles: Vec<EdgePole>,
    /// The emulated edges with the `[ε_r, κ]` they had.
    pub edges: Vec<(Axis, [usize; 3], [f64; 2])>,
    /// Number of edges made PEC (no counted sheet cell).
    pub pec: usize,
}

/// Does a primitive of the sheet `name` contain `p` (drawing units) and
/// win the priority there (ties go to the sheet)?
fn sheet_wins(model: &ContinuousStructure, name: &str, p: Vec3) -> bool {
    let hits = model.properties_at(p, CoordSystem::Cartesian, TypeMask::PHYSICAL);
    let Some(top) = hits.first() else {
        return false;
    };
    let prio = top.primitive.priority;
    // The sheet wins ties.
    hits.iter()
        .take_while(|h| h.primitive.priority == prio)
        .any(|h| {
            h.property.name == name && matches!(h.property.kind, PropertyKind::ConductingSheet(_))
        })
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
    if grid.is_cylindrical() {
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
    let lines: [&[f64]; 3] = core::array::from_fn(|a| model.grid.lines(a));
    for (e, name) in edges.into_iter().zip(names) {
        let d = e.axis.index();
        let nrm = e.normal.index();
        let t = 3 - d - nrm;
        let node = e.node;
        // Outer mesh planes: the boundary decides.
        if (0..3).any(|a| a != d && (node[a] == 0 || node[a] + 1 == n[a])) {
            continue;
        }
        let mut mid = Vec3::new(lines[0][node[0]], lines[1][node[1]], lines[2][node[2]]);
        mid.0[d] = 0.5 * (lines[d][node[d]] + lines[d][node[d] + 1]);
        if !sheet_wins(model, &name, mid) {
            continue;
        }
        // Counted neighbouring cells along t.
        let mut w = 0.0;
        if node[d] >= 1 {
            for c in [node[t].checked_sub(1), Some(node[t])]
                .into_iter()
                .flatten()
            {
                if c < 1 || c + 1 >= n[t] {
                    continue;
                }
                let mut p = mid;
                p.0[t] = 0.5 * (lines[t][c] + lines[t][c + 1]);
                if sheet_wins(model, &name, p) {
                    w += 0.5 * grid.delta(Axis::ALL[t], c);
                }
            }
        }
        let idx = grid.index(node);
        if w == 0.0 {
            materials.pec_mask_mut(e.axis)[idx] = true;
            out.pec += 1;
            continue;
        }
        let (sigma, th) = (e.conductivity, e.thickness);
        let row = TABLE[bin(sigma, th, f)];
        let l = grid.delta(e.axis, node[d]);
        let sq = w / l;
        let (st, lt) = (sigma * th, MU0 * th);
        let c = dt * dt * sq / (row[1] * lt);
        let g_e = row[0] * st * sq;
        let (ab, ac) = e.axis.others();
        let geo = grid.dual_delta(ab, node[ab.index()]) * grid.dual_delta(ac, node[ac.index()]) / l;
        out.edges.push((
            e.axis,
            node,
            [materials.eps_r(e.axis)[idx], materials.kappa(e.axis)[idx]],
        ));
        materials.eps_r_mut(e.axis)[idx] = c / (EPS0 * geo);
        materials.kappa_mut(e.axis)[idx] = g_e / geo;
        let m = EPS0 * geo;
        for (lb, rb) in [(row[2], row[3]), (row[4], row[5])] {
            let (lb, rb) = (lb * lt / sq, rb / (st * sq));
            out.poles.push(EdgePole {
                field: Field::Electric,
                axis: e.axis,
                node,
                pole: Pole::drude((1.0 / (m * lb)).sqrt(), rb / lb),
            });
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
