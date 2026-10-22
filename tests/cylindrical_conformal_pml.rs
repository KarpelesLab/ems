//! Conformal PEC walls inside the cylindrical PML of `ρ_max` (design
//! §5.13 "Absorbing layers", §5.15.2): a radial wedge guide whose walls
//! (radial planes between two α lines, solid metal beyond) run out into
//! the layer, carrying the radial TEM wave (`E_α`, `H_z`, PMC z plates).
//! The walls cut every α edge next to them and the z faces along them;
//! the small faces in the layer are merged, and the merge groups apply the
//! layer's stretch to their members themselves (their kernel `db` is 0, so
//! the layer's own correction vanishes there). Reflection against a domain
//! 560 cells longer, from which nothing returns within the gate.

mod common;

use csx::primitives::{BoxShape, LinPoly};
use csx::{ContinuousStructure, CoordSystem, Primitive, Property, Transform};
use ems::compile::{CompileOptions, compile};
use ems::engine::{ConformalReport, EdgeSource, Engine, Probe};
use ems::{Axis, Boundary, Face, Fdtd};

const DR: f64 = 0.5; // mm
const R0: f64 = 20.0; // inner radius, mm
const NA: usize = 32;
const A0: f64 = 0.2; // half opening of the sector, rad
/// Walls: 0.37 of a cell above α = 0.1 and 0.23 of a cell below −0.1.
const DA: f64 = 2.0 * A0 / NA as f64;
const WALLS: (f64, f64) = (-0.1 - 0.23 * DA, 0.1 + 0.37 * DA);

fn lines(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..=n)
        .map(|i| lo + (hi - lo) * i as f64 / n as f64)
        .collect()
}

/// The wedge on the sector `R0 … R0 + nr·DR`, `|α| ≤ A0`, PML(8) on
/// `ρ_max` if `pml`; `zpml`: `None` for one 1 mm z cell between PMC
/// plates, `Some(nz)` for `nz` z cells with PML(4) on both z faces (corners
/// with the ρ layer). A soft `E_α` source on every uncut α edge of ρ index
/// 8 (mid-height). `sheet`: the walls are zero-thickness radial plates
/// (conformal sheets) instead of solid metal.
fn wedge_engine(
    nr: usize,
    zpml: Option<usize>,
    conformal: bool,
    pml: bool,
    threads: usize,
    sheet: bool,
) -> (Engine, Option<ConformalReport>) {
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    let rho = lines(R0, R0 + nr as f64 * DR, nr);
    let alpha = lines(-A0, A0, NA);
    m.grid.set_lines(0, rho.iter().copied()).unwrap();
    m.grid.set_lines(1, alpha.iter().copied()).unwrap();
    let nz = zpml.unwrap_or(1);
    m.grid.set_lines(2, lines(0.0, nz as f64, nz)).unwrap();
    let metal = m.add_property(Property::metal("walls"));
    let far = 2.0 * rho[nr];
    let at = |a: f64| (far * a.cos(), far * a.sin());
    for (a0, a1) in [(WALLS.1, A0 + 0.1), (-A0 - 0.1, WALLS.0)] {
        if sheet {
            let wall = if a0 == WALLS.1 { a0 } else { a1 };
            let mut t = Transform::new();
            t.rotate_axis(2, wall);
            m.add_primitive(
                metal,
                Primitive::new(BoxShape::new([0.0, 0.0, -1.0], [far, 0.0, nz as f64 + 2.0]))
                    .with_transform(t)
                    .with_coord_system(CoordSystem::Cartesian),
            )
            .unwrap();
            continue;
        }
        m.add_primitive(
            metal,
            Primitive::new(LinPoly::new(
                2,
                -1.0,
                nz as f64 + 2.0,
                vec![(0.0, 0.0), at(a0), at(a1)],
            ))
            .with_coord_system(CoordSystem::Cartesian),
        )
        .unwrap();
    }
    let zb = if zpml.is_some() {
        Boundary::pml(4)
    } else {
        Boundary::Pmc
    };
    let mut fdtd = Fdtd::new()
        .threads(threads)
        .boundary(Face::ZMin, zb)
        .boundary(Face::ZMax, zb);
    if pml {
        fdtd = fdtd.boundary(Face::XMax, Boundary::pml(8));
    }
    let c = compile(
        &m,
        &fdtd,
        &CompileOptions::default().with_conformal(conformal),
    )
    .unwrap();
    let (mut e, h) = c.build().unwrap();
    let is = 8;
    let mut src = EdgeSource::new(common::sine_pulse(12e9, 8e9));
    for j in 0..NA {
        if alpha[j] > WALLS.0 && alpha[j + 1] < WALLS.1 {
            src = src.edge(Axis::Y, [is, j, nz / 2], 1.0);
        }
    }
    e.add_operator(src);
    (e, h.conformal)
}

/// [`wedge_engine`] between PMC plates: the `E_α` voltages at ρ index
/// `probe` on three α lines, concatenated, over 1 ns, and the conformal
/// report.
fn wedge(
    nr: usize,
    probe: usize,
    conformal: bool,
    pml: bool,
    threads: usize,
    sheet: bool,
) -> (Vec<f64>, Option<ConformalReport>) {
    let (mut e, rep) = wedge_engine(nr, None, conformal, pml, threads, sheet);
    let mid = NA / 2;
    let probes: Vec<_> = [mid, mid + 4, mid - 5]
        .map(|j| e.add_operator(Probe::voltage(Axis::Y, [probe, j, 0])))
        .into();
    e.run((1.0e-9 / e.dt()).ceil() as u64);
    let v = probes
        .iter()
        .flat_map(|p| e.operator(p).values().to_vec())
        .collect();
    (v, rep)
}

/// Largest |difference| relative to the largest |reference| value, dB.
fn rel_db(test: &[f64], reference: &[f64]) -> f64 {
    let n = test.len().min(reference.len());
    let d = (0..n).fold(0.0f64, |m, q| m.max((test[q] - reference[q]).abs()));
    let r = reference[..n].iter().fold(0.0f64, |m, x| m.max(x.abs()));
    20.0 * (d / r).log10()
}

/// Layer from ρ = 50 mm (index 60), probe at 45 mm (index 50).
fn reflection(conformal: bool, sheet: bool) -> (f64, Option<ConformalReport>) {
    let (inner, probe) = (60, 50);
    let (reference, _) = wedge(inner + 560, probe, conformal, false, 4, sheet);
    let (v, rep) = wedge(inner + 8, probe, conformal, true, 4, sheet);
    (rel_db(&v, &reference), rep)
}

#[test]
fn conformal_walls_in_the_rho_max_pml() {
    let (stair, _) = reflection(false, false);
    let (conf, rep) = reflection(true, false);
    let rep = rep.unwrap();
    eprintln!("wedge into the rho_max PML: staircase {stair:.1} dB, conformal {conf:.1} dB");
    eprintln!("{rep:?}");
    assert!(rep.verified && rep.merged > 0, "{rep:?}");
    assert!(conf < -60.0, "{conf}");
}

#[test]
fn conformal_rho_max_pml_threads_bit_identical() {
    let (a, _) = wedge(68, 50, true, true, 1, false);
    let (b, _) = wedge(68, 50, true, true, 3, false);
    assert_eq!(a, b);
    let (a, _) = wedge(68, 50, true, true, 1, true);
    let (b, _) = wedge(68, 50, true, true, 3, true);
    assert_eq!(a, b);
}

#[test]
fn conformal_sheet_walls_in_the_rho_max_pml() {
    // The walls as radial conformal sheets: the pieces of the cut α edges
    // and the regions of the cut faces in the layer take its stretch from
    // the sheet operator. (Unstretched, the run diverged.)
    let (conf, rep) = reflection(true, true);
    let rep = rep.unwrap();
    eprintln!("sheet wedge into the rho_max PML: {conf:.1} dB ({rep:?})");
    assert!(rep.verified && rep.sheet_edges > 0, "{rep:?}");
    assert!(conf < -60.0, "{conf}");
}

#[test]
fn conformal_walls_in_pml_corners_decay() {
    // 12 z cells with PML(4) on both z faces: the ρ_max and z layers
    // overlap where the walls run, the merged faces there take the z
    // stretch inside the ρ layer's filter (as the layer does). The energy
    // decays like the staircase run's (−77 dB at the end) without late
    // growth. Before: merged faces unstretched in the ρ layer diverged;
    // with hard merges (no deviation currents) the energy grew
    // exponentially from ~−95 dB after 7000 steps. The same with sheet
    // walls (pieces and regions stretched by the sheet operator), over
    // 30 000 steps.
    for (sheet, blocks) in [(false, 30), (true, 60)] {
        let (mut e, rep) = wedge_engine(68, Some(12), true, true, 4, sheet);
        assert!(rep.unwrap().verified);
        let mut w = Vec::new();
        for _ in 0..blocks {
            e.run(500);
            w.push(e.energy());
        }
        let peak = w.iter().fold(0f64, |m, &x| m.max(x));
        let late = w[20..].iter().fold(0f64, |m, &x| m.max(x));
        eprintln!(
            "corners (sheet walls {sheet}): late max {:.1} dB, end {:.1} dB",
            10.0 * (late / peak).log10(),
            10.0 * (w[blocks - 1] / peak).log10()
        );
        assert!(late < 1e-6 * peak, "{w:?}");
        assert!(w[blocks - 1] <= w[20], "{w:?}");
    }
}
