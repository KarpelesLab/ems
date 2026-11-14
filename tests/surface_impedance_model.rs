//! The surface impedance through the model and the compile step
//! ([`CompileOptions::with_lossy_metal`],
//! [`SheetModel::SurfaceImpedance`], [`CompileOptions::with_sheet_roughness`]):
//!
//! * a plate cavity drawn as csx `Metal` boxes made lossy copper,
//!   `Q = h/δ`;
//! * a `ConductingSheet` across a TEM line with the rational model against
//!   the exact slab, and its roughness factor on the reflection loss;
//! * sheets and lossy metal reaching into the CPML compile without the
//!   ladder's warning, and the defaults leave a model as it was.

mod common;

use common::*;
use csx::primitives::BoxShape;
use csx::properties::{Excitation, ProbeBox, ProbeType};
use csx::{ContinuousStructure, Primitive, Property};
use ems::compile::{CompileOptions, SheetModel, compile};
use ems::constants::{C0, Z0};
use ems::engine::sheet::{skin_depth, slab_abcd};
use ems::engine::surface::{Conductor, Roughness};
use ems::engine::{EdgeSource, Probe};
use ems::{Axis, Boundaries, Boundary, Complex64, Face, Fdtd, Simulation, Waveform};

/// `(f, Q)` of the TM110 ring-down of a 40 × 24 mm box between two metal
/// plates 0.5 mm apart, the plates csx `Metal` boxes one cell thick.
fn plate_cavity(opts: &CompileOptions) -> (f64, f64, Simulation) {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, linspace(0.0, 40.0, 21)).unwrap();
    m.grid.set_lines(1, linspace(0.0, 24.0, 13)).unwrap();
    m.grid.set_lines(2, linspace(0.0, 1.0, 5)).unwrap();
    let cu = m.add_property(Property::metal("plates"));
    for (z0, z1) in [(0.0, 0.25), (0.75, 1.0)] {
        m.add_primitive(
            cu,
            Primitive::new(BoxShape::new([0.0, 0.0, z0], [40.0, 24.0, z1])),
        )
        .unwrap();
    }
    let ex = m.add_property(Property::excitation("ex", Excitation::new([0.0, 0.0, 1.0])));
    m.add_primitive(
        ex,
        Primitive::new(BoxShape::new([20.0, 12.0, 0.25], [20.0, 12.0, 0.75])),
    )
    .unwrap();
    let pr = m.add_property(Property::probe("v", ProbeBox::new(ProbeType::Voltage)));
    m.add_primitive(
        pr,
        Primitive::new(BoxShape::new([14.0, 8.0, 0.25], [14.0, 8.0, 0.75])),
    )
    .unwrap();
    let f0 = 0.5 * C0 * (1.0f64 / 40e-3).hypot(1.0 / 24e-3);
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(f0, 1.5e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .max_timesteps(12_000)
        .no_end_criteria()
        .threads(1);
    let mut sim = Simulation::with_options(&m, &fdtd, opts).unwrap();
    let res = sim.run().unwrap();
    let p = res.probe("v").unwrap();
    let dt = p.times()[1] - p.times()[0];
    let v = p.scalar().unwrap();
    let tail = &v[(3e-9 / dt) as usize..];
    // Resonance, then the decay of the energy per period.
    let spec = |f: f64| spectrum(tail, dt, &[f])[0].norm();
    let (mut lo, mut hi) = (0.9 * f0, 1.1 * f0);
    for _ in 0..60 {
        let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if spec(a) < spec(b) {
            lo = a;
        } else {
            hi = b;
        }
    }
    let f = 0.5 * (lo + hi);
    let per = 1.0 / (f * dt);
    let n = (tail.len() as f64 / per).floor() as usize;
    let pts: Vec<(f64, f64)> = (0..n)
        .map(|p| {
            let (a, b) = ((p as f64 * per) as usize, ((p + 1) as f64 * per) as usize);
            let e: f64 = tail[a..b].iter().map(|x| x * x).sum::<f64>() / (b - a) as f64;
            ((p as f64 + 0.5) * per * dt, e.ln())
        })
        .collect();
    let k = pts.len() as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (mx, my) = (sx / k, sy / k);
    let slope = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>()
        / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
    (f, -core::f64::consts::TAU * f / slope, sim)
}

#[test]
fn lossy_metal_plate_cavity_q() {
    let _t = Timer::start("lossy_metal_plate_cavity_q");
    let h = 0.5e-3;
    for r in [
        Roughness::Smooth,
        Roughness::Huray {
            radius: 0.5e-6,
            ratio: 2.0,
        },
    ] {
        let opts = CompileOptions::default()
            .with_lossy_metal("plates", Conductor::new(5.8e7).with_roughness(r));
        let (f, q, sim) = plate_cavity(&opts);
        // (The 2 mm cells are λ/17 at the top of the pulse's band.)
        let w: Vec<_> = sim
            .warnings()
            .iter()
            .filter(|w| !w.starts_with("mesh warning"))
            .collect();
        assert!(w.is_empty(), "{w:?}");
        let op = sim
            .engine()
            .operator(sim.boundaries().surface.as_ref().unwrap());
        // The inner faces of the plates, less the edges on the walls.
        assert_eq!(op.solid_edges(), 2 * (20 * 11 + 19 * 12));
        assert_eq!((op.thin(), op.skipped()), (0, 0));
        let want = h / (skin_depth(5.8e7, f) * r.factor(skin_depth(5.8e7, f)));
        eprintln!(
            "{r:?}: f {:.4} GHz, Q {q:.1}, h/(δK) {want:.1} ({:+.2} %)",
            f / 1e9,
            100.0 * (q / want - 1.0)
        );
        assert!((q / want - 1.0).abs() < 0.015, "{r:?}: Q {q} vs {want}");
    }
}

#[test]
fn metal_stays_perfect_by_default() {
    // Without the option the plates are PEC: no operator, no decay.
    let (_, q, sim) = plate_cavity(&CompileOptions::default());
    assert!(sim.boundaries().surface.is_none());
    assert!(!(0.0..1e5).contains(&q), "Q {q}");
    // A name that is not a Metal property is reported.
    let c = compile(
        &ContinuousStructure::new_with_unit_grid(),
        &Fdtd::new().max_frequency(1e9),
        &CompileOptions::default().with_lossy_metal("copper", Conductor::new(5.8e7)),
    )
    .unwrap();
    assert!(
        c.warnings
            .iter()
            .any(|w| w.contains("lossy metal \"copper\"")),
        "{:?}",
        c.warnings
    );
}

/// Small helper trait: a 2 × 2 × 2-cell model.
trait UnitGrid {
    fn new_with_unit_grid() -> Self;
}

impl UnitGrid for ContinuousStructure {
    fn new_with_unit_grid() -> Self {
        let mut m = ContinuousStructure::new();
        for a in 0..3 {
            m.grid.set_lines(a, [0.0, 1.0, 2.0]).unwrap();
        }
        m
    }
}

const DZ: f64 = 0.25e-3;
const CELLS: usize = 480;
const SRC: usize = 40;
const REFL: usize = 140;
const SHEET: usize = 240;
const TRANS: usize = 400;
const STEPS: u64 = 2400;

#[derive(Clone, Copy, PartialEq)]
enum Sheet {
    None,
    Pec,
    Lossy(f64, f64),
}

/// The TEM line of `validation_sheet_conducting.rs` as a csx model with a
/// sheet across it: the time step and the E series before and behind it.
fn line(sheet: Sheet, opts: &CompileOptions) -> (f64, Vec<f64>, Vec<f64>) {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, [0.0, 5.0]).unwrap();
    m.grid.set_lines(1, [0.0, 5.0]).unwrap();
    m.grid
        .set_lines(2, (0..=CELLS).map(|q| q as f64 * DZ * 1e3))
        .unwrap();
    let z = SHEET as f64 * DZ * 1e3;
    let prop = match sheet {
        Sheet::None => None,
        Sheet::Pec => Some(Property::metal("sheet")),
        Sheet::Lossy(sigma, t) => Some(Property::conducting_sheet("sheet", sigma, t)),
    };
    if let Some(p) = prop {
        let id = m.add_property(p);
        m.add_primitive(
            id,
            Primitive::new(BoxShape::new([0.0, 0.0, z], [5.0, 5.0, z])),
        )
        .unwrap();
    }
    let fdtd = Fdtd::new().threads(2).max_frequency(15e9).boundaries(
        Boundaries::all(Boundary::Pec)
            .with(Face::YMin, Boundary::Pmc)
            .with(Face::YMax, Boundary::Pmc)
            .with(Face::ZMin, Boundary::pml(8))
            .with(Face::ZMax, Boundary::pml(8)),
    );
    let c = compile(&m, &fdtd, opts).unwrap();
    assert!(c.warnings.is_empty(), "{:?}", c.warnings);
    if matches!(sheet, Sheet::Lossy(..)) {
        let rational = opts.sheet_model == SheetModel::SurfaceImpedance;
        assert_eq!(c.surface_sheets.len(), if rational { 4 } else { 0 });
        assert_eq!(c.sheets.len(), if rational { 0 } else { 4 });
    }
    let (mut e, h) = c.build().unwrap();
    assert_eq!(
        h.surface.is_some(),
        matches!(sheet, Sheet::Lossy(..)) && opts.sheet_model == SheetModel::SurfaceImpedance
    );
    if let Some(h) = h.surface {
        assert_eq!(e.operator(&h).sheet_edges(), 2);
    }
    let l = Line::new(DZ, CELLS);
    let mut src = EdgeSource::new(sine_pulse(7e9, 4e9));
    for p in l.edges(SRC) {
        src = src.edge(Axis::X, p, 1.0);
    }
    e.add_operator(src);
    let pr = e.add_operator(Probe::voltage(Axis::X, l.edges(REFL)[0]));
    let pt = e.add_operator(Probe::voltage(Axis::X, l.edges(TRANS)[0]));
    e.run(STEPS);
    (
        e.dt(),
        e.operator(&pr).values().to_vec(),
        e.operator(&pt).values().to_vec(),
    )
}

/// `(T, Γ)` of the exact slab.
fn exact(sigma: f64, t: f64, f: f64) -> (Complex64, Complex64) {
    let [a, b, c, d] = slab_abcd(sigma, t, f);
    let den = a + b / Z0 + c * Z0 + d;
    (
        Complex64::new(2.0, 0.0) / den,
        (a + b / Z0 - c * Z0 - d) / den,
    )
}

#[test]
fn conducting_sheet_with_the_rational_model() {
    let _t = Timer::start("conducting_sheet_with_the_rational_model");
    let opts = CompileOptions::default().with_sheet_model(SheetModel::SurfaceImpedance);
    let f = linspace(2e9, 12e9, 21);
    let (dt, incident, reference) = line(Sheet::None, &opts);
    // Thick-ish sheet (t/δ = 1.8…4.4): transmission against the slab.
    let (sigma, t) = (1e6, 20e-6);
    let (_, _, tr) = line(Sheet::Lossy(sigma, t), &opts);
    let (a, b) = (spectrum(&reference, dt, &f), spectrum(&tr, dt, &f));
    let mut worst: f64 = 0.0;
    for ((f, a), b) in f.iter().zip(&a).zip(&b) {
        worst = worst.max((db((*b / *a).norm()) - db(exact(sigma, t, *f).0.norm())).abs());
    }
    eprintln!("rational csx sheet: max |T| error {worst:.4} dB");
    assert!(worst < 0.05, "{worst}");

    // Thick sheet (t/δ = 4.4…11): the reflection loss 2·R_s/η0, and
    // K times that with roughness.
    let (sigma, t) = (1e4, 0.5e-3);
    let (_, pec, _) = line(Sheet::Pec, &opts);
    let rp = spectrum(&diff(&pec, &incident), dt, &f);
    let loss = |opts: &CompileOptions| -> Vec<f64> {
        let (_, v, _) = line(Sheet::Lossy(sigma, t), opts);
        let rs = spectrum(&diff(&v, &incident), dt, &f);
        rp.iter()
            .zip(&rs)
            .map(|(a, b)| 1.0 - (*b / *a).norm())
            .collect()
    };
    let smooth = loss(&opts);
    let rough = Roughness::HammerstadJensen { rms: 60e-6 };
    let rough_loss = loss(&opts.clone().with_sheet_roughness("sheet", rough));
    let (mut ws, mut wr) = (0.0f64, 0.0f64);
    for ((f, s), r) in f.iter().zip(&smooth).zip(&rough_loss) {
        let want = 1.0 - exact(sigma, t, *f).1.norm();
        let k = rough.factor(skin_depth(sigma, *f));
        ws = ws.max((s / want - 1.0).abs());
        wr = wr.max((r / (k * want) - 1.0).abs());
    }
    eprintln!("1 − |Γ|: smooth within {ws:.4}, rough (K = 1.4…1.9) within {wr:.4} of K·2R_s/η0");
    assert!(ws < 0.02 && wr < 0.04, "{ws} {wr}");
}

/// A sheet strip over a metal ground, both running into the CPML on the
/// x faces.
fn strip_model() -> (ContinuousStructure, Fdtd) {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, linspace(0.0, 20.0, 41)).unwrap();
    m.grid.set_lines(1, linspace(0.0, 8.0, 17)).unwrap();
    m.grid.set_lines(2, linspace(0.0, 3.0, 7)).unwrap();
    let gnd = m.add_property(Property::metal("ground"));
    m.add_primitive(
        gnd,
        Primitive::new(BoxShape::new([0.0, 0.0, 0.0], [20.0, 8.0, 0.5])),
    )
    .unwrap();
    let strip = m.add_property(Property::conducting_sheet("strip", 5.8e7, 35e-6));
    m.add_primitive(
        strip,
        Primitive::new(BoxShape::new([0.0, 3.0, 1.0], [20.0, 5.0, 1.0])),
    )
    .unwrap();
    let fdtd = Fdtd::new().max_frequency(10e9).boundaries(
        Boundaries::all(Boundary::Pec)
            .with(Face::XMin, Boundary::pml(8))
            .with(Face::XMax, Boundary::pml(8)),
    );
    (m, fdtd)
}

#[test]
fn copper_through_the_cpml_compiles_without_the_ladder_warning() {
    let (m, fdtd) = strip_model();
    let pml = |w: &[String]| w.iter().any(|w| w.contains("inside the PML"));
    // As before: the ladder, with its warning; no surface operator.
    let c = compile(&m, &fdtd, &CompileOptions::default()).unwrap();
    assert!(pml(&c.warnings), "{:?}", c.warnings);
    assert!(c.solid_edges.is_empty() && c.surface_sheets.is_empty());
    assert!(!c.sheets.is_empty());
    assert!(c.build().unwrap().1.surface.is_none());
    // The surface impedance on both.
    let opts = CompileOptions::default()
        .with_sheet_model(SheetModel::SurfaceImpedance)
        .with_lossy_metal("ground", Conductor::new(5.8e7));
    let c = compile(&m, &fdtd, &opts).unwrap();
    assert!(!pml(&c.warnings), "{:?}", c.warnings);
    assert!(c.sheets.is_empty() && !c.surface_sheets.is_empty());
    let (mut e, h) = c.build().unwrap();
    let op = e.operator(h.surface.as_ref().unwrap());
    // The ground's upper face, less the edges on the outer walls: 40 × 15
    // x edges and 39 × 16 y edges; the strip: 40 × 5 and 4 × 39 (its ends
    // lie on the walls).
    assert_eq!(op.solid_edges(), 40 * 15 + 39 * 16);
    assert_eq!(op.sheet_edges(), 40 * 5 + 4 * 39);
    assert!(op.stretched() > 0);
    assert_eq!(op.thin(), 0);
    e.run(20);
}
