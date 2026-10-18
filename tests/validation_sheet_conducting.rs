//! Analytic validation (design §9) of the conducting-sheet model
//! ([`ems::engine::sheet`]): plane-wave transmission and reflection of a
//! zero-thickness lossy sheet of conductivity `σ` and thickness `t`
//! against the exact slab (`ABCD` chain matrix of the skin-effect line,
//! [`slab_abcd`]):
//!
//! ```text
//! T = 2/(A + B/η0 + C·η0 + D),   Γ = (A + B/η0 − C·η0 − D)/(A + B/η0 + C·η0 + D)
//! ```
//!
//! * thin sheet (`t ≪ δ`), where `T = 2/(2 + η0·σ·t)`;
//! * thick-ish sheet (`t` = 1.8…4.4 δ over the band), where the shielding
//!   is up to 16 dB better than the thin-sheet value;
//! * thick sheet (`t ≫ δ`), where the reflection loss `1 − |Γ| ≈ 2·R_s/η0`
//!   measures the one-sided surface resistance `R_s = 1/(σδ)`.
//!
//! The line is the quasi-1-D TEM line of the other validation tests (PEC /
//! PMC walls, CPML ends), 0.25 mm cells, evaluated over 2–12 GHz
//! (≥ 100 cells per wavelength, so mesh dispersion is negligible).
//! Transmission is the probe behind the sheet relative to the run without
//! sheet; reflection is the reflected wave (total minus incident) relative
//! to the one of a PEC sheet at the same node (`Γ = −1`), which removes the
//! propagation between probe and sheet.

mod common;

use common::*;
use ems::Axis;
use ems::Complex64;
use ems::constants::Z0;
use ems::engine::ConductingSheet;
use ems::engine::sheet::{SheetEdge, model_abcd, slab_abcd};

const DZ: f64 = 0.25e-3;
const CELLS: usize = 480;
const SRC: usize = 40;
const REFL: usize = 140;
const SHEET: usize = 240;
const TRANS: usize = 400;
const STEPS: u64 = 2400;
const F_MAX: f64 = 15e9;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    None,
    Pec,
    Sheet(f64, f64),
}

/// Probe series at `REFL` and `TRANS`, and the time step.
fn run(kind: Kind) -> (f64, Vec<f64>, Vec<f64>) {
    let mut line = Line::new(DZ, CELLS);
    line.src = SRC;
    line.probes = vec![REFL, TRANS];
    if kind == Kind::Pec {
        line.sheets = vec![SHEET];
    }
    let (mut e, hs) = line.build(sine_pulse(7e9, 4e9));
    if let Kind::Sheet(sigma, t) = kind {
        let mut s = ConductingSheet::new(Some(F_MAX));
        for node in line.edges(SHEET) {
            s.push(SheetEdge {
                axis: Axis::X,
                node,
                normal: Axis::Z,
                conductivity: sigma,
                thickness: t,
                upper_share: None,
            });
        }
        let h = s.add_to(&mut e).unwrap();
        assert_eq!(e.operator(&h).edges(), 2);
    }
    e.run(STEPS);
    let v: Vec<Vec<f64>> = hs.iter().map(|h| e.operator(h).values().to_vec()).collect();
    (e.dt(), v[0].clone(), v[1].clone())
}

fn exact(sigma: f64, t: f64, f: f64) -> (Complex64, Complex64) {
    let [a, b, c, d] = slab_abcd(sigma, t, f);
    let den = a + b / Z0 + c * Z0 + d;
    (
        Complex64::new(2.0, 0.0) / den,
        (a + b / Z0 - c * Z0 - d) / den,
    )
}

fn transmission_of([a, b, c, d]: [Complex64; 4]) -> Complex64 {
    Complex64::new(2.0, 0.0) / (a + b / Z0 + c * Z0 + d)
}

fn freqs() -> Vec<f64> {
    linspace(2e9, 12e9, 21)
}

/// Measured transmission spectrum of `kind` against `reference` (no sheet).
fn transmission(reference: &(f64, Vec<f64>, Vec<f64>), kind: Kind) -> Vec<Complex64> {
    let (dt, _, r) = reference;
    let (_, _, t) = run(kind);
    let f = freqs();
    let a = spectrum(r, *dt, &f);
    let b = spectrum(&t, *dt, &f);
    a.iter().zip(&b).map(|(a, b)| *b / *a).collect()
}

#[test]
fn thin_sheet_transmission() {
    let _t = Timer::start("thin_sheet_transmission");
    let reference = run(Kind::None);
    // The record holds the whole transmitted pulse.
    let r = &reference.2;
    let tail = peak(&r[r.len() - 50..]) / peak(r);
    assert!(tail < 1e-5, "{tail}");
    // η0·σ·t = 3.77: T ≈ 0.347 (−9.2 dB); δ ≥ 46 µm ≫ t.
    let (sigma, t) = (1e4, 1e-6);
    let tm = transmission(&reference, Kind::Sheet(sigma, t));
    let thin = 2.0 / (2.0 + Z0 * sigma * t);
    let mut worst: f64 = 0.0;
    for (f, m) in freqs().iter().zip(&tm) {
        let (te, _) = exact(sigma, t, *f);
        assert!(((te.norm() - thin) / thin).abs() < 1e-3);
        worst = worst.max((*m / te - 1.0).norm());
    }
    eprintln!("thin sheet: T = {thin:.4}, max |T_meas/T_exact − 1| = {worst:.2e}");
    assert!(worst < 2e-3, "{worst}");
}

#[test]
fn thick_sheet_transmission() {
    let _t = Timer::start("thick_sheet_transmission");
    let reference = run(Kind::None);
    // σ = 1e6 S/m, t = 20 µm: t/δ from 1.8 (2 GHz) to 4.4 (12 GHz).
    let (sigma, t) = (1e6, 20e-6);
    let tm = transmission(&reference, Kind::Sheet(sigma, t));
    let thin = 2.0 / (2.0 + Z0 * sigma * t);
    let (mut worst, mut worst_phase, mut thin_off): (f64, f64, f64) = (0.0, 0.0, 0.0);
    let mut worst_model: f64 = 0.0;
    for (f, m) in freqs().iter().zip(&tm) {
        let (te, _) = exact(sigma, t, *f);
        let tl = transmission_of(model_abcd(sigma, t, F_MAX, *f));
        worst = worst.max((db(m.norm()) - db(te.norm())).abs());
        worst_phase = worst_phase.max((*m / te).arg().abs());
        worst_model = worst_model.max((*m / tl - 1.0).norm());
        thin_off = thin_off.max((db(te.norm()) - db(thin)).abs());
        eprintln!(
            "{:5.1} GHz: |T| exact {:7.2} dB, ladder {:7.2} dB, measured {:7.2} dB",
            f / 1e9,
            db(te.norm()),
            db(tl.norm()),
            db(m.norm())
        );
    }
    eprintln!(
        "thick sheet: max |T| error {worst:.3} dB, phase {:.2}°, vs the ladder model {worst_model:.1e}; \
         thin-sheet formula off by up to {thin_off:.1} dB",
        worst_phase.to_degrees()
    );
    // The skin effect matters here: the thin-sheet value is far off.
    assert!(thin_off > 10.0, "{thin_off}");
    // The error against the exact slab is the ladder discretisation
    // (O((Δz/δ)²) per skin depth of attenuation); the time stepping
    // reproduces the ladder itself closely.
    assert!(worst < 0.35, "{worst} dB");
    assert!(worst_phase.to_degrees() < 2.0, "{worst_phase}");
    assert!(worst_model < 5e-3, "{worst_model}");
}

#[test]
fn thick_sheet_reflection_surface_resistance() {
    let _t = Timer::start("thick_sheet_reflection_surface_resistance");
    let incident = run(Kind::None).1;
    let (dt, pec, _) = run(Kind::Pec);
    // σ = 1e4 S/m, t = 0.5 mm: t/δ from 4.4 (2 GHz) to 11 (12 GHz);
    // R_s = 0.89–2.2 Ω, 1 − |Γ| ≈ 2·R_s/η0 = 0.5–1.2 %.
    let (sigma, t) = (1e4, 0.5e-3);
    let (_, sheet, _) = run(Kind::Sheet(sigma, t));
    let f = freqs();
    let rp = spectrum(&diff(&pec, &incident), dt, &f);
    let rs = spectrum(&diff(&sheet, &incident), dt, &f);
    let mut worst: f64 = 0.0;
    for ((f, a), b) in f.iter().zip(&rp).zip(&rs) {
        let g = -*b / *a;
        let (_, ge) = exact(sigma, t, *f);
        let (loss, loss_e) = (1.0 - g.norm(), 1.0 - ge.norm());
        let rel = (loss / loss_e - 1.0).abs();
        worst = worst.max(rel);
        eprintln!(
            "{:5.1} GHz: 1 − |Γ| exact {loss_e:.5}, measured {loss:.5} ({:+.2} %)",
            f / 1e9,
            100.0 * (loss / loss_e - 1.0)
        );
        assert!((g / ge - 1.0).norm() < 1e-3, "{f}: {g} vs {ge}");
    }
    eprintln!("thick sheet: max relative error of 1 − |Γ| {worst:.3}");
    assert!(worst < 0.03, "{worst}");
}

/// A partial sheet across a closed PEC cavity (3-D fields, sheet edges with
/// crossing dual edges and borders, graded mesh): the stored energy only
/// decays, and the result does not depend on the thread count.
#[test]
fn sheet_in_cavity_is_passive_and_deterministic() {
    use ems::engine::{Engine, Grid};
    let _t = Timer::start("sheet_in_cavity_is_passive_and_deterministic");
    let run = |threads: usize, sigma: f64, t: f64| {
        let g = Grid::new([
            linspace(0.0, 10e-3, 11),
            (0..=12).map(|i| 1e-3 * (i as f64).powf(1.1)).collect(),
            linspace(0.0, 9e-3, 10),
        ])
        .unwrap();
        let mut e = Engine::builder(g).threads(threads).build().unwrap();
        let mut s = ConductingSheet::new(Some(20e9));
        s.add_rect(Axis::Z, 4, [2, 3, 0], [8, 9, 0], sigma, t);
        let h = s.add_to(&mut e).unwrap();
        {
            let mut f = e.fields_mut();
            f.u.set(Axis::Y, [3, 4, 2], 1.0);
            f.u.set(Axis::X, [5, 6, 6], -0.7);
            f.u.set(Axis::Z, [4, 5, 3], 0.4);
        }
        let mut w = vec![e.energy()];
        for _ in 0..40 {
            e.run(50);
            w.push(e.energy());
        }
        let edges = e.operator(&h).edges();
        (w, e.u_component(Axis::X), e.i_component(Axis::Y), edges)
    };
    for (sigma, t) in [(1e4, 1e-6), (1e6, 20e-6), (5.8e7, 35e-6)] {
        let (w, ux, iy, edges) = run(1, sigma, t);
        assert_eq!(edges, 6 * 7 + 6 * 7);
        let w0 = w[0];
        for p in w.windows(2) {
            assert!(
                p[1] <= p[0] * (1.0 + 1e-6) + 1e-12 * w0,
                "{sigma} {t}: {w:?}"
            );
        }
        assert!(w.last().unwrap() < &w0, "{sigma} {t}: no loss");
        let (w4, ux4, iy4, _) = run(4, sigma, t);
        assert_eq!(w, w4);
        assert_eq!(ux, ux4);
        assert_eq!(iy, iy4);
        eprintln!(
            "σ = {sigma:e}, t = {t:e}: energy {:.3e} → {:.3e}",
            w0,
            w.last().unwrap()
        );
    }
}

/// The same thick-ish sheet as [`thick_sheet_transmission`], built as a
/// csx model (`ConductingSheet` box on a mesh plane) and compiled: the
/// compile step finds the sheet normal and installs the operator.
#[test]
fn csx_conducting_sheet_transmission() {
    use csx::primitives::BoxShape;
    use csx::{ContinuousStructure, Primitive, Property};
    use ems::compile::{CompileOptions, compile};
    use ems::engine::{EdgeSource, Probe};
    use ems::{Boundaries, Boundary, Face, Fdtd};
    let _t = Timer::start("csx_conducting_sheet_transmission");
    let (sigma, t) = (1e6, 20e-6);
    let run = |with_sheet: bool| {
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        m.grid.set_lines(0, [0.0, 5.0]).unwrap();
        m.grid.set_lines(1, [0.0, 5.0]).unwrap();
        m.grid
            .set_lines(2, (0..=CELLS).map(|q| q as f64 * DZ * 1e3))
            .unwrap();
        if with_sheet {
            let z = SHEET as f64 * DZ * 1e3;
            let id = m.add_property(Property::conducting_sheet("sheet", sigma, t));
            m.add_primitive(
                id,
                Primitive::new(BoxShape::new([0.0, 0.0, z], [5.0, 5.0, z])),
            )
            .unwrap();
        }
        let fdtd = Fdtd::new().threads(2).max_frequency(F_MAX).boundaries(
            Boundaries::all(Boundary::Pec)
                .with(Face::YMin, Boundary::Pmc)
                .with(Face::YMax, Boundary::Pmc)
                .with(Face::ZMin, Boundary::pml(8))
                .with(Face::ZMax, Boundary::pml(8)),
        );
        let c = compile(&m, &fdtd, &CompileOptions::default()).unwrap();
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        // Two E_x and two E_y edges (the latter on the PEC walls).
        assert_eq!(c.sheets.len(), if with_sheet { 4 } else { 0 });
        let (mut e, h) = c.build().unwrap();
        if let Some(h) = h.sheets {
            assert_eq!(e.operator(&h).edges(), 2);
        }
        assert_eq!(h.sheets.is_some(), with_sheet);
        let line = Line::new(DZ, CELLS);
        let mut src = EdgeSource::new(sine_pulse(7e9, 4e9));
        for p in line.edges(SRC) {
            src = src.edge(Axis::X, p, 1.0);
        }
        e.add_operator(src);
        let probe = e.add_operator(Probe::voltage(Axis::X, line.edges(TRANS)[0]));
        e.run(STEPS);
        (e.dt(), e.operator(&probe).values().to_vec())
    };
    let (dt, r) = run(false);
    let (_, m) = run(true);
    let f = freqs();
    let (a, b) = (spectrum(&r, dt, &f), spectrum(&m, dt, &f));
    let mut worst: f64 = 0.0;
    for ((f, a), b) in f.iter().zip(&a).zip(&b) {
        let (te, _) = exact(sigma, t, *f);
        worst = worst.max((db((*b / *a).norm()) - db(te.norm())).abs());
    }
    eprintln!("csx sheet: max |T| error {worst:.3} dB");
    assert!(worst < 0.35, "{worst}");
}
