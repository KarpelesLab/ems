//! Analytic validation (design §9): oblique-incidence reflection of the
//! CPML (< −40 dB at 45°) and of the first-order Mur boundary (reported;
//! theory `|R(θ)| = (1 − cos θ)/(1 + cos θ)`, 0.17 ≈ −15 dB at 45°).
//!
//! 2-D TM_z setup: one cell in z between PEC faces (E_z, H_x, H_y uniform
//! in z, an exact 2-D solution), 1 mm cells in x and y. A soft E_z line
//! source radiates a cylindrical pulse towards the x_max face under test,
//! `L` cells away. A probe `a` cells in front of that face and `L + a`
//! cells off-axis receives the specular reflection at exactly 45°
//! (image-source geometry); a second probe on the axis sees normal
//! incidence. The reference run extends the domain beyond x_max, and all
//! PEC walls of both runs are far enough that their echoes arrive after the
//! comparison window (Taflove & Hagness ch. 7, numerical reflection
//! experiments). Reflection = max |test − reference| / max |reference| at
//! the probe, within the window.

mod common;

use common::*;
use ems::constants::C0;
use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::boundary::mur::MurBoundary;
use ems::engine::{EdgeSource, Engine, Grid, Probe};
use ems::{Axis, Cpml, Face};

const D: f64 = 1e-3;
/// Source to face under test (cells).
const L: usize = 25;
/// Probe to face under test (cells).
const A: usize = 5;
/// Source position: cells to the x_min / y_min walls.
const SX: usize = 45;
const SY: usize = 50;
/// Cells from the source to the y_max wall.
const YMAX: usize = 65;
/// Reference: cells from the source to its x_max wall.
const XREF: usize = 60;
const TAU: f64 = 15e-12;

#[derive(Clone, Copy, PartialEq)]
enum Wall {
    /// Reference: plain PEC wall far away.
    Reference,
    /// PEC wall at the face (gating sanity check).
    Pec,
    Cpml(Cpml),
    Mur,
}

/// Probe series at the 45° and the normal-incidence probe.
fn run(face: Wall, steps: u64) -> [Vec<f64>; 2] {
    let nx = match face {
        Wall::Reference => SX + XREF,
        Wall::Cpml(p) => SX + L + p.cells,
        _ => SX + L,
    };
    let x: Vec<f64> = (0..=nx).map(|q| q as f64 * D).collect();
    let y: Vec<f64> = (0..=SY + YMAX).map(|q| q as f64 * D).collect();
    let g = Grid::new([x, y, vec![0.0, 10.0 * D]]).unwrap();
    // All faces PEC (default); z-faces PEC make the 1-cell slab exactly TM_z.
    let mut e = Engine::builder(g).threads(4).build().unwrap();
    match face {
        Wall::Cpml(p) => {
            e.add_operator(CpmlBoundary::new(&e, [(Face::XMax, p)]).unwrap());
        }
        Wall::Mur => {
            e.add_operator(MurBoundary::new(&e, Face::XMax, C0).unwrap());
        }
        _ => {}
    }
    e.add_operator(EdgeSource::new(dgauss(TAU)).edge(Axis::Z, [SX, SY, 0], 1.0));
    let px = SX + L - A;
    let h45 = e.add_operator(Probe::voltage(Axis::Z, [px, SY + L + A, 0]));
    let h0 = e.add_operator(Probe::voltage(Axis::Z, [px, SY, 0]));
    e.run(steps);
    [h45, h0].map(|h| e.operator(&h).values().to_vec())
}

/// Window in steps: the 45° echo (path √2·(L + A)) plus the pulse
/// (8·τ) plus margin. The nearest spurious wall echo needs ≥ 100 cells.
fn window(dt: f64) -> u64 {
    let cells = 2f64.sqrt() * (L + A) as f64 + 8.0 * TAU * C0 / D + 8.0;
    assert!(cells < 100.0);
    (cells * D / (C0 * dt)) as u64
}

/// Reflection (dB) at the 45° and normal probes for each face model.
fn reflections(faces: &[Wall]) -> Vec<[f64; 2]> {
    // The time step only depends on the cell sizes, the same for all runs.
    let dt = Engine::builder(Grid::uniform([1, 1, 1], [D, D, 10.0 * D]).unwrap())
        .threads(1)
        .build()
        .unwrap()
        .dt();
    let steps = window(dt);
    let r = run(Wall::Reference, steps);
    faces
        .iter()
        .map(|&f| {
            let t = run(f, steps);
            [0, 1].map(|q| db(peak(&diff(&t[q], &r[q])) / peak(&r[q])))
        })
        .collect()
}

#[test]
fn cpml_oblique_incidence() {
    let _t = Timer::start("cpml_oblique_incidence");
    let p = Cpml::cells(8);
    let r = reflections(&[Wall::Cpml(p), Wall::Pec]);
    eprintln!(
        "CPML (8 cells, m = 3): 45° {:.1} dB, normal {:.1} dB; PEC wall (gating check): 45° {:.1} dB, normal {:.1} dB",
        r[0][0], r[0][1], r[1][0], r[1][1]
    );
    assert!(r[0][0] < -40.0 && r[0][1] < -40.0, "{:?}", r[0]);
    // The window does contain the echo of the face under test.
    assert!(r[1][0] > -6.0 && r[1][1] > -6.0, "{:?}", r[1]);
}

#[test]
fn mur_oblique_incidence() {
    let _t = Timer::start("mur_oblique_incidence");
    let r = reflections(&[Wall::Mur]);
    let theory = db((1.0 - 0.5f64.sqrt()) / (1.0 + 0.5f64.sqrt()));
    eprintln!(
        "Mur 1st order: 45° {:.1} dB (plane-wave theory {theory:.1} dB), normal {:.1} dB",
        r[0][0], r[0][1]
    );
    // Clearly worse than at normal incidence, and near the plane-wave
    // value (the cylindrical wave and 1/√r spreading blur it a little).
    assert!(r[0][0] > r[0][1] + 6.0, "{:?}", r[0]);
    assert!(r[0][0] > -25.0 && r[0][0] < -8.0, "{:?}", r[0]);
    assert!(r[0][1] < -25.0, "{:?}", r[0]);
}
