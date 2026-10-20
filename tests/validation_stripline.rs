//! Validation of the strip-line port ([`StriplinePort`], design §5.5 / §9).
//!
//! 1. **Split parallel plate** (strip over the full width between PMC side
//!    walls, PEC grounds): the TEM mode is two parallel-plate lines of
//!    height `b/2` in parallel, `Z = η0·b/(4·w·√ε_r)` exactly, and the
//!    discrete line has this impedance with the 1-D Yee wavenumber, so the
//!    extraction must reproduce both to round-off; two ports facing each
//!    other give `S11 ≈ 0`, `S21 = e^{−jβL}`.
//! 2. **Stripline** (`w/b = 1`, ε_r = 2.2) with PEC side walls, half of it
//!    modelled with a PMC symmetry plane through the strip centre, the
//!    strip edges meshed with the thirds rule: `Z_L` versus Cohn's exact
//!    zero-thickness formula, `ε_eff = ε_r`, and the through-line `S11`.

mod common;

use common::*;
use csx::primitives::BoxShape;
use csx::properties::Material;
use csx::{ContinuousStructure, Primitive, Property};
use ems::constants::Z0;
use ems::port::{LinePortResult, StriplinePort};
use ems::post::s_param;
use ems::prelude::*;

fn lines(from: f64, to: f64, step: f64) -> Vec<f64> {
    let n = ((to - from) / step).round() as usize;
    (0..=n).map(|k| from + k as f64 * step).collect()
}

fn run(model: &ContinuousStructure, fdtd: &Fdtd) -> Results {
    let mut sim = Simulation::new(model, fdtd).unwrap();
    let w: Vec<_> = sim
        .warnings()
        .iter()
        .filter(|w| !w.contains("mesh"))
        .cloned()
        .collect();
    assert!(w.is_empty(), "{w:?}");
    let res = sim.run().unwrap();
    let o = res.outcome().unwrap();
    assert!(
        matches!(o.reason, ems::engine::StopReason::EnergyDecay { .. }),
        "{o:?}"
    );
    res
}

/// Worst `|S11|` (dB) of a through line; checks `S21 ≈ e^{−jβL}`.
fn through(r1: &LinePortResult, r2: &LinePortResult, len: f64, s21_tol: f64) -> f64 {
    let s11 = r1.s11();
    let s21 = s_param(r2, r1);
    let beta = r1.beta();
    let mut worst = f64::NEG_INFINITY;
    for k in 0..s11.len() {
        worst = worst.max(db(s11[k].norm()));
        let want = Complex64::cis(-beta[k] * len);
        assert!(
            (s21[k] - want).norm() < s21_tol,
            "f = {}: S21 = {}, want {want}",
            r1.freqs()[k],
            s21[k]
        );
    }
    worst
}

#[test]
fn stripline_port_split_parallel_plate_exact() {
    const EPS_R: f64 = 2.0;
    let (b, w) = (2.0, 2.0);
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    model.grid.set_lines(0, lines(0.0, 50.0, 0.5)).unwrap();
    model.grid.set_lines(1, lines(0.0, w, 0.5)).unwrap();
    model.grid.set_lines(2, lines(0.0, b, 0.25)).unwrap();
    let sub = model.add_property(Property::material("sub", Material::new(EPS_R, 0.0)));
    model
        .add_primitive(
            sub,
            Primitive::new(BoxShape::new([0.0, 0.0, 0.0], [50.0, w, b])),
        )
        .unwrap();
    // The strip runs through both CPML ends.
    let cu = model.add_property(Property::metal("strip"));
    model
        .add_primitive(
            cu,
            Primitive::new(BoxShape::new([0.0, 0.0, 1.0], [50.0, w, 1.0])),
        )
        .unwrap();
    let p1 = StriplinePort::new(1, [8.0, 0.0, 0.0], [20.0, w, b], Axis::X, Axis::Z)
        .eps_r(EPS_R)
        .metal(cu)
        .excite(1.0)
        .add_to(&mut model)
        .unwrap();
    let p2 = StriplinePort::new(2, [42.0, 0.0, 0.0], [30.0, w, b], Axis::X, Axis::Z)
        .eps_r(EPS_R)
        .metal(cu)
        .add_to(&mut model)
        .unwrap();
    assert_eq!(p1.probe_positions(), Some([13.5, 14.0, 14.5]));
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(4e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::XMin, Boundary::pml(8))
        .boundary(Face::XMax, Boundary::pml(8))
        .boundary(Face::YMin, Boundary::Pmc)
        .boundary(Face::YMax, Boundary::Pmc)
        .end_criteria(EndCriteria::energy_decay_db(-80.0))
        .max_timesteps(20_000);
    let res = run(&model, &fdtd);

    let freqs = linspace(1e9, 7e9, 13);
    let r1 = p1.analyze(&res, &freqs).unwrap();
    let r2 = p2.analyze(&res, &freqs).unwrap();
    let z_want = Z0 / EPS_R.sqrt() * b / (4.0 * w);
    let dt = res.dt();
    let (mut eb, mut ez) = (0.0f64, 0.0f64);
    for (k, &f) in freqs.iter().enumerate() {
        let k_yee = yee_k(f, dt, 0.5e-3, EPS_R, 0.0).re / 0.5e-3;
        let bd = k_yee * 0.5e-3;
        let tol = 1e-6 + 5e-8 / (bd * bd);
        for r in [&r1, &r2] {
            let (z, beta) = (r.z_line()[k], r.beta()[k]);
            let (db_, dz) = ((beta / k_yee - 1.0).abs(), (z - z_want).norm() / z_want);
            eb = eb.max(db_ / tol);
            ez = ez.max(dz / tol);
            assert!(dz < tol, "f = {f:e}: Z = {z}, want {z_want}");
            assert!(db_ < tol, "f = {f:e}: β = {beta}, Yee {k_yee}");
        }
    }
    println!("split parallel plate: β and Z errors up to {eb:.2} and {ez:.2} of the tolerance");
    let worst = through(&r1, &r2, 10e-3, 1e-4);
    println!("split parallel plate: worst |S11| = {worst:.1} dB");
    assert!(worst < -60.0, "{worst}");
}

/// The stripline of case 2 with cells `d` mm along the line and across
/// the strip; returns the mean (Z_full, ε_eff) of both ports and the
/// worst through-line |S11| \[dB\].
fn stripline(d: f64, b: f64) -> (f64, f64, f64, f64) {
    const EPS_R: f64 = 2.2;
    let w = b; // w/b = 1
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    // Half model: PMC at y = 0, strip on 0 ≤ y ≤ w/2, thirds rule at the
    // edge (w/2 = (n + 1/3)·dy), graded to 4·dy cells, PEC wall at
    // y ≈ w/2 + 3b.
    let n = ((0.5 * w / d) - 1.0 / 3.0).round().max(1.0);
    let dy = 0.5 * w / (n + 1.0 / 3.0);
    let mut y = lines(0.0, (n + 1.0) * dy, dy);
    let mut step = dy;
    while *y.last().unwrap() < 0.5 * w + 3.0 * b {
        step = (step * 1.3).min(4.0 * dy);
        let next = *y.last().unwrap() + step;
        y.push(next);
    }
    let ymax = *y.last().unwrap();
    let len = 40.0;
    model.grid.set_lines(0, lines(0.0, len, d)).unwrap();
    model.grid.set_lines(1, y).unwrap();
    model
        .grid
        .set_lines(2, lines(0.0, b, d.min(b / 8.0)))
        .unwrap();
    let sub = model.add_property(Property::material("sub", Material::new(EPS_R, 0.0)));
    model
        .add_primitive(
            sub,
            Primitive::new(BoxShape::new([0.0, 0.0, 0.0], [len, ymax, b])),
        )
        .unwrap();
    let cu = model.add_property(Property::metal("strip"));
    model
        .add_primitive(
            cu,
            Primitive::new(BoxShape::new([0.0, 0.0, b / 2.0], [len, w / 2.0, b / 2.0])),
        )
        .unwrap();
    let p1 = StriplinePort::new(1, [6.0, 0.0, 0.0], [17.0, w / 2.0, b], Axis::X, Axis::Z)
        .metal(cu)
        .feed_resistance(50.0)
        .excite(1.0)
        .add_to(&mut model)
        .unwrap();
    let p2 = StriplinePort::new(
        2,
        [len - 6.0, 0.0, 0.0],
        [23.0, w / 2.0, b],
        Axis::X,
        Axis::Z,
    )
    .metal(cu)
    .add_to(&mut model)
    .unwrap();
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(4e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::XMin, Boundary::pml(8))
        .boundary(Face::XMax, Boundary::pml(8))
        .boundary(Face::YMin, Boundary::Pmc)
        .end_criteria(EndCriteria::energy_decay_db(-50.0))
        .max_timesteps(20_000)
        .threads(4);
    let res = run(&model, &fdtd);
    let freqs = linspace(1e9, 7e9, 7);
    let r1 = p1.analyze(&res, &freqs).unwrap();
    let r2 = p2.analyze(&res, &freqs).unwrap();
    let z_ref = StriplinePort::z0_of(w, b, EPS_R);
    let mut zs = Vec::new();
    let mut es = Vec::new();
    for r in [&r1, &r2] {
        for (k, f) in freqs.iter().enumerate() {
            // The half model carries half the strip current.
            let z = 0.5 * r.z_line()[k];
            println!(
                "d = {d} mm, {:.1} GHz: Z = {z:.3} (Cohn {z_ref:.3}), ε_eff = {:.4}",
                f / 1e9,
                r.eps_eff()[k]
            );
            assert!(z.im.abs() < 0.01 * z.re, "{z}");
            zs.push(z.re);
            es.push(r.eps_eff()[k]);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let worst = through(&r1, &r2, (23.0 - 17.0) * 1e-3, 0.02);
    (mean(&zs), z_ref, mean(&es), worst)
}

#[test]
fn stripline_port_vs_cohn() {
    let (z, z_ref, e, worst) = stripline(0.5, 2.0);
    println!(
        "stripline w/b = 1: Z = {z:.2} Ω, Cohn {z_ref:.2} Ω ({:+.2} %), ε_eff = {e:.4}, \
         worst |S11| = {worst:.1} dB",
        100.0 * (z / z_ref - 1.0)
    );
    assert!((z / z_ref - 1.0).abs() < 0.03, "Z = {z}, Cohn {z_ref}");
    assert!((e / 2.2 - 1.0).abs() < 0.01, "ε_eff = {e}");
    assert!(worst < -35.0, "{worst}");
}

/// A finer mesh (release: `cargo test --release -- --ignored`).
#[test]
#[ignore]
fn stripline_port_vs_cohn_fine() {
    for d in [0.25, 0.125] {
        let (z, z_ref, e, worst) = stripline(d, 2.0);
        println!(
            "d = {d}: Z = {z:.3} Ω, Cohn {z_ref:.3} Ω ({:+.2} %), ε_eff = {e:.4}, worst |S11| = \
             {worst:.1} dB",
            100.0 * (z / z_ref - 1.0)
        );
        assert!((z / z_ref - 1.0).abs() < 0.015, "Z = {z}, Cohn {z_ref}");
    }
}

#[test]
fn stripline_port_errors() {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, lines(0.0, 20.0, 1.0)).unwrap();
    m.grid.set_lines(1, lines(-3.0, 3.0, 1.0)).unwrap();
    m.grid.set_lines(2, lines(0.0, 3.0, 1.0)).unwrap();
    // Mid-plane z = 1.5 is not a mesh line.
    let p = StriplinePort::new(1, [0.0, -1.0, 0.0], [10.0, 1.0, 3.0], Axis::X, Axis::Z);
    assert!(matches!(p.add_to(&mut m), Err(ems::Error::Invalid(_))));
    let p = StriplinePort::new(1, [0.0, -1.0, 0.0], [10.0, 1.0, 2.0], Axis::X, Axis::X);
    assert!(p.add_to(&mut m).is_err());
    let p = StriplinePort::new(1, [0.0, -1.0, 0.0], [10.0, 1.0, 2.0], Axis::X, Axis::Z)
        .feed_resistance(-1.0);
    assert!(p.add_to(&mut m).is_err());
    let ok = StriplinePort::new(1, [0.0, -1.0, 0.0], [10.0, 1.0, 2.0], Axis::X, Axis::Z)
        .feed_resistance(50.0)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let pr = ok.properties().unwrap();
    assert_eq!(pr.elements.as_ref().unwrap()[1].0, "port_resist_1B");
    assert_eq!(pr.excitations.as_ref().unwrap()[0].0, "port_excite_1A");
    assert!(ok.clone().add_to(&mut m).is_err());
    let again = StriplinePort::new(1, [0.0, -1.0, 0.0], [10.0, 1.0, 2.0], Axis::X, Axis::Z);
    assert!(again.add_to(&mut m).is_err()); // names taken
}
