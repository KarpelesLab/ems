//! Validation of the microstrip-line port ([`MslPort`], design §9).
//!
//! 1. **Parallel-plate line** (PEC ground and top wall, PMC side walls, the
//!    strip on the top wall over the full width): the TEM mode is exact,
//!    `Z = η0·h/(w·√ε_r)`, and the discrete Yee line has exactly this wave
//!    impedance with the 1-D numerical wavenumber (see the `msl` module
//!    docs), so the extraction must reproduce both to round-off and DFT
//!    accuracy. Two ports facing each other: `S11 ≈ 0`, `S21 = e^{−jβL}`.
//! 2. **Microstrip** (ε_r = 4.3, h = 0.8 mm, w = 1.5 mm) in a shielded box,
//!    half of it modelled with a PMC symmetry plane through the strip
//!    centre: the extracted impedance and effective permittivity versus the
//!    quasi-static closed form (Hammerstad 1975, as given in Pozar,
//!    *Microwave Engineering*, §3.8), and the two-port through-line
//!    S-parameters.

mod common;

use std::f64::consts::PI;

use common::*;
use csx::primitives::BoxShape;
use csx::properties::Material;
use csx::{ContinuousStructure, Primitive, Property};
use ems::constants::Z0;
use ems::port::{MslPort, MslPortResult};
use ems::post::s_param;
use ems::prelude::*;

fn lines(from: f64, to: f64, step: f64) -> Vec<f64> {
    let n = ((to - from) / step).round() as usize;
    (0..=n).map(|k| from + k as f64 * step).collect()
}

fn mean(x: impl IntoIterator<Item = f64>) -> f64 {
    let v: Vec<f64> = x.into_iter().collect();
    v.iter().sum::<f64>() / v.len() as f64
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

/// Checks `S11 ≈ 0` and `S21 ≈ e^{−jβL}` of a through line between two
/// port results; returns the worst `|S11|` in dB.
fn through(r1: &MslPortResult, r2: &MslPortResult, len: f64, s21_tol: f64) -> f64 {
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
fn msl_port_parallel_plate_exact() {
    const EPS_R: f64 = 2.0;
    let (h, w) = (1.0, 2.0);
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    model.grid.set_lines(0, lines(0.0, 60.0, 0.5)).unwrap();
    model.grid.set_lines(1, lines(0.0, w, 0.5)).unwrap();
    model.grid.set_lines(2, lines(0.0, h, 0.25)).unwrap();
    let sub = model.add_property(Property::material("sub", Material::new(EPS_R, 0.0)));
    model
        .add_primitive(
            sub,
            Primitive::new(BoxShape::new([0.0, 0.0, 0.0], [60.0, w, h])),
        )
        .unwrap();
    let p1 = MslPort::new(1, [10.0, 0.0, 0.0], [25.0, w, h], Axis::X, Axis::Z)
        .excite(1.0)
        .add_to(&mut model)
        .unwrap();
    let p2 = MslPort::new(2, [50.0, 0.0, 0.0], [35.0, w, h], Axis::X, Axis::Z)
        .add_to(&mut model)
        .unwrap();
    assert_eq!(p1.probe_positions(), Some([17.0, 17.5, 18.0]));
    assert_eq!(p2.probe_positions(), Some([43.0, 42.5, 42.0]));
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
    let z_want = Z0 / EPS_R.sqrt() * h / w;
    let dt = res.dt();
    let (mut eb, mut ez) = (0.0f64, 0.0f64);
    for (k, &f) in freqs.iter().enumerate() {
        let k_yee = yee_k(f, dt, 0.5e-3, EPS_R, 0.0).re / 0.5e-3;
        // β comes from a second difference of the voltages, so the f32
        // round-off of the fields enters as ~1e-8/(βd)².
        let bd = k_yee * 0.5e-3;
        let tol = 1e-6 + 5e-8 / (bd * bd);
        for r in [&r1, &r2] {
            let (z, b) = (r.z_line()[k], r.beta()[k]);
            let (db_, dz) = ((b / k_yee - 1.0).abs(), (z - z_want).norm() / z_want);
            eb = eb.max(db_ / tol);
            ez = ez.max(dz / tol);
            assert!(dz < tol, "f = {f:e}: Z = {z}, want {z_want}");
            assert!(db_ < tol, "f = {f:e}: β = {b}, Yee {k_yee}");
            assert!(r.alpha()[k].abs() < tol * b);
        }
        // The incident wave of port 1 is the only one at its plane (up to
        // the CPML reflection); V = V_inc there.
        assert!(r1.v_ref()[k].norm() < 1e-3 * r1.v_inc()[k].norm());
    }
    println!("parallel plate: β and Z errors up to {eb:.2} and {ez:.2} of the tolerance");
    let worst = through(&r1, &r2, 10e-3, 1e-4);
    println!("parallel plate: worst |S11| = {worst:.1} dB");
    assert!(worst < -60.0, "{worst}");
}

/// Hammerstad / Pozar closed form for a zero-thickness strip: (ε_eff, Z0).
fn hammerstad(eps_r: f64, w: f64, h: f64) -> (f64, f64) {
    let u = w / h;
    let e = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 / (1.0 + 12.0 / u).sqrt();
    let z0 = if u <= 1.0 {
        60.0 / e.sqrt() * (8.0 / u + u / 4.0).ln()
    } else {
        120.0 * PI / (e.sqrt() * (u + 1.393 + 0.667 * (u + 1.444).ln()))
    };
    (e, z0)
}

#[test]
fn msl_port_microstrip_vs_hammerstad() {
    const EPS_R: f64 = 4.3;
    const H: f64 = 0.8;
    const W: f64 = 1.5;
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    // Half model: PMC symmetry plane at y = 0, the strip on 0 ≤ y ≤ w/2.
    // Uniform cells `dy` across the strip with the thirds rule at its edge
    // (`w/2 = (2 + 1/3)·dy`), graded to 1 mm cells, then an 8-cell CPML
    // above and beside the line.
    let dy = 0.5 * W / (2.0 + 1.0 / 3.0);
    let mut y = lines(0.0, 3.0 * dy, dy);
    let mut z = lines(0.0, H, 0.2);
    for v in [&mut y, &mut z] {
        let mut d = *v.last().unwrap() - v[v.len() - 2];
        while *v.last().unwrap() < 13.0 {
            d = (d * 1.4).min(1.0);
            let next = *v.last().unwrap() + d;
            v.push(next);
        }
    }
    model.grid.set_lines(0, lines(-25.0, 25.0, 0.5)).unwrap();
    model.grid.set_lines(1, y.clone()).unwrap();
    model.grid.set_lines(2, z).unwrap();
    let ymax = *y.last().unwrap();
    let sub = model.add_property(Property::material("sub", Material::new(EPS_R, 0.0)));
    model
        .add_primitive(
            sub,
            Primitive::new(BoxShape::new([-25.0, 0.0, 0.0], [25.0, ymax, H])),
        )
        .unwrap();
    let cu = model.add_property(Property::metal("strip"));
    model
        .add_primitive(
            cu,
            Primitive::new(BoxShape::new([-25.0, 0.0, H], [25.0, W / 2.0, H])),
        )
        .unwrap();
    let p1 = MslPort::new(1, [-20.0, 0.0, 0.0], [-2.0, W / 2.0, H], Axis::X, Axis::Z)
        .metal(cu)
        .feed_resistance(50.0)
        .excite(1.0)
        .add_to(&mut model)
        .unwrap();
    let p2 = MslPort::new(2, [20.0, 0.0, 0.0], [2.0, W / 2.0, H], Axis::X, Axis::Z)
        .metal(cu)
        .add_to(&mut model)
        .unwrap();
    assert!(p1.properties().unwrap().element.is_some());
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(4e9, 4e9))
        .boundaries(Boundaries::all(Boundary::pml(8)))
        .boundary(Face::ZMin, Boundary::Pec)
        .boundary(Face::YMin, Boundary::Pmc)
        .end_criteria(EndCriteria::energy_decay_db(-50.0))
        .max_timesteps(30_000);
    let res = run(&model, &fdtd);

    let freqs = linspace(1e9, 6e9, 11);
    let r1 = p1.analyze(&res, &freqs).unwrap();
    let r2 = p2.analyze(&res, &freqs).unwrap();
    let (e_ham, z_ham) = hammerstad(EPS_R, W, H);
    for (k, f) in freqs.iter().enumerate() {
        println!(
            "{:.2} GHz: Z1 = {:.3}, Z2 = {:.3}, eps1 = {:.4}, eps2 = {:.4}",
            f / 1e9,
            r1.z_line()[k] * 0.5,
            r2.z_line()[k] * 0.5,
            r1.eps_eff()[k],
            r2.eps_eff()[k]
        );
    }
    // The half model carries half the strip current: Z_full = Z_half / 2.
    let low = |r: &MslPortResult| {
        let z = mean(r.z_line().iter().take(5).map(|z| 0.5 * z.re));
        (z, mean(r.eps_eff().into_iter().take(5)))
    };
    let ((z1, e1), (z2, e2)) = (low(&r1), low(&r2));
    println!(
        "microstrip (1–3 GHz): Z = {z1:.2} / {z2:.2} Ω, Hammerstad {z_ham:.2} Ω ({:+.2} %); \
         ε_eff = {e1:.3} / {e2:.3}, Hammerstad {e_ham:.3} ({:+.2} %)",
        (z2 / z_ham - 1.0) * 100.0,
        (e2 / e_ham - 1.0) * 100.0
    );
    for r in [&r1, &r2] {
        for zl in r.z_line() {
            assert!(zl.im.abs() < 0.02 * zl.re, "{zl}");
        }
    }
    assert!((z2 / z1 - 1.0).abs() < 0.01, "{z1} vs {z2}");
    assert!((e2 / e1 - 1.0).abs() < 0.01, "{e1} vs {e2}");
    assert!(
        (z2 / z_ham - 1.0).abs() < 0.03,
        "Z = {z2}, Hammerstad {z_ham}"
    );
    assert!(
        (e2 / e_ham - 1.0).abs() < 0.03,
        "ε_eff = {e2}, Hammerstad {e_ham}"
    );
    let worst = through(&r1, &r2, 4e-3, 0.02);
    println!("microstrip: worst |S11| = {worst:.1} dB");
    assert!(worst < -30.0, "{worst}");
}

#[test]
fn msl_port_errors() {
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    let port = || MslPort::new(1, [0.0, -1.0, 0.0], [10.0, 1.0, 1.0], Axis::X, Axis::Z);
    // No mesh yet.
    assert!(port().add_to(&mut model).is_err());
    model.grid.set_lines(0, lines(0.0, 10.0, 1.0)).unwrap();
    model.grid.set_lines(1, lines(-2.0, 2.0, 0.5)).unwrap();
    model.grid.set_lines(2, lines(0.0, 2.0, 0.5)).unwrap();
    // No mesh line within the strip width.
    assert!(
        MslPort::new(1, [0.0, 0.1, 0.0], [10.0, 0.4, 1.0], Axis::X, Axis::Z)
            .add_to(&mut model)
            .is_err()
    );
    // Same propagation and height axes; zero height.
    assert!(
        MslPort::new(1, [0.0, -1.0, 0.0], [10.0, 1.0, 1.0], Axis::X, Axis::X)
            .add_to(&mut model)
            .is_err()
    );
    assert!(
        MslPort::new(1, [0.0, -1.0, 1.0], [10.0, 1.0, 1.0], Axis::X, Axis::Z)
            .add_to(&mut model)
            .is_err()
    );
    assert!(port().feed_resistance(-5.0).add_to(&mut model).is_err());
    // Too short for three probe lines past the feed.
    assert!(
        MslPort::new(1, [0.0, -1.0, 0.0], [2.0, 1.0, 1.0], Axis::X, Axis::Z)
            .add_to(&mut model)
            .is_err()
    );
    let n0 = model.properties().len();
    let p = port().excite(1.0).add_to(&mut model).unwrap();
    assert_eq!(model.properties().len(), n0 + 7);
    let names = p.properties().unwrap();
    assert_eq!(names.metal.0, "port_metal_1");
    assert_eq!(names.currents[1].0, "port_it1B");
    assert!(names.element.is_none());
    // Duplicate number, and adding twice.
    assert!(port().add_to(&mut model).is_err());
    assert!(p.add_to(&mut model).is_err());
}

#[test]
fn msl_port_graded_mesh_is_rejected() {
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    let x: Vec<f64> = [0.0, 1.0, 2.2, 3.6, 5.2, 7.0, 9.0].to_vec();
    model.grid.set_lines(0, x).unwrap();
    model.grid.set_lines(1, lines(-2.0, 2.0, 0.5)).unwrap();
    model.grid.set_lines(2, lines(0.0, 2.0, 0.5)).unwrap();
    let e = MslPort::new(1, [0.0, -1.0, 0.0], [9.0, 1.0, 1.0], Axis::X, Axis::Z)
        .add_to(&mut model)
        .unwrap_err();
    assert!(e.to_string().contains("equally spaced"), "{e}");
}
