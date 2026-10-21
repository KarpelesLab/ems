//! The microstrip-line port ([`MslPort`]) on cylindrical grids (design
//! §5.5, §5.15), on the two uniform line geometries it accepts there,
//! each in a limit with an exact or closed-form reference:
//!
//! 1. **Along z, strip above the ground along ρ**, the strip covering the
//!    whole azimuth: the line is a coax (ground = inner conductor, the
//!    PEC `ρ_min` face; strip = outer conductor on the `ρ_max` face), so
//!    the extracted impedance is the discrete coax TEM value
//!    `η/(2π)·Σ Δρ_i/ρ_{i+½}` (as for [`CoaxPort`](ems::port::CoaxPort))
//!    and `β` the 1-D Yee wavenumber along z; a matched through line.
//! 2. **Along α, strip above a z ground**, a full-width strip between PMC
//!    `ρ` walls at a large radius: the curved parallel-plate line tends to
//!    the flat one, `Z = η·h/w` and `β = k` per metre of arc at the
//!    middle radius, to `O((w/R)²)`.

mod common;

use core::f64::consts::TAU;

use common::*;
use csx::primitives::BoxShape;
use csx::properties::Material;
use csx::{ContinuousStructure, CoordSystem, Primitive, Property};
use ems::constants::Z0;
use ems::port::{LinePortResult, MslPort};
use ems::post::s_param;
use ems::prelude::*;

fn lines(from: f64, to: f64, step: f64) -> Vec<f64> {
    let n = ((to - from) / step).round() as usize;
    (0..=n).map(|k| from + k as f64 * step).collect()
}

fn cylindrical(rho: Vec<f64>, alpha: Vec<f64>, z: Vec<f64>) -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, rho).unwrap();
    m.grid.set_lines(1, alpha).unwrap();
    m.grid.set_lines(2, z).unwrap();
    m
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
    sim.run().unwrap()
}

/// Worst `|S11|` (dB) and worst `|S21 − e^{−jβL}|` of a through line.
fn through(r1: &LinePortResult, r2: &LinePortResult, len: f64) -> (f64, f64) {
    let s11 = r1.s11();
    let s21 = s_param(r2, r1);
    let beta = r1.beta();
    let (mut w11, mut w21) = (f64::NEG_INFINITY, 0.0f64);
    for k in 0..s11.len() {
        w11 = w11.max(db(s11[k].norm()));
        w21 = w21.max((s21[k] - Complex64::cis(-beta[k] * len)).norm());
    }
    (w11, w21)
}

#[test]
fn msl_along_z_around_the_azimuth_is_a_coax() {
    const EPS_R: f64 = 2.0;
    let (a, b, dr, dz) = (1.0, 2.3, 0.1, 0.5);
    let mut m = cylindrical(
        lines(a, b, dr),
        (0..=8).map(|j| j as f64 * TAU / 8.0).collect(),
        lines(0.0, 60.0, dz),
    );
    let sub = m.add_property(Property::material("fill", Material::new(EPS_R, 0.0)));
    m.add_primitive(
        sub,
        Primitive::new(BoxShape::new([a, 0.0, 0.0], [b, TAU, 60.0])),
    )
    .unwrap();
    // Ground on the inner face (ρ = a), the strip on the outer face over
    // the whole azimuth.
    let p1 = MslPort::new(1, [a, 0.0, 6.0], [b, TAU, 26.0], Axis::Z, Axis::X)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let p2 = MslPort::new(2, [a, 0.0, 54.0], [b, TAU, 34.0], Axis::Z, Axis::X)
        .add_to(&mut m)
        .unwrap();
    assert_eq!(p1.probe_positions(), Some([15.5, 16.0, 16.5]));
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(5e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::ZMin, Boundary::pml(8))
        .boundary(Face::ZMax, Boundary::pml(8))
        .end_criteria(EndCriteria::energy_decay_db(-80.0))
        .max_timesteps(30_000);
    let res = run(&m, &fdtd);
    let freqs = linspace(2e9, 8e9, 7);
    let r1 = p1.analyze(&res, &freqs).unwrap();
    let r2 = p2.analyze(&res, &freqs).unwrap();
    let z_disc: f64 = lines(a, b, dr)
        .windows(2)
        .map(|w| (w[1] - w[0]) / (0.5 * (w[0] + w[1])))
        .sum::<f64>()
        * Z0
        / (TAU * EPS_R.sqrt());
    let dt = res.dt();
    let (mut ez, mut eb) = (0.0f64, 0.0f64);
    for (k, &f) in freqs.iter().enumerate() {
        let k_yee = yee_k(f, dt, dz * 1e-3, EPS_R, 0.0).re / (dz * 1e-3);
        for r in [&r1, &r2] {
            ez = ez.max((r.z_line()[k] - z_disc).norm() / z_disc);
            eb = eb.max((r.beta()[k] / k_yee - 1.0).abs());
        }
    }
    let (s11, s21) = through(&r1, &r2, 8e-3);
    println!(
        "MSL along z as a coax: discrete Z {z_disc:.3} Ω, |Z_L − Z| ≤ {ez:.2e}, β vs Yee ≤ \
         {eb:.2e}; |S11| ≤ {s11:.1} dB, |ΔS21| ≤ {s21:.2e}"
    );
    assert!(ez < 2e-4, "Z vs discrete {ez}");
    assert!(eb < 2e-4, "β vs Yee {eb}");
    assert!(s11 < -60.0, "|S11| = {s11} dB");
    assert!(s21 < 1e-3, "S21 error {s21}");
}

#[test]
fn msl_along_alpha_at_a_large_radius_is_a_parallel_plate_line() {
    let (r0, w, h) = (200.0, 2.0, 1.0);
    let da = 0.5 / r0; // 0.5 mm of arc per cell at the middle radius
    let n_alpha = 140;
    let alpha: Vec<f64> = (0..=n_alpha).map(|j| j as f64 * da).collect();
    let mut m = cylindrical(
        lines(r0 - w / 2.0, r0 + w / 2.0, 0.25),
        alpha.clone(),
        lines(0.0, h, 0.5),
    );
    // Ground z = 0 (PEC face), strip on the z = h PEC face over the full
    // width between the PMC ρ walls.
    let (f1, e1) = (alpha[20], alpha[50]);
    let (f2, e2) = (alpha[120], alpha[90]);
    let p1 = MslPort::new(
        1,
        [r0 - w / 2.0, f1, 0.0],
        [r0 + w / 2.0, e1, h],
        Axis::Y,
        Axis::Z,
    )
    .excite(1.0)
    .add_to(&mut m)
    .unwrap();
    let p2 = MslPort::new(
        2,
        [r0 - w / 2.0, f2, 0.0],
        [r0 + w / 2.0, e2, h],
        Axis::Y,
        Axis::Z,
    )
    .add_to(&mut m)
    .unwrap();
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(5e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::XMin, Boundary::Pmc)
        .boundary(Face::XMax, Boundary::Pmc)
        .boundary(Face::YMin, Boundary::pml(8))
        .boundary(Face::YMax, Boundary::pml(8))
        .end_criteria(EndCriteria::energy_decay_db(-80.0))
        .max_timesteps(30_000);
    let res = run(&m, &fdtd);
    let freqs = linspace(2e9, 8e9, 7);
    let r1 = p1.analyze(&res, &freqs).unwrap();
    let r2 = p2.analyze(&res, &freqs).unwrap();
    let z_pp = Z0 * h / w;
    let dt = res.dt();
    let (mut ez, mut eb) = (0.0f64, 0.0f64);
    for (k, &f) in freqs.iter().enumerate() {
        let k_yee = yee_k(f, dt, 0.5e-3, 1.0, 0.0).re / 0.5e-3;
        for r in [&r1, &r2] {
            ez = ez.max((r.z_line()[k] - z_pp).norm() / z_pp);
            eb = eb.max((r.beta()[k] / k_yee - 1.0).abs());
        }
    }
    let len = (alpha[90] - alpha[50]) * r0 * 1e-3;
    let (s11, s21) = through(&r1, &r2, len);
    println!(
        "MSL along α at R = {r0} mm: parallel plate {z_pp:.3} Ω, |Z_L − Z| ≤ {ez:.2e}, β vs \
         Yee ≤ {eb:.2e}; |S11| ≤ {s11:.1} dB, |ΔS21| ≤ {s21:.2e}"
    );
    assert!(ez < 1e-3, "Z vs parallel plate {ez}");
    assert!(eb < 1e-3, "β vs Yee {eb}");
    assert!(s11 < -50.0, "|S11| = {s11} dB");
    assert!(s21 < 3e-3, "S21 error {s21}");
}

/// Lines along ρ and grounds normal to α are not uniform: refused.
#[test]
fn msl_along_rho_is_refused() {
    let mut m = cylindrical(
        lines(1.0, 10.0, 0.5),
        (0..=8).map(|j| j as f64 * 0.1).collect(),
        lines(0.0, 2.0, 0.5),
    );
    for (dir, height) in [(Axis::X, Axis::Z), (Axis::Z, Axis::Y)] {
        let p = MslPort::new(1, [2.0, 0.1, 0.0], [8.0, 0.5, 1.0], dir, height);
        assert!(matches!(p.add_to(&mut m), Err(ems::Error::Unsupported(_))));
    }
}
