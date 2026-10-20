//! Validation of the coaxial-line port ([`CoaxPort`], design §5.5, §9).
//!
//! A matched through line between two ports (port 1 excited, port 2
//! passive, looking back), PML at both ends. The TEM line has
//! `Z0 = η0/(2π·√ε_r)·ln(b/a)` (Pozar §2.2) and `β = ω·√ε_r/c`.
//!
//! 1. **Cylindrical grid** (`ρ` from `a` to `b`, the two PEC `ρ` faces as
//!    the conductors, closed azimuth): the mesh is exact for the TEM mode
//!    and the discrete line has the impedance
//!    `η/(2π)·Σ Δρ_i/ρ_{i+½}` and the 1-D Yee wavenumber along `z`, so the
//!    extraction must reproduce both closely; `|S11|` of the through line is
//!    small and `S21 = e^{−jβL}`.
//! 2. **Cartesian grid** (staircased conductors, `#[ignore]`: run in
//!    release): the conductors written as mesh-aligned cells keep the line
//!    exactly TEM (`ε_eff = ε_r` up to the Yee dispersion), with the
//!    staircase changing `Z_L` by about −6 % at six cells per radius.

mod common;

use core::f64::consts::TAU;

use common::*;
use csx::primitives::BoxShape;
use csx::properties::Material;
use csx::{ContinuousStructure, CoordSystem, Primitive, Property};
use ems::constants::Z0;
use ems::port::{CoaxPort, LinePortResult};
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

const EPS_R: f64 = 2.0;

#[test]
fn coax_port_cylindrical_matched_line() {
    let (a, b, dr, dz) = (1.0, 2.3, 0.1, 0.5);
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, lines(a, b, dr)).unwrap();
    m.grid
        .set_lines(1, (0..=8).map(|j| j as f64 * TAU / 8.0))
        .unwrap();
    m.grid.set_lines(2, lines(0.0, 60.0, dz)).unwrap();
    let sub = m.add_property(Property::material("fill", Material::new(EPS_R, 0.0)));
    m.add_primitive(
        sub,
        Primitive::new(BoxShape::new([a, 0.0, 0.0], [b, TAU, 60.0])),
    )
    .unwrap();
    let p1 = CoaxPort::new(1, [0.0, 0.0, 6.0], [0.0, 0.0, 26.0], Axis::Z, a, b)
        .eps_r(EPS_R)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let p2 = CoaxPort::new(2, [0.0, 0.0, 54.0], [0.0, 0.0, 34.0], Axis::Z, a, b)
        .eps_r(EPS_R)
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

    let z_an = p1.z0();
    let z_disc: f64 = lines(a, b, dr)
        .windows(2)
        .map(|w| (w[1] - w[0]) / (0.5 * (w[0] + w[1])))
        .sum::<f64>()
        * Z0
        / (TAU * EPS_R.sqrt());
    let dt = res.dt();
    let (mut ez, mut eza, mut eb) = (0.0f64, 0.0f64, 0.0f64);
    for (k, &f) in freqs.iter().enumerate() {
        let k_yee = yee_k(f, dt, dz * 1e-3, EPS_R, 0.0).re / (dz * 1e-3);
        for r in [&r1, &r2] {
            let (z, beta) = (r.z_line()[k], r.beta()[k]);
            ez = ez.max((z - z_disc).norm() / z_disc);
            eza = eza.max((z - z_an).norm() / z_an);
            eb = eb.max((beta / k_yee - 1.0).abs());
        }
    }
    let (s11, s21) = through(&r1, &r2, 8e-3);
    println!(
        "cylindrical coax: Z0 {z_an:.3} Ω, discrete {z_disc:.3} Ω; |Z_L − discrete| ≤ {:.2e}, \
         |Z_L − Z0| ≤ {:.2e}, β vs Yee ≤ {eb:.2e}; |S11| ≤ {s11:.1} dB, |ΔS21| ≤ {s21:.2e}",
        ez, eza
    );
    assert!(ez < 2e-4, "Z vs discrete {ez}");
    assert!(eza < 2e-3, "Z vs analytic {eza}");
    assert!(eb < 2e-4, "β vs Yee {eb}");
    assert!(s11 < -60.0, "|S11| = {s11} dB");
    assert!(s21 < 1e-3, "S21 error {s21}");
}

/// The same line on a quarter sector `α ∈ [0, π/2]` with PMC α faces
/// (symmetry planes of the TEM mode): the port extracts the sector line,
/// whose impedance is 4 × the full line's (`z0` reports it), with the same
/// `β` and matched S-parameters.
#[test]
fn coax_port_cylindrical_quarter_sector() {
    let (a, b, dr, dz) = (1.0, 2.3, 0.1, 0.5);
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, lines(a, b, dr)).unwrap();
    m.grid
        .set_lines(1, (0..=2).map(|j| j as f64 * TAU / 8.0))
        .unwrap();
    m.grid.set_lines(2, lines(0.0, 60.0, dz)).unwrap();
    let sub = m.add_property(Property::material("fill", Material::new(EPS_R, 0.0)));
    m.add_primitive(
        sub,
        Primitive::new(BoxShape::new([a, 0.0, 0.0], [b, TAU / 4.0, 60.0])),
    )
    .unwrap();
    let p1 = CoaxPort::new(1, [0.0, 0.0, 6.0], [0.0, 0.0, 26.0], Axis::Z, a, b)
        .eps_r(EPS_R)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let p2 = CoaxPort::new(2, [0.0, 0.0, 54.0], [0.0, 0.0, 34.0], Axis::Z, a, b)
        .eps_r(EPS_R)
        .add_to(&mut m)
        .unwrap();
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(5e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::YMin, Boundary::Pmc)
        .boundary(Face::YMax, Boundary::Pmc)
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
        / (TAU * EPS_R.sqrt())
        * 4.0;
    assert!((p1.z0() / (4.0 * Z0 / (TAU * EPS_R.sqrt()) * 2.3f64.ln()) - 1.0).abs() < 1e-12);
    let dt = res.dt();
    let (mut ez, mut eb) = (0.0f64, 0.0f64);
    for (k, &f) in freqs.iter().enumerate() {
        let k_yee = yee_k(f, dt, dz * 1e-3, EPS_R, 0.0).re / (dz * 1e-3);
        for r in [&r1, &r2] {
            let (z, beta) = (r.z_line()[k], r.beta()[k]);
            ez = ez.max((z - z_disc).norm() / z_disc);
            eb = eb.max((beta / k_yee - 1.0).abs());
        }
    }
    let (s11, s21) = through(&r1, &r2, 8e-3);
    println!(
        "quarter-sector coax: discrete {z_disc:.3} Ω; |Z_L − discrete| ≤ {ez:.2e}, β vs Yee ≤ \
         {eb:.2e}; |S11| ≤ {s11:.1} dB, |ΔS21| ≤ {s21:.2e}"
    );
    assert!(ez < 2e-4, "Z vs discrete {ez}");
    assert!(eb < 2e-4, "β vs Yee {eb}");
    assert!(s11 < -60.0, "|S11| = {s11} dB");
    assert!(s21 < 1e-3, "S21 error {s21}");
}

/// Cartesian staircased coax along z: a = 1.5 mm, b = 3.45 mm (50 Ω in
/// air ×1/√ε_r), 0.25 mm cells, a shield to 4 mm.
#[test]
#[ignore = "slow in debug; run with --release -- --ignored"]
fn coax_port_cartesian_matched_line() {
    let (a, b, s) = (1.5, 3.45, 4.0);
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    let cell = 0.25;
    m.grid.set_lines(0, lines(-4.5, 4.5, cell)).unwrap();
    m.grid.set_lines(1, lines(-4.5, 4.5, cell)).unwrap();
    m.grid.set_lines(2, lines(0.0, 60.0, 0.5)).unwrap();
    let sub = m.add_property(Property::material("fill", Material::new(EPS_R, 0.0)));
    m.add_primitive(
        sub,
        Primitive::new(BoxShape::new([-4.5, -4.5, 0.0], [4.5, 4.5, 60.0])),
    )
    .unwrap();
    let p1 = CoaxPort::new(1, [0.0, 0.0, 6.0], [0.0, 0.0, 30.0], Axis::Z, a, b)
        .feed_extension(6.0)
        .shield(s)
        .eps_r(EPS_R)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let p2 = CoaxPort::new(2, [0.0, 0.0, 54.0], [0.0, 0.0, 30.0], Axis::Z, a, b)
        .feed_extension(6.0)
        .shield(s)
        .eps_r(EPS_R)
        .add_to(&mut m)
        .unwrap();
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(5e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::ZMin, Boundary::pml(8))
        .boundary(Face::ZMax, Boundary::pml(8))
        .end_criteria(EndCriteria::energy_decay_db(-60.0))
        .max_timesteps(30_000)
        .threads(4);
    let res = run(&m, &fdtd);
    let freqs = linspace(2e9, 8e9, 7);
    let r1 = p1.analyze(&res, &freqs).unwrap();
    let r2 = p2.analyze(&res, &freqs).unwrap();
    let z_an = p1.z0();
    let (mut eza, mut ee) = (0.0f64, 0.0f64);
    for k in 0..freqs.len() {
        for r in [&r1, &r2] {
            eza = eza.max((r.z_line()[k] - z_an).norm() / z_an);
            ee = ee.max((r.eps_eff()[k] / EPS_R - 1.0).abs());
        }
    }
    let (s11, s21) = through(&r1, &r2, 0.0);
    println!(
        "Cartesian coax: Z0 {z_an:.3} Ω, Z_L {:.3}; |Z_L − Z0| ≤ {:.2} %, ε_eff ≤ {:.2} %; \
         |S11| ≤ {s11:.1} dB, |ΔS21| ≤ {s21:.2e}",
        r1.z_line()[3].re,
        100.0 * eza,
        100.0 * ee
    );
    assert!(eza < 0.07, "Z vs analytic {eza}");
    assert!(ee < 3e-3, "ε_eff {ee}");
    assert!(s11 < -60.0, "|S11| = {s11} dB");
    assert!(s21 < 1e-3, "S21 error {s21}");
}

/// The port's own conductors on a cylindrical grid (inner conductor and
/// shield as `(ρ, α, z)` boxes inside a larger grid), a resistive
/// (Thevenin) feed, and both ports meeting at one reference plane.
#[test]
#[ignore = "about 35 s in debug; run with --release -- --ignored"]
fn coax_port_cylindrical_written_conductors_and_feed_resistor() {
    let (a, b, dr, dz) = (1.0, 2.3, 0.1, 0.5);
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, lines(0.7, 2.6, dr)).unwrap();
    m.grid
        .set_lines(1, (0..=8).map(|j| j as f64 * TAU / 8.0))
        .unwrap();
    m.grid.set_lines(2, lines(0.0, 40.0, dz)).unwrap();
    let p1 = CoaxPort::new(1, [0.0, 0.0, 5.0], [0.0, 0.0, 20.0], Axis::Z, a, b)
        .shield(2.6)
        .dielectric(EPS_R)
        .feed_extension(5.0)
        .feed_resistance(35.0)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let p2 = CoaxPort::new(2, [0.0, 0.0, 35.0], [0.0, 0.0, 20.0], Axis::Z, a, b)
        .shield(2.6)
        .dielectric(EPS_R)
        .feed_extension(5.0)
        .add_to(&mut m)
        .unwrap();
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
    let mut ez = 0.0f64;
    for k in 0..freqs.len() {
        for r in [&r1, &r2] {
            ez = ez.max((r.z_line()[k] - z_disc).norm() / z_disc);
        }
    }
    let (s11, s21) = through(&r1, &r2, 0.0);
    println!(
        "cylindrical coax, written conductors, R feed: |Z_L − discrete| ≤ {ez:.2e}; \
         |S11| ≤ {s11:.1} dB, |ΔS21| ≤ {s21:.2e}"
    );
    assert!(ez < 2e-3, "Z vs discrete {ez}");
    assert!(s11 < -50.0, "|S11| = {s11} dB");
    assert!(s21 < 5e-3, "S21 error {s21}");
}
