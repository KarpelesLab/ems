//! Validation of the curve port ([`CurvePort`], design §5.5).
//!
//! A loop over a ground plane (the PEC `z = 0` face of a small PEC box,
//! 1 mm cells): the port's feed edge stands on the ground, the curve runs
//! up, across (straight or with a diagonal, staircased run) and down to a
//! load resistor `R_L` on the ground. At frequencies where the loop is
//! electrically small the circuit is `R_L` in series with the loop
//! inductance, `Z_in ≈ R_L + jωL` (plus the stray capacitance of the
//! loop, which raises `Re Z_in` by ≈ `2ω²LC`: +7 % at 1 GHz, +1 % at
//! 0.4 GHz, so the check stays below 0.4 GHz): `Re Z_in → R_L` checks that the
//! current flows through the whole curve into the load (connectivity and
//! the V/I sign conventions), and `Im Z_in / ω` must be a constant,
//! positive inductance. A thin csx `Wire` of radius `r0` below the
//! staircase's lattice radius (`r_eq = 0.1985·Δ`, design §5.14) must add
//! about `(μ0/2π)·ln(r_eq/r0)` per unit length to the inductance of the
//! staircase `Curve` (the wire is thinner, its image is not).

use csx::primitives::BoxShape;
use csx::properties::LumpedElement;
use csx::{ContinuousStructure, Primitive, Property};
use ems::port::{CurvePort, LumpedPort};
use ems::prelude::*;

const R_LOAD: f64 = 100.0;
const FREQS: [f64; 3] = [0.1e9, 0.2e9, 0.4e9];

fn model() -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, (0..=16).map(f64::from)).unwrap();
    m.grid.set_lines(1, (0..=8).map(f64::from)).unwrap();
    m.grid.set_lines(2, (0..=8).map(f64::from)).unwrap();
    // Load resistor from (12, 4, 1) down to the ground.
    let load = m.add_property(Property::lumped_element(
        "load",
        LumpedElement::resistor(csx::Axis::Z, R_LOAD),
    ));
    m.add_primitive(
        load,
        Primitive::new(BoxShape::new([12.0, 4.0, 0.0], [12.0, 4.0, 1.0])),
    )
    .unwrap();
    m
}

fn fdtd() -> Fdtd {
    Fdtd::new()
        .waveform(Waveform::gaussian(0.0, 3e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .end_criteria(EndCriteria::energy_decay_db(-50.0))
        .max_timesteps(30_000)
        .threads(2)
}

fn z_in(port: &CurvePort, m: &ContinuousStructure) -> Vec<Complex64> {
    let mut sim = Simulation::new(m, &fdtd()).unwrap();
    let w: Vec<_> = sim
        .warnings()
        .iter()
        .filter(|w| !w.contains("mesh") && !w.contains("thin wires meet"))
        .cloned()
        .collect();
    assert!(w.is_empty(), "{w:?}");
    let res = sim.run().unwrap();
    port.analyze(&res, &FREQS).unwrap().z_in()
}

/// Checks `Z_in ≈ R_L + jωL`; returns `L` (mean over the frequencies).
fn check_rl(z: &[Complex64], what: &str) -> f64 {
    let l: Vec<f64> = z
        .iter()
        .zip(FREQS)
        .map(|(z, f)| z.im / (std::f64::consts::TAU * f))
        .collect();
    for ((z, f), l) in z.iter().zip(FREQS).zip(&l) {
        eprintln!(
            "{what}: {:.2} GHz Z_in = {z:.3} Ω, L = {:.3} nH",
            f / 1e9,
            l * 1e9
        );
        assert!(
            (z.re / R_LOAD - 1.0).abs() < 0.02,
            "{what}: Re Z_in = {} at {f}",
            z.re
        );
    }
    let mean = l.iter().sum::<f64>() / l.len() as f64;
    assert!(mean > 0.0, "{what}: L = {mean}");
    for x in &l {
        assert!((x / mean - 1.0).abs() < 0.05, "{what}: L = {l:?}");
    }
    mean
}

#[test]
fn curve_port_drives_a_loop_into_a_load() {
    // Straight runs: up 4 mm, across 8 mm, down to the load at z = 1.
    let mut m = model();
    let pts = [
        [4.0, 4.0, 0.0],
        [4.0, 4.0, 4.0],
        [12.0, 4.0, 4.0],
        [12.0, 4.0, 1.0],
    ];
    let port = CurvePort::new(1, 50.0, pts)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let l_curve = check_rl(&z_in(&port, &m), "staircase curve");

    // The same loop with a diagonal run (staircased by csx): still R_L in
    // series with a (similar) inductance.
    let mut m = model();
    let pts = [
        [4.0, 4.0, 0.0],
        [4.0, 4.0, 2.0],
        [8.0, 4.0, 4.0],
        [12.0, 4.0, 4.0],
        [12.0, 4.0, 1.0],
    ];
    let port = CurvePort::new(1, 50.0, pts)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let l_diag = check_rl(&z_in(&port, &m), "diagonal curve");
    assert!(
        (l_diag / l_curve - 1.0).abs() < 0.25,
        "{l_diag} vs {l_curve}"
    );

    // The diagonal loop as a thin csx `Wire` (an oblique segment: the
    // oblique-wire model of design §5.14.1) of radius r0 = Δ/10 and Δ/20:
    // per unit length of wire over ground the extra inductance relative to
    // the staircase (lattice radius r_eq = 0.1985·Δ) is (μ0/2π)·ln(r_eq/r0),
    // the image (return path) being unchanged; ℓ = 12.5 mm of wire.
    let wire_l = |r0: f64| {
        let mut m = model();
        let port = CurvePort::new(1, 50.0, pts)
            .excite(1.0)
            .wire_radius(r0)
            .add_to(&mut m)
            .unwrap();
        check_rl(&z_in(&port, &m), &format!("thin wire r0 = {r0} mm"))
    };
    let len = 1.0 + 20f64.sqrt() + 4.0 + 3.0;
    let per_mm = |r0: f64| ems::constants::MU0 / std::f64::consts::TAU * (0.1985 / r0).ln() * 1e-3;
    let mut extras = Vec::new();
    for r0 in [0.1, 0.05] {
        let got = wire_l(r0) - l_diag;
        let want = per_mm(r0) * len;
        eprintln!(
            "r0 = {r0} mm: extra L {:.3} nH, two-wire estimate {:.3} nH ({:+.0} %)",
            got * 1e9,
            want * 1e9,
            100.0 * (got / want - 1.0)
        );
        assert!((got / want - 1.0).abs() < 0.35, "extra L {got} vs {want}");
        extras.push(got);
    }
    // The extra inductance scales with ln(r_eq/r0).
    let ratio = extras[1] / extras[0];
    let want = (0.1985f64 / 0.05).ln() / (0.1985f64 / 0.1).ln();
    eprintln!("extra-L ratio {ratio:.3} (ln ratio {want:.3})");
    assert!((ratio / want - 1.0).abs() < 0.1, "{ratio} vs {want}");
}

/// A curve port is exactly a lumped port on its feed edge plus a PEC
/// curve: the same model built by hand gives bit-identical results.
#[test]
fn curve_port_equals_lumped_port_plus_curve() {
    let pts = [[4.0, 4.0, 0.0], [4.0, 4.0, 3.0], [12.0, 4.0, 1.0]];
    let mut m = model();
    let port = CurvePort::new(1, 50.0, pts)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let mut m2 = model();
    let lp = LumpedPort::new(1, 50.0, [4.0, 4.0, 0.0], [4.0, 4.0, 1.0], Axis::Z)
        .excite(1.0)
        .add_to(&mut m2)
        .unwrap();
    let metal = m2.add_property(Property::metal("port_metal_1"));
    m2.add_primitive(
        metal,
        Primitive::new(csx::primitives::Curve::new([
            [4.0, 4.0, 1.0],
            [4.0, 4.0, 3.0],
            [12.0, 4.0, 1.0],
        ])),
    )
    .unwrap();
    let fdtd = fdtd().max_timesteps(400).no_end_criteria();
    let r1 = Simulation::new(&m, &fdtd).unwrap().run().unwrap();
    let r2 = Simulation::new(&m2, &fdtd).unwrap().run().unwrap();
    let a = port.analyze(&r1, &FREQS).unwrap();
    let b = lp.analyze(&r2, &FREQS).unwrap();
    assert_eq!(a.z_in(), b.z_in());
}
