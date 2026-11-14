//! [`PoleWeighting`]: the poles of the edges in the surface of a
//! dispersive material, against a closed form.
//!
//! A parallel-plate line (PEC plates normal to y, PMC walls normal to z,
//! propagation along x) whose lower quarter is a Debye material: `E_y` is
//! tangential to the material's surface and, the cross section being
//! small against the wavelength, uniform over it, so the layers are
//! capacitances in parallel and the line has
//!
//! ```text
//! ε_eff(ω) = (h·ε_d(ω) + (H − h)·ε0)/H,     k = ω·√(μ0·ε_eff)
//! ```
//!
//! (the first term of the transverse resonance
//! `k_z1·tan(k_z1·h) + k_z2·tan(k_z2·(H − h)) = 0`, which the reference
//! solves; the next term is 1.5 % of the attenuation at 6 GHz). The
//! material's surface lies on a mesh line: the `E_y` edges in it have
//! half their dual face in the material.

mod common;

use common::*;
use csx::primitives::BoxShape;
use csx::properties::{DebyeMaterial, DebyePole, Excitation, ProbeBox, ProbeType};
use csx::{ContinuousStructure, Material, Primitive, Property};
use ems::compile::{CompileOptions, PoleWeighting};
use ems::constants::C0;
use ems::{Boundaries, Boundary, Complex64, Face, Fdtd, Simulation, Waveform};

const EPS_INF: f64 = 3.8;
const DELTA: f64 = 0.4;
const TAU: f64 = 3.2e-11;
/// Height of the line and of the material \[mm\]; cells of 0.25 mm.
const H: f64 = 2.0;
const FILL: f64 = 0.5;
/// Distance between the probes \[mm\].
const DIST: f64 = 40.0;

/// Attenuation \[Np/m\] of the line at `freqs` with the poles weighted
/// by `w`.
fn attenuation(w: PoleWeighting, freqs: &[f64]) -> Vec<f64> {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, linspace(0.0, 100.0, 201)).unwrap();
    m.grid.set_lines(1, linspace(0.0, 1.0, 3)).unwrap();
    m.grid.set_lines(2, linspace(0.0, H, 9)).unwrap();
    let diel = m.add_property(Property::debye_material(
        "laminate",
        DebyeMaterial {
            material: Material::new(EPS_INF, 0.0),
            poles: vec![DebyePole::new(DELTA, TAU)],
        },
    ));
    m.add_primitive(
        diel,
        Primitive::new(BoxShape::new([0.0, 0.0, 0.0], [100.0, 1.0, FILL])),
    )
    .unwrap();
    let ex = m.add_property(Property::excitation("ex", Excitation::new([0.0, 1.0, 0.0])));
    m.add_primitive(
        ex,
        Primitive::new(BoxShape::new([20.0, 0.0, 0.0], [20.0, 1.0, H])),
    )
    .unwrap();
    for (name, x) in [("v1", 35.0), ("v2", 35.0 + DIST)] {
        let p = m.add_property(Property::probe(name, ProbeBox::new(ProbeType::Voltage)));
        m.add_primitive(
            p,
            Primitive::new(BoxShape::new([x, 0.0, 1.0], [x, 1.0, 1.0])),
        )
        .unwrap();
    }
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(4e9, 3e9))
        .boundaries(
            Boundaries::all(Boundary::Pec)
                .with(Face::ZMin, Boundary::Pmc)
                .with(Face::ZMax, Boundary::Pmc)
                .with(Face::XMin, Boundary::pml(8))
                .with(Face::XMax, Boundary::pml(8)),
        )
        .max_timesteps(9000)
        .no_end_criteria()
        .threads(1);
    let opts = CompileOptions::default().with_pole_weighting(w);
    let mut sim = Simulation::with_options(&m, &fdtd, &opts).unwrap();
    let res = sim.run().unwrap();
    let (a, b) = (
        res.probe("v1").unwrap().dft(freqs).unwrap(),
        res.probe("v2").unwrap().dft(freqs).unwrap(),
    );
    a.iter()
        .zip(&b)
        .map(|(a, b)| -(b.norm() / a.norm()).ln() / (DIST * 1e-3))
        .collect()
}

/// `α` of the line \[Np/m\]: the root of the transverse resonance next
/// to the parallel capacitances, with `x·tan x = x²·(1 + x²/3 + 2x⁴/15)`
/// (`x` under 0.2 here).
fn exact(f: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * f;
    let k0 = w / C0;
    let eps_d =
        Complex64::new(EPS_INF, 0.0) + Complex64::new(DELTA, 0.0) / Complex64::new(1.0, w * TAU);
    let (h, g) = (FILL * 1e-3, (H - FILL) * 1e-3);
    let re = |x: f64| Complex64::new(x, 0.0);
    // `k_z·tan(k_z·d)` of a layer of thickness `d` with `k_z² = u`.
    let layer = |u: Complex64, d: f64| {
        let x2 = u * re(d * d);
        u * re(d) * (re(1.0) + x2 * re(1.0 / 3.0) + x2 * x2 * re(2.0 / 15.0))
    };
    let resonance = |b2: Complex64| layer(eps_d * re(k0 * k0) - b2, h) + layer(re(k0 * k0) - b2, g);
    let eff = (eps_d * re(FILL / H)) + re(1.0 - FILL / H);
    // Secant steps from the parallel capacitances.
    let (mut b0, mut b1) = (eff * re(k0 * k0), eff * re(1.001 * k0 * k0));
    for _ in 0..30 {
        let (f0, f1) = (resonance(b0), resonance(b1));
        let d = f1 - f0;
        if d.norm() == 0.0 {
            break;
        }
        let next = b1 - f1 * (b1 - b0) / d;
        (b0, b1) = (b1, next);
    }
    -csqrt(b1).im
}

/// With the averaged poles the line has the loss of its closed form; with
/// the full poles on the edges in the material's surface (the default,
/// openEMS's rule) the material is half a cell thicker for its loss, a
/// quarter more with two cells of it.
#[test]
fn averaged_poles_give_a_surface_edge_its_share() {
    let freqs: Vec<f64> = (2..=6).map(|g| g as f64 * 1e9).collect();
    let averaged = attenuation(PoleWeighting::Averaged, &freqs);
    let midpoint = attenuation(PoleWeighting::Midpoint, &freqs);
    for ((f, a), m) in freqs.iter().zip(&averaged).zip(&midpoint) {
        let want = exact(*f);
        eprintln!(
            "{:.0} GHz: α {want:.4} Np/m; averaged {a:.4} ({:+.2} %), midpoint {m:.4} ({:+.2} %)",
            f * 1e-9,
            (a / want - 1.0) * 100.0,
            (m / want - 1.0) * 100.0
        );
        assert!((a / want - 1.0).abs() < 0.005, "{f}: {a} {want}");
        assert!((m / want - 1.25).abs() < 0.02, "{f}: {m} {want}");
    }
}
