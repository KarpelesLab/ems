//! Analytic validation (design §9): a lumped port feeding a matched
//! parallel-plate line.
//!
//! Plates normal to x (PEC, `h` = 2 mm apart), PMC side walls normal to y
//! (`w` = 2 mm apart), so the TEM mode is exact and the characteristic
//! impedance is exactly `Z = η0·h/w` (Pozar, parallel-plate line; with
//! PMC walls there is no fringing). The line runs along z from a PMC
//! end wall, where the port sits (an x-directed [`LumpedPort`] over the
//! whole end cross-section, 2 series edges × 2 columns), into a CPML.
//! With `R = Z` the port is matched: |S11| < −30 dB. With `R = 2Z` the
//! port must see the same `Z_in` (≈ `Z`, exactly as below) and
//! `S11 ≈ −1/3`.
//!
//! # Reference plane
//!
//! The port current is the loop integral of H around the port edges' dual
//! face (`LumpedPort` docs), i.e. the resistor current minus the
//! displacement current of the port edges' own (half-cell) capacitance:
//! that capacitance belongs to the port, and the current is effectively
//! taken at the first H plane, `Δz/2` into the line, while the voltage is
//! taken at `z = 0`. For a pure forward wave of the discrete line this
//! gives exactly `Z_in = Z·e^{jK/2}` (`K = k̃Δz`, the Yee phase advance
//! per cell), so a perfectly matched port reads `|S11| = tan(K/4)`, first
//! order in `kΔz` (−31.6 dB at 10 GHz with 0.5 mm cells, −37.6 dB with
//! the 0.25 mm used here). The power actually reflected towards the
//! resistor is second order: the half-cell capacitance in parallel with
//! the line is the discrete line's real mid-shunt image impedance
//! `Z/cos(K/2)`. The test checks both the §9 bound and the exact
//! discrete `Z_in`.

mod common;

use common::*;
use ems::constants::Z0;
use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::{Engine, FaceCondition, FaceConditions, Grid, LumpedBox, LumpedPort};
use ems::post::PortResult;
use ems::{Axis, Cpml, Face};

const H: f64 = 2e-3;
const W: f64 = 2e-3;
const DZ: f64 = 0.25e-3;
const NZ: usize = 408;

fn z_line() -> f64 {
    Z0 * H / W
}

/// Runs the line driven by a port of resistance `r`; returns the analysis
/// at `freqs` and the time step.
fn run(r: f64, freqs: &[f64]) -> (PortResult, f64) {
    let z: Vec<f64> = (0..=NZ).map(|q| q as f64 * DZ).collect();
    let g = Grid::new([vec![0.0, 0.5 * H, H], vec![0.0, W], z]).unwrap();
    let faces = FaceConditions::all(FaceCondition::Pec)
        .with(Face::YMin, FaceCondition::Pmc)
        .with(Face::YMax, FaceCondition::Pmc)
        .with(Face::ZMin, FaceCondition::Pmc);
    let mut e = Engine::builder(g).faces(faces).threads(1).build().unwrap();
    e.add_operator(CpmlBoundary::new(&e, [(Face::ZMax, Cpml::cells(8))]).unwrap());
    let region = LumpedBox::new(Axis::X, [0, 0, 0], [2, 1, 0]).unwrap();
    let h = LumpedPort::new(region, r)
        .excite(sine_pulse(5.5e9, 4e9))
        .add_to(&mut e)
        .unwrap();
    e.run(2000);
    let p = e.operator(&h);
    let v = p.voltage();
    let tail = peak(&v[v.len() - 20..]) / peak(v);
    assert!(tail < 1e-5, "record too short: {tail:e}");
    (p.analyze(freqs).unwrap(), e.dt())
}

/// `Z_in = Z·e^{jK/2}` of the discrete line (see the module docs).
fn z_discrete(f: f64, dt: f64) -> ems::Complex64 {
    let k = yee_k(f, dt, DZ, 1.0, 0.0).re;
    ems::Complex64::cis(0.5 * k) * z_line()
}

#[test]
fn lumped_port_matched_parallel_plate() {
    let _t = Timer::start("lumped_port_matched_parallel_plate");
    let z = z_line();
    let freqs = linspace(1e9, 10e9, 19);
    let (m, dt) = run(z, &freqs);
    let s = m.s11();
    let zin = m.z_in();
    let worst = s.iter().map(|s| s.norm()).fold(0.0, f64::max);
    let zerr = zin
        .iter()
        .map(|zi| (*zi - z).norm() / z)
        .fold(0.0, f64::max);
    let zd = freqs
        .iter()
        .zip(&zin)
        .map(|(&f, zi)| (*zi / z_discrete(f, dt) - 1.0).norm())
        .fold(0.0, f64::max);
    eprintln!(
        "matched port R = Z = {z:.2} Ω: max |S11| = {:.1} dB over 1–10 GHz \
         (1 GHz {:.1} dB, 10 GHz {:.1} dB); max |Z_in − Z|/Z = {zerr:.2e}, \
         max |Z_in/(Z·e^(jK/2)) − 1| = {zd:.2e}",
        db(worst),
        db(s[0].norm()),
        db(s[s.len() - 1].norm()),
    );
    assert!(db(worst) < -30.0, "|S11| = {:.1} dB", db(worst));
    assert!(zd < 1e-3, "{zd}");

    // Mismatched port: the line still presents Z, S11 = (Z − 2Z)/(Z + 2Z).
    let (m2, _) = run(2.0 * z, &freqs);
    let (mut es, mut ez) = (0.0f64, 0.0f64);
    for ((s, zi), &f) in m2.s11().iter().zip(m2.z_in()).zip(&freqs) {
        es = es.max((*s - (-1.0 / 3.0)).norm());
        ez = ez.max((zi / z_discrete(f, dt) - 1.0).norm());
    }
    eprintln!("port R = 2Z: max |S11 + 1/3| = {es:.2e}, max |Z_in/(Z·e^(jK/2)) − 1| = {ez:.2e}");
    assert!(es < 0.03 && ez < 1e-3, "{es} {ez}");
}
