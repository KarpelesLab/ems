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
//! The port current is the current of the port's Thevenin branch
//! (`LumpedPort` docs, "Reference plane"), so `V` and `I` are both defined
//! at the port edges and the edges' own half-cell capacitance belongs to
//! the line. The line then starts with a half-cell shunt section, whose
//! input impedance is the discrete line's real mid-shunt image impedance
//! `Z/cos(K/2)` (`K = k̃Δz`, the Yee phase advance per cell): a matched
//! port reads `|S11| = tan²(K/4)`, second order in `kΔz` (−63 dB at
//! 10 GHz with 0.5 mm cells). (Measuring the H loop around the port edges
//! instead would put the current half a cell into the line,
//! `Z_in = Z·e^{jK/2}`, `|S11| = tan(K/4)`: first order, −31.6 dB.) The
//! test checks the exact discrete `Z_in`, the §9 bound and the
//! second-order convergence of `|S11|` from 0.5 mm to 0.25 mm cells.

mod common;

use common::*;
use ems::constants::Z0;
use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::{Engine, FaceCondition, FaceConditions, Grid, LumpedBox, LumpedPort};
use ems::post::PortResult;
use ems::{Axis, Cpml, Face};

const H: f64 = 2e-3;
const W: f64 = 2e-3;
/// Line length (CPML included) \[m\].
const LEN: f64 = 102e-3;
/// Simulated time \[s\] (2000 steps of the 0.25 mm mesh).
const T_END: f64 = 1.33e-9;

fn z_line() -> f64 {
    Z0 * H / W
}

/// Runs the line with cells `dz` driven by a port of resistance `r`;
/// returns the analysis at `freqs` and the time step.
fn run(r: f64, dz: f64, freqs: &[f64]) -> (PortResult, f64) {
    let nz = (LEN / dz).round() as usize;
    let z: Vec<f64> = (0..=nz).map(|q| q as f64 * dz).collect();
    let g = Grid::new([vec![0.0, 0.5 * H, H], vec![0.0, W], z]).unwrap();
    let faces = FaceConditions::all(FaceCondition::Pec)
        .with(Face::YMin, FaceCondition::Pmc)
        .with(Face::YMax, FaceCondition::Pmc)
        .with(Face::ZMin, FaceCondition::Pmc);
    let mut e = Engine::builder(g).faces(faces).threads(1).build().unwrap();
    let pml = (4e-3 / dz).round() as usize;
    e.add_operator(CpmlBoundary::new(&e, [(Face::ZMax, Cpml::cells(pml))]).unwrap());
    let region = LumpedBox::new(Axis::X, [0, 0, 0], [2, 1, 0]).unwrap();
    let h = LumpedPort::new(region, r)
        .excite(sine_pulse(5.5e9, 4e9))
        .add_to(&mut e)
        .unwrap();
    e.run((T_END / e.dt()).ceil() as u64);
    let p = e.operator(&h);
    let v = p.voltage();
    let tail = peak(&v[v.len() - 20..]) / peak(v);
    assert!(tail < 1e-5, "record too short: {tail:e}");
    (p.analyze(freqs).unwrap(), e.dt())
}

/// `Z_in = Z/cos(K/2)` of the discrete line (see the module docs).
fn z_discrete(f: f64, dt: f64, dz: f64) -> f64 {
    let k = yee_k(f, dt, dz, 1.0, 0.0).re;
    z_line() / (0.5 * k).cos()
}

#[test]
fn lumped_port_matched_parallel_plate() {
    let _t = Timer::start("lumped_port_matched_parallel_plate");
    let z = z_line();
    let freqs = linspace(1e9, 10e9, 19);
    let mut s10 = Vec::new();
    for dz in [0.5e-3, 0.25e-3] {
        let (m, dt) = run(z, dz, &freqs);
        let s = m.s11();
        let zin = m.z_in();
        let worst = s.iter().map(|s| s.norm()).fold(0.0, f64::max);
        let zd = freqs
            .iter()
            .zip(&zin)
            .map(|(&f, zi)| (*zi / z_discrete(f, dt, dz) - 1.0).norm())
            .fold(0.0, f64::max);
        let k10 = yee_k(10e9, dt, dz, 1.0, 0.0).re;
        let exact = (0.25 * k10).tan().powi(2);
        let last = s[s.len() - 1].norm();
        eprintln!(
            "dz = {:.2} mm, matched port R = Z = {z:.2} Ω: max |S11| = {:.1} dB over \
             1–10 GHz (1 GHz {:.1} dB, 10 GHz {:.1} dB, tan²(K/4) = {:.1} dB); \
             max |Z_in/(Z/cos(K/2)) − 1| = {zd:.2e}",
            dz * 1e3,
            db(worst),
            db(s[0].norm()),
            db(last),
            db(exact),
        );
        assert!(db(worst) < -55.0, "|S11| = {:.1} dB", db(worst));
        assert!(zd < 2e-4, "{zd}");
        s10.push(last);
    }
    let rate = (s10[0] / s10[1]).log2();
    eprintln!("|S11| at 10 GHz: convergence rate {rate:.2} (0.5 → 0.25 mm)");
    assert!((rate - 2.0).abs() < 0.3, "rate {rate}");

    // Mismatched port: the line presents Z/cos(K/2) ≈ Z, S11 ≈ −1/3.
    let dz = 0.25e-3;
    let (m2, dt) = run(2.0 * z, dz, &freqs);
    let (mut es, mut ez) = (0.0f64, 0.0f64);
    for ((s, zi), &f) in m2.s11().iter().zip(m2.z_in()).zip(&freqs) {
        let zd = z_discrete(f, dt, dz);
        let want = (zd - 2.0 * z) / (zd + 2.0 * z);
        es = es.max((*s - want).norm());
        ez = ez.max((zi / zd - 1.0).norm());
    }
    eprintln!(
        "port R = 2Z: max |S11 − S11_exact| = {es:.2e}, max |Z_in/(Z/cos(K/2)) − 1| = {ez:.2e}"
    );
    assert!(es < 1e-4 && ez < 2e-4, "{es} {ez}");
}
