//! Validation of the radial-line port ([`RadialLinePort`], design §5.5,
//! §5.15) against the analytic cylindrical waves of a parallel-plate
//! radial line (Marcuvitz, *Waveguide Handbook*, §2.6): `V ∝ H₀(kρ)`, the
//! outgoing wave impedance `Z₊ = j·η·h/(Δα·ρ)·H₀⁽²⁾/H₁⁽²⁾`.
//!
//! The plates are the PEC `z` faces of a cylindrical grid two cells
//! apart; a feed ring uniform in α and across the gap launches exactly
//! the TM₀₀ mode, which the Yee scheme keeps uniform in `z` and α, so the
//! runs are 1-D radial lines in 3-D form.
//!
//! 1. **Matched line** (outward port, PMC `ρ_min` face as an open inner
//!    end at the feed, cylindrical PML on `ρ_max`): `Γ ≈ 0`, `V/I` at the
//!    reference radius is the analytic `Z₊`, the five probe values fit
//!    the two cylindrical waves.
//! 2. **Short at `ρ = b`** (PEC `ρ_max` face): `Γ(r) = −H₀⁽²⁾(kb)·H₀⁽¹⁾(kr)
//!    / (H₀⁽¹⁾(kb)·H₀⁽²⁾(kr))`, `|Γ| = 1`.
//! 3. **Quarter sector** (PMC α walls): the same as 1 with `Δα = π/2`.
//!    The feed resistor is split so that the two wall columns (half-width
//!    dual cells) take half the current of an inner one; with a single
//!    element over all columns the feed is not uniform in α and its
//!    azimuthal near field reached the probes at −45 dB (fit residual
//!    1e-3 at 16 GHz), with the split −59 dB as on the full line.
//! 4. **Inward port** towards the axis (grid with the axis): the
//!    converging wave returns from the axis as an outgoing one, standing
//!    wave `J₀`, `Γ(r) = H₀⁽²⁾(kr)/H₀⁽¹⁾(kr)`.

mod common;

use core::f64::consts::{PI, TAU};

use common::*;
use csx::{ContinuousStructure, CoordSystem};
use ems::constants::C0;
use ems::port::{RadialLinePort, RadialLinePortResult};
use ems::prelude::*;

fn lines(from: f64, to: f64, step: f64) -> Vec<f64> {
    let n = ((to - from) / step).round() as usize;
    (0..=n).map(|k| from + k as f64 * step).collect()
}

/// A cylindrical model: ρ lines `rho`, `n_alpha` α cells over `span`,
/// plates at z = 0 and 2 mm (two cells).
fn model(rho: Vec<f64>, n_alpha: usize, span: f64) -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, rho).unwrap();
    m.grid
        .set_lines(1, (0..=n_alpha).map(|j| j as f64 * span / n_alpha as f64))
        .unwrap();
    m.grid.set_lines(2, [0.0, 1.0, 2.0]).unwrap();
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

fn fdtd(xmin: Boundary, xmax: Boundary) -> Fdtd {
    Fdtd::new()
        .waveform(Waveform::gaussian(10e9, 8e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::XMin, xmin)
        .boundary(Face::XMax, xmax)
        .end_criteria(EndCriteria::energy_decay_db(-60.0))
        .max_timesteps(40_000)
}

/// `(H₀⁽¹⁾(x), H₀⁽²⁾(x))` from the ascending series of `J₀` and `Y₀`
/// (Abramowitz & Stegun 9.1.10, 9.1.13), independent of the port's
/// quadrature; accurate to ~1e-12 for the `x ≤ 12` used here.
fn hankel0(x: f64) -> (Complex64, Complex64) {
    // (`f64::consts::EULER_GAMMA` is newer than the MSRV.)
    #[allow(clippy::approx_constant)]
    const EULER: f64 = 0.577_215_664_901_532_9;
    let q = x * x / 4.0;
    let (mut j0, mut s, mut term, mut harm) = (0.0, 0.0, 1.0, 0.0);
    for k in 0..80 {
        if k > 0 {
            let kf = f64::from(k);
            term *= -q / (kf * kf);
            harm += 1.0 / kf;
        }
        j0 += term;
        // Σ (−1)^{k+1} H_k (x²/4)^k/(k!)² = −Σ H_k·term.
        s -= harm * term;
    }
    let y0 = 2.0 / PI * (((x / 2.0).ln() + EULER) * j0 + s);
    (Complex64::new(j0, y0), Complex64::new(j0, -y0))
}

fn worst_db(r: &RadialLinePortResult) -> f64 {
    r.reflection()
        .iter()
        .map(|g| db(g.norm()))
        .fold(f64::NEG_INFINITY, f64::max)
}

/// The discrete wavenumber used by the fit, for an analytic reference
/// that shares the 1-D dispersion of the scheme.
fn k_of(r: &RadialLinePortResult, q: usize) -> f64 {
    r.wavenumber()[q]
}

#[test]
fn radial_port_matched_line() {
    let rho = lines(2.0, 50.0, 0.5);
    let mut m = model(rho, 4, TAU);
    let p = RadialLinePort::new(1, [0.0, 2.0], 2.0, 20.0)
        .feed_resistance(50.0)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    assert_eq!(p.probe_positions(), Some([10.5, 11.0, 11.5]));
    let res = run(&m, &fdtd(Boundary::Pmc, Boundary::pml(8)));
    let freqs = linspace(4e9, 16e9, 7);
    let r = p.analyze(&res, &freqs).unwrap();
    let (mut ez, mut er) = (0.0f64, 0.0f64);
    for (q, &f) in freqs.iter().enumerate() {
        let z_an = p.wave_impedance(f, 20.0).unwrap();
        let z = r.z_in()[q];
        ez = ez.max((z - z_an).norm() / z_an.norm());
        er = er.max(r.fit_residual()[q]);
        // Power balance of the normalised waves.
        let (a, b) = (r.a()[q], r.b()[q]);
        let pw = 0.5 * (a.norm_sqr() - b.norm_sqr());
        assert!((pw - r.accepted_power()[q]).abs() < 1e-9 * pw.abs());
    }
    let g = worst_db(&r);
    println!(
        "radial matched line: |Γ| ≤ {g:.1} dB, |V/I − Z₊| ≤ {ez:.2e}, fit residual ≤ {er:.1e}"
    );
    assert!(g < -55.0, "|Γ| = {g} dB");
    assert!(ez < 1e-2, "{ez}");
    assert!(er < 1e-2, "{er}");
}

#[test]
fn radial_port_short_circuit() {
    let b = 30.0;
    let rho = lines(2.0, b, 0.5);
    let mut m = model(rho, 4, TAU);
    let p = RadialLinePort::new(1, [0.0, 2.0], 2.0, 20.0)
        .feed_resistance(50.0)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let res = run(
        &m,
        &fdtd(Boundary::Pmc, Boundary::Pec).end_criteria(EndCriteria::energy_decay_db(-50.0)),
    );
    let freqs = linspace(4e9, 16e9, 7);
    let r = p.analyze(&res, &freqs).unwrap();
    let (mut em, mut ep) = (0.0f64, 0.0f64);
    for (q, _) in freqs.iter().enumerate() {
        let k = k_of(&r, q);
        let (h1b, h2b) = hankel0(k * b * 1e-3);
        let (h1r, h2r) = hankel0(k * 20e-3);
        let want = -(h2b * h1r) / (h1b * h2r);
        let g = r.reflection()[q];
        em = em.max((g.norm() - 1.0).abs());
        ep = ep.max((g / want).arg().abs());
    }
    println!("radial short: ||Γ| − 1| ≤ {em:.2e}, phase error ≤ {ep:.2e} rad");
    assert!(em < 1e-2, "{em}");
    assert!(ep < 2e-2, "{ep}");
}

#[test]
fn radial_port_quarter_sector() {
    let rho = lines(2.0, 50.0, 0.5);
    let mut m = model(rho, 2, PI / 2.0);
    let p = RadialLinePort::new(1, [0.0, 2.0], 2.0, 20.0)
        .feed_resistance(50.0)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let f = fdtd(Boundary::Pmc, Boundary::pml(8))
        .boundary(Face::YMin, Boundary::Pmc)
        .boundary(Face::YMax, Boundary::Pmc);
    let res = run(&m, &f);
    let freqs = linspace(4e9, 16e9, 7);
    let r = p.analyze(&res, &freqs).unwrap();
    let g = worst_db(&r);
    // The sector's wave impedance is 4× the full line's.
    let z = p.wave_impedance(10e9, 20.0).unwrap();
    let eta_h = ems::constants::Z0 * 2e-3;
    let zw = eta_h / (PI / 2.0 * 20e-3);
    println!("radial quarter sector: |Γ| ≤ {g:.1} dB, Z₊(10 GHz) = {z} (η·h/(Δα·ρ) = {zw:.2})");
    assert!(g < -55.0, "|Γ| = {g} dB");
    assert!((z.re / zw - 1.0).abs() < 0.05);
}

#[test]
fn radial_port_inward_to_the_axis() {
    let rho = lines(0.0, 50.0, 0.5);
    let mut m = model(rho, 4, TAU);
    // A soft ring source at 30 mm; the port looks inwards to 12 mm.
    let p = RadialLinePort::new(1, [0.0, 2.0], 30.0, 12.0)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    assert_eq!(p.probe_positions(), Some([21.5, 21.0, 20.5]));
    let res = run(&m, &fdtd(Boundary::Pec, Boundary::pml(8)));
    let freqs = linspace(4e9, 16e9, 7);
    let r = p.analyze(&res, &freqs).unwrap();
    let (mut em, mut ep) = (0.0f64, 0.0f64);
    for (q, &f) in freqs.iter().enumerate() {
        let k = k_of(&r, q);
        assert!((k / (TAU * f / C0) - 1.0).abs() < 2e-3);
        let (h1, h2) = hankel0(k * 12e-3);
        let want = h2 / h1;
        let g = r.reflection()[q];
        em = em.max((g.norm() - 1.0).abs());
        ep = ep.max((g / want).arg().abs());
    }
    println!("radial inward port: ||Γ| − 1| ≤ {em:.2e}, phase error ≤ {ep:.2e} rad");
    assert!(em < 1e-2, "{em}");
    assert!(ep < 2e-2, "{ep}");
}
