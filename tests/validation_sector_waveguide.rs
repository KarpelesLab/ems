//! Analytic validation (design §5.5, §9): modes of an empty **sectoral
//! waveguide** with [`SectorWaveguidePort`]s.
//!
//! A guide along z whose cross section is a circular sector of radius
//! `b` = 10 mm and opening `φ_s` (a cylindrical sector grid with the
//! axis; the α faces and `ρ_max` are PEC walls), CPML at both ends.
//! Port 1 excites the mode near the start, port 2 looks back from the far
//! end. Between the reference planes (distance `L`) the ideal result is
//! `S11 = 0`, `S21 = e^{−jβL}`, `β = √(k² − k_c²)` with `k_c·b` a zero of
//! `J_ν` or `J_ν'` of the fractional order `ν = mπ/φ_s` (for `φ_s = 2π/3`:
//! TE11 `ν = 3/2`, `k_c·b = j'_{3/2,1} = 2.4605`, cutoff 11.75 GHz; the
//! next modes TE01 at 18.3 GHz and TE21 at 20.1 GHz).
//!
//! As in `tests/validation_circ_waveguide.rs`, each run also reports the
//! effective cutoff of the discrete mode from the z dispersion relation
//! of the Yee scheme,
//! `k_c,eff² = sin²(ωΔt/2)/(cΔt/2)² − sin²(βΔz/2)/(Δz/2)²`: its offset
//! from `k_c` is the error of the discretised cross section (second order
//! in the cells), its spread over frequency the port's own error.

use core::f64::consts::PI;

use csx::{ContinuousStructure, CoordSystem};
use ems::constants::C0;
use ems::port::{SectorMode, SectorWaveguidePort};
use ems::post::{Complex64, s_param};
use ems::{Boundaries, Boundary, EndCriteria, Face, Fdtd, Simulation, Waveform};

const B: f64 = 10.0;

struct Run {
    freqs: Vec<f64>,
    s11: Vec<Complex64>,
    s21: Vec<Complex64>,
    l: f64,
    beta: Vec<f64>,
    kc: f64,
    dt: f64,
    dz: f64,
}

/// A matched guide of opening `span` with `nr` × `na` cells, the mode
/// `mode`, 1 mm cells along z over `len` mm.
fn run(span: f64, nr: usize, na: usize, mode: SectorMode, freqs: &[f64], f0: f64) -> Run {
    let (dz, len): (f64, f64) = (1.0, 50.0);
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid
        .set_lines(0, (0..=nr).map(|i| B * i as f64 / nr as f64))
        .unwrap();
    m.grid
        .set_lines(1, (0..=na).map(|j| j as f64 * span / na as f64))
        .unwrap();
    m.grid
        .set_lines(2, (0..=(len / dz).round() as usize).map(|i| i as f64 * dz))
        .unwrap();
    let pml = 8;
    let z1 = (pml as f64 + 2.0) * dz;
    let (r1, r2) = (z1 + 4.0 * dz, len - z1 - 4.0 * dz);
    let p1 = SectorWaveguidePort::new(1, z1, r1, B, mode)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let p2 = SectorWaveguidePort::new(2, len - z1, r2, B, mode)
        .add_to(&mut m)
        .unwrap();
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(f0, 3e9))
        .boundaries(
            Boundaries::all(Boundary::Pec)
                .with(Face::ZMin, Boundary::pml(pml))
                .with(Face::ZMax, Boundary::pml(pml)),
        )
        .end_criteria(EndCriteria::energy_decay_db(-50.0))
        .max_timesteps(60_000)
        .threads(4);
    let mut sim = Simulation::new(&m, &fdtd).unwrap();
    let w: Vec<_> = sim
        .warnings()
        .iter()
        .filter(|w| !w.starts_with("mesh "))
        .cloned()
        .collect();
    assert!(w.is_empty(), "{w:?}");
    let res = sim.run().unwrap();
    let a1 = p1.analyze(&res, freqs).unwrap();
    let s21 = s_param(&p2.analyze(&res, freqs).unwrap(), &a1);
    Run {
        freqs: freqs.to_vec(),
        s11: a1.s11(),
        s21,
        l: (r2 - r1) * 1e-3,
        beta: freqs.iter().map(|&f| p1.beta(f).unwrap()).collect(),
        kc: p1.cutoff_wavenumber().unwrap(),
        dt: res.dt(),
        dz: dz * 1e-3,
    }
}

/// β from the phase `−βl` of `s`, unwrapped with the analytic `beta`.
fn beta_from_phase(s: Complex64, l: f64, beta: f64) -> f64 {
    let mut phi = -s.im.atan2(s.re);
    phi += 2.0 * PI * ((beta * l - phi) / (2.0 * PI)).round();
    phi / l
}

/// Prints and checks a matched run; returns (worst |S11| dB, mean
/// relative `k_c,eff` error, its spread).
fn check_matched(r: &Run, s11_db: f64, beta_tol: f64, s21_tol: f64) -> (f64, f64, f64) {
    let mut worst = f64::NEG_INFINITY;
    let mut kcs = Vec::new();
    for (k, &f) in r.freqs.iter().enumerate() {
        let db = 20.0 * r.s11[k].norm().log10();
        let beta = beta_from_phase(r.s21[k], r.l, r.beta[k]);
        let err = beta / r.beta[k] - 1.0;
        let sq = |x: f64| x.sin().powi(2);
        let kc = (sq(PI * f * r.dt) / (0.5 * C0 * r.dt).powi(2)
            - sq(0.5 * beta * r.dz) / (0.5 * r.dz).powi(2))
        .sqrt();
        kcs.push(kc / r.kc - 1.0);
        eprintln!(
            "f = {:.2} GHz: |S11| = {db:.1} dB, |S21| = {:.4}, β = {beta:.3} rad/m \
             (analytic {:.3}, {:+.3}%), k_c,eff/k_c − 1 = {:+.3}%",
            f / 1e9,
            r.s21[k].norm(),
            r.beta[k],
            100.0 * err,
            100.0 * (kc / r.kc - 1.0)
        );
        worst = worst.max(db);
        assert!(db < s11_db, "{f}: |S11| = {db} dB");
        assert!(
            (r.s21[k].norm() - 1.0).abs() < s21_tol,
            "{f}: |S21| = {}",
            r.s21[k].norm()
        );
        assert!(err.abs() < beta_tol, "{f}: β error {err}");
    }
    let mean = kcs.iter().sum::<f64>() / kcs.len() as f64;
    let spread = kcs.iter().map(|x| (x - mean).abs()).fold(0.0, f64::max);
    eprintln!(
        "worst |S11| {worst:.1} dB, k_c,eff error {:+.3}% (spread ±{:.3}%)",
        100.0 * mean,
        100.0 * spread
    );
    (worst, mean, spread)
}

/// TE11 of a 120° sector (`ν = 3/2`, a Bessel function of half-integer
/// order) on a coarse grid (8 radial × 8 azimuthal cells).
#[test]
fn te11_120_degree_sector_matched() {
    let freqs = [13e9, 14e9, 15e9, 16e9, 17e9];
    let r = run(2.0 * PI / 3.0, 8, 8, SectorMode::Te(1, 1), &freqs, 15e9);
    assert!((r.kc * B * 1e-3 - 2.460_536).abs() < 1e-5, "{}", r.kc);
    let (_, kc_err, spread) = check_matched(&r, -30.0, 0.03, 0.02);
    assert!(kc_err.abs() < 0.015 && spread < 3e-3, "{kc_err} ± {spread}");
}

/// The effective cutoff converges to `j'_{3/2,1}/b` at second order:
/// halving the cells divides its error by about four.
#[test]
#[ignore = "slow in debug builds; run with --release --ignored"]
fn te11_120_degree_sector_converges() {
    let freqs = [13e9, 14e9, 15e9, 16e9, 17e9];
    let coarse = run(2.0 * PI / 3.0, 8, 8, SectorMode::Te(1, 1), &freqs, 15e9);
    let fine = run(2.0 * PI / 3.0, 16, 16, SectorMode::Te(1, 1), &freqs, 15e9);
    let (_, e1, _) = check_matched(&coarse, -30.0, 0.03, 0.02);
    let (_, e2, _) = check_matched(&fine, -35.0, 0.01, 0.01);
    eprintln!("k_c,eff error {e1:+.4e} → {e2:+.4e} (ratio {:.2})", e1 / e2);
    assert!((3.0..5.5).contains(&(e1 / e2)), "{e1} {e2}");
}

/// TM11 of a 72° sector (`ν = 5/2`, `k_c·b = j_{5/2,1}`, cutoff 27.5 GHz;
/// the guide is over-moded there, TE11 and TE01 propagate from 17.4 and
/// 18.3 GHz, which the mode-shaped port does not couple to). Near the
/// cutoff `β` magnifies the cross section's cutoff error
/// (`dβ/β = −(k_c/β)²·dk_c/k_c`), so the check is on `k_c,eff`.
#[test]
fn tm11_72_degree_sector_matched() {
    let freqs = [29e9, 31e9, 33e9, 35e9];
    let r = run(0.4 * PI, 12, 8, SectorMode::Tm(1, 1), &freqs, 32e9);
    // j_{5/2,1}: the first root of tan x = 3x/(3 − x²).
    assert!((r.kc * B * 1e-3 - 5.763_459_197).abs() < 1e-8, "{}", r.kc);
    let (_, kc_err, spread) = check_matched(&r, -25.0, 0.1, 0.03);
    assert!(kc_err.abs() < 0.01 && spread < 3e-3, "{kc_err} ± {spread}");
}
