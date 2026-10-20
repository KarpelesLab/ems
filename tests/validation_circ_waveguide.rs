//! Analytic validation (design §9): modes of an empty circular waveguide
//! with [`CircWaveguidePort`]s.
//!
//! A guide of radius `a` = 10 mm along z with CPML at both ends. Port 1
//! excites the mode near the start; port 2 is a passive port near the far
//! end, looking back. Between the reference planes (distance `L`) the
//! ideal result is `S11 = 0`, `S21 = e^{−jβL}`, `β = √(k² − k_c²)`. With
//! the far end shorted instead (PEC at distance `d` from port 1's
//! reference plane), `S11 = −e^{−2jβd}`.
//!
//! Besides the analytic `β`, each run reports the **effective cutoff** of
//! the discrete mode, from the z dispersion relation of the Yee scheme
//! with the extracted `β`,
//!
//! ```text
//! k_c,eff² = sin²(ωΔt/2)/(cΔt/2)² − sin²(βΔz/2)/(Δz/2)²,
//! ```
//!
//! which is constant over frequency for a clean single mode: its deviation
//! from `p/a` is the error of the discretised cross section (azimuthal
//! sampling and the radial cells on a cylindrical grid, the staircased
//! wall on a Cartesian grid), its spread the port's own error.
//!
//! Grids: cylindrical (axis on ρ = 0, closed azimuth, the wall is the
//! `ρ_max` PEC face) and Cartesian (a PEC block with a vacuum cylinder
//! carved out, staircased).

use core::f64::consts::{PI, TAU};

use csx::primitives::{BoxShape, Cylinder};
use csx::properties::Material;
use csx::{ContinuousStructure, CoordSystem, Primitive, Property};
use ems::constants::C0;
use ems::port::{CircMode, CircWaveguidePort};
use ems::post::{Complex64, s_param};
use ems::{Boundaries, Boundary, CompileOptions, EndCriteria, Face, Fdtd, Simulation, Waveform};

const A: f64 = 10.0;

#[derive(Clone, Copy)]
enum Mesh {
    /// Cylindrical: `nr` radial cells over `a`, `na` azimuthal cells.
    Cyl { nr: usize, na: usize },
    /// Cartesian cells of `d` mm, wall staircased or conformal.
    Cart { d: f64, conformal: bool },
}

struct Run {
    freqs: Vec<f64>,
    s11: Vec<Complex64>,
    s21: Option<Vec<Complex64>>,
    /// Distance between the reference planes, or to the short \[m\].
    l: f64,
    beta: Vec<f64>,
    kc: f64,
    dt: f64,
    dz: f64,
}

fn run(mesh: Mesh, mode: CircMode, freqs: &[f64], f0: f64, len: f64, shorted: bool) -> Run {
    run_with(mesh, mode, freqs, f0, len, shorted, A)
}

/// [`run`] with the ports' radius `pr` (the guide's is `A`).
fn run_with(
    mesh: Mesh,
    mode: CircMode,
    freqs: &[f64],
    f0: f64,
    len: f64,
    shorted: bool,
    pr: f64,
) -> Run {
    let dz = 1.0;
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    let zl: Vec<f64> = (0..=(len / dz).round() as usize)
        .map(|i| i as f64 * dz)
        .collect();
    let mut opts = CompileOptions::default();
    match mesh {
        Mesh::Cyl { nr, na } => {
            m.coord_system = CoordSystem::Cylindrical;
            m.grid.coord_system = CoordSystem::Cylindrical;
            m.grid
                .set_lines(0, (0..=nr).map(|i| A * i as f64 / nr as f64))
                .unwrap();
            m.grid
                .set_lines(1, (0..=na).map(|j| j as f64 * TAU / na as f64))
                .unwrap();
        }
        Mesh::Cart { d, conformal } => {
            let n = ((A + 1.5) / d).ceil() as i64;
            for a in 0..2 {
                m.grid.set_lines(a, (-n..=n).map(|i| i as f64 * d)).unwrap();
            }
            let e = n as f64 * d;
            let metal = m.add_property(Property::metal("wall"));
            m.add_primitive(
                metal,
                Primitive::new(BoxShape::new([-e, -e, 0.0], [e, e, len])),
            )
            .unwrap();
            let air = m.add_property(Property::material("air", Material::new(1.0, 0.0)));
            m.add_primitive(
                air,
                Primitive::new(Cylinder::new([0.0, 0.0, 0.0], [0.0, 0.0, len], A))
                    .with_priority(10),
            )
            .unwrap();
            opts = opts.with_conformal(conformal);
        }
    }
    m.grid.set_lines(2, zl).unwrap();
    let pml = 8;
    let z1 = (pml as f64 + 2.0) * dz;
    let (r1, r2) = (z1 + 4.0 * dz, len - z1 - 4.0 * dz);
    let p1 = CircWaveguidePort::new(1, [0., 0., z1], [0., 0., r1], pr, mode)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let p2 = (!shorted).then(|| {
        CircWaveguidePort::new(2, [0., 0., len - z1], [0., 0., r2], pr, mode)
            .add_to(&mut m)
            .unwrap()
    });
    let far = if shorted {
        Boundary::Pec
    } else {
        Boundary::pml(pml)
    };
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(f0, 3e9))
        .boundaries(
            Boundaries::all(Boundary::Pec)
                .with(Face::ZMin, Boundary::pml(pml))
                .with(Face::ZMax, far),
        )
        .end_criteria(EndCriteria::energy_decay_db(-50.0))
        .max_timesteps(60_000)
        .threads(4);
    let mut sim = Simulation::with_options(&m, &fdtd, &opts).unwrap();
    let w: Vec<_> = sim
        .warnings()
        .iter()
        .filter(|w| !w.starts_with("mesh "))
        .cloned()
        .collect();
    assert!(w.is_empty(), "{w:?}");
    let res = sim.run().unwrap();
    eprintln!("dt = {:.3e} s, {:?}", res.dt(), res.outcome());
    let a1 = p1.analyze(&res, freqs).unwrap();
    let s21 = p2.map(|p2| s_param(&p2.analyze(&res, freqs).unwrap(), &a1));
    Run {
        freqs: freqs.to_vec(),
        s11: a1.s11(),
        s21,
        l: (if shorted { len - r1 } else { r2 - r1 }) * 1e-3,
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

/// The effective cutoff wavenumber of the discrete mode at `f` with the
/// measured `beta` (see the module docs).
fn kc_eff(r: &Run, f: f64, beta: f64) -> f64 {
    let s = |x: f64| x.sin().powi(2);
    let k2 = s(PI * f * r.dt) / (0.5 * C0 * r.dt).powi(2);
    let b2 = s(0.5 * beta * r.dz) / (0.5 * r.dz).powi(2);
    (k2 - b2).sqrt()
}

/// Prints and checks a matched run; returns (worst |S11| dB, k_c,eff
/// relative error mean, k_c,eff spread).
fn check_matched(r: &Run, s11_db: f64, beta_tol: f64, s21_tol: f64) -> (f64, f64, f64) {
    let s21 = r.s21.as_ref().unwrap();
    let mut worst = f64::NEG_INFINITY;
    let mut kcs = Vec::new();
    for (k, &f) in r.freqs.iter().enumerate() {
        let db = 20.0 * r.s11[k].norm().log10();
        let beta = beta_from_phase(s21[k], r.l, r.beta[k]);
        let err = beta / r.beta[k] - 1.0;
        let kc = kc_eff(r, f, beta);
        kcs.push(kc / r.kc - 1.0);
        eprintln!(
            "f = {:.2} GHz: |S11| = {db:.1} dB, |S21| = {:.4}, β = {beta:.3} rad/m \
             (analytic {:.3}, {:+.3}%), k_c,eff/k_c − 1 = {:+.3}%",
            f / 1e9,
            s21[k].norm(),
            r.beta[k],
            100.0 * err,
            100.0 * (kc / r.kc - 1.0)
        );
        worst = worst.max(db);
        assert!(db < s11_db, "{f}: |S11| = {db} dB");
        assert!(
            (s21[k].norm() - 1.0).abs() < s21_tol,
            "{f}: |S21| = {}",
            s21[k].norm()
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

/// TE11 on a coarse cylindrical grid (8 radial × 12 azimuthal cells,
/// 1 mm along z): fast enough for debug builds.
#[test]
fn te11_matched_cylindrical() {
    let freqs = [10e9, 11e9, 12e9, 13e9];
    let r = run(
        Mesh::Cyl { nr: 8, na: 12 },
        CircMode::Te(1, 1),
        &freqs,
        11.5e9,
        50.0,
        false,
    );
    // The coarse cross section lowers the cutoff by 0.8 %, i.e. β is 2.6 %
    // high at 10 GHz; the effective cutoff is the sharper check.
    let (_, kc_err, spread) = check_matched(&r, -30.0, 0.04, 0.02);
    assert!(kc_err.abs() < 0.015 && spread < 2e-3, "{kc_err} ± {spread}");
}

/// TE11 on a finer cylindrical grid (20 × 32 cells).
#[test]
#[ignore = "slow in debug builds; run with --release --ignored"]
fn te11_matched_cylindrical_fine() {
    let freqs = [10e9, 11e9, 12e9, 13e9];
    let r = run(
        Mesh::Cyl { nr: 20, na: 32 },
        CircMode::Te(1, 1),
        &freqs,
        11.5e9,
        80.0,
        false,
    );
    check_matched(&r, -35.0, 0.01, 0.01);
}

/// TM01 (`p01 = 2.405`, f_c = 11.47 GHz) and TE01 (`p'01 = 3.832`,
/// f_c = 18.28 GHz) on the cylindrical grid: azimuthally uniform, so the
/// azimuthal sampling does not enter.
#[test]
#[ignore = "slow in debug builds; run with --release --ignored"]
fn tm01_te01_matched_cylindrical() {
    let r = run(
        Mesh::Cyl { nr: 20, na: 16 },
        CircMode::Tm(0, 1),
        &[13e9, 14e9, 15e9, 16e9],
        14.5e9,
        80.0,
        false,
    );
    check_matched(&r, -35.0, 0.01, 0.01);
    let r = run(
        Mesh::Cyl { nr: 20, na: 16 },
        CircMode::Te(0, 1),
        &[20e9, 21e9, 22e9, 23e9],
        21.5e9,
        80.0,
        false,
    );
    check_matched(&r, -35.0, 0.01, 0.01);
}

/// TE11 on a Cartesian grid (0.5 mm cells) with a staircased wall. Its
/// discrete cutoff is a few percent low, so near the cutoff the analytic
/// `β` and `Z_TE` are off (β is checked through `k_c,eff` instead). A
/// port radius tuned to the measured cutoff (`a·k_c/k_c,eff`) removes most
/// of the mismatch.
#[test]
#[ignore = "slow in debug builds; run with --release --ignored"]
fn te11_matched_cartesian_staircase() {
    let freqs = [10e9, 11e9, 12e9, 13e9];
    let mesh = Mesh::Cart {
        d: 0.5,
        conformal: false,
    };
    let r = run_with(mesh, CircMode::Te(1, 1), &freqs, 11.5e9, 60.0, false, A);
    let (s11, kc_err, _) = check_matched(&r, -20.0, 0.2, 0.02);
    assert!((-0.06..-0.01).contains(&kc_err), "{kc_err}");
    let tuned = A / (1.0 + kc_err);
    eprintln!("port radius tuned to {tuned:.3} mm");
    let r = run_with(mesh, CircMode::Te(1, 1), &freqs, 11.5e9, 60.0, false, tuned);
    // The staircased guide's own k_c,eff varies by ±0.5 % over the band
    // (the same β as with the untuned port), so the tuned radius matches
    // its mean only.
    let (s11_tuned, kc_err, _) = check_matched(&r, -35.0, 0.03, 0.01);
    assert!(s11_tuned < s11 - 10.0, "{s11_tuned} vs {s11}");
    assert!(kc_err.abs() < 1e-3, "{kc_err}");
}

#[test]
#[ignore = "slow in debug builds; run with --release --ignored"]
fn te11_short_circuit_phase() {
    let freqs = [10e9, 11e9, 12e9, 13e9];
    let r = run(
        Mesh::Cyl { nr: 20, na: 32 },
        CircMode::Te(1, 1),
        &freqs,
        11.5e9,
        60.0,
        true,
    );
    for (k, &f) in freqs.iter().enumerate() {
        let s = r.s11[k];
        let beta = beta_from_phase(-s, 2.0 * r.l, r.beta[k]);
        let err = beta / r.beta[k] - 1.0;
        eprintln!(
            "f = {:.1} GHz: |S11| = {:.4}, β = {beta:.3} rad/m (analytic {:.3}, {:+.3}%)",
            f / 1e9,
            s.norm(),
            r.beta[k],
            100.0 * err
        );
        assert!((s.norm() - 1.0).abs() < 0.02, "{f}: |S11| = {}", s.norm());
        assert!(err.abs() < 0.01, "{f}: β error {err}");
    }
}
