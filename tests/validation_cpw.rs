//! Validation of the coplanar-waveguide port ([`CpwPort`], design §5.5 /
//! §9). Centre strip `S` = 1 mm, gaps `W` = 0.5 mm (`k0 = 1/2`), grounds
//! out to PEC side walls at `|y|` = 3 mm, zero-thickness metal on `z = 0`
//! meshed with the thirds rule at all four edges.
//!
//! 1. **Shielded CPW in air** (PEC covers at `z = ±H`): a homogeneous TEM
//!    line, so `β` must be the 1-D Yee wavenumber (within 2e-4, what is
//!    left of the feed's near field at port 1) and `Z_L`
//!    the conformal-mapping value of a CPW between two covers,
//!    `Z = (η0/4)·K(k3')/K(k3)`, `k3 = tanh(πS/4H)/tanh(π(S + 2W)/4H)`
//!    (each cover half is the lower half of the conductor-backed formula;
//!    obtained here from [`CpwPort::quasi_static_of`]).
//! 2. (ignored, heavier) **Conductor-backed CPW** (ε_r = 4.4, h = 0.5 mm
//!    on the PEC `z_min` face, air above with a CPML 8 mm away): `Z_L`
//!    and `ε_eff` against Simons' quasi-static formula.
//! 3. (ignored, heavier) **Conventional CPW** on a 0.5 mm substrate with
//!    CPML above and below, against Wen/Simons.
//!
//! All cases: two ports facing each other on a through line, `S11 ≈ 0`,
//! probes 15 mm from the feed (the feed's near field, cut-off modes of the
//! 6 mm wide box, decays as `e^{−0.57/mm·s}`).

mod common;

use common::*;
use csx::primitives::BoxShape;
use csx::properties::Material;
use csx::{ContinuousStructure, Primitive, Property};
use ems::constants::Z0;
use ems::port::{CpwPort, LinePortResult};
use ems::post::s_param;
use ems::prelude::*;

const S: f64 = 1.0;
const W: f64 = 0.5;
const Y: f64 = 3.0;
const LEN: f64 = 80.0;
const ZTOP: f64 = 8.0;

fn lines(from: f64, to: f64, step: f64) -> Vec<f64> {
    let n = ((to - from) / step).round() as usize;
    (0..=n).map(|k| from + k as f64 * step).collect()
}

/// Width lines: uniform `dy` with the thirds rule at `±S/2` and
/// `±(S/2 + W)` (`S/2 = (2 + 1/3)·dy`), graded up to `4·dy` out to `±Y`.
fn width_lines() -> Vec<f64> {
    let dy = 0.5 * S / (2.0 + 1.0 / 3.0);
    let mut y = lines(0.0, 6.0 * dy, dy);
    let mut step = dy;
    while *y.last().unwrap() < Y - 1e-9 {
        step = (step * 1.3).min(4.0 * dy);
        let next = (*y.last().unwrap() + step).min(Y);
        y.push(next);
    }
    let mut all: Vec<f64> = y.iter().rev().map(|v| -v).collect();
    all.extend_from_slice(&y[1..]);
    all
}

struct Case {
    /// Cell size along the line \[mm\].
    dx: f64,
    /// z lines (the metal plane is z = 0).
    z: Vec<f64>,
    /// Substrate `(ε_r, z_lo, z_hi)`, if any.
    sub: Option<(f64, f64, f64)>,
    zmin: Boundary,
    zmax: Boundary,
    /// Energy end criterion \[dB\].
    decay_db: f64,
}

struct Out {
    r1: LinePortResult,
    r2: LinePortResult,
    worst_s11: f64,
    dt: f64,
}

fn run(c: &Case, freqs: &[f64]) -> Out {
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    model.grid.set_lines(0, lines(0.0, LEN, c.dx)).unwrap();
    model.grid.set_lines(1, width_lines()).unwrap();
    model.grid.set_lines(2, c.z.clone()).unwrap();
    if let Some((eps, lo, hi)) = c.sub {
        let sub = model.add_property(Property::material("sub", Material::new(eps, 0.0)));
        model
            .add_primitive(
                sub,
                Primitive::new(BoxShape::new([0.0, -Y, lo], [LEN, Y, hi])),
            )
            .unwrap();
    }
    // Strip and grounds through both CPML ends.
    let cu = model.add_property(Property::metal("cpw"));
    for (a, b) in [(-Y, -0.5 * S - W), (-0.5 * S, 0.5 * S), (0.5 * S + W, Y)] {
        model
            .add_primitive(
                cu,
                Primitive::new(BoxShape::new([0.0, a, 0.0], [LEN, b, 0.0])),
            )
            .unwrap();
    }
    let z_lo = c.z[0];
    let p1 = CpwPort::new(1, [10.0, -Y, z_lo], [40.0, Y, 0.0], Axis::X, Axis::Z, S, W)
        .metal(cu)
        .feed_resistance(50.0)
        .excite(1.0)
        .add_to(&mut model)
        .unwrap();
    let p2 = CpwPort::new(
        2,
        [LEN - 10.0, -Y, z_lo],
        [48.0, Y, 0.0],
        Axis::X,
        Axis::Z,
        S,
        W,
    )
    .metal(cu)
    .add_to(&mut model)
    .unwrap();
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(4e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::XMin, Boundary::pml(8))
        .boundary(Face::XMax, Boundary::pml(8))
        .boundary(Face::ZMin, c.zmin)
        .boundary(Face::ZMax, c.zmax)
        .end_criteria(EndCriteria::energy_decay_db(c.decay_db))
        .max_timesteps(30_000)
        .threads(4);
    let mut sim = Simulation::new(&model, &fdtd).unwrap();
    let w: Vec<_> = sim
        .warnings()
        .iter()
        .filter(|w| !w.contains("mesh"))
        .cloned()
        .collect();
    assert!(w.is_empty(), "{w:?}");
    let res = sim.run().unwrap();
    let r1 = p1.analyze(&res, freqs).unwrap();
    let r2 = p2.analyze(&res, freqs).unwrap();
    let s11 = r1.s11();
    let s21 = s_param(&r2, &r1);
    let beta = r1.beta();
    let mut worst = f64::NEG_INFINITY;
    for k in 0..freqs.len() {
        worst = worst.max(db(s11[k].norm()));
        let want = Complex64::cis(-beta[k] * 8e-3);
        assert!(
            (s21[k] - want).norm() < 0.02,
            "f = {}: S21 = {}, want {want}",
            freqs[k],
            s21[k]
        );
    }
    Out {
        r1,
        r2,
        worst_s11: worst,
        dt: res.dt(),
    }
}

fn mean(v: impl IntoIterator<Item = f64>) -> f64 {
    let v: Vec<f64> = v.into_iter().collect();
    v.iter().sum::<f64>() / v.len() as f64
}

/// Mean `Re Z_L` and `ε_eff` of both ports over the first `n` frequencies.
fn line_values(o: &Out, n: usize, label: &str) -> (f64, f64) {
    let mut z = Vec::new();
    let mut e = Vec::new();
    for r in [&o.r1, &o.r2] {
        for k in 0..n {
            let zl = r.z_line()[k];
            println!(
                "{label}: {:.1} GHz: Z_L = {zl:.3}, ε_eff = {:.4}",
                r.freqs()[k] / 1e9,
                r.eps_eff()[k]
            );
            assert!(zl.im.abs() < 0.02 * zl.re, "{zl}");
            z.push(zl.re);
            e.push(r.eps_eff()[k]);
        }
    }
    (mean(z), mean(e))
}

#[test]
fn cpw_port_shielded_air_tem() {
    const H: f64 = 1.0;
    let case = Case {
        dx: 1.0,
        z: lines(-H, H, 0.2),
        sub: None,
        zmin: Boundary::Pec,
        zmax: Boundary::Pec,
        decay_db: -80.0,
    };
    let freqs = linspace(1e9, 7e9, 7);
    let o = run(&case, &freqs);
    // Cover-shielded CPW from the open and backed formulas:
    // Z_open = η/(4·r0), Z_backed = η/(2·(r0 + r3)), Z_shielded = η/(4·r3).
    let (z_open, _) = CpwPort::quasi_static_of(S, W, 0.0, 1.0, false);
    let (z_backed, _) = CpwPort::quasi_static_of(S, W, H, 1.0, true);
    let r0 = Z0 / (4.0 * z_open);
    let r3 = Z0 / (2.0 * z_backed) - r0;
    let z_want = Z0 / (4.0 * r3);
    let (z, _) = line_values(&o, freqs.len(), "shielded air CPW");
    // Homogeneous TEM line: the 1-D Yee wavenumber along x.
    let mut eb = 0.0f64;
    for r in [&o.r1, &o.r2] {
        for (k, &f) in freqs.iter().enumerate() {
            let k_yee = yee_k(f, o.dt, 1e-3, 1.0, 0.0).re / 1e-3;
            let bd = k_yee * 1e-3;
            let err = (r.beta()[k] / k_yee - 1.0).abs();
            eb = eb.max(err);
            assert!(err < 2e-4 + 5e-8 / (bd * bd), "f = {f}: β error {err}");
        }
    }
    println!(
        "shielded air CPW: Z_L = {z:.3} Ω, conformal mapping {z_want:.3} Ω ({:+.2} %); β vs \
         Yee within {eb:.1e}; worst |S11| = {:.1} dB",
        100.0 * (z / z_want - 1.0),
        o.worst_s11
    );
    assert!((z / z_want - 1.0).abs() < 0.03, "Z = {z}, want {z_want}");
    assert!(o.worst_s11 < -40.0, "{}", o.worst_s11);
}

/// Heavier (fine `z` cells, a distant CPML): run in release with
/// `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn cpw_port_conductor_backed_vs_simons() {
    const EPS_R: f64 = 4.4;
    const H: f64 = 0.5;
    let mut z = lines(-H, 0.0, 0.1);
    let mut step: f64 = 0.1;
    while *z.last().unwrap() < ZTOP {
        step = (step * 1.3).min(0.5);
        let next = *z.last().unwrap() + step;
        z.push(next);
    }
    let case = Case {
        dx: 0.5,
        z,
        sub: Some((EPS_R, -H, 0.0)),
        zmin: Boundary::Pec,
        zmax: Boundary::pml(8),
        decay_db: -80.0,
    };
    let freqs = linspace(1e9, 5e9, 5);
    let o = run(&case, &freqs);
    let (z_want, e_want) = CpwPort::quasi_static_of(S, W, H, EPS_R, true);
    let (z, e) = line_values(&o, 3, "backed CPW");
    println!(
        "backed CPW (1–3 GHz): Z_L = {z:.3} Ω, Simons {z_want:.3} Ω ({:+.2} %); ε_eff = \
         {e:.4}, Simons {e_want:.4} ({:+.2} %); worst |S11| = {:.1} dB",
        100.0 * (z / z_want - 1.0),
        100.0 * (e / e_want - 1.0),
        o.worst_s11
    );
    assert!((z / z_want - 1.0).abs() < 0.04, "Z = {z}, want {z_want}");
    assert!(
        (e / e_want - 1.0).abs() < 0.04,
        "ε_eff = {e}, want {e_want}"
    );
    assert!(o.worst_s11 < -35.0, "{}", o.worst_s11);
}

/// Conventional CPW on a 0.5 mm, ε_r = 4.4 substrate in open space
/// (release: `cargo test --release -- --ignored`).
#[test]
#[ignore]
fn cpw_port_conventional_vs_wen() {
    const EPS_R: f64 = 4.4;
    const H: f64 = 0.5;
    let grade = |v: &mut Vec<f64>, to: f64, sign: f64| {
        let mut step: f64 = 0.1;
        while sign * *v.last().unwrap() < to {
            step = (step * 1.3).min(0.5);
            let next = *v.last().unwrap() + sign * step;
            v.push(next);
        }
    };
    let mut lo = lines(-H, 0.0, 0.1);
    lo.reverse();
    grade(&mut lo, ZTOP, -1.0);
    let mut hi = vec![0.0];
    grade(&mut hi, ZTOP, 1.0);
    lo.reverse();
    lo.extend_from_slice(&hi[1..]);
    let case = Case {
        dx: 0.5,
        z: lo,
        sub: Some((EPS_R, -H, 0.0)),
        zmin: Boundary::pml(8),
        zmax: Boundary::pml(8),
        decay_db: -80.0,
    };
    let freqs = linspace(1e9, 5e9, 5);
    let o = run(&case, &freqs);
    let (z_want, e_want) = CpwPort::quasi_static_of(S, W, H, EPS_R, false);
    let (z, e) = line_values(&o, 3, "conventional CPW");
    println!(
        "conventional CPW (1–3 GHz): Z_L = {z:.3} Ω, Wen/Simons {z_want:.3} Ω ({:+.2} %); \
         ε_eff = {e:.4}, Wen/Simons {e_want:.4} ({:+.2} %); worst |S11| = {:.1} dB",
        100.0 * (z / z_want - 1.0),
        100.0 * (e / e_want - 1.0),
        o.worst_s11
    );
    assert!((z / z_want - 1.0).abs() < 0.04, "Z = {z}, want {z_want}");
    assert!(
        (e / e_want - 1.0).abs() < 0.04,
        "ε_eff = {e}, want {e_want}"
    );
    assert!(o.worst_s11 < -35.0, "{}", o.worst_s11);
}

#[test]
fn cpw_port_errors() {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, lines(0.0, 20.0, 1.0)).unwrap();
    m.grid.set_lines(1, lines(-6.0, 6.0, 0.5)).unwrap();
    m.grid.set_lines(2, lines(0.0, 2.0, 0.5)).unwrap();
    let p = |s: f64, g: f64| {
        CpwPort::new(
            1,
            [0.0, -3.0, 0.0],
            [10.0, 3.0, 1.0],
            Axis::X,
            Axis::Z,
            s,
            g,
        )
    };
    // No room for the grounds; zero gap; metal plane off the mesh.
    assert!(p(4.0, 1.0).add_to(&mut m).is_err());
    assert!(p(2.0, 0.0).add_to(&mut m).is_err());
    let off = CpwPort::new(
        1,
        [0.0, -3.0, 0.0],
        [10.0, 3.0, 1.2],
        Axis::X,
        Axis::Z,
        2.0,
        0.5,
    );
    assert!(off.add_to(&mut m).is_err());
    let flat = CpwPort::new(
        1,
        [0.0, -3.0, 1.0],
        [10.0, 3.0, 1.0],
        Axis::X,
        Axis::Z,
        2.0,
        0.5,
    );
    assert!(flat.clone().backed().add_to(&mut m).is_err());
    let ok = flat
        .feed_resistance(50.0)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let pr = ok.properties().unwrap();
    assert_eq!(pr.elements.as_ref().unwrap()[0].0, "port_resist_1A");
    assert_eq!(pr.voltages[2].0, "port_ut1C");
    // In air (h = 0): ε_eff = 1.
    assert_eq!(ok.quasi_static().unwrap().1, 1.0);
    assert!(ok.clone().add_to(&mut m).is_err());
}
