//! Validation of the first-order Mur absorbing boundary (Mur 1981):
//! normal-incidence reflection of a TEM pulse in a PEC/PMC "waveguide",
//! energy leaving a box, long-run stability and thread determinism.

use ems::constants::C0;
use ems::engine::boundary::mur::MurBoundary;
use ems::engine::{EdgeSource, Engine, FaceCondition, FaceConditions, Grid, Probe};
use ems::{Axis, Face};

/// Gaussian pulse of 1/e half-width `tau`, delayed by `4·tau`.
fn gaussian(tau: f64) -> impl Fn(f64) -> f64 + Send + 'static {
    move |t: f64| {
        let x = (t - 4.0 * tau) / tau;
        (-x * x).exp()
    }
}

/// Derivative-of-Gaussian pulse (no DC content, so a soft source leaves no
/// static charge behind).
fn dgaussian(tau: f64) -> impl Fn(f64) -> f64 + Send + 'static {
    move |t: f64| {
        let x = (t - 4.0 * tau) / tau;
        -2.0 * x * (-x * x).exp()
    }
}

const D: f64 = 1e-3;
const TAU: f64 = 2.8e-11;

/// Quasi-1-D line of `cells` cells along `a` (1 cell across), carrying an
/// exact TEM plane wave: E along `b = a.next()`, H along `c = a.prev()`,
/// PEC on the `b` faces (E normal to them), PMC on the `c` faces (H normal
/// to them). The `a` faces are PEC, plus Mur if `mur`. A soft source on the
/// E edges of plane `src` launches a pulse both ways; returns the E voltage
/// at each probe plane for `steps` steps.
fn line(
    a: Axis,
    cells: usize,
    src: usize,
    probes: &[usize],
    mur: bool,
    steps: u64,
) -> Vec<Vec<f64>> {
    let (b, c) = (a.next(), a.prev());
    let mut n = [1; 3];
    n[a.index()] = cells;
    let grid = Grid::uniform(n, [D; 3]).unwrap();
    let faces = FaceConditions::default()
        .with(Face::new(c, false), FaceCondition::Pmc)
        .with(Face::new(c, true), FaceCondition::Pmc);
    let mut e = Engine::builder(grid)
        .faces(faces)
        .threads(1)
        .build()
        .unwrap();
    if mur {
        let m = MurBoundary::with_faces(&e, &[(Face::new(a, false), C0), (Face::new(a, true), C0)])
            .unwrap();
        e.add_operator(m);
    }
    let node = |pos: usize, q: usize| {
        let mut p = [0; 3];
        p[a.index()] = pos;
        p[c.index()] = q;
        p
    };
    e.add_operator(
        EdgeSource::new(gaussian(TAU))
            .edge(b, node(src, 0), 1.0)
            .edge(b, node(src, 1), 1.0),
    );
    let hs: Vec<_> = probes
        .iter()
        .map(|&p| e.add_operator(Probe::voltage(b, node(p, 0))))
        .collect();
    e.run(steps);
    // The wave must stay uniform across the line (exact TEM solution).
    for &p in probes {
        let (v0, v1) = (e.u(b, node(p, 0)), e.u(b, node(p, 1)));
        assert!(
            (v0 - v1).abs() <= 1e-5 * v0.abs().max(1e-6),
            "not TEM: {v0} vs {v1}"
        );
    }
    hs.iter().map(|h| e.operator(h).values().to_vec()).collect()
}

/// Reflection level in dB: peak of (Mur run − reference run in a much longer
/// line) relative to the peak incident signal, at each probe.
fn reflection_db(a: Axis) -> Vec<f64> {
    let (cells, src, probes) = (120, 60, [30, 90]);
    // Window: incident + reflections from both faces (up to 150 mm of
    // travel past the source) plus the pulse length, with margin.
    let dt = 0.99 * D / (C0 * 3f64.sqrt());
    let steps = (8.5e-10 / dt) as u64;
    // Reference: 150 cells more on each side; its own end reflections need
    // >= 2·150 + 90 mm of travel and arrive after the window.
    let ext = 150;
    let pr: Vec<_> = probes.iter().map(|p| p + ext).collect();
    let refr = line(a, cells + 2 * ext, src + ext, &pr, false, steps);
    let mur = line(a, cells, src, &probes, true, steps);
    refr.iter()
        .zip(&mur)
        .map(|(r, m)| {
            let peak = r.iter().fold(0.0f64, |x, v| x.max(v.abs()));
            let diff = r
                .iter()
                .zip(m)
                .fold(0.0f64, |x, (p, q)| x.max((p - q).abs()));
            assert!(peak > 0.0);
            20.0 * (diff / peak).log10()
        })
        .collect()
}

#[test]
fn normal_incidence_reflection_all_faces() {
    for a in Axis::ALL {
        let db = reflection_db(a);
        eprintln!("axis {a}: reflection at probes {db:.1?} dB");
        for r in db {
            assert!(r < -30.0, "axis {a}: reflection {r:.1} dB");
        }
    }
}

#[test]
fn reflection_much_better_than_pec() {
    // Sanity check of the measurement: without Mur the PEC face reflects
    // the full pulse (0 dB).
    let dt = 0.99 * D / (C0 * 3f64.sqrt());
    let steps = (8.5e-10 / dt) as u64;
    let r = line(Axis::X, 420, 210, &[180], false, steps);
    let p = line(Axis::X, 120, 60, &[30], false, steps);
    let peak = r[0].iter().fold(0.0f64, |x, v| x.max(v.abs()));
    let diff = r[0]
        .iter()
        .zip(&p[0])
        .fold(0.0f64, |x, (a, b)| x.max((a - b).abs()));
    let db = 20.0 * (diff / peak).log10();
    eprintln!("PEC reflection {db:.1} dB");
    assert!(db > -1.0);
}

/// Cube of `cells³` cells with Mur on all six faces and a dipole pulse in
/// the middle.
fn mur_box(cells: usize, threads: usize, graded: bool) -> Engine {
    let lines = |s: f64| -> Vec<f64> {
        (0..=cells)
            .map(|q| {
                let t = q as f64 / cells as f64;
                let w = if graded {
                    0.2 * (6.0 * t).sin() / 6.0
                } else {
                    0.0
                };
                s * cells as f64 * (t + w)
            })
            .collect()
    };
    let grid = Grid::new([lines(D), lines(1.1 * D), lines(0.9 * D)]).unwrap();
    let mut e = Engine::builder(grid).threads(threads).build().unwrap();
    let all: Vec<_> = Face::ALL.iter().map(|&f| (f, C0)).collect();
    e.add_operator(MurBoundary::with_faces(&e, &all).unwrap());
    let m = cells / 2;
    e.add_operator(
        EdgeSource::new(dgaussian(TAU))
            .edge(Axis::Z, [m, m, m], 1.0)
            .edge(Axis::X, [m - 2, m + 1, m], 0.3),
    );
    e
}

/// Runs until the pulse is gone; returns (peak energy, energy at the end).
fn drain(e: &mut Engine, steps: u64) -> (f64, f64) {
    let mut peak = 0.0f64;
    for _ in 0..steps {
        e.step();
        peak = peak.max(e.energy());
    }
    (peak, e.energy())
}

#[test]
fn energy_leaves_through_mur_faces() {
    let mut e = mur_box(16, 2, false);
    // Source over at 8·tau; the box is crossed in ~16·1.1 mm / c ≈ 60 ps.
    let steps = ((8.0 * TAU + 3.0e-10) / e.dt()) as u64;
    let (peak, end) = drain(&mut e, steps);
    eprintln!(
        "energy peak {peak:.3e}, after {steps} steps {end:.3e} ({:.2e})",
        end / peak
    );
    assert!(end < 1e-2 * peak, "energy did not leave: {end} / {peak}");
}

fn stability(steps: u64) {
    let mut e = mur_box(12, 2, true);
    let settle = ((8.0 * TAU + 3.0e-10) / e.dt()) as u64;
    let (peak, w0) = drain(&mut e, settle);
    // From here on nothing is excited: energy must not grow.
    let mut wmax = 0.0f64;
    for q in 0..steps {
        e.step();
        if q % 50 == 0 {
            wmax = wmax.max(e.energy());
        }
    }
    let w1 = e.energy();
    eprintln!("peak {peak:.3e}, after settle {w0:.3e}, max later {wmax:.3e}, end {w1:.3e}");
    // The residual is at f32 round-off level; any instability would grow
    // exponentially far beyond it.
    assert!(w1.is_finite());
    assert!(w0 < 1e-9 * peak, "pulse did not leave: {w0} / {peak}");
    assert!(wmax < 1e-9 * peak, "energy grew: {w0} -> {wmax}");
}

#[test]
fn long_run_stable() {
    stability(3000);
}

#[test]
#[ignore = "long: 1e5 steps; run with --ignored in release"]
fn long_run_stable_long() {
    stability(100_000);
}

#[test]
fn deterministic_across_threads() {
    let run = |threads| {
        let mut e = mur_box(10, threads, true);
        assert_eq!(e.threads(), threads);
        e.run(250);
        let mut v = Vec::new();
        for a in Axis::ALL {
            v.extend(e.u_component(a).iter().map(|x| x.to_bits()));
            v.extend(e.i_component(a).iter().map(|x| x.to_bits()));
        }
        v
    };
    let one = run(1);
    assert!(one.iter().any(|&b| b != 0));
    assert_eq!(one, run(4));
    assert_eq!(one, run(3));
}
