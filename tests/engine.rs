//! Physics validation of the engine core: PEC cavity resonances (uniform,
//! graded, dielectric/anisotropic fill), energy behaviour, end criteria,
//! determinism.

use std::ops::ControlFlow;

use ems::constants::C0;
use ems::engine::{
    CellMaterials, EdgeSource, Engine, FaceCondition, FaceConditions, Grid, Material,
    OperatorHandle, Probe, RunLimits, RunOutcome, StopReason, TimeStep,
};
use ems::{Axis, EndCriteria, Face};

/// Gaussian-modulated pulse centred at `f0` with ~`bw` bandwidth.
fn pulse(f0: f64, bw: f64) -> (impl Fn(f64) -> f64 + Send + 'static, f64) {
    let tau = 1.0 / (std::f64::consts::PI * bw);
    let t0 = 4.0 * tau;
    let f = move |t: f64| {
        let x = (t - t0) / tau;
        (-x * x).exp() * (2.0 * std::f64::consts::PI * f0 * (t - t0)).cos()
    };
    (f, 2.0 * t0)
}

/// |DFT| at `f` of the Hann-windowed samples with `t >= t_start`.
fn spectrum(times: &[f64], values: &[f64], t_start: f64, f: f64) -> f64 {
    let first = times.iter().position(|&t| t >= t_start).unwrap();
    let (t, v) = (&times[first..], &values[first..]);
    let n = t.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (q, (&tt, &x)) in t.iter().zip(v).enumerate() {
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * q as f64 / (n - 1.0)).cos();
        let ph = -2.0 * std::f64::consts::PI * f * tt;
        re += w * x * ph.cos();
        im += w * x * ph.sin();
    }
    re.hypot(im)
}

/// Frequency of the spectral peak within `[lo, hi]`: coarse scan, then
/// golden-section refinement. Also returns the peak magnitude.
fn peak(times: &[f64], values: &[f64], t_start: f64, lo: f64, hi: f64) -> (f64, f64) {
    let s = |f: f64| spectrum(times, values, t_start, f);
    let n = 200;
    let step = (hi - lo) / n as f64;
    let (best, _) = (0..=n)
        .map(|q| (q, s(lo + q as f64 * step)))
        .fold((0, f64::MIN), |a, b| if b.1 > a.1 { b } else { a });
    let (mut a, mut b) = (
        lo + (best as f64 - 1.0) * step,
        lo + (best as f64 + 1.0) * step,
    );
    let g = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..60 {
        let c = b - g * (b - a);
        let d = a + g * (b - a);
        if s(c) > s(d) {
            b = d;
        } else {
            a = c;
        }
    }
    let f = 0.5 * (a + b);
    (f, s(f))
}

fn cavity_freq(m: f64, n: f64, p: f64, dims: [f64; 3], eps_r: f64) -> f64 {
    C0 / (2.0 * eps_r.sqrt())
        * ((m / dims[0]).powi(2) + (n / dims[1]).powi(2) + (p / dims[2]).powi(2)).sqrt()
}

/// Lines from 0 to `len` with `cells` cells, sinusoidally graded so the cell
/// size varies by ±`alpha` around the mean.
fn graded(len: f64, cells: usize, alpha: f64) -> Vec<f64> {
    (0..=cells)
        .map(|q| {
            let t = q as f64 / cells as f64;
            len * (t - alpha * (2.0 * std::f64::consts::PI * t).sin()
                / (2.0 * std::f64::consts::PI))
        })
        .collect()
}

const DIMS: [f64; 3] = [0.10, 0.045, 0.07];

struct CavityRun {
    times: Vec<f64>,
    values: Vec<f64>,
    t_start: f64,
}

/// Runs a PEC cavity excited by a pulse on an E_y edge near (0.3a, 0.4d),
/// probing an E_y edge near (0.7a, 0.6d).
fn run_cavity(
    grid: Grid,
    mats: CellMaterials,
    steps: u64,
    threads: usize,
    scale: f64,
) -> CavityRun {
    let nearest = |axis: Axis, x: f64| {
        let l = grid.lines(axis);
        (0..l.len())
            .min_by(|&a, &b| (l[a] - x).abs().total_cmp(&(l[b] - x).abs()))
            .unwrap()
    };
    let src = [
        nearest(Axis::X, 0.3 * DIMS[0]),
        nearest(Axis::Y, 0.3 * DIMS[1]),
        nearest(Axis::Z, 0.4 * DIMS[2]),
    ];
    let prb = [
        nearest(Axis::X, 0.7 * DIMS[0]),
        nearest(Axis::Y, 0.5 * DIMS[1]),
        nearest(Axis::Z, 0.6 * DIMS[2]),
    ];
    let mut e = Engine::builder(grid)
        .materials(mats)
        .threads(threads)
        .build()
        .unwrap();
    let (f, t_end) = pulse(3.2e9 * scale, 2.5e9 * scale);
    e.add_operator(EdgeSource::new(f).edge(Axis::Y, src, 1.0));
    let h = e.add_operator(Probe::voltage(Axis::Y, prb));
    e.run(steps);
    let p = e.operator(&h);
    CavityRun {
        times: p.times().to_vec(),
        values: p.values().to_vec(),
        t_start: t_end,
    }
}

fn check_modes(r: &CavityRun, eps_r: f64, tol: f64) {
    for (m, n, p) in [(1.0, 0.0, 1.0), (2.0, 0.0, 1.0)] {
        let fa = cavity_freq(m, n, p, DIMS, eps_r);
        let (f, mag) = peak(&r.times, &r.values, r.t_start, 0.95 * fa, 1.05 * fa);
        let err = (f - fa) / fa;
        eprintln!(
            "mode ({m},{n},{p}): fdtd {f:.6e}, analytic {fa:.6e}, err {:.4}%, |X| {mag:.3e}",
            err * 100.0
        );
        assert!(
            err.abs() < tol,
            "mode ({m},{n},{p}) off by {:.3}%",
            err * 100.0
        );
        // A genuine resonance, not a window edge.
        assert!(f > 0.96 * fa && f < 1.04 * fa);
        let off = spectrum(&r.times, &r.values, r.t_start, 0.9 * fa);
        assert!(mag > 20.0 * off, "no clear peak for ({m},{n},{p})");
    }
}

#[test]
fn pec_cavity_uniform() {
    let d = 0.005;
    let g = Grid::uniform([20, 9, 14], [d, d, d]).unwrap();
    let m = CellMaterials::vacuum(&g);
    let r = run_cavity(g, m, 3000, 4, 1.0);
    check_modes(&r, 1.0, 0.005);
}

#[test]
fn pec_cavity_graded() {
    let g = Grid::new([
        graded(DIMS[0], 22, 0.3),
        graded(DIMS[1], 10, 0.2),
        graded(DIMS[2], 16, 0.3),
    ])
    .unwrap();
    let m = CellMaterials::vacuum(&g);
    let r = run_cavity(g, m, 4000, 4, 1.0);
    check_modes(&r, 1.0, 0.005);
}

#[test]
fn dielectric_cavity_scales() {
    // Only E_y exists in the (m, 0, p) modes, so only eps_y matters: an
    // anisotropic fill with eps_y = 2.25 must scale them by 1/1.5.
    let d = 0.005;
    let g = Grid::uniform([20, 9, 14], [d, d, d]).unwrap();
    let mut mat = Material::VACUUM;
    mat.eps_r = [4.0, 2.25, 9.0];
    let m = CellMaterials::new(&g, mat);
    let r = run_cavity(g, m, 4500, 4, 1.0 / 1.5);
    check_modes(&r, 2.25, 0.005);
}

fn small_cavity(mats: impl FnOnce(&Grid) -> CellMaterials, threads: usize) -> (Engine, f64) {
    let g = Grid::uniform([10, 8, 6], [0.01, 0.009, 0.011]).unwrap();
    let m = mats(&g);
    let mut e = Engine::builder(g)
        .materials(m)
        .threads(threads)
        .build()
        .unwrap();
    let (f, t_end) = pulse(3e9, 1e9);
    e.add_operator(
        EdgeSource::new(f)
            .edge(Axis::Z, [3, 2, 2], 1.0)
            .edge(Axis::X, [6, 5, 3], -0.5),
    );
    (e, t_end)
}

/// Runs past the source, then samples the energy invariant every
/// `interval` steps for `steps` steps; returns the largest relative
/// deviation from the first sample, and that of the staggered energy.
fn invariant_drift(e: &mut Engine, t_end: f64, steps: u64, interval: u64) -> (f64, f64) {
    e.run((t_end / e.dt()).ceil() as u64 + 1);
    let (w0, s0) = (e.energy(), e.energy_staggered());
    assert!(w0 > 0.0);
    let (mut dev, mut sdev) = (0.0f64, 0.0f64);
    for _ in 0..steps / interval {
        // Also check that the in-step value matches the read-only one.
        let w_ro = e.energy();
        let w = e.step_with_energy();
        assert_eq!(w.to_bits(), w_ro.to_bits());
        dev = dev.max((w / w0 - 1.0).abs());
        sdev = sdev.max((e.energy_staggered() / s0 - 1.0).abs());
        e.run(interval - 1);
    }
    eprintln!(
        "W0 = {w0:.6e} J: max |W/W0 − 1| = {dev:.3e} (staggered energy: {sdev:.3e}) over {steps} steps"
    );
    (dev, sdev)
}

fn energy_conserved(steps: u64) {
    let (mut e, t_end) = small_cavity(CellMaterials::vacuum, 2);
    let (dev, sdev) = invariant_drift(&mut e, t_end, steps, 10);
    assert!(dev < 1e-4, "lossless invariant drifted by {dev:e}");
    // The staggered energy is visibly not conserved.
    assert!(sdev > 100.0 * dev);
}

#[test]
fn lossless_energy_conserved_uniform() {
    energy_conserved(10_000);
}

#[test]
#[ignore = "long: 1e5 steps (design §9); run with --ignored in release"]
fn lossless_energy_conserved_long() {
    energy_conserved(100_000);
}

#[test]
fn lossless_energy_conserved_graded_dielectric() {
    let g = Grid::new([
        graded(0.1, 15, 0.3),
        graded(0.07, 11, 0.2),
        graded(0.05, 9, 0.25),
    ])
    .unwrap();
    let mut m = CellMaterials::vacuum(&g);
    m.fill_box([3..9, 2..8, 1..6], Material::dielectric(4.0));
    let mut aniso = Material::VACUUM.with_mu_r(1.5);
    aniso.eps_r = [2.0, 3.0, 1.5];
    m.fill_box([9..14, 0..11, 4..9], aniso);
    m.fill_pec_box([11..12, 3..8, 2..7]);
    let mut e = Engine::builder(g)
        .materials(m)
        .faces(FaceConditions::default().with(Face::YMax, FaceCondition::Pmc))
        .threads(3)
        .build()
        .unwrap();
    let (f, t_end) = pulse(3e9, 2e9);
    e.add_operator(
        EdgeSource::new(f)
            .edge(Axis::Z, [5, 4, 3], 1.0)
            .edge(Axis::Y, [10, 5, 6], 0.7),
    );
    let (dev, _) = invariant_drift(&mut e, t_end, 8000, 50);
    assert!(dev < 1e-4, "lossless invariant drifted by {dev:e}");
}

#[test]
fn lossy_energy_decays() {
    let kappa = 0.002;
    let (mut e, t_end) = small_cavity(
        |g| CellMaterials::new(g, Material::VACUUM.with_kappa(kappa)),
        2,
    );
    e.run((t_end / e.dt()).ceil() as u64 + 1);
    // The invariant is non-increasing with electric loss.
    let mut prev = e.energy();
    let (w1, t1) = (prev, e.time());
    for _ in 0..500 {
        let w = e.step_with_energy();
        assert!(w <= prev, "lossy energy grew: {prev} -> {w}");
        prev = w;
    }
    // Average over a window to remove the small ripple of the loss rate
    // (loss acts on the electric part only).
    let avg = |e: &mut Engine| {
        let mut s = 0.0;
        for _ in 0..100 {
            s += e.step_with_energy();
        }
        (s / 100.0, e.time() - 50.5 * e.dt())
    };
    let (wa, ta) = avg(&mut e);
    e.run(400);
    let (wb, tb) = avg(&mut e);
    assert!(wb < wa && wa < w1 && t1 < ta);
    // Uniform conductivity: E-field energy decays at kappa/eps0, and on
    // average half the energy is electric -> W ~ exp(-kappa/eps0 * t).
    let rate = (wa / wb).ln() / (tb - ta);
    let want = kappa / ems::constants::EPS0;
    eprintln!("decay rate {rate:.4e}, expected {want:.4e}");
    assert!(
        (rate / want - 1.0).abs() < 0.02,
        "decay rate {rate} vs {want}"
    );
}

#[test]
fn run_until_energy_decay() {
    let kappa = 0.002;
    let (mut e, t_end) = small_cavity(
        |g| CellMaterials::new(g, Material::VACUUM.with_kappa(kappa)),
        2,
    );
    let db = -30.0;
    let crit = EndCriteria::energy_decay_db(db).check_interval(20);
    let limits = RunLimits::new(100_000)
        .end_criteria(crit)
        .excitation_end(t_end);
    let mut w_max = 0.0;
    // First sample after the excitation: (time, energy).
    let mut after: Option<(f64, f64)> = None;
    let mut checks = 0u64;
    let out = e
        .run_until_with(&limits, |p| {
            assert_eq!(p.timestep, checks * 20);
            assert_eq!(p.steps, p.timestep + 1);
            checks += 1;
            w_max = f64::max(w_max, p.energy);
            if p.time > t_end && after.is_none() {
                after = Some((p.time, p.energy));
            }
            assert_eq!(p.max_energy, w_max);
            assert!(p.energy_db <= 0.0);
            assert!(p.mcells_per_s > 0.0);
            ControlFlow::Continue(())
        })
        .unwrap();
    let StopReason::EnergyDecay { ratio_db } = out.reason else {
        panic!("unexpected stop {:?}", out.reason);
    };
    assert!(ratio_db < db && ratio_db > db - 1.0, "{ratio_db}");
    assert_eq!(out.timestep, out.steps);
    assert_eq!(out.time, e.time());
    assert_eq!(out.max_energy, w_max);
    assert!(out.energy / out.max_energy < crit.energy_ratio);
    // The deciding sample is one step before the end.
    let t_stop = out.time - e.dt();
    let (ta, wa) = after.unwrap();
    assert!(t_stop > ta);
    // Free decay W(t) = W(ta)·exp(−κ/ε0·(t − ta)) after the source: the
    // expected crossing of W = ratio·W_max.
    let rate = kappa / ems::constants::EPS0;
    let want = ta + (wa / (crit.energy_ratio * w_max)).ln() / rate;
    eprintln!(
        "stopped at {t_stop:.4e} s after {} steps, expected ≈ {want:.4e} s (checks every {:.3e} s)",
        out.steps,
        20.0 * e.dt()
    );
    let window = 0.01 * (want - ta) + 20.0 * e.dt();
    assert!(
        (t_stop - want).abs() < window,
        "stop at {t_stop:e} s, expected {want:e} ± {window:e} s"
    );
}

#[test]
fn run_until_max_timesteps_and_abort() {
    // Lossless: the energy never decays, so the run hits max_timesteps.
    let (mut e, t_end) = small_cavity(CellMaterials::vacuum, 2);
    let limits = RunLimits::new(1234)
        .end_criteria(EndCriteria::energy_decay_db(-60.0))
        .excitation_end(t_end);
    let out = e.run_until(&limits).unwrap();
    assert_eq!(out.reason, StopReason::MaxTimesteps);
    assert_eq!((out.steps, out.timestep, e.timestep()), (1234, 1234, 1234));
    assert!(out.energy > 0.0 && out.energy <= out.max_energy);
    // A second call counts its own steps.
    let out = e.run_until(&RunLimits::new(7)).unwrap();
    assert_eq!(
        (out.reason, out.steps, out.timestep),
        (StopReason::MaxTimesteps, 7, 1241)
    );
    assert_eq!(
        e.run_until(&RunLimits::new(0)).unwrap().reason,
        StopReason::MaxTimesteps
    );
    // Abort from the callback.
    let mut calls = 0;
    let out = e
        .run_until_with(&RunLimits::new(10_000), |p| {
            calls += 1;
            if p.steps > 300 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .unwrap();
    assert_eq!(out.reason, StopReason::Aborted);
    assert_eq!(out.steps, 301);
    assert_eq!(calls, 7);
    // Invalid limits are rejected before stepping.
    let before = e.timestep();
    let bad = RunLimits::new(10).end_criteria(EndCriteria::energy_ratio(2.0));
    assert!(e.run_until(&bad).is_err());
    assert!(
        e.run_until(&RunLimits::new(10).excitation_end(f64::NAN))
            .is_err()
    );
    assert_eq!(e.timestep(), before);
}

#[test]
fn run_until_waits_for_excitation_and_energy() {
    // No field yet and the source starts at ~0: the criterion must not
    // fire on W_max = 0, nor before the excitation is over.
    let (mut e, t_end) = small_cavity(CellMaterials::vacuum, 1);
    let mut first_db = None;
    let limits = RunLimits::new(100)
        .end_criteria(EndCriteria::energy_ratio(1.0).check_interval(1))
        .excitation_end(t_end);
    let out = e
        .run_until_with(&limits, |p| {
            first_db.get_or_insert(p.energy_db);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(first_db, Some(0.0));
    assert!(100.0 * e.dt() < t_end);
    assert_eq!(out.reason, StopReason::MaxTimesteps);
}

fn energy_trace(threads: usize) -> (Vec<u64>, RunOutcome) {
    let g = Grid::new([
        graded(0.1, 13, 0.2),
        graded(0.08, 9, 0.1),
        graded(0.06, 7, 0.0),
    ])
    .unwrap();
    let mut m = CellMaterials::vacuum(&g);
    m.fill_box(
        [2..6, 1..5, 2..6],
        Material::dielectric(3.0).with_kappa(0.05),
    );
    m.fill_box(
        [7..12, 0..9, 0..3],
        Material::VACUUM.with_mu_r(2.0).with_sigma_m(50.0),
    );
    m.fill_pec_box([9..10, 2..7, 1..6]);
    let mut e = Engine::builder(g)
        .materials(m)
        .threads(threads)
        .build()
        .unwrap();
    assert_eq!(e.threads(), threads);
    let (f, t_end) = pulse(5e9, 4e9);
    e.add_operator(EdgeSource::new(f).edge(Axis::Z, [4, 4, 3], 1.0));
    let mut trace = Vec::new();
    let limits = RunLimits::new(5000)
        .end_criteria(EndCriteria::energy_decay_db(-20.0).check_interval(7))
        .excitation_end(t_end);
    let out = e
        .run_until_with(&limits, |p| {
            trace.push(p.energy.to_bits());
            ControlFlow::Continue(())
        })
        .unwrap();
    trace.push(e.energy().to_bits());
    (trace, out)
}

#[test]
fn energy_deterministic_across_threads() {
    let (t1, o1) = energy_trace(1);
    assert!(matches!(o1.reason, StopReason::EnergyDecay { .. }));
    for threads in [2, 4] {
        let (t, o) = energy_trace(threads);
        assert_eq!(t1, t);
        assert_eq!(
            (o1.reason, o1.steps, o1.energy.to_bits()),
            (o.reason, o.steps, o.energy.to_bits())
        );
    }
}

fn run_det(threads: usize, faces: FaceConditions) -> (Vec<Vec<f32>>, Vec<f64>, f64) {
    let g = Grid::new([
        graded(0.1, 13, 0.2),
        graded(0.08, 9, 0.1),
        graded(0.06, 7, 0.0),
    ])
    .unwrap();
    let mut m = CellMaterials::vacuum(&g);
    m.fill_box(
        [2..6, 1..5, 2..6],
        Material::dielectric(3.0).with_kappa(0.02),
    );
    m.fill_box(
        [7..12, 0..9, 0..3],
        Material::VACUUM.with_mu_r(2.0).with_sigma_m(50.0),
    );
    m.fill_pec_box([9..10, 2..7, 1..6]);
    let mut e = Engine::builder(g)
        .materials(m)
        .faces(faces)
        .timestep(TimeStep::Courant(0.95))
        .threads(threads)
        .build()
        .unwrap();
    assert_eq!(e.threads(), threads);
    let (f, _) = pulse(5e9, 4e9);
    e.add_operator(EdgeSource::new(f).edge(Axis::Z, [4, 4, 3], 1.0));
    let h: OperatorHandle<Probe> = e.add_operator(Probe::current(Axis::X, [5, 3, 3]).interval(3));
    e.run(300);
    let mut arrays = Vec::new();
    for a in Axis::ALL {
        arrays.push(e.u_component(a));
        arrays.push(e.i_component(a));
    }
    (arrays, e.operator(&h).values().to_vec(), e.energy())
}

#[test]
fn deterministic_across_threads() {
    let faces = FaceConditions::default()
        .with(Face::XMax, FaceCondition::Pmc)
        .with(Face::ZMin, FaceCondition::Pmc);
    let (a1, p1, w1) = run_det(1, faces);
    assert!(w1 > 0.0);
    for t in [2, 3, 4] {
        let (a, p, w) = run_det(t, faces);
        for (x, y) in a1.iter().zip(&a) {
            assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()));
        }
        assert_eq!(
            p1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            p.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(w1.to_bits(), w.to_bits());
    }
}

#[test]
fn deterministic_run_vs_step_no_operators() {
    // The batched no-operator path must match stepping with operators that
    // do nothing.
    struct Noop;
    impl ems::engine::Operator for Noop {}
    let build = |threads| {
        let g = Grid::uniform([9, 7, 5], [1e-3; 3]).unwrap();
        let mut e = Engine::builder(g).threads(threads).build().unwrap();
        let mut f = e.fields_mut();
        f.u.set(Axis::Z, [4, 3, 2], 1.0);
        f.i.set(Axis::X, [2, 2, 2], 1e-3);
        drop(f);
        e
    };
    let mut a = build(3);
    a.run(200);
    let mut b = build(1);
    b.add_operator(Noop);
    for _ in 0..200 {
        b.step();
    }
    assert_eq!(a.timestep(), 200);
    for ax in Axis::ALL {
        assert_eq!(a.u_component(ax), b.u_component(ax));
        assert_eq!(a.i_component(ax), b.i_component(ax));
    }
}

#[test]
fn pec_walls_and_padding_stay_zero() {
    let (mut e, _) = small_cavity(CellMaterials::vacuum, 3);
    e.run(200);
    let [nx, ny, nz] = e.grid().nodes();
    for j in 0..ny {
        for k in 0..nz {
            // Tangential E on the x faces, and the x-edge padding plane.
            assert_eq!(e.u(Axis::Y, [0, j, k]), 0.0);
            assert_eq!(e.u(Axis::Z, [nx - 1, j, k]), 0.0);
            assert_eq!(e.u(Axis::X, [nx - 1, j, k]), 0.0);
            // Normal H on a PEC face stays zero.
            assert_eq!(e.i(Axis::X, [0, j, k]), 0.0);
        }
    }
    assert!(e.energy() > 0.0);
}

#[test]
fn builder_validation() {
    let g = Grid::uniform([4, 4, 4], [1e-3; 3]).unwrap();
    let lim = ems::engine::courant_limit(&g);
    assert!(
        Engine::builder(g.clone())
            .timestep(TimeStep::Fixed(1.01 * lim))
            .build()
            .is_err()
    );
    let e = Engine::builder(g.clone())
        .timestep(TimeStep::Fixed(0.5 * lim))
        .threads(1)
        .build()
        .unwrap();
    assert_eq!(e.dt(), 0.5 * lim);
    let other = Grid::uniform([3, 4, 4], [1e-3; 3]).unwrap();
    assert!(
        Engine::builder(g.clone())
            .materials(CellMaterials::vacuum(&other))
            .build()
            .is_err()
    );
    let mut bad = CellMaterials::vacuum(&g);
    bad.set([0, 0, 0], Material::VACUUM.with_kappa(-1.0));
    assert!(Engine::builder(g).materials(bad).build().is_err());
}
