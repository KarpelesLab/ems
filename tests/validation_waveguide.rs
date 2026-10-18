//! Analytic validation (design §9): TE modes of an empty rectangular
//! waveguide with [`RectWaveguidePort`]s.
//!
//! A 20 mm × 10 mm guide (TE10 cutoff `f_c = c/(2a)` = 7.49 GHz) with PEC
//! walls (the transverse faces of the domain) and CPML at the ends.
//! Port 1 excites the mode near the start; port 2 is a passive port near
//! the far end, looking back. Between the reference planes (distance `L`)
//! the ideal result is `S11 = 0`, `S21 = e^{−jβL}` with
//! `β = √(k² − k_c²)`. With the far end shorted instead (PEC wall at
//! distance `d` from port 1's reference plane), `S11 = −e^{−2jβd}`.
//!
//! The β extracted from the phases is compared with the analytic value
//! (within 0.2 %, the Yee dispersion of 1 mm cells) and with the exact
//! dispersion relation of the Yee scheme for the sampled mode,
//!
//! ```text
//! sin²(β̃Δ/2)/Δ² = sin²(ωΔt/2)/(cΔt)² − sin²(mπΔ/2a)/Δ² − sin²(nπΔ/2b)/Δ²
//! ```
//!
//! (cubic cells `Δ`), which it matches to about 1e-4: what is left of the
//! analytic error is the scheme's dispersion, not the port.

use core::f64::consts::PI;

use csx::ContinuousStructure;
use ems::constants::C0;
use ems::port::RectWaveguidePort;
use ems::post::{Complex64, s_param};
use ems::{Axis, Boundaries, Boundary, EndCriteria, Face, Fdtd, Simulation, Waveform};

const A: f64 = 20.0;
const B: f64 = 10.0;
const FREQS: [f64; 4] = [9e9, 10e9, 11e9, 12e9];

struct Run {
    s11: Vec<Complex64>,
    /// `S21` (`None` without port 2).
    s21: Option<Vec<Complex64>>,
    /// Distance between the two reference planes, or from port 1's
    /// reference plane to the short \[m\].
    l: f64,
    /// Analytic β.
    beta: Vec<f64>,
    /// β of the discrete (Yee) dispersion relation.
    beta_yee: Vec<f64>,
}

/// The guide along `axis` (cells of `d` mm, length `len` mm), cross
/// section `A × B` on `axis.others()`, mode `mode`; far end matched
/// (port 2 + PML) or `shorted` (PEC wall, no port 2).
fn run(axis: Axis, mode: (u32, u32), d: f64, len: f64, shorted: bool) -> Run {
    run_dims(axis, mode, [A, B], d, len, shorted)
}

/// [`run`] with the cross section `[a, b]` (mm) on `axis.others()`.
fn run_dims(
    axis: Axis,
    mode: (u32, u32),
    [ga, gb]: [f64; 2],
    d: f64,
    len: f64,
    shorted: bool,
) -> Run {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    let lines = |l: f64| -> Vec<f64> {
        let n = (l / d).round() as usize;
        (0..=n).map(|i| i as f64 * d).collect()
    };
    let w = axis.index();
    let (u, v) = (axis.others().0.index(), axis.others().1.index());
    m.grid.set_lines(u, lines(ga)).unwrap();
    m.grid.set_lines(v, lines(gb)).unwrap();
    m.grid.set_lines(w, lines(len)).unwrap();
    let pml = 8;
    let z1 = (pml as f64 + 2.0) * d;
    let (r1, r2) = (z1 + 4.0 * d, len - z1 - 4.0 * d);
    let corner = |a: f64, b: f64, z: f64| {
        let mut p = [0.0; 3];
        p[u] = a;
        p[v] = b;
        p[w] = z;
        p
    };
    let p1 = RectWaveguidePort::new(1, corner(0., 0., z1), corner(ga, gb, r1), axis, mode)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let p2 = (!shorted).then(|| {
        RectWaveguidePort::new(2, corner(0., 0., len - z1), corner(ga, gb, r2), axis, mode)
            .add_to(&mut m)
            .unwrap()
    });
    let [lo, hi] = [Face::ALL[2 * w], Face::ALL[2 * w + 1]];
    let far = if shorted {
        Boundary::Pec
    } else {
        Boundary::pml(pml)
    };
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(10.5e9, 3e9))
        .boundaries(
            Boundaries::all(Boundary::Pec)
                .with(lo, Boundary::pml(pml))
                .with(hi, far),
        )
        .end_criteria(EndCriteria::energy_decay_db(-50.0))
        .max_timesteps(40_000)
        .threads(4);
    let mut sim = Simulation::new(&m, &fdtd).unwrap();
    assert!(sim.warnings().is_empty(), "{:?}", sim.warnings());
    let res = sim.run().unwrap();
    let a1 = p1.analyze(&res, &FREQS).unwrap();
    let s21 = p2.map(|p2| s_param(&p2.analyze(&res, &FREQS).unwrap(), &a1));
    Run {
        s11: a1.s11(),
        s21,
        l: (if shorted { len - r1 } else { r2 - r1 }) * 1e-3,
        beta: FREQS.iter().map(|&f| p1.beta(f).unwrap()).collect(),
        beta_yee: FREQS
            .iter()
            .map(|&f| {
                let (dt, dm) = (res.dt(), d * 1e-3);
                let (mm, nn) = (f64::from(mode.0), f64::from(mode.1));
                let s = |x: f64| x.sin().powi(2);
                let rhs = s(PI * f * dt) / (C0 * dt).powi(2)
                    - s(mm * PI * dm / (2.0 * ga * 1e-3)) / dm.powi(2)
                    - s(nn * PI * dm / (2.0 * gb * 1e-3)) / dm.powi(2);
                2.0 / dm * (dm * rhs.sqrt()).asin()
            })
            .collect(),
    }
}

/// β from the phase `−βl` of `s`, unwrapped with the analytic `beta`.
fn beta_from_phase(s: Complex64, l: f64, beta: f64) -> f64 {
    let mut phi = -s.im.atan2(s.re);
    phi += 2.0 * PI * ((beta * l - phi) / (2.0 * PI)).round();
    phi / l
}

fn check_matched(r: &Run, s11_db: f64, beta_tol: f64) {
    let s21 = r.s21.as_ref().unwrap();
    for (k, &f) in FREQS.iter().enumerate() {
        let db = 20.0 * r.s11[k].norm().log10();
        let beta = beta_from_phase(s21[k], r.l, r.beta[k]);
        let err = beta / r.beta[k] - 1.0;
        let err_yee = beta / r.beta_yee[k] - 1.0;
        eprintln!(
            "f = {:.1} GHz: |S11| = {db:.1} dB, |S21| = {:.4}, β = {beta:.3} rad/m \
             (analytic {:.3}, {:+.3}%; Yee {:+.2e})",
            f / 1e9,
            s21[k].norm(),
            r.beta[k],
            100.0 * err,
            err_yee
        );
        assert!(db < s11_db, "{f}: |S11| = {db} dB");
        assert!(
            (s21[k].norm() - 1.0).abs() < 0.01,
            "{f}: |S21| = {}",
            s21[k].norm()
        );
        assert!(err.abs() < beta_tol, "{f}: β error {err}");
        assert!(err_yee.abs() < 2e-4, "{f}: β vs Yee dispersion {err_yee}");
    }
}

#[test]
fn te10_matched_guide() {
    let r = run(Axis::Z, (1, 0), 1.0, 80.0, false);
    check_matched(&r, -50.0, 0.002);
}

/// The same guide along y: the cross section is on `(z, x)`, so the mode
/// functions and probe planes are exercised on another axis pair.
#[test]
fn te10_matched_guide_along_y() {
    let r = run(Axis::Y, (1, 0), 1.0, 80.0, false);
    check_matched(&r, -50.0, 0.002);
}

/// TE01 of a guide along x whose long side (20 mm) is along `v = z`: the
/// same physical mode, with the `n` index and the `e_u` component.
#[test]
fn te01_matched_guide_along_x() {
    let r = run_dims(Axis::X, (0, 1), [B, A], 1.0, 80.0, false);
    check_matched(&r, -50.0, 0.002);
}

#[test]
fn short_circuit_phase() {
    // S11 = −e^{−2jβd}: |S11| = 1 and the round-trip phase of the short.
    let r = run(Axis::Z, (1, 0), 1.0, 60.0, true);
    for (k, &f) in FREQS.iter().enumerate() {
        let s = r.s11[k];
        // −S11 = e^{−jβ·2d}.
        let beta = beta_from_phase(-s, 2.0 * r.l, r.beta[k]);
        let err = beta / r.beta[k] - 1.0;
        let err_yee = beta / r.beta_yee[k] - 1.0;
        eprintln!(
            "f = {:.1} GHz: |S11| = {:.4}, β = {beta:.3} rad/m (analytic {:.3}, {:+.3}%; \
             Yee {err_yee:+.2e})",
            f / 1e9,
            s.norm(),
            r.beta[k],
            100.0 * err
        );
        assert!((s.norm() - 1.0).abs() < 0.01, "{f}: |S11| = {}", s.norm());
        assert!(err.abs() < 0.002, "{f}: β error {err}");
        assert!(err_yee.abs() < 5e-4, "{f}: β vs Yee dispersion {err_yee}");
    }
}

/// The port round-trips through openEMS XML: the mode functions are csx
/// user attributes and the excitation weights csx weighting functions, so
/// the compiled probes and excitation are identical.
#[cfg(feature = "xml")]
#[test]
fn port_round_trips_through_xml() {
    use csx::OpenEmsDocument;
    use ems::compile::compile;

    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, (0..=20).map(f64::from)).unwrap();
    m.grid.set_lines(1, (0..=10).map(f64::from)).unwrap();
    m.grid.set_lines(2, (0..=30).map(f64::from)).unwrap();
    RectWaveguidePort::new(1, [0., 0., 3.], [20., 10., 7.], Axis::Z, (2, 1))
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let fdtd = Fdtd::new().waveform(Waveform::gaussian(10e9, 5e9));
    let mut doc = OpenEmsDocument {
        structure: m.clone(),
        ..Default::default()
    };
    doc.set_fdtd_settings(&fdtd.to_csx());
    let back = OpenEmsDocument::from_xml_str(&doc.to_xml_string()).unwrap();
    let a = compile(&m, &fdtd, &Default::default()).unwrap();
    let b = compile(&back.structure, &fdtd, &Default::default()).unwrap();
    assert_eq!(a.sources.probes.len(), 2);
    assert!(a.sources.probes.iter().all(|p| !p.mode_entries.is_empty()));
    assert_eq!(a.sources.probes, b.sources.probes);
    assert_eq!(a.sources.excitations, b.sources.excitations);
    assert!(!a.sources.excitations[0].amplitudes.is_empty());
}
