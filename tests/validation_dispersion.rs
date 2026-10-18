//! Analytic validation (design §9): numerical dispersion of a plane wave
//! in vacuum against the Yee dispersion relation.
//!
//! For a plane wave along a mesh axis the Yee scheme obeys exactly
//! (Taflove & Hagness, ch. 4)
//!
//! ```text
//! sin(ω·Δt/2)/(c·Δt) = sin(k̃·Δz/2)/Δz
//! ```
//!
//! On a coarse mesh (10 cells per wavelength at the top of the band) and a
//! Courant number well below 1, the numerical phase velocity `ω/k̃` is
//! visibly slower than `c` (≈ 1 %), so matching it to ~1e-5 validates the
//! exact discrete behaviour, not just the continuum limit. The wave runs
//! along the quasi-1-D TEM line of `common::Line` (exactly 1-D); `k̃` comes
//! from the phase difference of two probes 100 cells apart.

mod common;

use std::f64::consts::PI;

use common::*;
use ems::constants::C0;
use ems::engine::TimeStep;

const DZ: f64 = 1e-3;

/// Returns, for Courant number `s = c·Δt/Δz`, the max relative error of
/// the measured `k̃` vs the Yee relation, the max relative deviation of the
/// Yee `k̃` from the continuum `ω/c`, and the max `||H| − 1|` of the probe
/// transfer function (lossless propagation).
fn dispersion(s: f64) -> (f64, f64, f64) {
    let (src, p1, d) = (48, 68, 100);
    let mut l = Line::new(DZ, p1 + d + 40 + 8);
    l.width = 10e-3;
    l.timestep = TimeStep::Fixed(s * DZ / C0);
    l.src = src;
    l.probes = vec![p1, p1 + d];
    let steps = (2.0 * (d + 20 + 76) as f64 / s) as u64 + 100;
    let (dt, v) = l.run(sine_pulse(17.5e9, 10e9), steps);
    for x in &v {
        let tail = peak(&x[x.len() - 20..]) / peak(x);
        assert!(tail < 1e-6, "record too short: tail {tail:e}");
    }
    let freqs = linspace(5e9, 30e9, 26);
    let a = spectrum(&v[0], dt, &freqs);
    let b = spectrum(&v[1], dt, &freqs);
    let len = d as f64 * DZ;
    let (mut e_yee, mut dev, mut e_mag) = (0.0f64, 0.0f64, 0.0f64);
    for ((&f, &a), &b) in freqs.iter().zip(&a).zip(&b) {
        let h = b / a;
        let k0 = 2.0 * PI * f / C0;
        // Unwrap around the continuum value (the Yee value is < 1.5 %
        // away, i.e. < 1 rad over 100 cells).
        let ph = -h.arg();
        let m = ((k0 * len - ph) / (2.0 * PI)).round();
        let k = (ph + 2.0 * PI * m) / len;
        let ky = yee_k(f, dt, DZ, 1.0, 0.0).re / DZ;
        // Closed form of the relation, for the record.
        let ky2 = 2.0 / DZ * (DZ / (C0 * dt) * (PI * f * dt).sin()).asin();
        assert!((ky - ky2).abs() < 1e-9 * ky);
        e_yee = e_yee.max((k / ky - 1.0).abs());
        dev = dev.max((ky / k0 - 1.0).abs());
        e_mag = e_mag.max((h.norm() - 1.0).abs());
    }
    eprintln!(
        "dispersion, S = {s}: v_num/c down to {:.4} at 10 cells/λ; \
         max |k_meas/k_yee − 1| = {e_yee:.2e}, max ||H| − 1| = {e_mag:.1e}",
        1.0 / (1.0 + dev)
    );
    (e_yee, dev, e_mag)
}

/// The measured `k̃(ω)` matches the Yee relation to 2e-5 relative over
/// 5–30 GHz (60 … 10 cells per wavelength), while the continuum differs
/// by more than 1 % (S = 0.5) — so the test resolves the discrete
/// relation, not just the continuum.
#[test]
fn plane_wave_matches_yee_dispersion() {
    let _t = Timer::start("plane_wave_matches_yee_dispersion");
    for s in [0.5, 0.9] {
        let (e_yee, dev, e_mag) = dispersion(s);
        assert!(e_yee < 2e-5, "S = {s}: {e_yee}");
        assert!(e_mag < 1e-3, "S = {s}: {e_mag}");
        if s == 0.5 {
            assert!(dev > 0.01, "dispersion too weak to discriminate: {dev}");
        }
    }
}
