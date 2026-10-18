//! Analytic validation (design §9): attenuation of a plane wave in a
//! conductive dielectric.
//!
//! The quasi-1-D TEM line of `common::Line` is filled with ε_r = 2,
//! κ = 0.2 S/m (loss tangent 0.45 at 4 GHz, 1.8 at 1 GHz), CPML at both
//! ends. A soft source launches a pulse; two probes `D` cells apart
//! downstream see only the forward wave, so
//! `α(f) = ln(|E₁(f)|/|E₂(f)|)/(D·Δz)` and
//! `β(f) = −Δarg/(D·Δz)`. These are compared with the continuum
//! `γ = jω√(μ0·ε0·ε_r·(1 − jκ/(ωε)))` (§9: 1 %) and with the exact discrete
//! wavenumber of the semi-implicit lossy Yee scheme (`common::yee_k`).

mod common;

use std::f64::consts::PI;

use common::*;
use ems::engine::Material;

const DZ: f64 = 0.5e-3;
const EPS_R: f64 = 2.0;
const KAPPA: f64 = 0.2;

/// Measures `(α, β)` at `freqs` with probes `d` cells apart.
fn measure(freqs: &[f64], d: usize) -> (f64, Vec<(f64, f64)>) {
    let (src, p1) = (48, 68);
    let mut l = Line::new(DZ, p1 + d + 60 + 8);
    l.background = Material::dielectric(EPS_R).with_kappa(KAPPA);
    l.src = src;
    l.probes = vec![p1, p1 + d];
    let (dt, v) = l.run(sine_pulse(4.5e9, 3.5e9), 3000);
    // The low-frequency part of the response diffuses (algebraic tail,
    // ~t^{-3/2}); it is smooth and far below the band, so 1e-4 of the peak
    // at the end of the record is plenty.
    for x in &v {
        let tail = peak(&x[x.len() - 50..]) / peak(x);
        assert!(tail < 1e-4, "record too short: tail {tail:e}");
    }
    let a = spectrum(&v[0], dt, freqs);
    let b = spectrum(&v[1], dt, freqs);
    let len = d as f64 * DZ;
    let ab = a
        .iter()
        .zip(&b)
        .zip(freqs)
        .map(|((&a, &b), &f)| {
            let h = b / a;
            let alpha = -h.norm().ln() / len;
            // Unwrap the phase around the continuum estimate.
            let est = continuum_gamma(f, EPS_R, KAPPA).im * len;
            let ph = -h.arg();
            let m = ((est - ph) / (2.0 * PI)).round();
            (alpha, (ph + 2.0 * PI * m) / len)
        })
        .collect();
    (dt, ab)
}

/// α(f) within 1 % of the continuum value over 1–8 GHz, and within 1e-3
/// (relative) of the discrete Yee value; β likewise.
#[test]
fn conductive_medium_attenuation() {
    let _t = Timer::start("conductive_medium_attenuation");
    let freqs = linspace(1e9, 8e9, 15);
    let (dt, m) = measure(&freqs, 40);
    let (mut ea, mut eb, mut ya, mut yb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&f, &(a, b)) in freqs.iter().zip(&m) {
        let g = continuum_gamma(f, EPS_R, KAPPA);
        let k = yee_k(f, dt, DZ, EPS_R, KAPPA);
        let (ay, by) = (-k.im / DZ, k.re / DZ);
        eprintln!(
            "  {:.2} GHz: α = {a:.4} Np/m (continuum {:.4}, Yee {ay:.4}), \
             β = {b:.3} rad/m (continuum {:.3}, Yee {by:.3})",
            f / 1e9,
            g.re,
            g.im
        );
        ea = ea.max((a / g.re - 1.0).abs());
        eb = eb.max((b / g.im - 1.0).abs());
        ya = ya.max((a / ay - 1.0).abs());
        yb = yb.max((b / by - 1.0).abs());
    }
    eprintln!(
        "conductive medium (ε_r = {EPS_R}, κ = {KAPPA} S/m): max rel. error vs continuum \
         α {:.3} %, β {:.3} %; vs Yee α {ya:.2e}, β {yb:.2e}",
        100.0 * ea,
        100.0 * eb
    );
    assert!(ea < 0.01 && eb < 0.01, "{ea} {eb}");
    assert!(ya < 1e-3 && yb < 1e-3, "{ya} {yb}");
}
