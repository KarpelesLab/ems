//! Analytic validation (design §9): normal-incidence reflection from a
//! dielectric half-space and a finite slab (Fresnel / slab formula), and
//! from an anisotropic half-space for each polarisation.
//!
//! Setup: the quasi-1-D TEM line of `common::Line` (exact plane wave
//! between PEC/PMC walls, CPML at both ends). A probe between the source
//! and the structure records the total field; the incident field is the
//! same probe in an all-vacuum reference run, the reflected one is the
//! difference. `|Γ(f)| = |DFT(test − ref)| / |DFT(ref)|` (vacuum is
//! lossless, so the probe position only changes the phase).
//!
//! Each result is checked twice: against the continuum formula with the
//! §9 tolerance, and much more tightly against the exact reflection of the
//! discrete Yee line (`common::discrete_reflection`), which separates
//! discretisation error (expected, `O((kΔz)²)`) from implementation error.

mod common;

use common::*;
use ems::engine::Material;
use ems::{Axis, Complex64, Cpml};

const DZ: f64 = 0.5e-3;
/// Interface node (first dielectric cell).
const IFACE: usize = 148;
const SRC: usize = 48;
const PRB: usize = 88;
const F0: f64 = 4.5e9;
const BW: f64 = 3.5e9;

/// Analysis band 1–8 GHz: ≥ 37 cells per wavelength in ε_r = 4.
fn band() -> Vec<f64> {
    linspace(1e9, 8e9, 29)
}

/// Measured `Γ(f)` of `line` (with its layers) against the same line in
/// vacuum.
fn measure(line: &Line, steps: u64, freqs: &[f64]) -> (f64, Vec<Complex64>) {
    let (dt, t) = line.run(sine_pulse(F0, BW), steps);
    let mut r = line.clone();
    r.layers.clear();
    r.background = Material::VACUUM;
    let (_, inc) = r.run(sine_pulse(F0, BW), steps);
    let refl = diff(&t[0], &inc[0]);
    // Everything has left the line: the record holds the whole pulses.
    let tail = peak(&t[0][t[0].len() - 50..]) / peak(&inc[0]);
    assert!(tail < 1e-5, "record too short: tail {tail:e}");
    let a = spectrum(&inc[0], dt, freqs);
    let b = spectrum(&refl, dt, freqs);
    (dt, b.iter().zip(&a).map(|(&b, &a)| b / a).collect())
}

/// Line with a dielectric from `IFACE` to the end (into the far CPML,
/// whose σ is scaled by `1/√ε_r` to keep the layer's attenuation). The far
/// layer has 24 cells: with the default 8 its echo of the transmitted
/// wave (≈ −58 dB in ε_r = 6) shows up as a ripple of ~1e-3 in |Γ(f)|,
/// larger than the discretisation effects resolved here.
fn half_space(m: Material, pol: Axis) -> Line {
    let cells = IFACE + 100 + 24;
    let mut l = Line::new(DZ, cells);
    let eps = m.eps_r[pol.index()];
    l.pml[1] = Cpml::cells(24).sigma_factor(1.0 / eps.sqrt());
    l.pol = pol;
    l.layers.push((IFACE..cells, m));
    l.src = SRC;
    l.probes = vec![PRB];
    l
}

/// Cell profile around the interface for `discrete_reflection`.
fn profile(eps: &[(usize, f64)]) -> Vec<f64> {
    eps.iter()
        .flat_map(|&(n, e)| std::iter::repeat_n(e, n))
        .collect()
}

/// Checks a half-space of relative permittivity `eps` seen by `pol`;
/// returns (max relative error vs Fresnel, max |Γ − Γ_yee|).
fn check_half_space(m: Material, pol: Axis, label: &str) -> (f64, f64) {
    let eps = m.eps_r[pol.index()];
    let l = half_space(m, pol);
    let freqs = band();
    let (dt, g) = measure(&l, 1400, &freqs);
    let want = fresnel(eps).abs();
    let prof = profile(&[(4, 1.0), (4, eps)]);
    let (mut e_fr, mut e_yee) = (0.0f64, 0.0f64);
    for (&f, gm) in freqs.iter().zip(&g) {
        let gy = discrete_reflection(&prof, DZ, dt, f).norm();
        e_fr = e_fr.max((gm.norm() - want).abs() / want);
        e_yee = e_yee.max((gm.norm() - gy).abs());
    }
    let lo = g[0].norm();
    let hi = g[g.len() - 1].norm();
    eprintln!(
        "{label}: |Γ| = {lo:.5} (1 GHz) .. {hi:.5} (8 GHz), Fresnel {want:.5}; \
         max rel. error vs Fresnel {:.3} %, max |Γ − Γ_yee| {e_yee:.2e}",
        100.0 * e_fr
    );
    (e_fr, e_yee)
}

/// Half-space ε_r = 4: |Γ| within 1 % of (1 − √ε_r)/(1 + √ε_r) = 1/3 over
/// 1–8 GHz, and within 1e-4 of the exact discrete value.
#[test]
fn half_space_fresnel() {
    let _t = Timer::start("half_space_fresnel");
    let (e_fr, e_yee) = check_half_space(Material::dielectric(4.0), Axis::X, "half-space ε_r = 4");
    assert!(e_fr < 0.01, "{e_fr}");
    assert!(e_yee < 1e-4, "{e_yee}");
}

/// Slab ε_r = 4, 10 mm (20 cells) thick in vacuum: |Γ(f)| follows
/// `r(1 − e^{−2jβd})/(1 − r²e^{−2jβd})` over 1–8 GHz, including the null
/// at βd = π (7.5 GHz).
#[test]
fn slab_reflection_vs_analytic() {
    let _t = Timer::start("slab_reflection_vs_analytic");
    let (eps, n_slab) = (4.0, 20);
    let mut l = Line::new(DZ, IFACE + n_slab + 80 + 24);
    l.pml[1] = Cpml::cells(24);
    l.layers
        .push((IFACE..IFACE + n_slab, Material::dielectric(eps)));
    l.src = SRC;
    l.probes = vec![PRB];
    let freqs = band();
    let (dt, g) = measure(&l, 1600, &freqs);
    let d = n_slab as f64 * DZ;
    let prof = profile(&[(4, 1.0), (n_slab, eps), (4, 1.0)]);
    let (mut e_an, mut e_yee) = (0.0f64, 0.0f64);
    let (mut gmax, mut gmin) = (0.0f64, 1.0f64);
    for (&f, gm) in freqs.iter().zip(&g) {
        let ga = slab_reflection(f, eps, d).norm();
        let gy = discrete_reflection(&prof, DZ, dt, f).norm();
        e_an = e_an.max((gm.norm() - ga).abs());
        e_yee = e_yee.max((gm.norm() - gy).abs());
        gmax = gmax.max(gm.norm());
        gmin = gmin.min(gm.norm());
    }
    // Null depth at 7.5 GHz (band point 26).
    let null = g[26].norm();
    eprintln!(
        "slab ε_r = 4, d = 10 mm: |Γ| in [{gmin:.4}, {gmax:.4}] (analytic max 0.6), \
         |Γ(7.5 GHz)| = {null:.2e}; max |Γ − Γ_slab| {e_an:.2e}, max |Γ − Γ_yee| {e_yee:.2e}"
    );
    assert!(e_an < 0.01, "{e_an}");
    assert!(e_yee < 1e-4, "{e_yee}");
    // The band covers the maximum (|Γ| = 0.6 at βd = π/2) and the null.
    assert!(gmax > 0.59 && gmin < 0.02, "{gmin} {gmax}");
}

/// Diagonally anisotropic half-space ε = diag(2, 6, 1): the x-polarised
/// wave sees ε_x = 2, the y-polarised one ε_y = 6, each within 1 % of its
/// own Fresnel coefficient.
#[test]
fn anisotropic_half_space_per_polarisation() {
    let _t = Timer::start("anisotropic_half_space_per_polarisation");
    let m = Material {
        eps_r: [2.0, 6.0, 1.0],
        ..Material::VACUUM
    };
    let (ex, yx) = check_half_space(m, Axis::X, "anisotropic, E ∥ x (ε_x = 2)");
    let (ey, yy) = check_half_space(m, Axis::Y, "anisotropic, E ∥ y (ε_y = 6)");
    assert!(ex < 0.01 && ey < 0.01, "{ex} {ey}");
    assert!(yx < 1e-4 && yy < 1e-4, "{yx} {yy}");
}
