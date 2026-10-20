//! Conducting sheets on cylindrical grids (design §5.9, §5.15): the slab
//! model with the `(ρ, α, z)` metric, validated on coaxial structures
//! (`a = 2 mm`, `b = 6 mm`, closed azimuth) against transmission-line
//! results of the exact slab ([`slab_impedance`]):
//!
//! * `z = const` sheet (a resistive annulus across a matched line, ρ and α
//!   edges): transmission `T = 2/(2 + η0·σ·t)` (`t ≪ δ`; the discrete line
//!   has the same `Σ Δρ/ρ_{i+½}` in its impedance and in the sheet's
//!   conductance, so the relation holds on the mesh too);
//! * `ρ = const` sheet (the inner conductor of a half-wave coaxial cavity
//!   as a thick lossy tube, z and α edges): `Q = (ωL' + X')/R'` with the
//!   series impedance `R' + jX' = Z11/(2πa)` per length;
//! * `α = const` sheet (a thin resistive fin from `a` to `b` along the same
//!   cavity, ρ and z edges, on the α seam of the closed grid and off it):
//!   `Q = ωC'/G'` with the shunt conductance `G' = σt·Σ(Δρ/ρ²)/ln²(b/a)`
//!   per length (the loss of the TEM field on the fin).
//!
//! `L'`, `C'` are those of the discrete line (`ln(b/a) → Σ Δρ/ρ_{i+½}`).

use std::f64::consts::{PI, TAU};

use csx::primitives::BoxShape;
use csx::properties::{Excitation, ProbeBox, ProbeType};
use csx::{ContinuousStructure, CoordSystem, Primitive, Property};
use ems::constants::{C0, EPS0, MU0, Z0};
use ems::engine::sheet::slab_impedance;
use ems::{Boundary, Complex64, Face, Fdtd, Simulation, Waveform};

const A: f64 = 2.0; // mm
const B: f64 = 6.0;
const NR: usize = 16;
const NA: usize = 8;
const DZ: f64 = 0.5;

#[derive(Clone, Copy)]
enum Sheet {
    None,
    /// `z = const` annulus at `z` (mm).
    Ring(f64, f64, f64),
    /// Inner conductor as a tube at `ρ = A` over the whole length.
    Tube(f64, f64),
    /// Fin at `α = α0` (0: the seam) over the whole length.
    Fin(f64, f64, f64),
}

/// A coax from `z = 0` to `nz·DZ` with a source at `z_src`, voltage probes
/// (`∫E_ρ dρ` from `a` to `b`) at `zs` (all in mm), and the sheet.
/// Returns the probe series, the time step and the warnings.
fn run(
    sheet: Sheet,
    nz: usize,
    z_src: f64,
    zs: &[f64],
    fdtd: Fdtd,
) -> (f64, Vec<Vec<f64>>, Vec<String>) {
    let r0 = if matches!(sheet, Sheet::Tube(..)) {
        1.0
    } else {
        A
    };
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    let dr = (B - A) / NR as f64;
    let n0 = ((A - r0) / dr).round() as usize;
    m.grid
        .set_lines(0, (0..=NR + n0).map(|i| r0 + i as f64 * dr))
        .unwrap();
    m.grid
        .set_lines(1, (0..=NA).map(|j| j as f64 * TAU / NA as f64))
        .unwrap();
    m.grid
        .set_lines(2, (0..=nz).map(|k| k as f64 * DZ))
        .unwrap();
    let ex = m.add_property(Property::excitation("ex", Excitation::new([1.0, 0.0, 0.0])));
    m.add_primitive(
        ex,
        Primitive::new(BoxShape::new([A, 0.0, z_src], [B, TAU, z_src])),
    )
    .unwrap();
    for (q, &z) in zs.iter().enumerate() {
        let id = m.add_property(Property::probe(
            format!("v{q}"),
            ProbeBox::new(ProbeType::Voltage),
        ));
        let al = 3.0 * TAU / NA as f64;
        m.add_primitive(id, Primitive::new(BoxShape::new([A, al, z], [B, al, z])))
            .unwrap();
    }
    let zmax = nz as f64 * DZ;
    let (prop, p0, p1) = match sheet {
        Sheet::None => (None, [0.0; 3], [0.0; 3]),
        Sheet::Ring(s, t, z) => (Some((s, t)), [A, 0.0, z], [B, TAU, z]),
        Sheet::Tube(s, t) => (Some((s, t)), [A, 0.0, 0.0], [A, TAU, zmax]),
        Sheet::Fin(s, t, a0) => (Some((s, t)), [A, a0, 0.0], [B, a0, zmax]),
    };
    if let Some((s, t)) = prop {
        let id = m.add_property(Property::conducting_sheet("sheet", s, t));
        m.add_primitive(id, Primitive::new(BoxShape::new(p0, p1)))
            .unwrap();
    }
    let mut sim = Simulation::new(&m, &fdtd).unwrap();
    let warnings = sim.warnings().to_vec();
    let res = sim.run().unwrap();
    let dt = res.probes()[0].times()[1] - res.probes()[0].times()[0];
    let v = (0..zs.len())
        .map(|q| {
            res.probe(&format!("v{q}"))
                .unwrap()
                .scalar()
                .unwrap()
                .to_vec()
        })
        .collect();
    (dt, v, warnings)
}

fn dft(v: &[f64], dt: f64, f: f64) -> Complex64 {
    v.iter()
        .enumerate()
        .map(|(n, &x)| Complex64::cis(-TAU * f * n as f64 * dt) * x)
        .fold(Complex64::ZERO, |a, b| a + b)
}

/// Discrete TEM line: `Σ Δρ/ρ_{i+½}` (the `ln(b/a)` of the mesh).
fn log_ba() -> f64 {
    let dr = (B - A) / NR as f64;
    (0..NR).map(|i| dr / (A + (i as f64 + 0.5) * dr)).sum()
}

fn only_mesh_remarks(w: &[String]) {
    assert!(w.iter().all(|w| w.starts_with("mesh ")), "{w:?}");
}

#[test]
fn z_sheet_resistive_annulus_transmission() {
    // σt = 0.01 S: η0σt = 3.77, T = 0.347 (thin: δ ≥ 50 µm ≫ 1 µm).
    let (s, t) = (1e4, 1e-6);
    let zs = [50.0, 100.0];
    let fdtd = || {
        Fdtd::new()
            .waveform(Waveform::gaussian(5e9, 5e9))
            .boundary(Face::ZMin, Boundary::pml(8))
            .boundary(Face::ZMax, Boundary::pml(8))
            .max_frequency(10e9)
            .max_timesteps(2600)
            .no_end_criteria()
    };
    let (dt, v0, w) = run(Sheet::None, 240, 10.0, &zs, fdtd());
    only_mesh_remarks(&w);
    let (_, v1, w) = run(Sheet::Ring(s, t, 75.0), 240, 10.0, &zs, fdtd());
    only_mesh_remarks(&w);
    let thin = 2.0 / (2.0 + Z0 * s * t);
    let mut worst: f64 = 0.0;
    for f in [2e9, 4e9, 6e9, 8e9] {
        let tr = dft(&v1[1], dt, f) / dft(&v0[1], dt, f);
        // Exact slab: shunt admittance 2/(Z11 + Z12) per square.
        let (z11, z12) = slab_impedance(s, t, f);
        let y = (z11 + z12).inv() * 2.0;
        let exact = Complex64::new(2.0, 0.0) / (Complex64::new(2.0, 0.0) + y * Z0);
        let err = (tr - exact).norm() / exact.norm();
        println!(
            "{:.0} GHz: T {tr:.5} exact {exact:.5} (thin {thin:.5}), {err:.2e}",
            f / 1e9
        );
        worst = worst.max(err);
    }
    assert!(worst < 3e-3, "worst {worst}");
}

/// Ring-down of the TEM half-wave mode (`L = 30 mm`, ≈ 5 GHz) of a closed
/// coaxial cavity: `(f, Q)` from the energy decay of the probe signal per
/// period after the source.
fn cavity_q(sheet: Sheet) -> (f64, f64) {
    const NZ: usize = 60;
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(5e9, 2e9))
        .max_frequency(10e9)
        .max_timesteps(9000)
        .no_end_criteria();
    let (dt, v, w) = run(sheet, NZ, 15.0, &[10.0], fdtd);
    only_mesh_remarks(&w);
    let v = &v[0];
    // Resonance: peak of the spectrum of the tail.
    let t0 = (2e-9 / dt) as usize;
    let tail = &v[t0..];
    let spec = |f: f64| dft(tail, dt, f).norm();
    let (mut lo, mut hi) = (4e9, 6e9);
    for _ in 0..60 {
        let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if spec(a) < spec(b) {
            lo = a;
        } else {
            hi = b;
        }
    }
    let f = 0.5 * (lo + hi);
    // Energy per period, then a least-squares fit of ln E against t:
    // E ∝ exp(−ω t / Q).
    let per = 1.0 / (f * dt);
    let n = (tail.len() as f64 / per).floor() as usize;
    let pts: Vec<(f64, f64)> = (0..n)
        .map(|p| {
            let (a, b) = ((p as f64 * per) as usize, ((p + 1) as f64 * per) as usize);
            let e: f64 = tail[a..b].iter().map(|x| x * x).sum::<f64>() / (b - a) as f64;
            ((p as f64 + 0.5) * per * dt, e.ln())
        })
        .collect();
    let m = pts.len() as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (mx, my) = (sx / m, sy / m);
    let slope = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>()
        / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
    (f, -TAU * f / slope)
}

#[test]
fn rho_sheet_lossy_inner_conductor_q() {
    // Thick-ish tube: t/δ = 4.4 at 5 GHz (Z11 = Z_c·coth(γt), nearly the
    // surface impedance; the region inside is shielded to ~1 %).
    let (s, t) = (1e5, 1e-4);
    let (f, q) = cavity_q(Sheet::Tube(s, t));
    let lp = MU0 / TAU * log_ba();
    let zs = slab_impedance(s, t, f).0 / (TAU * A * 1e-3);
    let want = (TAU * f * lp + zs.im) / zs.re;
    let err = q / want - 1.0;
    println!(
        "tube: f {:.4} GHz, Q {q:.2} (want {want:.2}, {err:+.4})",
        f / 1e9
    );
    assert!((f / (C0 / 2.0 / 30e-3) - 1.0).abs() < 0.02, "f {f}");
    assert!(err.abs() < 0.01, "Q {q} vs {want}");
}

#[test]
fn alpha_sheet_resistive_fin_q_on_and_off_the_seam() {
    // Thin fin, σt = 1e-4 S (Q ≈ 57).
    let (s, t) = (1e2, 1e-6);
    let l = log_ba();
    let dr = (B - A) / NR as f64 * 1e-3;
    let inv2: f64 = (0..NR)
        .map(|i| dr / ((A * 1e-3 + (i as f64 + 0.5) * dr).powi(2)))
        .sum();
    let g = s * t * inv2 / (l * l);
    let cp = TAU * EPS0 / l;
    let mut qs = Vec::new();
    for a0 in [0.0, PI / 2.0] {
        let (f, q) = cavity_q(Sheet::Fin(s, t, a0));
        let want = TAU * f * cp / g;
        let err = q / want - 1.0;
        println!(
            "fin at α = {a0:.3}: f {:.4} GHz, Q {q:.2} (want {want:.2}, {err:+.4})",
            f / 1e9
        );
        assert!(err.abs() < 0.02, "Q {q} vs {want}");
        qs.push(q);
    }
    // The seam (two stored copies, one per side) behaves like an interior
    // α line.
    assert!((qs[0] / qs[1] - 1.0).abs() < 2e-3, "{qs:?}");
}

#[test]
fn cylindrical_sheets_are_thread_count_independent() {
    let fdtd = |n: usize| {
        Fdtd::new()
            .waveform(Waveform::gaussian(5e9, 2e9))
            .max_frequency(10e9)
            .max_timesteps(1500)
            .no_end_criteria()
            .threads(n)
    };
    for sheet in [Sheet::Fin(1e2, 1e-6, 0.0), Sheet::Tube(1e5, 1e-4)] {
        let (_, a, _) = run(sheet, 60, 15.0, &[10.0], fdtd(1));
        let (_, b, _) = run(sheet, 60, 15.0, &[10.0], fdtd(3));
        assert_eq!(a, b);
    }
}
