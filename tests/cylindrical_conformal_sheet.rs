//! Conformal thin PEC sheets on a cylindrical grid (design §5.13.1,
//! §5.15): zero-width cylindrical shells as circular cavity walls, TM010
//! against the Bessel zero.
//!
//! * A shell coaxial with the grid, of radius between two ρ lines (closed
//!   azimuth with the axis): it cuts every radial edge at `ρ = r`,
//!   including the two stored copies on the α = 0 seam, and its trace in
//!   the z faces is the straight parameter line `ρ = r`.
//! * A shell off the axis on an annular grid with a closed azimuth,
//!   crossing the α = 0 seam: arcs (α edges) are cut and the traces are
//!   curves in `(ρ, α)`.
//!
//! csx cannot discretise either shell (it is on no mesh surface), so there
//! is no staircase to compare with; the conformal error must be small
//! and converge, and the energy must stay constant after the source.

mod common;

use std::f64::consts::{PI, TAU};

use csx::primitives::CylindricalShell;
use csx::{ContinuousStructure, CoordSystem, Primitive, Property};
use ems::Axis;
use ems::Fdtd;
use ems::compile::{CompileOptions, compile};
use ems::constants::C0;
use ems::engine::{ConformalReport, EdgeSource, Probe};

const J01: f64 = 2.404_825_557_695_773;

fn spectrum(dt: f64, v: &[f64], first: usize, f: f64) -> f64 {
    let v = &v[first..];
    let n = v.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (q, &x) in v.iter().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * PI * q as f64 / (n - 1.0)).cos();
        let ph = -2.0 * PI * f * (q as f64 + first as f64) * dt;
        re += w * x * ph.cos();
        im += w * x * ph.sin();
    }
    re.hypot(im)
}

fn peak(dt: f64, v: &[f64], first: usize, lo: f64, hi: f64) -> f64 {
    let s = |f: f64| spectrum(dt, v, first, f);
    let n = 300;
    let step = (hi - lo) / n as f64;
    let best = (0..=n)
        .map(|q| (q, s(lo + q as f64 * step)))
        .fold((0, f64::MIN), |a, b| if b.1 > a.1 { b } else { a })
        .0;
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
    0.5 * (a + b)
}

fn nearest(l: &[f64], x: f64) -> usize {
    (0..l.len())
        .min_by(|&a, &b| (l[a] - x).abs().total_cmp(&(l[b] - x).abs()))
        .unwrap()
}

struct Case {
    /// ρ, α, z lines (mm, rad, mm).
    lines: [Vec<f64>; 3],
    /// Shell centre (x, y) and radius (mm).
    centre: (f64, f64),
    radius: f64,
}

/// Relative TM010 error of the shell cavity, the conformal report and the
/// energy at four instants after the source.
fn tm010(c: &Case, threads: usize) -> (f64, ConformalReport, Vec<f64>) {
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    for a in 0..3 {
        m.grid.set_lines(a, c.lines[a].iter().copied()).unwrap();
    }
    let z1 = c.lines[2][c.lines[2].len() - 1];
    let (cx, cy) = c.centre;
    let metal = m.add_property(Property::metal("shell"));
    m.add_primitive(
        metal,
        Primitive::new(CylindricalShell::new(
            [cx, cy, -1.0],
            [cx, cy, z1 + 1.0],
            c.radius,
            0.0,
        ))
        .with_coord_system(CoordSystem::Cartesian),
    )
    .unwrap();
    let mut fdtd = Fdtd::new();
    if threads > 0 {
        fdtd = fdtd.threads(threads);
    }
    let comp = compile(&m, &fdtd, &CompileOptions::default().with_conformal(true)).unwrap();
    assert!(
        comp.warnings.iter().all(|w| w.starts_with("mesh ")),
        "{:?}",
        comp.warnings
    );
    let (mut e, hd) = comp.build().unwrap();
    let pol = |x: f64, y: f64| {
        let a = y.atan2(x).rem_euclid(TAU);
        [nearest(&c.lines[0], x.hypot(y)), nearest(&c.lines[1], a)]
    };
    let a = c.radius;
    let src = pol(cx + 0.15 * a, cy + 0.1 * a);
    let prb = pol(cx - 0.3 * a, cy - 0.2 * a);
    let f = J01 * C0 / (2.0 * PI * a * 1e-3);
    let bw = 0.4 * f;
    e.add_operator(EdgeSource::new(common::sine_pulse(f, bw)).edge(
        Axis::Z,
        [src[0], src[1], 0],
        1.0,
    ));
    let p = e.add_operator(Probe::voltage(Axis::Z, [prb[0], prb[1], 1]));
    let steps = (60.0 / f / e.dt()) as u64;
    let mut energy = Vec::new();
    for _ in 0..4 {
        e.run(steps / 4);
        energy.push(e.energy());
    }
    let dt = e.dt();
    let first = (common::sine_pulse_duration(bw) / dt).ceil() as usize;
    let fm = peak(dt, e.operator(&p).values(), first, 0.85 * f, 1.15 * f);
    ((fm - f) / f, hd.conformal.unwrap(), energy)
}

fn lin(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..=n)
        .map(|i| lo + (hi - lo) * i as f64 / n as f64)
        .collect()
}

/// Coaxial shell `r` = 10.3 mm, `h` mm radial cells from the axis, 16 α
/// cells, 2 cells in z.
fn coaxial(h: f64) -> Case {
    let nr = (12.0 / h).round() as usize;
    Case {
        lines: [
            lin(0.0, nr as f64 * h, nr),
            lin(0.0, TAU, 16),
            lin(0.0, 2.0 * h, 2),
        ],
        centre: (0.0, 0.0),
        radius: 10.3,
    }
}

/// Shell of radius 6 mm centred at (12, 0.3) mm (across the α = 0 seam)
/// on the annulus 4 … 20 mm, `h` mm radial cells, α cells of `h` at
/// ρ = 12 mm.
fn off_axis(h: f64) -> Case {
    let nr = (16.0 / h).round() as usize;
    let na = (TAU * 12.0 / h).round() as usize;
    Case {
        lines: [
            lin(4.0, 4.0 + nr as f64 * h, nr),
            lin(0.0, TAU, na),
            lin(0.0, 2.0 * h, 2),
        ],
        centre: (12.0, 0.3),
        radius: 6.0,
    }
}

fn check(name: &str, cases: &[(f64, Case)], tol: &[f64]) -> Vec<f64> {
    let mut errs = Vec::new();
    for ((h, c), &t) in cases.iter().zip(tol) {
        let (err, rep, w) = tm010(c, 0);
        eprintln!(
            "{name}, h = {h} mm: TM010 {:+.4}% ({} cut edges, {} cut faces, {} currents, \
             {} merged, {} floored, s = {:.4}); energy {:?}",
            err * 100.0,
            rep.sheet_edges,
            rep.sheet_faces,
            rep.sheet_currents,
            rep.sheet_merged,
            rep.sheet_floored,
            rep.stability,
            w
        );
        assert!(rep.verified, "{rep:?}");
        assert!(rep.sheet_edges > 0, "{rep:?}");
        assert!((w[3] / w[1] - 1.0).abs() < 1e-4, "{w:?}");
        assert!(err.abs() < t, "{name}, h = {h}: {err}");
        errs.push(err.abs());
    }
    errs
}

#[test]
fn coaxial_shell_tm010() {
    check(
        "coaxial shell",
        &[(1.0, coaxial(1.0)), (0.5, coaxial(0.5))],
        &[3e-3, 1e-3],
    );
}

#[test]
fn off_axis_shell_across_the_seam_tm010() {
    check(
        "off-axis shell",
        &[(1.0, off_axis(1.0)), (0.5, off_axis(0.5))],
        &[4e-3, 2e-3],
    );
}

#[test]
fn cylindrical_sheets_are_thread_independent() {
    let c = off_axis(1.0);
    let (a, _, wa) = tm010(&c, 1);
    let (b, _, wb) = tm010(&c, 3);
    assert_eq!(a, b);
    assert_eq!(wa, wb);
}

/// Coaxial shell of radius `10 + frac·h` mm: the inner side of every cut
/// α face keeps the fraction `frac` of the face at every refinement.
fn coaxial_at(h: f64, frac: f64) -> Case {
    let mut c = coaxial(h);
    c.radius = 10.0 + frac * h;
    c
}

#[test]
fn coaxial_shell_small_inner_regions_converge() {
    // frac = 0.3: small regions on the cavity side of every cut α face
    // (H_φ there is interpolated along ρ; with its current measured at the
    // face centre instead of the region centroid the error converged at
    // first order: +0.208 / +0.081 / +0.038 %).
    let cases: Vec<(f64, Case)> = [1.0, 0.5].map(|h| (h, coaxial_at(h, 0.3))).into();
    let e = check("coaxial shell (0.3 of a cell)", &cases, &[3e-3, 8e-4]);
    let order = (e[0] / e[1]).log2();
    eprintln!("observed order {order:.2}");
    assert!(order > 1.6, "{e:?}");
}

/// Three refinements of the coaxial (inner fraction 0.3) and off-axis
/// shells (design §5.13.1; `--release -- --ignored --nocapture`).
#[test]
#[ignore]
fn cylindrical_sheet_table() {
    let cases: Vec<(f64, Case)> = [1.0, 0.5, 0.25].map(|h| (h, coaxial_at(h, 0.3))).into();
    check("coaxial shell (0.3 of a cell)", &cases, &[1.0; 3]);
    let cases: Vec<(f64, Case)> = [1.0, 0.5, 0.25].map(|h| (h, off_axis(h))).into();
    check("off-axis shell", &cases, &[1.0; 3]);
}
