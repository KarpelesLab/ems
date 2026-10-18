//! Conformal zero-thickness PEC sheets (`engine::conformal`, thin
//! sheets): cavities bounded by oblique and curved sheets against the
//! analytic resonances, staircase vs conformal; long-run energy
//! conservation; thread-count determinism.

mod common;

use std::f64::consts::{FRAC_PI_4, PI};

use csx::primitives::{BoxShape, Cylinder, CylindricalShell, LinPoly, SphericalShell};
use csx::{ContinuousStructure, Material, Primitive, Property, Transform};
use ems::compile::{CompileOptions, compile};
use ems::constants::C0;
use ems::engine::{ConformalReport, EdgeSource, Engine, Probe};
use ems::{Axis, Fdtd};

/// First zero of J0 (TM010 of a circular cylinder).
const J01: f64 = 2.404_825_557_695_773;
/// First zero of d/dx[x·j1(x)] (lowest TM mode of a sphere).
const SPHERE_TM1: f64 = 2.743_707_269_992_269;

/// |DFT| at `f` of the Hann-windowed samples from index `first`.
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

/// Spectral peak in `[lo, hi]`: coarse scan, then golden-section search.
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

fn lines(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..=n)
        .map(|q| lo + (hi - lo) * q as f64 / n as f64)
        .collect()
}

fn nearest(l: &[f64], x: f64) -> usize {
    (0..l.len())
        .min_by(|&a, &b| (l[a] - x).abs().total_cmp(&(l[b] - x).abs()))
        .unwrap()
}

/// An empty model (drawing unit mm) on the given mesh with one metal
/// property holding `prims`.
fn model(lines_xyz: [Vec<f64>; 3], prims: Vec<Primitive>) -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for (a, l) in lines_xyz.into_iter().enumerate() {
        m.grid.set_lines(a, l).unwrap();
    }
    let metal = m.add_property(Property::metal("sheet"));
    for p in prims {
        m.add_primitive(metal, p).unwrap();
    }
    m
}

struct Run {
    dt: f64,
    values: Vec<f64>,
    first: usize,
    report: Option<ConformalReport>,
    warnings: Vec<String>,
}

/// Compiles `m` (PEC outer faces) with or without conformal PEC, drives a
/// soft E_z pulse at node `src` and probes E_z at node `prb`.
fn run(
    m: &ContinuousStructure,
    conformal: bool,
    f0: f64,
    src: [usize; 3],
    prb: [usize; 3],
    steps: u64,
    threads: usize,
) -> (Run, Engine) {
    let fdtd = Fdtd::new().threads(threads);
    let c = compile(
        m,
        &fdtd,
        &CompileOptions::default().with_conformal(conformal),
    )
    .unwrap();
    let warnings = c.warnings.clone();
    let (mut e, h) = c.build().unwrap();
    let bw = 0.4 * f0;
    e.add_operator(EdgeSource::new(common::sine_pulse(f0, bw)).edge(Axis::Z, src, 1.0));
    let p = e.add_operator(Probe::voltage(Axis::Z, prb));
    e.run(steps);
    let dt = e.dt();
    let first = (common::sine_pulse_duration(bw) / dt).ceil() as usize;
    let values = e.operator(&p).values().to_vec();
    (
        Run {
            dt,
            values,
            first,
            report: h.conformal,
            warnings,
        },
        e,
    )
}

/// TM010 of a circular cavity (radius 10 mm, axis z, centre off the mesh
/// nodes, 2 cells in z) with `per_radius` cells per radius. `sheet`: the
/// wall is a zero-width cylindrical shell (conformal sheet); otherwise a
/// PEC block with a cylindrical hole, staircased.
fn cylinder_error(per_radius: usize, sheet: bool) -> (f64, Option<ConformalReport>) {
    let a = 10.0;
    let h = a / per_radius as f64;
    let (cx, cy) = (0.31 * h, 0.17 * h);
    let ext = a + 2.0 * h;
    let nx = 2 * (per_radius + 2);
    let lx = lines(-ext, ext, nx);
    let lz = lines(0.0, 2.0 * h, 2);
    let axis = ([cx, cy, -1.0], [cx, cy, 2.0 * h + 1.0]);
    let m = if sheet {
        let shell = Primitive::new(CylindricalShell::new(axis.0, axis.1, a, 0.0));
        model([lx.clone(), lx.clone(), lz], vec![shell])
    } else {
        let mut m = model(
            [lx.clone(), lx.clone(), lz],
            vec![Primitive::new(BoxShape::new(
                [-ext - 1.0, -ext - 1.0, -1.0],
                [ext + 1.0, ext + 1.0, 2.0 * h + 1.0],
            ))],
        );
        let air = m.add_property(Property::material("cavity", Material::new(1.0, 0.0)));
        m.add_primitive(
            air,
            Primitive::new(Cylinder::new(axis.0, axis.1, a)).with_priority(10),
        )
        .unwrap();
        m
    };
    let f = J01 * C0 / (2.0 * PI * a * 1e-3);
    let c = [nearest(&lx, cx), nearest(&lx, cy), 0];
    let p = [nearest(&lx, cx + 0.2 * a), nearest(&lx, cy - 0.1 * a), 1];
    let steps = 120 * per_radius as u64 * 5;
    let (r, _) = run(&m, sheet, f, c, p, steps, 2);
    if sheet {
        assert!(
            !r.warnings.iter().any(|w| w.contains("sheet")),
            "{:?}",
            r.warnings
        );
    }
    let fm = peak(r.dt, &r.values, r.first, 0.85 * f, 1.15 * f);
    ((fm - f) / f, r.report)
}

#[test]
fn cylindrical_shell_tm010_converges() {
    let mut conf = Vec::new();
    for n in [5, 10, 20] {
        let (es, _) = cylinder_error(n, false);
        let (ec, rep) = cylinder_error(n, true);
        let rep = rep.unwrap();
        eprintln!(
            "shell TM010, {n} cells/radius: staircase {:+.4}%, conformal sheet {:+.4}% \
             ({} cut edges, {} cut faces, {} currents, {} merged, {} floored, s = {:.4})",
            es * 100.0,
            ec * 100.0,
            rep.sheet_edges,
            rep.sheet_faces,
            rep.sheet_currents,
            rep.sheet_merged,
            rep.sheet_floored,
            rep.stability
        );
        assert!(rep.verified, "{rep:?}");
        assert!(rep.sheet_edges > 0 && rep.edges == 0, "{rep:?}");
        assert!(ec.abs() < 0.2 * es.abs(), "conformal not better at {n}");
        conf.push(ec.abs());
    }
    let order = (conf[0] / conf[2]).log2() / 2.0;
    eprintln!("observed order {order:.2}");
    assert!(order > 1.6, "order {order} ({conf:?})");
}

/// Lowest TM mode (`(m, n) = (2, 1)`, `f = c·√5/(2a)`) of the right
/// isosceles triangle cut from the square `[0, a]²` (a = 20 mm, 2 cells in
/// z) by a PEC plate along the diagonal, a zero-thickness box rotated by
/// 45° about z. The mesh has `n` cells in x and `n + 1` in y, so the plate
/// is oblique to the cells and passes through no node except the corners.
/// `sheet = false`: the other half is a staircased solid prism instead.
fn triangle_error(n: usize, sheet: bool) -> (f64, Option<ConformalReport>) {
    let a = 20.0;
    let lx = lines(0.0, a, n);
    let ly = lines(0.0, a, n + 1);
    let hz = a / n as f64;
    let lz = lines(0.0, 2.0 * hz, 2);
    let prim = if sheet {
        let mut t = Transform::new();
        t.rotate_axis(2, FRAC_PI_4);
        Primitive::new(BoxShape::new(
            [-1.0, 0.0, -1.0],
            [a * 2f64.sqrt() + 1.0, 0.0, 2.0 * hz + 1.0],
        ))
        .with_transform(t)
    } else {
        // Solid half below the diagonal (y < x).
        Primitive::new(LinPoly::new(
            2,
            -1.0,
            2.0 * hz + 2.0,
            vec![(-3.0, -3.0), (a + 3.0, -3.0), (a + 3.0, a + 3.0)],
        ))
    };
    let m = model([lx.clone(), ly.clone(), lz], vec![prim]);
    let f = C0 * 5f64.sqrt() / (2.0 * a * 1e-3);
    // Inside the upper triangle (y > x).
    let src = [nearest(&lx, 0.3 * a), nearest(&ly, 0.62 * a), 0];
    let prb = [nearest(&lx, 0.17 * a), nearest(&ly, 0.8 * a), 1];
    let steps = (80.0 / f / (0.5 * hz * 1e-3 / C0)) as u64;
    let (r, e) = run(&m, sheet, f, src, prb, steps, 2);
    if sheet {
        // The plate isolates the lower triangle (f32 rounding only).
        let inside = r.values.iter().fold(0f64, |m, v| m.max(v.abs()));
        let mut below = 0f64;
        for (i, x) in lx.iter().enumerate() {
            for (j, y) in ly.iter().enumerate() {
                if *y < x - 0.5 {
                    below = below.max(e.u(Axis::Z, [i, j, 1]).abs() as f64);
                }
            }
        }
        assert!(below < 1e-5 * inside, "{below} vs {inside}");
    }
    let fm = peak(r.dt, &r.values, r.first, 0.85 * f, 1.15 * f);
    ((fm - f) / f, r.report)
}

#[test]
fn diagonal_plate_triangle() {
    let mut conf = Vec::new();
    for n in [20, 40] {
        let (es, _) = triangle_error(n, false);
        let (ec, rep) = triangle_error(n, true);
        let rep = rep.unwrap();
        eprintln!(
            "triangle TM21, n = {n}: staircase {:+.4}%, conformal plate {:+.4}% \
             ({} cut edges, {} cut faces, {} currents, {} merged, {} floored, s = {:.4})",
            es * 100.0,
            ec * 100.0,
            rep.sheet_edges,
            rep.sheet_faces,
            rep.sheet_currents,
            rep.sheet_merged,
            rep.sheet_floored,
            rep.stability
        );
        assert!(rep.verified, "{rep:?}");
        assert!(ec.abs() < 0.2 * es.abs(), "{es} {ec}");
        conf.push(ec.abs());
    }
    let order = (conf[0] / conf[1]).log2();
    eprintln!("observed order {order:.2}");
    assert!(order > 1.5, "order {order} ({conf:?})");
}

/// A spherical-shell cavity (zero-width shell, radius 10 mm, centre off
/// the nodes) in a box with `per_radius` cells per radius.
fn sphere(per_radius: usize) -> (ContinuousStructure, f64, [usize; 3], [usize; 3]) {
    let a = 10.0;
    let h = a / per_radius as f64;
    let c = [0.23 * h, 0.37 * h, 0.11 * h];
    let ext = a + 2.0 * h;
    let l = lines(-ext, ext, 2 * (per_radius + 2));
    let shell = Primitive::new(SphericalShell::new(c, a, 0.0));
    let m = model([l.clone(), l.clone(), l.clone()], vec![shell]);
    let f = SPHERE_TM1 * C0 / (2.0 * PI * a * 1e-3);
    let src = [
        nearest(&l, c[0]),
        nearest(&l, c[1]),
        nearest(&l, c[2] - 0.5 * h),
    ];
    let prb = [
        nearest(&l, c[0] + 0.15 * a),
        nearest(&l, c[1] - 0.1 * a),
        nearest(&l, c[2] + 0.2 * a),
    ];
    (m, f, src, prb)
}

#[test]
fn spherical_shell_tm_mode_and_energy() {
    // Coarse: many corner cuts and slivers in all three orientations.
    let (m, f, src, prb) = sphere(5);
    let (mut r, mut e) = run(&m, true, f, src, prb, 4000, 2);
    let rep = r.report.take().unwrap();
    eprintln!("sphere shell, 5 cells/radius: {rep:?}");
    assert!(rep.verified, "{rep:?}");
    let fm = peak(r.dt, &r.values, r.first, 0.85 * f, 1.15 * f);
    let err = (fm - f) / f;
    eprintln!("sphere shell TM1: conformal {:+.4}%", err * 100.0);
    assert!(err.abs() < 1e-2, "{err}");
    // Lossless and symmetric: W stays constant after the source.
    let w0 = e.energy();
    assert!(w0 > 0.0);
    let mut dev: f64 = 0.0;
    for _ in 0..20 {
        e.run(1000);
        dev = dev.max((e.energy() / w0 - 1.0).abs());
    }
    eprintln!("energy drift over 20000 steps: {dev:.2e}");
    assert!(dev < 1e-4, "{dev}");
    // The in-step energy agrees with the look-ahead.
    let a = e.energy();
    let b = e.step_with_energy();
    assert_eq!(a, b);
}

/// A finite square plate (12 × 12 mm) tilted about two axes in a PEC box:
/// plate edges inside cut faces (slits), plate interior cuts in all three
/// orientations.
fn tilted_plate() -> (ContinuousStructure, [usize; 3], [usize; 3]) {
    let l = lines(-10.0, 10.0, 16);
    let mut t = Transform::new();
    t.rotate_axis(0, 0.35)
        .rotate_axis(2, 0.6)
        .translate([0.31, -0.22, 0.13].into());
    let plate = Primitive::new(BoxShape::new([-6.0, -6.0, 0.0], [6.0, 6.0, 0.0])).with_transform(t);
    let m = model([l.clone(), l.clone(), l.clone()], vec![plate]);
    (m, [5, 6, 4], [10, 9, 11])
}

#[test]
fn finite_tilted_plate_conserves_energy() {
    let (m, src, prb) = tilted_plate();
    let f = 12e9;
    let (mut r, mut e) = run(&m, true, f, src, prb, 300, 2);
    let rep = r.report.take().unwrap();
    eprintln!("tilted plate: {rep:?}");
    assert!(rep.verified && rep.sheet_edges > 0, "{rep:?}");
    let w0 = e.energy();
    assert!(w0 > 0.0);
    let mut dev: f64 = 0.0;
    for _ in 0..20 {
        e.run(1000);
        dev = dev.max((e.energy() / w0 - 1.0).abs());
    }
    eprintln!("energy drift over 20000 steps: {dev:.2e}");
    assert!(dev < 1e-4, "{dev}");
}

#[test]
fn unmerged_sheet_regions_are_unstable() {
    // Plain split cells (threshold 0) fail the stability check by far;
    // the default merging passes it (see spherical_shell_tm_mode_and_energy).
    let (m, ..) = sphere(5);
    let mut o = CompileOptions::default().with_conformal(true);
    o.conformal_options.merge_below = 0.0;
    o.conformal_options.max_rounds = 1;
    let c = compile(&m, &Fdtd::new().threads(1), &o).unwrap();
    let (_, h) = c.build().unwrap();
    let rep = h.conformal.unwrap();
    eprintln!("sphere shell without merging: s = {:.2}", rep.stability);
    assert!(!rep.verified && rep.stability > 2.0, "{rep:?}");
    assert_eq!(rep.sheet_merged, 0);
}

#[test]
fn sheets_are_thread_independent() {
    let (m, src, prb) = tilted_plate();
    let (a, _) = run(&m, true, 12e9, src, prb, 500, 1);
    let (b, _) = run(&m, true, 12e9, src, prb, 500, 4);
    assert_eq!(a.values, b.values);
    assert_eq!(a.report, b.report);
    let (m, f, src, prb) = sphere(5);
    let (a, _) = run(&m, true, f, src, prb, 500, 1);
    let (b, _) = run(&m, true, f, src, prb, 500, 3);
    assert_eq!(a.values, b.values);
    assert_eq!(a.report, b.report);
}

#[test]
fn sheets_need_the_conformal_option() {
    // Without the option, csx cannot discretise the oblique sheet (and
    // warns); with it the sheet is handled and the warning removed.
    let (m, ..) = sphere(5);
    let c = compile(&m, &Fdtd::new(), &CompileOptions::default()).unwrap();
    assert!(c.conformal.is_none());
    let c = compile(
        &m,
        &Fdtd::new(),
        &CompileOptions::default().with_conformal(true),
    )
    .unwrap();
    let g = c.conformal.unwrap();
    assert!(!g.sheet_edges().is_empty() && !g.sheet_faces().is_empty());
    assert!(g.edges().is_empty() && g.faces().is_empty());
    // A model without sheets has no sheet data.
    let mut m2 = model(
        [lines(0.0, 4.0, 4), lines(0.0, 4.0, 4), lines(0.0, 4.0, 4)],
        vec![],
    );
    let mp = m2.add_property(Property::metal("ball"));
    m2.add_primitive(
        mp,
        Primitive::new(csx::primitives::Sphere::new([2.1, 1.9, 2.05], 1.3)),
    )
    .unwrap();
    let c = compile(
        &m2,
        &Fdtd::new(),
        &CompileOptions::default().with_conformal(true),
    )
    .unwrap();
    let g = c.conformal.unwrap();
    assert!(g.sheet_edges().is_empty() && g.sheet_faces().is_empty() && !g.edges().is_empty());
}
