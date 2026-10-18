//! Conformal PEC (Dey–Mittra in integral form, `engine::conformal`):
//! resonances of curved PEC cavities on a Cartesian mesh against the
//! analytic values, staircase vs conformal; long-run energy conservation
//! with small cut cells; thread-count determinism.

mod common;

use std::f64::consts::PI;

use csx::primitives::{BoxShape, Cylinder, LinPoly, Sphere};
use csx::{ContinuousStructure, Material, Primitive, Property};
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

fn lines(lo: f64, hi: f64, h: f64) -> Vec<f64> {
    let n = ((hi - lo) / h).round() as usize;
    (0..=n).map(|q| lo + q as f64 * h).collect()
}

fn nearest(l: &[f64], x: f64) -> usize {
    (0..l.len())
        .min_by(|&a, &b| (l[a] - x).abs().total_cmp(&(l[b] - x).abs()))
        .unwrap()
}

/// A PEC block filling the mesh with a vacuum cavity of shape `cavity`
/// carved out (drawing unit mm).
fn model(lines_xyz: [Vec<f64>; 3], cavity: Primitive) -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    let (lo, hi): (Vec<f64>, Vec<f64>) = lines_xyz
        .iter()
        .map(|l| (l[0] - 1.0, l[l.len() - 1] + 1.0))
        .unzip();
    for (a, l) in lines_xyz.into_iter().enumerate() {
        m.grid.set_lines(a, l).unwrap();
    }
    let metal = m.add_property(Property::metal("block"));
    m.add_primitive(
        metal,
        BoxShape::new([lo[0], lo[1], lo[2]], [hi[0], hi[1], hi[2]]),
    )
    .unwrap();
    let air = m.add_property(Property::material("cavity", Material::new(1.0, 0.0)));
    m.add_primitive(air, cavity.with_priority(10)).unwrap();
    m
}

struct Run {
    dt: f64,
    values: Vec<f64>,
    first: usize,
    report: Option<ConformalReport>,
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
        },
        e,
    )
}

/// TM010 of a circular PEC cylinder (radius 10 mm, axis z, centre off the
/// mesh nodes) on a mesh with `per_radius` cells per radius, 2 cells in z.
fn cylinder_error(per_radius: usize, conformal: bool) -> (f64, Option<ConformalReport>) {
    let a = 10.0;
    let h = a / per_radius as f64;
    let (cx, cy) = (0.31 * h, 0.17 * h);
    let ext = a + 2.0 * h;
    let lx = lines(-ext, ext, h);
    let lz = lines(0.0, 2.0 * h, h);
    let cav = Primitive::new(Cylinder::new([cx, cy, -1.0], [cx, cy, 2.0 * h + 1.0], a));
    let m = model([lx.clone(), lx.clone(), lz], cav);
    let f = J01 * C0 / (2.0 * PI * a * 1e-3);
    let c = [nearest(&lx, cx), nearest(&lx, cy), 0];
    let p = [nearest(&lx, cx + 0.2 * a), nearest(&lx, cy - 0.1 * a), 1];
    let steps = 120 * per_radius as u64 * 5;
    let (r, _) = run(&m, conformal, f, c, p, steps, 2);
    let fm = peak(r.dt, &r.values, r.first, 0.85 * f, 1.15 * f);
    ((fm - f) / f, r.report)
}

#[test]
fn cylinder_tm010_converges_second_order() {
    let mut conf = Vec::new();
    for n in [5, 10, 20] {
        let (es, _) = cylinder_error(n, false);
        let (ec, rep) = cylinder_error(n, true);
        let rep = rep.unwrap();
        eprintln!(
            "TM010, {n} cells/radius: staircase {:+.4}%, conformal {:+.4}% \
             ({} edges, {} own faces, {} merged in {} groups, s = {:.4})",
            es * 100.0,
            ec * 100.0,
            rep.edges,
            rep.faces,
            rep.merged,
            rep.groups,
            rep.stability
        );
        assert!(rep.verified, "{rep:?}");
        assert!(ec.abs() < 0.1 * es.abs(), "conformal not better at {n}");
        conf.push(ec.abs());
    }
    assert!(
        conf[0] < 5e-3 && conf[1] < 1.2e-3 && conf[2] < 3e-4,
        "{conf:?}"
    );
    let order = (conf[0] / conf[2]).log2() / 2.0;
    eprintln!("observed order {order:.2}");
    assert!(order > 1.8, "order {order}");
}

/// TM110 of a rectangular PEC cavity (20 × 13 mm) rotated by 17° about z,
/// 2 cells in z, `h` mm cells: planar walls oblique to the mesh, where the
/// chord construction of the cut faces is exact.
fn tilted_error(h: f64, conformal: bool) -> f64 {
    let (la, lb): (f64, f64) = (20.0, 13.0);
    let th = 17f64.to_radians();
    let c0 = [0.3 * h, 0.2 * h];
    let ext = 0.5 * la.hypot(lb) + 2.0 * h;
    let lx = lines(-ext, ext, h);
    let lz = lines(0.0, 2.0 * h, h);
    let corner = |u: f64, v: f64| {
        (
            c0[0] + u * th.cos() - v * th.sin(),
            c0[1] + u * th.sin() + v * th.cos(),
        )
    };
    let verts = vec![
        corner(-0.5 * la, -0.5 * lb),
        corner(0.5 * la, -0.5 * lb),
        corner(0.5 * la, 0.5 * lb),
        corner(-0.5 * la, 0.5 * lb),
    ];
    let cav = Primitive::new(LinPoly::new(2, -1.0, 2.0 * h + 2.0, verts));
    let m = model([lx.clone(), lx.clone(), lz], cav);
    let f = 0.5 * C0 * ((1.0 / la).powi(2) + (1.0 / lb).powi(2)).sqrt() * 1e3;
    let src = [nearest(&lx, c0[0]), nearest(&lx, c0[1]), 0];
    let prb = [nearest(&lx, c0[0] + 2.0), nearest(&lx, c0[1] - 1.5), 1];
    let steps = (60.0 / f / (0.57 * h * 1e-3 / C0)) as u64;
    let (r, _) = run(&m, conformal, f, src, prb, steps, 2);
    if let Some(rep) = r.report {
        assert!(rep.verified, "{rep:?}");
    }
    let fm = peak(r.dt, &r.values, r.first, 0.85 * f, 1.15 * f);
    (fm - f) / f
}

#[test]
fn tilted_rectangular_cavity() {
    let mut e = Vec::new();
    for h in [1.0, 0.5] {
        let (es, ec) = (tilted_error(h, false), tilted_error(h, true));
        eprintln!(
            "tilted TM110, h = {h} mm: staircase {:+.4}%, conformal {:+.4}%",
            es * 100.0,
            ec * 100.0
        );
        assert!(ec.abs() < 0.1 * es.abs(), "{es} {ec}");
        e.push(ec.abs());
    }
    assert!(e[1] < 5e-4, "{e:?}");
}

/// Lowest TM mode of a PEC sphere (radius 10 mm, centre off the nodes).
fn sphere(per_radius: usize) -> (ContinuousStructure, f64, [usize; 3], [usize; 3]) {
    let a = 10.0;
    let h = a / per_radius as f64;
    let c = [0.23 * h, 0.37 * h, 0.11 * h];
    let ext = a + 2.0 * h;
    let l = lines(-ext, ext, h);
    let cav = Primitive::new(Sphere::new(c, a));
    let m = model([l.clone(), l.clone(), l.clone()], cav);
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
fn sphere_tm_mode() {
    for n in [5, 8] {
        let (m, f, src, prb) = sphere(n);
        let steps = 800 * n as u64;
        let mut err = [0.0; 2];
        for (q, conformal) in [false, true].into_iter().enumerate() {
            let (r, _) = run(&m, conformal, f, src, prb, steps, 2);
            let fm = peak(r.dt, &r.values, r.first, 0.85 * f, 1.15 * f);
            err[q] = (fm - f) / f;
            if let Some(rep) = r.report {
                eprintln!("  {rep:?}");
                assert!(rep.verified);
            }
        }
        eprintln!(
            "sphere TM1, {n} cells/radius: staircase {:+.4}%, conformal {:+.4}%",
            err[0] * 100.0,
            err[1] * 100.0
        );
        assert!(err[1].abs() < 6e-3, "{err:?}");
        assert!(err[1].abs() < 0.1 * err[0].abs(), "{err:?}");
    }
}

#[test]
fn small_cut_cells_conserve_energy() {
    // Coarse sphere: many slivers and corner cuts. The scheme is symmetric
    // (energy-conserving), so after the source has stopped W must stay
    // constant; any instability would show as growth.
    let (m, f, src, prb) = sphere(5);
    let (mut r, mut e) = run(&m, true, f, src, prb, 400, 2);
    let rep = r.report.take().unwrap();
    assert!(rep.verified, "{rep:?}");
    let w0 = e.energy();
    assert!(w0 > 0.0);
    let mut dev: f64 = 0.0;
    for _ in 0..20 {
        e.run(1000);
        dev = dev.max((e.energy() / w0 - 1.0).abs());
    }
    eprintln!("energy drift over 20000 steps: {dev:.2e} ({rep:?})");
    assert!(dev < 1e-4, "{dev}");
}

#[test]
fn conformal_is_thread_independent() {
    let (m, f, src, prb) = sphere(5);
    let (a, _) = run(&m, true, f, src, prb, 600, 1);
    let (b, _) = run(&m, true, f, src, prb, 600, 3);
    assert_eq!(a.values, b.values);
    assert_eq!(a.report, b.report);
}

#[test]
fn staircase_default_unchanged() {
    // Without the option the compile step produces no conformal data and
    // the engine runs the plain staircase model.
    let (m, f, src, prb) = sphere(5);
    let c = compile(&m, &Fdtd::new(), &CompileOptions::default()).unwrap();
    assert!(c.conformal.is_none());
    let (r, _) = run(&m, false, f, src, prb, 300, 2);
    assert!(r.report.is_none());
    assert!(r.values.iter().any(|&v| v != 0.0));
}
