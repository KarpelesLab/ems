//! Conformal PEC on a cylindrical grid (design §5.13, §5.15): a circular
//! PEC cavity whose axis is parallel to the grid's z axis but 30 mm off it,
//! so its wall is not a coordinate surface of the `(ρ, α, z)` mesh. TM010
//! and TE11 (PMC z walls) against the Bessel zeros, staircase vs conformal
//! (Dey–Mittra with the cylindrical metric in the cut z faces, small faces
//! interpolated), and the energy stays constant after the source.

mod common;

use std::f64::consts::PI;

use csx::primitives::{BoxShape, Cylinder};
use csx::{ContinuousStructure, CoordSystem, Material, Primitive, Property};
use ems::compile::{CompileOptions, compile};
use ems::constants::C0;
use ems::engine::{ConformalReport, EdgeSource, Probe};
use ems::{Axis, Fdtd};

const J01: f64 = 2.404_825_557_695_773;
/// First zero of J1' (TE11).
const J11P: f64 = 1.841_183_781_340_659;
const A: f64 = 10.0; // cavity radius, mm
const R0: f64 = 30.0; // distance of the cavity axis from the grid axis, mm

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

/// Relative TM010 error (`te`: TE11 between PMC z walls, whose `H_z`
/// lives on the cut z faces) and the conformal report, `h` mm radial cells
/// (α cells of the same arc length at `R0`), 2 cells in z.
fn cavity_error(
    h: f64,
    conformal: bool,
    te: bool,
    threads: usize,
) -> (f64, Option<ConformalReport>, Vec<f64>) {
    let (cx, cy) = (R0 + 0.31 * h, 0.17 * h);
    let nr = ((2.0 * A + 4.0 * h) / h).round() as usize;
    let r: Vec<f64> = (0..=nr).map(|i| R0 - A - 2.0 * h + i as f64 * h).collect();
    let da = h / R0;
    let half = 0.45f64;
    let na = (2.0 * half / da).round() as usize;
    let al: Vec<f64> = (0..=na)
        .map(|j| -half + j as f64 * 2.0 * half / na as f64)
        .collect();
    let z: Vec<f64> = (0..=2).map(|k| k as f64 * h).collect();
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, r.clone()).unwrap();
    m.grid.set_lines(1, al.clone()).unwrap();
    m.grid.set_lines(2, z.clone()).unwrap();
    let metal = m.add_property(Property::metal("block"));
    m.add_primitive(
        metal,
        BoxShape::new(
            [r[0] - 1.0, -half - 0.1, -1.0],
            [r[nr] + 1.0, half + 0.1, z[2] + 1.0],
        ),
    )
    .unwrap();
    let air = m.add_property(Property::material("cavity", Material::new(1.0, 0.0)));
    m.add_primitive(
        air,
        Primitive::new(Cylinder::new([cx, cy, -1.0], [cx, cy, z[2] + 1.0], A))
            .with_coord_system(CoordSystem::Cartesian)
            .with_priority(10),
    )
    .unwrap();
    let pol = |x: f64, y: f64| [nearest(&r, x.hypot(y)), nearest(&al, y.atan2(x))];
    let src = pol(cx, cy);
    let prb = pol(cx + 0.2 * A, cy - 0.1 * A);
    let mut fdtd = Fdtd::new();
    if te {
        fdtd = fdtd
            .boundary(ems::Face::ZMin, ems::Boundary::Pmc)
            .boundary(ems::Face::ZMax, ems::Boundary::Pmc);
    }
    if threads > 0 {
        fdtd = fdtd.threads(threads);
    }
    let c = compile(
        &m,
        &fdtd,
        &CompileOptions::default().with_conformal(conformal),
    )
    .unwrap();
    assert!(
        c.warnings.iter().all(|w| w.starts_with("mesh ")),
        "{:?}",
        c.warnings
    );
    let (mut e, hd) = c.build().unwrap();
    let (f, ax) = if te {
        (J11P * C0 / (2.0 * PI * A * 1e-3), Axis::X)
    } else {
        (J01 * C0 / (2.0 * PI * A * 1e-3), Axis::Z)
    };
    let bw = 0.4 * f;
    e.add_operator(EdgeSource::new(common::sine_pulse(f, bw)).edge(ax, [src[0], src[1], 0], 1.0));
    let p = e.add_operator(Probe::voltage(ax, [prb[0], prb[1], 1]));
    let steps = (60.0 / f / e.dt()) as u64;
    let mut energy = Vec::new();
    for _ in 0..4 {
        e.run(steps / 4);
        energy.push(e.energy());
    }
    let dt = e.dt();
    let first = (common::sine_pulse_duration(bw) / dt).ceil() as usize;
    let fm = peak(dt, e.operator(&p).values(), first, 0.85 * f, 1.15 * f);
    ((fm - f) / f, hd.conformal, energy)
}

fn offset_cylinder(te: bool) -> Vec<f64> {
    let mut conf = Vec::new();
    for h in [1.0, 0.5] {
        let (es, _, _) = cavity_error(h, false, te, 0);
        let (ec, rep, w) = cavity_error(h, true, te, 0);
        let rep = rep.unwrap();
        eprintln!(
            "offset cylinder {}, {} cells/radius: staircase {:+.4}%, conformal {:+.4}% \
             ({} edges, {} faces, {} merged in {} groups, s = {:.4}); energy after the \
             source {:?}",
            if te { "TE11" } else { "TM010" },
            A / h,
            es * 100.0,
            ec * 100.0,
            rep.edges,
            rep.faces,
            rep.merged,
            rep.groups,
            rep.stability,
            w
        );
        assert!(rep.verified, "{rep:?}");
        assert!(ec.abs() < 0.1 * es.abs(), "conformal not better at h = {h}");
        // No growth after the source (lossless cavity).
        assert!((w[3] / w[1] - 1.0).abs() < 1e-3, "{w:?}");
        conf.push(ec.abs());
    }
    conf
}

#[test]
fn offset_cylinder_tm010_conformal() {
    let conf = offset_cylinder(false);
    assert!(conf[0] < 2e-3 && conf[1] < 6e-4, "{conf:?}");
}

#[test]
fn offset_cylinder_te11_conformal() {
    // Second order since the small cut faces are interpolated instead of
    // merged (+0.59 / +0.28 % before, design §5.13).
    let conf = offset_cylinder(true);
    assert!(conf[0] < 1.5e-3 && conf[1] < 4e-4, "{conf:?}");
    assert!(conf[0] > 3.0 * conf[1], "{conf:?}");
}

#[test]
fn cylindrical_conformal_is_thread_independent() {
    let (a, _, wa) = cavity_error(1.0, true, false, 1);
    let (b, _, wb) = cavity_error(1.0, true, false, 3);
    assert_eq!(a, b);
    assert_eq!(wa, wb);
}
