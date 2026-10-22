//! Junctions of conformal thin PEC sheets (design §5.13.1, "Junctions"):
//! a rectangular cavity of four oblique plates divided by a septum that
//! ends on two of them (T junctions; corners where two plates end, L
//! junctions, or cross, X junctions), and a circular cavity in solid
//! metal with a septum sheet along a slanted diameter touching the wall.
//! The septum must isolate the two parts (no field through a sub-cell gap
//! at the junctions), and the driven part resonates at its analytic TM
//! frequency.

mod common;

use std::f64::consts::PI;

use csx::primitives::{BoxShape, Cylinder};
use csx::{ContinuousStructure, Material, Primitive, Property, Transform, Vec3};
use ems::compile::{CompileOptions, compile};
use ems::constants::C0;
use ems::engine::{ConformalReport, EdgeSource, Engine, Probe};
use ems::{Axis, Boundary, Face, Fdtd};

/// First zero of J1 (TM11 of a circle, the lowest TM mode of a half disc).
const J11: f64 = 3.831_705_970_207_512;
/// First zero of J1' (TE11, the lowest TE mode of a half disc).
const J11P: f64 = 1.841_183_781_340_659;

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

fn lines(lo: f64, hi: f64, h: f64) -> Vec<f64> {
    let n = ((hi - lo) / h).round() as usize;
    (0..=n).map(|q| lo + q as f64 * h).collect()
}

fn nearest(l: &[f64], x: f64) -> usize {
    (0..l.len())
        .min_by(|&a, &b| (l[a] - x).abs().total_cmp(&(l[b] - x).abs()))
        .unwrap()
}

/// The resonance error and the leak of a 2-D cross-section run (see
/// [`run`]).
struct Run {
    f: f64,
    leak: f64,
    report: ConformalReport,
}

/// Runs a 2-D cross-section (2 cells in z, PEC around; `te`: PMC z
/// faces, an in-plane dipole, else E_z) with source and probe at the
/// local points `src`, `prb` (cavity frame `to_world`): the resonance
/// error against `f`, and the largest |E| on the final state's nodes for
/// which `isolated` holds, relative to the probe's largest value.
#[allow(clippy::too_many_arguments)]
fn run(
    m: &ContinuousStructure,
    l: &[Vec<f64>; 2],
    f: f64,
    te: bool,
    to_world: &Transform,
    src: (f64, f64),
    prb: (f64, f64),
    isolated: &dyn Fn(f64, f64) -> bool,
    threads: usize,
) -> (Run, Engine) {
    let mut fdtd = Fdtd::new().threads(threads);
    if te {
        fdtd = fdtd
            .boundary(Face::ZMin, Boundary::Pmc)
            .boundary(Face::ZMax, Boundary::Pmc);
    }
    let c = compile(m, &fdtd, &CompileOptions::default().with_conformal(true)).unwrap();
    assert!(
        !c.warnings.iter().any(|w| w.contains("sheet")),
        "{:?}",
        c.warnings
    );
    let (mut e, h) = c.build().unwrap();
    let node = |p: (f64, f64), k: usize| {
        let w = to_world.apply(Vec3::new(p.0, p.1, 0.0));
        [nearest(&l[0], w[0]), nearest(&l[1], w[1]), k]
    };
    let bw = 0.4 * f;
    // TE: a generic in-plane dipole (source = probe direction).
    let dirs: &[(Axis, f64)] = if te {
        &[(Axis::X, 0.6), (Axis::Y, 0.8)]
    } else {
        &[(Axis::Z, 1.0)]
    };
    let mut source = EdgeSource::new(common::sine_pulse(f, bw));
    for &(a, w) in dirs {
        source = source.edge(a, node(src, if te { 1 } else { 0 }), w);
    }
    e.add_operator(source);
    let probes: Vec<_> = dirs
        .iter()
        .map(|&(a, w)| (e.add_operator(Probe::voltage(a, node(prb, 1))), w))
        .collect();
    e.run((60.0 / f / e.dt()) as u64);
    let dt = e.dt();
    let mut v: Vec<f64> = Vec::new();
    for (p, w) in &probes {
        let x = e.operator(p).values();
        v.resize(x.len(), 0.0);
        for (s, x) in v.iter_mut().zip(x) {
            *s += w * x;
        }
    }
    let first = (common::sine_pulse_duration(bw) / dt).ceil() as usize;
    let fm = peak(dt, &v, first, 0.85 * f, 1.15 * f);
    let inside = v.iter().fold(0f64, |m, x| m.max(x.abs()));
    // Edges with both ends isolated.
    let iso = |i: usize, j: usize| {
        let q = to_world.apply_inverse(Vec3::new(l[0][i], l[1][j], 0.0));
        isolated(q[0], q[1])
    };
    let mut out = 0f64;
    for i in 0..l[0].len() - 1 {
        for j in 0..l[1].len() - 1 {
            for (a, ok) in [
                (Axis::X, iso(i, j) && iso(i + 1, j)),
                (Axis::Y, iso(i, j) && iso(i, j + 1)),
                (Axis::Z, iso(i, j)),
            ] {
                if ok {
                    out = out.max(e.u(a, [i, j, 1]).abs() as f64);
                }
            }
        }
    }
    (
        Run {
            f: (fm - f) / f,
            leak: out / inside,
            report: h.conformal.unwrap(),
        },
        e,
    )
}

/// How the sheets of the rectangle meet.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Ends {
    /// Every plate ends exactly on the plates it meets (L corners, T
    /// septum ends).
    Exact,
    /// Every plate runs 1 mm past the plates it meets (X junctions).
    Overhang,
}

/// The rectangle `[−10, 10] × [−6.5, 6.5]` mm (local frame: rotated by
/// 17° about z, centre off the nodes) of four zero-thickness plates and a
/// septum at `x = 2`: the 8 × 13 mm part is driven, its lowest TM mode
/// `f = (c/2)·√(1/8² + 1/13²)`, lowest TE mode `c/(2·13 mm)`; mesh `h`.
fn rectangle(h: f64, ends: Ends, te: bool, threads: usize) -> (Run, Engine) {
    let mut t = Transform::new();
    t.rotate_axis(2, 17f64.to_radians())
        .translate(Vec3::new(0.31 * h, 0.17 * h, 0.0));
    let l = [lines(-14.0, 14.0, h), lines(-12.0, 12.0, h)];
    let lz = lines(0.0, 2.0 * h, h);
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, l[0].iter().copied()).unwrap();
    m.grid.set_lines(1, l[1].iter().copied()).unwrap();
    m.grid.set_lines(2, lz.iter().copied()).unwrap();
    let metal = m.add_property(Property::metal("walls"));
    let o = if ends == Ends::Overhang { 1.0 } else { 0.0 };
    let (z0, z1) = (-1.0, 2.0 * h + 1.0);
    let plates = [
        ([-10.0 - o, -6.5, z0], [10.0 + o, -6.5, z1]),
        ([-10.0 - o, 6.5, z0], [10.0 + o, 6.5, z1]),
        ([-10.0, -6.5 - o, z0], [-10.0, 6.5 + o, z1]),
        ([10.0, -6.5 - o, z0], [10.0, 6.5 + o, z1]),
        // The septum.
        ([2.0, -6.5 - o, z0], [2.0, 6.5 + o, z1]),
    ];
    for (a, b) in plates {
        m.add_primitive(
            metal,
            Primitive::new(BoxShape::new(a, b)).with_transform(t.clone()),
        )
        .unwrap();
    }
    let f = if te {
        0.5 * C0 / 13e-3
    } else {
        0.5 * C0 * (1.0 / 64.0 + 1.0 / 169.0f64).sqrt() / 1e-3
    };
    // Isolated: the other part of the box and the outside, a cell away
    // from the plates.
    let isolated = |x: f64, y: f64| {
        let inside = x.abs() < 10.0 - h && y.abs() < 6.5 - h;
        let outside = x.abs() > 10.0 + h || y.abs() > 6.5 + h;
        (inside && x < 2.0 - h) || outside
    };
    run(
        &m,
        &l,
        f,
        te,
        &t,
        (5.1, 1.3),
        (7.6, -3.1),
        &isolated,
        threads,
    )
}

#[test]
fn septum_in_a_box_of_plates_isolates() {
    for (ends, te) in [
        (Ends::Exact, true),
        (Ends::Overhang, true),
        (Ends::Exact, false),
    ] {
        let mut errs = Vec::new();
        for h in [1.0, 0.5] {
            let (r, _) = rectangle(h, ends, te, 4);
            eprintln!(
                "box of plates ({ends:?}), h = {h}: {} of the 8 × 13 part {:+.4}%, \
                 leak {:.1e} ({:?})",
                if te { "TE01" } else { "TM11" },
                100.0 * r.f,
                r.leak,
                r.report
            );
            assert!(r.report.verified, "{:?}", r.report);
            assert!(
                r.leak < 1e-5,
                "{ends:?} te {te} h = {h}: leak {:.2e}",
                r.leak
            );
            errs.push(r.f.abs());
        }
        assert!(errs[1] < 0.2e-2 && errs[1] < errs[0], "{errs:?}");
    }
}

/// A circular cavity (radius 10 mm, centre off the nodes) carved from a
/// PEC block (conformal solid metal), divided by a septum sheet along a
/// diameter at 23°: of length `2a` (ending on the wall) or running 1 mm
/// into the metal. The driven half's lowest TM mode is TM11 of the disc,
/// its lowest TE mode TE11.
fn half_disc(h: f64, overhang: bool, te: bool) -> (Run, Engine) {
    let a = 10.0;
    let (cx, cy) = (0.31 * h, 0.17 * h);
    let mut t = Transform::new();
    t.rotate_axis(2, 23f64.to_radians())
        .translate(Vec3::new(cx, cy, 0.0));
    let ext = a + 2.0 * h;
    let l = [lines(-ext, ext, h), lines(-ext, ext, h)];
    let lz = lines(0.0, 2.0 * h, h);
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, l[0].iter().copied()).unwrap();
    m.grid.set_lines(1, l[1].iter().copied()).unwrap();
    m.grid.set_lines(2, lz.iter().copied()).unwrap();
    let metal = m.add_property(Property::metal("wall"));
    m.add_primitive(
        metal,
        Primitive::new(BoxShape::new(
            [-ext - 1.0, -ext - 1.0, -1.0],
            [ext + 1.0, ext + 1.0, 2.0 * h + 1.0],
        )),
    )
    .unwrap();
    let air = m.add_property(Property::material("cavity", Material::new(1.0, 0.0)));
    m.add_primitive(
        air,
        Primitive::new(Cylinder::new([cx, cy, -1.0], [cx, cy, 2.0 * h + 1.0], a)).with_priority(10),
    )
    .unwrap();
    let sheet = m.add_property(Property::metal("septum"));
    let o = if overhang { 1.0 } else { 0.0 };
    m.add_primitive(
        sheet,
        Primitive::new(BoxShape::new(
            [-a - o, 0.0, -1.0],
            [a + o, 0.0, 2.0 * h + 1.0],
        ))
        .with_transform(t.clone())
        .with_priority(20),
    )
    .unwrap();
    let f = if te { J11P } else { J11 } * C0 / (2.0 * PI * a * 1e-3);
    let isolated = |x: f64, y: f64| x.hypot(y) < a - h && y < -h;
    run(&m, &l, f, te, &t, (2.9, 4.1), (-3.7, 5.2), &isolated, 4)
}

#[test]
fn septum_touching_solid_metal_isolates() {
    for (overhang, te) in [(false, true), (true, true), (false, false)] {
        let mut errs = Vec::new();
        for h in [1.0, 0.5] {
            let (r, _) = half_disc(h, overhang, te);
            eprintln!(
                "half disc (septum into the metal: {overhang}), h = {h}: {} {:+.4}%, \
                 leak {:.1e} ({:?})",
                if te { "TE11" } else { "TM11" },
                100.0 * r.f,
                r.leak,
                r.report
            );
            assert!(r.report.verified, "{:?}", r.report);
            assert!(
                r.leak < 1e-5,
                "overhang {overhang} h = {h}: leak {:.2e}",
                r.leak
            );
            errs.push(r.f.abs());
        }
        assert!(errs[1] < 0.3e-2 && errs[1] < errs[0], "{errs:?}");
    }
}

#[test]
fn junctions_are_thread_independent() {
    let (a, ea) = rectangle(1.0, Ends::Exact, true, 1);
    let (b, eb) = rectangle(1.0, Ends::Exact, true, 3);
    assert_eq!(a.f, b.f);
    assert_eq!(a.leak, b.leak);
    let n = ea.grid().nodes();
    for i in 0..n[0] {
        for j in 0..n[1] {
            assert_eq!(ea.u(Axis::Z, [i, j, 1]), eb.u(Axis::Z, [i, j, 1]));
        }
    }
}

/// The half disc on a cylindrical grid: a zero-width cylindrical shell
/// (radius 6 mm, centre 12 mm off the axis; annular grid with a closed
/// azimuth, α cells of `h` at 12 mm) divided by a septum plate along a
/// diameter at 23°, ending on the shell (T junctions of two sheets in the
/// `(ρ, α)` z faces). TE11 of the driven half (PMC z faces); the resonance
/// error, the leak into the other half and the report.
fn cyl_half_disc(h: f64) -> (f64, f64, ConformalReport) {
    use csx::CoordSystem;
    use csx::primitives::CylindricalShell;
    use std::f64::consts::TAU;
    let r = 6.0;
    let (cx, cy) = (12.0, 0.3);
    let nr = (16.0 / h).round() as usize;
    let na = (TAU * 12.0 / h).round() as usize;
    let lin = |lo: f64, hi: f64, n: usize| -> Vec<f64> {
        (0..=n)
            .map(|q| lo + (hi - lo) * q as f64 / n as f64)
            .collect()
    };
    let l = [
        lin(4.0, 4.0 + nr as f64 * h, nr),
        lin(0.0, TAU, na),
        lin(0.0, 2.0 * h, 2),
    ];
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    for (a, x) in l.iter().enumerate() {
        m.grid.set_lines(a, x.iter().copied()).unwrap();
    }
    let metal = m.add_property(Property::metal("walls"));
    m.add_primitive(
        metal,
        Primitive::new(CylindricalShell::new(
            [cx, cy, -1.0],
            [cx, cy, 2.0 * h + 1.0],
            r,
            0.0,
        ))
        .with_coord_system(CoordSystem::Cartesian),
    )
    .unwrap();
    let mut t = Transform::new();
    t.rotate_axis(2, 23f64.to_radians())
        .translate(Vec3::new(cx, cy, 0.0));
    m.add_primitive(
        metal,
        Primitive::new(BoxShape::new([-r, 0.0, -1.0], [r, 0.0, 2.0 * h + 1.0]))
            .with_transform(t.clone())
            .with_coord_system(CoordSystem::Cartesian),
    )
    .unwrap();
    let fdtd = Fdtd::new()
        .threads(4)
        .boundary(Face::ZMin, Boundary::Pmc)
        .boundary(Face::ZMax, Boundary::Pmc);
    let c = compile(&m, &fdtd, &CompileOptions::default().with_conformal(true)).unwrap();
    assert!(
        c.warnings.iter().all(|w| w.starts_with("mesh ")),
        "{:?}",
        c.warnings
    );
    let (mut e, hd) = c.build().unwrap();
    let node = |p: (f64, f64)| {
        let w = t.apply(Vec3::new(p.0, p.1, 0.0));
        let a = w[1].atan2(w[0]).rem_euclid(TAU);
        [nearest(&l[0], w[0].hypot(w[1])), nearest(&l[1], a), 1]
    };
    let f = J11P * C0 / (2.0 * PI * r * 1e-3);
    let bw = 0.4 * f;
    let dirs = [(Axis::X, 0.6), (Axis::Y, 0.8)];
    let mut src = EdgeSource::new(common::sine_pulse(f, bw));
    for (a, w) in dirs {
        src = src.edge(a, node((1.7, 2.5)), w);
    }
    e.add_operator(src);
    let probes: Vec<_> = dirs
        .map(|(a, w)| (e.add_operator(Probe::voltage(a, node((-2.2, 3.1)))), w))
        .into();
    e.run((60.0 / f / e.dt()) as u64);
    let dt = e.dt();
    let mut v: Vec<f64> = Vec::new();
    for (p, w) in &probes {
        let x = e.operator(p).values();
        v.resize(x.len(), 0.0);
        for (s, x) in v.iter_mut().zip(x) {
            *s += w * x;
        }
    }
    let first = (common::sine_pulse_duration(bw) / dt).ceil() as usize;
    let fm = peak(dt, &v, first, 0.85 * f, 1.15 * f);
    let inside = v.iter().fold(0f64, |m, x| m.max(x.abs()));
    // Edges with both ends in the other half, a cell from the walls.
    let iso = |i: usize, j: usize| {
        let (rho, a) = (l[0][i], l[1][j]);
        let q = t.apply_inverse(Vec3::new(rho * a.cos(), rho * a.sin(), 0.0));
        q[0].hypot(q[1]) < r - h && q[1] < -h
    };
    let mut out = 0f64;
    for i in 0..l[0].len() - 1 {
        for j in 0..l[1].len() - 1 {
            for (ax, ok) in [
                (Axis::X, iso(i, j) && iso(i + 1, j)),
                (Axis::Y, iso(i, j) && iso(i, j + 1)),
                (Axis::Z, iso(i, j)),
            ] {
                if ok {
                    out = out.max(e.u(ax, [i, j, 1]).abs() as f64);
                }
            }
        }
    }
    ((fm - f) / f, out / inside, hd.conformal.unwrap())
}

#[test]
fn septum_on_a_cylindrical_grid_isolates() {
    let mut errs = Vec::new();
    for h in [1.0, 0.5] {
        let (err, leak, rep) = cyl_half_disc(h);
        eprintln!(
            "cylindrical grid, half disc of sheets, h = {h}: TE11 {:+.4}%, leak {leak:.1e} \
             ({rep:?})",
            100.0 * err
        );
        assert!(rep.verified, "{rep:?}");
        assert!(leak < 1e-5, "h = {h}: leak {leak:.2e}");
        errs.push(err.abs());
    }
    assert!(errs[1] < 0.3e-2 && errs[1] < errs[0], "{errs:?}");
}
