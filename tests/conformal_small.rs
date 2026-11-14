//! Conformal PEC on round copper of about a cell (design §5.13.3): a via
//! stub.
//!
//! A post from the floor of a closed box through 1 mm of `ε_r` = 4, with a
//! pad in the round clearance of a plane on top of the laminate (a via
//! and its pad in an antipad), or ending on a plane without a hole (a
//! shorting post). The box's lowest mode has its `E_z` maximum on the
//! post, so its frequency follows the inductance of the post and the
//! capacitance of the pad: 15.46 GHz for the stub and 22.28 GHz for the
//! shorting post on 0.05 mm cells, where the staircase and the conformal
//! cells agree within 1.2 %. The tests run 0.3 mm cells: the post
//! (0.3 mm) one cell across, the pad (0.6 mm) two, the clearance ring
//! (0.2 mm) less than one.
//!
//! * The open area of a face the post crosses between two opposite sides
//!   used to come out whole (the walk around the perimeter pairs the
//!   wrong transitions): the faces let the flux through the copper. They
//!   are integrated now.
//! * A post or a pad crossed along a diameter is no surface along the
//!   mesh; with `ConformalOutlines::Slanted` its edges were left to the
//!   staircase.
//! * What stays: a post in the middle of a cell, touching no line of
//!   nodes, has nothing to carry its current but the line of nodes the
//!   model gives it, and reads like the staircase.

mod common;

use std::f64::consts::PI;

use csx::primitives::{BoxShape, Cylinder, Polygon};
use csx::{ContinuousStructure, Material, Primitive, Property};
use ems::compile::{CompileOptions, ConformalOutlines, compile};
use ems::engine::{ConformalReport, EdgeSource, Probe};
use ems::{Axis, Fdtd};

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

/// Spectral peaks in `[lo, hi]`.
fn peaks(dt: f64, v: &[f64], first: usize, lo: f64, hi: f64) -> Vec<(f64, f64)> {
    let s = |f: f64| spectrum(dt, v, first, f);
    let n = 600;
    let step = (hi - lo) / n as f64;
    let c: Vec<f64> = (0..=n).map(|q| s(lo + q as f64 * step)).collect();
    let top = c.iter().copied().fold(0.0, f64::max);
    let mut out = Vec::new();
    for q in 1..n {
        if c[q] > c[q - 1] && c[q] >= c[q + 1] && c[q] > 0.02 * top {
            let (mut a, mut b) = (lo + (q as f64 - 1.0) * step, lo + (q as f64 + 1.0) * step);
            let g = 0.5 * (5f64.sqrt() - 1.0);
            for _ in 0..50 {
                let (x1, x2) = (b - g * (b - a), a + g * (b - a));
                if s(x1) > s(x2) {
                    b = x2;
                } else {
                    a = x1;
                }
            }
            out.push((0.5 * (a + b), c[q] / top));
        }
    }
    out
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

#[derive(Clone, Copy)]
struct Stub {
    /// Cell size in x and y \[mm\].
    h: f64,
    conformal: bool,
    /// Pad, clearance and post radii \[mm\] (0: none; a negative
    /// clearance: a plane without a hole).
    pad: f64,
    anti: f64,
    post: f64,
    /// Offset of the centre from a node, in cells.
    off: [f64; 2],
    merge_below: f64,
    solids: ConformalOutlines,
    threads: usize,
}

impl Stub {
    fn new(h: f64, conformal: bool) -> Self {
        Self {
            h,
            conformal,
            pad: 0.3,
            anti: 0.5,
            post: 0.15,
            off: [0.5, 0.2],
            merge_below: 0.5,
            solids: ConformalOutlines::Slanted,
            threads: 3,
        }
    }
}

fn disc(c: [f64; 2], r: f64) -> Vec<(f64, f64)> {
    let n = 96;
    let t = 2.0 * PI / n as f64;
    let rr = r * (t / t.sin()).sqrt();
    (0..n)
        .map(|q| {
            let a = t * q as f64;
            (c[0] + rr * a.cos(), c[1] + rr * a.sin())
        })
        .collect()
}

/// The plane with a round hole, as two half frames.
fn frame(c: [f64; 2], a: f64, big: f64) -> Vec<Vec<(f64, f64)>> {
    let half = |sign: f64| {
        let n = 96;
        let mut v: Vec<(f64, f64)> = (0..=n)
            .map(|q| {
                let t = PI / 2.0 + PI * q as f64 / n as f64;
                (c[0] + sign * a * t.cos(), c[1] + a * t.sin())
            })
            .collect();
        v.extend([
            (c[0], c[1] - big),
            (c[0] - sign * big, c[1] - big),
            (c[0] - sign * big, c[1] + big),
            (c[0], c[1] + big),
        ]);
        v
    };
    vec![half(1.0), half(-1.0)]
}

/// A via stub in a closed box: a post from the floor through 1 mm of
/// `ε_r` = 4 to a pad in the round clearance of a plane, air above.
/// Returns the spectral peaks \[Hz\] of `E_z` beside the post, with their
/// heights relative to the highest.
fn stub(s: Stub) -> (Vec<(f64, f64)>, Option<ConformalReport>) {
    let (ext, sub) = (3.0, 1.0);
    let h = s.h;
    let n = (ext / h).round();
    let l = lines(-n * h, n * h, h);
    let mut lz = lines(0.0, sub, 0.125);
    lz.extend([1.125, 1.25, 1.4, 1.6, 1.9, 2.3, 2.8, 3.4]);
    let c = [s.off[0] * h, s.off[1] * h];
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for (a, v) in [l.clone(), l.clone(), lz.clone()].into_iter().enumerate() {
        m.grid.set_lines(a, v).unwrap();
    }
    let fr4 = m.add_property(Property::material("substrate", Material::new(4.0, 0.0)));
    m.add_primitive(fr4, BoxShape::new([-ext, -ext, 0.0], [ext, ext, sub]))
        .unwrap();
    let metal = m.add_property(Property::metal("copper"));
    if s.anti < 0.0 {
        m.add_primitive(
            metal,
            Primitive::new(BoxShape::new([-ext, -ext, sub], [ext, ext, sub])).with_priority(10),
        )
        .unwrap();
    }
    if s.anti > 0.0 {
        for p in frame(c, s.anti, 10.0) {
            m.add_primitive(
                metal,
                Primitive::new(Polygon::new(2, sub, p)).with_priority(10),
            )
            .unwrap();
        }
    }
    if s.pad > 0.0 {
        m.add_primitive(
            metal,
            Primitive::new(Polygon::new(2, sub, disc(c, s.pad))).with_priority(10),
        )
        .unwrap();
    }
    if s.post > 0.0 {
        m.add_primitive(
            metal,
            Primitive::new(Cylinder::new([c[0], c[1], 0.0], [c[0], c[1], sub], s.post))
                .with_priority(10),
        )
        .unwrap();
        let (x, y) = (l[nearest(&l, c[0])], l[nearest(&l, c[1])]);
        m.add_primitive(
            metal,
            Primitive::new(BoxShape::new([x, y, 0.0], [x, y, sub])).with_priority(10),
        )
        .unwrap();
    }
    let fdtd = Fdtd::new().threads(s.threads).max_frequency(30e9);
    let mut opts = CompileOptions::default();
    if s.conformal {
        opts = opts
            .with_conformal(true)
            .with_conformal_outlines(ConformalOutlines::Slanted)
            .with_conformal_solids(s.solids);
        opts.conformal_options.merge_below = s.merge_below;
    }
    let cm = compile(&m, &fdtd, &opts).unwrap();
    let (mut e, hnd) = cm.build().unwrap();
    let src = [nearest(&l, c[0] + 1.0), nearest(&l, c[1] + 0.3), 3];
    let prb = [nearest(&l, c[0] - 0.9), nearest(&l, c[1] + 0.5), 3];
    let f0 = 14e9;
    let bw = 10e9;
    e.add_operator(EdgeSource::new(common::sine_pulse(f0, bw)).edge(Axis::Z, src, 1.0));
    let p = e.add_operator(Probe::voltage(Axis::Z, prb));
    let dt = e.dt();
    let first = (common::sine_pulse_duration(bw) / dt).ceil() as usize;
    e.run(first as u64 + (60.0 / 10e9 / dt) as u64);
    let v = e.operator(&p).values().to_vec();
    (peaks(dt, &v, first, 3e9, 26e9), hnd.conformal)
}

fn main_peak(p: &[(f64, f64)]) -> f64 {
    p.iter()
        .copied()
        .fold((0.0, 0.0), |a, b| if b.1 > a.1 { b } else { a })
        .0
}

fn run(h: f64, conformal: bool, pad: f64, anti: f64, off: [f64; 2]) -> (f64, usize, usize) {
    let (pk, rep) = stub(Stub {
        pad,
        anti,
        off,
        ..Stub::new(h, conformal)
    });
    if let Some(r) = &rep {
        assert!(r.verified, "{r:?}");
    }
    (
        main_peak(&pk),
        rep.map_or(0, |r| r.edges),
        rep.map_or(0, |r| r.faces + r.interpolated + r.merged + r.kept),
    )
}

/// A post one cell across, shorting the floor to a plane: on a line of
/// nodes or between two of them the conformal cells are 1.2 and 2.3 %
/// below the post on 0.1 mm cells, the staircase (a wire of the lattice
/// radius, whatever the post) 5.3 and 5.9 %. Before, both read as the
/// staircase: on the line of nodes the surface crosses every edge along
/// its normal and was taken for a surface along the mesh; between the
/// lines the faces through the post kept their whole area.
#[test]
fn shorting_post_of_a_cell() {
    let fine = run(0.1, true, 0.0, -1.0, [0.5, 0.2]).0;
    assert!((fine / 22.28e9 - 1.0).abs() < 0.006, "{fine}");
    for off in [[0.0, 0.0], [0.5, 0.2]] {
        let (stair, ..) = run(0.3, false, 0.0, -1.0, off);
        let (conf, edges, faces) = run(0.3, true, 0.0, -1.0, off);
        let (es, ec) = (stair / fine - 1.0, conf / fine - 1.0);
        eprintln!(
            "shorting post at {off:?}: staircase {:.3} GHz ({:+.2} %), conformal {:.3} GHz \
             ({:+.2} %), {edges} cut edges, {faces} faces; 0.1 mm cells {:.3} GHz",
            stair * 1e-9,
            es * 100.0,
            conf * 1e-9,
            ec * 100.0,
            fine * 1e-9
        );
        assert!(edges >= 8 && faces >= 20, "{edges} {faces}");
        assert!(es < -0.05, "staircase {es}");
        assert!(ec.abs() < 0.03, "conformal {ec}");
    }
}

/// The stub with its pad in the clearance of the plane, the centre at
/// four places in its cell: the conformal cells are within 2.6 % of the
/// stub on 0.1 mm cells (−0.4, −2.4, −0.3, −2.6 %), where they read −8.8
/// and −4.7 % in the first two places before and the staircase reads
/// −12.5, +29, −5.2 and +33 % (a pad of four edges, or one that the
/// post's line of nodes does not reach).
#[test]
fn via_stub_with_pad_and_clearance_of_a_cell() {
    let fine = run(0.1, true, 0.3, 0.5, [0.5, 0.2]).0;
    assert!((fine / 15.46e9 - 1.0).abs() < 0.006, "{fine}");
    let mut worst_stair = 0.0f64;
    for off in [[0.0, 0.0], [0.5, 0.2], [0.5, 0.5], [0.25, 0.1]] {
        let (stair, ..) = run(0.3, false, 0.3, 0.5, off);
        let (conf, edges, faces) = run(0.3, true, 0.3, 0.5, off);
        let (es, ec) = (stair / fine - 1.0, conf / fine - 1.0);
        eprintln!(
            "via stub at {off:?}: staircase {:.3} GHz ({:+.2} %), conformal {:.3} GHz ({:+.2} %), \
             {edges} cut edges, {faces} faces; 0.1 mm cells {:.3} GHz",
            stair * 1e-9,
            es * 100.0,
            conf * 1e-9,
            ec * 100.0,
            fine * 1e-9
        );
        worst_stair = worst_stair.max(es.abs());
        assert!(es.abs() > 0.04, "staircase {es}");
        assert!(ec.abs() < 0.03, "conformal {ec}");
    }
    assert!(worst_stair > 0.2, "{worst_stair}");
}

/// The conformal cells of the stub do not depend on the thread count.
#[test]
fn stub_does_not_depend_on_the_thread_count() {
    let one = stub(Stub {
        threads: 1,
        ..Stub::new(0.3, true)
    });
    let three = stub(Stub::new(0.3, true));
    assert_eq!(one.0, three.0);
    assert_eq!(one.1, three.1);
}

/// Cell sizes from 0.3 to 0.05 mm (`S_CELLS`), the centre at
/// `S_OFFX`, `S_OFFY` of a cell.
#[test]
#[ignore = "slow; run with --release -- --ignored --nocapture"]
fn stub_table() {
    let env = |k: &str, d: f64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (pad, anti) = (env("S_PAD", 0.3), env("S_ANTI", 0.5));
    let cells: Vec<f64> = std::env::var("S_CELLS")
        .unwrap_or("0.3,0.2,0.15,0.1,0.075,0.05".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let off = [env("S_OFFX", 0.5), env("S_OFFY", 0.2)];
    for &h in &cells {
        for conformal in [false, true] {
            let (f, edges, faces) = run(h, conformal, pad, anti, off);
            eprintln!(
                "h = {h} mm, conformal {conformal}: {:.4} GHz ({edges} cut edges, {faces} faces)",
                f * 1e-9
            );
        }
    }
}
