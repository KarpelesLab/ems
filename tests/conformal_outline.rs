//! Outlines of sheets on mesh planes with conformal PEC (design §5.13.2):
//! the open lengths of the in-plane edges and the open areas of the
//! in-plane faces a slanted or curved outline crosses.
//!
//! * **Analytic.** A stack of identical PEC sheets, one on every mesh
//!   plane of a slab two cells thick between PMC faces, each with the
//!   same hole: for fields uniform across the slab (`E` in the planes,
//!   `H` normal to them) the stack is a cavity with the hole's cross
//!   section, whose lowest resonances are the TE cutoffs of that cross
//!   section. Circle (TE11) and rotated rectangle (TE10), against the
//!   formulas, with the staircase next to them. The in-plane faces carry
//!   the modes' `H`, so this measures the outline scheme alone.
//! * **A strip.** A half-wave microstrip resonator (PEC strip on a
//!   substrate in a closed box) lying along x and at 45° to the mesh: the
//!   resonance must not depend on the direction.
//! * **Conducting sheets.** The same resonator of a lossy sheet: frequency
//!   and Q against the strip along x; edges partly covered keep the
//!   sheet's loss.
//! * Axis-aligned outlines are discretised as without the option.

mod common;

use std::f64::consts::PI;

use csx::primitives::{BoxShape, Polygon};
use csx::{ContinuousStructure, Material, Primitive, Property};
use ems::compile::{CompileOptions, ConformalOutlines, compile};
use ems::constants::C0;
use ems::engine::{ConformalReport, EdgeSource, Probe};
use ems::{Axis, Boundary, Face, Fdtd};

/// First zero of `J1'` (TE11 of a circular guide).
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

fn options(outlines: Option<ConformalOutlines>) -> CompileOptions {
    match outlines {
        None => CompileOptions::default(),
        Some(o) => CompileOptions::default()
            .with_conformal(true)
            .with_conformal_outlines(o),
    }
}

// ---------------------------------------------------------------------
// Sheet stacks: TE cutoffs of the hole.

/// A cross section: the polygons (mm) whose union is everything but the
/// hole, the hole's lowest TE cutoff \[Hz\], the mesh half-width \[mm\]
/// and a point of the hole for the source.
struct Hole {
    plates: Vec<Vec<(f64, f64)>>,
    f: f64,
    ext: f64,
    at: [f64; 2],
}

/// Circle of radius 10 mm, centre off the nodes, as two half frames that
/// overlap along `x = c_x` (the circle is a 720-gon: 6e-6 in radius).
fn circle(h: f64) -> Hole {
    let a = 10.0;
    let c = [0.31 * h, 0.17 * h];
    let big = 100.0;
    let eps = 0.02f64;
    let half = |sign: f64| {
        let n = 360;
        let mut v: Vec<(f64, f64)> = (0..=n)
            .map(|q| {
                let t = PI / 2.0 - eps + (PI + 2.0 * eps) * q as f64 / n as f64;
                (c[0] + sign * a * t.cos(), c[1] + a * t.sin())
            })
            .collect();
        let x = c[0] - sign * a * (PI / 2.0 - eps).cos();
        v.extend([
            (x, c[1] - big),
            (c[0] - sign * big, c[1] - big),
            (c[0] - sign * big, c[1] + big),
            (x, c[1] + big),
        ]);
        v
    };
    Hole {
        plates: vec![half(1.0), half(-1.0)],
        f: J11P * C0 / (2.0 * PI * a * 1e-3),
        ext: a + 2.0 * h,
        at: c,
    }
}

/// Rectangle 20 × 13 mm rotated by 17°: four half planes around it.
fn rotated_rectangle(h: f64) -> Hole {
    let (la, lb): (f64, f64) = (20.0, 13.0);
    let th = 17f64.to_radians();
    let c0 = [0.3 * h, 0.2 * h];
    let corner = |u: f64, v: f64| {
        (
            c0[0] + u * th.cos() - v * th.sin(),
            c0[1] + u * th.sin() + v * th.cos(),
        )
    };
    let big = 100.0;
    let plates = vec![
        vec![
            corner(-big, -0.5 * lb),
            corner(big, -0.5 * lb),
            corner(big, -big),
            corner(-big, -big),
        ],
        vec![
            corner(-big, 0.5 * lb),
            corner(big, 0.5 * lb),
            corner(big, big),
            corner(-big, big),
        ],
        vec![
            corner(-0.5 * la, -big),
            corner(-0.5 * la, big),
            corner(-big, big),
            corner(-big, -big),
        ],
        vec![
            corner(0.5 * la, -big),
            corner(0.5 * la, big),
            corner(big, big),
            corner(big, -big),
        ],
    ];
    let (x, y) = corner(0.13 * la, 0.11 * lb);
    Hole {
        plates,
        f: 0.5 * C0 / (la * 1e-3),
        ext: 0.5 * la.hypot(lb) + 2.0 * h,
        at: [x, y],
    }
}

/// Relative error of the hole's cutoff with cells of `h` mm, and the
/// conformal report.
fn cutoff_error(
    hole: &Hole,
    h: f64,
    outlines: Option<ConformalOutlines>,
) -> (f64, Option<ConformalReport>) {
    let l = lines(-hole.ext, hole.ext, h);
    let lz = lines(0.0, 2.0 * h, h);
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for (a, v) in [l.clone(), l.clone(), lz.clone()].into_iter().enumerate() {
        m.grid.set_lines(a, v).unwrap();
    }
    let metal = m.add_property(Property::metal("stack"));
    for &z in &lz {
        for p in &hole.plates {
            m.add_primitive(metal, Polygon::new(2, z, p.clone()))
                .unwrap();
        }
    }
    let fdtd = Fdtd::new()
        .threads(3)
        .boundary(Face::ZMin, Boundary::Pmc)
        .boundary(Face::ZMax, Boundary::Pmc);
    let cm = compile(&m, &fdtd, &options(outlines)).unwrap();
    let node = [nearest(&l, hole.at[0]), nearest(&l, hole.at[1]), 1];
    let bw = 0.4 * hole.f;
    let mut sum: Vec<f64> = Vec::new();
    let (mut dt, mut report) = (0.0, None);
    // E along x and along y at the point, summed: both partners of a
    // degenerate pair with equal weight.
    for axis in [Axis::X, Axis::Y] {
        let (mut e, hnd) = cm.build().unwrap();
        e.add_operator(EdgeSource::new(common::sine_pulse(hole.f, bw)).edge(axis, node, 1.0));
        let p = e.add_operator(Probe::voltage(axis, node));
        e.run((60.0 / hole.f / e.dt()) as u64);
        dt = e.dt();
        let v = e.operator(&p).values();
        sum.resize(v.len(), 0.0);
        for (s, x) in sum.iter_mut().zip(v) {
            *s += x;
        }
        if let Some(r) = &hnd.conformal {
            assert!(r.verified, "{r:?}");
        }
        report = hnd.conformal;
    }
    let first = (common::sine_pulse_duration(bw) / dt).ceil() as usize;
    let fm = peak(dt, &sum, first, 0.85 * hole.f, 1.15 * hole.f);
    ((fm - hole.f) / hole.f, report)
}

/// Errors `(staircase, conformal)` in percent per cell size.
fn stack_table(name: &str, cells: &[f64], mk: impl Fn(f64) -> Hole) -> Vec<(f64, f64)> {
    cells
        .iter()
        .map(|&h| {
            let hole = mk(h);
            let (es, _) = cutoff_error(&hole, h, None);
            let (ec, rep) = cutoff_error(&hole, h, Some(ConformalOutlines::Slanted));
            let rep = rep.unwrap();
            eprintln!(
                "{name}, h = {h} mm: staircase {:+.4} %, conformal {:+.4} % ({} cut edges, {} \
                 faces, {} interpolated, {} merged, s = {:.4})",
                es * 100.0,
                ec * 100.0,
                rep.edges,
                rep.faces,
                rep.interpolated,
                rep.merged,
                rep.stability
            );
            (es * 100.0, ec * 100.0)
        })
        .collect()
}

#[test]
fn circular_hole_te11_converges_second_order() {
    let e = stack_table("circular hole TE11", &[1.0, 0.5], circle);
    // The solid-metal scheme on the same cross section reads +0.031 /
    // +0.0086 % (design §5.13): the stack is the same discrete problem.
    assert!(e[0].0.abs() > 5.0 && e[1].0.abs() > 2.5, "{e:?}");
    assert!(e[0].1.abs() < 0.06 && e[1].1.abs() < 0.015, "{e:?}");
    assert!(e[1].1.abs() < 0.4 * e[0].1.abs(), "{e:?}");
}

#[test]
fn rotated_rectangular_hole_te10() {
    let e = stack_table(
        "rotated rectangular hole TE10",
        &[1.0, 0.5],
        rotated_rectangle,
    );
    assert!(e[0].0.abs() > 6.0 && e[1].0.abs() > 3.0, "{e:?}");
    assert!(e[0].1.abs() < 0.25 && e[1].1.abs() < 0.06, "{e:?}");
}

#[test]
#[ignore = "slow; run with --release -- --ignored --nocapture"]
fn stack_tables() {
    stack_table("circular hole TE11", &[1.0, 0.5, 0.25], circle);
    stack_table(
        "rotated rectangular hole TE10",
        &[1.0, 0.5, 0.25],
        rotated_rectangle,
    );
}

// ---------------------------------------------------------------------
// Microstrip resonators.

/// What the strip is made of.
#[derive(Clone, Copy, PartialEq)]
enum Copper {
    Pec,
    /// Conducting sheet: conductivity \[S/m\], thickness \[m\].
    Sheet(f64, f64),
}

struct Strip {
    /// Resonance \[Hz\].
    f: f64,
    /// Quality factor from the decay of the probe signal (lossy runs).
    q: f64,
    report: Option<ConformalReport>,
    sheet_cuts: usize,
    /// The probe series.
    values: Vec<f64>,
}

/// One resonator run.
#[derive(Clone, Copy)]
struct Case {
    /// Angle of the strip to the x axis.
    deg: f64,
    /// Cell size in x and y \[mm\].
    h: f64,
    copper: Copper,
    /// `None`: without conformal PEC.
    outlines: Option<ConformalOutlines>,
    /// Length of the record after the pulse, in periods of 6.5 GHz.
    periods: f64,
    /// Cubic cells of `h` instead of the graded z mesh (the largest time
    /// step of the grid).
    cubic: bool,
    threads: usize,
}

impl Case {
    fn new(deg: f64, h: f64, copper: Copper, outlines: Option<ConformalOutlines>) -> Self {
        Self {
            deg,
            h,
            copper,
            outlines,
            periods: 40.0,
            cubic: false,
            threads: 3,
        }
    }
}

/// A strip 12 mm long and 1 mm wide on 0.5 mm of `ε_r` = 4 over the PEC
/// floor of a closed box, turned by `deg` about its centre, cells of `h`
/// mm in x and y. The lowest (half-wave) resonance is driven at one end
/// and read at the other.
fn strip(case: Case) -> Strip {
    let Case {
        deg,
        h,
        copper,
        outlines,
        periods,
        cubic,
        threads,
    } = case;
    let (len, w, sub) = (12.0, 1.0, 0.5);
    let ext = 10.0;
    // The strip's centre off the nodes; with `deg` = 0 its long sides
    // fall between mesh lines.
    let c = [0.37 * h, 0.21 * h];
    let l = lines(-ext, ext, h);
    let mut lz = lines(0.0, sub, sub / 2.0);
    if cubic {
        lz = lines(0.0, 16.0 * h, h);
    } else {
        lz.extend([0.8, 1.2, 1.7, 2.3, 3.0, 3.8]);
    }
    let t = deg.to_radians();
    let at = |u: f64, v: f64| {
        (
            c[0] + u * t.cos() - v * t.sin(),
            c[1] + u * t.sin() + v * t.cos(),
        )
    };
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for (a, v) in [l.clone(), l.clone(), lz.clone()].into_iter().enumerate() {
        m.grid.set_lines(a, v).unwrap();
    }
    let fr4 = m.add_property(Property::material("substrate", Material::new(4.0, 0.0)));
    m.add_primitive(fr4, BoxShape::new([-ext, -ext, 0.0], [ext, ext, sub]))
        .unwrap();
    let prop = match copper {
        Copper::Pec => Property::metal("strip"),
        Copper::Sheet(sigma, thickness) => Property::conducting_sheet("strip", sigma, thickness),
    };
    let id = m.add_property(prop);
    let verts = vec![
        at(-0.5 * len, -0.5 * w),
        at(0.5 * len, -0.5 * w),
        at(0.5 * len, 0.5 * w),
        at(-0.5 * len, 0.5 * w),
    ];
    m.add_primitive(
        id,
        Primitive::new(Polygon::new(2, sub, verts)).with_priority(10),
    )
    .unwrap();
    let fdtd = Fdtd::new().threads(threads).max_frequency(12e9);
    let cm = compile(&m, &fdtd, &options(outlines)).unwrap();
    let sheet_cuts = cm.sheet_cuts.len();
    let (mut e, hnd) = cm.build().unwrap();
    // E_z under the strip, 1 mm from each end.
    let end = |s: f64| {
        let (x, y) = at(s * (0.5 * len - 1.0), 0.0);
        [nearest(&l, x), nearest(&l, y), 0]
    };
    let f0 = 6.5e9;
    let bw = 0.3 * f0;
    e.add_operator(EdgeSource::new(common::sine_pulse(f0, bw)).edge(Axis::Z, end(-1.0), 1.0));
    let p = e.add_operator(Probe::voltage(Axis::Z, end(1.0)));
    let dt = e.dt();
    let first = (common::sine_pulse_duration(bw) / dt).ceil() as usize;
    e.run(first as u64 + (periods / f0 / dt) as u64);
    let v = e.operator(&p).values().to_vec();
    let f = peak(dt, &v, first, 0.8 * f0, 1.2 * f0);
    // Decay: the mode's amplitude over the two halves of the record
    // (demodulated at the resonance), a·e^{−πft/Q}.
    let amp = |lo: usize, hi: usize| {
        let (mut re, mut im) = (0.0, 0.0);
        for (q, &x) in v[lo..hi].iter().enumerate() {
            let ph = 2.0 * PI * f * (q + lo) as f64 * dt;
            let wnd = 0.5 - 0.5 * (2.0 * PI * q as f64 / (hi - lo - 1) as f64).cos();
            re += wnd * x * ph.cos();
            im += wnd * x * ph.sin();
        }
        re.hypot(im)
    };
    let mid = (first + v.len()) / 2;
    let (a1, a2) = (amp(first, mid), amp(mid, v.len()));
    let q = PI * f * ((mid - first) as f64 * dt) / (a1 / a2).ln();
    if let Some(r) = &hnd.conformal {
        assert!(r.verified, "{r:?}");
    }
    Strip {
        f,
        q,
        report: hnd.conformal,
        sheet_cuts,
        values: v,
    }
}

const SLANTED: Option<ConformalOutlines> = Some(ConformalOutlines::Slanted);
const ALL: Option<ConformalOutlines> = Some(ConformalOutlines::All);
/// 35 µm at 2e5 S/m: 2.6 skin depths at 6.7 GHz, `R_s` = 0.36 Ω.
const LOSSY: Copper = Copper::Sheet(2e5, 35e-6);

#[test]
fn pec_strip_resonance_does_not_depend_on_direction() {
    let h = 0.25;
    // The reference: the strip along x with its sides cut (6.800 GHz;
    // 6.776 and 6.806 GHz at 0.5 and 0.125 mm).
    let along = strip(Case::new(0.0, h, Copper::Pec, ALL));
    let stair = strip(Case::new(45.0, h, Copper::Pec, None));
    let conf = strip(Case::new(45.0, h, Copper::Pec, SLANTED));
    let rep = conf.report.unwrap();
    let (es, ec) = (stair.f / along.f - 1.0, conf.f / along.f - 1.0);
    eprintln!(
        "PEC strip, h = {h} mm: along x {:.4} GHz; at 45°: staircase {:.4} GHz ({:+.2} %), \
         conformal {:.4} GHz ({:+.2} %); {} cut edges, {} faces, {} interpolated, s = {:.4}",
        along.f * 1e-9,
        stair.f * 1e-9,
        es * 100.0,
        conf.f * 1e-9,
        ec * 100.0,
        rep.edges,
        rep.faces,
        rep.interpolated,
        rep.stability
    );
    assert!(rep.edges > 100, "{rep:?}");
    assert!(es < -0.04, "staircase {es}");
    assert!(ec.abs() < 0.02, "conformal {ec}");
}

#[test]
fn conducting_strip_at_45_degrees() {
    let lossy = |h: f64, o| Case {
        periods: 30.0,
        ..Case::new(45.0, h, LOSSY, o)
    };
    let stair = strip(lossy(0.25, None));
    let conf = strip(lossy(0.25, SLANTED));
    let fine = strip(lossy(0.125, SLANTED));
    let pec = strip(Case::new(45.0, 0.25, Copper::Pec, SLANTED));
    eprintln!(
        "sheet strip at 45°: staircase {:.4} GHz, Q {:.1}; conformal {:.4} GHz, Q {:.1} ({} cut \
         sheet edges); at 0.125 mm {:.4} GHz, Q {:.1}; PEC conformal {:.4} GHz",
        stair.f * 1e-9,
        stair.q,
        conf.f * 1e-9,
        conf.q,
        conf.sheet_cuts,
        fine.f * 1e-9,
        fine.q,
        pec.f * 1e-9
    );
    assert!(conf.sheet_cuts > 100);
    // The sheet's internal inductance lowers the PEC resonance a little.
    let df = conf.f / pec.f - 1.0;
    assert!((-0.025..0.0).contains(&df), "{df}");
    assert!((fine.f / conf.f - 1.0).abs() < 0.02);
    // The loss does not depend on the cells (the staircase reads 1.5
    // times this Q).
    assert!(
        (conf.q / fine.q - 1.0).abs() < 0.12,
        "{} {}",
        conf.q,
        fine.q
    );
    assert!(stair.q > 1.25 * conf.q, "{} {}", stair.q, conf.q);
}

#[test]
fn conducting_outlines_are_stable_on_cubic_cells() {
    // The largest time step (stability factor 0.987), copper and a
    // resistive film, long records: no growth.
    for (copper, deg) in [
        (Copper::Sheet(5.8e7, 35e-6), 45.0),
        (LOSSY, 17.0),
        (Copper::Sheet(1e4, 1e-6), 30.0),
    ] {
        let s = strip(Case {
            periods: 100.0,
            cubic: true,
            ..Case::new(deg, 0.25, copper, ALL)
        });
        let rep = s.report.unwrap();
        assert!(rep.stability > 0.98 && rep.verified, "{rep:?}");
        let n = s.values.len();
        let (head, tail) = (
            common::peak(&s.values[..n / 4]),
            common::peak(&s.values[3 * n / 4..]),
        );
        assert!(tail.is_finite() && tail < head, "{deg}°: {head} {tail}");
    }
}

#[test]
fn axis_aligned_outlines_keep_the_staircase() {
    // Long sides between mesh lines, ends too.
    for cu in [Copper::Pec, LOSSY] {
        let short = |o| Case {
            periods: 12.0,
            ..Case::new(0.0, 0.25, cu, o)
        };
        let off = strip(short(None));
        let on = strip(short(SLANTED));
        let rep = on.report.unwrap();
        assert_eq!((rep.edges, rep.faces, on.sheet_cuts), (0, 0, 0));
        assert_eq!(off.values, on.values);
        // Every outline: the strip's sides are cut.
        let all = strip(short(ALL));
        assert!(all.report.unwrap().edges > 50);
        assert_ne!(off.values, all.values);
    }
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    for cu in [Copper::Pec, LOSSY] {
        let run = |threads| {
            strip(Case {
                periods: 6.0,
                threads,
                ..Case::new(45.0, 0.25, cu, SLANTED)
            })
        };
        assert_eq!(run(1).values, run(3).values);
    }
}

#[test]
#[ignore = "slow; run with --release -- --ignored --nocapture"]
fn strip_table() {
    for cu in [Copper::Pec, LOSSY] {
        for h in [0.5, 0.25, 0.125] {
            let run = |deg: f64, o| {
                let s = strip(Case {
                    periods: 60.0,
                    ..Case::new(deg, h, cu, o)
                });
                format!("{:.4} GHz Q {:.1}", s.f * 1e-9, s.q)
            };
            eprintln!(
                "{} strip h = {h} mm: along x {} (all outlines {}); 45°: staircase {}, conformal \
                 {}; 17°: staircase {}, conformal {}",
                if cu == Copper::Pec { "PEC" } else { "sheet" },
                run(0.0, None),
                run(0.0, ALL),
                run(45.0, None),
                run(45.0, SLANTED),
                run(17.0, None),
                run(17.0, SLANTED)
            );
        }
    }
}
