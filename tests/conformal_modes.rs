//! Convergence of the conformal PEC scheme (design §5.13) for E tangential
//! to the wall: TE and TM modes of curved and oblique cavities, the TEM
//! impedance of a coaxial line with conformal conductors, and a circular
//! waveguide with conformal walls running into CPML ends.
//!
//! Cavities are a PEC block with a vacuum region carved out, centres off
//! the mesh nodes; the resonance is the windowed-spectrum peak of a probe
//! after a soft pulse. 2-D cross sections have 2 cells along z between PMC
//! walls (TE: `H_z` modes, the waveguide cutoffs) or PEC walls (TM: `E_z`
//! modes). The fast tests run 10 and 20 cells per radius; the `#[ignore]`d
//! `*_table` tests add 40 (run with `--release -- --ignored --nocapture`)
//! and print the tables of design §5.13.

mod common;

use std::f64::consts::PI;

use csx::primitives::{BoxShape, Cylinder, LinPoly, Sphere};
use csx::{ContinuousStructure, Material, Primitive, Property};
use ems::compile::{CompileOptions, compile};
use ems::constants::{C0, Z0};
use ems::engine::{ConformalReport, EdgeSource, Probe, StopReason};
use ems::port::{CircMode, CircWaveguidePort};
use ems::{Axis, Boundaries, Boundary, EndCriteria, Face, Fdtd, Simulation, Waveform};

/// Zeros of `J_m` / `J_m'` and of the spherical-cavity characteristic
/// functions (Pozar; Harrington, *Time-Harmonic Electromagnetic Fields*).
const J01: f64 = 2.404_825_557_695_773;
const J11: f64 = 3.831_705_970_207_512;
const J11P: f64 = 1.841_183_781_340_659;
const J21P: f64 = 3.054_236_928_227_14;
/// Lowest TM mode of a sphere: first zero of `d/dx[x·j1(x)]`.
const SPHERE_TM1: f64 = 2.743_707_269_992_269;
/// Lowest TE mode of a sphere: first zero of `j1`.
const SPHERE_TE1: f64 = 4.493_409_457_909_064;

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

/// A resonance to measure: the model (PEC block, cavity carved out), the
/// analytic frequency, the excitations and the z-face condition.
///
/// Each run drives a soft source on a set of weighted edges (a field
/// direction `d` at a point `P`) and probes the same combination, so by
/// reciprocity a mode `m` appears with amplitude `∝ (d·e_m(P))²`. A
/// degenerate mode is split by the mesh into eigenvectors of arbitrary
/// orientation; the runs are chosen so that `Σ_runs (d·e_m(P))²` is the
/// same for every orientation (orthogonal directions at the centre, points
/// rotated by 90° for `m = 1` or 45° for `m = 2`), and their signals are
/// summed: the split partners then have equal amplitudes and the spectral
/// peak is their mean frequency.
struct Cavity {
    model: ContinuousStructure,
    f: f64,
    runs: Vec<Run>,
    /// PMC z faces (TE modes of a 2-D cross section).
    pmc_z: bool,
}

/// Weighted edges of one run: `(component, node, weight)`.
type Run = Vec<(Axis, [usize; 3], f64)>;

/// PEC block over the mesh `l` with `cavity` carved out (drawing unit mm).
fn block(l: [Vec<f64>; 3], cavity: Primitive) -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    let lo: Vec<f64> = l.iter().map(|l| l[0] - 1.0).collect();
    let hi: Vec<f64> = l.iter().map(|l| l[l.len() - 1] + 1.0).collect();
    for (a, l) in l.into_iter().enumerate() {
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

/// Relative frequency error and the conformal report (staircase without
/// `conformal`).
fn resonance(c: &Cavity, conformal: bool) -> (f64, Option<ConformalReport>) {
    let mut fdtd = Fdtd::new().threads(4);
    if c.pmc_z {
        fdtd = fdtd
            .boundary(Face::ZMin, Boundary::Pmc)
            .boundary(Face::ZMax, Boundary::Pmc);
    }
    let cm = compile(
        &c.model,
        &fdtd,
        &CompileOptions::default().with_conformal(conformal),
    )
    .unwrap();
    let bw = 0.4 * c.f;
    let mut sum: Vec<f64> = Vec::new();
    let (mut dt, mut report) = (0.0, None);
    for run in &c.runs {
        let (mut e, h) = cm.build().unwrap();
        let mut src = EdgeSource::new(common::sine_pulse(c.f, bw));
        for &(a, p, w) in run {
            src = src.edge(a, p, w);
        }
        e.add_operator(src);
        let probes: Vec<_> = run
            .iter()
            .map(|&(a, p, w)| (e.add_operator(Probe::voltage(a, p)), w))
            .collect();
        e.run((60.0 / c.f / e.dt()) as u64);
        dt = e.dt();
        for (p, w) in &probes {
            let v = e.operator(p).values();
            sum.resize(v.len(), 0.0);
            for (s, x) in sum.iter_mut().zip(v) {
                *s += w * x;
            }
        }
        if let Some(r) = &h.conformal {
            assert!(r.verified, "{r:?}");
        }
        report = h.conformal;
    }
    let first = (common::sine_pulse_duration(bw) / dt).ceil() as usize;
    let fm = peak(dt, &sum, first, 0.85 * c.f, 1.15 * c.f);
    ((fm - c.f) / c.f, report)
}

/// The run of field direction `d` (unit, `(x, y, z)`) at the point `p`
/// \[mm\] on the mesh `l`: the edges of the nearest node along the
/// non-zero components of `d`.
fn dipole(l: [&[f64]; 3], p: [f64; 3], d: [f64; 3]) -> Run {
    let node = [0, 1, 2].map(|a| nearest(l[a], p[a]));
    Axis::ALL
        .into_iter()
        .zip(d)
        .filter(|&(_, w)| w != 0.0)
        .map(|(a, w)| (a, node, w))
        .collect()
}

/// 2-D cross section `shape` (a z extrusion over 2 cells of `h` mm) in a
/// square mesh of half-width `ext`, between PMC z faces (`te`: `H_z`
/// modes, in-plane `d`) or PEC ones (`E_z`); `runs`: `(point (x, y) [mm],
/// in-plane direction)` (the direction is ignored for TM).
fn cross_section(
    h: f64,
    ext: f64,
    shape: impl Fn(f64) -> Primitive,
    f: f64,
    te: bool,
    runs: &[([f64; 2], [f64; 2])],
) -> Cavity {
    let l = lines(-ext, ext, h);
    let lz = lines(0.0, 2.0 * h, h);
    let model = block([l.clone(), l.clone(), lz.clone()], shape(2.0 * h));
    let runs = runs
        .iter()
        .map(|&(p, d)| {
            let d = if te {
                [d[0], d[1], 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            // Mid-height node (k = 1) for TE, the bottom E_z edge for TM.
            let z = if te { lz[1] } else { lz[0] };
            dipole([&l, &l, &lz], [p[0], p[1], z], d)
        })
        .collect();
    Cavity {
        model,
        f,
        runs,
        pmc_z: te,
    }
}

const A: f64 = 10.0;

/// Mode of a circular cross section.
#[derive(Clone, Copy)]
enum Circ {
    Te11,
    Te21,
    Tm010,
    Tm110,
}

/// Circular cross section of radius `A` (centre off the nodes), `n` cells
/// per radius.
fn circle(n: usize, mode: Circ) -> Cavity {
    let h = A / n as f64;
    let c = [0.31 * h, 0.17 * h];
    let at = |r: f64, deg: f64| {
        let t = deg.to_radians();
        [c[0] + r * A * t.cos(), c[1] + r * A * t.sin()]
    };
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let (x, te, runs) = match mode {
        // E uniform at the centre: x and y.
        Circ::Te11 => (
            J11P,
            true,
            vec![(at(0.0, 0.0), [1.0, 0.0]), (at(0.0, 0.0), [0.0, 1.0])],
        ),
        // E_φ at φ = 0° and 45° (cos 2φ and sin 2φ patterns).
        Circ::Te21 => (
            J21P,
            true,
            vec![(at(0.5, 0.0), [0.0, 1.0]), (at(0.5, 45.0), [-s, s])],
        ),
        Circ::Tm010 => (J01, false, vec![(at(0.3, 20.0), [0.0; 2])]),
        // E_z at φ = 0° and 90° (cos φ and sin φ).
        Circ::Tm110 => (
            J11,
            false,
            vec![(at(0.5, 0.0), [0.0; 2]), (at(0.5, 90.0), [0.0; 2])],
        ),
    };
    let f = x * C0 / (2.0 * PI * A * 1e-3);
    cross_section(
        h,
        A + 2.0 * h,
        |zh| Primitive::new(Cylinder::new([c[0], c[1], -1.0], [c[0], c[1], zh + 1.0], A)),
        f,
        te,
        &runs,
    )
}

/// Rectangle 20 × 13 mm rotated by 17° about z, cells of `h` mm; mode
/// `(p, q)`: TE_pq cutoff (`te`) or TM_pq (non-degenerate).
fn rotated_rectangle(h: f64, p: f64, q: f64, te: bool) -> Cavity {
    let (la, lb): (f64, f64) = (20.0, 13.0);
    let th = 17f64.to_radians();
    let c0 = [0.3 * h, 0.2 * h];
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
    let f = 0.5 * C0 * ((p / la).powi(2) + (q / lb).powi(2)).sqrt() * 1e3;
    let (x, y) = corner(0.13 * la, 0.11 * lb);
    // A generic in-plane direction couples to TE10 and TE01 alike.
    let d = [0.6, 0.8];
    cross_section(
        h,
        0.5 * la.hypot(lb) + 2.0 * h,
        |zh| Primitive::new(LinPoly::new(2, -1.0, zh + 1.0, verts.clone())),
        f,
        te,
        &[([x, y], d)],
    )
}

/// Sphere of radius `A`, `n` cells per radius: the lowest TM mode
/// (electric-dipole-like: E uniform at the centre; runs x, y, z there) or
/// TE mode (magnetic-dipole-like: E azimuthal about the dipole axis; runs
/// `E_y` at `(r, 0, 0)`, `E_z` at `(0, r, 0)`, `E_x` at `(0, 0, r)`), each
/// triply degenerate.
fn sphere(n: usize, te: bool) -> Cavity {
    let h = A / n as f64;
    let c = [0.23 * h, 0.37 * h, 0.11 * h];
    let l = lines(-A - 2.0 * h, A + 2.0 * h, h);
    let model = block(
        [l.clone(), l.clone(), l.clone()],
        Primitive::new(Sphere::new(c, A)),
    );
    let at = |d: [f64; 3]| [0, 1, 2].map(|a| c[a] + d[a] * A);
    let ll = [&l[..], &l[..], &l[..]];
    let (x, runs) = if te {
        let r = 0.5;
        (
            SPHERE_TE1,
            vec![
                dipole(ll, at([r, 0.0, 0.0]), [0.0, 1.0, 0.0]),
                dipole(ll, at([0.0, r, 0.0]), [0.0, 0.0, 1.0]),
                dipole(ll, at([0.0, 0.0, r]), [1.0, 0.0, 0.0]),
            ],
        )
    } else {
        (
            SPHERE_TM1,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                .into_iter()
                .map(|d| dipole(ll, at([0.0; 3]), d))
                .collect(),
        )
    };
    Cavity {
        model,
        f: x * C0 / (2.0 * PI * A * 1e-3),
        runs,
        pmc_z: false,
    }
}

/// Errors (staircase, conformal) and the conformal report.
fn both(c: &Cavity) -> (f64, f64, ConformalReport) {
    let (es, _) = resonance(c, false);
    let (ec, r) = resonance(c, true);
    (es, ec, r.unwrap())
}

/// Prints a convergence row and returns the conformal errors.
fn table(name: &str, cells: &[usize], mk: impl Fn(usize) -> Cavity) -> Vec<f64> {
    let mut row_s = Vec::new();
    let mut row_c = Vec::new();
    let mut s = Vec::new();
    for &n in cells {
        let (es, ec, r) = both(&mk(n));
        row_s.push(format!("{:+.3}", 100.0 * es));
        row_c.push(format!("{:+.4}", 100.0 * ec));
        s.push(r.stability);
        assert!(
            ec.abs() < 0.2 * es.abs(),
            "{name} at {n}: staircase {es}, conformal {ec}"
        );
        eprintln!("  {name} n = {n}: {r:?}");
    }
    let ec: Vec<f64> = row_c.iter().map(|x| x.parse::<f64>().unwrap()).collect();
    let k = ec.len() - 1;
    let order = (ec[k - 1] / ec[k]).abs().log2();
    eprintln!(
        "| {name} | {} | {} % | {} % | {order:.2} | {:.3}–{:.3} |",
        cells
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(" / "),
        row_s.join(" / "),
        row_c.join(" / "),
        s.iter().cloned().fold(f64::MAX, f64::min),
        s.iter().cloned().fold(f64::MIN, f64::max)
    );
    ec.iter().map(|x| x / 100.0).collect()
}

/// Ratio of successive errors must show second order (×4 per halving; at
/// least ×3, the coarse step being the least asymptotic).
fn assert_second_order(name: &str, e: &[f64]) {
    for w in e.windows(2) {
        assert!(w[0].abs() > 3.0 * w[1].abs(), "{name}: {e:?}");
    }
}

#[test]
fn circle_te_modes_converge_second_order() {
    let e = table("circle TE11", &[10, 20], |n| circle(n, Circ::Te11));
    assert!(e[0].abs() < 1.2e-3 && e[1].abs() < 3e-4, "{e:?}");
    assert_second_order("TE11", &e);
    let e = table("circle TE21", &[10, 20], |n| circle(n, Circ::Te21));
    assert!(e[0].abs() < 2e-3 && e[1].abs() < 5e-4, "{e:?}");
    assert_second_order("TE21", &e);
}

#[test]
fn circle_tm_modes_converge_second_order() {
    let e = table("circle TM010", &[10, 20], |n| circle(n, Circ::Tm010));
    assert!(e[0].abs() < 1.2e-3 && e[1].abs() < 3e-4, "{e:?}");
    assert_second_order("TM010", &e);
    let e = table("circle TM110", &[10, 20], |n| circle(n, Circ::Tm110));
    assert!(e[0].abs() < 3e-3 && e[1].abs() < 8e-4, "{e:?}");
    assert_second_order("TM110", &e);
}

#[test]
fn rotated_rectangle_cutoffs() {
    // Planar walls: the chord construction is exact, the remaining error
    // is the interpolation of the small faces and the bulk Yee dispersion.
    for (name, p, q, te) in [
        ("rectangle TE10", 1.0, 0.0, true),
        ("rectangle TE01", 0.0, 1.0, true),
        ("rectangle TM11", 1.0, 1.0, false),
    ] {
        let e = table(name, &[1, 2], |k| {
            rotated_rectangle(1.0 / k as f64, p, q, te)
        });
        assert!(e[1].abs() < 5e-4, "{name}: {e:?}");
    }
}

#[test]
fn sphere_modes() {
    let e = table("sphere TM1", &[5, 10], |n| sphere(n, false));
    assert!(e[1].abs() < 1e-3, "{e:?}");
    let e = table("sphere TE1", &[10, 14], |n| sphere(n, true));
    assert!(e[1].abs() < 1.5e-3, "{e:?}");
}

#[test]
#[ignore = "slow; run with --release -- --ignored --nocapture"]
fn cavity_table() {
    eprintln!("| mode | cells | staircase | conformal | order (last step) | s |");
    eprintln!("|---|---|---|---|---|---|");
    let cells = [10, 20, 40];
    table("circle TE11", &cells, |n| circle(n, Circ::Te11));
    table("circle TE21", &cells, |n| circle(n, Circ::Te21));
    table("circle TM010", &cells, |n| circle(n, Circ::Tm010));
    table("circle TM110", &cells, |n| circle(n, Circ::Tm110));
    let h = [1, 2, 4];
    table("rectangle TE10 (cells/mm)", &h, |k| {
        rotated_rectangle(1.0 / k as f64, 1.0, 0.0, true)
    });
    table("rectangle TE01 (cells/mm)", &h, |k| {
        rotated_rectangle(1.0 / k as f64, 0.0, 1.0, true)
    });
    table("rectangle TM11 (cells/mm)", &h, |k| {
        rotated_rectangle(1.0 / k as f64, 1.0, 1.0, false)
    });
    table("sphere TM1", &cells, |n| sphere(n, false));
    table("sphere TE1", &cells, |n| sphere(n, true));
}

/// TEM impedance of a coaxial line (inner radius `a` = 2 mm, outer `b` =
/// 5 mm, axis off the nodes) with conformal (or staircased) cylinders, `n`
/// cells per inner radius, CPML at the z ends: `|V/I|` over 2–6 GHz, `V`
/// summed along a mesh line between the conductors, `I` around a dual
/// loop, against `η0/(2π)·ln(b/a)`. Returns the mean relative error.
fn coax_z(n: usize, conformal: bool) -> f64 {
    let (a, b) = (2.0, 5.0);
    let h = a / n as f64;
    let c = [0.31 * h, 0.17 * h];
    let ext = b + 2.0 * h;
    let l = lines(-ext, ext, h);
    let dz = 1.0;
    let lz = lines(0.0, 30.0, dz);
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, l.clone()).unwrap();
    m.grid.set_lines(1, l.clone()).unwrap();
    m.grid.set_lines(2, lz.clone()).unwrap();
    let metal = m.add_property(Property::metal("coax"));
    m.add_primitive(
        metal,
        BoxShape::new([-ext - 1.0, -ext - 1.0, -1.0], [ext + 1.0, ext + 1.0, 31.0]),
    )
    .unwrap();
    let air = m.add_property(Property::material("gap", Material::new(1.0, 0.0)));
    m.add_primitive(
        air,
        Primitive::new(Cylinder::new([c[0], c[1], -1.0], [c[0], c[1], 31.0], b)).with_priority(10),
    )
    .unwrap();
    let inner = m.add_property(Property::metal("inner"));
    m.add_primitive(
        inner,
        Primitive::new(Cylinder::new([c[0], c[1], -1.0], [c[0], c[1], 31.0], a)).with_priority(20),
    )
    .unwrap();
    let fdtd = Fdtd::new()
        .threads(4)
        .boundary(Face::ZMin, Boundary::pml(8))
        .boundary(Face::ZMax, Boundary::pml(8));
    let cm = compile(
        &m,
        &fdtd,
        &CompileOptions::default().with_conformal(conformal),
    )
    .unwrap();
    let cut: std::collections::HashMap<(Axis, [usize; 3]), f64> = cm
        .conformal
        .as_ref()
        .map(|cp| {
            cp.edges()
                .iter()
                .map(|e| ((e.axis, e.node), e.fraction))
                .collect()
        })
        .unwrap_or_default();
    let (mut e, _) = cm.build().unwrap();
    // Radial soft source E_ρ ∝ 1/ρ on the plane z = 10 mm (x and y edges
    // between the conductors; `u = E·l_open` on cut edges).
    let k_src = nearest(&lz, 10.0);
    let (f0, bw) = (4e9, 2.5e9);
    let mut src = EdgeSource::new(common::sine_pulse(f0, bw));
    for (ax, d) in [(Axis::X, 0), (Axis::Y, 1)] {
        for i in 0..l.len() - 1 {
            for j in 0..l.len() {
                let node = if d == 0 { [i, j, k_src] } else { [j, i, k_src] };
                let mut mid = [l[node[0]] - c[0], l[node[1]] - c[1]];
                mid[d] += 0.5 * h;
                let r = mid[0].hypot(mid[1]);
                if !(a..=b).contains(&r) && !cut.contains_key(&(ax, node)) {
                    continue;
                }
                let f = cut.get(&(ax, node)).copied().unwrap_or(1.0);
                if !(a * 0.9..=b * 1.1).contains(&r) {
                    continue;
                }
                src = src.edge(ax, node, f * h * 1e-3 * mid[d] / (r * r));
            }
        }
    }
    e.add_operator(src);
    // V: x edges along the mesh line through the axis (nearest y line),
    // from the inner conductor to the outer one (PEC edges read 0); I: the
    // dual loop through the dual lines nearest to ±3.5 mm, in the dual
    // plane k + ½.
    let k = nearest(&lz, 18.0);
    let jy = nearest(&l, c[1]);
    let ix0 = nearest(&l, c[0]);
    let vpath: Vec<[usize; 3]> = (ix0..l.len() - 1).map(|i| [i, jy, k]).collect();
    let dual = |x: f64| {
        (0..l.len() - 1).min_by(|&p, &q| {
            let dp = (0.5 * (l[p] + l[p + 1]) - x).abs();
            let dq = (0.5 * (l[q] + l[q + 1]) - x).abs();
            dp.total_cmp(&dq)
        })
    };
    let (i0, i1) = (dual(c[0] - 3.5).unwrap(), dual(c[0] + 3.5).unwrap());
    let (j0, j1) = (dual(c[1] - 3.5).unwrap(), dual(c[1] + 3.5).unwrap());
    let mut loop_terms: Vec<(Axis, [usize; 3], f64)> = Vec::new();
    for i in i0 + 1..=i1 {
        loop_terms.push((Axis::X, [i, j0, k], 1.0));
        loop_terms.push((Axis::X, [i, j1, k], -1.0));
    }
    for j in j0 + 1..=j1 {
        loop_terms.push((Axis::Y, [i1, j, k], 1.0));
        loop_terms.push((Axis::Y, [i0, j, k], -1.0));
    }
    let vp: Vec<_> = vpath
        .iter()
        .map(|&p| e.add_operator(Probe::voltage(Axis::X, p)))
        .collect();
    let ip: Vec<_> = loop_terms
        .iter()
        .map(|&(ax, p, _)| e.add_operator(Probe::current(ax, p)))
        .collect();
    let steps = (4e-9 / e.dt()) as u64;
    e.run(steps);
    let dt = e.dt();
    let sum = |hs: Vec<Vec<f64>>, w: &[f64]| -> Vec<f64> {
        let n = hs[0].len();
        (0..n)
            .map(|t| hs.iter().zip(w).map(|(h, w)| w * h[t]).sum())
            .collect()
    };
    let v = sum(
        vp.iter().map(|p| e.operator(p).values().to_vec()).collect(),
        &vec![1.0; vp.len()],
    );
    let w: Vec<f64> = loop_terms.iter().map(|t| t.2).collect();
    let i = sum(
        ip.iter().map(|p| e.operator(p).values().to_vec()).collect(),
        &w,
    );
    let dft = |x: &[f64], f: f64| -> (f64, f64) {
        x.iter().enumerate().fold((0.0, 0.0), |(re, im), (t, &v)| {
            let ph = -2.0 * PI * f * t as f64 * dt;
            (re + v * ph.cos(), im + v * ph.sin())
        })
    };
    let z0 = Z0 / (2.0 * PI) * (b / a).ln();
    let mut err = 0.0;
    let freqs = [2e9, 3e9, 4e9, 5e9, 6e9];
    for &f in &freqs {
        let (vr, vi) = dft(&v, f);
        let (ir, ii) = dft(&i, f);
        let z = vr.hypot(vi) / ir.hypot(ii);
        err += (z / z0 - 1.0) / freqs.len() as f64;
    }
    err
}

#[test]
fn coax_tem_impedance() {
    let mut e = Vec::new();
    for n in [4, 8] {
        let (es, ec) = (coax_z(n, false), coax_z(n, true));
        eprintln!(
            "coax Z, {n} cells per inner radius: staircase {:+.3} %, conformal {:+.4} %",
            100.0 * es,
            100.0 * ec
        );
        assert!(ec.abs() < 0.2 * es.abs(), "{es} {ec}");
        e.push(ec);
    }
    assert!(e[1].abs() < 2e-3, "{e:?}");
}

#[test]
#[ignore = "slow; run with --release -- --ignored --nocapture"]
fn coax_table() {
    for n in [5, 10, 20] {
        let (es, ec) = (coax_z(n, false), coax_z(n, true));
        eprintln!(
            "| coax Z_TEM | {n} | {:+.3} % | {:+.4} % |",
            100.0 * es,
            100.0 * ec
        );
    }
}

/// A circular guide (radius 10 mm, 1 mm cells, the wall's extreme points
/// on mesh nodes) with conformal walls running into CPML at both ends,
/// TE11 port excitation: the energy must decay like the staircase run's.
/// Before the fix the CPML read `ε_r ~ 1/f_e` from the cut edges and
/// switched itself off, and the open edges on the wall held a resonance:
/// the energy stayed at its peak (design §5.13, "Absorbing boundaries").
#[test]
fn circular_guide_with_pml_decays() {
    let (a, len, d): (f64, f64, f64) = (10.0, 50.0, 1.0);
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    let n = ((a + 1.5) / d).ceil() as i64;
    for ax in 0..2 {
        m.grid
            .set_lines(ax, (-n..=n).map(|i| i as f64 * d))
            .unwrap();
    }
    m.grid.set_lines(2, lines(0.0, len, 1.0)).unwrap();
    let e = n as f64 * d;
    let metal = m.add_property(Property::metal("wall"));
    m.add_primitive(
        metal,
        Primitive::new(BoxShape::new([-e, -e, 0.0], [e, e, len])),
    )
    .unwrap();
    let air = m.add_property(Property::material("air", Material::new(1.0, 0.0)));
    m.add_primitive(
        air,
        Primitive::new(Cylinder::new([0.0, 0.0, 0.0], [0.0, 0.0, len], a)).with_priority(10),
    )
    .unwrap();
    CircWaveguidePort::new(1, [0., 0., 10.0], [0., 0., 14.0], a, CircMode::Te(1, 1))
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(11.5e9, 3e9))
        .boundaries(
            Boundaries::all(Boundary::Pec)
                .with(Face::ZMin, Boundary::pml(8))
                .with(Face::ZMax, Boundary::pml(8)),
        )
        .end_criteria(EndCriteria::energy_decay_db(-50.0))
        .max_timesteps(12_000)
        .threads(4);
    let mut steps = [0u64; 2];
    for (q, conformal) in [false, true].into_iter().enumerate() {
        let opts = CompileOptions::default().with_conformal(conformal);
        let mut sim = Simulation::with_options(&m, &fdtd, &opts).unwrap();
        let res = sim.run().unwrap();
        let out = res.outcome().unwrap();
        eprintln!("conformal {conformal}: {out:?}");
        assert!(
            matches!(out.reason, StopReason::EnergyDecay { .. }),
            "conformal {conformal}: {out:?}"
        );
        steps[q] = out.steps;
    }
    // (Measured: 6401 staircase, 5801 conformal.)
    assert!(steps[1] < 2 * steps[0], "{steps:?}");
}
