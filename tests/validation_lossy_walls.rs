//! Loss of solid lossy metal on walls oblique to the mesh, against a
//! closed form: a box cavity carved out of a block of lossy metal and
//! turned about z.
//!
//! The box `a × b × h` (`h` along z) has the modes
//!
//! * **TM110** (`E_z = sin(πu/a)·sin(πv/b)`, uniform in z): the currents
//!   on the side walls run along z, those on floor and ceiling in the
//!   plane;
//! * **TE101** (`E_v = sin(πu/a)·sin(πz/h)`, uniform along `v`): the
//!   currents on the walls `u = 0, a` run horizontally along the wall
//!   (the current of a strip's rim), those on the walls `v = 0, b` both
//!   ways.
//!
//! With the surface resistance `R_s = 1/(σδ)` on every wall, `Q = ω·W/P`
//! with `P = (R_s/2)·∮|H_tan|²` (Pozar, *Microwave Engineering*, 4th ed.,
//! §6.3, eq. 6.46 for the TE101 mode) is, for a mode with `E` along the
//! side of length `c` and half a period along each of the others `p`, `q`,
//!
//! ```text
//! Q = (c/δ) · k²·p·q / (k²·p·q + 2π²·c·(p/q² + q/p²)),   k² = π²(1/p² + 1/q²)
//! ```
//!
//! (floor and ceiling alone: `c/δ`). The surface reactance equals the
//! resistance and lowers the resonance by `f/(2Q)`.

mod common;

use std::f64::consts::PI;

use csx::primitives::{BoxShape, LinPoly};
use csx::{ContinuousStructure, Primitive, Property};
use ems::compile::{CompileOptions, ConformalOutlines, compile};
use ems::constants::C0;
use ems::engine::sheet::skin_depth;
use ems::engine::surface::Conductor;
use ems::engine::{EdgeSource, Probe};
use ems::{Axis, Fdtd};

/// Sides of the box \[mm\].
const BOX: [f64; 3] = [20.0, 14.0, 10.0];
const SIGMA: f64 = 2e4;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    /// `E_z`, uniform in z.
    Tm110,
    /// `E` along the box's `b`, uniform along it.
    Te101,
}

/// Resonance \[Hz\] and Q of the perfect box's mode with lossy walls.
fn analytic(mode: Mode) -> (f64, f64) {
    let [a, b, h] = BOX.map(|v| v * 1e-3);
    let (c, p, q) = match mode {
        Mode::Tm110 => (h, a, b),
        Mode::Te101 => (b, a, h),
    };
    let k2 = PI * PI * (1.0 / (p * p) + 1.0 / (q * q));
    let f = C0 * k2.sqrt() / (2.0 * PI);
    let delta = skin_depth(SIGMA, f);
    let q0 =
        (c / delta) * k2 * p * q / (k2 * p * q + 2.0 * PI * PI * c * (p / (q * q) + q / (p * p)));
    (f, q0)
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

/// Resonance \[Hz\] and Q of the box turned by `deg` on cells of `cell`
/// mm, from the ring-down over `periods`.
fn cavity(mode: Mode, deg: f64, cell: f64, periods: f64, sigma: Option<f64>) -> (f64, f64) {
    let [a, b, h] = BOX;
    let ext = (0.5 * a.hypot(b) / cell).ceil() * cell + 4.0 * cell;
    let l = lines(-ext, ext, cell);
    let lz = lines(-2.0 * cell, h + 2.0 * cell, cell);
    let t = deg.to_radians();
    // At 0° the walls lie on mesh lines; turned, the centre is off the
    // nodes.
    let c0 = if deg == 0.0 {
        [0.0, 0.0]
    } else {
        [0.3 * cell, 0.2 * cell]
    };
    let at = |u: f64, v: f64| {
        (
            c0[0] + u * t.cos() - v * t.sin(),
            c0[1] + u * t.sin() + v * t.cos(),
        )
    };
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for (ax, v) in [l.clone(), l.clone(), lz.clone()].into_iter().enumerate() {
        m.grid.set_lines(ax, v).unwrap();
    }
    // Floor, ceiling and four walls, three cells thick: metal drawn as
    // metal (its faces along the mesh are its own edges).
    let metal = m.add_property(Property::metal("block"));
    let th = 3.0 * cell;
    let far = ext + 1.0;
    for (z0, z1) in [(-2.0 * cell - 1.0, 0.0), (h, h + 2.0 * cell + 1.0)] {
        m.add_primitive(metal, BoxShape::new([-far, -far, z0], [far, far, z1]))
            .unwrap();
    }
    let (ha, hb) = (0.5 * a, 0.5 * b);
    for [u0, u1, v0, v1] in [
        [ha, ha + th, -hb - th, hb + th],
        [-ha - th, -ha, -hb - th, hb + th],
        [-ha - th, ha + th, hb, hb + th],
        [-ha - th, ha + th, -hb - th, -hb],
    ] {
        let verts = vec![at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)];
        m.add_primitive(
            metal,
            Primitive::new(LinPoly::new(2, -cell, h + 2.0 * cell, verts)),
        )
        .unwrap();
    }
    let (f0, _) = analytic(mode);
    let fdtd = Fdtd::new().threads(3).max_frequency(1.3 * f0);
    let mut opts = CompileOptions::default()
        .with_conformal(true)
        .with_conformal_solids(ConformalOutlines::Slanted);
    if let Some(s) = sigma {
        opts = opts.with_lossy_metal("block", Conductor::new(s));
    }
    let cm = compile(&m, &fdtd, &opts).unwrap();
    let (mut e, hnd) = cm.build().unwrap();
    if let Some(r) = &hnd.conformal {
        assert!(r.verified, "{r:?}");
    }
    // Source and probe at two points of the box, weighted along the
    // mode's E.
    let dir = match mode {
        Mode::Tm110 => [0.0, 0.0, 1.0],
        Mode::Te101 => [-t.sin(), t.cos(), 0.0],
    };
    let edges = |u: f64, v: f64, z: f64| {
        let (x, y) = at(u, v);
        let node = [nearest(&l, x), nearest(&l, y), nearest(&lz, z)];
        Axis::ALL
            .into_iter()
            .zip(dir)
            .filter(|(_, w)| w.abs() > 1e-9)
            .map(move |(ax, w)| (ax, node, w))
            .collect::<Vec<_>>()
    };
    let bw = 0.12 * f0;
    let mut src = EdgeSource::new(common::sine_pulse(f0, bw));
    for (ax, node, w) in edges(-0.13 * a, 0.11 * b, 0.4 * h) {
        src = src.edge(ax, node, w);
    }
    e.add_operator(src);
    let probes: Vec<_> = edges(0.17 * a, -0.09 * b, 0.55 * h)
        .into_iter()
        .map(|(ax, node, w)| (e.add_operator(Probe::voltage(ax, node)), w))
        .collect();
    let dt = e.dt();
    let first = (common::sine_pulse_duration(bw) / dt).ceil() as usize;
    e.run(first as u64 + (periods / f0 / dt) as u64);
    let mut v: Vec<f64> = Vec::new();
    for (p, w) in &probes {
        let x = e.operator(p).values();
        v.resize(x.len(), 0.0);
        for (s, x) in v.iter_mut().zip(x) {
            *s += w * x;
        }
    }
    // The resonance: the peak of the windowed spectrum.
    let spec = |f: f64, lo: usize, hi: usize| {
        let (mut re, mut im) = (0.0, 0.0);
        for (q, &x) in v[lo..hi].iter().enumerate() {
            let ph = 2.0 * PI * f * (q + lo) as f64 * dt;
            let wnd = 0.5 - 0.5 * (2.0 * PI * q as f64 / (hi - lo - 1) as f64).cos();
            re += wnd * x * ph.cos();
            im += wnd * x * ph.sin();
        }
        re.hypot(im)
    };
    let (mut lo, mut hi) = (0.95 * f0, 1.05 * f0);
    for _ in 0..50 {
        let (p, q) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if spec(p, first, v.len()) < spec(q, first, v.len()) {
            lo = p;
        } else {
            hi = q;
        }
    }
    let f = 0.5 * (lo + hi);
    let mid = (first + v.len()) / 2;
    let (a1, a2) = (spec(f, first, mid), spec(f, mid, v.len()));
    let q = PI * f * ((mid - first) as f64 * dt) / (a1 / a2).ln();
    (f, q)
}

/// The walls' loss does not depend on their direction in the mesh. A
/// wall oblique to it was a staircase for its loss: each edge beside it
/// carried the half cells of the mesh lines, not the wall that crosses
/// its dual face, and Q read 14 % low at 30° and 8 % low at 17°; at 45°
/// the run grew (the edges beside the cut cells had half the capacitance
/// two of them on one small face need).
#[test]
fn oblique_walls_have_the_loss_of_walls_along_the_mesh() {
    for (mode, deg, tol) in [
        (Mode::Tm110, 0.0, 0.01),
        (Mode::Tm110, 45.0, 0.015),
        (Mode::Tm110, 30.0, 0.015),
        (Mode::Te101, 17.0, 0.015),
    ] {
        let (f0, q0) = analytic(mode);
        let (f, q) = cavity(mode, deg, 1.0, 40.0, Some(SIGMA));
        eprintln!(
            "{mode:?} at {deg}°: {:.4} GHz ({:+.3} %), Q {q:.2} of {q0:.2} ({:+.2} %)",
            f * 1e-9,
            (f / f0 - 1.0) * 100.0,
            (q / q0 - 1.0) * 100.0
        );
        assert!(
            q.is_finite() && (q / q0 - 1.0).abs() < tol,
            "{mode:?} {deg}: {q} {q0}"
        );
        assert!((f / f0 - 1.0).abs() < 0.01, "{mode:?} {deg}: {f} {f0}");
    }
}

/// The table of design §5.9.1: Q of both modes against the closed form,
/// by angle and cell.
#[test]
#[ignore = "slow; run with --release -- --ignored --nocapture"]
fn wall_table() {
    for mode in [Mode::Tm110, Mode::Te101] {
        let (f0, q0) = analytic(mode);
        eprintln!("{mode:?}: f {:.4} GHz, Q {q0:.2}", f0 * 1e-9);
        for cell in [1.0, 0.5] {
            for deg in [0.0, 45.0, 17.0, 30.0] {
                let (fp, _) = cavity(mode, deg, cell, 40.0, None);
                let (f, q) = cavity(mode, deg, cell, 40.0, Some(SIGMA));
                eprintln!(
                    "  {cell} mm, {deg:>4}°: perfect {:+.3} %, lossy {:+.3} % of it (−1/2Q: {:+.3} %), \
                     Q {q:.2} ({:+.2} %)",
                    (fp / f0 - 1.0) * 100.0,
                    (f / fp - 1.0) * 100.0,
                    -50.0 / q0,
                    (q / q0 - 1.0) * 100.0
                );
            }
        }
    }
}
