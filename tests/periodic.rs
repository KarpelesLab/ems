//! Periodic boundaries (design §5.6.1): plane waves in a box periodic in
//! x and y, a periodic cavity, energy conservation and determinism, a
//! dielectric slab at normal incidence, and the supercell check (a
//! periodic structure simulated with one or with two periods gives the
//! same fields).

mod common;

use std::f64::consts::PI;

use common::*;
use ems::constants::C0;
use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::{
    BlochPair, CellMaterials, EdgeSource, Engine, FaceCondition, FaceConditions, Grid, Material,
    OperatorHandle, Probe, TimeStep,
};
use ems::{Axis, Complex64, Cpml, Face};

/// Face conditions with the x and/or y pairs periodic and PEC elsewhere.
fn faces(x: bool, y: bool) -> FaceConditions {
    let mut f = FaceConditions::all(FaceCondition::Pec);
    for (a, on) in [(Axis::X, x), (Axis::Y, y)] {
        if on {
            f[Face::new(a, false)] = FaceCondition::Periodic;
            f[Face::new(a, true)] = FaceCondition::Periodic;
        }
    }
    f
}

#[test]
fn one_sided_periodic_is_rejected() {
    let g = Grid::uniform([4, 4, 4], [1e-3; 3]).unwrap();
    let f = FaceConditions::all(FaceCondition::Pec).with(Face::YMax, FaceCondition::Periodic);
    let err = Engine::builder(g).faces(f).build().unwrap_err();
    assert!(err.to_string().contains("ymin"), "{err}");
}

// ---------------------------------------------------------------------
// Plane waves in a periodic box.
// ---------------------------------------------------------------------

const DZ: f64 = 0.5e-3;
const IFACE: usize = 148;
const SRC: usize = 48;
const PRB: usize = 88;
const F0: f64 = 4.5e9;
const BW: f64 = 3.5e9;
/// Width of the reference `common::Line`.
const LINE_W: f64 = 5e-3;

/// A box periodic in x and y (unit cell `xs` × `ys`, repeated `periods`
/// times along x) and CPML-terminated along z, excited by a uniform soft
/// E_x sheet at z node `SRC`.
#[derive(Clone)]
struct PeriodicBox {
    xs: Vec<f64>,
    ys: Vec<f64>,
    periods: usize,
    zcells: usize,
    /// `(x cells of the unit cell, y cells, z cells, material)`.
    blocks: Vec<([std::ops::Range<usize>; 3], Material)>,
    dt: f64,
    threads: usize,
    /// z node of the probe plane.
    prb: usize,
}

impl PeriodicBox {
    fn x_lines(&self) -> Vec<f64> {
        let l = self.xs[self.xs.len() - 1] - self.xs[0];
        let mut v = vec![self.xs[0]];
        for p in 0..self.periods {
            v.extend(self.xs[1..].iter().map(|x| x + p as f64 * l));
        }
        v
    }

    /// Engine plus E_x probes at z node `prb` on every (i, j) of the
    /// first period.
    fn build(&self, pulse: impl Fn(f64) -> f64 + Send + 'static) -> (Engine, Vec<Probe>, Grid) {
        let xs = self.x_lines();
        let z: Vec<f64> = (0..=self.zcells).map(|q| q as f64 * DZ).collect();
        let g = Grid::new([xs, self.ys.clone(), z]).unwrap();
        let ucx = self.xs.len() - 1;
        let mut m = CellMaterials::vacuum(&g);
        for p in 0..self.periods {
            for ([x, y, z], mat) in &self.blocks {
                m.fill_box(
                    [x.start + p * ucx..x.end + p * ucx, y.clone(), z.clone()],
                    *mat,
                );
            }
        }
        let mut e = Engine::builder(g.clone())
            .materials(m)
            .faces(faces(true, true))
            .timestep(TimeStep::Fixed(self.dt))
            .threads(self.threads)
            .build()
            .unwrap();
        e.add_operator(
            CpmlBoundary::new(
                &e,
                [(Face::ZMin, Cpml::cells(8)), (Face::ZMax, Cpml::cells(24))],
            )
            .unwrap(),
        );
        let [nx, ny, _] = g.nodes();
        let mut s = EdgeSource::new(pulse);
        for i in 0..nx - 1 {
            for j in 0..ny {
                // Uniform E: voltage ∝ edge length, normalised to the line.
                s = s.edge(Axis::X, [i, j, SRC], g.delta(Axis::X, i) / LINE_W);
            }
        }
        e.add_operator(s);
        let probes = (0..ucx)
            .flat_map(|i| (0..ny).map(move |j| (i, j)))
            .map(|(i, j)| Probe::voltage(Axis::X, [i, j, self.prb]))
            .collect();
        (e, probes, g)
    }

    /// Runs `steps` steps; returns the E_x field series \[V/m\] of the
    /// probes of [`build`](Self::build), as `(i, j, series)`.
    fn run(&self, steps: u64) -> Vec<(usize, usize, Vec<f64>)> {
        let (mut e, probes, g) = self.build(sine_pulse(F0, BW));
        let ny = g.nodes()[1];
        let hs: Vec<OperatorHandle<Probe>> =
            probes.into_iter().map(|p| e.add_operator(p)).collect();
        e.run(steps);
        hs.iter()
            .enumerate()
            .map(|(q, h)| {
                let (i, j) = (q / ny, q % ny);
                let dx = g.delta(Axis::X, i);
                let v = e.operator(h).values().iter().map(|u| u / dx).collect();
                (i, j, v)
            })
            .collect()
    }
}

/// Graded unit cell: 4 x cells with different first and last widths, 3 y
/// cells.
fn graded_box() -> PeriodicBox {
    PeriodicBox {
        xs: vec![0.0, 0.6e-3, 1.5e-3, 2.1e-3, 3.0e-3],
        ys: vec![0.0, 1.0e-3, 1.8e-3, 3.0e-3],
        periods: 1,
        zcells: IFACE + 100 + 24,
        blocks: Vec::new(),
        dt: 0.9 * 0.5e-3 / C0 / 3f64.sqrt() * 0.95,
        threads: 2,
        prb: PRB,
    }
}

/// The reference 1-D line (`common::Line`) with the same `dz`, `dt` and
/// z profile, E_x probe at `PRB` \[V/m\].
fn line_reference(
    b: &PeriodicBox,
    layers: &[(std::ops::Range<usize>, Material)],
    steps: u64,
) -> Vec<f64> {
    let mut l = Line::new(DZ, b.zcells);
    l.pml[1] = Cpml::cells(24);
    l.src = SRC;
    l.probes = vec![PRB];
    l.layers = layers.to_vec();
    l.timestep = TimeStep::Fixed(b.dt);
    let (_, v) = l.run(sine_pulse(F0, BW), steps);
    v[0].iter().map(|u| u / LINE_W).collect()
}

/// A uniform E_x sheet source in a box periodic in x and y launches an
/// exact plane wave: the field is the same on every edge of the probe
/// plane (interior and periodic-face copies, on a graded mesh), and equal
/// to that of the 1-D reference line.
#[test]
fn uniform_plane_wave() {
    let _t = Timer::start("uniform_plane_wave");
    let b = graded_box();
    let steps = 900;
    let got = b.run(steps);
    let want = line_reference(&b, &[], steps);
    let pk = peak(&want);
    let (mut spread, mut err) = (0.0f64, 0.0f64);
    for (_, _, v) in &got {
        spread = spread.max(peak(&diff(v, &got[0].2)) / pk);
        err = err.max(peak(&diff(v, &want)) / pk);
    }
    eprintln!(
        "{} probes: max spread across the plane {spread:.2e}, max deviation from the 1-D line {err:.2e} (of peak)",
        got.len()
    );
    assert!(spread < 1e-5, "{spread:e}");
    assert!(err < 1e-4, "{err:e}");
}

/// A dielectric slab (ε_r = 4, 20 cells) filling the periodic cross
/// section reflects like the slab formula, and like the exact discrete
/// line to 1e-4.
#[test]
fn slab_reflection_normal_incidence() {
    let _t = Timer::start("slab_reflection_normal_incidence");
    let n_slab = 20;
    let mut b = graded_box();
    b.blocks.push((
        [0..4, 0..3, IFACE..IFACE + n_slab],
        Material::dielectric(4.0),
    ));
    let steps = 1600;
    let t = b.run(steps);
    let mut r = b.clone();
    r.blocks.clear();
    let inc = r.run(steps);
    let freqs = linspace(1e9, 8e9, 29);
    let a = spectrum(&inc[0].2, b.dt, &freqs);
    let prof: Vec<f64> = [(4, 1.0), (n_slab, 4.0), (4, 1.0)]
        .iter()
        .flat_map(|&(n, e)| std::iter::repeat_n(e, n))
        .collect();
    let d = n_slab as f64 * DZ;
    let (mut e_an, mut e_yee) = (0.0f64, 0.0f64);
    for (_, _, v) in [&t[0], &t[t.len() - 1]] {
        let refl = diff(v, &inc[0].2);
        let rb = spectrum(&refl, b.dt, &freqs);
        for ((&f, &rb), &a) in freqs.iter().zip(&rb).zip(&a) {
            let g: Complex64 = rb / a;
            e_an = e_an.max((g.norm() - slab_reflection(f, 4.0, d).norm()).abs());
            e_yee = e_yee.max((g.norm() - discrete_reflection(&prof, DZ, b.dt, f).norm()).abs());
        }
    }
    eprintln!(
        "slab in periodic box: max ||Γ| − |Γ_slab|| {e_an:.2e}, max ||Γ| − |Γ_yee|| {e_yee:.2e}"
    );
    assert!(e_an < 0.01, "{e_an}");
    assert!(e_yee < 1e-4, "{e_yee}");
}

/// Supercell check: a structure that is inhomogeneous within the unit
/// cell (a dielectric block covering part of x and y, on a graded mesh),
/// simulated with one period or with two periods along x, gives the same
/// fields at the same positions. The one-period box has the period
/// boundary where the two-period box has an ordinary interior plane, so
/// this checks the periodic merge against the regular update.
#[test]
fn supercell_matches_unit_cell() {
    let _t = Timer::start("supercell_matches_unit_cell");
    let mut b = graded_box();
    b.blocks
        .push(([1..3, 0..2, IFACE..IFACE + 16], Material::dielectric(6.0)));
    // Inside the block, where the field varies across the cell.
    b.prb = IFACE + 8;
    let steps = 1200;
    let one = b.run(steps);
    let mut two = b.clone();
    two.periods = 2;
    two.threads = 3;
    let both = two.run(steps);
    let pk = one.iter().map(|(_, _, v)| peak(v)).fold(0.0, f64::max);
    let mut dev = 0.0f64;
    for ((i, j, v), (i2, j2, w)) in one.iter().zip(&both) {
        assert_eq!((i, j), (i2, j2));
        dev = dev.max(peak(&diff(v, w)) / pk);
    }
    // Inhomogeneity actually matters: the probe plane is not uniform.
    let spread = one
        .iter()
        .map(|(_, _, v)| peak(&diff(v, &one[0].2)) / pk)
        .fold(0.0, f64::max);
    eprintln!(
        "one vs two periods: max deviation {dev:.2e} of peak (field spread across the cell {spread:.2e})"
    );
    assert!(spread > 1e-3, "{spread}");
    assert!(dev < 1e-5, "{dev:e}");
}

// ---------------------------------------------------------------------
// Cavity: periodic in x, PEC in y and z.
// ---------------------------------------------------------------------

/// Gaussian-modulated cosine pulse and its duration.
fn pulse(f0: f64, bw: f64) -> (impl Fn(f64) -> f64 + Send + 'static, f64) {
    let tau = 1.0 / (PI * bw);
    let t0 = 4.0 * tau;
    let f = move |t: f64| {
        let x = (t - t0) / tau;
        (-x * x).exp() * (2.0 * PI * f0 * (t - t0)).cos()
    };
    (f, 2.0 * t0)
}

/// |DFT| at `f` of the Hann-windowed samples with `t >= t_start`.
fn amplitude(times: &[f64], values: &[f64], t_start: f64, f: f64) -> f64 {
    let first = times.iter().position(|&t| t >= t_start).unwrap();
    let (t, v) = (&times[first..], &values[first..]);
    let n = t.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (q, (&tt, &x)) in t.iter().zip(v).enumerate() {
        let w = 0.5 - 0.5 * (2.0 * PI * q as f64 / (n - 1.0)).cos();
        let ph = -2.0 * PI * f * tt;
        re += w * x * ph.cos();
        im += w * x * ph.sin();
    }
    re.hypot(im)
}

/// Spectral peak in `[lo, hi]` (scan, then golden section).
fn peak_freq(times: &[f64], values: &[f64], t_start: f64, lo: f64, hi: f64) -> f64 {
    let s = |f: f64| amplitude(times, values, t_start, f);
    let n = 200;
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

/// Period, PEC height and depth of the periodic cavity \[m\].
const CAV: [f64; 3] = [0.06, 0.03, 0.05];

/// Resonances of the cavity periodic in x (period `a`) between PEC walls
/// in y and z, from an E_y source; probe E_y elsewhere. Returns the probe
/// series, dt and the source duration.
fn periodic_cavity(xlines: Vec<f64>, threads: usize, steps: u64) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let n = [xlines.len() - 1, 12, 20];
    let g = Grid::new([
        xlines,
        (0..=n[1])
            .map(|q| q as f64 * CAV[1] / n[1] as f64)
            .collect(),
        (0..=n[2])
            .map(|q| q as f64 * CAV[2] / n[2] as f64)
            .collect(),
    ])
    .unwrap();
    let mut e = Engine::builder(g)
        .faces(faces(true, false))
        .threads(threads)
        .build()
        .unwrap();
    let (f, t_end) = pulse(6.5e9, 4e9);
    let nx = n[0];
    e.add_operator(EdgeSource::new(f).edge(Axis::Y, [nx / 5, 5, 7], 1.0));
    let h = e.add_operator(Probe::voltage(Axis::Y, [3 * nx / 5, 6, 12]));
    e.run(steps);
    let p = e.operator(&h);
    (p.times().to_vec(), p.values().to_vec(), e.dt(), t_end)
}

/// Mode `(m, p)` of the periodic cavity: `E_y ∝ e^{±j2πm x/a}·sin(pπz/d)`.
fn modes() -> [(f64, f64); 3] {
    [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
}

/// On a uniform mesh the resonances match the exact discrete dispersion
/// relation of the Yee scheme with `k_x = 2πm/a`,
/// `sin²(ωdt/2)/(c·dt)² = Σ sin²(k_iΔ_i/2)/Δ_i²`, to 1e-4.
#[test]
fn periodic_cavity_uniform_matches_yee_dispersion() {
    let _t = Timer::start("periodic_cavity_uniform_matches_yee_dispersion");
    let nx = 24;
    let dx = CAV[0] / nx as f64;
    let xl = (0..=nx).map(|q| q as f64 * dx).collect();
    let (t, v, dt, t_end) = periodic_cavity(xl, 2, 6000);
    let dz = CAV[2] / 20.0;
    for (m, p) in modes() {
        let kx = 2.0 * PI * m / CAV[0];
        let kz = PI * p / CAV[2];
        let s =
            (kx * dx / 2.0).sin().powi(2) / (dx * dx) + (kz * dz / 2.0).sin().powi(2) / (dz * dz);
        let f_yee = (C0 * dt * s.sqrt()).asin() / (PI * dt);
        let f_an = C0 / (2.0 * PI) * (kx * kx + kz * kz).sqrt();
        let f = peak_freq(&t, &v, t_end, 0.97 * f_yee, 1.03 * f_yee);
        let err = (f - f_yee) / f_yee;
        eprintln!(
            "mode (m={m}, p={p}): fdtd {f:.6e}, Yee {f_yee:.6e} (err {err:.2e}), continuum {f_an:.6e}"
        );
        assert!(err.abs() < 1e-4, "{err:e}");
    }
}

/// On a graded periodic mesh (first and last cell of different size) the
/// resonances (40 cells per period) are within 0.5 % of the continuum `f = c/(2π)·√(k_x² + k_z²)`.
#[test]
fn periodic_cavity_graded() {
    let _t = Timer::start("periodic_cavity_graded");
    let nx = 40;
    let xl: Vec<f64> = (0..=nx)
        .map(|q| {
            let t = q as f64 / nx as f64;
            // Graded and asymmetric: Δ_0 ≠ Δ_{n−1}.
            CAV[0] * (t + 0.08 * (PI * t).sin() * (1.0 - t))
        })
        .collect();
    assert!((xl[1] - xl[0] - (xl[nx] - xl[nx - 1])).abs() > 1e-4);
    let (t, v, _, t_end) = periodic_cavity(xl, 3, 6000);
    for (m, p) in modes() {
        let kx = 2.0 * PI * m / CAV[0];
        let kz = PI * p / CAV[2];
        let f_an = C0 / (2.0 * PI) * (kx * kx + kz * kz).sqrt();
        let f = peak_freq(&t, &v, t_end, 0.97 * f_an, 1.03 * f_an);
        let err = (f - f_an) / f_an;
        eprintln!(
            "graded mode (m={m}, p={p}): fdtd {f:.6e}, continuum {f_an:.6e} (err {:.3} %)",
            100.0 * err
        );
        assert!(err.abs() < 5e-3, "{err:e}");
    }
}

// ---------------------------------------------------------------------
// Energy and determinism.
// ---------------------------------------------------------------------

/// Closed box periodic in x and y (PEC in z), graded, with a dielectric
/// block touching the periodic faces, sources inside and on a periodic
/// face plane.
fn closed_box(threads: usize) -> (Engine, f64) {
    let xl = vec![0.0, 1.0e-3, 2.5e-3, 3.2e-3, 4.0e-3, 5.5e-3, 6.0e-3];
    let yl = vec![0.0, 1.2e-3, 2.0e-3, 3.5e-3, 4.0e-3];
    let zl: Vec<f64> = (0..=8).map(|q| q as f64 * 0.8e-3).collect();
    let g = Grid::new([xl, yl, zl]).unwrap();
    let mut m = CellMaterials::vacuum(&g);
    m.fill_box([0..2, 2..4, 1..4], Material::dielectric(3.0));
    m.fill_box([5..6, 0..1, 3..7], Material::dielectric(2.0).with_mu_r(1.5));
    let mut e = Engine::builder(g)
        .materials(m)
        .faces(faces(true, true))
        .threads(threads)
        .build()
        .unwrap();
    let (f, t_end) = pulse(30e9, 20e9);
    e.add_operator(
        EdgeSource::new(f)
            .edge(Axis::Z, [2, 2, 3], 1.0)
            .edge(Axis::Y, [0, 1, 4], 0.7)
            .edge(Axis::X, [3, 4, 5], -0.4),
    );
    (e, t_end)
}

/// The leapfrog invariant is conserved in a closed periodic box after the
/// sources have stopped (`|W/W0 − 1|` < 1e-5 over 10 000 steps), and the
/// in-step energy equals the read-only one.
#[test]
fn periodic_energy_conserved() {
    let _t = Timer::start("periodic_energy_conserved");
    let (mut e, t_end) = closed_box(2);
    e.run((t_end / e.dt()).ceil() as u64 + 1);
    let w0 = e.energy();
    assert!(w0 > 0.0);
    let mut dev = 0.0f64;
    for _ in 0..1000 {
        let ro = e.energy();
        let w = e.step_with_energy();
        assert_eq!(w.to_bits(), ro.to_bits());
        dev = dev.max((w / w0 - 1.0).abs());
        e.run(9);
    }
    eprintln!("periodic closed box: W0 = {w0:.4e} J, max |W/W0 − 1| = {dev:.2e} over 10000 steps");
    assert!(dev < 1e-5, "{dev:e}");
}

/// Bit-identical fields for 1, 2, 4 and 6 threads (the x period crosses
/// slab boundaries), and the copies of every periodic-face edge are equal.
#[test]
fn periodic_deterministic_across_threads() {
    let run = |threads: usize| {
        let (mut e, _) = closed_box(threads);
        e.run(700);
        Axis::ALL
            .iter()
            .flat_map(|&a| [e.u_component(a), e.i_component(a)])
            .collect::<Vec<_>>()
    };
    let r1 = run(1);
    for t in [2, 4, 6] {
        assert!(r1 == run(t), "threads = {t} differs");
    }
    // Copies: E_y and E_z at x = 0 and x = x_max, E_x and E_z at y = 0
    // and y = y_max.
    let (e, _) = closed_box(1);
    let [nx, ny, nz] = e.grid().nodes();
    let idx = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    for (a, c) in [(1, 0), (2, 0), (0, 1), (2, 1)] {
        let u = &r1[2 * a];
        for p in 0..nx.max(ny) {
            for k in 0..nz {
                let (lo, hi) = if c == 0 {
                    if p >= ny {
                        continue;
                    }
                    (idx(0, p, k), idx(nx - 1, p, k))
                } else {
                    if p >= nx {
                        continue;
                    }
                    (idx(p, 0, k), idx(p, ny - 1, k))
                };
                assert_eq!(
                    u[lo].to_bits(),
                    u[hi].to_bits(),
                    "component {a}, copy pair {p},{k}"
                );
            }
        }
    }
    let n_nonzero = r1[2].iter().filter(|&&v| v != 0.0).count();
    assert!(n_nonzero > 0);
}

// ---------------------------------------------------------------------
// Through Simulation (csx model + Fdtd settings).
// ---------------------------------------------------------------------

/// `Boundary::Periodic` from the solver settings: a csx E_x sheet
/// excitation spanning the periodic cross section gives the same voltage
/// along x on the periodic face (y = 0) and inside (y = 2 mm). The
/// settings validate opposite pairs and refuse to export to openEMS XML.
#[test]
fn simulation_with_periodic_boundaries() {
    use csx::primitives::BoxShape;
    use csx::properties::{Excitation, ProbeBox, ProbeType};
    use csx::{ContinuousStructure, Primitive, Property};
    use ems::{Boundaries, Boundary, Fdtd, Simulation, Waveform};

    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, [0.0, 0.7, 2.0, 3.1, 4.0]).unwrap();
    m.grid.set_lines(1, [0.0, 1.0, 2.0, 3.0]).unwrap();
    m.grid.set_lines(2, (0..=80).map(f64::from)).unwrap();
    let ex = m.add_property(Property::excitation(
        "src",
        Excitation::new([1.0, 0.0, 0.0]),
    ));
    m.add_primitive(
        ex,
        Primitive::new(BoxShape::new([0.0, 0.0, 20.0], [4.0, 3.0, 20.0])),
    )
    .unwrap();
    for (name, y) in [("v_face", 0.0), ("v_mid", 2.0)] {
        let p = m.add_property(Property::probe(name, ProbeBox::new(ProbeType::Voltage)));
        m.add_primitive(
            p,
            Primitive::new(BoxShape::new([0.0, y, 50.0], [4.0, y, 50.0])),
        )
        .unwrap();
    }
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(10e9, 8e9))
        .boundaries(Boundaries::all(Boundary::Periodic))
        .boundary(Face::ZMin, Boundary::pml(8))
        .boundary(Face::ZMax, Boundary::pml(8))
        .max_timesteps(600);
    assert!(fdtd.get_boundaries().is_periodic(Axis::X));
    assert!(!fdtd.get_boundaries().is_periodic(Axis::Z));
    let err = fdtd.to_csx_checked().unwrap_err();
    assert!(err.to_string().contains("periodic"), "{err}");

    let mut sim = Simulation::new(&m, &fdtd).unwrap();
    for w in sim.warnings() {
        eprintln!("warning: {w}");
        assert!(!w.contains("eriodic"), "{w}");
    }
    let res = sim.run().unwrap();
    let a = res.probe("v_face").unwrap().scalar().unwrap().to_vec();
    let b = res.probe("v_mid").unwrap().scalar().unwrap().to_vec();
    let pk = peak(&a);
    let dev = peak(&diff(&a, &b)) / pk;
    eprintln!("Simulation: peak {pk:.3e} V, face vs interior {dev:.2e}");
    assert!(pk > 0.0);
    assert!(dev < 1e-5, "{dev:e}");

    // One-sided periodic settings are rejected.
    let bad = Fdtd::new()
        .waveform(Waveform::gaussian(10e9, 8e9))
        .boundary(Face::XMin, Boundary::Periodic);
    assert!(bad.validate().is_err());
    assert!(Simulation::new(&m, &bad).is_err());
}

// ---------------------------------------------------------------------
// Bloch (sine–cosine pair).
// ---------------------------------------------------------------------

/// The periodic cavity of [`periodic_cavity`] (uniform, 24 cells per
/// period) as a Bloch pair with phase `psi` per period along x.
fn bloch_cavity(psi: f64, steps: u64) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let n = [24, 12, 20];
    let g = Grid::uniform(n, [CAV[0] / 24.0, CAV[1] / 12.0, CAV[2] / 20.0]).unwrap();
    let b = Engine::builder(g).faces(faces(true, false)).threads(2);
    let mut pair = BlochPair::new(b, [psi, 0.0, 0.0]).unwrap();
    let (f, t_end) = pulse(6.5e9, 4e9);
    pair.real_mut()
        .add_operator(EdgeSource::new(f).edge(Axis::Y, [5, 5, 7], 1.0));
    let h = pair
        .real_mut()
        .add_operator(Probe::voltage(Axis::Y, [14, 6, 12]));
    pair.run(steps);
    let p = pair.real().operator(&h);
    (
        p.times().to_vec(),
        p.values().to_vec(),
        pair.real().dt(),
        t_end,
    )
}

/// Bloch phase ψ = π/2 per period: the cavity resonates at the Bloch
/// wavenumbers `k_x = (ψ + 2πm)/a`, m = 0 and −1, matching the exact Yee
/// dispersion relation to 1e-4; the zero-phase modes (k_x = 0, 2π/a) are
/// absent.
#[test]
fn bloch_cavity_dispersion() {
    let _t = Timer::start("bloch_cavity_dispersion");
    let psi = PI / 2.0;
    let (t, v, dt, t_end) = bloch_cavity(psi, 6000);
    let (dx, dz) = (CAV[0] / 24.0, CAV[2] / 20.0);
    let f_yee = |kx: f64| {
        let kz = PI / CAV[2];
        let s =
            (kx * dx / 2.0).sin().powi(2) / (dx * dx) + (kz * dz / 2.0).sin().powi(2) / (dz * dz);
        (C0 * dt * s.sqrt()).asin() / (PI * dt)
    };
    for m in [0.0, -1.0] {
        let kx = (psi + 2.0 * PI * m) / CAV[0];
        let fy = f_yee(kx);
        let f = peak_freq(&t, &v, t_end, 0.98 * fy, 1.02 * fy);
        let err = (f - fy) / fy;
        eprintln!("Bloch ψ = π/2, k_x = {kx:.2} rad/m: fdtd {f:.6e}, Yee {fy:.6e} (err {err:.2e})");
        assert!(err.abs() < 1e-4, "{err:e}");
    }
    // The k_x = 0 mode of the zero-phase cavity is not excited.
    let f0 = f_yee(0.0);
    let on = amplitude(&t, &v, t_end, f_yee(psi / CAV[0]));
    let off = amplitude(&t, &v, t_end, f0);
    eprintln!("|X| at the Bloch mode {on:.3e}, at the k_x = 0 mode {off:.3e}");
    assert!(off < 0.02 * on, "{off} vs {on}");
}

/// With ψ = 0 and nothing in the imaginary engine, the pair reproduces a
/// plain periodic engine (an H source, so that no `post_e` hook acts on
/// the fields).
#[test]
fn bloch_zero_phase_is_periodic() {
    let g = Grid::uniform([10, 6, 8], [1e-3, 1.5e-3, 1e-3]).unwrap();
    let mk = || {
        Engine::builder(g.clone())
            .faces(faces(true, true))
            .threads(2)
    };
    let src = || EdgeSource::new(pulse(40e9, 30e9).0).kind(ems::engine::SourceKind::SoftH);
    let mut e = mk().build().unwrap();
    e.add_operator(src().edge(Axis::Z, [3, 2, 4], 1e-3));
    let mut pair = BlochPair::new(mk(), [0.0; 3]).unwrap();
    pair.real_mut()
        .add_operator(src().edge(Axis::Z, [3, 2, 4], 1e-3));
    e.run(400);
    pair.run(400);
    for a in Axis::ALL {
        assert!(e.u_component(a) == pair.real().u_component(a), "u {a}");
        assert!(e.i_component(a) == pair.real().i_component(a), "i {a}");
        assert!(pair.imag().u_component(a).iter().all(|&x| x == 0.0));
    }
}

/// The complex energy (sum of both engines' invariants) is conserved in a
/// closed box with Bloch phases on x and y, after the source has stopped.
#[test]
fn bloch_energy_conserved() {
    let _t = Timer::start("bloch_energy_conserved");
    let xl = vec![0.0, 1.0e-3, 2.5e-3, 3.2e-3, 4.0e-3, 5.5e-3, 6.0e-3];
    let yl = vec![0.0, 1.2e-3, 2.0e-3, 3.5e-3, 4.0e-3];
    let zl: Vec<f64> = (0..=8).map(|q| q as f64 * 0.8e-3).collect();
    let g = Grid::new([xl, yl, zl]).unwrap();
    let mut m = CellMaterials::vacuum(&g);
    m.fill_box([0..2, 2..4, 1..4], Material::dielectric(3.0));
    let b = Engine::builder(g)
        .materials(m)
        .faces(faces(true, true))
        .threads(2);
    let mut pair = BlochPair::new(b, [1.0, -0.4, 0.0]).unwrap();
    let (f, t_end) = pulse(30e9, 20e9);
    pair.real_mut()
        .add_operator(EdgeSource::new(f).edge(Axis::Z, [2, 2, 3], 1.0).edge(
            Axis::Y,
            [3, 1, 4],
            0.7,
        ));
    pair.run((t_end / pair.real().dt()).ceil() as u64 + 1);
    let w0 = pair.energy();
    let (wr, wi) = (pair.real().energy(), pair.imag().energy());
    assert!(wi > 0.01 * wr, "imaginary part not excited: {wi} vs {wr}");
    let mut dev = 0.0f64;
    for _ in 0..500 {
        pair.run(10);
        dev = dev.max((pair.energy() / w0 - 1.0).abs());
    }
    eprintln!(
        "Bloch closed box: W0 = {w0:.4e} J (im share {:.2}), max |W/W0 − 1| = {dev:.2e} over 5000 steps",
        wi / w0
    );
    assert!(dev < 1e-5, "{dev:e}");
}

#[test]
fn bloch_pair_validation() {
    let g = Grid::uniform([4, 4, 4], [1e-3; 3]).unwrap();
    let none = Engine::builder(g.clone());
    assert!(BlochPair::new(none, [0.0; 3]).is_err());
    let b = || Engine::builder(g.clone()).faces(faces(true, false));
    assert!(BlochPair::new(b(), [0.0, 0.5, 0.0]).is_err());
    assert!(BlochPair::new(b(), [f64::NAN, 0.0, 0.0]).is_err());
    let p = BlochPair::new(b(), [0.3, 0.0, 0.0]).unwrap();
    assert_eq!(p.phase(), [0.3, 0.0, 0.0]);
    let psi = BlochPair::phase_for(&g, [100.0, 0.0, 0.0]);
    assert!((psi[0] - 0.4).abs() < 1e-12);
}
