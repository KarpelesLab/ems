//! Azimuthal multigrid near the cylindrical axis (design §5.15.1): the
//! time step grows, the circular cavity still resonates at the
//! Bessel-zero frequencies, the energy invariant holds over long runs and
//! the results do not depend on the thread count.

use std::f64::consts::{PI, TAU};

use ems::Axis;
use ems::constants::C0;
use ems::engine::{EdgeSource, Engine, Grid, Probe, TimeStep, courant_limit};

fn pulse(f0: f64, bw: f64) -> (impl Fn(f64) -> f64 + Send + 'static, f64) {
    let tau = 1.0 / (PI * bw);
    let t0 = 4.0 * tau;
    let f = move |t: f64| {
        let x = (t - t0) / tau;
        (-x * x).exp() * (2.0 * PI * f0 * (t - t0)).sin()
    };
    (f, 2.0 * t0)
}

/// |DFT| at `f` of the Hann-windowed samples with `t >= t_start`.
fn spectrum(times: &[f64], values: &[f64], t_start: f64, f: f64) -> f64 {
    let first = times.iter().position(|&t| t >= t_start).unwrap();
    let (t, v) = (&times[first..], &values[first..]);
    let n = t.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (q, (&tt, &x)) in t.iter().zip(v).enumerate() {
        let w = 0.5 - 0.5 * (TAU * q as f64 / (n - 1.0)).cos();
        let ph = -TAU * f * tt;
        re += w * x * ph.cos();
        im += w * x * ph.sin();
    }
    re.hypot(im)
}

/// Frequency of the spectral peak within `[lo, hi]`.
fn peak(times: &[f64], values: &[f64], t_start: f64, lo: f64, hi: f64) -> f64 {
    let s = |f: f64| spectrum(times, values, t_start, f);
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

fn lines(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..=n)
        .map(|i| lo + (hi - lo) * i as f64 / n as f64)
        .collect()
}

/// Circular cavity of radius `A` and height `H` with an axis.
const A: f64 = 10e-3;
const H: f64 = 15e-3;

/// 20 × 24 × 15 cells (`Δρ` = 0.5 mm, as in `tests/cylindrical.rs`), with
/// every 2nd α line kept inside ρ = 1.5 mm (line 3) and every 4th inside
/// 0.5 mm (line 1): the innermost arcs are 4 times longer.
fn cavity_grid(multigrid: bool) -> Grid {
    let g = Grid::cylindrical([lines(0.0, A, 20), lines(0.0, TAU, 24), lines(0.0, H, 15)]).unwrap();
    if multigrid {
        g.with_multigrid(&[3, 1]).unwrap()
    } else {
        g
    }
}

struct Run {
    engine: Engine,
    probes: Vec<ems::engine::OperatorHandle<Probe>>,
    t_off: f64,
}

/// The cavity driven by a short pulse on edges of every component, all on
/// kept lines, some inside the coarse zones and on the zone boundaries.
fn cavity(threads: usize) -> Run {
    let mut engine = Engine::builder(cavity_grid(true))
        .threads(threads)
        .build()
        .unwrap();
    let (p, t_off) = pulse(15e9, 12e9);
    let src = EdgeSource::new(p)
        .edge(Axis::Z, [9, 3, 4], 1.0)
        .edge(Axis::Z, [6, 2, 7], 1.0)
        .edge(Axis::X, [6, 6, 9], 1.0)
        .edge(Axis::X, [13, 7, 5], 1.0)
        .edge(Axis::Y, [13, 17, 5], 1.0)
        .edge(Axis::Y, [3, 5, 8], 1.0)
        .edge(Axis::Z, [8, 5, 3], 1.0)
        .edge(Axis::Z, [2, 4, 5], 1.0)
        .edge(Axis::X, [1, 2, 6], 1.0)
        .edge(Axis::X, [0, 8, 10], 1.0)
        .edge(Axis::Z, [3, 7, 9], 1.0)
        .edge(Axis::Z, [1, 2, 7], 1.0)
        .edge(Axis::Y, [1, 13, 3], 1.0)
        .edge(Axis::Z, [0, 0, 6], 1.0);
    engine.add_operator(src);
    let mut probes = Vec::new();
    for (axis, node) in [
        (Axis::Z, [7, 6, 3]),
        (Axis::Z, [0, 0, 8]),
        (Axis::Z, [12, 5, 9]),
        (Axis::X, [11, 2, 11]),
        (Axis::X, [5, 10, 4]),
        (Axis::Y, [4, 9, 6]),
        (Axis::Y, [16, 20, 10]),
        (Axis::Z, [2, 10, 4]),
        (Axis::X, [2, 8, 3]),
    ] {
        probes.push(engine.add_operator(Probe::voltage(axis, node)));
    }
    Run {
        engine,
        probes,
        t_off,
    }
}

const J01: f64 = 2.404_825_557_695_773;
const J11: f64 = 3.831_705_970_207_512;
const JP11: f64 = 1.841_183_781_340_659;
const JP21: f64 = 3.054_236_928_227_14;
const JP01: f64 = 3.831_705_970_207_512;

fn mode(x: f64, p: u32) -> f64 {
    C0 / TAU * ((x / A).powi(2) + (p as f64 * PI / H).powi(2)).sqrt()
}

#[test]
fn timestep_grows() {
    let fine = courant_limit(&cavity_grid(false));
    let mg = courant_limit(&cavity_grid(true));
    // The remaining limit is of the order of the Cartesian CFL of the
    // cells next to the zone boundary, 1/(c·√(1/Δρ² + 1/Δz² + 1/(ρΔα)²)).
    let arc = 1.75e-3 * TAU / 24.0;
    let cfl =
        1.0 / (C0 * (1.0 / 0.5e-3f64.powi(2) + 1.0 / 1e-3f64.powi(2) + 1.0 / arc.powi(2)).sqrt());
    println!(
        "dt single grid {fine:.4e} s, multigrid {mg:.4e} s: x{:.2} (CFL estimate {cfl:.4e} s)",
        mg / fine
    );
    assert!(mg / fine > 3.5, "{}", mg / fine);
    assert!(mg < 1.01 * cfl, "{mg:e} vs {cfl:e}");
    // Removing the zones gives the single-grid limit back.
    let off = cavity_grid(true).with_multigrid(&[]).unwrap();
    assert_eq!(courant_limit(&off), fine);
}

#[test]
fn multigrid_cavity_resonances() {
    let mut r = cavity(0);
    let dt = r.engine.dt();
    let t_end = r.t_off + 4e-9;
    let steps = (t_end / dt).ceil() as u64;
    r.engine.run(steps);
    // Same tolerances as the single-grid test (`tests/cylindrical.rs`):
    // 0.5 %, TE211 1 % (discrete azimuthal wavenumber of the 24 α cells).
    let expected = [
        ("TM010", mode(J01, 0), 5e-3),
        ("TE111", mode(JP11, 1), 5e-3),
        ("TM011", mode(J01, 1), 5e-3),
        ("TE211", mode(JP21, 1), 1e-2),
        ("TM110", mode(J11, 0), 5e-3),
        ("TE011", mode(JP01, 1), 5e-3),
    ];
    for (name, f, tol) in expected {
        let (lo, hi) = (0.985 * f, 1.015 * f);
        let mut best = (0.0, f64::NAN);
        for h in &r.probes {
            let pr = r.engine.operator(h);
            let fp = peak(pr.times(), pr.values(), r.t_off, lo, hi);
            let m = spectrum(pr.times(), pr.values(), r.t_off, fp);
            if m > best.0 {
                best = (m, fp);
            }
        }
        let err = best.1 / f - 1.0;
        println!(
            "{name}: {:.4} GHz (analytic {:.4}), {:+.3} %",
            best.1 / 1e9,
            f / 1e9,
            100.0 * err
        );
        assert!(err.abs() < tol, "{name}: {err:+e}");
    }
}

#[test]
fn multigrid_energy_conserved_and_threads_agree() {
    let mut a = cavity(1);
    let mut b = cavity(3);
    let n0 = (a.t_off / a.engine.dt()).ceil() as u64;
    a.engine.run(n0);
    b.engine.run(n0);
    let w0 = a.engine.energy();
    assert!(w0 > 0.0);
    let mut worst = 0.0f64;
    for _ in 0..10 {
        a.engine.run(500);
        b.engine.run(500);
        let w = a.engine.energy();
        worst = worst.max((w / w0 - 1.0).abs());
    }
    println!("energy drift {worst:e}");
    assert!(worst < 1e-5, "{worst:e}");
    for axis in Axis::ALL {
        assert_eq!(a.engine.u_component(axis), b.engine.u_component(axis));
        assert_eq!(a.engine.i_component(axis), b.engine.i_component(axis));
    }
    // Removed edges stay 0, the fine α edges of a coarse α edge are equal
    // (uniform Δα), and the slaves on the zone boundaries (lines 3 and 1)
    // interpolate their masters.
    let e = &a.engine;
    let close = |s: f32, m: f64| (s as f64 - m).abs() <= 1e-6 * m.abs().max(1e-30);
    for k in 0..15 {
        for j in 0..24 {
            if j % 2 == 1 {
                assert_eq!(e.u(Axis::X, [2, j, k]), 0.0);
                assert_eq!(e.u(Axis::Z, [2, j, k]), 0.0);
                assert_eq!(e.u(Axis::Z, [1, j, k]), 0.0);
                let m =
                    0.5 * (e.u(Axis::Z, [3, j - 1, k]) as f64 + e.u(Axis::Z, [3, j + 1, k]) as f64);
                assert!(close(e.u(Axis::Z, [3, j, k]), m));
            } else {
                assert_eq!(e.u(Axis::Y, [2, j, k]), e.u(Axis::Y, [2, j + 1, k]));
                assert_eq!(e.u(Axis::Y, [1, j, k]), e.u(Axis::Y, [1, j + 1, k]));
            }
            if j % 4 != 0 {
                assert_eq!(e.u(Axis::X, [0, j, k]), 0.0);
            }
            if j % 4 == 2 {
                let m =
                    0.5 * (e.u(Axis::Z, [1, j - 2, k]) as f64 + e.u(Axis::Z, [1, j + 2, k]) as f64);
                assert!(close(e.u(Axis::Z, [1, j, k]), m));
            }
        }
    }
}

#[test]
fn multigrid_stable_at_the_courant_limit() {
    // Random initial fields, run at the computed limit itself for a long
    // time: the energy invariant stays put (no growth). Three levels on a
    // closed grid with an axis (cyclic zone-boundary solve), a PEC-walled
    // sector (open chain), and an annulus without axis.
    let cases = [
        ("closed, axis", 0.0, TAU, 24, vec![1, 2, 4]),
        ("sector", 0.0, 0.5 * PI, 16, vec![2, 4]),
        ("annulus", 1e-3, TAU, 24, vec![3]),
    ];
    for (name, r0, span, na, rings) in cases {
        let single = Grid::cylindrical([
            lines(r0, 6e-3, 6),
            lines(0.0, span, na),
            lines(0.0, 4e-3, 4),
        ])
        .unwrap();
        let fine = courant_limit(&single);
        let grid = single.with_multigrid(&rings).unwrap();
        let mut e = Engine::builder(grid)
            .timestep(TimeStep::Courant(1.0))
            .threads(2)
            .build()
            .unwrap();
        {
            let mut seed: u32 = 12345;
            let mut f = e.fields_mut();
            for axis in Axis::ALL {
                for i in 0..7 {
                    for j in 0..=na {
                        for k in 0..5 {
                            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            let v = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                            f.u.set(axis, [i, j, k], v);
                        }
                    }
                }
            }
        }
        // One step projects the random state onto the multigrid subspace.
        e.step();
        let w0 = e.energy();
        let mut worst = 0.0f64;
        for _ in 0..20 {
            e.run(1000);
            worst = worst.max((e.energy() / w0 - 1.0).abs());
        }
        println!(
            "{name}: dt x{:.2} over single grid, worst energy deviation over 20000 steps {worst:e}",
            e.dt() / fine
        );
        assert!(e.dt() > 1.5 * fine, "{name}");
        assert!(worst < 1e-4, "{name}: {worst:e}");
    }
}

#[test]
fn invalid_multigrid() {
    let g = cavity_grid(false);
    assert!(g.clone().with_multigrid(&[0]).is_err());
    assert!(g.clone().with_multigrid(&[20]).is_err());
    assert!(g.clone().with_multigrid(&[5, 5]).is_err());
    // 24 cells: 8 = 2³ divides 24, 16 does not.
    assert!(g.clone().with_multigrid(&[2, 4, 6]).is_ok());
    assert!(g.clone().with_multigrid(&[2, 4, 6, 8]).is_err());
    let cart = Grid::uniform([4, 4, 4], [1e-3; 3]).unwrap();
    assert!(cart.with_multigrid(&[2]).is_err());
}

/// The circular cavity built in csx (drawing unit mm, 20 × 24 × 15 cells)
/// with multigrid radii 1.5 and 0.5 mm, through [`Simulation`].
#[test]
fn csx_cavity_with_multigrid() {
    use csx::primitives::BoxShape;
    use csx::properties::{Excitation, ProbeBox, ProbeType};
    use csx::{ContinuousStructure, CoordSystem, Primitive, Property};
    use ems::{Fdtd, Simulation, Waveform};

    let alpha = |j: usize| j as f64 * TAU / 24.0;
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid
        .set_lines(0, (0..=20).map(|i| i as f64 * 0.5))
        .unwrap();
    m.grid.set_lines(1, (0..=24).map(alpha)).unwrap();
    m.grid.set_lines(2, (0..=15).map(f64::from)).unwrap();
    let ez = m.add_property(Property::excitation("ez", Excitation::new([0.0, 0.0, 1.0])));
    m.add_primitive(
        ez,
        Primitive::new(BoxShape::new([4.5, alpha(3), 2.0], [4.5, alpha(3), 4.0])),
    )
    .unwrap();
    let er = m.add_property(Property::excitation("er", Excitation::new([1.0, 0.0, 0.0])));
    m.add_primitive(
        er,
        Primitive::new(BoxShape::new([3.0, alpha(7), 9.0], [4.0, alpha(7), 9.0])),
    )
    .unwrap();
    for (name, a, b) in [
        ("vz", [7.0, alpha(5), 3.0], [7.0, alpha(5), 8.0]),
        ("vr", [5.0, alpha(11), 11.0], [6.5, alpha(11), 11.0]),
    ] {
        let id = m.add_property(Property::probe(name, ProbeBox::new(ProbeType::Voltage)));
        m.add_primitive(id, Primitive::new(BoxShape::new(a, b)))
            .unwrap();
    }
    let fdtd = |mg: &[f64]| {
        Fdtd::new()
            .waveform(Waveform::gaussian(15e9, 10e9))
            .max_time(2e-9)
            .no_end_criteria()
            .multigrid(mg.to_vec())
    };
    let single = Simulation::new(&m, &fdtd(&[])).unwrap();
    let mut sim = Simulation::new(&m, &fdtd(&[1.5, 0.5])).unwrap();
    assert_eq!(sim.grid().multigrid(), &[3, 1]);
    assert!(
        sim.warnings().iter().all(|w| w.starts_with("mesh ")),
        "{:?}",
        sim.warnings()
    );
    let res = sim.run().unwrap();
    let gain = res.dt() / courant_limit(single.grid()) / 0.99;
    println!("dt gain x{gain:.2}");
    assert!(gain > 3.5, "{gain}");
    let t0 = 2.0 * fdtd(&[]).get_waveform().unwrap().duration().unwrap();
    let probe = |n: &str| {
        let p = res.probe(n).unwrap();
        (p.times().to_vec(), p.scalar().unwrap().to_vec())
    };
    let (vz, vr) = (probe("vz"), probe("vr"));
    for (name, f, p) in [
        ("TM010", mode(J01, 0), &vz),
        ("TM011", mode(J01, 1), &vz),
        ("TE111", mode(JP11, 1), &vr),
        ("TM110", mode(J11, 0), &vz),
    ] {
        let got = peak(&p.0, &p.1, t0, 0.985 * f, 1.015 * f);
        let err = got / f - 1.0;
        println!(
            "{name}: {:.4} GHz (analytic {:.4}) {:+.3} %",
            got / 1e9,
            f / 1e9,
            100.0 * err
        );
        assert!(err.abs() < 5e-3, "{name} {err:+e}");
    }
    // A radius off the mesh lines snaps with a warning; one at the axis
    // is refused.
    let sim = Simulation::new(&m, &fdtd(&[1.4])).unwrap();
    assert_eq!(sim.grid().multigrid(), &[3]);
    assert!(sim.warnings().iter().any(|w| w.contains("snapped")));
    assert!(Simulation::new(&m, &fdtd(&[0.1])).is_err());
}
