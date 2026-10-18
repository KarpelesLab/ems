//! Cylindrical-coordinate engine (design §5.15): circular PEC cavity
//! resonances against the Bessel-zero formulas, energy conservation and
//! stability with an axis, thread-count determinism, and a coaxial line.

use std::f64::consts::{PI, TAU};

use ems::constants::{C0, Z0};
use ems::engine::{
    DumpPosition, DumpQuantity, DumpRegion, EdgeSource, Engine, FaceCondition, FaceConditions,
    FieldSampler, Grid, Probe, ProbeQuantity, TimeStep,
    boundary::{CpmlBoundary, MurBoundary},
};
use ems::{Axis, Cpml, Face};

/// Gaussian-modulated sine centred at `f0` with ~`bw` bandwidth (zero
/// mean, so a soft E source leaves no net charge); returns the pulse and
/// its duration.
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

fn cavity_grid(nr: usize, na: usize, nz: usize) -> Grid {
    Grid::cylindrical([lines(0.0, A, nr), lines(0.0, TAU, na), lines(0.0, H, nz)]).unwrap()
}

/// The cavity driven by a short pulse on one edge of each component, with
/// probes on a few edges of each component.
struct Run {
    engine: Engine,
    probes: Vec<ems::engine::OperatorHandle<Probe>>,
    t_off: f64,
}

fn cavity(threads: usize) -> Run {
    let grid = cavity_grid(20, 24, 15);
    let mut engine = Engine::builder(grid).threads(threads).build().unwrap();
    let (p, t_off) = pulse(15e9, 12e9);
    // An off-axis z edge (TM), a ρ and an α edge (TE), all asymmetric in α.
    let src = EdgeSource::new(p)
        .edge(Axis::Z, [9, 3, 4], 1.0)
        .edge(Axis::X, [6, 7, 9], 1.0)
        .edge(Axis::Y, [13, 17, 5], 1.0)
        .edge(Axis::Z, [0, 0, 6], 1.0);
    engine.add_operator(src);
    let mut probes = Vec::new();
    for (axis, node) in [
        (Axis::Z, [7, 5, 3]),
        (Axis::Z, [0, 0, 8]),
        (Axis::X, [11, 2, 11]),
        (Axis::Y, [4, 9, 6]),
        (Axis::Y, [16, 20, 10]),
    ] {
        probes.push(engine.add_operator(Probe::voltage(axis, node)));
    }
    Run {
        engine,
        probes,
        t_off,
    }
}

/// Bessel zeros `j_mn` and `j'_mn`.
const J01: f64 = 2.404_825_557_695_773;
const J11: f64 = 3.831_705_970_207_512;
const JP11: f64 = 1.841_183_781_340_659;
const JP21: f64 = 3.054_236_928_227_14;
const JP01: f64 = 3.831_705_970_207_512;

fn mode(x: f64, p: u32) -> f64 {
    C0 / TAU * ((x / A).powi(2) + (p as f64 * PI / H).powi(2)).sqrt()
}

#[test]
fn circular_cavity_resonances() {
    let mut r = cavity(0);
    let dt = r.engine.dt();
    let t_end = r.t_off + 4e-9;
    let steps = (t_end / dt).ceil() as u64;
    r.engine.run(steps);
    // Tolerances: 0.5 %, except TE211, whose m = 2 azimuthal variation
    // sees the discrete wavenumber (2/Δα)·sin(mΔα/2) of the 24 α cells
    // (−1.1 % on m, −0.61 % on the frequency).
    let expected = [
        ("TM010", mode(J01, 0), 5e-3),
        ("TE111", mode(JP11, 1), 5e-3),
        ("TM011", mode(J01, 1), 5e-3),
        ("TE211", mode(JP21, 1), 1e-2),
        ("TM110", mode(J11, 0), 5e-3),
        ("TE011", mode(JP01, 1), 5e-3),
    ];
    for (name, f, tol) in expected {
        // Best-matching probe: the one with the strongest peak near f.
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
fn energy_conserved_and_threads_agree() {
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
    // The copies of the axis z edge are equal, α edges on the axis are 0.
    let [_, na, nz] = a.engine.grid().nodes();
    for k in 0..nz - 1 {
        let v = a.engine.u(Axis::Z, [0, 0, k]);
        for j in 0..na {
            assert_eq!(a.engine.u(Axis::Z, [0, j, k]), v);
            assert_eq!(a.engine.u(Axis::Y, [0, j, k]), 0.0);
        }
    }
    // The closed azimuth: both copies of the α = 0 plane agree.
    for i in 0..20 {
        for k in 0..nz {
            assert_eq!(
                a.engine.u(Axis::X, [i, 0, k]),
                a.engine.u(Axis::X, [i, na - 1, k])
            );
        }
    }
}

#[test]
fn stable_at_the_courant_limit() {
    // Random initial fields on a coarse grid with an axis, run at the
    // computed limit itself: the energy invariant stays put.
    let grid = cavity_grid(6, 12, 4);
    let mut e = Engine::builder(grid)
        .timestep(TimeStep::Courant(1.0))
        .threads(1)
        .build()
        .unwrap();
    {
        let mut seed: u32 = 12345;
        let mut f = e.fields_mut();
        let [nr, na, nz] = [7, 13, 5];
        for axis in Axis::ALL {
            for i in 0..nr {
                for j in 0..na {
                    for k in 0..nz {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let v = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                        f.u.set(axis, [i, j, k], v);
                    }
                }
            }
        }
    }
    e.step();
    let w0 = e.energy();
    e.run(20_000);
    let w = e.energy();
    println!("dt {:e}, energy ratio {}", e.dt(), w / w0);
    assert!((w / w0 - 1.0).abs() < 1e-4, "{}", w / w0);
}

#[test]
fn coaxial_line_impedance() {
    // Coax a = 1 mm, b = 3 mm along z, Mur at both ends, closed azimuth.
    let (ra, rb) = (1e-3, 3e-3);
    let nr = 10;
    let nz = 120;
    let dz = 0.5e-3;
    let rho = lines(ra, rb, nr);
    let grid = Grid::cylindrical([
        rho.clone(),
        lines(0.0, TAU, 8),
        lines(0.0, nz as f64 * dz, nz),
    ])
    .unwrap();
    let faces = FaceConditions::all(FaceCondition::Pec);
    let mut e = Engine::builder(grid)
        .faces(faces)
        .threads(2)
        .build()
        .unwrap();
    let mur = MurBoundary::with_faces(&e, &[(Face::ZMin, C0), (Face::ZMax, C0)]).unwrap();
    e.add_operator(mur);
    // TEM source: E_ρ ∝ 1/ρ on a ring at k = 5, line voltage 1.
    let (p, _) = pulse(6e9, 5e9);
    let mut src = EdgeSource::new(p);
    for i in 0..nr {
        let w = (rho[i + 1] - rho[i]) / (0.5 * (rho[i] + rho[i + 1]));
        for j in 0..9 {
            src = src.edge(Axis::X, [i, j, 5], w);
        }
    }
    e.add_operator(src);
    let k = 60;
    let mut v = Probe::new(ProbeQuantity::Voltage);
    for i in 0..nr {
        v = v.entry(Axis::X, [i, 2, k], 1.0);
    }
    let mut cur = Probe::new(ProbeQuantity::Current);
    for j in 0..9 {
        cur = cur.entry(Axis::Y, [4, j, k], 1.0);
    }
    let (hv, hi) = (e.add_operator(v), e.add_operator(cur));
    let ring = |k: usize| {
        let mut p = Probe::new(ProbeQuantity::Voltage);
        for i in 0..nr {
            p = p.entry(Axis::X, [i, 2, k], 1.0);
        }
        p
    };
    let (h40, h80) = (e.add_operator(ring(40)), e.add_operator(ring(80)));
    let steps = (3e-9 / e.dt()).ceil() as u64;
    e.run(steps);
    let (pv, pi) = (e.operator(&hv), e.operator(&hi));
    // Discrete TEM impedance (midpoint rule for ∫dρ/ρ) and the analytic one.
    let discrete: f64 = (0..nr)
        .map(|i| (rho[i + 1] - rho[i]) / (0.5 * (rho[i] + rho[i + 1])))
        .sum::<f64>()
        * Z0
        / TAU;
    let analytic = Z0 / TAU * (rb / ra).ln();
    for f in [3e9, 6e9, 9e9] {
        // Plain DFT (both records decay to zero; a window would weigh V
        // and I, half a cell and half a step apart, differently).
        let z = dft(pv, f).hypot_ratio(dft(pi, f));
        println!(
            "{:.1} GHz: |Z| = {z:.3} Ω (discrete {discrete:.3}, analytic {analytic:.3})",
            f / 1e9
        );
        assert!((z / discrete - 1.0).abs() < 2e-3, "{z} vs {discrete}");
        assert!((z / analytic - 1.0).abs() < 1e-2, "{z} vs {analytic}");
    }
    // Phase velocity between k = 40 and k = 80 from the DFT phase: c up
    // to the (small) dispersion of the mesh.
    let (p40, p80) = (e.operator(&h40), e.operator(&h80));
    for f in [3e9, 6e9, 9e9] {
        let (a40, a80) = (phase(p40, f), phase(p80, f));
        let dphi = (a40 - a80).rem_euclid(TAU);
        let v = TAU * f * 40.0 * dz / dphi;
        println!("{:.1} GHz: phase velocity {:.6} c", f / 1e9, v / C0);
        assert!((v / C0 - 1.0).abs() < 2e-3, "{}", v / C0);
    }
}

/// Plain DFT of a probe at `f`, as `(re, im)`.
fn dft(p: &Probe, f: f64) -> Dft {
    let (mut re, mut im) = (0.0, 0.0);
    for (&t, &x) in p.times().iter().zip(p.values()) {
        re += x * (TAU * f * t).cos();
        im -= x * (TAU * f * t).sin();
    }
    Dft(re, im)
}

#[derive(Clone, Copy)]
struct Dft(f64, f64);

impl Dft {
    fn hypot_ratio(self, o: Dft) -> f64 {
        self.0.hypot(self.1) / o.0.hypot(o.1)
    }
}

/// Phase of the DFT of a probe at `f`.
fn phase(p: &Probe, f: f64) -> f64 {
    let d = dft(p, f);
    d.1.atan2(d.0)
}

/// Largest |difference| of two equally sampled series over their common
/// length, relative to the largest |reference| value.
fn rel_diff(test: &[f64], reference: &[f64]) -> f64 {
    let n = test.len().min(reference.len());
    let d = (0..n).fold(0.0f64, |m, q| m.max((test[q] - reference[q]).abs()));
    let r = reference[..n].iter().fold(0.0f64, |m, x| m.max(x.abs()));
    d / r
}

/// TEM voltage at k = 60 on a coax (a = 1 mm, b = 3 mm) with `nz` cells
/// of 0.5 mm, 8-cell CPML on both z faces, soft source at k = 20, over
/// 1.2 ns.
fn coax_cpml(nz: usize) -> Vec<f64> {
    let rho = lines(1e-3, 3e-3, 10);
    let grid = Grid::cylindrical([
        rho.clone(),
        lines(0.0, TAU, 8),
        lines(0.0, nz as f64 * 0.5e-3, nz),
    ])
    .unwrap();
    let mut e = Engine::builder(grid).threads(2).build().unwrap();
    let cpml = CpmlBoundary::new(
        &e,
        [(Face::ZMin, Cpml::cells(8)), (Face::ZMax, Cpml::cells(8))],
    )
    .unwrap();
    e.add_operator(cpml);
    let (p, _) = pulse(6e9, 5e9);
    let mut src = EdgeSource::new(p);
    for i in 0..10 {
        let w = (rho[i + 1] - rho[i]) / (0.5 * (rho[i] + rho[i + 1]));
        for j in 0..9 {
            src = src.edge(Axis::X, [i, j, 20], w);
        }
    }
    e.add_operator(src);
    let mut v = Probe::new(ProbeQuantity::Voltage);
    for i in 0..10 {
        v = v.entry(Axis::X, [i, 2, 60], 1.0);
    }
    let h = e.add_operator(v);
    e.run((1.2e-9 / e.dt()).ceil() as u64);
    e.operator(&h).values().to_vec()
}

#[test]
fn cpml_on_z_faces_absorbs_tem() {
    // The CPML layer of the short line starts 52 cells past the probe; the
    // reference line is long enough that nothing returns within 1.2 ns.
    let r = rel_diff(&coax_cpml(120), &coax_cpml(420));
    let db = 20.0 * r.log10();
    println!("coax CPML(8) reflection {db:.1} dB");
    assert!(db < -60.0, "{db:.1} dB");
}

/// E_z at ρ index `probe` of a z-invariant radial (TM₀) wave from a soft
/// source on the axis, between PEC plates 1 mm apart, with an 8-cell CPML
/// on the ρ_max face (`nr` cells of 0.5 mm), over 0.9 ns.
fn radial_cpml(nr: usize, probe: usize) -> Vec<f64> {
    radial(nr, probe, true)
}

fn radial(nr: usize, probe: usize, use_cpml: bool) -> Vec<f64> {
    let grid = Grid::cylindrical([
        lines(0.0, nr as f64 * 0.5e-3, nr),
        lines(0.0, TAU, 8),
        lines(0.0, 1e-3, 2),
    ])
    .unwrap();
    let mut e = Engine::builder(grid).threads(2).build().unwrap();
    if use_cpml {
        let cpml = CpmlBoundary::new(&e, [(Face::XMax, Cpml::cells(8))]).unwrap();
        e.add_operator(cpml);
    } else {
        e.add_operator(MurBoundary::new(&e, Face::XMax, C0).unwrap());
    }
    let (p, _) = pulse(15e9, 10e9);
    let mut src = EdgeSource::new(p);
    for j in 0..9 {
        for k in 0..2 {
            src = src.edge(Axis::Z, [0, j, k], 1.0);
        }
    }
    e.add_operator(src);
    let h = e.add_operator(Probe::voltage(Axis::Z, [probe, 3, 0]));
    e.run((0.9e-9 / e.dt()).ceil() as u64);
    e.operator(&h).values().to_vec()
}

#[test]
fn rho_face_absorbers() {
    // Mur on the ρ_max face uses the plane-wave one-way equation, so a
    // cylindrical wave reflects at the order of 1/(kρ): about −31 dB at
    // ρ = 26 mm (5–25 GHz), −37 dB at 56 mm, −43 dB at 116 mm (the
    // Cartesian-style CPML alike). A PML there is the cylindrical PML
    // (§5.15.2, tests/cyl_pml.rs). Probe at ρ = 50 mm, layer from
    // ρ = 56 mm; the reference ends at 260 mm.
    let reference = radial(520, 100, true);
    let cpml = 20.0 * rel_diff(&radial_cpml(120, 100), &reference).log10();
    let mur = 20.0 * rel_diff(&radial(120, 100, false), &reference).log10();
    println!("rho_max reflection: PML(8) {cpml:.1} dB, Mur {mur:.1} dB");
    assert!(cpml < -80.0 && mur < -34.0, "{cpml:.1} / {mur:.1} dB");
}

#[test]
fn dumps_sample_physical_components() {
    // A uniform E_α = 2 V/m (u_α = 2·ρ_i·Δα_j) and H_α = 3 A/m
    // (i_α = 3·ρ_{i+½}·Δα*_j) read back as such, at nodes and cell centres
    // (next to the axis from the outer edge alone); E_α on the axis itself
    // is 0 (the edges do not exist).
    let grid =
        Grid::cylindrical([lines(0.0, 4e-3, 4), lines(0.0, 1.0, 4), lines(0.0, 2e-3, 2)]).unwrap();
    let mut e = Engine::builder(grid.clone())
        .face(Face::YMin, FaceCondition::Pmc)
        .face(Face::YMax, FaceCondition::Pmc)
        .threads(1)
        .build()
        .unwrap();
    {
        let mut f = e.fields_mut();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..3 {
                    let n = [i, j, k];
                    if j < 4 {
                        f.u.set(Axis::Y, n, (2.0 * grid.edge_length(Axis::Y, n)) as f32);
                    }
                    if i < 4 && k < 2 {
                        f.i.set(Axis::Y, n, (3.0 * grid.dual_edge_length(Axis::Y, n)) as f32);
                    }
                }
            }
        }
    }
    let region = DumpRegion::new([0, 0, 0], [4, 4, 2]);
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6 * b.abs().max(1.0);
    for pos in [DumpPosition::Nodes, DumpPosition::CellCenters] {
        let se = FieldSampler::new(e.grid(), region, DumpQuantity::E, pos).unwrap();
        let sh = FieldSampler::new(e.grid(), region, DumpQuantity::H, pos).unwrap();
        let (ve, vh) = {
            let f = e.fields_mut();
            (se.sample(&f), sh.sample(&f))
        };
        let [dx, dy, dz] = se.dims();
        for p in 0..dx {
            let rho = se.coords(Axis::X)[p];
            for q in 0..dy {
                for r in 0..dz {
                    let v = ve[(p * dy + q) * dz + r][1] as f64;
                    if rho == 0.0 {
                        assert_eq!(v, 0.0);
                    } else {
                        assert!(close(v, 2.0), "{pos:?} E_α {v} at ρ {rho}");
                    }
                    let h = vh[(p * dy + q) * dz + r][1] as f64;
                    assert!(close(h, 3.0), "{pos:?} H_α {h} at ρ {rho}");
                }
            }
        }
        // α coordinates in radians.
        assert!(close(
            se.coords(Axis::Y)[se.dims()[1] - 1],
            if pos == DumpPosition::Nodes {
                1.0
            } else {
                0.875
            }
        ));
    }
}
