//! CPML validation: normal-incidence reflection (uniform and graded PML
//! mesh, CFS parameters, dielectric background), a 3D point source against
//! a large reference box, energy decay and long-run stability, and thread
//! determinism.
//!
//! Reflections are measured by time gating against a reference run in a
//! domain so large that its own boundary reflections arrive only after the
//! comparison window (Taflove & Hagness, *Computational Electrodynamics*,
//! 3rd ed., ch. 7, numerical-reflection experiments).

use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::{
    CellMaterials, EdgeSource, Engine, FaceCondition, FaceConditions, Grid, Material, Probe,
};
use ems::{Axis, Cpml, Face};

/// Gaussian-modulated pulse centred at `f0` with ~`bw` bandwidth.
fn pulse(f0: f64, bw: f64) -> impl Fn(f64) -> f64 + Send + 'static {
    let tau = 1.0 / (std::f64::consts::PI * bw);
    let t0 = 4.0 * tau;
    move |t: f64| {
        let x = (t - t0) / tau;
        (-x * x).exp() * (2.0 * std::f64::consts::PI * f0 * (t - t0)).cos()
    }
}

/// Derivative-of-Gaussian pulse (zero time integral, so a soft source
/// leaves no static charge behind), width `tau`, duration `8·tau`.
fn dgauss(tau: f64) -> impl Fn(f64) -> f64 + Send + 'static {
    let t0 = 4.0 * tau;
    move |t: f64| {
        let x = (t - t0) / tau;
        -2.0 * x * (-x * x).exp()
    }
}

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

/// Max |a − b| / max |b|.
fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let d = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    let m = b.iter().map(|y| y.abs()).fold(0.0, f64::max);
    d / m
}

/// Lines `z0, z0 + d, …` for `n` cells.
fn line(z0: f64, d: f64, n: usize) -> Vec<f64> {
    (0..=n).map(|q| z0 + q as f64 * d).collect()
}

const DZ: f64 = 1e-3;

/// Quasi-1D TEM line along z: one 10 mm × 10 mm cell in x and y, PEC x
/// walls and PMC y walls, so the E_x / H_y plane wave is exact. `lead`
/// cells precede the interior (CPML or plain medium). Source at interior
/// line `src`, probe at interior line `prb`. Returns the probe samples.
#[allow(clippy::too_many_arguments)]
fn line_run(
    z: Vec<f64>,
    eps_r: f64,
    pml: Option<Cpml>,
    lead: usize,
    src: usize,
    prb: usize,
    steps: u64,
) -> Vec<f64> {
    let g = Grid::new([vec![0.0, 0.01], vec![0.0, 0.01], z]).unwrap();
    let faces = FaceConditions::all(FaceCondition::Pec)
        .with(Face::YMin, FaceCondition::Pmc)
        .with(Face::YMax, FaceCondition::Pmc);
    let m = CellMaterials::new(&g, Material::dielectric(eps_r));
    let mut e = Engine::builder(g)
        .materials(m)
        .faces(faces)
        .threads(1)
        .build()
        .unwrap();
    if let Some(p) = pml {
        let b = CpmlBoundary::new(&e, [(Face::ZMin, p), (Face::ZMax, p)]).unwrap();
        e.add_operator(b);
    }
    let f = pulse(10e9 / eps_r.sqrt(), 10e9 / eps_r.sqrt());
    let ks = lead + src;
    e.add_operator(EdgeSource::new(f).edge(Axis::X, [0, 0, ks], 1.0).edge(
        Axis::X,
        [0, 1, ks],
        1.0,
    ));
    let h = e.add_operator(Probe::voltage(Axis::X, [0, 0, lead + prb]));
    e.run(steps);
    e.operator(&h).values().to_vec()
}

/// Reflection of a normally incident pulse from a CPML whose cells (from
/// the interface outwards) have the sizes `pml_cells`, in a background of
/// relative permittivity `eps_r`.
fn normal_reflection(pml_cells: &[f64], p: Cpml, eps_r: f64) -> f64 {
    let n = pml_cells.len();
    let interior = 150;
    let (src, prb) = (30, 100);
    // Test: PML | interior | PML.
    let mut z = vec![0.0];
    for &d in pml_cells.iter().rev() {
        z.push(z.last().unwrap() + d);
    }
    let z_int = *z.last().unwrap();
    z.extend(line(z_int, DZ, interior).into_iter().skip(1));
    for &d in pml_cells {
        z.push(z.last().unwrap() + d);
    }
    // Window: the incident pulse and both PML echoes pass the probe, but
    // the reference's PEC ends (≥ 1800 cells of round trip away) do not.
    let steps = (900.0 * eps_r.sqrt()) as u64;
    let mut p = p;
    p.cells = n;
    let test = line_run(z, eps_r, Some(p), n, src, prb, steps);
    let pad = 1000;
    let zr = line(0.0, DZ, interior + 2 * pad);
    let refr = line_run(zr, eps_r, None, pad, src, prb, steps);
    rel_err(&test, &refr)
}

#[test]
fn normal_incidence_uniform() {
    let r = normal_reflection(&[DZ; 8], Cpml::cells(8), 1.0);
    eprintln!("normal incidence, 8 cells m = 3: {:.1} dB", db(r));
    assert!(db(r) < -70.0, "reflection {:.1} dB", db(r));
    // Sanity check of the gating: without loss the PEC-backed layer
    // reflects totally.
    let r0 = normal_reflection(&[DZ; 8], Cpml::cells(8).sigma_factor(0.0), 1.0);
    assert!(db(r0) > -1.0, "window misses the echo: {:.1} dB", db(r0));
}

#[test]
fn normal_incidence_graded_mesh() {
    // PML cells growing outwards by 1.25×: 1, 1.25, 1.56, … 4.8 mm.
    let cells: Vec<f64> = (0..8).map(|q| DZ * 1.25f64.powi(q)).collect();
    let r = normal_reflection(&cells, Cpml::cells(8), 1.0);
    eprintln!("normal incidence, 8 cells graded 1.25x: {:.1} dB", db(r));
    assert!(db(r) < -45.0, "reflection {:.1} dB", db(r));
    let cells: Vec<f64> = (0..8).map(|q| DZ * 1.1f64.powi(q)).collect();
    let r = normal_reflection(&cells, Cpml::cells(8), 1.0);
    eprintln!("normal incidence, 8 cells graded 1.1x: {:.1} dB", db(r));
    assert!(db(r) < -50.0, "reflection {:.1} dB", db(r));
}

#[test]
fn normal_incidence_cfs() {
    let p = Cpml::cells(8).kappa_max(3.0).alpha_max(0.02);
    let r = normal_reflection(&[DZ; 8], p, 1.0);
    eprintln!(
        "normal incidence, CFS (κ_max 3, α_max 0.02): {:.1} dB",
        db(r)
    );
    assert!(db(r) < -60.0, "reflection {:.1} dB", db(r));
}

/// Homogeneous dielectric background, including the layer: `σ_max` is
/// tuned to the layer's refractive index automatically, so the default
/// layer is as good as in vacuum (−85.7 dB for ε_r = 4 and 12; the vacuum
/// `σ_max`, i.e. `sigma_factor = n`, gives −80.5 / −73.4 dB).
#[test]
fn normal_incidence_dielectric_background() {
    for eps in [4.0f64, 12.0] {
        let r = normal_reflection(&[DZ; 8], Cpml::cells(8), eps);
        eprintln!(
            "normal incidence, eps_r = {eps} background: {:.1} dB",
            db(r)
        );
        assert!(db(r) < -80.0, "ε_r = {eps}: reflection {:.1} dB", db(r));
        let v = normal_reflection(&[DZ; 8], Cpml::cells(8).sigma_factor(eps.sqrt()), eps);
        eprintln!("  with the vacuum sigma_max: {:.1} dB", db(v));
        assert!(db(v) > db(r) + 3.0, "{} {}", db(v), db(r));
    }
}

/// 3D cubic box of `interior` cells of size `d` plus `pml` CPML cells (or
/// plain vacuum PEC-backed padding when `cpml` is false) on every face. An
/// E_z soft source at the centre; voltage probes at the given offsets (in
/// cells) from the source node. Returns one sample vector per probe.
fn box_run(
    interior: usize,
    pml: usize,
    cpml: Option<Cpml>,
    probes: &[[isize; 3]],
    steps: u64,
    threads: usize,
) -> Vec<Vec<f64>> {
    let d = 2e-3;
    let n = interior + 2 * pml;
    let g = Grid::uniform([n; 3], [d; 3]).unwrap();
    let mut e = Engine::builder(g).threads(threads).build().unwrap();
    if let Some(p) = cpml {
        let b = CpmlBoundary::new(
            &e,
            Face::ALL.map(|f| {
                let mut p = p;
                p.cells = pml;
                (f, p)
            }),
        )
        .unwrap();
        e.add_operator(b);
    }
    let c = n / 2;
    e.add_operator(EdgeSource::new(dgauss(25e-12)).edge(Axis::Z, [c, c, c], 1.0));
    let at = |o: [isize; 3]| o.map(|v| (c as isize + v) as usize);
    let hs: Vec<_> = probes
        .iter()
        .map(|&o| e.add_operator(Probe::voltage(Axis::Z, at(o))))
        .collect();
    e.run(steps);
    hs.iter().map(|h| e.operator(h).values().to_vec()).collect()
}

/// Point source in a box with CPML on all six faces vs a large PEC box,
/// within a window that ends before the large box's echoes return. The
/// probes are close to the layer, one on axis and two towards the edge and
/// corner (oblique incidence on two and three layers). Returns the relative
/// error per probe for each parameter set in `params`.
fn point_source(interior: usize, steps: u64, pad: usize, params: &[Cpml]) -> Vec<Vec<f64>> {
    let h = interior as isize / 2;
    let probes = [[h - 3, 0, 0], [h - 3, h - 3, 0], [h - 3, h - 3, h - 4]];
    let refr = box_run(interior, pad, None, &probes, steps, 4);
    params
        .iter()
        .map(|&p| {
            let test = box_run(interior, 8, Some(p), &probes, steps, 4);
            test.iter().zip(&refr).map(|(t, r)| rel_err(t, r)).collect()
        })
        .collect()
}

#[test]
fn point_source_3d() {
    // 12^3 interior + 8-cell CPML; the reference box has 22 cells of padding
    // per side: its walls echo back after >= 2*28 - 3 = 53 cells of travel,
    // the window is 80*c*dt = 46 cells. The pulse is ~30 cells long.
    let lossless = Cpml::default().sigma_factor(0.0);
    let errs = point_source(12, 80, 22, &[Cpml::default(), lossless]);
    for (q, r) in errs[0].iter().enumerate() {
        eprintln!("3D point source, probe {q}: {:.1} dB", db(*r));
    }
    assert!(errs[0].iter().all(|&r| db(r) < -60.0), "{errs:?}");
    // Gating sanity check: a lossless layer (PEC 8 cells behind the
    // interface) must show up in the window.
    assert!(errs[1].iter().all(|&r| db(r) > -25.0), "{errs:?}");
}

#[test]
#[ignore = "heavy: 3D reference box of 152^3 cells"]
fn point_source_3d_large() {
    let errs = point_source(24, 220, 64, &[Cpml::default()]);
    for (q, r) in errs[0].iter().enumerate() {
        eprintln!("3D point source (large), probe {q}: {:.1} dB", db(*r));
    }
    assert!(errs[0].iter().all(|&r| db(r) < -60.0), "{errs:?}");
}

/// Energy decay after the pulse has left a small CPML-terminated box, and
/// no late-time growth.
fn energy_decay(interior: usize, p: Cpml, steps: u64, threads: usize) {
    let n = interior + 2 * p.cells;
    let g = Grid::uniform([n; 3], [2e-3; 3]).unwrap();
    let mut e = Engine::builder(g).threads(threads).build().unwrap();
    let b = CpmlBoundary::new(&e, Face::ALL.map(|f| (f, p))).unwrap();
    e.add_operator(b);
    let c = n / 2;
    e.add_operator(EdgeSource::new(dgauss(25e-12)).edge(Axis::Z, [c, c, c], 1.0));
    let mut w_max = 0.0f64;
    for _ in 0..400 {
        e.step();
        w_max = w_max.max(e.energy());
    }
    e.run(600);
    let w1 = e.energy();
    eprintln!("energy: max {w_max:.3e}, after 1000 steps {w1:.3e}");
    assert!(
        w1 < 1e-5 * w_max,
        "energy did not decay: {w1:e} vs {w_max:e}"
    );
    e.run(steps - 1000);
    let w2 = e.energy();
    eprintln!("energy after {steps} steps {w2:.3e}");
    assert!(
        w2.is_finite() && w2 <= w1.max(1e-12 * w_max),
        "late-time growth: {w1:e} -> {w2:e}"
    );
}

#[test]
fn energy_decays_and_stays_small() {
    energy_decay(2, Cpml::cells(3), 20_000, 1);
}

#[test]
#[ignore = "long: 200k steps"]
fn energy_decays_and_stays_small_long() {
    energy_decay(12, Cpml::cells(8), 200_000, 4);
    energy_decay(
        12,
        Cpml::cells(8).kappa_max(4.0).alpha_max(0.05),
        200_000,
        4,
    );
}

/// All field arrays after a CPML run, followed by the samples of two
/// probes inside the x_min / y_max layers and an energy sample. With
/// `probes_first` the probes are registered before the CPML operator.
fn det_run(threads: usize, probes_first: bool) -> Vec<Vec<f32>> {
    // Graded mesh, different layer thicknesses, CFS parameters, a lossy
    // dielectric block reaching into the x_max layer.
    let lines = |len: f64, n: usize| -> Vec<f64> {
        (0..=n)
            .map(|q| {
                let t = q as f64 / n as f64;
                len * (t + 0.05 * (6.0 * t).sin())
            })
            .collect()
    };
    let g = Grid::new([lines(0.04, 21), lines(0.03, 15), lines(0.03, 14)]).unwrap();
    let mut m = CellMaterials::vacuum(&g);
    m.fill_box(
        [10..21, 3..9, 4..8],
        Material::dielectric(3.0).with_kappa(0.05),
    );
    let mut e = Engine::builder(g)
        .materials(m)
        .threads(threads)
        .build()
        .unwrap();
    let p = Cpml::cells(5).kappa_max(2.0).alpha_max(0.01);
    let layers = [
        (Face::XMin, p),
        (Face::XMax, Cpml::cells(6)),
        (Face::YMin, p),
        (Face::YMax, p),
        (Face::ZMax, Cpml::cells(4)),
    ];
    let b = CpmlBoundary::new(&e, layers).unwrap();
    let probes = |e: &mut Engine| {
        [
            e.add_operator(Probe::voltage(Axis::Y, [2, 7, 6])),
            e.add_operator(Probe::current(Axis::Z, [9, 13, 6])),
        ]
    };
    let hs = if probes_first {
        let hs = probes(&mut e);
        e.add_operator(b);
        hs
    } else {
        e.add_operator(b);
        probes(&mut e)
    };
    e.add_operator(EdgeSource::new(dgauss(20e-12)).edge(Axis::Y, [9, 7, 6], 1.0));
    e.run(150);
    let w = e.step_with_energy();
    e.run(149);
    let mut out: Vec<Vec<f32>> = Axis::ALL
        .iter()
        .flat_map(|&a| [e.u_component(a), e.i_component(a)])
        .collect();
    for h in hs {
        out.push(e.operator(&h).values().iter().map(|&v| v as f32).collect());
    }
    out.push(vec![w as f32, (w - w as f32 as f64) as f32]);
    out
}

/// The parallel CPML pass gives bit-identical fields, probe samples and
/// energy for any thread count (including partitions that cut the x-normal
/// layers), and serial hooks see corrected values whatever the
/// registration order.
#[test]
fn deterministic_across_threads() {
    let a = det_run(1, false);
    assert!(a.iter().all(|v| v.iter().any(|&x| x != 0.0)));
    for (threads, first) in [(2, false), (4, true), (7, false), (1, true)] {
        let b = det_run(threads, first);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.len(), y.len());
            assert!(
                x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
                "{threads} threads (probes first: {first}) differ"
            );
        }
    }
}
