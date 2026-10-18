//! Cylindrical PML on the ρ_max face (design §5.15.2): reflection of
//! outgoing cylindrical waves against a much larger reference domain,
//! compared with the Cartesian-style layer; long-run stability; thread
//! determinism with z-face CPML corners.

use std::f64::consts::{PI, TAU};

use ems::engine::{
    EdgeSource, Engine, FaceCondition, FaceConditions, Grid, Probe, boundary::CpmlBoundary,
};
use ems::{Axis, Cpml, Face};

/// Gaussian-modulated sine centred at `f0` with ~`bw` bandwidth.
fn pulse(f0: f64, bw: f64) -> impl Fn(f64) -> f64 + Send + 'static {
    let tau = 1.0 / (PI * bw);
    let t0 = 4.0 * tau;
    move |t: f64| {
        let x = (t - t0) / tau;
        (-x * x).exp() * (2.0 * PI * f0 * (t - t0)).sin()
    }
}

fn lines(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..=n)
        .map(|i| lo + (hi - lo) * i as f64 / n as f64)
        .collect()
}

/// Polarisation of the z-invariant wave between the z plates.
#[derive(Clone, Copy, PartialEq)]
enum Pol {
    /// PEC plates: E_z, H_ρ, H_α.
    Tm,
    /// PMC plates: H_z, E_ρ, E_α.
    Te,
}

/// Layer on ρ_max.
#[derive(Clone, Copy)]
enum Layer {
    Cylindrical(Cpml),
    Plane(Cpml),
}

const DR: f64 = 0.5e-3;
const R0: f64 = 2e-3;
const NA: usize = 16;

/// Annulus `R0 … R0 + nr·DR` (PEC inner wall), closed azimuth of `NA`
/// cells, one 1 mm z cell between PEC (TM) or PMC (TE) plates. Sources
/// off-centre in α near the inner wall (so all azimuthal orders up to the
/// mesh limit are excited); returns the voltages of three probe edges at
/// ρ index `probe`, concatenated, over `t_end`.
fn radial(
    nr: usize,
    probe: usize,
    pol: Pol,
    layer: Option<Layer>,
    t_end: f64,
    threads: usize,
) -> Vec<f64> {
    let grid = Grid::cylindrical([
        lines(R0, R0 + nr as f64 * DR, nr),
        lines(0.0, TAU, NA),
        lines(0.0, 1e-3, 1),
    ])
    .unwrap();
    let mut faces = FaceConditions::all(FaceCondition::Pec);
    if pol == Pol::Te {
        faces[Face::ZMin] = FaceCondition::Pmc;
        faces[Face::ZMax] = FaceCondition::Pmc;
    }
    let mut e = Engine::builder(grid)
        .faces(faces)
        .threads(threads)
        .build()
        .unwrap();
    match layer {
        Some(Layer::Cylindrical(p)) => {
            let c = CpmlBoundary::new(&e, [(Face::XMax, p)]).unwrap();
            assert!(c.is_cylindrical_rho_max());
            e.add_operator(c);
        }
        Some(Layer::Plane(p)) => {
            let c = CpmlBoundary::new(&e, [(Face::XMax, p)])
                .unwrap()
                .plane_rho_max();
            e.add_operator(c);
        }
        None => {}
    }
    let src = match pol {
        Pol::Tm => EdgeSource::new(pulse(15e9, 10e9))
            .edge(Axis::Z, [3, 0, 0], 1.0)
            .edge(Axis::Z, [3, 0, 1], 1.0)
            .edge(Axis::Z, [5, 3, 0], 0.5)
            .edge(Axis::Z, [5, 3, 1], 0.5),
        Pol::Te => EdgeSource::new(pulse(15e9, 10e9))
            .edge(Axis::Y, [3, 0, 0], 1.0)
            .edge(Axis::Y, [3, 0, 1], 1.0)
            .edge(Axis::X, [4, 3, 0], 0.5)
            .edge(Axis::X, [4, 3, 1], 0.5),
    };
    e.add_operator(src);
    let probes: Vec<_> = match pol {
        Pol::Tm => vec![(Axis::Z, 0), (Axis::Z, 5), (Axis::Z, 11)],
        Pol::Te => vec![(Axis::Y, 0), (Axis::Y, 5), (Axis::X, 11)],
    }
    .into_iter()
    .map(|(a, j)| e.add_operator(Probe::voltage(a, [probe, j, 0])))
    .collect();
    e.run((t_end / e.dt()).ceil() as u64);
    probes
        .iter()
        .flat_map(|h| e.operator(h).values().to_vec())
        .collect()
}

/// Largest |difference| over the common length, relative to the largest
/// |reference| value, in dB.
fn rel_db(test: &[f64], reference: &[f64]) -> f64 {
    let n = test.len().min(reference.len());
    let d = (0..n).fold(0.0f64, |m, q| m.max((test[q] - reference[q]).abs()));
    let r = reference[..n].iter().fold(0.0f64, |m, x| m.max(x.abs()));
    20.0 * (d / r).log10()
}

/// Reflection of the ρ_max layer `p` (interface at ρ index `inner`,
/// probe at `probe`) against a reference 560 cells larger, from which
/// nothing returns to the probe within 1 ns: `(cylindrical, plane)` in dB.
fn reflection(pol: Pol, inner: usize, probe: usize, p: Cpml) -> (f64, f64) {
    let t_end = 1.0e-9;
    let reference = radial(inner + 560, probe, pol, None, t_end, 2);
    let nr = inner + p.cells;
    let new = radial(nr, probe, pol, Some(Layer::Cylindrical(p)), t_end, 2);
    let old = radial(nr, probe, pol, Some(Layer::Plane(p)), t_end, 2);
    (rel_db(&new, &reference), rel_db(&old, &reference))
}

/// Layer from ρ = 58 mm (index 112), probe at 47 mm (index 90), 5–25 GHz,
/// both polarisations, all azimuthal orders the 16 α cells resolve.
#[test]
fn cylindrical_pml_absorbs_outgoing_waves() {
    for pol in [Pol::Tm, Pol::Te] {
        let name = if pol == Pol::Tm { "TM" } else { "TE" };
        let (new, old) = reflection(pol, 112, 90, Cpml::cells(8));
        println!("{name}: ρ_max reflection, PML(8): cylindrical {new:.1} dB, plane {old:.1} dB");
        assert!(new < -80.0, "{name}: {new:.1} dB");
        assert!(old > -45.0, "{name}: plane layer {old:.1} dB");
    }
}

#[test]
fn reflection_vs_thickness_and_radius() {
    // Thin layers are limited by the round trip through the PEC-backed
    // layer, exp(−1.6·n) at σ_max = 0.8·(m+1)/(Z0·Δ): −56 dB for n = 4.
    let mut last = 0.0;
    for n in [4, 8, 16] {
        let (new, old) = reflection(Pol::Tm, 112, 90, Cpml::cells(n));
        println!("TM, ρ0 = 58 mm, PML({n}): cylindrical {new:.1} dB, plane {old:.1} dB");
        assert!(new < last - 10.0, "PML({n}): {new:.1} dB");
        last = new;
    }
    // Small radius (layer from 26 mm, probe at 16 mm): the plane layer
    // gets worse as 1/(kρ), the cylindrical one stays far below.
    let (new, old) = reflection(Pol::Te, 48, 28, Cpml::cells(8));
    println!("TE, ρ0 = 26 mm, PML(8): cylindrical {new:.1} dB, plane {old:.1} dB");
    assert!(new < -70.0 && old > -35.0, "{new:.1} / {old:.1} dB");
}

/// Field energy of the annulus (40 + 8 cells), sampled every 100 steps
/// for the first 2000 steps and then every 500 steps up to `steps`, with a zero-mean pulse or (`dc`) a Gaussian that leaves
/// static fields behind (the classic trigger of late-time growth).
fn long_run(pol: Pol, p: Cpml, dc: bool, steps: u64) -> Vec<f64> {
    let grid = Grid::cylindrical([
        lines(R0, R0 + 48.0 * DR, 48),
        lines(0.0, TAU, NA),
        lines(0.0, 1e-3, 1),
    ])
    .unwrap();
    let mut faces = FaceConditions::all(FaceCondition::Pec);
    if pol == Pol::Te {
        faces[Face::ZMin] = FaceCondition::Pmc;
        faces[Face::ZMax] = FaceCondition::Pmc;
    }
    let mut e = Engine::builder(grid)
        .faces(faces)
        .threads(1)
        .build()
        .unwrap();
    e.add_operator(CpmlBoundary::new(&e, [(Face::XMax, p)]).unwrap());
    let gauss = |t: f64| {
        let x = (t - 60e-12) / 20e-12;
        (-x * x).exp()
    };
    let src = if dc {
        EdgeSource::new(gauss)
    } else {
        EdgeSource::new(pulse(15e9, 10e9))
    };
    let axis = if pol == Pol::Tm { Axis::Z } else { Axis::Y };
    e.add_operator(src.edge(axis, [5, 2, 0], 1.0).edge(Axis::X, [7, 9, 0], 1.0));
    let mut w = Vec::new();
    while e.timestep() < steps {
        e.run(if e.timestep() < 2000 { 100 } else { 500 });
        w.push(e.energy_staggered());
    }
    w
}

#[test]
fn energy_decays_and_long_run_is_stable() {
    for (pol, p) in [
        (Pol::Tm, Cpml::cells(8)),
        (Pol::Te, Cpml::cells(8)),
        (Pol::Tm, Cpml::cells(8).kappa_max(4.0).alpha_max(0.05)),
    ] {
        // Zero-mean pulse: everything leaves through the layer (slowly: a
        // 2-D wave has a long wake) down to the static remainder of the
        // sampled soft source, and does not come back.
        let w = long_run(pol, p, false, 16_000);
        let peak = w.iter().fold(0.0f64, |m, &x| m.max(x));
        let last = *w.last().unwrap();
        let mid = 20 + 6000 / 500; // step 8000
        let tail = w[mid..].iter().fold(0.0f64, |m, &x| m.max(x));
        println!(
            "pulse: energy peak {peak:.3e}, after 8k steps ≤ {:.1} dB, after 16k {:.1} dB",
            10.0 * (tail / peak).log10(),
            10.0 * (last / peak).log10()
        );
        assert!(last < 1e-5 * peak, "{w:?}");
        // No late growth: the last quarter stays below the third.
        let max = |v: &[f64]| v.iter().fold(0.0f64, |m, &x| m.max(x));
        let q = w.len() - (w.len() - mid) / 2;
        assert!(max(&w[q..]) <= max(&w[mid..q]), "{w:?}");
        // Static remainder: settles (a slowly damped ripple of ~1e-5),
        // never grows.
        let w = long_run(pol, p, true, 12_000);
        let spread = |v: &[f64]| {
            let (lo, hi) = v
                .iter()
                .fold((f64::MAX, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
            (hi - lo) / hi
        };
        let w = &w[20..]; // from step 2000
        let n = w.len();
        let (early, late) = (spread(&w[..n / 2]), spread(&w[n / 2..]));
        println!(
            "DC: static energy {:.6e}, relative spread {early:.1e} then {late:.1e}",
            w[n - 1]
        );
        assert!(w.iter().all(|x| x.is_finite()));
        assert!(late < early && spread(w) < 1e-3, "{w:?}");
    }
}

/// A 3-D annulus with the cylindrical ρ_max layer and CPML on both z faces
/// (the corners use both stretches); probe values over `steps`.
fn corner_run(threads: usize, steps: u64) -> (Vec<f64>, f64) {
    let grid = Grid::cylindrical([
        lines(R0, R0 + 20e-3, 20),
        lines(0.0, TAU, 12),
        lines(0.0, 24e-3, 24),
    ])
    .unwrap();
    let mut e = Engine::builder(grid).threads(threads).build().unwrap();
    let p = Cpml::cells(6);
    e.add_operator(
        CpmlBoundary::new(&e, [(Face::XMax, p), (Face::ZMin, p), (Face::ZMax, p)]).unwrap(),
    );
    e.add_operator(
        EdgeSource::new(pulse(10e9, 8e9))
            .edge(Axis::Z, [4, 0, 12], 1.0)
            .edge(Axis::X, [6, 3, 10], 1.0)
            .edge(Axis::Y, [8, 7, 14], 1.0),
    );
    let probes: Vec<_> = [
        (Axis::Z, [17, 1, 20]),
        (Axis::X, [16, 6, 21]),
        (Axis::Y, [18, 9, 3]),
    ]
    .into_iter()
    .map(|(a, n)| e.add_operator(Probe::voltage(a, n)))
    .collect();
    e.run(steps);
    let v = probes
        .iter()
        .flat_map(|h| e.operator(h).values().to_vec())
        .collect();
    (v, e.energy_staggered())
}

#[test]
fn corners_threads_bit_identical() {
    let (a, wa) = corner_run(1, 400);
    for t in [2, 3, 5] {
        let (b, wb) = corner_run(t, 400);
        assert_eq!(a, b, "{t} threads");
        assert_eq!(wa, wb);
    }
}

/// 3-D annulus `R0 … R0 + nr` mm (1 mm cells), 16 α cells, `nz` z cells
/// with the model shifted by `dz` cells, with PML(8) on ρ_max and both z
/// faces (`layer`) or PEC (reference). Probes near the ρ/z corners;
/// values over `t_end`.
fn box3d(nr: usize, nz: usize, dz: usize, layer: Option<Layer>, t_end: f64) -> Vec<f64> {
    let grid = Grid::cylindrical([
        lines(R0, R0 + nr as f64 * 1e-3, nr),
        lines(0.0, TAU, 16),
        lines(0.0, nz as f64 * 1e-3, nz),
    ])
    .unwrap();
    let mut e = Engine::builder(grid).threads(4).build().unwrap();
    let faces = |p: Cpml| [(Face::XMax, p), (Face::ZMin, p), (Face::ZMax, p)];
    match layer {
        Some(Layer::Cylindrical(p)) => {
            e.add_operator(CpmlBoundary::new(&e, faces(p)).unwrap());
        }
        Some(Layer::Plane(p)) => {
            e.add_operator(CpmlBoundary::new(&e, faces(p)).unwrap().plane_rho_max());
        }
        None => {}
    }
    e.add_operator(
        EdgeSource::new(pulse(8e9, 6e9))
            .edge(Axis::Z, [3, 0, dz + 20], 1.0)
            .edge(Axis::X, [4, 5, dz + 18], 1.0)
            .edge(Axis::Y, [4, 11, dz + 22], 1.0),
    );
    let probes: Vec<_> = [
        (Axis::Z, [26, 1, dz + 28]),
        (Axis::X, [25, 6, dz + 29]),
        (Axis::Y, [27, 9, dz + 11]),
        (Axis::Z, [20, 13, dz + 20]),
    ]
    .into_iter()
    .map(|(a, n)| e.add_operator(Probe::voltage(a, n)))
    .collect();
    e.run((t_end / e.dt()).ceil() as u64);
    probes
        .iter()
        .flat_map(|h| e.operator(h).values().to_vec())
        .collect()
}

#[test]
fn corner_reflection_3d() {
    // Nothing returns from the reference walls within 0.35 ns (105 mm). The
    // corners need the two-axis stretch: simply adding the z-face CPML
    // correction to the cylindrical one there is unstable.
    let t_end = 0.35e-9;
    let reference = box3d(100, 40 + 110, 55, None, t_end);
    let p = Cpml::cells(8);
    let new = rel_db(
        &box3d(40, 40, 0, Some(Layer::Cylindrical(p)), t_end),
        &reference,
    );
    let old = rel_db(&box3d(40, 40, 0, Some(Layer::Plane(p)), t_end), &reference);
    println!("3-D, PML(8) on ρ_max and z: cylindrical {new:.1} dB, plane {old:.1} dB");
    assert!(new < -65.0 && old > -40.0, "{new:.1} / {old:.1} dB");
}
