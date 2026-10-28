//! Plane waves (TF/SF) in layered and dispersive backgrounds: the spectral
//! incident field of `engine::PlaneWave` (design §5.11).
//!
//! * Leakage: an empty TF box crossing dielectric layers (a half-space, a
//!   slab, a grounded substrate, a graded mesh along the layers) in a
//!   CPML-terminated cube, at normal and oblique incidence, and in
//!   homogeneous dispersive (Debye, Drude, Lorentz) backgrounds; the
//!   scattered-field region must stay quiet.
//! * Physics: the field in the box against the discrete 1-D Yee response
//!   (normal incidence, exact), the continuum Fresnel coefficients
//!   (oblique), and the discrete dispersive propagation factor.

mod common;

use common::{
    Timer, db, discrete_response, sine_pulse, sine_pulse_duration, spectrum, yee_k_complex,
    yee_omega,
};
use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::dispersive::{Field, Scheme};
use ems::engine::{
    CellMaterials, Dispersive, Engine, Grid, LayerEnd, Material, PlaneWave, PlaneWaveSpec, Pole,
};
use ems::{Axis, Complex64, Cpml, Face};

/// One run: a cube of `n` cells (1 mm, or the given z lines), materials
/// per z cell, optional poles on all cells, a TF box `lo..=hi`.
struct Case {
    n: usize,
    z_lines: Option<Vec<f64>>,
    cell: Box<dyn Fn(usize) -> Material>,
    poles: Vec<Pole>,
    scheme: Scheme,
    zmin_pec: bool,
    lo: usize,
    hi: usize,
    dir: [f64; 3],
    pol: [f64; 3],
    f0: f64,
    /// Pulse bandwidth (default `f0/2`).
    bw: f64,
    layers: Option<[LayerEnd; 2]>,
    /// E edges recorded every step (axis, node).
    probes: Vec<(Axis, [usize; 3])>,
}

struct Out {
    /// Worst scattered-field E over the incident peak in the box, dB.
    leak: f64,
    probes: Vec<Vec<f64>>,
    /// `a(t)` at the probe times.
    a: Vec<f64>,
    dt: f64,
    grid: Grid,
}

impl Case {
    fn new(n: usize, lo: usize, hi: usize, dir: [f64; 3], pol: [f64; 3]) -> Self {
        Self {
            n,
            z_lines: None,
            cell: Box::new(|_| Material::VACUUM),
            poles: Vec::new(),
            scheme: Scheme::Trapezoidal,
            zmin_pec: false,
            lo,
            hi,
            dir,
            pol,
            f0: 8e9,
            bw: 4e9,
            layers: Some([LayerEnd::Open; 2]),
            probes: Vec::new(),
        }
    }

    fn run(&self) -> Out {
        self.try_run().unwrap()
    }

    fn try_run(&self) -> ems::Result<Out> {
        let n = self.n;
        let l: Vec<f64> = (0..=n).map(|i| i as f64 * 1e-3).collect();
        let z = self.z_lines.clone().unwrap_or_else(|| l.clone());
        let grid = Grid::new([l.clone(), l, z]).unwrap();
        let mut m = CellMaterials::new(&grid, Material::VACUUM);
        for k in 0..n {
            m.fill_box([0..n, 0..n, k..k + 1], (self.cell)(k));
        }
        let mut e = Engine::builder(grid.clone())
            .materials(m)
            .threads(1)
            .build()
            .unwrap();
        let faces: Vec<_> = Face::ALL
            .into_iter()
            .filter(|&f| !(self.zmin_pec && f == Face::ZMin))
            .map(|f| (f, Cpml::cells(8)))
            .collect();
        e.add_operator(CpmlBoundary::new(&e, faces).unwrap());
        let bw = self.bw;
        let mut spec = PlaneWaveSpec::new([self.lo; 3], [self.hi; 3], self.dir, self.pol)
            .with_match_frequency(self.f0)
            .with_duration(sine_pulse_duration(bw));
        if let Some([a, b]) = self.layers {
            spec = spec.with_layers(Axis::Z).with_layer_ends(a, b);
        }
        if !self.poles.is_empty() {
            let mut d = Dispersive::new().with_scheme(self.scheme);
            d.fill_cells(&grid, [0..n, 0..n, 0..n], Field::Electric, &self.poles)
                .unwrap();
            spec = spec.with_dispersion(d.entries(), d.scheme());
            d.add_to(&mut e).unwrap();
        }
        let t0 = std::time::Instant::now();
        let pw = PlaneWave::new(&e, &spec, sine_pulse(self.f0, bw))?;
        eprintln!("  set-up {:.2} s", t0.elapsed().as_secs_f64());
        assert!(pw.is_spectral());
        e.add_operator(pw);
        let slow = (self.cell)(0).eps_r[0]
            .max((self.cell)(n - 1).eps_r[0])
            .sqrt()
            * 3.0;
        let steps = ((sine_pulse_duration(bw) + slow * n as f64 * 1e-3 / 3e8) / e.dt()) as u64;
        let (p0, p1) = (9, n - 9);
        let (lo, hi) = (self.lo, self.hi);
        let tf = |a: usize, p: [usize; 3]| {
            (0..3).all(|t| {
                if t == a {
                    (lo..hi).contains(&p[t])
                } else {
                    (lo..=hi).contains(&p[t])
                }
            })
        };
        let k0 = if self.zmin_pec { 1 } else { p0 };
        let (mut sf, mut inside) = (Vec::new(), Vec::new());
        for a in 0..3 {
            for i in p0..p1 {
                for j in p0..p1 {
                    for k in k0..p1 {
                        let p = [i, j, k];
                        let len = grid.delta(Axis::ALL[a], p[a]);
                        if tf(a, p) {
                            inside.push((a, grid.index(p), len));
                        } else {
                            sf.push((a, grid.index(p), len));
                        }
                    }
                }
            }
        }
        let a = sine_pulse(self.f0, bw);
        let mut out = Out {
            leak: 0.0,
            probes: vec![Vec::new(); self.probes.len()],
            a: Vec::new(),
            dt: e.dt(),
            grid: grid.clone(),
        };
        let (mut worst, mut inc_peak) = (0.0f64, 0.0f64);
        for step in 0..steps {
            e.step();
            for (q, &(ax, p)) in self.probes.iter().enumerate() {
                out.probes[q].push(e.u(ax, p) as f64 / grid.delta(ax, p[ax.index()]));
            }
            out.a.push(a((step + 1) as f64 * e.dt()));
            if step % 2 == 1 {
                continue;
            }
            let u = Axis::ALL.map(|a| e.u_component(a));
            for &(a, idx, len) in &sf {
                worst = worst.max((u[a][idx] as f64 / len).abs());
            }
            for &(a, idx, len) in &inside {
                inc_peak = inc_peak.max((u[a][idx] as f64 / len).abs());
            }
        }
        out.leak = db(worst / inc_peak);
        Ok(out)
    }
}

fn layer(eps: f64, from: usize, to: usize) -> Box<dyn Fn(usize) -> Material> {
    Box::new(move |k| {
        if (from..to).contains(&k) {
            Material::dielectric(eps)
        } else {
            Material::VACUUM
        }
    })
}

fn deg(a: f64) -> (f64, f64) {
    let r = a.to_radians();
    (r.sin(), r.cos())
}

#[test]
fn half_space_leakage() {
    let _t = Timer::start("half_space_leakage");
    let (s, c) = deg(30.0);
    let mut out = Vec::new();
    for (name, dir, pol) in [
        ("normal", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ("30° TE", [s, 0.0, c], [0.0, 1.0, 0.0]),
        ("30° TM", [s, 0.0, c], [c, 0.0, -s]),
        ("(1,1,2) mixed", [1.0, 1.0, 2.0], [1.0, 1.0, -1.0]),
        ("−z 30° TM", [s, 0.0, -c], [c, 0.0, s]),
    ] {
        let mut case = Case::new(36, 13, 23, dir, pol);
        case.cell = layer(4.0, 18, 36);
        let l = case.run().leak;
        eprintln!("ε_r = 4 half-space across the box, {name}: {l:.1} dB");
        out.push(l);
    }
    assert!(out.iter().all(|&l| l < -100.0), "{out:?}");
}

#[test]
fn slab_grounded_and_graded_leakage() {
    let _t = Timer::start("slab_grounded_and_graded_leakage");
    let (s, c) = deg(40.0);
    // A slab crossing the box bottom face and ending inside it.
    let mut a = Case::new(36, 13, 23, [s, 0.0, c], [c, 0.0, -s]);
    a.cell = layer(6.0, 10, 16);
    let la = a.run().leak;
    // A grounded substrate (PEC z_min, ε_r = 3 up to z = 16 mm) under a
    // wave from above, the box crossing the substrate surface.
    let mut b = Case::new(36, 12, 24, [s, 0.0, -c], [0.0, 1.0, 0.0]);
    b.cell = layer(3.0, 0, 16);
    b.zmin_pec = true;
    b.layers = Some([LayerEnd::Pec, LayerEnd::Open]);
    let lb = b.run().leak;
    b.pol = [c, 0.0, s];
    let lb_tm = b.run().leak;
    // The half-space on a graded z mesh (cells 0.7 … 1.3 mm).
    let mut g = Case::new(36, 13, 23, [s, 0.0, c], [0.0, 1.0, 0.0]);
    g.z_lines = Some(
        (0..=36)
            .scan(0.0, |z, i| {
                let v = *z;
                *z += 1e-3 * (1.0 + 0.3 * (i as f64 * 0.5).sin());
                Some(v)
            })
            .collect(),
    );
    g.cell = layer(4.0, 18, 36);
    let lg = g.run().leak;
    eprintln!(
        "40°: slab across the box face {la:.1} dB, grounded substrate TE {lb:.1} dB / TM \
         {lb_tm:.1} dB, half-space on a graded z mesh {lg:.1} dB"
    );
    assert!(la < -100.0 && lb < -100.0 && lb_tm < -100.0 && lg < -100.0);
}

/// Normal incidence: the field in the box is the exact discrete 1-D Yee
/// response of the stack (independent transfer recursion), normalised to
/// the incident wave at the reference corner.
#[test]
fn normal_incidence_matches_the_discrete_stack() {
    let _t = Timer::start("normal_incidence_matches_the_discrete_stack");
    let mut case = Case::new(36, 13, 23, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    case.cell = layer(5.0, 17, 20);
    case.probes = (13..=23).map(|k| (Axis::X, [18, 18, k])).collect();
    let out = case.run();
    eprintln!("slab leakage {:.1} dB", out.leak);
    // Cells from the reference node (13) up; the stack ends in vacuum.
    let eps: Vec<Complex64> = (13..36).map(|k| (case.cell)(k).eps_r[0].into()).collect();
    let mu = vec![Complex64::ONE; eps.len()];
    let freqs = [5e9, 7e9, 9e9, 11e9];
    let sa = spectrum(&out.a, out.dt, &freqs);
    let mut worst = 0.0f64;
    for (q, pr) in out.probes.iter().enumerate() {
        let sp = spectrum(pr, out.dt, &freqs);
        for (i, &f) in freqs.iter().enumerate() {
            let (_, e) = discrete_response(&eps, &mu, 1e-3, out.dt, f);
            worst = worst.max((sp[i] / sa[i] - e[q]).norm());
        }
    }
    eprintln!("normal incidence on an ε_r = 5 slab: |E/a − E_discrete| ≤ {worst:.2e}");
    assert!(worst < 1e-4, "{worst}");
}

/// Oblique incidence on a half-space: the transmitted field against the
/// continuum Fresnel coefficients (second-order discretisation error).
#[test]
fn oblique_transmission_matches_fresnel() {
    let _t = Timer::start("oblique_transmission_matches_fresnel");
    let eps2 = 4.0f64;
    let (s, c) = deg(35.0);
    let f = 6e9;
    let mut worst = 0.0f64;
    for tm in [false, true] {
        let pol = if tm { [c, 0.0, -s] } else { [0.0, 1.0, 0.0] };
        let mut case = Case::new(36, 13, 23, [s, 0.0, c], pol);
        case.cell = layer(eps2, 18, 36);
        case.f0 = 6e9;
        // Tangential E (x or y) at a node 3 cells into the dielectric,
        // and at the reference node.
        let ax = if tm { Axis::X } else { Axis::Y };
        let mut p = [13, 13, 21];
        let mut r = [13, 13, 13];
        p[ax.index()] = 13;
        r[ax.index()] = 13;
        case.probes = vec![(ax, p), (ax, r)];
        let out = case.run();
        let sp = spectrum(&out.probes[0], out.dt, &[f])[0];
        let sa = spectrum(&out.a, out.dt, &[f])[0];
        // Continuum: t (tangential E ratio) and the phase path.
        let n2 = eps2.sqrt();
        let st = s / n2;
        let ct = (1.0 - st * st).sqrt();
        let (z1, z2) = if tm {
            (c, ct / n2)
        } else {
            (1.0 / c, 1.0 / (n2 * ct))
        };
        let t = 2.0 * z2 / (z1 + z2);
        let pol_t = if tm { c } else { 1.0 };
        let k0 = 2.0 * std::f64::consts::PI * f / 3e8;
        let g = &out.grid;
        let x = g.dual_line(Axis::X, 13) - g.lines(Axis::X)[13];
        let xp = if tm { x } else { 0.0 };
        let phase = k0 * (s * xp + c * 5e-3 + n2 * ct * 3e-3);
        let want = Complex64::cis(-phase) * (t * pol_t);
        let err = (sp / sa / want - 1.0).norm();
        eprintln!(
            "35° {}: E_t/a = {:.4} ∠{:.3}, Fresnel {:.4} ∠{:.3} (rel. error {err:.2e})",
            if tm { "TM" } else { "TE" },
            (sp / sa).norm(),
            (sp / sa).arg(),
            want.norm(),
            want.arg()
        );
        worst = worst.max(err);
    }
    assert!(worst < 2e-2, "{worst}");
}

/// Homogeneous dispersive backgrounds: no leakage, and the propagation
/// factor in the box is the discrete one of `ε∞ + χ_d(ω)`.
#[test]
fn dispersive_backgrounds() {
    let _t = Timer::start("dispersive_backgrounds");
    let w = 2.0 * std::f64::consts::PI;
    let media: [(&str, f64, Pole, Scheme); 4] = [
        (
            "Debye",
            2.0,
            Pole::debye(2.0, 1.0 / (w * 6e9)),
            Scheme::Trapezoidal,
        ),
        (
            "Drude",
            1.0,
            Pole::drude(w * 3e9, w * 0.5e9),
            Scheme::Trapezoidal,
        ),
        (
            "Lorentz",
            1.5,
            Pole::lorentz(w * 12e9, w * 25e9, w * 1e9),
            Scheme::Trapezoidal,
        ),
        (
            "Lorentz, leapfrog",
            1.5,
            Pole::lorentz(w * 12e9, w * 25e9, w * 1e9),
            Scheme::Leapfrog,
        ),
    ];
    let (s, c) = deg(30.0);
    let mut worst_leak = f64::NEG_INFINITY;
    let mut worst_k = 0.0f64;
    for (name, eps_inf, pole, scheme) in media {
        for (dir, pol) in [
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([c, s, 0.0], [0.0, 0.0, 1.0]),
        ] {
            let mut case = Case::new(36, 13, 23, dir, pol);
            case.cell = Box::new(move |_| Material::dielectric(eps_inf));
            case.poles = vec![pole];
            case.scheme = scheme;
            case.layers = None;
            // Well clear of the Drude cutoff (3 GHz): no slow components.
            case.f0 = 10e9;
            case.bw = 2e9;
            let normal = dir[1] == 0.0;
            if normal {
                case.probes = vec![(Axis::Z, [15, 18, 18]), (Axis::Z, [21, 18, 18])];
            }
            let out = case.run();
            worst_leak = worst_leak.max(out.leak);
            let mut msg = String::new();
            if normal {
                let freqs = [8e9, 10e9, 12e9];
                let s1 = spectrum(&out.probes[0], out.dt, &freqs);
                let s2 = spectrum(&out.probes[1], out.dt, &freqs);
                for (i, &f) in freqs.iter().enumerate() {
                    let chi = match scheme {
                        Scheme::Trapezoidal => pole.discrete_susceptibility(f, out.dt),
                        Scheme::Leapfrog => pole.leapfrog_susceptibility(f, out.dt),
                        _ => unreachable!("scheme {scheme:?}"),
                    };
                    let k =
                        yee_k_complex(yee_omega(f, out.dt), 1e-3, chi + eps_inf, Complex64::ONE);
                    let want = (Complex64::new(0.0, -6.0) * k).exp();
                    worst_k = worst_k.max((s2[i] / s1[i] - want).norm() / want.norm());
                }
                msg = format!(", propagation factor error ≤ {worst_k:.1e}");
            }
            eprintln!(
                "{name} background ({}): leakage {:.1} dB{msg}",
                if normal { "normal" } else { "30°" },
                out.leak
            );
        }
    }
    assert!(worst_leak < -100.0, "{worst_leak}");
    assert!(worst_k < 1e-4, "{worst_k}");
    // A pulse reaching down to the Drude cutoff has slow components that
    // arrive at the box before t = 0 and ring for long: refused rather
    // than leaking.
    let mut case = Case::new(20, 7, 13, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    case.cell = Box::new(|_| Material::VACUUM);
    case.poles = vec![Pole::drude(w * 4e9, w * 1e9)];
    case.layers = None;
    let err = case.try_run().err().expect("refused");
    let msg = err.to_string();
    assert!(
        msg.contains("before t = 0") || msg.contains("does not decay"),
        "{msg}"
    );
}

/// A dispersive (Drude) layer crossing the box.
#[test]
fn dispersive_layer_leakage() {
    let _t = Timer::start("dispersive_layer_leakage");
    let w = 2.0 * std::f64::consts::PI;
    let n = 36;
    let (s, c) = deg(25.0);
    let grid = Grid::uniform([n; 3], [1e-3; 3]).unwrap();
    let mut e = Engine::builder(grid.clone()).threads(1).build().unwrap();
    e.add_operator(CpmlBoundary::new(&e, Face::ALL.map(|f| (f, Cpml::cells(8)))).unwrap());
    let mut d = Dispersive::new();
    d.fill_cells(
        &grid,
        [0..n, 0..n, 17..21],
        Field::Electric,
        &[Pole::drude(w * 5e9, w * 2e9)],
    )
    .unwrap();
    let spec = PlaneWaveSpec::new([13; 3], [23; 3], [s, 0.0, c], [c, 0.0, -s])
        .with_match_frequency(8e9)
        .with_duration(sine_pulse_duration(4e9))
        .with_layers(Axis::Z)
        .with_dispersion(d.entries(), d.scheme());
    d.add_to(&mut e).unwrap();
    let pw = PlaneWave::new(&e, &spec, sine_pulse(8e9, 4e9)).unwrap();
    e.add_operator(pw);
    let (mut worst, mut peak) = (0.0f64, 0.0f64);
    for _ in 0..700 {
        e.step();
        for a in Axis::ALL {
            // Outside the box (x below it) and inside.
            worst = worst.max((e.u(a, [11, 18, 19]) as f64 / 1e-3).abs());
            worst = worst.max((e.u(a, [18, 18, 25]) as f64 / 1e-3).abs());
            peak = peak.max((e.u(a, [18, 18, 19]) as f64 / 1e-3).abs());
        }
    }
    let l = db(worst / peak);
    eprintln!("Drude layer across the box, 25° TM: {l:.1} dB");
    assert!(l < -100.0, "{l}");
}

/// A layered background without the layered declaration is rejected,
/// as is a wave entering through a closed end.
#[test]
fn layered_errors() {
    let grid = Grid::uniform([20; 3], [1e-3; 3]).unwrap();
    let mut m = CellMaterials::new(&grid, Material::VACUUM);
    m.fill_box([0..20, 0..20, 10..20], Material::dielectric(4.0));
    let e = Engine::builder(grid)
        .materials(m)
        .threads(1)
        .build()
        .unwrap();
    let spec =
        PlaneWaveSpec::new([5; 3], [15; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).with_duration(1e-9);
    assert!(PlaneWave::new(&e, &spec, |_| 0.0).is_err());
    // Stratified along x is not what the medium does.
    let err = PlaneWave::new(&e, &spec.clone().with_layers(Axis::X), |_| 0.0).unwrap_err();
    assert!(err.to_string().contains("layered"), "{err}");
    let ok = PlaneWave::new(&e, &spec.clone().with_layers(Axis::Z), |t| {
        (-((t - 2e-10) / 5e-11f64).powi(2)).exp()
    });
    assert!(ok.is_ok(), "{:?}", ok.err());
    let closed = spec
        .with_layers(Axis::Z)
        .with_layer_ends(LayerEnd::Pec, LayerEnd::Open);
    let err = PlaneWave::new(&e, &closed, |_| 0.0).unwrap_err();
    assert!(err.to_string().contains("closed"), "{err}");
}

/// Through `Simulation`: a csx model whose plane-wave box crosses a
/// substrate gets the layered incident field natively; an openEMS file
/// (openEMS semantics) keeps rejecting it.
#[test]
fn csx_layered_background() {
    use csx::primitives::BoxShape;
    use csx::properties::{Excitation, ExcitationType, ProbeBox, ProbeType};
    use csx::{ContinuousStructure, Primitive, Property, Scalar};
    use ems::{Boundaries, Boundary, Fdtd, Waveform};

    let _t = Timer::start("csx_layered_background");
    let (s, c) = deg(30.0);
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for a in 0..3 {
        m.grid.set_lines(a, (0..=36).map(f64::from)).unwrap();
    }
    let sub = m.add_property(Property::material("sub", csx::Material::new(4.0, 0.0)));
    m.add_primitive(
        sub,
        Primitive::new(BoxShape::new([-1.0, -1.0, 18.0], [37.0, 37.0, 37.0])),
    )
    .unwrap();
    let mut ex = Excitation::new([c, 0.0, -s]).with_type(ExcitationType::PlaneWave);
    ex.prop_dir = [s, 0.0, c].map(Scalar::new);
    let id = m.add_property(Property::excitation("pw", ex));
    m.add_primitive(id, Primitive::new(BoxShape::new([12.0; 3], [24.0; 3])))
        .unwrap();
    for (name, p) in [
        ("before", [10.0, 18.0, 15.0]),
        ("above", [18.0, 18.0, 27.0]),
        ("inside", [18.0, 18.0, 20.0]),
    ] {
        let id = m.add_property(Property::probe(name, ProbeBox::new(ProbeType::EField)));
        m.add_primitive(id, Primitive::new(BoxShape::new(p, p)))
            .unwrap();
    }
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(8e9, 4e9))
        .boundaries(Boundaries::all(Boundary::pml(8)))
        .max_timesteps(900);
    let mut sim = ems::Simulation::new(&m, &fdtd).unwrap();
    let pw = sim.plane_wave("pw").expect("installed");
    assert!(
        pw.is_spectral() && pw.spec().layers == Some(Axis::Z),
        "{:?}",
        pw.spec()
    );
    let res = sim.run().unwrap();
    let peak = |n: &str| {
        res.probe(n)
            .unwrap()
            .vector()
            .unwrap()
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt())
            .fold(0.0, f64::max)
    };
    let inside = peak("inside");
    let leak = peak("before").max(peak("above"));
    eprintln!(
        "csx layered: inside {inside:.3}, scattered-field probes {:.1} dB",
        db(leak / inside)
    );
    assert!(inside > 0.3 && db(leak / inside) < -100.0);
    // openEMS semantics: the inhomogeneous surface is an error, as before.
    let err = ems::Simulation::new(&m, &fdtd.file_equivalent())
        .unwrap_err()
        .to_string();
    assert!(err.contains("not homogeneous"), "{err}");
}
