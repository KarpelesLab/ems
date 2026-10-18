//! Total-field / scattered-field plane-wave source (`engine::PlaneWave`).
//!
//! * Leakage: an empty TF box in a CPML-terminated cube; the scattered-field
//!   region must stay (nearly) empty.
//! * A PEC plate and a dielectric slab in the TF region of a quasi-1-D line:
//!   the scattered field outside the box is the reflected wave.

mod common;

use common::{Timer, db, discrete_reflection, sine_pulse, sine_pulse_duration, spectrum, yee_k};
use ems::constants::C0;
use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::{
    CellMaterials, Engine, FaceCondition, FaceConditions, Grid, Material, PlaneWave, PlaneWaveSpec,
};
use ems::{Axis, Complex64, Cpml, Face};

/// Cube of `n` 1 mm cells with an 8-cell CPML on all faces and a TF box of
/// nodes `lo..=hi`; returns the worst scattered-field E over the run
/// relative to the incident peak, in dB.
fn leakage(n: usize, lo: usize, hi: usize, dir: [f64; 3], pol: [f64; 3]) -> f64 {
    leakage_on(Grid::uniform([n; 3], [1e-3; 3]).unwrap(), lo, hi, dir, pol)
}

/// [`leakage`] on a cubic grid of the same node count (`n + 1` per axis).
fn leakage_on(grid: Grid, lo: usize, hi: usize, dir: [f64; 3], pol: [f64; 3]) -> f64 {
    let n = grid.cells()[0];
    let mut e = Engine::builder(grid.clone()).threads(1).build().unwrap();
    e.add_operator(CpmlBoundary::new(&e, Face::ALL.map(|f| (f, Cpml::cells(8)))).unwrap());
    let (f0, bw) = (8e9, 4e9);
    let spec = PlaneWaveSpec::new([lo; 3], [hi; 3], dir, pol).with_match_frequency(f0);
    let pw = e.add_operator(PlaneWave::new(&e, &spec, sine_pulse(f0, bw)).unwrap());
    let steps = ((sine_pulse_duration(bw) + 3.0 * n as f64 * 1e-3 / C0) / e.dt()) as u64;
    // Scattered-field E edges between the box and the CPML (one cell gap
    // to the PML), as (axis, flat index).
    let (p0, p1) = (9, n - 9);
    let tf = |a: usize, p: [usize; 3]| {
        (0..3).all(|t| {
            if t == a {
                (lo..hi).contains(&p[t])
            } else {
                (lo..=hi).contains(&p[t])
            }
        })
    };
    let mut sf = Vec::new();
    for a in 0..3 {
        for i in p0..p1 {
            for j in p0..p1 {
                for k in p0..p1 {
                    let p = [i, j, k];
                    if !tf(a, p) {
                        let len = grid.delta(Axis::ALL[a], p[a]);
                        sf.push((a, grid.index(p), len));
                    }
                }
            }
        }
    }
    let amp = pol.iter().map(|x| x * x).sum::<f64>().sqrt();
    let (mut worst, mut inc_peak) = (0.0f64, 0.0f64);
    let centre = Axis::ALL.map(|a| grid.dual_line(a, (lo + hi) / 2));
    for _ in 0..steps / 2 {
        e.run(2);
        let u = Axis::ALL.map(|a| e.u_component(a));
        for &(a, idx, len) in &sf {
            worst = worst.max((u[a][idx] as f64 / len).abs());
        }
        let inc = e.operator(&pw).incident_e(centre).unwrap();
        inc_peak = inc_peak.max(inc.iter().map(|x| x * x).sum::<f64>().sqrt());
    }
    // The pulse peaks at about 0.87 (sine carrier under the envelope).
    assert!(
        inc_peak / amp > 0.8 && inc_peak / amp <= 1.0,
        "incident peak {inc_peak}"
    );
    db(worst / inc_peak)
}

#[test]
fn normal_incidence_does_not_leak() {
    let _t = Timer::start("normal_incidence_does_not_leak");
    let l = leakage(36, 13, 23, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    eprintln!("normal incidence leakage: {l:.1} dB");
    assert!(l < -80.0, "{l:.1} dB");
    let l = leakage(36, 13, 23, [0.0, 0.0, -1.0], [0.0, 2.0, 0.0]);
    eprintln!("normal incidence (−z, E_y = 2) leakage: {l:.1} dB");
    assert!(l < -80.0, "{l:.1} dB");
}

#[test]
fn oblique_incidence_leakage() {
    let _t = Timer::start("oblique_incidence_leakage");
    let (s, c) = (30f64.to_radians().sin(), 30f64.to_radians().cos());
    let l30 = leakage(36, 13, 23, [c, s, 0.0], [0.0, 0.0, 1.0]);
    let l45 = leakage(36, 13, 23, [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]);
    let l3d = leakage(36, 13, 23, [1.0, 1.0, 1.0], [1.0, -1.0, 0.0]);
    eprintln!("oblique leakage: 30° (TE) {l30:.1} dB, 45° (TM) {l45:.1} dB, (1,1,1) {l3d:.1} dB");
    // Diagonals of a cubic mesh are matched exactly (all samples sit
    // half-way between auxiliary nodes, so the interpolation is a uniform
    // filter of an exact discrete solution).
    assert!(l45 < -100.0 && l3d < -100.0);
    assert!(l30 < -60.0);
}

#[test]
fn graded_mesh_leakage() {
    let _t = Timer::start("graded_mesh_leakage");
    // Cells from 0.8 to 1.2 mm (smoothly varying) on every axis.
    let lines: Vec<f64> = (0..=36)
        .scan(0.0, |x, i| {
            let v = *x;
            *x += 1e-3 * (1.0 + 0.2 * (i as f64 * 0.35).sin());
            Some(v)
        })
        .collect();
    let graded = Grid::new([lines.clone(), lines.clone(), lines]).unwrap();
    let (s, c) = (30f64.to_radians().sin(), 30f64.to_radians().cos());
    let lg0 = leakage_on(graded.clone(), 13, 23, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let lg30 = leakage_on(graded, 13, 23, [c, s, 0.0], [0.0, 0.0, 1.0]);
    eprintln!("graded mesh leakage: normal {lg0:.1} dB, 30° {lg30:.1} dB");
    assert!(lg0 < -45.0 && lg30 < -45.0);
}

/// Quasi-1-D line along x (one transverse cell, PEC y walls, PMC z walls,
/// E along y) of `n` cells of `dx`, CPML at both ends, TF box over nodes
/// `a..=b`. `cells` sets the material per x cell.
struct Line {
    dx: f64,
    n: usize,
    a: usize,
    b: usize,
}

impl Line {
    fn engine(&self, mats: impl Fn(&mut CellMaterials), pec_plane: Option<usize>) -> Engine {
        let x: Vec<f64> = (0..=self.n).map(|i| i as f64 * self.dx).collect();
        let g = Grid::new([x, vec![0.0, self.dx], vec![0.0, self.dx]]).unwrap();
        let faces = FaceConditions::all(FaceCondition::Pec)
            .with(Face::ZMin, FaceCondition::Pmc)
            .with(Face::ZMax, FaceCondition::Pmc);
        let mut m = CellMaterials::new(&g, Material::VACUUM);
        mats(&mut m);
        if let Some(p) = pec_plane {
            m.set_pec_edge(Axis::Y, [p, 0, 0]);
            m.set_pec_edge(Axis::Y, [p, 0, 1]);
        }
        let mut e = Engine::builder(g)
            .materials(m)
            .faces(faces)
            .threads(1)
            .build()
            .unwrap();
        e.add_operator(
            CpmlBoundary::new(
                &e,
                [(Face::XMin, Cpml::cells(10)), (Face::XMax, Cpml::cells(10))],
            )
            .unwrap(),
        );
        e
    }

    fn spec(&self) -> PlaneWaveSpec {
        PlaneWaveSpec::new(
            [self.a, 0, 0],
            [self.b, 1, 1],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )
    }

    fn pos(&self, i: usize) -> [f64; 3] {
        [i as f64 * self.dx, 0.5 * self.dx, 0.0]
    }
}

#[test]
fn pec_plate_reflects_the_incident_wave() {
    let _t = Timer::start("pec_plate_reflects_the_incident_wave");
    let line = Line {
        dx: 1e-3,
        n: 160,
        a: 40,
        b: 120,
    };
    let plate = 80;
    let mut e = line.engine(|_| {}, Some(plate));
    let (f0, bw) = (8e9, 4e9);
    let pw = e.add_operator(PlaneWave::new(&e, &line.spec(), sine_pulse(f0, bw)).unwrap());
    let (xs, xb, xt) = (37, 125, 100); // SF before, SF after, TF shadow
    let steps = ((sine_pulse_duration(bw) + 0.3 / C0) / e.dt()) as u64;
    let (mut s, mut inc) = (Vec::new(), Vec::new());
    let (mut shadow, mut after, mut peak) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..steps {
        e.step();
        let op = e.operator(&pw);
        let ein = |i: usize| op.incident_e(line.pos(i)).unwrap()[1];
        let u = |i: usize| e.u(Axis::Y, [i, 0, 0]) as f64 / line.dx;
        s.push(u(xs));
        // The auxiliary grid starts just before the box: record the
        // incident wave at the box face and shift it to xs below.
        inc.push(ein(line.a));
        peak = peak.max(ein(xt).abs());
        shadow = shadow.max(u(xt).abs());
        // Behind the plate the total field is 0: scattered = −incident.
        after = after.max((u(xb) + ein(xb)).abs());
    }
    let (shadow, after) = (db(shadow / peak), db(after / peak));
    let dt = e.dt();
    let freqs: Vec<f64> = (0..=10).map(|q| 4e9 + q as f64 * 0.8e9).collect();
    let (ss, si) = (spectrum(&s, dt, &freqs), spectrum(&inc, dt, &freqs));
    let mut worst = 0.0f64;
    for (q, &f) in freqs.iter().enumerate() {
        // Reflected = −incident delayed by the round trip xs → plate → xs.
        let k = yee_k(f, dt, line.dx, 1.0, 0.0).re;
        let want = -Complex64::cis(-2.0 * k * (plate - xs) as f64);
        let inc_xs = si[q] * Complex64::cis(k * (line.a - xs) as f64);
        worst = worst.max((ss[q] / inc_xs - want).norm());
    }
    eprintln!(
        "PEC plate: |R − R_yee| ≤ {worst:.2e}, shadow {shadow:.1} dB, \
         scattered + incident behind the box {after:.1} dB"
    );
    assert!(worst < 1e-4, "{worst}");
    // Both limited by the CPML reflection of the outgoing −incident wave.
    assert!(shadow < -80.0 && after < -80.0);
}

#[test]
fn dielectric_slab_reflection() {
    let _t = Timer::start("dielectric_slab_reflection");
    let line = Line {
        dx: 0.5e-3,
        n: 240,
        a: 40,
        b: 200,
    };
    let (s0, s1, eps) = (100, 120, 4.0); // 10 mm slab
    let mut e = line.engine(
        |m| m.fill_box([s0..s1, 0..1, 0..1], Material::dielectric(eps)),
        None,
    );
    let (f0, bw) = (6e9, 3e9);
    let pw = e.add_operator(PlaneWave::new(&e, &line.spec(), sine_pulse(f0, bw)).unwrap());
    let xs = 30;
    let steps = ((sine_pulse_duration(bw) + 0.4 / C0) / e.dt()) as u64;
    let (mut s, mut inc) = (Vec::new(), Vec::new());
    for _ in 0..steps {
        e.step();
        s.push(e.u(Axis::Y, [xs, 0, 0]) as f64 / line.dx);
        inc.push(e.operator(&pw).incident_e(line.pos(line.a)).unwrap()[1]);
    }
    let dt = e.dt();
    let freqs: Vec<f64> = (0..=10).map(|q| 3e9 + q as f64 * 0.6e9).collect();
    let (ss, si) = (spectrum(&s, dt, &freqs), spectrum(&inc, dt, &freqs));
    let (mut worst_d, mut worst_c) = (0.0f64, 0.0f64);
    // Cells from xs − 2 to past the slab, for the discrete reference.
    let cells: Vec<f64> = (xs - 2..s1 + 4)
        .map(|c| if (s0..s1).contains(&c) { eps } else { 1.0 })
        .collect();
    for (q, &f) in freqs.iter().enumerate() {
        let k = yee_k(f, dt, line.dx, 1.0, 0.0).re;
        // Measured reflection referred to node xs − 2 (the start of `cells`).
        let inc_xs = si[q] * Complex64::cis(k * (line.a - xs) as f64);
        let r = ss[q] / inc_xs * Complex64::cis(-2.0 * k * 2.0);
        let rd = discrete_reflection(&cells, line.dx, dt, f);
        worst_d = worst_d.max((r - rd).norm());
        // Continuum: magnitude only (the Yee phase differs slightly).
        let rc = common::slab_reflection(f, eps, (s1 - s0) as f64 * line.dx);
        worst_c = worst_c.max((r.norm() - rc.norm()).abs());
    }
    eprintln!(
        "dielectric slab: |R − R_yee| ≤ {worst_d:.2e}, ||R| − |R_continuum|| ≤ {worst_c:.2e}"
    );
    assert!(worst_d < 1e-4, "{worst_d}");
    assert!(worst_c < 0.01, "{worst_c}");
}

#[test]
fn invalid_specs_are_rejected() {
    let grid = Grid::uniform([10; 3], [1e-3; 3]).unwrap();
    let e = Engine::builder(grid).threads(1).build().unwrap();
    let ok = |s: PlaneWaveSpec| PlaneWave::new(&e, &s, |_| 0.0).is_ok();
    let base = PlaneWaveSpec::new([2; 3], [8; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(ok(base.clone()));
    let mut s = base.clone();
    s.polarization = [1.0, 1.0, 0.0];
    assert!(!ok(s), "polarisation along k");
    let mut s = base.clone();
    s.stop = [8, 2, 8];
    assert!(!ok(s), "flat box");
    let mut s = base.clone();
    s.stop = [8, 8, 11];
    assert!(!ok(s), "outside the mesh");
    let mut s = base.clone();
    s.direction = [0.0; 3];
    assert!(!ok(s), "no direction");
    assert!(!ok(base.clone().with_background(-1.0, 1.0)));
    assert!(
        !ok(base.with_match_frequency(1e14)),
        "beyond the mesh resolution"
    );
}

/// Oblique incidence on a graded mesh gives the same fields for any thread
/// count.
#[test]
fn deterministic_across_threads() {
    let lines: Vec<f64> = (0..=24)
        .scan(0.0, |x, i| {
            let v = *x;
            *x += 1e-3 * (1.0 + 0.2 * (i as f64 * 0.5).sin());
            Some(v)
        })
        .collect();
    let grid = Grid::new([lines.clone(), lines.clone(), lines]).unwrap();
    let run = |threads: usize| {
        let mut e = Engine::builder(grid.clone())
            .threads(threads)
            .build()
            .unwrap();
        let spec = PlaneWaveSpec::new([6; 3], [18; 3], [1.0, 0.5, 0.2], [0.0, -0.4, 1.0]);
        e.add_operator(PlaneWave::new(&e, &spec, sine_pulse(10e9, 5e9)).unwrap());
        e.run(150);
        Axis::ALL.map(|a| (e.u_component(a), e.i_component(a)))
    };
    let one = run(1);
    assert!(one[2].0.iter().any(|&v| v != 0.0));
    assert!(run(4) == one);
}

/// A csx model with a type-10 (plane wave) excitation, run through
/// [`ems::Simulation`].
mod csx_model {
    use csx::primitives::BoxShape;
    use csx::properties::{Excitation, ExcitationType, ProbeBox, ProbeType};
    use csx::{ContinuousStructure, Primitive, Property, Scalar};
    use ems::{Boundaries, Boundary, Fdtd, Waveform};

    pub fn model() -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        for a in 0..3 {
            m.grid.set_lines(a, (0..=36).map(f64::from)).unwrap();
        }
        let mut ex = Excitation::new([0.0, 0.0, 1.0]).with_type(ExcitationType::PlaneWave);
        ex.prop_dir = [1.0, 0.0, 0.0].map(Scalar::new);
        let id = m.add_property(Property::excitation("pw", ex));
        m.add_primitive(id, Primitive::new(BoxShape::new([12.0; 3], [24.0; 3])))
            .unwrap();
        for (name, x) in [("before", 10.0), ("centre", 18.0), ("after", 27.0)] {
            let p = m.add_property(Property::probe(name, ProbeBox::new(ProbeType::EField)));
            m.add_primitive(
                p,
                Primitive::new(BoxShape::new([x, 18.0, 18.0], [x, 18.0, 18.0])),
            )
            .unwrap();
        }
        m
    }

    pub fn fdtd() -> Fdtd {
        Fdtd::new()
            .waveform(Waveform::gaussian(8e9, 4e9))
            .boundaries(Boundaries::all(Boundary::pml(8)))
            .max_timesteps(600)
    }
}

#[test]
fn csx_plane_wave_excitation() {
    let _t = Timer::start("csx_plane_wave_excitation");
    let m = csx_model::model();
    let fdtd = csx_model::fdtd();
    let mut sim = ems::Simulation::new(&m, &fdtd).unwrap();
    assert!(
        sim.warnings().iter().all(|w| !w.contains("\"pw\"")),
        "{:?}",
        sim.warnings()
    );
    let pw = sim.plane_wave("pw").expect("plane wave operator");
    assert_eq!(pw.spec().start, [12; 3]);
    assert_eq!(pw.spec().stop, [24; 3]);
    assert_eq!(pw.spacing(), 1e-3);
    let res = sim.run().unwrap();
    let series = |n: &str| {
        let p = res.probe(n).unwrap();
        let v: Vec<f64> = p.vector().unwrap().iter().map(|v| v[2]).collect();
        (p.times().to_vec(), v)
    };
    let (t, centre) = series("centre");
    let wave = fdtd.get_waveform().unwrap();
    // Incident E_z = a(t − ξ/c), ξ = 6 mm from the entry face.
    let dt = res.dt();
    let err = t
        .iter()
        .zip(&centre)
        .map(|(&t, &v)| (v - wave.value(t - 6e-3 / C0, dt)).abs())
        .fold(0.0, f64::max);
    let leak = ["before", "after"]
        .map(|n| common::peak(&series(n).1))
        .into_iter()
        .fold(0.0, f64::max);
    let peak = common::peak(&centre);
    eprintln!(
        "csx plane wave: centre peak {peak:.4}, max |E − a(t − ξ/c)| {err:.2e}, \
         scattered-field probes {:.1} dB",
        db(leak / peak)
    );
    assert!((peak - 1.0).abs() < 0.02, "{peak}");
    assert!(err < 0.02, "{err}");
    assert!(db(leak / peak) < -80.0);
}

#[test]
fn csx_plane_wave_warnings() {
    use csx::Scalar;
    use csx::primitives::BoxShape;
    use csx::properties::{Excitation, ExcitationType};
    use csx::{Primitive, Property};

    let mut m = csx_model::model();
    // Excite partly along PropDir: the component is removed.
    let mut ex = Excitation::new([1.0, 0.0, 1.0]).with_type(ExcitationType::PlaneWave);
    ex.prop_dir = [1.0, 0.0, 0.0].map(Scalar::new);
    let id = m.add_property(Property::excitation("skew", ex.clone()));
    m.add_primitive(id, Primitive::new(BoxShape::new([14.0; 3], [20.0; 3])))
        .unwrap();
    // A flat box has no total-field volume.
    let id = m.add_property(Property::excitation("flat", ex.clone()));
    m.add_primitive(
        id,
        Primitive::new(BoxShape::new([14.0, 14.0, 14.0], [20.0, 20.0, 14.0])),
    )
    .unwrap();
    // No propagation direction.
    ex.prop_dir = [0.0; 3].map(Scalar::new);
    let id = m.add_property(Property::excitation("nodir", ex));
    m.add_primitive(id, Primitive::new(BoxShape::new([14.0; 3], [20.0; 3])))
        .unwrap();
    let sim = ems::Simulation::new(&m, &csx_model::fdtd().max_timesteps(1)).unwrap();
    let w = sim.warnings().join("\n");
    assert!(
        w.contains("\"skew\": excitation vector component along"),
        "{w}"
    );
    assert!(w.contains("\"flat\": primitive has no volume"), "{w}");
    assert!(w.contains("\"nodir\": no propagation direction"), "{w}");
    assert!(!w.contains("is not discretised"), "{w}");
    let skew = sim.plane_wave("skew").unwrap();
    assert_eq!(skew.spec().polarization, [0.0, 0.0, 1.0]);
    assert!(sim.plane_wave("flat").is_none() && sim.plane_wave("nodir").is_none());
}

#[cfg(feature = "xml")]
#[test]
fn xml_plane_wave_round_trip() {
    use csx::OpenEmsDocument;

    let model = csx_model::model();
    let fdtd = csx_model::fdtd().max_timesteps(400);
    let probe = |mut sim: ems::Simulation| {
        let r = sim.run().unwrap();
        r.probe("centre").unwrap().vector().unwrap().to_vec()
    };
    let direct = probe(ems::Simulation::new(&model, &fdtd).unwrap());
    let mut doc = OpenEmsDocument {
        structure: model,
        ..Default::default()
    };
    doc.set_fdtd_settings(&fdtd.to_csx());
    let xml = doc.to_xml_string();
    assert!(xml.contains("PropDir"), "{xml}");
    let sim = ems::Simulation::from_xml_str(&xml).unwrap();
    assert!(sim.plane_wave("pw").is_some(), "{:?}", sim.warnings());
    let back = probe(sim);
    let ez: Vec<f64> = back.iter().map(|v| v[2]).collect();
    assert!(common::peak(&ez) > 0.5);
    assert_eq!(back, direct);
}
