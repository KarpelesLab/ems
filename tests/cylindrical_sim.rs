//! Cylindrical csx models through [`Simulation`] (design §5.15): a circular
//! cavity built in `(ρ, α, z)` resonates at the Bessel-zero frequencies,
//! the openEMS XML form gives bit-identical results, and the
//! Cartesian-only features are refused or dropped with warnings.

use std::f64::consts::{PI, TAU};

use csx::primitives::BoxShape;
use csx::properties::{Excitation, LumpedElement, ProbeBox, ProbeType};
use csx::{ContinuousStructure, CoordSystem, Primitive, Property};
use ems::constants::{C0, Z0};
use ems::port::LumpedPort;
use ems::{Axis, Boundary, Complex64, Face, Fdtd, ProbeResult, Simulation, Waveform};

/// |DFT| at `f` of the Hann-windowed samples with `t >= t_start`.
fn spectrum(p: &ProbeResult, t_start: f64, f: f64) -> f64 {
    let v = p.scalar().unwrap();
    let first = p.times().iter().position(|&t| t >= t_start).unwrap();
    let (t, v) = (&p.times()[first..], &v[first..]);
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

fn peak(p: &ProbeResult, t_start: f64, lo: f64, hi: f64) -> f64 {
    let s = |f: f64| spectrum(p, t_start, f);
    let n = 100;
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
        let (c, d) = (b - g * (b - a), a + g * (b - a));
        if s(c) > s(d) {
            b = d;
        } else {
            a = c;
        }
    }
    0.5 * (a + b)
}

/// Circular PEC cavity, radius 10 mm, height 15 mm (drawing unit mm),
/// meshed in cylindrical coordinates with an axis and a closed azimuth.
const NA: usize = 24;

fn alpha(j: usize) -> f64 {
    j as f64 * TAU / NA as f64
}

fn cavity() -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid
        .set_lines(0, (0..=20).map(|i| i as f64 * 0.5))
        .unwrap();
    m.grid.set_lines(1, (0..=NA).map(alpha)).unwrap();
    m.grid.set_lines(2, (0..=15).map(f64::from)).unwrap();
    // A z excitation off the axis (TM modes) and a ρ one (TE modes).
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
    m
}

fn fdtd() -> Fdtd {
    Fdtd::new()
        .waveform(Waveform::gaussian(15e9, 10e9))
        .max_timesteps(9000)
        .no_end_criteria()
}

const A: f64 = 10e-3;
const H: f64 = 15e-3;

fn mode(x: f64, p: u32) -> f64 {
    C0 / TAU * ((x / A).powi(2) + (p as f64 * PI / H).powi(2)).sqrt()
}

#[test]
fn csx_cylinder_cavity_resonates() {
    let mut sim = Simulation::new(&cavity(), &fdtd()).unwrap();
    assert!(sim.grid().is_cylindrical() && sim.grid().has_axis() && sim.grid().is_closed());
    // Only mesh-check remarks (the time step is set by the arcs at the axis).
    assert!(
        sim.warnings().iter().all(|w| w.starts_with("mesh ")),
        "{:?}",
        sim.warnings()
    );
    let res = sim.run().unwrap();
    let t0 = 2.0 * fdtd().get_waveform().unwrap().duration().unwrap();
    let vz = res.probe("vz").unwrap();
    let vr = res.probe("vr").unwrap();
    for (name, f, p) in [
        ("TM010", mode(2.404_825_557_695_773, 0), vz),
        ("TM011", mode(2.404_825_557_695_773, 1), vz),
        ("TE111", mode(1.841_183_781_340_659, 1), vr),
        ("TM110", mode(3.831_705_970_207_512, 0), vz),
    ] {
        let got = peak(p, t0, 0.985 * f, 1.015 * f);
        let err = got / f - 1.0;
        println!(
            "{name}: {:.4} GHz (analytic {:.4}) {:+.3} %",
            got / 1e9,
            f / 1e9,
            100.0 * err
        );
        assert!(err.abs() < 5e-3, "{name} {err:+e}");
    }
}

#[test]
fn absorbing_faces_that_do_not_exist_are_dropped() {
    let f = fdtd()
        .max_timesteps(10)
        .boundary(Face::XMin, Boundary::mur())
        .boundary(Face::YMin, Boundary::pml(4))
        .boundary(Face::YMax, Boundary::pml(4))
        .boundary(Face::ZMax, Boundary::mur());
    let mut sim = Simulation::new(&cavity(), &f).unwrap();
    let w = sim.warnings().join("\n");
    assert!(
        w.contains("xmin ignored: on this cylindrical grid it is the axis"),
        "{w}"
    );
    assert!(w.contains("ymin ignored"), "{w}");
    assert!(w.contains("ymax ignored"), "{w}");
    assert!(!w.contains("zmax ignored"), "{w}");
    sim.run().unwrap();
}

#[test]
fn cartesian_only_features_are_refused() {
    let mut m = cavity();
    let mut sim = Simulation::new(&m, &fdtd().max_timesteps(10)).unwrap();
    // NF2FF boxes work (a sector box here, design §5.10); a mirror plane
    // through the box is refused.
    let b = ems::nf2ff::Nf2ffBox::new([1.0, 0.5, 1.0], [8.0, 6.0, 14.0], [10e9]);
    assert!(sim.add_nf2ff(&b).is_ok());
    let b = b.mirror(Axis::X, 0.0, ems::nf2ff::MirrorKind::Pec);
    assert!(matches!(sim.add_nf2ff(&b), Err(ems::Error::Invalid(_))));
    let port = ems::port::MslPort::new(
        1,
        [0.0, 0.0, 0.0],
        [5.0, 1.0, 1.0],
        ems::Axis::X,
        ems::Axis::Z,
    );
    assert!(matches!(
        port.add_to(&mut m),
        Err(ems::Error::Unsupported(_))
    ));
}

#[cfg(feature = "xml")]
#[test]
fn xml_round_trip_is_bit_identical() {
    use csx::OpenEmsDocument;
    // A file's Gaussian is openEMS's.
    let (model, fdtd) = (
        cavity(),
        fdtd()
            .waveform(Waveform::openems_gaussian(15e9, 10e9))
            .max_timesteps(1500),
    );
    // A file runs with openEMS's conventions (its axis update, design
    // §5.16.2, and its time step, §5.16.1): the same as the API with
    // `openems_run` and openEMS's criterion.
    let direct_fdtd = fdtd
        .clone()
        .threads(2)
        .openems_run(true)
        .timestep_method(ems::engine::TimestepMethod::Rennings);
    let mut direct = Simulation::new(&model, &direct_fdtd).unwrap();
    let a = direct.run().unwrap();
    let mut doc = OpenEmsDocument {
        structure: model.clone(),
        ..Default::default()
    };
    let mut settings = fdtd.to_csx();
    settings.coord_system = Some(1); // as openEMS writes cylindrical files
    doc.set_fdtd_settings(&settings);
    let xml = doc.to_xml_string();
    assert!(xml.contains("CoordSystem=\"1\""), "{xml}");
    let mut from_file = Simulation::from_xml_str(&xml).unwrap();
    assert!(from_file.grid().is_cylindrical());
    assert_eq!(from_file.engine().dt(), direct.engine().dt());
    let b = from_file.run().unwrap();
    for name in ["vz", "vr"] {
        assert_eq!(
            a.probe(name).unwrap().values(),
            b.probe(name).unwrap().values(),
            "{name}"
        );
    }
}

/// Coax a = 1 mm, b = 3 mm, 20 mm long with open (PMC) ends: a 50 Ω port
/// at z = 2 over the α range `port_alpha` and a 100 Ω load spanning the
/// whole circle at z = 18. Returns Z_in at `freqs`.
fn coax_z_in(port_alpha: (f64, f64), freqs: &[f64]) -> Vec<Complex64> {
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid
        .set_lines(
            0,
            (0..=COAX_NR).map(|i| 1.0 + 2.0 * i as f64 / COAX_NR as f64),
        )
        .unwrap();
    m.grid
        .set_lines(1, (0..=8).map(|j| j as f64 * TAU / 8.0))
        .unwrap();
    m.grid.set_lines(2, (0..=20).map(f64::from)).unwrap();
    let (a0, a1) = port_alpha;
    let port = LumpedPort::new(1, 50.0, [1.0, a0, 2.0], [3.0, a1, 2.0], Axis::X)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let load = m.add_property(Property::new(
        "load",
        csx::properties::PropertyKind::LumpedElement(LumpedElement {
            direction: csx::Axis::X,
            r: csx::Scalar::new(100.0),
            ..Default::default()
        }),
    ));
    m.add_primitive(
        load,
        Primitive::new(BoxShape::new([1.0, 0.0, 18.0], [3.0, TAU, 18.0])),
    )
    .unwrap();
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(0.0, 3e9))
        .boundary(Face::ZMin, Boundary::Pmc)
        .boundary(Face::ZMax, Boundary::Pmc)
        .max_timesteps(60_000)
        .end_criteria(ems::EndCriteria::energy_decay_db(-60.0));
    let mut sim = Simulation::new(&m, &fdtd).unwrap();
    assert!(
        sim.warnings().iter().all(|w| w.starts_with("mesh ")),
        "{:?}",
        sim.warnings()
    );
    let res = sim.run().unwrap();
    port.analyze(&res, freqs).unwrap().z_in()
}

const COAX_NR: usize = 10;

#[test]
fn coax_lumped_ports_and_full_circle_load() {
    // A full-circle radial port and load follow the transmission-line
    // formula (discrete TEM impedance; 0.25 % at 0.25 GHz to 1.3 % at
    // 2 GHz). Counting the duplicate α column of the load (or of a port on
    // the α = 0 face) would show as a 1/8 error. (Before ems re-selected
    // the edges of boxes past α = π, csx had given the port and the load
    // only their α ≤ π half, which happened to read 0.1–0.7 %.)
    let freqs = [0.25e9, 0.5e9, 1e9, 2e9];
    let z_in = coax_z_in((0.0, TAU), &freqs);
    let z0: f64 = (0..COAX_NR)
        .map(|i| {
            let n = COAX_NR as f64;
            let (a, b) = (1.0 + 2.0 * i as f64 / n, 1.0 + 2.0 * (i + 1) as f64 / n);
            (b - a) / (0.5 * (a + b))
        })
        .sum::<f64>()
        * Z0
        / TAU;
    let j = Complex64::I;
    for (z, f) in z_in.iter().zip(freqs) {
        let beta = TAU * f / C0;
        let open = |l: f64| -j * z0 / (beta * l).tan();
        let par = |a: Complex64, b: Complex64| a * b / (a + b);
        let line = |zl: Complex64, l: f64| {
            let t = (beta * l).tan();
            z0 * (zl + j * z0 * t) / (z0 + j * zl * t)
        };
        let z_load = par(Complex64::new(100.0, 0.0), open(2e-3));
        let want = par(open(2e-3), line(z_load, 16e-3));
        let err = (*z - want).norm() / want.norm();
        println!(
            "{:.2} GHz: Z_in {z:.2} (line model {want:.2}), {:.2} %",
            f / 1e9,
            100.0 * err
        );
        assert!(err < 0.015, "{z} vs {want}");
    }
    // A one-column port on the periodic α = 0 face (stored twice) behaves
    // exactly like the same port at an interior α.
    let edge = coax_z_in((0.0, 0.0), &freqs);
    let inner = coax_z_in((PI / 2.0, PI / 2.0), &freqs);
    for (a, b) in edge.iter().zip(&inner) {
        println!("one column: α = 0 {a:.4}, α = π/2 {b:.4}");
        assert!((*a - *b).norm() < 1e-4 * b.norm(), "{a} vs {b}");
    }
}

#[test]
fn axis_port_is_one_edge_whatever_its_alpha_range() {
    // A z port on the axis: the box may name one α copy or all of them;
    // both are the same physical edge, so the results are the same.
    let run = |a1: f64| {
        let mut m = cavity();
        let port = LumpedPort::new(1, 50.0, [0.0, 0.0, 6.0], [0.0, a1, 9.0], Axis::Z)
            .excite(1.0)
            .add_to(&mut m)
            .unwrap();
        let mut sim = Simulation::new(&m, &fdtd().max_timesteps(1500)).unwrap();
        let res = sim.run().unwrap();
        port.analyze(&res, &[11e9, 15e9]).unwrap().z_in()
    };
    let (one, all) = (run(0.0), run(TAU));
    println!("axis port Z_in: {one:?} / {all:?}");
    // Equal up to the rounding of the per-copy element values.
    for (a, b) in one.iter().zip(&all) {
        assert!((*a - *b).norm() <= 1e-12 * a.norm(), "{a} vs {b}");
    }
}
