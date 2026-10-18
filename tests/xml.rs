//! Running openEMS XML documents (`Simulation::from_xml*`, design §9 "XML
//! round trip"): a hand-written PEC cavity resonates at the analytic
//! frequencies, a lumped-port file gives the expected input impedance, and
//! a model built in Rust gives bit-identical probe results directly and
//! after a round trip through csx's XML writer.
#![cfg(feature = "xml")]

use std::f64::consts::PI;
use std::path::PathBuf;

use csx::primitives::BoxShape;
use csx::properties::{LumpedElement, Material, ProbeBox, ProbeType};
use csx::{ContinuousStructure, OpenEmsDocument, Primitive, Property};
use ems::constants::C0;
use ems::port::LumpedPort;
use ems::{Axis, Boundary, EndCriteria, Face, Fdtd, ProbeResult, Simulation, Waveform};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

/// |DFT| at `f` of the Hann-windowed samples with `t >= t_start`.
fn spectrum(p: &ProbeResult, t_start: f64, f: f64) -> f64 {
    let v = p.scalar().unwrap();
    let first = p.times().iter().position(|&t| t >= t_start).unwrap();
    let (t, v) = (&p.times()[first..], &v[first..]);
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

/// Spectral peak in `[lo, hi]`: grid scan plus golden-section refinement.
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

#[test]
fn hand_written_pec_cavity_resonates() {
    let mut sim = Simulation::from_xml(data("cavity_pec.xml")).unwrap();
    // Only mesh-check remarks (5 mm cells are ~10 cells per wavelength
    // at f_max).
    assert!(
        sim.warnings().iter().all(|w| w.starts_with("mesh ")),
        "{:?}",
        sim.warnings()
    );
    assert_eq!(sim.grid().nodes(), [21, 10, 15]);
    assert_eq!(sim.run_limits().max_timesteps, 5000);
    assert!(sim.run_limits().end_criteria.is_none());
    let res = sim.run().unwrap();
    assert_eq!(res.outcome().unwrap().steps, 5000);
    let p = res.probe("probe").unwrap();
    assert_eq!(p.kind(), ems::ProbeKind::Voltage);
    let t_start = Waveform::gaussian(3.2e9, 2.5e9).duration().unwrap();
    let (a, d): (f64, f64) = (0.100, 0.070);
    for (m, n) in [(1.0, 1.0), (2.0, 1.0)] {
        let fa = C0 / 2.0 * ((m / a).powi(2) + (n / d).powi(2)).sqrt();
        let f = peak(p, t_start, 0.95 * fa, 1.05 * fa);
        let err = (f - fa) / fa;
        eprintln!(
            "TE{m}0{n}: {f:.5e} Hz vs {fa:.5e} Hz ({:+.3}%)",
            err * 100.0
        );
        assert!(err.abs() < 5e-3, "TE{m}0{n} off by {:.3}%", err * 100.0);
        assert!(spectrum(p, t_start, f) > 20.0 * spectrum(p, t_start, 0.9 * fa));
    }
}

/// The resistor loop of tests/sim_operators.rs (a 4 mm × 2 mm loop in a
/// 20 mm PEC box, port column at x = 8, 100 Ω load at x = 12).
fn loop_model() -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for a in 0..3 {
        m.grid.set_lines(a, (0..=20).map(f64::from)).unwrap();
    }
    for z in [9.0, 11.0] {
        let id = m.add_property(Property::metal(format!("strip{z}")));
        m.add_primitive(
            id,
            Primitive::new(BoxShape::new([8.0, 10.0, z], [12.0, 10.0, z])),
        )
        .unwrap();
    }
    let id = m.add_property(Property::lumped_element(
        "load",
        LumpedElement::resistor(csx::Axis::Z, 100.0),
    ));
    m.add_primitive(
        id,
        Primitive::new(BoxShape::new([12.0, 10.0, 9.0], [12.0, 10.0, 11.0])),
    )
    .unwrap();
    m
}

fn add_port(m: &mut ContinuousStructure) -> LumpedPort {
    LumpedPort::new(1, 50.0, [8.0, 10.0, 9.0], [8.0, 10.0, 11.0], Axis::Z)
        .excite(1.0)
        .add_to(m)
        .unwrap()
}

fn loop_fdtd() -> Fdtd {
    Fdtd::new()
        .waveform(Waveform::gaussian(0.0, 3e9))
        .max_timesteps(4000)
        .no_end_criteria()
}

/// Writes `model` + `fdtd` as an openEMS document string via csx.
fn to_xml(model: &ContinuousStructure, fdtd: &Fdtd) -> String {
    let mut doc = OpenEmsDocument {
        structure: model.clone(),
        ..Default::default()
    };
    doc.set_fdtd_settings(&fdtd.to_csx());
    doc.to_xml_string()
}

/// Regenerates `tests/data/lumped_port.xml` from the Rust model
/// (`cargo test --test xml -- --ignored write_lumped_port_fixture`).
#[test]
#[ignore = "regenerates a test fixture"]
fn write_lumped_port_fixture() {
    let mut m = loop_model();
    add_port(&mut m);
    std::fs::write(data("lumped_port.xml"), to_xml(&m, &loop_fdtd())).unwrap();
}

#[test]
fn lumped_port_file_gives_expected_z_in() {
    let mut sim = Simulation::from_xml(data("lumped_port.xml")).unwrap();
    assert!(sim.warnings().is_empty(), "{:?}", sim.warnings());
    assert!(sim.lumped_element("port_resist_1").is_some());
    assert!(sim.lumped_element("load").is_some());
    let res = sim.run().unwrap();
    assert_eq!(res.outcome().unwrap().steps, 4000);
    // The port helper only needs the property names to analyse the file's
    // probes.
    let port = add_port(&mut loop_model());
    let freqs = [50e6, 100e6, 200e6, 500e6];
    let r = port.analyze(&res, &freqs).unwrap();
    for (z, f) in r.z_in().iter().zip(freqs) {
        let l = z.im / (2.0 * PI * f);
        eprintln!("f {f:.2e}: Z_in {z}  L {l:.3e}");
        assert!((z.re / 100.0 - 1.0).abs() < 0.01, "Re Z_in {}", z.re);
        assert!(l > 3e-9 && l < 5e-9, "loop inductance {l}");
    }
}

/// A richer model for the round trip: the port loop plus a dielectric
/// block, field and voltage probes, Mur and PML faces, end criteria.
fn round_trip_model() -> (ContinuousStructure, Fdtd) {
    let mut m = loop_model();
    add_port(&mut m);
    let id = m.add_property(Property::material("diel", Material::new(3.5, 0.02)));
    m.add_primitive(
        id,
        Primitive::new(BoxShape::new([2.5, 3.0, 4.0], [7.0, 16.5, 15.0])).with_priority(1),
    )
    .unwrap();
    let id = m.add_property(Property::probe(
        "e_probe",
        ProbeBox::new(ProbeType::EField).with_frequencies([1e9, 2e9]),
    ));
    m.add_primitive(
        id,
        Primitive::new(BoxShape::new([5.0, 6.0, 7.0], [5.0, 6.0, 7.0])),
    )
    .unwrap();
    let id = m.add_property(Property::probe(
        "v_probe",
        ProbeBox::new(ProbeType::Voltage).with_weight(-2.0),
    ));
    m.add_primitive(
        id,
        Primitive::new(BoxShape::new([4.0, 4.0, 3.0], [4.0, 4.0, 12.0])),
    )
    .unwrap();
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(2e9, 1.5e9))
        .max_timesteps(1500)
        .end_criteria(EndCriteria::energy_ratio(1e-4))
        .timestep_factor(0.95)
        .boundary(Face::XMax, Boundary::pml(4))
        .boundary(Face::ZMax, Boundary::mur());
    (m, fdtd)
}

#[test]
fn rust_model_round_trips_bit_identically() {
    let (model, fdtd) = round_trip_model();
    let mut direct = Simulation::new(&model, &fdtd.clone().threads(3)).unwrap();
    let a = direct.run().unwrap();

    let xml = to_xml(&model, &fdtd);
    let doc = OpenEmsDocument::from_xml_str(&xml).unwrap();
    let mut from_file =
        Simulation::from_document_with(&doc, &Default::default(), |f| f.threads(2)).unwrap();
    assert_eq!(from_file.warnings(), direct.warnings());
    assert_eq!(from_file.engine().dt(), direct.engine().dt());
    assert_eq!(from_file.run_limits(), direct.run_limits());
    let b = from_file.run().unwrap();

    assert_eq!(a.outcome().unwrap().steps, b.outcome().unwrap().steps);
    assert_eq!(a.outcome().unwrap().reason, b.outcome().unwrap().reason);
    assert_eq!(a.probes().len(), 4);
    assert_eq!(a.probes().len(), b.probes().len());
    for (pa, pb) in a.probes().iter().zip(b.probes()) {
        assert_eq!(pa.name(), pb.name());
        assert_eq!(pa.frequencies(), pb.frequencies());
        assert_eq!(pa.times(), pb.times(), "{}", pa.name());
        assert_eq!(pa.values(), pb.values(), "{}", pa.name());
    }

    // The string entry point gives the same as the document one.
    let mut s = Simulation::from_xml_str(&xml).unwrap();
    let c = s.run().unwrap();
    assert_eq!(
        c.probe("port_ut1").unwrap().values(),
        a.probe("port_ut1").unwrap().values()
    );
}

#[test]
fn document_without_fdtd_uses_defaults() {
    let model = loop_model();
    let xml = model.to_xml_string();
    let sim = Simulation::from_xml_str(&xml).unwrap();
    assert!(
        sim.warnings()[0].contains("no <FDTD>"),
        "{:?}",
        sim.warnings()
    );
    assert_eq!(
        sim.fdtd().get_max_timesteps(),
        Fdtd::OPENEMS_DEFAULT_TIMESTEPS
    );
    assert!(Simulation::from_xml_str("<notOpenEMS/>").is_err());
    assert!(Simulation::from_xml(data("does_not_exist.xml")).is_err());
}
