//! csx model properties mapped onto engine operators by `Simulation`:
//! lumped ports built with `ems::port::LumpedPort` (Thevenin behaviour,
//! equivalence with the engine-level port), probe sign conventions, hard
//! vs soft sources, field probes and frequency-domain dumps.

use csx::primitives::BoxShape;
use csx::properties::{DumpBox, DumpType, Excitation, ExcitationType, ProbeBox, ProbeType};
use csx::{ContinuousStructure, Primitive, Property};
use ems::engine::{LumpedBox, LumpedPort as EnginePort};
use ems::port::LumpedPort;
use ems::post::Complex64;
use ems::{Axis, Fdtd, Simulation, Waveform};

/// The simulation's warnings except those of the mesh check.
fn non_mesh_warnings(sim: &Simulation) -> Vec<&String> {
    sim.warnings()
        .iter()
        .filter(|w| !w.starts_with("mesh "))
        .collect()
}

/// Model with `n` 1 mm cells per axis (lines at integer mm).
fn cube(n: usize) -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for a in 0..3 {
        m.grid.set_lines(a, (0..=n).map(|i| i as f64)).unwrap();
    }
    m
}

fn add_box(m: &mut ContinuousStructure, p: Property, lo: [f64; 3], hi: [f64; 3]) {
    let id = m.add_property(p);
    m.add_primitive(id, Primitive::new(BoxShape::new(lo, hi)))
        .unwrap();
}

/// The 4 mm × 2 mm loop of tests/lumped.rs in the y = 10 plane of a 20 mm
/// PEC box: port column at x = 8, load column at x = 12, z = 9..11, joined
/// by PEC lines at z = 9 and z = 11.
fn loop_model() -> ContinuousStructure {
    let mut m = cube(20);
    for z in [9.0, 11.0] {
        add_box(
            &mut m,
            Property::metal(format!("strip{z}")),
            [8.0, 10.0, z],
            [12.0, 10.0, z],
        );
    }
    m
}

fn add_load(m: &mut ContinuousStructure, r: f64) {
    add_box(
        m,
        Property::lumped_element(
            "load",
            csx::properties::LumpedElement::resistor(csx::Axis::Z, r),
        ),
        [12.0, 10.0, 9.0],
        [12.0, 10.0, 11.0],
    );
}

fn loop_fdtd(steps: u64) -> Fdtd {
    Fdtd::new()
        .waveform(Waveform::gaussian(0.0, 3e9))
        .max_timesteps(steps)
        .threads(2)
}

const FREQS: [f64; 4] = [50e6, 100e6, 200e6, 500e6];

/// The loop terminated in a csx lumped 100 Ω resistor and fed by a csx
/// lumped port: `Z_in ≈ R_L` (plus the ~3.9 nH loop inductance) at low
/// frequency, as with the engine-level port (tests/lumped.rs (b)).
#[test]
fn csx_port_into_resistor_loop() {
    let mut m = loop_model();
    add_load(&mut m, 100.0);
    let port = LumpedPort::new(1, 50.0, [8.0, 10.0, 9.0], [8.0, 10.0, 11.0], Axis::Z)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let mut sim = Simulation::new(&m, &loop_fdtd(4000)).unwrap();
    assert!(sim.warnings().is_empty(), "{:?}", sim.warnings());
    // The metal lines are PEC edges.
    assert_eq!(
        sim.engine().coefficients().cb(Axis::X)[sim.grid().index([9, 10, 9])],
        0.0
    );
    let res = sim.run().unwrap();
    assert_eq!(res.outcome().unwrap().steps, 4000);
    let r = port.analyze(&res, &FREQS).unwrap();
    let z = r.z_in();
    for (k, &f) in FREQS.iter().enumerate() {
        let l_eff = z[k].im / (2.0 * std::f64::consts::PI * f);
        eprintln!("f {f:.2e}: Z_in {}  L_eff {l_eff:.3e}", z[k]);
        assert!((z[k].re / 100.0 - 1.0).abs() < 0.01, "Re Z_in {}", z[k].re);
        assert!(l_eff > 3e-9 && l_eff < 5e-9, "loop inductance {l_eff}");
    }
    let s = r.s11()[0];
    assert!(
        (s - Complex64::new(1.0 / 3.0, 0.0)).norm() < 0.01,
        "S11 {s}"
    );
    // Probes are reachable by name and id.
    let props = port.properties().unwrap();
    let v = res.probe("port_ut1").unwrap();
    assert_eq!(res.probe_by_id(props.voltage.1).unwrap().name(), "port_ut1");
    assert_eq!(v.kind(), ems::ProbeKind::Voltage);
    assert_eq!(v.times().len(), 4000);
    let i = res.probe("port_it1").unwrap();
    assert!((i.times()[0] - 0.5 * res.dt()).abs() < 1e-24);
    assert!((v.times()[0] - res.dt()).abs() < 1e-24);
}

/// The same port built from csx properties and as an engine `LumpedPort`
/// on the same compiled model gives the same V and I series (the csx
/// excitation over the lumped resistor is a Thevenin source, sampled with
/// openEMS's timing: half a step earlier than the engine port's). With
/// openEMS run semantics (XML files) it is scaled to openEMS's soft source
/// instead; see tests/openems_core.rs.
#[test]
fn csx_port_matches_engine_port() {
    let steps = 1500;
    let mut m = loop_model();
    add_load(&mut m, 100.0);
    let bare = m.clone();
    let port = LumpedPort::new(1, 50.0, [8.0, 10.0, 9.0], [8.0, 10.0, 11.0], Axis::Z)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let fdtd = loop_fdtd(steps);
    let mut a = Simulation::new(&m, &fdtd).unwrap();
    let ra = a.run().unwrap();

    let mut b = Simulation::new(&bare, &fdtd).unwrap();
    let bx = LumpedBox::new(Axis::Z, [8, 10, 9], [8, 10, 11]).unwrap();
    let dt = b.engine().dt();
    let wave = Waveform::gaussian(0.0, 3e9);
    let h = EnginePort::new(bx, 50.0)
        .excite(move |t| wave.value(t - 0.5 * dt, dt))
        .add_to(b.engine_mut())
        .unwrap();
    b.run().unwrap();
    let p = b.engine().operator(&h);
    // The port current is minus the port resistor's terminal current.
    let el = ra.lumped_current("port_resist_1").unwrap();
    assert_eq!(el.kind(), ems::ProbeKind::Current);
    assert_eq!(
        Some(el.id()),
        m.property_id("port_resist_1").map(|p| p.get())
    );
    assert!((el.times()[0] - 0.5 * ra.dt()).abs() < 1e-24);
    let va = ra.probe("port_ut1").unwrap().scalar().unwrap();
    let ia: Vec<f64> = el.scalar().unwrap().iter().map(|i| -i).collect();
    assert_eq!(va.len(), p.voltage().len());
    let vmax = p.voltage().iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let imax = p.current().iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(vmax > 0.1 && imax > 1e-3);
    for n in 0..va.len() {
        assert!((va[n] - p.voltage()[n]).abs() < 1e-5 * vmax, "V[{n}]");
        assert!((ia[n] - p.current()[n]).abs() < 1e-5 * imax, "I[{n}]");
    }
    let za = port.analyze(&ra, &FREQS).unwrap().z_in();
    let zb = p.analyze(&FREQS).unwrap().z_in();
    for (x, y) in za.iter().zip(&zb) {
        assert!((*x - *y).norm() < 1e-3 * y.norm(), "{x} vs {y}");
    }
}

/// Probe sign conventions: swapping start and stop negates voltage and
/// current probes; the probe weight scales the result.
#[test]
fn probe_directions_and_weights() {
    let mut m = loop_model();
    add_load(&mut m, 100.0);
    LumpedPort::new(1, 50.0, [8.0, 10.0, 9.0], [8.0, 10.0, 11.0], Axis::Z)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let probe = |t: ProbeType| ProbeBox::new(t);
    add_box(
        &mut m,
        Property::probe("v_up", probe(ProbeType::Voltage)),
        [12.0, 10.0, 9.0],
        [12.0, 10.0, 11.0],
    );
    add_box(
        &mut m,
        Property::probe("v_down", probe(ProbeType::Voltage).with_weight(2.0)),
        [12.0, 10.0, 11.0],
        [12.0, 10.0, 9.0],
    );
    let cur = |w: f64| {
        probe(ProbeType::Current)
            .with_norm_dir(csx::Axis::Z)
            .with_weight(w)
    };
    add_box(
        &mut m,
        Property::probe("i_up", cur(1.0)),
        [12.0, 10.0, 9.0],
        [12.0, 10.0, 11.0],
    );
    add_box(
        &mut m,
        Property::probe("i_down", cur(1.0)),
        [12.0, 10.0, 11.0],
        [12.0, 10.0, 9.0],
    );
    let mut sim = Simulation::new(&m, &loop_fdtd(600)).unwrap();
    // Short run: only the mesh check's "too few steps" warning.
    let w = non_mesh_warnings(&sim);
    assert!(w.is_empty(), "{w:?}");
    let res = sim.run().unwrap();
    let s = |n: &str| res.probe(n).unwrap().scalar().unwrap().to_vec();
    let (vu, vd, iu, id) = (s("v_up"), s("v_down"), s("i_up"), s("i_down"));
    let (vport, iport) = (s("port_ut1"), s("port_it1"));
    assert!(vu.iter().any(|&v| v.abs() > 1e-3));
    for n in 0..vu.len() {
        assert_eq!(vd[n], -2.0 * vu[n]);
        assert_eq!(id[n], -iu[n]);
    }
    // Around the loop: the load current (start → stop along +z at the
    // load, i.e. down the port) equals the port's terminal current at low
    // frequency, and the load voltage is R_L times it.
    let f = [100e6];
    let dft = |n: &str| res.probe(n).unwrap().dft(&f).unwrap()[0];
    let (vl, il, ip) = (dft("v_up"), dft("i_up"), dft("port_it1"));
    assert!((il - ip).norm() < 0.02 * ip.norm(), "{il} vs {ip}");
    assert!(
        ((vl / il).re / 100.0 - 1.0).abs() < 0.01,
        "Z_load {}",
        vl / il
    );
    assert!(iport.len() == vport.len());
}

fn line_model() -> ContinuousStructure {
    cube(8)
}

/// openEMS timing: a field at time `T` gets the signal at `T − dt`. Hard E
/// source: the edge voltages are overwritten with `E·l·a(n·dt)` for the
/// state at `(n+1)·dt`, and an E-field probe at the lower node of the
/// driven edges reads `E·a(t − dt)` (the native E_z edge above the node).
/// Soft E source: the first sample is `E·l·a(0) = 0` and the second
/// `E·l·a(dt)`. Hard H source: the dual-edge current is `H·l*·a`.
#[test]
fn hard_and_soft_sources_and_field_probe() {
    let f0 = 1e9;
    let amp = 3.0; // V/m
    let wave = Waveform::sinusoid(f0);
    let run = |kind: ExcitationType| {
        let mut m = line_model();
        let ex = Excitation::new([0.0, 0.0, amp]).with_type(kind);
        add_box(
            &mut m,
            Property::excitation("src", ex),
            [4.0, 4.0, 3.0],
            [4.0, 4.0, 5.0],
        );
        add_box(
            &mut m,
            Property::probe("v", ProbeBox::new(ProbeType::Voltage)),
            [4.0, 4.0, 3.0],
            [4.0, 4.0, 5.0],
        );
        add_box(
            &mut m,
            Property::probe("e", ProbeBox::new(ProbeType::EField)),
            [4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0],
        );
        let fdtd = Fdtd::new()
            .waveform(wave.clone())
            .max_timesteps(50)
            .threads(1);
        let mut sim = Simulation::new(&m, &fdtd).unwrap();
        assert!(sim.warnings().is_empty(), "{:?}", sim.warnings());
        sim.run().unwrap()
    };
    let hard = run(ExcitationType::EFieldHard);
    let dt = hard.dt();
    let v = hard.probe("v").unwrap();
    let e = hard.probe("e").unwrap();
    for (k, (&t, &x)) in v.times().iter().zip(v.scalar().unwrap()).enumerate() {
        assert!((t - (k as f64 + 1.0) * dt).abs() < 1e-22);
        let want = 2.0 * (amp * 1e-3 * wave.value(t - dt, dt)) as f32 as f64;
        assert!((x - want).abs() <= 1e-6 * amp * 1e-3, "{x} vs {want}");
        let ez = e.vector().unwrap()[k][2];
        assert!(
            (ez - amp * wave.value(t - dt, dt)).abs() <= 1e-5 * amp,
            "{ez}"
        );
    }
    assert_eq!(e.kind(), ems::ProbeKind::EField);

    let soft = run(ExcitationType::EFieldSoft);
    let v = soft.probe("v").unwrap().scalar().unwrap();
    assert_eq!(v[0], 0.0);
    let want = 2.0 * amp * 1e-3 * wave.value(dt, dt);
    assert!(
        (v[1] - want).abs() < 1e-6 * want.abs(),
        "{} vs {want}",
        v[1]
    );
    // A soft source radiates: later samples differ from the hard ones.
    let vh = hard.probe("v").unwrap().scalar().unwrap();
    assert!((v[40] - vh[40]).abs() > 1e-3 * vh[40].abs());

    // Hard H source on one H_z face; H probe at its node reads it back
    // through the H interpolation (bilinear across x, y).
    let mut m = line_model();
    let ex = Excitation::new([0.0, 0.0, 2.0]).with_type(ExcitationType::HFieldHard);
    add_box(
        &mut m,
        Property::excitation("hsrc", ex),
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 4.0],
    );
    let fdtd = Fdtd::new()
        .waveform(wave.clone())
        .max_timesteps(20)
        .threads(1);
    let mut sim = Simulation::new(&m, &fdtd).unwrap();
    assert!(sim.warnings().is_empty(), "{:?}", sim.warnings());
    let src = sim.source("hsrc").unwrap();
    let edges: Vec<_> = src.edges().collect();
    assert_eq!(edges.len(), 1);
    assert_eq!((edges[0].0, edges[0].1), (Axis::Z, [4, 4, 4]));
    sim.run().unwrap();
    let t = (19.0 + 0.5) * dt;
    let want = 2.0 * 1e-3 * wave.value(t - dt, dt);
    let got = sim.engine().i(Axis::Z, [4, 4, 4]) as f64;
    assert!((got - want).abs() < 1e-6 * 2e-3, "{got} vs {want}");
}

/// Disabled excitations are ignored; plane waves without a direction and
/// unsupported dumps produce warnings.
#[test]
fn unsupported_and_disabled_properties() {
    let mut m = line_model();
    let mut ex = Excitation::new([0.0, 0.0, 1.0]);
    ex.enabled = false;
    add_box(
        &mut m,
        Property::excitation("off", ex),
        [4.0, 4.0, 3.0],
        [4.0, 4.0, 5.0],
    );
    let pw = Excitation::new([0.0, 0.0, 1.0]).with_type(ExcitationType::PlaneWave);
    add_box(
        &mut m,
        Property::excitation("pw", pw),
        [1.0, 1.0, 1.0],
        [6.0, 6.0, 6.0],
    );
    add_box(
        &mut m,
        Property::dump("sar", DumpBox::new(DumpType::Sar1g)),
        [0.0, 0.0, 0.0],
        [8.0, 8.0, 8.0],
    );
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(1e9, 1e9))
        .max_timesteps(5)
        .threads(1);
    let sim = Simulation::new(&m, &fdtd).unwrap();
    let w = sim.warnings();
    assert!(sim.sources().excitations.iter().all(|e| e.name != "off"));
    assert!(sim.source("off").is_none());
    assert!(
        w.iter()
            .any(|s| s.contains("\"off\"") && s.contains("disabled")),
        "{w:?}"
    );
    assert!(w.iter().any(|s| s.contains("\"pw\"")), "{w:?}");
    assert!(w.iter().any(|s| s.contains("\"sar\"")), "{w:?}");
}

/// A frequency-domain E dump through the Simulation matches the DFT of an
/// E-field probe at one of its nodes; a time-domain dump records every
/// `K`-th state and writes VTR files.
#[test]
fn frequency_and_time_dumps() {
    let f = 2e9;
    let mut m = line_model();
    add_box(
        &mut m,
        Property::excitation("src", Excitation::new([0.0, 0.0, 1.0])),
        [4.0, 4.0, 3.0],
        [4.0, 4.0, 5.0],
    );
    let mut fd = DumpBox::new(DumpType::EFieldFd);
    fd.probe.frequencies = vec![f, 1.5e9];
    add_box(
        &mut m,
        Property::dump("fd", fd),
        [2.0, 2.0, 4.0],
        [6.0, 6.0, 4.0],
    );
    let mut td = DumpBox::new(DumpType::HField);
    td.sub_sampling = Some([2, 2, 1]);
    add_box(
        &mut m,
        Property::dump("td", td),
        [0.0, 0.0, 4.0],
        [8.0, 8.0, 4.0],
    );
    add_box(
        &mut m,
        Property::probe("e", ProbeBox::new(ProbeType::EField).with_frequencies([f])),
        [3.0, 5.0, 4.0],
        [3.0, 5.0, 4.0],
    );
    let wave = Waveform::gaussian(2e9, 1e9);
    let steps = 600;
    // Oversampling large enough that the dumps sample every step.
    let fdtd = Fdtd::new()
        .waveform(wave.clone())
        .max_timesteps(steps)
        .oversampling(1000)
        .threads(1);
    let mut sim = Simulation::new(&m, &fdtd).unwrap();
    // Short run: only the mesh check's "too few steps" warning.
    let w = non_mesh_warnings(&sim);
    assert!(w.is_empty(), "{w:?}");
    let res = sim.run().unwrap();
    let pe = res.probe("e").unwrap();
    let d = res.dump("fd").unwrap();
    let fdump = d.freq().unwrap();
    // The dump samples every K-th step (openEMS rule); the same DFT of
    // the probe series (E at n·dt).
    let dt = res.dt();
    let k = (fdump.sample_interval() / dt).round() as u64;
    assert!(k > 1);
    let probe = pe
        .times()
        .iter()
        .zip(pe.vector().unwrap())
        .filter(|(t, _)| ((**t / dt).round() as u64).is_multiple_of(k))
        .map(|(t, x)| Complex64::cis(-std::f64::consts::TAU * f * t) * x[2] * (k as f64 * dt))
        .fold(Complex64::ZERO, |a, b| a + b);
    assert_eq!(fdump.sampler().dims(), [5, 5, 1]);
    // Node (3, 5, 4) is sample (1, 3, 0).
    let v = fdump.values(0)[5 + 3][2];
    assert!(probe.norm() > 0.0);
    assert!((v - probe).norm() < 1e-6 * probe.norm(), "{v} vs {probe}");

    let td = res.dump("td").unwrap();
    assert_eq!(td.sampler().dims(), [5, 5, 1]);
    assert_eq!(td.snapshots().unwrap().len() as u64, steps + 1);

    // Default oversampling (4): every K-th state, K from f_max = 3 GHz.
    let fdtd = fdtd.oversampling(4);
    let mut sim = Simulation::new(&m, &fdtd).unwrap();
    let res = sim.run().unwrap();
    let k = (1.0 / (2.0 * 3e9 * 4.0 * res.dt())).floor() as u64;
    assert!(k > 1);
    let snaps = res.dump("td").unwrap().snapshots().unwrap();
    assert_eq!(snaps[1].step, k);
    assert_eq!(snaps.len() as u64, steps / k + 1);
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("sim_operators_td");
    res.dump("td").unwrap().write_vtr(&dir, "td").unwrap();
    assert!(dir.join("td.pvd").exists());
    res.dump("fd").unwrap().write_vtr(&dir, "fd").unwrap();
    assert!(dir.join("fd_f0.vtr").exists());
    assert!(sim.time_dump("td").is_some() && sim.freq_dump("fd").is_some());
}

/// `Caps` puts PEC end plates on a multi-column lumped element: the
/// transverse edges of its end planes become PEC, its interior does not.
#[test]
fn lumped_caps() {
    let run = |caps: bool| {
        let mut m = line_model();
        let mut le = csx::properties::LumpedElement::resistor(csx::Axis::Z, 50.0);
        le.caps = caps;
        add_box(
            &mut m,
            Property::lumped_element("r", le),
            [3.0, 4.0, 3.0],
            [5.0, 4.0, 5.0],
        );
        let sim = Simulation::new(&m, &Fdtd::new().max_timesteps(1).threads(1)).unwrap();
        assert!(sim.warnings().is_empty(), "{:?}", sim.warnings());
        let el = sim.lumped_element("r").unwrap();
        assert_eq!(el.edge_values().len(), 3 * 2);
        assert_eq!(el.series_edges(), 2);
        // csx distributes R = 50 Ω over 2 series × 3 parallel edges.
        assert!((el.edge_values()[0].r.unwrap() - 75.0).abs() < 1e-9);
        let g = sim.grid();
        let co = sim.engine().coefficients();
        [3, 4, 5].map(|z| co.cb(Axis::X)[g.index([3, 4, z])] == 0.0)
    };
    assert_eq!(run(true), [true, false, true]);
    assert_eq!(run(false), [false, false, false]);
}

/// Field probes read the native E components of the (snapped) start corner
/// of their box, like openEMS (`E_z` of the edge above the node); dumps
/// honour the probe settings' start / stop time; a hard
/// source on a lumped element's edge warns; a multi-column port's voltage
/// probe sits on a mesh line (no snapping warning).
#[test]
fn field_probe_centre_dump_window_and_warnings() {
    let wave = Waveform::sinusoid(1e9);
    let mut m = line_model();
    add_box(
        &mut m,
        Property::excitation(
            "src",
            Excitation::new([0.0, 0.0, 3.0]).with_type(ExcitationType::EFieldHard),
        ),
        [4.0, 4.0, 3.0],
        [4.0, 4.0, 5.0],
    );
    let point = |m: &mut ContinuousStructure, name: &str, lo: [f64; 3], hi: [f64; 3]| {
        add_box(
            m,
            Property::probe(name, ProbeBox::new(ProbeType::EField)),
            lo,
            hi,
        );
    };
    point(&mut m, "p4", [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]);
    point(&mut m, "p5", [5.0, 4.0, 4.0], [5.0, 4.0, 4.0]);
    point(&mut m, "p3", [4.0, 4.0, 3.0], [4.0, 4.0, 3.0]);
    point(&mut m, "box_z", [4.0, 4.0, 3.0], [4.0, 4.0, 5.0]);
    point(&mut m, "box_x", [4.0, 4.0, 4.0], [5.0, 4.0, 4.0]);
    let (t0, t1) = (20e-12, 60e-12);
    let mut td = DumpBox::new(DumpType::EField);
    td.probe.start_time = csx::Scalar::new(t0);
    td.probe.stop_time = csx::Scalar::new(t1);
    add_box(
        &mut m,
        Property::dump("td", td),
        [0.0, 0.0, 4.0],
        [8.0, 8.0, 4.0],
    );
    let fdtd = Fdtd::new()
        .waveform(wave)
        .max_timesteps(60)
        .oversampling(1000)
        .threads(1);
    let mut sim = Simulation::new(&m, &fdtd).unwrap();
    assert!(sim.warnings().is_empty(), "{:?}", sim.warnings());
    let res = sim.run().unwrap();
    let v = |n: &str| res.probe(n).unwrap().vector().unwrap().to_vec();
    let (p4, p5, bz, bx) = (v("p4"), v("p5"), v("box_z"), v("box_x"));
    assert!(p4.iter().any(|x| x[2].abs() > 1.0));
    let p3 = v("p3");
    for k in 0..p4.len() {
        assert_eq!(bz[k], p3[k], "a box reads its start corner");
        assert_eq!(bx[k], p4[k], "a box reads its start corner");
        assert!(p5[k][2].abs() <= p4[k][2].abs() + 1e-12);
    }
    let snaps = res.dump("td").unwrap().snapshots().unwrap();
    let dt = res.dt();
    // openEMS window: (n + 1)·dt ≥ start and (n − 1)·dt < stop.
    let want: Vec<u64> = (0..=60u64)
        .filter(|&n| (n + 1) as f64 * dt >= t0 && (n as f64 - 1.0) * dt < t1)
        .collect();
    assert!(want.len() > 5);
    assert_eq!(snaps.iter().map(|s| s.step).collect::<Vec<_>>(), want);

    // A hard E source on a lumped element's edges overrides it: warning.
    let mut m = line_model();
    add_box(
        &mut m,
        Property::lumped_element(
            "r",
            csx::properties::LumpedElement::resistor(csx::Axis::Z, 50.0),
        ),
        [4.0, 4.0, 3.0],
        [4.0, 4.0, 5.0],
    );
    add_box(
        &mut m,
        Property::excitation(
            "hard",
            Excitation::new([0.0, 0.0, 1.0]).with_type(ExcitationType::EFieldHard),
        ),
        [4.0, 4.0, 3.0],
        [4.0, 4.0, 5.0],
    );
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(1e9, 1e9))
        .max_timesteps(1)
        .threads(1);
    let sim = Simulation::new(&m, &fdtd).unwrap();
    let w = sim.warnings();
    assert!(
        w.iter()
            .any(|s| s.contains("\"hard\"") && s.contains("hard E source on 2 edge(s)")),
        "{w:?}"
    );

    // A two-column port: the voltage probe sits on a mesh line.
    let mut m = line_model();
    LumpedPort::new(1, 50.0, [3.0, 4.0, 3.0], [4.0, 4.0, 5.0], Axis::Z)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let sim = Simulation::new(&m, &fdtd).unwrap();
    let w = non_mesh_warnings(&sim);
    assert!(w.is_empty(), "{w:?}");
}
