//! A model compiled once for several runs ([`ems::CompiledModel`]): every
//! simulation built from it equals a fresh compile and run, bit for bit.
//!
//! The model is a board region with everything that carries state between
//! steps: a Debye laminate (dispersive poles), lossy copper (conducting
//! sheets) or perfect copper, two lumped ports (lumped elements with
//! Thevenin sources), CPML on every face, probes.

use csx::primitives::BoxShape;
use csx::properties::{DebyeMaterial, DebyePole};
use csx::{ContinuousStructure, Primitive, Property};
use ems::port::LumpedPort;
use ems::{
    Axis, Boundaries, Boundary, CompileOptions, CompiledModel, Cpml, EndCriteria, Fdtd, Results,
    Simulation, Waveform,
};

/// A microstrip on a ground plane (mm): 24 × 20 × 14 cells, ports at both
/// ends of the strip, those in `excited` with their excitation.
fn board(lossy: bool, excited: &[u32]) -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid
        .set_lines(0, (0..=24).map(|i| i as f64 * 0.5))
        .unwrap();
    m.grid
        .set_lines(1, (0..=20).map(|i| i as f64 * 0.5))
        .unwrap();
    m.grid
        .set_lines(2, (0..=14).map(|i| i as f64 * 0.4))
        .unwrap();
    let laminate = if lossy {
        Property::debye_material(
            "laminate",
            DebyeMaterial {
                material: csx::Material::new(4.1, 1e-4),
                poles: vec![DebyePole::new(0.12, 1.6e-11), DebyePole::new(0.1, 1.6e-10)],
            },
        )
    } else {
        Property::material("laminate", csx::Material::new(4.3, 0.0))
    };
    let id = m.add_property(laminate);
    let slab = BoxShape::new([0.0, 0.0, 2.0], [12.0, 10.0, 3.6]);
    m.add_primitive(id, Primitive::new(slab)).unwrap();
    let copper = |name: &str| {
        if lossy {
            Property::conducting_sheet(name, 5.8e7, 35e-6)
        } else {
            Property::metal(name)
        }
    };
    for (name, start, stop) in [
        ("ground", [0.0, 0.0, 2.0], [12.0, 10.0, 2.0]),
        ("strip", [3.0, 4.0, 3.6], [9.0, 6.0, 3.6]),
    ] {
        let id = m.add_property(copper(name));
        let sheet = Primitive::new(BoxShape::new(start, stop)).with_priority(10);
        m.add_primitive(id, sheet).unwrap();
    }
    for (n, x) in [(1, 3.0), (2, 9.0)] {
        let mut port = LumpedPort::new(n, 50.0, [x, 4.0, 3.6], [x, 6.0, 2.0], Axis::Z).priority(30);
        if excited.contains(&n) {
            port = port.excite(1.0);
        }
        port.add_to(&mut m).unwrap();
    }
    m
}

fn fdtd() -> Fdtd {
    Fdtd::new()
        .waveform(Waveform::gaussian(5e9, 5e9))
        .boundaries(Boundaries::all(Boundary::Pml(Cpml::cells(4))))
        .end_criteria(EndCriteria::energy_decay_db(-60.0).check_interval(16))
        .max_timesteps(400)
        .threads(1)
}

/// Everything a run leaves behind, as bits: the probes and lumped-element
/// currents by name, the fields, the run outcome.
#[derive(Debug, PartialEq)]
struct Record {
    series: Vec<(String, Vec<u64>, Vec<u64>)>,
    fields: Vec<u32>,
    outcome: (u64, u64, u64),
}

fn record(sim: &Simulation, res: &Results, names: &[&str]) -> Record {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut series = Vec::new();
    for &name in names {
        let p = res
            .probe(name)
            .or_else(|| res.lumped_current(name))
            .unwrap_or_else(|| panic!("no series {name:?}"));
        let values = p.scalar().expect("scalar probe");
        assert!(values.iter().any(|v| *v != 0.0), "{name} recorded nothing");
        series.push((name.to_string(), bits(p.times()), bits(values)));
    }
    let engine = sim.engine();
    let fields = Axis::ALL
        .iter()
        .flat_map(|&a| {
            let mut v = engine.u_component(a);
            v.extend(engine.i_component(a));
            v
        })
        .map(f32::to_bits)
        .collect();
    let o = res.outcome().expect("a finished run");
    Record {
        series,
        fields,
        outcome: (o.steps, o.energy.to_bits(), o.max_energy.to_bits()),
    }
}

const SERIES: [&str; 6] = [
    "port_ut1",
    "port_it1",
    "port_ut2",
    "port_it2",
    "port_resist_1",
    "port_resist_2",
];

fn run(mut sim: Simulation) -> Record {
    let res = sim.run().unwrap();
    record(&sim, &res, &SERIES)
}

/// One port driven per run on one compiled model, against a fresh compile
/// of the model that has only that port's excitation.
fn one_port_per_run(lossy: bool) {
    let opts = CompileOptions::default();
    let compiled = CompiledModel::new(&board(lossy, &[1, 2]), &fdtd(), &opts).unwrap();
    assert_eq!(compiled.excitations(), ["port_excite_1", "port_excite_2"]);
    let mut first = None;
    // Port 1, port 2, and port 1 again after the other run.
    for n in [1, 2, 1] {
        let name = format!("port_excite_{n}");
        let sim = compiled.simulation_driving(&[name.as_str()]).unwrap();
        assert_eq!(sim.sources().excitations.len(), 1);
        let shared = run(sim);
        let fresh = run(Simulation::with_options(&board(lossy, &[n]), &fdtd(), &opts).unwrap());
        assert!(shared == fresh, "port {n} (lossy: {lossy}) differs");
        if n == 1 {
            match &first {
                None => first = Some(shared),
                Some(f) => assert!(*f == shared, "the second run of port 1 differs"),
            }
        } else {
            assert!(
                first.as_ref() != Some(&shared),
                "ports 1 and 2 gave one result"
            );
        }
    }
}

#[test]
fn one_port_per_run_lossy_board() {
    one_port_per_run(true);
}

#[test]
fn one_port_per_run_perfect_copper() {
    one_port_per_run(false);
}

/// All excitations driven: `simulation` (twice) and `into_simulation`
/// against `Simulation::with_options`, warnings included.
#[test]
fn every_excitation_is_with_options() {
    let (model, opts) = (board(true, &[1, 2]), CompileOptions::default());
    let fresh = Simulation::with_options(&model, &fdtd(), &opts).unwrap();
    let compiled = CompiledModel::new(&model, &fdtd(), &opts).unwrap();
    let again = compiled.simulation().unwrap();
    assert_eq!(fresh.warnings(), again.warnings());
    // (Unset lumped values are NaN: compared as text.)
    assert!(format!("{:?}", fresh.sources()) == format!("{:?}", again.sources()));
    let fresh = run(fresh);
    assert!(fresh == run(again));
    assert!(fresh == run(compiled.simulation().unwrap()));
    assert!(fresh == run(compiled.into_simulation().unwrap()));
}

#[test]
fn undriven_and_unknown_excitations() {
    let opts = CompileOptions::default();
    let compiled = CompiledModel::new(&board(false, &[1, 2]), &fdtd(), &opts).unwrap();
    // Nothing driven: the fields stay zero.
    let mut sim = compiled.simulation_driving(&[]).unwrap();
    sim.step(50);
    assert!(
        Axis::ALL
            .iter()
            .all(|&a| sim.engine().u_component(a).iter().all(|v| *v == 0.0))
    );
    let err = compiled.simulation_driving(&["port_excite_3"]).unwrap_err();
    assert!(err.to_string().contains("port_excite_3"), "{err}");
}
