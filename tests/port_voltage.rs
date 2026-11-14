//! The voltage of a multi-column lumped port: the centre column (default)
//! against the mean over the columns
//! ([`ems::port::LumpedPort::average_voltage`]).
//!
//! A microstrip 2 mm wide and 1.6 mm above its ground plane on a 0.5 mm
//! mesh, a lumped port of 5 columns under each end, port 1 driven. The
//! mean voltage and the element current satisfy the port's circuit
//! equation `V = V_s − R·I` exactly, so the undriven port 2 has no
//! incident wave at its own resistance; the centre column does not, as far
//! as the columns differ. The test states what the option changes, not
//! which analysis is closer to the structure's S-parameters.

use csx::primitives::BoxShape;
use csx::{ContinuousStructure, Primitive, Property};
use ems::port::LumpedPort;
use ems::{Axis, Boundaries, Boundary, Cpml, EndCriteria, Fdtd, Simulation, Waveform};

const R: f64 = 50.0;

/// The line with its two ports; `average`: with the column probes.
fn line(average: bool) -> (ContinuousStructure, Vec<LumpedPort>) {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    m.grid
        .set_lines(0, (0..=48).map(|i| i as f64 * 0.5))
        .unwrap();
    m.grid
        .set_lines(1, (0..=28).map(|i| i as f64 * 0.5))
        .unwrap();
    m.grid
        .set_lines(2, (0..=30).map(|i| i as f64 * 0.4))
        .unwrap();
    let id = m.add_property(Property::material("laminate", csx::Material::new(4.3, 0.0)));
    let slab = BoxShape::new([0.0, 0.0, 4.8], [24.0, 14.0, 6.4]);
    m.add_primitive(id, Primitive::new(slab)).unwrap();
    for (name, start, stop) in [
        ("ground", [0.0, 0.0, 4.8], [24.0, 14.0, 4.8]),
        ("strip", [6.0, 6.0, 6.4], [18.0, 8.0, 6.4]),
    ] {
        let id = m.add_property(Property::metal(name));
        let sheet = Primitive::new(BoxShape::new(start, stop)).with_priority(10);
        m.add_primitive(id, sheet).unwrap();
    }
    let ports = [(1, 6.0), (2, 18.0)]
        .into_iter()
        .map(|(n, x)| {
            let mut port = LumpedPort::new(n, R, [x, 6.0, 6.4], [x, 8.0, 4.8], Axis::Z)
                .priority(30)
                .average_voltage(average);
            if n == 1 {
                port = port.excite(1.0);
            }
            port.add_to(&mut m).unwrap()
        })
        .collect();
    (m, ports)
}

fn fdtd() -> Fdtd {
    Fdtd::new()
        .waveform(Waveform::gaussian(4e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pml(Cpml::cells(8))))
        .end_criteria(EndCriteria::energy_decay_db(-50.0))
        .max_timesteps(20_000)
        .threads(1)
}

#[test]
fn mean_over_the_columns_against_the_centre_column() {
    let (model, mean_ports) = line(true);
    let (plain_model, centre_ports) = line(false);
    let columns = &mean_ports[0].properties().unwrap().voltage_columns;
    assert_eq!(columns.len(), 5);
    assert_eq!(columns[0].0, "port_ut1_col0");
    assert!(
        centre_ports[0]
            .properties()
            .unwrap()
            .voltage_columns
            .is_empty()
    );

    let mut sim = Simulation::new(&model, &fdtd()).unwrap();
    let res = sim.run().unwrap();
    let outcome = res.outcome().unwrap();
    assert!(
        matches!(outcome.reason, ems::engine::StopReason::EnergyDecay { .. }),
        "{outcome:?}"
    );

    // The option only adds probes: the default model's run records the
    // same centre voltage and element current, bit for bit.
    let mut plain = Simulation::new(&plain_model, &fdtd()).unwrap();
    let plain_res = plain.run().unwrap();
    for name in ["port_ut1", "port_ut2", "port_it2"] {
        let (a, b) = (res.probe(name).unwrap(), plain_res.probe(name).unwrap());
        assert!(a.scalar().unwrap() == b.scalar().unwrap(), "{name}");
    }
    let current = |r: &ems::Results| {
        r.lumped_current("port_resist_2")
            .unwrap()
            .scalar()
            .unwrap()
            .to_vec()
    };
    assert!(current(&res) == current(&plain_res));

    // In time: the mean voltage of the undriven port is R times its
    // element current (taken half a step apart, as the element's
    // trapezoidal update takes them); the centre column's is not.
    let i = current(&res);
    let series = |name: &str| res.probe(name).unwrap().scalar().unwrap().to_vec();
    let cols: Vec<Vec<f64>> = (0..5)
        .map(|k| series(&format!("port_ut2_col{k}")))
        .collect();
    let mean: Vec<f64> = (0..cols[0].len())
        .map(|n| cols.iter().map(|c| c[n]).sum::<f64>() / 5.0)
        .collect();
    let centre = series("port_ut2");
    assert!(centre == cols[2], "the centre column is the default probe");
    // The port runs from the strip (start, +) down to the ground along −z:
    // the element current along +z is −V/R.
    let residual = |v: &[f64]| {
        let peak = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let worst = (0..i.len().min(v.len()) - 1)
            .map(|n| (0.5 * (v[n] + v[n + 1]) + R * i[n + 1]).abs())
            .fold(0.0f64, f64::max);
        worst / peak
    };
    let (res_mean, res_centre) = (residual(&mean), residual(&centre));
    println!("time-domain |V + R i| / max|V|: mean {res_mean:.2e}, centre {res_centre:.2e}");
    assert!(res_mean < 1e-12, "{res_mean}");
    assert!(res_centre > 1e-2, "{res_centre}");

    // In frequency: the incident wave of the undriven port, and how far
    // the two analyses are apart.
    let freqs: Vec<f64> = (1..=8).map(|k| k as f64 * 1e9).collect();
    let analyze = |ports: &[LumpedPort]| {
        let p: Vec<_> = ports
            .iter()
            .map(|p| p.analyze(&res, &freqs).unwrap())
            .collect();
        let (a1, b1, a2, b2) = (p[0].a(), p[0].b(), p[1].a(), p[1].b());
        let s11: Vec<f64> = (0..freqs.len()).map(|k| (b1[k] / a1[k]).norm()).collect();
        let s21: Vec<f64> = (0..freqs.len()).map(|k| (b2[k] / a1[k]).norm()).collect();
        let idle: Vec<f64> = (0..freqs.len()).map(|k| (a2[k] / b2[k]).norm()).collect();
        (s11, s21, idle)
    };
    let (m11, m21, m_idle) = analyze(&mean_ports);
    let (c11, c21, c_idle) = analyze(&centre_ports);
    let max = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(*x));
    let min = |v: &[f64]| v.iter().fold(f64::INFINITY, |m, x| m.min(*x));
    let apart = |a: &[f64], b: &[f64]| {
        max(&a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .collect::<Vec<_>>())
    };
    println!("|a2/b2| mean: {m_idle:.6?}");
    println!("|a2/b2| centre: {c_idle:.6?}");
    println!("|S11| mean {m11:.4?}\n|S11| centre {c11:.4?}");
    println!("|S21| mean {m21:.4?}\n|S21| centre {c21:.4?}");
    println!(
        "apart: |S11| {:.4}, |S21| {:.4}",
        apart(&m11, &c11),
        apart(&m21, &c21)
    );
    // Measured: |a2/b2| 2e-6 … 1.2e-4 (mean; the half step between V and
    // I) against 0.003 … 0.024 (centre) from 1 to 8 GHz; the analyses are
    // 0.0195 apart in |S11| and 0.0087 in |S21| at most.
    assert!(max(&m_idle) < 3e-4, "{m_idle:?}");
    assert!(min(&c_idle) > 2e-3 && max(&c_idle) > 2e-2, "{c_idle:?}");
    assert!((0.015..0.025).contains(&apart(&m11, &c11)));
    assert!((0.005..0.012).contains(&apart(&m21, &c21)));
}
