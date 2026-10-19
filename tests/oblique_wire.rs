//! Oblique thin wires ([`ems::engine::oblique_wire`]): energy and
//! stability over a long run at the Courant limit, thread determinism,
//! the csx mapping, and a coarse tilted dipole against the same dipole
//! along z.
//!
//! The dipole accuracy study (several orientations against the
//! axis-aligned material model) is in `tests/oblique_wire_dipole.rs`.

mod common;

use common::*;
use csx::primitives::{BoxShape, Wire};
use csx::{ContinuousStructure, Primitive, Property};
use ems::compile::{CompileOptions, compile};
use ems::engine::oblique_wire::{ObliqueWire, ObliqueWires};
use ems::engine::{EdgeMaterials, EdgeSource, Engine, Grid, TimeStep};
use ems::port::LumpedPort;
use ems::{Axis, Fdtd, Simulation, Waveform};
use ems::{Boundaries, Boundary, EndCriteria};

const MM: f64 = 1e-3;

fn mm(p: [f64; 3]) -> [f64; 3] {
    p.map(|v| v * MM)
}

/// Closed PEC box of 1 mm cells with a bent wire, a junction, a wire
/// standing on the floor (connected end) with a lossless feed (no
/// resistance, no source), and very thin wires: after a soft source the
/// energy (fields plus wire state) stays constant over 15 000 steps at
/// Courant 0.999, and 1 and 4 threads agree bit for bit.
#[test]
fn energy_stability_and_threads() {
    let _t = Timer::start("oblique energy_stability_and_threads");
    let run = |threads: usize| {
        let g = Grid::uniform([16, 16, 24], [MM; 3]).unwrap();
        let m = EdgeMaterials::vacuum(&g);
        let r0 = 1e-3 * MM;
        let wires = [
            ObliqueWire::new(
                [
                    mm([4.3, 5.1, 4.2]),
                    mm([8.2, 8.7, 12.1]),
                    mm([11.6, 9.4, 19.3]),
                ],
                r0,
            ),
            ObliqueWire::new([mm([8.2, 8.7, 12.1]), mm([12.4, 4.2, 10.5])], 0.05 * MM),
            ObliqueWire::new(
                [
                    mm([4.5, 12.0, 0.0]),
                    mm([5.0, 11.5, 1.5]),
                    mm([6.1, 10.2, 8.3]),
                ],
                r0,
            )
            .feed(0, 0.0),
        ];
        let nz = g.nodes()[2];
        let net = ObliqueWires::new(&g, &m, &wires, |a, n| a != Axis::Z && n[2] == 0).unwrap();
        let conn: Vec<usize> = (0..net.nodes().len())
            .filter(|&j| net.connected_nodes()[j])
            .collect();
        assert_eq!(conn.len(), 1, "one connected end");
        assert_eq!(net.nodes()[conn[0]], mm([4.5, 12.0, 0.0]));
        assert!(nz > 0);
        let mut e = Engine::builder(g)
            .edge_materials(m)
            .threads(threads)
            .timestep(TimeStep::Courant(0.999))
            .build()
            .unwrap();
        let h = e.add_operator(net);
        e.add_operator(
            EdgeSource::new(sine_pulse(20e9, 15e9))
                .edge(Axis::Z, [8, 8, 11], 1.0)
                .edge(Axis::X, [5, 11, 4], 0.5),
        );
        e.run(400); // source done
        let w0 = e.energy();
        let ww = e.operator(&h).wire_energy();
        let (mut wmin, mut wmax) = (w0, w0);
        for _ in 0..30 {
            e.run(500);
            let w = e.energy();
            wmin = wmin.min(w);
            wmax = wmax.max(w);
        }
        let peak = Axis::ALL
            .iter()
            .flat_map(|&a| e.u_component(a))
            .fold(0f32, |m, v| m.max(v.abs()));
        let cur = e.operator(&h).currents().to_vec();
        (w0, ww, wmin, wmax, peak, e.u_component(Axis::X), cur)
    };
    let (w0, ww, wmin, wmax, peak, ux1, c1) = run(1);
    eprintln!(
        "energy {w0:e} (wire state {ww:e}), drift [{:.1e}, {:.1e}] over 15000 steps, \
         peak u {peak:e}",
        wmin / w0 - 1.0,
        wmax / w0 - 1.0
    );
    assert!(w0 > 0.0 && ww > 1e-3 * w0 && peak.is_finite());
    assert!((wmax / w0 - 1.0).abs() < 1e-5 && (wmin / w0 - 1.0).abs() < 1e-5);
    let (.., ux4, c4) = run(4);
    assert!(
        ux1.iter()
            .zip(&ux4)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(c1.iter().zip(&c4).all(|(a, b)| a.to_bits() == b.to_bits()));
}

const GAP: f64 = 5.0;
const TIP: f64 = 85.0;

/// Resonance (`X = 0` from below), resistance there, and dX/df [Ω/GHz].
fn resonance(freqs: &[f64], z: &[ems::Complex64]) -> (f64, f64, f64) {
    let k = (1..freqs.len())
        .find(|&k| z[k - 1].im < 0.0 && z[k].im >= 0.0)
        .expect("resonance in band");
    let t = -z[k - 1].im / (z[k].im - z[k - 1].im);
    let f = freqs[k - 1] + t * (freqs[k] - freqs[k - 1]);
    let r = z[k - 1].re + t * (z[k].re - z[k - 1].re);
    let s = (z[k + 10].im - z[k - 10].im) / ((freqs[k + 10] - freqs[k - 10]) / 1e9);
    (f, r, s)
}

/// 170 mm dipole of radius `a` [mm] along the unit vector `t`, delta-gap
/// fed (50 Ω), on a uniform mesh of `d` mm cells with `margin` mm around.
fn dipole(t: [f64; 3], a: f64, d: f64, margin: f64) -> (f64, f64, f64) {
    let n = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    let t = t.map(|v| v / n);
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = MM;
    for (ax, tv) in t.iter().enumerate() {
        let k = ((TIP * tv.abs() + margin) / d).ceil() as i64;
        model
            .grid
            .set_lines(ax, (-k..=k).map(|i| i as f64 * d))
            .unwrap();
    }
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(0.9e9, 0.45e9))
        .boundaries(Boundaries::all(Boundary::pml(6)))
        .end_criteria(EndCriteria::energy_decay_db(-30.0))
        .max_timesteps(20_000);
    let mut sim = Simulation::new(&model, &fdtd).unwrap();
    let p = |s: f64| mm(t.map(|v| v * s));
    let w = ObliqueWire::new([p(-TIP), p(-GAP), p(GAP), p(TIP)], a * MM)
        .feed(1, 50.0)
        .excite(Waveform::gaussian(0.9e9, 0.45e9), 1.0);
    let grid = sim.grid().clone();
    let net = ObliqueWires::new(&grid, &EdgeMaterials::vacuum(&grid), &[w], |_, _| false).unwrap();
    let h = sim.engine_mut().add_operator(net);
    sim.run().unwrap();
    let freqs: Vec<f64> = (0..=300).map(|k| 0.7e9 + k as f64 * 1e6).collect();
    let z = sim.engine().operator(&h).feed_impedance(0, &freqs);
    resonance(&freqs, &z)
}

/// A coarse (20 mm cells) dipole tilted by 45° in xz and in 3-D: within
/// 1.5 % of the same wire along z in resonance, resistance and reactance
/// slope.
#[test]
fn tilted_dipole_matches_axial() {
    let _t = Timer::start("oblique tilted_dipole_matches_axial");
    let a = 0.5;
    let (f0, r0, s0) = dipole([0.0, 0.0, 1.0], a, 20.0, 210.0);
    eprintln!(
        "along z: f_res {:.4} GHz, R {r0:.2} Ω, dX/df {s0:.0} Ω/GHz",
        f0 / 1e9
    );
    for t in [[1.0, 0.0, 1.0], [1.0, 2.0, 3.0]] {
        let (f, r, s) = dipole(t, a, 20.0, 210.0);
        eprintln!(
            "along {t:?}: f_res {:.4} GHz ({:+.2} %), R {r:.2} Ω ({:+.2} %), dX/df {s:.0} \
             ({:+.2} %)",
            f / 1e9,
            100.0 * (f / f0 - 1.0),
            100.0 * (r / r0 - 1.0),
            100.0 * (s / s0 - 1.0)
        );
        assert!((f / f0 - 1.0).abs() < 0.015);
        assert!((r / r0 - 1.0).abs() < 0.03);
        assert!((s / s0 - 1.0).abs() < 0.03);
    }
}

/// csx `Wire`s with oblique segments become oblique thin wires (no csx
/// staircase, no warning); ends on a lumped port and on metal are
/// connected; too thick a wire keeps the staircase with a warning.
#[test]
fn csx_mapping() {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = MM;
    for a in 0..3 {
        m.grid.set_lines(a, (0..=20).map(f64::from)).unwrap();
    }
    let w = m.add_property(Property::metal("wire"));
    // A V-shaped dipole fed by a lumped port on the z edge (10,10,9..10).
    m.add_primitive(w, Wire::new([[10.0, 10.0, 10.0], [13.0, 11.0, 16.0]], 0.02))
        .unwrap();
    m.add_primitive(w, Wire::new([[10.0, 10.0, 9.0], [7.5, 9.0, 3.5]], 0.02))
        .unwrap();
    // A wire standing on a plate at z = 2.
    let plate = m.add_property(Property::metal("plate"));
    m.add_primitive(
        plate,
        Primitive::new(BoxShape::new([1.0, 1.0, 2.0], [6.0, 6.0, 2.0])),
    )
    .unwrap();
    m.add_primitive(w, Wire::new([[3.3, 3.6, 2.0], [4.1, 5.2, 7.7]], 0.02))
        .unwrap();
    // Too thick for the model (radius 0.4 cells).
    m.add_primitive(w, Wire::new([[15.0, 3.0, 3.0], [17.0, 6.0, 9.0]], 0.4))
        .unwrap();
    let port = LumpedPort::new(1, 50.0, [10.0, 10.0, 9.0], [10.0, 10.0, 10.0], Axis::Z)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let fdtd = Fdtd::new()
        .threads(1)
        .waveform(Waveform::gaussian(5e9, 4e9))
        .max_timesteps(300)
        .no_end_criteria();
    let c = compile(&m, &fdtd, &CompileOptions::default()).unwrap();
    let ws: Vec<&String> = c
        .warnings
        .iter()
        .filter(|w| !w.starts_with("mesh"))
        .collect();
    assert!(ws.iter().any(|w| w.contains("too large")), "{ws:?}");
    assert!(ws.iter().all(|w| w.contains("primitive #3")), "{ws:?}");
    let net = c.oblique_wires.as_ref().expect("oblique wires");
    let nodes = net.nodes();
    let conn: Vec<[f64; 3]> = (0..nodes.len())
        .filter(|&j| net.connected_nodes()[j])
        .map(|j| nodes[j].map(|v| (v / MM * 1e6).round() / 1e6))
        .collect();
    assert_eq!(
        conn,
        vec![[10.0, 10.0, 10.0], [10.0, 10.0, 9.0], [3.3, 3.6, 2.0]],
        "connected ends"
    );
    assert_eq!(net.segment_wires().iter().max(), Some(&2));
    // The thin oblique wires leave no csx staircase; the thick one does.
    let pec: usize = Axis::ALL
        .iter()
        .map(|&a| c.materials.pec_mask(a).iter().filter(|&&p| p).count())
        .sum();
    let plate_edges = 2 * 5 * 6;
    assert!(pec > plate_edges, "thick wire staircase");
    let mut m2 = m.clone();
    let wp = m2.property_id("wire").unwrap();
    m2.property_mut(wp).unwrap().primitives.truncate(3);
    let c2 = compile(&m2, &fdtd, &CompileOptions::default()).unwrap();
    let pec2: usize = Axis::ALL
        .iter()
        .map(|&a| c2.materials.pec_mask(a).iter().filter(|&&p| p).count())
        .sum();
    assert_eq!(pec2, plate_edges);
    // It runs, and the port sees the wire.
    let mut s = Simulation::new(&m2, &fdtd).unwrap();
    let z = port.analyze(&s.run().unwrap(), &[4e9]).unwrap().z_in()[0];
    assert!(z.re.is_finite() && z.re > 0.0, "{z:?}");
}
