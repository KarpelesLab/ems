//! Oblique thin wires on cylindrical grids (design §5.14.1, §5.15):
//! the energy of a closed annular cavity with wires across the α = 0 seam
//! stays constant at the time-step limit (1 vs 3 threads bit-identical),
//! and a 170 mm dipole (radius 0.5 mm, 50 Ω delta gap) on a 20 mm
//! `(ρ, α, z)` grid around `x = R0` matches the same dipole on a 20 mm
//! Cartesian grid.

mod common;

use std::f64::consts::TAU;

use common::*;
use csx::primitives::Wire;
use csx::{ContinuousStructure, CoordSystem, Primitive, Property};
use ems::compile::{CompileOptions, compile};
use ems::engine::oblique_wire::{ObliqueWire, ObliqueWires};
use ems::engine::{EdgeMaterials, EdgeSource, Engine, Grid, TimeStep};
use ems::{Axis, Boundaries, Boundary, EndCriteria, Fdtd, Simulation, Waveform};

const MM: f64 = 1e-3;

fn mm(p: [f64; 3]) -> [f64; 3] {
    p.map(|v| v * MM)
}

fn lin(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..=n)
        .map(|i| lo + (hi - lo) * i as f64 / n as f64)
        .collect()
}

/// Closed annular PEC cavity (ρ 10…26 mm, closed azimuth of 24 cells,
/// z 0…24 mm, 1 mm cells radially and in z) with a bent wire across the
/// α = 0 seam, a junction, a wire standing on the floor (connected end,
/// lossless feed) and a very thin wire: after a soft source the energy
/// (fields plus wire state) stays constant over 10 000 steps at Courant
/// 0.999, and 1 and 3 threads agree bit for bit.
#[test]
fn energy_stability_and_threads_across_the_seam() {
    let run = |threads: usize| {
        let g = Grid::cylindrical([
            lin(10.0 * MM, 26.0 * MM, 16),
            lin(0.0, TAU, 24),
            lin(0.0, 24.0 * MM, 24),
        ])
        .unwrap();
        let m = EdgeMaterials::vacuum(&g);
        let r0 = 1e-3 * MM;
        let wires = [
            ObliqueWire::new(
                [
                    mm([17.0, -4.0, 4.2]),
                    mm([18.5, 3.1, 12.1]),
                    mm([14.2, 9.4, 19.3]),
                ],
                r0,
            ),
            ObliqueWire::new([mm([18.5, 3.1, 12.1]), mm([22.0, -2.5, 10.5])], 0.05 * MM),
            ObliqueWire::new(
                [
                    mm([-15.0, 6.0, 0.0]),
                    mm([-14.0, 7.0, 1.5]),
                    mm([-16.1, 3.2, 8.3]),
                ],
                r0,
            )
            .feed(0, 0.0),
        ];
        let net = ObliqueWires::new(&g, &m, &wires, |a, n| a != Axis::Z && n[2] == 0).unwrap();
        let conn = (0..net.nodes().len())
            .filter(|&j| net.connected_nodes()[j])
            .count();
        assert_eq!(conn, 1, "one connected end");
        let mut e = Engine::builder(g)
            .edge_materials(m)
            .threads(threads)
            .timestep(TimeStep::Courant(0.999))
            .build()
            .unwrap();
        let h = e.add_operator(net);
        e.add_operator(
            EdgeSource::new(sine_pulse(20e9, 15e9))
                .edge(Axis::Z, [8, 1, 11], 1.0)
                .edge(Axis::X, [5, 20, 4], 0.5),
        );
        e.run(400);
        let w0 = e.energy();
        let ww = e.operator(&h).wire_energy();
        let (mut wmin, mut wmax) = (w0, w0);
        for _ in 0..20 {
            e.run(500);
            let w = e.energy();
            wmin = wmin.min(w);
            wmax = wmax.max(w);
        }
        let ux = e.u_component(Axis::X).to_vec();
        let cur = e.operator(&h).currents().to_vec();
        (w0, ww, wmin, wmax, ux, cur)
    };
    let (w0, ww, wmin, wmax, ux1, c1) = run(1);
    eprintln!(
        "cylindrical cavity: energy {w0:e} (wire state {ww:e}), drift [{:.1e}, {:.1e}] over \
         10000 steps",
        wmin / w0 - 1.0,
        wmax / w0 - 1.0
    );
    assert!(w0 > 0.0 && ww > 1e-4 * w0);
    assert!((wmax / w0 - 1.0).abs() < 1e-5 && (wmin / w0 - 1.0).abs() < 1e-5);
    let (.., ux3, c3) = run(3);
    assert!(
        ux1.iter()
            .zip(&ux3)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(c1.iter().zip(&c3).all(|(a, b)| a.to_bits() == b.to_bits()));
}

const GAP: f64 = 5.0;
const TIP: f64 = 85.0;
const D: f64 = 20.0;
const MARGIN: f64 = 210.0;

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

fn fdtd() -> Fdtd {
    Fdtd::new()
        .waveform(Waveform::gaussian(0.9e9, 0.45e9))
        .boundaries(Boundaries::all(Boundary::pml(6)))
        .end_criteria(EndCriteria::energy_decay_db(-30.0))
        .max_timesteps(20_000)
}

fn unit(t: [f64; 3]) -> [f64; 3] {
    let n = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    t.map(|v| v / n)
}

/// The dipole along `t` through `centre` [mm] on `model`'s grid.
fn feed_z(model: &ContinuousStructure, t: [f64; 3], centre: [f64; 3]) -> (f64, f64, f64) {
    let mut sim = Simulation::new(model, &fdtd()).unwrap();
    let p = |s: f64| mm(core::array::from_fn(|k| centre[k] + s * t[k]));
    let w = ObliqueWire::new([p(-TIP), p(-GAP), p(GAP), p(TIP)], 0.5 * MM)
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

/// The Cartesian reference: the dipole along `t` through the origin (a
/// node) on 20 mm cells.
fn cartesian(t: [f64; 3]) -> (f64, f64, f64) {
    let t = unit(t);
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = MM;
    for (ax, tv) in t.iter().enumerate() {
        let k = ((TIP * tv.abs() + MARGIN) / D).ceil() as i64;
        m.grid
            .set_lines(ax, (-k..=k).map(|i| i as f64 * D))
            .unwrap();
    }
    feed_z(&m, t, [0.0; 3])
}

/// The dipole along `t` through `(R0, 0, 0)` on a `(ρ, α, z)` grid of
/// 20 mm cells (`R0·Δα` = 20 mm), a sector around it.
fn cylindrical(t: [f64; 3], r0: f64) -> (f64, f64, f64) {
    let t = unit(t);
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = MM;
    let ext = |k: usize| TIP * t[k].abs() + MARGIN;
    let nr = (ext(0) / D).ceil() as i64;
    m.grid
        .set_lines(0, (-nr..=nr).map(|i| r0 + i as f64 * D))
        .unwrap();
    let da = D / r0;
    let na = (ext(1) / D).ceil() as i64;
    m.grid
        .set_lines(1, (-na..=na).map(|j| j as f64 * da))
        .unwrap();
    let nz = (ext(2) / D).ceil() as i64;
    m.grid
        .set_lines(2, (-nz..=nz).map(|k| k as f64 * D))
        .unwrap();
    feed_z(&m, t, [r0, 0.0, 0.0])
}

/// Along z, tilted 45° in the ρz plane and along (1, 2, 3) through
/// `x = R0` (a node): against the same wire on the Cartesian grid. At
/// `R0` = 600 mm (the correction window, 5 cells, is a sixth of `R0`)
/// within 0.3 % (resonance), 1 % (resistance) and 1.5 % (reactance
/// slope); at 1200 mm (nearly Cartesian) alike.
#[test]
fn dipole_on_a_cylindrical_grid_matches_cartesian() {
    for t in [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 2.0, 3.0]] {
        let (f0, r0, s0) = cartesian(t);
        eprintln!(
            "Cartesian, along {t:?}: f_res {:.4} GHz, R {r0:.2} Ω, dX/df {s0:.0} Ω/GHz",
            f0 / 1e9
        );
        let radii: &[f64] = if t[1] == 2.0 {
            &[600.0, 1200.0]
        } else {
            &[600.0]
        };
        for &rc in radii {
            let (f, r, s) = cylindrical(t, rc);
            eprintln!(
                "  cylindrical, R0 = {rc} mm: f_res {:.4} GHz ({:+.2} %), R {r:.2} Ω ({:+.2} %), \
                 dX/df {s:.0} ({:+.2} %)",
                f / 1e9,
                100.0 * (f / f0 - 1.0),
                100.0 * (r / r0 - 1.0),
                100.0 * (s / s0 - 1.0)
            );
            assert!((f / f0 - 1.0).abs() < 0.003, "{t:?} at {rc}");
            assert!((r / r0 - 1.0).abs() < 0.01, "{t:?} at {rc}");
            assert!((s / s0 - 1.0).abs() < 0.015, "{t:?} at {rc}");
        }
    }
}

/// csx `Wire`s on a cylindrical grid go to the oblique model (no
/// staircase, no warning), whatever their direction.
#[test]
fn csx_wires_on_a_cylindrical_grid() {
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = MM;
    m.grid.set_lines(0, lin(10.0, 30.0, 20)).unwrap();
    m.grid.set_lines(1, lin(0.0, TAU, 64)).unwrap();
    m.grid.set_lines(2, lin(0.0, 20.0, 20)).unwrap();
    let wire = m.add_property(Property::metal("wire"));
    for (a, b) in [
        ([20.0, 0.0, 5.0], [20.0, 0.0, 15.0]),
        ([14.0, 3.0, 4.0], [22.0, -6.0, 16.0]),
    ] {
        m.add_primitive(
            wire,
            Primitive::new(Wire::new([a, b], 0.05)).with_coord_system(CoordSystem::Cartesian),
        )
        .unwrap();
    }
    let c = compile(&m, &Fdtd::new(), &CompileOptions::default()).unwrap();
    assert!(
        c.warnings.iter().all(|w| w.starts_with("mesh ")),
        "{:?}",
        c.warnings
    );
    let ow = c.oblique_wires.as_ref().expect("oblique wires");
    assert!(ow.segments() > 10);
    assert!(
        Axis::ALL
            .iter()
            .all(|&a| !c.materials.pec_mask(a).iter().any(|&p| p))
    );
}
