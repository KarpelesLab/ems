//! Validation of the thin-wire model ([`ems::engine::wire`]).
//!
//! * **Coaxial line** (TEM): a wire on the axis of a rectangular PEC tube.
//!   The per-unit-length inductance and capacitance of the wire follow
//!   `ln(1/r0)`, so the characteristic impedance changes with the radius
//!   by exactly `Z0(r0) − Z0(staircase) = η0/(2π)·ln(r_eq/r0)` on square
//!   cells, while the wave keeps travelling at the speed of the Yee line.
//!   The absolute impedance is compared with the closed form for a round
//!   wire in a square tube of side `D`, `Z0 = η0/(2π)·ln(1.0787·D/d)`
//!   (Wheeler).
//! * **Energy and stability** over a long run, and thread determinism.
//! * **csx mapping**: `Wire` primitives on metal become thin wires, feed
//!   gaps join them, unsupported cases warn, XML round trip.

mod common;

use common::*;
use csx::primitives::Wire;
use csx::{ContinuousStructure, Primitive, Property};
use ems::compile::{CompileOptions, compile};
use ems::constants::Z0;
use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::wire::{ThinWire, lattice_radius};
use ems::engine::{EdgeMaterials, EdgeSource, Engine, Grid, Probe, ProbeQuantity, TimeStep};
use ems::port::LumpedPort;
use ems::post::dft;
use ems::{Axis, Cpml, Face};
use ems::{Fdtd, Simulation, Waveform};
use std::f64::consts::PI;

const DZ: f64 = 1e-3;
const NZ: usize = 260;
const SRC: usize = 10;
const P1: usize = 100;
const P2: usize = 110;

/// Result of one coax run: `|V/I|` per frequency at `P1`, and the
/// measured phase advance per cell between `P1` and `P2`.
struct Coax {
    z: Vec<f64>,
    k: Vec<f64>,
    dt: f64,
}

/// Tube of `nx × ny` cells of `hx × hy`, wire on node (`nx/2`, `ny/2`)
/// along z, CPML at z_max; `radius = None` is the plain staircase.
fn coax(nx: usize, ny: usize, hx: f64, hy: f64, radius: Option<f64>, freqs: &[f64]) -> Coax {
    let g = Grid::new([
        (0..=nx).map(|i| i as f64 * hx).collect(),
        (0..=ny).map(|j| j as f64 * hy).collect(),
        (0..=NZ).map(|k| k as f64 * DZ).collect(),
    ])
    .unwrap();
    let (iw, jw) = (nx / 2, ny / 2);
    let mut m = EdgeMaterials::vacuum(&g);
    let w = ThinWire::new(Axis::Z, [iw, jw, 0], NZ, radius.unwrap_or(1.0));
    w.set_pec(&mut m, &[]);
    if radius.is_some() {
        w.apply(&g, &mut m).unwrap();
    }
    let mut e = Engine::builder(g)
        .edge_materials(m)
        .threads(1)
        .build()
        .unwrap();
    e.add_operator(CpmlBoundary::new(&e, [(Face::ZMax, Cpml::cells(8))]).unwrap());
    let mut s = EdgeSource::new(sine_pulse(5e9, 2.5e9));
    for i in iw..nx {
        s = s.edge(Axis::X, [i, jw, SRC], 1.0);
    }
    e.add_operator(s);
    let volt = |k: usize| {
        (iw..nx).fold(Probe::new(ProbeQuantity::Voltage), |p, i| {
            p.entry(Axis::X, [i, jw, k], 1.0)
        })
    };
    let v1 = e.add_operator(volt(P1));
    let v2 = e.add_operator(volt(P2));
    let cur = e.add_operator(
        Probe::new(ProbeQuantity::Current)
            .entry(Axis::Y, [iw, jw, P1], 1.0)
            .entry(Axis::Y, [iw - 1, jw, P1], -1.0)
            .entry(Axis::X, [iw, jw, P1], -1.0)
            .entry(Axis::X, [iw, jw - 1, P1], 1.0),
    );
    e.run(2000);
    let dt = e.dt();
    let sv1 = dft(dt, dt, e.operator(&v1).values(), freqs);
    let sv2 = dft(dt, dt, e.operator(&v2).values(), freqs);
    let si = dft(0.5 * dt, dt, e.operator(&cur).values(), freqs);
    Coax {
        z: sv1.iter().zip(&si).map(|(v, i)| (*v / *i).norm()).collect(),
        k: sv1
            .iter()
            .zip(&sv2)
            .map(|(a, b)| -(*b / *a).arg() / (P2 - P1) as f64)
            .collect(),
        dt,
    }
}

fn freqs() -> Vec<f64> {
    linspace(2e9, 8e9, 7)
}

/// Mean and max relative deviation of `x` from `want`.
fn rel(x: &[f64], want: f64) -> (f64, f64) {
    let m = x.iter().sum::<f64>() / x.len() as f64;
    let d = x.iter().map(|v| (v / want - 1.0).abs()).fold(0.0, f64::max);
    (m, d)
}

#[test]
fn coax_impedance_follows_the_radius() {
    let _t = Timer::start("coax_impedance_follows_the_radius");
    let f = freqs();
    let h = 1e-3;
    let r_eq = lattice_radius(h, h);
    let eta = Z0 / (2.0 * PI);
    // Wheeler, square tube of side D = 10 mm, wire diameter d.
    let wheeler = |r: f64| eta * (1.0787 * 10.0 * h / (2.0 * r)).ln();
    let stair = coax(10, 10, h, h, None, &f);
    let (z_st, dev) = rel(&stair.z, stair.z[0]);
    eprintln!(
        "staircase: Z0 {z_st:.3} Ω (spread {dev:.1e}), Wheeler at r_eq {:.3} Ω",
        wheeler(r_eq)
    );
    assert!(dev < 2e-3, "staircase Z0 varies by {dev:e}");
    assert!((z_st / wheeler(r_eq) - 1.0).abs() < 5e-3);
    for div in [5.0, 50.0, 1000.0] {
        let r0 = r_eq / div;
        let c = coax(10, 10, h, h, Some(r0), &f);
        let want = z_st + eta * div.ln();
        let (z, dev) = rel(&c.z, want);
        // Wave speed: the 1-D Yee line.
        let kerr =
            c.k.iter()
                .zip(&f)
                .map(|(k, &f)| (k / yee_k(f, c.dt, DZ, 1.0, 0.0).re - 1.0).abs())
                .fold(0.0, f64::max);
        eprintln!(
            "r0 = r_eq/{div}: Z0 {z:.3} Ω, expected {want:.3} (max dev {dev:.1e}), \
             Wheeler {:.3}; phase constant vs Yee {kerr:.1e}",
            wheeler(r0)
        );
        assert!(dev < 2e-3, "Z0 {z} vs {want}");
        assert!((z / wheeler(r0) - 1.0).abs() < 5e-3);
        assert!(kerr < 1e-3, "phase constant off by {kerr:e}");
    }
}

#[test]
fn coax_rectangular_cells() {
    let _t = Timer::start("coax_rectangular_cells");
    let f = freqs();
    let (hx, hy) = (1e-3, 1.5e-3);
    let r_eq = lattice_radius(hx, hy);
    let eta = Z0 / (2.0 * PI);
    let stair = coax(12, 8, hx, hy, None, &f);
    let z_st = stair.z.iter().sum::<f64>() / stair.z.len() as f64;
    for div in [10.0, 100.0] {
        let c = coax(12, 8, hx, hy, Some(r_eq / div), &f);
        let want = z_st + eta * f64::ln(div);
        let (z, dev) = rel(&c.z, want);
        eprintln!("1 × 1.5 mm cells, r0 = r_eq/{div}: Z0 {z:.3} Ω vs {want:.3} ({dev:.1e})");
        assert!(dev < 1e-2, "Z0 {z} vs {want}");
    }
}

/// Closed PEC tube with a very thin wire: the energy invariant holds and
/// the fields stay bounded over a long run; 1 and 4 threads agree bit for
/// bit.
#[test]
fn energy_stability_and_threads() {
    let _t = Timer::start("energy_stability_and_threads");
    let run = |threads: usize| {
        let g = Grid::uniform([10, 10, 40], [1e-3; 3]).unwrap();
        let mut m = EdgeMaterials::vacuum(&g);
        let w = ThinWire::new(Axis::Z, [5, 5, 5], 30, lattice_radius(1e-3, 1e-3) / 1000.0);
        // A gap in the middle, as a dipole feed.
        w.set_pec(&mut m, &[15]);
        let gfac = w.apply(&g, &mut m).unwrap();
        assert!(gfac > 5.0);
        let mut e = Engine::builder(g)
            .edge_materials(m)
            .threads(threads)
            // At the Courant limit: the modified elements must not need a
            // smaller step.
            .timestep(TimeStep::Courant(0.999))
            .build()
            .unwrap();
        e.add_operator(
            EdgeSource::new(sine_pulse(8e9, 4e9))
                .edge(Axis::Z, [5, 5, 20], 1.0)
                .edge(Axis::X, [5, 5, 12], 0.5),
        );
        e.run(600); // source done
        let w0 = e.energy();
        let mut wmin = w0;
        let mut wmax = w0;
        for _ in 0..40 {
            e.run(500);
            let w = e.energy();
            wmin = wmin.min(w);
            wmax = wmax.max(w);
        }
        let peak = Axis::ALL
            .iter()
            .flat_map(|&a| e.u_component(a))
            .fold(0f32, |m, v| m.max(v.abs()));
        (w0, wmin, wmax, peak, e.u_component(Axis::X))
    };
    let (w0, wmin, wmax, peak, ux1) = run(1);
    eprintln!(
        "energy {w0:e}, drift [{:.1e}, {:.1e}] over 20000 steps, peak u {peak:e}",
        wmin / w0 - 1.0,
        wmax / w0 - 1.0
    );
    assert!(w0 > 0.0 && peak.is_finite());
    assert!((wmax / w0 - 1.0).abs() < 1e-5 && (wmin / w0 - 1.0).abs() < 1e-5);
    let (.., ux4) = run(4);
    assert!(
        ux1.iter()
            .zip(&ux4)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

/// 20³ cells of 1 mm (drawing unit mm), a dipole of two `Wire` arms
/// along z on (10, 10) with a lumped port in the one-cell gap, plus
/// `extra` primitives on a second metal property.
fn wire_model(radius: f64, extra: Vec<Primitive>) -> (ContinuousStructure, LumpedPort) {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for a in 0..3 {
        m.grid.set_lines(a, (0..=20).map(f64::from)).unwrap();
    }
    let w = m.add_property(Property::metal("wire"));
    for (a, b) in [(4.0, 9.0), (10.0, 16.0)] {
        m.add_primitive(w, Wire::new([[10.0, 10.0, a], [10.0, 10.0, b]], radius))
            .unwrap();
    }
    if !extra.is_empty() {
        let other = m.add_property(Property::metal("other"));
        for p in extra {
            m.add_primitive(other, p).unwrap();
        }
    }
    let port = LumpedPort::new(1, 50.0, [10.0, 10.0, 9.0], [10.0, 10.0, 10.0], Axis::Z)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    (m, port)
}

fn wire_fdtd() -> Fdtd {
    Fdtd::new()
        .threads(1)
        .waveform(Waveform::gaussian(5e9, 4e9))
        .max_timesteps(400)
        .no_end_criteria()
}

#[test]
fn csx_mapping() {
    let r_eq = lattice_radius(1.0, 1.0); // mm
    let (m, _) = wire_model(0.02, Vec::new());
    let c = compile(&m, &wire_fdtd(), &CompileOptions::default()).unwrap();
    let mesh = |w: &String| w.starts_with("mesh");
    assert!(c.warnings.iter().all(mesh), "{:?}", c.warnings);
    // One wire through the gap, 4..16.
    assert_eq!(c.thin_wires.len(), 1, "{:?}", c.thin_wires);
    let w = c.thin_wires[0];
    assert_eq!(
        (w.axis(), w.start(), w.segments()),
        (Axis::Z, [10, 10, 4], 12)
    );
    assert!((w.radius() - 0.02e-3).abs() < 1e-15);
    let g = w.factor(&c.grid).unwrap();
    assert!((g - (1.0 + 2.0 / PI * (r_eq / 0.02).ln())).abs() < 1e-9);
    // The gap edge is not PEC; the node above it is interior.
    assert!(!c.materials.is_pec_edge(Axis::Z, [10, 10, 9]));
    let i = c.grid.index([10, 10, 10]);
    assert!((c.materials.eps_r(Axis::X)[i] - 1.0 / g).abs() < 1e-12);

    // Unsupported pieces keep the staircase, with warnings; the oblique
    // wire goes to the oblique thin-wire model (tests/oblique_wire.rs).
    let extra = vec![
        Primitive::new(Wire::new([[3.0, 3.0, 3.0], [6.0, 7.0, 3.0]], 0.02)),
        Primitive::new(Wire::new([[3.5, 15.0, 2.0], [3.5, 15.0, 8.0]], 0.02)),
        Primitive::new(Wire::new([[15.0, 3.0, 2.0], [15.0, 3.0, 8.0]], 0.3)),
    ];
    let (m, _) = wire_model(0.02, extra);
    let c = compile(&m, &wire_fdtd(), &CompileOptions::default()).unwrap();
    let ws: Vec<&String> = c.warnings.iter().filter(|w| !mesh(w)).collect();
    for want in ["not on mesh lines", "not below the staircase radius"] {
        assert!(ws.iter().any(|w| w.contains(want)), "{want}: {ws:?}");
    }
    // Nothing about the oblique wire (primitive #0 of "other").
    assert!(ws.iter().all(|w| !w.contains("primitive #0")), "{ws:?}");
    assert_eq!(c.thin_wires.len(), 1);
    assert!(c.oblique_wires.as_ref().is_some_and(|w| w.segments() > 0));

    // The thin wire changes the port impedance against the staircase.
    let z = |r: f64| {
        let (m, port) = wire_model(r, Vec::new());
        let mut s = Simulation::new(&m, &wire_fdtd()).unwrap();
        port.analyze(&s.run().unwrap(), &[3e9]).unwrap().z_in()[0]
    };
    let (zt, zs) = (z(0.02), z(0.0));
    assert!((zt - zs).norm() > 1.0, "{zt:?} vs {zs:?}");
}

/// Through XML (csx writes `<Wire WireRadius>`): identical results.
#[cfg(feature = "xml")]
#[test]
fn xml_round_trip() {
    use csx::OpenEmsDocument;
    let (m, port) = wire_model(0.02, Vec::new());
    let mut doc = OpenEmsDocument {
        structure: m.clone(),
        ..Default::default()
    };
    // What an openEMS file means (Fdtd::from_csx sets both).
    let fdtd = wire_fdtd()
        .waveform(Waveform::openems_gaussian(5e9, 4e9))
        .timestep_method(ems::engine::TimestepMethod::Rennings)
        .openems_run(true);
    doc.set_fdtd_settings(&fdtd.to_csx());
    let mut a = Simulation::new(&m, &fdtd).unwrap();
    let mut b = Simulation::from_xml_str(&doc.to_xml_string()).unwrap();
    let (ra, rb) = (a.run().unwrap(), b.run().unwrap());
    assert_eq!(
        port.analyze(&ra, &[3e9]).unwrap().z_in(),
        port.analyze(&rb, &[3e9]).unwrap().z_in()
    );
}
