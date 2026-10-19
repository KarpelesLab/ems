//! Oblique thin-wire dipoles ([`ems::engine::oblique_wire`]), run with
//! `--ignored` in release.
//!
//! The same physical wire (170 mm tip to tip, radius `a` = 0.496 mm, a
//! 10 mm feed gap with a 50 Ω delta-gap source) along z, tilted in a
//! plane and tilted in 3-D, on a uniform 10 mm mesh. Resonance, input
//! resistance and reactance slope must match the axis-aligned material
//! model ([`ems::engine::wire`]: csx `Wire` arms on a mesh line plus a
//! lumped port, itself within 0.3 % of a fine staircase reference, see
//! `tests/thin_wire_dipole.rs`) and must not depend on the orientation
//! and position of the wire relative to the mesh.

use csx::TypeMask;
use csx::mesh::{MeshBuilder, MeshOptions};
use csx::primitives::{BoxShape, Wire};
use csx::{ContinuousStructure, Primitive, Property};
use ems::engine::oblique_wire::{ObliqueWire, ObliqueWires};
use ems::engine::wire::lattice_radius;
use ems::prelude::*;

const GAP: f64 = 5.0;
const TIP: f64 = 85.0;

#[derive(Debug)]
struct Res {
    f_res: f64,
    r_res: f64,
    slope: f64,
}

fn fdtd() -> Fdtd {
    Fdtd::new()
        .waveform(Waveform::gaussian(0.9e9, 0.45e9))
        .boundaries(Boundaries::all(Boundary::pml(8)))
        .end_criteria(EndCriteria::energy_decay_db(-40.0))
        .max_timesteps(60_000)
}

fn freqs() -> Vec<f64> {
    (0..=400).map(|k| 0.65e9 + k as f64 * 1e6).collect()
}

fn resonance(z: &[Complex64]) -> Res {
    let freqs = freqs();
    let k = (1..freqs.len())
        .find(|&k| z[k - 1].im < 0.0 && z[k].im >= 0.0)
        .expect("resonance in band");
    let t = -z[k - 1].im / (z[k].im - z[k - 1].im);
    let f_res = freqs[k - 1] + t * (freqs[k] - freqs[k - 1]);
    let r_res = z[k - 1].re + t * (z[k].re - z[k - 1].re);
    let slope = (z[k + 10].im - z[k - 10].im) / ((freqs[k + 10] - freqs[k - 10]) / 1e9);
    Res {
        f_res,
        r_res,
        slope,
    }
}

fn unit(t: [f64; 3]) -> [f64; 3] {
    let n = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    t.map(|v| v / n)
}

/// A uniform mesh of `d` mm cells with lines at `offset + k·d`, covering
/// the dipole along `t` through `centre` plus a 150 mm margin (PML
/// inside).
fn mesh(model: &mut ContinuousStructure, t: [f64; 3], centre: [f64; 3], d: f64, offset: [f64; 3]) {
    model.grid.delta_unit = 1e-3;
    for a in 0..3 {
        let ext = TIP * t[a].abs() + 150.0 + (centre[a] - offset[a]).abs();
        let k = (ext / d).ceil() as i64;
        model
            .grid
            .set_lines(a, (-k..=k).map(|i| offset[a] + i as f64 * d))
            .unwrap();
    }
}

/// Dipole along `t` through `centre` [mm], oblique model with segments of
/// at most `seg` mm (default: half a cell), on `d` mm cells.
fn oblique(t: [f64; 3], centre: [f64; 3], radius: f64, seg: Option<f64>, d: f64) -> Res {
    let t = unit(t);
    let mut model = ContinuousStructure::new();
    mesh(&mut model, t, centre, d, [0.0; 3]);
    let mut sim = Simulation::new(&model, &fdtd()).unwrap();
    let p = |s: f64| -> [f64; 3] { core::array::from_fn(|a| (centre[a] + s * t[a]) * 1e-3) };
    let mut w = ObliqueWire::new([p(-TIP), p(-GAP), p(GAP), p(TIP)], radius * 1e-3)
        .feed(1, 50.0)
        .excite(Waveform::gaussian(0.9e9, 0.45e9), 1.0);
    if let Some(l) = seg {
        w = w.max_segment(l * 1e-3);
    }
    let grid = sim.grid().clone();
    let mats = ems::EdgeMaterials::vacuum(&grid);
    let net = ObliqueWires::new(&grid, &mats, &[w], |_, _| false).unwrap();
    let h = sim.engine_mut().add_operator(net);
    sim.run().unwrap();
    resonance(&sim.engine().operator(&h).feed_impedance(0, &freqs()))
}

/// The axis-aligned material model on 10 mm cells, as in
/// `tests/thin_wire_dipole.rs`.
fn axis_aligned(radius: f64) -> Res {
    let d = 10.0;
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    let wire = model.add_property(Property::metal("wire"));
    let arms = [(GAP, TIP), (-TIP, -GAP)];
    for (a, b) in arms {
        let prim = Primitive::new(BoxShape::new([0.0, 0.0, a], [0.0, 0.0, b]));
        model.add_primitive(wire, prim).unwrap();
    }
    let port = LumpedPort::new(1, 50.0, [0.0, 0.0, -GAP], [0.0, 0.0, GAP], Axis::Z)
        .excite(1.0)
        .add_to(&mut model)
        .unwrap();
    let near = [-2.0 * d, -d, 0.0, d, 2.0 * d];
    let nz = (2.0 * TIP / d).round() as usize;
    MeshBuilder::new(
        MeshOptions::new(10.0)
            .with_ratio(1.4)
            .with_margin(150.0)
            .with_mask(TypeMask::PHYSICAL),
    )
    .with_lines(0, near)
    .with_lines(1, near)
    .with_lines(2, (0..=nz).map(|i| -TIP + i as f64 * d))
    .apply(&mut model)
    .unwrap();
    model.property_mut(wire).unwrap().primitives = arms
        .iter()
        .map(|&(a, b)| Primitive::new(Wire::new([[0.0, 0.0, a], [0.0, 0.0, b]], radius)))
        .collect();
    let mut sim = Simulation::new(&model, &fdtd()).unwrap();
    let res = sim.run().unwrap();
    resonance(&port.analyze(&res, &freqs()).unwrap().z_in())
}

/// Through csx: a lumped port on the z edge (0, 0, ±5 mm) and two csx
/// `Wire` arms from its terminals along ±`t` (so the arms are offset
/// from each other by the gap), 10 mm cells.
fn csx_tilted(t: [f64; 3], radius: f64) -> (Res, Vec<String>) {
    let t = unit(t);
    let mut model = ContinuousStructure::new();
    mesh(&mut model, t, [0.0; 3], 10.0, [0.0, 0.0, GAP]);
    let wire = model.add_property(Property::metal("wire"));
    let arm = TIP - GAP;
    for sgn in [1.0, -1.0] {
        let a = [0.0, 0.0, sgn * GAP];
        let b: [f64; 3] = core::array::from_fn(|k| a[k] + sgn * arm * t[k]);
        model
            .add_primitive(wire, Primitive::new(Wire::new([a, b], radius)))
            .unwrap();
    }
    let port = LumpedPort::new(1, 50.0, [0.0, 0.0, -GAP], [0.0, 0.0, GAP], Axis::Z)
        .excite(1.0)
        .add_to(&mut model)
        .unwrap();
    let mut sim = Simulation::new(&model, &fdtd()).unwrap();
    let warnings = sim.warnings().to_vec();
    let res = sim.run().unwrap();
    (
        resonance(&port.analyze(&res, &freqs()).unwrap().z_in()),
        warnings,
    )
}

fn show(name: &str, r: &Res, reference: &Res) -> [f64; 3] {
    let d = [
        r.f_res / reference.f_res - 1.0,
        r.r_res / reference.r_res - 1.0,
        r.slope / reference.slope - 1.0,
    ];
    eprintln!(
        "{name}: f_res {:.4} GHz ({:+.2} %), R {:.2} Ω ({:+.2} %), dX/df {:.0} Ω/GHz ({:+.1} %)",
        r.f_res / 1e9,
        100.0 * d[0],
        r.r_res,
        100.0 * d[1],
        r.slope,
        100.0 * d[2]
    );
    d
}

/// Tilted and off-mesh dipoles against the axis-aligned material model.
#[test]
#[ignore]
fn tilted_dipoles() {
    let a = lattice_radius(2.5, 2.5);
    let reference = axis_aligned(a);
    show("axis-aligned material model", &reference, &reference);
    let (s30, s45) = (30f64.to_radians(), 45f64.to_radians());
    let cases: [(&str, [f64; 3], [f64; 3]); 6] = [
        (
            "oblique model, along z on a mesh line",
            [0.0, 0.0, 1.0],
            [0.0; 3],
        ),
        (
            "oblique model, along z at a cell centre",
            [0.0, 0.0, 1.0],
            [5.0, 5.0, 0.0],
        ),
        (
            "oblique model, 30° in xz",
            [s30.sin(), 0.0, s30.cos()],
            [0.0; 3],
        ),
        (
            "oblique model, 45° in xz",
            [s45.sin(), 0.0, s45.cos()],
            [0.0; 3],
        ),
        (
            "oblique model, 30° in xz, off-mesh",
            [s30.sin(), 0.0, s30.cos()],
            [3.1, 4.3, 2.7],
        ),
        (
            "oblique model, along (1, 2, 3), off-mesh",
            [1.0, 2.0, 3.0],
            [1.3, 2.1, 0.7],
        ),
    ];
    let mut f = Vec::new();
    for (name, t, c) in cases {
        let r = oblique(t, c, a, None, 10.0);
        let d = show(name, &r, &reference);
        assert!(d[0].abs() < 0.015, "{name}: resonance {d:?}");
        assert!(d[1].abs() < 0.025, "{name}: resistance {d:?}");
        assert!(d[2].abs() < 0.03, "{name}: slope {d:?}");
        f.push(r.f_res);
    }
    let hi = f.iter().copied().fold(f64::MIN, f64::max);
    let lo = f.iter().copied().fold(f64::MAX, f64::min);
    eprintln!(
        "resonance spread over orientations and positions: {:.2} %",
        100.0 * (hi / lo - 1.0)
    );
    assert!(hi / lo - 1.0 < 0.01);
    // Through csx, with the gap as a lumped port and tilted arms.
    let (r, w) = csx_tilted([s30.sin(), 0.0, s30.cos()], a);
    assert!(w.iter().all(|w| w.starts_with("mesh")), "{w:?}");
    let d = show("csx: lumped port, arms tilted 30° in xz", &r, &reference);
    assert!(d[0].abs() < 0.04 && d[1].abs() < 0.08, "{d:?}");
}

/// Convergence in the wire segment length and the mesh, along z.
#[test]
#[ignore]
fn segments_and_mesh() {
    let a = lattice_radius(2.5, 2.5);
    let reference = axis_aligned(a);
    let mut f = Vec::new();
    for seg in [10.0, 5.0, 2.5, 1.25] {
        let r = oblique([0.0, 0.0, 1.0], [0.0; 3], a, Some(seg), 10.0);
        show(
            &format!("segments of {seg} mm on 10 mm cells"),
            &r,
            &reference,
        );
        f.push(r.f_res);
    }
    // Half a cell (the default) is within 0.2 % of the converged value.
    assert!((f[1] / f[3] - 1.0).abs() < 2e-3);
    for d in [5.0, 10.0, 20.0] {
        for c in [0.0, 0.5 * d] {
            let r = oblique([0.0, 0.0, 1.0], [c, c, 0.0], a, None, d);
            let at = if c == 0.0 {
                "on a mesh line"
            } else {
                "cell centre"
            };
            let e = show(&format!("{d} mm cells, {at}"), &r, &reference);
            assert!(e[0].abs() < 0.02, "{e:?}");
        }
    }
}

/// A 45° dipole on a coarse mesh with Mur boundaries, 50 000 steps (many
/// times the ring-down) without end criteria: the energy decays to the
/// static residue of the pulse's DC content (about 4·10⁻⁸ of the peak)
/// and then stays constant. (With CPML the residue grows slowly, with or
/// without the wire: a plain staircase dipole with a lumped port shows
/// the same late-time CPML drift.)
#[test]
#[ignore]
fn long_run_open() {
    let t = unit([1.0, 0.0, 1.0]);
    let mut model = ContinuousStructure::new();
    mesh(&mut model, t, [0.0; 3], 20.0, [0.0; 3]);
    let fdtd = fdtd()
        .boundaries(Boundaries::all(Boundary::mur()))
        .no_end_criteria();
    let mut sim = Simulation::new(&model, &fdtd).unwrap();
    let p = |s: f64| -> [f64; 3] { t.map(|v| v * s * 1e-3) };
    let w = ObliqueWire::new([p(-TIP), p(-GAP), p(GAP), p(TIP)], 0.1e-3)
        .feed(1, 50.0)
        .excite(Waveform::gaussian(0.9e9, 0.45e9), 1.0);
    let grid = sim.grid().clone();
    let mats = ems::EdgeMaterials::vacuum(&grid);
    let net = ObliqueWires::new(&grid, &mats, &[w], |_, _| false).unwrap();
    sim.engine_mut().add_operator(net);
    let mut peak = 0f64;
    let mut e = Vec::new();
    for _ in 0..100 {
        sim.step(500);
        let w = sim.engine().energy();
        peak = peak.max(w);
        e.push(w);
    }
    let (w20, w50) = (e[39], e[99]);
    eprintln!(
        "peak energy {peak:e}; at 20 000 steps {:.3e}, at 50 000 steps {:.3e} of the peak",
        w20 / peak,
        w50 / peak
    );
    assert!(w50.is_finite() && w50 < 1e-6 * peak);
    assert!((w50 / w20 - 1.0).abs() < 1e-3);
}
