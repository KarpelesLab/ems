//! Thin-wire dipoles through the csx path ([`ems::engine::wire`]): a
//! center-fed dipole, 170 mm tip to tip with a 10 mm feed gap holding a
//! 50 Ω lumped port, built from csx `Wire` primitives on meshes whose
//! cells are 2…40× the wire radius.
//!
//! **Reference.** A plain staircase wire (PEC edges) behaves like a round
//! wire of its lattice radius `r_eq = 0.1985·Δ`, and its open ends sit
//! half a cell beyond the end nodes. A staircase dipole on a fine mesh
//! with `r_eq(Δ_fine) = a`, laid out so that its effective tips are at
//! ±85 mm, is therefore a converged-mesh reference for a dipole of radius
//! `a`. (The plain staircase results on 1.25 / 2.5 / 5 / 10 mm cells laid
//! out this way change smoothly with `ln(r_eq)`, which supports the
//! half-cell convention.)
//!
//! The thin-wire model must reproduce the reference resonance, input
//! resistance and reactance slope on coarser meshes, and follow the
//! radius (thinner: higher resonant frequency, steeper reactance), which
//! a plain staircase cannot.

use csx::mesh::{MeshBuilder, MeshOptions};
use csx::primitives::{BoxShape, Wire};
use csx::{ContinuousStructure, Primitive, Property, TypeMask};
use ems::engine::wire::lattice_radius;
use ems::prelude::*;

/// Half the feed gap and the arm tip [mm].
const GAP: f64 = 5.0;
const TIP: f64 = 85.0;

struct Res {
    f_res: f64,
    r_res: f64,
    /// dX/df at resonance [Ω/GHz].
    slope: f64,
    warnings: Vec<String>,
}

/// Dipole on `d` mm cells along and around the wire. `radius` [mm] gives
/// csx `Wire` arms from the gap to the tips; `None` gives zero-thickness
/// lines laid out as the staircase reference (gap of one cell, end nodes
/// half a cell inside the tips).
fn dipole(d: f64, radius: Option<f64>) -> Res {
    let (gap, tip, z0) = match radius {
        Some(_) => (GAP, TIP, -TIP),
        None => (0.5 * d, TIP - 0.5 * d, -TIP - 0.5 * d),
    };
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    let wire = model.add_property(Property::metal("wire"));
    // Mesh with zero-thickness arms: csx's mesh builder would otherwise put
    // mesh lines on the wire surface (its bounding box). The wires replace
    // them afterwards.
    let arms = [(gap, tip), (-tip, -gap)];
    for (a, b) in arms {
        let prim = Primitive::new(BoxShape::new([0.0, 0.0, a], [0.0, 0.0, b]));
        model.add_primitive(wire, prim).unwrap();
    }
    let port = LumpedPort::new(1, 50.0, [0.0, 0.0, -gap], [0.0, 0.0, gap], Axis::Z)
        .excite(1.0)
        .add_to(&mut model)
        .unwrap();
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(0.9e9, 0.45e9))
        .boundaries(Boundaries::all(Boundary::pml(8)))
        .end_criteria(EndCriteria::energy_decay_db(-40.0))
        .max_timesteps(60_000);
    let near = [-2.0 * d, -d, 0.0, d, 2.0 * d];
    let nz = (-2.0 * z0 / d).round() as usize;
    MeshBuilder::new(
        MeshOptions::new(10.0)
            .with_ratio(1.4)
            .with_margin(150.0)
            .with_mask(TypeMask::PHYSICAL),
    )
    .with_lines(0, near)
    .with_lines(1, near)
    .with_lines(2, (0..=nz).map(|i| z0 + i as f64 * d))
    .apply(&mut model)
    .unwrap();
    if let Some(r) = radius {
        model.property_mut(wire).unwrap().primitives = arms
            .iter()
            .map(|&(a, b)| Primitive::new(Wire::new([[0.0, 0.0, a], [0.0, 0.0, b]], r)))
            .collect();
    }
    let mut sim = Simulation::new(&model, &fdtd).unwrap();
    let warnings = sim.warnings().to_vec();
    let res = sim.run().unwrap();
    let freqs: Vec<f64> = (0..=400).map(|k| 0.65e9 + k as f64 * 1e6).collect();
    let z = port.analyze(&res, &freqs).unwrap().z_in();
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
        warnings,
    }
}

fn show(name: &str, r: &Res) {
    eprintln!(
        "{name}: f_res {:.4} GHz, R {:.2} Ω, dX/df {:.0} Ω/GHz",
        r.f_res / 1e9,
        r.r_res,
        r.slope
    );
}

/// Relative deviations of resonance, resistance and slope from `reference`,
/// printed and checked against `tol` (resonance), `2·tol` (resistance) and
/// `5·tol` (slope).
fn compare(name: &str, r: &Res, reference: &Res, tol: f64) {
    assert!(r.warnings.is_empty(), "{name}: {:?}", r.warnings);
    let d = [
        r.f_res / reference.f_res - 1.0,
        r.r_res / reference.r_res - 1.0,
        r.slope / reference.slope - 1.0,
    ];
    eprintln!(
        "{name}: f_res {:.4} GHz ({:+.2} %), R {:.2} Ω ({:+.2} %), dX/df {:.0} ({:+.1} %)",
        r.f_res / 1e9,
        100.0 * d[0],
        r.r_res,
        100.0 * d[1],
        r.slope,
        100.0 * d[2]
    );
    assert!(d[0].abs() < tol, "{name}: resonance off by {}", d[0]);
    assert!(d[1].abs() < 2.0 * tol, "{name}: resistance off by {}", d[1]);
    assert!(d[2].abs() < 5.0 * tol, "{name}: slope off by {}", d[2]);
}

#[test]
fn dipole_matches_fine_staircase() {
    let t0 = std::time::Instant::now();
    // Radius 0.496 mm: staircase reference on 2.5 mm cells.
    let a = lattice_radius(2.5, 2.5);
    let reference = dipole(2.5, None);
    assert!(reference.warnings.is_empty(), "{:?}", reference.warnings);
    show(
        &format!("reference: staircase on 2.5 mm (a = {a:.3} mm)"),
        &reference,
    );
    let thin: Vec<Res> = [5.0, 10.0].map(|d| dipole(d, Some(a))).into();
    for (d, r) in [5.0, 10.0].iter().zip(&thin) {
        compare(&format!("thin wire a on {d} mm cells"), r, &reference, 6e-3);
    }
    // The plain staircase on 10 mm cells models a ≈ 2 mm.
    show("staircase on 10 mm (a ≈ 1.99 mm)", &dipole(10.0, None));
    // A 2.5× thinner wire on the 10 mm mesh: higher resonance, steeper
    // reactance.
    let thinner = dipole(10.0, Some(a / 2.5));
    show("thin wire a/2.5 on 10 mm cells", &thinner);
    assert!(thinner.f_res > thin[1].f_res && thinner.slope > 1.1 * thin[1].slope);
    eprintln!(
        "[dipole_matches_fine_staircase] {:.2} s",
        t0.elapsed().as_secs_f64()
    );
}

/// Like [`dipole_matches_fine_staircase`] for a 0.248 mm wire, against a
/// 1.25 mm staircase (heavier; run with `--ignored`, in release).
#[test]
#[ignore]
fn dipole_matches_finer_staircase() {
    let a = lattice_radius(1.25, 1.25);
    let reference = dipole(1.25, None);
    show(
        &format!("reference: staircase on 1.25 mm (a = {a:.3} mm)"),
        &reference,
    );
    for d in [2.5, 5.0, 10.0] {
        compare(
            &format!("thin wire a on {d} mm cells"),
            &dipole(d, Some(a)),
            &reference,
            1e-2,
        );
    }
}
