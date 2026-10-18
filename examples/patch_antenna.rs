//! A 2.4 GHz rectangular patch antenna fed by a lumped port: S11
//! resonance and input impedance.
//!
//! ```text
//! cargo run --release --example patch_antenna
//! ```
//!
//! Substrate `ε_r = 3.38`, `h = 1.524 mm` (a common microwave laminate),
//! 60 mm × 60 mm with a ground plane underneath; patch `L = 33.5 mm`
//! (resonant length, along x) × `W = 42 mm`, sized with the transmission
//! line model (Balanis, *Antenna Theory*, §14.2: `W = c/(2f)·sqrt(2/(ε_r+1))`,
//! `L = c/(2f·sqrt(ε_eff)) − 2ΔL`). A 50 Ω lumped port between ground
//! and patch, 5.5 mm off the centre along x, acts as a probe feed (the
//! inset point where `R_edge·cos²(π·x₀/L)` is about 50 Ω).
//!
//! The mesh is generated with csx's [`MeshBuilder`] and checked with
//! [`ems::mesh::check`]; the domain is closed by an 8-cell CPML.

use csx::mesh::{MeshBuilder, MeshOptions};
use csx::primitives::BoxShape;
use csx::properties::Material;
use csx::{ContinuousStructure, Primitive, Property, TypeMask};
use ems::prelude::*;

const EPS_R: f64 = 3.38;
const H: f64 = 1.524;
const SUB: f64 = 60.0;
const PATCH_L: f64 = 33.5;
const PATCH_W: f64 = 42.0;
const FEED_X: f64 = -5.5;

fn main() -> Result<()> {
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    let sub = model.add_property(Property::material("substrate", Material::new(EPS_R, 0.0)));
    model.add_primitive(
        sub,
        Primitive::new(BoxShape::new(
            [-SUB / 2.0, -SUB / 2.0, 0.0],
            [SUB / 2.0, SUB / 2.0, H],
        )),
    )?;
    let metal = model.add_property(Property::metal("copper"));
    model.add_primitive(
        metal,
        Primitive::new(BoxShape::new(
            [-SUB / 2.0, -SUB / 2.0, 0.0],
            [SUB / 2.0, SUB / 2.0, 0.0],
        ))
        .with_priority(10),
    )?;
    model.add_primitive(
        metal,
        Primitive::new(BoxShape::new(
            [-PATCH_L / 2.0, -PATCH_W / 2.0, H],
            [PATCH_L / 2.0, PATCH_W / 2.0, H],
        ))
        .with_priority(10),
    )?;
    let port = LumpedPort::new(1, 50.0, [FEED_X, 0.0, 0.0], [FEED_X, 0.0, H], Axis::Z)
        .excite(1.0)
        .priority(5)
        .add_to(&mut model)?;

    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(2.4e9, 1.0e9))
        .boundaries(Boundaries::all(Boundary::pml(8)))
        .end_criteria(EndCriteria::energy_decay_db(-35.0))
        .max_timesteps(60_000);

    // Mesh: λ/20 in the substrate at f_max in x/y (≈ 2.4 mm → 2 mm), 3 mm
    // in z, 4 cells through the substrate, a 40 mm air margin (about λ/3
    // at 2.4 GHz) that also holds the PML.
    let max_xy = ems::mesh::suggest_max_cell(&model, &fdtd, 20.0)?.min(2.0);
    MeshBuilder::new(
        MeshOptions::new(max_xy)
            .with_max_res([max_xy, max_xy, 3.0])
            .with_ratio(1.4)
            .with_margin(40.0)
            .with_thirds([max_xy, max_xy, 3.0])
            .with_mask(TypeMask::PHYSICAL),
    )
    .add_lines(2, (1..4).map(|i| H * i as f64 / 4.0))
    .apply(&mut model)?;

    let report = ems::mesh::check(&model, &fdtd)?;
    print!("{report}");

    let mut sim = Simulation::new(&model, &fdtd)?;
    for w in sim.warnings() {
        println!("warning: {w}");
    }
    let res = sim.run()?;
    let o = res.outcome().expect("outcome");
    println!(
        "{:?} after {} steps in {:.2} s",
        o.reason,
        o.steps,
        o.elapsed.as_secs_f64()
    );

    let freqs: Vec<f64> = (0..=240).map(|k| 1.8e9 + k as f64 * 5e6).collect();
    let r = port.analyze(&res, &freqs)?;
    let s11 = r.s11();
    let z = r.z_in();
    let db: Vec<f64> = s11.iter().map(|s| 20.0 * s.norm().log10()).collect();
    let k = (0..freqs.len())
        .min_by(|&a, &b| db[a].total_cmp(&db[b]))
        .expect("frequencies");
    println!(" f [GHz]  |S11| [dB]   Z_in [Ω]");
    for q in (0..freqs.len()).step_by(20) {
        println!("{:>7.3}  {:>9.2}   {:.1}", freqs[q] / 1e9, db[q], z[q]);
    }
    println!(
        "resonance: f = {:.4} GHz, |S11| = {:.1} dB, Z_in = {:.1} Ω",
        freqs[k] / 1e9,
        db[k],
        z[k]
    );
    let bw: Vec<f64> = freqs
        .iter()
        .zip(&db)
        .filter(|(_, d)| **d < -10.0)
        .map(|(f, _)| *f)
        .collect();
    if let (Some(lo), Some(hi)) = (bw.first(), bw.last()) {
        println!(
            "-10 dB band: {:.3} .. {:.3} GHz ({:.1} %)",
            lo / 1e9,
            hi / 1e9,
            (hi - lo) / freqs[k] * 100.0
        );
    }
    Ok(())
}
