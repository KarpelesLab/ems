//! A thin-wire half-wave dipole fed by a lumped port: resonance and input
//! impedance.
//!
//! ```text
//! cargo run --release --example dipole
//! ```
//!
//! The dipole is a 152.5 mm long PEC line along z (two arms of zero
//! thickness on the mesh line `x = y = 0`) with a one-cell (2.5 mm) gap in
//! the middle holding a 50 Ω lumped port. A wire of PEC edges behaves like
//! a round wire of radius ≈ 0.135·Δ (Taflove & Hagness, ch. 10, thin-wire
//! models), here ≈ 0.34 mm, so `L/a ≈ 450`.
//!
//! For a thin dipole the first resonance (`X_in = 0`) is slightly below
//! `L = λ/2`, at `L ≈ 0.47…0.48 λ`, with `R_in ≈ 65…73 Ω` (the 73 Ω of
//! the infinitely thin half-wave dipole, Balanis, *Antenna Theory*,
//! §4.6). The mesh is generated with csx's [`MeshBuilder`], fine along and
//! around the wire and graded to λ/20 at `f_max` outside; an 8-cell CPML
//! closes the domain.
//!
//! An NF2FF box ([`ems::nf2ff`]) around the dipole gives the directivity
//! and gain at resonance: about 1.64 (2.15 dBi) for a thin half-wave
//! dipole (Balanis §4.6), with gain = directivity in vacuum.

use csx::mesh::{MeshBuilder, MeshOptions};
use csx::primitives::BoxShape;
use csx::{ContinuousStructure, Primitive, Property, TypeMask};
use ems::nf2ff::Nf2ffBox;
use ems::prelude::*;

/// Cell size along and around the wire [mm].
const D: f64 = 2.5;
/// Cells per arm.
const ARM_CELLS: usize = 30;

fn main() -> Result<()> {
    let gap = D / 2.0; // half gap
    let tip = gap + ARM_CELLS as f64 * D;
    let length = 2.0 * tip;

    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    let wire = model.add_property(Property::metal("wire"));
    for (a, b) in [(gap, tip), (-tip, -gap)] {
        model.add_primitive(
            wire,
            Primitive::new(BoxShape::new([0.0, 0.0, a], [0.0, 0.0, b])),
        )?;
    }
    let port = LumpedPort::new(1, 50.0, [0.0, 0.0, -gap], [0.0, 0.0, gap], Axis::Z)
        .excite(1.0)
        .add_to(&mut model)?;

    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(1e9, 0.5e9))
        .boundaries(Boundaries::all(Boundary::pml(8)))
        .end_criteria(EndCriteria::energy_decay_db(-40.0))
        .max_timesteps(50_000);

    let max_res = ems::mesh::suggest_max_cell(&model, &fdtd, 20.0)?; // ≈ 10 mm
    let near = [-2.0 * D, -D, 0.0, D, 2.0 * D];
    MeshBuilder::new(
        MeshOptions::new(max_res)
            .with_ratio(1.4)
            .with_margin(150.0)
            .with_mask(TypeMask::PHYSICAL),
    )
    .with_lines(0, near)
    .with_lines(1, near)
    .with_lines(2, (0..=2 * ARM_CELLS + 1).map(|i| -tip + i as f64 * D))
    .apply(&mut model)?;

    let mut sim = Simulation::new(&model, &fdtd)?;
    // NF2FF box 2–3 cells around the wire, every 5 MHz near resonance.
    let nf_freqs: Vec<f64> = (0..=60).map(|k| 0.8e9 + k as f64 * 5e6).collect();
    let b = tip + 2.0 * D;
    let nf = sim.add_nf2ff(&Nf2ffBox::new(
        [-2.0 * D, -2.0 * D, -b],
        [2.0 * D, 2.0 * D, b],
        nf_freqs.clone(),
    ))?;
    for w in sim.warnings() {
        println!("warning: {w}");
    }
    let n = sim.grid().nodes();
    println!(
        "dipole L = {length} mm, mesh {} x {} x {} nodes, dt = {:.3e} s",
        n[0],
        n[1],
        n[2],
        sim.engine().dt()
    );
    let res = sim.run()?;
    let o = res.outcome().expect("outcome");
    println!(
        "{:?} after {} steps in {:.2} s",
        o.reason,
        o.steps,
        o.elapsed.as_secs_f64()
    );

    let freqs: Vec<f64> = (0..=400).map(|k| 0.6e9 + k as f64 * 2e6).collect();
    let r = port.analyze(&res, &freqs)?;
    let z = r.z_in();
    let s11 = r.s11();
    println!(" f [GHz]   Z_in [Ω]        |S11| [dB] (50 Ω)");
    for q in (0..freqs.len()).step_by(50) {
        println!(
            "{:>7.3}   {:<16}  {:>6.2}",
            freqs[q] / 1e9,
            format!("{:.1}", z[q]),
            20.0 * s11[q].norm().log10()
        );
    }
    // First resonance: X_in crosses zero from below.
    let Some(k) = (1..freqs.len()).find(|&k| z[k - 1].im < 0.0 && z[k].im >= 0.0) else {
        println!("no resonance found in the band");
        return Ok(());
    };
    let t = -z[k - 1].im / (z[k].im - z[k - 1].im);
    let f_res = freqs[k - 1] + t * (freqs[k] - freqs[k - 1]);
    let r_res = z[k - 1].re + t * (z[k].re - z[k - 1].re);
    let lambda = C0 / f_res * 1e3;
    println!(
        "resonance: f = {:.4} GHz (L = {:.3} λ), R_in = {:.1} Ω",
        f_res / 1e9,
        length / lambda,
        r_res
    );

    // Directivity and gain at the NF2FF frequency nearest to resonance.
    let ff = sim.nf2ff(&nf);
    let q = (0..nf_freqs.len())
        .min_by(|&a, &b| {
            (nf_freqs[a] - f_res)
                .abs()
                .total_cmp(&(nf_freqs[b] - f_res).abs())
        })
        .expect("frequencies");
    let p_acc = port.analyze(&res, &nf_freqs[q..=q])?.accepted_power()[0];
    let d = ff.max_directivity(q);
    let g = ff.gain(q, d.theta, d.phi, p_acc);
    println!(
        "far field at {:.3} GHz: D = {:.3} ({:.2} dBi) at θ = {:.1}°, G = {:.2} dBi, \
         P_rad/P_acc = {:.4}",
        nf_freqs[q] / 1e9,
        d.directivity,
        d.dbi(),
        d.theta.to_degrees(),
        10.0 * g.log10(),
        ff.radiated_power(q) / p_acc
    );
    Ok(())
}
