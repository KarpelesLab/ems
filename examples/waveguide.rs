//! A WR-90 rectangular waveguide (22.86 mm × 10.16 mm) with a dielectric
//! slab across the guide, between two TE10 [`RectWaveguidePort`]s:
//! |S11| and |S21| versus the transmission-line model of the slab.
//!
//! ```text
//! cargo run --release --example waveguide
//! ```
//!
//! The guide runs along z with PEC walls (the x and y faces of the
//! domain) and 8-cell CPML at both ends. The slab (`ε_r = 2.2`,
//! `t = 10 mm`) fills the cross section in the middle. For TE10 each
//! section is a transmission line with `β = √(ε_r k² − (π/a)²)` and wave
//! impedance `Z_TE = ωμ0/β` (Pozar, *Microwave Engineering*, rectangular
//! waveguide), so the slab terminated by the matched empty guide has
//!
//! ```text
//! Z_in = Z2·(Z1 + jZ2·tan β2t)/(Z2 + jZ1·tan β2t),   S11 = (Z_in − Z1)/(Z_in + Z1)
//! ```
//!
//! and `|S21|² = 1 − |S11|²` (lossless). The magnitudes do not depend on
//! where the reference planes are. With the 0.7 mm mesh used here the
//! simulated magnitudes agree with the model to about 0.003 over the
//! WR-90 band (8.2–12.4 GHz).

use std::f64::consts::PI;

use csx::primitives::BoxShape;
use csx::properties::Material;
use csx::{ContinuousStructure, Primitive, Property};
use ems::constants::{C0, MU0};
use ems::post::s_param;
use ems::prelude::*;

const A: f64 = 22.86;
const B: f64 = 10.16;
const LEN: f64 = 120.0;
const EPS_R: f64 = 2.2;
const T: f64 = 10.0;

/// |S11| of the slab from the transmission-line model.
fn slab_s11(f: f64) -> f64 {
    let kc = PI / (A * 1e-3);
    let k = 2.0 * PI * f / C0;
    let w = 2.0 * PI * f;
    let b1 = (k * k - kc * kc).sqrt();
    let b2 = (EPS_R * k * k - kc * kc).sqrt();
    let (z1, z2) = (w * MU0 / b1, w * MU0 / b2);
    let tn = Complex64::new(0.0, (b2 * T * 1e-3).tan());
    let zin = (tn * z2 + z1) / (tn * z1 + z2) * z2;
    ((zin - z1) / (zin + z1)).norm()
}

fn main() -> Result<()> {
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    let uniform = |l: f64, n: usize| (0..=n).map(move |i| l * i as f64 / n as f64);
    model.grid.set_lines(0, uniform(A, 32))?;
    model.grid.set_lines(1, uniform(B, 14))?;
    model.grid.set_lines(2, uniform(LEN, 192))?;

    let z0 = 0.5 * (LEN - T);
    let slab = model.add_property(Property::material("slab", Material::new(EPS_R, 0.0)));
    model.add_primitive(
        slab,
        Primitive::new(BoxShape::new([0., 0., z0], [A, B, z0 + T])),
    )?;
    let p1 = RectWaveguidePort::new(1, [0., 0., 10.], [A, B, 15.], Axis::Z, (1, 0))
        .excite(1.0)
        .add_to(&mut model)?;
    let p2 = RectWaveguidePort::new(2, [0., 0., LEN - 10.], [A, B, LEN - 15.], Axis::Z, (1, 0))
        .add_to(&mut model)?;
    println!(
        "WR-90: TE10 cutoff {:.3} GHz, slab ε_r = {EPS_R}, t = {T} mm",
        p1.cutoff_frequency().unwrap() / 1e9
    );

    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(10.3e9, 2.5e9))
        .boundaries(
            Boundaries::all(Boundary::Pec)
                .with(Face::ZMin, Boundary::pml(8))
                .with(Face::ZMax, Boundary::pml(8)),
        )
        .end_criteria(EndCriteria::energy_decay_db(-50.0));
    let mut sim = Simulation::new(&model, &fdtd)?;
    for w in sim.warnings() {
        println!("warning: {w}");
    }
    let res = sim.run()?;
    let o = res.outcome().unwrap();
    println!("{} steps ({:?})", o.steps, o.reason);

    let freqs: Vec<f64> = (0..=8).map(|i| 8.2e9 + 0.525e9 * i as f64).collect();
    let a1 = p1.analyze(&res, &freqs)?;
    let a2 = p2.analyze(&res, &freqs)?;
    let s11 = a1.s11();
    let s21 = s_param(&a2, &a1);
    println!("  f [GHz]  |S11| sim  model    |S21| sim  model");
    for (k, &f) in freqs.iter().enumerate() {
        let m11 = slab_s11(f);
        println!(
            "  {:6.3}   {:.4}    {:.4}   {:.4}     {:.4}",
            f / 1e9,
            s11[k].norm(),
            m11,
            s21[k].norm(),
            (1.0 - m11 * m11).sqrt()
        );
    }
    Ok(())
}
