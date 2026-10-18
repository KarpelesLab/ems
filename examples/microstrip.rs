//! A 50 Ω-ish microstrip line on an FR4-like substrate between two lumped
//! ports: S11/S21 and the line impedance and effective permittivity
//! extracted from them, versus the closed-form Hammerstad formulas.
//!
//! ```text
//! cargo run --release --example microstrip
//! ```
//!
//! Geometry (mm): substrate `ε_r = 4.3`, `h = 0.8`; strip `w = 1.5`,
//! `l = 40` from `x = −20` to `x = 20`; ground plane on the `z = 0` PEC
//! boundary; 8-cell CPML on all other faces. Each end has a 50 Ω lumped
//! port from the ground to the strip across the full strip width
//! ([`LumpedPort`]); port 1 is excited.
//!
//! From the two-port S-parameters of a uniform line with reference
//! impedance `Z_ref` (Pozar, *Microwave Engineering*, §4.4 conversions):
//!
//! ```text
//! Z0 = Z_ref · sqrt(((1 + S11)² − S21²) / ((1 − S11)² − S21²))
//! Γ  = (Z0 − Z_ref)/(Z0 + Z_ref),   T = e^{−γl} = (S11 + S21 − Γ)/(1 − (S11 + S21)·Γ)
//! ε_eff = (c0·β/ω)²,   β = −unwrap(arg T)/l
//! ```
//!
//! The reference is the quasi-static closed form for a zero-thickness strip
//! (E. O. Hammerstad, "Equations for microstrip circuit design", 1975; as
//! given in Pozar, §3.8). The extraction includes the parasitics of the
//! lumped ports, so a few percent of difference are expected.

use std::f64::consts::PI;

use csx::mesh::{MeshBuilder, MeshOptions};
use csx::primitives::BoxShape;
use csx::properties::Material;
use csx::{ContinuousStructure, Primitive, Property, TypeMask};
use ems::post::s_param;
use ems::prelude::*;

const EPS_R: f64 = 4.3;
const H: f64 = 0.8;
const W: f64 = 1.5;
const LEN: f64 = 40.0;

fn csqrt(z: Complex64) -> Complex64 {
    let (r, t) = z.to_polar();
    Complex64::from_polar(r.sqrt(), 0.5 * t)
}

/// Hammerstad / Pozar closed form: (ε_eff, Z0).
fn hammerstad(eps_r: f64, w: f64, h: f64) -> (f64, f64) {
    let u = w / h;
    let e = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 / (1.0 + 12.0 / u).sqrt();
    let z0 = if u <= 1.0 {
        60.0 / e.sqrt() * (8.0 / u + u / 4.0).ln()
    } else {
        120.0 * PI / (e.sqrt() * (u + 1.393 + 0.667 * (u + 1.444).ln()))
    };
    (e, z0)
}

fn main() -> Result<()> {
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    let sub = model.add_property(Property::material("FR4", Material::new(EPS_R, 0.0)));
    model.add_primitive(
        sub,
        Primitive::new(BoxShape::new([-25.0, -10.0, 0.0], [25.0, 10.0, H])),
    )?;
    let cu = model.add_property(Property::metal("copper"));
    model.add_primitive(
        cu,
        Primitive::new(BoxShape::new(
            [-LEN / 2.0, -W / 2.0, H],
            [LEN / 2.0, W / 2.0, H],
        ))
        .with_priority(10),
    )?;
    model.add_primitive(
        cu,
        Primitive::new(BoxShape::new([-25.0, -10.0, 0.0], [25.0, 10.0, 0.0])).with_priority(10),
    )?;
    let p1 = LumpedPort::new(
        1,
        50.0,
        [-LEN / 2.0, -W / 2.0, 0.0],
        [-LEN / 2.0, W / 2.0, H],
        Axis::Z,
    )
    .excite(1.0)
    .add_to(&mut model)?;
    let p2 = LumpedPort::new(
        2,
        50.0,
        [LEN / 2.0, -W / 2.0, 0.0],
        [LEN / 2.0, W / 2.0, H],
        Axis::Z,
    )
    .add_to(&mut model)?;

    // Mesh: 1 mm maximum (λ/20 in the substrate at 8 GHz), 0.25 mm across
    // the strip, 4 cells through the substrate, air margin for the PML.
    MeshBuilder::new(
        MeshOptions::new(1.0)
            .with_ratio(1.4)
            .with_mask(TypeMask::PHYSICAL),
    )
    .add_lines(0, [-37.0, 37.0])
    .add_lines(1, (0..=6).map(|i| -W / 2.0 + 0.25 * i as f64))
    .add_lines(1, [-22.0, 22.0])
    .add_lines(2, (0..=4).map(|i| H * i as f64 / 4.0))
    .add_lines(2, [20.0])
    .apply(&mut model)?;

    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(4e9, 4e9))
        .boundaries(Boundaries::all(Boundary::pml(8)))
        .boundary(Face::ZMin, Boundary::Pec)
        .end_criteria(EndCriteria::energy_decay_db(-50.0))
        .max_timesteps(100_000);
    let mut sim = Simulation::new(&model, &fdtd)?;
    for w in sim.warnings() {
        println!("warning: {w}");
    }
    let n = sim.grid().nodes();
    println!(
        "mesh {} x {} x {} nodes, dt = {:.3e} s",
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

    let freqs: Vec<f64> = (1..=36).map(|k| k as f64 * 0.2e9).collect();
    let r1 = p1.analyze(&res, &freqs)?;
    let r2 = p2.analyze(&res, &freqs)?;
    let s11 = r1.s11();
    let s21 = s_param(&r2, &r1);
    let (e_ham, z_ham) = hammerstad(EPS_R, W, H);
    let z_ref = 50.0;
    let len = LEN * 1e-3;

    println!("Hammerstad: Z0 = {z_ham:.2} Ω, eps_eff = {e_ham:.3}");
    println!(" f [GHz]  |S11| [dB]  |S21| [dB]   Z0 [Ω]   eps_eff");
    let mut phase_prev = 0.0;
    let mut unwrapped = 0.0;
    let (mut z_sum, mut e_sum, mut cnt) = (0.0, 0.0, 0);
    for (k, &f) in freqs.iter().enumerate() {
        let (s, t) = (s11[k], s21[k]);
        let one = Complex64::new(1.0, 0.0);
        let num = (one + s) * (one + s) - t * t;
        let den = (one - s) * (one - s) - t * t;
        let mut z0 = csqrt(num / den) * z_ref;
        if z0.re < 0.0 {
            z0 = -z0;
        }
        let g = (z0 - z_ref) / (z0 + z_ref);
        let tr = (s + t - g) / (one - (s + t) * g);
        // Unwrap the transmission phase from low frequency upwards.
        let ph = tr.arg();
        let mut d = ph - phase_prev;
        while d > PI {
            d -= 2.0 * PI;
        }
        while d < -PI {
            d += 2.0 * PI;
        }
        unwrapped += d;
        phase_prev = ph;
        let beta = -unwrapped / len;
        let eps_eff = (C0 * beta / (2.0 * PI * f)).powi(2);
        if k % 4 == 3 {
            println!(
                "{:>7.2}  {:>10.2}  {:>10.2}  {:>7.2}  {:>8.3}",
                f / 1e9,
                20.0 * s.norm().log10(),
                20.0 * t.norm().log10(),
                z0.norm(),
                eps_eff
            );
        }
        if (1e9..=6e9).contains(&f) {
            z_sum += z0.norm();
            e_sum += eps_eff;
            cnt += 1;
        }
    }
    let (z_avg, e_avg) = (z_sum / cnt as f64, e_sum / cnt as f64);
    println!(
        "1-6 GHz average: Z0 = {z_avg:.2} Ω ({:+.1} % vs Hammerstad), eps_eff = {e_avg:.3} ({:+.1} %)",
        (z_avg / z_ham - 1.0) * 100.0,
        (e_avg / e_ham - 1.0) * 100.0
    );
    Ok(())
}
