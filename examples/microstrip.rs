//! A 50 Ω-ish microstrip line on an FR4-like substrate between two
//! microstrip ports ([`MslPort`]): S11/S21 and the line impedance and
//! effective permittivity extracted by the ports, versus the closed-form
//! Hammerstad formulas.
//!
//! ```text
//! cargo run --release --example microstrip
//! ```
//!
//! Geometry (mm): substrate `ε_r = 4.3`, `h = 0.8`; strip `w = 1.5` from
//! `x = −30` to `x = 30`, running into the 8-cell CPML at both ends; ground
//! plane on the `z = 0` PEC boundary. Port 1 (excited, with a 50 Ω feed
//! resistor) is fed at `x = −25` and has its reference plane at `x = −10`;
//! port 2 is its mirror image. The mesh uses the thirds rule at the strip
//! edges (csx [`MeshOptions::with_thirds`]) and uniform 0.5 mm cells along
//! the line, as the ports require.
//!
//! Each port extracts the propagation constant `γ` and the line impedance
//! `Z_L` from its probes ([`MslPortResult`]); with `Z_L` as reference
//! impedance a uniform matched line gives `S11 ≈ 0` and `S21 = e^{−γL}`
//! (`L` = 20 mm between the reference planes).
//!
//! The reference is the quasi-static closed form for a zero-thickness strip
//! (E. O. Hammerstad, "Equations for microstrip circuit design", 1975; as
//! given in Pozar, *Microwave Engineering*, §3.8).

use std::f64::consts::PI;

use csx::mesh::{MeshBuilder, MeshOptions};
use csx::primitives::BoxShape;
use csx::properties::Material;
use csx::{ContinuousStructure, Primitive, Property, TypeMask};
use ems::port::{MslPort, MslPortResult};
use ems::post::s_param;
use ems::prelude::*;

const EPS_R: f64 = 4.3;
const H: f64 = 0.8;
const W: f64 = 1.5;
const X: f64 = 30.0;
/// Distance between the reference planes \[mm\].
const LEN: f64 = 20.0;

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
        Primitive::new(BoxShape::new([-X, -15.0, 0.0], [X, 15.0, H])),
    )?;
    let cu = model.add_property(Property::metal("copper"));
    model.add_primitive(
        cu,
        Primitive::new(BoxShape::new([-X, -W / 2.0, H], [X, W / 2.0, H])).with_priority(10),
    )?;

    let p1 = MslPort::new(
        1,
        [-25.0, -W / 2.0, 0.0],
        [-LEN / 2.0, W / 2.0, H],
        Axis::X,
        Axis::Z,
    )
    .metal(cu)
    .feed_resistance(50.0)
    .excite(1.0);
    let p2 = MslPort::new(
        2,
        [25.0, -W / 2.0, 0.0],
        [LEN / 2.0, W / 2.0, H],
        Axis::X,
        Axis::Z,
    )
    .metal(cu);

    // Mesh (before adding the ports): 0.5 mm along the line, the thirds
    // rule with 0.25 mm across the strip edges (not along x, where the
    // strip ends with the substrate), 4 cells through the substrate, up to
    // 0.8 mm (y) and 1 mm (z) elsewhere, 12 mm of air above.
    let mut opts = MeshOptions::new(1.0)
        .with_max_res([0.5, 0.8, 1.0])
        .with_ratio(1.4)
        .with_thirds([f64::NAN, 0.25, f64::NAN])
        .with_mask(TypeMask::PHYSICAL);
    opts.margin = [[0.0, 0.0], [0.0, 0.0], [0.0, 12.0]];
    let mut mesher = MeshBuilder::new(opts).with_lines(2, (0..=4).map(|i| H * i as f64 / 4.0));
    for p in [&p1, &p2] {
        for (axis, lines) in p.mesh_lines().into_iter().enumerate() {
            mesher = mesher.with_lines(axis, lines);
        }
    }
    mesher.apply(&mut model)?;
    let p1 = p1.add_to(&mut model)?;
    let p2 = p2.add_to(&mut model)?;

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
    let r1: MslPortResult = p1.analyze(&res, &freqs)?;
    let r2 = p2.analyze(&res, &freqs)?;
    let s11 = r1.s11();
    let s21 = s_param(&r2, &r1);
    let (beta, eps_eff) = (r1.beta(), r1.eps_eff());
    let (e_ham, z_ham) = hammerstad(EPS_R, W, H);

    println!("Hammerstad: Z0 = {z_ham:.2} Ω, eps_eff = {e_ham:.3}");
    println!(" f [GHz]  |S11| [dB]  |S21| [dB]  ∠S21 + βL [deg]   Z_L [Ω]   eps_eff");
    let (mut z_sum, mut e_sum, mut cnt) = (0.0, 0.0, 0);
    for (k, &f) in freqs.iter().enumerate() {
        let z = r1.z_line()[k];
        if k % 4 == 3 {
            let dphi = (s21[k] * Complex64::cis(beta[k] * LEN * 1e-3)).arg();
            println!(
                "{:>7.2}  {:>10.2}  {:>10.3}  {:>15.3}  {:>8.2}  {:>8.3}",
                f / 1e9,
                20.0 * s11[k].norm().log10(),
                20.0 * s21[k].norm().log10(),
                dphi.to_degrees(),
                z.re,
                eps_eff[k]
            );
        }
        if (1e9..=3e9).contains(&f) {
            z_sum += z.re;
            e_sum += eps_eff[k];
            cnt += 1;
        }
    }
    let (z_avg, e_avg) = (z_sum / cnt as f64, e_sum / cnt as f64);
    println!(
        "1-3 GHz average: Z_L = {z_avg:.2} Ω ({:+.1} % vs Hammerstad), eps_eff = {e_avg:.3} ({:+.1} %)",
        (z_avg / z_ham - 1.0) * 100.0,
        (e_avg / e_ham - 1.0) * 100.0
    );
    Ok(())
}
