//! NF2FF over a stratified ground (design §5.10): an FDTD short dipole
//! over a dielectric half-space or a grounded slab, the NF2FF box in the
//! air above, far fields with the layered-medium (asymptotic) Green's
//! function.
//!
//! * Upper hemisphere against the textbook vertical-dipole pattern
//!   `sinθ·|1 + R_v·e^{−2jkh cosθ}|` (Balanis, *Antenna Theory*, §4.8).
//! * Power balance: the far-field power above plus the power transmitted
//!   into the half-space (including the part beyond the critical angle,
//!   from the evanescent spectrum of the source) equals the power leaving
//!   the box.
//! * Grounded slab: a vacuum "slab" on PEC conserves power; with a
//!   dielectric the far field does not depend on the box (the surface wave
//!   carries the power the far field misses).

mod common;

use std::f64::consts::PI;

use csx::primitives::BoxShape;
use csx::{ContinuousStructure, Primitive, Property};
use ems::engine::Medium;
use ems::nf2ff::{Ground, Nf2ffBox, Nf2ffResult};
use ems::prelude::*;

/// Mesh lines `-n·d ..= n·d` (mm).
fn lines(n: i32, d: f64) -> Vec<f64> {
    (-n..=n).map(|i| f64::from(i) * d).collect()
}

/// A z-directed one-cell lumped port at z = 0 … 10 mm, a dielectric
/// (`eps_r`) below `z_i` mm, 10 mm cells, ±18 cells with an 8-cell PML;
/// `pec_below` cuts the domain at z = −80 mm with a PEC face (a grounded
/// slab). One NF2FF result per box `(z_bottom, half width, z_top)` (mm).
fn run(
    eps_r: f64,
    z_i: f64,
    pec_below: bool,
    freqs: &[f64],
    boxes: &[[f64; 3]],
) -> Vec<Nf2ffResult> {
    let d = 10.0;
    let half = 18;
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    model.grid.set_lines(0, lines(half, d)).unwrap();
    model.grid.set_lines(1, lines(half, d)).unwrap();
    let z: Vec<f64> = if pec_below {
        (-8..=half).map(|i| f64::from(i) * d).collect()
    } else {
        lines(half, d)
    };
    model.grid.set_lines(2, z).unwrap();
    let sub = model.add_property(Property::material("ground", csx::Material::new(eps_r, 0.0)));
    let e = f64::from(half + 1) * d;
    model
        .add_primitive(
            sub,
            Primitive::new(BoxShape::new([-e, -e, -e], [e, e, z_i])),
        )
        .unwrap();
    LumpedPort::new(1, 50.0, [0.0, 0.0, 0.0], [0.0, 0.0, d], Axis::Z)
        .excite(1.0)
        .add_to(&mut model)
        .unwrap();
    let mut b = Boundaries::all(Boundary::pml(8));
    if pec_below {
        b.set(Face::ZMin, Boundary::Pec);
    }
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(0.9e9, 0.45e9))
        .boundaries(b)
        .end_criteria(EndCriteria::energy_decay_db(-50.0))
        .max_timesteps(20_000);
    let mut sim = Simulation::new(&model, &fdtd).unwrap();
    let hs: Vec<_> = boxes
        .iter()
        .map(|&[zb, w, zt]| {
            sim.add_nf2ff(&Nf2ffBox::new([-w, -w, zb], [w, w, zt], freqs.to_vec()))
                .unwrap()
        })
        .collect();
    let bad: Vec<_> = sim
        .warnings()
        .iter()
        .filter(|w| w.contains("NF2FF"))
        .collect();
    assert!(bad.is_empty(), "{bad:?}");
    sim.run().unwrap();
    hs.iter().map(|h| sim.nf2ff(h)).collect()
}

/// Power in the upper hemisphere (midpoint rule, `nt` θ steps; the
/// pattern is φ-symmetric).
fn upper_power(ff: &Nf2ffResult, k: usize, nt: usize) -> f64 {
    let dt = 0.5 * PI / nt as f64;
    (0..nt)
        .map(|a| {
            let t = (a as f64 + 0.5) * dt;
            ff.radiation_intensity(k, t, 0.0) * t.sin()
        })
        .sum::<f64>()
        * dt
        * 2.0
        * PI
}

#[test]
fn dipole_over_a_dielectric_half_space() {
    let _t = common::Timer::start("dipole_over_a_dielectric_half_space");
    let eps = 4.0;
    let z_i = -40.0;
    let freqs = [0.8e9, 1e9];
    let ff = run(eps, z_i, false, &freqs, &[[z_i + 10.0, 60.0, 70.0]])
        .remove(0)
        .with_ground(Ground::half_space(
            Axis::Z,
            z_i * 1e-3,
            Medium::lossless(eps, 1.0),
        ))
        .unwrap();
    // The dipole's centre (z = 5 mm) over the interface.
    let h = (5.0 - z_i) * 1e-3;
    for (k, &f) in freqs.iter().enumerate() {
        let kw = 2.0 * PI * f / 299_792_458.0;
        let pattern = |t: f64| {
            let (st, ct) = t.sin_cos();
            let root = (eps - st * st).sqrt();
            let rv = (eps * ct - root) / (eps * ct + root);
            st * (Complex64::ONE + Complex64::cis(-2.0 * kw * h * ct) * rv).norm()
        };
        let t0 = 60f64.to_radians();
        let norm = ff.far_field(k, t0, 0.0).e_theta.norm() / pattern(t0);
        let mut worst = 0.0f64;
        for deg in [15.0, 30.0, 45.0, 75.0, 85.0] {
            let t = f64::to_radians(deg);
            for phi in [0.0, 1.3] {
                let got = ff.far_field(k, t, phi);
                worst = worst.max((got.e_theta.norm() / norm - pattern(t)).abs() / pattern(t0));
                assert!(got.e_phi.norm() < 1e-2 * norm);
            }
        }
        let p_box = ff.radiated_power(k);
        // Fine θ steps: the transmitted lobe at the critical angle is
        // sharp.
        let p_far = ff.radiated_power_far(k, 360, 4);
        let up = upper_power(&ff, k, 90);
        let d = ff.max_directivity(k);
        eprintln!(
            "f = {:.1} GHz: upper pattern vs textbook ≤ {worst:.1e} (of the 60° value), \
             P_far/P_box = {:.4}, into the ground {:.1} %, D_max = {:.2} at θ = {:.1}°",
            f / 1e9,
            p_far / p_box,
            100.0 * (1.0 - up / p_far),
            d.directivity,
            d.theta.to_degrees()
        );
        assert!(worst < 3e-3, "{worst}");
        assert!((p_far / p_box - 1.0).abs() < 2e-2, "{}", p_far / p_box);
        // A dense half-space takes most of the power.
        assert!(up / p_far < 0.3);
    }
}

#[test]
fn dipole_over_a_grounded_slab() {
    let _t = common::Timer::start("dipole_over_a_grounded_slab");
    let z_i = -40.0;
    let freqs = [1e9];
    let slab =
        |eps: f64| Ground::grounded_slab(Axis::Z, z_i * 1e-3, 0.04, Medium::lossless(eps, 1.0));
    // A vacuum layer on the PEC face 80 mm down: no surface wave.
    let ff = run(1.0, z_i, true, &freqs, &[[z_i + 10.0, 60.0, 70.0]])
        .remove(0)
        .with_ground(slab(1.0))
        .unwrap();
    let vac = ff.radiated_power_far(0, 180, 4) / ff.radiated_power(0);
    // ε_r = 3: two boxes give the same far field.
    let ffs = run(
        3.0,
        z_i,
        true,
        &freqs,
        &[[z_i + 10.0, 60.0, 70.0], [z_i + 30.0, 80.0, 90.0]],
    );
    // (The same phase centre for both.)
    let [a, b] = [0, 1].map(|i| {
        ffs[i]
            .clone()
            .with_center([0.0, 0.0, 0.02])
            .with_ground(slab(3.0))
            .unwrap()
    });
    let peak = a.max_directivity(0);
    let ua = a.radiation_intensity(0, peak.theta, 0.0);
    let mut worst = 0.0f64;
    for deg in [10.0, 30.0, 50.0, 70.0, 80.0, 88.0] {
        let t = f64::to_radians(deg);
        let (fa, fb) = (a.far_field(0, t, 0.4), b.far_field(0, t, 0.4));
        let du = (fa.e_theta - fb.e_theta).norm_sqr() / (2.0 * a.wave_impedance());
        worst = worst.max((du / ua).sqrt());
    }
    let eff = a.radiated_power_far(0, 180, 4) / a.radiated_power(0);
    eprintln!(
        "grounded slab: vacuum layer P_far/P_box = {vac:.4}; ε_r = 3: box-to-box far-field \
         difference ≤ {worst:.1e} of the peak, space-wave share {:.1} % (the rest is the TM0 \
         surface wave), D_max = {:.2} at θ = {:.1}°",
        100.0 * eff,
        peak.directivity,
        peak.theta.to_degrees()
    );
    assert!((vac - 1.0).abs() < 2e-2, "{vac}");
    assert!(worst < 2e-2, "{worst}");
    assert!(eff < 1.0);
    // No far field below the ground plane.
    assert_eq!(a.radiation_intensity(0, 2.5, 0.0), 0.0);
}
