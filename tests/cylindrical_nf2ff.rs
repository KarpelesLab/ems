//! NF2FF on a cylindrical grid (design §5.10, §5.15): a short dipole on
//! the axis of a `(ρ, α, z)` grid with an axis and a closed azimuth,
//! cylindrical PML on `ρ_max` and CPML on the z faces. The Huygens surface
//! is a cylinder (`ρ = const` side and two `z = const` caps); its fields
//! are rotated to Cartesian components. Directivity 1.5 at θ = 90°, a
//! `sin²θ` pattern without φ dependence, and the box flux equals the
//! far-field integral and the Hertzian-dipole power of the port current.

use std::f64::consts::{PI, TAU};

use csx::{ContinuousStructure, CoordSystem};
use ems::nf2ff::{MirrorKind, Nf2ffBox, Nf2ffResult};
use ems::prelude::*;

const NA: usize = 16;
const D: f64 = 10.0; // mm

/// Returns the far field and the port result at `freqs`; `ground`: the
/// domain starts at z = 0 on a PEC plane (monopole with a z mirror).
fn run(freqs: &[f64], ground: bool, threads: usize) -> (Nf2ffResult, PortResult, Vec<String>) {
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid
        .set_lines(0, (0..=18).map(|i| f64::from(i) * D))
        .unwrap();
    m.grid
        .set_lines(1, (0..=NA).map(|j| j as f64 * TAU / NA as f64))
        .unwrap();
    let z: Vec<f64> = if ground {
        (0..=18).map(|i| f64::from(i) * D).collect()
    } else {
        (-18..=18).map(|i| f64::from(i) * D).collect()
    };
    m.grid.set_lines(2, z).unwrap();
    let port = LumpedPort::new(1, 50.0, [0.0, 0.0, 0.0], [0.0, 0.0, D], Axis::Z)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let mut b = Boundaries::all(Boundary::Pec);
    b.set(Face::XMax, Boundary::pml(8));
    b.set(Face::ZMax, Boundary::pml(8));
    b.set(
        Face::ZMin,
        if ground {
            Boundary::Pec
        } else {
            Boundary::pml(8)
        },
    );
    let mut fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(1e9, 0.5e9))
        .boundaries(b)
        .end_criteria(EndCriteria::energy_decay_db(-50.0))
        .max_timesteps(40_000);
    if threads > 0 {
        fdtd = fdtd.threads(threads);
    }
    let mut sim = Simulation::new(&m, &fdtd).unwrap();
    let s = 6.0 * D;
    let nb = if ground {
        Nf2ffBox::new([0.0, 0.0, 0.0], [s, TAU, s], freqs.to_vec()).mirror(
            Axis::Z,
            0.0,
            MirrorKind::Pec,
        )
    } else {
        Nf2ffBox::new([0.0, 0.0, -s], [s, TAU, s], freqs.to_vec())
    };
    let h = sim.add_nf2ff(&nb).unwrap();
    let w = sim.warnings().to_vec();
    let res = sim.run().unwrap();
    let pr = port.analyze(&res, freqs).unwrap();
    (sim.nf2ff(&h), pr, w)
}

#[test]
fn short_dipole_on_the_axis() {
    let freqs = [0.7e9, 1e9];
    let (ff, port, w) = run(&freqs, false, 0);
    let bad: Vec<_> = w.iter().filter(|w| w.contains("NF2FF")).collect();
    assert!(bad.is_empty(), "{bad:?}");
    assert_eq!(ff.center(), [0.0, 0.0, 0.0]);
    for (k, &f) in freqs.iter().enumerate() {
        let p_box = ff.radiated_power(k);
        let p_far = ff.radiated_power_far(k, 36, 12);
        let d = ff.max_directivity(k);
        // Hertzian dipole of the port current over the one-cell gap.
        let il = port.current()[k].norm() * D * 1e-3;
        let kw = ff.wavenumber(k);
        let p_hertz = Z0 * kw * kw * il * il / (12.0 * PI);
        eprintln!(
            "f = {:.2} GHz: D_max = {:.4} at θ = {:.2}°, P_far/P_box = {:.5}, \
             P_box/P_Hertz = {:.4}",
            f / 1e9,
            d.directivity,
            d.theta.to_degrees(),
            p_far / p_box,
            p_box / p_hertz
        );
        assert!((d.directivity / 1.5 - 1.0).abs() < 0.02, "{d:?}");
        assert!((d.theta - PI / 2.0).abs() < 0.05, "{d:?}");
        assert!((p_far / p_box - 1.0).abs() < 0.01, "{p_far} vs {p_box}");
        assert!((p_box / p_hertz - 1.0).abs() < 0.03, "{p_box} vs {p_hertz}");
        let u90 = ff.radiation_intensity(k, PI / 2.0, 0.3);
        for &t in &[0.2, 0.5, 0.8, 1.2, 2.0, 2.7] {
            for &p in &[0.0, 1.0, 2.5, 4.0] {
                let f = ff.far_field(k, t, p);
                let want = u90 * t.sin().powi(2);
                let err = (f.intensity() - want).abs() / u90;
                assert!(err < 0.01, "θ = {t}, φ = {p}: {} vs {want}", f.intensity());
                assert!(f.e_phi.norm() < 0.01 * f.e_theta.norm().max(1e-30));
            }
        }
    }
}

#[test]
fn monopole_over_ground_with_a_z_mirror() {
    let (ff, _, _) = run(&[1e9], true, 0);
    let d = ff.max_directivity(0);
    let p_box = ff.radiated_power(0);
    let p_far = ff.radiated_power_far(0, 36, 12);
    eprintln!(
        "monopole: D_max = {:.4} at θ = {:.2}°, P_far/P_box = {:.5}",
        d.directivity,
        d.theta.to_degrees(),
        p_far / p_box
    );
    assert!((d.directivity / 3.0 - 1.0).abs() < 0.02, "{d:?}");
    assert!((p_far / p_box - 1.0).abs() < 0.01);
    // Only z mirrors on cylindrical grids.
    let m = ems::nf2ff::Mirror::new(Axis::X, 0.0, MirrorKind::Pec);
    assert!(matches!(ff.with_mirror(m), Err(ems::Error::Unsupported(_))));
}
