//! NF2FF validation: short (Hertzian) dipole directivity and pattern,
//! half-wave dipole directivity, power balance (box flux, far-field
//! integral, port accepted power), image theory over a PEC ground and
//! thread determinism.

use std::f64::consts::PI;

use csx::primitives::BoxShape;
use csx::{ContinuousStructure, Primitive, Property};
use ems::nf2ff::{MirrorKind, Nf2ffBox, Nf2ffResult};
use ems::prelude::*;

/// Uniform mesh lines `-n·d ..= n·d` (drawing units: mm).
fn lines(n: i32, d: f64) -> Vec<f64> {
    (-n..=n).map(|i| f64::from(i) * d).collect()
}

/// A z-directed lumped port of `gap_cells` cells (cell size `d` mm) from
/// the origin up, with optional PEC arms of `arm` cells each, in a cube of
/// `±half` cells with an 8-cell CPML. With `ground`, the domain is cut at
/// z = 0 by a PEC boundary and the port sits on it (a monopole).
struct Dipole {
    d: f64,
    half: i32,
    gap_cells: i32,
    arm: i32,
    ground: bool,
    f0: f64,
    threads: usize,
    /// Relative permittivity of the background filling the domain.
    eps_r: f64,
}

impl Dipole {
    fn run(&self, box_cells: i32, freqs: &[f64]) -> (Nf2ffResult, PortResult) {
        let d = self.d;
        let mut model = ContinuousStructure::new();
        model.grid.delta_unit = 1e-3;
        model.grid.set_lines(0, lines(self.half, d)).unwrap();
        model.grid.set_lines(1, lines(self.half, d)).unwrap();
        let z: Vec<f64> = if self.ground {
            (0..=self.half).map(|i| f64::from(i) * d).collect()
        } else {
            lines(self.half, d)
        };
        model.grid.set_lines(2, z).unwrap();
        if self.eps_r != 1.0 {
            let bg = model.add_property(Property::material(
                "background",
                csx::Material::new(self.eps_r, 0.0),
            ));
            let e = f64::from(self.half + 1) * d;
            model
                .add_primitive(bg, Primitive::new(BoxShape::new([-e; 3], [e; 3])))
                .unwrap();
        }
        // The port spans mesh lines 0 .. gap_cells.
        let (z0, z1) = (0.0, f64::from(self.gap_cells) * d);
        if self.arm > 0 {
            let wire = model.add_property(Property::metal("wire"));
            let len = f64::from(self.arm) * d;
            model
                .add_primitive(
                    wire,
                    Primitive::new(BoxShape::new([0.0, 0.0, z1], [0.0, 0.0, z1 + len])),
                )
                .unwrap();
            if !self.ground {
                model
                    .add_primitive(
                        wire,
                        Primitive::new(BoxShape::new([0.0, 0.0, z0 - len], [0.0, 0.0, z0])),
                    )
                    .unwrap();
            }
        }
        let port = LumpedPort::new(1, 50.0, [0.0, 0.0, z0], [0.0, 0.0, z1], Axis::Z)
            .excite(1.0)
            .add_to(&mut model)
            .unwrap();
        let mut b = Boundaries::all(Boundary::pml(8));
        if self.ground {
            b.set(Face::ZMin, Boundary::Pec);
        }
        let mut fdtd = Fdtd::new()
            .waveform(Waveform::gaussian(self.f0, 0.5 * self.f0))
            .boundaries(b)
            .end_criteria(EndCriteria::energy_decay_db(-50.0))
            .max_timesteps(20_000);
        if self.threads > 0 {
            fdtd = fdtd.threads(self.threads);
        }
        let mut sim = Simulation::new(&model, &fdtd).unwrap();
        let s = f64::from(box_cells) * d;
        let mut nb = Nf2ffBox::new([-s, -s, -s], [s, s, s], freqs.to_vec());
        if self.ground {
            nb = Nf2ffBox::new([-s, -s, 0.0], [s, s, s], freqs.to_vec()).mirror(
                Axis::Z,
                0.0,
                MirrorKind::Pec,
            );
        }
        let h = sim.add_nf2ff(&nb).unwrap();
        let bad: Vec<_> = sim
            .warnings()
            .iter()
            .filter(|w| w.contains("NF2FF"))
            .collect();
        assert!(bad.is_empty(), "{bad:?}");
        let res = sim.run().unwrap();
        let pr = port.analyze(&res, freqs).unwrap();
        (sim.nf2ff(&h), pr)
    }
}

fn short_dipole(ground: bool, threads: usize) -> Dipole {
    Dipole {
        d: 10.0,
        half: 18,
        gap_cells: 1,
        arm: 0,
        ground,
        f0: 1e9,
        threads,
        eps_r: 1.0,
    }
}

#[test]
fn short_dipole_directivity_and_pattern() {
    let (ff, _) = short_dipole(false, 0).run(6, &[0.7e9, 1e9]);
    for k in 0..2 {
        let p_box = ff.radiated_power(k);
        let p_far = ff.radiated_power_far(k, 36, 12);
        let d = ff.max_directivity(k);
        eprintln!(
            "short dipole f = {:.2} GHz: D_max = {:.4} ({:.3} dBi) at θ = {:.2}°, P_far/P_box = {:.5}",
            ff.freqs()[k] / 1e9,
            d.directivity,
            d.dbi(),
            d.theta.to_degrees(),
            p_far / p_box
        );
        assert!((d.directivity / 1.5 - 1.0).abs() < 0.02, "{d:?}");
        assert!((d.theta - PI / 2.0).abs() < 0.05, "{d:?}");
        assert!((p_far / p_box - 1.0).abs() < 0.01, "{p_far} vs {p_box}");
        // sin²θ pattern, no cross-polarisation, no φ dependence.
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
fn monopole_over_ground_doubles_directivity() {
    let (ff, _) = short_dipole(true, 0).run(6, &[1e9]);
    let d = ff.max_directivity(0);
    let p_box = ff.radiated_power(0);
    let p_far = ff.radiated_power_far(0, 36, 12);
    eprintln!(
        "short monopole: D_max = {:.4} ({:.3} dBi) at θ = {:.2}°, P_far/P_box = {:.5}",
        d.directivity,
        d.dbi(),
        d.theta.to_degrees(),
        p_far / p_box
    );
    assert!((d.directivity / 3.0 - 1.0).abs() < 0.02, "{d:?}");
    assert!(d.theta <= PI / 2.0 + 1e-9);
    assert!((p_far / p_box - 1.0).abs() < 0.01, "{p_far} vs {p_box}");
}

#[test]
fn half_wave_dipole_directivity_and_power_balance() {
    // 5 mm cells, 2 × 15-cell arms + 1-cell gap: L = 155 mm, resonant near
    // 0.92 GHz (L ≈ 0.475 λ).
    let dip = Dipole {
        d: 5.0,
        half: 34,
        gap_cells: 1,
        arm: 15,
        ground: false,
        f0: 0.9e9,
        threads: 0,
        eps_r: 1.0,
    };
    let freqs = [0.92e9];
    let (ff, port) = dip.run(22, &freqs);
    let d = ff.max_directivity(0);
    let p_box = ff.radiated_power(0);
    let p_far = ff.radiated_power_far(0, 36, 12);
    let p_acc = port.accepted_power()[0];
    let g = ff.gain(0, d.theta, d.phi, p_acc);
    eprintln!(
        "half-wave dipole: Z_in = {:.1} Ω, D_max = {:.4} ({:.3} dBi), G = {:.4}, \
         P_far/P_box = {:.5}, P_box/P_acc = {:.5}",
        port.z_in()[0],
        d.directivity,
        d.dbi(),
        g,
        p_far / p_box,
        p_box / p_acc
    );
    // Thin half-wave dipole: D = 1.64 (2.15 dBi), Balanis §4.6.
    assert!((d.directivity / 1.64 - 1.0).abs() < 0.02, "{d:?}");
    assert!((p_far / p_box - 1.0).abs() < 0.01, "{p_far} vs {p_box}");
    assert!((p_box / p_acc - 1.0).abs() < 0.02, "{p_box} vs {p_acc}");
    assert!((g / d.directivity - 1.0).abs() < 0.02);
}

#[test]
fn nf2ff_is_thread_independent() {
    let small = Dipole {
        d: 10.0,
        half: 14,
        gap_cells: 1,
        arm: 0,
        ground: false,
        f0: 1e9,
        threads: 1,
        eps_r: 1.0,
    };
    let (a, _) = small.run(4, &[1e9]);
    let (b, _) = Dipole {
        threads: 3,
        ..small
    }
    .run(4, &[1e9]);
    for &(t, p) in &[(0.3, 0.1), (1.5, 2.0), (2.5, 4.0)] {
        assert_eq!(a.far_field(0, t, p), b.far_field(0, t, p));
    }
    assert_eq!(a.radiated_power(0), b.radiated_power(0));
}

/// Radiated power and far field of a short dipole of moment `I·l` in a
/// lossless medium: `P = η·k²·|Il|²/(12π)`, `|rE_θ(90°)| = η·k·|Il|/(4π)`.
fn hertzian(ff: &Nf2ffResult, k: usize, il: f64) -> (f64, f64) {
    let (kw, eta) = (ff.wavenumber(k), ff.wave_impedance());
    (
        eta * kw * kw * il * il / (12.0 * PI),
        eta * kw * il / (4.0 * PI),
    )
}

/// The short dipole in vacuum and embedded in an ε_r = 4 background (cells
/// halved, so both have λ/30 cells): directivity 1.5, and box power and
/// far field against the Hertzian dipole of the measured port current
/// (`P ∝ η·k² = √ε_r·η0·k0²` at equal current; `|rE| ∝ η·k` is the same).
#[test]
fn short_dipole_in_dielectric_background() {
    let freqs = [1e9];
    let mut out = Vec::new();
    for (eps_r, d) in [(1.0, 10.0), (4.0, 5.0)] {
        let dip = Dipole {
            d,
            eps_r,
            ..short_dipole(false, 0)
        };
        let (ff, port) = dip.run(6, &freqs);
        let i = port.current()[0].norm();
        let (p_h, e_h) = hertzian(&ff, 0, i * d * 1e-3);
        let p_box = ff.radiated_power(0);
        let p_far = ff.radiated_power_far(0, 36, 12);
        let e90 = ff.far_field(0, PI / 2.0, 0.4).e_theta.norm();
        let dmax = ff.max_directivity(0);
        eprintln!(
            "short dipole, ε_r = {eps_r}: detected ε_r = {:.6}, D_max = {:.4}, \
             P_far/P_box = {:.5}, P_box/P_hertz = {:.4}, |rE_θ|/hertz = {:.4}, \
             P_box/|I|² = {:.4e} Ω",
            ff.background().eps_r,
            dmax.directivity,
            p_far / p_box,
            p_box / p_h,
            e90 / e_h,
            p_box / (i * i)
        );
        assert!((ff.background().eps_r / eps_r - 1.0).abs() < 1e-5);
        assert!((dmax.directivity / 1.5 - 1.0).abs() < 0.02, "{dmax:?}");
        assert!((p_far / p_box - 1.0).abs() < 0.01);
        assert!((p_box / p_h - 1.0).abs() < 0.03, "{p_box} vs {p_h}");
        assert!((e90 / e_h - 1.0).abs() < 0.02, "{e90} vs {e_h}");
        out.push((p_box / (i * i * d * d), e90 / (i * d)));
    }
    // Same moment: power ratio √ε_r = 2, far-field magnitude ratio 1.
    let (pr, er) = (out[1].0 / out[0].0, out[1].1 / out[0].1);
    eprintln!("ε_r = 4 vs vacuum at equal moment: P ratio {pr:.4}, |rE| ratio {er:.4}");
    assert!((pr / 2.0 - 1.0).abs() < 0.03, "{pr}");
    assert!((er - 1.0).abs() < 0.02, "{er}");
}

/// Background detection on the NF2FF box: an inhomogeneous surface warns
/// (vacuum assumed), a lossy one is refused, an explicit background wins.
#[test]
fn nf2ff_background_detection() {
    let build = |kappa: f64, cross: bool| {
        let mut model = ContinuousStructure::new();
        model.grid.delta_unit = 1e-3;
        for a in 0..3 {
            model.grid.set_lines(a, lines(14, 10.0)).unwrap();
        }
        let bg = model.add_property(Property::material("bg", csx::Material::new(2.0, kappa)));
        let (lo, hi) = if cross {
            ([-150.0, -150.0, -150.0], [150.0, 150.0, 0.0])
        } else {
            ([-150.0; 3], [150.0; 3])
        };
        model
            .add_primitive(bg, Primitive::new(BoxShape::new(lo, hi)))
            .unwrap();
        let fdtd = Fdtd::new()
            .waveform(Waveform::gaussian(1e9, 0.5e9))
            .boundaries(Boundaries::all(Boundary::pml(4)))
            .max_timesteps(10);
        Simulation::new(&model, &fdtd).unwrap()
    };
    let nb = Nf2ffBox::new([-60.0; 3], [60.0; 3], [1e9]);
    let mut sim = build(0.0, false);
    let h = sim.add_nf2ff(&nb).unwrap();
    assert!((sim.nf2ff(&h).background().eps_r / 2.0 - 1.0).abs() < 1e-5);
    let mut sim = build(0.0, true);
    let h = sim.add_nf2ff(&nb).unwrap();
    assert_eq!(sim.nf2ff(&h).background().eps_r, 1.0);
    assert!(
        sim.warnings()
            .iter()
            .any(|w| w.contains("NF2FF") && w.contains("not homogeneous")),
        "{:?}",
        sim.warnings()
    );
    let mut sim = build(0.01, false);
    assert!(matches!(
        sim.add_nf2ff(&nb),
        Err(ems::Error::Unsupported(_))
    ));
    let h = sim.add_nf2ff(&nb.clone().background(2.0, 1.0)).unwrap();
    assert_eq!(sim.nf2ff(&h).background().eps_r, 2.0);
}
