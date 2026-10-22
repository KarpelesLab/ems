//! The coplanar-waveguide and strip-line ports ([`CpwPort`],
//! [`StriplinePort`]) on cylindrical grids (design §5.5, §5.15), on the
//! uniform line geometries they accept there (as [`MslPort`] does,
//! `tests/validation_msl_cyl.rs`), each against an exact discrete or a
//! closed-form reference:
//!
//! 1. **Strip line along z** between two coaxial ground cylinders, the
//!    strip a cylinder over the whole azimuth: two coaxial lines in
//!    parallel, `Z = Z_in·Z_out/(Z_in + Z_out)` with the discrete coax
//!    values `η/(2π)·Σ Δρ_i/ρ_{i+½}`; `β` the 1-D Yee wavenumber.
//! 2. **CPW along z** on a cylinder (`ρ = const` metal, widths and gaps
//!    angles) inside a shielded closed azimuth: the TEM mode of a
//!    z-uniform Yee line is the discrete electrostatic solution of its
//!    cross-section (edge capacitances `ε·(dual length)/(edge length)`
//!    in `(ρ, α)`), so `Z = 1/(c·C')` with `C'` from a 2-D Laplace solve
//!    on the same mesh.
//! 3. **Strip line along α** between two `z` planes at a large radius,
//!    full width between PMC `ρ` walls: the parallel-plate value
//!    `η·b/(4w)`, `β = k` per metre of arc, to `O((w/R)²)`.
//! 4. **CPW along α** on a `z` plane at a large radius: the flat CPW of
//!    the same discrete cross-section (`(ρ, z)` as Cartesian), to
//!    `O((w/R)²)`.

mod common;

use core::f64::consts::{PI, TAU};

use common::*;
use csx::primitives::BoxShape;
use csx::properties::Material;
use csx::{ContinuousStructure, CoordSystem, Primitive, Property};
use ems::constants::{C0, Z0};
use ems::port::{CpwPort, LinePortResult, MslPort, StriplinePort};
use ems::post::s_param;
use ems::prelude::*;

fn lines(from: f64, to: f64, step: f64) -> Vec<f64> {
    let n = ((to - from) / step).round() as usize;
    (0..=n).map(|k| from + k as f64 * step).collect()
}

fn cylindrical(rho: Vec<f64>, alpha: Vec<f64>, z: Vec<f64>) -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, rho).unwrap();
    m.grid.set_lines(1, alpha).unwrap();
    m.grid.set_lines(2, z).unwrap();
    m
}

fn run(model: &ContinuousStructure, fdtd: &Fdtd) -> Results {
    let mut sim = Simulation::new(model, fdtd).unwrap();
    let w: Vec<_> = sim
        .warnings()
        .iter()
        .filter(|w| !w.contains("mesh"))
        .cloned()
        .collect();
    assert!(w.is_empty(), "{w:?}");
    sim.run().unwrap()
}

/// Draws the metal boxes `boxes` (opposite corners, drawing units): the
/// line over the whole length, through both absorbing ends (each port
/// writes its metal over its own length only).
fn line_metal(m: &mut ContinuousStructure, boxes: &[([f64; 3], [f64; 3])]) {
    let id = m.add_property(Property::metal("line"));
    for &(a, b) in boxes {
        m.add_primitive(id, Primitive::new(BoxShape::new(a, b)))
            .unwrap();
    }
}

/// Worst `|S11|` (dB) and worst `|S21 − e^{−jβL}|` of a through line.
fn through(r1: &LinePortResult, r2: &LinePortResult, len: f64) -> (f64, f64) {
    let s11 = r1.s11();
    let s21 = s_param(r2, r1);
    let beta = r1.beta();
    let (mut w11, mut w21) = (f64::NEG_INFINITY, 0.0f64);
    for k in 0..s11.len() {
        w11 = w11.max(db(s11[k].norm()));
        w21 = w21.max((s21[k] - Complex64::cis(-beta[k] * len)).norm());
    }
    (w11, w21)
}

/// Worst relative `|Z_L − z|` and worst `|β/k_Yee − 1|` of both ports
/// (`dl` the cell along the line \[m\]).
fn line_errors(rs: [&LinePortResult; 2], freqs: &[f64], dt: f64, dl: f64, z: f64) -> (f64, f64) {
    let (mut ez, mut eb) = (0.0f64, 0.0f64);
    for (k, &f) in freqs.iter().enumerate() {
        let k_yee = yee_k(f, dt, dl, 1.0, 0.0).re / dl;
        for r in rs {
            ez = ez.max((r.z_line()[k] - z).norm() / z);
            eb = eb.max((r.beta()[k] / k_yee - 1.0).abs());
        }
    }
    (ez, eb)
}

/// The discrete coax impedance between radii `a` and `b` on the ρ lines
/// `rho` (vacuum).
fn coax_z(rho: &[f64], a: f64, b: f64) -> f64 {
    rho.windows(2)
        .filter(|w| w[0] >= a - 1e-9 && w[1] <= b + 1e-9)
        .map(|w| (w[1] - w[0]) / (0.5 * (w[0] + w[1])))
        .sum::<f64>()
        * Z0
        / TAU
}

/// The per-unit-length vacuum capacitance \[F/m\] of a 2-D node network:
/// nodes `(i, j)` on lines `u` (index `i`) and `v` (index `j`, periodic
/// if `periodic`), edge capacitances `ε0·(dual length)/(edge length)`
/// with the cylindrical `(ρ, α)` geometry if `cyl`, else Cartesian;
/// `fixed(i, j)` the conductor potentials (`Some(1)` on the strip,
/// `Some(0)` on grounds). Conjugate gradients on the free nodes.
fn capacitance(
    u: &[f64],
    v: &[f64],
    periodic: bool,
    cyl: bool,
    fixed: impl Fn(usize, usize) -> Option<f64>,
) -> f64 {
    let (nu, nv) = (u.len(), if periodic { v.len() - 1 } else { v.len() });
    let du = |i: usize| u[i + 1] - u[i];
    let dv_edge = |j: usize| v[j + 1] - v[j];
    let dual = |d: &dyn Fn(usize) -> f64, i: usize, n: usize, per: bool| -> f64 {
        let lo = if i > 0 {
            d(i - 1)
        } else if per {
            d(n - 1)
        } else {
            0.0
        };
        let hi = if i + 1 < n || per { d(i) } else { 0.0 };
        0.5 * (lo + hi)
    };
    // Edges: (node a, node b, capacitance).
    let id = |i: usize, j: usize| i * nv + j;
    let mut edges = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            // u edge (i → i+1, j).
            if i + 1 < nu {
                let dvs = dual(&dv_edge, j, nv, periodic);
                let c = if cyl {
                    0.5 * (u[i] + u[i + 1]) * dvs / du(i)
                } else {
                    dvs / du(i)
                };
                edges.push((id(i, j), id(i + 1, j), c));
            }
            // v edge (i, j → j+1).
            if j + 1 < nv || periodic {
                let jn = (j + 1) % nv;
                let dus = dual(&du, i, nu, false);
                let c = if cyl {
                    dus / (u[i] * dv_edge(j))
                } else {
                    dus / dv_edge(j)
                };
                edges.push((id(i, j), id(i, jn), c));
            }
        }
    }
    let n = nu * nv;
    let mut phi = vec![0.0; n];
    let mut free = vec![true; n];
    for i in 0..nu {
        for j in 0..nv {
            if let Some(p) = fixed(i, j) {
                phi[id(i, j)] = p;
                free[id(i, j)] = false;
            }
        }
    }
    // A·x = b on the free nodes: A = Σ c (e_a − e_b)(e_a − e_b)ᵀ.
    let apply = |x: &[f64], out: &mut [f64]| {
        out.iter_mut().for_each(|o| *o = 0.0);
        for &(a, b, c) in &edges {
            let d = c * (x[a] - x[b]);
            out[a] += d;
            out[b] -= d;
        }
    };
    let mut r = vec![0.0; n];
    apply(&phi, &mut r);
    for k in 0..n {
        r[k] = if free[k] { -r[k] } else { 0.0 };
    }
    let mut p = r.clone();
    let mut ap = vec![0.0; n];
    let mut rr: f64 = r.iter().map(|x| x * x).sum();
    let rr0 = rr;
    for _ in 0..20 * n {
        if rr <= 1e-28 * rr0 {
            break;
        }
        apply(&p, &mut ap);
        for k in 0..n {
            if !free[k] {
                ap[k] = 0.0;
            }
        }
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        let alpha = rr / pap;
        for k in 0..n {
            phi[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        let rr1: f64 = r.iter().map(|x| x * x).sum();
        let beta = rr1 / rr;
        for k in 0..n {
            p[k] = r[k] + beta * p[k];
        }
        rr = rr1;
    }
    let eps0 = 1.0 / (Z0 * C0);
    eps0 * edges
        .iter()
        .map(|&(a, b, c)| c * (phi[a] - phi[b]).powi(2))
        .sum::<f64>()
}

#[test]
fn stripline_along_z_between_coaxial_grounds_is_two_coaxes() {
    let (a, b, dr, dz) = (1.0, 3.0, 0.1, 0.5);
    let s = 0.5 * (a + b);
    let rho = lines(a, b, dr);
    let mut m = cylindrical(
        rho.clone(),
        (0..=8).map(|j| j as f64 * TAU / 8.0).collect(),
        lines(0.0, 60.0, dz),
    );
    // Grounds: the PEC ρ faces; the strip on the mid radius s over the
    // whole azimuth.
    let p1 = StriplinePort::new(1, [a, 0.0, 6.0], [b, TAU, 26.0], Axis::Z, Axis::X)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let p2 = StriplinePort::new(2, [a, 0.0, 54.0], [b, TAU, 34.0], Axis::Z, Axis::X)
        .add_to(&mut m)
        .unwrap();
    line_metal(&mut m, &[([s, 0.0, 0.0], [s, TAU, 60.0])]);
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(5e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::ZMin, Boundary::pml(8))
        .boundary(Face::ZMax, Boundary::pml(8))
        .end_criteria(EndCriteria::energy_decay_db(-80.0))
        .max_timesteps(30_000);
    let res = run(&m, &fdtd);
    let freqs = linspace(2e9, 8e9, 7);
    let r1 = p1.analyze(&res, &freqs).unwrap();
    let r2 = p2.analyze(&res, &freqs).unwrap();
    let (zi, zo) = (coax_z(&rho, a, s), coax_z(&rho, s, b));
    let z_disc = zi * zo / (zi + zo);
    let (ez, eb) = line_errors([&r1, &r2], &freqs, res.dt(), dz * 1e-3, z_disc);
    let (s11, s21) = through(&r1, &r2, 8e-3);
    println!(
        "strip line along z between coaxial grounds: discrete Z {z_disc:.3} Ω, |Z_L − Z| ≤ \
         {ez:.2e}, β vs Yee ≤ {eb:.2e}; |S11| ≤ {s11:.1} dB, |ΔS21| ≤ {s21:.2e}"
    );
    assert!(ez < 2e-4, "Z vs discrete {ez}");
    assert!(eb < 2e-4, "β vs Yee {eb}");
    assert!(s11 < -60.0, "|S11| = {s11} dB");
    assert!(s21 < 1e-3, "S21 error {s21}");
}

#[test]
fn cpw_along_z_on_a_cylinder_matches_the_discrete_tem_line() {
    // A small shield: the cut-off modes (from 16.5 GHz, the outer annulus
    // with the fin) are far above the band and have decayed at the probes,
    // 27 mm from the feeds.
    let (r_in, r_m, r_out, dr, dz) = (1.0, 1.3, 1.6, 0.1, 0.5);
    let n_alpha = 48;
    let da = TAU / n_alpha as f64;
    let rho = lines(r_in, r_out, dr);
    let alpha: Vec<f64> = (0..=n_alpha).map(|j| j as f64 * da).collect();
    let len = 130.0;
    let mut m = cylindrical(rho.clone(), alpha.clone(), lines(0.0, len, dz));
    // Centre strip 4.8 cells wide around α = π, gaps 2.2 cells; the
    // grounds fill the rest of the azimuth. The metal edges lie 0.4 cells
    // past the last line in the metal (see `CpwPort`'s mesh requirements).
    let (sw, gap) = (4.8 * da, 2.2 * da);
    let port = |n: u32, z0: f64, z1: f64| {
        CpwPort::new(
            n,
            [r_in, 0.0, z0],
            [r_m, TAU, z1],
            Axis::Z,
            Axis::X,
            sw,
            gap,
        )
    };
    let p1 = port(1, 6.0, 60.0).excite(1.0).add_to(&mut m).unwrap();
    let p2 = port(2, 124.0, 70.0).add_to(&mut m).unwrap();
    let (e1, e2) = (PI - 0.5 * sw, PI + 0.5 * sw);
    let (e0, e3) = (e1 - gap, e2 + gap);
    line_metal(
        &mut m,
        &[
            ([r_m, 0.0, 0.0], [r_m, e0, len]),
            ([r_m, e1, 0.0], [r_m, e2, len]),
            ([r_m, e3, 0.0], [r_m, TAU, len]),
            // A fin on the seam joins the grounds to both shields (one
            // TEM mode, strip against the rest).
            ([r_in, 0.0, 0.0], [r_out, 0.0, len]),
        ],
    );
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(5e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::ZMin, Boundary::pml(8))
        .boundary(Face::ZMax, Boundary::pml(8))
        .end_criteria(EndCriteria::energy_decay_db(-80.0))
        .max_timesteps(30_000);
    let res = run(&m, &fdtd);
    let freqs = linspace(2e9, 8e9, 7);
    let r1 = p1.analyze(&res, &freqs).unwrap();
    let r2 = p2.analyze(&res, &freqs).unwrap();
    // The discrete reference: conductors are the PEC ρ faces and the
    // nodes of the metal plane's α edges whose midpoints are in metal.
    let im = rho.iter().position(|&r| (r - r_m).abs() < 1e-9).unwrap();
    let edge_kind = |j: usize| {
        let mid = 0.5 * (alpha[j] + alpha[j + 1]);
        if (e1..=e2).contains(&mid) {
            Some(1.0)
        } else if mid <= e0 || mid >= e3 {
            Some(0.0)
        } else {
            None
        }
    };
    let rho_m: Vec<f64> = rho.iter().map(|r| r * 1e-3).collect();
    let c = capacitance(&rho_m, &alpha, true, true, |i, j| {
        if i == 0 || i + 1 == rho.len() || j == 0 {
            return Some(0.0);
        }
        if i != im {
            return None;
        }
        let prev = (j + n_alpha - 1) % n_alpha;
        edge_kind(j).or_else(|| edge_kind(prev))
    });
    let z_disc = 1.0 / (C0 * c);
    let (ez, eb) = line_errors([&r1, &r2], &freqs, res.dt(), dz * 1e-3, z_disc);
    let (s11, s21) = through(&r1, &r2, 10e-3);
    let (zq, _) = p1.quasi_static().unwrap();
    println!(
        "CPW along z on a cylinder: discrete Z {z_disc:.3} Ω (flat CPW in free space with \
         the arcs {zq:.1} Ω), |Z_L − Z| ≤ {ez:.2e}, β vs Yee ≤ {eb:.2e}; |S11| ≤ {s11:.1} \
         dB, |ΔS21| ≤ {s21:.2e}"
    );
    assert!(ez < 2e-4, "Z vs discrete {ez}");
    assert!(eb < 2e-4, "β vs Yee {eb}");
    assert!(s11 < -70.0, "|S11| = {s11} dB");
    assert!(s21 < 5e-4, "S21 error {s21}");
}

#[test]
fn stripline_along_alpha_at_a_large_radius_is_a_parallel_plate_line() {
    let (r0, w, b) = (200.0, 2.0, 2.0);
    let da = 0.5 / r0;
    let n_alpha = 140;
    let alpha: Vec<f64> = (0..=n_alpha).map(|j| j as f64 * da).collect();
    let mut m = cylindrical(
        lines(r0 - w / 2.0, r0 + w / 2.0, 0.25),
        alpha.clone(),
        lines(0.0, b, 0.5),
    );
    // Grounds: the PEC z faces; the strip on z = b/2 over the full width
    // between the PMC ρ walls.
    let port = |n: u32, f: f64, e: f64| {
        StriplinePort::new(
            n,
            [r0 - w / 2.0, f, 0.0],
            [r0 + w / 2.0, e, b],
            Axis::Y,
            Axis::Z,
        )
    };
    let p1 = port(1, alpha[20], alpha[50])
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let p2 = port(2, alpha[120], alpha[90]).add_to(&mut m).unwrap();
    line_metal(
        &mut m,
        &[(
            [r0 - w / 2.0, alpha[0], b / 2.0],
            [r0 + w / 2.0, alpha[n_alpha], b / 2.0],
        )],
    );
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(5e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::XMin, Boundary::Pmc)
        .boundary(Face::XMax, Boundary::Pmc)
        .boundary(Face::YMin, Boundary::pml(8))
        .boundary(Face::YMax, Boundary::pml(8))
        .end_criteria(EndCriteria::energy_decay_db(-80.0))
        .max_timesteps(30_000);
    let res = run(&m, &fdtd);
    let freqs = linspace(2e9, 8e9, 7);
    let r1 = p1.analyze(&res, &freqs).unwrap();
    let r2 = p2.analyze(&res, &freqs).unwrap();
    let z_pp = Z0 * b / (4.0 * w);
    let (ez, eb) = line_errors([&r1, &r2], &freqs, res.dt(), 0.5e-3, z_pp);
    let len = (alpha[90] - alpha[50]) * r0 * 1e-3;
    let (s11, s21) = through(&r1, &r2, len);
    println!(
        "strip line along α at R = {r0} mm: parallel plates {z_pp:.3} Ω, |Z_L − Z| ≤ \
         {ez:.2e}, β vs Yee ≤ {eb:.2e}; |S11| ≤ {s11:.1} dB, |ΔS21| ≤ {s21:.2e}"
    );
    assert!(ez < 1e-3, "Z vs parallel plates {ez}");
    assert!(eb < 1e-3, "β vs Yee {eb}");
    assert!(s11 < -50.0, "|S11| = {s11} dB");
    assert!(s21 < 3e-3, "S21 error {s21}");
}

#[test]
fn cpw_along_alpha_at_a_large_radius_is_the_flat_line() {
    let r0 = 200.0;
    let (dr, dzz) = (0.25, 0.25);
    // A 5 × 2.5 mm box: the lowest cut-off mode (30 GHz) has decayed at
    // the probes, 17 mm from the feeds.
    let (rlo, rhi, zlo, zm, zhi) = (r0 - 2.5, r0 + 2.5, 0.0, 1.25, 2.5);
    let da = 0.5 / r0;
    let n_alpha = 200;
    let alpha: Vec<f64> = (0..=n_alpha).map(|j| j as f64 * da).collect();
    let rho = lines(rlo, rhi, dr);
    let zl = lines(zlo, zhi, dzz);
    let mut m = cylindrical(rho.clone(), alpha.clone(), zl.clone());
    // Metal on z = zm: centre strip 1.2 mm around ρ = r0, gaps 0.8 mm
    // (edges 0.4 cells past the last line in the metal); grounds to the
    // PEC ρ walls; PEC z faces.
    let (sw, gap) = (1.2, 0.8);
    let port = |n: u32, f: f64, e: f64| {
        CpwPort::new(n, [rlo, f, zlo], [rhi, e, zm], Axis::Y, Axis::Z, sw, gap)
    };
    let p1 = port(1, alpha[12], alpha[80])
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    let p2 = port(2, alpha[188], alpha[120]).add_to(&mut m).unwrap();
    let (e1, e2) = (r0 - 0.5 * sw, r0 + 0.5 * sw);
    let (e0, e3) = (e1 - gap, e2 + gap);
    let (a50, a90) = (alpha[0], alpha[n_alpha]);
    line_metal(
        &mut m,
        &[
            ([rlo, a50, zm], [e0, a90, zm]),
            ([e1, a50, zm], [e2, a90, zm]),
            ([e3, a50, zm], [rhi, a90, zm]),
        ],
    );
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(5e9, 4e9))
        .boundaries(Boundaries::all(Boundary::Pec))
        .boundary(Face::YMin, Boundary::pml(8))
        .boundary(Face::YMax, Boundary::pml(8))
        .end_criteria(EndCriteria::energy_decay_db(-80.0))
        .max_timesteps(30_000);
    let res = run(&m, &fdtd);
    let freqs = linspace(2e9, 8e9, 7);
    let r1 = p1.analyze(&res, &freqs).unwrap();
    let r2 = p2.analyze(&res, &freqs).unwrap();
    // Flat discrete reference on the (ρ, z) cross-section.
    let km = zl.iter().position(|&z| (z - zm).abs() < 1e-9).unwrap();
    let edge_kind = |i: usize| {
        let mid = 0.5 * (rho[i] + rho[i + 1]);
        if (e1..=e2).contains(&mid) {
            Some(1.0)
        } else if mid <= e0 || mid >= e3 {
            Some(0.0)
        } else {
            None
        }
    };
    let rho_m: Vec<f64> = rho.iter().map(|r| r * 1e-3).collect();
    let z_m: Vec<f64> = zl.iter().map(|z| z * 1e-3).collect();
    let c = capacitance(&rho_m, &z_m, false, false, |i, k| {
        if i == 0 || i + 1 == rho.len() || k == 0 || k + 1 == zl.len() {
            return Some(0.0);
        }
        if k != km {
            return None;
        }
        edge_kind(i).or_else(|| i.checked_sub(1).and_then(edge_kind))
    });
    let z_flat = 1.0 / (C0 * c);
    let (ez, eb) = line_errors([&r1, &r2], &freqs, res.dt(), 0.5e-3, z_flat);
    let len = (alpha[120] - alpha[80]) * r0 * 1e-3;
    let (s11, s21) = through(&r1, &r2, len);
    println!(
        "CPW along α at R = {r0} mm: flat discrete Z {z_flat:.3} Ω, |Z_L − Z| ≤ {ez:.2e}, β \
         vs Yee ≤ {eb:.2e}; |S11| ≤ {s11:.1} dB, |ΔS21| ≤ {s21:.2e}"
    );
    assert!(ez < 1e-3, "Z vs flat {ez}");
    assert!(eb < 2e-4, "β vs Yee {eb}");
    assert!(s11 < -70.0, "|S11| = {s11} dB");
    assert!(s21 < 1e-4, "S21 error {s21}");
}

/// Lines along ρ and heights along α are not uniform: refused, by all
/// three line ports alike.
#[test]
fn cylindrical_lines_along_rho_are_refused() {
    let mut m = cylindrical(
        lines(1.0, 10.0, 0.5),
        (0..=8).map(|j| j as f64 * 0.1).collect(),
        lines(0.0, 2.0, 0.5),
    );
    let sub = m.add_property(Property::material("sub", Material::new(2.0, 0.0)));
    let _ = sub;
    for (dir, height) in [(Axis::X, Axis::Z), (Axis::Z, Axis::Y)] {
        let (a, b) = ([2.0, 0.1, 0.0], [8.0, 0.5, 1.0]);
        let unsupported = |r: ems::Result<()>| matches!(r, Err(ems::Error::Unsupported(_)));
        assert!(unsupported(
            MslPort::new(1, a, b, dir, height)
                .add_to(&mut m)
                .map(|_| ())
        ));
        assert!(unsupported(
            StriplinePort::new(1, a, b, dir, height)
                .add_to(&mut m)
                .map(|_| ())
        ));
        assert!(unsupported(
            CpwPort::new(1, a, b, dir, height, 0.1, 0.05)
                .add_to(&mut m)
                .map(|_| ())
        ));
    }
}
