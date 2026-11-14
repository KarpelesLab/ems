//! Analytic validation (design §9) of the surface-impedance operator
//! ([`ems::engine::surface`]):
//!
//! * plane-wave reflection of a solid lossy block,
//!   `Γ = (Z_s − η0)/(Z_s + η0)`;
//! * the sheet cases of `validation_sheet_conducting.rs` (thin, thick-ish
//!   and thick sheet against the exact slab) with the rational sheet;
//! * the quality factor of a plate cavity with solid copper plates,
//!   `Q = h/δ` (and `h/(δ·K)` with roughness);
//! * the attenuation of a parallel-plate line with solid lossy plates,
//!   `α = R_s/(η0·h)`, running into the CPML;
//! * a lossy strip between lossy plates running into the CPML: the layer
//!   reflects it like a perfectly conducting one;
//! * a resistive strip between the plates of a capacitor: the discharge
//!   time `R·C` with the strip's own width (its border edges carry half a
//!   cell);
//! * passivity and thread-count independence in a 3-D cavity.

mod common;

use common::*;
use ems::Complex64;
use ems::constants::{C0, MU0, Z0};
use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::sheet::{SheetEdge, skin_depth, slab_abcd};
use ems::engine::surface::{
    Conductor, ImpedanceFit, Roughness, SolidEdge, SurfaceImpedance, box_edges,
    half_space_impedance,
};
use ems::engine::{
    EdgeMaterials, EdgeSource, Engine, FaceCondition, FaceConditions, FusedSteps, Grid, Operator,
    Probe,
};
use ems::{Axis, Cpml, Face};

const DZ: f64 = 0.25e-3;
const CELLS: usize = 480;
const SRC: usize = 40;
const REFL: usize = 140;
const SHEET: usize = 240;
const TRANS: usize = 400;
const STEPS: u64 = 2400;
const F_MAX: f64 = 15e9;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    None,
    /// A PEC block from `SHEET` on.
    Pec,
    /// The block with a surface impedance.
    Solid(f64),
    /// A PEC sheet at `SHEET`.
    PecSheet,
    /// A rational sheet `(σ, t)` at `SHEET`.
    Sheet(f64, f64),
}

/// Probe series at `REFL` and `TRANS`, and the time step.
fn run(kind: Kind) -> (f64, Vec<f64>, Vec<f64>) {
    let mut line = Line::new(DZ, CELLS);
    line.src = SRC;
    line.probes = vec![REFL, TRANS];
    match kind {
        Kind::Pec | Kind::Solid(_) => line.sheets = (SHEET..SHEET + 3).collect(),
        Kind::PecSheet => line.sheets = vec![SHEET],
        _ => {}
    }
    let (mut e, hs) = line.build(sine_pulse(7e9, 4e9));
    let mut s = SurfaceImpedance::new(Some(F_MAX));
    match kind {
        Kind::Solid(sigma) => {
            for k in SHEET..SHEET + 3 {
                for node in line.edges(k) {
                    s.push_solid(SolidEdge {
                        axis: Axis::X,
                        node,
                        conductor: Conductor::new(sigma),
                        outside: None,
                    });
                }
            }
            let h = s.add_to(&mut e).unwrap();
            let op = e.operator(&h);
            // The front face carries the impedance; the back face of the
            // block looks into the other half of the line.
            assert_eq!((op.solid_edges(), op.interior(), op.thin()), (4, 2, 0));
        }
        Kind::Sheet(sigma, t) => {
            for node in line.edges(SHEET) {
                s.push_sheet(
                    SheetEdge {
                        axis: Axis::X,
                        node,
                        normal: Axis::Z,
                        conductivity: sigma,
                        thickness: t,
                        upper_share: None,
                    },
                    Roughness::Smooth,
                );
            }
            let h = s.add_to(&mut e).unwrap();
            assert_eq!(e.operator(&h).sheet_edges(), 2);
        }
        _ => {}
    }
    e.run(STEPS);
    let v: Vec<Vec<f64>> = hs.iter().map(|h| e.operator(h).values().to_vec()).collect();
    (e.dt(), v[0].clone(), v[1].clone())
}

fn freqs() -> Vec<f64> {
    linspace(2e9, 12e9, 21)
}

/// `(T, Γ)` of the exact slab.
fn exact(sigma: f64, t: f64, f: f64) -> (Complex64, Complex64) {
    let [a, b, c, d] = slab_abcd(sigma, t, f);
    let den = a + b / Z0 + c * Z0 + d;
    (
        Complex64::new(2.0, 0.0) / den,
        (a + b / Z0 - c * Z0 - d) / den,
    )
}

#[test]
fn solid_block_reflection() {
    let _t = Timer::start("solid_block_reflection");
    let incident = run(Kind::None).1;
    let (dt, pec, _) = run(Kind::Pec);
    // σ = 1e4 S/m: R_s = 0.89–2.2 Ω, 1 − |Γ| ≈ 2·R_s/η0 = 0.5–1.2 %.
    let sigma = 1e4;
    let (_, solid, behind) = run(Kind::Solid(sigma));
    assert!(peak(&behind) < 1e-12, "field behind the block");
    let f = freqs();
    let rp = spectrum(&diff(&pec, &incident), dt, &f);
    let rs = spectrum(&diff(&solid, &incident), dt, &f);
    let mut worst: f64 = 0.0;
    for ((f, a), b) in f.iter().zip(&rp).zip(&rs) {
        let g = -*b / *a;
        let zs = half_space_impedance(&Conductor::new(sigma), *f);
        let ge = (zs - Z0) / (zs + Z0);
        let (loss, loss_e) = (1.0 - g.norm(), 1.0 - ge.norm());
        worst = worst.max((loss / loss_e - 1.0).abs());
        eprintln!(
            "{:5.1} GHz: 1 − |Γ| exact {loss_e:.5}, measured {loss:.5} ({:+.2} %)",
            f / 1e9,
            100.0 * (loss / loss_e - 1.0)
        );
        assert!((g / ge - 1.0).norm() < 1e-3, "{f}: {g} vs {ge}");
    }
    eprintln!("solid block: max relative error of 1 − |Γ| {worst:.4}");
    assert!(worst < 0.02, "{worst}");
}

#[test]
fn rational_sheet_transmission() {
    let _t = Timer::start("rational_sheet_transmission");
    let (dt, _, reference) = run(Kind::None);
    let f = freqs();
    let a = spectrum(&reference, dt, &f);
    // Thin: η0·σ·t = 3.77, T = 2/(2 + η0σt). Thick-ish: t/δ = 1.8…4.4.
    for (sigma, t, tol_db) in [(1e4, 1e-6, 2e-3), (1e6, 20e-6, 0.05)] {
        let (_, _, tr) = run(Kind::Sheet(sigma, t));
        let b = spectrum(&tr, dt, &f);
        let (mut worst, mut worst_ph) = (0.0f64, 0.0f64);
        for ((f, a), b) in f.iter().zip(&a).zip(&b) {
            let (m, (te, _)) = (*b / *a, exact(sigma, t, *f));
            worst = worst.max((db(m.norm()) - db(te.norm())).abs());
            worst_ph = worst_ph.max((m / te).arg().abs().to_degrees());
        }
        eprintln!("sheet σ = {sigma:e}, t = {t:e}: |T| within {worst:.4} dB, {worst_ph:.3}°");
        assert!(
            worst < tol_db && worst_ph < 0.5,
            "{sigma} {t}: {worst} {worst_ph}"
        );
    }
}

#[test]
fn rational_thick_sheet_reflection() {
    let _t = Timer::start("rational_thick_sheet_reflection");
    let incident = run(Kind::None).1;
    let (dt, pec, _) = run(Kind::PecSheet);
    let (sigma, t) = (1e4, 0.5e-3);
    let (_, sheet, _) = run(Kind::Sheet(sigma, t));
    let f = freqs();
    let rp = spectrum(&diff(&pec, &incident), dt, &f);
    let rs = spectrum(&diff(&sheet, &incident), dt, &f);
    let mut worst: f64 = 0.0;
    for ((f, a), b) in f.iter().zip(&rp).zip(&rs) {
        let g = -*b / *a;
        let (_, ge) = exact(sigma, t, *f);
        worst = worst.max(((1.0 - g.norm()) / (1.0 - ge.norm()) - 1.0).abs());
        assert!((g / ge - 1.0).norm() < 1e-3, "{f}: {g} vs {ge}");
    }
    eprintln!("thick rational sheet: max relative error of 1 − |Γ| {worst:.4}");
    assert!(worst < 0.02, "{worst}");
}

/// `(f, Q)` of the ring-down `v` sampled at `dt`: the spectral peak in
/// `lo..hi` and the decay of the energy per period.
fn ring_down(v: &[f64], dt: f64, mut lo: f64, mut hi: f64) -> (f64, f64) {
    let spec = |f: f64| spectrum(v, dt, &[f])[0].norm();
    for _ in 0..60 {
        let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if spec(a) < spec(b) {
            lo = a;
        } else {
            hi = b;
        }
    }
    let f = 0.5 * (lo + hi);
    let per = 1.0 / (f * dt);
    let n = (v.len() as f64 / per).floor() as usize;
    let pts: Vec<(f64, f64)> = (0..n)
        .map(|p| {
            let (a, b) = ((p as f64 * per) as usize, ((p + 1) as f64 * per) as usize);
            let e: f64 = v[a..b].iter().map(|x| x * x).sum::<f64>() / (b - a) as f64;
            ((p as f64 + 0.5) * per * dt, e.ln())
        })
        .collect();
    let m = pts.len() as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (mx, my) = (sx / m, sy / m);
    let slope = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>()
        / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
    (f, -core::f64::consts::TAU * f / slope)
}

/// TM110 of a 40 × 24 mm box between two solid plates `h = 0.5 mm` apart
/// (PEC side walls, where the mode's only E component vanishes): `(f, Q)`.
fn plate_cavity(c: Conductor) -> (f64, f64) {
    let g = Grid::new([
        linspace(0.0, 40e-3, 21),
        linspace(0.0, 24e-3, 13),
        linspace(0.0, 1e-3, 5),
    ])
    .unwrap();
    let mut m = EdgeMaterials::vacuum(&g);
    let plates = [([0, 0, 0], [20, 12, 1]), ([0, 0, 3], [20, 12, 4])];
    for (a, b) in plates {
        for (axis, node) in box_edges(a, b) {
            m.set_pec_edge(axis, node);
        }
    }
    let mut e = Engine::builder(g)
        .edge_materials(m)
        .threads(1)
        .build()
        .unwrap();
    let mut s = SurfaceImpedance::new(Some(10e9));
    for (a, b) in plates {
        s.add_box(a, b, c);
    }
    let h = s.add_to(&mut e).unwrap();
    // The inner faces of the two plates, less the edges on the walls.
    assert_eq!(e.operator(&h).solid_edges(), 2 * (20 * 11 + 19 * 12));
    assert_eq!(e.operator(&h).thin(), 0);
    let f0 = 0.5 * C0 * (1.0f64 / 40e-3).hypot(1.0 / 24e-3);
    let mut src = EdgeSource::new(sine_pulse(f0, 1e9));
    for k in 1..3 {
        src = src.edge(Axis::Z, [10, 6, k], 1.0);
    }
    e.add_operator(src);
    let p = e.add_operator(Probe::voltage(Axis::Z, [7, 4, 1]));
    e.run(12_000);
    let dt = e.dt();
    let v = e.operator(&p).values();
    let t0 = (3e-9 / dt) as usize;
    ring_down(&v[t0..], dt, 0.9 * f0, 1.1 * f0)
}

#[test]
fn plate_cavity_q_smooth_and_rough() {
    let _t = Timer::start("plate_cavity_q_smooth_and_rough");
    let h = 0.5e-3;
    for (r, tol) in [
        (Roughness::Smooth, 0.01),
        (Roughness::HammerstadJensen { rms: 0.5e-6 }, 0.02),
        (
            Roughness::Huray {
                radius: 0.5e-6,
                ratio: 2.0,
            },
            0.03,
        ),
    ] {
        let c = Conductor::new(5.8e7).with_roughness(r);
        let (f, q) = plate_cavity(c);
        let delta = skin_depth(5.8e7, f);
        let k = r.factor(delta);
        // Q = ω·(μ0·h + 2·X/ω)/(2·R): h/(δ·K) but for the internal
        // inductance (a few 1e-3; the fit's own with roughness).
        let want = h / (delta * k);
        let fit = ImpedanceFit::half_space(&c, 1e8, 1e10).unwrap().eval(f);
        let want_fit = (core::f64::consts::TAU * f * MU0 * h + 2.0 * fit.im) / (2.0 * fit.re);
        eprintln!(
            "{r:?}: f {:.4} GHz, K {k:.3}, Q {q:.1}, h/(δK) {want:.1} ({:+.2} %), \
             with the fit's reactance {want_fit:.1} ({:+.2} %)",
            f / 1e9,
            100.0 * (q / want - 1.0),
            100.0 * (q / want_fit - 1.0)
        );
        assert!((q / want - 1.0).abs() < tol, "{r:?}: Q {q} vs {want}");
        assert!(
            (q / want_fit - 1.0).abs() < 0.01,
            "{r:?}: Q {q} vs {want_fit}"
        );
    }
}

/// A line along y between two solid plates (normal z, `h` apart, one cell
/// thick), PMC side walls or a strip of `strip` cells between PEC walls,
/// CPML on both y ends: the E_z series at two nodes, the time step and
/// the number of stretched terms.
struct PlateLine {
    /// Cells across x; the strip (if any) covers `strip` in the middle.
    nx: usize,
    strip: Option<core::ops::Range<usize>>,
    lossy: Option<Conductor>,
    /// Install the operator before the CPML (no stretch).
    before_cpml: bool,
    /// Bandwidth of the 6 GHz pulse \[Hz\].
    bw: f64,
    /// Cells along y.
    ny: usize,
    /// A sheet `(σ, t)` on the plane `x = nx/2` between the plates, over
    /// the whole length.
    wall: Option<(f64, f64)>,
    /// The strip is a sheet `(σ, t)` on a mesh plane, not a solid.
    sheet: Option<(f64, f64)>,
    threads: usize,
}

const PL_D: f64 = 0.5e-3;

impl PlateLine {
    fn run(&self, steps: u64) -> (f64, Vec<f64>, Vec<f64>, usize) {
        // z: plate, 4 cells (2 mm), plate. With a strip: it is one cell
        // thick in the middle (cells 2..3 of 0..5 between the plates).
        let nz = if self.strip.is_some() {
            7
        } else {
            std::env::var("NZ").map_or(6, |v| v.parse().unwrap())
        };
        let g = Grid::uniform([self.nx, self.ny, nz], [PL_D; 3]).unwrap();
        let mut m = EdgeMaterials::vacuum(&g);
        let mut boxes = vec![
            ([0, 0, 0], [self.nx, self.ny, 1]),
            ([0, 0, nz - 1], [self.nx, self.ny, nz]),
        ];
        // The edges of a sheet strip (PEC if the line is lossless).
        let mut sheet_edges = Vec::new();
        match (&self.strip, self.sheet) {
            (Some(s), None) => boxes.push(([s.start, 0, 3], [s.end, self.ny, 4])),
            (Some(s), Some(_)) => {
                sheet_edges = box_edges([s.start, 0, 3], [s.end, self.ny, 3]);
                if self.lossy.is_none() {
                    for &(axis, node) in &sheet_edges {
                        m.set_pec_edge(axis, node);
                    }
                }
            }
            _ => {}
        }
        for &(a, b) in &boxes {
            for (axis, node) in box_edges(a, b) {
                m.set_pec_edge(axis, node);
            }
        }
        let side = if self.strip.is_some() {
            FaceCondition::Pec
        } else {
            FaceCondition::Pmc
        };
        let faces = FaceConditions::all(FaceCondition::Pec)
            .with(Face::XMin, side)
            .with(Face::XMax, side);
        let mut e = Engine::builder(g)
            .edge_materials(m)
            .faces(faces)
            .threads(self.threads)
            .build()
            .unwrap();
        let surface = |e: &mut Engine| {
            if let Some((sigma, t)) = self.wall {
                let mut s = SurfaceImpedance::new(Some(12e9));
                for (axis, node) in box_edges([self.nx / 2, 0, 1], [self.nx / 2, self.ny, nz - 1]) {
                    s.push_sheet(
                        SheetEdge {
                            axis,
                            node,
                            normal: Axis::X,
                            conductivity: sigma,
                            thickness: t,
                            upper_share: None,
                        },
                        Roughness::Smooth,
                    );
                }
                let h = s.add_to(e).unwrap();
                assert!(e.operator(&h).sheet_edges() > 0);
                return e.operator(&h).stretched();
            }
            let Some(c) = self.lossy else { return 0 };
            let mut s = SurfaceImpedance::new(Some(12e9));
            for &(a, b) in &boxes {
                s.add_box(a, b, c);
            }
            if let Some((sigma, t)) = self.sheet {
                for &(axis, node) in &sheet_edges {
                    s.push_sheet(
                        SheetEdge {
                            axis,
                            node,
                            normal: Axis::Z,
                            conductivity: sigma,
                            thickness: t,
                            upper_share: None,
                        },
                        Roughness::Smooth,
                    );
                }
            }
            let h = s.add_to(e).unwrap();
            assert_eq!(e.operator(&h).thin(), 0);
            e.operator(&h).stretched()
        };
        let mut stretched = 0;
        if self.before_cpml {
            // (In the serial hooks, after the layer's own update of these
            // edges: run by the slabs, the operator would come before the
            // layer, which would then stretch them as ordinary edges.)
            e.add_operator(Idle);
            stretched = surface(&mut e);
        }
        let layers = [Face::YMin, Face::YMax].map(|f| (f, Cpml::cells(8)));
        let cpml = CpmlBoundary::new(&e, layers).unwrap();
        e.add_operator(cpml);
        if !self.before_cpml {
            stretched = surface(&mut e);
        }
        // Source and probes between the lower plate and the strip (or the
        // upper plate), over the whole width.
        let mut src = EdgeSource::new(sine_pulse(6e9, self.bw));
        let cols: Vec<usize> = match &self.strip {
            Some(s) => (s.start..=s.end).collect(),
            None => (0..=self.nx).collect(),
        };
        let top = if self.strip.is_some() { 3 } else { nz - 1 };
        for &i in &cols {
            for k in 1..top {
                src = src.edge(Axis::Z, [i, 20, k], 1.0);
            }
        }
        e.add_operator(src);
        let mid = cols[cols.len() / 2];
        let p1 = e.add_operator(Probe::voltage(Axis::Z, [mid, 40, 1]));
        let p2 = e.add_operator(Probe::voltage(Axis::Z, [mid, 120, 1]));
        e.run(steps);
        (
            e.dt(),
            e.operator(&p1).values().to_vec(),
            e.operator(&p2).values().to_vec(),
            stretched,
        )
    }
}

#[test]
fn parallel_plate_attenuation() {
    let _t = Timer::start("parallel_plate_attenuation");
    let sigma = 1e6;
    let line = |lossy: Option<Conductor>| PlateLine {
        nx: 1,
        strip: None,
        lossy,
        before_cpml: false,
        bw: 4e9,
        ny: 140,
        wall: None,
        sheet: None,
        threads: 1,
    };
    let (dt, a1, a2, _) = line(None).run(1500);
    let (_, b1, b2, _) = line(Some(Conductor::new(sigma))).run(1500);
    let f = linspace(3e9, 9e9, 13);
    let (h, len) = (
        (std::env::var("NZ").map_or(6.0, |v| v.parse().unwrap()) - 2.0) * PL_D,
        80.0 * PL_D,
    );
    let ratio = |x: &[f64], y: &[f64]| -> Vec<f64> {
        let (x, y) = (spectrum(x, dt, &f), spectrum(y, dt, &f));
        x.iter().zip(&y).map(|(x, y)| (*y / *x).norm()).collect()
    };
    let (pec, lossy) = (ratio(&a1, &a2), ratio(&b1, &b2));
    let mut worst: f64 = 0.0;
    for ((f, p), l) in f.iter().zip(&pec).zip(&lossy) {
        let rs = 1.0 / (sigma * skin_depth(sigma, *f));
        let want = rs / (Z0 * h) * len;
        let got = -(l / p).ln();
        worst = worst.max((got / want - 1.0).abs());
        eprintln!(
            "{:4.1} GHz: α·L {got:.5} Np, R_s/(η0·h)·L {want:.5} ({:+.2} %)",
            f / 1e9,
            100.0 * (got / want - 1.0)
        );
    }
    assert!(worst < 0.02, "{worst}");
}

#[test]
fn lossy_strips_run_into_the_cpml() {
    let _t = Timer::start("lossy_strips_run_into_the_cpml");
    // The echo of the far layer at the first probe: the signal less the
    // one of a line twice as long, whose echo arrives after 0.8 ns.
    let cu = Conductor::new(5.8e7);
    // A solid strip, then a sheet strip (35 µm, then 2 µm: thinner than
    // the skin depth below 1 GHz).
    for sheet in [None, Some((5.8e7, 35e-6)), Some((5.8e7, 2e-6))] {
        let echo = |lossy: Option<Conductor>, before_cpml: bool| {
            let run = |ny: usize| {
                PlateLine {
                    nx: 14,
                    strip: Some(5..9),
                    lossy,
                    before_cpml,
                    bw: 8e9,
                    ny,
                    wall: None,
                    sheet,
                    threads: 2,
                }
                .run(780)
            };
            let (short, long) = (run(140), run(280));
            (db(peak(&diff(&short.1, &long.1)) / peak(&long.1)), short.3)
        };
        let (rp, _) = echo(None, false);
        let (rl, stretched) = echo(Some(cu), false);
        let (ru, none) = echo(Some(cu), true);
        eprintln!(
            "{sheet:?}: echo of the layer: PEC {rp:.1} dB, lossy {rl:.1} dB \
             ({stretched} stretched terms), lossy without the stretch {ru:.1} dB"
        );
        assert_eq!(none, 0);
        // A straight solid conductor has no derivative along the layer
        // normal in its surface loops; a sheet has (its crossing dual
        // edges).
        assert_eq!(stretched > 0, sheet.is_some());
        assert!(rl < rp + 2.0, "lossy {rl} dB against PEC {rp} dB");
    }
}

/// A sheet without effect (`σ·t = 1e-18 S`) along the line, in the plane
/// of E and of the direction of propagation, running through both layers:
/// its edges are updated by the operator, with the layer's stretch on the
/// circulations along the layer normal. The line behaves as without the
/// sheet; without the stretch (operator installed before the CPML) the
/// layers reflect.
#[test]
fn sheet_edges_are_stretched_like_the_kernel() {
    let _t = Timer::start("sheet_edges_are_stretched_like_the_kernel");
    let line = |wall: Option<(f64, f64)>, before_cpml: bool| PlateLine {
        nx: 2,
        strip: None,
        lossy: None,
        before_cpml,
        bw: 4e9,
        ny: 140,
        wall,
        sheet: None,
        threads: 1,
    };
    let none = line(None, false).run(1500);
    let wall = Some((1e-12, 1e-6));
    let with = line(wall, false).run(1500);
    let plain = line(wall, true).run(1500);
    assert!(with.3 > 0 && plain.3 == 0);
    let err = |x: &(f64, Vec<f64>, Vec<f64>, usize)| {
        (peak(&diff(&x.1, &none.1)) / peak(&none.1)).max(peak(&diff(&x.2, &none.2)) / peak(&none.2))
    };
    let (e_with, e_plain) = (err(&with), err(&plain));
    eprintln!(
        "transparent sheet through the CPML: {:.1} dB from the line without it \
         ({} stretched terms), {:.1} dB without the stretch",
        db(e_with),
        with.3,
        db(e_plain)
    );
    assert!(e_with < 1e-4, "{e_with}");
    assert!(e_plain > 1e-2, "{e_plain}");
}

/// A plate capacitor (10 × 10 mm, 2 mm apart, magnetic side walls) with a
/// resistive strip between its plates, two cells (2 mm) wide: it discharges
/// with `τ = R·C`, `R = h/(σt·w)`. The strip's three lines of edges carry
/// ½ + 1 + ½ cells of sheet; with a whole dual cell on each (the ladder's
/// width at a border) `τ` would be a third shorter.
#[test]
fn resistive_strip_discharges_a_plate_capacitor() {
    let _t = Timer::start("resistive_strip_discharges_a_plate_capacitor");
    let g = Grid::uniform([10, 10, 2], [1e-3; 3]).unwrap();
    let mut faces = FaceConditions::all(FaceCondition::Pmc);
    faces[Face::ZMin] = FaceCondition::Pec;
    faces[Face::ZMax] = FaceCondition::Pec;
    let mut e = Engine::builder(g).faces(faces).threads(1).build().unwrap();
    let (sigma, t) = (10.0, 5e-6);
    let mut s = SurfaceImpedance::new(Some(1e9));
    for (axis, node) in box_edges([5, 4, 0], [5, 6, 2]) {
        s.push_sheet(
            SheetEdge {
                axis,
                node,
                normal: Axis::X,
                conductivity: sigma,
                thickness: t,
                upper_share: None,
            },
            Roughness::Smooth,
        );
    }
    let h = s.add_to(&mut e).unwrap();
    // Six z edges and the two y edges between the plates (those on the
    // plates are PEC).
    assert_eq!(
        (e.operator(&h).sheet_edges(), e.operator(&h).skipped()),
        (8, 4)
    );
    {
        let mut f = e.fields_mut();
        for i in 0..=10 {
            for j in 0..=10 {
                for k in 0..2 {
                    f.u.set(Axis::Z, [i, j, k], 0.5);
                }
            }
        }
    }
    let p = e.add_operator(Probe::voltage(Axis::Z, [1, 8, 0]));
    e.run(6000);
    let dt = e.dt();
    let v = e.operator(&p).values();
    // Least-squares slope of ln v over 2…11 ns.
    let pts: Vec<(f64, f64)> = v
        .iter()
        .enumerate()
        .map(|(n, x)| (n as f64 * dt, *x))
        .filter(|(t, x)| *t > 2e-9 && *x > 0.0)
        .map(|(t, x)| (t, x.ln()))
        .collect();
    assert!(pts.len() > 4000, "{}", pts.len());
    let m = pts.len() as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (mx, my) = (sx / m, sy / m);
    let slope = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>()
        / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
    let tau = -1.0 / slope;
    let (r, c) = (
        2e-3 / (sigma * t * 2e-3),
        ems::constants::EPS0 * 1e-4 / 2e-3,
    );
    eprintln!(
        "τ = {:.4} ns, R·C = {:.4} ns ({:+.2} %)",
        tau * 1e9,
        r * c * 1e9,
        100.0 * (tau / (r * c) - 1.0)
    );
    assert!((tau / (r * c) - 1.0).abs() < 0.02, "{tau} vs {}", r * c);
}

/// A solid block and a rational sheet in a closed cavity on a graded mesh:
/// the energy only decays and the fields do not depend on the thread count.
#[test]
fn passive_and_deterministic() {
    let _t = Timer::start("passive_and_deterministic");
    let run = |threads: usize| {
        let g = Grid::new([
            linspace(0.0, 12e-3, 13),
            (0..=12).map(|i| 1e-3 * (i as f64).powf(1.1)).collect(),
            linspace(0.0, 10e-3, 11),
        ])
        .unwrap();
        let mut m = EdgeMaterials::vacuum(&g);
        let block = ([3, 3, 2], [7, 8, 4]);
        for (axis, node) in box_edges(block.0, block.1) {
            m.set_pec_edge(axis, node);
        }
        let mut e = Engine::builder(g)
            .edge_materials(m)
            .threads(threads)
            .build()
            .unwrap();
        let mut s = SurfaceImpedance::new(Some(20e9));
        s.add_box(
            block.0,
            block.1,
            Conductor::new(5.8e7).with_roughness(Roughness::HammerstadJensen { rms: 1e-6 }),
        );
        for (d, o) in [(Axis::X, Axis::Y), (Axis::Y, Axis::X)] {
            for a in 2..9 {
                for b in 3..=9 {
                    let mut node = [0, 0, 7];
                    node[d.index()] = a;
                    node[o.index()] = b;
                    s.push_sheet(
                        SheetEdge {
                            axis: d,
                            node,
                            normal: Axis::Z,
                            conductivity: 1e6,
                            thickness: 20e-6,
                            upper_share: None,
                        },
                        Roughness::Smooth,
                    );
                }
            }
        }
        let h = s.add_to(&mut e).unwrap();
        {
            let mut f = e.fields_mut();
            f.u.set(Axis::Y, [3, 4, 5], 1.0);
            f.u.set(Axis::X, [5, 6, 8], -0.7);
            f.u.set(Axis::Z, [9, 5, 3], 0.4);
        }
        let mut w = vec![e.energy()];
        for _ in 0..40 {
            e.run(50);
            w.push(e.energy());
        }
        let op = e.operator(&h);
        let counts = (op.solid_edges(), op.sheet_edges(), op.interior(), op.thin());
        (w, e.u_component(Axis::X), e.i_component(Axis::Y), counts)
    };
    let (w, ux, iy, counts) = run(1);
    eprintln!(
        "{counts:?}: energy {:.3e} → {:.3e}",
        w[0],
        w.last().unwrap()
    );
    assert_eq!(counts.1, 2 * 7 * 7);
    assert_eq!(counts.3, 0);
    for p in w.windows(2) {
        assert!(p[1] <= p[0] * (1.0 + 1e-6) + 1e-12 * w[0], "{w:?}");
    }
    assert!(w.last().unwrap() < &(0.999 * w[0]), "no loss");
    let (w3, ux3, iy3, _) = run(3);
    assert_eq!(w, w3);
    assert_eq!(ux, ux3);
    assert_eq!(iy, iy3);
}

/// An operator that does nothing, with every serial hook: registered
/// before the surface impedance it keeps that in the serial hooks.
struct Idle;

impl Operator for Idle {}

/// The edges run by the engine's threads, each on its slab, give the
/// values of the serial hooks bit for bit: for any thread count, in fused
/// steps too. A solid block and a rational sheet on a graded mesh, the
/// sheet running into a CPML on four faces (stretched circulations and
/// stretched H corrections), slab boundaries through both.
#[test]
fn slabs_equal_the_serial_hooks() {
    let _t = Timer::start("slabs_equal_the_serial_hooks");
    let run = |threads: usize, serial: bool, fused: FusedSteps| {
        let g = Grid::new([
            linspace(0.0, 14e-3, 15),
            (0..=14).map(|i| 1e-3 * (i as f64).powf(1.1)).collect(),
            linspace(0.0, 10e-3, 11),
        ])
        .unwrap();
        let mut m = EdgeMaterials::vacuum(&g);
        let block = ([4, 4, 2], [9, 9, 4]);
        for (axis, node) in box_edges(block.0, block.1) {
            m.set_pec_edge(axis, node);
        }
        let mut e = Engine::builder(g)
            .edge_materials(m)
            .threads(threads)
            .build()
            .unwrap();
        e.set_fused_steps(fused);
        let layers = [Face::XMin, Face::XMax, Face::YMin, Face::YMax].map(|f| (f, Cpml::cells(3)));
        let cpml = CpmlBoundary::new(&e, layers).unwrap();
        e.add_operator(cpml);
        if serial {
            e.add_operator(Idle);
        }
        let mut s = SurfaceImpedance::new(Some(20e9));
        s.add_box(
            block.0,
            block.1,
            Conductor::new(5.8e7).with_roughness(Roughness::HammerstadJensen { rms: 1e-6 }),
        );
        for (d, o) in [(Axis::X, Axis::Y), (Axis::Y, Axis::X)] {
            for a in 0..14 {
                for b in 1..14 {
                    let mut node = [0, 0, 7];
                    node[d.index()] = a;
                    node[o.index()] = b;
                    s.push_sheet(
                        SheetEdge {
                            axis: d,
                            node,
                            normal: Axis::Z,
                            conductivity: 1e6,
                            thickness: 20e-6,
                            upper_share: None,
                        },
                        Roughness::Smooth,
                    );
                }
            }
        }
        let h = s.add_to(&mut e).unwrap();
        assert_eq!(e.operator(&h).runs_in_slabs(), !serial);
        assert!(e.operator(&h).stretched() > 100);
        {
            let mut f = e.fields_mut();
            f.u.set(Axis::Y, [3, 4, 5], 1.0);
            f.u.set(Axis::X, [5, 6, 8], -0.7);
            f.u.set(Axis::Z, [10, 5, 3], 0.4);
            // On a sheet edge: both faces take it.
            f.u.set(Axis::X, [6, 7, 7], 0.3);
        }
        let mut w = vec![e.energy().to_bits()];
        for _ in 0..8 {
            e.run(40);
            w.push(e.energy().to_bits());
            // What a source adds to a sheet edge between steps.
            let mut f = e.fields_mut();
            let v = f.u.get(Axis::Y, [7, 5, 7]);
            f.u.set(Axis::Y, [7, 5, 7], v + 0.01);
        }
        let fields: Vec<u32> = Axis::ALL
            .iter()
            .flat_map(|&a| {
                let mut v = e.u_component(a);
                v.extend(e.i_component(a));
                v
            })
            .map(f32::to_bits)
            .collect();
        let lower = e
            .operator(&h)
            .lower_voltage(Axis::X, [7, 7, 7])
            .unwrap()
            .to_bits();
        (w, fields, lower)
    };
    let want = run(1, true, FusedSteps::Auto);
    assert!(want.1.iter().any(|&v| f32::from_bits(v) != 0.0));
    assert_eq!(
        run(3, true, FusedSteps::Auto),
        want,
        "serial hooks, 3 threads"
    );
    for threads in [1, 2, 3, 5, 7] {
        for fused in [FusedSteps::Never, FusedSteps::Always] {
            assert_eq!(
                run(threads, false, fused),
                want,
                "{threads} threads, {fused:?}"
            );
        }
    }
}
