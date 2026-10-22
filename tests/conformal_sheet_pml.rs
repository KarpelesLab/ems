//! Conformal thin PEC sheets inside absorbing layers (design §5.13.1,
//! "Absorbing layers"): two parallel plates (zero-thickness boxes, tilted
//! about the guide axis x or level just off the mesh planes) run into a
//! CPML on `x_max`. The pieces and regions of the cut cells in the layer
//! get the layer's stretch from the sheet operator (their kernel
//! coefficients are 0, so the CPML's own correction vanishes there).
//! Reflection against a domain 160 cells longer, from which nothing
//! returns within the gate; long-run stability.

mod common;

use csx::primitives::BoxShape;
use csx::{ContinuousStructure, Primitive, Property, Transform, Vec3};
use ems::compile::{CompileOptions, compile};
use ems::engine::{ConformalReport, EdgeSource, Engine, Probe};
use ems::{Axis, Boundary, Face, Fdtd};

const H: f64 = 1.0; // mm
/// Half the plate distance, mm.
const B: f64 = 2.1;

/// The plates' tilt about x (rad); 0: level at `z = 4.07 ± 2.1` (0.03 and
/// 0.17 of a cell off the mesh planes: slivers along the whole guide).
#[derive(Debug, Clone, Copy)]
struct Plates(f64);

const TILTED: Plates = Plates(0.35);
const LEVEL: Plates = Plates(0.0);

impl Plates {
    fn transform(self) -> Transform {
        let mut t = Transform::new();
        t.rotate_axis(0, self.0)
            .translate(Vec3::new(0.0, 0.13, 4.07));
        t
    }
}

fn lines(lo: f64, n: usize) -> Vec<f64> {
    (0..=n).map(|i| lo + i as f64 * H).collect()
}

/// The guide with `nx` cells along x (CPML(8) on `x_max` if `pml`), y in
/// `[−5, 5]`, z in `[0, 8]` mm (PEC around), a soft TEM-like source (E
/// normal to the plates) at x index 4 between the plates; the engine, the
/// report and the source nodes.
fn guide(
    plates: Plates,
    nx: usize,
    pml: bool,
    threads: usize,
) -> (Engine, Option<ConformalReport>, Vec<[usize; 3]>) {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    let (lx, ly, lz) = (lines(0.0, nx), lines(-5.0, 10), lines(0.0, 8));
    m.grid.set_lines(0, lx.iter().copied()).unwrap();
    m.grid.set_lines(1, ly.iter().copied()).unwrap();
    m.grid.set_lines(2, lz.iter().copied()).unwrap();
    let metal = m.add_property(Property::metal("plates"));
    let t = plates.transform();
    let len = lx[nx] + 2.0;
    for z in [-B, B] {
        m.add_primitive(
            metal,
            Primitive::new(BoxShape::new([-1.0, -12.0, z], [len, 12.0, z]))
                .with_transform(t.clone()),
        )
        .unwrap();
    }
    let mut fdtd = Fdtd::new().threads(threads);
    if pml {
        fdtd = fdtd.boundary(Face::XMax, Boundary::pml(8));
    }
    let opts = CompileOptions::default().with_conformal(true);
    let c = compile(&m, &fdtd, &opts).unwrap();
    let (mut e, h) = c.build().unwrap();
    // Nodes well between the plates.
    let between = |p: Vec3| t.apply_inverse(p)[2].abs() < B - 0.3 * H;
    let (s, c) = plates.0.sin_cos();
    let mut src = EdgeSource::new(common::sine_pulse(20e9, 6e9));
    let mut nodes = Vec::new();
    for j in 0..ly.len() - 1 {
        for k in 0..lz.len() - 1 {
            let p = Vec3::new(lx[4], ly[j], lz[k]);
            let q = Vec3::new(lx[4], ly[j + 1], lz[k + 1]);
            if between(p) && between(q) && ly[j].abs() < 3.0 {
                src = src.edge(Axis::Y, [4, j, k], -s).edge(Axis::Z, [4, j, k], c);
                nodes.push([0, j, k]);
            }
        }
    }
    e.add_operator(src);
    (e, h.conformal, nodes)
}

/// E_y and E_z at x index `probe` on three source nodes' (y, z), over
/// 1 ns, concatenated.
fn signals(
    plates: Plates,
    nx: usize,
    probe: usize,
    pml: bool,
    threads: usize,
) -> (Vec<f64>, ConformalReport) {
    let (mut e, rep, nodes) = guide(plates, nx, pml, threads);
    let pick = [nodes[0], nodes[nodes.len() / 2], nodes[nodes.len() - 1]];
    let probes: Vec<_> = pick
        .iter()
        .flat_map(|&[_, j, k]| [Axis::Y, Axis::Z].map(|a| (a, [probe, j, k])))
        .map(|(a, p)| e.add_operator(Probe::voltage(a, p)))
        .collect();
    e.run((1.0e-9 / e.dt()).ceil() as u64);
    let v = probes
        .iter()
        .flat_map(|p| e.operator(p).values().to_vec())
        .collect();
    (v, rep.unwrap())
}

/// Largest |difference| relative to the largest |reference| value, dB.
fn rel_db(test: &[f64], reference: &[f64]) -> f64 {
    let n = test.len().min(reference.len());
    let d = (0..n).fold(0.0f64, |m, q| m.max((test[q] - reference[q]).abs()));
    let r = reference[..n].iter().fold(0.0f64, |m, x| m.max(x.abs()));
    20.0 * (d / r).log10()
}

#[test]
fn sheet_plates_into_cpml_reflection() {
    // Layer from x = 32 mm (index 32), probe at 26 mm (index 26). (Plates
    // on the mesh planes, z = 4 ± 2, csx's PEC mask: −78.5 dB.)
    for plates in [TILTED, LEVEL] {
        let (reference, _) = signals(plates, 40 + 160, 26, false, 4);
        let (v, rep) = signals(plates, 40, 26, true, 4);
        let r = rel_db(&v, &reference);
        eprintln!("{plates:?} into CPML: reflection {r:.1} dB ({rep:?})");
        assert!(rep.verified && rep.sheet_interpolated > 0, "{rep:?}");
        assert!(r < -70.0, "{plates:?}: {r}");
    }
}

#[test]
fn sheet_plates_into_cpml_long_run() {
    // 36 000 steps; the energy decays to the static level the source
    // leaves (−92.5 dB with the plates on the mesh planes) and stays. With
    // the regions in the layer merged instead of interpolated, the level
    // plates grew from −74 dB after 4000 steps (~1 dB per 1000 steps).
    for plates in [TILTED, LEVEL] {
        let (mut e, rep, _) = guide(plates, 40, true, 4);
        assert!(rep.unwrap().verified);
        let mut w = Vec::new();
        for _ in 0..36 {
            e.run(1000);
            w.push(e.energy());
        }
        let peak = w.iter().fold(0f64, |m, &x| m.max(x));
        let db: Vec<f64> = w.iter().map(|x| 10.0 * (x / peak).log10()).collect();
        eprintln!("{plates:?}, 36 000 steps: {db:.1?}");
        let late = w[12..].iter().fold(0f64, |m, &x| m.max(x));
        assert!(late < 1e-8 * peak, "{db:?}");
        assert!(w[35] <= 1.01 * w[12], "{db:?}");
    }
}

#[test]
fn sheet_plates_into_cpml_threads_bit_identical() {
    let (a, _) = signals(TILTED, 40, 26, true, 1);
    let (b, _) = signals(TILTED, 40, 26, true, 3);
    assert_eq!(a, b);
}
