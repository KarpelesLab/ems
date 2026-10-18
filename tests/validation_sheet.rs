//! Analytic validation (design §9): a zero-thickness PEC sheet across a
//! parallel-plate line blocks transmission (< −60 dB), while the same
//! sheet with an aperture lets a substantial part through (sanity check
//! of the measurement).
//!
//! Line: 8 × 8 mm cross-section of 1 mm cells, PEC plates normal to x,
//! PMC walls normal to y (TEM mode E_x exact), CPML at both z ends. A soft
//! source on every E_x edge of one z plane launches the TEM pulse; the
//! sheet is the set of tangential (E_x, E_y) edges of one z plane marked
//! PEC in the engine material map ([`CellMaterials::set_pec_edge`]).
//! Transmission is the probe signal (plate-to-plate voltage behind the
//! sheet) relative to the same probe without sheet, both as a peak ratio
//! and as the worst |T(f)| over 2–18 GHz.

mod common;

use common::*;
use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::{CellMaterials, EdgeSource, Engine, FaceCondition, FaceConditions, Grid, Probe};
use ems::{Axis, Cpml, Face};

const N: usize = 8;
const NZ: usize = 128;
const SRC: usize = 30;
const SHEET: usize = 60;
const PRB: usize = 90;

#[derive(Clone, Copy, PartialEq)]
enum Sheet {
    None,
    Full,
    /// Square aperture `x, y ∈ (lo, hi)` mm.
    Aperture(usize, usize),
}

/// Returns `(dt, probe series)`.
fn run(sheet: Sheet) -> (f64, Vec<f64>) {
    let g = Grid::uniform([N, N, NZ], [1e-3; 3]).unwrap();
    let faces = FaceConditions::all(FaceCondition::Pec)
        .with(Face::YMin, FaceCondition::Pmc)
        .with(Face::YMax, FaceCondition::Pmc);
    let mut m = CellMaterials::vacuum(&g);
    let open = |a: usize, b: usize| match sheet {
        Sheet::Aperture(lo, hi) => a >= lo && a < hi && b > lo && b < hi,
        _ => false,
    };
    if sheet != Sheet::None {
        // E_x edge (i..i+1, j) and E_y edge (i, j..j+1) of the plane.
        for i in 0..=N {
            for j in 0..=N {
                if i < N && !open(i, j) {
                    m.set_pec_edge(Axis::X, [i, j, SHEET]);
                }
                if j < N && !open(j, i) {
                    m.set_pec_edge(Axis::Y, [i, j, SHEET]);
                }
            }
        }
    }
    let mut e = Engine::builder(g)
        .materials(m)
        .faces(faces)
        .threads(4)
        .build()
        .unwrap();
    let p = Cpml::cells(8);
    e.add_operator(CpmlBoundary::new(&e, [(Face::ZMin, p), (Face::ZMax, p)]).unwrap());
    let mut s = EdgeSource::new(sine_pulse(10e9, 6e9));
    let mut probe = Probe::new(ems::engine::ProbeQuantity::Voltage);
    for i in 0..N {
        for j in 0..=N {
            s = s.edge(Axis::X, [i, j, SRC], 1.0);
        }
        probe = probe.entry(Axis::X, [i, N / 2, PRB], 1.0);
    }
    e.add_operator(s);
    let h = e.add_operator(probe);
    e.run(500);
    (e.dt(), e.operator(&h).values().to_vec())
}

/// (peak ratio, max |T(f)| over 2–18 GHz) of `sheet` vs no sheet.
fn transmission(sheet: Sheet, reference: &(f64, Vec<f64>)) -> (f64, f64) {
    let (dt, r) = reference;
    let (_, t) = run(sheet);
    let freqs = linspace(2e9, 18e9, 33);
    let a = spectrum(r, *dt, &freqs);
    let b = spectrum(&t, *dt, &freqs);
    let tf = a
        .iter()
        .zip(&b)
        .map(|(a, b)| b.norm() / a.norm())
        .fold(0.0, f64::max);
    (peak(&t) / peak(r), tf)
}

#[test]
fn pec_sheet_blocks_transmission() {
    let _t = Timer::start("pec_sheet_blocks_transmission");
    let reference = run(Sheet::None);
    // The record holds the whole transmitted pulse.
    let r = &reference.1;
    assert!(peak(&r[r.len() - 20..]) < 1e-6 * peak(r));
    let (pk, tf) = transmission(Sheet::Full, &reference);
    eprintln!(
        "full PEC sheet: peak ratio {pk:e} ({:.1} dB), max |T(f)| {tf:e} ({:.1} dB)",
        db(pk),
        db(tf)
    );
    assert!(db(pk) < -60.0 && db(tf) < -60.0, "{pk} {tf}");
    let (pk, tf) = transmission(Sheet::Aperture(2, 6), &reference);
    eprintln!(
        "PEC sheet with 4 × 4 mm aperture: peak ratio {pk:.3} ({:.1} dB), max |T(f)| {tf:.3} ({:.1} dB)",
        db(pk),
        db(tf)
    );
    // A 4 × 4 mm hole in an 8 × 8 mm cross-section (well below the
    // aperture resonance) passes ~ −20 dB, rising with frequency.
    assert!(db(pk) > -30.0, "aperture blocks too much: {pk}");
    assert!(pk < 0.9, "aperture has no effect: {pk}");
}
