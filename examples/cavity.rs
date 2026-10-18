//! Resonances of an empty rectangular PEC cavity versus the analytic
//! values.
//!
//! ```text
//! cargo run --release --example cavity
//! ```
//!
//! A 100 mm × 45 mm × 70 mm box (PEC walls = the domain boundaries) on a
//! uniform 2.5 mm mesh is excited by a short Gaussian pulse on a single
//! E_y edge and probed by a voltage probe on another E_y edge. The TE_m0p
//! modes (E along y only) have
//! `f = c0/2 · sqrt((m/a)² + (p/d)²)` (Pozar, *Microwave Engineering*,
//! §6.3); the peaks of the windowed probe spectrum are compared with them.
//! The small negative offsets are the numerical dispersion of the Yee
//! scheme (Taflove & Hagness, ch. 4), which shrinks quadratically with the
//! cell size.

use std::f64::consts::PI;

use csx::primitives::BoxShape;
use csx::properties::{Excitation, ProbeBox, ProbeType};
use csx::{ContinuousStructure, Primitive, Property};
use ems::prelude::*;

/// Cavity size [mm].
const A: f64 = 100.0;
const B: f64 = 45.0;
const D: f64 = 70.0;
/// Cell size [mm].
const CELL: f64 = 2.5;

fn mm_lines(len: f64) -> impl Iterator<Item = f64> {
    let n = (len / CELL).round() as usize;
    (0..=n).map(|i| i as f64 * CELL)
}

/// |DFT| of the Hann-windowed samples `x` (times `t`) at `f`.
fn spectrum(t: &[f64], x: &[f64], f: f64) -> f64 {
    let n = t.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (q, (&tt, &v)) in t.iter().zip(x).enumerate() {
        let w = 0.5 - 0.5 * (2.0 * PI * q as f64 / (n - 1.0)).cos();
        let (s, c) = (2.0 * PI * f * tt).sin_cos();
        re += w * v * c;
        im -= w * v * s;
    }
    re.hypot(im)
}

/// Frequency of the spectral maximum in `[lo, hi]` (scan + golden section).
fn peak(t: &[f64], x: &[f64], lo: f64, hi: f64) -> f64 {
    let s = |f| spectrum(t, x, f);
    let n = 80;
    let step = (hi - lo) / n as f64;
    let best = (0..=n)
        .map(|q| (q, s(lo + q as f64 * step)))
        .fold((0, f64::MIN), |a, b| if b.1 > a.1 { b } else { a })
        .0 as f64;
    let (mut a, mut b) = (lo + (best - 1.0) * step, lo + (best + 1.0) * step);
    let g = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..50 {
        let (c, d) = (b - g * (b - a), a + g * (b - a));
        if s(c) > s(d) {
            b = d;
        } else {
            a = c;
        }
    }
    0.5 * (a + b)
}

fn main() -> Result<()> {
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    model.grid.set_lines(0, mm_lines(A))?;
    model.grid.set_lines(1, mm_lines(B))?;
    model.grid.set_lines(2, mm_lines(D))?;

    // One E_y edge as source at (0.3a, 0.4d), one as probe at (0.7a, 0.6d)
    // (no TE_m0p mode up to m, p = 3 has a node at either).
    let src = model.add_property(Property::excitation(
        "source",
        Excitation::new([0.0, 1.0, 0.0]),
    ));
    model.add_primitive(
        src,
        Primitive::new(BoxShape::new([30.0, 20.0, 27.5], [30.0, 22.5, 27.5])),
    )?;
    let probe = model.add_property(Property::probe("probe", ProbeBox::new(ProbeType::Voltage)));
    model.add_primitive(
        probe,
        Primitive::new(BoxShape::new([70.0, 20.0, 42.5], [70.0, 22.5, 42.5])),
    )?;

    let wave = Waveform::gaussian(3e9, 2.5e9);
    let fdtd = Fdtd::new()
        .waveform(wave.clone())
        .max_timesteps(30_000)
        .no_end_criteria(); // lossless: the energy never decays
    let mut sim = Simulation::new(&model, &fdtd)?;
    for w in sim.warnings() {
        println!("warning: {w}");
    }
    let n = sim.grid().nodes();
    println!(
        "cavity {A} x {B} x {D} mm, {} x {} x {} cells of {CELL} mm, dt = {:.3e} s",
        n[0] - 1,
        n[1] - 1,
        n[2] - 1,
        sim.engine().dt()
    );
    let res = sim.run()?;
    let o = res.outcome().expect("outcome");
    println!("{} steps in {:.2} s", o.steps, o.elapsed.as_secs_f64());

    // Skip the excitation itself.
    let p = res.probe("probe").expect("probe");
    let t_end = wave.duration().unwrap_or(0.0);
    let first = p.times().partition_point(|&t| t < t_end);
    let (t, x) = (&p.times()[first..], &p.scalar().expect("voltage")[first..]);

    println!("mode      analytic [GHz]   FDTD [GHz]   error");
    let (a, d) = (A * 1e-3, D * 1e-3);
    for (m, q) in [(1, 1), (2, 1), (1, 2), (3, 1), (2, 2)] {
        let fa = C0 / 2.0 * ((m as f64 / a).powi(2) + (q as f64 / d).powi(2)).sqrt();
        let f = peak(t, x, 0.97 * fa, 1.03 * fa);
        println!(
            "TE{m}0{q}    {:>10.5}       {:>10.5}   {:+.3} %",
            fa / 1e9,
            f / 1e9,
            (f / fa - 1.0) * 100.0
        );
    }
    Ok(())
}
