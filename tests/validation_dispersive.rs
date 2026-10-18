//! Analytic validation of dispersive media (ADE operator,
//! `ems::engine::dispersive`): Drude half-space and Debye slab reflection,
//! Lorentz absorption line in transmission, a magnetically dispersive
//! matched slab, long-run energy conservation, thread determinism and the
//! csx-model path.
//!
//! Setup as in `validation_dielectric.rs`: the quasi-1-D TEM line of
//! `common::Line`, a probe between source and structure (reflection) or
//! behind it (transmission), against an all-vacuum reference run.
//!
//! Every result is compared twice: with the continuum formula, using the
//! analytic `ε(ω) = ε∞ + Σχ(ω)` (errors at discretisation level), and with
//! the exact response of the discrete Yee line in the medium the scheme
//! realises, `ε∞ + Σχ(Ω_t)` with `Ω_t = (2/dt)·tan(ω·dt/2)`
//! (`Pole::discrete_susceptibility`, `common::discrete_response`), which
//! must agree to the level of the time-signal truncation and CPML echo.

mod common;

use std::f64::consts::PI;

use common::*;
use ems::compile::{CompileOptions, compile};
use ems::engine::dispersive::{Dispersive, Field};
use ems::engine::{CellMaterials, EdgeSource, Engine, Grid, Material, Pole, Probe};
use ems::{Axis, Boundaries, Boundary, Complex64, Cpml, Face, Fdtd};

const DZ: f64 = 0.5e-3;
const IFACE: usize = 148;
const SRC: usize = 48;
const PRB: usize = 88;
const F0: f64 = 4.5e9;
const BW: f64 = 3.5e9;
const TAU2PI: f64 = 2.0 * PI;

fn band() -> Vec<f64> {
    linspace(1e9, 8e9, 29)
}

/// A dispersive structure on the line: cells `range` filled with
/// `eps_inf` (and `mu_inf`) plus electric / magnetic poles.
#[derive(Clone)]
struct Medium {
    eps_inf: f64,
    mu_inf: f64,
    electric: Vec<Pole>,
    magnetic: Vec<Pole>,
}

impl Medium {
    fn electric(eps_inf: f64, poles: &[Pole]) -> Self {
        Medium {
            eps_inf,
            mu_inf: 1.0,
            electric: poles.to_vec(),
            magnetic: Vec::new(),
        }
    }

    /// Continuum `(ε, μ)` at `f`.
    fn continuum(&self, f: f64) -> (Complex64, Complex64) {
        let w = TAU2PI * f;
        let e: Complex64 = self.electric.iter().map(|p| p.susceptibility(w)).sum();
        let m: Complex64 = self.magnetic.iter().map(|p| p.susceptibility(w)).sum();
        (e + self.eps_inf, m + self.mu_inf)
    }

    /// `(ε, μ)` realised by the scheme at `f`.
    fn discrete(&self, f: f64, dt: f64) -> (Complex64, Complex64) {
        let e: Complex64 = self
            .electric
            .iter()
            .map(|p| p.discrete_susceptibility(f, dt))
            .sum();
        let m: Complex64 = self
            .magnetic
            .iter()
            .map(|p| p.discrete_susceptibility(f, dt))
            .sum();
        (e + self.eps_inf, m + self.mu_inf)
    }
}

/// Runs `line` (vacuum background, its layers ignored) with `medium` in
/// z-cells `range`; returns `dt` and the probe series.
fn run_line(
    line: &Line,
    medium: Option<(&Medium, core::ops::Range<usize>)>,
    steps: u64,
) -> (f64, Vec<Vec<f64>>) {
    let mut l = line.clone();
    l.layers.clear();
    if let Some((m, r)) = &medium {
        l.layers.push((
            r.clone(),
            Material {
                eps_r: [m.eps_inf; 3],
                mu_r: [m.mu_inf; 3],
                ..Material::VACUUM
            },
        ));
    }
    let (mut e, hs) = l.build(sine_pulse(F0, BW));
    if let Some((m, r)) = medium {
        let g = e.grid().clone();
        let mut d = Dispersive::new();
        let cells = [0..1, 0..1, r];
        d.fill_cells(&g, cells.clone(), Field::Electric, &m.electric)
            .unwrap();
        d.fill_cells(&g, cells, Field::Magnetic, &m.magnetic)
            .unwrap();
        let h = d.add_to(&mut e).unwrap();
        assert!(e.operator(&h).edges() > 0);
    }
    e.run(steps);
    let v = hs.iter().map(|h| e.operator(h).values().to_vec()).collect();
    (e.dt(), v)
}

/// Spectral ratio `(test − ref)/ref` (reflection) or `test/ref`
/// (transmission) at the probe.
fn ratio(test: &[f64], reference: &[f64], dt: f64, freqs: &[f64], refl: bool) -> Vec<Complex64> {
    let tail = peak(&test[test.len() - 50..]) / peak(reference);
    assert!(tail < 1e-4, "record too short: tail {tail:e}");
    let a = spectrum(reference, dt, freqs);
    let num = if refl {
        diff(test, reference)
    } else {
        test.to_vec()
    };
    let b = spectrum(&num, dt, freqs);
    b.iter().zip(&a).map(|(&b, &a)| b / a).collect()
}

/// Per-cell profiles from local cell `c0` to the end, for
/// `discrete_response`: vacuum except `range` (global cells).
fn profiles(
    m: &Medium,
    f: f64,
    dt: f64,
    c0: usize,
    n: usize,
    range: core::ops::Range<usize>,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let (e, mu) = m.discrete(f, dt);
    let mut pe = vec![Complex64::ONE; n];
    let mut pm = vec![Complex64::ONE; n];
    for c in range {
        pe[c - c0] = e;
        pm[c - c0] = mu;
    }
    (pe, pm)
}

struct Errors {
    continuum: f64,
    discrete: f64,
}

/// Reflection from a half-space of `m` starting at `IFACE` (into the far
/// CPML).
fn half_space(m: &Medium, label: &str) -> Errors {
    let cells = IFACE + 100 + 24;
    let mut l = Line::new(DZ, cells);
    l.pml[1] = Cpml::cells(24).sigma_factor(1.0 / m.eps_inf.sqrt());
    l.src = SRC;
    l.probes = vec![PRB];
    let steps = 2400;
    let (dt, t) = run_line(&l, Some((m, IFACE..cells)), steps);
    let (_, r) = run_line(&l, None, steps);
    let freqs = band();
    let g = ratio(&t[0], &r[0], dt, &freqs, true);
    let (mut ec, mut ed) = (0.0f64, 0.0f64);
    for (&f, gm) in freqs.iter().zip(&g) {
        let (e, mu) = m.continuum(f);
        let gc = half_space_r(e, mu).norm();
        // Discrete: a long stretch of the medium (lossy or evanescent, so
        // the exit condition is immaterial) after 4 vacuum cells.
        let (pe, pm) = profiles(m, f, dt, 0, 8, 4..8);
        let (gd, _) = discrete_response(&pe, &pm, DZ, dt, f);
        ec = ec.max((gm.norm() - gc).abs());
        ed = ed.max((gm.norm() - gd.norm()).abs());
    }
    eprintln!(
        "{label}: |Γ| = {:.4} (1 GHz) .. {:.4} (8 GHz); max ||Γ| − |Γ_continuum|| {ec:.2e}, \
         max ||Γ| − |Γ_discrete|| {ed:.2e}",
        g[0].norm(),
        g[g.len() - 1].norm()
    );
    Errors {
        continuum: ec,
        discrete: ed,
    }
}

/// Reflection and transmission of a slab of `m`, `n_slab` cells from
/// `IFACE`; transmission probe 20 cells behind it.
fn slab(m: &Medium, n_slab: usize, steps: u64, label: &str) -> (Errors, Errors, Vec<f64>) {
    let cells = IFACE + n_slab + 80 + 24;
    let mut l = Line::new(DZ, cells);
    l.pml[1] = Cpml::cells(24);
    l.src = SRC;
    let behind = IFACE + n_slab + 20;
    l.probes = vec![PRB, behind];
    let range = IFACE..IFACE + n_slab;
    let (dt, t) = run_line(&l, Some((m, range.clone())), steps);
    let (_, r) = run_line(&l, None, steps);
    let freqs = band();
    let g = ratio(&t[0], &r[0], dt, &freqs, true);
    let tr = ratio(&t[1], &r[1], dt, &freqs, false);
    let d = n_slab as f64 * DZ;
    let (mut rc, mut rd, mut tc, mut td) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let c0 = IFACE - 4;
    let n = n_slab + 8 + 20;
    for (q, &f) in freqs.iter().enumerate() {
        let (e, mu) = m.continuum(f);
        let (gc, tcn) = slab_rt(f, e, mu, d);
        let (pe, pm) = profiles(m, f, dt, c0, n, range.clone());
        let (gd, ed) = discrete_response(&pe, &pm, DZ, dt, f);
        // Unit incident wave e^{−jK0·k} from local node 0.
        let k0 = yee_k(f, dt, DZ, 1.0, 0.0);
        let kl = (behind - c0) as f64;
        let tdn = ed[behind - c0] / (-Complex64::I * k0 * kl).exp();
        rc = rc.max((g[q].norm() - gc.norm()).abs());
        rd = rd.max((g[q].norm() - gd.norm()).abs());
        tc = tc.max((tr[q].norm() - tcn.norm()).abs());
        td = td.max((tr[q].norm() - tdn.norm()).abs());
    }
    let tmag: Vec<f64> = tr.iter().map(|t| t.norm()).collect();
    eprintln!(
        "{label}: reflection max err vs continuum {rc:.2e}, vs discrete {rd:.2e}; \
         transmission max err vs continuum {tc:.2e}, vs discrete {td:.2e}"
    );
    (
        Errors {
            continuum: rc,
            discrete: rd,
        },
        Errors {
            continuum: tc,
            discrete: td,
        },
        tmag,
    )
}

/// Drude medium of the csx test: ε∞ = 2, openEMS f_plasma = 4 GHz,
/// τ = 0.2 ns, i.e. `ε = 2·(1 − f_p²/(f² − jf/(2πτ)))`.
fn drude_medium() -> Medium {
    let eps_inf = 2.0;
    Medium::electric(
        eps_inf,
        &[Pole::drude(TAU2PI * 4e9 * eps_inf.sqrt(), 1.0 / 0.2e-9)],
    )
}

/// (a) Drude half-space: |Γ(f)| ≈ 1 below the plasma frequency (4 GHz),
/// dropping above it; matches the continuum `ε(ω)` within 1e-2 and the
/// discrete-ε(ω) response within 2e-3.
#[test]
fn drude_half_space_reflection() {
    let _t = Timer::start("drude_half_space_reflection");
    let e = half_space(
        &drude_medium(),
        "Drude half-space (ε∞ = 2, f_p = 4 GHz, τ = 0.2 ns)",
    );
    assert!(e.continuum < 3e-3, "{}", e.continuum);
    assert!(e.discrete < 5e-4, "{}", e.discrete);
}

/// (b) Debye slab (ε∞ = 2, Δε = 4, τ = 20 ps), 10 mm.
#[test]
fn debye_slab_reflection() {
    let _t = Timer::start("debye_slab_reflection");
    let m = Medium::electric(2.0, &[Pole::debye(4.0, 20e-12)]);
    let (r, t, _) = slab(
        &m,
        20,
        2000,
        "Debye slab (ε∞ = 2, Δε = 4, τ = 20 ps, d = 10 mm)",
    );
    assert!(
        r.continuum < 5e-3 && t.continuum < 5e-3,
        "{} {}",
        r.continuum,
        t.continuum
    );
    assert!(
        r.discrete < 5e-4 && t.discrete < 5e-4,
        "{} {}",
        r.discrete,
        t.discrete
    );
}

/// (c) Lorentz absorption line (ε∞ = 1, f0 = 5 GHz, Δε = 2,
/// γ = 2π·1 GHz), 10 mm slab: the transmission dip sits at the line and
/// |T(f)| matches the analytic slab transmission.
#[test]
fn lorentz_slab_absorption() {
    let _t = Timer::start("lorentz_slab_absorption");
    let w0 = TAU2PI * 5e9;
    let m = Medium::electric(1.0, &[Pole::lorentz(w0 * 2f64.sqrt(), w0, TAU2PI * 1e9)]);
    let (r, t, tmag) = slab(
        &m,
        20,
        4000,
        "Lorentz slab (f0 = 5 GHz, Δε = 2, γ = 2π·1 GHz)",
    );
    let freqs = band();
    let (qmin, tmin) = tmag
        .iter()
        .enumerate()
        .fold((0, f64::MAX), |a, (q, &v)| if v < a.1 { (q, v) } else { a });
    let d = 20.0 * DZ;
    let tan_min = freqs
        .iter()
        .map(|&f| (f, slab_rt(f, m.continuum(f).0, Complex64::ONE, d).1.norm()))
        .fold((0.0, f64::MAX), |a, b| if b.1 < a.1 { b } else { a });
    eprintln!(
        "  |T| minimum {tmin:.4} at {:.2} GHz (analytic {:.4} at {:.2} GHz)",
        freqs[qmin] / 1e9,
        tan_min.1,
        tan_min.0 / 1e9
    );
    assert!((freqs[qmin] - tan_min.0).abs() < 0.3e9);
    assert!(tmin < 0.3);
    assert!(
        r.continuum < 5e-3 && t.continuum < 5e-3,
        "{} {}",
        r.continuum,
        t.continuum
    );
    assert!(
        r.discrete < 5e-4 && t.discrete < 5e-4,
        "{} {}",
        r.discrete,
        t.discrete
    );
}

/// Magnetic Lorentz poles: a slab with ε(ω) = μ(ω) (same Lorentz pole on
/// both) is impedance matched, so |Γ| ≈ 0 while the transmission shows
/// the absorption line.
#[test]
fn magnetic_matched_slab() {
    let _t = Timer::start("magnetic_matched_slab");
    let w0 = TAU2PI * 6e9;
    let p = Pole::lorentz(w0, w0, TAU2PI * 1e9);
    let m = Medium {
        eps_inf: 1.0,
        mu_inf: 1.0,
        electric: vec![p],
        magnetic: vec![p],
    };
    let (r, t, tmag) = slab(
        &m,
        20,
        4000,
        "matched ε = μ Lorentz slab (f0 = 6 GHz, Δ = 1)",
    );
    let tmin = tmag.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!("  |T| minimum {tmin:.4}");
    // Continuum Γ = 0: the measured |Γ| is the discretisation mismatch
    // (E/H staggering), reproduced by the discrete response.
    assert!(r.continuum < 2e-2, "{}", r.continuum);
    assert!(
        r.discrete < 1e-4 && t.discrete < 1e-4,
        "{} {}",
        r.discrete,
        t.discrete
    );
    assert!(t.continuum < 1e-2, "{}", t.continuum);
    assert!(tmin < 0.2);
}

// ------------------------------------------------------------ stability

/// A closed PEC cavity, vacuum with a dispersive block, an initial field
/// and no source.
fn cavity(
    threads: usize,
    lossless: bool,
    magnetic: bool,
) -> (Engine, ems::engine::OperatorHandle<Dispersive>) {
    let g = Grid::uniform([10, 8, 6], [1e-3; 3]).unwrap();
    let mut mats = CellMaterials::new(&g, Material::VACUUM);
    let blk = [2..8, 1..6, 1..5];
    mats.fill_box(blk.clone(), Material::dielectric(1.5));
    let mut e = Engine::builder(g.clone())
        .materials(mats)
        .threads(threads)
        .build()
        .unwrap();
    let gam = if lossless { 0.0 } else { 3e9 };
    let w = TAU2PI * 20e9;
    let mut d = Dispersive::new();
    d.fill_cells(
        &g,
        blk.clone(),
        Field::Electric,
        &[
            Pole::lorentz(w, w, gam),
            Pole::drude(0.7 * w, gam),
            Pole::debye(2.0, if lossless { 0.0 } else { 5e-12 }),
        ],
    )
    .unwrap();
    if magnetic {
        d.fill_cells(
            &g,
            blk,
            Field::Magnetic,
            &[Pole::lorentz(0.5 * w, 1.5 * w, gam)],
        )
        .unwrap();
    }
    let h = d.add_to(&mut e).unwrap();
    {
        let mut f = e.fields_mut();
        for (q, n) in [[3, 3, 2], [5, 2, 3], [1, 4, 2], [6, 5, 3]]
            .iter()
            .enumerate()
        {
            f.u.set(Axis::ALL[q % 3], *n, 1.0 + q as f32);
        }
    }
    (e, h)
}

fn energy_drift(steps: u64) -> (f64, f64) {
    energy_drift_with(steps, false)
}

fn energy_drift_with(steps: u64, magnetic: bool) -> (f64, f64) {
    let (mut e, _) = cavity(1, true, magnetic);
    e.run(10);
    let w0 = e.energy();
    let (mut lo, mut hi) = (w0, w0);
    let chunk = 50;
    for _ in 0..steps / chunk {
        e.run(chunk);
        let w = e.energy();
        lo = lo.min(w);
        hi = hi.max(w);
    }
    ((hi - lo) / w0, e.operator_energy() / w0)
}

/// (d) Lossless Lorentz + Drude + Debye(τ = 0) block in a closed cavity:
/// field + pole energy is conserved (relative 1e-5) over 4000 steps,
/// while a sizeable share of it sits in the poles.
#[test]
fn lossless_poles_conserve_energy() {
    let _t = Timer::start("lossless_poles_conserve_energy");
    let (drift, share) = energy_drift(4000);
    eprintln!("lossless poles, 4000 steps: energy drift {drift:.2e}, pole share {share:.3}");
    assert!(drift < 1e-5, "{drift}");
    assert!(share > 0.01, "{share}");
}

/// Lossless electric and magnetic poles: the engine energy (which
/// includes the magnetic branch energy at `(n − ½)·dt`) is conserved to
/// the same level as the electric-only case, and the value
/// `step_with_energy` computes during a real step equals `energy()`.
#[test]
fn lossless_magnetic_poles_conserve_energy() {
    let (drift, share) = energy_drift_with(4000, true);
    eprintln!(
        "lossless electric + magnetic poles, 4000 steps: drift {drift:.2e}, pole share {share:.3}"
    );
    assert!(drift < 1e-5, "{drift}");
    let (mut e, h) = cavity(1, true, true);
    e.run(37);
    assert!(e.operator(&h).magnetic_energy() > 0.0);
    let w = e.energy();
    let ws = e.step_with_energy();
    assert_eq!(w.to_bits(), ws.to_bits(), "{w} vs {ws}");
}

/// Long-run variant: 100 000 steps.
#[test]
#[ignore = "long run (≈1e5 steps)"]
fn lossless_poles_conserve_energy_long() {
    let (drift, _) = energy_drift(100_000);
    eprintln!("lossless poles, 1e5 steps: energy drift {drift:.2e}");
    assert!(drift < 1e-4, "{drift}");
}

/// Lossy poles: the energy never increases and decays.
#[test]
fn lossy_poles_dissipate() {
    let (mut e, _) = cavity(1, false, false);
    e.run(2);
    let mut w = e.energy();
    let w0 = w;
    for _ in 0..200 {
        e.run(10);
        let wn = e.energy();
        assert!(wn <= w * (1.0 + 1e-6), "{wn} > {w}");
        w = wn;
    }
    eprintln!("lossy poles: W/W0 after 2000 steps {:.3e}", w / w0);
    assert!(w < 0.5 * w0);
}

/// Electric and magnetic poles (lossy and lossless), long run: bounded,
/// and with losses never increasing.
#[test]
fn magnetic_poles_stay_bounded() {
    for lossless in [true, false] {
        let (mut e, _) = cavity(1, lossless, true);
        e.run(2);
        let w0 = e.energy();
        let mut wmax = 0.0f64;
        let mut w = w0;
        for _ in 0..60 {
            e.run(50);
            let wn = e.energy();
            if !lossless {
                assert!(wn <= w * (1.0 + 1e-6), "{wn} > {w}");
            }
            w = wn;
            wmax = wmax.max(wn);
        }
        eprintln!(
            "magnetic poles (lossless {lossless}): max W/W0 over 3000 steps {:.4}",
            wmax / w0
        );
        assert!(wmax < 1.05 * w0, "{wmax} {w0}");
    }
}

/// (d) Bit-identical results for 1 and 4 threads, with electric and
/// magnetic poles crossing the slab boundaries.
#[test]
fn deterministic_across_threads() {
    let run = |threads: usize| {
        let (mut e, h) = cavity(threads, false, true);
        assert_eq!(e.threads(), threads);
        e.run(300);
        let u: Vec<Vec<f32>> = Axis::ALL.iter().map(|&a| e.u_component(a)).collect();
        let i: Vec<Vec<f32>> = Axis::ALL.iter().map(|&a| e.i_component(a)).collect();
        let w = (
            e.operator(&h).electric_energy(),
            e.operator(&h).magnetic_energy(),
            e.energy(),
        );
        (u, i, w)
    };
    let (u1, i1, w1) = run(1);
    let (u4, i4, w4) = run(4);
    assert!(u1 == u4 && i1 == i4);
    assert_eq!(w1.0.to_bits(), w4.0.to_bits());
    assert_eq!(w1.1.to_bits(), w4.1.to_bits());
    assert_eq!(w1.2.to_bits(), w4.2.to_bits());
}

// ------------------------------------------------------------ csx path

/// (e) The Drude half-space as a csx `LorentzMaterial` (openEMS
/// parameters: Epsilon = 2, EpsilonPlasmaFrequency = 4 GHz,
/// EpsilonRelaxTime = 0.2 ns) compiled from a csx model gives the same
/// probe signal as the engine-level setup, and no dispersion warning.
#[test]
fn csx_lorentz_material_matches_engine_setup() {
    let _t = Timer::start("csx_lorentz_material_matches_engine_setup");
    let cells = IFACE + 40 + 8;
    let steps = 700;
    // Engine-level.
    let mut l = Line::new(DZ, cells);
    l.src = SRC;
    l.probes = vec![PRB, IFACE + 5];
    let (_, want) = run_line(&l, Some((&drude_medium(), IFACE..cells)), steps);

    // csx model: same line, millimetre drawing units.
    let mut model = csx::ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    model.grid.set_lines(0, [0.0, 5.0]).unwrap();
    model.grid.set_lines(1, [0.0, 5.0]).unwrap();
    model
        .grid
        .set_lines(2, (0..=cells).map(|q| q as f64 * DZ * 1e3))
        .unwrap();
    let mut lm = csx::properties::LorentzMaterial::default();
    lm.material.epsilon.set(2.0);
    let mut pole = csx::properties::LorentzPole::zero();
    pole.eps_plasma_frequency.set(4e9);
    pole.eps_relax_time.set(0.2e-9);
    lm.poles.push(pole);
    let id = model.add_property(csx::Property::lorentz_material("drude", lm));
    model
        .add_primitive(
            id,
            csx::Primitive::new(csx::primitives::BoxShape::new(
                [-1.0, -1.0, IFACE as f64 * DZ * 1e3],
                [6.0, 6.0, cells as f64 * DZ * 1e3 + 1.0],
            )),
        )
        .unwrap();
    let fdtd = Fdtd::new().threads(1).boundaries(
        Boundaries::all(Boundary::Pec)
            .with(Face::YMin, Boundary::Pmc)
            .with(Face::YMax, Boundary::Pmc)
            .with(Face::ZMin, Boundary::Pml(Cpml::cells(8)))
            .with(Face::ZMax, Boundary::Pml(Cpml::cells(8))),
    );
    let c = compile(&model, &fdtd, &CompileOptions::default()).unwrap();
    assert!(
        !c.warnings.iter().any(|w| w.contains("dispers")),
        "{:?}",
        c.warnings
    );
    assert!(!c.dispersive.is_empty());
    let (mut e, h) = c.build().unwrap();
    assert!(h.dispersive.is_some());
    let mut s = EdgeSource::new(sine_pulse(F0, BW));
    for p in l.edges(SRC) {
        s = s.edge(Axis::X, p, 1.0);
    }
    e.add_operator(s);
    let ps: Vec<_> = l
        .probes
        .iter()
        .map(|&k| e.add_operator(Probe::voltage(Axis::X, l.edges(k)[0])))
        .collect();
    e.run(steps);
    for (q, p) in ps.iter().enumerate() {
        let got = e.operator(p).values();
        let err = peak(&diff(got, &want[q])) / peak(&want[q]);
        eprintln!("csx path vs engine setup, probe {q}: max rel. difference {err:.2e}");
        assert!(err < 1e-5, "{err}");
    }
}
