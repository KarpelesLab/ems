//! Oblique incidence on periodic structures (design §5.6.2): Bloch pairs
//! driven by the periodic plane-wave source, at engine level (against the
//! exact discrete and the continuum slab response) and through
//! `Simulation` (csx models with an oblique plane-wave excitation).

mod common;

use std::f64::consts::PI;

use common::*;
use ems::constants::C0;
use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::{
    BlochPair, CellMaterials, Engine, FaceCondition, FaceConditions, Grid, Material,
    OperatorHandle, PeriodicPlaneWave, PeriodicPlaneWaveSpec, Probe, TimeStep, WavePart,
};
use ems::{Axis, Complex64, Cpml, Face};

/// Cell size (all axes) and z cells.
const D: f64 = 0.5e-3;
const NZ: usize = 240;
/// Entry plane, reflection probe (scattered field) and transmission
/// probe (total field), z nodes of a forward run.
const ENTRY: usize = 50;
const REFL: usize = 30;
const TRANS: usize = 200;
/// Slab cells along z (10 mm).
const SLAB: std::ops::Range<usize> = 110..130;
const SLAB_EPS: f64 = 4.0;
/// Design frequency: 30° incidence.
const FD: f64 = 10e9;
/// Pulse: band ≈ 7–14 GHz, far from the cut-off at FD/2.
const F0: f64 = 10.5e9;
const FC: f64 = 3.5e9;
/// The time step of every run here.
const DT: f64 = 0.95 * D / C0 / 1.732_050_807_568_877_2;
const STEPS: u64 = 7000;

fn kt30() -> f64 {
    2.0 * PI * FD / C0 * 0.5
}

fn freqs() -> Vec<f64> {
    linspace(7e9, 14e9, 15)
}

fn periodic_faces() -> FaceConditions {
    FaceConditions::all(FaceCondition::Pec)
        .with(Face::XMin, FaceCondition::Periodic)
        .with(Face::XMax, FaceCondition::Periodic)
        .with(Face::YMin, FaceCondition::Periodic)
        .with(Face::YMax, FaceCondition::Periodic)
}

fn gauss() -> impl Fn(f64) -> f64 + Send + Clone + 'static {
    let w = ems::Waveform::gaussian(F0, FC);
    move |t| w.value(t, 0.0)
}

/// Complex spectra of an engine-level run.
struct Run {
    refl: Vec<Complex64>,
    trans: Vec<Complex64>,
    inc: Vec<Complex64>,
}

impl Run {
    fn r(&self, k: usize) -> f64 {
        self.refl[k].norm() / self.inc[k].norm()
    }
    fn t(&self, k: usize) -> f64 {
        self.trans[k].norm() / self.inc[k].norm()
    }
}

/// An oblique run in a 4 × 2-cell unit cell, CPML along z: transverse
/// wavevector `k_t`, tangential polarisation along `pol`, slab (if
/// `slab`), propagation along `+z` (`forward`) or mirrored (`−z`, every z
/// position reflected). Returns the complex spectra of the `pol`
/// component of E at the reflection and transmission probes, and of the
/// incident E at the entry plane.
fn oblique(k_t: [f64; 3], pol: Axis, slab: bool, forward: bool, threads: usize) -> Run {
    let g = Grid::uniform([4, 2, NZ], [D; 3]).unwrap();
    let z = |q: usize| if forward { q } else { NZ - q };
    let mut m = CellMaterials::vacuum(&g);
    if slab {
        let cells = if forward {
            SLAB
        } else {
            NZ - SLAB.end..NZ - SLAB.start
        };
        m.fill_box([0..4, 0..2, cells], Material::dielectric(SLAB_EPS));
    }
    let b = Engine::builder(g.clone())
        .materials(m)
        .faces(periodic_faces())
        .timestep(TimeStep::Fixed(DT))
        .threads(threads);
    let mut pair = BlochPair::new(b, BlochPair::phase_for(&g, k_t)).unwrap();
    let mut p = [0.0; 3];
    p[pol.index()] = 1.0;
    let (lo, hi) = if forward {
        (ENTRY, NZ)
    } else {
        (0, NZ - ENTRY)
    };
    let spec = PeriodicPlaneWaveSpec::new(Axis::Z, lo, hi, forward, k_t, p);
    let mut inc_h = None;
    let mut probes: Vec<[OperatorHandle<Probe>; 2]> = Vec::new();
    let (re, im) = pair.engines_mut();
    for (part, e) in [(WavePart::Real, re), (WavePart::Imag, im)] {
        let c = CpmlBoundary::new(
            e,
            [(Face::ZMin, Cpml::cells(10)), (Face::ZMax, Cpml::cells(10))],
        )
        .unwrap();
        e.add_operator(c);
        let pw = PeriodicPlaneWave::new(e, &spec, part, gauss()).unwrap();
        let h = e.add_operator(pw);
        if part == WavePart::Real {
            inc_h = Some(h);
        }
        probes.push([REFL, TRANS].map(|q| e.add_operator(Probe::voltage(pol, [1, 1, z(q)]))));
    }
    pair.run(STEPS);
    let f = freqs();
    let series = |e: &Engine, h: &OperatorHandle<Probe>| -> Vec<f64> {
        e.operator(h).values().iter().map(|u| u / D).collect()
    };
    let cplx = |k: usize| -> Vec<Complex64> {
        let re = spectrum(&series(pair.real(), &probes[0][k]), DT, &f);
        let im = spectrum(&series(pair.imag(), &probes[1][k]), DT, &f);
        re.iter()
            .zip(&im)
            .map(|(&r, &i)| r + Complex64::I * i)
            .collect()
    };
    let inc_op = pair.real().operator(&inc_h.unwrap());
    let part = |g: fn(Complex64) -> f64| -> Vec<f64> {
        inc_op
            .incident()
            .iter()
            .map(|v| g(v[pol.index()]))
            .collect()
    };
    let inc = spectrum(&part(|c| c.re), DT, &f)
        .iter()
        .zip(spectrum(&part(|c| c.im), DT, &f))
        .map(|(&r, i)| r + Complex64::I * i)
        .collect();
    Run {
        refl: cplx(0),
        trans: cplx(1),
        inc,
    }
}

/// Fresnel reflection and transmission of a lossless slab (`eps_r`,
/// thickness `d`) in vacuum at transverse wavenumber `kt`, TE or TM, as
/// ratios of the tangential field of the reflected / transmitted wave.
fn slab(f: f64, kt: f64, eps_r: f64, d: f64, te: bool) -> (Complex64, Complex64) {
    let k0 = 2.0 * PI * f / C0;
    let kz1 = Complex64::new(k0 * k0 - kt * kt, 0.0).sqrt();
    let kz2 = Complex64::new(eps_r * k0 * k0 - kt * kt, 0.0).sqrt();
    // Wave admittances of the tangential fields (up to a common factor).
    let (y1, y2) = if te {
        (kz1, kz2)
    } else {
        (k0 * k0 / kz1, eps_r * k0 * k0 / kz2)
    };
    let g = (y1 - y2) / (y1 + y2);
    let p = (Complex64::new(0.0, -2.0) * kz2 * d).exp();
    let r = g * (1.0 - p) / (1.0 - g * g * p);
    let t = (1.0 - g * g) * (Complex64::new(0.0, -1.0) * kz2 * d).exp() / (1.0 - g * g * p);
    (r, t)
}

/// The exact reflection and transmission of the discrete (Yee) scheme for
/// the slab cells `SLAB` (tangential edges on the slab faces carry the
/// mean of the two cells), spacing `D`, time step `DT`, discrete
/// transverse wavenumber of `kt`: the 1-D recurrence of the Bloch-phased
/// Yee equations, marched from the transmitted side. TE: tangential E ⊥
/// `k_t`; TM: tangential H ⊥ `k_t`.
fn yee_slab(f: f64, kt: f64, eps_r: f64, te: bool) -> (Complex64, Complex64) {
    use ems::constants::{EPS0, MU0};
    let om = 2.0 / DT * (PI * f * DT).sin();
    let k = 2.0 / D * (0.5 * kt * D).sin();
    let j = Complex64::I;
    let cell_eps = |c: isize| {
        if c >= 0 && SLAB.contains(&(c as usize)) {
            eps_r
        } else {
            1.0
        }
    };
    let node_eps = |n: isize| 0.5 * (cell_eps(n - 1) + cell_eps(n)) * EPS0;
    // H(n+½) = (E(n+1) − E(n))/(D·a(n+½)),  (H(n+½) − H(n−½))/D = b(n)·E(n).
    let a = |c: isize| -> Complex64 {
        if te {
            j * (om * MU0)
        } else {
            -j * (om * MU0 - k * k / (om * cell_eps(c) * EPS0))
        }
    };
    let b = |n: isize| -> Complex64 {
        if te {
            j * (om * node_eps(n) - k * k / (om * MU0))
        } else {
            -j * (om * node_eps(n))
        }
    };
    // Vacuum propagation per cell: cos κ = 1 + D²·a·b/2.
    let kappa = (1.0 + 0.5 * D * D * (a(-10) * b(-10)).re).acos();
    let (lo, hi) = (SLAB.start as isize - 5, SLAB.end as isize + 5);
    let w = |n: isize| Complex64::cis(-kappa * n as f64);
    let (mut e1, mut e0) = (w(hi + 1), w(hi));
    let mut h = (e1 - e0) / (a(hi) * D);
    let mut n = hi;
    while n > lo {
        let hm = h - b(n) * e0 * D;
        let em = e0 - a(n - 1) * hm * D;
        (e1, e0, h) = (e0, em, hm);
        n -= 1;
    }
    // E(n) = A·w(n) + B/w(n) at n = lo, lo + 1.
    let (wl, wl1) = (w(lo), w(lo + 1));
    let det = wl / wl1 - wl1 / wl;
    let aa = (e0 / wl1 - e1 / wl) / det;
    let bb = (e1 * wl - e0 * wl1) / det;
    (bb / aa, 1.0 / aa)
}

/// Engine-level slab at 30° (at FD): |R|, |T| against the exact discrete
/// response, the continuum (Fresnel) response, and power conservation.
fn check_slab(te: bool) {
    let kt = kt30();
    let pol = if te { Axis::Y } else { Axis::X };
    let run = oblique([kt, 0.0, 0.0], pol, true, true, 2);
    let (mut yee, mut fres, mut power) = (0.0f64, 0.0f64, 0.0f64);
    for (k, &f) in freqs().iter().enumerate() {
        let (r, t) = (run.r(k), run.t(k));
        let (rf, tf) = slab(f, kt, SLAB_EPS, 10e-3, te);
        let (ry, ty) = yee_slab(f, kt, SLAB_EPS, te);
        eprintln!(
            "{:6.2} GHz  θ {:4.1}°  |R| {r:.5} (Yee {:.5}, Fresnel {:.5})  |T| {t:.5} \
             (Yee {:.5}, Fresnel {:.5})",
            f / 1e9,
            (kt * C0 / (2.0 * PI * f)).asin().to_degrees(),
            ry.norm(),
            rf.norm(),
            ty.norm(),
            tf.norm()
        );
        yee = yee.max((r - ry.norm()).abs()).max((t - ty.norm()).abs());
        if f <= 12.5e9 {
            fres = fres.max((r - rf.norm()).abs()).max((t - tf.norm()).abs());
        }
        power = power.max((r * r + t * t - 1.0).abs());
    }
    let mode = if te { "TE" } else { "TM" };
    eprintln!(
        "{mode}: max deviation from the discrete response {yee:.1e}, from Fresnel (≤ 12.5 GHz) \
         {fres:.1e}, | |R|²+|T|² − 1 | ≤ {power:.1e}"
    );
    assert!(yee < 3e-4, "{yee}");
    assert!(fres < 1e-2, "{fres}");
    assert!(power < 5e-4, "{power}");
}

#[test]
fn oblique_slab_te() {
    let _t = Timer::start("oblique_slab_te");
    check_slab(true);
}

#[test]
fn oblique_slab_tm() {
    let _t = Timer::start("oblique_slab_tm");
    check_slab(false);
}

/// Without a scatterer the scattered-field region stays empty and the
/// incident wave passes unchanged, for both propagation directions and
/// polarisations, with the wavevector along x or y.
#[test]
fn empty_cell_leakage() {
    let _t = Timer::start("empty_cell_leakage");
    let kt = kt30();
    for (k_t, pol, forward) in [
        ([kt, 0.0, 0.0], Axis::Y, true),
        ([0.0, kt, 0.0], Axis::Y, false),
        ([0.6 * kt, 0.8 * kt, 0.0], Axis::X, true),
    ] {
        let run = oblique(k_t, pol, false, forward, 1);
        let (mut leak, mut pass) = (0.0f64, 0.0f64);
        for k in 0..freqs().len() {
            leak = leak.max(run.r(k));
            pass = pass.max((run.t(k) - 1.0).abs());
        }
        eprintln!(
            "k_t {k_t:.1?}, E along {pol}, forward {forward}: leakage {:.1} dB, \
             |T| − 1 ≤ {pass:.1e}",
            db(leak)
        );
        assert!(leak < 1e-4, "{leak}");
        assert!(pass < 1e-4, "{pass}");
    }
}

/// `k_t = 0`: the Bloch pair reproduces a plain periodic engine with the
/// same source bit for bit (real part), and the imaginary part stays zero.
#[test]
fn zero_wavevector_is_plain_periodic() {
    let _t = Timer::start("zero_wavevector_is_plain_periodic");
    let g = Grid::uniform([4, 2, 120], [D; 3]).unwrap();
    let mut m = CellMaterials::vacuum(&g);
    m.fill_box([0..4, 0..2, 60..70], Material::dielectric(3.0));
    let builder = || {
        Engine::builder(g.clone())
            .materials(m.clone())
            .faces(periodic_faces())
            .timestep(TimeStep::Fixed(DT))
            .threads(2)
    };
    let spec = PeriodicPlaneWaveSpec::new(Axis::Z, 30, 120, true, [0.0; 3], [1.0, 0.0, 0.0]);
    let install = |e: &mut Engine, part| {
        let c = CpmlBoundary::new(
            e,
            [(Face::ZMin, Cpml::cells(8)), (Face::ZMax, Cpml::cells(8))],
        )
        .unwrap();
        e.add_operator(c);
        e.add_operator(PeriodicPlaneWave::new(e, &spec, part, gauss()).unwrap());
        e.add_operator(Probe::voltage(Axis::X, [2, 1, 20]))
    };
    let mut plain = builder().build().unwrap();
    let hp = install(&mut plain, WavePart::Real);
    let mut pair = BlochPair::new(builder(), [0.0; 3]).unwrap();
    let (re, im) = pair.engines_mut();
    let hr = install(re, WavePart::Real);
    let hi = install(im, WavePart::Imag);
    plain.run(1500);
    pair.run(1500);
    let vp = plain.operator(&hp).values();
    assert!(vp.iter().any(|&v| v != 0.0));
    assert_eq!(vp, pair.real().operator(&hr).values());
    assert!(pair.imag().operator(&hi).values().iter().all(|&v| v == 0.0));
}

/// Results are bit-identical for any thread count.
#[test]
fn bloch_determinism() {
    let _t = Timer::start("bloch_determinism");
    let kt = kt30();
    let a = oblique([kt, 0.0, 0.0], Axis::Y, true, true, 1);
    let b = oblique([kt, 0.0, 0.0], Axis::Y, true, true, 3);
    assert_eq!(a.refl, b.refl);
    assert_eq!(a.trans, b.trans);
    assert_eq!(a.inc, b.inc);
}

/// Through `Simulation`: csx models of a periodic cell with an oblique
/// plane-wave excitation.
mod sim {
    use super::*;
    use csx::primitives::BoxShape;
    use csx::properties::{Excitation, ExcitationType, ProbeBox, ProbeType};
    use csx::{ContinuousStructure, Primitive, Property, Scalar};
    use ems::{Boundaries, Boundary, Fdtd, Simulation, Waveform};

    /// What the cell contains.
    pub enum Scatterer {
        Nothing,
        /// The 10 mm slab of `SLAB_EPS` (cells `SLAB`).
        Slab,
        /// A PEC patch of `w` cells (square) in the `cells` × `cells` unit
        /// cell, at z node 120.
        Patch {
            cells: usize,
            w: usize,
        },
    }

    /// The cell in mm: `cx` × `cy` cells of `D`, z as in the engine
    /// tests; a plane-wave box from `ENTRY` to the end, direction at 30°
    /// in the xz plane (normal incidence with `normal`), TE (E along y) or
    /// TM; voltage probes on one E edge of the `pol` component at `REFL`
    /// and `TRANS`.
    pub fn model(s: &Scatterer, te: bool, normal: bool) -> (ContinuousStructure, Axis) {
        let (cx, cy) = match s {
            Scatterer::Patch { cells, .. } => (*cells, *cells),
            _ => (4, 2),
        };
        let d = D * 1e3;
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        for (a, n) in [(0, cx), (1, cy), (2, NZ)] {
            m.grid.set_lines(a, (0..=n).map(|i| i as f64 * d)).unwrap();
        }
        let (lx, ly, lz) = (cx as f64 * d, cy as f64 * d, NZ as f64 * d);
        match s {
            Scatterer::Nothing => {}
            Scatterer::Slab => {
                let id = m.add_property(Property::material(
                    "slab",
                    csx::Material::new(SLAB_EPS, 0.0),
                ));
                let (z0, z1) = (SLAB.start as f64 * d, SLAB.end as f64 * d);
                m.add_primitive(
                    id,
                    Primitive::new(BoxShape::new([0.0, 0.0, z0], [lx, ly, z1])),
                )
                .unwrap();
            }
            Scatterer::Patch { cells, w } => {
                let id = m.add_property(Property::metal("patch"));
                let a = (cells - w) as f64 / 2.0 * d;
                let z = 120.0 * d;
                m.add_primitive(
                    id,
                    Primitive::new(BoxShape::new(
                        [a, a, z],
                        [a + *w as f64 * d, a + *w as f64 * d, z],
                    )),
                )
                .unwrap();
            }
        }
        let (s30, c30) = if normal {
            (0.0, 1.0)
        } else {
            (0.5, 0.75f64.sqrt())
        };
        let (excite, pol) = if te {
            ([0.0, 1.0, 0.0], Axis::Y)
        } else {
            ([c30, 0.0, -s30], Axis::X)
        };
        let mut ex = Excitation::new(excite).with_type(ExcitationType::PlaneWave);
        ex.prop_dir = [s30, 0.0, c30].map(Scalar::new);
        let id = m.add_property(Property::excitation("pw", ex));
        m.add_primitive(
            id,
            Primitive::new(BoxShape::new([0.0, 0.0, ENTRY as f64 * d], [lx, ly, lz])),
        )
        .unwrap();
        for (name, q) in [("refl", REFL), ("trans", TRANS)] {
            let p = m.add_property(Property::probe(name, ProbeBox::new(ProbeType::Voltage)));
            let mut a = [1.0 * d, 1.0 * d, q as f64 * d];
            let mut b = a;
            b[pol.index()] += d;
            if pol == Axis::Y {
                a[1] = 0.0;
                b[1] = d;
            }
            m.add_primitive(p, Primitive::new(BoxShape::new(a, b)))
                .unwrap();
        }
        (m, pol)
    }

    pub fn fdtd(threads: usize) -> Fdtd {
        let pml = Boundary::pml(10);
        Fdtd::new()
            .waveform(Waveform::gaussian(F0, FC))
            .boundaries(
                Boundaries::all(Boundary::Periodic)
                    .with(Face::ZMin, pml)
                    .with(Face::ZMax, pml),
            )
            .timestep(DT)
            .max_timesteps(STEPS)
            .threads(threads)
            .bloch_frequency(FD)
    }

    /// `|R|`, `|T|` per frequency: probe spectra over the incident wave's
    /// (Bloch runs) or the waveform's (plain runs, where the TF/SF
    /// reference corner is the entry plane).
    pub fn rt(sim: &mut Simulation, pol: Axis) -> (Vec<f64>, Vec<f64>) {
        let res = sim.run().unwrap();
        let f = freqs();
        let inc: Vec<Complex64> = match res.incident("pw") {
            Some(w) => w.dft(&f).iter().map(|v| v[pol.index()]).collect(),
            None => {
                let wf = Waveform::gaussian(F0, FC);
                let a: Vec<f64> = (1..=res.probe("refl").unwrap().times().len())
                    .map(|n| wf.value(n as f64 * DT, DT))
                    .collect();
                let p = sim.plane_wave("pw").unwrap().spec().polarization[pol.index()];
                spectrum(&a, DT, &f).iter().map(|&x| x * p).collect()
            }
        };
        let ratio = |name: &str| -> Vec<f64> {
            let v = res.probe(name).unwrap().dft(&f).unwrap();
            v.iter()
                .zip(&inc)
                .map(|(v, i)| v.norm() / D / i.norm())
                .collect()
        };
        (ratio("refl"), ratio("trans"))
    }
}

/// Bloch mode through `Simulation`: a TM slab at 30° (wavevector derived
/// from the plane wave's direction at the design frequency) against the
/// exact discrete and the Fresnel response.
#[test]
fn simulation_oblique_slab() {
    let _t = Timer::start("simulation_oblique_slab");
    let (m, pol) = sim::model(&sim::Scatterer::Slab, false, false);
    let mut s = ems::Simulation::new(&m, &sim::fdtd(2)).unwrap();
    eprintln!("warnings: {:?}", s.warnings());
    assert!(s.is_bloch());
    let k = s.bloch_wavevector().unwrap();
    assert!(
        (k[0] - kt30()).abs() < 1e-9 * kt30() && k[1] == 0.0 && k[2] == 0.0,
        "{k:?}"
    );
    assert!(s.periodic_plane_wave("pw").is_some());
    assert!(
        s.warnings()
            .iter()
            .all(|w| w.starts_with("mesh ") || !w.contains("\"pw\"")),
        "{:?}",
        s.warnings()
    );
    let (r, t) = sim::rt(&mut s, pol);
    let (mut yee, mut fres) = (0.0f64, 0.0f64);
    for (q, &f) in freqs().iter().enumerate() {
        let (ry, ty) = yee_slab(f, kt30(), SLAB_EPS, false);
        let (rf, tf) = slab(f, kt30(), SLAB_EPS, 10e-3, false);
        yee = yee
            .max((r[q] - ry.norm()).abs())
            .max((t[q] - ty.norm()).abs());
        if f <= 12.5e9 {
            fres = fres
                .max((r[q] - rf.norm()).abs())
                .max((t[q] - tf.norm()).abs());
        }
    }
    eprintln!("Simulation TM slab: vs discrete {yee:.1e}, vs Fresnel (≤ 12.5 GHz) {fres:.1e}");
    assert!(yee < 1e-3, "{yee}");
    assert!(fres < 1e-2, "{fres}");
}

/// Normal incidence: the Bloch machinery forced with `k_t = 0` gives the
/// reflection and transmission of the plain periodic run (zero-phase
/// periodic faces, TF/SF plane wave).
#[test]
fn simulation_normal_incidence_matches_periodic() {
    let _t = Timer::start("simulation_normal_incidence_matches_periodic");
    let (m, pol) = sim::model(&sim::Scatterer::Slab, true, true);
    let mut plain = ems::Simulation::new(&m, &sim::fdtd(2)).unwrap();
    assert!(!plain.is_bloch());
    let mut bloch = ems::Simulation::new(&m, &sim::fdtd(2).bloch_wavevector([0.0; 3])).unwrap();
    assert!(bloch.is_bloch());
    let (rp, tp) = sim::rt(&mut plain, pol);
    let (rb, tb) = sim::rt(&mut bloch, pol);
    let mut dev = 0.0f64;
    for q in 0..rp.len() {
        dev = dev.max((rp[q] - rb[q]).abs()).max((tp[q] - tb[q]).abs());
    }
    eprintln!("normal incidence: Bloch (k_t = 0) vs plain periodic {dev:.1e}");
    assert!(dev < 1e-4, "{dev}");
}

/// A frequency-selective surface (square PEC patches, 3 mm period, at 30°
/// TE, below the first grating lobe): lossless, so |R|² + |T|² = 1 for
/// the specular order; the run ends on the energy criterion.
#[test]
fn simulation_fss_power_balance() {
    let _t = Timer::start("simulation_fss_power_balance");
    let (m, pol) = sim::model(&sim::Scatterer::Patch { cells: 6, w: 4 }, true, false);
    let fdtd = sim::fdtd(2).end_criteria(ems::EndCriteria::energy_decay_db(-80.0));
    let mut s = ems::Simulation::new(&m, &fdtd).unwrap();
    let (r, t) = sim::rt(&mut s, pol);
    let mut worst = 0.0f64;
    for (q, &f) in freqs().iter().enumerate() {
        eprintln!(
            "{:6.2} GHz  |R| {:.4}  |T| {:.4}  |R|²+|T|² {:.5}",
            f / 1e9,
            r[q],
            t[q],
            r[q] * r[q] + t[q] * t[q]
        );
        worst = worst.max((r[q] * r[q] + t[q] * t[q] - 1.0).abs());
    }
    let res = s.results();
    eprintln!("FSS: | |R|²+|T|² − 1 | ≤ {worst:.1e}");
    assert!(worst < 2e-3, "{worst}");
    // A real scatterer: neither transparent nor a full mirror.
    assert!(r.iter().any(|&x| x > 0.2) && t.iter().any(|&x| x > 0.2));
    assert!(res.is_bloch());
}

/// Bloch runs through `Simulation` are deterministic (1 vs 3 threads),
/// and complex probe results carry their imaginary part.
#[test]
fn simulation_determinism() {
    let _t = Timer::start("simulation_determinism");
    let (m, _) = sim::model(&sim::Scatterer::Slab, true, false);
    let run = |n| {
        let mut s = ems::Simulation::new(&m, &sim::fdtd(n).max_timesteps(800)).unwrap();
        s.run().unwrap()
    };
    let (a, b) = (run(1), run(3));
    let pa = a.probe("refl").unwrap();
    assert!(pa.imag().is_some());
    assert_eq!(pa, b.probe("refl").unwrap());
    assert_eq!(a.probe("trans").unwrap(), b.probe("trans").unwrap());
    assert_eq!(a.incident("pw"), b.incident("pw"));
}

/// Settings checks and warnings of the Bloch mode.
#[test]
fn simulation_bloch_settings() {
    let (m, _) = sim::model(&sim::Scatterer::Nothing, true, false);
    // A Bloch wavevector along a non-periodic axis.
    let bad = sim::fdtd(1).bloch_wavevector([0.0, 0.0, 10.0]);
    assert!(bad.validate().is_err());
    assert!(ems::Simulation::new(&m, &bad).is_err());
    // No openEMS representation.
    assert!(sim::fdtd(1).to_csx_checked().is_err());
    // An explicit wavevector overrides the direction (warning).
    let s = ems::Simulation::new(&m, &sim::fdtd(1).bloch_wavevector([50.0, 0.0, 0.0])).unwrap();
    assert_eq!(s.bloch_wavevector(), Some([50.0, 0.0, 0.0]));
    assert!(
        s.warnings()
            .iter()
            .any(|w| w.contains("explicit Bloch wavevector")),
        "{:?}",
        s.warnings()
    );
    // NF2FF is not available in a Bloch run.
    let mut s = s;
    let b = ems::nf2ff::Nf2ffBox::new([0.0, 0.0, 60.0], [2.0, 1.0, 140.0], vec![10e9]);
    assert!(s.add_nf2ff(&b).is_err());
    // Without periodic faces an oblique plane wave stays a plain TF/SF source.
    let f = sim::fdtd(1).boundaries(
        ems::Boundaries::all(ems::Boundary::Pmc)
            .with(Face::ZMin, ems::Boundary::pml(8))
            .with(Face::ZMax, ems::Boundary::pml(8)),
    );
    let s = ems::Simulation::new(&m, &f).unwrap();
    assert!(!s.is_bloch());
}
