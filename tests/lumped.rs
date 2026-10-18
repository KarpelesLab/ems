//! Lumped elements and lumped ports in closed PEC boxes (no absorbing
//! boundaries needed): DC limits, input impedance of a small loop loaded
//! by lumped R / L / C / RLC, reciprocity and determinism.

use ems::Axis;
use ems::engine::{
    CellMaterials, Engine, Grid, LumpedBox, LumpedElement, LumpedPort, OperatorHandle, Topology,
};
use ems::post::{Complex64, PortResult, s_param};
use ems::waveform::Waveform;

const D: f64 = 1e-3;

/// Cube of `cells` cells of size `d`. With 20 mm the first cavity
/// resonance is at c/(2·20 mm)·√2 ≈ 10.6 GHz, far above the frequencies
/// analysed here.
fn cube(cells: usize, d: f64) -> Grid {
    Grid::uniform([cells; 3], [d; 3]).unwrap()
}

/// A rectangular loop in the y = `y` plane: a z-directed port column at
/// x = `port_x` and a z-directed load column at x = `load_x`, both spanning
/// z = `z0..=z1`, joined by zero-thickness PEC strips (single x-edge lines)
/// at z = `z0` and z = `z1`. Node indices.
struct Loop {
    port_x: usize,
    load_x: usize,
    y: usize,
    z0: usize,
    z1: usize,
}

/// 4 mm × 2 mm loop in the 20 mm cube (1 mm cells).
const LOOP: Loop = Loop {
    port_x: 8,
    load_x: 12,
    y: 10,
    z0: 9,
    z1: 11,
};

impl Loop {
    fn scaled(&self, s: usize) -> Loop {
        Loop {
            port_x: self.port_x * s,
            load_x: self.load_x * s,
            y: self.y * s,
            z0: self.z0 * s,
            z1: self.z1 * s,
        }
    }
    fn column(&self, x: usize) -> LumpedBox {
        LumpedBox::new(Axis::Z, [x, self.y, self.z0], [x, self.y, self.z1]).unwrap()
    }
    fn port_box(&self) -> LumpedBox {
        self.column(self.port_x)
    }
    fn load_box(&self) -> LumpedBox {
        self.column(self.load_x)
    }
    fn materials(&self, g: &Grid) -> CellMaterials {
        let mut m = CellMaterials::vacuum(g);
        for z in [self.z0, self.z1] {
            m.fill_pec_box([self.port_x..self.load_x + 1, self.y..self.y + 1, z..z + 1]);
        }
        m
    }
}

fn engine(g: Grid, m: CellMaterials, threads: usize) -> Engine {
    Engine::builder(g)
        .materials(m)
        .threads(threads)
        .build()
        .unwrap()
}

/// Port resistance (and reference impedance) of the loop tests.
const Z0: f64 = 50.0;

/// What terminates the load column of the loop.
#[derive(Clone, Copy)]
enum Load<'a> {
    /// PEC column: a short.
    Short,
    /// Nothing: the bare gap.
    Open,
    /// A lumped element over the load column.
    Element(&'a dyn Fn(LumpedBox) -> LumpedElement),
}

/// Loop geometry and resolution of a run: `scale` cells per millimetre.
#[derive(Clone, Copy)]
struct Setup {
    scale: usize,
    steps: u64,
    threads: usize,
}

/// 1 mm cells, ~7.6 ns: long after the 1.2 ns pulse and the L/R and RC
/// transients of the loop have died out.
const QUICK: Setup = Setup {
    scale: 1,
    steps: 4000,
    threads: 2,
};

/// Runs the loop with `load`, excited by a baseband Gaussian (`fc` = 3 GHz,
/// 1 V) through the 50 Ω port. Returns the analysis at `freqs` and the
/// recorded port voltage and current.
fn run_loop(load: Load<'_>, s: Setup, freqs: &[f64]) -> (PortResult, Vec<f64>, Vec<f64>) {
    let lp = LOOP.scaled(s.scale);
    let g = cube(20 * s.scale, D / s.scale as f64);
    let mut m = lp.materials(&g);
    if let Load::Short = load {
        m.fill_pec_box([lp.load_x..lp.load_x + 1, lp.y..lp.y + 1, lp.z0..lp.z1 + 1]);
    }
    let mut e = engine(g, m, s.threads);
    let h: OperatorHandle<LumpedPort> = LumpedPort::new(lp.port_box(), Z0)
        .excite_waveform(Waveform::gaussian(0.0, 3e9), 1.0)
        .add_to(&mut e)
        .unwrap();
    if let Load::Element(load) = load {
        load(lp.load_box()).add_to(&mut e).unwrap();
    }
    e.run(s.steps);
    let p = e.operator(&h);
    (
        p.analyze(freqs).unwrap(),
        p.voltage().to_vec(),
        p.current().to_vec(),
    )
}

fn omega(f: f64) -> f64 {
    2.0 * std::f64::consts::PI * f
}

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Mean of the last `n` samples.
fn tail_mean(x: &[f64], n: usize) -> f64 {
    x[x.len() - n..].iter().sum::<f64>() / n as f64
}

/// Smooth 0 → 1 ramp over `tr` seconds (raised cosine), then constant.
fn ramp(tr: f64) -> impl Fn(f64) -> f64 + Send + 'static {
    move |t| {
        if t >= tr {
            1.0
        } else if t <= 0.0 {
            0.0
        } else {
            0.5 * (1.0 - (std::f64::consts::PI * t / tr).cos())
        }
    }
}

/// (a) DC limits of a port with a ramped 1 V source and R = 50 Ω:
/// `V → V_s·Z_L/(R + Z_L)`, `I → V_s/(R + Z_L)` for an open gap
/// (`Z_L = ∞`), a wall-to-wall column (shorted through the PEC box,
/// `Z_L = 0`), and the strip loop with a 150 Ω lumped load.
#[test]
fn port_dc_limits() {
    let tr = 0.5e-9;
    let r = 50.0;
    let settle = |e: &mut Engine, h: &OperatorHandle<LumpedPort>| {
        e.run(3000);
        let p = e.operator(h);
        (tail_mean(p.voltage(), 200), tail_mean(p.current(), 200))
    };
    // Open: a floating two-edge gap in the middle of a 12 mm box.
    let g = cube(12, D);
    let m = CellMaterials::vacuum(&g);
    let mut e = engine(g, m, 2);
    let b = LumpedBox::new(Axis::Z, [6, 6, 5], [6, 6, 7]).unwrap();
    let h = LumpedPort::new(b, r)
        .excite(ramp(tr))
        .add_to(&mut e)
        .unwrap();
    let (v, i) = settle(&mut e, &h);
    eprintln!("open: V = {v:.6}, I = {i:.3e}");
    assert!((v - 1.0).abs() < 1e-4, "open-circuit V = {v}");
    assert!(i.abs() < 1e-6, "open-circuit I = {i}");

    // Short: a column from the z = 0 wall to the z = 4 wall of a flat box.
    let g = Grid::uniform([12, 12, 4], [D; 3]).unwrap();
    let m = CellMaterials::vacuum(&g);
    let mut e = engine(g, m, 2);
    let b = LumpedBox::new(Axis::Z, [6, 6, 0], [6, 6, 4]).unwrap();
    let h = LumpedPort::new(b, r)
        .excite(ramp(tr))
        .add_to(&mut e)
        .unwrap();
    let (v, i) = settle(&mut e, &h);
    eprintln!("short: V = {v:.3e}, I = {i:.6e}");
    assert!(v.abs() < 1e-4, "short-circuit V = {v}");
    assert!((i * r - 1.0).abs() < 1e-4, "short-circuit I = {i}");

    // Loop with a 150 Ω load: V → 0.75 V, I → 5 mA.
    let g = cube(20, D);
    let m = LOOP.materials(&g);
    let mut e = engine(g, m, 2);
    let h = LumpedPort::new(LOOP.port_box(), r)
        .excite(ramp(tr))
        .add_to(&mut e)
        .unwrap();
    LumpedElement::resistor(LOOP.load_box(), 150.0)
        .add_to(&mut e)
        .unwrap();
    let (v, i) = settle(&mut e, &h);
    eprintln!("150 ohm: V = {v:.6}, I = {i:.6e}");
    assert!((v - 0.75).abs() < 1e-4, "loaded V = {v}");
    assert!((i - 5e-3).abs() < 1e-6, "loaded I = {i}");
}

/// Analysis frequencies, all well below the first cavity resonance.
const FREQS: [f64; 6] = [50e6, 100e6, 200e6, 500e6, 1e9, 2e9];

/// (b) The loop terminated in a lumped 100 Ω resistor. Below ~500 MHz the
/// input impedance is `R_L` plus the series inductance of the 4 × 2 mm loop
/// (~3.9 nH: `Im Z/ω` is constant, as for a pure inductance), and S11 is
/// `(R_L − Z0)/(R_L + Z0) = 1/3`. The small rise of `Re Z` with frequency
/// is the ~70 fF strip-to-strip capacitance in parallel with the load
/// (`Re Z ≈ R_L·(1 + (ωR_L C)²)` to first order, +1 % at 1 GHz).
#[test]
fn loop_resistor_impedance() {
    let rl = 100.0;
    let (r, _, _) = run_loop(
        Load::Element(&|b| LumpedElement::resistor(b, rl)),
        QUICK,
        &FREQS,
    );
    let z = r.z_in();
    let s = r.s11();
    for (k, &f) in FREQS.iter().enumerate() {
        let l_eff = z[k].im / omega(f);
        eprintln!(
            "f {f:.2e}: Z_in {}  L_eff {l_eff:.4e} H  S11 {}",
            z[k], s[k]
        );
        assert!(l_eff > 3e-9 && l_eff < 5e-9, "loop inductance {l_eff}");
        if f <= 500e6 {
            assert!((z[k].re / rl - 1.0).abs() < 0.01, "Re Z_in {}", z[k].re);
        }
        if f <= 1e9 {
            assert!((z[k].re / rl - 1.0).abs() < 0.02, "Re Z_in {}", z[k].re);
        }
        // S11 is exactly the reflection of Z_in.
        let want = (z[k] - Z0) / (z[k] + Z0);
        assert!((s[k] - want).norm() < 1e-9);
    }
    let s_dc = (rl - Z0) / (rl + Z0);
    assert!((s[0] - c(s_dc, 0.0)).norm() < 0.01, "S11 {}", s[0]);
    // Power balance: incident = reflected + accepted.
    let (pi, pr, pa) = (r.incident_power(), r.reflected_power(), r.accepted_power());
    for k in 0..FREQS.len() {
        assert!((pi[k] - pr[k] - pa[k]).abs() < 1e-9 * pi[k]);
        assert!(pa[k] > 0.0);
    }
}

/// Short–open–load calibration of the loop as a one-port error network:
/// for any linear passive network between the port and the load column,
/// `Z_in = (a·Z_L + b)/(c·Z_L + 1)` (the bilinear one-port model behind
/// the three-term VNA error model, e.g. Pozar, *Microwave Engineering*, on
/// network analyser calibration). The three standards (PEC short, bare
/// gap, lumped 100 Ω) fix `a, b, c` at each frequency; any other load is
/// then de-embedded as `Z_L = (b − Z_in)/(c·Z_in − a)`. This isolates the
/// lumped-element models from the loop parasitics.
struct Calibration {
    a: Vec<Complex64>,
    b: Vec<Complex64>,
    c: Vec<Complex64>,
}

const R_STD: f64 = 100.0;

impl Calibration {
    fn measure(s: Setup) -> Self {
        let zs = run_loop(Load::Short, s, &FREQS).0.z_in();
        let zo = run_loop(Load::Open, s, &FREQS).0.z_in();
        let zr = run_loop(
            Load::Element(&|b| LumpedElement::resistor(b, R_STD)),
            s,
            &FREQS,
        )
        .0
        .z_in();
        let mut cal = Calibration {
            a: Vec::new(),
            b: zs.clone(),
            c: Vec::new(),
        };
        for k in 0..FREQS.len() {
            let cc = (zs[k] - zr[k]) / ((zr[k] - zo[k]) * R_STD);
            cal.c.push(cc);
            cal.a.push(zo[k] * cc);
        }
        cal
    }

    fn deembed(&self, z_in: &[Complex64]) -> Vec<Complex64> {
        (0..FREQS.len())
            .map(|k| (self.b[k] - z_in[k]) / (self.c[k] * z_in[k] - self.a[k]))
            .collect()
    }
}

/// De-embeds the lumped element built by `load` and compares it with the
/// nominal impedance `want(ω)`; returns the worst relative error.
fn check_element(
    cal: &Calibration,
    s: Setup,
    name: &str,
    load: &dyn Fn(LumpedBox) -> LumpedElement,
    want: &dyn Fn(f64) -> Complex64,
    tol: f64,
) -> f64 {
    let (r, _, _) = run_loop(Load::Element(load), s, &FREQS);
    let zl = cal.deembed(&r.z_in());
    let mut worst = 0.0f64;
    for (k, &f) in FREQS.iter().enumerate() {
        let w = want(omega(f));
        let err = (zl[k] - w).norm() / w.norm();
        eprintln!("{name}: f {f:.2e}: Z_L {}  want {w}  err {err:.2e}", zl[k]);
        worst = worst.max(err);
    }
    assert!(worst < tol, "{name}: worst relative error {worst:.3e}");
    worst
}

/// Nominal series impedance `R + jωL + 1/(jωC)` (`cap = 0`: no capacitor).
fn z_series(r: f64, l: f64, cap: f64) -> impl Fn(f64) -> Complex64 {
    move |w| {
        let xc = if cap > 0.0 { -1.0 / (w * cap) } else { 0.0 };
        c(r, w * l + xc)
    }
}

/// Nominal parallel impedance `1/(1/R + 1/(jωL) + jωC)` (0 = absent).
fn z_parallel(r: f64, l: f64, cap: f64) -> impl Fn(f64) -> Complex64 {
    move |w| {
        let mut y = c(0.0, w * cap);
        if r > 0.0 {
            y += 1.0 / r;
        }
        if l > 0.0 {
            y += c(0.0, -1.0 / (w * l));
        }
        1.0 / y
    }
}

/// (c) Lumped R, C (coefficient folds), L (branch state), series RL,
/// series RLC and parallel RLC on the load column, de-embedded through the
/// short–open–load calibration, match their nominal impedances.
fn lumped_elements_deembedded(s: Setup, tol: f64) {
    let cal = Calibration::measure(s);
    type Case<'a> = (
        &'a str,
        &'a dyn Fn(LumpedBox) -> LumpedElement,
        Box<dyn Fn(f64) -> Complex64>,
    );
    let cases: [Case<'_>; 6] = [
        (
            "R 30",
            &|b| LumpedElement::resistor(b, 30.0),
            Box::new(z_series(30.0, 0.0, 0.0)),
        ),
        (
            "C 1p",
            &|b| LumpedElement::new(b).c(1e-12),
            Box::new(z_parallel(0.0, 0.0, 1e-12)),
        ),
        (
            "L 10n",
            &|b| LumpedElement::new(b).l(10e-9),
            Box::new(z_parallel(0.0, 10e-9, 0.0)),
        ),
        (
            "series RL",
            &|b| {
                LumpedElement::new(b)
                    .r(40.0)
                    .l(3e-9)
                    .topology(Topology::Series)
            },
            Box::new(z_series(40.0, 3e-9, 0.0)),
        ),
        (
            "series RLC",
            &|b| {
                LumpedElement::new(b)
                    .r(20.0)
                    .l(5e-9)
                    .c(2e-12)
                    .topology(Topology::Series)
            },
            Box::new(z_series(20.0, 5e-9, 2e-12)),
        ),
        (
            "parallel RLC",
            &|b| LumpedElement::new(b).r(200.0).l(8e-9).c(0.5e-12),
            Box::new(z_parallel(200.0, 8e-9, 0.5e-12)),
        ),
    ];
    for (name, load, want) in cases {
        check_element(&cal, s, name, load, &*want, tol);
    }
}

#[test]
fn lumped_elements_deembedded_quick() {
    lumped_elements_deembedded(QUICK, 1e-3);
}

#[test]
#[ignore = "heavy: 0.5 mm mesh (64k cells, 8000 steps per run); run with --ignored in release"]
fn lumped_elements_deembedded_fine() {
    lumped_elements_deembedded(
        Setup {
            scale: 2,
            steps: 8000,
            threads: 4,
        },
        2e-3,
    );
}

/// (c) The loop inductance resonates with a lumped 1 pF capacitor near
/// `1/(2π√(L·C))`, with `L` from the shorted loop. The resonance sits a few
/// percent lower because the strip capacitance adds to `C`.
#[test]
fn loop_lc_resonance() {
    let l = run_loop(Load::Short, QUICK, &[100e6]).0.z_in()[0].im / omega(100e6);
    let cap = 1e-12;
    let f_lc = 1.0 / (2.0 * std::f64::consts::PI * (l * cap).sqrt());
    let freqs: Vec<f64> = (0..=200).map(|q| 1.5e9 + q as f64 * 7.5e6).collect();
    let (r, _, _) = run_loop(
        Load::Element(&|b| LumpedElement::new(b).c(cap)),
        QUICK,
        &freqs,
    );
    let x: Vec<f64> = r.z_in().iter().map(|z| z.im).collect();
    let q = x
        .windows(2)
        .position(|w| w[0] < 0.0 && w[1] >= 0.0)
        .expect("series resonance in the scan");
    let f_r = freqs[q] - x[q] * (freqs[q + 1] - freqs[q]) / (x[q + 1] - x[q]);
    eprintln!("L = {l:.4e} H, f_LC = {f_lc:.4e}, resonance {f_r:.4e}");
    assert!(f_r < f_lc && f_r > 0.95 * f_lc, "{f_r} vs {f_lc}");
    // At resonance the input is the (lossless) load: Re Z_in ≈ 0.
    let zr = r.z_in()[q].re.abs();
    assert!(zr < 0.5, "Re Z_in at resonance {zr}");
}

/// Two-port: ports 1 (50 Ω) and 2 (75 Ω) at the ends of a 10 mm strip
/// pair, a series-RC shunt in the middle and a lumped capacitor near port 1
/// (so the network is not symmetric). Exciting port `excite`, returns
/// `S_oi = b_o/a_i` (other port passive), `S_ii` and port 2's voltage.
fn two_port(excite: usize, threads: usize) -> (Vec<Complex64>, Vec<Complex64>, Vec<f64>) {
    let g = cube(20, D);
    let (y, z0, z1) = (10, 9, 11);
    let mut m = CellMaterials::vacuum(&g);
    for z in [z0, z1] {
        m.fill_pec_box([5..16, y..y + 1, z..z + 1]);
    }
    let col = |x| LumpedBox::new(Axis::Z, [x, y, z0], [x, y, z1]).unwrap();
    let mut e = engine(g, m, threads);
    let w = Waveform::gaussian(0.0, 3e9);
    let mk = |x, r, on: bool| {
        let p = LumpedPort::new(col(x), r);
        if on {
            p.excite_waveform(w.clone(), 1.0)
        } else {
            p
        }
    };
    let p1 = mk(5, 50.0, excite == 1).add_to(&mut e).unwrap();
    let p2 = mk(15, 75.0, excite == 2).add_to(&mut e).unwrap();
    LumpedElement::new(col(7))
        .c(0.5e-12)
        .add_to(&mut e)
        .unwrap();
    LumpedElement::new(col(10))
        .r(30.0)
        .c(2e-12)
        .topology(Topology::Series)
        .add_to(&mut e)
        .unwrap();
    e.run(5000);
    let r1 = e.operator(&p1).analyze(&FREQS).unwrap();
    let r2 = e.operator(&p2).analyze(&FREQS).unwrap();
    let (inc, out) = if excite == 1 { (&r1, &r2) } else { (&r2, &r1) };
    (
        s_param(out, inc),
        inc.s11(),
        e.operator(&p2).voltage().to_vec(),
    )
}

/// (d) Reciprocity of a passive two-port: S21 = S12 (the reference
/// impedances differ, but power-wave S-parameters with real references are
/// symmetric for any reciprocal network; Kurokawa 1965). Also passivity.
#[test]
fn two_port_reciprocity() {
    let (s21, s11, _) = two_port(1, 2);
    let (s12, s22, _) = two_port(2, 2);
    for k in 0..FREQS.len() {
        let d = (s21[k] - s12[k]).norm();
        eprintln!(
            "f {:.2e}: S21 {} S12 {} |diff| {d:.2e}; |S11| {:.4} |S22| {:.4}",
            FREQS[k],
            s21[k],
            s12[k],
            s11[k].norm(),
            s22[k].norm()
        );
        assert!(s21[k].norm() > 0.05);
        assert!(d < 2e-3 * s21[k].norm(), "S21 != S12 at {}", FREQS[k]);
        assert!(s11[k].norm_sqr() + s21[k].norm_sqr() <= 1.0 + 1e-6);
        assert!(s22[k].norm_sqr() + s12[k].norm_sqr() <= 1.0 + 1e-6);
    }
}

/// (e) Lumped elements and ports are bit-identical for any thread count.
#[test]
fn lumped_deterministic_across_threads() {
    let (a, _, va) = two_port(1, 1);
    for t in [3, 5] {
        let (b, _, vb) = two_port(1, t);
        assert_eq!(a, b);
        assert_eq!(
            va.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vb.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    let load = |b| {
        LumpedElement::new(b)
            .r(20.0)
            .l(5e-9)
            .topology(Topology::Series)
    };
    let (_, v1, i1) = run_loop(
        Load::Element(&load),
        Setup {
            threads: 1,
            ..QUICK
        },
        &FREQS,
    );
    let (_, v4, i4) = run_loop(
        Load::Element(&load),
        Setup {
            threads: 4,
            ..QUICK
        },
        &FREQS,
    );
    assert_eq!(v1, v4);
    assert_eq!(i1, i4);
}

/// A lossless closed PEC cavity loaded by lossless lumped elements (a
/// parallel L ∥ C and a series L + C), started from a charged element:
/// the total energy `W_fields + W_branch` ([`Engine::energy`]) is
/// conserved, the field part alone is not (energy sloshes into the
/// branches), and the in-step energy matches the read-only one.
#[test]
fn lc_loaded_cavity_conserves_total_energy() {
    for threads in [1, 3] {
        let g = cube(12, D);
        let mut e = Engine::builder(g).threads(threads).build().unwrap();
        let par = LumpedBox::new(Axis::Z, [4, 6, 4], [4, 6, 8]).unwrap();
        let ser = LumpedBox::new(Axis::X, [6, 3, 7], [9, 3, 7]).unwrap();
        let hp = LumpedElement::new(par)
            .l(2e-9)
            .c(0.2e-12)
            .add_to(&mut e)
            .unwrap();
        let hs = LumpedElement::new(ser)
            .l(3e-9)
            .c(0.1e-12)
            .topology(Topology::Series)
            .add_to(&mut e)
            .unwrap();
        {
            let mut f = e.fields_mut();
            for p in par.edges() {
                f.u.set(Axis::Z, p, 0.25);
            }
        }
        e.run(5);
        let w0 = e.energy();
        let b0 = e.operator_energy();
        assert!(w0 > 0.0);
        let (mut dev, mut fdev, mut bmax) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..400 {
            let w_ro = e.energy();
            let w_ops = e.operator_energy();
            let parts = e.operator(&hp).branch_energy() + e.operator(&hs).branch_energy();
            assert_eq!(w_ops.to_bits(), parts.to_bits());
            let w = e.step_with_energy();
            assert_eq!(w.to_bits(), w_ro.to_bits());
            dev = dev.max((w / w0 - 1.0).abs());
            fdev = fdev.max(((w - w_ops) / (w0 - b0) - 1.0).abs());
            bmax = bmax.max(w_ops / w);
            e.run(9);
        }
        eprintln!(
            "{threads} threads: W0 = {w0:.4e} J, max |W/W0 − 1| = {dev:.2e}, \
             fields only {fdev:.2e}, max branch share {bmax:.3}"
        );
        assert!(dev < 1e-4, "total energy drifted by {dev:e}");
        assert!(bmax > 0.05, "branches hold too little energy: {bmax}");
        assert!(fdev > 100.0 * dev, "fields-only energy {fdev:e} vs {dev:e}");
    }
}
