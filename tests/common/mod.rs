//! Shared helpers of the analytic validation suite (`tests/validation_*.rs`):
//! pulses, a quasi-1D TEM line runner, the discrete (Yee) and continuum
//! reference formulas, and small complex-number utilities.
//!
//! Every test file uses a different subset, hence the `dead_code` allow.
#![allow(dead_code)]

use std::f64::consts::PI;

use ems::Complex64;
use ems::constants::{C0, EPS0, MU0};
use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::{
    CellMaterials, EdgeSource, Engine, FaceCondition, FaceConditions, Grid, Material, Probe,
    TimeStep,
};
use ems::post::dft;
use ems::{Axis, Cpml, Face};

/// Sine-modulated Gaussian centred at `f0` with 1/e spectral half-width
/// `bw` (spectrum ∝ `exp(−((f − f0)/bw)²)` near `f0`), delayed by four
/// envelope widths. Odd about its centre, so it has exactly zero DC
/// content: a soft source driven by it leaves no static charge behind.
pub fn sine_pulse(f0: f64, bw: f64) -> impl Fn(f64) -> f64 + Send + 'static {
    let tau = 1.0 / (PI * bw);
    let t0 = 4.0 * tau;
    move |t: f64| {
        let x = (t - t0) / tau;
        (-x * x).exp() * (2.0 * PI * f0 * (t - t0)).sin()
    }
}

/// Duration of [`sine_pulse`] (`8·τ`) \[s\].
pub fn sine_pulse_duration(bw: f64) -> f64 {
    8.0 / (PI * bw)
}

/// Derivative-of-Gaussian pulse of width `tau`, delayed by `4·tau`
/// (zero DC content).
pub fn dgauss(tau: f64) -> impl Fn(f64) -> f64 + Send + 'static {
    let t0 = 4.0 * tau;
    move |t: f64| {
        let x = (t - t0) / tau;
        -2.0 * x * (-x * x).exp()
    }
}

/// Amplitude ratio in dB.
pub fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

/// `n` equally spaced frequencies from `f0` to `f1` inclusive.
pub fn linspace(f0: f64, f1: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|q| f0 + (f1 - f0) * q as f64 / (n - 1) as f64)
        .collect()
}

/// Peak absolute value.
pub fn peak(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |m, v| m.max(v.abs()))
}

/// Element-wise `a − b`.
pub fn diff(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// DFT of a probe series sampled at `t_n = (n + 1)·dt` (E probes from
/// step 0).
pub fn spectrum(x: &[f64], dt: f64, freqs: &[f64]) -> Vec<Complex64> {
    dft(dt, dt, x, freqs)
}

/// Principal square root.
pub fn csqrt(z: Complex64) -> Complex64 {
    let (r, t) = z.to_polar();
    Complex64::from_polar(r.sqrt(), 0.5 * t)
}

/// Principal natural logarithm.
pub fn cln(z: Complex64) -> Complex64 {
    let (r, t) = z.to_polar();
    Complex64::new(r.ln(), t)
}

/// Principal arcsine, `asin z = −j·ln(j·z + √(1 − z²))`.
pub fn casin(z: Complex64) -> Complex64 {
    let j = Complex64::I;
    -j * cln(j * z + csqrt(1.0 - z * z))
}

/// The Yee time-difference operator `Ω = (2/dt)·sin(ω·dt/2)` \[rad/s\]:
/// the leapfrog scheme sees `∂/∂t` as `jΩ`.
pub fn yee_omega(f: f64, dt: f64) -> f64 {
    2.0 / dt * (PI * f * dt).sin()
}

/// Numerical wavenumber of the 1-D Yee scheme with cell size `dz` and
/// time step `dt` in a medium (`eps_r`, conductivity `kappa`, μ = μ0),
/// as the complex phase advance per cell `K = k̃·dz` (wave `e^{−jK·k}`,
/// `Im K ≤ 0`). From the semi-implicit update (loss averaged over `n`,
/// `n+1`, which turns `σ` into `σ·cos(ω·dt/2)`):
///
/// ```text
/// 4·sin²(K/2) = Ω²·μ0·ε·dz² − jΩ·μ0·κ·cos(ω·dt/2)·dz²
/// ```
///
/// (Taflove & Hagness, ch. 4, numerical dispersion, with the lossy
/// update of ch. 3.)
pub fn yee_k(f: f64, dt: f64, dz: f64, eps_r: f64, kappa: f64) -> Complex64 {
    let om = yee_omega(f, dt);
    let c = (PI * f * dt).cos();
    let x = Complex64::new(
        om * om * MU0 * EPS0 * eps_r * dz * dz,
        -om * MU0 * kappa * c * dz * dz,
    );
    let k = casin(csqrt(x) * 0.5) * 2.0;
    // Forward branch (Re K ≥ 0; then Im K ≤ 0 for a lossy medium).
    if k.re < 0.0 { -k } else { k }
}

/// Continuum propagation constant `γ = α + jβ = jω·√(μ0·ε·(1 − jκ/(ωε)))`.
pub fn continuum_gamma(f: f64, eps_r: f64, kappa: f64) -> Complex64 {
    let w = 2.0 * PI * f;
    let eps = Complex64::new(EPS0 * eps_r, -kappa / w);
    Complex64::I * w * csqrt(eps * MU0)
}

/// Exact reflection coefficient of the discrete 1-D Yee line for a
/// layered profile, at frequency `f`.
///
/// `cells` lists the relative permittivity of consecutive z-cells of size
/// `dz`; the first two and the last two must belong to the (lossless,
/// uniform) incidence and exit media. Node `k` between cells `k−1` and `k`
/// carries the averaged `(ε_{k−1} + ε_k)/2` (the engine's edge average).
/// The frequency-domain node equation of the scheme,
/// `E_{k−1} − 2E_k + E_{k+1} = −Ω²μ0·ε_k·dz²·E_k`, is integrated from a
/// pure outgoing wave at the exit side back to the incidence side, where
/// the field is split into incident and reflected discrete waves.
pub fn discrete_reflection(cells: &[f64], dz: f64, dt: f64, f: f64) -> Complex64 {
    let n = cells.len();
    assert!(n >= 4);
    let om = yee_omega(f, dt);
    let x = |eps: f64| om * om * MU0 * EPS0 * eps * dz * dz;
    let kr = yee_k(f, dt, dz, cells[n - 1], 0.0);
    let kl = yee_k(f, dt, dz, cells[0], 0.0);
    // Nodes 0..=n; node k sits between cells k−1 and k.
    let mut e = vec![Complex64::ZERO; n + 1];
    let j = Complex64::I;
    e[n] = (-j * kr * n as f64).exp();
    e[n - 1] = (-j * kr * (n - 1) as f64).exp();
    for k in (1..n).rev() {
        let eps = 0.5 * (cells[k - 1] + cells[k]);
        e[k - 1] = e[k] * (2.0 - x(eps)) - e[k + 1];
    }
    // e_k = A·e^{−jK k} + B·e^{jK k} at k = 0, 1.
    let (p, q) = ((-j * kl).exp(), (j * kl).exp());
    // A + B = e0, A·p + B·q = e1.
    let b = (e[1] - e[0] * p) / (q - p);
    let a = e[0] - b;
    b / a
}

/// Continuum reflection of a lossless dielectric slab (`eps_r`, thickness
/// `d`) in vacuum at normal incidence:
/// `Γ = r·(1 − e^{−2jβd}) / (1 − r²·e^{−2jβd})`, `r = (1 − n)/(1 + n)`.
pub fn slab_reflection(f: f64, eps_r: f64, d: f64) -> Complex64 {
    let n = eps_r.sqrt();
    let r = (1.0 - n) / (1.0 + n);
    let beta = 2.0 * PI * f * n / C0;
    let p = Complex64::cis(-2.0 * beta * d);
    (1.0 - p) * r / (1.0 - p * (r * r))
}

/// Normal-incidence Fresnel coefficient vacuum → `eps_r` half-space.
pub fn fresnel(eps_r: f64) -> f64 {
    let n = eps_r.sqrt();
    (1.0 - n) / (1.0 + n)
}

/// A quasi-1-D TEM line along z: one cell of `width` across in x and y,
/// PEC on the faces normal to the polarisation `pol` (X or Y) and PMC on
/// the other transverse pair, so the E_pol / H plane wave is an exact
/// solution of the discrete scheme (the 1-D Yee line). The z faces are PEC
/// with a CPML on each end (see [`Line::pml`]).
#[derive(Debug, Clone)]
pub struct Line {
    /// Cell size along z \[m\].
    pub dz: f64,
    /// Number of z cells, CPML included.
    pub cells: usize,
    /// CPML on the z_min and z_max ends (`cells == 0`: plain PEC end).
    pub pml: [Cpml; 2],
    /// Transverse cell size \[m\].
    pub width: f64,
    /// Polarisation (E direction), `Axis::X` or `Axis::Y`.
    pub pol: Axis,
    /// Background material.
    pub background: Material,
    /// `(z-cell range, material)` overrides, applied in order.
    pub layers: Vec<(core::ops::Range<usize>, Material)>,
    /// z node of the soft source.
    pub src: usize,
    /// z nodes of the E probes.
    pub probes: Vec<usize>,
    /// z nodes carrying a zero-thickness PEC sheet across the line.
    pub sheets: Vec<usize>,
    /// Time step choice.
    pub timestep: TimeStep,
}

impl Line {
    /// Line with `cells` z cells of `dz`, vacuum, 8-cell CPML, x
    /// polarisation, 5 mm wide.
    pub fn new(dz: f64, cells: usize) -> Self {
        Line {
            dz,
            cells,
            pml: [Cpml::cells(8); 2],
            width: 5e-3,
            pol: Axis::X,
            background: Material::VACUUM,
            layers: Vec::new(),
            src: 0,
            probes: Vec::new(),
            sheets: Vec::new(),
            timestep: TimeStep::default(),
        }
    }

    /// The two (uniform) E edges of the line at z node `k`.
    pub fn edges(&self, k: usize) -> [[usize; 3]; 2] {
        match self.pol {
            Axis::X => [[0, 0, k], [0, 1, k]],
            Axis::Y => [[0, 0, k], [1, 0, k]],
            Axis::Z => panic!("polarisation must be transverse"),
        }
    }

    /// Builds the engine with source and probes; returns the engine and
    /// the probe handles.
    pub fn build(
        &self,
        pulse: impl Fn(f64) -> f64 + Send + 'static,
    ) -> (Engine, Vec<ems::engine::OperatorHandle<Probe>>) {
        let z: Vec<f64> = (0..=self.cells).map(|q| q as f64 * self.dz).collect();
        let g = Grid::new([vec![0.0, self.width], vec![0.0, self.width], z]).unwrap();
        let other = match self.pol {
            Axis::X => Axis::Y,
            _ => Axis::X,
        };
        let faces = FaceConditions::all(FaceCondition::Pec)
            .with(Face::new(other, false), FaceCondition::Pmc)
            .with(Face::new(other, true), FaceCondition::Pmc);
        let mut m = CellMaterials::new(&g, self.background);
        for (r, mat) in &self.layers {
            m.fill_box([0..1, 0..1, r.clone()], *mat);
        }
        for &k in &self.sheets {
            for p in self.edges(k) {
                m.set_pec_edge(self.pol, p);
            }
        }
        let mut e = Engine::builder(g)
            .materials(m)
            .faces(faces)
            .timestep(self.timestep)
            .threads(1)
            .build()
            .unwrap();
        let layers: Vec<_> = [Face::ZMin, Face::ZMax]
            .into_iter()
            .zip(self.pml)
            .filter(|(_, p)| p.cells > 0)
            .collect();
        if !layers.is_empty() {
            e.add_operator(CpmlBoundary::new(&e, layers).unwrap());
        }
        let mut s = EdgeSource::new(pulse);
        for p in self.edges(self.src) {
            s = s.edge(self.pol, p, 1.0);
        }
        e.add_operator(s);
        let hs = self
            .probes
            .iter()
            .map(|&k| e.add_operator(Probe::voltage(self.pol, self.edges(k)[0])))
            .collect();
        (e, hs)
    }

    /// Runs `steps` steps; returns the time step and one series per probe.
    pub fn run(
        &self,
        pulse: impl Fn(f64) -> f64 + Send + 'static,
        steps: u64,
    ) -> (f64, Vec<Vec<f64>>) {
        let (mut e, hs) = self.build(pulse);
        e.run(steps);
        let v = hs.iter().map(|h| e.operator(h).values().to_vec()).collect();
        (e.dt(), v)
    }
}

/// Wall-clock guard printing the runtime of a test on drop.
pub struct Timer(&'static str, std::time::Instant);

impl Timer {
    /// Starts timing test `name`.
    pub fn start(name: &'static str) -> Self {
        Timer(name, std::time::Instant::now())
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        eprintln!("[{}] {:.2} s", self.0, self.1.elapsed().as_secs_f64());
    }
}
