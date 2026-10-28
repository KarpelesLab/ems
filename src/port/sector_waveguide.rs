//! Sectoral waveguide port ([`SectorWaveguidePort`]): a guide whose cross
//! section is a circular sector, on a cylindrical sector grid.

use core::f64::consts::PI;

use csx::primitives::BoxShape;
use csx::properties::{Excitation, ProbeBox, ProbeType, WeightFn};
use csx::{ContinuousStructure, CoordSystem, Primitive, Property, PropertyId, Scalar};

use super::bessel;
use super::circ_waveguide::CircWaveguidePortProperties;
use crate::constants::{C0, EPS0, MU0};
use crate::post::{Complex64, PortResult};
use crate::sim::Results;
use crate::{Error, Result};

/// A mode of a sectoral waveguide of opening `φ_s`: `m` half periods
/// across the opening (azimuthal order `ν = mπ/φ_s`), `n`-th radial root
/// (`n ≥ 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SectorMode {
    /// TE_mn (`H_z ∝ J_ν(k_c ρ)·cos(ν(α − α_0))`, `k_c·b = j'_{ν,n}`).
    Te(u32, u32),
    /// TM_mn (`E_z ∝ J_ν(k_c ρ)·sin(ν(α − α_0))`, `k_c·b = j_{ν,n}`,
    /// `m ≥ 1`).
    Tm(u32, u32),
}

impl SectorMode {
    /// `(m, n)`.
    pub fn indices(self) -> (u32, u32) {
        match self {
            SectorMode::Te(m, n) | SectorMode::Tm(m, n) => (m, n),
        }
    }

    /// Whether the mode is TE.
    pub fn is_te(self) -> bool {
        matches!(self, SectorMode::Te(..))
    }
}

impl core::fmt::Display for SectorMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (m, n) = self.indices();
        let t = if self.is_te() { "TE" } else { "TM" };
        write!(f, "{t}{m}{n}")
    }
}

/// A port for one TE_mn or TM_mn mode of a **sectoral waveguide** (a
/// guide along z whose cross section is the circular sector `ρ ≤ b`,
/// `α_0 ≤ α ≤ α_0 + φ_s`, with PEC walls), written into a csx model like
/// [`CircWaveguidePort`](super::CircWaveguidePort): a mode-shaped E-field
/// `Excitation` and a pair of mode-matching probes.
///
/// ```
/// use csx::{ContinuousStructure, CoordSystem};
/// use ems::port::{SectorMode, SectorWaveguidePort};
/// use std::f64::consts::PI;
///
/// let mut model = ContinuousStructure::new();
/// model.coord_system = CoordSystem::Cylindrical;
/// model.grid.coord_system = CoordSystem::Cylindrical;
/// model.grid.delta_unit = 1e-3;
/// model.grid.set_lines(0, (0..=40).map(|i| f64::from(i) * 0.25))?;
/// // A 120° sector.
/// model.grid.set_lines(1, (0..=24).map(|j| f64::from(j) * PI / 36.0))?;
/// model.grid.set_lines(2, (0..=80).map(f64::from))?;
/// let port = SectorWaveguidePort::new(1, 10.0, 14.0, 10.0, SectorMode::Te(1, 1))
///     .excite(1.0)
///     .add_to(&mut model)?;
/// // ν = 3/2: k_c·b = j'_{3/2,1} = 2.4605.
/// assert!((port.order().unwrap() - 1.5).abs() < 1e-12);
/// assert!((port.cutoff_wavenumber().unwrap() * 0.01 - 2.460_5).abs() < 1e-4);
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Geometry
///
/// * The grid is a cylindrical **sector** with the axis (first `ρ` line
///   0): its α span `φ_s` (`< 2π`) is the opening of the guide, and both
///   α faces must be PEC walls (the boundary settings are not checked).
///   `radius` (drawing units) is the outer radius `b`: the `ρ_max` PEC
///   face (the last `ρ` line) or a metal wall in the model.
/// * The guide runs along z: the excitation plane is `z = start_z`, the
///   reference plane `z = stop_z`, and the port looks into the structure
///   in the direction `start_z → stop_z`.
/// * As for the other waveguide ports the excitation is a soft source
///   that radiates both ways (end the guide behind it in a PML), and the
///   reference plane should be at least one cell downstream of it.
///
/// # Mode
///
/// With `ν = mπ/φ_s`, `ψ = J_ν(k_c ρ)·cos(ν(α − α_0))` (TE) or
/// `J_ν(k_c ρ)·sin(ν(α − α_0))` (TM), the transverse mode function is
/// (Marcuvitz, *Waveguide Handbook*, §2.5; Harrington, *Time-Harmonic
/// Electromagnetic Fields*, §5-8, for the wedge and sector guides)
///
/// ```text
/// TE:  e = −N·∇ψ × ẑ:  e_ρ = N·(ν/ρ)·J_ν·sin(ν(α−α_0)),  e_α = N·k_c·J_ν'·cos(ν(α−α_0))
/// TM:  e =  N·∇ψ:      e_ρ = N·k_c·J_ν'·sin(ν(α−α_0)),   e_α = N·(ν/ρ)·J_ν·cos(ν(α−α_0))
/// h = ẑ × e (propagation along +z; −ẑ × e towards −z)
/// ```
///
/// so that the tangential E vanishes on both α walls (`e_ρ`, `E_z ∝
/// sin(ν(α − α_0))`) and on `ρ = b` (`J_ν'(k_c b) = 0` for TE,
/// `J_ν(k_c b) = 0` for TM). TE_0n is the circular TE_0n (`e_α = −N·k_c·J_1`).
/// `N` normalises `∫|e|² dA = 1` over the sector, with the Lommel
/// integrals `∫₀^b J_ν(k_c ρ)²ρ dρ = b²/2·(1 − ν²/j'²)·J_ν(j')²` (TE) and
/// `b²/2·J_ν'(j)²` (TM), which hold for real `ν` (Abramowitz & Stegun
/// 11.4.5 and its derivation), times `φ_s/2` (`ν > 0`) or `φ_s` (TE_0n).
/// `J_ν` of non-integer order is a power series (A&S 9.1.10), written
/// into the csx expressions of the mode functions; orders `0 < ν < 1`
/// (openings wider than `mπ`: the field is singular at the axis, `∝
/// ρ^{ν−1}`) are refused. In a medium `ε_r`, `μ_r`
/// ([`filled`](Self::filled)) `β = √(k² − k_c²)` and the wave impedance
/// (the reference impedance of [`analyze`](Self::analyze)) is
/// `Z_TE = ωμ/β` or `Z_TM = β/(ωε)`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SectorWaveguidePort {
    number: u32,
    start_z: f64,
    stop_z: f64,
    radius: f64,
    mode: SectorMode,
    eps_r: f64,
    mu_r: f64,
    amplitude: Option<f64>,
    delay: f64,
    priority: i32,
    /// `(ν, k_c [1/m])`, set by `add_to`.
    order: Option<(f64, f64)>,
    props: Option<CircWaveguidePortProperties>,
}

impl SectorWaveguidePort {
    /// Port `number` for `mode`, excitation plane `z = start_z`, reference
    /// plane `z = stop_z`, outer radius `radius` (drawing units); see the
    /// [type docs](Self).
    pub fn new(number: u32, start_z: f64, stop_z: f64, radius: f64, mode: SectorMode) -> Self {
        Self {
            number,
            start_z,
            stop_z,
            radius,
            mode,
            eps_r: 1.0,
            mu_r: 1.0,
            amplitude: None,
            delay: 0.0,
            priority: 0,
            order: None,
            props: None,
        }
    }

    /// Excites the mode with field `amplitude·e` \[V/m, e in 1/m\] times
    /// the simulation waveform.
    pub fn excite(mut self, amplitude: f64) -> Self {
        self.amplitude = Some(amplitude);
        self
    }

    /// Delay of the excitation \[s\] (default 0).
    pub fn delay(mut self, delay: f64) -> Self {
        self.delay = delay;
        self
    }

    /// Priority of the port's primitives (default 0).
    pub fn priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// The guide is filled with a homogeneous lossless medium `ε_r`, `μ_r`
    /// (default vacuum), for the cutoff, `β` and the wave impedance only.
    pub fn filled(mut self, eps_r: f64, mu_r: f64) -> Self {
        self.eps_r = eps_r;
        self.mu_r = mu_r;
        self
    }

    /// Port number.
    pub fn number(&self) -> u32 {
        self.number
    }

    /// The mode.
    pub fn mode(&self) -> SectorMode {
        self.mode
    }

    /// Whether the port is excited.
    pub fn is_excited(&self) -> bool {
        self.amplitude.is_some()
    }

    /// The properties written by [`add_to`](Self::add_to) (`None` before).
    pub fn properties(&self) -> Option<&CircWaveguidePortProperties> {
        self.props.as_ref()
    }

    /// The azimuthal order `ν = mπ/φ_s` (after [`add_to`](Self::add_to)).
    pub fn order(&self) -> Option<f64> {
        self.order.map(|o| o.0)
    }

    /// Cutoff wavenumber `k_c` \[1/m\] (after [`add_to`](Self::add_to)).
    pub fn cutoff_wavenumber(&self) -> Option<f64> {
        self.order.map(|o| o.1)
    }

    fn velocity(&self) -> f64 {
        C0 / (self.eps_r * self.mu_r).sqrt()
    }

    /// Cutoff frequency \[Hz\] (after [`add_to`](Self::add_to)).
    pub fn cutoff_frequency(&self) -> Option<f64> {
        Some(self.cutoff_wavenumber()? * self.velocity() / (2.0 * PI))
    }

    /// Propagation constant `β(f)` \[rad/m\]; `None` before `add_to` and
    /// at or below the cutoff.
    pub fn beta(&self, f: f64) -> Option<f64> {
        let kc = self.cutoff_wavenumber()?;
        let k = 2.0 * PI * f / self.velocity();
        (k > kc).then(|| (k * k - kc * kc).sqrt())
    }

    /// Wave impedance \[Ω\]: `Z_TE = ωμ/β` or `Z_TM = β/(ωε)`; `None`
    /// where [`beta`](Self::beta) is.
    pub fn z_wave(&self, f: f64) -> Option<f64> {
        let beta = self.beta(f)?;
        let w = 2.0 * PI * f;
        Some(if self.mode.is_te() {
            w * MU0 * self.mu_r / beta
        } else {
            beta / (w * EPS0 * self.eps_r)
        })
    }

    /// Writes the port's properties into `model` and returns the port
    /// with their names and ids.
    ///
    /// # Errors
    /// [`Error::Unsupported`] on a grid that is not a cylindrical sector
    /// with the axis; [`Error::Invalid`] for `n = 0`, a TM mode with
    /// `m = 0`, an order `0 < ν < 1`, a non-positive radius, coinciding
    /// planes, non-finite or non-positive values, a port already added, or
    /// a property name already present in the model.
    pub fn add_to(mut self, model: &mut ContinuousStructure) -> Result<Self> {
        let num = self.number;
        let err = |m: String| Error::Invalid(format!("port {num}: {m}"));
        if self.props.is_some() {
            return Err(err("already added".into()));
        }
        let (m, n) = self.mode.indices();
        if n == 0 || (!self.mode.is_te() && m == 0) {
            return Err(err(format!("{} is not a sector-guide mode", self.mode)));
        }
        if ![self.start_z, self.stop_z, self.delay]
            .iter()
            .all(|v| v.is_finite())
            || self.amplitude.is_some_and(|a| !a.is_finite())
        {
            return Err(err("values must be finite".into()));
        }
        if self.start_z == self.stop_z {
            return Err(err("the excitation and reference planes coincide".into()));
        }
        if !(self.radius.is_finite() && self.radius > 0.0) {
            return Err(err(format!(
                "radius {} must be finite and > 0",
                self.radius
            )));
        }
        if !(self.eps_r.is_finite() && self.eps_r > 0.0 && self.mu_r.is_finite() && self.mu_r > 0.0)
        {
            return Err(err(format!(
                "filling ε_r = {}, μ_r = {} must be finite and > 0",
                self.eps_r, self.mu_r
            )));
        }
        let rho = model.grid.lines(0);
        let al = model.grid.lines(1);
        if model.grid.coord_system != CoordSystem::Cylindrical
            || rho.first() != Some(&0.0)
            || al.len() < 2
        {
            return Err(Error::Unsupported(format!(
                "port {num}: a sector guide needs a cylindrical grid with the axis"
            )));
        }
        let (a0, a1) = (al[0], al[al.len() - 1]);
        let span = a1 - a0;
        if span >= 2.0 * PI * (1.0 - 1e-9) {
            return Err(Error::Unsupported(format!(
                "port {num}: the grid closes the azimuth; use CircWaveguidePort"
            )));
        }
        let nu = f64::from(m) * PI / span;
        if nu > 0.0 && nu < 1.0 {
            return Err(err(format!(
                "{} of a {span} rad sector has the order ν = {nu} < 1 (a field singular at \
                 the axis)",
                self.mode
            )));
        }
        let names = [
            format!("port_excite_{num}"),
            format!("port_ut{num}"),
            format!("port_it{num}"),
        ];
        if let Some(dup) = names.iter().find(|s| model.property_by_name(s).is_some()) {
            return Err(Error::Invalid(format!(
                "the model already has a property named {dup:?}"
            )));
        }
        let unit = model.grid.delta_unit;
        let te = self.mode.is_te();
        let p = if te {
            bessel::jv_prime_zero(nu, n)
        } else {
            bessel::jv_zero(nu, n)
        };
        let b_m = self.radius * unit;
        let kc = p / b_m;
        self.order = Some((nu, kc));
        let sign = (self.stop_z - self.start_z).signum();
        let (e, h) = mode_functions(te, nu, p, kc, b_m, span, a0, unit, sign);

        // The cross section on a plane, out to the line enclosing ρ = b.
        let tol = 1e-9 * (1.0 + self.radius);
        let r_hi = rho
            .iter()
            .copied()
            .find(|&r| r >= self.radius - tol)
            .unwrap_or(self.radius);
        let plane = |z: f64| -> ([f64; 3], [f64; 3]) { ([0.0, a0, z], [r_hi, a1, z]) };
        let prim = |(s, t): ([f64; 3], [f64; 3])| {
            Primitive::new(BoxShape::new(s, t)).with_priority(self.priority)
        };
        let [excite, ut, it] = names;
        let excitation = match self.amplitude {
            Some(amp) => {
                let mut ex = Excitation::new([amp, amp, 0.0]).with_number(num);
                for (d, f) in e.iter().enumerate().take(2) {
                    ex.weight[d] = WeightFn::new(f.clone());
                }
                ex.delay = Scalar::new(self.delay);
                let id = model.add_property(Property::excitation(excite.clone(), ex));
                model.add_primitive(id, prim(plane(self.start_z)))?;
                Some((excite, id))
            }
            None => None,
        };
        let ref_plane = plane(self.stop_z);
        let mut probe = |name: &str, ty: ProbeType, f: &[String; 3]| -> Result<PropertyId> {
            let mut pb = ProbeBox::new(ty).with_norm_dir(csx::Axis::Z);
            pb.number = num;
            let mut p = Property::probe(name.to_owned(), pb);
            for (c, s) in ["X", "Y", "Z"].iter().zip(f) {
                p.set_attribute(format!("ModeFunction{c}"), s.clone());
            }
            p.set_attribute(crate::compile::MODE_SAMPLING_ATTR, "exact");
            let id = model.add_property(p);
            model.add_primitive(id, prim(ref_plane))?;
            Ok(id)
        };
        let v_id = probe(&ut, ProbeType::WaveguideVoltage, &e)?;
        let i_id = probe(&it, ProbeType::WaveguideCurrent, &h)?;
        self.props = Some(CircWaveguidePortProperties {
            excitation,
            voltage: (ut, v_id),
            current: (it, i_id),
        });
        Ok(self)
    }

    /// Frequency-domain analysis at `freqs` \[Hz\], as for
    /// [`CircWaveguidePort::analyze`](super::CircWaveguidePort::analyze):
    /// `V(f)`, `I(f)` from the mode probes, the wave impedance as the
    /// reference impedance.
    ///
    /// # Errors
    /// [`Error::Invalid`] if the port was not added, a probe is missing
    /// from `results`, or a frequency is at or below the cutoff.
    pub fn analyze(&self, results: &Results, freqs: &[f64]) -> Result<PortResult> {
        let p = self
            .props
            .as_ref()
            .ok_or_else(|| Error::Invalid(format!("port {} was not added", self.number)))?;
        let z: Vec<Complex64> = freqs
            .iter()
            .map(|&f| {
                self.z_wave(f)
                    .map(|z| Complex64::new(z, 0.0))
                    .ok_or_else(|| {
                        Error::Invalid(format!(
                            "port {}: {f} Hz is not above the {} cutoff {} Hz",
                            self.number,
                            self.mode,
                            self.cutoff_frequency().unwrap_or(f64::NAN)
                        ))
                    })
            })
            .collect::<Result<_>>()?;
        let get = |name: &str| {
            results
                .probe(name)
                .ok_or_else(|| Error::Invalid(format!("no probe {name:?} in the results")))
        };
        let v = get(&p.voltage.0)?.dft(freqs)?;
        let i = get(&p.current.0)?.dft(freqs)?;
        PortResult::new_per_frequency(freqs.to_vec(), v, i, z)
    }
}

/// `Σ_k c_k·Y^k` as a csx expression in Horner form (`Y` an expression).
fn horner_expr(c: &[f64], y: &str) -> String {
    let mut s = String::new();
    for (k, ck) in c.iter().enumerate() {
        if k > 0 {
            s.push_str(&format!("+{y}*("));
        }
        s.push_str(&format!("({ck:e})"));
    }
    s.push_str(&")".repeat(c.len().saturating_sub(1)));
    s
}

/// The csx expressions of `e` and `h` (`ρ, α, z` components; `rho` and `a`
/// the weight variables in drawing units and radians) of a sector mode:
/// TE if `te`, order `nu`, root `p`, `k_c = kc`, outer radius `b_m` \[m\],
/// opening `span` from `a0`, drawing unit `unit`, propagation sign `sign`
/// (see the [type docs](SectorWaveguidePort#mode)).
#[allow(clippy::too_many_arguments)]
fn mode_functions(
    te: bool,
    nu: f64,
    p: f64,
    kc: f64,
    b_m: f64,
    span: f64,
    a0: f64,
    unit: f64,
    sign: f64,
) -> ([String; 3], [String; 3]) {
    let ang = if nu == 0.0 { span } else { 0.5 * span };
    let rad = if te {
        0.5 * b_m * b_m * (1.0 - nu * nu / (p * p)) * bessel::jv(nu, p).powi(2)
    } else {
        0.5 * b_m * b_m * bessel::jv_prime(nu, p).powi(2)
    };
    let norm = 1.0 / (kc * (ang * rad).sqrt());
    let (e_rho, e_a) = if nu == 0.0 {
        // TE_0n: e_α = N·k_c·J_0' = −N·k_c·J_1.
        let kr = format!("{:e}*rho", kc * unit);
        ("0".to_owned(), format!("{:e}*jn(1,{kr})", -norm * kc))
    } else {
        // J_ν(k_c ρ)/ρ = (k_c/2)·P·S(Y) and J_ν'(k_c ρ) = P·D(Y), with
        // P = (k_c ρ/2)^{ν−1} and Y = (k_c ρ/2)² (ρ in metres).
        let c = bessel::jv_series(nu, p);
        let d: Vec<f64> = c
            .iter()
            .enumerate()
            .map(|(k, &ck)| ck * (k as f64 + 0.5 * nu))
            .collect();
        let half = format!("({:e}*rho)", 0.5 * kc * unit);
        let y = format!("({half}*{half})");
        let pw = format!("pow({half},{:e})", nu - 1.0);
        let jr = format!("{pw}*({})", horner_expr(&c, &y));
        let jp = format!("{pw}*({})", horner_expr(&d, &y));
        let arg = format!("{nu:e}*(a-({a0:e}))");
        let (s, co) = (format!("sin({arg})"), format!("cos({arg})"));
        if te {
            (
                format!("{:e}*{jr}*{s}", norm * nu * 0.5 * kc),
                format!("{:e}*{jp}*{co}", norm * kc),
            )
        } else {
            (
                format!("{:e}*{jp}*{s}", norm * kc),
                format!("{:e}*{jr}*{co}", norm * nu * 0.5 * kc),
            )
        }
    };
    let scaled = |k: f64, s: &str| -> String {
        if s == "0" {
            "0".into()
        } else {
            format!("{k:e}*({s})")
        }
    };
    // h = ±ẑ × e: h_ρ = −e_α, h_α = e_ρ.
    let h = [scaled(-sign, &e_a), scaled(sign, &e_rho), "0".into()];
    ([e_rho, e_a, "0".into()], h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use csx::expr::Expr;
    use csx::properties::WEIGHT_VARS;

    fn sector(span: f64) -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.coord_system = CoordSystem::Cylindrical;
        m.grid.coord_system = CoordSystem::Cylindrical;
        m.grid.delta_unit = 1e-3;
        m.grid
            .set_lines(0, (0..=40).map(|i| f64::from(i) * 0.25))
            .unwrap();
        m.grid
            .set_lines(1, (0..=24).map(|j| f64::from(j) * span / 24.0))
            .unwrap();
        m.grid.set_lines(2, (0..=40).map(f64::from)).unwrap();
        m
    }

    fn compile(src: &str) -> csx::expr::CompiledExpr {
        Expr::parse(src)
            .unwrap()
            .compile(&WEIGHT_VARS, |_| None)
            .unwrap()
    }

    fn attrs(m: &ContinuousStructure, id: PropertyId) -> [csx::expr::CompiledExpr; 3] {
        ["X", "Y", "Z"].map(|c| {
            compile(
                m.property(id)
                    .unwrap()
                    .attribute(&format!("ModeFunction{c}"))
                    .unwrap(),
            )
        })
    }

    /// The mode expressions: `∫|e|² dA = 1` over the sector, `h = ẑ × e`,
    /// tangential E zero on the walls, and the radial dependence of the
    /// Bessel functions.
    #[test]
    fn mode_functions() {
        let span = 2.0 * PI / 3.0;
        for mode in [
            SectorMode::Te(1, 1),
            SectorMode::Tm(1, 1),
            SectorMode::Te(2, 1),
            SectorMode::Te(0, 1),
            SectorMode::Tm(1, 2),
        ] {
            let m = &mut sector(span);
            let port = SectorWaveguidePort::new(1, 5.0, 8.0, 10.0, mode)
                .add_to(m)
                .unwrap();
            let (nu, kc) = port.order.unwrap();
            let pr = port.properties().unwrap();
            let e = attrs(m, pr.voltage.1);
            let h = attrs(m, pr.current.1);
            let at = |f: &csx::expr::CompiledExpr, r: f64, a: f64| {
                f.eval(&csx::properties::weight_vars(csx::Vec3::new(
                    r * a.cos(),
                    r * a.sin(),
                    0.0,
                )))
            };
            let (nr, na) = (200, 96);
            let mut sum = 0.0;
            for ir in 0..nr {
                let r = (ir as f64 + 0.5) * 10.0 / nr as f64;
                for ia in 0..na {
                    let a = (ia as f64 + 0.5) * span / na as f64;
                    let (er, ea) = (at(&e[0], r, a), at(&e[1], r, a));
                    let (hr, ha) = (at(&h[0], r, a), at(&h[1], r, a));
                    assert!((hr + ea).abs() < 1e-9 * (1.0 + ea.abs()));
                    assert!((ha - er).abs() < 1e-9 * (1.0 + er.abs()));
                    sum +=
                        (er * er + ea * ea) * (r * 1e-3) * (1e-2 / nr as f64) * (span / na as f64);
                }
            }
            assert!((sum - 1.0).abs() < 2e-3, "{mode}: ∫|e|² = {sum}");
            let scale = (at(&e[0], 5.0, 0.5).abs() + at(&e[1], 5.0, 0.5).abs()).max(1.0);
            for r in [1.0, 4.3, 9.0] {
                assert!(at(&e[0], r, 0.0).abs() < 1e-9 * scale, "{mode}");
                assert!(at(&e[0], r, span).abs() < 1e-9 * scale, "{mode}");
            }
            if !mode.is_te() {
                assert!(at(&e[1], 10.0, 0.3).abs() < 1e-9 * scale, "{mode}");
            }
            if nu > 0.0 {
                let (r, a, r1) = (3.7, 0.4, 6.1);
                let want = if mode.is_te() {
                    bessel::jv_prime(nu, kc * r * 1e-3) / bessel::jv_prime(nu, kc * r1 * 1e-3)
                } else {
                    bessel::jv(nu, kc * r * 1e-3) / r / (bessel::jv(nu, kc * r1 * 1e-3) / r1)
                };
                let got = at(&e[1], r, a) / at(&e[1], r1, a);
                assert!((got / want - 1.0).abs() < 1e-10, "{mode}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn refused_cases() {
        let mut m = sector(1.5 * PI);
        // ν = 2/3: singular at the axis.
        assert!(
            SectorWaveguidePort::new(1, 5.0, 8.0, 10.0, SectorMode::Te(1, 1))
                .add_to(&mut m)
                .is_err()
        );
        assert!(
            SectorWaveguidePort::new(2, 5.0, 8.0, 10.0, SectorMode::Tm(0, 1))
                .add_to(&mut m)
                .is_err()
        );
        let mut closed = sector(2.0 * PI);
        assert!(matches!(
            SectorWaveguidePort::new(1, 5.0, 8.0, 10.0, SectorMode::Te(0, 1)).add_to(&mut closed),
            Err(Error::Unsupported(_))
        ));
    }
}
