//! Circular waveguide port ([`CircWaveguidePort`]).

use core::f64::consts::PI;

use csx::primitives::BoxShape;
use csx::properties::{Excitation, ProbeBox, ProbeType, WeightFn};
use csx::{ContinuousStructure, CoordSystem, Primitive, Property, PropertyId, Scalar};

use super::bessel;
use crate::axis::Axis;
use crate::constants::{C0, EPS0, MU0};
use crate::post::{Complex64, PortResult};
use crate::sim::Results;
use crate::{Error, Result};

/// A mode of a circular waveguide: `n` azimuthal periods, `m`-th radial
/// root (`m ≥ 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircMode {
    /// TE_nm (`H_z ∝ J_n(k_c ρ)`, `k_c·a = p'_nm`, the `m`-th zero of `J_n'`).
    Te(u32, u32),
    /// TM_nm (`E_z ∝ J_n(k_c ρ)`, `k_c·a = p_nm`, the `m`-th zero of `J_n`).
    Tm(u32, u32),
}

impl CircMode {
    /// `(n, m)`.
    pub fn indices(self) -> (u32, u32) {
        match self {
            CircMode::Te(n, m) | CircMode::Tm(n, m) => (n, m),
        }
    }

    /// Whether the mode is TE.
    pub fn is_te(self) -> bool {
        matches!(self, CircMode::Te(..))
    }

    /// The cutoff root `k_c·a`: `p'_nm` (TE) or `p_nm` (TM); `None` for
    /// `m = 0`.
    pub fn root(self) -> Option<f64> {
        match self {
            _ if self.indices().1 == 0 => None,
            CircMode::Te(n, m) => Some(bessel::jn_prime_zero(n, m)),
            CircMode::Tm(n, m) => Some(bessel::jn_zero(n, m)),
        }
    }
}

impl core::fmt::Display for CircMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (n, m) = self.indices();
        let t = if self.is_te() { "TE" } else { "TM" };
        write!(f, "{t}{n}{m}")
    }
}

/// The csx property names and ids of an added [`CircWaveguidePort`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircWaveguidePortProperties {
    /// The mode `Excitation` (if excited): `port_excite_<n>`.
    pub excitation: Option<(String, PropertyId)>,
    /// The `WaveguideVoltage` `ProbeBox`: `port_ut<n>`.
    pub voltage: (String, PropertyId),
    /// The `WaveguideCurrent` `ProbeBox`: `port_it<n>`.
    pub current: (String, PropertyId),
}

/// A circular waveguide port for one TE_nm or TM_nm mode, written into a
/// csx model as a mode-shaped E-field `Excitation` and a pair of
/// mode-matching probes (`WaveguideVoltage` / `WaveguideCurrent`), like
/// [`RectWaveguidePort`](super::RectWaveguidePort).
///
/// ```
/// use csx::ContinuousStructure;
/// use ems::port::{CircMode, CircWaveguidePort};
///
/// let mut model = ContinuousStructure::new();
/// model.grid.delta_unit = 1e-3;
/// for a in 0..2 {
///     model.grid.set_lines(a, (-11..=11).map(f64::from))?;
/// }
/// model.grid.set_lines(2, (0..=60).map(f64::from))?;
/// // Guide of radius 10 mm along +z: excitation at z = 10, reference
/// // plane at z = 14, TE11 (E along x at the centre).
/// let port = CircWaveguidePort::new(1, [0., 0., 10.], [0., 0., 14.], 10.0, CircMode::Te(1, 1))
///     .excite(1.0)
///     .add_to(&mut model)?;
/// let fc = port.cutoff_frequency().unwrap();
/// assert!((fc / 8.7848e9 - 1.0).abs() < 1e-4);
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Geometry
///
/// * `start` and `stop` (drawing units) are the centre of the guide on
///   the **excitation plane** and on the **reference plane**; they differ
///   along exactly one axis, the propagation axis, and the port looks
///   into the structure in the direction `start → stop`. `radius` (drawing
///   units) is the inner radius `a` of the guide.
/// * **Cartesian grids**: any propagation axis; the transverse axes are
///   `(u, v) = axis.others()` and the azimuth `φ` is measured from `u`
///   towards `v`. The guide wall (a PEC tube) is part of the model, not
///   of the port: a staircased wall has a different cutoff from the
///   ideal circle (see design §5.5), which the analysis does not know.
/// * **Cylindrical grids**: the guide is coaxial with the grid, `start`
///   and `stop` must lie on the axis (`ρ = 0`), the propagation axis is
///   `z` and the azimuth must be closed (`2π`). The wall can be the
///   `ρ_max` PEC face (`radius` = the last `ρ` line) or metal.
/// * As for the rectangular port, the excitation is a soft source that
///   radiates both ways (end the guide behind it in a PML), the two planes
///   must differ, and the reference plane should be at least one cell
///   downstream of the excitation.
///
/// # Mode
///
/// With `ψ = J_n(k_c ρ)·sin(n(φ − φ0))` (TE, `n > 0`), `J_0(k_c ρ)` (TE,
/// `n = 0`) or `J_n(k_c ρ)·cos(n(φ − φ0))` (TM), `ρ` in metres from the
/// centre and `φ0` the [`polarization`](Self::polarization) angle, the
/// transverse mode function is (Pozar, *Microwave Engineering*, §3.4;
/// Marcuvitz, *Waveguide Handbook*, §2.3)
///
/// ```text
/// TE:  e = N·∇ψ × ẑ:  e_ρ =  N·(n/ρ)·J_n·cos(n(φ−φ0)),  e_φ = −N·k_c·J_n'·sin(n(φ−φ0))
/// TM:  e = N·∇ψ:      e_ρ =  N·k_c·J_n'·cos(n(φ−φ0)),   e_φ = −N·(n/ρ)·J_n·sin(n(φ−φ0))
/// h = d̂ × e
/// ```
///
/// (TE0m: `e_φ = N·k_c·J_1(k_c ρ)`), so that `φ0 = 0` puts the TE11
/// field along `u` (x for a z guide) at the centre. `N` normalises
/// `∫|e|² dA = 1`: by Green's identity `∫|∇ψ|² = k_c² ∫ψ²`, and
/// (Abramowitz & Stegun 11.4.5)
///
/// ```text
/// ∫₀^a J_n(k_c ρ)² ρ dρ = a²/2 · (1 − n²/p'²)·J_n(p')²   (TE)
///                       = a²/2 · J_n'(p)²                (TM)
/// ```
///
/// times `π` (`n > 0`) or `2π` (`n = 0`) for the azimuth. The cutoff is
/// `k_c = p/a` with the Bessel root `p` ([`CircMode::root`]); in a medium
/// `ε_r`, `μ_r` ([`filled`](Self::filled)) `β = √(k² − k_c²)` and the wave
/// impedance (the reference impedance of [`analyze`](Self::analyze)) is
/// `Z_TE = ωμ/β` or `Z_TM = β/(ωε)`.
///
/// The mode functions are written as csx expressions (`jn(n, x)`, with
/// `J_n' = (J_{n−1} − J_{n+1})/2`) in drawing-unit coordinates; on a
/// Cartesian grid they are masked to `ρ ≤ a` with `if()`, on a
/// cylindrical grid they use the `rho` and `a` weight variables and give
/// the `(ρ, α)` components directly.
#[derive(Debug, Clone, PartialEq)]
pub struct CircWaveguidePort {
    number: u32,
    start: [f64; 3],
    stop: [f64; 3],
    radius: f64,
    mode: CircMode,
    phi0: f64,
    eps_r: f64,
    mu_r: f64,
    amplitude: Option<f64>,
    delay: f64,
    priority: i32,
    /// `k_c` \[1/m\], set by `add_to`.
    kc: Option<f64>,
    props: Option<CircWaveguidePortProperties>,
}

impl CircWaveguidePort {
    /// Port `number` for `mode` with the guide centre at `start`
    /// (excitation plane) and `stop` (reference plane) and inner radius
    /// `radius` (drawing units); see the [type docs](Self).
    pub fn new(number: u32, start: [f64; 3], stop: [f64; 3], radius: f64, mode: CircMode) -> Self {
        Self {
            number,
            start,
            stop,
            radius,
            mode,
            phi0: 0.0,
            eps_r: 1.0,
            mu_r: 1.0,
            amplitude: None,
            delay: 0.0,
            priority: 0,
            kc: None,
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

    /// Rotates the mode pattern by `phi0` \[rad\] about the axis (default
    /// 0; for TE11, `π/2` puts the centre field along `v`). No effect on
    /// `n = 0` modes.
    pub fn polarization(mut self, phi0: f64) -> Self {
        self.phi0 = phi0;
        self
    }

    /// The guide is filled with a homogeneous lossless medium `ε_r`, `μ_r`
    /// (default vacuum). Used for the cutoff, `β` and the wave impedance
    /// only; the material itself is part of the model.
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
    pub fn mode(&self) -> CircMode {
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

    fn velocity(&self) -> f64 {
        C0 / (self.eps_r * self.mu_r).sqrt()
    }

    /// Cutoff wavenumber `k_c` \[1/m\] (after [`add_to`](Self::add_to)).
    pub fn cutoff_wavenumber(&self) -> Option<f64> {
        self.kc
    }

    /// Cutoff frequency \[Hz\] (after [`add_to`](Self::add_to)).
    pub fn cutoff_frequency(&self) -> Option<f64> {
        Some(self.kc? * self.velocity() / (2.0 * PI))
    }

    /// Propagation constant `β(f)` \[rad/m\]; `None` before `add_to` and
    /// at or below the cutoff.
    pub fn beta(&self, f: f64) -> Option<f64> {
        let kc = self.kc?;
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
    /// [`Error::Invalid`] for `m = 0`, a non-positive radius, planes that
    /// coincide or differ along more than one axis, non-finite or
    /// non-positive values, a port already added, or a property name
    /// already present in the model. [`Error::Unsupported`] on a
    /// cylindrical grid if the port is not on the axis along `z` or the
    /// azimuth is not closed.
    pub fn add_to(mut self, model: &mut ContinuousStructure) -> Result<Self> {
        let num = self.number;
        if self.props.is_some() {
            return Err(Error::Invalid(format!("port {num} was already added")));
        }
        let Some(p) = self.mode.root() else {
            return Err(Error::Invalid(format!(
                "port {num}: {} is not a mode (m ≥ 1)",
                self.mode
            )));
        };
        if self.start.iter().chain(&self.stop).any(|v| !v.is_finite())
            || !self.delay.is_finite()
            || !self.phi0.is_finite()
            || self.amplitude.is_some_and(|a| !a.is_finite())
        {
            return Err(Error::Invalid("port values must be finite".into()));
        }
        if !(self.radius.is_finite() && self.radius > 0.0) {
            return Err(Error::Invalid(format!(
                "port {num}: radius {} must be finite and > 0",
                self.radius
            )));
        }
        if !(self.eps_r.is_finite() && self.eps_r > 0.0 && self.mu_r.is_finite() && self.mu_r > 0.0)
        {
            return Err(Error::Invalid(format!(
                "port {num}: filling ε_r = {}, μ_r = {} must be finite and > 0",
                self.eps_r, self.mu_r
            )));
        }
        let cyl = match model.grid.coord_system {
            CoordSystem::Cartesian => false,
            CoordSystem::Cylindrical => true,
            #[allow(unreachable_patterns)]
            _ => {
                return Err(Error::Unsupported(format!(
                    "port {num}: unsupported coordinate system"
                )));
            }
        };
        let unit = model.grid.delta_unit;
        let axis = if cyl {
            if self.start[0] != 0.0 || self.stop[0] != 0.0 {
                return Err(Error::Unsupported(format!(
                    "port {num}: on a cylindrical grid the guide must be on the axis (ρ = 0)"
                )));
            }
            if self.start[2] == self.stop[2] {
                return Err(Error::Invalid(format!(
                    "port {num}: the excitation and reference planes coincide"
                )));
            }
            let al = model.grid.lines(1);
            let span = match (al.first(), al.last()) {
                (Some(a), Some(b)) => b - a,
                _ => 0.0,
            };
            if (span - 2.0 * PI).abs() > 1e-9 * 2.0 * PI {
                return Err(Error::Unsupported(format!(
                    "port {num}: a circular waveguide port needs a closed azimuth (2π)"
                )));
            }
            Axis::Z
        } else {
            let diff: Vec<usize> = (0..3).filter(|&d| self.start[d] != self.stop[d]).collect();
            match diff[..] {
                [d] => Axis::from_index(d).expect("axis index in 0..3"),
                [] => {
                    return Err(Error::Invalid(format!(
                        "port {num}: the excitation and reference planes coincide"
                    )));
                }
                _ => {
                    return Err(Error::Invalid(format!(
                        "port {num}: start and stop must differ along one axis only"
                    )));
                }
            }
        };
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
        let a_m = self.radius * unit;
        let kc = p / a_m;
        self.kc = Some(kc);
        let w = axis.index();
        let sign = (self.stop[w] - self.start[w]).signum();
        let (e, h) = self.mode_functions(cyl, axis, unit, p, sign);

        // Boxes: the cross section on the excitation / reference plane.
        let alpha = {
            let al = model.grid.lines(1);
            (
                al.first().copied().unwrap_or(0.0),
                al.last().copied().unwrap_or(0.0),
            )
        };
        let plane = |at: f64| -> ([f64; 3], [f64; 3]) {
            if cyl {
                ([0.0, alpha.0, at], [self.radius, alpha.1, at])
            } else {
                let (mut lo, mut hi) = (self.start, self.start);
                let (u, v) = axis.others();
                for d in [u.index(), v.index()] {
                    lo[d] -= self.radius;
                    hi[d] += self.radius;
                }
                lo[w] = at;
                hi[w] = at;
                (lo, hi)
            }
        };
        let prim = |(s, t): ([f64; 3], [f64; 3])| {
            Primitive::new(BoxShape::new(s, t)).with_priority(self.priority)
        };
        let [excite, ut, it] = names;
        let excitation = match self.amplitude {
            Some(amp) => {
                let mut vec = [0.0; 3];
                for d in 0..3 {
                    if e[d] != "0" {
                        vec[d] = amp;
                    }
                }
                let mut ex = Excitation::new(vec).with_number(num);
                for (d, f) in e.iter().enumerate() {
                    if f != "0" {
                        ex.weight[d] = WeightFn::new(f.clone());
                    }
                }
                ex.delay = Scalar::new(self.delay);
                let id = model.add_property(Property::excitation(excite.clone(), ex));
                model.add_primitive(id, prim(plane(self.start[w])))?;
                Some((excite, id))
            }
            None => None,
        };
        let csx_axis = csx::Axis::from_index(w).expect("axis index in 0..3");
        let ref_plane = plane(self.stop[w]);
        let mut probe = |name: &str, ty: ProbeType, f: &[String; 3]| -> Result<PropertyId> {
            let mut pb = ProbeBox::new(ty).with_norm_dir(csx_axis);
            pb.number = num;
            let mut p = Property::probe(name.to_owned(), pb);
            for (c, s) in ["X", "Y", "Z"].iter().zip(f) {
                p.set_attribute(format!("ModeFunction{c}"), s.clone());
            }
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

    /// The csx expressions of `e` and `h` (grid component order), for root
    /// `p` and propagation sign `sign`.
    fn mode_functions(
        &self,
        cyl: bool,
        axis: Axis,
        unit: f64,
        p: f64,
        sign: f64,
    ) -> ([String; 3], [String; 3]) {
        let (n, _) = self.mode.indices();
        let nf = f64::from(n);
        let a_m = self.radius * unit;
        let kc = p / a_m;
        let ang = if n == 0 { 2.0 * PI } else { PI };
        let rad = if self.mode.is_te() {
            0.5 * a_m * a_m * (1.0 - nf * nf / (p * p)) * bessel::jn(n as i32, p).powi(2)
        } else {
            0.5 * a_m * a_m * bessel::jn_prime(n as i32, p).powi(2)
        };
        let norm = 1.0 / (kc * (ang * rad).sqrt());
        let var = ["x", "y", "z"];
        let (u, v) = axis.others();
        let (r, phi) = if cyl {
            ("rho".to_owned(), "a".to_owned())
        } else {
            let du = format!("({}-({:e}))", var[u.index()], self.start[u.index()]);
            let dv = format!("({}-({:e}))", var[v.index()], self.start[v.index()]);
            (format!("hypot({du},{dv})"), format!("atan2({dv},{du})"))
        };
        // Argument k_c·ρ with ρ in drawing units.
        let kr = format!("{:e}*{r}", kc * unit);
        let j = |k: i64| format!("jn({k},{kr})");
        let jp = format!("({}-{})/2", j(i64::from(n) - 1), j(i64::from(n) + 1));
        // J_n(k_c ρ)/ρ [1/m], with its limit at ρ = 0.
        let lim = if n == 1 { 0.5 * kc } else { 0.0 };
        let jr = format!("if({r}>0,{}/({r}*{unit:e}),{lim:e})", j(i64::from(n)));
        let arg = format!("{nf:e}*({phi}-({:e}))", self.phi0);
        let (c, s) = (format!("cos({arg})"), format!("sin({arg})"));
        let (e_rho, e_phi) = match (self.mode.is_te(), n) {
            (true, 0) => ("0".to_owned(), format!("{:e}*{}", norm * kc, j(1))),
            (true, _) => (
                format!("{:e}*{jr}*{c}", norm * nf),
                format!("{:e}*{jp}*{s}", -norm * kc),
            ),
            (false, 0) => (format!("{:e}*{jp}", norm * kc), "0".to_owned()),
            (false, _) => (
                format!("{:e}*{jp}*{c}", norm * kc),
                format!("{:e}*{jr}*{s}", -norm * nf),
            ),
        };
        let neg = |s: &str| -> String {
            if s == "0" {
                "0".into()
            } else {
                format!("-({s})")
            }
        };
        let scaled = |k: f64, s: &str| -> String {
            if s == "0" {
                "0".into()
            } else {
                format!("{k:e}*({s})")
            }
        };
        let mut e = [const { String::new() }; 3];
        let mut h = [const { String::new() }; 3];
        if cyl {
            e = [e_rho.clone(), e_phi.clone(), "0".into()];
            // h = ±ẑ × e: h_ρ = −e_φ, h_φ = e_ρ.
            h = [scaled(-sign, &e_phi), scaled(sign, &e_rho), "0".into()];
        } else {
            let comb = |a: &str, ca: &str, b: &str, cb: &str, minus: bool| -> String {
                let ta = (a != "0").then(|| format!("({a})*{ca}({phi})"));
                let tb = (b != "0").then(|| format!("({b})*{cb}({phi})"));
                let body = match (ta, tb) {
                    (Some(x), Some(y)) => format!("{x}{}{y}", if minus { "-" } else { "+" }),
                    (Some(x), None) => x,
                    (None, Some(y)) => {
                        if minus {
                            neg(&y)
                        } else {
                            y
                        }
                    }
                    (None, None) => return "0".into(),
                };
                format!("if({r}<={:e},{body},0)", self.radius)
            };
            // e_u = e_ρ cos φ − e_φ sin φ, e_v = e_ρ sin φ + e_φ cos φ.
            let eu = comb(&e_rho, "cos", &e_phi, "sin", true);
            let ev = comb(&e_rho, "sin", &e_phi, "cos", false);
            e[axis.index()] = "0".into();
            h[axis.index()] = "0".into();
            // h = d̂ × e: h_u = −e_v, h_v = e_u (for d̂ = +axis).
            h[u.index()] = scaled(-sign, &ev);
            h[v.index()] = scaled(sign, &eu);
            e[u.index()] = eu;
            e[v.index()] = ev;
        }
        (e, h)
    }

    /// Frequency-domain analysis at `freqs` \[Hz\]: `V(f)` and `I(f)` are
    /// the DFTs of the mode probes and the reference impedance is the
    /// wave impedance ([`z_wave`](Self::z_wave)), as for
    /// [`RectWaveguidePort::analyze`](super::RectWaveguidePort::analyze).
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

#[cfg(test)]
mod tests {
    use super::*;
    use csx::expr::Expr;
    use csx::properties::WEIGHT_VARS;

    fn cart_model() -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        for a in 0..3 {
            m.grid.set_lines(a, (-12..=12).map(f64::from)).unwrap();
        }
        m
    }

    fn cyl_model() -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.coord_system = CoordSystem::Cylindrical;
        m.grid.coord_system = CoordSystem::Cylindrical;
        m.grid.delta_unit = 1e-3;
        m.grid.set_lines(0, (0..=10).map(f64::from)).unwrap();
        m.grid
            .set_lines(1, (0..=16).map(|j| j as f64 * 2.0 * PI / 16.0))
            .unwrap();
        m.grid.set_lines(2, (0..=20).map(f64::from)).unwrap();
        m
    }

    fn compile(src: &str) -> csx::expr::CompiledExpr {
        Expr::parse(src)
            .unwrap()
            .compile(&WEIGHT_VARS, |_| None)
            .unwrap()
    }

    fn attrs(m: &ContinuousStructure, id: PropertyId) -> [String; 3] {
        ["X", "Y", "Z"].map(|c| {
            m.property(id)
                .unwrap()
                .attribute(&format!("ModeFunction{c}"))
                .unwrap()
                .to_owned()
        })
    }

    /// `∫|e|² dA = 1` over the disc (polar midpoint rule), `h = d̂ × e`,
    /// and the Cartesian and cylindrical expressions agree.
    fn check(mode: CircMode, phi0: f64, axis: Axis, dir: f64) {
        let r0 = 10.0;
        let mut c = [0.5, -1.0, 2.0];
        c[axis.index()] = 0.0;
        let (s, mut t) = (c, c);
        t[axis.index()] = 4.0 * dir;
        let mut m = cart_model();
        let port = CircWaveguidePort::new(2, s, t, r0, mode)
            .polarization(phi0)
            .excite(1.0)
            .add_to(&mut m)
            .unwrap();
        let pr = port.properties().unwrap();
        let e = attrs(&m, pr.voltage.1).map(|s| compile(&s));
        let h = attrs(&m, pr.current.1).map(|s| compile(&s));
        let mut cm = cyl_model();
        let cp = CircWaveguidePort::new(2, [0., 0., 2.], [0., 0., 2. + 4.0 * dir], r0, mode)
            .polarization(phi0)
            .add_to(&mut cm)
            .unwrap();
        let ce = attrs(&cm, cp.properties().unwrap().voltage.1).map(|s| compile(&s));
        let ch = attrs(&cm, cp.properties().unwrap().current.1).map(|s| compile(&s));
        let (u, v) = axis.others();
        let (nr, na) = (200, 128);
        let mut sum = 0.0;
        for ir in 0..nr {
            let rho = (ir as f64 + 0.5) * r0 / nr as f64;
            for ia in 0..na {
                let phi = (ia as f64 + 0.5) * 2.0 * PI / na as f64;
                let mut q = c;
                q[u.index()] += rho * phi.cos();
                q[v.index()] += rho * phi.sin();
                let vars = csx::properties::weight_vars(csx::Vec3::new(q[0], q[1], q[2]));
                let ev = e.each_ref().map(|f| f.eval(&vars));
                let hv = h.each_ref().map(|f| f.eval(&vars));
                assert_eq!(ev[axis.index()], 0.0);
                let (eu, evv) = (ev[u.index()], ev[v.index()]);
                let (hu, hv_) = (hv[u.index()], hv[v.index()]);
                assert!((hu + dir * evv).abs() < 1e-9 && (hv_ - dir * eu).abs() < 1e-9);
                sum += (eu * eu + evv * evv)
                    * (rho * 1e-3)
                    * (r0 * 1e-3 / nr as f64)
                    * (2.0 * PI / na as f64);
                // Cylindrical: (ρ, α) components at (ρ cos φ, ρ sin φ).
                let cv = csx::properties::weight_vars(csx::Vec3::new(
                    rho * phi.cos(),
                    rho * phi.sin(),
                    0.0,
                ));
                let (er, ep) = (ce[0].eval(&cv), ce[1].eval(&cv));
                let (cu, cvv) = (
                    er * phi.cos() - ep * phi.sin(),
                    er * phi.sin() + ep * phi.cos(),
                );
                let scale = eu.abs().max(evv.abs()).max(1.0);
                assert!(
                    (cu - eu).abs() < 1e-9 * scale && (cvv - evv).abs() < 1e-9 * scale,
                    "{mode} at ρ = {rho}, φ = {phi}: ({cu}, {cvv}) vs ({eu}, {evv})"
                );
                let (hr, hp) = (ch[0].eval(&cv), ch[1].eval(&cv));
                assert!(
                    (hr + dir * ep).abs() < 1e-9 * scale && (hp - dir * er).abs() < 1e-9 * scale
                );
            }
        }
        assert!((sum - 1.0).abs() < 1e-3, "{mode}: ∫|e|² = {sum}");
        // Outside the disc the Cartesian functions vanish.
        let mut q = c;
        q[u.index()] += 10.5;
        let vars = csx::properties::weight_vars(csx::Vec3::new(q[0], q[1], q[2]));
        assert!(e.iter().all(|f| f.eval(&vars) == 0.0));
    }

    #[test]
    fn mode_functions_normalised() {
        for mode in [
            CircMode::Te(1, 1),
            CircMode::Te(0, 1),
            CircMode::Te(2, 1),
            CircMode::Te(1, 2),
            CircMode::Tm(0, 1),
            CircMode::Tm(1, 1),
            CircMode::Tm(2, 1),
        ] {
            check(mode, 0.0, Axis::Z, 1.0);
            check(mode, 0.3, Axis::Z, -1.0);
        }
        check(CircMode::Te(1, 1), 0.0, Axis::X, 1.0);
        check(CircMode::Tm(1, 1), 0.2, Axis::Y, -1.0);
    }

    #[test]
    fn te11_centre_field_and_dispersion() {
        let mut m = cart_model();
        let port = CircWaveguidePort::new(1, [0., 0., 0.], [0., 0., 3.], 10.0, CircMode::Te(1, 1))
            .add_to(&mut m)
            .unwrap();
        assert!(port.properties().unwrap().excitation.is_none());
        let e = attrs(&m, port.properties().unwrap().voltage.1).map(|s| compile(&s));
        let vars = csx::properties::weight_vars(csx::Vec3::new(0.0, 0.0, 0.0));
        let (ex, ey) = (e[0].eval(&vars), e[1].eval(&vars));
        assert!(ex > 0.0 && ey.abs() < 1e-12 * ex, "({ex}, {ey})");
        let fc = port.cutoff_frequency().unwrap();
        assert!((fc / (1.841_183_781 * C0 / (2.0 * PI * 0.01)) - 1.0).abs() < 1e-9);
        assert!(port.beta(0.99 * fc).is_none());
        let f = 2.0 * fc;
        let k = 2.0 * PI * f / C0;
        assert!((port.beta(f).unwrap() - k * 0.75f64.sqrt()).abs() < 1e-9 * k);
        let z = port.z_wave(f).unwrap();
        assert!((z - crate::constants::Z0 / 0.75f64.sqrt()).abs() < 1e-9 * z);
        let mut m = cart_model();
        let tm = CircWaveguidePort::new(1, [0., 0., 0.], [0., 0., 3.], 10.0, CircMode::Tm(0, 1))
            .filled(2.0, 1.0)
            .add_to(&mut m)
            .unwrap();
        let fc = tm.cutoff_frequency().unwrap();
        let z = tm.z_wave(2.0 * fc).unwrap();
        assert!((z - crate::constants::Z0 / 2f64.sqrt() * 0.75f64.sqrt()).abs() < 1e-9 * z);
    }

    #[test]
    fn invalid_ports() {
        let mut m = cart_model();
        let p = |s: [f64; 3], t: [f64; 3], r: f64, mode| CircWaveguidePort::new(1, s, t, r, mode);
        let te = CircMode::Te(1, 1);
        assert!(
            p([0.; 3], [0., 0., 2.], 5.0, CircMode::Te(1, 0))
                .add_to(&mut m)
                .is_err()
        );
        assert!(p([0.; 3], [0., 0., 2.], 0.0, te).add_to(&mut m).is_err());
        assert!(p([0.; 3], [0.; 3], 5.0, te).add_to(&mut m).is_err());
        assert!(p([0.; 3], [1., 0., 2.], 5.0, te).add_to(&mut m).is_err());
        let ok = p([0.; 3], [0., 0., 2.], 5.0, te).add_to(&mut m).unwrap();
        assert!(ok.clone().add_to(&mut m).is_err());
        assert!(p([0.; 3], [0., 0., 2.], 5.0, te).add_to(&mut m).is_err()); // names
        let mut c = cyl_model();
        assert!(matches!(
            p([1., 0., 2.], [1., 0., 5.], 5.0, te).add_to(&mut c),
            Err(Error::Unsupported(_))
        ));
        c.grid.set_lines(1, [0.0, 1.0, 2.0]).unwrap();
        assert!(matches!(
            p([0., 0., 2.], [0., 0., 5.], 5.0, te).add_to(&mut c),
            Err(Error::Unsupported(_))
        ));
    }
}
