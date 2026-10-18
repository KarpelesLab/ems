//! Rectangular waveguide port ([`RectWaveguidePort`]).

use core::f64::consts::PI;

use csx::primitives::BoxShape;
use csx::properties::{Excitation, ProbeBox, ProbeType, WeightFn};
use csx::{ContinuousStructure, Primitive, Property, PropertyId, Scalar};

use crate::axis::Axis;
use crate::constants::{C0, MU0};
use crate::post::{Complex64, PortResult};
use crate::sim::Results;
use crate::{Error, Result};

/// The csx property names and ids of an added [`RectWaveguidePort`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveguidePortProperties {
    /// The mode `Excitation` (if excited): `port_excite_<n>`.
    pub excitation: Option<(String, PropertyId)>,
    /// The `WaveguideVoltage` `ProbeBox`: `port_ut<n>`.
    pub voltage: (String, PropertyId),
    /// The `WaveguideCurrent` `ProbeBox`: `port_it<n>`.
    pub current: (String, PropertyId),
}

/// A rectangular waveguide port for one TE_mn mode, written into a csx
/// model as a mode-shaped E-field `Excitation` and a pair of
/// mode-matching probes (`WaveguideVoltage` / `WaveguideCurrent`, see
/// [`Simulation`](crate::Simulation)).
///
/// ```
/// use csx::ContinuousStructure;
/// use ems::port::RectWaveguidePort;
/// use ems::Axis;
///
/// let mut model = ContinuousStructure::new();
/// model.grid.delta_unit = 1e-3;
/// model.grid.set_lines(0, (0..=20).map(f64::from))?;
/// model.grid.set_lines(1, (0..=10).map(f64::from))?;
/// model.grid.set_lines(2, (0..=60).map(f64::from))?;
/// // WR-like 20 mm × 10 mm guide along +z: excitation at z = 10,
/// // reference plane at z = 14, TE10.
/// let port = RectWaveguidePort::new(1, [0., 0., 10.], [20., 10., 14.], Axis::Z, (1, 0))
///     .excite(1.0)
///     .add_to(&mut model)?;
/// let fc = port.cutoff_frequency().unwrap();
/// assert!((fc - 299_792_458.0 / 0.04).abs() < 1.0);
/// assert_eq!(port.properties().unwrap().voltage.0, "port_ut1");
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Geometry
///
/// * `start` and `stop` (drawing units) span the guide's inner cross
///   section on the two transverse axes `(u, v) = axis.others()`
///   (`(y, z)` for x, `(z, x)` for y, `(x, y)` for z): width
///   `a = |stop_u − start_u|`, height `b = |stop_v − start_v|`. The guide
///   walls themselves (PEC faces of the domain or metal) are part of the
///   model, not of the port.
/// * Along `axis`, `start` is the **excitation plane** and `stop` the
///   **reference plane** where `V` and `I` are measured; the port looks
///   into the structure in the direction `start → stop` (`+axis` if
///   `stop[axis] > start[axis]`). The two planes must differ, and the
///   reference plane should be at least one cell downstream of the
///   excitation (the soft source's sheet current makes H discontinuous at
///   its own plane).
/// * The excitation is a soft source and radiates in both directions:
///   behind the excitation plane the guide should end in an absorbing
///   boundary (PML), so the backward wave and any wave reflected by the
///   structure leave the domain. A passive port (no
///   [`excite`](Self::excite)) writes only its probes.
///
/// # Mode
///
/// TE_mn with `m` half-waves along `u` and `n` along `v` (`(m, n) ≠ (0,
/// 0)`); with local coordinates `u' = u − u_min`, `v' = v − v_min` in
/// metres, the transverse mode function is (Pozar, *Microwave
/// Engineering*, rectangular waveguide TE modes)
///
/// ```text
/// e_u = −N·(n/b)·cos(mπu'/a)·sin(nπv'/b)
/// e_v =  N·(m/a)·sin(mπu'/a)·cos(nπv'/b)
/// h   = d̂ × e                       (d̂ = ±axis, the port direction)
/// ```
///
/// normalised to `∫|e|² dA = 1` (unit 1/m), so that for a single wave
/// travelling along `d̂` the modal voltage `V = ∫E_t·e dA` and current
/// `I = ∫H_t·h dA` satisfy `V/I = Z_TE`. The excitation field is
/// `amplitude·e` \[V/m\]; the launched wave amplitude depends on the soft
/// source (S-parameters do not).
///
/// The guide may be filled with a homogeneous medium ([`filled`](Self::filled));
/// the cutoff is `f_c = c/(2π√(ε_r μ_r))·√((mπ/a)² + (nπ/b)²)`, the
/// propagation constant `β = √(k² − k_c²)` and the wave impedance
/// `Z_TE = ωμ/β`, the reference impedance of [`analyze`](Self::analyze).
#[derive(Debug, Clone, PartialEq)]
pub struct RectWaveguidePort {
    number: u32,
    start: [f64; 3],
    stop: [f64; 3],
    axis: Axis,
    mode: (u32, u32),
    eps_r: f64,
    mu_r: f64,
    amplitude: Option<f64>,
    delay: f64,
    priority: i32,
    /// `[a, b]` in metres, set by `add_to`.
    dims: Option<[f64; 2]>,
    props: Option<WaveguidePortProperties>,
}

impl RectWaveguidePort {
    /// Port `number` for mode TE_`mode` between the box corners `start`
    /// and `stop` (drawing units), propagating along `axis`; see the
    /// [type docs](Self).
    pub fn new(number: u32, start: [f64; 3], stop: [f64; 3], axis: Axis, mode: (u32, u32)) -> Self {
        Self {
            number,
            start,
            stop,
            axis,
            mode,
            eps_r: 1.0,
            mu_r: 1.0,
            amplitude: None,
            delay: 0.0,
            priority: 0,
            dims: None,
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
    /// (default vacuum). Used for the cutoff, `β` and `Z_TE` only; the
    /// material itself is part of the model.
    pub fn filled(mut self, eps_r: f64, mu_r: f64) -> Self {
        self.eps_r = eps_r;
        self.mu_r = mu_r;
        self
    }

    /// Port number.
    pub fn number(&self) -> u32 {
        self.number
    }

    /// The mode `(m, n)`.
    pub fn mode(&self) -> (u32, u32) {
        self.mode
    }

    /// Whether the port is excited.
    pub fn is_excited(&self) -> bool {
        self.amplitude.is_some()
    }

    /// Guide width `a` and height `b` \[m\] (after [`add_to`](Self::add_to)).
    pub fn dimensions(&self) -> Option<[f64; 2]> {
        self.dims
    }

    /// The properties written by [`add_to`](Self::add_to) (`None` before).
    pub fn properties(&self) -> Option<&WaveguidePortProperties> {
        self.props.as_ref()
    }

    /// Cutoff wavenumber `k_c` \[1/m\] (after `add_to`).
    fn kc(&self) -> Option<f64> {
        let [a, b] = self.dims?;
        let (m, n) = (f64::from(self.mode.0), f64::from(self.mode.1));
        Some((m * PI / a).hypot(n * PI / b))
    }

    fn velocity(&self) -> f64 {
        C0 / (self.eps_r * self.mu_r).sqrt()
    }

    /// Cutoff frequency \[Hz\] (after [`add_to`](Self::add_to)).
    pub fn cutoff_frequency(&self) -> Option<f64> {
        Some(self.kc()? * self.velocity() / (2.0 * PI))
    }

    /// Propagation constant `β(f)` \[rad/m\]; `None` before `add_to` and
    /// at or below the cutoff (where the mode is evanescent).
    pub fn beta(&self, f: f64) -> Option<f64> {
        let kc = self.kc()?;
        let k = 2.0 * PI * f / self.velocity();
        (k > kc).then(|| (k * k - kc * kc).sqrt())
    }

    /// Wave impedance `Z_TE(f) = ωμ/β` \[Ω\]; `None` where
    /// [`beta`](Self::beta) is.
    pub fn z_te(&self, f: f64) -> Option<f64> {
        Some(2.0 * PI * f * MU0 * self.mu_r / self.beta(f)?)
    }

    /// Writes the port's properties into `model` and returns the port
    /// with their names and ids.
    ///
    /// # Errors
    /// [`Error::Invalid`] for mode `(0, 0)`, a cross section of zero
    /// width or height, coinciding excitation and reference planes,
    /// non-finite or non-positive values, a port already added, or a
    /// property name already present in the model.
    pub fn add_to(mut self, model: &mut ContinuousStructure) -> Result<Self> {
        let n = self.number;
        if self.props.is_some() {
            return Err(Error::Invalid(format!("port {n} was already added")));
        }
        if self.mode == (0, 0) {
            return Err(Error::Invalid(format!("port {n}: TE00 is not a mode")));
        }
        if self.start.iter().chain(&self.stop).any(|v| !v.is_finite())
            || !self.delay.is_finite()
            || self.amplitude.is_some_and(|a| !a.is_finite())
        {
            return Err(Error::Invalid("port values must be finite".into()));
        }
        if !(self.eps_r.is_finite() && self.eps_r > 0.0 && self.mu_r.is_finite() && self.mu_r > 0.0)
        {
            return Err(Error::Invalid(format!(
                "port {n}: filling ε_r = {}, μ_r = {} must be finite and > 0",
                self.eps_r, self.mu_r
            )));
        }
        let unit = model.grid.delta_unit;
        let w = self.axis.index();
        let (ua, va) = self.axis.others();
        let (u, v) = (ua.index(), va.index());
        let a = (self.stop[u] - self.start[u]).abs() * unit;
        let b = (self.stop[v] - self.start[v]).abs() * unit;
        if !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) {
            return Err(Error::Invalid(format!(
                "port {n}: the cross section must have a positive width and height"
            )));
        }
        let span = self.stop[w] - self.start[w];
        if span == 0.0 {
            return Err(Error::Invalid(format!(
                "port {n}: the excitation plane (start) and the reference plane (stop) \
                 coincide along {}",
                self.axis
            )));
        }
        let names = [
            format!("port_excite_{n}"),
            format!("port_ut{n}"),
            format!("port_it{n}"),
        ];
        if let Some(dup) = names.iter().find(|s| model.property_by_name(s).is_some()) {
            return Err(Error::Invalid(format!(
                "the model already has a property named {dup:?}"
            )));
        }
        self.dims = Some([a, b]);
        let [excite, ut, it] = names;

        // Mode function components (csx axis order) as expressions.
        let (mm, nn) = (f64::from(self.mode.0), f64::from(self.mode.1));
        let (iu, iv) = (
            if mm == 0.0 { a } else { 0.5 * a },
            if nn == 0.0 { b } else { 0.5 * b },
        );
        let (su, sv) = (
            if mm == 0.0 { 0.0 } else { 0.5 * a },
            if nn == 0.0 { 0.0 } else { 0.5 * b },
        );
        let norm = 1.0 / ((nn / b).powi(2) * iu * sv + (mm / a).powi(2) * su * iv).sqrt();
        let u0 = self.start[u].min(self.stop[u]);
        let v0 = self.start[v].min(self.stop[v]);
        let (ad, bd) = (a / unit, b / unit);
        let var = ["x", "y", "z"];
        let trig = |f: &str, k: f64, x: &str, x0: f64, len: f64| -> String {
            if k == 0.0 {
                // cos(0) = 1 (sin(0) terms are excluded by the caller).
                "1".into()
            } else {
                format!("{f}({:e}*({x}-({x0:e})))", k * PI / len)
            }
        };
        let comp = |coef: f64, fu: &str, fv: &str| -> String {
            if coef == 0.0 {
                return "0".into();
            }
            format!(
                "{coef:e}*{}*{}",
                trig(fu, mm, var[u], u0, ad),
                trig(fv, nn, var[v], v0, bd)
            )
        };
        let e_u = -norm * nn / b;
        let e_v = norm * mm / a;
        let sign = span.signum();
        let mut e = [const { String::new() }; 3];
        let mut h = [const { String::new() }; 3];
        e[w] = "0".into();
        h[w] = "0".into();
        e[u] = comp(e_u, "cos", "sin");
        e[v] = comp(e_v, "sin", "cos");
        // h = d̂ × e with (u, v, d̂) right-handed for d̂ = +axis:
        // h_u = −e_v, h_v = e_u.
        h[u] = comp(-sign * e_v, "sin", "cos");
        h[v] = comp(sign * e_u, "cos", "sin");

        let (mut lo, mut hi) = (self.start, self.stop);
        lo[w] = self.start[w];
        hi[w] = self.start[w];
        let prim = |s: [f64; 3], t: [f64; 3]| {
            Primitive::new(BoxShape::new(s, t)).with_priority(self.priority)
        };
        let excitation = match self.amplitude {
            Some(amp) => {
                let mut vec = [0.0; 3];
                for d in [u, v] {
                    if e[d] != "0" {
                        vec[d] = amp;
                    }
                }
                let mut ex = Excitation::new(vec).with_number(n);
                for d in [u, v] {
                    if e[d] != "0" {
                        ex.weight[d] = WeightFn::new(e[d].clone());
                    }
                }
                ex.delay = Scalar::new(self.delay);
                let id = model.add_property(Property::excitation(excite.clone(), ex));
                model.add_primitive(id, prim(lo, hi))?;
                Some((excite, id))
            }
            None => None,
        };
        let (mut ps, mut pt) = (self.start, self.stop);
        ps[w] = self.stop[w];
        pt[w] = self.stop[w];
        let csx_axis = csx::Axis::from_index(w).expect("axis index in 0..3");
        let mut probe = |name: &str, ty: ProbeType, f: &[String; 3]| -> Result<PropertyId> {
            let mut pb = ProbeBox::new(ty).with_norm_dir(csx_axis);
            pb.number = n;
            let mut p = Property::probe(name.to_owned(), pb);
            for (c, s) in ["X", "Y", "Z"].iter().zip(f) {
                p.set_attribute(format!("ModeFunction{c}"), s.clone());
            }
            let id = model.add_property(p);
            model.add_primitive(id, prim(ps, pt))?;
            Ok(id)
        };
        let v_id = probe(&ut, ProbeType::WaveguideVoltage, &e)?;
        let i_id = probe(&it, ProbeType::WaveguideCurrent, &h)?;
        self.props = Some(WaveguidePortProperties {
            excitation,
            voltage: (ut, v_id),
            current: (it, i_id),
        });
        Ok(self)
    }

    /// Frequency-domain analysis at `freqs` \[Hz\]: `V(f)` and `I(f)` are
    /// the DFTs of the port's mode probes (each with its exact sample
    /// times, [`ProbeResult::dft`](crate::ProbeResult::dft)) and the
    /// reference impedance is `Z_TE(f)` ([`PortResult::with_z_refs`]).
    /// `S11 = b/a` then compares the waves travelling away from and
    /// towards the structure at the reference plane.
    ///
    /// # Errors
    /// [`Error::Invalid`] if the port was not added, a probe is missing
    /// from `results`, or a frequency is at or below the cutoff (the mode
    /// does not propagate; filter with [`beta`](Self::beta)).
    pub fn analyze(&self, results: &Results, freqs: &[f64]) -> Result<PortResult> {
        let p = self
            .props
            .as_ref()
            .ok_or_else(|| Error::Invalid(format!("port {} was not added", self.number)))?;
        let z: Vec<Complex64> = freqs
            .iter()
            .map(|&f| {
                self.z_te(f).map(|z| Complex64::new(z, 0.0)).ok_or_else(|| {
                    Error::Invalid(format!(
                        "port {}: {f} Hz is not above the TE{}{} cutoff {} Hz",
                        self.number,
                        self.mode.0,
                        self.mode.1,
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
        PortResult::with_z_refs(freqs.to_vec(), v, i, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use csx::expr::Expr;
    use csx::properties::WEIGHT_VARS;

    fn model() -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        m.grid.set_lines(0, (0..=20).map(f64::from)).unwrap();
        m.grid.set_lines(1, (0..=10).map(f64::from)).unwrap();
        m.grid.set_lines(2, (0..=40).map(f64::from)).unwrap();
        m
    }

    fn eval(src: &str, p: [f64; 3]) -> f64 {
        let e = Expr::parse(src)
            .unwrap()
            .compile(&WEIGHT_VARS, |_| None)
            .unwrap();
        e.eval(&[p[0], p[1], p[2], 0.0, 0.0, 0.0, 0.0])
    }

    /// `∫|e|² dA` by the midpoint rule, and `h = ẑ × e` pointwise.
    fn check_mode(mode: (u32, u32), dir: f64) {
        let mut m = model();
        let (s, t) = if dir > 0.0 { (2.0, 5.0) } else { (30.0, 26.0) };
        let port = RectWaveguidePort::new(3, [0., 0., s], [20., 10., t], Axis::Z, mode)
            .excite(1.0)
            .add_to(&mut m)
            .unwrap();
        let pr = port.properties().unwrap();
        let attr = |id: PropertyId, c: &str| {
            m.property(id)
                .unwrap()
                .attribute(&format!("ModeFunction{c}"))
                .unwrap()
                .to_owned()
        };
        let e = ["X", "Y", "Z"].map(|c| attr(pr.voltage.1, c));
        let h = ["X", "Y", "Z"].map(|c| attr(pr.current.1, c));
        assert_eq!(e[2], "0");
        let (nx, ny) = (40, 20);
        let mut sum = 0.0;
        for ix in 0..nx {
            for iy in 0..ny {
                let p = [
                    (ix as f64 + 0.5) * 20.0 / nx as f64,
                    (iy as f64 + 0.5) * 10.0 / ny as f64,
                    0.0,
                ];
                let (ex, ey) = (eval(&e[0], p), eval(&e[1], p));
                let (hx, hy) = (eval(&h[0], p), eval(&h[1], p));
                assert!((hx + dir * ey).abs() < 1e-9 && (hy - dir * ex).abs() < 1e-9);
                sum += (ex * ex + ey * ey) * (20e-3 / nx as f64) * (10e-3 / ny as f64);
            }
        }
        assert!((sum - 1.0).abs() < 1e-4, "{mode:?}: ∫|e|² = {sum}");
    }

    #[test]
    fn mode_functions_normalised() {
        for mode in [(1, 0), (0, 1), (1, 1), (2, 1)] {
            check_mode(mode, 1.0);
            check_mode(mode, -1.0);
        }
    }

    #[test]
    fn te10_profile_and_dispersion() {
        let mut m = model();
        let port = RectWaveguidePort::new(1, [0., 0., 2.], [20., 10., 5.], Axis::Z, (1, 0))
            .add_to(&mut m)
            .unwrap();
        // Passive: probes only.
        assert!(port.properties().unwrap().excitation.is_none());
        assert_eq!(m.properties().len(), 2);
        let ey = m
            .property(port.properties().unwrap().voltage.1)
            .unwrap()
            .attribute("ModeFunctionY")
            .unwrap()
            .to_owned();
        // e_y = √(2/(ab)) sin(πx/a), peak at the centre.
        let want = (2.0f64 / (20e-3 * 10e-3)).sqrt();
        assert!((eval(&ey, [10.0, 3.0, 0.0]) - want).abs() < 1e-9 * want);
        assert!(eval(&ey, [0.0, 3.0, 0.0]).abs() < 1e-9 * want);
        let fc = port.cutoff_frequency().unwrap();
        assert!((fc - C0 / 0.04).abs() < 1e-6);
        assert!(port.beta(0.99 * fc).is_none());
        let f = 2.0 * fc;
        let k = 2.0 * PI * f / C0;
        let beta = port.beta(f).unwrap();
        assert!((beta - k * 0.75f64.sqrt()).abs() < 1e-9 * k);
        let z = port.z_te(f).unwrap();
        assert!((z - crate::constants::Z0 / 0.75f64.sqrt()).abs() < 1e-9 * z);
    }

    #[test]
    fn invalid_ports() {
        let mut m = model();
        let p = |s: [f64; 3], t: [f64; 3], mode| RectWaveguidePort::new(1, s, t, Axis::Z, mode);
        assert!(
            p([0., 0., 2.], [20., 10., 5.], (0, 0))
                .add_to(&mut m)
                .is_err()
        );
        assert!(
            p([0., 0., 2.], [0., 10., 5.], (1, 0))
                .add_to(&mut m)
                .is_err()
        );
        assert!(
            p([0., 0., 2.], [20., 10., 2.], (1, 0))
                .add_to(&mut m)
                .is_err()
        );
        let ok = p([0., 0., 2.], [20., 10., 5.], (1, 0))
            .add_to(&mut m)
            .unwrap();
        assert!(ok.clone().add_to(&mut m).is_err());
        assert!(
            p([0., 0., 2.], [20., 10., 5.], (1, 0))
                .add_to(&mut m)
                .is_err()
        ); // names
    }
}
