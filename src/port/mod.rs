//! Port helpers that **emit csx properties** into a model (design §2):
//! the ports round-trip through openEMS XML and run unchanged from XML.
//!
//! Not to be confused with the engine-level
//! [`engine::LumpedPort`](crate::engine::LumpedPort), an operator installed
//! directly on an [`Engine`](crate::Engine); [`LumpedPort`] here builds the
//! same physical port out of csx properties, which the
//! [`Simulation`](crate::Simulation) maps onto engine operators.

use csx::primitives::BoxShape;
use csx::properties::{Excitation, LumpedElement, ProbeBox, ProbeType};
use csx::{ContinuousStructure, Primitive, Property, PropertyId, Scalar};

use crate::axis::Axis;
use crate::post::{Complex64, PortResult};
use crate::sim::Results;
use crate::{Error, Result};

/// The csx property names and ids of an added [`LumpedPort`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortProperties {
    /// The `LumpedElement` (the port resistance): `port_resist_<n>`.
    pub element: (String, PropertyId),
    /// The `Excitation` (if excited): `port_excite_<n>`.
    pub excitation: Option<(String, PropertyId)>,
    /// The voltage `ProbeBox`: `port_ut<n>`.
    pub voltage: (String, PropertyId),
    /// The current `ProbeBox`: `port_it<n>`.
    pub current: (String, PropertyId),
}

/// A lumped port written into a csx model: a `LumpedElement` resistor
/// `R`, an optional E-field `Excitation`, and voltage and current
/// `ProbeBox`es over one box (the usual openEMS script construction).
///
/// ```
/// use csx::ContinuousStructure;
/// use ems::port::LumpedPort;
/// use ems::Axis;
///
/// let mut model = ContinuousStructure::new();
/// model.grid.delta_unit = 1e-3;
/// for a in 0..3 {
///     model.grid.set_lines(a, (0..=10).map(|i| i as f64))?;
/// }
/// let port = LumpedPort::new(1, 50.0, [5.0, 5.0, 4.0], [5.0, 5.0, 6.0], Axis::Z)
///     .excite(1.0)
///     .add_to(&mut model)?;
/// assert_eq!(port.properties().unwrap().voltage.0, "port_ut1");
/// assert_eq!(model.properties().len(), 4);
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Conventions
///
/// * The port runs from `start` to `stop` along `axis` (drawing units);
///   `start[axis] > stop[axis]` reverses it. `start` is the `+` terminal.
/// * The **excitation** (if any) is a soft E source of
///   `amplitude / L` V/m along `start → stop` (`L = |stop − start|` along
///   `axis`, in metres), so that over the resistor it is a Thevenin source
///   of `amplitude` volts per column (see [`Simulation`](crate::Simulation)):
///   an open port reads `V = amplitude·waveform(t)`.
/// * The **voltage** probe is the line from `start` to `stop` along `axis`
///   through the transverse centre of the box: `V = ∫E·dl`.
/// * The **current** probe covers the whole box with normal `axis` and
///   weight `−1`: `I` is the current leaving the `+` terminal into the
///   structure, so `P = V·I` is the power delivered to it (the convention
///   of [`PortResult`]).
/// * The element has PEC end caps (csx `Caps`, default) and the given
///   priority (default 0) for all four primitives.
#[derive(Debug, Clone, PartialEq)]
pub struct LumpedPort {
    number: u32,
    resistance: f64,
    start: [f64; 3],
    stop: [f64; 3],
    axis: Axis,
    amplitude: Option<f64>,
    delay: f64,
    priority: i32,
    props: Option<PortProperties>,
}

impl LumpedPort {
    /// Port number `number` with resistance `r` \[Ω\] (also the reference
    /// impedance of [`analyze`](Self::analyze)) between the box corners
    /// `start` and `stop` (drawing units), directed along `axis`.
    pub fn new(number: u32, r: f64, start: [f64; 3], stop: [f64; 3], axis: Axis) -> Self {
        Self {
            number,
            resistance: r,
            start,
            stop,
            axis,
            amplitude: None,
            delay: 0.0,
            priority: 0,
            props: None,
        }
    }

    /// Excites the port with a Thevenin voltage of `amplitude` \[V\] times
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

    /// Port number.
    pub fn number(&self) -> u32 {
        self.number
    }

    /// Port resistance \[Ω\].
    pub fn resistance(&self) -> f64 {
        self.resistance
    }

    /// Whether the port is excited.
    pub fn is_excited(&self) -> bool {
        self.amplitude.is_some()
    }

    /// The properties written by [`add_to`](Self::add_to) (`None` before).
    pub fn properties(&self) -> Option<&PortProperties> {
        self.props.as_ref()
    }

    /// Writes the port's properties into `model` and returns the port with
    /// their names and ids.
    ///
    /// # Errors
    /// [`Error::Invalid`] for a non-positive resistance, a zero-length
    /// port, non-finite values, a port already added, or a property name
    /// already present in the model.
    pub fn add_to(mut self, model: &mut ContinuousStructure) -> Result<Self> {
        if self.props.is_some() {
            return Err(Error::Invalid(format!(
                "port {} was already added",
                self.number
            )));
        }
        if !(self.resistance.is_finite() && self.resistance > 0.0) {
            return Err(Error::Invalid(format!(
                "port resistance {} must be finite and > 0",
                self.resistance
            )));
        }
        if self.start.iter().chain(&self.stop).any(|v| !v.is_finite())
            || !self.delay.is_finite()
            || self.amplitude.is_some_and(|a| !a.is_finite())
        {
            return Err(Error::Invalid("port values must be finite".into()));
        }
        let a = self.axis.index();
        let span = self.stop[a] - self.start[a];
        let length = span.abs() * model.grid.delta_unit;
        if !(length > 0.0 && length.is_finite()) {
            return Err(Error::Invalid(format!(
                "port {} has zero length along {}",
                self.number, self.axis
            )));
        }
        let n = self.number;
        let names = [
            format!("port_resist_{n}"),
            format!("port_excite_{n}"),
            format!("port_ut{n}"),
            format!("port_it{n}"),
        ];
        if let Some(dup) = names.iter().find(|s| model.property_by_name(s).is_some()) {
            return Err(Error::Invalid(format!(
                "the model already has a property named {dup:?}"
            )));
        }
        let [resist, excite, ut, it] = names;
        let csx_axis = csx::Axis::from_index(a).expect("axis index in 0..3");
        let prim = |s: [f64; 3], t: [f64; 3]| {
            Primitive::new(BoxShape::new(s, t)).with_priority(self.priority)
        };
        let (start, stop) = (self.start, self.stop);

        let el = LumpedElement::resistor(csx_axis, self.resistance);
        let el_id = model.add_property(Property::lumped_element(resist.clone(), el));
        model.add_primitive(el_id, prim(start, stop))?;

        let excitation = match self.amplitude {
            Some(amp) => {
                let mut v = [0.0; 3];
                v[a] = span.signum() * amp / length;
                let mut ex = Excitation::new(v).with_number(n);
                ex.delay = Scalar::new(self.delay);
                let id = model.add_property(Property::excitation(excite.clone(), ex));
                model.add_primitive(id, prim(start, stop))?;
                Some((excite, id))
            }
            None => None,
        };

        let mut mid = [0.0; 3];
        for d in 0..3 {
            mid[d] = 0.5 * (start[d] + stop[d]);
        }
        let (mut vs, mut vt) = (mid, mid);
        vs[a] = start[a];
        vt[a] = stop[a];
        let mut vp = ProbeBox::new(ProbeType::Voltage);
        vp.number = n;
        let v_id = model.add_property(Property::probe(ut.clone(), vp));
        model.add_primitive(v_id, prim(vs, vt))?;

        let mut ip = ProbeBox::new(ProbeType::Current)
            .with_weight(-1.0)
            .with_norm_dir(csx_axis);
        ip.number = n;
        let i_id = model.add_property(Property::probe(it.clone(), ip));
        model.add_primitive(i_id, prim(start, stop))?;

        self.props = Some(PortProperties {
            element: (resist, el_id),
            excitation,
            voltage: (ut, v_id),
            current: (it, i_id),
        });
        Ok(self)
    }

    /// Frequency-domain analysis at `freqs` \[Hz\] from the port's probes in
    /// `results`, with reference impedance `Z_ref = R`.
    ///
    /// # Errors
    /// See [`analyze_with`](Self::analyze_with).
    pub fn analyze(&self, results: &Results, freqs: &[f64]) -> Result<PortResult> {
        self.analyze_with(results, freqs, Complex64::new(self.resistance, 0.0))
    }

    /// Frequency-domain analysis with reference impedance `z_ref`: `V(f)`
    /// and `I(f)` are the DFTs of the voltage and current probes with their
    /// own sample times ([`ProbeResult::dft`](crate::ProbeResult::dft)), so
    /// the half-step offset of the current is exact.
    ///
    /// # Errors
    /// [`Error::Invalid`] if the port was not added, its probes are missing
    /// from `results` or have no samples, or `z_ref` is invalid.
    pub fn analyze_with(
        &self,
        results: &Results,
        freqs: &[f64],
        z_ref: Complex64,
    ) -> Result<PortResult> {
        let p = self
            .props
            .as_ref()
            .ok_or_else(|| Error::Invalid(format!("port {} was not added", self.number)))?;
        let get = |name: &str| {
            results
                .probe(name)
                .ok_or_else(|| Error::Invalid(format!("no probe {name:?} in the results")))
        };
        let v = get(&p.voltage.0)?.dft(freqs)?;
        let i = get(&p.current.0)?.dft(freqs)?;
        PortResult::new(freqs.to_vec(), v, i, z_ref)
    }
}
