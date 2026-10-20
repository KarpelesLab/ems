//! Microstrip-line (MSL) port: a section of strip line with voltage and
//! current probes from which the line's propagation constant and
//! characteristic impedance are extracted, and the forward and backward
//! waves are computed at a chosen reference plane.
//!
//! # Line model
//!
//! Along the propagation coordinate `s` (metres from the feed plane,
//! increasing into the structure) the quasi-TEM mode of a uniform line is
//!
//! ```text
//! V(s) = A·e^{−γs} + B·e^{γs},   I(s) = (A·e^{−γs} − B·e^{γs}) / Z_L
//! ```
//!
//! with `γ = α + jβ` and line impedance `Z_L` (Pozar, *Microwave
//! Engineering*, §2.1). The port records three voltages `V1, V2, V3` at
//! equally spaced mesh lines `s2 − d, s2, s2 + d`, and the currents `Ia`,
//! `Ib` on the dual planes halfway between them (where the Yee scheme
//! measures the H loop around the strip). From the expressions above,
//! for any `A`, `B`:
//!
//! ```text
//! V1 + V3 = 2·cosh(γd)·V2              ⇒  sinh²(γd/2) = (V1 + V3 − 2·V2) / (4·V2)
//! V3 − V1 = −2·sinh(γd/2)·Z_L·(Ia + Ib) ⇒  Z_L = −(V3 − V1) / (2·sinh(γd/2)·(Ia + Ib))
//! I(s2)   = (Ia + Ib) / (2·cosh(γd/2))
//! ```
//!
//! The root is chosen with `β ≥ 0` (`βd < π` is assumed, i.e. more than two
//! cells per guided wavelength). The waves at `s2` are
//! `A' = (V2 + Z_L·I(s2))/2`, `B' = (V2 − Z_L·I(s2))/2`, and at the
//! reference plane `s_ref` they are `A'·e^{−γ(s_ref − s2)}` and
//! `B'·e^{γ(s_ref − s2)}`; [`MslPort::analyze`] reports the voltage and
//! current there as a [`PortResult`](crate::post::PortResult).
//!
//! The relations are exact for any spatially exponential mode sampled at
//! these positions, so they also hold for the discrete line of the Yee
//! scheme on a uniform mesh section (whose `γ` includes the numerical
//! dispersion and whose `Z_L` is the discrete wave impedance): the
//! extraction introduces no discretisation error of its own. This is why
//! the three voltage lines must be **equally spaced**. `β` comes from a
//! second difference of the voltages, so the f32 round-off of the fields
//! limits its relative accuracy to roughly `1e-8/(βd)²` at low frequencies.
//!
//! The fields launched by the feed that are not the line mode (radiation
//! and surface waves of an open structure, below-cutoff modes of a shielded
//! one) and the near fields of the structure must have decayed at the
//! probes. They decay slowly: in the microstrip validation (`h` = 0.8 mm,
//! `w/h` ≈ 2) probes 5·h past the feed read ε_eff 5 % high, at ≈ 11·h
//! within 1 %. Keep **about ten substrate heights** between the feed and
//! the middle of the port. Near frequencies where `V2` vanishes (a
//! standing-wave node at the middle probe) the extraction is
//! ill-conditioned.

use csx::primitives::BoxShape;
use csx::properties::{Excitation, LumpedElement, ProbeBox, ProbeType};
use csx::{ContinuousStructure, Primitive, Property, PropertyId, Scalar};

use super::line::{self, Layout, LinePortResult, choose_probes, line_index};
use crate::axis::Axis;
use crate::post::Complex64;
use crate::sim::Results;
use crate::{Error, Result};

/// The csx property names and ids of an added [`MslPort`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MslPortProperties {
    /// The strip metal (the given property, or `port_metal_<n>`).
    pub metal: (String, PropertyId),
    /// The feed resistor, if any: `port_resist_<n>`.
    pub element: Option<(String, PropertyId)>,
    /// The `Excitation` (if excited): `port_excite_<n>`.
    pub excitation: Option<(String, PropertyId)>,
    /// The voltage `ProbeBox`es `port_ut<n>A`, `B`, `C`, in propagation
    /// order.
    pub voltages: [(String, PropertyId); 3],
    /// The current `ProbeBox`es `port_it<n>A`, `B`, in propagation order.
    pub currents: [(String, PropertyId); 2],
}

/// A microstrip-line port written into a csx model: the strip over the
/// port length, an optional feed (excitation and/or resistor) at its start,
/// and three voltage and two current `ProbeBox`es; see the
/// [module documentation](self) for the analysis.
///
/// ```
/// use csx::ContinuousStructure;
/// use ems::port::MslPort;
/// use ems::Axis;
///
/// let mut model = ContinuousStructure::new();
/// model.grid.delta_unit = 1e-3;
/// model.grid.set_lines(0, (0..=20).map(f64::from))?;
/// model.grid.set_lines(1, (-5..=5).map(f64::from))?;
/// model.grid.set_lines(2, (0..=6).map(f64::from))?;
/// // Strip 2 mm wide, 1 mm above the ground (z = 0), from x = 0 to 10.
/// let port = MslPort::new(1, [0.0, -1.0, 0.0], [10.0, 1.0, 1.0], Axis::X, Axis::Z)
///     .excite(1.0)
///     .add_to(&mut model)?;
/// let p = port.properties().unwrap();
/// assert_eq!(p.voltages[1].0, "port_ut1B");
/// assert_eq!(port.probe_positions(), Some([4.0, 5.0, 6.0]));
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Geometry
///
/// * `start` and `stop` are opposite corners of the port box (drawing
///   units). Along the **propagation axis** `dir` the port runs from the
///   feed plane `start[dir]` to `stop[dir]`; the wave enters the
///   structure in that direction (`stop[dir] < start[dir]` points it
///   towards `−dir`).
/// * Along the **height axis** `height`, the ground is at `start[height]`
///   and the strip at `stop[height]` (either order). The ground itself is
///   not written: it is part of the model (a metal plane or a PEC
///   boundary).
/// * The remaining axis spans the strip width.
///
/// # Written properties
///
/// * The strip: a zero-thickness box over the port length and width in
///   the given metal property ([`metal`](Self::metal)) or in a new metal
///   `port_metal_<n>`.
/// * With [`feed_resistance`](Self::feed_resistance), a `LumpedElement`
///   resistor `port_resist_<n>` from the strip to the ground across the
///   strip width at the feed plane (a matched source or termination).
/// * With [`excite`](Self::excite), an E-field excitation `port_excite_<n>`
///   on the same box, directed from the strip to the ground with
///   `amplitude / h` V/m (`h` the strip height in metres). Over a feed
///   resistor it is a Thevenin source of `amplitude` volts (like
///   [`LumpedPort`](super::LumpedPort)); without one it is a soft source
///   that launches waves in both directions.
/// * Voltages `port_ut<n>A/B/C`: strip-to-ground lines (`V = V_strip −
///   V_ground`) at the three equally spaced mesh lines, past the feed and
///   not beyond `stop[dir]`, whose middle is nearest the middle of the
///   port, at the width-axis mesh line nearest the strip centre.
/// * Currents `port_it<n>A/B`: H loops around the strip (the width-axis
///   mesh lines within the strip, in its plane) on the dual planes between
///   those lines, counted positive along the propagation direction.
///
/// # Mesh requirements
///
/// The model must be meshed **before** [`add_to`](Self::add_to): the feed
/// plane and the ground and strip heights must be mesh lines
/// ([`mesh_lines`](Self::mesh_lines) lists them), at least one width-axis
/// line must lie within the strip, and the port must contain three equally
/// spaced mesh lines past the feed. `add_to` fails otherwise.
///
/// The strip edges need not be mesh lines. A zero-thickness strip whose
/// edges are mesh lines behaves as if it were wider (the field singularity
/// at the edge is smeared over the adjacent cells): with lines at the edges,
/// the line impedance comes out several percent low. The usual remedy is
/// the "thirds rule": the mesh lines nearest each edge at one third of a
/// cell inside the strip and two thirds outside.
///
/// # Reference plane
///
/// The waves are reported at `stop[dir]` (the end of the port towards the
/// structure), moved by [`reference_shift`](Self::reference_shift) along
/// the propagation direction.
#[derive(Debug, Clone, PartialEq)]
pub struct MslPort {
    number: u32,
    start: [f64; 3],
    stop: [f64; 3],
    dir: Axis,
    height: Axis,
    metal: Option<PropertyId>,
    feed_r: Option<f64>,
    amplitude: Option<f64>,
    delay: f64,
    priority: i32,
    shift: f64,
    props: Option<MslPortProperties>,
    layout: Option<Layout>,
}

impl MslPort {
    /// Port number `number` over the box `start`–`stop` (drawing units),
    /// propagating along `dir`, with the strip above the ground along
    /// `height` (see [Geometry](#geometry)).
    pub fn new(number: u32, start: [f64; 3], stop: [f64; 3], dir: Axis, height: Axis) -> Self {
        Self {
            number,
            start,
            stop,
            dir,
            height,
            metal: None,
            feed_r: None,
            amplitude: None,
            delay: 0.0,
            priority: 0,
            shift: 0.0,
            props: None,
            layout: None,
        }
    }

    /// Draws the strip in the existing metal property `id` instead of a
    /// new `port_metal_<n>`.
    pub fn metal(mut self, id: PropertyId) -> Self {
        self.metal = Some(id);
        self
    }

    /// Adds a feed resistor of `r` \[Ω\] at the feed plane; `r = 0` removes
    /// it (the default).
    pub fn feed_resistance(mut self, r: f64) -> Self {
        self.feed_r = (r != 0.0).then_some(r);
        self
    }

    /// Excites the port with `amplitude` \[V\] times the simulation
    /// waveform (see [Written properties](#written-properties)).
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

    /// Moves the reference plane by `shift` (drawing units) from
    /// `stop[dir]` along the propagation direction (positive: further into
    /// the structure). Default 0.
    pub fn reference_shift(mut self, shift: f64) -> Self {
        self.shift = shift;
        self
    }

    /// Port number.
    pub fn number(&self) -> u32 {
        self.number
    }

    /// Whether the port is excited.
    pub fn is_excited(&self) -> bool {
        self.amplitude.is_some()
    }

    /// The properties written by [`add_to`](Self::add_to) (`None` before).
    pub fn properties(&self) -> Option<&MslPortProperties> {
        self.props.as_ref()
    }

    /// The positions of the three voltage probes along the propagation
    /// axis (drawing units, propagation order), once added.
    pub fn probe_positions(&self) -> Option<[f64; 3]> {
        self.layout.map(|l| l.v_pos)
    }

    /// The coordinates (drawing units, per axis index) that must be mesh
    /// lines: the feed plane and `stop[dir]`, and the ground and strip
    /// heights. The port also needs three equally spaced lines between
    /// `start[dir]` and `stop[dir]` and at least one line within the strip
    /// width (see [Mesh requirements](#mesh-requirements)).
    pub fn mesh_lines(&self) -> [Vec<f64>; 3] {
        let mut out: [Vec<f64>; 3] = Default::default();
        for d in [self.dir.index(), self.height.index()] {
            out[d] = vec![self.start[d], self.stop[d]];
            out[d].sort_by(f64::total_cmp);
            out[d].dedup();
        }
        out
    }

    fn width_axis(&self) -> Result<Axis> {
        if self.dir == self.height {
            return Err(Error::Invalid(format!(
                "MSL port {}: propagation and height axes are both {}",
                self.number, self.dir
            )));
        }
        Ok(Axis::ALL
            .into_iter()
            .find(|&a| a != self.dir && a != self.height)
            .expect("three axes"))
    }

    /// Writes the port's properties into `model` and returns the port with
    /// their names and ids.
    ///
    /// # Errors
    /// [`Error::Invalid`] for equal propagation and height axes, a zero
    /// length or height, a negative width, non-finite values, a
    /// non-positive feed resistance, a port already added, a property name
    /// already present in the model, an unknown metal property, or a mesh
    /// that does not meet the [requirements](#mesh-requirements).
    pub fn add_to(mut self, model: &mut ContinuousStructure) -> Result<Self> {
        let n = self.number;
        if self.props.is_some() {
            return Err(Error::Invalid(format!("MSL port {n} was already added")));
        }
        if model.grid.coord_system != csx::CoordSystem::Cartesian {
            return Err(Error::Unsupported(format!(
                "MSL port {n}: MSL ports need a Cartesian grid"
            )));
        }
        let w_axis = self.width_axis()?;
        if self.start.iter().chain(&self.stop).any(|v| !v.is_finite())
            || !self.delay.is_finite()
            || !self.shift.is_finite()
            || self.amplitude.is_some_and(|a| !a.is_finite())
        {
            return Err(Error::Invalid(format!(
                "MSL port {n}: values must be finite"
            )));
        }
        if let Some(r) = self.feed_r
            && !(r.is_finite() && r > 0.0)
        {
            return Err(Error::Invalid(format!(
                "MSL port {n}: feed resistance {r} must be finite and > 0"
            )));
        }
        let (p, h, w) = (self.dir.index(), self.height.index(), w_axis.index());
        let du = model.grid.delta_unit;
        let (feed, end) = (self.start[p], self.stop[p]);
        if end == feed {
            return Err(Error::Invalid(format!(
                "MSL port {n} has zero length along {}",
                self.dir
            )));
        }
        let (h_gnd, h_strip) = (self.start[h], self.stop[h]);
        if h_gnd == h_strip {
            return Err(Error::Invalid(format!(
                "MSL port {n} has zero height along {}",
                self.height
            )));
        }
        let (w_lo, w_hi) = (
            self.start[w].min(self.stop[w]),
            self.start[w].max(self.stop[w]),
        );

        // Mesh checks.
        for (axis, x, what) in [
            (self.dir, feed, "feed plane"),
            (self.height, h_gnd, "ground"),
            (self.height, h_strip, "strip height"),
        ] {
            if line_index(model.grid.lines(axis.index()), x).is_none() {
                return Err(Error::Invalid(format!(
                    "MSL port {n}: the {what} {axis} = {x} is not a mesh line \
                     (mesh the model first; see MslPort::mesh_lines)"
                )));
            }
        }
        let layout =
            choose_probes(model.grid.lines(p), feed, end, self.shift, du).ok_or_else(|| {
                Error::Invalid(format!(
                    "MSL port {n}: needs three equally spaced mesh lines along {} \
                     past the feed at {feed} and not beyond {end}",
                    self.dir
                ))
            })?;
        // The width-axis lines within the strip (edges within tolerance),
        // and the one nearest its centre.
        let in_strip: Vec<f64> = model
            .grid
            .lines(w)
            .iter()
            .copied()
            .filter(|&x| {
                (w_lo..=w_hi).contains(&x)
                    || line_index(&[x], w_lo).is_some()
                    || line_index(&[x], w_hi).is_some()
            })
            .collect();
        let (Some(&s_lo), Some(&s_hi)) = (in_strip.first(), in_strip.last()) else {
            return Err(Error::Invalid(format!(
                "MSL port {n}: no {w_axis} mesh line within the strip {w_lo} ..= {w_hi}"
            )));
        };
        let w_mid = 0.5 * (w_lo + w_hi);
        let w_c = in_strip
            .iter()
            .copied()
            .min_by(|x, y| (x - w_mid).abs().total_cmp(&(y - w_mid).abs()))
            .expect("non-empty");

        // Names.
        let names = [
            format!("port_metal_{n}"),
            format!("port_resist_{n}"),
            format!("port_excite_{n}"),
            format!("port_ut{n}A"),
            format!("port_ut{n}B"),
            format!("port_ut{n}C"),
            format!("port_it{n}A"),
            format!("port_it{n}B"),
        ];
        let used = |i: usize| match i {
            0 => self.metal.is_none(),
            1 => self.feed_r.is_some(),
            2 => self.amplitude.is_some(),
            _ => true,
        };
        if let Some(dup) = names
            .iter()
            .enumerate()
            .find(|&(i, s)| used(i) && model.property_by_name(s).is_some())
        {
            return Err(Error::Invalid(format!(
                "the model already has a property named {:?}",
                dup.1
            )));
        }
        let metal = match self.metal {
            Some(id) => {
                let prop = model.property(id).ok_or_else(|| {
                    Error::Invalid(format!("MSL port {n}: no metal property {id}"))
                })?;
                (prop.name.clone(), id)
            }
            None => {
                let id = model.add_property(Property::metal(names[0].clone()));
                (names[0].clone(), id)
            }
        };

        let csx_axis = |a: usize| csx::Axis::from_index(a).expect("axis index in 0..3");
        let prim = |s: [f64; 3], t: [f64; 3]| {
            Primitive::new(BoxShape::new(s, t)).with_priority(self.priority)
        };
        let point = |xp: f64, xw: f64, xh: f64| {
            let mut v = [0.0; 3];
            v[p] = xp;
            v[w] = xw;
            v[h] = xh;
            v
        };

        // Strip.
        model.add_primitive(
            metal.1,
            prim(point(feed, w_lo, h_strip), point(end, w_hi, h_strip)),
        )?;

        // Feed: from the strip (+) to the ground.
        let (f0, f1) = (point(feed, w_lo, h_strip), point(feed, w_hi, h_gnd));
        let element = match self.feed_r {
            Some(r) => {
                let el = LumpedElement::resistor(csx_axis(h), r);
                let id = model.add_property(Property::lumped_element(names[1].clone(), el));
                model.add_primitive(id, prim(f0, f1))?;
                Some((names[1].clone(), id))
            }
            None => None,
        };
        let excitation = match self.amplitude {
            Some(amp) => {
                let height = (h_strip - h_gnd).abs() * du;
                let mut v = [0.0; 3];
                v[h] = (h_gnd - h_strip).signum() * amp / height;
                let mut ex = Excitation::new(v).with_number(n);
                ex.delay = Scalar::new(self.delay);
                let id = model.add_property(Property::excitation(names[2].clone(), ex));
                model.add_primitive(id, prim(f0, f1))?;
                Some((names[2].clone(), id))
            }
            None => None,
        };

        // Probes.
        let voltages = [0, 1, 2].map(|k| {
            let mut pb = ProbeBox::new(ProbeType::Voltage);
            pb.number = n;
            let id = model.add_property(Property::probe(names[3 + k].clone(), pb));
            (names[3 + k].clone(), id)
        });
        for (k, (_, id)) in voltages.iter().enumerate() {
            let x = layout.v_pos[k];
            model.add_primitive(*id, prim(point(x, w_c, h_strip), point(x, w_c, h_gnd)))?;
        }
        let currents = [0, 1].map(|k| {
            let mut pb = ProbeBox::new(ProbeType::Current).with_norm_dir(csx_axis(p));
            pb.number = n;
            let id = model.add_property(Property::probe(names[6 + k].clone(), pb));
            (names[6 + k].clone(), id)
        });
        for (k, (_, id)) in currents.iter().enumerate() {
            let (a, b) = (layout.v_pos[k], layout.v_pos[k + 1]);
            model.add_primitive(*id, prim(point(a, s_lo, h_strip), point(b, s_hi, h_strip)))?;
        }
        self.props = Some(MslPortProperties {
            metal,
            element,
            excitation,
            voltages,
            currents,
        });
        self.layout = Some(layout);
        Ok(self)
    }

    /// Frequency-domain analysis at `freqs` \[Hz\] with the extracted line
    /// impedance `Z_L(f)` as reference impedance.
    ///
    /// # Errors
    /// See [`analyze_with`](Self::analyze_with); also [`Error::Invalid`]
    /// if `Re Z_L` is not positive at some frequency (typically outside the
    /// excited band).
    pub fn analyze(&self, results: &Results, freqs: &[f64]) -> Result<MslPortResult> {
        let (p, l) = self.added()?;
        line::analyze(
            results,
            freqs,
            &p.probe_names(),
            l,
            None,
            "MSL",
            self.number,
        )
    }

    /// Frequency-domain analysis with the reference impedance `z_ref`.
    ///
    /// # Errors
    /// [`Error::Invalid`] if the port was not added, a probe is missing
    /// from `results` or has no samples, or `z_ref` is invalid.
    pub fn analyze_with(
        &self,
        results: &Results,
        freqs: &[f64],
        z_ref: Complex64,
    ) -> Result<MslPortResult> {
        let (p, l) = self.added()?;
        line::analyze(
            results,
            freqs,
            &p.probe_names(),
            l,
            Some(z_ref),
            "MSL",
            self.number,
        )
    }

    fn added(&self) -> Result<(&MslPortProperties, &Layout)> {
        match (&self.props, &self.layout) {
            (Some(p), Some(l)) => Ok((p, l)),
            _ => Err(Error::Invalid(format!(
                "MSL port {} was not added",
                self.number
            ))),
        }
    }
}

impl MslPortProperties {
    fn probe_names(&self) -> line::ProbeNames<'_> {
        line::ProbeNames {
            voltages: self.voltages.each_ref().map(|v| v.0.as_str()),
            currents: self.currents.each_ref().map(|v| v.0.as_str()),
        }
    }
}

/// The analysis of an [`MslPort`] (a [`LinePortResult`]).
pub type MslPortResult = LinePortResult;
