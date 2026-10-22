//! Symmetric strip-line port ([`StriplinePort`]).
//!
//! A strip midway between two ground planes carries a pure TEM mode in a
//! homogeneous dielectric. For a strip of zero thickness and width `w`
//! between grounds a distance `b` apart, the conformal mapping of Cohn
//! (S. B. Cohn, "Characteristic impedance of the shielded-strip
//! transmission line", IRE Trans. MTT-2, 1954) gives the exact
//!
//! ```text
//! Z0 = η0/(4·√ε_r) · K(k)/K(k'),   k = sech(πw/2b),   k' = tanh(πw/2b)
//! ```
//!
//! with `K` the complete elliptic integral of the first kind
//! (Pozar, *Microwave Engineering*, §3.7, quotes the curve fit of this
//! result). For a strip of finite thickness `t`, Wheeler's formula
//! (H. A. Wheeler, IEEE Trans. MTT-26, 1978; Wadell, *Transmission Line
//! Design Handbook*, §3.5) is the usual closed form; the port's strip is a
//! zero-thickness sheet, so [`StriplinePort::z0`] uses Cohn's.
//!
//! The probes and the extraction are those of the microstrip port (see the
//! [`msl`](super::msl) module): three strip-to-ground voltages at equally
//! spaced mesh lines and two H loops around the strip between them.

use core::f64::consts::PI;

use csx::primitives::BoxShape;
use csx::properties::{Excitation, LumpedElement, ProbeBox, ProbeType};
use csx::{ContinuousStructure, Primitive, Property, PropertyId, Scalar};

use super::elliptic::k_ratio_with;
use super::line::{self, Layout, LinePortResult, choose_probes, line_index};
use crate::axis::Axis;
use crate::constants::Z0;
use crate::post::Complex64;
use crate::sim::Results;
use crate::{Error, Result};

/// The csx property names and ids of an added [`StriplinePort`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StriplinePortProperties {
    /// The strip metal (the given property, or `port_metal_<n>`).
    pub metal: (String, PropertyId),
    /// The feed resistors (each `2R`), if any: `port_resist_<n>A` (strip
    /// to the lower ground) and `port_resist_<n>B` (strip to the upper
    /// ground).
    pub elements: Option<[(String, PropertyId); 2]>,
    /// The excitations (if excited): `port_excite_<n>A` (lower half) and
    /// `port_excite_<n>B` (upper half).
    pub excitations: Option<[(String, PropertyId); 2]>,
    /// The voltage `ProbeBox`es `port_ut<n>A`, `B`, `C` (strip to the
    /// lower ground), in propagation order.
    pub voltages: [(String, PropertyId); 3],
    /// The current `ProbeBox`es `port_it<n>A`, `B`, in propagation order.
    pub currents: [(String, PropertyId); 2],
}

impl StriplinePortProperties {
    fn probe_names(&self) -> line::ProbeNames<'_> {
        line::ProbeNames {
            voltages: self.voltages.each_ref().map(|v| v.0.as_str()),
            currents: self.currents.each_ref().map(|v| v.0.as_str()),
        }
    }
}

/// A symmetric strip-line port written into a csx model: the strip over
/// the port length midway between the grounds, an optional two-sided
/// feed at its start, and three voltage and two current `ProbeBox`es (the
/// line analysis of [`MslPort`](super::MslPort)).
///
/// ```
/// use csx::ContinuousStructure;
/// use ems::port::StriplinePort;
/// use ems::Axis;
///
/// let mut model = ContinuousStructure::new();
/// model.grid.delta_unit = 1e-3;
/// model.grid.set_lines(0, (0..=20).map(f64::from))?;
/// model.grid.set_lines(1, (-5..=5).map(f64::from))?;
/// model.grid.set_lines(2, (0..=4).map(f64::from))?;
/// // Strip 2 mm wide between grounds at z = 0 and z = 4 (strip at z = 2).
/// let port = StriplinePort::new(1, [0.0, -1.0, 0.0], [10.0, 1.0, 4.0], Axis::X, Axis::Z)
///     .excite(1.0)
///     .add_to(&mut model)?;
/// assert_eq!(port.probe_positions(), Some([4.0, 5.0, 6.0]));
/// // w/b = 0.5 in air: Cohn's exact Z0.
/// assert!((port.z0().unwrap() - 100.43).abs() < 0.01);
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Geometry
///
/// * `start` and `stop` are opposite corners of the port box (drawing
///   units). Along the propagation axis `dir` the port runs from the feed
///   plane `start[dir]` to `stop[dir]` (the wave enters the structure in
///   that direction).
/// * Along the height axis `height`, the two ground planes are at
///   `start[height]` (the "lower" ground) and `stop[height]` (the
///   "upper" one); the strip lies on the mid-plane between them. The
///   grounds and the dielectric are part of the model, not of the port.
/// * The remaining axis spans the strip width `w`.
/// * **Cylindrical grids** (`(ρ, α, z)` axes `X`, `Y`, `Z`): the same
///   boxes in `(ρ, α, z)`, for the lines that are uniform along their
///   propagation axis (as for [`MslPort`](super::MslPort)): along z
///   between two coaxial ground cylinders (height along ρ, the width an α
///   range: an angle, the arc at the strip radius), along α between them
///   (width along z) or between two `z` planes (width along ρ). Lines
///   along ρ and heights along α are refused. Along α the feed, `stop`,
///   [`reference_shift`](Self::reference_shift) and the probe positions
///   are angles and `γ` is per metre of arc at the strip radius (height
///   along ρ) or at the centre of the strip's ρ range (height along z).
///   [`z0`](Self::z0) is then Cohn's flat value with the arc width (a
///   large-radius approximation).
///
/// # Written properties
///
/// * The strip: a zero-thickness box on the mid-plane over the port length
///   and width, in the given metal ([`metal`](Self::metal)) or a new metal
///   `port_metal_<n>`.
/// * With [`feed_resistance`](Self::feed_resistance)`(R)`, two
///   `LumpedElement` resistors of `2R` at the feed plane, from the strip to
///   each ground across the strip width: in parallel, a source resistance
///   `R`.
/// * With [`excite`](Self::excite), two E excitations over the same boxes,
///   directed from the strip to each ground with `amplitude / (b/2)` V/m.
///   Over the feed resistors, each half is a Thevenin source of `amplitude`
///   volts behind `2R`, together `amplitude` behind `R`; without them it is
///   a soft source that launches waves in both directions.
/// * Voltages `port_ut<n>A/B/C` from the strip to the lower ground at
///   three equally spaced mesh lines (the width-axis line nearest the strip
///   centre), currents `port_it<n>A/B` as H loops around the strip on the
///   dual planes between them (as for [`MslPort`](super::MslPort)).
///
/// # Mesh requirements
///
/// The model must be meshed before [`add_to`](Self::add_to): the feed
/// plane, both grounds and the mid-plane must be mesh lines
/// ([`mesh_lines`](Self::mesh_lines)), at least one width-axis line must
/// lie within the strip, and the port must contain three equally spaced
/// lines past the feed. The strip edges need not be mesh lines; with lines
/// on the edges a zero-thickness strip acts wider (lower impedance), the
/// usual remedy is the "thirds rule" (lines at one third of a cell inside
/// and two thirds outside each edge).
///
/// # Reference plane
///
/// At `stop[dir]`, moved by [`reference_shift`](Self::reference_shift)
/// along the propagation direction.
#[derive(Debug, Clone, PartialEq)]
pub struct StriplinePort {
    number: u32,
    start: [f64; 3],
    stop: [f64; 3],
    dir: Axis,
    height: Axis,
    eps_r: f64,
    metal: Option<PropertyId>,
    feed_r: Option<f64>,
    amplitude: Option<f64>,
    delay: f64,
    priority: i32,
    shift: f64,
    /// `[w, b]` in metres, set by `add_to`.
    dims: Option<[f64; 2]>,
    props: Option<StriplinePortProperties>,
    layout: Option<Layout>,
}

impl StriplinePort {
    /// Port number `number` over the box `start`–`stop` (drawing units),
    /// propagating along `dir`, with the grounds at `start[height]` and
    /// `stop[height]` (see [Geometry](#geometry)).
    pub fn new(number: u32, start: [f64; 3], stop: [f64; 3], dir: Axis, height: Axis) -> Self {
        Self {
            number,
            start,
            stop,
            dir,
            height,
            eps_r: 1.0,
            metal: None,
            feed_r: None,
            amplitude: None,
            delay: 0.0,
            priority: 0,
            shift: 0.0,
            dims: None,
            props: None,
            layout: None,
        }
    }

    /// Relative permittivity of the dielectric between the grounds
    /// (default 1). Used for [`z0`](Self::z0) only; the material itself is
    /// part of the model.
    pub fn eps_r(mut self, eps_r: f64) -> Self {
        self.eps_r = eps_r;
        self
    }

    /// Draws the strip in the existing metal property `id` instead of a
    /// new `port_metal_<n>`.
    pub fn metal(mut self, id: PropertyId) -> Self {
        self.metal = Some(id);
        self
    }

    /// Adds a feed of source resistance `r` \[Ω\] (two resistors `2r`, one
    /// to each ground); `r = 0` removes it (the default).
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
    /// `stop[dir]` along the propagation direction. Default 0.
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
    pub fn properties(&self) -> Option<&StriplinePortProperties> {
        self.props.as_ref()
    }

    /// The positions of the three voltage probes along the propagation
    /// axis (drawing units, propagation order), once added.
    pub fn probe_positions(&self) -> Option<[f64; 3]> {
        self.layout.map(|l| l.v_pos)
    }

    /// Strip width `w` and ground spacing `b` \[m\] (after
    /// [`add_to`](Self::add_to)).
    pub fn dimensions(&self) -> Option<[f64; 2]> {
        self.dims
    }

    /// Cohn's exact characteristic impedance of the zero-thickness strip
    /// \[Ω\] (see the [module docs](self)); `None` before
    /// [`add_to`](Self::add_to).
    pub fn z0(&self) -> Option<f64> {
        let [w, b] = self.dims?;
        Some(Self::z0_of(w, b, self.eps_r))
    }

    /// Cohn's exact impedance \[Ω\] of a zero-thickness strip of width `w`
    /// midway between grounds `b` apart (one common unit) in a dielectric
    /// `eps_r` (see the [module docs](self)).
    pub fn z0_of(w: f64, b: f64, eps_r: f64) -> f64 {
        cohn_z0(w, b, eps_r)
    }

    /// The coordinates (drawing units, per axis index) that must be mesh
    /// lines: the feed plane and `stop[dir]`, both grounds and the
    /// mid-plane (see [Mesh requirements](#mesh-requirements)).
    pub fn mesh_lines(&self) -> [Vec<f64>; 3] {
        let mut out: [Vec<f64>; 3] = Default::default();
        let (p, h) = (self.dir.index(), self.height.index());
        out[p] = vec![self.start[p], self.stop[p]];
        out[h] = vec![
            self.start[h],
            0.5 * (self.start[h] + self.stop[h]),
            self.stop[h],
        ];
        for v in &mut out {
            v.sort_by(f64::total_cmp);
            v.dedup();
        }
        out
    }

    /// Writes the port's properties into `model` and returns the port with
    /// their names and ids.
    ///
    /// # Errors
    /// [`Error::Unsupported`] on a cylindrical grid for a line along ρ or
    /// a height along α; [`Error::Invalid`]
    /// for equal propagation and height axes, a zero length, height or
    /// width, a non-positive reference radius, non-finite values, a non-positive feed resistance or
    /// permittivity, a port already added, a property name already present
    /// in the model, an unknown metal property, or a mesh that does not
    /// meet the [requirements](#mesh-requirements).
    pub fn add_to(mut self, model: &mut ContinuousStructure) -> Result<Self> {
        let n = self.number;
        let err = |m: String| Error::Invalid(format!("stripline port {n}: {m}"));
        if self.props.is_some() {
            return Err(err("already added".into()));
        }
        if self.dir == self.height {
            return Err(err(format!(
                "propagation and height axes are both {}",
                self.dir
            )));
        }
        if self.start.iter().chain(&self.stop).any(|v| !v.is_finite())
            || !self.delay.is_finite()
            || !self.shift.is_finite()
            || self.amplitude.is_some_and(|a| !a.is_finite())
        {
            return Err(err("values must be finite".into()));
        }
        if !(self.eps_r.is_finite() && self.eps_r > 0.0) {
            return Err(err(format!("ε_r = {} must be finite and > 0", self.eps_r)));
        }
        if let Some(r) = self.feed_r
            && !(r.is_finite() && r > 0.0)
        {
            return Err(err(format!("feed resistance {r} must be finite and > 0")));
        }
        let cyl = line::cylindrical_line(model, self.dir, self.height, "stripline", n)?;
        let w_axis = Axis::ALL
            .into_iter()
            .find(|&a| a != self.dir && a != self.height)
            .expect("three axes");
        let (p, h, w) = (self.dir.index(), self.height.index(), w_axis.index());
        let du = model.grid.delta_unit;
        let (feed, end) = (self.start[p], self.stop[p]);
        if end == feed {
            return Err(err(format!("zero length along {}", self.dir)));
        }
        let (g_lo, g_hi) = (self.start[h], self.stop[h]);
        let mid = 0.5 * (g_lo + g_hi);
        if g_lo == g_hi {
            return Err(err(format!("zero ground spacing along {}", self.height)));
        }
        let (w_lo, w_hi) = (
            self.start[w].min(self.stop[w]),
            self.start[w].max(self.stop[w]),
        );
        if w_hi <= w_lo {
            return Err(err(format!("zero strip width along {w_axis}")));
        }
        for (axis, x, what) in [
            (self.dir, feed, "feed plane"),
            (self.height, g_lo, "lower ground"),
            (self.height, g_hi, "upper ground"),
            (self.height, mid, "strip mid-plane"),
        ] {
            if line_index(model.grid.lines(axis.index()), x).is_none() {
                return Err(err(format!(
                    "the {what} {axis} = {x} is not a mesh line (mesh the model first; \
                     see StriplinePort::mesh_lines)"
                )));
            }
        }
        // Cylindrical grids: a width along α is an angle (the arc at the
        // strip radius), and along α the line coordinate is the arc at the
        // strip radius (height along ρ) or at the centre of the strip's ρ
        // range (height along z).
        let arc = |r: f64| -> Result<f64> {
            if r.is_nan() || r <= 0.0 {
                return Err(err(format!("the reference radius {r} must be > 0")));
            }
            Ok(r)
        };
        let w_scale = if cyl && w_axis == Axis::Y {
            arc(mid)?
        } else {
            1.0
        };
        let du_line = if cyl && self.dir == Axis::Y {
            let r = if self.height == Axis::X {
                mid
            } else {
                0.5 * (w_lo + w_hi)
            };
            du * arc(r)?
        } else {
            du
        };
        let layout = choose_probes(model.grid.lines(p), feed, end, self.shift, du_line)
            .ok_or_else(|| {
                err(format!(
                    "needs three equally spaced mesh lines along {} past the feed at {feed} \
                     and not beyond {end}",
                    self.dir
                ))
            })?;
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
            return Err(err(format!(
                "no {w_axis} mesh line within the strip {w_lo} ..= {w_hi}"
            )));
        };
        let w_mid = 0.5 * (w_lo + w_hi);
        let w_c = in_strip
            .iter()
            .copied()
            .min_by(|x, y| (x - w_mid).abs().total_cmp(&(y - w_mid).abs()))
            .expect("non-empty");

        let names = [
            format!("port_metal_{n}"),
            format!("port_resist_{n}A"),
            format!("port_resist_{n}B"),
            format!("port_excite_{n}A"),
            format!("port_excite_{n}B"),
            format!("port_ut{n}A"),
            format!("port_ut{n}B"),
            format!("port_ut{n}C"),
            format!("port_it{n}A"),
            format!("port_it{n}B"),
        ];
        let used = |i: usize| match i {
            0 => self.metal.is_none(),
            1 | 2 => self.feed_r.is_some(),
            3 | 4 => self.amplitude.is_some(),
            _ => true,
        };
        if let Some((_, dup)) = names
            .iter()
            .enumerate()
            .find(|&(i, s)| used(i) && model.property_by_name(s).is_some())
        {
            return Err(Error::Invalid(format!(
                "the model already has a property named {dup:?}"
            )));
        }
        let metal = match self.metal {
            Some(id) => {
                let prop = model
                    .property(id)
                    .ok_or_else(|| err(format!("no metal property {id}")))?;
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
        model.add_primitive(metal.1, prim(point(feed, w_lo, mid), point(end, w_hi, mid)))?;

        // Feed halves: strip (+) to the lower and to the upper ground.
        let halves = [g_lo, g_hi].map(|g| (point(feed, w_lo, mid), point(feed, w_hi, g)));
        let elements = match self.feed_r {
            Some(r) => {
                let mut out = Vec::with_capacity(2);
                for (k, &(a, b)) in halves.iter().enumerate() {
                    let el = LumpedElement::resistor(csx_axis(h), 2.0 * r);
                    let name = names[1 + k].clone();
                    let id = model.add_property(Property::lumped_element(name.clone(), el));
                    model.add_primitive(id, prim(a, b))?;
                    out.push((name, id));
                }
                Some(<[_; 2]>::try_from(out).expect("two halves"))
            }
            None => None,
        };
        let excitations = match self.amplitude {
            Some(amp) => {
                let half = 0.5 * (g_hi - g_lo).abs() * du;
                let mut out = Vec::with_capacity(2);
                for (k, (&(a, b), g)) in halves.iter().zip([g_lo, g_hi]).enumerate() {
                    let mut v = [0.0; 3];
                    v[h] = (g - mid).signum() * amp / half;
                    let mut ex = Excitation::new(v).with_number(n);
                    ex.delay = Scalar::new(self.delay);
                    let name = names[3 + k].clone();
                    let id = model.add_property(Property::excitation(name.clone(), ex));
                    model.add_primitive(id, prim(a, b))?;
                    out.push((name, id));
                }
                Some(<[_; 2]>::try_from(out).expect("two halves"))
            }
            None => None,
        };

        let voltages = [0, 1, 2].map(|k| {
            let mut pb = ProbeBox::new(ProbeType::Voltage);
            pb.number = n;
            let id = model.add_property(Property::probe(names[5 + k].clone(), pb));
            (names[5 + k].clone(), id)
        });
        for (k, (_, id)) in voltages.iter().enumerate() {
            let x = layout.v_pos[k];
            model.add_primitive(*id, prim(point(x, w_c, mid), point(x, w_c, g_lo)))?;
        }
        let currents = [0, 1].map(|k| {
            let mut pb = ProbeBox::new(ProbeType::Current).with_norm_dir(csx_axis(p));
            pb.number = n;
            let id = model.add_property(Property::probe(names[8 + k].clone(), pb));
            (names[8 + k].clone(), id)
        });
        for (k, (_, id)) in currents.iter().enumerate() {
            let (a, b) = (layout.v_pos[k], layout.v_pos[k + 1]);
            model.add_primitive(*id, prim(point(a, s_lo, mid), point(b, s_hi, mid)))?;
        }
        self.dims = Some([(w_hi - w_lo) * du * w_scale, (g_hi - g_lo).abs() * du]);
        self.props = Some(StriplinePortProperties {
            metal,
            elements,
            excitations,
            voltages,
            currents,
        });
        self.layout = Some(layout);
        Ok(self)
    }

    /// Frequency-domain analysis at `freqs` \[Hz\] with the extracted line
    /// impedance `Z_L(f)` as reference impedance (see
    /// [`MslPort::analyze`](super::MslPort::analyze)).
    ///
    /// # Errors
    /// See [`analyze_with`](Self::analyze_with); also [`Error::Invalid`]
    /// if `Re Z_L` is not positive at some frequency.
    pub fn analyze(&self, results: &Results, freqs: &[f64]) -> Result<LinePortResult> {
        let (p, l) = self.added()?;
        line::analyze(
            results,
            freqs,
            &p.probe_names(),
            l,
            None,
            "stripline",
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
    ) -> Result<LinePortResult> {
        let (p, l) = self.added()?;
        line::analyze(
            results,
            freqs,
            &p.probe_names(),
            l,
            Some(z_ref),
            "stripline",
            self.number,
        )
    }

    fn added(&self) -> Result<(&StriplinePortProperties, &Layout)> {
        match (&self.props, &self.layout) {
            (Some(p), Some(l)) => Ok((p, l)),
            _ => Err(Error::Invalid(format!(
                "stripline port {} was not added",
                self.number
            ))),
        }
    }
}

/// Cohn's zero-thickness strip-line impedance for strip width `w` and
/// ground spacing `b` (any common unit).
fn cohn_z0(w: f64, b: f64, eps_r: f64) -> f64 {
    let x = PI * w / (2.0 * b);
    // k = sech x, k' = tanh x, both without cancellation.
    let (k, kp) = (1.0 / x.cosh(), x.tanh());
    Z0 / (4.0 * eps_r.sqrt()) * k_ratio_with(k, kp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohn_limits() {
        // Wide strip: two parallel-plate halves, Z → η·b/(4w) (plus the
        // fringing capacitance, which lowers it).
        let z = cohn_z0(20.0, 1.0, 1.0);
        let pp = Z0 / 4.0 / 20.0;
        assert!(z < pp && z > 0.9 * pp, "{z} vs {pp}");
        // Large w/b: K(k) → π/2, K(k') → ln(4/k) = πw/2b + ln 2, so
        // Z = η/(4·(w/b + 2·ln2/π)) (the fringing of the two edges).
        let z = cohn_z0(2.0, 1.0, 1.0);
        let wide = Z0 / (4.0 * (2.0 + 2.0 * 2f64.ln() / PI));
        assert!((z - wide).abs() < 0.01 * wide, "{z} vs {wide}");
        // Scaling with ε_r.
        assert!((cohn_z0(1.0, 1.0, 4.0) - 0.5 * cohn_z0(1.0, 1.0, 1.0)).abs() < 1e-12);
    }
}
