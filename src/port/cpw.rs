//! Coplanar-waveguide (CPW) port ([`CpwPort`]).
//!
//! A centre strip of width `S` between two coplanar ground planes, with
//! gaps `W`, on a substrate of thickness `h` and permittivity `ε_r`
//! (optionally backed by a ground plane at the substrate's far side).
//!
//! # Quasi-static impedance
//!
//! For zero-thickness metal and grounds of unbounded width, conformal
//! mapping gives (C. P. Wen, "Coplanar waveguide: a surface strip
//! transmission line suitable for nonreciprocal gyromagnetic device
//! applications", IEEE Trans. MTT-17, 1969; R. N. Simons, *Coplanar
//! Waveguide Circuits, Components, and Systems*, Wiley 2001, §2.2 and
//! §2.4), with `K` the complete elliptic integral of the first kind and
//! `k' = √(1 − k²)`:
//!
//! ```text
//! k0 = S/(S + 2W)
//!
//! conventional CPW (substrate of thickness h, air on both sides of it):
//!   k1    = sinh(πS/4h) / sinh(π(S + 2W)/4h)
//!   ε_eff = 1 + (ε_r − 1)/2 · K(k1)/K(k1') · K(k0')/K(k0)
//!   Z0    = 30π/√ε_eff · K(k0')/K(k0)
//!
//! conductor-backed CPW (ground plane at the far side of the substrate):
//!   k3    = tanh(πS/4h) / tanh(π(S + 2W)/4h)
//!   q     = K(k3)/K(k3') · K(k0')/K(k0)
//!   ε_eff = (1 + ε_r·q) / (1 + q)
//!   Z0    = 60π/√ε_eff / (K(k0)/K(k0') + K(k3)/K(k3'))
//! ```
//!
//! (`30π` and `60π` are `η0/4` and `η0/2`; the port uses `η0`.) In air
//! the conventional CPW is an exact TEM line, `ε_eff = 1`. The formulas
//! neglect the finite ground width and metal thickness; grounds a few
//! times `S + 2W` wide are close to the limit.
//!
//! # Probes
//!
//! The extraction is that of the microstrip port (see the
//! [`msl`](super::msl) module): three strip-to-ground voltages across one
//! gap at equally spaced mesh lines and two H loops around the centre
//! strip between them.

use core::f64::consts::PI;

use csx::primitives::BoxShape;
use csx::properties::{Excitation, LumpedElement, ProbeBox, ProbeType};
use csx::{ContinuousStructure, Primitive, Property, PropertyId, Scalar};

use super::elliptic::k_ratio;
use super::line::{self, Layout, LinePortResult, choose_probes, line_index};
use crate::axis::Axis;
use crate::constants::Z0;
use crate::post::Complex64;
use crate::sim::Results;
use crate::{Error, Result};

/// The csx property names and ids of an added [`CpwPort`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpwPortProperties {
    /// The metal of the centre strip and the coplanar grounds (the given
    /// property, or `port_metal_<n>`).
    pub metal: (String, PropertyId),
    /// The feed resistors (each `2R`), if any: `port_resist_<n>A` across
    /// the gap on the `+width` side of the strip, `port_resist_<n>B`
    /// across the `−width` one.
    pub elements: Option<[(String, PropertyId); 2]>,
    /// The excitations (if excited): `port_excite_<n>A` (`+width` gap) and
    /// `port_excite_<n>B` (`−width` gap).
    pub excitations: Option<[(String, PropertyId); 2]>,
    /// The voltage `ProbeBox`es `port_ut<n>A`, `B`, `C` (strip to the
    /// `+width` ground), in propagation order.
    pub voltages: [(String, PropertyId); 3],
    /// The current `ProbeBox`es `port_it<n>A`, `B` (around the centre
    /// strip), in propagation order.
    pub currents: [(String, PropertyId); 2],
}

impl CpwPortProperties {
    fn probe_names(&self) -> line::ProbeNames<'_> {
        line::ProbeNames {
            voltages: self.voltages.each_ref().map(|v| v.0.as_str()),
            currents: self.currents.each_ref().map(|v| v.0.as_str()),
        }
    }
}

/// A coplanar-waveguide port written into a csx model: the centre strip
/// and both coplanar grounds over the port length, an optional feed
/// across the two gaps at its start, and three voltage and two current
/// `ProbeBox`es (the line analysis of [`MslPort`](super::MslPort)); see
/// the [module docs](self) for the quasi-static reference.
///
/// ```
/// use csx::ContinuousStructure;
/// use ems::port::CpwPort;
/// use ems::Axis;
///
/// let mut model = ContinuousStructure::new();
/// model.grid.delta_unit = 1e-3;
/// model.grid.set_lines(0, (0..=20).map(f64::from))?;
/// model.grid.set_lines(1, (-12..=12).map(|i| f64::from(i) * 0.5))?;
/// model.grid.set_lines(2, (0..=4).map(|i| f64::from(i) * 0.5))?;
/// // Centre strip 2 mm, gaps 1 mm, grounds out to |y| = 6 mm, on a
/// // 1 mm substrate (z = 0 … 1, metal at z = 1).
/// let port = CpwPort::new(1, [0.0, -6.0, 0.0], [10.0, 6.0, 1.0], Axis::X, Axis::Z, 2.0, 1.0)
///     .eps_r(4.0)
///     .excite(1.0)
///     .add_to(&mut model)?;
/// let p = port.properties().unwrap();
/// assert_eq!(p.excitations.as_ref().unwrap()[1].0, "port_excite_1B");
/// let (z0, eps_eff) = port.quasi_static().unwrap();
/// // k0 = 1/2; the 1 mm substrate gives ε_eff below (ε_r + 1)/2.
/// assert!((z0 - 85.02).abs() < 0.01 && (eps_eff - 2.008).abs() < 1e-3);
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Geometry
///
/// * `start` and `stop` are opposite corners of the port box (drawing
///   units). Along the propagation axis `dir` the port runs from the feed
///   plane `start[dir]` to `stop[dir]` (the wave enters the structure in
///   that direction).
/// * Along the height axis `height`, the metal lies on the plane
///   `stop[height]`; `start[height]` is the substrate's far side, so
///   `h = |stop[height] − start[height]|` is the substrate thickness (used
///   only for the quasi-static reference; it may be 0 for a CPW in air)
///   and, for a [`backed`](Self::backed) CPW, the position of the backing
///   ground. The substrate and the backing are part of the model.
/// * The remaining (width) axis: the box spans the whole metal. The centre
///   strip of width `S` (`centre_width`) is centred in the box, the gaps
///   `W` (`gap`) are on both sides of it, and the grounds fill the rest of
///   the box on each side (their width must be positive).
/// * **Cylindrical grids** (`(ρ, α, z)` axes `X`, `Y`, `Z`): the same
///   boxes in `(ρ, α, z)`, for the lines that are uniform along their
///   propagation axis (as for [`MslPort`](super::MslPort)): along z with
///   the metal on a `ρ` surface (widths and gaps angles, arcs at the
///   metal radius), along α with the metal on a `ρ` surface (width along
///   z) or on a `z` plane (width along ρ). Lines along ρ and metal planes
///   normal to α are refused. Along α the feed, `stop`,
///   [`reference_shift`](Self::reference_shift) and the probe positions
///   are angles and `γ` is per metre of arc at the metal radius (metal on
///   a `ρ` surface) or at the centre of the box's ρ range (`z` plane).
///   [`quasi_static`](Self::quasi_static) is then the flat CPW with the
///   arc lengths (a large-radius approximation).
///
/// # Written properties
///
/// * The centre strip and the two grounds: zero-thickness boxes on the
///   metal plane over the port length, in the given metal
///   ([`metal`](Self::metal)) or a new metal `port_metal_<n>`.
/// * With [`feed_resistance`](Self::feed_resistance)`(R)`, two
///   `LumpedElement` resistors `2R` along the width axis across the two
///   gaps at the feed plane (in parallel, a source resistance `R`).
/// * With [`excite`](Self::excite), E excitations over the same two gap
///   lines, directed from the strip to each ground with `amplitude / W`
///   V/m: over the resistors, a Thevenin source of `amplitude` volts
///   behind `R`; without them a soft source radiating both ways.
/// * Voltages `port_ut<n>A/B/C`: from the width-axis line nearest the
///   strip centre to the first line on the `+width` ground, along the
///   width axis on the metal plane, at three equally spaced mesh lines.
/// * Currents `port_it<n>A/B`: H loops around the centre strip (the
///   width-axis lines within it) on the dual planes between them.
///
/// The feed covers the mesh edges whose midpoints lie in the gaps (csx
/// semantics of a line primitive), which are also the edges not covered
/// by the metal, so it is consistent with the strip and grounds whether
/// or not their edges are mesh lines.
///
/// # Mesh requirements
///
/// The model must be meshed before [`add_to`](Self::add_to): the feed
/// plane and the metal plane must be mesh lines
/// ([`mesh_lines`](Self::mesh_lines)), at least one width-axis line must
/// lie within the centre strip and one on the `+width` ground, and the
/// port must contain three equally spaced lines past the feed. The metal
/// edges need not be mesh lines; with lines on them, the zero-thickness
/// strip and grounds act as if the gaps were narrower (lower impedance);
/// the usual remedy is the "thirds rule" at each of the four edges (lines
/// one third of a cell inside the metal and two thirds outside). Keep each
/// metal edge within half a cell past the last mesh line in the metal: a
/// metal edge further out (beyond the midpoint of the next cell) makes the
/// transverse mesh edge that it cuts PEC (its midpoint is in the metal)
/// while the line beyond keeps its free edges along the line, so the
/// metal border is no equipotential and the line is a slow-wave structure
/// (`ε_eff` 1.4 in air for 2-cell gaps, design §5.5).
///
/// # Reference plane
///
/// At `stop[dir]`, moved by [`reference_shift`](Self::reference_shift)
/// along the propagation direction.
#[derive(Debug, Clone, PartialEq)]
pub struct CpwPort {
    number: u32,
    start: [f64; 3],
    stop: [f64; 3],
    dir: Axis,
    height: Axis,
    centre_width: f64,
    gap: f64,
    eps_r: f64,
    backed: bool,
    metal: Option<PropertyId>,
    feed_r: Option<f64>,
    amplitude: Option<f64>,
    delay: f64,
    priority: i32,
    shift: f64,
    /// `[S, W, h]` in metres, set by `add_to`.
    dims: Option<[f64; 3]>,
    props: Option<CpwPortProperties>,
    layout: Option<Layout>,
}

impl CpwPort {
    /// Port number `number` over the box `start`–`stop` (drawing units),
    /// propagating along `dir`, with the metal on the plane
    /// `stop[height]`, a centre strip `centre_width` wide and gaps `gap`
    /// (drawing units); see [Geometry](#geometry).
    pub fn new(
        number: u32,
        start: [f64; 3],
        stop: [f64; 3],
        dir: Axis,
        height: Axis,
        centre_width: f64,
        gap: f64,
    ) -> Self {
        Self {
            number,
            start,
            stop,
            dir,
            height,
            centre_width,
            gap,
            eps_r: 1.0,
            backed: false,
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

    /// Substrate permittivity (default 1), for the quasi-static reference
    /// only; the substrate itself is part of the model.
    pub fn eps_r(mut self, eps_r: f64) -> Self {
        self.eps_r = eps_r;
        self
    }

    /// Uses the conductor-backed formula (a ground plane at
    /// `start[height]`, which the model provides).
    pub fn backed(mut self) -> Self {
        self.backed = true;
        self
    }

    /// Draws the strip and grounds in the existing metal property `id`
    /// instead of a new `port_metal_<n>`.
    pub fn metal(mut self, id: PropertyId) -> Self {
        self.metal = Some(id);
        self
    }

    /// Adds a feed of source resistance `r` \[Ω\] (two resistors `2r`
    /// across the gaps); `r = 0` removes it (the default).
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
    pub fn properties(&self) -> Option<&CpwPortProperties> {
        self.props.as_ref()
    }

    /// The positions of the three voltage probes along the propagation
    /// axis (drawing units, propagation order), once added.
    pub fn probe_positions(&self) -> Option<[f64; 3]> {
        self.layout.map(|l| l.v_pos)
    }

    /// The coordinates (drawing units, per axis index) that must be mesh
    /// lines: the feed plane and `stop[dir]`, and the metal plane.
    pub fn mesh_lines(&self) -> [Vec<f64>; 3] {
        let mut out: [Vec<f64>; 3] = Default::default();
        let (p, h) = (self.dir.index(), self.height.index());
        out[p] = vec![self.start[p], self.stop[p]];
        out[p].sort_by(f64::total_cmp);
        out[p].dedup();
        out[h] = vec![self.stop[h]];
        out
    }

    /// The quasi-static `(Z0 [Ω], ε_eff)` of the port's cross section
    /// (after [`add_to`](Self::add_to)); see the [module docs](self).
    pub fn quasi_static(&self) -> Option<(f64, f64)> {
        let [s, w, h] = self.dims?;
        Some(Self::quasi_static_of(s, w, h, self.eps_r, self.backed))
    }

    /// The quasi-static `(Z0 [Ω], ε_eff)` of a CPW with centre width `s`,
    /// gaps `w`, substrate thickness `h` (one common unit; `h = 0` means no
    /// substrate, allowed only without backing) and permittivity `eps_r`,
    /// conductor-backed if `backed` (see the [module docs](self)).
    ///
    /// # Panics
    /// If `backed` and `h = 0`.
    pub fn quasi_static_of(s: f64, w: f64, h: f64, eps_r: f64, backed: bool) -> (f64, f64) {
        let k0 = s / (s + 2.0 * w);
        let r0 = k_ratio(k0); // K(k0)/K(k0')
        if backed {
            assert!(h > 0.0, "a conductor-backed CPW needs h > 0");
            let k3 = (PI * s / (4.0 * h)).tanh() / (PI * (s + 2.0 * w) / (4.0 * h)).tanh();
            let r3 = k_ratio(k3);
            let q = r3 / r0;
            let e = (1.0 + eps_r * q) / (1.0 + q);
            (Z0 / 2.0 / e.sqrt() / (r0 + r3), e)
        } else {
            let e = if h > 0.0 {
                // sinh ratio via exponentials (no overflow for small h).
                let (a, b) = (PI * s / (4.0 * h), PI * (s + 2.0 * w) / (4.0 * h));
                let k1 = (a - b).exp() * (1.0 - (-2.0 * a).exp()) / (1.0 - (-2.0 * b).exp());
                1.0 + 0.5 * (eps_r - 1.0) * k_ratio(k1) / r0
            } else {
                1.0
            };
            (Z0 / 4.0 / e.sqrt() / r0, e)
        }
    }

    /// Writes the port's properties into `model` and returns the port with
    /// their names and ids.
    ///
    /// # Errors
    /// [`Error::Unsupported`] on a cylindrical grid for a line along ρ or
    /// a metal plane normal to α; [`Error::Invalid`]
    /// for equal propagation and height axes, a zero length, non-positive
    /// strip or gap widths, a non-positive reference radius, a box too narrow for the grounds, `h = 0` with
    /// [`backed`](Self::backed), non-finite values, a non-positive feed
    /// resistance or permittivity, a port already added, a property name
    /// already present in the model, an unknown metal property, or a mesh
    /// that does not meet the [requirements](#mesh-requirements).
    pub fn add_to(mut self, model: &mut ContinuousStructure) -> Result<Self> {
        let n = self.number;
        let err = |m: String| Error::Invalid(format!("CPW port {n}: {m}"));
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
            || !self.centre_width.is_finite()
            || !self.gap.is_finite()
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
        if !(self.centre_width > 0.0 && self.gap > 0.0) {
            return Err(err(format!(
                "centre width {} and gap {} must be > 0",
                self.centre_width, self.gap
            )));
        }
        let cyl = line::cylindrical_line(model, self.dir, self.height, "CPW", n)?;
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
        let hm = self.stop[h];
        // Cylindrical grids: widths along α are angles (arcs at the metal
        // radius), and along α the line coordinate is the arc at the
        // metal radius (height along ρ) or at the centre of the box's ρ
        // range (height along z).
        let arc = |r: f64| -> Result<f64> {
            if r.is_nan() || r <= 0.0 {
                return Err(err(format!("the reference radius {r} must be > 0")));
            }
            Ok(r)
        };
        let w_scale = if cyl && w_axis == Axis::Y {
            arc(hm)?
        } else {
            1.0
        };
        let du_line = if cyl && self.dir == Axis::Y {
            let r = if self.height == Axis::X {
                hm
            } else {
                0.5 * (self.start[w] + self.stop[w])
            };
            du * arc(r)?
        } else {
            du
        };
        let sub_h = (self.stop[h] - self.start[h]).abs();
        if self.backed && sub_h == 0.0 {
            return Err(err(
                "a conductor-backed CPW needs a substrate thickness > 0".into(),
            ));
        }
        let (w_lo, w_hi) = (
            self.start[w].min(self.stop[w]),
            self.start[w].max(self.stop[w]),
        );
        let c = 0.5 * (w_lo + w_hi);
        let (sw, gap) = (self.centre_width, self.gap);
        // Strip [e1, e2], gaps [e0, e1] and [e2, e3], grounds outside.
        let (e1, e2) = (c - 0.5 * sw, c + 0.5 * sw);
        let (e0, e3) = (e1 - gap, e2 + gap);
        if !(w_lo < e0 && e3 < w_hi) {
            return Err(err(format!(
                "the box {w_lo} ..= {w_hi} along {w_axis} leaves no room for the grounds \
                 (strip and gaps span {e0} ..= {e3})"
            )));
        }
        for (axis, x, what) in [
            (self.dir, feed, "feed plane"),
            (self.height, hm, "metal plane"),
        ] {
            if line_index(model.grid.lines(axis.index()), x).is_none() {
                return Err(err(format!(
                    "the {what} {axis} = {x} is not a mesh line (mesh the model first; \
                     see CpwPort::mesh_lines)"
                )));
            }
        }
        let layout = choose_probes(model.grid.lines(p), feed, end, self.shift, du_line)
            .ok_or_else(|| {
                err(format!(
                    "needs three equally spaced mesh lines along {} past the feed at {feed} \
                     and not beyond {end}",
                    self.dir
                ))
            })?;
        let within = |x: f64, a: f64, b: f64| {
            (a..=b).contains(&x) || line_index(&[x], a).is_some() || line_index(&[x], b).is_some()
        };
        let wl = model.grid.lines(w);
        let in_strip: Vec<f64> = wl.iter().copied().filter(|&x| within(x, e1, e2)).collect();
        let (Some(&s_lo), Some(&s_hi)) = (in_strip.first(), in_strip.last()) else {
            return Err(err(format!(
                "no {w_axis} mesh line within the centre strip {e1} ..= {e2}"
            )));
        };
        let w_c = in_strip
            .iter()
            .copied()
            .min_by(|x, y| (x - c).abs().total_cmp(&(y - c).abs()))
            .expect("non-empty");
        let Some(g_line) = wl.iter().copied().find(|&x| within(x, e3, w_hi)) else {
            return Err(err(format!(
                "no {w_axis} mesh line on the ground {e3} ..= {w_hi}"
            )));
        };

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
        let point = |xp: f64, xw: f64| {
            let mut v = [0.0; 3];
            v[p] = xp;
            v[w] = xw;
            v[h] = hm;
            v
        };
        for (a, b) in [(w_lo, e0), (e1, e2), (e3, w_hi)] {
            model.add_primitive(metal.1, prim(point(feed, a), point(end, b)))?;
        }

        // Feed lines across the gaps, from the strip (+) to each ground:
        // A on the +width side, B on the −width side.
        let gaps = [(e2, e3, 1.0), (e1, e0, -1.0)];
        let elements = match self.feed_r {
            Some(r) => {
                let mut out = Vec::with_capacity(2);
                for (k, &(a, b, _)) in gaps.iter().enumerate() {
                    let el = LumpedElement::resistor(csx_axis(w), 2.0 * r);
                    let name = names[1 + k].clone();
                    let id = model.add_property(Property::lumped_element(name.clone(), el));
                    model.add_primitive(id, prim(point(feed, a), point(feed, b)))?;
                    out.push((name, id));
                }
                Some(<[_; 2]>::try_from(out).expect("two gaps"))
            }
            None => None,
        };
        let excitations = match self.amplitude {
            Some(amp) => {
                let mut out = Vec::with_capacity(2);
                for (k, &(a, b, sign)) in gaps.iter().enumerate() {
                    let mut v = [0.0; 3];
                    v[w] = sign * amp / (gap * du * w_scale);
                    let mut ex = Excitation::new(v).with_number(n);
                    ex.delay = Scalar::new(self.delay);
                    let name = names[3 + k].clone();
                    let id = model.add_property(Property::excitation(name.clone(), ex));
                    model.add_primitive(id, prim(point(feed, a), point(feed, b)))?;
                    out.push((name, id));
                }
                Some(<[_; 2]>::try_from(out).expect("two gaps"))
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
            model.add_primitive(*id, prim(point(x, w_c), point(x, g_line)))?;
        }
        let currents = [0, 1].map(|k| {
            let mut pb = ProbeBox::new(ProbeType::Current).with_norm_dir(csx_axis(p));
            pb.number = n;
            let id = model.add_property(Property::probe(names[8 + k].clone(), pb));
            (names[8 + k].clone(), id)
        });
        for (k, (_, id)) in currents.iter().enumerate() {
            let (a, b) = (layout.v_pos[k], layout.v_pos[k + 1]);
            model.add_primitive(*id, prim(point(a, s_lo), point(b, s_hi)))?;
        }
        self.dims = Some([sw * du * w_scale, gap * du * w_scale, sub_h * du]);
        self.props = Some(CpwPortProperties {
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
            "CPW",
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
            "CPW",
            self.number,
        )
    }

    fn added(&self) -> Result<(&CpwPortProperties, &Layout)> {
        match (&self.props, &self.layout) {
            (Some(p), Some(l)) => Ok((p, l)),
            _ => Err(Error::Invalid(format!(
                "CPW port {} was not added",
                self.number
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port::elliptic::ellip_k;

    #[test]
    fn quasi_static_limits() {
        // In air: ε_eff = 1 and Z0 = (η0/4)·K(k0')/K(k0); k0 = 1/2.
        let (z, e) = CpwPort::quasi_static_of(1.0, 0.5, 0.0, 1.0, false);
        let k0: f64 = 0.5;
        let want = Z0 / 4.0 * ellip_k((1.0 - k0 * k0).sqrt()) / ellip_k(k0);
        assert!((z - want).abs() < 1e-9 && e == 1.0, "{z} {e}");
        // Thick substrate: k1 → k0, ε_eff → (ε_r + 1)/2.
        let (_, e) = CpwPort::quasi_static_of(1.0, 0.5, 1e3, 9.0, false);
        assert!((e - 5.0).abs() < 1e-3, "{e}");
        // Air-filled backed CPW: ε_eff = 1; a far backing recovers the
        // open CPW, a close one lowers Z.
        let (zb, e) = CpwPort::quasi_static_of(1.0, 0.5, 1e3, 1.0, true);
        assert!(
            (e - 1.0).abs() < 1e-12 && (zb / want - 1.0).abs() < 1e-3,
            "{zb}"
        );
        let (zc, _) = CpwPort::quasi_static_of(1.0, 0.5, 0.2, 1.0, true);
        assert!(zc < 0.7 * want);
        // Thin substrate without backing: ε_eff → 1.
        let (_, e) = CpwPort::quasi_static_of(1.0, 0.5, 1e-3, 9.0, false);
        assert!((e - 1.0).abs() < 1e-6, "{e}");
    }
}
