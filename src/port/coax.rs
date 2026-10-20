//! Coaxial-line port ([`CoaxPort`]).
//!
//! A section of coaxial line with three voltage and two current probes,
//! analysed with the transmission-line extraction shared with
//! [`MslPort`](super::MslPort) (see the [`msl`](super::msl) module for the
//! line model): the propagation constant `γ` and the line impedance `Z_L`
//! come from the probes, the waves are reported at a reference plane.
//!
//! The TEM mode of a coaxial line with inner radius `a`, outer radius `b`
//! and a homogeneous filling `ε_r` has (Pozar, *Microwave Engineering*,
//! §2.2 and §3.4)
//!
//! ```text
//! E_ρ = V / (ρ·ln(b/a)),   H_φ = I / (2πρ),   Z0 = η0/(2π·√ε_r)·ln(b/a)
//! ```
//!
//! which is what [`CoaxPort::z0`] returns and what the feed imposes.

use core::f64::consts::TAU;

use csx::primitives::{BoxShape, Cylinder};
use csx::properties::{Excitation, LumpedElement, Material, ProbeBox, ProbeType, WeightFn};
use csx::{ContinuousStructure, CoordSystem, Primitive, Property, PropertyId, Scalar};

use super::line::{self, Layout, LinePortResult, choose_probes, line_index};
use crate::axis::Axis;
use crate::constants::Z0;
use crate::post::Complex64;
use crate::sim::Results;
use crate::{Error, Result};

/// The csx property names and ids of an added [`CoaxPort`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoaxPortProperties {
    /// The conductors (the given property, or `port_metal_<n>`).
    pub metal: (String, PropertyId),
    /// The dielectric filling, if written: `port_material_<n>`.
    pub material: Option<(String, PropertyId)>,
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

/// A coaxial-line port written into a csx model: the conductors over the
/// port length, an optional dielectric filling, an optional feed at its
/// start and three voltage and two current `ProbeBox`es, analysed like an
/// [`MslPort`](super::MslPort) (line model and extraction: the
/// [`msl`](super::msl) module): the propagation constant `γ` and line
/// impedance `Z_L` come from the probes, the waves are reported at a
/// reference plane.
///
/// The TEM mode of a coaxial line with inner radius `a`, outer radius `b`
/// and a homogeneous filling `ε_r` has (Pozar, *Microwave Engineering*,
/// §2.2 and §3.4) `E_ρ = V/(ρ·ln(b/a))`, `H_φ = I/(2πρ)` and
/// `Z0 = η0/(2π·√ε_r)·ln(b/a)` ([`z0`](Self::z0)).
///
/// ```
/// use core::f64::consts::TAU;
/// use csx::{ContinuousStructure, CoordSystem};
/// use ems::port::CoaxPort;
/// use ems::Axis;
///
/// // Cylindrical grid: ρ from the inner (1 mm) to the outer (2.3 mm)
/// // conductor, a closed azimuth, z along the line.
/// let mut model = ContinuousStructure::new();
/// model.coord_system = CoordSystem::Cylindrical;
/// model.grid.coord_system = CoordSystem::Cylindrical;
/// model.grid.delta_unit = 1e-3;
/// model.grid.set_lines(0, (0..=13).map(|i| 1.0 + 0.1 * i as f64))?;
/// model.grid.set_lines(1, (0..=8).map(|j| j as f64 * TAU / 8.0))?;
/// model.grid.set_lines(2, (0..=40).map(f64::from))?;
/// let port = CoaxPort::new(1, [0.0, 0.0, 5.0], [0.0, 0.0, 15.0], Axis::Z, 1.0, 2.3)
///     .excite(1.0)
///     .add_to(&mut model)?;
/// assert_eq!(port.probe_positions(), Some([9.0, 10.0, 11.0]));
/// assert!((port.z0() - 49.94).abs() < 0.01);
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Geometry
///
/// * `start` and `stop` (drawing units) are two points on the line's
///   axis: along the propagation axis `dir` the port runs from the **feed
///   plane** `start[dir]` to `stop[dir]` (the wave enters the structure in
///   that direction); the transverse coordinates of `start` are the axis
///   (those of `stop` must agree).
/// * `r_inner` (`a`) and `r_outer` (`b`) are the radius of the inner
///   conductor and the inner radius of the outer conductor (drawing
///   units), `0 < a < b`.
/// * **Cartesian grids**: any `dir`. The conductors are staircased on the
///   mesh as boxes of whole cells: the inner one the cells whose centre is
///   within `a` of the axis, the outer one (with
///   [`shield`](Self::shield)) the cells whose centre is between `b` and
///   the shield radius. (A csx `Cylinder` would not do:
///   csx makes an edge PEC when its centre is in metal, which leaves nodes
///   at the conductor potential with free longitudinal edges; the line is
///   then not TEM and measurably slow, `ε_eff` 12 % high at six cells per
///   radius. If the model continues the line beyond the port, draw it the
///   same way, e.g. with a second port or with boxes.)
/// * **Cylindrical grids**: `dir` must be `z` and the line's axis is the
///   grid's `ρ = 0` axis (the transverse coordinates of `start` are
///   ignored). The conductors are `(ρ, α, z)` boxes over the grid's whole
///   `α` range: the inner one for `ρ ≤ a`, the shield for `b ≤ ρ ≤`
///   shield radius. A grid may also start at `ρ = a` and end at `ρ = b`
///   with PEC faces as the conductors (the default boundaries); the
///   written primitives then only touch those faces.
///
/// # Written properties
///
/// * The conductors in the given metal property ([`metal`](Self::metal))
///   or in a new metal `port_metal_<n>`, over the port length (plus
///   [`feed_extension`](Self::feed_extension) behind the feed), with
///   priority `priority + 1`: the inner conductor always, the outer
///   conductor only with [`shield`](Self::shield) (otherwise the model
///   provides it: metal, or the `ρ_max` PEC face of a cylindrical grid).
/// * With [`dielectric`](Self::dielectric), a material `port_material_<n>`
///   filling `a ≤ ρ ≤ b` over the port length.
/// * With [`excite`](Self::excite), an E-field excitation
///   `port_excite_<n>` over the annulus `a < ρ < b` of the feed plane with
///   the TEM profile `E_ρ = amplitude/(ρ·ln(b/a))` (radially outwards,
///   from the inner conductor, the `+` terminal, to the outer one): on a
///   Cartesian grid the `u`, `v` components `E_ρ·(u − u0)/ρ` and
///   `E_ρ·(v − v0)/ρ` as csx weighting functions masked to the annulus,
///   on a cylindrical grid the `ρ` component `1/rho`. Without a feed
///   resistor it is a soft source that launches waves in both
///   directions: the line behind the feed plane must be absorbing (end
///   in a PML), like the guide behind a
///   [`RectWaveguidePort`](super::RectWaveguidePort).
/// * With [`feed_resistance`](Self::feed_resistance) (cylindrical grids
///   only), a radial `LumpedElement` resistor `port_resist_<n>` over the
///   same annulus and the whole `α` range; with the excitation it is a
///   Thevenin source (as for [`LumpedPort`](super::LumpedPort)).
/// * Voltages `port_ut<n>A/B/C`: radial lines at three equally spaced
///   mesh lines past the feed (chosen as for the MSL port), from a mesh
///   node inside the inner conductor to one inside (or on) the outer
///   conductor: `V = V_inner − V_outer`. The edges inside the metal carry
///   no voltage, so the path ends do not need to be on the surfaces. On a
///   Cartesian grid the path runs along `u = dir.others().0` through the
///   `v` line nearest the axis, from the `u` line nearest the axis to the
///   first line at `ρ ≥ b` (with a shield: the line nearest the middle of
///   the shield); on a cylindrical grid along `ρ` on the `α` line nearest
///   `start[α]`, from the last `ρ` line `≤ a` to the first `≥ b`.
/// * Currents `port_it<n>A/B`: H loops around the inner conductor on the
///   dual planes between the voltage lines, positive along the
///   propagation direction. Cartesian: the smallest box of mesh lines
///   containing every node within `a` of the axis, whose loop (half a
///   cell outside) must stay inside `ρ < b`. Cylindrical: the box from
///   the first `ρ` line to the last line `≤ a` over all `α` lines. On a
///   closed azimuth the `H_α` edges on the `α = 0` and `α = 2π` lines are
///   two stored copies of one edge, each holding half of its dual length,
///   so the loop must contain both; its radial sides, which do not cancel
///   at the seam, lie inside or on the inner conductor where `H_ρ = 0`.
///
/// # Mesh requirements
///
/// The model must be meshed **before** [`add_to`](Self::add_to). The feed
/// plane must be a mesh line and the port must contain three equally
/// spaced lines past it. Cartesian: a node within `a` of the axis on the
/// probe path and a line at `ρ ≥ b`; at least about four cells across the
/// inner radius and a gap `b − a` of a few cells, so that the current
/// loop fits between the conductors. Cylindrical: a `ρ` line at or below
/// `a` and one at or above `b`; with `a` and `b` on mesh lines (see
/// [`mesh_lines`](Self::mesh_lines)) the conductors are exact. `add_to`
/// fails otherwise.
///
/// # Reference plane
///
/// The waves are reported at `stop[dir]` (the end of the port towards
/// the structure), moved by [`reference_shift`](Self::reference_shift)
/// along the propagation direction.
///
/// # Accuracy
///
/// The extraction reproduces the discrete line exactly (see the `msl`
/// module), so the extracted `Z_L` is the impedance of the line *as
/// meshed*. On a cylindrical grid that is the discrete TEM impedance
/// `η/(2π)·Σ Δρ_i/ρ_{i+½}`, which tends to [`z0`](Self::z0) as `Δρ²`; on a
/// Cartesian grid the line stays exactly TEM (whole-cell conductors, see
/// [Geometry](#geometry)) but the staircase acts like conductors whose
/// surfaces lie about `0.2·Δ` outside the nominal radii: `Z_L` is low by
/// 5.9 % at 6 cells per inner radius and 3.4 % at 12 (first order). Use a
/// cylindrical grid where the impedance must be exact.
///
/// The feed's near field must have decayed at the probes: on a
/// cylindrical grid with `b − a` = 1.3 mm, the extracted `β` is off by
/// 7e-4 with the middle probe 7 mm past the feed and by 4e-5 at 10 mm;
/// keep **about eight times `b − a`** between the feed and the middle of
/// the port.
#[derive(Debug, Clone, PartialEq)]
pub struct CoaxPort {
    number: u32,
    start: [f64; 3],
    stop: [f64; 3],
    dir: Axis,
    r_inner: f64,
    r_outer: f64,
    shield: Option<f64>,
    eps_r: f64,
    write_dielectric: bool,
    metal: Option<PropertyId>,
    feed_r: Option<f64>,
    amplitude: Option<f64>,
    delay: f64,
    priority: i32,
    shift: f64,
    extension: f64,
    props: Option<CoaxPortProperties>,
    layout: Option<Layout>,
}

impl CoaxPort {
    /// Port `number` from the feed plane at `start` to `stop` (points on
    /// the line's axis, drawing units) along `dir`, with inner conductor
    /// radius `r_inner` and outer conductor inner radius `r_outer`
    /// (drawing units); see [Geometry](#geometry).
    pub fn new(
        number: u32,
        start: [f64; 3],
        stop: [f64; 3],
        dir: Axis,
        r_inner: f64,
        r_outer: f64,
    ) -> Self {
        Self {
            number,
            start,
            stop,
            dir,
            r_inner,
            r_outer,
            shield: None,
            eps_r: 1.0,
            write_dielectric: false,
            metal: None,
            feed_r: None,
            amplitude: None,
            delay: 0.0,
            priority: 0,
            shift: 0.0,
            extension: 0.0,
            props: None,
            layout: None,
        }
    }

    /// Writes the outer conductor as a shield from `r_outer` to
    /// `r_shield` (drawing units, `> r_outer`). Default: not written.
    pub fn shield(mut self, r_shield: f64) -> Self {
        self.shield = Some(r_shield);
        self
    }

    /// Fills the line with a lossless dielectric `ε_r` written as
    /// `port_material_<n>` over the port length (`ε_r = 1` writes nothing).
    /// Also sets [`eps_r`](Self::eps_r).
    pub fn dielectric(mut self, eps_r: f64) -> Self {
        self.eps_r = eps_r;
        self.write_dielectric = eps_r != 1.0;
        self
    }

    /// The relative permittivity of the filling used by [`z0`](Self::z0)
    /// only (the material is part of the model). Default 1.
    pub fn eps_r(mut self, eps_r: f64) -> Self {
        self.eps_r = eps_r;
        self
    }

    /// Draws the conductors in the existing metal property `id` instead of
    /// a new `port_metal_<n>`.
    pub fn metal(mut self, id: PropertyId) -> Self {
        self.metal = Some(id);
        self
    }

    /// Adds a radial feed resistor of `r` \[Ω\] over the feed annulus
    /// (cylindrical grids only); `r = 0` removes it (the default).
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

    /// Priority of the port's primitives (default 0; the conductors use
    /// `priority + 1`).
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

    /// Extends the conductors (and the dielectric) by `len` drawing units
    /// behind the feed plane, e.g. through the absorbing boundary behind a
    /// soft-source feed. Default 0.
    pub fn feed_extension(mut self, len: f64) -> Self {
        self.extension = len;
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
    pub fn properties(&self) -> Option<&CoaxPortProperties> {
        self.props.as_ref()
    }

    /// The positions of the three voltage probes along the propagation
    /// axis (drawing units, propagation order), once added.
    pub fn probe_positions(&self) -> Option<[f64; 3]> {
        self.layout.map(|l| l.v_pos)
    }

    /// Analytic TEM impedance `Z0 = η0/(2π·√ε_r)·ln(b/a)` \[Ω\].
    pub fn z0(&self) -> f64 {
        Z0 / (TAU * self.eps_r.sqrt()) * (self.r_outer / self.r_inner).ln()
    }

    /// The coordinates (drawing units, per axis index) that should be mesh
    /// lines: the feed plane and `stop[dir]` on `dir`; on a Cartesian grid
    /// the axis position on the two transverse axes, on a cylindrical grid
    /// the radii `a`, `b` (and the shield radius) on `ρ`.
    pub fn mesh_lines(&self, coord_system: CoordSystem) -> [Vec<f64>; 3] {
        let mut out: [Vec<f64>; 3] = Default::default();
        let d = self.dir.index();
        out[d] = vec![self.start[d], self.stop[d]];
        if coord_system == CoordSystem::Cylindrical {
            out[0] = vec![self.r_inner, self.r_outer];
            out[0].extend(self.shield);
        } else {
            let (u, v) = self.dir.others();
            out[u.index()] = vec![self.start[u.index()]];
            out[v.index()] = vec![self.start[v.index()]];
        }
        for o in &mut out {
            o.sort_by(f64::total_cmp);
            o.dedup();
        }
        out
    }

    /// Writes the port's properties into `model` and returns the port with
    /// their names and ids.
    ///
    /// # Errors
    /// [`Error::Invalid`] for invalid radii (`0 < a < b < shield`), a zero
    /// length, transverse coordinates of `start` and `stop` that differ
    /// (Cartesian), non-finite values, a non-positive `ε_r` or feed
    /// resistance, a port already added, a property name already present
    /// in the model, an unknown metal property, or a mesh that does not
    /// meet the [requirements](#mesh-requirements);
    /// [`Error::Unsupported`] for a cylindrical grid with `dir` other than
    /// `z`, or a feed resistor on a Cartesian grid.
    pub fn add_to(mut self, model: &mut ContinuousStructure) -> Result<Self> {
        let n = self.number;
        if self.props.is_some() {
            return Err(Error::Invalid(format!("coax port {n} was already added")));
        }
        let cyl = match model.grid.coord_system {
            CoordSystem::Cartesian => false,
            CoordSystem::Cylindrical => true,
            #[allow(unreachable_patterns)]
            _ => {
                return Err(Error::Unsupported(format!(
                    "coax port {n}: unsupported coordinate system"
                )));
            }
        };
        if cyl && self.dir != Axis::Z {
            return Err(Error::Unsupported(format!(
                "coax port {n}: on a cylindrical grid the line must run along z"
            )));
        }
        if !cyl && self.feed_r.is_some() {
            return Err(Error::Unsupported(format!(
                "coax port {n}: a feed resistor needs a cylindrical grid (radial element)"
            )));
        }
        if self.start.iter().chain(&self.stop).any(|v| !v.is_finite())
            || !self.delay.is_finite()
            || !self.shift.is_finite()
            || !(self.extension.is_finite() && self.extension >= 0.0)
            || !self.r_inner.is_finite()
            || !self.r_outer.is_finite()
            || self.shield.is_some_and(|s| !s.is_finite())
            || self.amplitude.is_some_and(|a| !a.is_finite())
        {
            return Err(Error::Invalid(format!(
                "coax port {n}: values must be finite"
            )));
        }
        let (a, b) = (self.r_inner, self.r_outer);
        if !(a > 0.0 && b > a) {
            return Err(Error::Invalid(format!(
                "coax port {n}: radii must satisfy 0 < r_inner ({a}) < r_outer ({b})"
            )));
        }
        if let Some(s) = self.shield
            && s <= b
        {
            return Err(Error::Invalid(format!(
                "coax port {n}: shield radius {s} must exceed r_outer {b}"
            )));
        }
        if !(self.eps_r.is_finite() && self.eps_r > 0.0) {
            return Err(Error::Invalid(format!(
                "coax port {n}: ε_r = {} must be finite and > 0",
                self.eps_r
            )));
        }
        if let Some(r) = self.feed_r
            && !(r.is_finite() && r > 0.0)
        {
            return Err(Error::Invalid(format!(
                "coax port {n}: feed resistance {r} must be finite and > 0"
            )));
        }
        let p = self.dir.index();
        let (ua, va) = self.dir.others();
        let (u, v) = (ua.index(), va.index());
        if !cyl && (self.start[u] != self.stop[u] || self.start[v] != self.stop[v]) {
            return Err(Error::Invalid(format!(
                "coax port {n}: start and stop must lie on one axis along {}",
                self.dir
            )));
        }
        let du = model.grid.delta_unit;
        let (feed, end) = (self.start[p], self.stop[p]);
        if end == feed {
            return Err(Error::Invalid(format!(
                "coax port {n} has zero length along {}",
                self.dir
            )));
        }
        if line_index(model.grid.lines(p), feed).is_none() {
            return Err(Error::Invalid(format!(
                "coax port {n}: the feed plane {} = {feed} is not a mesh line (mesh the \
                 model first; see CoaxPort::mesh_lines)",
                self.dir
            )));
        }
        let layout =
            choose_probes(model.grid.lines(p), feed, end, self.shift, du).ok_or_else(|| {
                Error::Invalid(format!(
                    "coax port {n}: needs three equally spaced mesh lines along {} past the \
                     feed at {feed} and not beyond {end}",
                    self.dir
                ))
            })?;
        let geo = if cyl {
            cylindrical_geometry(model, a, b, self.shield, self.start[1], n)?
        } else {
            cartesian_geometry(
                model,
                [u, v],
                [self.start[u], self.start[v]],
                a,
                b,
                self.shield,
                n,
            )?
        };

        // Names.
        let names = [
            format!("port_metal_{n}"),
            format!("port_material_{n}"),
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
            1 => self.write_dielectric,
            2 => self.feed_r.is_some(),
            3 => self.amplitude.is_some(),
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
                    Error::Invalid(format!("coax port {n}: no metal property {id}"))
                })?;
                (prop.name.clone(), id)
            }
            None => {
                let id = model.add_property(Property::metal(names[0].clone()));
                (names[0].clone(), id)
            }
        };
        let prio = self.priority;
        let boxed = |s: [f64; 3], t: [f64; 3], pr: i32| {
            Primitive::new(BoxShape::new(s, t)).with_priority(pr)
        };
        // A point at transverse "radius" coordinates (Cartesian: u, v;
        // cylindrical: ρ, α) and propagation coordinate `x`.
        let point = |t0: f64, t1: f64, x: f64| {
            let mut q = [0.0; 3];
            if cyl {
                q[0] = t0;
                q[1] = t1;
            } else {
                q[u] = t0;
                q[v] = t1;
            }
            q[p] = x;
            q
        };
        let on_axis = |x: f64| {
            let mut q = self.start;
            q[p] = x;
            q
        };
        let (a0, a1) = geo.alpha_span;

        // Conductors and dielectric, from `back` (behind the feed).
        let back = feed - (end - feed).signum() * self.extension;
        if cyl {
            model.add_primitive(
                metal.1,
                boxed(point(0.0, a0, back), point(a, a1, end), prio + 1),
            )?;
            if let Some(s) = self.shield {
                model.add_primitive(
                    metal.1,
                    boxed(point(b, a0, back), point(s, a1, end), prio + 1),
                )?;
            }
        } else {
            // Mesh-aligned staircase cells (see `staircase`).
            for &([u0, v0], [u1, v1]) in &geo.cells {
                model.add_primitive(
                    metal.1,
                    boxed(point(u0, v0, back), point(u1, v1, end), prio + 1),
                )?;
            }
        }
        let material = if self.write_dielectric {
            let id = model.add_property(Property::material(
                names[1].clone(),
                Material::new(self.eps_r, 0.0),
            ));
            if cyl {
                model.add_primitive(id, boxed(point(a, a0, back), point(b, a1, end), prio))?;
            } else {
                model.add_primitive(
                    id,
                    Primitive::new(Cylinder::new(on_axis(back), on_axis(end), b))
                        .with_priority(prio),
                )?;
            }
            Some((names[1].clone(), id))
        } else {
            None
        };

        // Feed over the annulus at the feed plane.
        let (f0, f1) = if cyl {
            (point(a, a0, feed), point(b, a1, feed))
        } else {
            let (cu, cv) = (self.start[u], self.start[v]);
            (point(cu - b, cv - b, feed), point(cu + b, cv + b, feed))
        };
        let element = match self.feed_r {
            Some(r) => {
                let el = LumpedElement::resistor(csx::Axis::X, r);
                let id = model.add_property(Property::lumped_element(names[2].clone(), el));
                model.add_primitive(id, boxed(f0, f1, prio))?;
                Some((names[2].clone(), id))
            }
            None => None,
        };
        let excitation = match self.amplitude {
            Some(amp) => {
                // E_ρ = amp/(ρ·ln(b/a)) with ρ in metres = amp/(ln·du)·(1/ρ_du).
                let scale = amp / ((b / a).ln() * du);
                let mut vec = [0.0; 3];
                let mut ex;
                if cyl {
                    vec[0] = scale;
                    ex = Excitation::new(vec).with_number(n);
                    ex.weight[0] = WeightFn::new(format!("if((rho>{a:e})&(rho<{b:e}),1/rho,0)"));
                } else {
                    vec[u] = scale;
                    vec[v] = scale;
                    ex = Excitation::new(vec).with_number(n);
                    let var = ["x", "y", "z"];
                    let (cu, cv) = (self.start[u], self.start[v]);
                    let du_ = format!("({}-({cu:e}))", var[u]);
                    let dv_ = format!("({}-({cv:e}))", var[v]);
                    let r2 = format!("({du_}^2+{dv_}^2)");
                    let mask = format!("({r2}>{:e})&({r2}<{:e})", a * a, b * b);
                    ex.weight[u] = WeightFn::new(format!("if({mask},{du_}/{r2},0)"));
                    ex.weight[v] = WeightFn::new(format!("if({mask},{dv_}/{r2},0)"));
                }
                ex.delay = Scalar::new(self.delay);
                let id = model.add_property(Property::excitation(names[3].clone(), ex));
                model.add_primitive(id, boxed(f0, f1, prio))?;
                Some((names[3].clone(), id))
            }
            None => None,
        };

        // Probes.
        let voltages = [0, 1, 2].map(|k| {
            let mut pb = ProbeBox::new(ProbeType::Voltage);
            pb.number = n;
            let id = model.add_property(Property::probe(names[4 + k].clone(), pb));
            (names[4 + k].clone(), id)
        });
        for (k, (_, id)) in voltages.iter().enumerate() {
            let x = layout.v_pos[k];
            let (s0, s1, t) = geo.v_path;
            model.add_primitive(*id, boxed(point(s0, t, x), point(s1, t, x), prio))?;
        }
        let csx_p = csx::Axis::from_index(p).expect("axis index in 0..3");
        let currents = [0, 1].map(|k| {
            let mut pb = ProbeBox::new(ProbeType::Current).with_norm_dir(csx_p);
            pb.number = n;
            let id = model.add_property(Property::probe(names[7 + k].clone(), pb));
            (names[7 + k].clone(), id)
        });
        for (k, (_, id)) in currents.iter().enumerate() {
            let (x0, x1) = (layout.v_pos[k], layout.v_pos[k + 1]);
            let ([l0, l1], [h0, h1]) = geo.loop_box;
            model.add_primitive(*id, boxed(point(l0, l1, x0), point(h0, h1, x1), prio))?;
        }
        self.props = Some(CoaxPortProperties {
            metal,
            material,
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
    pub fn analyze(&self, results: &Results, freqs: &[f64]) -> Result<LinePortResult> {
        let (p, l) = self.added()?;
        line::analyze(
            results,
            freqs,
            &p.probe_names(),
            l,
            None,
            "coax",
            self.number,
        )
    }

    /// Frequency-domain analysis with the reference impedance `z_ref`
    /// (e.g. [`z0`](Self::z0)).
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
            "coax",
            self.number,
        )
    }

    fn added(&self) -> Result<(&CoaxPortProperties, &Layout)> {
        match (&self.props, &self.layout) {
            (Some(p), Some(l)) => Ok((p, l)),
            _ => Err(Error::Invalid(format!(
                "coax port {} was not added",
                self.number
            ))),
        }
    }
}

impl CoaxPortProperties {
    fn probe_names(&self) -> line::ProbeNames<'_> {
        line::ProbeNames {
            voltages: self.voltages.each_ref().map(|v| v.0.as_str()),
            currents: self.currents.each_ref().map(|v| v.0.as_str()),
        }
    }
}

/// Transverse probe geometry, in the transverse coordinates of the grid
/// (Cartesian `(u, v)`, cylindrical `(ρ, α)`).
struct Geometry {
    /// Voltage path: from the first to the second coordinate along the
    /// first transverse axis, at the third on the second axis.
    v_path: (f64, f64, f64),
    /// Current-loop box corners `([t0, t1], [t0, t1])`.
    loop_box: ([f64; 2], [f64; 2]),
    /// `α` range of the conductors and the feed (cylindrical), unused on
    /// Cartesian grids.
    alpha_span: (f64, f64),
    /// Cartesian conductor cells: transverse box corners `([u0, v0], [u1,
    /// v1])` (empty on cylindrical grids).
    cells: Vec<([f64; 2], [f64; 2])>,
}

/// Tolerance of a mesh-line comparison: 1e-6 of the smallest cell.
fn tol(lines: &[f64]) -> f64 {
    let cell = lines
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    if cell.is_finite() { 1e-6 * cell } else { 1e-9 }
}

fn cylindrical_geometry(
    model: &ContinuousStructure,
    a: f64,
    b: f64,
    shield: Option<f64>,
    alpha: f64,
    n: u32,
) -> Result<Geometry> {
    let rho = model.grid.lines(0);
    let al = model.grid.lines(1);
    if rho.len() < 2 || al.len() < 2 {
        return Err(Error::Invalid(format!(
            "coax port {n}: the grid needs at least two ρ and α lines"
        )));
    }
    let tr = tol(rho);
    let inner = rho.iter().copied().rfind(|&r| r <= a + tr);
    let Some(inner) = inner else {
        return Err(Error::Invalid(format!(
            "coax port {n}: no ρ mesh line at or inside the inner radius {a}"
        )));
    };
    let outer = match shield {
        Some(s) => {
            let mid = 0.5 * (b + s);
            rho.iter()
                .copied()
                .filter(|&r| r >= b - tr && r <= s + tr)
                .min_by(|x, y| (x - mid).abs().total_cmp(&(y - mid).abs()))
        }
        None => rho.iter().copied().find(|&r| r >= b - tr),
    };
    let Some(outer) = outer else {
        return Err(Error::Invalid(format!(
            "coax port {n}: no ρ mesh line at or outside the outer radius {b}"
        )));
    };
    let alpha_line = al
        .iter()
        .copied()
        .min_by(|x, y| (x - alpha).abs().total_cmp(&(y - alpha).abs()))
        .expect("non-empty");
    let (first, last) = (al[0], al[al.len() - 1]);
    Ok(Geometry {
        v_path: (inner, outer, alpha_line),
        loop_box: ([rho[0], first], [inner, last]),
        alpha_span: (first, last),
        cells: Vec::new(),
    })
}

fn cartesian_geometry(
    model: &ContinuousStructure,
    [u, v]: [usize; 2],
    [cu, cv]: [f64; 2],
    a: f64,
    b: f64,
    shield: Option<f64>,
    n: u32,
) -> Result<Geometry> {
    let (lu, lv) = (model.grid.lines(u), model.grid.lines(v));
    if lu.len() < 2 || lv.len() < 2 {
        return Err(Error::Invalid(format!(
            "coax port {n}: the grid needs at least two lines on each transverse axis"
        )));
    }
    let (tu, tv) = (tol(lu), tol(lv));
    let nearest = |l: &[f64], c: f64| {
        l.iter()
            .copied()
            .min_by(|x, y| (x - c).abs().total_cmp(&(y - c).abs()))
            .expect("non-empty")
    };
    let (u0, v0) = (nearest(lu, cu), nearest(lv, cv));
    let dv = v0 - cv;
    if (u0 - cu).hypot(dv) > a + tu.max(tv) {
        return Err(Error::Invalid(format!(
            "coax port {n}: no mesh node within the inner radius {a} of the axis \
             (see CoaxPort::mesh_lines)"
        )));
    }
    let r = |x: f64| (x - cu).hypot(dv);
    let end = match shield {
        Some(s) => {
            let mid = cu + (0.5 * (b + s)).max(0.0);
            lu.iter()
                .copied()
                .filter(|&x| x > cu && r(x) >= b - tu && r(x) <= s + tu)
                .min_by(|x, y| (x - mid).abs().total_cmp(&(y - mid).abs()))
        }
        None => lu.iter().copied().find(|&x| x > cu && r(x) >= b - tu),
    };
    let Some(end) = end else {
        return Err(Error::Invalid(format!(
            "coax port {n}: no {} mesh line at ρ ≥ {b} on the probe path",
            Axis::from_index(u).expect("axis")
        )));
    };
    // Smallest box of lines containing every node within `a` of the axis.
    let span = |l: &[f64], c: f64, t: f64| -> Option<(usize, usize)> {
        let lo = l.iter().position(|&x| x >= c - a - t)?;
        let hi = l.iter().rposition(|&x| x <= c + a + t)?;
        (lo <= hi && lo > 0 && hi + 1 < l.len()).then_some((lo, hi))
    };
    let (Some((ulo, uhi)), Some((vlo, vhi))) = (span(lu, cu, tu), span(lv, cv, tv)) else {
        return Err(Error::Invalid(format!(
            "coax port {n}: the inner conductor needs mesh lines across it and one \
             more line on each side"
        )));
    };
    // The loop runs on the dual lines half a cell outside the box.
    let du0 = cu - 0.5 * (lu[ulo] + lu[ulo - 1]);
    let du1 = 0.5 * (lu[uhi] + lu[uhi + 1]) - cu;
    let dv0 = cv - 0.5 * (lv[vlo] + lv[vlo - 1]);
    let dv1 = 0.5 * (lv[vhi] + lv[vhi + 1]) - cv;
    let corner = du0.max(du1).hypot(dv0.max(dv1));
    if corner >= b {
        return Err(Error::Invalid(format!(
            "coax port {n}: the current loop around the inner conductor (corner at \
             ρ = {corner:.4}) does not fit inside r_outer = {b}; refine the mesh"
        )));
    }
    let mut cells = staircase(lu, lv, |r| r <= a, (cu, cv));
    if cells.is_empty() {
        return Err(Error::Invalid(format!(
            "coax port {n}: no mesh cell centre lies inside the inner conductor (radius \
             {a}); refine the mesh"
        )));
    }
    if let Some(s) = shield {
        cells.extend(staircase(lu, lv, |r| r >= b && r <= s, (cu, cv)));
    }
    Ok(Geometry {
        v_path: (u0, end, v0),
        loop_box: ([lu[ulo], lv[vlo]], [lu[uhi], lv[vhi]]),
        alpha_span: (0.0, 0.0),
        cells,
    })
}

/// The mesh cells (merged into runs along `u`) whose centre, at distance
/// `r` from the axis `(cu, cv)`, satisfies `inside(r)`.
///
/// csx makes an edge PEC when its *centre* is in metal, so a curved
/// primitive (a `Cylinder`) gives PEC edges whose end nodes are outside
/// the metal; those nodes then take the conductor's potential but keep
/// their free longitudinal E edge, which is not compatible with a TEM
/// field and slows the line down (measured: `ε_eff` 12 % high, `Z` 3 %
/// high for 6 cells per radius). With mesh-aligned cells every PEC edge
/// has both end nodes, and their longitudinal edges, in the metal, and the
/// staircased line is exactly TEM.
fn staircase(
    lu: &[f64],
    lv: &[f64],
    inside: impl Fn(f64) -> bool,
    (cu, cv): (f64, f64),
) -> Vec<([f64; 2], [f64; 2])> {
    let mut out = Vec::new();
    for j in 0..lv.len() - 1 {
        let mut run: Option<usize> = None;
        let yc = 0.5 * (lv[j] + lv[j + 1]) - cv;
        for i in 0..lu.len() {
            let ok = i + 1 < lu.len() && inside((0.5 * (lu[i] + lu[i + 1]) - cu).hypot(yc));
            match (ok, run) {
                (true, None) => run = Some(i),
                (false, Some(i0)) => {
                    out.push(([lu[i0], lv[j]], [lu[i], lv[j + 1]]));
                    run = None;
                }
                _ => {}
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cyl_model(na: usize) -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.coord_system = CoordSystem::Cylindrical;
        m.grid.coord_system = CoordSystem::Cylindrical;
        m.grid.delta_unit = 1e-3;
        m.grid
            .set_lines(0, (0..=13).map(|i| 1.0 + 0.1 * i as f64))
            .unwrap();
        m.grid
            .set_lines(1, (0..=na).map(|j| j as f64 * TAU / na as f64))
            .unwrap();
        m.grid.set_lines(2, (0..=40).map(f64::from)).unwrap();
        m
    }

    fn cart_model() -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        let l: Vec<f64> = (-16..=16).map(|i| i as f64 * 0.25).collect();
        m.grid.set_lines(0, l.clone()).unwrap();
        m.grid.set_lines(1, l).unwrap();
        m.grid.set_lines(2, (0..=40).map(f64::from)).unwrap();
        m
    }

    fn prim_box(m: &ContinuousStructure, id: PropertyId) -> ([f64; 3], [f64; 3]) {
        let p = &m.property(id).unwrap().primitives[0];
        match &p.shape {
            csx::primitives::Shape::Box(b) => (b.start.value().0, b.stop.value().0),
            s => panic!("{s:?}"),
        }
    }

    #[test]
    fn cylindrical_layout() {
        let mut m = cyl_model(8);
        let port = CoaxPort::new(1, [0.0, 0.0, 5.0], [0.0, 0.0, 15.0], Axis::Z, 1.0, 2.3)
            .excite(1.0)
            .feed_resistance(50.0)
            .add_to(&mut m)
            .unwrap();
        let p = port.properties().unwrap();
        assert!(p.element.is_some() && p.excitation.is_some() && p.material.is_none());
        let (s, t) = prim_box(&m, p.voltages[1].1);
        assert_eq!((s[0], t[0], s[2]), (1.0, 2.3, 10.0));
        // Loop: ρ up to the inner conductor, all α lines.
        let (s, t) = prim_box(&m, p.currents[0].1);
        assert_eq!((s[0], t[0]), (1.0, 1.0));
        assert_eq!(s[1], 0.0);
        assert!((t[1] - TAU).abs() < 1e-12);
        assert_eq!((s[2], t[2]), (9.0, 10.0));
        let z = Z0 / TAU * 2.3f64.ln();
        assert!((port.z0() - z).abs() < 1e-12);
    }

    #[test]
    fn cartesian_layout() {
        let mut m = cart_model();
        let port = CoaxPort::new(2, [0.0, 0.0, 30.0], [0.0, 0.0, 20.0], Axis::Z, 1.5, 3.45)
            .shield(4.0)
            .dielectric(2.0)
            .excite(1.0)
            .add_to(&mut m)
            .unwrap();
        assert_eq!(port.probe_positions(), Some([26.0, 25.0, 24.0]));
        let p = port.properties().unwrap();
        assert!(p.material.is_some());
        let (s, t) = prim_box(&m, p.voltages[0].1);
        assert_eq!((s[0], t[0], s[1]), (0.0, 3.75, 0.0));
        let (s, t) = prim_box(&m, p.currents[0].1);
        assert_eq!((s[0], s[1], t[0], t[1]), (-1.5, -1.5, 1.5, 1.5));
        // Excitation weights: radial 1/ρ inside the annulus only.
        let ex = m.property(p.excitation.as_ref().unwrap().1).unwrap();
        let csx::PropertyKind::Excitation(e) = &ex.kind else {
            unreachable!()
        };
        let at = |x: f64, y: f64| e.amplitude_at(0, csx::Vec3::new(x, y, 30.0));
        let s = 1.0 / ((3.45f64 / 1.5).ln() * 1e-3);
        assert!((at(2.0, 0.0) - s / 2.0).abs() < 1e-9 * s);
        assert!((at(0.0, 2.0)).abs() < 1e-9 * s);
        assert_eq!(at(1.0, 0.0), 0.0);
        assert_eq!(at(3.5, 0.0), 0.0);
    }

    #[test]
    fn invalid_ports() {
        let mut m = cyl_model(8);
        let c = |a: f64, b: f64| CoaxPort::new(1, [0.0, 0.0, 5.0], [0.0, 0.0, 15.0], Axis::Z, a, b);
        assert!(c(0.0, 2.0).add_to(&mut m).is_err());
        assert!(c(2.0, 1.0).add_to(&mut m).is_err());
        assert!(c(1.0, 2.0).shield(1.5).add_to(&mut m).is_err());
        assert!(c(1.0, 2.0).eps_r(-1.0).add_to(&mut m).is_err());
        assert!(c(1.0, 2.0).feed_resistance(-5.0).add_to(&mut m).is_err());
        // Outer radius beyond the grid.
        assert!(c(1.0, 5.0).add_to(&mut m).is_err());
        // Feed plane off the mesh, zero length.
        assert!(
            CoaxPort::new(1, [0.0, 0.0, 5.5], [0.0, 0.0, 15.0], Axis::Z, 1.0, 2.0)
                .add_to(&mut m)
                .is_err()
        );
        assert!(
            CoaxPort::new(1, [0.0, 0.0, 5.0], [0.0, 0.0, 5.0], Axis::Z, 1.0, 2.0)
                .add_to(&mut m)
                .is_err()
        );
        // Not along z on a cylindrical grid.
        assert!(matches!(
            CoaxPort::new(1, [1.0, 0.0, 5.0], [2.0, 0.0, 5.0], Axis::X, 1.0, 2.0).add_to(&mut m),
            Err(Error::Unsupported(_))
        ));
        let ok = c(1.0, 2.0).add_to(&mut m).unwrap();
        assert!(ok.clone().add_to(&mut m).is_err());
        assert!(c(1.0, 2.0).add_to(&mut m).is_err()); // names

        let mut m = cart_model();
        let c = |a: f64, b: f64| CoaxPort::new(1, [0.0, 0.0, 5.0], [0.0, 0.0, 15.0], Axis::Z, a, b);
        assert!(matches!(
            c(1.5, 3.45).feed_resistance(50.0).add_to(&mut m),
            Err(Error::Unsupported(_))
        ));
        // Loop does not fit between the conductors.
        assert!(c(1.5, 1.8).add_to(&mut m).is_err());
        // Off-axis stop.
        assert!(
            CoaxPort::new(1, [0.0, 0.0, 5.0], [0.5, 0.0, 15.0], Axis::Z, 1.5, 3.45)
                .add_to(&mut m)
                .is_err()
        );
        assert!(c(1.5, 3.45).add_to(&mut m).is_ok());
    }
}
