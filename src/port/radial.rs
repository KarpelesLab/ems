//! Radial-line port ([`RadialLinePort`]): a parallel-plate radial
//! waveguide on a cylindrical grid.
//!
//! # Line model
//!
//! Two parallel plates `z = z₋` (the `−` terminal) and `z = z₊` (the `+`
//! terminal), `h = |z₊ − z₋|` apart, filled with a homogeneous medium
//! (`η = η0·√(μ_r/ε_r)`, `k = ω·√(με)`), carry the radial TEM-like mode
//! TM₀₀: `E_z` uniform across the gap and around the azimuth, `H_α`. Its
//! voltage `V(ρ) = V₊ − V₋ = −∫E_z dz` and its current `I(ρ)`, the radial
//! current in the `+` plate (outwards), obey the telegrapher equations of
//! a non-uniform line, `dV/dρ = −jωμ·h/(Δα·ρ)·I`, `dI/dρ = −jωε·Δα·ρ/h·V`
//! (`Δα` the azimuth the grid covers, 2π for a closed one), whose
//! solutions are cylindrical waves (Marcuvitz, *Waveguide Handbook*,
//! §2.6; Ramo, Whinnery & Van Duzer, §9.4):
//!
//! ```text
//! V(ρ) = A·H₀⁽²⁾(kρ) + B·H₀⁽¹⁾(kρ)
//! I(ρ) = −j·(Δα·ρ/(η·h))·[A·H₁⁽²⁾(kρ) + B·H₁⁽¹⁾(kρ)]
//! ```
//!
//! `H⁽²⁾` the outgoing and `H⁽¹⁾` the incoming wave (time factor
//! `e^{jωt}`). The outgoing wave has the complex wave impedance
//! `Z₊(ρ) = j·η·h/(Δα·ρ)·H₀⁽²⁾(kρ)/H₁⁽²⁾(kρ)`, which tends to the real
//! `η·h/(Δα·ρ)` for `kρ ≫ 1`, the incoming one `−Z₊*`.
//!
//! # Extraction
//!
//! The port records `V` at three equally spaced radii past the feed and
//! `I` on the two dual radii between them (the `H_α` rings where the Yee
//! scheme measures the current). The two amplitudes are fitted to these
//! five values (least squares, the currents scaled by `η·h/(Δα·ρ)`),
//! with the wavenumber of the discrete scheme along the probe spacing `d`:
//! `k = (2/d)·asin(k_t·d/2)`, `k_t = Ω·√(με)` and `Ω = (2/Δt)·sin(ωΔt/2)`,
//! which removes the leading numerical dispersion of the 1-D wave (the
//! remaining discretisation error of the cylindrical spreading is of
//! order `(Δρ/ρ)²`). The waves are reported at the reference radius:
//! `V_inc`, `V_ref` and the travelling-wave reflection coefficient
//! `Γ = V_ref/V_inc`, and power-normalised amplitudes
//! `a = V_inc·√Re(1/Z_inc)`, `b = V_ref·√Re(1/Z_inc)` with
//! `½(|a|² − |b|²) = ½·Re(V·I*)` exactly (the cross terms of the two
//! cylindrical waves carry no power), so that `b/a = Γ` and `|b/a|²` is
//! the reflected power fraction ([`RadialLinePortResult`]). A real
//! reference impedance gives the usual [`PortResult`]
//! ([`RadialLinePortResult::port`]).

use core::f64::consts::TAU;

use csx::primitives::BoxShape;
use csx::properties::{Excitation, LumpedElement, Material, ProbeBox, ProbeType};
use csx::{ContinuousStructure, CoordSystem, Primitive, Property, PropertyId, Scalar};

use super::bessel;
use super::line::{Layout, choose_probes, line_index};
use crate::constants::{C0, Z0};
use crate::post::{Complex64, PortResult};
use crate::sim::Results;
use crate::{Error, Result};

/// The csx property names and ids of an added [`RadialLinePort`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadialLinePortProperties {
    /// The plates, if written (the given property, or `port_metal_<n>`).
    pub metal: Option<(String, PropertyId)>,
    /// The dielectric filling, if written: `port_material_<n>`.
    pub material: Option<(String, PropertyId)>,
    /// The feed resistor, if any: `port_resist_<n>` (on a sector grid
    /// the inner α columns), and on a sector grid `port_resist_<n>a`,
    /// `port_resist_<n>b` on the two wall columns.
    pub elements: Vec<(String, PropertyId)>,
    /// The `Excitation` (if excited): `port_excite_<n>`.
    pub excitation: Option<(String, PropertyId)>,
    /// The voltage `ProbeBox`es `port_ut<n>A`, `B`, `C`, in propagation
    /// order.
    pub voltages: [(String, PropertyId); 3],
    /// The current `ProbeBox`es `port_it<n>A`, `B`, in propagation order.
    pub currents: [(String, PropertyId); 2],
}

/// A radial-line port written into a csx model on a cylindrical grid: a
/// parallel-plate radial waveguide between two `z` planes, an optional
/// feed ring (excitation and/or resistor) at the feed radius, three
/// voltage and two current `ProbeBox`es, analysed with cylindrical waves
/// (see the [module documentation](self)).
///
/// ```
/// use core::f64::consts::TAU;
/// use csx::{ContinuousStructure, CoordSystem};
/// use ems::port::RadialLinePort;
///
/// let mut model = ContinuousStructure::new();
/// model.coord_system = CoordSystem::Cylindrical;
/// model.grid.coord_system = CoordSystem::Cylindrical;
/// model.grid.delta_unit = 1e-3;
/// model.grid.set_lines(0, (0..=60).map(|i| 2.0 + 0.5 * i as f64))?;
/// model.grid.set_lines(1, (0..=4).map(|j| j as f64 * TAU / 4.0))?;
/// model.grid.set_lines(2, [0.0, 1.0, 2.0])?;
/// // Plates z = 0 (−) and z = 2 mm (+), fed at ρ = 2 mm, reference at 12 mm.
/// let port = RadialLinePort::new(1, [0.0, 2.0], 2.0, 12.0)
///     .feed_resistance(50.0)
///     .excite(1.0)
///     .add_to(&mut model)?;
/// assert_eq!(port.probe_positions(), Some([6.5, 7.0, 7.5]));
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Geometry
///
/// * `plates = [z₋, z₊]` (drawing units): the `−` and `+` plates. They
///   are part of the model (PEC `z` faces or metal) unless
///   [`plates`](Self::plates) writes them.
/// * The port runs from the **feed radius** `r_feed` to the reference
///   radius `r_end` (drawing units, `≥ 0`); the incident wave propagates
///   from the feed towards `r_end`: outwards (`H₀⁽²⁾`) if `r_end > r_feed`,
///   inwards (`H₀⁽¹⁾`) otherwise. The current is counted in that direction
///   in the `+` plate.
/// * The port covers the grid's whole α range. On a sector grid (α span
///   below 2π) the α faces must be PMC walls (or periodic), on which the
///   mode's `E_z` lives; the port is then the sector of the radial line
///   (`Δα` in the formulas is the sector's span).
///
/// # Written properties
///
/// * With [`plates`](Self::plates), the plates as zero-thickness boxes
///   over the port's radial range and the whole α range, in the given
///   metal property or in a new metal `port_metal_<n>`, priority
///   `priority + 1`.
/// * With [`dielectric`](Self::dielectric), a material `port_material_<n>`
///   between the plates over the port's radial range.
/// * With [`feed_resistance`](Self::feed_resistance), a `LumpedElement`
///   resistor `port_resist_<n>` along `z` from the `+` to the `−` plate on
///   the feed ring (all α columns in parallel: `R` is the total; on a
///   sector grid split into the inner columns and the wall columns
///   `port_resist_<n>a`, `b`, see
///   [`feed_resistance`](Self::feed_resistance)).
/// * With [`excite`](Self::excite), an E-field excitation `port_excite_<n>`
///   on the same ring, directed from the `+` to the `−` plate with
///   `amplitude/h` V/m. Over the feed resistor it is a Thevenin source of
///   `amplitude` volts (like [`LumpedPort`](super::LumpedPort)); without
///   one a soft source launching waves both ways. The ring must not lie
///   on a PEC face (`E_z` vanishes there): put it inside the domain, or on
///   a PMC `ρ_min` face (an open inner end).
/// * Voltages `port_ut<n>A/B/C`: `z` lines from the `+` to the `−` plate
///   (`V = V₊ − V₋`) at the three equally spaced `ρ` lines past the feed,
///   not beyond `r_end`, whose middle is nearest the middle of the port,
///   on the α line nearest the middle of the α range.
/// * Currents `port_it<n>A/B`: H rings `∮H_α ρ dα` on the dual radii
///   between those lines (flat current boxes normal to `z` from the first
///   `ρ` line out to the inner of the two lines, over the whole α range, a
///   quarter cell into the cell nearest the middle of the gap, which ems
///   and openEMS both read as that cell's dual plane): the total current
///   through that disc, which is the radial current in the `+` plate.
///
/// # Mesh requirements
///
/// The model must be meshed **before** [`add_to`](Self::add_to): the
/// plates and the feed radius must be mesh lines ([`mesh_lines`](Self::mesh_lines)),
/// there must be a cell between the plates, and three
/// equally spaced `ρ` lines past the feed (`ρ > 0`). The fields of the
/// feed that are not the TM₀₀ mode (a feed that is not uniform in α or
/// across the gap) must have decayed at the probes.
///
/// # Reference plane
///
/// The waves are reported at `r_end`, moved by
/// [`reference_shift`](Self::reference_shift) along the propagation
/// direction.
#[derive(Debug, Clone, PartialEq)]
pub struct RadialLinePort {
    number: u32,
    plates: [f64; 2],
    r_feed: f64,
    r_end: f64,
    eps_r: f64,
    mu_r: f64,
    write_dielectric: bool,
    write_plates: bool,
    metal: Option<PropertyId>,
    feed_r: Option<f64>,
    amplitude: Option<f64>,
    delay: f64,
    priority: i32,
    shift: f64,
    props: Option<RadialLinePortProperties>,
    layout: Option<Layout>,
    /// Set by `add_to`: the α span \[rad\], the gap `h` \[m\] and the
    /// drawing unit \[m\].
    span: f64,
    gap: f64,
    unit: f64,
}

impl RadialLinePort {
    /// Port `number` between the plates `plates = [z₋, z₊]`, from the feed
    /// radius `r_feed` to `r_end` (drawing units); see
    /// [Geometry](#geometry).
    pub fn new(number: u32, plates: [f64; 2], r_feed: f64, r_end: f64) -> Self {
        Self {
            number,
            plates,
            r_feed,
            r_end,
            eps_r: 1.0,
            mu_r: 1.0,
            write_dielectric: false,
            write_plates: false,
            metal: None,
            feed_r: None,
            amplitude: None,
            delay: 0.0,
            priority: 0,
            shift: 0.0,
            props: None,
            layout: None,
            span: TAU,
            gap: 0.0,
            unit: 1.0,
        }
    }

    /// Writes the plates (see [Written properties](#written-properties)).
    /// Default: the model provides them.
    pub fn plates(mut self) -> Self {
        self.write_plates = true;
        self
    }

    /// Draws the plates (with [`plates`](Self::plates)) in the existing
    /// metal property `id` instead of a new `port_metal_<n>`.
    pub fn metal(mut self, id: PropertyId) -> Self {
        self.metal = Some(id);
        self
    }

    /// Fills the port with a lossless dielectric `ε_r` written as
    /// `port_material_<n>` (`ε_r = 1` writes nothing); also sets the
    /// medium of the analysis.
    pub fn dielectric(mut self, eps_r: f64) -> Self {
        self.eps_r = eps_r;
        self.write_dielectric = eps_r != 1.0;
        self
    }

    /// The medium between the plates used by the analysis only (the
    /// material is part of the model). Default vacuum.
    pub fn medium(mut self, eps_r: f64, mu_r: f64) -> Self {
        self.eps_r = eps_r;
        self.mu_r = mu_r;
        self
    }

    /// Adds a feed resistor of `r` \[Ω\] (total) on the feed ring; `r = 0`
    /// removes it (the default). On a sector grid the α cells must be
    /// uniform: the two wall columns, whose dual cells are half as wide,
    /// get resistors of their own with half the conductance of an inner
    /// column (`2N·r` for `N` cells, the inner ones `N/(N−1)·r` in all),
    /// so that the feed current is uniform in α.
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

    /// Priority of the port's primitives (default 0; the plates use
    /// `priority + 1`).
    pub fn priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Moves the reference radius by `shift` (drawing units) from `r_end`
    /// along the propagation direction. Default 0.
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
    pub fn properties(&self) -> Option<&RadialLinePortProperties> {
        self.props.as_ref()
    }

    /// The radii of the three voltage probes (drawing units, propagation
    /// order), once added.
    pub fn probe_positions(&self) -> Option<[f64; 3]> {
        self.layout.map(|l| l.v_pos)
    }

    /// The reference radius (drawing units).
    pub fn reference_radius(&self) -> f64 {
        self.r_end + (self.r_end - self.r_feed).signum() * self.shift
    }

    /// The coordinates (drawing units, per axis index) that must be mesh
    /// lines: the feed radius on `ρ`, the plates on `z`.
    pub fn mesh_lines(&self) -> [Vec<f64>; 3] {
        let mut z = self.plates.to_vec();
        z.sort_by(f64::total_cmp);
        z.dedup();
        [vec![self.r_feed], Vec::new(), z]
    }

    fn outward(&self) -> bool {
        self.r_end > self.r_feed
    }

    /// The analytic wave impedance `Z_inc(ρ)` \[Ω\] of the incident wave at
    /// radius `r` (drawing units, `> 0`) and frequency `f` \[Hz\] in the
    /// continuum (`k = 2πf·√(με)/c0`): `Z₊(ρ)` (see the
    /// [module docs](self)) for an outward port, `−Z₊*` for an inward one.
    ///
    /// # Errors
    /// [`Error::Invalid`] before [`add_to`](Self::add_to) or for `r ≤ 0`.
    pub fn wave_impedance(&self, f: f64, r: f64) -> Result<Complex64> {
        self.added()?;
        if !(r > 0.0 && r.is_finite()) {
            return Err(Error::Invalid(format!(
                "radial-line port {}: radius {r} must be > 0",
                self.number
            )));
        }
        let k = TAU * f * (self.eps_r * self.mu_r).sqrt() / C0;
        let rho = r * self.unit;
        let (inc, _) = self.waves(k, rho);
        Ok(inc.0 / (inc.1 * self.i_factor(rho)))
    }

    /// `η·h` \[Ω·m\].
    fn eta_h(&self) -> f64 {
        Z0 * (self.mu_r / self.eps_r).sqrt() * self.gap
    }

    /// `I = factor·(A·H₁ᶦⁿᶜ + B·H₁ʳᵉᶠ)` at radius `rho` \[m\], counted along
    /// the propagation direction.
    fn i_factor(&self, rho: f64) -> Complex64 {
        let s = if self.outward() { 1.0 } else { -1.0 };
        Complex64::new(0.0, -s * self.span * rho / self.eta_h())
    }

    /// `(H₀, H₁)` of the incident and of the reflected wave at `kρ`.
    #[allow(clippy::type_complexity)]
    fn waves(&self, k: f64, rho: f64) -> ((Complex64, Complex64), (Complex64, Complex64)) {
        let x = k * rho;
        let c = |p: (f64, f64)| Complex64::new(p.0, p.1);
        let (h10, h20) = bessel::hankel(0, x);
        let (h11, h21) = bessel::hankel(1, x);
        let (out, inc) = ((c(h20), c(h21)), (c(h10), c(h11)));
        if self.outward() {
            (out, inc)
        } else {
            (inc, out)
        }
    }

    /// Writes the port's properties into `model` and returns the port with
    /// their names and ids.
    ///
    /// # Errors
    /// [`Error::Unsupported`] on a Cartesian grid; [`Error::Invalid`] for
    /// equal plates or radii, negative radii, non-finite values, a
    /// non-positive `ε_r`, `μ_r` or feed resistance, a port already added,
    /// a property name already present in the model, an unknown metal
    /// property, or a mesh that does not meet the
    /// [requirements](#mesh-requirements).
    pub fn add_to(mut self, model: &mut ContinuousStructure) -> Result<Self> {
        let n = self.number;
        if self.props.is_some() {
            return Err(Error::Invalid(format!(
                "radial-line port {n} was already added"
            )));
        }
        if model.grid.coord_system != CoordSystem::Cylindrical {
            return Err(Error::Unsupported(format!(
                "radial-line port {n}: needs a cylindrical grid"
            )));
        }
        if self
            .plates
            .iter()
            .chain([&self.r_feed, &self.r_end, &self.delay, &self.shift])
            .any(|v| !v.is_finite())
            || self.amplitude.is_some_and(|a| !a.is_finite())
        {
            return Err(Error::Invalid(format!(
                "radial-line port {n}: values must be finite"
            )));
        }
        let [z_m, z_p] = self.plates;
        if z_m == z_p {
            return Err(Error::Invalid(format!(
                "radial-line port {n}: the plates coincide (z = {z_m})"
            )));
        }
        if !(self.r_feed >= 0.0 && self.r_end >= 0.0) || self.r_feed == self.r_end {
            return Err(Error::Invalid(format!(
                "radial-line port {n}: radii {} and {} must be ≥ 0 and differ",
                self.r_feed, self.r_end
            )));
        }
        for (what, v) in [("ε_r", self.eps_r), ("μ_r", self.mu_r)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::Invalid(format!(
                    "radial-line port {n}: {what} = {v} must be finite and > 0"
                )));
            }
        }
        if let Some(r) = self.feed_r
            && !(r.is_finite() && r > 0.0)
        {
            return Err(Error::Invalid(format!(
                "radial-line port {n}: feed resistance {r} must be finite and > 0"
            )));
        }
        let du = model.grid.delta_unit;
        let [rho, al, zl] = [0, 1, 2].map(|a| model.grid.lines(a).to_vec());
        let (rho, al, zl) = (&rho[..], &al[..], &zl[..]);
        if al.len() < 2 {
            return Err(Error::Invalid(format!(
                "radial-line port {n}: the grid needs at least two α lines"
            )));
        }
        for (x, lines, what) in [
            (self.r_feed, rho, "feed radius ρ"),
            (z_m, zl, "− plate z"),
            (z_p, zl, "+ plate z"),
        ] {
            if line_index(lines, x).is_none() {
                return Err(Error::Invalid(format!(
                    "radial-line port {n}: the {what} = {x} is not a mesh line (mesh the \
                     model first; see RadialLinePort::mesh_lines)"
                )));
            }
        }
        let layout = choose_probes(rho, self.r_feed, self.r_end, self.shift, du)
            .filter(|l| l.v_pos.iter().all(|&r| r > 0.0))
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "radial-line port {n}: needs three equally spaced ρ lines (ρ > 0) past \
                     the feed at {} and not beyond {}",
                    self.r_feed, self.r_end
                ))
            })?;
        // The z cell nearest the middle of the gap (for the current rings).
        let (z_lo, z_hi) = (z_m.min(z_p), z_m.max(z_p));
        let z_mid = 0.5 * (z_lo + z_hi);
        let tz = 1e-6 * (z_hi - z_lo);
        let cell = zl
            .windows(2)
            .filter(|w| w[0] >= z_lo - tz && w[1] <= z_hi + tz)
            .min_by(|a, b| {
                (0.5 * (a[0] + a[1]) - z_mid)
                    .abs()
                    .total_cmp(&(0.5 * (b[0] + b[1]) - z_mid).abs())
            })
            .map(|w| (w[0], w[1]))
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "radial-line port {n}: no mesh cell between the plates"
                ))
            })?;
        let (a0, a1) = (al[0], al[al.len() - 1]);
        let a_mid = 0.5 * (a0 + a1);
        let alpha_c = al
            .iter()
            .copied()
            .min_by(|x, y| (x - a_mid).abs().total_cmp(&(y - a_mid).abs()))
            .expect("non-empty");
        // A sector (α span below 2π): the number of α cells, which must be
        // uniform for a feed resistor (see `feed_resistance`).
        let closed = ((a1 - a0) - TAU).abs() <= 1e-9 * TAU;
        let sector_cells = (!closed).then_some(al.len() - 1);
        if let Some(c) = sector_cells
            && self.feed_r.is_some()
        {
            let d = (a1 - a0) / c as f64;
            if al.windows(2).any(|w| ((w[1] - w[0]) - d).abs() > 1e-6 * d) {
                return Err(Error::Invalid(format!(
                    "radial-line port {n}: a feed resistor on a sector grid needs uniform α \
                     cells"
                )));
            }
        }

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
            0 => self.write_plates && self.metal.is_none(),
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
        let prio = self.priority;
        let boxed = |s: [f64; 3], t: [f64; 3], pr: i32| {
            Primitive::new(BoxShape::new(s, t)).with_priority(pr)
        };
        let (r_lo, r_hi) = (self.r_feed.min(self.r_end), self.r_feed.max(self.r_end));

        let metal = if self.write_plates {
            let m = match self.metal {
                Some(id) => {
                    let prop = model.property(id).ok_or_else(|| {
                        Error::Invalid(format!("radial-line port {n}: no metal property {id}"))
                    })?;
                    (prop.name.clone(), id)
                }
                None => {
                    let id = model.add_property(Property::metal(names[0].clone()));
                    (names[0].clone(), id)
                }
            };
            for z in [z_m, z_p] {
                model.add_primitive(m.1, boxed([r_lo, a0, z], [r_hi, a1, z], prio + 1))?;
            }
            Some(m)
        } else {
            None
        };
        let material = if self.write_dielectric {
            let id = model.add_property(Property::material(
                names[1].clone(),
                Material::new(self.eps_r, 0.0),
            ));
            model.add_primitive(id, boxed([r_lo, a0, z_lo], [r_hi, a1, z_hi], prio))?;
            Some((names[1].clone(), id))
        } else {
            None
        };

        // Feed ring, from the + plate to the − plate.
        let (f0, f1) = ([self.r_feed, a0, z_p], [self.r_feed, a1, z_m]);
        let mut elements = Vec::new();
        if let Some(r) = self.feed_r {
            let mut add = |name: String, r: f64, b0: f64, b1: f64| -> Result<()> {
                let el = LumpedElement::resistor(csx::Axis::Z, r);
                let id = model.add_property(Property::lumped_element(name.clone(), el));
                model.add_primitive(
                    id,
                    boxed([self.r_feed, b0, z_p], [self.r_feed, b1, z_m], prio),
                )?;
                elements.push((name, id));
                Ok(())
            };
            match sector_cells {
                // Closed azimuth: every column is alike (the seam's two
                // copies are one edge).
                None => add(names[2].clone(), r, a0, a1)?,
                // Sector: the columns on the walls have half the dual
                // width, so half the conductance of the others.
                Some(c) => {
                    let cf = c as f64;
                    if c > 1 {
                        add(names[2].clone(), r * cf / (cf - 1.0), al[1], al[c - 1])?;
                    }
                    add(format!("{}a", names[2]), 2.0 * cf * r, a0, a0)?;
                    add(format!("{}b", names[2]), 2.0 * cf * r, a1, a1)?;
                }
            }
        }
        let h = (z_p - z_m).abs() * du;
        let excitation = match self.amplitude {
            Some(amp) => {
                let mut ex =
                    Excitation::new([0.0, 0.0, (z_m - z_p).signum() * amp / h]).with_number(n);
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
            let r = layout.v_pos[k];
            model.add_primitive(*id, boxed([r, alpha_c, z_p], [r, alpha_c, z_m], prio))?;
        }
        // A flat box a quarter cell above the cell's lower line: ems puts
        // the loop in that cell (the layer of the nearest node), openEMS
        // in the nearest dual plane, the same one. Its current is counted
        // along +z; a current from the − to the + plate through the disc
        // is the outward current in the + plate (weight −1 when the +
        // plate is below, and again for an inward port).
        let z_box = cell.0 + 0.25 * (cell.1 - cell.0);
        let weight = if (z_p > z_m) == self.outward() {
            1.0
        } else {
            -1.0
        };
        let currents = [0, 1].map(|k| {
            let mut pb = ProbeBox::new(ProbeType::Current)
                .with_norm_dir(csx::Axis::Z)
                .with_weight(weight);
            pb.number = n;
            let id = model.add_property(Property::probe(names[7 + k].clone(), pb));
            (names[7 + k].clone(), id)
        });
        for (k, (_, id)) in currents.iter().enumerate() {
            let r = layout.v_pos[k].min(layout.v_pos[k + 1]);
            model.add_primitive(*id, boxed([rho[0], a0, z_box], [r, a1, z_box], prio))?;
        }
        self.props = Some(RadialLinePortProperties {
            metal,
            material,
            elements,
            excitation,
            voltages,
            currents,
        });
        self.layout = Some(layout);
        self.span = a1 - a0;
        self.gap = h;
        self.unit = du;
        Ok(self)
    }

    fn added(&self) -> Result<(&RadialLinePortProperties, &Layout)> {
        match (&self.props, &self.layout) {
            (Some(p), Some(l)) => Ok((p, l)),
            _ => Err(Error::Invalid(format!(
                "radial-line port {} was not added",
                self.number
            ))),
        }
    }

    /// Frequency-domain analysis at `freqs` \[Hz\] (see the
    /// [module documentation](self)).
    ///
    /// # Errors
    /// [`Error::Invalid`] if the port was not added, a probe is missing
    /// from `results` or has no samples, or a frequency is at or above the
    /// discrete cutoff of the probe spacing (`k_t·d ≥ 2`).
    pub fn analyze(&self, results: &Results, freqs: &[f64]) -> Result<RadialLinePortResult> {
        let (p, l) = self.added()?;
        let get = |name: &str| {
            results
                .probe(name)
                .ok_or_else(|| Error::Invalid(format!("no probe {name:?} in the results")))?
                .dft(freqs)
        };
        let v_sp = p.voltages.each_ref().map(|v| get(&v.0));
        let i_sp = p.currents.each_ref().map(|v| get(&v.0));
        let [v1, v2, v3] = v_sp;
        let [ia, ib] = i_sp;
        let (v1, v2, v3, ia, ib) = (v1?, v2?, v3?, ia?, ib?);
        let dt = results.dt();
        let du = self.unit;
        let r_v = l.v_pos.map(|r| r * du);
        let r_i = [0.5 * (r_v[0] + r_v[1]), 0.5 * (r_v[1] + r_v[2])];
        let r_ref = self.reference_radius() * du;
        let d = l.spacing;
        let nm = (self.eps_r * self.mu_r).sqrt();
        let mut out = RadialLinePortResult {
            freqs: freqs.to_vec(),
            ..Default::default()
        };
        for (q, &f) in freqs.iter().enumerate() {
            let om = 2.0 / dt * (core::f64::consts::PI * f * dt).sin();
            let s = om * nm / C0 * d / 2.0;
            if s.is_nan() || s >= 1.0 {
                return Err(Error::Invalid(format!(
                    "radial-line port {}: f = {f:e} Hz is above the discrete cutoff of the \
                     probe spacing",
                    self.number
                )));
            }
            let k = 2.0 / d * s.asin();
            // Rows [coef A, coef B] = measurement, currents scaled by
            // η·h/(Δα·ρ).
            let mut rows: Vec<([Complex64; 2], Complex64)> = Vec::with_capacity(5);
            for (m, &r) in r_v.iter().enumerate() {
                let (inc, refl) = self.waves(k, r);
                rows.push(([inc.0, refl.0], [v1[q], v2[q], v3[q]][m]));
            }
            for (m, &r) in r_i.iter().enumerate() {
                let (inc, refl) = self.waves(k, r);
                let fac = self.i_factor(r);
                let scale = self.eta_h() / (self.span * r);
                rows.push((
                    [inc.1 * fac * scale, refl.1 * fac * scale],
                    [ia[q], ib[q]][m] * scale,
                ));
            }
            // Normal equations M^H M x = M^H y.
            let (mut m00, mut m01, mut m11) = (0.0, Complex64::ZERO, 0.0);
            let (mut y0, mut y1) = (Complex64::ZERO, Complex64::ZERO);
            for (c, y) in &rows {
                m00 += c[0].norm_sqr();
                m11 += c[1].norm_sqr();
                m01 += c[0].conj() * c[1];
                y0 += c[0].conj() * *y;
                y1 += c[1].conj() * *y;
            }
            let det = m00 * m11 - m01.norm_sqr();
            let a = (y0 * m11 - m01 * y1) / det;
            let b = (y1 * m00 - m01.conj() * y0) / det;
            let (mut res, mut norm) = (0.0, 0.0);
            for (c, y) in &rows {
                res += (c[0] * a + c[1] * b - *y).norm_sqr();
                norm += y.norm_sqr();
            }
            let (inc, refl) = self.waves(k, r_ref);
            let fac = self.i_factor(r_ref);
            let (vi, vr) = (a * inc.0, b * refl.0);
            out.v.push(vi + vr);
            out.i.push(fac * (a * inc.1 + b * refl.1));
            out.v_inc.push(vi);
            out.v_ref.push(vr);
            out.z_wave.push(inc.0 / (inc.1 * fac));
            out.k.push(k);
            out.residual
                .push(if norm > 0.0 { (res / norm).sqrt() } else { 0.0 });
        }
        Ok(out)
    }
}

/// The analysis of a [`RadialLinePort`] at its reference radius (see the
/// [module documentation](super::radial) for the definitions).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RadialLinePortResult {
    freqs: Vec<f64>,
    v: Vec<Complex64>,
    i: Vec<Complex64>,
    v_inc: Vec<Complex64>,
    v_ref: Vec<Complex64>,
    z_wave: Vec<Complex64>,
    k: Vec<f64>,
    residual: Vec<f64>,
}

impl RadialLinePortResult {
    /// Analysis frequencies \[Hz\].
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// Voltage `V = V_inc + V_ref` at the reference radius.
    pub fn voltage(&self) -> &[Complex64] {
        &self.v
    }

    /// Current at the reference radius (along the propagation direction
    /// in the `+` plate).
    pub fn current(&self) -> &[Complex64] {
        &self.i
    }

    /// Incident cylindrical wave `V_inc` at the reference radius.
    pub fn v_inc(&self) -> &[Complex64] {
        &self.v_inc
    }

    /// Reflected cylindrical wave `V_ref` at the reference radius.
    pub fn v_ref(&self) -> &[Complex64] {
        &self.v_ref
    }

    /// Travelling-wave reflection coefficient `Γ = V_ref/V_inc` at the
    /// reference radius (`= b/a`).
    pub fn reflection(&self) -> Vec<Complex64> {
        self.v_inc
            .iter()
            .zip(&self.v_ref)
            .map(|(&a, &b)| b / a)
            .collect()
    }

    /// Wave impedance of the incident wave at the reference radius (with
    /// the discrete wavenumber, [`wavenumber`](Self::wavenumber)).
    pub fn z_wave(&self) -> &[Complex64] {
        &self.z_wave
    }

    /// Input impedance `V/I` at the reference radius.
    pub fn z_in(&self) -> Vec<Complex64> {
        self.v.iter().zip(&self.i).map(|(&v, &i)| v / i).collect()
    }

    /// Power-normalised incident amplitude `a = V_inc·√Re(1/Z_inc)`.
    pub fn a(&self) -> Vec<Complex64> {
        self.v_inc
            .iter()
            .zip(&self.z_wave)
            .map(|(&v, &z)| v * z.inv().re.sqrt())
            .collect()
    }

    /// Power-normalised reflected amplitude `b = V_ref·√Re(1/Z_inc)`.
    pub fn b(&self) -> Vec<Complex64> {
        self.v_ref
            .iter()
            .zip(&self.z_wave)
            .map(|(&v, &z)| v * z.inv().re.sqrt())
            .collect()
    }

    /// Accepted power `½·Re(V·I*) = ½(|a|² − |b|²)`.
    pub fn accepted_power(&self) -> Vec<f64> {
        self.v
            .iter()
            .zip(&self.i)
            .map(|(&v, &i)| 0.5 * (v * i.conj()).re)
            .collect()
    }

    /// The wavenumber of the fit \[1/m\] (the discrete one, see the
    /// [module documentation](super::radial)).
    pub fn wavenumber(&self) -> &[f64] {
        &self.k
    }

    /// Relative residual of the fit of the two cylindrical waves to the
    /// five probe values: near 0 for a clean TM₀₀ field at the probes.
    pub fn fit_residual(&self) -> &[f64] {
        &self.residual
    }

    /// `V`, `I` at the reference radius as a [`PortResult`] with the
    /// reference impedance `z_ref` (e.g. the asymptotic `η·h/(Δα·ρ)`).
    ///
    /// # Errors
    /// As [`PortResult::new`].
    pub fn port(&self, z_ref: Complex64) -> Result<PortResult> {
        PortResult::new(self.freqs.clone(), self.v.clone(), self.i.clone(), z_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.coord_system = CoordSystem::Cylindrical;
        m.grid.coord_system = CoordSystem::Cylindrical;
        m.grid.delta_unit = 1e-3;
        m.grid
            .set_lines(0, (0..=40).map(|i| 1.0 + 0.5 * f64::from(i)))
            .unwrap();
        m.grid
            .set_lines(1, (0..=4).map(|j| f64::from(j) * TAU / 4.0))
            .unwrap();
        m.grid.set_lines(2, [0.0, 1.0, 2.0, 3.0]).unwrap();
        m
    }

    /// The cylindrical waves solve the telegrapher equations: the fit of
    /// exact samples returns the amplitudes, and `½(|a|² − |b|²)` is the
    /// accepted power.
    #[test]
    fn waves_and_power() {
        let mut m = model();
        let p = RadialLinePort::new(1, [0.0, 3.0], 1.0, 15.0)
            .medium(2.0, 1.0)
            .add_to(&mut m)
            .unwrap();
        assert_eq!(p.probe_positions(), Some([7.5, 8.0, 8.5]));
        let k = 300.0;
        let (a, b) = (Complex64::new(1.0, 0.3), Complex64::new(-0.2, 0.1));
        let v = |r: f64| {
            let (i, o) = p.waves(k, r);
            a * i.0 + b * o.0
        };
        let cur = |r: f64| {
            let (i, o) = p.waves(k, r);
            p.i_factor(r) * (a * i.1 + b * o.1)
        };
        // dV/dρ = −jωμ·h/(Δα·ρ)·I with ω·μ = k·η.
        let (r, e) = (8e-3, 1e-7);
        let dv = (v(r + e) - v(r - e)) / (2.0 * e);
        let eta = Z0 / 2f64.sqrt();
        let want = Complex64::new(0.0, -k * eta * 3e-3 / (TAU * r)) * cur(r);
        assert!((dv - want).norm() < 1e-6 * want.norm(), "{dv} vs {want}");
        // Power: ½Re(V I*) = ½(|V+|² − |V−|²)·Re(1/Z+).
        let (i, o) = p.waves(k, r);
        let z = i.0 / (i.1 * p.i_factor(r));
        let pw = 0.5 * (v(r) * cur(r).conj()).re;
        let want = 0.5 * ((a * i.0).norm_sqr() - (b * o.0).norm_sqr()) * z.inv().re;
        assert!((pw - want).abs() < 1e-12 * want.abs(), "{pw} vs {want}");
        // The asymptotic impedance.
        let z = p.wave_impedance(50e9, 15.0).unwrap();
        let zw = eta * 3e-3 / (TAU * 15e-3);
        assert!(
            (z.re / zw - 1.0).abs() < 0.01 && z.im.abs() < 0.05 * zw,
            "{z} vs {zw}"
        );
    }

    #[test]
    fn written_properties() {
        let mut m = model();
        let p = RadialLinePort::new(2, [3.0, 0.0], 1.0, 15.0)
            .plates()
            .dielectric(2.0)
            .feed_resistance(50.0)
            .excite(1.0)
            .add_to(&mut m)
            .unwrap();
        let pr = p.properties().unwrap();
        assert!(pr.metal.is_some() && pr.material.is_some());
        assert!(pr.elements.len() == 1 && pr.excitation.is_some());
        assert_eq!(pr.currents[1].0, "port_it2B");
        // Cartesian grids are refused; a feed radius off the mesh too.
        let mut c = ContinuousStructure::new();
        assert!(matches!(
            RadialLinePort::new(1, [0.0, 1.0], 1.0, 5.0).add_to(&mut c),
            Err(Error::Unsupported(_))
        ));
        let mut m = model();
        assert!(
            RadialLinePort::new(1, [0.0, 3.0], 1.2, 15.0)
                .add_to(&mut m)
                .is_err()
        );
        assert!(
            RadialLinePort::new(1, [1.0, 1.0], 1.0, 15.0)
                .add_to(&mut m)
                .is_err()
        );
    }
}
