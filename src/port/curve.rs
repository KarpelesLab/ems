//! Curve port ([`CurvePort`]): a lumped port between two points that are
//! not on one mesh line, connected along a polyline.

use csx::primitives::{Curve, Wire};
use csx::{ContinuousStructure, Primitive, Property, PropertyId};

use super::{LumpedPort, PortProperties};
use crate::axis::Axis;
use crate::post::{Complex64, PortResult};
use crate::sim::Results;
use crate::{Error, Result};

/// The csx property names and ids of an added [`CurvePort`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CurvePortProperties {
    /// The feed: the resistor, excitation and probes of the
    /// [`LumpedPort`] on the feed edge (`port_resist_<n>`,
    /// `port_excite_<n>`, `port_ut<n>`, `port_it<n>`).
    pub feed: PortProperties,
    /// The metal of the curve (the given property, or `port_metal_<n>`),
    /// if the curve continues past the feed edge.
    pub metal: Option<(String, PropertyId)>,
}

/// A lumped port along a curve (a polyline), written into a csx model: a
/// one-cell [`LumpedPort`] (resistor `R`, optional excitation, V/I
/// probes) on the first mesh edge of the curve — the **feed edge** — and a
/// PEC curve (csx `Curve` in a metal property, or a thin csx `Wire` with
/// [`wire_radius`](Self::wire_radius)) along the rest of the polyline.
///
/// It connects two terminals that do not lie on a common mesh line (e.g.
/// the feed of a bent or oblique wire, a probe feed from a ground plane to
/// an off-axis point) with a single source and a single, well-defined
/// port voltage and current.
///
/// ```
/// use csx::ContinuousStructure;
/// use ems::port::CurvePort;
///
/// let mut model = ContinuousStructure::new();
/// model.grid.delta_unit = 1e-3;
/// for a in 0..3 {
///     model.grid.set_lines(a, (0..=10).map(f64::from))?;
/// }
/// // From the + terminal at (2, 2, 0) up to z = 4, then diagonally to (6, 5, 4).
/// let port = CurvePort::new(1, 50.0, [[2.0, 2.0, 0.0], [2.0, 2.0, 4.0], [6.0, 5.0, 4.0]])
///     .excite(1.0)
///     .add_to(&mut model)?;
/// assert_eq!(port.feed_edge(), Some(([2.0, 2.0, 0.0], [2.0, 2.0, 1.0], ems::Axis::Z)));
/// assert_eq!(port.properties().unwrap().feed.voltage.0, "port_ut1");
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Conventions
///
/// * The polyline runs from the **`+` terminal** (its first point) to the
///   `−` terminal (its last point), at least two points. The model must be
///   meshed before [`add_to`](Self::add_to): every vertex is moved to the
///   nearest mesh node (drawing units; `(ρ, α, z)` on a cylindrical grid),
///   so the curve is the staircase through those nodes (csx walks
///   each segment through the nearest nodes).
/// * The **feed edge** is the first cell of the curve: from the first node,
///   one mesh cell along the axis on which the first segment advances by
///   the most cells (ties: x before y before z), towards the second vertex.
///   The first segment must therefore span at least one cell.
/// * On the feed edge the port is exactly a [`LumpedPort`] from the first
///   node to the next one (same properties, names and sign conventions): a
///   resistor `R`, an excitation of `amplitude` volts (Thevenin source),
///   the voltage probe along the edge and the element current.
/// * From the end of the feed edge to the last vertex the curve is PEC
///   (a csx `Curve` in the metal property, written into
///   [`metal`](Self::metal) or a new `port_metal_<n>`; with
///   [`wire_radius`](Self::wire_radius) a csx `Wire`, i.e. the thin-wire or
///   oblique-wire model of design §5.14). It carries no voltage, so the
///   port voltage `V = ∫E·dl` along the **whole** curve from the `+` to the
///   `−` terminal equals the feed-edge voltage, and the port current is the
///   current through the curve (the feed edge is in series with it). The
///   curve's own inductance and its coupling to the structure belong to
///   the structure, i.e. the reference plane is the feed edge.
/// * Whatever is at the `−` terminal (a ground plane, another conductor)
///   is part of the model.
/// * With [`wire_radius`](Self::wire_radius), a polyline with an oblique
///   segment is one oblique-wire network (bends included) and adds the
///   expected `(μ0/2π)·ln(r_eq/r0)` per unit length to the staircase's
///   inductance (`tests/validation_curve.rs`). A polyline of axis-aligned
///   runs only becomes separate straight thin wires that meet at the bends
///   (the compile step warns): their end corrections overestimate the
///   inductance of a bent wire (a loop read 9.3 nH instead of the
///   staircase's 5.6 nH at `r0 ≈ r_eq`), so prefer the plain `Curve`
///   there.
///
/// Why one feed edge instead of a resistor spread along the curve: a csx
/// `LumpedElement` acts on edges of a single direction, and a source spread
/// over a staircase would need one element per straight run with its
/// share of `R` and of the voltage; the series connection is equivalent at
/// the terminals, while a single edge keeps the port a
/// [`LumpedPort`] with its second-order reference plane (design §5.5).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CurvePort {
    number: u32,
    resistance: f64,
    points: Vec<[f64; 3]>,
    amplitude: Option<f64>,
    delay: f64,
    priority: i32,
    metal: Option<PropertyId>,
    wire_radius: Option<f64>,
    feed: Option<LumpedPort>,
    feed_edge: Option<([f64; 3], [f64; 3], Axis)>,
    metal_props: Option<(String, PropertyId)>,
}

impl CurvePort {
    /// Port `number` with resistance `r` \[Ω\] along the polyline `points`
    /// (drawing units), from the `+` terminal (first point) to the `−`
    /// terminal (last point).
    pub fn new(number: u32, r: f64, points: impl IntoIterator<Item = [f64; 3]>) -> Self {
        Self {
            number,
            resistance: r,
            points: points.into_iter().collect(),
            amplitude: None,
            delay: 0.0,
            priority: 0,
            metal: None,
            wire_radius: None,
            feed: None,
            feed_edge: None,
            metal_props: None,
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

    /// Draws the curve in the existing metal property `id` instead of a
    /// new `port_metal_<n>`.
    pub fn metal(mut self, id: PropertyId) -> Self {
        self.metal = Some(id);
        self
    }

    /// Draws the curve as a thin csx `Wire` of radius `r` (drawing units)
    /// instead of a `Curve` (design §5.14: the thin-wire model for
    /// axis-aligned runs, the oblique-wire model if a segment is oblique).
    pub fn wire_radius(mut self, r: f64) -> Self {
        self.wire_radius = Some(r);
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

    /// The feed edge `(start, stop, axis)` (drawing units, `start` the `+`
    /// terminal), once added.
    pub fn feed_edge(&self) -> Option<([f64; 3], [f64; 3], Axis)> {
        self.feed_edge
    }

    /// The properties written by [`add_to`](Self::add_to) (`None` before).
    pub fn properties(&self) -> Option<CurvePortProperties> {
        Some(CurvePortProperties {
            feed: self.feed.as_ref()?.properties()?.clone(),
            metal: self.metal_props.clone(),
        })
    }

    /// Writes the port's properties into `model` and returns the port with
    /// their names and ids.
    ///
    /// # Errors
    /// [`Error::Invalid`] for fewer than two points, non-finite values, a
    /// non-positive resistance or wire radius, an unmeshed model, a first
    /// segment shorter than one cell, an unknown metal property, a port
    /// already added, or a property name already present in the model.
    pub fn add_to(mut self, model: &mut ContinuousStructure) -> Result<Self> {
        let n = self.number;
        if self.feed.is_some() {
            return Err(Error::Invalid(format!("curve port {n} was already added")));
        }
        if self.points.len() < 2 {
            return Err(Error::Invalid(format!(
                "curve port {n} needs at least two points"
            )));
        }
        if self.points.iter().flatten().any(|v| !v.is_finite())
            || !self.delay.is_finite()
            || self.amplitude.is_some_and(|a| !a.is_finite())
        {
            return Err(Error::Invalid(format!(
                "curve port {n}: values must be finite"
            )));
        }
        if let Some(r) = self.wire_radius
            && !(r.is_finite() && r > 0.0)
        {
            return Err(Error::Invalid(format!(
                "curve port {n}: wire radius {r} must be finite and > 0"
            )));
        }
        // Snap every vertex to the nearest mesh node.
        let mut nodes: Vec<[usize; 3]> = Vec::with_capacity(self.points.len());
        for p in &self.points {
            let mut idx = [0; 3];
            for (d, x) in p.iter().enumerate() {
                idx[d] = nearest(model.grid.lines(d), *x).ok_or_else(|| {
                    Error::Invalid(format!(
                        "curve port {n}: the model has no mesh lines on axis {d} \
                         (mesh it first)"
                    ))
                })?;
            }
            nodes.push(idx);
        }
        let lines: [Vec<f64>; 3] = [0, 1, 2].map(|d| model.grid.lines(d).to_vec());
        let coord = |idx: [usize; 3]| [0, 1, 2].map(|d| lines[d][idx[d]]);
        let (n0, n1) = (nodes[0], nodes[1]);
        let steps = [0, 1, 2].map(|d| n1[d] as i64 - n0[d] as i64);
        let a = (0..3)
            .max_by_key(|&d| (steps[d].abs(), std::cmp::Reverse(d)))
            .expect("three axes");
        if steps[a] == 0 {
            return Err(Error::Invalid(format!(
                "curve port {n}: the first segment is shorter than one mesh cell"
            )));
        }
        let mut f = n0;
        f[a] = (n0[a] as i64 + steps[a].signum()) as usize;
        let axis = Axis::from_index(a).expect("axis index in 0..3");
        let (p0, p1) = (coord(n0), coord(f));

        // The rest of the curve: from the end of the feed edge on.
        let mut rest = vec![f];
        rest.extend(nodes.iter().skip(1).copied().filter(|&q| q != f));
        rest.dedup();
        let wire_name = format!("port_metal_{n}");
        if rest.len() > 1 && self.metal.is_none() && model.property_by_name(&wire_name).is_some() {
            return Err(Error::Invalid(format!(
                "the model already has a property named {wire_name:?}"
            )));
        }
        if let Some(id) = self.metal
            && model.property(id).is_none()
        {
            return Err(Error::Invalid(format!(
                "curve port {n}: no metal property {id}"
            )));
        }
        let mut feed = LumpedPort::new(n, self.resistance, p0, p1, axis)
            .delay(self.delay)
            .priority(self.priority);
        if let Some(amp) = self.amplitude {
            feed = feed.excite(amp);
        }
        let feed = feed.add_to(model)?;
        let metal = if rest.len() > 1 {
            let (name, id) = match self.metal {
                Some(id) => (model.property(id).expect("checked above").name.clone(), id),
                None => (
                    wire_name.clone(),
                    model.add_property(Property::metal(wire_name)),
                ),
            };
            let pts: Vec<[f64; 3]> = rest.iter().map(|&q| coord(q)).collect();
            let prim = match self.wire_radius {
                Some(r) => Primitive::new(Wire::new(pts, r)),
                None => Primitive::new(Curve::new(pts)),
            };
            model.add_primitive(id, prim.with_priority(self.priority))?;
            Some((name, id))
        } else {
            None
        };
        self.feed = Some(feed);
        self.feed_edge = Some((p0, p1, axis));
        self.metal_props = metal;
        Ok(self)
    }

    /// Frequency-domain analysis with reference impedance `Z_ref = R`
    /// (see [`LumpedPort::analyze`]).
    ///
    /// # Errors
    /// See [`analyze_with`](Self::analyze_with).
    pub fn analyze(&self, results: &Results, freqs: &[f64]) -> Result<PortResult> {
        self.analyze_with(results, freqs, Complex64::new(self.resistance, 0.0))
    }

    /// Frequency-domain analysis with reference impedance `z_ref`: the
    /// feed edge's [`LumpedPort::analyze_with`].
    ///
    /// # Errors
    /// [`Error::Invalid`] if the port was not added, its probes are missing
    /// from `results`, or `z_ref` is invalid.
    pub fn analyze_with(
        &self,
        results: &Results,
        freqs: &[f64],
        z_ref: Complex64,
    ) -> Result<PortResult> {
        self.feed
            .as_ref()
            .ok_or_else(|| Error::Invalid(format!("curve port {} was not added", self.number)))?
            .analyze_with(results, freqs, z_ref)
    }
}

/// Index of the line nearest `x`.
fn nearest(lines: &[f64], x: f64) -> Option<usize> {
    lines
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - x).abs().total_cmp(&(b.1 - x).abs()))
        .map(|(k, _)| k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        for a in 0..3 {
            m.grid.set_lines(a, (0..=10).map(f64::from)).unwrap();
        }
        m
    }

    #[test]
    fn feed_edge_and_curve() {
        let mut m = model();
        // Vertices off the mesh are snapped; the first segment advances
        // most along −y.
        let p = CurvePort::new(2, 75.0, [[5.2, 6.9, 3.0], [4.0, 2.0, 3.4], [1.0, 1.0, 8.0]])
            .add_to(&mut m)
            .unwrap();
        assert_eq!(
            p.feed_edge(),
            Some(([5.0, 7.0, 3.0], [5.0, 6.0, 3.0], Axis::Y))
        );
        let props = p.properties().unwrap();
        assert!(props.feed.excitation.is_none());
        let (name, id) = props.metal.unwrap();
        assert_eq!(name, "port_metal_2");
        let prim = &m.property(id).unwrap().primitives[0];
        let csx::primitives::Shape::Curve(c) = &prim.shape else {
            panic!("{prim:?}")
        };
        let pts: Vec<[f64; 3]> = c
            .points
            .iter()
            .map(|q| {
                let v = q.value();
                [v[0], v[1], v[2]]
            })
            .collect();
        assert_eq!(pts, vec![[5.0, 6.0, 3.0], [4.0, 2.0, 3.0], [1.0, 1.0, 8.0]]);
        // Resistor, voltage, current and metal.
        assert_eq!(m.properties().len(), 4);
    }

    #[test]
    fn single_cell_curve_is_a_lumped_port() {
        let mut m = model();
        let p = CurvePort::new(1, 50.0, [[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
            .excite(1.0)
            .wire_radius(0.1)
            .add_to(&mut m)
            .unwrap();
        assert!(p.properties().unwrap().metal.is_none());
        assert_eq!(m.properties().len(), 4);
        let mut m2 = model();
        LumpedPort::new(1, 50.0, [1.0, 1.0, 1.0], [1.0, 1.0, 2.0], Axis::Z)
            .excite(1.0)
            .add_to(&mut m2)
            .unwrap();
        assert_eq!(m.properties(), m2.properties());
    }

    #[test]
    fn invalid_curves() {
        let mut m = model();
        let c = |pts: Vec<[f64; 3]>| CurvePort::new(1, 50.0, pts);
        assert!(c(vec![[1.0, 1.0, 1.0]]).add_to(&mut m).is_err());
        assert!(
            c(vec![[1.0, 1.0, 1.0], [1.2, 1.0, 1.3]])
                .add_to(&mut m)
                .is_err()
        );
        assert!(
            c(vec![[1.0, 1.0, f64::NAN], [1.0, 1.0, 3.0]])
                .add_to(&mut m)
                .is_err()
        );
        assert!(
            CurvePort::new(1, 0.0, [[1.0, 1.0, 1.0], [1.0, 1.0, 3.0]])
                .add_to(&mut m)
                .is_err()
        );
        assert!(
            c(vec![[1.0, 1.0, 1.0], [1.0, 1.0, 3.0]])
                .wire_radius(-1.0)
                .add_to(&mut m)
                .is_err()
        );
        let ok = c(vec![[1.0, 1.0, 1.0], [1.0, 1.0, 3.0]])
            .add_to(&mut m)
            .unwrap();
        assert!(ok.clone().add_to(&mut m).is_err());
        assert!(
            c(vec![[3.0, 1.0, 1.0], [3.0, 1.0, 3.0]])
                .add_to(&mut m)
                .is_err()
        ); // names
        assert!(m.properties().len() == 4);
    }
}
