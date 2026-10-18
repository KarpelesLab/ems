//! Lumped circuit elements (R, L, C, Thevenin sources) on a box of E edges.
//!
//! # Geometry
//!
//! A [`LumpedBox`] is a closed box of grid nodes `lo ..= hi` with a
//! direction `axis`. Along `axis` it spans `M = hi[axis] − lo[axis] ≥ 1`
//! primary edges in series; across it, every node `(b, c)` of the
//! transverse rectangle carries one such column of edges, giving
//! `N = (hi[b] − lo[b] + 1)·(hi[c] − lo[c] + 1)` columns in parallel.
//!
//! # Distribution over the edges
//!
//! An element of impedance `Z` across the box is spread uniformly: each
//! column carries `N·Z`, split into `M` equal edge impedances, so every edge
//! gets `Z_e = N·Z/M` (design §5.5). Hence per edge
//!
//! ```text
//! g_e = M/(N·R),   L_e = N·L/M,   C_e = M·C/N,   R_e (series) = N·R/M,
//! ```
//!
//! and a Thevenin source `V_s` across the box contributes `V_s/M` per edge.
//!
//! # Update equations
//!
//! The lumped current is added to the edge's Ampère law in integral form
//! (design §5.2): `C·du/dt + G·u + i_lumped = Σi`. This is the lumped-element
//! FDTD extension of Piket-May, Taflove and Baron, "FD-TD modeling of
//! digital signal propagation in 3-D circuits with passive and active
//! loads", IEEE Trans. MTT 42(8), 1994, and Taflove & Hagness,
//! *Computational Electrodynamics*, 3rd ed., ch. 15 (lumped resistor,
//! capacitor, inductor and resistive voltage source), written for edge
//! voltages instead of field values.
//!
//! * **Resistor and parallel capacitor** are pure coefficient changes. The
//!   edge's present `C` and `G` are recovered from its `ca`, `cb`
//!   (`C = dt·(1 + ca)/(2·cb)`, `G = (1 − ca)/cb`), so the element composes
//!   with any material already on the edge; `g_e` and `C_e` are added and
//!   `ca`, `cb` recomputed with the semi-implicit loss average
//!   ([`fold_parallel`]).
//! * **Resistive voltage source** (Thevenin `V_s` in series with the
//!   resistor): the resistor current is `g_e·(u − V_s/M)`, so the semi-implicit
//!   update becomes `u^{n+1} = ca·u^n + cb·(Σi^{n+½} + g_e·V_s^{n+½}/M)`
//!   (Taflove & Hagness, resistive voltage source). The extra term is added
//!   after the E kernel; `V_s` is sampled at `t = (n + ½)·dt`, the centre of
//!   the E update.
//! * **Inductors and series RLC branches** need auxiliary state (an
//!   auxiliary differential equation per edge). The branch obeys
//!   `L_e·dj/dt + R_e·j + q/C_e = u − V_s/M`, `dq/dt = j`, with the branch
//!   current `j` entering Ampère's law. It is discretised with the trapezoidal
//!   rule centred at `(n + ½)·dt`, *coupled implicitly* with the edge
//!   voltage (the semi-implicit treatment of Piket-May et al. extended to a
//!   whole branch):
//!
//!   ```text
//!   L_e·(j^{n+1} − j^n)/dt + R_e·j̄ + (q^{n+1} + q^n)/(2C_e) = (u^{n+1} + u^n)/2 − V_s^{n+½}/M
//!   q^{n+1} = q^n + dt·j̄,   j̄ = (j^{n+1} + j^n)/2
//!   u^{n+1} = u* − cb·j̄,    u* = ca·u^n + cb·Σi^{n+½} (the kernel result)
//!   ```
//!
//!   This is linear in `u^{n+1}` and `j̄` and is solved in closed form per
//!   edge. The trapezoidal rule maps the passive branch impedance to a
//!   passive discrete one, so the lumped part adds no stability limit; for
//!   a pure resistor it reproduces the coefficient-folded update exactly.
//!
//! # Supported elements
//!
//! [`Topology::Parallel`]: any combination of `R ∥ L ∥ C` (R and C folded,
//! L through a one-element branch); a source must have an `R`, with which it
//! is in series. [`Topology::Series`]: any combination of `R + L + C` in
//! series (a pure R is folded, everything else uses the branch); the source
//! sits in the branch.
//!
//! An edge may carry several folded elements (they simply add), but at most
//! one element with a branch: branches on the same edge are not combined
//! implicitly.
//!
//! Besides the box form, an element can be given as an explicit **edge
//! list** with per-edge values ([`LumpedElement::from_edges`], used for csx
//! `LumpedElement`s, whose values csx distributes itself); every edge then
//! uses exactly the per-edge equations above with its own `g_e`, `C_e`,
//! `L_e`, `R_e`, and sources carry an explicit voltage per edge
//! ([`LumpedElement::edge_source`]) instead of `V_s/M`.
//!
//! # Terminal current
//!
//! With [`record_current`](LumpedElement::record_current) the element
//! records, every step, the current through it along `+axis`,
//!
//! ```text
//! i^{n+½} = (1/M)·Σ_e [ g_e·(ū_e − v_e) + C_e·(u_e^{n+1} − u_e^n)/dt + j̄_e ],
//! ū_e = (u_e^{n+1} + u_e^n)/2,
//! ```
//!
//! summed over all edges (columns in parallel) and divided by the number
//! `M` of series positions: the resistor (with its series source voltage
//! `v_e`), the folded capacitor and the branch current, exactly the terms
//! that the element adds to each edge's Ampère law. It is centred at
//! `t = (n + ½)·dt`, like the H currents. For a uniform element this is
//! the circuit current of the element's terminals; the edges' own
//! background capacitance and conductance (material) are *not* included,
//! they belong to the surrounding structure. [`LumpedPort`] uses it as the
//! port current (see [its docs](super::port#reference-plane)).
//!
//! [`LumpedPort`]: super::LumpedPort
//!
//! # Energy
//!
//! Folded capacitance is part of the edge's `C` and hence of the field
//! energy ([`Engine::energy`]). A branch stores, per edge,
//!
//! ```text
//! W_b^n = ½·L_e·(j^n)² + ½·(q^n)²/C_e
//! ```
//!
//! reported by [`branch_energy`](LumpedElement::branch_energy) and through
//! [`Operator::energy`] added to [`Engine::energy`]. With the field
//! invariant `W^n` (engine module docs), the E update gives
//! `W^{n+1} − W^n = −dt·ū·j̄` per branch edge (`ū = (u^{n+1} + u^n)/2`,
//! lossless background), and the trapezoidal branch update gives
//! `W_b^{n+1} − W_b^n = dt·j̄·(ū − V_s/M) − dt·R_e·j̄²`. So
//! `W + Σ W_b` changes only by the source work and the branch resistor
//! loss: for lossless L and C it is conserved (up to the `f32` rounding of
//! `u`), whereas `W` alone oscillates as energy moves into the branch.

use super::Engine;
use super::excite::Amplitude;
use super::fields::Fields;
use super::operator::{Hooks, Operator, OperatorHandle, StepContext};
use crate::axis::Axis;
use crate::waveform::Waveform;
use crate::{Error, Result};

/// A closed box of grid nodes with a direction, the support of a lumped
/// element or port (see the [module documentation](self)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LumpedBox {
    axis: Axis,
    lo: [usize; 3],
    hi: [usize; 3],
}

impl LumpedBox {
    /// Box of nodes `lo ..= hi` (inclusive on every axis), directed along
    /// `axis`. Requires `lo[axis] < hi[axis]` (at least one edge) and
    /// `lo ≤ hi` on the transverse axes.
    pub fn new(axis: Axis, lo: [usize; 3], hi: [usize; 3]) -> Result<Self> {
        let a = axis.index();
        if lo[a] >= hi[a] {
            return Err(Error::Invalid(format!(
                "lumped box {lo:?}..={hi:?} is degenerate along its direction {axis}"
            )));
        }
        if (0..3).any(|t| lo[t] > hi[t]) {
            return Err(Error::Invalid(format!(
                "lumped box {lo:?}..={hi:?} has lo > hi"
            )));
        }
        Ok(Self { axis, lo, hi })
    }

    /// Direction of the element.
    pub fn axis(&self) -> Axis {
        self.axis
    }

    /// Lowest node (inclusive).
    pub fn lo(&self) -> [usize; 3] {
        self.lo
    }

    /// Highest node (inclusive).
    pub fn hi(&self) -> [usize; 3] {
        self.hi
    }

    /// Number `M` of edges in series along the direction.
    pub fn series_edges(&self) -> usize {
        let a = self.axis.index();
        self.hi[a] - self.lo[a]
    }

    /// Number `N` of parallel edge columns.
    pub fn columns(&self) -> usize {
        let (b, c) = self.axis.others();
        let (b, c) = (b.index(), c.index());
        (self.hi[b] - self.lo[b] + 1) * (self.hi[c] - self.lo[c] + 1)
    }

    /// Start nodes of all `M·N` edges, column by column (deterministic
    /// order).
    pub fn edges(&self) -> Vec<[usize; 3]> {
        let a = self.axis.index();
        let (b, c) = self.axis.others();
        let (b, c) = (b.index(), c.index());
        let mut v = Vec::with_capacity(self.series_edges() * self.columns());
        for pb in self.lo[b]..=self.hi[b] {
            for pc in self.lo[c]..=self.hi[c] {
                for pa in self.lo[a]..self.hi[a] {
                    let mut p = [0; 3];
                    p[a] = pa;
                    p[b] = pb;
                    p[c] = pc;
                    v.push(p);
                }
            }
        }
        v
    }

    /// Checks that the box fits in a grid with `nodes` nodes per axis.
    pub fn check_fits(&self, nodes: [usize; 3]) -> Result<()> {
        if (0..3).any(|t| self.hi[t] >= nodes[t]) {
            return Err(Error::Invalid(format!(
                "lumped box {:?}..={:?} exceeds the grid ({nodes:?} nodes)",
                self.lo, self.hi
            )));
        }
        Ok(())
    }
}

/// How the R, L and C of a [`LumpedElement`] are connected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Topology {
    /// `R ∥ L ∥ C` (default).
    #[default]
    Parallel,
    /// `R + L + C` in series.
    Series,
}

/// Time function of a lumped source.
pub(crate) enum Source {
    Fn(Amplitude),
    Waveform(Waveform, f64),
}

impl Source {
    fn value(&self, t: f64, dt: f64) -> f64 {
        match self {
            Source::Fn(f) => f(t),
            Source::Waveform(w, a) => a * w.value(t, dt),
        }
    }

    fn validate(&self) -> Result<()> {
        if let Source::Waveform(w, a) = self {
            w.validate()?;
            if !a.is_finite() {
                return Err(Error::Invalid(format!(
                    "source amplitude {a} is not finite"
                )));
            }
        }
        Ok(())
    }
}

/// Adds a conductance `g` \[S\] and capacitance `c` \[F\] in parallel to an
/// E edge whose update coefficients are `(ca, cb)` for time step `dt`, and
/// returns the new `(ca, cb)`.
///
/// The edge's existing `C` and `G` are recovered from the coefficients
/// (`C = dt·(1 + ca)/(2·cb)`, `G = (1 − ca)/cb`, the inverse of
/// `ca = (C − G·dt/2)/(C + G·dt/2)`, `cb = dt/(C + G·dt/2)`), so this
/// composes with any material conductivity and with other lumped elements.
/// The recovery is exact up to the `f32` rounding of `ca`, `cb`
/// (relative ~1e-7 of `C/dt`).
///
/// # Panics
/// If `cb == 0` (a PEC or padding edge has no `C` to add to).
pub fn fold_parallel(ca: f32, cb: f32, dt: f64, g: f64, c: f64) -> (f32, f32) {
    assert!(cb != 0.0, "cannot fold a lumped element into a PEC edge");
    let (ca, cb) = (ca as f64, cb as f64);
    let cap = dt * (1.0 + ca) / (2.0 * cb) + c;
    let cond = (1.0 - ca) / cb + g;
    let den = cap + 0.5 * cond * dt;
    (((cap - 0.5 * cond * dt) / den) as f32, (dt / den) as f32)
}

/// The values of one edge of an edge-list lumped element
/// ([`LumpedElement::from_edges`]): the edge's own share of the element,
/// already distributed (no further `M`/`N` scaling is applied).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeValues {
    /// Start node of the E edge along the element's axis.
    pub node: [usize; 3],
    /// Resistance of this edge \[Ω\], if any.
    pub r: Option<f64>,
    /// Inductance of this edge \[H\], if any.
    pub l: Option<f64>,
    /// Capacitance of this edge \[F\], if any.
    pub c: Option<f64>,
}

impl EdgeValues {
    fn present(&self) -> [bool; 3] {
        [self.r.is_some(), self.l.is_some(), self.c.is_some()]
    }
}

/// Per-edge folded conductance and capacitance, and branch `(R, L, C)`.
type EdgeParts = (f64, f64, Option<(f64, f64, f64)>);

/// Where an element sits: a box (values distributed over its edges) or an
/// explicit edge list (values given per edge).
#[derive(Debug, Clone)]
enum Support {
    Box(LumpedBox),
    Edges {
        axis: Axis,
        edges: Vec<EdgeValues>,
        /// Number of distinct positions along the axis (edges in series).
        series: usize,
    },
}

/// Auxiliary state of a series `R_e + L_e + C_e` branch on each edge.
#[derive(Debug, Clone)]
struct Branch {
    /// `2·L_e/dt` per edge.
    two_l_dt: Vec<f64>,
    /// `1/C_e` per edge (0 without capacitor).
    inv_c: Vec<f64>,
    /// `Z = 2·L_e/dt + R_e + dt/(2·C_e)` per edge.
    z: Vec<f64>,
    /// Branch current `j^n` per edge \[A\].
    j: Vec<f64>,
    /// Branch charge `q^n` per edge \[C\].
    q: Vec<f64>,
    /// `u^n` saved before the E update.
    u_prev: Vec<f64>,
    /// `j̄ = (j^{n+1} + j^n)/2` of the last update, per edge.
    j_avg: Vec<f64>,
}

/// Recorder of the element's terminal current (see the
/// [module docs](self#terminal-current)).
#[derive(Debug, Clone)]
struct CurrentRecord {
    /// Folded conductance `g_e` per edge \[S\].
    g: Vec<f64>,
    /// Folded capacitance `C_e` per edge \[F\].
    c: Vec<f64>,
    /// `u^n` saved before the E update.
    u_prev: Vec<f64>,
    /// Step index of the first sample.
    first_step: Option<u64>,
    /// `i^{n+½}` per recorded step \[A\].
    values: Vec<f64>,
}

struct Installed {
    dt: f64,
    edges: Vec<[usize; 3]>,
    /// Final `cb` of each edge (after folding).
    cb: Vec<f64>,
    /// Per-edge conductance in series with the source (0 where the source
    /// sits in the branch).
    g_src: Vec<f64>,
    /// Edges in series along the axis (`M`).
    series: usize,
    /// Source time functions with their per-edge voltage weights.
    terms: Vec<(Source, Vec<f64>)>,
    /// Scratch: source voltage per edge of the current step.
    v: Vec<f64>,
    /// Whether the source drives the branch (series topology with branch).
    src_in_branch: bool,
    branch: Option<Branch>,
    record: Option<CurrentRecord>,
}

/// A lumped R/L/C element, optionally with Thevenin voltage sources, over
/// a [`LumpedBox`] or an explicit list of edges; see the
/// [module documentation](self) for the model.
///
/// Build it, then install it with [`add_to`](Self::add_to), which changes
/// the edge coefficients and registers the operator.
///
/// **Polarity.** The source voltage `V_s` acts along `+axis`: with the
/// element open-circuited, `Σu` along a column (`∫E·dl` from `lo` to `hi`)
/// tends to `V_s`, i.e. the `lo` end is the `+` terminal.
///
/// **Edge lists.** [`from_edges`](Self::from_edges) takes the per-edge
/// values directly (e.g. as distributed by `csx::discretize`), with the
/// same physics per edge as the box form; sources are given with a voltage
/// weight per edge ([`edge_source`](Self::edge_source)).
pub struct LumpedElement {
    support: Support,
    r: Option<f64>,
    l: Option<f64>,
    c: Option<f64>,
    topology: Topology,
    source: Option<Source>,
    edge_sources: Vec<(Source, Vec<f64>)>,
    record: bool,
    state: Option<Installed>,
}

impl core::fmt::Debug for LumpedElement {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LumpedElement")
            .field("support", &self.support)
            .field("r", &self.r)
            .field("l", &self.l)
            .field("c", &self.c)
            .field("topology", &self.topology)
            .field("source", &self.source.is_some())
            .field("edge_sources", &self.edge_sources.len())
            .field("record", &self.record)
            .finish_non_exhaustive()
    }
}

fn check_value(name: &str, v: f64) -> Result<()> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(Error::Invalid(format!(
            "lumped {name} = {v} must be finite and > 0"
        )))
    }
}

impl LumpedElement {
    /// Element over `region` with no R, L or C yet (parallel topology).
    pub fn new(region: LumpedBox) -> Self {
        Self {
            support: Support::Box(region),
            r: None,
            l: None,
            c: None,
            topology: Topology::Parallel,
            source: None,
            edge_sources: Vec::new(),
            record: false,
            state: None,
        }
    }

    /// A resistor `r` \[Ω\] over `region`.
    pub fn resistor(region: LumpedBox, r: f64) -> Self {
        Self::new(region).r(r)
    }

    /// Element on an explicit list of E edges along `axis`, with values
    /// given per edge (not distributed further) and connected per edge
    /// according to `topology`.
    ///
    /// All edges must have the same set of R/L/C present (so the element
    /// either uses branch state on every edge or on none), and no edge may
    /// be listed twice.
    pub fn from_edges(axis: Axis, topology: Topology, edges: Vec<EdgeValues>) -> Result<Self> {
        let Some(first) = edges.first() else {
            return Err(Error::Invalid("lumped element has no edges".into()));
        };
        let pattern = first.present();
        if edges.iter().any(|e| e.present() != pattern) {
            return Err(Error::Invalid(
                "lumped element edges must all have the same set of R, L, C".into(),
            ));
        }
        let mut nodes: Vec<[usize; 3]> = edges.iter().map(|e| e.node).collect();
        nodes.sort_unstable();
        if nodes.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::Invalid("lumped element lists an edge twice".into()));
        }
        let a = axis.index();
        let mut along: Vec<usize> = nodes.iter().map(|n| n[a]).collect();
        along.sort_unstable();
        along.dedup();
        Ok(Self {
            support: Support::Edges {
                axis,
                series: along.len(),
                edges,
            },
            r: None,
            l: None,
            c: None,
            topology,
            source: None,
            edge_sources: Vec::new(),
            record: false,
            state: None,
        })
    }

    /// Sets the resistance \[Ω\] (box elements).
    pub fn r(mut self, r: f64) -> Self {
        self.r = Some(r);
        self
    }

    /// Sets the inductance \[H\] (box elements).
    pub fn l(mut self, l: f64) -> Self {
        self.l = Some(l);
        self
    }

    /// Sets the capacitance \[F\] (box elements).
    pub fn c(mut self, c: f64) -> Self {
        self.c = Some(c);
        self
    }

    /// Sets how R, L, C are connected (default [`Topology::Parallel`]).
    pub fn topology(mut self, t: Topology) -> Self {
        self.topology = t;
        self
    }

    /// Thevenin source `V_s(t)` \[V\] across the box, in series with the
    /// resistor (parallel topology) or the branch (series topology),
    /// sampled at `t = (n + ½)·dt`. Each of the `M` series edges gets
    /// `V_s/M`. For an edge-list element use
    /// [`edge_source`](Self::edge_source).
    pub fn source(mut self, v: impl Fn(f64) -> f64 + Send + 'static) -> Self {
        self.source = Some(Source::Fn(Box::new(v)));
        self
    }

    /// Thevenin source `amplitude · waveform(t)` \[V\]; see
    /// [`source`](Self::source). [`Waveform::Dirac`] fires at the step
    /// whose E update is centred on `t = dt/2`.
    pub fn waveform(mut self, waveform: Waveform, amplitude: f64) -> Self {
        self.source = Some(Source::Waveform(waveform, amplitude));
        self
    }

    /// Adds a Thevenin source with per-edge voltages `weights[e]·v(t)`
    /// \[V\] (one weight per edge, in the order of
    /// [`edge_nodes`](Self::edge_nodes)), in series with each edge's
    /// resistor or branch like [`source`](Self::source). Several sources
    /// add up.
    pub fn edge_source(
        mut self,
        weights: Vec<f64>,
        v: impl Fn(f64) -> f64 + Send + 'static,
    ) -> Self {
        self.edge_sources.push((Source::Fn(Box::new(v)), weights));
        self
    }

    /// [`edge_source`](Self::edge_source) with the time function
    /// `amplitude · waveform(t)`.
    pub fn edge_source_waveform(
        mut self,
        weights: Vec<f64>,
        waveform: Waveform,
        amplitude: f64,
    ) -> Self {
        self.edge_sources
            .push((Source::Waveform(waveform, amplitude), weights));
        self
    }

    /// Records the element's terminal current every step (see
    /// [Terminal current](self#terminal-current)); read it with
    /// [`current`](Self::current). Adds a `pre_e` and a `post_e` hook.
    pub fn record_current(mut self) -> Self {
        self.record = true;
        self
    }

    pub(crate) fn set_source(&mut self, s: Option<Source>) {
        self.source = s;
    }

    /// The element's box (`None` for an edge-list element).
    pub fn region(&self) -> Option<&LumpedBox> {
        match &self.support {
            Support::Box(b) => Some(b),
            Support::Edges { .. } => None,
        }
    }

    /// Direction of the element's edges.
    pub fn axis(&self) -> Axis {
        match &self.support {
            Support::Box(b) => b.axis(),
            Support::Edges { axis, .. } => *axis,
        }
    }

    /// Start nodes of the element's edges (box: column by column; edge
    /// list: as given).
    pub fn edge_nodes(&self) -> Vec<[usize; 3]> {
        match &self.support {
            Support::Box(b) => b.edges(),
            Support::Edges { edges, .. } => edges.iter().map(|e| e.node).collect(),
        }
    }

    /// Number `M` of edges in series along the axis (for an edge list:
    /// the number of distinct positions along the axis).
    pub fn series_edges(&self) -> usize {
        match &self.support {
            Support::Box(b) => b.series_edges(),
            Support::Edges { series, .. } => *series,
        }
    }

    /// The per-edge values of an edge-list element (empty for a box).
    pub fn edge_values(&self) -> &[EdgeValues] {
        match &self.support {
            Support::Box(_) => &[],
            Support::Edges { edges, .. } => edges,
        }
    }

    /// Resistance \[Ω\] of a box element, if any.
    pub fn resistance(&self) -> Option<f64> {
        self.r
    }

    /// Inductance \[H\] of a box element, if any.
    pub fn inductance(&self) -> Option<f64> {
        self.l
    }

    /// Capacitance \[F\] of a box element, if any.
    pub fn capacitance(&self) -> Option<f64> {
        self.c
    }

    /// The topology.
    pub fn get_topology(&self) -> Topology {
        self.topology
    }

    /// Which of R, L, C are present.
    fn present(&self) -> [bool; 3] {
        match &self.support {
            Support::Box(_) => [self.r.is_some(), self.l.is_some(), self.c.is_some()],
            Support::Edges { edges, .. } => edges[0].present(),
        }
    }

    fn has_source(&self) -> bool {
        self.source.is_some() || !self.edge_sources.is_empty()
    }

    /// Whether the element uses auxiliary branch state (inductor or series
    /// RLC) rather than a pure coefficient change.
    pub fn has_branch(&self) -> bool {
        let [_, l, c] = self.present();
        match self.topology {
            Topology::Parallel => l,
            Topology::Series => l || c,
        }
    }

    /// Validates the values (finite, `> 0`, at least one of R/L/C, a
    /// parallel source needs an R, one weight per edge for edge sources).
    pub fn validate(&self) -> Result<()> {
        match &self.support {
            Support::Box(_) => {
                for (name, v) in [("R", self.r), ("L", self.l), ("C", self.c)] {
                    if let Some(v) = v {
                        check_value(name, v)?;
                    }
                }
            }
            Support::Edges { edges, .. } => {
                for e in edges {
                    for (name, v) in [("R", e.r), ("L", e.l), ("C", e.c)] {
                        if let Some(v) = v {
                            check_value(name, v)?;
                        }
                    }
                }
            }
        }
        let [r, l, c] = self.present();
        if !(r || l || c) {
            return Err(Error::Invalid(
                "lumped element needs at least one of R, L, C".into(),
            ));
        }
        if self.has_source() && self.topology == Topology::Parallel && !r {
            return Err(Error::Invalid(
                "a source on a parallel lumped element needs a resistance".into(),
            ));
        }
        if let Some(s) = &self.source {
            s.validate()?;
        }
        let ne = self.edge_nodes().len();
        for (s, w) in &self.edge_sources {
            s.validate()?;
            if w.len() != ne {
                return Err(Error::Invalid(format!(
                    "edge source has {} weights for {ne} edges",
                    w.len()
                )));
            }
            if let Some(x) = w.iter().find(|x| !x.is_finite()) {
                return Err(Error::Invalid(format!(
                    "edge source weight {x} is not finite"
                )));
            }
        }
        Ok(())
    }

    /// Checks that every edge exists in a grid of `nodes` nodes.
    fn check_fits(&self, nodes: [usize; 3]) -> Result<()> {
        match &self.support {
            Support::Box(b) => b.check_fits(nodes),
            Support::Edges { axis, edges, .. } => {
                let a = axis.index();
                for e in edges {
                    let n = e.node;
                    if (0..3).any(|t| n[t] >= nodes[t]) || n[a] + 1 >= nodes[a] {
                        return Err(Error::Invalid(format!(
                            "lumped edge {axis}:{n:?} is outside the grid ({nodes:?} nodes)"
                        )));
                    }
                }
                Ok(())
            }
        }
    }

    /// Per-edge `(g_e, C_e, branch (R_e, L_e, C_e))`, in edge order.
    fn edge_parts(&self) -> Vec<EdgeParts> {
        let has_branch = self.has_branch();
        let parts = |r: Option<f64>, l: Option<f64>, c: Option<f64>| match self.topology {
            Topology::Parallel => (
                r.map_or(0.0, |r| 1.0 / r),
                c.unwrap_or(0.0),
                l.map(|l| (0.0, l, f64::INFINITY)),
            ),
            Topology::Series if !has_branch => (r.map_or(0.0, |r| 1.0 / r), 0.0, None),
            Topology::Series => (
                0.0,
                0.0,
                Some((
                    r.unwrap_or(0.0),
                    l.unwrap_or(0.0),
                    c.unwrap_or(f64::INFINITY),
                )),
            ),
        };
        match &self.support {
            Support::Box(b) => {
                let (m, n) = (b.series_edges() as f64, b.columns() as f64);
                // Z_e = N·Z/M (module docs); conductances and capacitances
                // scale inversely.
                let p = match self.topology {
                    Topology::Parallel => (
                        self.r.map_or(0.0, |r| m / (n * r)),
                        self.c.map_or(0.0, |c| m * c / n),
                        self.l.map(|l| (0.0, n * l / m, f64::INFINITY)),
                    ),
                    Topology::Series if !has_branch => {
                        (self.r.map_or(0.0, |r| m / (n * r)), 0.0, None)
                    }
                    Topology::Series => (
                        0.0,
                        0.0,
                        Some((
                            self.r.map_or(0.0, |r| n * r / m),
                            self.l.map_or(0.0, |l| n * l / m),
                            self.c.map_or(f64::INFINITY, |c| m * c / n),
                        )),
                    ),
                };
                vec![p; b.edges().len()]
            }
            Support::Edges { edges, .. } => edges.iter().map(|e| parts(e.r, e.l, e.c)).collect(),
        }
    }

    /// Changes the edge coefficients of `engine` and sets up the auxiliary
    /// state; the engine is left unchanged on error.
    pub(crate) fn install(&mut self, engine: &mut Engine) -> Result<()> {
        if self.state.is_some() {
            return Err(Error::Invalid("lumped element already installed".into()));
        }
        self.validate()?;
        let nodes = engine.grid().nodes();
        self.check_fits(nodes)?;
        let dt = engine.dt();
        let edges = self.edge_nodes();
        let parts = self.edge_parts();
        let axis = self.axis();
        let idx = |p: [usize; 3]| (p[0] * nodes[1] + p[1]) * nodes[2] + p[2];
        let mut co = engine.coefficients_mut();
        let (ca, cb) = co.e_mut(axis);
        if let Some(p) = edges.iter().find(|&&p| cb[idx(p)] == 0.0) {
            return Err(Error::Invalid(format!(
                "lumped element edge {axis}:{p:?} is PEC (metal or a PEC outer face)"
            )));
        }
        let mut cbs = Vec::with_capacity(edges.len());
        for (&p, &(g_e, c_e, _)) in edges.iter().zip(&parts) {
            let i = idx(p);
            if g_e != 0.0 || c_e != 0.0 {
                (ca[i], cb[i]) = fold_parallel(ca[i], cb[i], dt, g_e, c_e);
            }
            cbs.push(cb[i] as f64);
        }
        drop(co);
        let ne = edges.len();
        let has_branch = self.has_branch();
        let branch = has_branch.then(|| {
            let mut b = Branch {
                two_l_dt: Vec::with_capacity(ne),
                inv_c: Vec::with_capacity(ne),
                z: Vec::with_capacity(ne),
                j: vec![0.0; ne],
                q: vec![0.0; ne],
                u_prev: vec![0.0; ne],
                j_avg: vec![0.0; ne],
            };
            for &(_, _, br) in &parts {
                let (r, l, c) = br.unwrap_or((0.0, 0.0, f64::INFINITY));
                let inv_c = if c.is_finite() { 1.0 / c } else { 0.0 };
                b.two_l_dt.push(2.0 * l / dt);
                b.inv_c.push(inv_c);
                b.z.push(2.0 * l / dt + r + 0.5 * dt * inv_c);
            }
            b
        });
        let src_in_branch = has_branch && self.topology == Topology::Series;
        let g_src = parts
            .iter()
            .map(|&(g, _, _)| if src_in_branch { 0.0 } else { g })
            .collect();
        let series = self.series_edges();
        let record = self.record.then(|| CurrentRecord {
            g: parts.iter().map(|p| p.0).collect(),
            c: parts.iter().map(|p| p.1).collect(),
            u_prev: vec![0.0; ne],
            first_step: None,
            values: Vec::new(),
        });
        let mut terms = Vec::new();
        if let Some(s) = self.source.take() {
            terms.push((s, vec![1.0 / series as f64; ne]));
        }
        terms.append(&mut self.edge_sources);
        self.state = Some(Installed {
            dt,
            edges,
            cb: cbs,
            g_src,
            series,
            v: vec![0.0; if terms.is_empty() { 0 } else { ne }],
            terms,
            src_in_branch,
            branch,
            record,
        });
        Ok(())
    }

    /// Installs the element into `engine` (coefficient changes and, for
    /// sources and branches, an operator) and returns its handle.
    ///
    /// Fails, leaving the engine unchanged, if the values are invalid, the
    /// element does not fit the grid or any of its edges is PEC (`cb = 0`,
    /// e.g. metal or a PEC outer face).
    pub fn add_to(mut self, engine: &mut Engine) -> Result<OperatorHandle<LumpedElement>> {
        self.install(engine)?;
        Ok(engine.add_operator(self))
    }

    /// Per-edge conductance `g_e` in series with the source, of the first
    /// edge (0 if none or before installation).
    pub fn edge_conductance(&self) -> f64 {
        self.state
            .as_ref()
            .and_then(|s| s.g_src.first().copied())
            .unwrap_or(0.0)
    }

    /// Average branch current `j̄` of the last step, summed over all edges
    /// and divided by the number `M` of series edges (i.e. the element's
    /// branch current along `+axis` for the inductor / series branch),
    /// \[A\]; 0 without a branch.
    pub fn branch_current(&self) -> f64 {
        let Some(s) = &self.state else { return 0.0 };
        let Some(b) = &s.branch else { return 0.0 };
        b.j_avg.iter().sum::<f64>() / s.series as f64
    }

    /// Recorded terminal currents `i^{n+½}` along `+axis` \[A\] (see
    /// [Terminal current](self#terminal-current)); empty unless
    /// [`record_current`](Self::record_current) was set.
    pub fn current(&self) -> &[f64] {
        self.state
            .as_ref()
            .and_then(|s| s.record.as_ref())
            .map_or(&[], |r| &r.values)
    }

    /// Time of the first [`current`](Self::current) sample,
    /// `(n0 + ½)·dt` \[s\]; samples are `dt` apart.
    pub fn current_t0(&self) -> f64 {
        let Some(s) = &self.state else { return 0.0 };
        let n0 = s.record.as_ref().and_then(|r| r.first_step).unwrap_or(0);
        (n0 as f64 + 0.5) * s.dt
    }

    /// Sample times of [`current`](Self::current) \[s\].
    pub fn current_times(&self) -> Vec<f64> {
        let t0 = self.current_t0();
        let dt = self.state.as_ref().map_or(0.0, |s| s.dt);
        (0..self.current().len())
            .map(|n| t0 + n as f64 * dt)
            .collect()
    }

    /// Energy stored in the branch state, `Σ ½·L_e·j² + ½·q²/C_e` over
    /// the edges at `t = n·dt` \[J\] (see [Energy](self#energy)); 0
    /// without a branch or before installation.
    pub fn branch_energy(&self) -> f64 {
        let Some(s) = &self.state else { return 0.0 };
        let Some(b) = &s.branch else { return 0.0 };
        (0..b.j.len())
            .map(|e| {
                let half_l = 0.25 * b.two_l_dt[e] * s.dt;
                half_l * b.j[e] * b.j[e] + 0.5 * b.inv_c[e] * b.q[e] * b.q[e]
            })
            .sum()
    }

    /// Serial hooks used: `pre_e` and `post_e` with a branch or a current
    /// record, `post_e` with only a source, none for a pure coefficient
    /// change.
    pub(crate) fn used_hooks(&self) -> Hooks {
        let (branch, source) = match &self.state {
            Some(s) => (
                s.branch.is_some() || s.record.is_some(),
                !s.terms.is_empty(),
            ),
            None => (self.has_branch() || self.record, self.has_source()),
        };
        if branch {
            Hooks::PRE_E | Hooks::POST_E
        } else if source {
            Hooks::POST_E
        } else {
            Hooks::NONE
        }
    }

    pub(crate) fn pre_e_impl(&mut self, fields: &mut Fields<'_>) {
        let axis = self.axis();
        let Some(s) = &mut self.state else { return };
        if let Some(b) = &mut s.branch {
            for (u, &p) in b.u_prev.iter_mut().zip(&s.edges) {
                *u = fields.u.get(axis, p) as f64;
            }
        }
        if let Some(r) = &mut s.record {
            for (u, &p) in r.u_prev.iter_mut().zip(&s.edges) {
                *u = fields.u.get(axis, p) as f64;
            }
        }
    }

    pub(crate) fn post_e_impl(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        let axis = self.axis();
        let Some(s) = &mut self.state else { return };
        let mut active = false;
        if !s.terms.is_empty() {
            s.v.fill(0.0);
            for (src, w) in &s.terms {
                let val = src.value(ctx.h_time(), s.dt);
                if val != 0.0 {
                    active = true;
                    for (v, &w) in s.v.iter_mut().zip(w) {
                        *v += w * val;
                    }
                }
            }
        }
        if active && !s.src_in_branch {
            for (e, &p) in s.edges.iter().enumerate() {
                let add = s.cb[e] * s.g_src[e] * s.v[e];
                if add != 0.0 {
                    let u = fields.u.get_mut(axis, p);
                    *u = (*u as f64 + add) as f32;
                }
            }
        }
        if let Some(b) = &mut s.branch {
            // Series source sits in the branch; parallel source is in the
            // folded resistor.
            for (e, (&p, &cb)) in s.edges.iter().zip(&s.cb).enumerate() {
                let v_br = if active && s.src_in_branch {
                    s.v[e]
                } else {
                    0.0
                };
                let u_star = fields.u.get(axis, p) as f64;
                let k = b.two_l_dt[e] * b.j[e] - b.q[e] * b.inv_c[e] - v_br + 0.5 * b.u_prev[e];
                let z = b.z[e];
                let u_new = (z * u_star - cb * k) / (z + 0.5 * cb);
                let jb = (k + 0.5 * u_new) / z;
                b.j[e] = 2.0 * jb - b.j[e];
                b.q[e] += s.dt * jb;
                b.j_avg[e] = jb;
                fields.u.set(axis, p, u_new as f32);
            }
        }
        if let Some(r) = &mut s.record {
            if r.first_step.is_none() {
                r.first_step = Some(ctx.step);
            }
            // Deterministic: fixed edge order, f64 accumulation.
            let mut sum = 0.0;
            for (e, &p) in s.edges.iter().enumerate() {
                let (u0, u1) = (r.u_prev[e], fields.u.get(axis, p) as f64);
                let v = if active && !s.src_in_branch {
                    s.v[e]
                } else {
                    0.0
                };
                sum += r.g[e] * (0.5 * (u0 + u1) - v) + r.c[e] * (u1 - u0) / s.dt;
                if let Some(b) = &s.branch {
                    sum += b.j_avg[e];
                }
            }
            r.values.push(sum / s.series as f64);
        }
    }
}

impl Operator for LumpedElement {
    fn hooks(&self) -> Hooks {
        self.used_hooks()
    }

    /// [`branch_energy`](Self::branch_energy).
    fn energy(&self) -> f64 {
        self.branch_energy()
    }

    fn pre_e(&mut self, fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        self.pre_e_impl(fields);
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        self.post_e_impl(fields, ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::EPS0;
    use crate::engine::{
        CellMaterials, Coefficients, FaceConditions, Grid, Material, primary_edge_cg,
    };

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    #[test]
    fn box_counts_and_validation() {
        let b = LumpedBox::new(Axis::Z, [1, 2, 3], [2, 4, 6]).unwrap();
        assert_eq!(b.series_edges(), 3);
        assert_eq!(b.columns(), 2 * 3);
        let e = b.edges();
        assert_eq!(e.len(), 18);
        assert_eq!(e[0], [1, 2, 3]);
        assert_eq!(e[2], [1, 2, 5]);
        assert!(LumpedBox::new(Axis::Z, [1, 2, 3], [1, 2, 3]).is_err());
        assert!(LumpedBox::new(Axis::X, [2, 0, 0], [1, 0, 0]).is_err());
        assert!(LumpedBox::new(Axis::X, [0, 1, 0], [1, 0, 0]).is_err());
        assert!(b.check_fits([3, 5, 7]).is_ok());
        assert!(b.check_fits([3, 5, 6]).is_err());
        let r = LumpedBox::new(Axis::Y, [0, 0, 0], [0, 1, 0]).unwrap();
        assert!(LumpedElement::resistor(r, -1.0).validate().is_err());
        assert!(LumpedElement::resistor(r, f64::NAN).validate().is_err());
        assert!(LumpedElement::new(r).validate().is_err());
        assert!(
            LumpedElement::new(r)
                .c(1e-12)
                .source(|_| 1.0)
                .validate()
                .is_err()
        );
        assert!(
            LumpedElement::new(r)
                .c(1e-12)
                .topology(Topology::Series)
                .source(|_| 1.0)
                .validate()
                .is_ok()
        );
        assert!(LumpedElement::new(r).l(1e-9).has_branch());
        assert!(!LumpedElement::new(r).r(1.0).c(1e-12).has_branch());
        assert!(
            LumpedElement::new(r)
                .r(1.0)
                .c(1e-12)
                .topology(Topology::Series)
                .has_branch()
        );
    }

    /// Folding `g` into an edge of a conductive material must give the same
    /// coefficients as a material with `G + g` from the start.
    #[test]
    fn fold_composes_with_conductive_material() {
        let g = Grid::uniform([4, 4, 4], [1e-3; 3]).unwrap();
        let kappa = 0.7;
        let mats = CellMaterials::new(&g, Material::dielectric(2.0).with_kappa(kappa));
        let dt = 1e-12;
        let co = Coefficients::compute(&g, &mats, &FaceConditions::default(), dt).unwrap();
        let node = [1, 2, 1];
        let i = g.index(node);
        let (c0, g0) = primary_edge_cg(&g, &mats, Axis::Z, node);
        assert!(rel(c0, EPS0 * 2.0 * 1e-3) < 1e-12);
        let (gl, cl) = (0.02, 3e-14);
        let (ca, cb) = fold_parallel(co.ca(Axis::Z)[i], co.cb(Axis::Z)[i], dt, gl, cl);
        let (cc, gg) = (c0 + cl, g0 + gl);
        let want_ca = (cc - 0.5 * gg * dt) / (cc + 0.5 * gg * dt);
        let want_cb = dt / (cc + 0.5 * gg * dt);
        assert!(rel(ca as f64, want_ca) < 1e-6, "{ca} vs {want_ca}");
        assert!(rel(cb as f64, want_cb) < 1e-6, "{cb} vs {want_cb}");
        // Folding nothing is the identity (up to f32 round trip).
        let (ca0, cb0) = fold_parallel(co.ca(Axis::Z)[i], co.cb(Axis::Z)[i], dt, 0.0, 0.0);
        assert!(rel(ca0 as f64, co.ca(Axis::Z)[i] as f64) < 1e-6);
        assert!(rel(cb0 as f64, co.cb(Axis::Z)[i] as f64) < 1e-6);
        // Two folds add up.
        let (a1, b1) = fold_parallel(co.ca(Axis::Z)[i], co.cb(Axis::Z)[i], dt, 0.01, 1e-14);
        let (a2, b2) = fold_parallel(a1, b1, dt, 0.01, 2e-14);
        assert!(rel(a2 as f64, want_ca) < 1e-5);
        assert!(rel(b2 as f64, want_cb) < 1e-5);
        // Recovered capacitance is C0 + cl.
        assert!(rel(Coefficients::capacitance(ca, cb, dt), cc) < 1e-6);
    }

    #[test]
    fn install_distributes_over_edges() {
        let g = Grid::uniform([6, 6, 6], [1e-3; 3]).unwrap();
        let mut e = Engine::builder(g.clone()).threads(1).build().unwrap();
        let dt = e.dt();
        let before = e.coefficients().cb(Axis::X).to_vec();
        let cab = e.coefficients().ca(Axis::X).to_vec();
        // 3 series x edges, 2 x 2 columns, R = 30 Ω, C = 2 pF.
        let bx = LumpedBox::new(Axis::X, [1, 2, 2], [4, 3, 3]).unwrap();
        let h = LumpedElement::new(bx)
            .r(30.0)
            .c(2e-12)
            .source(|_| 1.0)
            .add_to(&mut e)
            .unwrap();
        let ge = 3.0 / (4.0 * 30.0);
        let ce = 3.0 * 2e-12 / 4.0;
        assert!(rel(e.operator(&h).edge_conductance(), ge) < 1e-15);
        let co = e.coefficients();
        for p in bx.edges() {
            let i = g.index(p);
            let (wa, wb) = fold_parallel(cab[i], before[i], dt, ge, ce);
            assert_eq!((co.ca(Axis::X)[i], co.cb(Axis::X)[i]), (wa, wb));
        }
        // Outside the box nothing changed.
        let i = g.index([0, 2, 2]);
        assert_eq!(co.cb(Axis::X)[i], before[i]);
        drop(co);
        // PEC edge rejected, engine untouched.
        let pec = LumpedBox::new(Axis::Y, [0, 1, 1], [0, 2, 1]).unwrap();
        let snapshot = e.coefficients().clone();
        assert!(LumpedElement::resistor(pec, 50.0).add_to(&mut e).is_err());
        assert_eq!(*e.coefficients(), snapshot);
        let out = LumpedBox::new(Axis::Y, [1, 1, 1], [1, 7, 1]).unwrap();
        assert!(LumpedElement::resistor(out, 50.0).add_to(&mut e).is_err());
    }

    /// The trapezoidal branch with only R must reproduce the folded
    /// resistor bit-for-bit up to rounding.
    #[test]
    fn branch_resistor_matches_fold() {
        let run = |series_rc: bool| {
            let g = Grid::uniform([5, 5, 5], [1e-3; 3]).unwrap();
            let mut e = Engine::builder(g).threads(1).build().unwrap();
            let bx = LumpedBox::new(Axis::Z, [2, 2, 1], [2, 2, 3]).unwrap();
            let el = LumpedElement::resistor(bx, 20.0)
                .source(|t| (t * 3e10).sin())
                .topology(Topology::Series);
            // A series R with a huge C behaves like the plain resistor.
            let el = if series_rc { el.c(1.0) } else { el };
            assert_eq!(el.has_branch(), series_rc);
            el.add_to(&mut e).unwrap();
            e.run(300);
            e.u_component(Axis::Z)
        };
        let a = run(false);
        let b = run(true);
        let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.0);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() <= 1e-4 * scale, "{x} vs {y}");
        }
    }

    /// An edge list with the box's distributed values and per-edge source
    /// weights `1/M` reproduces the box element (parallel R ∥ L ∥ C and
    /// series R + L + C, both with a source).
    #[test]
    fn edge_list_matches_box() {
        let (r, l, c) = (40.0, 2e-9, 1e-12);
        for topo in [Topology::Parallel, Topology::Series] {
            let bx = LumpedBox::new(Axis::Z, [2, 2, 1], [3, 2, 3]).unwrap();
            let (m, n) = (2.0, 2.0);
            let src = |t: f64| (t * 2e10).sin();
            let run = |el: LumpedElement| {
                let g = Grid::uniform([5, 5, 5], [1e-3; 3]).unwrap();
                let mut e = Engine::builder(g).threads(1).build().unwrap();
                let h = el.add_to(&mut e).unwrap();
                e.run(200);
                (e.u_component(Axis::Z), e.operator(&h).branch_current())
            };
            let a = run(LumpedElement::new(bx)
                .r(r)
                .l(l)
                .c(c)
                .topology(topo)
                .source(src));
            let edges: Vec<EdgeValues> = bx
                .edges()
                .into_iter()
                .map(|node| EdgeValues {
                    node,
                    r: Some(n * r / m),
                    l: Some(n * l / m),
                    c: Some(m * c / n),
                })
                .collect();
            let w = vec![1.0 / m; edges.len()];
            let el = LumpedElement::from_edges(Axis::Z, topo, edges)
                .unwrap()
                .edge_source(w, src);
            assert!(el.region().is_none());
            assert_eq!(el.series_edges(), 2);
            let b = run(el);
            let scale = a.0.iter().fold(0.0f32, |s, v| s.max(v.abs()));
            assert!(scale > 0.0);
            for (x, y) in a.0.iter().zip(&b.0) {
                assert!((x - y).abs() <= 1e-5 * scale, "{topo:?}: {x} vs {y}");
            }
            assert!((a.1 - b.1).abs() <= 1e-5 * a.1.abs().max(1e-12));
        }
        // Mixed R/L/C presence and duplicate edges are rejected.
        let e = |node, r| EdgeValues {
            node,
            r,
            l: None,
            c: None,
        };
        assert!(
            LumpedElement::from_edges(
                Axis::Z,
                Topology::Parallel,
                vec![e([1, 1, 1], Some(1.0)), e([1, 1, 2], None)]
            )
            .is_err()
        );
        assert!(
            LumpedElement::from_edges(
                Axis::Z,
                Topology::Parallel,
                vec![e([1, 1, 1], Some(1.0)), e([1, 1, 1], Some(1.0))]
            )
            .is_err()
        );
        assert!(LumpedElement::from_edges(Axis::Z, Topology::Parallel, vec![]).is_err());
    }
}
