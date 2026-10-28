//! Lumped port: a resistive Thevenin source with voltage and current
//! probes (design §5.5).
//!
//! # Model
//!
//! A [`LumpedPort`] over a [`LumpedBox`] is a [`LumpedElement`] resistor
//! `R` (per-edge `g_e = M/(N·R)`) with an optional source `V_s(t)` in
//! series (Taflove & Hagness, *Computational Electrodynamics*, 3rd ed.,
//! ch. 15, resistive voltage source; Piket-May, Taflove and Baron, IEEE
//! Trans. MTT 42(8), 1994). A port without source is a matched passive
//! termination that still records `V` and `I`.
//!
//! # Conventions
//!
//! * **Voltage** `V = ∫E·dl` from `lo` to `hi` along `+axis`: the sum of the
//!   `M` edge voltages of a column, averaged over the `N` columns (for a
//!   uniform gap all columns agree; otherwise this is the mean column
//!   voltage). The `lo` end is the `+` terminal: an open-circuited port
//!   settles at `V = V_s`. Sampled after the E update of step `n`, at
//!   `t = (n + 1)·dt`.
//! * **Current** `I`: the terminal current **leaving the `+` terminal into
//!   the structure** (so `P = V·I > 0` is power delivered to the
//!   structure). It is the current through the Thevenin branch (source in
//!   series with `R`), i.e. minus the resistor's
//!   [terminal current](super::lumped#terminal-current) along `+axis`:
//!
//!   ```text
//!   I^{n+½} = (V_s^{n+½} − (V^n + V^{n+1})/2) / R
//!   ```
//!
//!   for a uniform port (the semi-implicit resistor current, exactly the
//!   term in the edges' Ampère law). Centred at `t = (n + ½)·dt`.
//!
//! With these conventions an ideal load `Z_L` gives `V = Z_L·I`, and the
//! source sees `V = V_s − R·I` up to the second-order time average of `V`.
//!
//! # Reference plane
//!
//! The port edges are ordinary Yee edges: besides the resistor they carry
//! their own background capacitance `C_p` (the material over the edges'
//! dual face, e.g. half a cell of line for a port on a boundary), and
//! Ampère's law on them reads
//!
//! ```text
//! C_p·(u^{n+1} − u^n)/dt + G_p·ū + i_R = ∮H·dl.
//! ```
//!
//! Both `V` and `I` are therefore defined *at the port edges*, and `C_p`,
//! `G_p` belong to the structure seen by the port, as in circuit theory:
//! the source branch current `−i_R` is split into the displacement and
//! conduction current of the port cell and the current `−∮H·dl` that
//! leaves it through the surrounding H loop.
//!
//! Measuring `I = −∮H·dl` instead (the H loop around the port edges)
//! excludes the port cell's own displacement current: that current is
//! taken half a cell away from the voltage. For a port feeding a matched
//! TEM line with phase advance `K = k̃·Δ` per cell, the loop current gives
//! `Z_in = Z·e^{jK/2}` and `|S11| = tan(K/4) ≈ K/4`, a first-order error in
//! the mesh size, although the power reflected towards the resistor is
//! second order. With the branch current, the port cell is a half-cell
//! (mid-shunt) termination of the discrete line, whose input impedance is
//! the line's real image impedance: with the Yee dispersion relation
//! `sin(K/2) = (ω̃·Δ/2)·√(L'C')` (`ω̃ = (2/dt)·sin(ω·dt/2)`) and
//! `C_p = C'·Δ/2`,
//!
//! ```text
//! Y_in = jω̃·C_p + e^{−jK/2}/Z = j·sin(K/2)/Z + (cos(K/2) − j·sin(K/2))/Z
//!      = cos(K/2)/Z,
//! ```
//!
//! so `Z_in = Z/cos(K/2)` and `|S11| = tan²(K/4)`: second order, and the
//! actual reflection seen by the resistor. (Image impedances of a
//! discretised line: e.g. Pozar, *Microwave Engineering*, periodic
//! structures / image parameter method; the lumped-port treatment follows
//! Taflove & Hagness, *Computational Electrodynamics*, 3rd ed., ch. 15, and
//! Piket-May, Taflove and Baron, IEEE Trans. MTT 42(8), 1994.)
//!
//! [`LumpedPort::analyze`] turns the two series into a
//! [`PortResult`] (DFTs with their own sample times, power waves,
//! S-parameters, impedance).

use super::Engine;
use super::fields::Fields;
use super::lumped::{LumpedBox, LumpedElement, Source};
use super::operator::{Hooks, Operator, OperatorHandle, StepContext};
use crate::axis::Axis;
use crate::post::{Complex64, PortResult, dft};
use crate::waveform::Waveform;
use crate::{Error, Result};

/// A lumped port operator; see the [module documentation](self).
#[derive(Debug)]
#[non_exhaustive]
pub struct LumpedPort {
    region: LumpedBox,
    element: LumpedElement,
    resistance: f64,
    edges: Vec<[usize; 3]>,
    dt: f64,
    /// Step index of the first sample.
    first_step: Option<u64>,
    v: Vec<f64>,
    i: Vec<f64>,
}

impl LumpedPort {
    /// Passive port of resistance `r` \[Ω\] over `region`.
    pub fn new(region: LumpedBox, r: f64) -> Self {
        Self {
            region,
            element: LumpedElement::resistor(region, r).record_current(),
            resistance: r,
            edges: Vec::new(),
            dt: 0.0,
            first_step: None,
            v: Vec::new(),
            i: Vec::new(),
        }
    }

    /// Excites the port with the Thevenin source `V_s(t)` \[V\], sampled at
    /// `t = (n + ½)·dt`.
    pub fn excite(mut self, v: impl Fn(f64) -> f64 + Send + 'static) -> Self {
        self.element.set_source(Some(Source::Fn(Box::new(v))));
        self
    }

    /// Excites the port with `amplitude · waveform(t)` \[V\].
    pub fn excite_waveform(mut self, waveform: Waveform, amplitude: f64) -> Self {
        self.element
            .set_source(Some(Source::Waveform(waveform, amplitude)));
        self
    }

    /// The port's box.
    pub fn region(&self) -> &LumpedBox {
        &self.region
    }

    /// Port resistance `R` \[Ω\] (also the default reference impedance).
    pub fn resistance(&self) -> f64 {
        self.resistance
    }

    /// Installs the port (coefficients and operator) and returns its handle.
    /// Fails, leaving the engine unchanged, under the same conditions as
    /// [`LumpedElement::add_to`].
    pub fn add_to(mut self, engine: &mut Engine) -> Result<OperatorHandle<LumpedPort>> {
        let region = self.region;
        region.check_fits(engine.grid().nodes())?;
        self.element.install(engine)?;
        self.dt = engine.dt();
        self.edges = region.edges();
        Ok(engine.add_operator(self))
    }

    /// Time step \[s\] (0 before installation).
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Recorded port voltages `V` \[V\] (see the module docs).
    pub fn voltage(&self) -> &[f64] {
        &self.v
    }

    /// Recorded port currents `I` \[A\] (see the module docs).
    pub fn current(&self) -> &[f64] {
        &self.i
    }

    /// Time of the first voltage sample, `(n0 + 1)·dt` \[s\]; samples are
    /// `dt` apart.
    pub fn voltage_t0(&self) -> f64 {
        (self.first_step.unwrap_or(0) as f64 + 1.0) * self.dt
    }

    /// Time of the first current sample, `(n0 + ½)·dt` \[s\]; samples are
    /// `dt` apart.
    pub fn current_t0(&self) -> f64 {
        (self.first_step.unwrap_or(0) as f64 + 0.5) * self.dt
    }

    /// Voltage sample times \[s\].
    pub fn voltage_times(&self) -> Vec<f64> {
        let t0 = self.voltage_t0();
        (0..self.v.len()).map(|n| t0 + n as f64 * self.dt).collect()
    }

    /// Current sample times \[s\].
    pub fn current_times(&self) -> Vec<f64> {
        let t0 = self.current_t0();
        (0..self.i.len()).map(|n| t0 + n as f64 * self.dt).collect()
    }

    /// Frequency-domain analysis at `freqs` \[Hz\] with the reference
    /// impedance `Z_ref = R`.
    pub fn analyze(&self, freqs: &[f64]) -> Result<PortResult> {
        self.analyze_with(freqs, Complex64::new(self.resistance, 0.0))
    }

    /// Frequency-domain analysis at `freqs` \[Hz\] with reference impedance
    /// `z_ref` (`Re z_ref > 0`).
    ///
    /// `V(f)` and `I(f)` are the DFTs ([`dft()`]) of the recorded series with
    /// their own absolute sample times (`voltage_t0`, `current_t0`), so the
    /// half-step offset of the current is accounted for exactly by the
    /// transform kernel.
    pub fn analyze_with(&self, freqs: &[f64], z_ref: Complex64) -> Result<PortResult> {
        if self.v.is_empty() {
            return Err(Error::Invalid("port has no samples yet".into()));
        }
        let v = dft(self.voltage_t0(), self.dt, &self.v, freqs);
        let i = dft(self.current_t0(), self.dt, &self.i, freqs);
        PortResult::new(freqs.to_vec(), v, i, z_ref)
    }
}

/// Dual edges `(axis, node, weight)` whose weighted sum is `−∮H·dl`
/// (right-handed about `+axis`) around the E edges along `axis` whose
/// transverse node coordinates lie in `lo ..= hi`, in the dual plane
/// crossing the edge layer `layer` (edges from node `layer` to
/// `layer + 1` along `axis`). `lo[axis]`, `hi[axis]` are ignored.
///
/// The E edge along `a` at node `p` is updated with
/// `(i_c[p] − i_c[p − e_b]) − (i_b[p] − i_b[p − e_c])` (`(a, b, c)`
/// cyclic). Summed over the rectangle of columns, interior terms cancel and
/// the boundary loop remains. Low-side entries outside the domain are
/// dropped (they read as zero in the kernel); high-side entries at the
/// domain edge are padding and stay zero.
pub(crate) fn loop_entries(
    axis: Axis,
    lo: [usize; 3],
    hi: [usize; 3],
    layer: usize,
) -> Vec<(Axis, [usize; 3], f64)> {
    let (ab, ac) = axis.others();
    let (a, b, c) = (axis.index(), ab.index(), ac.index());
    let node = |pb: usize, pc: usize| {
        let mut p = [0; 3];
        p[a] = layer;
        p[b] = pb;
        p[c] = pc;
        p
    };
    let mut v = Vec::new();
    for pc in lo[c]..=hi[c] {
        v.push((ac, node(hi[b], pc), -1.0));
        if lo[b] > 0 {
            v.push((ac, node(lo[b] - 1, pc), 1.0));
        }
    }
    for pb in lo[b]..=hi[b] {
        v.push((ab, node(pb, hi[c]), 1.0));
        if lo[c] > 0 {
            v.push((ab, node(pb, lo[c] - 1), -1.0));
        }
    }
    v
}

impl Operator for LumpedPort {
    fn hooks(&self) -> Hooks {
        Hooks::PRE_E | Hooks::POST_E
    }

    /// `pre_e` only saves the edge voltages.
    fn movable_hooks(&self) -> Hooks {
        Hooks::PRE_E
    }

    /// The element's branch energy ([`LumpedElement::branch_energy`]).
    fn energy(&self) -> f64 {
        self.element.branch_energy()
    }

    fn pre_e(&mut self, fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        self.element.pre_e_impl(fields);
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if self.first_step.is_none() {
            self.first_step = Some(ctx.step);
        }
        self.element.post_e_impl(fields, ctx);
        let i = self.element.current().last().copied().unwrap_or(0.0);
        self.i.push(-i);
        let axis = self.region.axis();
        let n = self.region.columns() as f64;
        let s: f64 = self
            .edges
            .iter()
            .map(|&p| fields.u.get(axis, p) as f64)
            .sum();
        self.v.push(s / n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mid_loop(region: &LumpedBox) -> Vec<(Axis, [usize; 3], f64)> {
        let a = region.axis().index();
        let layer = region.lo()[a] + (region.series_edges() - 1) / 2;
        loop_entries(region.axis(), region.lo(), region.hi(), layer)
    }

    #[test]
    fn loop_single_column() {
        // z column at (3, 4), z 1..=4 (M = 3): middle layer 2.
        let r = LumpedBox::new(Axis::Z, [3, 4, 1], [3, 4, 4]).unwrap();
        let l = mid_loop(&r);
        // curl_z = (iy[i] − iy[i−1]) − (ix[j] − ix[j−1]); I = −curl.
        let want = vec![
            (Axis::Y, [3, 4, 2], -1.0),
            (Axis::Y, [2, 4, 2], 1.0),
            (Axis::X, [3, 4, 2], 1.0),
            (Axis::X, [3, 3, 2], -1.0),
        ];
        assert_eq!(l, want);
        // At the low domain faces the outside entries are dropped.
        let r = LumpedBox::new(Axis::X, [0, 0, 0], [2, 1, 0]).unwrap();
        let l = mid_loop(&r);
        // (a, b, c) = (x, y, z); layer (2 − 1)/2 = 0.
        assert_eq!(
            l,
            vec![
                (Axis::Z, [0, 1, 0], -1.0),
                (Axis::Y, [0, 0, 0], 1.0),
                (Axis::Y, [0, 1, 0], 1.0),
            ]
        );
    }
}
