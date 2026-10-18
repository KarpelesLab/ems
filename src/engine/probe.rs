//! Probe operators.

use super::fields::Fields;
use super::operator::{Operator, StepContext};

/// Which field a [`Probe`] samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeQuantity {
    /// E edge voltages `u`, sampled after the E update at `(n + 1)·dt`.
    Voltage,
    /// H dual-edge currents `i`, sampled after the H update at
    /// `(n + ½)·dt`.
    Current,
}

/// Records the weighted sum `Σ w·x` of a set of field entries every
/// `interval` steps, with the exact sample times.
///
/// With unit weights along a chain of E edges this is the line integral of
/// E (a voltage); along a loop of H dual edges, the enclosed current.
#[derive(Debug, Clone)]
pub struct Probe {
    quantity: ProbeQuantity,
    entries: Vec<(usize, [usize; 3], f64)>,
    interval: u64,
    times: Vec<f64>,
    values: Vec<f64>,
}

impl Probe {
    /// Empty probe of `quantity`, sampling every step.
    pub fn new(quantity: ProbeQuantity) -> Self {
        Self {
            quantity,
            entries: Vec::new(),
            interval: 1,
            times: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Probe of the single E edge along `axis` at `node`.
    pub fn voltage(axis: usize, node: [usize; 3]) -> Self {
        Self::new(ProbeQuantity::Voltage).entry(axis, node, 1.0)
    }

    /// Probe of the single H dual edge along `axis` at `node`.
    pub fn current(axis: usize, node: [usize; 3]) -> Self {
        Self::new(ProbeQuantity::Current).entry(axis, node, 1.0)
    }

    /// Adds component `axis` at `node` with `weight`.
    pub fn entry(mut self, axis: usize, node: [usize; 3], weight: f64) -> Self {
        self.entries.push((axis, node, weight));
        self
    }

    /// Samples only every `interval` steps (≥ 1; steps where
    /// `n % interval == 0`).
    pub fn interval(mut self, interval: u64) -> Self {
        self.interval = interval.max(1);
        self
    }

    /// The sampled quantity.
    pub fn quantity(&self) -> ProbeQuantity {
        self.quantity
    }

    /// Sample times \[s\].
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Sample values (V or A), parallel to [`times`](Self::times).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    fn sample(&mut self, f: &super::fields::FieldArrays<'_>, t: f64) {
        // Deterministic: fixed entry order, f64 accumulation.
        let v = self
            .entries
            .iter()
            .map(|&(a, n, w)| w * f.get(a, n) as f64)
            .sum();
        self.times.push(t);
        self.values.push(v);
    }
}

impl Operator for Probe {
    fn post_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if self.quantity == ProbeQuantity::Current && ctx.step.is_multiple_of(self.interval) {
            self.sample(&fields.i, ctx.h_time());
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if self.quantity == ProbeQuantity::Voltage && ctx.step.is_multiple_of(self.interval) {
            self.sample(&fields.u, ctx.e_time());
        }
    }
}
