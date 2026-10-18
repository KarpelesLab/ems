//! Probe operators.

use super::dump::{DumpQuantity, FieldSampler};
use super::fields::Fields;
use super::operator::{Hooks, Operator, StepContext};
use crate::axis::Axis;

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
    entries: Vec<(Axis, [usize; 3], f64)>,
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
    pub fn voltage(axis: Axis, node: [usize; 3]) -> Self {
        Self::new(ProbeQuantity::Voltage).entry(axis, node, 1.0)
    }

    /// Probe of the single H dual edge along `axis` at `node`.
    pub fn current(axis: Axis, node: [usize; 3]) -> Self {
        Self::new(ProbeQuantity::Current).entry(axis, node, 1.0)
    }

    /// Adds component `axis` at `node` with `weight`.
    pub fn entry(mut self, axis: Axis, node: [usize; 3], weight: f64) -> Self {
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
    fn hooks(&self) -> Hooks {
        match self.quantity {
            ProbeQuantity::Current => Hooks::POST_H,
            ProbeQuantity::Voltage => Hooks::POST_E,
        }
    }

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

/// Records the interpolated field vector (E \[V/m\] or H \[A/m\]) at the
/// sample points of a [`FieldSampler`] every step: E after the E update at
/// `(n + 1)·dt`, H after the H update at `(n + ½)·dt`.
///
/// Values and interpolation are those of the dumps
/// ([`dump` module docs](super::dump#interpolation)); with a one-point
/// [`DumpRegion`](super::DumpRegion) this is a field point probe. With
/// several points the recorded vector is their mean.
#[derive(Debug, Clone)]
pub struct FieldProbe {
    sampler: FieldSampler,
    times: Vec<f64>,
    values: Vec<[f64; 3]>,
}

impl FieldProbe {
    /// Probe of `sampler`'s quantity.
    pub fn new(sampler: FieldSampler) -> Self {
        Self {
            sampler,
            times: Vec::new(),
            values: Vec::new(),
        }
    }

    /// The sampler.
    pub fn sampler(&self) -> &FieldSampler {
        &self.sampler
    }

    /// Sample times \[s\].
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Sampled vectors, parallel to [`times`](Self::times).
    pub fn values(&self) -> &[[f64; 3]] {
        &self.values
    }

    fn sample(&mut self, fields: &Fields<'_>, t: f64) {
        let mut acc = [0.0f64; 3];
        // Deterministic: fixed point order, f64 accumulation.
        self.sampler.sample_with(fields, |_, v| {
            for (a, x) in acc.iter_mut().zip(v) {
                *a += x;
            }
        });
        let n = self.sampler.len().max(1) as f64;
        self.times.push(t);
        self.values.push(acc.map(|a| a / n));
    }
}

impl Operator for FieldProbe {
    fn hooks(&self) -> Hooks {
        match self.sampler.quantity() {
            DumpQuantity::E => Hooks::POST_E,
            DumpQuantity::H => Hooks::POST_H,
        }
    }

    fn post_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if self.sampler.quantity() == DumpQuantity::H {
            self.sample(fields, ctx.h_time());
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if self.sampler.quantity() == DumpQuantity::E {
            self.sample(fields, ctx.e_time());
        }
    }
}
