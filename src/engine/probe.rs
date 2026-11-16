//! Probe operators.

use super::dump::{DumpPosition, DumpQuantity, DumpRegion, FieldSampler};
use super::fields::Fields;
use super::grid::Grid;
use super::operator::{Hooks, Operator, StepContext};
use crate::axis::Axis;
use crate::{Error, Result};

/// Which field a [`Probe`] samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProbeQuantity {
    /// E edge voltages `u`, sampled after the E update at `(n + 1)·dt`.
    Voltage,
    /// H dual-edge currents `i`, sampled after the H update at
    /// `(n + ½)·dt`.
    Current,
}

/// One sample point of a mode-purity measurement ([`Probe::mode_purity`]):
/// the two tangential components of the field at the point, as weighted
/// sums of edge values, the mode function there and the area the point
/// stands for.
#[derive(Debug, Clone, PartialEq)]
pub struct ModeSample {
    /// Area weight `A` \[m²\].
    pub area: f64,
    /// The two tangential components of the mode function at the point
    /// (any scale).
    pub mode: [f64; 2],
    /// The two tangential field components at the point (V/m or A/m):
    /// `Σ c·x` over the entries `(axis, node, c)` of the sampled quantity.
    pub field: [Vec<(Axis, [usize; 3], f64)>; 2],
}

/// The sample times of a recording: the rule they follow, where they
/// follow one. A probe samples every `interval` steps at `(n + ½)·dt` or
/// `(n + 1)·dt`, so its times are `(first + k·interval + offset)·dt`: the
/// values [`StepContext::h_time`] and [`StepContext::e_time`] give, by the
/// same expression. They are held as that rule, and as a list only once
/// asked for as one ([`as_slice`](Self::as_slice)) or if a sample does
/// not follow the rule: a list of them is as long as the samples, for
/// every probe of a run.
#[derive(Debug, Clone, Default)]
pub(crate) struct SampleTimes {
    rule: Rule,
    /// Number of samples under the rule.
    n: usize,
    /// The times as a list, made on demand.
    list: std::sync::OnceLock<Vec<f64>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
enum Rule {
    /// No sample yet, or the times are in the list alone.
    #[default]
    None,
    /// `(first + k·interval + offset)·dt` (`interval` 0 until the second
    /// sample).
    Steps {
        first: u64,
        interval: u64,
        offset: f64,
        dt: f64,
    },
    /// `t0 + k·dt`.
    Linear { t0: f64, dt: f64 },
}

impl SampleTimes {
    /// The times `t0 + k·dt` of `n` samples.
    pub(crate) fn linear(t0: f64, dt: f64, n: usize) -> Self {
        Self {
            rule: Rule::Linear { t0, dt },
            n,
            list: std::sync::OnceLock::new(),
        }
    }

    /// The times `list`.
    pub(crate) fn listed(list: Vec<f64>) -> Self {
        Self {
            rule: Rule::None,
            n: 0,
            list: list.into(),
        }
    }

    /// Number of samples.
    pub(crate) fn len(&self) -> usize {
        match self.rule {
            Rule::None => self.list.get().map_or(0, Vec::len),
            _ => self.n,
        }
    }

    /// The time of sample `k` (`k < len`).
    pub(crate) fn get(&self, k: usize) -> f64 {
        match self.rule {
            Rule::None => self.list.get().expect("listed times")[k],
            Rule::Steps {
                first,
                interval,
                offset,
                dt,
            } => ((first + k as u64 * interval) as f64 + offset) * dt,
            Rule::Linear { t0, dt } => t0 + k as f64 * dt,
        }
    }

    /// The first time, if any.
    pub(crate) fn first(&self) -> Option<f64> {
        (self.len() > 0).then(|| self.get(0))
    }

    /// The times as a list (made at the first call and kept).
    pub(crate) fn as_slice(&self) -> &[f64] {
        self.list
            .get_or_init(|| (0..self.n).map(|k| self.get(k)).collect())
    }

    /// The number of leading times for which `pred` holds (the times
    /// ascend: as `partition_point` of the list).
    pub(crate) fn partition_point(&self, pred: impl Fn(f64) -> bool) -> usize {
        let (mut lo, mut hi) = (0, self.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(self.get(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// The times of the samples `range`.
    pub(crate) fn window(&self, range: core::ops::Range<usize>) -> Self {
        let n = range.len();
        let rule = match self.rule {
            Rule::None => {
                return Self {
                    rule: Rule::None,
                    n: 0,
                    list: self.as_slice()[range].to_vec().into(),
                };
            }
            Rule::Steps {
                first,
                interval,
                offset,
                dt,
            } => Rule::Steps {
                first: first + range.start as u64 * interval,
                interval,
                offset,
                dt,
            },
            // (Not `t0 + start·dt` as a new start: the same expression
            // for every time, whatever the window.)
            Rule::Linear { .. } if range.start != 0 => {
                return Self {
                    rule: Rule::None,
                    n: 0,
                    list: self.as_slice()[range].to_vec().into(),
                };
            }
            r @ Rule::Linear { .. } => r,
        };
        Self {
            rule,
            n,
            list: std::sync::OnceLock::new(),
        }
    }

    /// Appends the time `(step + offset)·dt`.
    fn push(&mut self, step: u64, offset: f64, dt: f64) {
        let same = |o: f64, d: f64| o.to_bits() == offset.to_bits() && d.to_bits() == dt.to_bits();
        let follows = match &mut self.rule {
            Rule::None => self.list.get().is_none_or(Vec::is_empty),
            Rule::Steps {
                first,
                interval,
                offset: o,
                dt: d,
            } => {
                if self.n == 1 && step > *first && same(*o, *d) {
                    *interval = step - *first;
                    true
                } else {
                    same(*o, *d) && *interval > 0 && step == *first + self.n as u64 * *interval
                }
            }
            Rule::Linear { .. } => false,
        };
        if follows {
            if self.rule == Rule::None {
                self.rule = Rule::Steps {
                    first: step,
                    interval: 0,
                    offset,
                    dt,
                };
            }
            self.n += 1;
            // (A list made earlier is one sample short now.)
            if self.list.get().is_some() {
                self.list = std::sync::OnceLock::new();
            }
        } else {
            // Not the rule's next time: the list from here on.
            let mut list = self.as_slice().to_vec();
            list.push((step as f64 + offset) * dt);
            *self = Self {
                rule: Rule::None,
                n: 0,
                list: list.into(),
            };
        }
    }
}

impl PartialEq for SampleTimes {
    /// The same times.
    fn eq(&self, o: &Self) -> bool {
        self.len() == o.len() && (0..self.len()).all(|k| self.get(k) == o.get(k))
    }
}

/// Records the weighted sum `Σ w·x` of a set of field entries every
/// `interval` steps, with the exact sample times.
///
/// With unit weights along a chain of E edges this is the line integral of
/// E (a voltage); along a loop of H dual edges, the enclosed current.
///
/// With [`mode_purity`](Self::mode_purity) it also records, at the same
/// samples, how much of the sampled transverse field is the mode.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Probe {
    quantity: ProbeQuantity,
    entries: Vec<(Axis, [usize; 3], f64)>,
    interval: u64,
    times: SampleTimes,
    values: Vec<f64>,
    purity: Option<(Vec<ModeSample>, f64)>,
    purity_values: Vec<f64>,
    /// Sum in `f32` ([`f32_sum`](Self::f32_sum)).
    f32_sum: bool,
    /// Storage locations of `entries` and of the purity sample terms,
    /// from the first sample (the engine's layout is fixed).
    locs: Option<ProbeLocs>,
}

/// Field storage locations of a probe's entries and purity terms.
#[derive(Debug, Clone)]
struct ProbeLocs {
    /// The layout they were computed for.
    n: [usize; 3],
    starts: Vec<usize>,
    entries: Vec<(usize, usize)>,
    terms: Vec<[Vec<(usize, usize)>; 2]>,
}

impl Probe {
    /// Empty probe of `quantity`, sampling every step.
    pub fn new(quantity: ProbeQuantity) -> Self {
        Self {
            quantity,
            entries: Vec::new(),
            interval: 1,
            times: SampleTimes::default(),
            values: Vec::new(),
            purity: None,
            purity_values: Vec::new(),
            f32_sum: false,
            locs: None,
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

    /// Accumulates the weighted sum in `f32`, term by term in entry
    /// order (`s ← s + f32(w)·x`, rounded each time), instead of in
    /// `f64`: openEMS's current probes (black box, design §5.16.9).
    pub(crate) fn f32_sum(mut self, on: bool) -> Self {
        self.f32_sum = on;
        self
    }

    /// Samples only every `interval` steps (≥ 1; steps where
    /// `n % interval == 0`).
    pub fn interval(mut self, interval: u64) -> Self {
        self.interval = interval.max(1);
        self
    }

    /// Also records the mode purity at every sample, scaled by `weight`:
    ///
    /// ```text
    /// p = weight · (Σ_s A_s·F_s·f_s)² / (Σ_s A_s·|F_s|² · Σ_s A_s·|f_s|²)
    /// ```
    ///
    /// over the `samples` `s` (field `F_s`, mode function `f_s`, area
    /// `A_s`; two tangential components each): the squared normalised
    /// overlap of the sampled field with the mode, 1 for a field of the
    /// mode's shape and 0 for one orthogonal to it (or no field) — by
    /// the Cauchy–Schwarz inequality it lies in `[0, 1]` before scaling.
    /// This is the `mode_purity` column of openEMS's waveguide probe files
    /// (see [`compile`](crate::compile#waveguide-probes)), which is scaled
    /// by the probe weight like the value.
    pub fn mode_purity(mut self, samples: Vec<ModeSample>, weight: f64) -> Self {
        self.purity = Some((samples, weight));
        self
    }

    /// The recorded mode purity, parallel to [`times`](Self::times)
    /// (`None` without [`mode_purity`](Self::mode_purity)).
    pub fn purity(&self) -> Option<&[f64]> {
        self.purity.as_ref().map(|_| self.purity_values.as_slice())
    }

    /// The sampled quantity.
    pub fn quantity(&self) -> ProbeQuantity {
        self.quantity
    }

    /// Sample times \[s\].
    pub fn times(&self) -> &[f64] {
        self.times.as_slice()
    }

    /// The recorded values, which the probe gives up (its times stay).
    pub(crate) fn take_values(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.values)
    }

    /// The sample times as they are held (a rule, not a list).
    pub(crate) fn sample_times(&self) -> &SampleTimes {
        &self.times
    }

    /// Sample values (V or A), parallel to [`times`](Self::times).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Records a sample at `(step + offset)·dt`.
    fn sample(&mut self, f: &super::fields::FieldArrays<'_>, step: u64, offset: f64, dt: f64) {
        let lay = f.layout();
        // Rebuilt for another engine's layout (a probe cloned after use).
        if self
            .locs
            .as_ref()
            .is_none_or(|l| l.n != lay.n || l.starts != lay.slab_start)
        {
            self.locs = Some(ProbeLocs {
                n: lay.n,
                starts: lay.slab_start.clone(),
                entries: self
                    .entries
                    .iter()
                    .map(|&(_, n, _)| lay.locate(n))
                    .collect(),
                terms: self.purity.as_ref().map_or_else(Vec::new, |(samples, _)| {
                    samples
                        .iter()
                        .map(|s| {
                            s.field
                                .each_ref()
                                .map(|t| t.iter().map(|&(_, n, _)| lay.locate(n)).collect())
                        })
                        .collect()
                }),
            });
        }
        let locs = self.locs.as_ref().expect("built above");
        // Deterministic: fixed entry order, f64 (or f32) accumulation.
        let terms = self.entries.iter().zip(&locs.entries);
        let v = if self.f32_sum {
            f64::from(
                terms
                    .map(|(&(a, _, w), &l)| w as f32 * f.at(a, l))
                    .fold(0.0f32, |s, x| s + x),
            )
        } else {
            terms.map(|(&(a, _, w), &l)| w * f.at(a, l) as f64).sum()
        };
        let p = self.purity_core(&|s, c, k| {
            let (a, _, _) = self.purity.as_ref().expect("purity samples").0[s].field[c][k];
            f64::from(f.at(a, locs.terms[s][c][k]))
        });
        self.times.push(step, offset, dt);
        self.values.push(v);
        if let Some(p) = p {
            self.purity_values.push(p);
        }
    }

    /// The weighted mode purity of the field values `get` (`None` without
    /// [`mode_purity`](Self::mode_purity)).
    fn purity_with(&self, get: &dyn Fn(Axis, [usize; 3]) -> f64) -> Option<f64> {
        let samples = &self.purity.as_ref()?.0;
        self.purity_core(&|s, c, k| {
            let (a, n, _) = samples[s].field[c][k];
            get(a, n)
        })
    }

    /// [`purity_with`](Self::purity_with) with term `k` of component `c`
    /// of sample `s` read by `value(s, c, k)`.
    fn purity_core(&self, value: &dyn Fn(usize, usize, usize) -> f64) -> Option<f64> {
        let (samples, weight) = self.purity.as_ref()?;
        let (mut dot, mut ff, mut mm) = (0.0f64, 0.0f64, 0.0f64);
        for (si, s) in samples.iter().enumerate() {
            let mut e = [0.0f64; 2];
            for (ci, (c, terms)) in e.iter_mut().zip(&s.field).enumerate() {
                *c = terms
                    .iter()
                    .enumerate()
                    .map(|(k, &(_, _, w))| w * value(si, ci, k))
                    .sum();
            }
            dot += s.area * (e[0] * s.mode[0] + e[1] * s.mode[1]);
            ff += s.area * (e[0] * e[0] + e[1] * e[1]);
            mm += s.area * (s.mode[0] * s.mode[0] + s.mode[1] * s.mode[1]);
        }
        let p = if ff > 0.0 && mm > 0.0 {
            dot * dot / (ff * mm)
        } else {
            0.0
        };
        Some(weight * p)
    }

    /// The sample a current probe would record at `(N + ½)·dt` for an
    /// engine that has completed `N` steps, from the H update of step `N`
    /// evaluated locally ([`FieldSampler::sample_lookahead`]'s rules:
    /// without CPML and H-source corrections, `None` where unavailable),
    /// without running it: `(time, value, mode purity)`. openEMS records
    /// this state at the end of a run. `None` for voltage probes and when
    /// step `N` is not a sampling step.
    pub fn lookahead(&self, engine: &super::Engine) -> Option<(f64, f64, Option<f64>)> {
        let n = engine.timestep();
        if self.quantity != ProbeQuantity::Current || !n.is_multiple_of(self.interval) {
            return None;
        }
        super::dump::with_lookahead_currents(engine, |get| {
            let terms = self.entries.iter();
            let v = if self.f32_sum {
                f64::from(
                    terms
                        .map(|&(a, node, w)| w as f32 * get(a, node) as f32)
                        .fold(0.0f32, |s, x| s + x),
                )
            } else {
                terms.map(|&(a, node, w)| w * get(a, node)).sum()
            };
            ((n as f64 + 0.5) * engine.dt(), v, self.purity_with(get))
        })
    }
}

impl Operator for Probe {
    fn hooks(&self) -> Hooks {
        match self.quantity {
            ProbeQuantity::Current => Hooks::POST_H,
            ProbeQuantity::Voltage => Hooks::POST_E,
        }
    }

    /// Current probes only read the currents.
    fn movable_hooks(&self) -> Hooks {
        Hooks::POST_H
    }

    /// Only at the sampling steps.
    fn hooks_at(&self, ctx: &StepContext<'_>) -> Hooks {
        if ctx.step.is_multiple_of(self.interval) {
            self.hooks()
        } else {
            Hooks::NONE
        }
    }

    fn post_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if self.quantity == ProbeQuantity::Current && ctx.step.is_multiple_of(self.interval) {
            self.sample(&fields.i, ctx.step, 0.5, ctx.dt);
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if self.quantity == ProbeQuantity::Voltage && ctx.step.is_multiple_of(self.interval) {
            self.sample(&fields.u, ctx.step, 1.0, ctx.dt);
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
/// several points the recorded vector is their mean ([`new`](Self::new))
/// or a weighted sum ([`at`](Self::at)).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FieldProbe {
    sampler: FieldSampler,
    /// Weight per sample point (empty: mean).
    weights: Vec<f64>,
    times: Vec<f64>,
    values: Vec<[f64; 3]>,
}

impl FieldProbe {
    /// Probe of `sampler`'s quantity (the mean over its points).
    pub fn new(sampler: FieldSampler) -> Self {
        Self {
            sampler,
            weights: Vec::new(),
            times: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Probe of `quantity` at the arbitrary point `p` \[m\]: the
    /// node-interpolated field ([`DumpPosition::Nodes`]) of the up to
    /// 2 × 2 × 2 nodes around `p`, combined trilinearly with the
    /// coordinates of `p` inside its cell. On a mesh line (within `1e-9`
    /// of the local cell size) only that line's nodes are used, so a
    /// point on a node is exactly that node's value.
    ///
    /// # Errors
    /// [`Error::Invalid`] if `p` is not finite or lies outside the grid.
    pub fn at(grid: &Grid, quantity: DumpQuantity, p: [f64; 3]) -> Result<Self> {
        let mut lo = [0; 3];
        let mut hi = [0; 3];
        let mut w: [Vec<f64>; 3] = Default::default();
        for a in Axis::ALL {
            let d = a.index();
            let l = grid.lines(a);
            let x = p[d];
            let (first, last) = (l[0], l[l.len() - 1]);
            let tol = |j: usize| {
                1e-9 * if l.len() > 1 {
                    grid.delta(a, j.min(l.len() - 2))
                } else {
                    1.0
                }
            };
            if !(x.is_finite() && x >= first - tol(0) && x <= last + tol(l.len() - 1)) {
                return Err(Error::Invalid(format!(
                    "field probe point {p:?} is outside the grid along {a}"
                )));
            }
            // Cell j = [l[j], l[j+1]] containing x.
            let j = l.partition_point(|&v| v <= x).saturating_sub(1);
            if (x - l[j]).abs() <= tol(j) || j + 1 == l.len() {
                (lo[d], hi[d], w[d]) = (j, j, vec![1.0]);
            } else if (l[j + 1] - x).abs() <= tol(j) {
                (lo[d], hi[d], w[d]) = (j + 1, j + 1, vec![1.0]);
            } else {
                let t = (x - l[j]) / (l[j + 1] - l[j]);
                (lo[d], hi[d], w[d]) = (j, j + 1, vec![1.0 - t, t]);
            }
        }
        let sampler =
            FieldSampler::new(grid, DumpRegion::new(lo, hi), quantity, DumpPosition::Nodes)?;
        let mut weights = Vec::with_capacity(sampler.len());
        for wx in &w[0] {
            for wy in &w[1] {
                for wz in &w[2] {
                    weights.push(wx * wy * wz);
                }
            }
        }
        debug_assert_eq!(weights.len(), sampler.len());
        Ok(Self {
            sampler,
            weights,
            times: Vec::new(),
            values: Vec::new(),
        })
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
        let weights = &self.weights;
        // Deterministic: fixed point order, f64 accumulation.
        self.sampler.sample_with(fields, |i, v| {
            let w = weights.get(i).copied().unwrap_or(1.0);
            for (a, x) in acc.iter_mut().zip(v) {
                *a += w * x;
            }
        });
        let n = if self.weights.is_empty() {
            self.sampler.len().max(1) as f64
        } else {
            1.0
        };
        self.times.push(t);
        self.values.push(acc.map(|a| a / n));
    }
}

impl Operator for FieldProbe {
    fn hooks(&self) -> Hooks {
        if self.sampler.quantity().from_currents() {
            Hooks::POST_H
        } else {
            Hooks::POST_E
        }
    }

    /// Probes of H (and the other quantities of the currents) only read
    /// the currents.
    fn movable_hooks(&self) -> Hooks {
        if self.sampler.quantity().from_currents() {
            Hooks::POST_H
        } else {
            Hooks::NONE
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The times of regular samples are held as their rule and read as
    /// the list they were; a sample off the rule turns them into a list.
    #[test]
    fn sample_times_follow_the_steps() {
        let dt = 1.37e-12;
        let mut t = SampleTimes::default();
        assert_eq!((t.len(), t.first()), (0, None));
        let mut want = Vec::new();
        for step in (6..60).step_by(3) {
            t.push(step, 0.5, dt);
            want.push((step as f64 + 0.5) * dt);
            assert_eq!(t.as_slice(), &want[..]);
        }
        assert!(matches!(t.rule, Rule::Steps { interval: 3, .. }));
        assert_eq!(t.first(), Some(want[0]));
        assert_eq!(t.partition_point(|x| x < want[7]), 7);
        let w = t.window(4..11);
        assert_eq!(w.as_slice(), &want[4..11]);
        assert!(w == SampleTimes::listed(want[4..11].to_vec()));
        // Off the rule: a step left out.
        t.push(100, 0.5, dt);
        want.push(100.5 * dt);
        assert_eq!(t.rule, Rule::None);
        assert_eq!(t.as_slice(), &want[..]);
        t.push(103, 0.5, dt);
        want.push(103.5 * dt);
        assert_eq!((t.len(), t.as_slice()), (want.len(), &want[..]));
        assert_eq!(t.window(2..5).as_slice(), &want[2..5]);
        // `t0 + k·dt`.
        let l = SampleTimes::linear(0.5 * dt, dt, 4);
        assert_eq!(
            l.as_slice(),
            &[
                0.5 * dt,
                0.5 * dt + dt,
                0.5 * dt + 2.0 * dt,
                0.5 * dt + 3.0 * dt
            ]
        );
        assert_eq!(l.window(1..3).as_slice(), &l.as_slice()[1..3]);
    }
}
