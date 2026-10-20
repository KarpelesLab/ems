//! Excitation operators.

use super::fields::Fields;
use super::operator::{Hooks, Operator, StepContext};
use crate::axis::Axis;

/// Time function of a source, `t` \[s\] → amplitude \[V\].
pub type Amplitude = Box<dyn Fn(f64) -> f64 + Send>;

/// What an [`EdgeSource`] drives and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SourceKind {
    /// Additive ("soft") source on E edges (default): after the E update
    /// of step `n` (`u^n → u^{n+1}`), `u += w·a(t)` with `t = n·dt`, the
    /// start of that update. This is openEMS's timing (design §5.7):
    /// with the same time step, ems and openEMS probe signals agree to
    /// 1e-6, against 7e-2 for `t = (n + ½)·dt`.
    #[default]
    SoftE,
    /// Hard source on E edges: after the E update of step `n`,
    /// `u = w·a(t)` with `t = (n + 1)·dt`, the time of the updated `u`.
    HardE,
    /// Additive source on H dual edges: after the H update of step `n`
    /// (`i^{n−½} → i^{n+½}`), `i += w·a(t)` with `t = (n − ½)·dt`, the
    /// start of that update (openEMS's timing, like [`SoftE`](Self::SoftE)).
    SoftH,
    /// Hard source on H dual edges: after the H update of step `n`,
    /// `i = w·a(t)` with `t = (n + ½)·dt`, the time of the updated `i`.
    HardH,
}

impl SourceKind {
    /// Whether the source acts on the H currents.
    pub fn is_magnetic(self) -> bool {
        matches!(self, SourceKind::SoftH | SourceKind::HardH)
    }

    /// Whether the source overwrites the field instead of adding to it.
    pub fn is_hard(self) -> bool {
        matches!(self, SourceKind::HardE | SourceKind::HardH)
    }
}

/// Voltage (or current) source on a set of E edges (or H dual edges).
///
/// By default ([`SourceKind::SoftE`]) it adds `weight · amplitude(t)` to the
/// voltage `u` of each listed edge after every E update ("soft" or additive
/// source, Taflove & Hagness, *Computational Electrodynamics*, 3rd ed.,
/// ch. 5 on source conditions). The amplitude is sampled at `t = n·dt`,
/// the start of the E update it is applied with, as openEMS does
/// (design §5.7). See [`SourceKind`] for hard sources and H sources and
/// their sample times.
///
/// A soft source on a PEC edge (`ca = cb = 0`) behaves like a hard source:
/// the kernel clears the edge and the source writes the amplitude.
pub struct EdgeSource {
    kind: SourceKind,
    edges: Vec<(Axis, [usize; 3], f32)>,
    amplitude: Amplitude,
}

impl core::fmt::Debug for EdgeSource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EdgeSource")
            .field("kind", &self.kind)
            .field("edges", &self.edges)
            .finish_non_exhaustive()
    }
}

impl EdgeSource {
    /// Soft E source with time function `amplitude` and no edges yet.
    pub fn new(amplitude: impl Fn(f64) -> f64 + Send + 'static) -> Self {
        Self {
            kind: SourceKind::SoftE,
            edges: Vec::new(),
            amplitude: Box::new(amplitude),
        }
    }

    /// Sets the kind of source (default [`SourceKind::SoftE`]).
    pub fn kind(mut self, kind: SourceKind) -> Self {
        self.kind = kind;
        self
    }

    /// The kind of source.
    pub fn get_kind(&self) -> SourceKind {
        self.kind
    }

    /// Adds the primary edge (or, for H kinds, the dual edge) along `axis`
    /// at `node`, with `weight` (use a negative weight to excite against
    /// the axis direction).
    pub fn edge(mut self, axis: Axis, node: [usize; 3], weight: f64) -> Self {
        self.edges.push((axis, node, weight as f32));
        self
    }

    /// The excited edges `(axis, node, weight)`.
    pub fn edges(&self) -> impl Iterator<Item = (Axis, [usize; 3], f64)> + '_ {
        self.edges.iter().map(|&(a, n, w)| (a, n, w as f64))
    }

    fn apply(&self, arr: &mut super::fields::FieldArrays<'_>, t: f64) {
        let a = (self.amplitude)(t) as f32;
        if self.kind.is_hard() {
            for &(axis, node, w) in &self.edges {
                arr.set(axis, node, w * a);
            }
        } else {
            for &(axis, node, w) in &self.edges {
                *arr.get_mut(axis, node) += w * a;
            }
        }
    }
}

impl Operator for EdgeSource {
    fn hooks(&self) -> Hooks {
        if self.kind.is_magnetic() {
            Hooks::POST_H
        } else {
            Hooks::POST_E
        }
    }

    fn post_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        let t = match self.kind {
            SourceKind::SoftH => ctx.time() - 0.5 * ctx.dt,
            SourceKind::HardH => ctx.h_time(),
            _ => return,
        };
        self.apply(&mut fields.i, t);
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        let t = match self.kind {
            SourceKind::SoftE => ctx.time(),
            SourceKind::HardE => ctx.e_time(),
            _ => return,
        };
        self.apply(&mut fields.u, t);
    }
}
