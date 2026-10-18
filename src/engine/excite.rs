//! Excitation operators.

use super::fields::Fields;
use super::operator::{Operator, StepContext};

/// Time function of a source, `t` \[s\] → amplitude \[V\].
pub type Amplitude = Box<dyn Fn(f64) -> f64 + Send>;

/// Soft voltage source on a set of E edges.
///
/// After every E update it adds `weight · amplitude(t)` to the voltage `u`
/// of each listed edge ("soft" or additive source, Taflove & Hagness,
/// *Computational Electrodynamics*, 3rd ed., ch. 5 on source conditions).
/// The amplitude is sampled at `t = (n + ½)·dt`, the centre of the E update
/// it is applied with (design §5.7).
///
/// A soft source on a PEC edge (`ca = cb = 0`) behaves like a hard source:
/// the kernel clears the edge and the source writes the amplitude.
pub struct EdgeSource {
    edges: Vec<(usize, [usize; 3], f32)>,
    amplitude: Amplitude,
}

impl core::fmt::Debug for EdgeSource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EdgeSource")
            .field("edges", &self.edges)
            .finish_non_exhaustive()
    }
}

impl EdgeSource {
    /// Source with time function `amplitude` and no edges yet.
    pub fn new(amplitude: impl Fn(f64) -> f64 + Send + 'static) -> Self {
        Self {
            edges: Vec::new(),
            amplitude: Box::new(amplitude),
        }
    }

    /// Adds the primary edge along `axis` starting at `node`, with `weight`
    /// (use a negative weight to excite against the axis direction).
    pub fn edge(mut self, axis: usize, node: [usize; 3], weight: f64) -> Self {
        self.edges.push((axis, node, weight as f32));
        self
    }

    /// The excited edges `(axis, node, weight)`.
    pub fn edges(&self) -> impl Iterator<Item = (usize, [usize; 3], f64)> + '_ {
        self.edges.iter().map(|&(a, n, w)| (a, n, w as f64))
    }
}

impl Operator for EdgeSource {
    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        let a = (self.amplitude)(ctx.h_time()) as f32;
        for &(axis, node, w) in &self.edges {
            *fields.u.get_mut(axis, node) += w * a;
        }
    }
}
