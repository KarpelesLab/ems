//! Operator hooks: sparse work run between the half-steps.

use core::any::Any;
use core::marker::PhantomData;

use super::coeffs::Coefficients;
use super::fields::Fields;
use super::grid::Grid;

/// Information about the step being performed, passed to every hook.
///
/// Step `n` (= [`step`](Self::step)) advances the H currents from
/// `t = (n − ½)·dt` to `(n + ½)·dt` and then the E voltages from `n·dt` to
/// `(n + 1)·dt`.
#[derive(Debug, Clone, Copy)]
pub struct StepContext<'a> {
    /// Index `n` of the step in progress (0 for the first step).
    pub step: u64,
    /// Time step \[s\].
    pub dt: f64,
    /// The mesh.
    pub grid: &'a Grid,
    /// The update coefficients (read only while stepping).
    pub coefficients: &'a Coefficients,
}

impl StepContext<'_> {
    /// `n·dt`: time of the E voltages before this step (also the time at
    /// which the H update samples them).
    pub fn time(&self) -> f64 {
        self.step as f64 * self.dt
    }

    /// `(n + ½)·dt`: time of the H currents after this step's H update. E
    /// sources are sampled here (design §5.7), as the E update is centred on
    /// this instant.
    pub fn h_time(&self) -> f64 {
        (self.step as f64 + 0.5) * self.dt
    }

    /// `(n + 1)·dt`: time of the E voltages after this step's E update.
    pub fn e_time(&self) -> f64 {
        (self.step as f64 + 1.0) * self.dt
    }
}

/// A sparse engine extension: boundaries (Mur, CPML), sources, lumped
/// elements, probes, dumps.
///
/// Each step runs, in this order:
///
/// 1. `pre_h` of every operator (registration order),
/// 2. the H kernel (all threads),
/// 3. `post_h` of every operator,
/// 4. `pre_e` of every operator,
/// 5. the E kernel (all threads),
/// 6. `post_e` of every operator.
///
/// Hooks run on the thread that called [`Engine::step`](super::Engine::step)
/// / [`run`](super::Engine::run), with exclusive access to all fields; the
/// workers are parked meanwhile. All hooks default to doing nothing.
///
/// Operators are `'static + Send` so the engine can be moved between
/// threads and results can be read back through a typed
/// [`OperatorHandle`].
pub trait Operator: Any + Send {
    /// Before the H update of step `ctx.step`.
    fn pre_h(&mut self, _fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {}
    /// After the H update (`i` is at `ctx.h_time()`).
    fn post_h(&mut self, _fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {}
    /// Before the E update.
    fn pre_e(&mut self, _fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {}
    /// After the E update (`u` is at `ctx.e_time()`).
    fn post_e(&mut self, _fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {}
}

/// Typed handle to an operator registered with
/// [`Engine::add_operator`](super::Engine::add_operator), used to read its
/// state (e.g. probe data) back.
#[derive(Debug)]
pub struct OperatorHandle<T> {
    pub(crate) engine: u64,
    pub(crate) index: usize,
    pub(crate) _t: PhantomData<fn() -> T>,
}

impl<T> Clone for OperatorHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for OperatorHandle<T> {}
