//! Operator hooks: sparse work run between the half-steps.

use core::any::Any;
use core::marker::PhantomData;
use std::sync::Arc;

use super::coeffs::Coefficients;
use super::fields::Fields;
use super::grid::Grid;
use super::slab::{SlabHooks, SlabPartition};
use crate::axis::Axis;

/// A set of serial operator hooks, returned by [`Operator::hooks`] to tell
/// the engine which ones an operator implements.
///
/// Combine with `|`. The engine skips undeclared hooks and the phases (and
/// barriers) in which no registered operator has work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hooks(u8);

impl Hooks {
    /// No serial hook.
    pub const NONE: Self = Self(0);
    /// [`Operator::pre_h`].
    pub const PRE_H: Self = Self(1);
    /// [`Operator::post_h`].
    pub const POST_H: Self = Self(2);
    /// [`Operator::pre_e`].
    pub const PRE_E: Self = Self(4);
    /// [`Operator::post_e`].
    pub const POST_E: Self = Self(8);
    /// All four hooks (the default).
    pub const ALL: Self = Self(15);

    /// Whether every hook of `other` is in `self`.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether any hook of `other` is in `self`.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for Hooks {
    type Output = Self;
    fn bitor(self, o: Self) -> Self {
        Self(self.0 | o.0)
    }
}

impl core::ops::BitOrAssign for Hooks {
    fn bitor_assign(&mut self, o: Self) {
        self.0 |= o.0;
    }
}

impl core::ops::BitAnd for Hooks {
    type Output = Self;
    fn bitand(self, o: Self) -> Self {
        Self(self.0 & o.0)
    }
}

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
/// Optional extensions (all defaulted, so a simple operator implements only
/// the hooks it needs):
///
/// * [`hooks`](Self::hooks) declares which of the four hooks are used, so
///   the engine can skip empty phases and their barriers;
/// * [`slab_hooks`](Self::slab_hooks) provides bulk work run in parallel by
///   the worker pool inside the half-steps ([`SlabHooks`]): after the
///   kernel of each slab (2. and 5. above) and before any serial `post_*`
///   hook, whatever the registration order;
/// * [`energy`](Self::energy) reports energy stored in the operator's own
///   state (e.g. lumped inductors and capacitors), added to
///   [`Engine::energy`](super::Engine::energy).
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

    /// The serial hooks this operator implements (default
    /// [`Hooks::ALL`]). Queried once, at registration; undeclared hooks
    /// are never called. Declaring fewer hooks only saves time: an
    /// operator whose hooks are all empty may return [`Hooks::NONE`].
    fn hooks(&self) -> Hooks {
        Hooks::ALL
    }

    /// The hooks this operator needs at step `ctx.step` (default
    /// [`hooks`](Self::hooks)): a subset of [`hooks`](Self::hooks), asked
    /// before every run of steps. The engine stops its workers for a
    /// serial phase only at steps where some operator needs it, so an
    /// operator that acts only on some steps (a probe sampling every `K`
    /// steps, a soft source whose signal is zero) saves the barriers of
    /// the others. A declared hook may still be called at a step where
    /// it was not asked for (another operator needed that phase), so
    /// hooks must stay correct whenever they are called.
    fn hooks_at(&self, ctx: &StepContext<'_>) -> Hooks {
        let _ = ctx;
        self.hooks()
    }

    /// The hooks among [`POST_H`](Hooks::POST_H) and
    /// [`PRE_E`](Hooks::PRE_E) that only *read* the fields and that the
    /// engine may therefore move out of the gap between the H and E
    /// updates (default [`Hooks::NONE`]). Queried once, at registration.
    ///
    /// * A movable `post_h` reads `i` only (never `u`) and writes no
    ///   field. The engine may call it after the E update and its slab
    ///   hooks, which leave `i` unchanged, and before any `post_e` hook.
    /// * A movable `pre_e` reads `u` only (never `i`) and writes no field.
    ///   The engine may call it before the H update, after the `pre_h`
    ///   hooks: the H update and its slab hooks leave `u` unchanged.
    /// * The two must not depend on each other through the operator's own
    ///   state (the engine may call `pre_e` before `post_h`).
    ///
    /// At a step where every operator that needs `post_h` or `pre_e`
    /// ([`hooks_at`](Self::hooks_at)) declares those hooks movable, the
    /// engine moves them (calling those of every operator that declares
    /// them movable, in registration order) and runs H and E without a
    /// serial phase between them, fused when possible (see
    /// [`SlabHooks::fusion`]). Probes of currents and lumped elements
    /// are such operators.
    fn movable_hooks(&self) -> Hooks {
        Hooks::NONE
    }

    /// Slab-parallel work for the engine's x-slab `partition` (default
    /// none). Called once, at registration; the returned object is run by
    /// the worker threads (see [`SlabHooks`]). The operator may keep a
    /// clone of the `Arc` to read results back.
    fn slab_hooks(&mut self, _partition: &SlabPartition<'_>) -> Option<Arc<dyn SlabHooks>> {
        None
    }

    /// Energy \[J\] held in the operator's own state at `t = n·dt`, the
    /// time of the E voltages after `n` completed steps (default 0).
    ///
    /// [`Engine::energy`](super::Engine::energy) and
    /// [`Engine::step_with_energy`](super::Engine::step_with_energy) add
    /// the values of all operators, in registration order, to the field
    /// invariant. Implement it so that the sum is the conserved quantity of
    /// the coupled scheme (see [`LumpedElement`](super::LumpedElement)).
    fn energy(&self) -> f64 {
        0.0
    }

    /// Whether [`lookahead_current`](Self::lookahead_current) may return
    /// a value (default `false`; queried once per look-ahead).
    fn has_lookahead_currents(&self) -> bool {
        false
    }

    /// The current `i^{N+½}` of the H dual edge along `axis` at `node`,
    /// for an operator that completes the H update of that edge itself
    /// (in a hook), from the kernel's value `kernel` of the H update
    /// evaluated locally on the current state (`da·i − db·∮u`); `None`
    /// for the edges it does not own (default). Used by the look-ahead H
    /// samples of probes and dumps after the last step
    /// ([`Probe::lookahead`](super::Probe::lookahead)).
    fn lookahead_current(&self, axis: Axis, node: [usize; 3], kernel: f32) -> Option<f32> {
        let _ = (axis, node, kernel);
        None
    }
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
