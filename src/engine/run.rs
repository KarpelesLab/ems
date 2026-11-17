//! Run loop with end criteria (design §5.8) and progress reporting.

use core::ops::ControlFlow;
use std::time::{Duration, Instant};

use super::Engine;
use crate::fdtd::{EndCriteria, Settle};
use crate::{Error, Result};

/// When [`Engine::run_until`] stops.
///
/// ```
/// use ems::engine::RunLimits;
/// use ems::{EndCriteria, Waveform};
///
/// let wf = Waveform::gaussian(2e9, 1e9);
/// let limits = RunLimits::new(100_000)
///     .end_criteria(EndCriteria::energy_decay_db(-40.0))
///     .excitation_end(wf.duration());
/// assert_eq!(limits.max_timesteps, 100_000);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RunLimits {
    /// Maximum number of steps this call runs (counted from the engine's
    /// [`timestep`](Engine::timestep) at the start of the call).
    pub max_timesteps: u64,
    /// Energy-decay criterion; `None` runs to `max_timesteps` (unless
    /// aborted).
    pub end_criteria: Option<EndCriteria>,
    /// Time \[s\] after which the excitation is over, typically
    /// [`Waveform::duration`](crate::Waveform::duration). The energy
    /// criterion only applies once the engine time is strictly greater.
    /// `None` means there is no excitation to wait for (the criterion
    /// applies from the first check). For an excitation that never ends,
    /// use `Some(f64::INFINITY)` or no end criteria.
    pub excitation_end: Option<f64>,
    /// Evaluate the energy of the state *after* every `K` steps of the call
    /// (with [`Engine::energy`], one extra read-only pass) and stop right
    /// there, instead of as a by-product of the next step (default
    /// `false`). This is openEMS's schedule: with `K` its Nyquist interval
    /// the run stops after a multiple of `K` steps.
    pub check_after_interval: bool,
    /// With [`check_after_interval`](Self::check_after_interval): one more
    /// evaluation after this many steps of the call, between the regular
    /// ones (default `None`). openEMS checks once more after half its
    /// Gaussian excitation's length, `⌊L/2⌋` steps (black box: runs whose
    /// threshold falls between the checks stop there).
    pub extra_check: Option<u64>,
    /// The energy the criterion compares (default
    /// [`EnergyMeasure::Invariant`]).
    pub energy_measure: EnergyMeasure,
}

/// What the energy end criterion measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum EnergyMeasure {
    /// The leapfrog energy invariant [`Engine::energy`] (fields weighted
    /// by their edge capacitances and inductances, plus operator state),
    /// less the energy inside lossy conductors
    /// ([`Engine::held_energy`]), which no field outside them can draw
    /// on and which decays at the conductor's own rate.
    #[default]
    Invariant,
    /// openEMS's measure [`Engine::energy_openems`]:
    /// `ε0·Σ u² + μ0·Σ i²` over every edge, unweighted, with `u` at `n·dt`
    /// and `i` at `(n + ½)·dt`. On Cartesian grids the dual edges of the
    /// outer planes normal to their own axis count with openEMS's
    /// full-cell length (`4·i²`, ems's being half cells), and the E
    /// on the three upper faces (x, y or z at its last line) is left out
    /// (black box, design §5.16.1). It is evaluated on the state (an extra
    /// read-only H pass) also without
    /// [`check_after_interval`](RunLimits::check_after_interval).
    OpenEms,
}

impl RunLimits {
    /// At most `max_timesteps` steps, no energy criterion.
    pub fn new(max_timesteps: u64) -> Self {
        Self {
            max_timesteps,
            end_criteria: None,
            excitation_end: None,
            check_after_interval: false,
            extra_check: None,
            energy_measure: EnergyMeasure::Invariant,
        }
    }

    /// Builder: sets [`energy_measure`](Self::energy_measure).
    pub fn energy_measure(mut self, m: EnergyMeasure) -> Self {
        self.energy_measure = m;
        self
    }

    /// Builder: sets [`check_after_interval`](Self::check_after_interval).
    pub fn check_after_interval(mut self, on: bool) -> Self {
        self.check_after_interval = on;
        self
    }

    /// Builder: sets [`extra_check`](Self::extra_check).
    pub fn extra_check(mut self, steps: impl Into<Option<u64>>) -> Self {
        self.extra_check = steps.into();
        self
    }

    /// Builder: sets the energy-decay criterion.
    pub fn end_criteria(mut self, c: impl Into<Option<EndCriteria>>) -> Self {
        self.end_criteria = c.into();
        self
    }

    /// Builder: sets the excitation end time \[s\].
    pub fn excitation_end(mut self, t: impl Into<Option<f64>>) -> Self {
        self.excitation_end = t.into();
        self
    }

    /// Steps between energy evaluations: the criterion's
    /// [`check_interval`](EndCriteria::check_interval), or
    /// [`EndCriteria::DEFAULT_CHECK_INTERVAL`] without one.
    pub fn check_interval(&self) -> u64 {
        self.end_criteria
            .map_or(EndCriteria::DEFAULT_CHECK_INTERVAL, |c| c.check_interval)
    }

    /// Validates the end criteria and the excitation end time (not NaN).
    pub fn validate(&self) -> Result<()> {
        if let Some(c) = &self.end_criteria {
            c.validate()?;
        }
        if self.excitation_end.is_some_and(f64::is_nan) {
            return Err(Error::Invalid("excitation end time is NaN".into()));
        }
        Ok(())
    }
}

/// Why [`Engine::run_until`] stopped.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum StopReason {
    /// The excitation was over and `W/W_max` fell below the threshold;
    /// `ratio_db = 10·log10(W/W_max)` at the deciding check
    /// (`-∞` if `W ≤ 0`).
    EnergyDecay {
        /// Energy ratio at the deciding check \[dB\].
        ratio_db: f64,
    },
    /// `max_timesteps` steps were run.
    MaxTimesteps,
    /// The progress callback returned [`ControlFlow::Break`].
    Aborted,
    /// The energy became NaN, infinite or negative (unstable setup, e.g.
    /// an operator that adds energy without bound). The invariant is a
    /// positive quadratic form of the fields while the scheme is stable;
    /// a mode that grows drives it below zero before the fields overflow,
    /// and `W < 0` must not read as a field that has decayed.
    Diverged,
}

/// Snapshot passed to the progress callback of
/// [`Engine::run_until_with`] after every energy check.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Progress {
    /// Engine [`timestep`](Engine::timestep) `n` of the energy sample.
    pub timestep: u64,
    /// `n·dt` \[s\].
    pub time: f64,
    /// Steps run by this call so far (the engine is one step past the
    /// sample: `timestep + 1`).
    pub steps: u64,
    /// The criterion's energy `W^n` \[J\]: the invariant
    /// [`Engine::energy`], less [`Engine::held_energy`] for
    /// [`EnergyMeasure::Invariant`].
    pub energy: f64,
    /// Largest `W` seen in this call \[J\].
    pub max_energy: f64,
    /// `10·log10(W/W_max)` \[dB\]: 0 while no energy has been seen,
    /// `-∞` if `W ≤ 0`.
    pub energy_db: f64,
    /// Wall time since the start of the call.
    pub elapsed: Duration,
    /// Average throughput of this call so far, cells × steps / wall time,
    /// in millions of cell updates per second.
    pub mcells_per_s: f64,
}

/// Result of [`Engine::run_until`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RunOutcome {
    /// Why the run stopped.
    pub reason: StopReason,
    /// Steps run by this call.
    pub steps: u64,
    /// Engine [`timestep`](Engine::timestep) at the end.
    pub timestep: u64,
    /// Engine [`time`](Engine::time) at the end \[s\].
    pub time: f64,
    /// Last measured energy \[J\] (`0` if none was measured).
    pub energy: f64,
    /// Largest measured energy \[J\].
    pub max_energy: f64,
    /// Wall time of the call.
    pub elapsed: Duration,
    /// Steps of the call at the first check that met the energy
    /// criterion; the run went on from there if the criterion has a
    /// [`settle`](EndCriteria::settle) test. `None` if none did.
    pub criterion_steps: Option<u64>,
}

impl Engine {
    /// Runs until the end criteria are met or `limits.max_timesteps`
    /// steps have been run (design §5.8). Equivalent to
    /// [`run_until_with`](Self::run_until_with) with a callback that never
    /// aborts.
    pub fn run_until(&mut self, limits: &RunLimits) -> Result<RunOutcome> {
        self.run_until_with(limits, |_| ControlFlow::Continue(()))
    }

    /// Runs like [`run_until`](Self::run_until), calling `progress` after
    /// every energy check; returning [`ControlFlow::Break`] stops the run
    /// with [`StopReason::Aborted`].
    ///
    /// # Algorithm
    ///
    /// Every `K = limits.check_interval()` steps, starting with the first
    /// step of the call, the step is run with
    /// [`step_with_energy`](Self::step_with_energy), which yields the
    /// energy invariant `W^n` of the state before it (`t = n·dt`); the
    /// criterion's `W` is that less [`held_energy`](Self::held_energy)
    /// of the same state, and `W_max` is updated. Then, in this order:
    ///
    /// 1. `progress` is called; `Break` → [`StopReason::Aborted`];
    /// 2. a non-finite `W`, or a negative one beyond round-off
    ///    (`W < −1e-12·W_max`) → [`StopReason::Diverged`];
    /// 3. if end criteria are set, `t > excitation_end` (or no excitation
    ///    end is given), `W_max > 0` and `W/W_max < energy_ratio` →
    ///    [`StopReason::EnergyDecay`]; with a
    ///    [`settle`](EndCriteria::settle) test only once the probes'
    ///    spectra are still or its cap is reached ([`Settle`]), the
    ///    energy being below the threshold at that check too;
    /// 4. otherwise the next `K − 1` steps run without energy evaluation
    ///    (batched, one barrier per half-step when no operator has serial
    ///    hooks).
    ///
    /// The engine therefore stops one step after the deciding sample. With
    /// [`RunLimits::check_after_interval`], each check instead follows `K`
    /// steps and evaluates the state reached ([`Engine::energy`]), so the
    /// run stops at the deciding state.
    /// When `max_timesteps` steps have run → [`StopReason::MaxTimesteps`].
    /// While no energy has been seen (`W_max = 0`) the criterion never
    /// fires. Energy is computed (and `progress` called) even without end
    /// criteria; with neither a criterion nor a callback, use
    /// [`run`](Self::run).
    ///
    /// The energy evaluation costs about one division per edge in one H
    /// half-step every `K` steps.
    ///
    /// # Errors
    /// [`Error::Invalid`] if `limits` fail [`RunLimits::validate`]; no step
    /// is run then.
    pub fn run_until_with<F>(&mut self, limits: &RunLimits, progress: F) -> Result<RunOutcome>
    where
        F: FnMut(&Progress) -> ControlFlow<()>,
    {
        drive(self, limits, progress)
    }
}

/// What [`drive`] steps: an [`Engine`], or the two engines of a Bloch
/// pair.
pub(crate) trait Stepper {
    /// Steps run so far (`n`).
    fn timestep(&self) -> u64;
    /// Time step \[s\].
    fn dt(&self) -> f64;
    /// Time `n·dt` \[s\].
    fn time(&self) -> f64;
    /// Cells per step (for the throughput).
    fn cells(&self) -> f64;
    /// One step, returning the energy invariant of the state before it.
    fn step_with_energy(&mut self) -> f64;
    /// `n` steps without energy evaluation.
    fn run(&mut self, n: u64);
    /// The part of the invariant of the current state that the end
    /// criterion leaves out ([`Engine::held_energy`]).
    fn held_energy(&self) -> f64 {
        0.0
    }
    /// The energy of the current state in `measure` without stepping, if
    /// the stepper supports it.
    fn energy_now(&mut self, measure: EnergyMeasure) -> Option<f64> {
        let _ = measure;
        None
    }
    /// Calls `f` with the sample times and values of every voltage and
    /// current probe, always in the same order
    /// ([`Settle`](crate::fdtd::Settle)); none by default.
    fn signals(&self, f: &mut dyn FnMut(&[f64], &[f64])) {
        let _ = f;
    }
}

/// The running spectra of a stepper's probes at one frequency, for
/// [`Settle`].
struct Settling {
    rule: Settle,
    /// Per probe: samples taken in, `X`, `A`.
    sums: Vec<(usize, f64, f64, f64)>,
    history: Vec<Checked>,
}

/// A check of [`Settling`]: the steps run, the energy and every probe's
/// `X`.
type Checked = (u64, f64, Vec<(f64, f64)>);

impl Settling {
    fn new(rule: Settle) -> Self {
        Self {
            rule,
            sums: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Takes in the samples since the last check, after `steps` steps
    /// with the energy `w`.
    fn update<S: Stepper + ?Sized>(&mut self, s: &S, steps: u64, w: f64) {
        let om = 2.0 * core::f64::consts::PI * self.rule.frequency;
        let (sums, mut q) = (&mut self.sums, 0);
        s.signals(&mut |times, values| {
            if sums.len() == q {
                sums.push((0, 0.0, 0.0, 0.0));
            }
            let (done, re, im, abs) = &mut sums[q];
            for (t, x) in times.iter().zip(values).skip(*done) {
                let (sin, cos) = (om * t).sin_cos();
                *re += x * cos;
                *im -= x * sin;
                *abs += x.abs();
            }
            *done = times.len().min(values.len());
            q += 1;
        });
        self.history
            .push((steps, w, sums.iter().map(|p| (p.1, p.2)).collect()));
    }

    /// Whether what every probe's spectrum still lacks, by the window
    /// that ends at the last check, is within the tolerance.
    fn settled(&self) -> bool {
        let Some((steps, w, now)) = self.history.last() else {
            return true;
        };
        let span = (self.rule.window * *steps as f64).ceil() as u64;
        let from = steps.saturating_sub(span);
        let (w0, then) = match self.history.partition_point(|h| h.0 <= from) {
            0 => return false,
            n => (self.history[n - 1].1, &self.history[n - 1].2),
        };
        // The energy's fall over the window, `ln(W₀/W)`: twice the
        // signals'. No field left: nothing to come.
        if *w <= 0.0 {
            return true;
        }
        let fall = (w0 / w).ln();
        if fall.is_nan() || fall <= 0.0 {
            return false;
        }
        self.sums.iter().enumerate().all(|(q, p)| {
            // (A probe added during the run has no value then: zero.)
            let (re, im) = then.get(q).copied().unwrap_or((0.0, 0.0));
            let moved = (now[q].0 - re).hypot(now[q].1 - im);
            2.0 * moved <= fall * self.rule.tolerance * p.3
        })
    }
}

impl Stepper for Engine {
    fn timestep(&self) -> u64 {
        self.step
    }
    fn dt(&self) -> f64 {
        self.dt
    }
    fn time(&self) -> f64 {
        Engine::time(self)
    }
    fn cells(&self) -> f64 {
        self.grid.num_cells() as f64
    }
    fn step_with_energy(&mut self) -> f64 {
        Engine::step_with_energy(self)
    }
    fn run(&mut self, n: u64) {
        Engine::run(self, n);
    }
    fn held_energy(&self) -> f64 {
        Engine::held_energy(self)
    }
    fn energy_now(&mut self, measure: EnergyMeasure) -> Option<f64> {
        Some(self.energy_in(measure))
    }
    fn signals(&self, f: &mut dyn FnMut(&[f64], &[f64])) {
        for r in &self.ops {
            let op: &dyn core::any::Any = &*r.op;
            if let Some(p) = op.downcast_ref::<super::Probe>() {
                f(p.times(), p.values());
            }
        }
    }
}

/// A negative energy invariant of more than this fraction of the largest
/// seen is a diverging run, not the round-off of a field that is gone.
const NEGATIVE_ENERGY: f64 = 1e-12;

/// The loop of [`Engine::run_until_with`] for any [`Stepper`].
pub(crate) fn drive<S, F>(s: &mut S, limits: &RunLimits, mut progress: F) -> Result<RunOutcome>
where
    S: Stepper + ?Sized,
    F: FnMut(&Progress) -> ControlFlow<()>,
{
    limits.validate()?;
    let interval = limits.check_interval();
    let max = limits.max_timesteps;
    let cells = s.cells();
    let start = Instant::now();
    let (mut steps, mut w, mut w_max) = (0u64, 0.0f64, 0.0f64);
    // The openEMS measure needs the state's own evaluation.
    let measure = limits.energy_measure;
    let supported = s.energy_now(measure).is_some();
    let after = limits.check_after_interval && supported;
    let before = !after && measure != EnergyMeasure::Invariant && supported;
    // The invariant counts what lossy conductors hold inside; the
    // criterion does not (`Operator::energy_held`).
    let held = |s: &S| match measure {
        EnergyMeasure::Invariant => s.held_energy(),
        EnergyMeasure::OpenEms => 0.0,
    };
    let mut settling = limits
        .end_criteria
        .and_then(|c| c.settle)
        .map(Settling::new);
    let mut met: Option<u64> = None;
    let reason = loop {
        if steps >= max {
            break StopReason::MaxTimesteps;
        }
        let n;
        if after {
            let regular = (steps / interval + 1) * interval;
            let next = limits
                .extra_check
                .filter(|&x| x > steps && x < regular)
                .unwrap_or(regular)
                .min(max);
            s.run(next - steps);
            steps = next;
            if !steps.is_multiple_of(interval) && limits.extra_check != Some(steps) {
                continue;
            }
            n = s.timestep();
            w = s.energy_now(measure).expect("checked above") - held(s);
        } else if before {
            n = s.timestep();
            w = s.energy_now(measure).expect("checked above");
            s.run(1);
            steps += 1;
        } else {
            n = s.timestep();
            let inside = held(s);
            w = s.step_with_energy() - inside;
            steps += 1;
        }
        w_max = w_max.max(w);
        let ratio = if w_max > 0.0 { w / w_max } else { 1.0 };
        let ratio_db = if ratio > 0.0 {
            10.0 * ratio.log10()
        } else {
            f64::NEG_INFINITY
        };
        let t = n as f64 * s.dt();
        let elapsed = start.elapsed();
        let p = Progress {
            timestep: n,
            time: t,
            steps,
            energy: w,
            max_energy: w_max,
            energy_db: ratio_db,
            elapsed,
            mcells_per_s: cells * steps as f64 / elapsed.as_secs_f64().max(1e-9) / 1e6,
        };
        if progress(&p).is_break() {
            break StopReason::Aborted;
        }
        if !w.is_finite() || w < -NEGATIVE_ENERGY * w_max {
            break StopReason::Diverged;
        }
        if let Some(z) = &mut settling {
            z.update(s, steps, w);
        }
        if let Some(c) = &limits.end_criteria
            && limits.excitation_end.is_none_or(|end| t > end)
            && w_max > 0.0
            && ratio < c.energy_ratio
        {
            let first = *met.get_or_insert(steps);
            // With a settle test: on from the first such check until the
            // probes' spectra are still, or the cap.
            let done = settling.as_ref().is_none_or(|z| {
                steps as f64 >= first as f64 * (1.0 + z.rule.max_extension) || z.settled()
            });
            if done {
                break StopReason::EnergyDecay { ratio_db };
            }
        }
        if !after {
            let chunk = (interval - 1).min(max - steps);
            s.run(chunk);
            steps += chunk;
        }
    };
    Ok(RunOutcome {
        reason,
        steps,
        timestep: s.timestep(),
        time: s.time(),
        energy: w,
        max_energy: w_max,
        elapsed: start.elapsed(),
        criterion_steps: met,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stepper with a scripted energy per check, and one probe whose
    /// sample `k` is `signal(k)` (none without).
    struct Script {
        n: u64,
        w: Vec<f64>,
        signal: Option<fn(u64) -> f64>,
    }

    impl Stepper for Script {
        fn timestep(&self) -> u64 {
            self.n
        }
        fn dt(&self) -> f64 {
            1.0
        }
        fn time(&self) -> f64 {
            self.n as f64
        }
        fn cells(&self) -> f64 {
            1.0
        }
        fn step_with_energy(&mut self) -> f64 {
            let w = self.w[(self.n as usize).min(self.w.len() - 1)];
            self.n += 1;
            w
        }
        fn run(&mut self, n: u64) {
            self.n += n;
        }
        fn signals(&self, f: &mut dyn FnMut(&[f64], &[f64])) {
            if let Some(x) = self.signal {
                let t: Vec<f64> = (0..self.n).map(|k| k as f64).collect();
                let v: Vec<f64> = (0..self.n).map(x).collect();
                f(&t, &v);
            }
        }
    }

    /// The settle test: a run whose energy is below its threshold from
    /// step 40 goes on while its probe's slow tail has more to give the
    /// spectrum than the tolerance, ends where it has not, and no later
    /// than the cap.
    #[test]
    fn a_slow_tail_keeps_the_run_going() {
        // A pulse of 20 steps and a tail of a hundredth of it that halves
        // every `HALF` steps, the energy with it.
        const HALF: f64 = 60.0;
        fn tail(k: u64) -> f64 {
            0.01 * (-(k as f64) * core::f64::consts::LN_2 / HALF).exp()
        }
        fn x(k: u64) -> f64 {
            tail(k) + if k < 20 { 1.0 } else { 0.0 }
        }
        let run = |settle: Option<Settle>, signal: Option<fn(u64) -> f64>, steady: bool| {
            let c = EndCriteria::energy_decay_db(-30.0)
                .check_interval(1)
                .settle(settle);
            let limits = RunLimits::new(100_000).end_criteria(c);
            let w = (0..5000)
                .map(|k| match k {
                    0..40 => 1.0,
                    _ if steady => 1e-4,
                    _ => tail(k) * tail(k),
                })
                .collect();
            let mut s = Script { n: 0, w, signal };
            drive(&mut s, &limits, |_| ControlFlow::Continue(())).unwrap()
        };
        let plain = run(None, Some(x), false);
        assert_eq!((plain.steps, plain.criterion_steps), (41, Some(41)));
        // Without a probe there is nothing to wait for.
        let none = run(Some(Settle::at(0.0)), None, false);
        assert_eq!(none.steps, 41);
        // Over the window of N/8 steps the spectrum moves by the tail's
        // sum there and the energy falls by 2^(−2·N/480): the estimate
        // is the tail's own remainder, below 1e-4 of A ≈ 20.9 at step
        // 560, below 1e-5 at 775.
        let far = |s: Settle| s.max_extension(50.0);
        let o = run(Some(far(Settle::at(0.0))), Some(x), false);
        assert!(matches!(o.reason, StopReason::EnergyDecay { .. }), "{o:?}");
        assert_eq!((o.steps, o.criterion_steps), (560, Some(41)), "{o:?}");
        let tight = run(Some(far(Settle::at(0.0).tolerance(1e-5))), Some(x), false);
        assert_eq!(tight.steps, 775, "{tight:?}");
        // The cap ends it; so it does where the energy stands still,
        // which says nothing of what is to come.
        let capped = run(Some(Settle::at(0.0).max_extension(1.0)), Some(x), false);
        assert_eq!(capped.steps, 82, "{capped:?}");
        let still = run(Some(Settle::at(0.0).max_extension(3.0)), Some(x), true);
        assert_eq!(still.steps, 164, "{still:?}");
        // At a frequency the tail turns many times in a window it gives
        // less, and the run is shorter.
        let f = run(Some(far(Settle::at(0.05))), Some(x), false);
        assert!(f.steps < o.steps, "{f:?}");
        assert!(Settle::at(1.0).window(1.0).validate().is_err());
        assert!(Settle::at(-1.0).validate().is_err());
        assert!(Settle::at(0.0).tolerance(0.0).validate().is_err());
    }

    fn outcome(w: &[f64]) -> RunOutcome {
        let limits =
            RunLimits::new(100).end_criteria(EndCriteria::energy_decay_db(-30.0).check_interval(1));
        let mut s = Script {
            n: 0,
            w: w.to_vec(),
            signal: None,
        };
        drive(&mut s, &limits, |_| ControlFlow::Continue(())).unwrap()
    }

    /// A negative invariant is a diverging run, not a decayed one: a
    /// scheme with a growing mode passes below zero on its way up, and
    /// `W/W_max < ratio` would end the run as converged there.
    #[test]
    fn negative_energy_is_divergence_not_decay() {
        let o = outcome(&[1.0, 4.0, 4.5, -0.2, 3.0]);
        assert_eq!(o.reason, StopReason::Diverged, "{o:?}");
        assert_eq!(o.steps, 4);
        // A field that is gone: zero, or round-off of either sign.
        for last in [0.0, 1e-20, -1e-20] {
            let o = outcome(&[1.0, 4.0, 2.0, last]);
            assert!(
                matches!(o.reason, StopReason::EnergyDecay { .. }),
                "{last}: {o:?}"
            );
        }
        assert_eq!(outcome(&[1.0, f64::NAN]).reason, StopReason::Diverged);
        assert_eq!(outcome(&[1.0, 2.0, 1.5]).reason, StopReason::MaxTimesteps);
    }
}
