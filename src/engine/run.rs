//! Run loop with end criteria (design §5.8) and progress reporting.

use core::ops::ControlFlow;
use std::time::{Duration, Instant};

use super::Engine;
use crate::fdtd::EndCriteria;
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
}

impl RunLimits {
    /// At most `max_timesteps` steps, no energy criterion.
    pub fn new(max_timesteps: u64) -> Self {
        Self {
            max_timesteps,
            end_criteria: None,
            excitation_end: None,
        }
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
    /// The energy became NaN or infinite (unstable setup, e.g. an operator
    /// that adds energy without bound).
    Diverged,
}

/// Snapshot passed to the progress callback of
/// [`Engine::run_until_with`] after every energy check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
    /// Engine [`timestep`](Engine::timestep) `n` of the energy sample.
    pub timestep: u64,
    /// `n·dt` \[s\].
    pub time: f64,
    /// Steps run by this call so far (the engine is one step past the
    /// sample: `timestep + 1`).
    pub steps: u64,
    /// Energy invariant `W^n` \[J\] (see [`Engine::energy`]).
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
    /// energy invariant `W^n` of the state before it (`t = n·dt`), and
    /// `W_max` is updated. Then, in this order:
    ///
    /// 1. `progress` is called; `Break` → [`StopReason::Aborted`];
    /// 2. a non-finite `W` → [`StopReason::Diverged`];
    /// 3. if end criteria are set, `t > excitation_end` (or no excitation
    ///    end is given), `W_max > 0` and `W/W_max < energy_ratio` →
    ///    [`StopReason::EnergyDecay`];
    /// 4. otherwise the next `K − 1` steps run without energy evaluation
    ///    (batched, one barrier per half-step when no operator has serial
    ///    hooks).
    ///
    /// The engine therefore stops one step after the deciding sample.
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
    pub fn run_until_with<F>(&mut self, limits: &RunLimits, mut progress: F) -> Result<RunOutcome>
    where
        F: FnMut(&Progress) -> ControlFlow<()>,
    {
        limits.validate()?;
        let interval = limits.check_interval();
        let max = limits.max_timesteps;
        let cells = self.grid.num_cells() as f64;
        let start = Instant::now();
        let (mut steps, mut w, mut w_max) = (0u64, 0.0f64, 0.0f64);
        let reason = loop {
            if steps >= max {
                break StopReason::MaxTimesteps;
            }
            let n = self.step;
            w = self.step_with_energy();
            steps += 1;
            w_max = w_max.max(w);
            let ratio = if w_max > 0.0 { w / w_max } else { 1.0 };
            let ratio_db = if ratio > 0.0 {
                10.0 * ratio.log10()
            } else {
                f64::NEG_INFINITY
            };
            let t = n as f64 * self.dt;
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
            if !w.is_finite() {
                break StopReason::Diverged;
            }
            if let Some(c) = &limits.end_criteria
                && limits.excitation_end.is_none_or(|end| t > end)
                && w_max > 0.0
                && ratio < c.energy_ratio
            {
                break StopReason::EnergyDecay { ratio_db };
            }
            let chunk = (interval - 1).min(max - steps);
            self.run(chunk);
            steps += chunk;
        };
        Ok(RunOutcome {
            reason,
            steps,
            timestep: self.step,
            time: self.time(),
            energy: w,
            max_energy: w_max,
            elapsed: start.elapsed(),
        })
    }
}
