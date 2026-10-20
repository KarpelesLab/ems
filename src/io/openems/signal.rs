//! The excitation signal files openEMS writes before a run: `et` (the
//! signal at the E sample times `k·dt`) and `ht` (at the H sample times
//! `(k + ½)·dt`), plus the signal length it reports.
//!
//! As observed with openEMS v0.37 (fixtures in `tests/fixtures/openems/`,
//! see `tests/openems_core.rs`): no header, one row per sample, time and
//! value separated by a tab, both printed with 6 significant digits (C
//! `%g`):
//!
//! ```text
//! 0    0
//! 1.92583e-12    8.09555e-05
//! 3.85167e-12    8.01543e-05
//! ```
//!
//! The number of rows is the signal length openEMS reports on its console
//! ("Excitation signal length is: L timesteps"), [`ExcitationSignal`]:
//!
//! * Gaussian pulse: `⌊2T/dt⌋ + 1` samples (`T` its delay, see
//!   [`Waveform`]), cut to the number of time steps of the run;
//! * Dirac and step: 2 samples;
//! * custom expression: the number of time steps of the run;
//! * sinusoid: openEMS reports the **period** `P = round(1/(f0·dt))` in
//!   steps instead and writes two periods (`2P` rows).
//!
//! The values are those of [`Waveform::value`] with the time step `dt`
//! (openEMS's sampled signals; for the sinusoid the file's second period
//! repeats the first, so the sample at `t = P·dt` is written as 0 where
//! openEMS prints a value of order 1e-16).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::probes::fmt_g;
use crate::{Error, Result, Waveform};

/// How openEMS describes the excitation's extent in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalSpan {
    /// "Excitation signal length is: `n` timesteps".
    Length(u64),
    /// "Excitation signal period is: `n` timesteps" (sinusoid).
    Period(u64),
}

impl SignalSpan {
    /// The number of steps (length or period).
    pub fn steps(self) -> u64 {
        match self {
            SignalSpan::Length(n) | SignalSpan::Period(n) => n,
        }
    }
}

/// The sampled excitation signal of a run, as openEMS writes it to `et`
/// and `ht` (see the [module docs](self)).
#[derive(Debug, Clone, PartialEq)]
pub struct ExcitationSignal {
    /// The length (or period) openEMS reports.
    pub span: SignalSpan,
    /// For a Gaussian cut to the run: the uncut length in steps.
    pub uncut: Option<u64>,
    /// `(t, value)` at the E sample times `k·dt`.
    pub e: Vec<(f64, f64)>,
    /// `(t, value)` at the H sample times `(k + ½)·dt`.
    pub h: Vec<(f64, f64)>,
}

impl ExcitationSignal {
    /// The signal of `waveform` for time step `dt` in a run of
    /// `max_timesteps` steps.
    pub fn new(waveform: &Waveform, dt: f64, max_timesteps: u64) -> Self {
        let mut uncut = None;
        let (span, rows) = match *waveform {
            Waveform::Gaussian { .. } | Waveform::OpenEmsGaussian { .. } => {
                let full = waveform
                    .duration()
                    .map_or(1, |d| (d / dt).floor().max(0.0) as u64 + 1);
                let n = if full > max_timesteps {
                    uncut = Some(full);
                    max_timesteps
                } else {
                    full
                };
                (SignalSpan::Length(n), n)
            }
            Waveform::Sinusoid { f0 } => {
                let p = (1.0 / (f0 * dt)).round().max(1.0) as u64;
                (SignalSpan::Period(p), 2 * p)
            }
            Waveform::Dirac | Waveform::Step => (SignalSpan::Length(2), 2),
            Waveform::Custom(_) | Waveform::Expression(_) => {
                (SignalSpan::Length(max_timesteps), max_timesteps)
            }
        };
        let series = |offset: f64| -> Vec<(f64, f64)> {
            (0..rows)
                .map(|k| {
                    let t = (k as f64 + offset) * dt;
                    (t, waveform.value(t, dt))
                })
                .collect()
        };
        ExcitationSignal {
            span,
            uncut,
            e: series(0.0),
            h: series(0.5),
        }
    }

    /// The text of a signal file (`rows`: [`e`](Self::e) or
    /// [`h`](Self::h)).
    pub fn text(rows: &[(f64, f64)]) -> String {
        let mut s = String::with_capacity(rows.len() * 24);
        for &(t, v) in rows {
            let _ = writeln!(s, "{}\t{}", fmt_g(t, 6), fmt_g(v, 6));
        }
        s
    }

    /// Writes `dir/et` and `dir/ht`; returns their paths.
    ///
    /// # Errors
    /// I/O errors.
    pub fn write(&self, dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
        let dir = dir.as_ref();
        let mut out = Vec::new();
        for (name, rows) in [("et", &self.e), ("ht", &self.h)] {
            let p = dir.join(name);
            std::fs::write(&p, Self::text(rows))
                .map_err(|e| Error::Io(format!("{}: {e}", p.display())))?;
            out.push(p);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths() {
        let dt = 1.92583e-12;
        // openEMS: 992 samples for f0 = 3 GHz, fc = 1.5 GHz (fixture
        // `end_criteria`), cut to the run when longer.
        let g = Waveform::openems_gaussian(3e9, 1.5e9);
        let s = ExcitationSignal::new(&g, dt, 5000);
        assert_eq!(
            (s.span, s.uncut, s.e.len()),
            (SignalSpan::Length(992), None, 992)
        );
        let s = ExcitationSignal::new(&g, dt, 400);
        assert_eq!((s.span, s.uncut), (SignalSpan::Length(400), Some(992)));
        assert_eq!(s.h.len(), 400);
        let s = ExcitationSignal::new(&Waveform::dirac(), dt, 40);
        assert_eq!(s.e, vec![(0.0, 0.0), (dt, 1.0)]);
        assert_eq!(s.h[1], (1.5 * dt, 1.0));
        let s = ExcitationSignal::new(&Waveform::step(), dt, 40);
        assert_eq!(s.e, vec![(0.0, 1.0), (dt, 1.0)]);
        let s = ExcitationSignal::new(&Waveform::sinusoid(2e9), 5e-10 / 260.0, 600);
        assert_eq!((s.span, s.e.len()), (SignalSpan::Period(260), 520));
        let text = ExcitationSignal::text(&s.e[..2]);
        assert_eq!(text, "0\t0\n1.92308e-12\t0.0241637\n");
    }
}
