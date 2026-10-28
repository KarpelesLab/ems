//! The run statistics files of openEMS's `--dump-statistics` option.
//!
//! As observed with openEMS v0.37:
//!
//! * `openEMS_run_stats.txt`: a header line, then one row per progress
//!   report during the run (wall time since the start of the run \[s\],
//!   time step, speed \[cells/s\], field energy \[J\]), tab-separated, 6
//!   significant digits (`%g`):
//!
//!   ```text
//!   %time    timestep    speed    energy
//!   4.01699    1056    5.67828e+07    4.96784e-18
//!   ```
//!
//! * `openEMS_stats.txt`, written after the run: one value per line with
//!   its description after a tab, values with 16 significant digits:
//!
//!   ```text
//!   216000    % number of cells
//!   1.925833194728316e-12    % timestep (s)
//!   6000    % number of iterations
//!   1.155499916836989e-08    % total numerical time (s)
//!   20.66677    % simulation time (s)
//!   62709363.87253547    % speed (cells/s)
//!   ```
//!
//! openEMS counts "cells" as the number of mesh nodes (the product of the
//! line counts), so its speed is nodes × steps per second; [`RunStats`]
//! follows that. The energy column is the energy measure of the solver
//! (ems: [`Engine::energy`](crate::engine::Engine::energy)), which differs
//! slightly from openEMS's (design §5.16.1).

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use super::probes::fmt_g;
use crate::{Error, Result};

/// File name of the per-report statistics.
pub const RUN_STATS_FILE: &str = "openEMS_run_stats.txt";
/// File name of the summary statistics.
pub const STATS_FILE: &str = "openEMS_stats.txt";

/// The summary of a run (`openEMS_stats.txt`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RunStats {
    /// Mesh nodes (openEMS's "number of cells").
    pub cells: u64,
    /// Time step \[s\].
    pub dt: f64,
    /// Steps run.
    pub iterations: u64,
    /// Wall time of the run \[s\].
    pub wall_time: f64,
}

impl RunStats {
    /// The statistics of a run of `iterations` steps of `dt` on `cells`
    /// mesh nodes that took `wall_time` seconds.
    pub fn new(cells: u64, dt: f64, iterations: u64, wall_time: f64) -> Self {
        Self {
            cells,
            dt,
            iterations,
            wall_time,
        }
    }

    /// Simulated time `iterations · dt` \[s\].
    pub fn numerical_time(&self) -> f64 {
        self.iterations as f64 * self.dt
    }

    /// Speed in cells (nodes) per second of wall time.
    pub fn speed(&self) -> f64 {
        self.cells as f64 * self.iterations as f64 / self.wall_time.max(1e-9)
    }

    /// The text of `openEMS_stats.txt`.
    pub fn text(&self) -> String {
        let mut s = String::new();
        for (v, what) in [
            (self.cells as f64, "number of cells"),
            (self.dt, "timestep (s)"),
            (self.iterations as f64, "number of iterations"),
            (self.numerical_time(), "total numerical time (s)"),
            (self.wall_time, "simulation time (s)"),
            (self.speed(), "speed (cells/s)"),
        ] {
            let _ = writeln!(s, "{}\t% {what}", fmt_g(v, 16));
        }
        s
    }

    /// Writes `dir/openEMS_stats.txt`; returns its path.
    ///
    /// # Errors
    /// I/O errors.
    pub fn write(&self, dir: impl AsRef<Path>) -> Result<PathBuf> {
        let p = dir.as_ref().join(STATS_FILE);
        std::fs::write(&p, self.text()).map_err(|e| Error::Io(format!("{}: {e}", p.display())))?;
        Ok(p)
    }
}

/// `openEMS_run_stats.txt`, written while the run goes: the header when
/// created, a row per [`report`](Self::report).
#[derive(Debug)]
#[non_exhaustive]
pub struct RunStatsLog {
    path: PathBuf,
    file: std::fs::File,
}

impl RunStatsLog {
    /// Creates (truncates) `dir/openEMS_run_stats.txt` and writes the
    /// header line.
    ///
    /// # Errors
    /// I/O errors.
    pub fn create(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join(RUN_STATS_FILE);
        let io = |e: std::io::Error| Error::Io(format!("{}: {e}", path.display()));
        let mut file = std::fs::File::create(&path).map_err(io)?;
        file.write_all(b"%time\ttimestep\tspeed\tenergy\n")
            .map_err(io)?;
        Ok(Self { path, file })
    }

    /// The row of one progress report: wall time \[s\], time step,
    /// speed \[cells/s\], energy \[J\].
    pub fn row(time: f64, timestep: u64, speed: f64, energy: f64) -> String {
        format!(
            "{}\t{timestep}\t{}\t{}\n",
            fmt_g(time, 6),
            fmt_g(speed, 6),
            fmt_g(energy, 6)
        )
    }

    /// Appends one row (see [`row`](Self::row)).
    ///
    /// # Errors
    /// I/O errors.
    pub fn report(&mut self, time: f64, timestep: u64, speed: f64, energy: f64) -> Result<()> {
        self.file
            .write_all(Self::row(time, timestep, speed, energy).as_bytes())
            .map_err(|e| Error::Io(format!("{}: {e}", self.path.display())))
    }

    /// The file's path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        let s = RunStats {
            cells: 216_000,
            dt: 1.925833194728316e-12,
            iterations: 6000,
            wall_time: 20.66677,
        };
        let t = s.text();
        let lines: Vec<&str> = t.lines().collect();
        assert_eq!(lines[0], "216000\t% number of cells");
        assert_eq!(lines[1], "1.925833194728316e-12\t% timestep (s)");
        assert_eq!(lines[2], "6000\t% number of iterations");
        assert!(
            lines[3].starts_with("1.1554999168369")
                && lines[3].ends_with("e-08\t% total numerical time (s)")
        );
        assert_eq!(lines[4], "20.66677\t% simulation time (s)");
        assert!(lines[5].starts_with("6270936") && lines[5].ends_with("\t% speed (cells/s)"));
        assert_eq!(
            RunStatsLog::row(4.01699, 1056, 5.67828e7, 4.96784e-18),
            "4.01699\t1056\t5.67828e+07\t4.96784e-18\n"
        );
    }
}
