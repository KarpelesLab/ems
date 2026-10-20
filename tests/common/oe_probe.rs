//! Minimal reader for openEMS time-domain probe text files (the committed
//! oracle fixtures in `tests/fixtures/openems/`), plus the comparison
//! helpers of the black-box parity tests.
//!
//! The format, as observed in openEMS output: `%`-prefixed header lines
//! (generator line with version and date, start / stop coordinates, column
//! names), then one whitespace-separated row per sample, time in seconds
//! first. Header lines are skipped entirely, so the tests never depend on
//! the version or timestamp.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// One probe file: sample times and value columns.
#[derive(Debug, Clone)]
pub struct OeProbe {
    /// Sample times \[s\].
    pub t: Vec<f64>,
    /// Columns after the time (1 for voltage/current, 3 for field probes).
    pub cols: Vec<Vec<f64>>,
}

impl OeProbe {
    /// Parses the file at `path`.
    pub fn read(path: impl AsRef<Path>) -> OeProbe {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let mut t = Vec::new();
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('%') {
                continue;
            }
            let mut it = line.split_whitespace().map(|v| {
                v.parse::<f64>()
                    .unwrap_or_else(|e| panic!("{}: bad number {v:?}: {e}", path.display()))
            });
            t.push(it.next().expect("time column"));
            let vals: Vec<f64> = it.collect();
            if cols.is_empty() {
                cols = vec![Vec::new(); vals.len()];
            }
            assert_eq!(vals.len(), cols.len(), "{}: ragged rows", path.display());
            for (c, v) in cols.iter_mut().zip(vals) {
                c.push(v);
            }
        }
        OeProbe { t, cols }
    }

    /// The value column of a scalar probe (voltage, current; for the
    /// waveguide probes the first of their two columns, the second being
    /// the mode purity).
    pub fn values(&self) -> &[f64] {
        assert!(matches!(self.cols.len(), 1 | 2), "not a scalar probe");
        &self.cols[0]
    }
}

/// Directory of fixture case `name`.
pub fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/openems")
        .join(name)
}

/// The `key: value` entries of a case's `meta.txt` with key `key`.
pub fn meta(name: &str, key: &str) -> Vec<String> {
    let p = fixture_dir(name).join("meta.txt");
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let pre = format!("{key}:");
    text.lines()
        .filter_map(|l| l.strip_prefix(&pre).map(|v| v.trim().to_string()))
        .collect()
}

/// The openEMS time step recorded in a case's `meta.txt`.
pub fn meta_dt(name: &str) -> f64 {
    meta(name, "dt")[0].parse().expect("dt in meta.txt")
}

/// Samples the ems series `(times, values)` at the openEMS sample times
/// `t` (linear interpolation; the sample times coincide when the time
/// steps agree) and returns the relative L2 difference
/// `‖ems − oe‖ / ‖oe‖` and the largest time mismatch to the nearest ems
/// sample, in units of `dt`.
pub fn rel_l2(t: &[f64], oe: &[f64], ems_t: &[f64], ems: &[f64], dt: f64) -> (f64, f64) {
    let (mut num, mut den, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    for (&ti, &vi) in t.iter().zip(oe) {
        if ti > *ems_t.last().unwrap() + 1e-6 * dt {
            break;
        }
        let k = ems_t.partition_point(|&x| x < ti);
        let v = if k == 0 {
            // Before the first ems sample: the field is still zero.
            worst = worst.max((ems_t[0] - ti) / dt * if vi == 0.0 { 0.0 } else { 1.0 });
            0.0
        } else if k >= ems_t.len() {
            ems[ems_t.len() - 1]
        } else {
            let (t0, t1) = (ems_t[k - 1], ems_t[k]);
            let d = (ti - t0).min(t1 - ti);
            worst = worst.max(d / dt);
            let w = (ti - t0) / (t1 - t0);
            ems[k - 1] * (1.0 - w) + ems[k] * w
        };
        num += (v - vi) * (v - vi);
        den += vi * vi;
    }
    ((num / den.max(1e-300)).sqrt(), worst)
}
