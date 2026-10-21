//! Black-box parity with openEMS on core XML semantics.
//!
//! Every case under `tests/fixtures/openems/<case>/` whose `model.xml` is an
//! "ems core parity fixture" holds openEMS v0.37 output for that model
//! (probe files, excitation signals `et`/`ht`, console output; see
//! `meta.txt` and `scripts/openems-oracle/gen-core.sh`). Each test runs ems
//! on the same XML, writes the probes in openEMS's format
//! ([`ems::io::openems::write_probes`]) and compares:
//!
//! * the time step (printed by openEMS with 6 digits) and the number of
//!   steps run;
//! * the excitation signal samples (`et` at `k·dt`, `ht` at `(k + ½)·dt`);
//! * per probe file: the header (coordinates, indices, columns) exactly,
//!   the sample times, and the values (relative L2 difference below the
//!   case's tolerance; ~1e-6 is typical where the formulation agrees).
//!
//! The tolerances leave room for openEMS output regenerated on another
//! machine; the version / date line is not compared.

#![cfg(feature = "xml")]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use ems::io::openems::{ProbeDomain, ProbeFile, read_probe_file};
use ems::{Results, Simulation};

fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/openems")
        .join(case)
}

/// Files of a fixture that are not probe files.
const NOT_PROBES: [&str; 5] = ["model.xml", "meta.txt", "console.txt", "et", "ht"];

/// A number from the console line starting with `prefix` (the first
/// whitespace-separated token after it).
fn console_value(case: &str, prefix: &str) -> Option<f64> {
    let text = std::fs::read_to_string(fixture(case).join("console.txt")).unwrap();
    text.lines().find_map(|l| {
        let i = l.find(prefix)?;
        l[i + prefix.len()..]
            .split_whitespace()
            .next()?
            .trim_end_matches(['s', ';'])
            .parse()
            .ok()
    })
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = a.iter().map(|x| x * x).sum();
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

/// Unique scratch directory for the ems output of `case`.
fn out_dir(case: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ems-openems-core-{}-{case}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

struct Check {
    case: &'static str,
    /// Relative L2 tolerance of the time-domain values (and the FD values).
    tol: f64,
    /// Per-probe tolerances overriding `tol`.
    per_probe: &'static [(&'static str, f64)],
}

impl Check {
    fn new(case: &'static str) -> Self {
        Check {
            case,
            tol: 1e-4,
            per_probe: &[],
        }
    }

    fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    fn per_probe(mut self, p: &'static [(&'static str, f64)]) -> Self {
        self.per_probe = p;
        self
    }

    fn run(&self) -> (Simulation, Results) {
        let mut sim = Simulation::from_xml(fixture(self.case).join("model.xml")).unwrap();
        let res = sim.run().unwrap();
        (sim, res)
    }

    /// Runs the case and checks everything; returns the results.
    fn check(&self) -> Results {
        let case = self.case;
        let (sim, res) = self.run();
        let dt = res.dt();

        // Time step: openEMS prints 6 significant digits.
        let dt_oe = console_value(case, "FDTD timestep is:").unwrap();
        assert!(
            (dt / dt_oe - 1.0).abs() < 6e-6,
            "{case}: dt {dt:e} vs openEMS {dt_oe:e}"
        );

        // Steps run.
        let steps = res.outcome().unwrap().steps;
        let want = console_value(case, "reached after")
            .or_else(|| console_value(case, "Max. number of timesteps:"))
            .unwrap();
        assert_eq!(steps as f64, want, "{case}: steps");

        // Excitation signal samples.
        let wf = sim.fdtd().get_waveform().unwrap().clone();
        for (file, offset) in [("et", 0.0), ("ht", 0.5)] {
            let text = std::fs::read_to_string(fixture(case).join(file)).unwrap();
            let rows: Vec<(f64, f64)> = text
                .lines()
                .filter_map(|l| {
                    let mut it = l.split_whitespace().map(|v| v.parse::<f64>().unwrap());
                    Some((it.next()?, it.next()?))
                })
                .collect();
            assert!(!rows.is_empty());
            let peak = rows.iter().fold(0.0f64, |m, r| m.max(r.1.abs()));
            // openEMS lists two periods of a sinusoid but runs with the
            // first one repeated (whose sample 0 is zero); compare one.
            let n = match wf {
                ems::Waveform::Sinusoid { f0 } => (1.0 / (f0 * dt)).round() as usize,
                _ => rows.len(),
            };
            for (k, &(t, v)) in rows.iter().enumerate().take(n) {
                let tk = (k as f64 + offset) * dt;
                assert!(
                    (t - tk).abs() <= 1e-5 * tk.max(dt),
                    "{case} {file}[{k}]: t {t:e}"
                );
                let got = wf.value(tk, dt);
                assert!(
                    (got - v).abs() <= 2e-5 * peak.max(1e-30) + 1e-6 * v.abs(),
                    "{case} {file}[{k}]: {got} vs openEMS {v}"
                );
            }
        }

        // Probe files.
        let dir = out_dir(case);
        ems::io::openems::write_probes(&res, &dir).unwrap();
        let names = |d: &Path, skip: &[&str]| -> BTreeSet<String> {
            std::fs::read_dir(d)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|n| !skip.contains(&n.as_str()))
                .collect()
        };
        let oe = names(&fixture(case), &NOT_PROBES);
        let ours = names(&dir, &[]);
        assert_eq!(ours, oe, "{case}: probe files");
        for name in &oe {
            let a_text = std::fs::read_to_string(fixture(case).join(name)).unwrap();
            let b_text = std::fs::read_to_string(dir.join(name)).unwrap();
            // Header lines after the version line: identical.
            let hdr = |t: &str| -> Vec<String> {
                t.lines()
                    .skip(1)
                    .filter(|l| l.starts_with('%'))
                    .map(str::to_string)
                    .collect()
            };
            assert_eq!(hdr(&a_text), hdr(&b_text), "{case} {name}: header");
            let a = ProbeFile::parse(&a_text).unwrap();
            let b = ProbeFile::parse(&b_text).unwrap();
            assert_eq!((a.domain, a.kind), (b.domain, b.kind), "{case} {name}");
            let tol = self
                .per_probe
                .iter()
                .find(|p| p.0 == name.trim_end_matches("_FD"))
                .map_or(self.tol, |p| p.1);
            match a.domain {
                ProbeDomain::Time => {
                    // H files included: ems adds the H after the last
                    // step like openEMS (`io::openems::write_probes`).
                    let n = b.rows.len();
                    assert_eq!(a.rows.len(), n, "{case} {name}: rows");
                    assert!(n > 0);
                    let ta = &a.abscissa()[..n];
                    let tb = b.abscissa();
                    for (x, y) in ta.iter().zip(&tb) {
                        assert!(
                            (x - y).abs() <= 1e-6 * x.abs().max(dt),
                            "{case} {name}: t {x} vs {y}"
                        );
                    }
                    let flat = |f: &ProbeFile| -> Vec<f64> {
                        f.rows[..n].iter().flat_map(|r| r[1..].to_vec()).collect()
                    };
                    let e = rel_l2(&flat(&a), &flat(&b));
                    eprintln!("{case} {name}: {} rows, rel. L2 {e:.2e}", n);
                    assert!(e < tol, "{case} {name}: rel. L2 {e:e} > {tol:e}");
                }
                ProbeDomain::Frequency => {
                    assert_eq!(a.abscissa(), b.abscissa(), "{case} {name}");
                    let flat = |f: &ProbeFile| -> Vec<f64> {
                        f.rows.iter().flat_map(|r| r[1..].to_vec()).collect()
                    };
                    let e = rel_l2(&flat(&a), &flat(&b));
                    eprintln!("{case} {name}: FD rel. difference {e:.2e}");
                    assert!(e < tol.max(1e-4), "{case} {name}: FD {e:e} > {tol:e}");
                }
            }
        }
        // The reader reads every file openEMS wrote.
        for name in &oe {
            read_probe_file(fixture(case).join(name)).unwrap();
        }
        let _ = std::fs::remove_dir_all(&dir);
        res
    }
}

#[test]
fn gauss_soft_e() {
    Check::new("gauss_soft_e").check();
}

#[test]
fn dirac_hard_e() {
    Check::new("dirac_hard_e").check();
}

#[test]
fn sine_delay_soft_h() {
    Check::new("sine_delay_soft_h").check();
}

#[test]
fn step_hard_h() {
    Check::new("step_hard_h").check();
}

#[test]
fn custom_weighted() {
    Check::new("custom_weighted").check();
}

#[test]
fn current_probes() {
    // openEMS disables a current probe whose box extends along its normal;
    // so does ems for files (warning).
    Check::new("current_probes").check();
}

#[test]
fn current_probe_ties() {
    // Transverse corners exactly half-way between lines snap inwards (the
    // headers and loops of i_half, i_rev, i_half2; i_pt_cross reads 0),
    // zero-length Cylinder / CylindricalShell probes are point boxes.
    Check::new("current_probe_ties").check();
}

#[test]
fn probe_skip() {
    // Voltage boxes that are not lines and current boxes extending along
    // their normal are skipped, as openEMS does.
    let (sim, _) = Check::new("probe_skip").run();
    let skipped = sim
        .warnings()
        .iter()
        .filter(|w| w.contains("as openEMS does"))
        .count();
    assert_eq!(skipped, 8, "{:?}", sim.warnings());
    Check::new("probe_skip").check();
}

#[test]
fn nonuniform_materials() {
    Check::new("nonuniform_materials").check();
}

#[test]
fn cfl_method() {
    Check::new("cfl_method").check();
}

#[test]
fn cell_constant_material() {
    Check::new("cell_constant_material").check();
}

#[test]
fn lumped_port() {
    Check::new("lumped_port").check();
}

#[test]
fn lumped_rc() {
    // The series RLC branch is the same trapezoidal (bilinear) update in
    // both codes; `urlc` over the low-impedance branch is ~1e-3 of the
    // field around it, so f32 round-off limits it to ~1.3e-3 (with an
    // excitation over the element, `lumped_exc`, 1e-6).
    Check::new("lumped_rc").per_probe(&[("urlc", 2e-3)]).check();
}

#[test]
fn lumped_l() {
    // openEMS's explicit inductor update (ems now uses it for files). The
    // voltage over the inductor is ~2e-5 of the field around it (the
    // inductor shorts the edge), so the f32 round-off of the neighbouring
    // fields limits `ut` to ~5e-4.
    Check::new("lumped_l").per_probe(&[("ut", 1.5e-3)]).check();
}

#[test]
fn lumped_exc() {
    // Excitations over series and parallel elements (plain soft source,
    // branch driven without it) and explicit parallel inductors.
    Check::new("lumped_exc").tol(1e-4).check();
}

#[test]
fn lumped_graded() {
    // R||C spread like a material over edges of different sizes; inductor
    // and series element in equal parts.
    Check::new("lumped_graded").tol(1e-4).check();
}

#[test]
fn lumped_graded_rl() {
    // Parallel elements with an inductor and no capacitor: openEMS limits
    // the edge capacitance (by R on `rl`, by L on `rn`); with a capacitor
    // (`rlc`) it does not.
    Check::new("lumped_graded_rl").tol(1e-4).check();
}

#[test]
fn pmc_mur() {
    Check::new("pmc_mur").check();
}

#[test]
fn pml() {
    // openEMS's PML_8 (ems runs openEMS's own layer for files).
    Check::new("pml").tol(5e-5).check();
}

#[test]
fn end_criteria() {
    let res = Check::new("end_criteria").check();
    // A multiple of the 57-step Nyquist interval.
    assert_eq!(res.outcome().unwrap().steps % 57, 0);
    // The energy decay at the stop step: openEMS prints −32.2833 dB; ems
    // counts the Mur face edges like openEMS (−32.2843 dB; −32.2868 dB
    // without them).
    let text = std::fs::read_to_string(fixture("end_criteria").join("console.txt")).unwrap();
    let oe: f64 = text
        .split("reached after")
        .nth(1)
        .and_then(|r| r.split('(').nth(1))
        .and_then(|r| r.split("dB").next())
        .unwrap()
        .parse()
        .unwrap();
    let ems::engine::StopReason::EnergyDecay { ratio_db } = res.outcome().unwrap().reason else {
        panic!("not stopped by the energy criterion");
    };
    eprintln!("end_criteria: openEMS {oe} dB, ems {ratio_db:.4} dB");
    assert!((ratio_db - oe).abs() < 2e-3, "{ratio_db} vs {oe}");
}

#[test]
fn max_time() {
    Check::new("max_time").check();
}
