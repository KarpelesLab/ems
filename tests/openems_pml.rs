//! openEMS `PML_<n>` layers against the black-box openEMS (fixtures
//! `tests/fixtures/openems/pml_*`, written by
//! `scripts/openems-oracle/gen-pml.sh`, never by the test suite; design
//! §5.16.2). ems runs these faces with openEMS's own layer
//! ([`ems::engine::boundary::upml::OpenEmsPml`]) for files:
//!
//! * `pml_face`: one layer, oblique incidence, probes inside it;
//! * `pml_exact`: openEMS's arithmetic, bit for bit (fixed time step,
//!   overlapping layers, a lossy block in the layers);
//! * `pml_corner`: four layers overlapping in edges and three-layer
//!   corners, a graded layer, a lossy block and a metal plate in the
//!   layers, raw E/H line dumps through the corners (feature `hdf5`);
//! * `pml_medium`: layers in a lossy magnetic medium (conductivities
//!   dropped inside the layer);
//! * `pml_lumped`: lumped elements across and inside a layer (their R
//!   and C act outside it only);
//! * `pml_end`: the energy end criterion (stop step and level);
//! * `pml_cyl`: a cylindrical grid with the axis, layers on `ρ_max` and
//!   both z faces;
//! * `pml_wg_iris`, `pml_msl_filter`: the end-to-end waveguide iris and
//!   microstrip filter with PML ends.
//!
//! The `ems` binary runs each model as `openEMS` would; probe series,
//! steps run, the end-criterion line and the dumps are compared.
#![cfg(feature = "xml")]

#[cfg(feature = "hdf5")]
#[path = "common/h5cmp.rs"]
mod h5cmp;

use std::path::{Path, PathBuf};
use std::process::Command;

use ems::io::openems::{ProbeDomain, ProbeFile, read_model, read_probe_file};
use ems::{CompileOptions, Simulation};

fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/openems")
        .join(case)
}

/// The `key:` line of a case's `meta.txt`.
fn meta(case: &str, key: &str) -> String {
    let text = std::fs::read_to_string(fixture(case).join("meta.txt")).unwrap();
    let pre = format!("{key}:");
    text.lines()
        .find_map(|l| l.strip_prefix(&pre).map(|v| v.trim().to_string()))
        .unwrap_or_else(|| panic!("{case}/meta.txt: no {key}"))
}

/// Runs the `ems` binary (three threads) on the case's model in a fresh
/// scratch directory; returns the directory and the console output.
fn run(case: &str) -> (PathBuf, String) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("openems_pml")
        .join(case);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(fixture(case).join("model.xml"), dir.join("model.xml")).unwrap();
    let mut args = vec!["model.xml".to_string(), "--numThreads=3".to_string()];
    if !cfg!(feature = "hdf5") {
        args.push("--disable-dumps".into());
    }
    let out = Command::new(env!("CARGO_BIN_EXE_ems"))
        .args(&args)
        .current_dir(&dir)
        .output()
        .expect("run the ems binary");
    let console = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{case}: ems failed\n{console}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    (dir, console)
}

/// The run made as many steps as openEMS's.
fn check_steps(case: &str, console: &str) {
    let got: u64 = console
        .lines()
        .find_map(|l| l.strip_prefix("Time for "))
        .and_then(|l| l.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .expect("Time for … iterations line");
    let want: u64 = meta(case, "steps").parse().unwrap();
    assert_eq!(got, want, "{case}: steps run");
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Relative L2 difference of probe file `name` (every value column against
/// the norm of the largest one), asserted below `tol`.
fn compare_probe(case: &str, dir: &Path, name: &str, tol: f64) -> f64 {
    let a = read_probe_file(fixture(case).join(name)).unwrap();
    let b = read_probe_file(dir.join(name)).unwrap();
    assert_eq!((a.domain, a.kind), (b.domain, b.kind), "{case}/{name}");
    // ems adds no look-ahead H row on cylindrical grids.
    assert!(
        a.rows.len().abs_diff(b.rows.len()) <= 1,
        "{case}/{name}: rows {} vs {}",
        a.rows.len(),
        b.rows.len()
    );
    let n = a.rows.len().min(b.rows.len());
    for (x, y) in a.abscissa().iter().zip(b.abscissa()).take(n) {
        assert!(
            (x - y).abs() <= 1e-6 * x.abs().max(1e-30),
            "{case}/{name}: abscissa {x} vs {y}"
        );
    }
    let cols = if a.kind.is_mode_matching() && a.domain == ProbeDomain::Time {
        1
    } else {
        a.rows[0].len() - 1
    };
    let col =
        |f: &ProbeFile, c: usize| -> Vec<f64> { f.rows[..n].iter().map(|r| r[1 + c]).collect() };
    let scale = (0..cols).map(|c| norm(&col(&a, c))).fold(0.0, f64::max);
    let worst = (0..cols)
        .map(|c| {
            let d: Vec<f64> = col(&a, c)
                .iter()
                .zip(col(&b, c))
                .map(|(p, q)| p - q)
                .collect();
            norm(&d) / scale.max(1e-300)
        })
        .fold(0.0, f64::max);
    eprintln!("{case}/{name}: {n} rows, relative L2 {worst:.2e}");
    assert!(
        worst < tol,
        "{case}/{name}: relative L2 {worst:e} > {tol:e}"
    );
    worst
}

/// Runs `case` and compares every kept probe file (not the HDF5 dumps)
/// and the steps run.
fn check(case: &str, tol: f64) -> (PathBuf, String) {
    let (dir, console) = run(case);
    check_steps(case, &console);
    for f in meta(case, "keep").split_whitespace() {
        if !f.ends_with(".h5") {
            compare_probe(case, &dir, f, tol);
        }
    }
    (dir, console)
}

/// One layer at oblique incidence: 9e-7…3e-6 outside the layer and at
/// depths 2 and 5 cells inside it.
#[test]
fn one_face() {
    check("pml_face", 2e-5);
}

/// openEMS's layer arithmetic (design §5.16.8): three overlapping layers
/// (one graded), a lossy dielectric-magnetic block reaching into all of
/// them, a fixed time step. Every probe, including those in the edge and
/// corner overlaps, is openEMS's to the printed digits (was 1e-6…5e-6
/// with the collapsed single-stage update and ems's constants).
#[test]
fn exact_arithmetic() {
    check("pml_exact", 1e-6);
}

/// Overlapping layers (edges, three-layer corners), a graded layer, a
/// lossy block, a metal plate and a soft source inside layers: probes in
/// the overlaps 4e-7…5e-6; the raw line dumps through two corners 2e-6
/// (E) and 5e-5 (H) of their largest value. The H difference is in the
/// late ring-down below 1e-4 of the peak: a source inside a layer leaves
/// a static field there (openEMS keeps its contribution in E, not in the
/// flux), and its round-off (1e-6) differs between the codes; with the
/// source outside the layer the H dump agrees to 3e-6.
#[test]
fn corners() {
    let case = "pml_corner";
    let (_dir, _) = check(case, 2e-5);
    #[cfg(feature = "hdf5")]
    for f in ["E_line.h5", "H_line.h5"] {
        let (want, got) = (fixture(case).join(f), _dir.join(f));
        // Structure and attributes; the values against the largest one of
        // the whole series (the late, decayed snapshots are round-off).
        let d = h5cmp::compare_h5(&want, &got, &[], &[]);
        assert!(d.attrs < 1e-6, "{case}/{f}: {d:?}");
        let (a, b) = (h5cmp::read_tree(&want), h5cmp::read_tree(&got));
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (k, (_, _, va)) in a
            .objects
            .iter()
            .filter(|(k, _)| k.starts_with("/FieldData"))
        {
            for (x, y) in va.iter().zip(&b.objects[k].2) {
                num = num.max((x - y).abs());
                den = den.max(x.abs());
            }
        }
        eprintln!("{case}/{f}: {:.2e} of the largest value", num / den);
        assert!(num < 1e-4 * den, "{case}/{f}: {num:e} of {den:e}");
    }
}

/// Lumped elements reaching into a layer: openEMS's layer replaces the
/// update of its edges, so an element's R and C act only outside it (the
/// caps and an excitation over the element still do). Measured: the
/// probes 5e-7…1.1e-5, the field probe deep in the layer 7e-5 (diverged,
/// ~1e7, with the R and C folded into the layer's coefficients).
#[test]
fn lumped_in_layer() {
    let (dir, console) = run("pml_lumped");
    check_steps("pml_lumped", &console);
    for (f, tol) in [
        ("u_src", 1e-5),
        ("u_mid", 2e-5),
        ("u_cross", 5e-5),
        ("u_cross_in", 5e-5),
        ("u_rc", 2e-5),
        ("u_port", 1e-5),
        ("i_port", 3e-5),
        ("e_in", 3e-4),
        ("h_in", 2e-4),
    ] {
        compare_probe("pml_lumped", &dir, f, tol);
    }
}

/// A lossy magnetic medium filling the layers (openEMS drops κ and σ_m
/// inside them): 4e-7…1.5e-6.
#[test]
fn lossy_medium() {
    check("pml_medium", 1e-5);
}

/// The energy end criterion counts the layers with their true fields:
/// the same stop step (1956) and level as openEMS (−50.509 against
/// openEMS's −50.518 dB: its usual 0.005–0.01 dB offset, design §5.16.1;
/// with the CPML ems stopped at 1908); probes 3e-6.
#[test]
fn end_criterion() {
    let case = "pml_end";
    let (_, console) = check(case, 2e-5);
    let level = |l: &str| -> f64 {
        let s = l.rsplit('(').next().unwrap();
        s.trim_end_matches("dB)").parse().unwrap()
    };
    let want = level(&meta(case, "end"));
    let got = console
        .lines()
        .find(|l| l.contains("end-criteria"))
        .map(level)
        .expect("end-criteria line");
    assert!(
        (got - want).abs() < 0.02,
        "{case}: stop level {got} vs {want} dB"
    );
}

/// Cylindrical grid (closed azimuth, no axis): layers on `ρ_max` and both
/// z faces, 1.4e-6…7e-6.
#[test]
fn cylindrical() {
    check("pml_cyl", 3e-5);
}

/// The WR-90 iris with PML ends (6.6 GHz cutoff, 7–13 GHz pulse): 3e-7…7e-7
/// (2.4e-2 with the CPML) and the energy criterion stops at the same step.
#[test]
fn waveguide_iris() {
    check("pml_wg_iris", 1e-5);
}

/// The microstrip filter with PML line ends (lossy substrate and strip in
/// the layers): ports 2e-7…5e-7 (3e-3 with the CPML), the stub field 1e-5.
#[test]
fn microstrip_filter() {
    check("pml_msl_filter", 5e-5);
}

/// The layer runs in slab hooks: bit-identical results for any thread
/// count.
#[test]
fn thread_count_invariant() {
    let (doc, _) = read_model(fixture("pml_corner").join("model.xml")).unwrap();
    let probes = |threads: usize| {
        let mut sim = Simulation::from_document_with(&doc, &CompileOptions::default(), |f| {
            f.threads(threads)
        })
        .unwrap();
        let res = sim.run().unwrap();
        res.probes()
            .iter()
            .map(|p| p.values().clone())
            .collect::<Vec<_>>()
    };
    let one = probes(1);
    assert!(!one.is_empty());
    for t in [2, 5] {
        assert_eq!(probes(t), one, "{t} threads");
    }
}
