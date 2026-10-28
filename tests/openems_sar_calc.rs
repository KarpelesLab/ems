//! openEMS `sar_calc` parity (`ems::io::openems::sar_calc`,
//! `ems-sar-calc`): the fixtures in `tests/fixtures/openems/sar_calc_*`
//! were written by the black-box `sar_calc` tool
//! (`scripts/openems-oracle/gen-sar.sh`) from openEMS's raw SAR dump of
//! the `dump_sar` fixture and from synthetic raw files (`sar-synth.py`).
//!
//! Every `command:` line of a case's `meta.txt` is replayed with
//! `ems-sar-calc` (same options) and the whole output file is compared:
//! structure, every attribute but `@proc_time`, every dataset to float
//! precision. One known exception: the IEEE C95.3 face cubes of
//! `dump_sar`, whose six orientations have equal volumes there and are
//! ordered by openEMS's rounding (see `ems::post::sar::AveragingMethod`).
#![cfg(feature = "hdf5")]

#[path = "common/h5cmp.rs"]
mod h5cmp;

use std::path::{Path, PathBuf};
use std::process::Command;

use ems::io::openems::sar_calc::{SarCalcOptions, run};
use ems::post::sar::AveragingMethod;
use h5cmp::{compare_h5, read_tree};

fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/openems")
        .join(case)
}

fn tmp_dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("openems_sar_calc")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The `sar_calc` command lines of a fixture case (arguments after
/// `sar_calc`).
fn commands(case: &str) -> Vec<Vec<String>> {
    let meta = std::fs::read_to_string(fixture(case).join("meta.txt")).unwrap();
    meta.lines()
        .filter_map(|l| l.strip_prefix("command: sar_calc "))
        .map(|l| l.split_whitespace().map(String::from).collect())
        .collect()
}

/// Number of voxels of dataset `p` that differ by more than `tol` of the
/// largest value.
fn differing(want: &Path, got: &Path, p: &str, tol: f64) -> usize {
    let (a, b) = (read_tree(want), read_tree(got));
    let (x, y) = (&a.objects[p].2, &b.objects[p].2);
    let m = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    x.iter()
        .zip(y)
        .filter(|(a, b)| (*a - *b).abs() > tol * m)
        .count()
}

/// Replays the commands of `case` with `ems-sar-calc` and compares the
/// outputs; `loose` lists (file, largest number of differing voxels of
/// `f0`) for the known exceptions.
fn replay(case: &str, loose: &[(&str, usize)]) {
    let dir = fixture(case);
    let out = tmp_dir(case);
    let exe = env!("CARGO_BIN_EXE_ems-sar-calc");
    let cmds = commands(case);
    assert!(!cmds.is_empty(), "{case}: no commands in meta.txt");
    for args in cmds {
        let mut args = args.clone();
        let mut file = String::new();
        for i in 0..args.len() {
            match args[i].as_str() {
                "-i" => args[i + 1] = dir.join(&args[i + 1]).display().to_string(),
                "-o" => {
                    file = args[i + 1].clone();
                    args[i + 1] = out.join(&file).display().to_string();
                }
                _ => {}
            }
        }
        let st = Command::new(exe).args(&args).output().unwrap();
        assert!(st.status.success(), "{case} {args:?}: {st:?}");
        let (want, got) = (dir.join(&file), out.join(&file));
        let limit = loose.iter().find(|(f, _)| *f == file).map(|l| l.1);
        let skip: &[&str] = if limit.is_some() {
            &["/FieldData/FD/f0"]
        } else {
            &[]
        };
        let d = compare_h5(&want, &got, &["proc_time"], skip);
        eprintln!("{case}/{file}: {d:?}");
        assert!(d.attrs < 1e-6, "{case}/{file}: attributes {d:?}");
        // 62704 solves the cube masses to 1e-6: an iterate on either side
        // of that tolerance can end both codes on different iterates.
        assert!(d.data < 2e-6, "{case}/{file}: data {d:?}");
        if let Some(n) = limit {
            let k = differing(&want, &got, "/FieldData/FD/f0", 1e-6);
            eprintln!("{case}/{file}: {k} voxels differ");
            assert!(k <= n, "{case}/{file}: {k} voxels differ");
        }
    }
}

/// The raw SAR dump openEMS wrote for `dump_sar` (7×7×6 tissue cells of
/// 2 mm, 1 g: every cube clipped by air, touching faces).
#[test]
fn raw_dump_matches_openems() {
    replay("sar_calc_dump", &[("c95.h5", 16)]);
}

/// A uniform synthetic tissue block (1 mm cells, 0.05 g, two
/// frequencies).
#[test]
fn tissue_block_matches_openems() {
    replay("sar_calc_block", &[]);
}

/// Synthetic raw files of `sar-synth.py` (graded meshes, random
/// densities, irregular tissue, thin slabs with face cubes only,
/// anisotropic and small cubes, `-a` auto range).
#[test]
fn synthetic_cases_match_openems() {
    replay("sar_calc_synth", &[]);
}

/// Edge cases of `sar-synth.py` (cubes smaller than a cell, the IEEE
/// validity of cubes of about two cells on a graded mesh, face cubes
/// shorter than their voxel), each method; exact, cube statistics
/// included.
#[test]
fn edge_cases_match_openems() {
    replay("sar_calc_edge", &[]);
}

/// The library entry point writes what the binary writes.
#[test]
fn library_matches_binary() {
    let raw = fixture("sar_calc_block").join("raw.h5");
    let out = tmp_dir("lib");
    let mut o = SarCalcOptions::new(&raw, out.join("lib.h5"), 0.05);
    o.method = AveragingMethod::IeeeC95_3;
    o.threads = 3;
    o.autorange = Some(3.0);
    let r = run(&o).unwrap();
    assert!(r.range.is_some());
    let exe = env!("CARGO_BIN_EXE_ems-sar-calc");
    let cli = out.join("cli.h5");
    let st = Command::new(exe)
        .args([
            "-n",
            "1",
            "--method=IEEE_C95_3",
            "-m",
            "0.05",
            "-a",
            "3",
            "-i",
        ])
        .arg(&raw)
        .arg("--output")
        .arg(&cli)
        .output()
        .unwrap();
    assert!(st.status.success(), "{st:?}");
    let d = compare_h5(&o.output, &cli, &["proc_time"], &[]);
    assert_eq!(d.worst(), 0.0, "{d:?}");
}

/// The binary rejects bad command lines.
#[test]
fn command_line_errors() {
    let exe = env!("CARGO_BIN_EXE_ems-sar-calc");
    for bad in [
        vec!["-o", "x.h5", "-m", "1"],
        vec!["-i", "x.h5", "-m", "1"],
        vec!["-i", "x.h5", "-o", "y.h5", "-m", "1", "--method", "FOO"],
        vec!["-i", "x.h5", "-o", "y.h5", "-m", "1", "--bogus"],
        vec!["-i", "x.h5", "-o", "y.h5", "-m", "1", "-a", "0"],
    ] {
        let st = Command::new(exe).args(&bad).output().unwrap();
        assert!(!st.status.success(), "{bad:?}");
    }
}
