//! openEMS `sar_calc` parity (`ems::io::openems::sar_calc`,
//! `ems-sar-calc`): the fixtures in `tests/fixtures/openems/sar_calc_*`
//! were written by the black-box `sar_calc` tool
//! (`scripts/openems-oracle/gen-sar.sh`) from openEMS's raw SAR dump of
//! the `dump_sar` fixture and from a synthetic tissue block.
//!
//! The file layout must match exactly. The values: method `SIMPLE` (also
//! openEMS's in-run method) equals openEMS where the centred cube is a
//! cube of uniform tissue clear of air and of the domain boundary, and is
//! within a few percent elsewhere (openEMS solves the cube size to ±5 % of
//! the mass only); the IEEE methods use ems's own procedure, which agrees
//! with openEMS on most voxels of the block but not at a thin tissue
//! block's surface (see `ems::post::sar::AveragingMethod`).
#![cfg(feature = "hdf5")]

#[path = "common/h5cmp.rs"]
mod h5cmp;

use std::path::{Path, PathBuf};

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

fn opts(input: PathBuf, output: PathBuf, mass_g: f64, method: AveragingMethod) -> SarCalcOptions {
    SarCalcOptions {
        input,
        output,
        mass_g,
        method,
        threads: 1,
        verbose: false,
        export_cube_stats: false,
        legacy: false,
    }
}

/// Values of dataset `p` of both files, and the reference maximum.
fn values(want: &Path, got: &Path, p: &str) -> (Vec<f64>, Vec<f64>, f64) {
    let (a, b) = (read_tree(want), read_tree(got));
    let x = a.objects[p].2.clone();
    let y = b.objects[p].2.clone();
    let m = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    (x, y, m)
}

const CUBE_COUNTS: [&str; 4] = ["valid_cubes", "used_cubes", "unused_cubes", "air_cubes"];

#[test]
fn raw_dump_layouts_and_simple_values() {
    let dir = fixture("sar_calc_dump");
    let raw = fixture("dump_sar").join("sar_raw.h5");
    let out = tmp_dir("dump");
    for (file, method, cubes, legacy) in [
        ("simple.h5", AveragingMethod::Simple, false, false),
        ("simple_e.h5", AveragingMethod::Simple, true, false),
        ("legacy.h5", AveragingMethod::Simple, false, true),
        ("ieee62704.h5", AveragingMethod::Ieee62704, false, false),
        ("c95.h5", AveragingMethod::IeeeC95_3, false, false),
    ] {
        let mut o = opts(raw.clone(), out.join(file), 1.0, method);
        o.export_cube_stats = cubes;
        o.legacy = legacy;
        run(&o).unwrap();
        let want = dir.join(file);
        if method == AveragingMethod::Simple {
            // Same structure, totals and cube counts; values within 3 %.
            let d = compare_h5(
                &want,
                &o.output,
                &["proc_time", "maxSAR"],
                &["/FieldData/FD/f0"],
            );
            assert!(d.attrs < 1e-6, "{file}: {d:?}");
            let (x, y, m) = values(&want, &o.output, "/FieldData/FD/f0");
            let worst = x
                .iter()
                .zip(&y)
                .fold(0.0f64, |w, (a, b)| w.max((a - b).abs()));
            eprintln!("{file}: SIMPLE differs by {:e} of the peak", worst / m);
            assert!(worst < 0.03 * m, "{file}: {}", worst / m);
        } else {
            let mut loose = vec!["proc_time", "maxSAR", "maxSAR_idx"];
            loose.extend(CUBE_COUNTS);
            compare_h5(&want, &o.output, &loose, &["/FieldData/FD/f0"]);
        }
    }
}

#[test]
fn tissue_block_simple_is_exact_inside() {
    let dir = fixture("sar_calc_block");
    let raw = dir.join("raw.h5");
    let out = tmp_dir("block");
    let o = opts(
        raw.clone(),
        out.join("simple.h5"),
        0.05,
        AveragingMethod::Simple,
    );
    run(&o).unwrap();
    let want = dir.join("simple.h5");
    // The f1 peak is at the surface: maxSAR (and its voxel) may differ.
    let d = compare_h5(
        &want,
        &o.output,
        &["proc_time", "maxSAR", "maxSAR_idx"],
        &["/FieldData/FD/f0", "/FieldData/FD/f1"],
    );
    assert!(d.attrs < 1e-6, "{d:?}");
    // 12 x 12 x 10 cells, tissue block 1..=10 x 1..=10 x 1..=8, cubes of
    // 50 cells (half side 1.84 cells): unclipped for 3..=8 x 3..=8 x 3..=6.
    let inside = |i: usize| {
        let (x, y, z) = (i / 120, (i / 10) % 12, i % 10);
        (3..=8).contains(&x) && (3..=8).contains(&y) && (3..=6).contains(&z)
    };
    for f in ["/FieldData/FD/f0", "/FieldData/FD/f1"] {
        let (x, y, m) = values(&want, &o.output, f);
        let (mut interior, mut all) = (0.0f64, 0.0f64);
        let mut n = 0;
        for (i, (a, b)) in x.iter().zip(&y).enumerate() {
            let e = (a - b).abs() / m;
            all = all.max(e);
            if inside(i) {
                interior = interior.max(e);
                n += 1;
            }
        }
        eprintln!("{f}: SIMPLE interior {interior:e}, all {all:e}");
        assert_eq!(n, 144);
        assert!(interior < 1e-6, "{f}: {interior}");
        assert!(all < 0.02, "{f}: {all}");
    }
    // IEEE_62704: same cube counts and peak voxel; most voxels agree.
    let o = opts(
        raw,
        out.join("ieee62704.h5"),
        0.05,
        AveragingMethod::Ieee62704,
    );
    run(&o).unwrap();
    let want = dir.join("ieee62704.h5");
    compare_h5(
        &want,
        &o.output,
        &["proc_time", "maxSAR"],
        &["/FieldData/FD/f0", "/FieldData/FD/f1"],
    );
    for f in ["/FieldData/FD/f0", "/FieldData/FD/f1"] {
        let (x, y, m) = values(&want, &o.output, f);
        let same = x
            .iter()
            .zip(&y)
            .filter(|(a, b)| (*a - *b).abs() <= 1e-6 * m)
            .count();
        eprintln!("{f}: IEEE_62704 equal on {same}/{}", x.len());
        assert!(same * 100 >= 95 * x.len(), "{f}: {same}");
    }
}

/// The binary takes `sar_calc`'s command line.
#[test]
fn command_line() {
    let raw = fixture("sar_calc_block").join("raw.h5");
    let out = tmp_dir("cli");
    let exe = env!("CARGO_BIN_EXE_ems-sar-calc");
    let o = out.join("cli.h5");
    let st = std::process::Command::new(exe)
        .args(["-n", "1", "--method=SIMPLE", "-m", "0.05", "-i"])
        .arg(&raw)
        .arg("--output")
        .arg(&o)
        .output()
        .unwrap();
    assert!(st.status.success(), "{st:?}");
    let lib = out.join("lib.h5");
    run(&opts(
        raw.clone(),
        lib.clone(),
        0.05,
        AveragingMethod::Simple,
    ))
    .unwrap();
    let d = compare_h5(&lib, &o, &["proc_time"], &[]);
    assert_eq!(d.worst(), 0.0);
    for bad in [
        vec!["-o", "x.h5", "-m", "1"],
        vec!["-i", "x.h5", "-m", "1"],
        vec!["-i", "x.h5", "-o", "y.h5", "-m", "1", "--method", "FOO"],
        vec!["-i", "x.h5", "-o", "y.h5", "-m", "1", "--bogus"],
    ] {
        let st = std::process::Command::new(exe).args(&bad).output().unwrap();
        assert!(!st.status.success(), "{bad:?}");
    }
}
