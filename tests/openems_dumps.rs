//! openEMS-compatible dump files: ems runs the fixture models of
//! `tests/fixtures/openems/dump_*` (written by a black-box openEMS, see
//! `scripts/openems-oracle/gen-dumps.sh`) and writes its dumps with
//! `ems::io::openems::dump`; every file openEMS wrote must exist with the
//! same structure (groups, datasets, shapes, datatypes, attributes; VTK
//! element structure) and the same values within a float32 tolerance.
//!
//! ems applies openEMS's excitation timing to models read from files
//! (design §5.16.1), so the models run as written.
#![cfg(all(feature = "xml", feature = "hdf5"))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ems::Simulation;
use ems::io::openems::dump::{DumpFileOptions, write_dump, write_sar};

#[path = "common/h5cmp.rs"]
mod h5cmp;
use h5cmp::compare_h5;

fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/openems")
        .join(case)
}

/// Runs `case` with ems and writes all dumps into a fresh directory.
fn run_ems(case: &str, variant: &str, opts: &DumpFileOptions) -> PathBuf {
    let out = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("openems_dumps")
        .join(case)
        .join(variant);
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    let mut sim = Simulation::from_xml(fixture(case).join("model.xml")).unwrap();
    let res = sim.run().unwrap();
    for d in res.dumps() {
        write_dump(&out, d, opts).unwrap();
    }
    for s in res.sar_dumps() {
        write_sar(&out, s).unwrap();
    }
    out
}

// ---------------------------------------------------------------------
// VTK comparison.

/// The XML structure of a VTK file without the binary payloads.
fn vtk_skeleton(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| {
            let t = l.trim();
            !(t.is_empty()
                || (!t.starts_with('<')
                    && t.chars()
                        .all(|c| c.is_ascii_alphanumeric() || "+/=".contains(c))))
        })
        .map(str::to_string)
        .collect()
}

#[cfg(feature = "hdf5-deflate")]
fn b64(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            _ => 63,
        }
    };
    let mut out = Vec::new();
    for ch in s.as_bytes().chunks(4) {
        let n = ch.iter().filter(|&&c| c != b'=').count();
        let mut v = 0u32;
        for (i, &c) in ch.iter().enumerate() {
            if c != b'=' {
                v |= val(c) << (18 - 6 * i);
            }
        }
        out.extend(&v.to_be_bytes()[1..n.max(1)]);
    }
    out
}

/// Decodes the compressed binary arrays of a VTK file (f32 or f64 each).
#[cfg(feature = "hdf5-deflate")]
fn vtk_arrays(text: &str) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    let mut ty = "";
    for l in text.lines() {
        let t = l.trim();
        if t.starts_with("<DataArray") {
            ty = if t.contains("Float64") { "f64" } else { "f32" };
            continue;
        }
        if t.is_empty() || t.starts_with('<') || !t.ends_with('=') && t.len() < 8 {
            continue;
        }
        if !t
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "+/=".contains(c))
        {
            continue;
        }
        let first = b64(&t[..8]);
        let nb = u32::from_le_bytes(first[..4].try_into().unwrap()) as usize;
        let hlen = (3 + nb) * 4;
        let hb = hlen.div_ceil(3) * 4;
        let head = b64(&t[..hb]);
        let sizes: Vec<usize> = (0..nb)
            .map(|i| u32::from_le_bytes(head[12 + 4 * i..16 + 4 * i].try_into().unwrap()) as usize)
            .collect();
        let data = b64(&t[hb..]);
        let mut raw = Vec::new();
        let mut pos = 0;
        for s in sizes {
            raw.extend(
                hdf5rs::filters::deflate::zlib_decompress(&data[pos..pos + s], 1 << 15, 1 << 26)
                    .unwrap(),
            );
            pos += s;
        }
        out.push(if ty == "f64" {
            raw.chunks(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        } else {
            raw.chunks(4)
                .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap())))
                .collect()
        });
    }
    out
}

/// Compares two VTK files; returns the relative difference of the field
/// array (the first one) and asserts the coordinates agree.
fn compare_vtk(want: &Path, got: &Path) -> f64 {
    let (a, b) = (
        std::fs::read_to_string(want).unwrap(),
        std::fs::read_to_string(got).unwrap(),
    );
    let name = want.file_name().unwrap().to_string_lossy();
    assert_eq!(vtk_skeleton(&a), vtk_skeleton(&b), "{name}: VTK structure");
    #[cfg(feature = "hdf5-deflate")]
    {
        let (xa, xb) = (vtk_arrays(&a), vtk_arrays(&b));
        assert_eq!(xa.len(), xb.len(), "{name}: array count");
        for (x, y) in xa.iter().zip(&xb).skip(1) {
            for (p, q) in x.iter().zip(y) {
                assert!(
                    (p - q).abs() <= 1e-6 * p.abs().max(1e-9),
                    "{name}: coordinates"
                );
            }
        }
        let den = xa[0].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let num = xa[0]
            .iter()
            .zip(&xb[0])
            .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
        if den > 0.0 { num / den } else { num }
    }
    #[cfg(not(feature = "hdf5-deflate"))]
    0.0
}

// ---------------------------------------------------------------------
// Cases.

/// The fixture files of `dir` (HDF5 and VTK).
fn fixture_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e == "h5" || e == "vtr" || e == "vts")
        })
        .collect();
    v.sort();
    v
}

/// Compares every fixture file of `case/sub` with the ems output; returns
/// the worst relative difference per file, skipping the numbers of files
/// in `values_skip`.
fn check(
    case: &str,
    sub: &str,
    opts: DumpFileOptions,
    values_skip: &[&str],
) -> BTreeMap<String, f64> {
    let dir = if sub.is_empty() {
        fixture(case)
    } else {
        fixture(case).join(sub)
    };
    let out = run_ems(case, sub, &opts);
    let mut errs = BTreeMap::new();
    for f in fixture_files(&dir) {
        let n = f.file_name().unwrap().to_string_lossy().to_string();
        let g = out.join(&n);
        assert!(g.exists(), "{case}: ems did not write {n}");
        let e = if n.ends_with(".h5") {
            compare_h5(
                &f,
                &g,
                &[
                    "proc_time",
                    "maxSAR",
                    "maxSAR_idx",
                    "valid_cubes",
                    "used_cubes",
                    "unused_cubes",
                ],
                &[],
            )
            .worst()
        } else {
            compare_vtk(&f, &g)
        };
        let e = if values_skip.iter().any(|s| n.starts_with(s)) {
            0.0
        } else {
            e
        };
        errs.insert(n, e);
    }
    let worst = errs.values().fold(0.0f64, |a, &b| a.max(b));
    eprintln!("{case}/{sub}: worst relative difference {worst:e}");
    errs
}

fn assert_close(case: &str, errs: &BTreeMap<String, f64>, tol: f64) {
    for (n, e) in errs {
        assert!(
            *e <= tol,
            "{case}/{n}: relative difference {e:e} > {tol:e}\n{errs:#?}"
        );
    }
}

#[test]
fn time_domain_dumps() {
    let e = check("dump_td", "", DumpFileOptions::new(), &[]);
    assert_eq!(e.len(), 18 + 12);
    assert_close("dump_td", &e, 1e-5);
}

#[test]
fn frequency_domain_dumps() {
    let e = check("dump_fd", "", DumpFileOptions::new(), &[]);
    assert_close("dump_fd", &e, 1e-4);
}

#[test]
fn legacy_hdf5_layout() {
    let e = check(
        "dump_legacy",
        "",
        DumpFileOptions::new().legacy_hdf5(true),
        &[],
    );
    assert_close("dump_legacy", &e, 1e-5);
}

#[test]
fn sampling_rules() {
    let e = check("dump_sampling", "", DumpFileOptions::new(), &[]);
    assert!(!fixture("dump_sampling").join("outside.h5").exists());
    assert_close("dump_sampling", &e, 1e-5);
}

/// A Dirac excitation: openEMS's Nyquist rate is one step whatever
/// `f_max` says, so the time-domain dump records every step (ems wrote
/// every 12th before).
#[test]
fn dirac_dumps() {
    let e = check("dump_dirac", "", DumpFileOptions::new(), &[]);
    assert_close("dump_dirac", &e, 1e-5);
}

#[test]
fn cylindrical_dumps() {
    let e = check("dump_cyl", "", DumpFileOptions::new(), &[]);
    assert_close("dump_cyl", &e, 1e-5);
    let e = check(
        "dump_cyl",
        "native",
        DumpFileOptions::new().native_field_dumps(true),
        &[],
    );
    assert_close("dump_cyl/native", &e, 1e-5);
}

/// Closed azimuth with the axis (`dump_cyl_closed`): openEMS writes the
/// `N` distinct α lines (and dual lines), interpolates node values across
/// the seam, and updates the axis `E_z` with a ring sum that counts the
/// seam line twice; ems reproduces all three for files (design §5.15,
/// §5.16.2): every E and H dump within 2e-6 of its peak (before: a 7th α
/// line, and 1.5e-2…8e-2 near the axis and at the seam).
#[test]
fn closed_cylindrical_dumps() {
    let e = check("dump_cyl_closed", "", DumpFileOptions::new(), &[]);
    assert_close("dump_cyl_closed", &e, 1e-5);
}

/// SAR files: local, raw and 1 g averaged SAR (openEMS's `SIMPLE` cube
/// solver, every cube of this small block clipped by air) match.
#[test]
fn sar_dumps() {
    let e = check("dump_sar", "", DumpFileOptions::new(), &[]);
    assert_close("dump_sar", &e, 1e-5);
}

/// SAR files on a cylindrical grid (`cyl_sar`, gen-cyl.sh): openEMS's
/// cell volume is `ρ_i·Δα·Δρ·Δz` with the radius of the cell's lower ρ
/// line, and it scales the α cell widths by the drawing unit; ems writes
/// the same (its own SAR volumes use the exact annular cells). The cell
/// fields of the two codes differ by about 1e-4 of their peak on this
/// model, which shows as 3.5e-3 of the peak SAR in the weaker tissue
/// field (both codes' local SAR equal `κ|E|²/(2ρ)` of their raw files to
/// 1e-7).
///
/// The 1 g average: openEMS averages the `(ρ, α, z)` cells as Cartesian
/// cells of widths `Δρ`, `Δα·u`, `Δz` weighing with those volumes (its
/// in-run dump equals `sar_calc` on the raw dump, and `sar_calc` on the
/// same file with the axes renamed to `x, y, z`), which ems reproduces
/// with its openEMS averaging on those cells (`SarVolume::with_cell_volumes`,
/// `averaged_openems`): `ems-sar-calc` on openEMS's raw dump gives
/// openEMS's file to float precision; ems's own in-run file differs by the
/// field difference of the two codes on this model (as the local SAR).
#[test]
fn cylindrical_sar_dumps() {
    let e = check("cyl_sar", "", DumpFileOptions::new(), &["sar_1g"]);
    assert_close("cyl_sar", &e, 6e-3);
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("openems_dumps/cyl_sar");
    let want = fixture("cyl_sar").join("sar_1g.h5");
    let d = sar_values(&want, &out.join("sar_1g.h5"));
    eprintln!("cyl_sar 1 g: {d:e} of the peak");
    assert!(d < 6e-3, "{d}");
    // ems-sar-calc on openEMS's raw dump: the same average.
    let o = ems::io::openems::sar_calc::SarCalcOptions {
        input: fixture("cyl_sar").join("sar_raw.h5"),
        output: out.join("sar_calc_1g.h5"),
        mass_g: 1.0,
        method: ems::post::sar::AveragingMethod::Simple,
        threads: 1,
        verbose: false,
        export_cube_stats: false,
        legacy: false,
        autorange: None,
    };
    ems::io::openems::sar_calc::run(&o).unwrap();
    let d = sar_values(&want, &o.output);
    eprintln!("cyl_sar sar_calc 1 g: {d:e} of the peak");
    assert!(d < 1e-6, "{d}");
}

/// Largest difference of the `f0` SAR datasets of two files, relative to
/// the peak of the first.
fn sar_values(want: &Path, got: &Path) -> f64 {
    let (a, b) = (h5cmp::read_tree(want), h5cmp::read_tree(got));
    let p = "/FieldData/FD/f0";
    let (x, y) = (&a.objects[p].2, &b.objects[p].2);
    assert_eq!(x.len(), y.len());
    let m = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    x.iter()
        .zip(y)
        .fold(0.0f64, |d, (u, v)| d.max((u - v).abs()))
        / m
}
