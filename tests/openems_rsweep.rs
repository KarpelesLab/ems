//! Models of the randomized differential sweep against the black-box
//! openEMS (fixtures `tests/fixtures/openems/rs_*`, written by
//! `scripts/openems-oracle/gen-rsweep.sh`, never by the test suite;
//! design §5.16.3). Each is a small random model (graded mesh, mixed
//! PEC/PMC/Mur/PML faces, materials, dispersive media, metal, conducting
//! sheets, lumped elements, E/H sources, probes and raw line dumps) that
//! exposed an openEMS behaviour ems now follows for files; the case's
//! `model.xml` comment names it. The `ems` binary runs each model as
//! `openEMS` would; probe series, steps run and the dumps are compared.
#![cfg(feature = "xml")]

#[cfg(feature = "hdf5")]
#[path = "common/h5cmp.rs"]
mod h5cmp;

use std::path::{Path, PathBuf};
use std::process::Command;

use ems::io::openems::{ProbeDomain, ProbeFile, read_probe_file};

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
        .join("openems_rsweep")
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

/// The raw HDF5 dumps of `case` against openEMS's: the largest
/// difference over the whole series below `tol` of the largest value.
#[cfg(feature = "hdf5")]
fn check_dumps(case: &str, dir: &Path, tol: f64) {
    for f in meta(case, "keep")
        .split_whitespace()
        .filter(|f| f.ends_with(".h5"))
    {
        let (want, got) = (fixture(case).join(f), dir.join(f));
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
        assert!(num < tol * den, "{case}/{f}: {num:e} of {den:e}");
    }
}

/// Runs a case: probes to `tol` (relative L2), dumps to `dump_tol`.
fn case(case: &str, tol: f64, dump_tol: f64) {
    let (_dir, _) = check(case, tol);
    #[cfg(feature = "hdf5")]
    check_dumps(case, &_dir, dump_tol);
    let _ = dump_tol;
}

/// A Drude box reaching into PML layers: openEMS's layer replaces the
/// update of its edges and the poles have no effect there (the static
/// `ε` remains); the raw H line dump snaps to the cells of a graded mesh.
/// 1e-6…5e-6 (was 1.2e-1 and 3.4e-1).
#[test]
fn drude_in_pml() {
    case("rs_0", 2e-5, 3e-5);
}

/// A Lorentz box touching a Mur face: openEMS's Mur face reads its
/// neighbours before the dispersive correction. 1e-6…2e-5 (was 0.03…0.4).
#[test]
fn lorentz_at_mur() {
    case("rs_1", 1e-4, 1e-4);
}

/// A metal sheet under a higher-priority dielectric (the dielectric wins
/// the sheet's edges). 1.4e-6…6.5e-6 (was up to 0.97).
#[test]
fn sheet_under_material() {
    case("rs_3", 3e-5, 1e-4);
}

/// A domain edge of two Mur faces (only the lower axis's face updates
/// it), with a raw H dump at the corner: 2.5e-7…8.7e-6 (dump was 0.64).
#[test]
fn mur_domain_edge() {
    case("rs_11", 5e-5, 1e-5);
}

/// A hard E source one cell from a Mur face (the source edge reads as 0
/// for the face). 1.4e-6…3.1e-5 (was ~1).
#[test]
fn hard_source_at_mur() {
    case("rs_12", 1e-4, 1e-5);
}

/// A soft H source: the look-ahead H of the last dump step includes it.
#[test]
fn h_source_lookahead() {
    case("rs_18", 1e-4, 1e-5);
}

/// A metal sheet and a dielectric of equal priority (the one defined
/// first wins); raw H dump. 1.2e-6…6.3e-5 (was up to 0.4).
#[test]
fn sheet_material_tie() {
    case("rs_29", 1e-4, 2e-4);
}

/// Conducting-sheet border edges on a graded mesh (half the dual width).
/// Exact (openEMS's sheet arithmetic, design §5.16.13; was 3.5e-6…2.9e-4,
/// and 74 before openEMS's sheet model).
#[test]
fn sheet_borders() {
    case("rs_99", 1e-6, 1e-4);
}

/// A passive series RLC over several columns: openEMS limits the edge
/// capacitance by the branch impedance. 5e-6…3.3e-4 (weak E probes;
/// was 0.3).
#[test]
fn series_rlc_columns() {
    case("rs_109", 1e-3, 1e-4);
}

/// A conducting sheet starting on the second line of a Mur face's axis:
/// openEMS leaves out the lines the boundary claims. Exact (openEMS's
/// sheet arithmetic, design §5.16.13; was 3e-7…6e-5, unstable before).
#[test]
fn sheet_at_mur_lines() {
    case("rs_116", 1e-6, 1e-4);
}

/// A lumped capacitor with caps over a metal sheet of lower priority, an
/// H source inside it: the element wins its edges over the metal (at any
/// priority) and its caps span its whole box. Exact (was 1.2…2.0).
#[test]
fn lumped_over_metal() {
    case("rs_157", 1e-6, 1e-4);
}

/// A soft E source on a Mur face: openEMS holds the face (no Mur update)
/// through E sample `L + 10` of an excitation of `L` steps. Exact (was
/// 5e-4…2.9e-2 with the face held one sample longer).
#[test]
fn mur_face_hold() {
    case("rs_162", 1e-6, 1e-4);
}

/// An R‖L element in the last cell before a PMC face (cut to 320 steps:
/// openEMS's capacitance limit makes it grow unstably later, in both
/// codes): openEMS keeps the H edge of the element's direction and node
/// in the PMC layer it otherwise zeroes. 1e-6…4e-6 (was 4.7e-4…1.8e-2).
#[test]
fn lumped_at_pmc() {
    case("rs_55", 2e-5, 2e-5);
}

/// A conducting sheet touching a dielectric of equal priority defined
/// before it: the dielectric wins the sheet edges at its midpoints (ties
/// go to the property defined first); a Drude box, a lumped port.
/// 4.6e-7…2.1e-5 (was 6.1e-3…9.8e-3).
#[test]
fn sheet_material_tie_order() {
    case("rs_152", 1e-4, 1e-4);
}

/// A conducting sheet of higher priority through a metal box: its edges
/// in the metal are sheet edges, not PEC. Exact (dumps were 1.3e-3).
#[test]
fn sheet_over_metal() {
    case("rs_63", 1e-6, 1e-5);
}

/// A conducting sheet parallel to a PML face, inside the layer: openEMS
/// makes it PEC. Exact (was 6e-5…6.3e-2).
#[test]
fn sheet_in_layer() {
    case("rs_77", 1e-6, 1e-4);
}

/// Lumped resistors in Lorentz media: openEMS forms the pole
/// coefficients from the edge with the element folded in. Exact (was
/// 2.6e-5…2.4e-4, the dumps 7.2e-4).
#[test]
fn lumped_in_lorentz() {
    case("rs_51", 1e-6, 1e-5);
}

/// A lumped capacitor in a Debye medium (hand-written): the same for
/// openEMS's Debye update. Exact (was 5.0e-2…5.9e-1).
#[test]
fn lumped_in_debye() {
    case("rs_51_debye", 1e-6, 1e-5);
}
