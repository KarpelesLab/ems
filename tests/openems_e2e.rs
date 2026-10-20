//! End-to-end parity with openEMS on realistic models that exercise many
//! features together (fixtures `tests/fixtures/openems/e2e_*`, written by
//! the black-box openEMS with `scripts/openems-oracle/gen-e2e.sh`, never by
//! the test suite):
//!
//! * `e2e_patch`: probe-fed 2.4 GHz patch antenna on lossy FR4, lumped
//!   50 Ω port, PML, NF2FF box (far field through the `nf2ff` job);
//! * `e2e_msl_filter`: microstrip open-stub low-pass filter, soft feed,
//!   three-voltage / two-current probe ports, Mur on every open face
//!   (point-sampled media on layered faces, corners), FD field dump;
//! * `e2e_waveguide_iris`: WR-90 guide with an inductive iris and a
//!   dielectric post, weighted TE10 excitation, mode-matching probes, Mur
//!   ends, energy end criterion;
//! * `e2e_coax_monopole`: quarter-wave monopole fed by a PTFE coax on a
//!   cylindrical grid (closed azimuth, axis inside metal), 1/ρ weighted
//!   excitation, Mur faces;
//! * `e2e_drude_slab`: plane wave on a Drude plasma slab backed by a lossy
//!   Lorentz layer, E/H probes with FD samples, FD field dump.
//!
//! The `ems` binary runs each model as `openEMS` would (same files in the
//! working directory) and its outputs are compared with openEMS's: the
//! probe series and spectra, the field dumps, and derived engineering
//! quantities (input impedance and resonance, transfer functions,
//! reflection, far field). Tolerances are a few times the measured
//! agreement, given in each test's doc comment; the documented formulation
//! differences (design §5.16: PML, plane-wave injection, cylindrical α
//! current loops) set them where they apply. The patch and filter models
//! fix openEMS's time step (ems's Rennings rule is 0.5–2 % off on these
//! layered, doubly graded meshes, design §5.16.1).
#![cfg(feature = "xml")]

#[cfg(feature = "hdf5")]
#[path = "common/h5cmp.rs"]
mod h5cmp;

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::process::Command;

use ems::Complex64;
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

/// Runs the `ems` binary on the case's model in a fresh scratch directory
/// and returns the directory and the console output. Without the `hdf5`
/// feature the (HDF5) dumps are disabled.
fn run(case: &str) -> (PathBuf, String) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("openems_e2e")
        .join(case);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(fixture(case).join("model.xml"), dir.join("model.xml")).unwrap();
    let mut args = vec!["model.xml".to_string(), "--numThreads=4".to_string()];
    args.extend(meta(case, "args").split_whitespace().map(str::to_string));
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

/// The run made as many steps as openEMS's (`steps:` of `meta.txt`;
/// `Time for N iterations` on the console).
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

fn oe(case: &str, name: &str) -> ProbeFile {
    read_probe_file(fixture(case).join(name)).unwrap()
}

fn got(dir: &Path, name: &str) -> ProbeFile {
    read_probe_file(dir.join(name)).unwrap()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The value columns of probe file `name` of both codes over the rows
/// both have (ems adds no look-ahead row on cylindrical grids), after
/// checking kinds and sample times (or frequencies). Mode-matching files
/// without their purity column.
fn columns(case: &str, dir: &Path, name: &str) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let (a, b) = (oe(case, name), got(dir, name));
    assert_eq!((a.domain, a.kind), (b.domain, b.kind), "{case}/{name}");
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
    (
        (0..cols).map(|c| col(&a, c)).collect(),
        (0..cols).map(|c| col(&b, c)).collect(),
    )
}

/// Relative L2 difference of probe file `name` (all columns, measured
/// against the norm of the largest column: the small transverse
/// components of a guided field are round-off), asserted below `tol`.
fn compare_probe(case: &str, dir: &Path, name: &str, tol: f64) -> f64 {
    let (a, b) = columns(case, dir, name);
    let scale = a.iter().map(|c| norm(c)).fold(0.0, f64::max);
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| {
            let d: Vec<f64> = x.iter().zip(y).map(|(p, q)| p - q).collect();
            norm(&d) / scale.max(1e-300)
        })
        .fold(0.0, f64::max);
    eprintln!(
        "{case}/{name}: {} rows, relative L2 {worst:.2e}",
        a[0].len()
    );
    assert!(
        worst < tol,
        "{case}/{name}: relative L2 {worst:e} > {tol:e}"
    );
    worst
}

/// The DFT `Σ x(t_k)·e^{−j2πf·t_k}·Δt` of a time series at `f`.
fn dft(t: &[f64], x: &[f64], f: f64) -> Complex64 {
    let dt = t[1] - t[0];
    let (mut re, mut im) = (0.0, 0.0);
    for (&tk, &xk) in t.iter().zip(x) {
        let w = -2.0 * PI * f * tk;
        re += xk * w.cos() * dt;
        im += xk * w.sin() * dt;
    }
    Complex64::new(re, im)
}

/// The spectrum of value column 0 of a time-domain probe file at `freqs`,
/// over its first `n` rows.
fn spectrum(p: &ProbeFile, n: usize, freqs: &[f64]) -> Vec<Complex64> {
    let (t, x) = (p.abscissa(), p.column(0));
    freqs.iter().map(|&f| dft(&t[..n], &x[..n], f)).collect()
}

/// Pointwise ratio of two spectra.
fn ratio(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    a.iter().zip(b).map(|(x, y)| *x / *y).collect()
}

/// Largest relative difference of two complex curves.
fn worst_rel(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).norm() / x.norm())
        .fold(0.0, f64::max)
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| a + (b - a) * k as f64 / (n - 1) as f64)
        .collect()
}

fn db(z: Complex64) -> f64 {
    20.0 * z.norm().log10()
}

/// The common row count of probe files `names` in both codes.
fn rows(case: &str, dir: &Path, names: &[&str]) -> usize {
    names
        .iter()
        .map(|n| oe(case, n).rows.len().min(got(dir, n).rows.len()))
        .min()
        .unwrap()
}

/// An HDF5 dump or far-field file of both codes: same structure, data
/// within `tol` of the largest value of each dataset.
#[cfg(feature = "hdf5")]
fn compare_h5(case: &str, dir: &Path, name: &str, loose: &[&str], tol: f64) {
    let d = h5cmp::compare_h5(&fixture(case).join(name), &dir.join(name), loose, &[]);
    eprintln!("{case}/{name}: {d:?}");
    assert!(d.data < tol, "{case}/{name}: {d:?}");
}

// ---------------------------------------------------------------- patch

/// Patch antenna. Measured: port and edge probes 6e-5…2e-4 (PML, design
/// §5.16.2), the H probe 6e-4; input impedance 2.0–2.8 GHz within 9e-4
/// and the same resonance (|S11| minimum against 50 Ω: 2.29 GHz,
/// −15.7 dB, on a 2.5 MHz grid); far field from the box dumps 1e-3,
/// `Dmax` 2.4e-4, `Prad` 6.4e-4.
#[test]
fn patch_antenna() {
    let case = "e2e_patch";
    let (dir, console) = run(case);
    check_steps(case, &console);
    for p in ["port_ut", "port_it", "ut_edge", "ut_edge_fd"] {
        compare_probe(case, &dir, p, 2e-3);
    }
    compare_probe(case, &dir, "h_above", 5e-3);
    compare_probe(case, &dir, "ut_edge_fd_FD", 2e-3);
    // Input impedance (the probe's current flows out of the patch) and
    // resonance of the coarse patch.
    let freqs = linspace(2.0e9, 2.8e9, 321);
    let n = rows(case, &dir, &["port_ut", "port_it"]);
    let z = |u: ProbeFile, i: ProbeFile| -> Vec<Complex64> {
        ratio(&spectrum(&u, n, &freqs), &spectrum(&i, n, &freqs))
            .into_iter()
            .map(|z| Complex64::new(-z.re, -z.im))
            .collect()
    };
    let za = z(oe(case, "port_ut"), oe(case, "port_it"));
    let zb = z(got(&dir, "port_ut"), got(&dir, "port_it"));
    let s11 = |z: &[Complex64]| -> Vec<f64> {
        let z0 = Complex64::new(50.0, 0.0);
        z.iter().map(|&z| db((z - z0) / (z + z0))).collect()
    };
    let (sa, sb) = (s11(&za), s11(&zb));
    let argmin = |v: &[f64]| (0..v.len()).min_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap();
    let (ia, ib) = (argmin(&sa), argmin(&sb));
    eprintln!(
        "{case}: resonance {:.4} GHz ({:.2} dB) vs {:.4} GHz ({:.2} dB)",
        freqs[ia] / 1e9,
        sa[ia],
        freqs[ib] / 1e9,
        sb[ib]
    );
    assert!(
        sa[ia] < -10.0,
        "{case}: the patch must resonate in the band"
    );
    assert_eq!(ia, ib, "{case}: resonance frequency");
    let w = worst_rel(&za, &zb);
    eprintln!("{case}: input impedance differs by {w:.2e}");
    assert!(w < 5e-3, "{case}: Zin {w:e}");
    #[cfg(feature = "hdf5")]
    far_field(case, &dir);
}

/// The case's nf2ff job on ems's box dumps against openEMS's
/// `farfield.h5`.
#[cfg(feature = "hdf5")]
fn far_field(case: &str, dir: &Path) {
    use ems::io::openems::nf2ff::{Nf2ffJob, run_job};
    let mut job = Nf2ffJob::read_file(fixture(case).join("nf2ff.xml")).unwrap();
    for p in &mut job.planes {
        p.e_file = dir.join(&p.e_file);
        p.h_file = dir.join(&p.h_file);
    }
    job.outfile = dir.join(&job.outfile);
    let ff = run_job(&job).unwrap();
    compare_h5(case, dir, "farfield.h5", &["Prad", "Dmax"], 5e-3);
    let want = h5cmp::read_tree(&fixture(case).join("farfield.h5"));
    let attr = |k: &str| h5cmp::numbers(&want.objects["/nf2ff"].1[k].1);
    for (got, want) in ff.dmax.iter().zip(attr("Dmax")) {
        assert!(
            (got / want - 1.0).abs() < 2e-3,
            "{case}: Dmax {got} vs {want}"
        );
    }
    for (got, want) in ff.prad.iter().zip(attr("Prad")) {
        assert!(
            (got / want - 1.0).abs() < 5e-3,
            "{case}: Prad {got} vs {want}"
        );
    }
}

// ----------------------------------------------------------- msl filter

/// Microstrip stub filter with Mur faces. Measured: port probes 2e-7…5e-7,
/// the E probe by the stub 1e-5, the FD dump in the substrate 1.3e-3 (of
/// its largest value; the small components near the strip); transfer
/// function `U2/U1` 2–9 GHz within 8e-6 dB.
#[test]
fn msl_stub_filter() {
    let case = "e2e_msl_filter";
    let (dir, console) = run(case);
    check_steps(case, &console);
    for p in [
        "port_ut1A",
        "port_ut1B",
        "port_ut1C",
        "port_it1A",
        "port_it1B",
        "port_ut2A",
        "port_ut2B",
        "port_ut2C",
        "port_it2A",
        "port_it2B",
    ] {
        compare_probe(case, &dir, p, 5e-6);
    }
    compare_probe(case, &dir, "ez_stub", 5e-4);
    compare_probe(case, &dir, "ez_stub_FD", 5e-4);
    let freqs = linspace(2e9, 9e9, 141);
    let n = rows(case, &dir, &["port_ut1B", "port_ut2B"]);
    let h = |f: &dyn Fn(&str) -> ProbeFile| {
        ratio(
            &spectrum(&f("port_ut2B"), n, &freqs),
            &spectrum(&f("port_ut1B"), n, &freqs),
        )
    };
    let ha = h(&|p| oe(case, p));
    let hb = h(&|p| got(&dir, p));
    let w = ha
        .iter()
        .zip(&hb)
        .map(|(a, b)| (db(*a) - db(*b)).abs())
        .fold(0.0, f64::max);
    eprintln!("{case}: |U2/U1| differs by {w:.2e} dB");
    assert!(w < 1e-4, "{case}: transfer function {w} dB");
    #[cfg(feature = "hdf5")]
    compare_h5(case, &dir, "Ef_sub.h5", &[], 5e-3);
}

// ------------------------------------------------------- waveguide iris

/// WR-90 iris with Mur ends. Measured: mode probes 4e-7 (purity 1e-6
/// absolute where the mode carries a signal), the E probe 7.5e-7, the end
/// criterion stops at openEMS's step; `U2/U1` 8–12 GHz within 7e-7.
#[test]
fn waveguide_iris() {
    let case = "e2e_waveguide_iris";
    let (dir, console) = run(case);
    check_steps(case, &console);
    for p in ["port_ut1", "port_it1", "port_ut2", "port_it2", "ey_mid"] {
        compare_probe(case, &dir, p, 5e-6);
    }
    for p in ["port_ut1", "port_it1", "port_ut2", "port_it2"] {
        let (a, b) = (oe(case, p), got(&dir, p));
        let (pa, pb) = (a.mode_purity().unwrap(), b.mode_purity().unwrap());
        // Where the mode carries a signal (the purity of round-off noise
        // is ill-conditioned).
        let v = a.column(0);
        let vmax = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let w = pa
            .iter()
            .zip(&pb)
            .zip(&v)
            .filter(|(_, x)| x.abs() > 1e-2 * vmax)
            .map(|((x, y), _)| (x - y).abs())
            .fold(0.0, f64::max);
        eprintln!("{case}/{p}: mode purity differs by {w:.2e}");
        assert!(w < 1e-4, "{case}/{p}: mode purity differs by {w}");
    }
    let freqs = linspace(8e9, 12e9, 81);
    let n = rows(case, &dir, &["port_ut1", "port_ut2"]);
    let h = |f: &dyn Fn(&str) -> ProbeFile| {
        ratio(
            &spectrum(&f("port_ut2"), n, &freqs),
            &spectrum(&f("port_ut1"), n, &freqs),
        )
    };
    let w = worst_rel(&h(&|p| oe(case, p)), &h(&|p| got(&dir, p)));
    eprintln!("{case}: U2/U1 differs by {w:.2e}");
    assert!(w < 1e-5, "{case}: transfer function {w:e}");
}

// -------------------------------------------------------- coax monopole

/// Coax-fed monopole, cylindrical grid. Measured: coax voltage 1.7e-4,
/// far E field 2.2e-4, near H field 2.1e-4 (the axis region above the
/// monopole, design §5.15); the coax current reads 3/4 of ems's in
/// openEMS, whose loop over a full-azimuth box covers 3 of the 4 α cells
/// (design §5.16.2), 1.8e-4 after that factor; input impedance within
/// 7e-4 with it.
#[test]
fn coax_monopole() {
    let case = "e2e_coax_monopole";
    let (dir, console) = run(case);
    check_steps(case, &console);
    compare_probe(case, &dir, "port_ut", 2e-3);
    compare_probe(case, &dir, "ez_far", 2e-3);
    compare_probe(case, &dir, "ez_far_FD", 2e-3);
    compare_probe(case, &dir, "h_near", 2e-3);
    // openEMS's (N − 1)/N current over a closed azimuth of N = 4 cells.
    let (a, b) = columns(case, &dir, "port_it");
    let b: Vec<f64> = b[0].iter().map(|x| 0.75 * x).collect();
    let d: Vec<f64> = a[0].iter().zip(&b).map(|(x, y)| x - y).collect();
    let e = norm(&d) / norm(&a[0]);
    eprintln!("{case}/port_it: relative L2 {e:.2e} after the 3/4 factor");
    assert!(e < 2e-3, "{case}/port_it: {e:e}");
    let freqs = linspace(1.5e9, 3.5e9, 41);
    let n = rows(case, &dir, &["port_ut", "port_it"]);
    let za = ratio(
        &spectrum(&oe(case, "port_ut"), n, &freqs),
        &spectrum(&oe(case, "port_it"), n, &freqs),
    );
    let zb: Vec<Complex64> = ratio(
        &spectrum(&got(&dir, "port_ut"), n, &freqs),
        &spectrum(&got(&dir, "port_it"), n, &freqs),
    )
    .into_iter()
    .map(|z| z / 0.75)
    .collect();
    let w = worst_rel(&za, &zb);
    eprintln!("{case}: input impedance differs by {w:.2e}");
    assert!(w < 5e-3, "{case}: Zin {w:e}");
}

// ----------------------------------------------------------- drude slab

/// Drude/Lorentz slab under a plane wave. Measured: E/H probes 8.5e-4…
/// 9.8e-4 and their FD samples 7e-4 (openEMS's own plane-wave injection
/// error, design §5.16.2), the FD dump along the line 2.4e-3; the
/// reflection `E_refl/E_inc` and transmission `E_trans/E_inc` 3–8 GHz
/// within 9e-4 and 6e-4.
#[test]
fn drude_slab_plane_wave() {
    let case = "e2e_drude_slab";
    let (dir, console) = run(case);
    check_steps(case, &console);
    for p in [
        "ex_inc",
        "ex_refl",
        "ex_trans",
        "hy_trans",
        "ux_in_slab",
        "ex_refl_FD",
        "ex_trans_FD",
    ] {
        compare_probe(case, &dir, p, 3e-3);
    }
    let freqs = linspace(3e9, 8e9, 51);
    let n = rows(case, &dir, &["ex_inc", "ex_refl", "ex_trans"]);
    for out in ["ex_refl", "ex_trans"] {
        let r = |f: &dyn Fn(&str) -> ProbeFile| {
            ratio(
                &spectrum(&f(out), n, &freqs),
                &spectrum(&f("ex_inc"), n, &freqs),
            )
        };
        let w = worst_rel(&r(&|p| oe(case, p)), &r(&|p| got(&dir, p)));
        eprintln!("{case}: {out}/ex_inc differs by {w:.2e}");
        assert!(w < 5e-3, "{case}: {out} {w:e}");
    }
    #[cfg(feature = "hdf5")]
    compare_h5(case, &dir, "E_fd.h5", &[], 1e-2);
}
