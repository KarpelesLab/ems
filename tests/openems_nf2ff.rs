//! openEMS `nf2ff` tool parity (`ems::io::openems::nf2ff`, `ems-nf2ff`):
//! the fixtures in `tests/fixtures/openems/nf2ff_*` were written by the
//! black-box openEMS (`scripts/openems-oracle/gen-nf2ff.sh`).
//!
//! * `nf2ff_synth`: synthetic near-field planes (FD, legacy FD, TD) and
//!   job files exercising the XML options; ems must write the same file
//!   (structure and values) as the nf2ff tool on the same inputs.
//! * `nf2ff_dipole`: openEMS ran a short dipole with NF2FF box dumps and
//!   transformed them; ems runs the same model, writes the box dumps in
//!   the openEMS layout and transforms them with the same nf2ff XML.
#![cfg(all(feature = "xml", feature = "hdf5"))]

#[path = "common/h5cmp.rs"]
mod h5cmp;

use std::path::{Path, PathBuf};

use ems::Simulation;
use ems::io::openems::dump::{DumpFileOptions, write_dump};
use ems::io::openems::nf2ff::{Nf2ffJob, run_job};
use h5cmp::{compare_h5, read_tree};

fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/openems")
        .join(case)
}

fn tmp_dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("openems_nf2ff")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The job of `xml`, with its file names resolved against `inputs` and
/// the output redirected into `out` (nf2ff resolves them against the
/// working directory, which the tests do not change).
fn job_in(xml: &Path, inputs: &Path, out: &Path) -> Nf2ffJob {
    let mut job = Nf2ffJob::read_file(xml).unwrap();
    for p in &mut job.planes {
        p.e_file = inputs.join(&p.e_file);
        p.h_file = inputs.join(&p.h_file);
    }
    job.outfile = out.join(&job.outfile);
    job
}

#[test]
fn synthetic_planes_match_the_nf2ff_tool() {
    let dir = fixture("nf2ff_synth");
    let out = tmp_dir("synth");
    let mut jobs: Vec<PathBuf> = std::fs::read_dir(dir.join("jobs"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    jobs.sort();
    assert_eq!(jobs.len(), 4);
    for xml in jobs {
        let job = job_in(&xml, &dir, &out);
        run_job(&job).unwrap();
        let name = job.outfile.file_name().unwrap().to_owned();
        let d = compare_h5(&dir.join(&name), &job.outfile, &[], &[]);
        eprintln!("{}: {d:?}", name.to_string_lossy());
        // openEMS computes in single precision somewhere: ~1e-7 typical.
        assert!(d.worst() < 1e-5, "{}: {d:?}", name.to_string_lossy());
    }
}

#[test]
fn dipole_box_dumps_transform_like_openems() {
    let dir = fixture("nf2ff_dipole");
    let out = tmp_dir("dipole");
    let mut sim = Simulation::from_xml(dir.join("model.xml")).unwrap();
    let res = sim.run().unwrap();
    for d in res.dumps() {
        write_dump(&out, d, &DumpFileOptions::new()).unwrap();
    }
    let job = job_in(&dir.join("nf2ff.xml"), &out, &out);
    let ff = run_job(&job).unwrap();
    // Same structure; values compared below (E_φ of a z dipole is ~0, so
    // it is measured against the E_θ scale).
    compare_h5(
        &dir.join("farfield.h5"),
        &job.outfile,
        &["Prad", "Dmax"],
        &["nf2ff"],
    );
    let (a, b) = (read_tree(&dir.join("farfield.h5")), read_tree(&job.outfile));
    let vals = |t: &h5cmp::H5Tree, p: &str| t.objects[p].2.clone();
    let mut worst: f64 = 0.0;
    for k in 0..2 {
        let scale = vals(&a, &format!("/nf2ff/E_theta/FD/f{k}"))
            .iter()
            .fold(0.0f64, |m, x| m.max(x.abs()));
        for q in ["E_theta", "E_phi"] {
            let p = format!("/nf2ff/{q}/FD/f{k}");
            for (x, y) in vals(&a, &p).iter().zip(vals(&b, &p)) {
                worst = worst.max((x - y).abs() / scale);
            }
        }
        let p = format!("/nf2ff/P_rad/FD/f{k}");
        let pa = vals(&a, &p);
        let pmax = pa.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for (x, y) in pa.iter().zip(vals(&b, &p)) {
            worst = worst.max((x - y).abs() / pmax);
        }
    }
    eprintln!("dipole: far field differs by {worst:e}, Dmax {:?}", ff.dmax);
    // The near fields come from two solvers (ems vs openEMS, ~5e-4 apart
    // on the box faces); the radiated far field is far less sensitive.
    assert!(worst < 1e-4, "{worst}");
    // The box flux of the strongly reactive near field of a small box is
    // (1 % here), so Prad and Dmax are compared loosely.
    let want_dmax = h5cmp::numbers(&a.objects["/nf2ff"].1["Dmax"].1);
    for (got, want) in ff.dmax.iter().zip(&want_dmax) {
        assert!((got / want - 1.0).abs() < 0.02, "Dmax {got} vs {want}");
        // A short dipole: directivity 1.5.
        assert!((got - 1.5).abs() < 0.02, "{got}");
    }
}

/// `ems-nf2ff <xml>`, file names relative to the working directory (a
/// copy of the inputs: the job writes its output next to them).
#[test]
fn command_line() {
    let exe = env!("CARGO_BIN_EXE_ems-nf2ff");
    let dir = fixture("nf2ff_synth");
    let work = tmp_dir("cli");
    std::fs::create_dir_all(work.join("jobs")).unwrap();
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.ends_with(".h5") && !name.starts_with("out_") {
            std::fs::copy(&p, work.join(&name)).unwrap();
        }
    }
    std::fs::copy(dir.join("jobs/mirrors.xml"), work.join("jobs/mirrors.xml")).unwrap();
    let st = std::process::Command::new(exe)
        .arg("jobs/mirrors.xml")
        .current_dir(&work)
        .output()
        .unwrap();
    assert!(st.status.success(), "{st:?}");
    let d = compare_h5(
        &dir.join("out_mirrors.h5"),
        &work.join("out_mirrors.h5"),
        &[],
        &[],
    );
    assert!(d.worst() < 1e-5, "{d:?}");
    for args in [vec![], vec!["missing.xml"]] {
        let st = std::process::Command::new(exe)
            .args(&args)
            .current_dir(&work)
            .output()
            .unwrap();
        assert!(!st.status.success(), "{args:?}");
    }
}
