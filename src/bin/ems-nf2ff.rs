//! `ems-nf2ff <nf2ff-xml-file>`: the near-field to far-field transform of
//! openEMS's `nf2ff` tool (same XML file, same input dumps, same output
//! HDF5 layout); see `ems::io::openems::nf2ff`.

use std::process::ExitCode;

use ems::io::openems::nf2ff::{Nf2ffJob, far_field, read_plane, write_far_field};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 || args[0] == "-h" || args[0] == "--help" {
        eprintln!("Usage: ems-nf2ff <nf2ff-xml-file>");
        return ExitCode::FAILURE;
    }
    match run(&args[0]) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ems-nf2ff: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(path: &str) -> ems::Result<()> {
    let job = Nf2ffJob::read_file(path)?;
    if job.verbose > 0 {
        eprintln!("ems-nf2ff: verbose level {}", job.verbose);
    }
    let mut planes = Vec::with_capacity(job.planes.len());
    for p in &job.planes {
        if job.verbose > 0 {
            eprintln!(
                "ems-nf2ff: reading planes {} & {}",
                p.e_file.display(),
                p.h_file.display()
            );
        }
        planes.push(read_plane(p, &job.frequencies)?);
    }
    let ff = far_field(&job, &planes)?;
    if job.verbose > 0 {
        for (k, f) in job.frequencies.iter().enumerate() {
            eprintln!(
                "ems-nf2ff: f = {f:e} Hz ({}/{}): Prad = {:e} W, Dmax = {}",
                k + 1,
                job.frequencies.len(),
                ff.prad[k],
                ff.dmax[k]
            );
        }
    }
    write_far_field(&job.outfile, &ff)
}
