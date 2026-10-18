//! Runs an openEMS XML model file and writes its probe and dump results.
//!
//! ```text
//! cargo run --release --example run_xml -- model.xml [--threads N] [--out DIR] [--conformal]
//! ```
//!
//! `--conformal` resolves solid metal below the cell size (conformal PEC,
//! `ems::CompileOptions::conformal`) instead of the staircase mask.
//!
//! Prints the compile warnings, progress (energy decay, throughput) and how
//! the run ended. Every probe is written as `DIR/<probe>.csv` (time series,
//! `t` plus the value or the `x`, `y`, `z` field components) and every dump
//! as VTK `.vtr` files with a `.pvd` index (`DIR/<dump>_*.vtr`). The
//! terminal current of every lumped element is written as
//! `DIR/<element>.csv` (`<element>_current.csv` if a probe has the same
//! name; column `i`, along the element's direction; for a
//! lumped port's `port_resist_<n>` the port current into the structure is
//! `−i`). Use that, not the H-loop probe `port_it<n>`, as the port current
//! when analysing ports by hand: it is defined at the same edges as the
//! voltage `port_ut<n>` (second-order reference plane, see
//! `ems::port::LumpedPort`). `DIR` defaults to `<model>_results` next to
//! the model file.
//!
//! With the `hdf5` feature (`--features hdf5`), dumps whose DumpBox asks for
//! HDF5 (`FileType="1"`) are written as `DIR/<dump>.h5` instead (ems layout,
//! see `ems::io::h5`), and all probes also go to `DIR/probes.h5`. Without
//! it, such dumps fall back to `.vtr` with a warning.
//!
//! SAR dumps (`SARLocal`, `SAR1g`, `SAR10g`, `SARRaw`) are written per
//! frequency as `DIR/<dump>_f<k>.vtr` (cell data `SAR`, `class`), or `.h5`
//! for HDF5 dumps with the feature. Their values are spectral densities of
//! the transient run (see `ems::post::sar`): normalise them by the accepted
//! port power for SAR per watt.

use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use ems::io::CsvWriter;
use ems::{CompileOptions, ProbeKind, ProbeValues, Simulation};

struct Args {
    model: PathBuf,
    threads: Option<usize>,
    out: Option<PathBuf>,
    conformal: bool,
}

fn usage() -> String {
    "usage: run_xml MODEL.xml [--threads N] [--out DIR] [--conformal]".into()
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let (mut model, mut threads, mut out) = (None, None, None);
    let mut conformal = false;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--threads" | "-j" => {
                let v = it.next().ok_or_else(usage)?;
                let n: usize = v.parse().map_err(|_| format!("bad thread count {v:?}"))?;
                if n == 0 {
                    return Err("thread count must be at least 1".into());
                }
                threads = Some(n);
            }
            "--out" | "-o" => out = Some(PathBuf::from(it.next().ok_or_else(usage)?)),
            "--conformal" => conformal = true,
            "--help" | "-h" => return Err(usage()),
            _ if a.starts_with('-') => return Err(format!("unknown option {a:?}\n{}", usage())),
            _ if model.is_none() => model = Some(PathBuf::from(a)),
            _ => return Err(usage()),
        }
    }
    Ok(Args {
        model: model.ok_or_else(usage)?,
        threads,
        out,
        conformal,
    })
}

/// A file name made of the safe characters of `name`.
fn file_stem(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-_.".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() { "unnamed".into() } else { s }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let doc = csx::OpenEmsDocument::read_file(&args.model)?;
    let threads = args.threads;
    let opts = CompileOptions::default().with_conformal(args.conformal);
    let mut sim = Simulation::from_document_with(&doc, &opts, |f| match threads {
        Some(n) => f.threads(n),
        None => f,
    })?;
    let g = sim.grid();
    let n = g.nodes();
    println!(
        "{}: {} x {} x {} nodes ({:.3} Mcells), dt = {:.4e} s, {} threads, compiled in {:.2} s",
        args.model.display(),
        n[0],
        n[1],
        n[2],
        g.num_cells() as f64 / 1e6,
        sim.engine().dt(),
        sim.engine().threads(),
        t0.elapsed().as_secs_f64()
    );
    for w in sim.warnings() {
        println!("  ! {w}");
    }
    let limits = sim.run_limits();
    println!(
        "running up to {} steps{}",
        limits.max_timesteps,
        match limits.end_criteria {
            Some(c) => format!(", until the energy decays by {:.1} dB", c.decay_db()),
            None => String::new(),
        }
    );

    let mut last = Instant::now();
    let res = sim.run_with(|p| {
        if last.elapsed() >= Duration::from_secs(1) {
            last = Instant::now();
            println!(
                "  step {:>9}  t = {:.4e} s  energy {:7.2} dB  {:8.1} Mcells/s",
                p.timestep, p.time, p.energy_db, p.mcells_per_s
            );
        }
        ControlFlow::Continue(())
    })?;
    let o = res.outcome().expect("run sets the outcome");
    let secs = o.elapsed.as_secs_f64();
    println!(
        "done: {:?} after {} steps (t = {:.4e} s) in {:.2} s, {:.1} Mcells/s",
        o.reason,
        o.steps,
        o.time,
        secs,
        sim.grid().num_cells() as f64 * o.steps as f64 / secs.max(1e-9) / 1e6
    );

    let out = args.out.unwrap_or_else(|| {
        let stem = args
            .model
            .file_stem()
            .map_or("model".into(), |s| s.to_string_lossy().into_owned());
        args.model.with_file_name(format!("{stem}_results"))
    });
    std::fs::create_dir_all(&out)?;
    for p in res.probes() {
        let path = out.join(format!("{}.csv", file_stem(p.name())));
        let comment = format!("probe {:?} ({:?}), dt = {:e} s", p.name(), p.kind(), p.dt());
        let w = CsvWriter::new().comment(comment).column("t", p.times());
        match p.values() {
            ProbeValues::Scalar(v) => {
                let name = if p.kind() == ProbeKind::Current {
                    "i"
                } else {
                    "u"
                };
                w.column(name, v).write(&path)?;
            }
            ProbeValues::Vector(v) => {
                let c: Vec<Vec<f64>> = (0..3).map(|k| v.iter().map(|x| x[k]).collect()).collect();
                w.column("x", &c[0])
                    .column("y", &c[1])
                    .column("z", &c[2])
                    .write(&path)?;
            }
        }
        println!("wrote {}", path.display());
    }
    let probe_files: std::collections::HashSet<String> =
        res.probes().iter().map(|p| file_stem(p.name())).collect();
    for p in res.lumped_currents() {
        let mut stem = file_stem(p.name());
        if probe_files.contains(&stem) {
            stem.push_str("_current"); // a probe already uses the name
        }
        let path = out.join(format!("{stem}.csv"));
        let comment = format!(
            "terminal current of lumped element {:?} along its direction, dt = {:e} s \
             (port current into the structure = -i)",
            p.name(),
            p.dt()
        );
        if let ProbeValues::Scalar(v) = p.values() {
            CsvWriter::new()
                .comment(comment)
                .column("t", p.times())
                .column("i", v)
                .write(&path)?;
            println!("wrote {}", path.display());
        }
    }
    #[cfg(feature = "hdf5")]
    {
        let path = out.join("probes.h5");
        ems::io::h5::write_probes(&path, res.probes().iter().chain(res.lumped_currents()))?;
        println!("wrote {}", path.display());
    }
    for d in res.dumps() {
        let base = file_stem(d.name());
        if d.file_type() == csx::properties::DumpFileType::Hdf5 {
            #[cfg(feature = "hdf5")]
            {
                let path = out.join(format!("{base}.h5"));
                d.write_h5(&path)?;
                println!("wrote {}", path.display());
                continue;
            }
            #[cfg(not(feature = "hdf5"))]
            println!(
                "  ! dump {:?} asks for HDF5 but run_xml was built without the `hdf5` feature; writing .vtr",
                d.name()
            );
        }
        d.write_vtr(&out, &base)?;
        println!("wrote {}/{base}_*.vtr", out.display());
    }
    for d in res.sar_dumps() {
        let base = file_stem(d.name());
        let h5 = cfg!(feature = "hdf5") && d.file_type() == csx::properties::DumpFileType::Hdf5;
        for (k, f) in d.frequencies().iter().enumerate() {
            let sar = d.field(k)?;
            let path = out.join(format!("{base}_f{k}.{}", if h5 { "h5" } else { "vtr" }));
            #[cfg(feature = "hdf5")]
            if h5 {
                sar.write_h5(&path)?;
            }
            if !h5 {
                sar.write_vtr(&path)?;
            }
            let peak = sar.max().map_or(0.0, |m| m.value);
            println!(
                "wrote {} ({f:e} Hz: peak {peak:.4e} W/kg·s², absorbed {:.4e} W·s²)",
                path.display(),
                d.volume(k).total_power()
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
