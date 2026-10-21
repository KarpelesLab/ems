//! `ems`: runs an openEMS model file like the `openEMS` executable does.
//!
//! ```text
//! ems <FDTD_XML_FILE> [<options>...]
//! ```
//!
//! A drop-in replacement for openEMS v0.37's command-line program (design
//! §7.2): the same options (`ems --help`), the same files written to the
//! current directory, the same exit codes.
//!
//! * **Files.** The probe text files (`<probe>`, `<probe>_FD`; waveguide
//!   probes with their `mode_purity` column), the field dumps of every
//!   `DumpBox` in the format its `FileType` asks for (VTK, or HDF5 when ems
//!   is built with the `hdf5` feature; without it a model with HDF5 dumps
//!   is rejected before the run, unless `--disable-dumps`), the SAR dumps,
//!   and the excitation signals `et` / `ht`; see `ems::io::openems`.
//! * **Options.** As listed by `openEMS --help`, with its syntaxes
//!   (`--opt=value` and `--opt value`, unique prefixes of long options,
//!   `-v`, `-vv`, `-vvv`, `-v2`, `--verbose[=N]`):
//!   * `--disable-dumps`: no `DumpBox` is compiled or written (SAR
//!     included);
//!   * `--debug-material`, `--debug-PEC`, `--debug-operator`,
//!     `--debug-boxes`: `material_dump.vtr`, `PEC_dump.vtp`,
//!     `operator_dump.vtr` (`.vts` structured grids on cylindrical
//!     grids), `box_dump_<name>.vtk` (`ems::io::openems::debug`); `--debug-CSX`: `debugCSX.xml`, the
//!     model's `<ContinuousStructure>` as written by csx;
//!   * `--engine=<fastest|basic|sse|sse-compressed|multithreaded|gpu|gpu-reference>`:
//!     ems has a single engine; `basic`, `sse` and `sse-compressed` (the
//!     single-threaded openEMS engines) run it on one thread unless
//!     `--numThreads` is given, the others with the automatic thread
//!     count. Unknown names are ignored, as openEMS does;
//!   * `--numThreads=N`: engine threads (0: automatic);
//!   * `--no-simulation`: set up, write the debug and signal files, stop
//!     (exit status 1, as openEMS);
//!   * `--dump-statistics`: `openEMS_run_stats.txt` (a row per progress
//!     report) and `openEMS_stats.txt` (`ems::io::openems::stats`);
//!   * `--exact-endcriteria`: accepted; ems always evaluates the end
//!     criteria every Nyquist interval of the excitation, so its runs stop
//!     at the same step on any machine;
//!   * `--showProbeDiscretization`: prints the snapped box of every probe
//!     and dump;
//!   * `--nativeFieldDumps`, `--legacyHDF5Dumps`: the dump file variants of
//!     `ems::io::openems::dump::DumpFileOptions`;
//!   * `-v`…`-vvv`: more console output (operator summary, probe sampling,
//!     the model's properties).
//! * **Console.** The key lines of openEMS's output in the same form
//!   (mesh size, time step and Nyquist rate, excitation length, number of
//!   time steps, progress reports, end criteria, time and speed), so logs
//!   parse the same; compile warnings go to stderr.
//! * **Exit status.** 0 after a run; 1 if the model cannot be read or set
//!   up, a file cannot be written, or with `--no-simulation`; 255 without
//!   a model file (usage printed); 134 for a command-line error (openEMS
//!   aborts there).
//!
//! Differences: the model file is the first argument that is not an
//! option (openEMS takes the first argument, whatever it is); the probe
//! files are written at the end of the run rather than during it; the
//! `--debug-*` files hold ems's own values (see
//! `ems::io::openems::debug`).

mod args;
mod boxes;

use std::ops::ControlFlow;
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use args::{ENGINES, Options};
use ems::csx;
use ems::csx::properties::{DumpFileType, PropertyKind};
use ems::engine::StopReason;
use ems::io::openems::debug;
use ems::io::openems::dump::DumpFileOptions;
use ems::io::openems::{ExcitationSignal, RunStats, RunStatsLog, SignalSpan};
use ems::{CompileOptions, Simulation};

/// Exit status of a command-line error (openEMS aborts: `SIGABRT`).
const EXIT_USAGE_ERROR: u8 = 134;
/// Exit status without a model file (openEMS returns −1).
const EXIT_NO_FILE: u8 = 255;
/// Seconds between progress reports (as openEMS).
const PROGRESS_INTERVAL: Duration = Duration::from_secs(4);

/// C `printf("%g")`.
fn g(v: f64) -> String {
    ems::io::openems::probes::format_g(v, 6)
}

/// C `printf("%.{prec}e")` (two-digit exponent with sign).
fn e(v: f64, prec: usize) -> String {
    let s = format!("{v:.prec$e}");
    match s.split_once('e') {
        Some((m, x)) => {
            let (sign, digits) = match x.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', x),
            };
            format!("{m}e{sign}{digits:0>2}")
        }
        None => s,
    }
}

fn banner() {
    let line = format!(" {}", "-".repeat(70));
    println!("{line}");
    println!(
        " | ems v{} -- FDTD field solver, openEMS-compatible command line",
        env!("CARGO_PKG_VERSION")
    );
    println!(" | MIT license, https://github.com/KarpelesLab/ems");
    println!("{line}");
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    banner();
    if argv.is_empty() {
        print!("\n{}", args::usage());
        return ExitCode::from(EXIT_NO_FILE);
    }
    let opts = match Options::parse(argv) {
        Ok(o) => o,
        Err(err) => {
            eprintln!("ems: error: {err} (see ems --help)");
            return ExitCode::from(EXIT_USAGE_ERROR);
        }
    };
    if opts.help {
        print!("\n{}", args::usage());
        return ExitCode::SUCCESS;
    }
    let Some(file) = opts.file.clone() else {
        eprintln!("ems: error: no FDTD XML file given");
        print!("\n{}", args::usage());
        return ExitCode::from(EXIT_NO_FILE);
    };
    match run(&opts, Path::new(&file)) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("ems: error: {err}");
            ExitCode::FAILURE
        }
    }
}

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The thread count for the options (`None`: automatic).
fn threads(opts: &Options) -> Option<usize> {
    if opts.num_threads > 0 {
        return Some(opts.num_threads);
    }
    match opts.engine.as_deref() {
        Some("basic" | "sse" | "sse-compressed") => Some(1),
        _ => None,
    }
}

fn announce(opts: &Options) {
    let say = |s: &str| println!("ems - {s}");
    if opts.debug_csx {
        say("dumping the CSX geometry to 'debugCSX.xml'");
    }
    if opts.debug_pec {
        say(&format!(
            "dumping the PEC edges to '{}'",
            debug::PEC_DUMP_FILE
        ));
    }
    if opts.debug_boxes {
        say("dumping the probe and dump boxes to 'box_dump_*.vtk'");
    }
    if opts.debug_material {
        say(&format!(
            "dumping the materials to '{}' (.vts on cylindrical grids)",
            debug::MATERIAL_DUMP_FILE
        ));
    }
    if opts.debug_operator {
        say(&format!(
            "dumping the operator to '{}' (.vts on cylindrical grids)",
            debug::OPERATOR_DUMP_FILE
        ));
    }
    if opts.dump_statistics {
        say("dumping the run statistics to 'openEMS_run_stats.txt' and 'openEMS_stats.txt'");
    }
    if opts.disable_dumps {
        say("all field dumps disabled");
    }
    match opts.engine.as_deref() {
        None | Some("fastest") => {}
        Some(e) if ENGINES.contains(&e) => say(&format!(
            "engine '{e}' requested: ems has a single engine, run {}",
            match threads(opts) {
                Some(1) => "on one thread".to_string(),
                Some(n) => format!("on {n} threads"),
                None => "multithreaded".to_string(),
            }
        )),
        Some(e) => say(&format!("unknown engine '{e}' ignored")),
    }
    if opts.no_simulation {
        say("preprocessing only, no simulation");
    }
    if opts.show_probe_discretization {
        say("showing the probe discretization");
    }
    if opts.exact_endcriteria {
        say("exact end criteria (ems always evaluates them every Nyquist interval)");
    }
    if opts.native_field_dumps {
        say("field dumps with the native field components");
    }
    if opts.legacy_hdf5_dumps {
        say("HDF5 field dumps in the legacy layout");
    }
    if opts.num_threads > 0 {
        say(&format!("fixed number of threads: {}", opts.num_threads));
    }
    if opts.verbose > 0 {
        say(&format!("verbose level {}", opts.verbose));
    }
}

fn run(opts: &Options, file: &Path) -> Res<ExitCode> {
    announce(opts);
    let cwd = Path::new(".");
    if opts.verbose > 0 {
        println!("Reading the openEMS file {} ...", file.display());
    }
    // openEMS's reading rules (boundary names, DumpMode default, ...).
    let (mut doc, read_warnings) = ems::io::openems::read_model(file)
        .map_err(|err| format!("cannot load {}: {err}", file.display()))?;
    for w in &read_warnings {
        eprintln!("Warning: {w}");
    }
    // --disable-dumps: the dumps are not compiled, but their boxes are
    // still reported (as openEMS does), from a model of the grid and the
    // dumps only.
    let mut disabled_dumps = None;
    if opts.disable_dumps {
        if opts.show_probe_discretization || opts.debug_boxes {
            disabled_dumps = Some(filtered(&doc.structure, is_dump));
        }
        doc.structure = filtered(&doc.structure, |p| !is_dump(p));
    }
    if opts.verbose >= 3 {
        for p in doc.structure.properties() {
            println!(
                " property {:?}: {} with {} primitive(s)",
                p.name,
                p.kind.type_name(),
                p.primitives.len()
            );
        }
    }
    let hdf5_dumps: Vec<&str> = doc
        .structure
        .properties()
        .iter()
        .filter_map(|p| match &p.kind {
            PropertyKind::DumpBox(d) if d.file_type == DumpFileType::Hdf5 => Some(p.name.as_str()),
            _ => None,
        })
        .collect();
    if !cfg!(feature = "hdf5") && !hdf5_dumps.is_empty() {
        return Err(format!(
            "the dumps {hdf5_dumps:?} ask for HDF5 files (FileType=\"1\"), but this ems was \
             built without the `hdf5` feature; rebuild it with `--features hdf5`, or run with \
             --disable-dumps"
        )
        .into());
    }

    if opts.debug_csx {
        doc.structure.write_file(cwd.join("debugCSX.xml"))?;
    }
    println!("Create FDTD operator (ems)");
    let t_setup = Instant::now();
    let n_threads = threads(opts);
    let copts = CompileOptions::default();
    let mut sim = Simulation::from_document_with(&doc, &copts, |f| match n_threads {
        Some(n) => f.threads(n),
        None => f,
    })
    .map_err(|err| format!("cannot set up {}: {err}", file.display()))?;
    for w in sim.warnings() {
        eprintln!("Warning: {w}");
    }
    if opts.debug_material || opts.debug_pec {
        // The edge materials of the compile step (the engine keeps only
        // its coefficients).
        let c = ems::compile::compile(&doc.structure, sim.fdtd(), &copts)?;
        if opts.debug_material {
            debug::write_material_dump(cwd, &c.grid, &c.materials)?;
        }
        if opts.debug_pec {
            debug::write_pec_dump(cwd, &c.grid, &c.materials)?;
        }
    }
    if opts.debug_operator {
        let grid = sim.grid();
        let map = sim.node_map();
        let mut exc = Vec::new();
        for x in &sim.sources().excitations {
            if !x.excitation_type.is_electric() {
                continue;
            }
            for &(edge, amp) in &x.amplitudes {
                let (axis, node) = map.edge(edge);
                exc.push((axis, node, amp * grid.edge_length(axis, node)));
            }
        }
        debug::write_operator_dump(cwd, grid, &sim.engine().coefficients(), &exc)?;
    }

    // ---- the key figures.
    let grid = sim.grid();
    let nodes = grid.nodes();
    let cells = nodes.iter().product::<usize>() as u64;
    println!(
        "FDTD simulation size: {}x{}x{} --> {cells} FDTD cells",
        nodes[0], nodes[1], nodes[2]
    );
    let dt = sim.engine().dt();
    let nyquist = sim.fdtd().openems_nyquist_interval(dt);
    println!(
        "FDTD timestep is: {} s; Nyquist rate: {nyquist} timesteps @{} Hz",
        g(dt),
        g(1.0 / (2.0 * nyquist as f64 * dt))
    );
    let limits = sim.run_limits();
    let max_steps = limits.max_timesteps;
    if let Some(w) = sim.fdtd().get_waveform() {
        let sig = ExcitationSignal::new(w, dt, max_steps);
        if let Some(full) = sig.uncut {
            println!(
                "Excitation: the Gaussian pulse would be {full} timesteps ({} s) long; cut to the \
                 max. number of timesteps",
                g(full as f64 * dt)
            );
        }
        let (what, n) = match sig.span {
            SignalSpan::Length(n) => ("length", n),
            SignalSpan::Period(n) => ("period", n),
        };
        println!(
            "Excitation signal {what} is: {n} timesteps ({}s)",
            g(n as f64 * dt)
        );
        println!(
            "Max. number of timesteps: {max_steps} ( --> {} * Excitation signal {what})",
            g(max_steps as f64 / n.max(1) as f64)
        );
        if max_steps < 3 * n {
            eprintln!(
                "Warning: the max. number of timesteps is smaller than three times the \
                 excitation signal {what}; you may want to choose a higher number of max. \
                 timesteps"
            );
        }
        sig.write(cwd)?;
    } else {
        println!("Max. number of timesteps: {max_steps} (no excitation)");
    }
    if opts.verbose > 0 {
        print_operator_summary(&sim, cells, nyquist, t_setup.elapsed());
    }
    if opts.no_simulation {
        return Ok(ExitCode::FAILURE);
    }

    println!(
        "Create FDTD engine (ems, {} thread{})",
        sim.engine().threads(),
        if sim.engine().threads() == 1 { "" } else { "s" }
    );
    let before = sim.results();
    if opts.show_probe_discretization || opts.debug_boxes {
        let unit = sim.delta_unit();
        let cyl = sim.grid().is_cylindrical();
        let du = |v: [f64; 3]| -> String {
            let c: Vec<String> = (0..3)
                .map(|a| g(if cyl && a == 1 { v[a] } else { v[a] / unit }))
                .collect();
            format!("({})", c.join(","))
        };
        let idx = |v: [usize; 3]| format!("[{},{},{}]", v[0], v[1], v[2]);
        let extra = match &disabled_dumps {
            Some(model) => Some(
                ems::compile::compile(model, sim.fdtd(), &copts)?
                    .sources
                    .dumps,
            ),
            None => None,
        };
        for b in boxes::snapped_boxes(&sim, &before, extra.as_deref()) {
            if opts.show_probe_discretization {
                println!(
                    "{}: snapped {} coords: {} -> {}   {} -> {}",
                    b.name,
                    if b.dual { "dual" } else { "primary" },
                    du(b.start_m),
                    du(b.stop_m),
                    idx(b.start),
                    idx(b.stop)
                );
            }
            if opts.debug_boxes
                && let Some(name) = &b.dump_name
            {
                debug::write_box_dump(cwd, name, b.start_m, b.stop_m)?;
            }
        }
    }
    if opts.verbose >= 2 {
        let k = before.openems_probe_interval();
        for p in before.probes() {
            println!(
                "probe '{}': time-domain sampling every {k} timestep(s) (Nyquist/{})",
                p.name(),
                (nyquist / k.max(1)).max(1)
            );
        }
    }
    let mut stats_log = if opts.dump_statistics {
        Some(RunStatsLog::create(cwd)?)
    } else {
        None
    };

    println!("Running the FDTD engine ...");
    if opts.exact_endcriteria || opts.verbose > 0 {
        println!(
            "Evaluating the end criteria every {} timestep(s) (the excitation's Nyquist rate)",
            nyquist
        );
    }
    let start = Instant::now();
    let mut last = (Instant::now(), 0u64);
    let mut io_error = None;
    let res = sim.run_with(|p| {
        if last.0.elapsed() >= PROGRESS_INTERVAL {
            let dt_wall = last.0.elapsed().as_secs_f64();
            let steps = p.steps.saturating_sub(last.1);
            let speed = cells as f64 * steps as f64 / dt_wall.max(1e-9);
            println!(
                "[@{:>9}s] Timestep: {:>12} || Speed: {:>6.1} MC/s ({} s/TS) || Energy: ~{} ({:>6.2}dB)",
                start.elapsed().as_secs(),
                p.timestep,
                speed / 1e6,
                e(dt_wall / steps.max(1) as f64, 3),
                e(p.energy, 2),
                p.energy_db
            );
            if let Some(log) = &mut stats_log
                && let Err(err) = log.report(start.elapsed().as_secs_f64(), p.timestep, speed, p.energy)
            {
                io_error = Some(err);
                return ControlFlow::Break(());
            }
            last = (Instant::now(), p.steps);
        }
        ControlFlow::Continue(())
    })?;
    if let Some(err) = io_error {
        return Err(err.into());
    }
    let o = *res.outcome().expect("a run sets the outcome");
    let crit_db = limits.end_criteria.map(|c| c.decay_db());
    match o.reason {
        StopReason::EnergyDecay { ratio_db } => println!(
            "Run: end-criteria of {}dB reached after {} timesteps ({}dB)",
            g(crit_db.unwrap_or(f64::NAN)),
            o.steps,
            g(ratio_db)
        ),
        StopReason::MaxTimesteps => match crit_db {
            Some(db) => eprintln!(
                "Warning: max. number of timesteps was reached before the end-criteria of {}dB \
                 was reached; you may want to choose a higher number of max. timesteps",
                g(db)
            ),
            None => println!("Run: max. number of timesteps reached"),
        },
        StopReason::Diverged => eprintln!(
            "Warning: the field energy diverged after {} timesteps (unstable setup)",
            o.steps
        ),
        _ => {}
    }
    let secs = o.elapsed.as_secs_f64();
    println!(
        "Time for {} iterations with {cells} cells : {} sec",
        o.steps,
        g(secs)
    );
    println!(
        "Speed: {} MCells/s",
        g(cells as f64 * o.steps as f64 / secs.max(1e-9) / 1e6)
    );
    if opts.dump_statistics {
        RunStats {
            cells,
            dt,
            iterations: o.steps,
            wall_time: secs,
        }
        .write(cwd)?;
    }

    // ---- files.
    let written = ems::io::openems::write_probes(&res, cwd)?;
    let dopts = DumpFileOptions::new()
        .legacy_hdf5(opts.legacy_hdf5_dumps)
        .native_field_dumps(opts.native_field_dumps);
    let mut dumps = 0;
    for d in res.dumps() {
        dumps += d.write_openems(cwd, &dopts)?.len();
    }
    for d in res.sar_dumps() {
        dumps += d.write_openems(cwd)?.len();
    }
    if opts.verbose > 0 {
        println!(
            "Wrote {} probe file(s) and {dumps} dump file(s)",
            written.len()
        );
    }
    Ok(ExitCode::SUCCESS)
}

/// The structure `s` with only the properties for which `keep` holds
/// (property ids are renumbered).
fn filtered(
    s: &csx::ContinuousStructure,
    keep: impl Fn(&csx::Property) -> bool,
) -> csx::ContinuousStructure {
    let mut out = csx::ContinuousStructure::new();
    out.grid = s.grid.clone();
    out.parameters = s.parameters.clone();
    out.background = s.background.clone();
    out.coord_system = s.coord_system;
    out.base_dir = s.base_dir.clone();
    out.extra_attributes = s.extra_attributes.clone();
    out.extra_elements = s.extra_elements.clone();
    for p in s.properties().iter().filter(|p| keep(p)) {
        out.add_property(p.clone());
    }
    out
}

fn is_dump(p: &csx::Property) -> bool {
    matches!(p.kind, PropertyKind::DumpBox(_))
}

/// `-v`: a summary of what was set up.
fn print_operator_summary(sim: &Simulation, cells: u64, nyquist: u64, setup: Duration) {
    let grid = sim.grid();
    let n = grid.nodes();
    let c = grid.cells();
    let dt = sim.engine().dt();
    println!("------- ems operator -------");
    println!(
        "Mesh lines\t\t: {}x{}x{} = {cells} nodes ({} MCells)",
        n[0],
        n[1],
        n[2],
        g(cells as f64 / 1e6)
    );
    println!(
        "Mesh cells\t\t: {}x{}x{} = {}",
        c[0],
        c[1],
        c[2],
        grid.num_cells()
    );
    println!(
        "Coordinates\t\t: {}",
        if grid.is_cylindrical() {
            "cylindrical"
        } else {
            "Cartesian"
        }
    );
    println!("Timestep (s)\t\t: {}", g(dt));
    println!("Nyquist rate (TS)\t: {nyquist}");
    println!("Nyquist rate (s)\t: {}", g(nyquist as f64 * dt));
    println!("Engine threads\t\t: {}", sim.engine().threads());
    let src = sim.sources();
    let (e_edges, h_faces) = src.excitations.iter().fold((0, 0), |(e, h), x| {
        if x.excitation_type.is_electric() {
            (e + x.amplitudes.len(), h)
        } else if x.excitation_type.is_magnetic() {
            (e, h + x.amplitudes.len())
        } else {
            (e, h)
        }
    });
    println!("E-field excitations\t: {e_edges} edge(s)");
    println!("H-field excitations\t: {h_faces} face(s)");
    println!("Plane waves\t\t: {}", src.plane_waves.len());
    println!("Lumped element edges\t: {}", src.lumped.len());
    println!("Probes\t\t\t: {}", src.probes.len());
    println!("Dumps\t\t\t: {} ({} SAR)", src.dumps.len(), src.sar.len());
    println!("Setup time (s)\t\t: {}", g(setup.as_secs_f64()));
    println!("----------------------------");
}
