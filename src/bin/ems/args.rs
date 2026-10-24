//! Command-line parsing: the options of `openEMS --help` (v0.37.0-32,
//! openEMS-Project be30220, which added `--gpu-device`), with the
//! syntaxes openEMS accepts (`--opt=value`, `--opt value`, unique
//! prefixes of long options, `-v`/`-vv`/`-vvv`/`-v2`).

/// The engine names openEMS knows (`--engine`).
pub const ENGINES: [&str; 7] = [
    "fastest",
    "basic",
    "sse",
    "sse-compressed",
    "multithreaded",
    "gpu",
    "gpu-reference",
];

/// Parsed options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// The model file (the first argument that is not an option).
    pub file: Option<String>,
    pub help: bool,
    pub disable_dumps: bool,
    pub debug_material: bool,
    pub debug_pec: bool,
    pub debug_operator: bool,
    pub debug_boxes: bool,
    pub debug_csx: bool,
    /// `--engine` as given (`None`: `fastest`).
    pub engine: Option<String>,
    /// `--gpu-device` (−1: the best device). Accepted and reported; ems
    /// has no GPU engine.
    pub gpu_device: i64,
    /// `--numThreads` (0: automatic).
    pub num_threads: usize,
    pub no_simulation: bool,
    pub dump_statistics: bool,
    pub exact_endcriteria: bool,
    pub show_probe_discretization: bool,
    pub native_field_dumps: bool,
    pub legacy_hdf5_dumps: bool,
    /// Verbose level 0–3.
    pub verbose: u32,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            file: None,
            help: false,
            disable_dumps: false,
            debug_material: false,
            debug_pec: false,
            debug_operator: false,
            debug_boxes: false,
            debug_csx: false,
            engine: None,
            gpu_device: -1,
            num_threads: 0,
            no_simulation: false,
            dump_statistics: false,
            exact_endcriteria: false,
            show_probe_discretization: false,
            native_field_dumps: false,
            legacy_hdf5_dumps: false,
            verbose: 0,
        }
    }
}

/// What an option takes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arg {
    /// A flag.
    None,
    /// A required value.
    Required,
    /// An optional value (`--verbose[=N]`).
    Optional,
}

/// The long options: name, argument.
const LONG: [(&str, Arg); 17] = [
    ("help", Arg::None),
    ("disable-dumps", Arg::None),
    ("debug-material", Arg::None),
    ("debug-PEC", Arg::None),
    ("debug-operator", Arg::None),
    ("debug-boxes", Arg::None),
    ("debug-CSX", Arg::None),
    ("engine", Arg::Required),
    ("gpu-device", Arg::Required),
    ("numThreads", Arg::Required),
    ("no-simulation", Arg::None),
    ("dump-statistics", Arg::None),
    ("exact-endcriteria", Arg::None),
    ("showProbeDiscretization", Arg::None),
    ("nativeFieldDumps", Arg::None),
    ("legacyHDF5Dumps", Arg::None),
    ("verbose", Arg::Optional),
];

/// A command-line error (openEMS reports it and exits with status 1;
/// see `main`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgError(pub String);

impl std::fmt::Display for ArgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The long option `name` matches exactly, or as the unique prefix of
/// one (openEMS accepts abbreviations).
fn lookup(name: &str) -> Result<(&'static str, Arg), ArgError> {
    if let Some(&(n, a)) = LONG.iter().find(|(n, _)| *n == name) {
        return Ok((n, a));
    }
    let hits: Vec<_> = LONG
        .iter()
        .filter(|(n, _)| n.starts_with(name) && !name.is_empty())
        .collect();
    match hits.as_slice() {
        [(n, a)] => Ok((n, *a)),
        [] => Err(ArgError(format!("unrecognised option '--{name}'"))),
        _ => Err(ArgError(format!(
            "option '--{name}' is ambiguous and matches {}",
            hits.iter()
                .map(|(n, _)| format!("'--{n}'"))
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

fn parse_level(opt: &str, v: &str) -> Result<u32, ArgError> {
    v.parse::<u32>().map_err(|_| {
        ArgError(format!(
            "the argument ('{v}') for option '--{opt}' is invalid"
        ))
    })
}

impl Options {
    /// Parses the arguments after the program name.
    ///
    /// # Errors
    /// Unknown or ambiguous options, missing or invalid values, values
    /// for flags.
    pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Self, ArgError> {
        let args: Vec<String> = args.into_iter().collect();
        let mut o = Options::default();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            i += 1;
            if let Some(body) = a.strip_prefix("--") {
                if body.is_empty() {
                    // `--`: the rest are file names.
                    if o.file.is_none() {
                        o.file = args.get(i).cloned();
                    }
                    break;
                }
                let (name, inline) = match body.split_once('=') {
                    Some((n, v)) => (n, Some(v.to_string())),
                    None => (body, None),
                };
                let (name, kind) = lookup(name)?;
                let value = match kind {
                    Arg::None => {
                        if inline.is_some() {
                            return Err(ArgError(format!(
                                "option '--{name}' does not take any arguments"
                            )));
                        }
                        None
                    }
                    Arg::Required => match inline {
                        Some(v) => Some(v),
                        None if i < args.len() => {
                            i += 1;
                            Some(args[i - 1].clone())
                        }
                        None => {
                            return Err(ArgError(format!(
                                "the required argument for option '--{name}' is missing"
                            )));
                        }
                    },
                    Arg::Optional => match inline {
                        Some(v) => Some(v),
                        // A following number is the value.
                        None if args.get(i).is_some_and(|v| {
                            !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit())
                        }) =>
                        {
                            i += 1;
                            Some(args[i - 1].clone())
                        }
                        None => None,
                    },
                };
                o.apply(name, value)?;
            } else if let Some(short) = a.strip_prefix('-').filter(|s| !s.is_empty()) {
                o.short(short, &args, &mut i)?;
            } else if o.file.is_none() {
                o.file = Some(a.clone());
            }
            // Further file names are ignored, as openEMS does.
        }
        Ok(o)
    }

    /// `-h`, `-v`, `-vv`, `-vvv`, `-v2`, `-v 2`.
    fn short(&mut self, s: &str, args: &[String], i: &mut usize) -> Result<(), ArgError> {
        match s {
            "h" => self.help = true,
            _ if s.starts_with('v') => {
                let rest = &s[1..];
                if rest.is_empty() {
                    self.verbose = match args.get(*i) {
                        Some(v) if !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) => {
                            *i += 1;
                            parse_level("verbose", v)?
                        }
                        _ => 1,
                    };
                } else if rest.bytes().all(|b| b == b'v') {
                    self.verbose = 1 + rest.len() as u32;
                } else {
                    let v = rest.strip_prefix('=').unwrap_or(rest);
                    self.verbose = parse_level("verbose", v)?;
                }
            }
            _ => return Err(ArgError(format!("unrecognised option '-{s}'"))),
        }
        Ok(())
    }

    fn apply(&mut self, name: &str, value: Option<String>) -> Result<(), ArgError> {
        match name {
            "help" => self.help = true,
            "disable-dumps" => self.disable_dumps = true,
            "debug-material" => self.debug_material = true,
            "debug-PEC" => self.debug_pec = true,
            "debug-operator" => self.debug_operator = true,
            "debug-boxes" => self.debug_boxes = true,
            "debug-CSX" => self.debug_csx = true,
            "engine" => self.engine = value,
            "gpu-device" => {
                let v = value.unwrap_or_default();
                self.gpu_device = v.trim().parse().map_err(|_| {
                    ArgError(format!(
                        "the argument ('{v}') for option '--gpu-device' is invalid"
                    ))
                })?;
            }
            "numThreads" => {
                let v = value.unwrap_or_default();
                self.num_threads = v.trim().parse().map_err(|_| {
                    ArgError(format!(
                        "the argument ('{v}') for option '--numThreads' is invalid"
                    ))
                })?;
            }
            "no-simulation" => self.no_simulation = true,
            "dump-statistics" => self.dump_statistics = true,
            "exact-endcriteria" => self.exact_endcriteria = true,
            "showProbeDiscretization" => self.show_probe_discretization = true,
            "nativeFieldDumps" => self.native_field_dumps = true,
            "legacyHDF5Dumps" => self.legacy_hdf5_dumps = true,
            "verbose" => {
                self.verbose = match value {
                    Some(v) => parse_level("verbose", &v)?,
                    None => 1,
                };
            }
            _ => unreachable!("option table and handler agree"),
        }
        Ok(())
    }
}

/// The usage text (`-h`, `--help`).
pub fn usage() -> String {
    let lines = [
        "Usage: ems <FDTD_XML_FILE> [<options>...]",
        "",
        "Runs an openEMS FDTD model file and writes the files openEMS writes",
        "(probe files, field dumps, et/ht) to the current directory.",
        "",
        "Options:",
        "  -h [ --help ]                   print this text and exit",
        "  --disable-dumps                 skip all field dumps (DumpBox, incl. SAR)",
        "  --debug-material                write the edge materials to material_dump.vtr",
        "  --debug-PEC                     write the PEC edges to PEC_dump.vtp",
        "  --debug-operator                write the update coefficients to operator_dump.vtr",
        "  --debug-boxes                   write the probe and dump boxes to box_dump_*.vtk",
        "  --debug-CSX                     write the model geometry to debugCSX.xml",
        "  --engine arg (=fastest)         engine type: fastest, basic, sse, sse-compressed,",
        "                                  multithreaded, gpu, gpu-reference; ems has one",
        "                                  (multithreaded) engine: basic, sse and",
        "                                  sse-compressed run it on one thread",
        "  --gpu-device arg (=-1)          GPU device of --engine=gpu; accepted, ems has no",
        "                                  GPU engine",
        "  --numThreads arg (=0)           number of engine threads (0: automatic)",
        "  --no-simulation                 set up only (write the debug and signal files),",
        "                                  do not run",
        "  --dump-statistics               write openEMS_run_stats.txt and openEMS_stats.txt",
        "  --exact-endcriteria             accepted; ems always checks the end criteria every",
        "                                  Nyquist interval, independent of the wall clock",
        "  --showProbeDiscretization       print the snapped probe and dump boxes",
        "  --nativeFieldDumps              cylindrical VTK dumps in (rho, alpha, z) components",
        "  --legacyHDF5Dumps               HDF5 dumps in the legacy layout (Octave/Matlab)",
        "  -v [ --verbose ] [=arg(=1)]     verbose level 1 to 3 (also -v, -vv, -vvv)",
    ];
    let mut s = lines.join("\n");
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(a: &[&str]) -> Result<Options, ArgError> {
        Options::parse(a.iter().map(|s| s.to_string()))
    }

    #[test]
    fn syntaxes() {
        let o = p(&["m.xml", "--engine=basic", "--numThreads", "3", "-vv"]).unwrap();
        assert_eq!(o.file.as_deref(), Some("m.xml"));
        assert_eq!(o.engine.as_deref(), Some("basic"));
        assert_eq!((o.num_threads, o.verbose), (3, 2));
        let o = p(&["--no-sim", "m.xml", "other.xml", "--verbose", "3"]).unwrap();
        assert!(o.no_simulation);
        assert_eq!((o.file.as_deref(), o.verbose), (Some("m.xml"), 3));
        assert_eq!(p(&["m.xml", "-v2"]).unwrap().verbose, 2);
        assert_eq!(p(&["m.xml", "--verbose"]).unwrap().verbose, 1);
        assert_eq!(p(&["m.xml", "-v", "m2"]).unwrap().verbose, 1);
        assert!(p(&["m.xml", "--bogus"]).is_err());
        assert!(p(&["m.xml", "--debug"]).is_err()); // ambiguous
        assert!(p(&["m.xml", "--numThreads=abc"]).is_err());
        assert!(p(&["m.xml", "--disable-dumps=1"]).is_err());
        assert!(p(&["m.xml", "--engine"]).is_err());
        assert!(p(&["-h"]).unwrap().help);
        assert_eq!(p(&["m.xml"]).unwrap().gpu_device, -1);
        assert_eq!(p(&["m.xml", "--gpu", "2"]).unwrap().gpu_device, 2);
        assert_eq!(p(&["m.xml", "--gpu-device=-5"]).unwrap().gpu_device, -5);
        assert!(p(&["m.xml", "--gpu-device=abc"]).is_err());
        let o = p(&[
            "m.xml",
            "--debug-PEC",
            "--debug-CSX",
            "--legacy",
            "--native",
        ])
        .unwrap();
        assert!(o.debug_pec && o.debug_csx && o.legacy_hdf5_dumps && o.native_field_dumps);
    }
}
