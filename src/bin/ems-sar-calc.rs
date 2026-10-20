//! `ems-sar-calc`: mass-averaged SAR from a raw SAR dump, with the command
//! line of openEMS's `sar_calc` (same options, same output layout); see
//! `ems::io::openems::sar_calc`.

use std::process::ExitCode;

use ems::io::openems::sar_calc::{SarCalcOptions, run};
use ems::post::sar::AveragingMethod;

const USAGE: &str = "\
Usage: ems-sar-calc -i <raw.h5> -o <out.h5> -m <grams> [options]
Options:
  -h [ --help ]               print usage message
  -i [ --input ] arg          pathname to input hdf5 file
  -o [ --output ] arg         pathname for output hdf5 file
  --method arg (=SIMPLE)      set SAR method: IEEE_C95_3, IEEE_62704, SIMPLE
  -m [ --mass ] arg           averaging mass in g
  -a [ --autorange ] arg      autorange, limit the calculation to cells within
                              N dB of the peak local SAR (>0)
  -n [ --numThreads ] arg     number of threads
  -v [ --verbose ]            verbose
  -p [ --progress ]           show progress
  -e [ --export_cube_stats ]  Export Cube Statistics
  --legacyHDF5Dumps           Dumping using the legacy HDF5 file format";

/// Options taking a value: (short, long).
const VALUED: [(&str, &str); 6] = [
    ("i", "input"),
    ("o", "output"),
    ("", "method"),
    ("m", "mass"),
    ("a", "autorange"),
    ("n", "numThreads"),
];
/// Flags: (short, long).
const FLAGS: [(&str, &str); 5] = [
    ("h", "help"),
    ("v", "verbose"),
    ("p", "progress"),
    ("e", "export_cube_stats"),
    ("", "legacyHDF5Dumps"),
];

/// Parses the command line into (long name, value) pairs.
fn parse(args: &[String]) -> Result<Vec<(&'static str, String)>, String> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let (name, inline) = if let Some(l) = a.strip_prefix("--") {
            match l.split_once('=') {
                Some((n, v)) => (n.to_string(), Some(v.to_string())),
                None => (l.to_string(), None),
            }
        } else if let Some(s) = a.strip_prefix('-').filter(|s| !s.is_empty()) {
            let (c, rest) = s.split_at(1);
            let long = VALUED
                .iter()
                .chain(&FLAGS)
                .find(|(sh, _)| *sh == c)
                .map(|(_, l)| (*l).to_string())
                .ok_or_else(|| format!("unrecognised option '{a}'"))?;
            (long, (!rest.is_empty()).then(|| rest.to_string()))
        } else {
            return Err(format!("unexpected argument '{a}'"));
        };
        if let Some(&(_, l)) = VALUED.iter().find(|(_, l)| *l == name) {
            let v = match inline {
                Some(v) => v,
                None => it.next().cloned().ok_or_else(|| {
                    format!("the required argument for option '--{l}' is missing")
                })?,
            };
            out.push((l, v));
        } else if let Some(&(_, l)) = FLAGS.iter().find(|(_, l)| *l == name) {
            out.push((l, String::new()));
        } else {
            return Err(format!("unrecognised option '{a}'"));
        }
    }
    Ok(out)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match main_inner(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn main_inner(args: &[String]) -> Result<(), String> {
    let opts = parse(args)?;
    let get = |n: &str| {
        opts.iter()
            .rev()
            .find(|(k, _)| *k == n)
            .map(|(_, v)| v.as_str())
    };
    if get("help").is_some() {
        println!("{USAGE}");
        return Ok(());
    }
    let input = get("input").ok_or("the option '--input' is required but missing")?;
    let output = get("output").ok_or("the option '--output' is required but missing")?;
    let mass_g: f64 = get("mass")
        .ok_or("ems-sar-calc: the averaging mass (-m) is required")?
        .parse()
        .map_err(|_| "ems-sar-calc: invalid mass".to_string())?;
    if !(mass_g.is_finite() && mass_g > 0.0) {
        return Err("ems-sar-calc: SAR calculation is invalid due to missing values".into());
    }
    let method_name = get("method").unwrap_or("SIMPLE");
    let method = AveragingMethod::from_name(method_name)
        .ok_or_else(|| format!("ems-sar-calc: {method_name} is an unknown averaging method"))?;
    let autorange = match get("autorange") {
        Some(a) => Some(
            a.parse::<f64>()
                .ok()
                .filter(|a| *a > 0.0 && a.is_finite())
                .ok_or_else(|| format!("ems-sar-calc: invalid auto range {a}"))?,
        ),
        None => None,
    };
    let threads = match get("numThreads") {
        Some(n) => n
            .parse()
            .map_err(|_| format!("ems-sar-calc: invalid thread count {n}"))?,
        None => 0,
    };
    let o = SarCalcOptions {
        input: input.into(),
        output: output.into(),
        mass_g,
        method,
        threads,
        verbose: get("verbose").is_some(),
        export_cube_stats: get("export_cube_stats").is_some(),
        legacy: get("legacyHDF5Dumps").is_some(),
        autorange,
    };
    if o.verbose {
        println!("Averaging method: {}", method.name());
        println!("Input file: {input}");
        println!("Averaging mass: {}", mass_g * 1e-3);
    }
    let out = run(&o).map_err(|e| format!("ems-sar-calc: {e}"))?;
    let (raw, res) = (&out.raw, &out.frequencies);
    if let (Some(db), Some(r)) = (autorange, &out.range) {
        if o.verbose {
            println!("Calculate auto range with {db}dB from the local SAR max.");
            println!("Auto range: padded by {}m", r.padding);
            for a in 0..3 {
                println!(
                    "SetSubRange: Dir {a} Start {} Stop {}",
                    r.start[a], r.stop[a]
                );
            }
        }
        for f in res {
            if let Some(t) = f.threshold.filter(|&t| t > f.max.0) {
                println!(
                    "Warning, the auto range threshold ({t} W/kg) is above the peak SAR that was \
                     found ({} W/kg) at f={}Hz. The true peak may be outside of the evaluated \
                     range, increase the auto range (dB) or disable it...",
                    f.max.0, f.frequency
                );
            }
        }
    }
    if o.export_cube_stats && res.len() > 1 {
        println!("Unable to store cube statistics for more than one frequency of interest!!");
    }
    println!("Output file: {output}");
    let mass = raw.total_mass();
    println!("Total mass: {mass}");
    for r in res {
        println!("********");
        println!("Frequency: {}", r.frequency);
        println!("Total power: {}", r.power);
        println!("Avg SAR: {}", r.power / mass);
        println!("max. SAR: {}", r.max.0);
        println!(
            "max. SAR Pos.Idx.: {}, {}, {}",
            r.max.1[0], r.max.1[1], r.max.1[2]
        );
    }
    if o.verbose
        && let Some(r) = res.first()
    {
        println!(
            "Stats: Valid={} Used={} Unused={} Air-Voxel={}",
            r.cubes[0], r.cubes[1], r.cubes[2], r.cubes[3]
        );
    }
    Ok(())
}
