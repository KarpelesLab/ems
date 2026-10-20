//! The snapped probe and dump boxes (`--showProbeDiscretization`,
//! `--debug-boxes`), as openEMS reports them.
//!
//! Observed with openEMS v0.37: voltage, E-field and waveguide probes and
//! dumps on the primary mesh, current and H-field probes on the dual mesh
//! (the probe's reported position), H and B dumps without interpolation
//! and all cell-interpolated dumps on the dual lines (cell centres,
//! including the virtual one past the last line), SAR dumps on the
//! primary lines; dumps snapped like their samples
//! (`ems::io::openems::dump::sample_indices`, without sub-sampling).

use ems::compile::ProbeSpec;
use ems::csx::properties::{DumpBox, DumpMode, ProbeType};
use ems::io::openems::dump::sample_indices;
use ems::{Axis, Results, Simulation};

/// One snapped box.
pub struct SnappedBox {
    /// Property name.
    pub name: String,
    /// The `box_dump_<…>.vtk` name part (`""` for probes), or `None` if
    /// openEMS writes no box file for it.
    pub dump_name: Option<String>,
    /// Dual-mesh indices.
    pub dual: bool,
    pub start: [usize; 3],
    pub stop: [usize; 3],
    /// Corner coordinates in metres (radians along α on cylindrical
    /// grids).
    pub start_m: [f64; 3],
    pub stop_m: [f64; 3],
}

/// Position of primary line `i` or dual line `i` (between `i` and
/// `i + 1`; past the last line by half the last cell).
fn position(lines: &[f64], i: usize, dual: bool) -> f64 {
    if !dual {
        return lines[i];
    }
    match lines.len() {
        0 => 0.0,
        1 => lines[0],
        n if i + 1 < n => 0.5 * (lines[i] + lines[i + 1]),
        n => lines[n - 1] + 0.5 * (lines[n - 1] - lines[n - 2]),
    }
}

/// The boxes of every probe, then every dump, in model order (`dumps`:
/// those dumps instead of the simulation's, e.g. the disabled ones).
pub fn snapped_boxes(
    sim: &Simulation,
    res: &Results,
    dumps: Option<&[ProbeSpec<DumpBox>]>,
) -> Vec<SnappedBox> {
    let g = sim.grid();
    let lines = |a: Axis| g.lines(a);
    let make =
        |name: &str, dump_name: Option<String>, dual: bool, start: [usize; 3], stop: [usize; 3]| {
            let at = |idx: [usize; 3]| Axis::ALL.map(|a| position(lines(a), idx[a.index()], dual));
            SnappedBox {
                name: name.to_string(),
                dump_name,
                dual,
                start,
                stop,
                start_m: at(start),
                stop_m: at(stop),
            }
        };
    let mut out = Vec::new();
    for p in &sim.sources().probes {
        let loc = res.probe(&p.name).and_then(|r| r.location());
        let (dual, dump) = match p.settings.probe_type {
            ProbeType::Voltage | ProbeType::EField => (false, None),
            ProbeType::Current => (true, Some(String::new())),
            ProbeType::HField => (true, None),
            ProbeType::WaveguideVoltage | ProbeType::WaveguideCurrent => {
                out.push(make(&p.name, Some(String::new()), false, p.start, p.stop));
                continue;
            }
            _ => continue,
        };
        // Probes that were not mapped (e.g. a current probe without a
        // normal) have no position.
        let Some(l) = loc else { continue };
        out.push(make(&p.name, dump, dual, l.start, l.stop));
    }
    for d in dumps.unwrap_or(&sim.sources().dumps) {
        let s = &d.settings;
        let id = s.dump_type.id();
        let sar = (20..=29).contains(&id);
        let magnetic = matches!(id, 1 | 5 | 11 | 15);
        let dual = !sar
            && (s.dump_mode == DumpMode::Cell
                || (s.dump_mode == DumpMode::NoInterpolation && magnetic));
        let mut start = [0; 3];
        let mut stop = [0; 3];
        let mut inside = true;
        for a in Axis::ALL {
            let l = lines(a);
            let cand: Vec<f64> = if dual {
                (0..l.len()).map(|i| position(l, i, true)).collect()
            } else {
                l.to_vec()
            };
            let domain = (l[0], l[l.len() - 1]);
            let k = a.index();
            match sample_indices(&cand, domain, d.corners[0][k], d.corners[1][k], 1, None) {
                Some(ix) => {
                    start[k] = ix[0];
                    stop[k] = ix[ix.len() - 1];
                }
                None => inside = false,
            }
        }
        if inside {
            out.push(make(&d.name, Some(d.name.clone()), dual, start, stop));
        }
    }
    out
}
