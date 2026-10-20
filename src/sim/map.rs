//! Mapping of the discretised csx sources, lumped elements, probes and
//! dumps ([`ModelSources`]) onto engine operators. The semantics are
//! documented on [`Simulation`](super::Simulation).

use std::collections::HashMap;

use csx::properties::{DumpMode, ExcitationType, LumpedElementType, ProbeType};

use super::results::ProbeKind;
use crate::Result;
use crate::axis::Axis;
use crate::compile::{ModelSources, NodeMap};
use crate::engine::cylindrical;
use crate::engine::port::loop_entries;
use crate::engine::{
    DumpPosition, DumpQuantity, DumpTiming, EdgeDft, EdgeSource, EdgeValues, Engine, FieldProbe,
    FieldSampler, FreqDump, LumpedElement, OperatorHandle, PeriodicPlaneWave, PlaneWave, Probe,
    ProbeQuantity, Schedule, SourceKind, TimeDump, Topology, WavePart,
};
use crate::fdtd::Fdtd;
use crate::io::openems::dump as oe;

/// The engine operator behind a model probe.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProbeOp {
    /// Voltage or current: a weighted sum of edge values.
    Scalar(OperatorHandle<Probe>),
    /// E or H field vector at a point.
    Field(OperatorHandle<FieldProbe>),
}

/// A mapped ProbeBox.
#[derive(Debug, Clone)]
pub(crate) struct ProbeEntry {
    pub name: String,
    pub id: u32,
    pub number: u32,
    pub kind: ProbeKind,
    /// Result weight, applied to field probes when reading (already in the
    /// entries of scalar probes).
    pub weight: f64,
    pub op: ProbeOp,
    pub frequencies: Vec<f64>,
    pub start_time: f64,
    pub stop_time: f64,
}

/// The engine operator behind a model dump.
#[derive(Debug, Clone, Copy)]
pub(crate) enum DumpOp {
    Time(OperatorHandle<TimeDump>),
    Freq(OperatorHandle<FreqDump>),
}

/// A mapped DumpBox.
#[derive(Debug, Clone)]
pub(crate) struct DumpEntry {
    pub name: String,
    pub id: u32,
    pub file_type: csx::properties::DumpFileType,
    pub dump_type: csx::properties::DumpType,
    pub op: DumpOp,
}

/// A mapped SAR dump: index into [`ModelSources::sar`] and its edge DFT.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SarEntry {
    pub spec: usize,
    pub op: OperatorHandle<EdgeDft>,
    /// The cell-interpolated E DFT of the openEMS SAR files.
    pub oe: Option<OperatorHandle<FreqDump>>,
}

/// Everything [`map_sources`] registered.
#[derive(Debug, Clone, Default)]
pub(crate) struct Mapped {
    pub lumped: Vec<(String, OperatorHandle<LumpedElement>)>,
    /// csx property id of each entry of `lumped` (filled by the caller).
    pub lumped_ids: Vec<u32>,
    pub sources: Vec<(String, OperatorHandle<EdgeSource>)>,
    pub plane_waves: Vec<(String, OperatorHandle<PlaneWave>)>,
    /// Plane waves of a Bloch run (periodic-cell sources).
    pub periodic_waves: Vec<(String, OperatorHandle<PeriodicPlaneWave>)>,
    pub probes: Vec<ProbeEntry>,
    pub dumps: Vec<DumpEntry>,
    pub sar: Vec<SarEntry>,
}

/// A csx lumped element being assembled.
struct Group {
    name: String,
    axis: Axis,
    topology: Topology,
    caps: bool,
    edges: Vec<EdgeValues>,
    /// Whether an excitation on its edges can act as a Thevenin source.
    can_source: bool,
    /// Per-edge voltage weights of each overlapping excitation, keyed by
    /// excitation index.
    sources: Vec<(usize, Vec<f64>)>,
    /// Cylindrical grids: every edge csx listed, copies included.
    raw: Vec<[usize; 3]>,
}

/// `(series, parallel)` edge counts of a set of edge start nodes along
/// `axis`: distinct positions along the axis, and edges per position.
fn counts(axis: Axis, nodes: impl Iterator<Item = [usize; 3]>) -> (f64, f64) {
    let mut n = 0usize;
    let mut along = std::collections::BTreeSet::new();
    for p in nodes {
        n += 1;
        along.insert(p[axis.index()]);
    }
    let s = along.len().max(1) as f64;
    (s, n as f64 / s)
}

/// Cylindrical grids: csx spreads an element over the edges it lists,
/// `R·p/s`, `L·p/s`, `C·s/p` per edge (`s` in series, `p` in parallel),
/// counting every stored copy of an edge (the α copies on the axis, both
/// α faces of a closed grid) as a parallel edge. Only one copy per
/// physical edge is kept, so the per-edge values are rescaled to the kept
/// counts (`R_e' = R_e·(s/p)·(p'/s')`), which keeps the element's totals.
fn redistribute(g: &mut Group) {
    let (s0, p0) = counts(g.axis, g.raw.iter().copied());
    let (s1, p1) = counts(g.axis, g.edges.iter().map(|e| e.node));
    if g.edges.is_empty() || (s0 == s1 && p0 == p1) {
        return;
    }
    let f = (s0 / p0) * (p1 / s1);
    for e in &mut g.edges {
        e.r = e.r.map(|r| r * f);
        e.l = e.l.map(|l| l * f);
        e.c = e.c.map(|c| c / f);
    }
}

/// Positive finite value, or `None` (csx uses NaN for "unset").
fn value(v: f64, what: &str, name: &str, warnings: &mut Vec<String>) -> Option<f64> {
    if v.is_nan() {
        None
    } else if v.is_finite() && v > 0.0 {
        Some(v)
    } else {
        warnings.push(format!(
            "lumped element {name:?}: {what} = {v} is not positive; treated as absent"
        ));
        None
    }
}

/// Sampling interval (steps) of dumps: the Nyquist interval of the highest
/// frequency of interest divided by the oversampling factor (default 4).
pub(crate) fn dump_interval(fdtd: &Fdtd, dt: f64) -> u64 {
    let Some(f) = fdtd
        .get_max_frequency()
        .filter(|f| f.is_finite() && *f > 0.0)
    else {
        return 1;
    };
    let os = fdtd.get_oversampling().unwrap_or(4).max(1) as f64;
    let k = (1.0 / (2.0 * f * os * dt)).floor();
    if k.is_finite() && k >= 1.0 {
        k as u64
    } else {
        1
    }
}

/// The openEMS Nyquist step count of dumps: from the excitation's highest
/// frequency (`f0 + fc` for a Gaussian, `f0` for sinusoids and custom
/// signals; openEMS ignores the `<FDTD f_max>` attribute here), else the
/// explicit maximum frequency; 1 without either.
fn openems_nyquist(fdtd: &Fdtd, dt: f64) -> u64 {
    let f = fdtd
        .get_waveform()
        .and_then(|w| match w {
            crate::Waveform::Expression(e) if e.f0() > 0.0 => Some(e.f0()),
            w => w.max_frequency(),
        })
        .or_else(|| fdtd.get_explicit_max_frequency())
        .filter(|f| f.is_finite() && *f > 0.0);
    f.map_or(1, |f| oe::nyquist_steps(f, dt))
}

/// The sample indices of a dump along each axis (openEMS snapping, see
/// [`oe::sample_indices`]); `None` if the box is outside the mesh.
fn dump_indices(
    grid: &crate::Grid,
    d: &crate::compile::ProbeSpec<csx::properties::DumpBox>,
    pos: DumpPosition,
    delta_unit: f64,
) -> Option<[Vec<usize>; 3]> {
    let s = &d.settings;
    let mut out: [Vec<usize>; 3] = Default::default();
    for a in Axis::ALL {
        let i = a.index();
        let l = grid.lines(a);
        let n = l.len();
        let cand: Vec<f64> = if pos == DumpPosition::CellCenters {
            (0..n)
                .map(|j| {
                    if j + 1 < n {
                        0.5 * (l[j] + l[j + 1])
                    } else if n >= 2 {
                        l[n - 1] + 0.5 * (l[n - 1] - l[n - 2])
                    } else {
                        l[0]
                    }
                })
                .collect()
        } else {
            l.to_vec()
        };
        let stride = s.sub_sampling.map_or(1, |v| v[i].max(1) as usize);
        let unit = if grid.is_cylindrical() && a == Axis::Y {
            1.0
        } else {
            delta_unit
        };
        let res = s
            .opt_resolution
            .as_ref()
            .map(|r| r[i].value() * unit)
            .filter(|r| r.is_finite() && *r > 0.0);
        out[i] = oe::sample_indices(
            &cand,
            (l[0], l[n - 1]),
            d.corners[0][i],
            d.corners[1][i],
            stride,
            res,
        )?;
    }
    Some(out)
}

/// The engine of a Bloch run being mapped: which part of the complex field
/// it carries, and the transverse wavevector.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlochMap {
    pub part: WavePart,
    pub k: [f64; 3],
}

/// Registers the operators for `src` on `engine`; see the
/// [`Simulation`](super::Simulation) docs for the semantics. In a Bloch
/// run (`bloch`), the imaginary engine gets no excitations (sources,
/// Thevenin sources) other than its part of the plane waves, plane waves
/// become [`PeriodicPlaneWave`]s, and SAR dumps are not supported.
pub(crate) fn map_sources(
    engine: &mut Engine,
    src: &ModelSources,
    node_map: &NodeMap,
    fdtd: &Fdtd,
    warnings: &mut Vec<String>,
    bloch: Option<BlochMap>,
) -> Result<Mapped> {
    let mut mapped = Mapped::default();
    let dt = engine.dt();
    let nodes = engine.grid().nodes();
    let idx = |p: [usize; 3]| (p[0] * nodes[1] + p[1]) * nodes[2] + p[2];

    // ---- lumped elements, grouped by property name (first-seen order).
    let mut groups: Vec<Group> = Vec::new();
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    // (axis, node) → (group, edge position).
    let mut lumped_edge: HashMap<(Axis, [usize; 3]), (usize, usize)> = HashMap::new();
    let cyl_grid = engine.grid().clone();
    let cylindrical = cyl_grid.is_cylindrical();
    // Cylindrical grids: (group, canonical edge) pairs already placed.
    let mut placed = std::collections::HashSet::new();
    for le in &src.lumped {
        let (axis, raw_node) = node_map.edge(le.edge);
        // A lumped element acts on the physical edge: on a cylindrical grid
        // its canonical copy carries it (see
        // `engine::cylindrical::e_edge_copies`); further copies are skipped
        // and the per-edge values redistributed below.
        let node = if cylindrical {
            cylindrical::e_edge_copies(&cyl_grid, axis, raw_node)[0]
        } else {
            raw_node
        };
        let g = *by_name.entry(le.name.as_str()).or_insert_with(|| {
            let topology = match le.le_type {
                None | Some(LumpedElementType::Parallel) => Topology::Parallel,
                Some(LumpedElementType::Series) => Topology::Series,
                Some(other) => {
                    warnings.push(format!(
                        "lumped element {:?}: unknown LEtype {other}; parallel assumed",
                        le.name
                    ));
                    Topology::Parallel
                }
            };
            groups.push(Group {
                name: le.name.clone(),
                axis,
                topology,
                caps: le.caps,
                edges: Vec::new(),
                can_source: false,
                sources: Vec::new(),
                raw: Vec::new(),
            });
            groups.len() - 1
        });
        let grp = &mut groups[g];
        if cylindrical {
            grp.raw.push(raw_node);
            if !placed.insert((g, axis, node)) {
                continue;
            }
        }
        if axis != grp.axis {
            warnings.push(format!(
                "lumped element {:?}: edge {axis}:{node:?} not along {}; ignored",
                le.name, grp.axis
            ));
            continue;
        }
        if engine.coefficients().cb(axis)[idx(node)] == 0.0 {
            warnings.push(format!(
                "lumped element {:?}: edge {axis}:{node:?} is PEC (shorted); ignored",
                le.name
            ));
            continue;
        }
        if lumped_edge.contains_key(&(axis, node)) {
            warnings.push(format!(
                "lumped element {:?}: edge {axis}:{node:?} already carries a lumped element; ignored",
                le.name
            ));
            continue;
        }
        let ev = EdgeValues {
            node,
            r: value(le.r, "R", &le.name, warnings),
            l: value(le.l, "L", &le.name, warnings),
            c: value(le.c, "C", &le.name, warnings),
        };
        if ev.r.is_none() && ev.l.is_none() && ev.c.is_none() {
            continue;
        }
        lumped_edge.insert((axis, node), (g, grp.edges.len()));
        grp.edges.push(ev);
    }
    if cylindrical {
        for g in &mut groups {
            redistribute(g);
        }
    }
    for g in &mut groups {
        g.can_source = match g.topology {
            Topology::Parallel => g.edges.first().is_some_and(|e| e.r.is_some()),
            Topology::Series => !g.edges.is_empty(),
        };
    }

    // ---- excitations.
    let waveform = fdtd.get_waveform();
    if waveform.is_none() && !(src.excitations.is_empty() && src.plane_waves.is_empty()) {
        warnings
            .push("the settings have no excitation waveform; all excitations are ignored".into());
    }
    let grid = engine.grid().clone();
    let mut soft_e = Vec::new();
    let mut others = Vec::new();
    // The imaginary engine of a Bloch run is driven through the periodic
    // faces only (and its part of the plane waves).
    let imag = bloch.is_some_and(|b| b.part == WavePart::Imag);
    for (k, ex) in src.excitations.iter().enumerate() {
        let Some(w) = waveform.filter(|_| !imag) else {
            break;
        };
        let kind = match ex.excitation_type {
            ExcitationType::EFieldSoft => SourceKind::SoftE,
            ExcitationType::EFieldHard => SourceKind::HardE,
            ExcitationType::HFieldSoft => SourceKind::SoftH,
            ExcitationType::HFieldHard => SourceKind::HardH,
            // Reported by the compile step (csx does not discretise them).
            _ => continue,
        };
        let (wf, delay) = (w.clone(), ex.delay);
        let mut op = EdgeSource::new(move |t| wf.value(t - delay, dt)).kind(kind);
        let mut thevenin = 0;
        let mut hard_on_lumped = 0usize;
        // Cylindrical grids: physical edges already handled (canonical copy).
        let mut seen = std::collections::HashSet::new();
        for &(edge, a) in &ex.amplitudes {
            let (axis, node) = node_map.edge(edge);
            let copies = if grid.is_cylindrical() && !kind.is_magnetic() {
                let c = cylindrical::e_edge_copies(&grid, axis, node);
                if !seen.insert((axis, c[0])) {
                    continue;
                }
                c
            } else {
                vec![node]
            };
            let node = copies[0];
            let len = if kind.is_magnetic() {
                grid.dual_edge_length(axis, node)
            } else {
                grid.edge_length(axis, node)
            };
            let weight = a * len;
            if kind == SourceKind::SoftE
                && let Some(&(g, e)) = lumped_edge.get(&(axis, node))
                && groups[g].can_source
            {
                let grp = &mut groups[g];
                let ne = grp.edges.len();
                let pos = match grp.sources.iter().position(|(j, _)| *j == k) {
                    Some(p) => p,
                    None => {
                        grp.sources.push((k, vec![0.0; ne]));
                        grp.sources.len() - 1
                    }
                };
                grp.sources[pos].1[e] += weight;
                thevenin += 1;
                continue;
            }
            if kind == SourceKind::SoftE && lumped_edge.contains_key(&(axis, node)) {
                warnings.push(format!(
                    "excitation {:?}: edge {axis}:{node:?} has a lumped element without \
                     resistance; excited as a plain soft source",
                    ex.name
                ));
            }
            if kind == SourceKind::HardE && lumped_edge.contains_key(&(axis, node)) {
                hard_on_lumped += 1;
            }
            for &c in &copies {
                op = op.edge(axis, c, weight);
            }
        }
        if hard_on_lumped > 0 {
            warnings.push(format!(
                "excitation {:?}: hard E source on {hard_on_lumped} edge(s) of a lumped \
                 element; the source overwrites the edge voltage and the element has no \
                 effect there",
                ex.name
            ));
        }
        let has_edges = op.edges().next().is_some();
        if has_edges {
            if kind == SourceKind::SoftE {
                soft_e.push((ex.name.clone(), op));
            } else {
                others.push((ex.name.clone(), op));
            }
        } else if thevenin == 0 {
            warnings.push(format!("excitation {:?}: no edges; ignored", ex.name));
        }
    }

    // ---- install lumped elements (caps first), then sources.
    for g in groups {
        if g.edges.is_empty() {
            warnings.push(format!(
                "lumped element {:?}: no usable edges; ignored",
                g.name
            ));
            continue;
        }
        if g.caps {
            add_caps(engine, g.axis, &g.edges);
        }
        let ne = g.edges.len();
        let mut el = LumpedElement::from_edges(g.axis, g.topology, g.edges)?.record_current();
        for (k, weights) in g.sources {
            debug_assert_eq!(weights.len(), ne);
            let ex = &src.excitations[k];
            let wf = waveform.expect("sources only with a waveform").clone();
            let delay = ex.delay;
            el = el.edge_source(weights, move |t| wf.value(t - delay, dt));
        }
        let h = el.add_to(engine)?;
        mapped.lumped.push((g.name, h));
    }
    // Soft E sources add after the lumped updates; hard sources overwrite
    // last; H sources act in post_h.
    for (name, op) in soft_e.into_iter().chain(others) {
        let h = engine.add_operator(op);
        mapped.sources.push((name, h));
    }
    // Plane waves last: their corrections add to the final kernel values.
    for pw in &src.plane_waves {
        let Some(w) = waveform else { break };
        // openEMS's phase reference (design §5.16); not measured for the
        // periodic (Bloch) variant, which keeps ems's.
        let lag = match bloch {
            None => crate::compile::planewave::openems_lag(&pw.spec, engine.grid(), dt),
            Some(_) => 0.0,
        };
        let (wf, delay) = (w.clone(), pw.delay + lag);
        let amplitude = move |t| wf.value(t - delay, dt);
        if let Some(b) = bloch {
            let spec = match super::bloch::periodic_spec(pw, fdtd, nodes, b.k) {
                Ok(s) => s,
                Err(w) => {
                    warnings.push(w);
                    continue;
                }
            };
            if fdtd.get_bloch_wavevector().is_some()
                && let Some(w) = super::bloch::direction_mismatch(pw, fdtd, &spec)
            {
                warnings.push(w);
            }
            match PeriodicPlaneWave::new(engine, &spec, b.part, amplitude) {
                Ok(op) => mapped
                    .periodic_waves
                    .push((pw.name.clone(), engine.add_operator(op))),
                Err(e) => warnings.push(format!("plane wave {:?}: {e}; ignored", pw.name)),
            }
            continue;
        }
        match PlaneWave::new(engine, &pw.spec, amplitude) {
            Ok(op) => mapped
                .plane_waves
                .push((pw.name.clone(), engine.add_operator(op))),
            Err(e) => warnings.push(format!("plane wave {:?}: {e}; ignored", pw.name)),
        }
    }

    // ---- probes.
    for p in &src.probes {
        let s = &p.settings;
        let weight = s.weight.value();
        let kind = match s.probe_type {
            ProbeType::Voltage => ProbeKind::Voltage,
            ProbeType::Current => ProbeKind::Current,
            ProbeType::EField => ProbeKind::EField,
            ProbeType::HField => ProbeKind::HField,
            ProbeType::WaveguideVoltage | ProbeType::WaveguideCurrent => {
                // Mode-weighted sums computed by the compile step.
                let (q, kind) = if s.probe_type == ProbeType::WaveguideVoltage {
                    (ProbeQuantity::Voltage, ProbeKind::Voltage)
                } else {
                    (ProbeQuantity::Current, ProbeKind::Current)
                };
                let mut pr = Probe::new(q);
                for &(axis, node, w) in &p.mode_entries {
                    pr = pr.entry(axis, node, w * weight);
                }
                mapped.probes.push(ProbeEntry {
                    name: p.name.clone(),
                    id: p.id,
                    number: s.number,
                    kind,
                    weight,
                    op: ProbeOp::Scalar(engine.add_operator(pr)),
                    frequencies: s.frequencies.clone(),
                    start_time: s.start_time.value(),
                    stop_time: s.stop_time.value(),
                });
                continue;
            }
            // Reported by the compile step.
            _ => continue,
        };
        let op = match kind {
            ProbeKind::Voltage => {
                let mut pr = Probe::new(ProbeQuantity::Voltage);
                let path = voltage_path(p.start, p.stop);
                if path.is_empty() {
                    warnings.push(format!(
                        "voltage probe {:?}: start and stop snap to the same node; it reads 0",
                        p.name
                    ));
                }
                for (axis, node, w) in path {
                    pr = pr.entry(axis, node, w * weight);
                }
                ProbeOp::Scalar(engine.add_operator(pr))
            }
            ProbeKind::Current => {
                let Some(entries) = current_loop(p.start, p.stop, s.norm_dir, nodes) else {
                    warnings.push(format!(
                        "current probe {:?}: no normal direction (set NormDir or use a \
                         box that is flat along one axis); ignored",
                        p.name
                    ));
                    continue;
                };
                let mut pr = Probe::new(ProbeQuantity::Current);
                for (axis, node, w) in entries {
                    pr = pr.entry(axis, node, w * weight);
                }
                ProbeOp::Scalar(engine.add_operator(pr))
            }
            ProbeKind::EField | ProbeKind::HField => {
                let q = if kind == ProbeKind::EField {
                    DumpQuantity::E
                } else {
                    DumpQuantity::H
                };
                // The centre of the (snapped) box: the node itself for a
                // point probe, interpolated otherwise.
                let centre = Axis::ALL.map(|a| {
                    let l = grid.lines(a);
                    let d = a.index();
                    0.5 * (l[p.start[d]] + l[p.stop[d]])
                });
                ProbeOp::Field(engine.add_operator(FieldProbe::at(&grid, q, centre)?))
            }
        };
        mapped.probes.push(ProbeEntry {
            name: p.name.clone(),
            id: p.id,
            number: s.number,
            kind,
            weight,
            op,
            frequencies: s.frequencies.clone(),
            start_time: s.start_time.value(),
            stop_time: s.stop_time.value(),
        });
    }

    // ---- dumps (openEMS sampling rules, see `io::openems::dump`).
    let k = dump_interval(fdtd, dt);
    let nyquist = openems_nyquist(fdtd, dt);
    let td_interval = oe::time_domain_interval(nyquist, fdtd.get_oversampling().unwrap_or(4));
    for d in &src.dumps {
        let s = &d.settings;
        let id = s.dump_type.id();
        let fd = (10..=15).contains(&id);
        let q = match id {
            0 | 10 => DumpQuantity::E,
            1 | 11 => DumpQuantity::H,
            2 | 12 => DumpQuantity::J,
            3 | 13 => DumpQuantity::RotH,
            4 | 14 => DumpQuantity::D,
            5 | 15 => DumpQuantity::B,
            // SAR dumps below; unknown types are reported by the compile step.
            _ => continue,
        };
        if fd && s.probe.frequencies.is_empty() {
            continue;
        }
        let pos = match s.dump_mode {
            DumpMode::Cell => DumpPosition::CellCenters,
            DumpMode::NoInterpolation => DumpPosition::Native,
            _ => DumpPosition::Nodes,
        };
        let Some(indices) = dump_indices(&grid, d, pos, src.delta_unit) else {
            warnings.push(format!(
                "dump {:?}: box is outside the mesh; disabled",
                d.name
            ));
            continue;
        };
        let sampler = FieldSampler::openems(&grid, indices, q, pos)?.timing(DumpTiming::OpenEms);
        let (first, last) =
            oe::step_window(s.probe.start_time.value(), s.probe.stop_time.value(), dt);
        let op = if fd {
            let kf = oe::frequency_domain_interval(nyquist, &s.probe.frequencies, dt);
            let op = FreqDump::new(sampler, s.probe.frequencies.clone())
                .interval(kf)
                .step_window(first, last);
            DumpOp::Freq(engine.add_operator(op))
        } else {
            let op = TimeDump::new(sampler)
                .schedule(Schedule::Every(td_interval))
                .step_window(first, last);
            DumpOp::Time(engine.add_operator(op))
        };
        mapped.dumps.push(DumpEntry {
            name: d.name.clone(),
            id: d.id,
            file_type: s.file_type,
            dump_type: s.dump_type,
            op,
        });
    }

    // ---- SAR dumps (edge DFTs, evaluated by the results).
    for (spec, s) in src.sar.iter().enumerate() {
        if bloch.is_some() {
            warnings.push(format!(
                "SAR dump {:?}: not supported in a Bloch run; ignored",
                s.name
            ));
            continue;
        }
        let op = EdgeDft::new(s.cells, s.frequencies.clone()).interval(k);
        let oe = match &s.openems {
            Some(c) => {
                let sampler = FieldSampler::openems(
                    &grid,
                    c.indices.clone(),
                    DumpQuantity::E,
                    DumpPosition::CellCenters,
                )?
                .timing(DumpTiming::OpenEms);
                let (first, last) =
                    src.dumps
                        .iter()
                        .find(|d| d.id == s.id)
                        .map_or((0, None), |d| {
                            let p = &d.settings.probe;
                            oe::step_window(p.start_time.value(), p.stop_time.value(), dt)
                        });
                let kf = oe::frequency_domain_interval(nyquist, &s.frequencies, dt);
                let f = FreqDump::new(sampler, s.frequencies.clone())
                    .interval(kf)
                    .step_window(first, last);
                Some(engine.add_operator(f))
            }
            None => None,
        };
        mapped.sar.push(SarEntry {
            spec,
            op: engine.add_operator(op),
            oe,
        });
    }
    Ok(mapped)
}

/// PEC end caps of a lumped element: the transverse E edges of its two end
/// planes, inside the transverse extent of its edges.
fn add_caps(engine: &mut Engine, axis: Axis, edges: &[EdgeValues]) {
    let a = axis.index();
    let mut lo = [usize::MAX; 3];
    let mut hi = [0; 3];
    for e in edges {
        for d in 0..3 {
            lo[d] = lo[d].min(e.node[d]);
            hi[d] = hi[d].max(e.node[d] + usize::from(d == a));
        }
    }
    let nodes = engine.grid().nodes();
    let idx = |p: [usize; 3]| (p[0] * nodes[1] + p[1]) * nodes[2] + p[2];
    let (b, c) = axis.others();
    let mut co = engine.coefficients_mut();
    for (t, o) in [(b, c), (c, b)] {
        let (ti, oi) = (t.index(), o.index());
        if hi[ti] == lo[ti] {
            continue;
        }
        let (ca, cb) = co.e_mut(t);
        for plane in [lo[a], hi[a]] {
            for pt in lo[ti]..hi[ti] {
                for po in lo[oi]..=hi[oi] {
                    let mut p = [0; 3];
                    p[a] = plane;
                    p[ti] = pt;
                    p[oi] = po;
                    let i = idx(p);
                    ca[i] = 0.0;
                    cb[i] = 0.0;
                }
            }
        }
    }
}

/// E edges `(axis, node, ±1)` of the staircase path from `start` to `stop`:
/// along x (at the start's y, z), then y (at the stop's x), then z.
pub(crate) fn voltage_path(start: [usize; 3], stop: [usize; 3]) -> Vec<(Axis, [usize; 3], f64)> {
    let mut v = Vec::new();
    let mut p = start;
    for axis in Axis::ALL {
        let a = axis.index();
        let (from, to) = (p[a], stop[a]);
        if from <= to {
            for i in from..to {
                p[a] = i;
                v.push((axis, p, 1.0));
            }
        } else {
            for i in (to..from).rev() {
                p[a] = i;
                v.push((axis, p, -1.0));
            }
        }
        p[a] = to;
    }
    v
}

/// Dual edges `(axis, node, weight)` whose weighted sum is the current
/// through the probe box from `start` towards `stop` along the normal
/// (`+normal` for a flat box), or `None` without a normal direction.
pub(crate) fn current_loop(
    start: [usize; 3],
    stop: [usize; 3],
    norm_dir: Option<csx::Axis>,
    nodes: [usize; 3],
) -> Option<Vec<(Axis, [usize; 3], f64)>> {
    let n = match norm_dir {
        Some(d) => Axis::from_index(d.index())?,
        None => {
            let flat: Vec<Axis> = Axis::ALL
                .into_iter()
                .filter(|a| start[a.index()] == stop[a.index()])
                .collect();
            match flat[..] {
                [a] => a,
                _ => return None,
            }
        }
    };
    let ni = n.index();
    if nodes[ni] < 2 {
        return None;
    }
    let lo = [0, 1, 2].map(|d| start[d].min(stop[d]));
    let hi = [0, 1, 2].map(|d| start[d].max(stop[d]));
    let layer = if lo[ni] == hi[ni] {
        lo[ni].min(nodes[ni] - 2)
    } else {
        lo[ni] + (hi[ni] - lo[ni] - 1) / 2
    };
    let sign = if start[ni] <= stop[ni] { 1.0 } else { -1.0 };
    // loop_entries gives −∮H (right-handed about +n).
    Some(
        loop_entries(n, lo, hi, layer)
            .into_iter()
            .map(|(a, p, w)| (a, p, -w * sign))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voltage_path_signs() {
        let p = voltage_path([1, 2, 3], [1, 2, 5]);
        assert_eq!(
            p,
            vec![(Axis::Z, [1, 2, 3], 1.0), (Axis::Z, [1, 2, 4], 1.0)]
        );
        let p = voltage_path([1, 2, 5], [1, 2, 3]);
        assert_eq!(
            p,
            vec![(Axis::Z, [1, 2, 4], -1.0), (Axis::Z, [1, 2, 3], -1.0)]
        );
        let p = voltage_path([2, 0, 0], [0, 1, 0]);
        assert_eq!(
            p,
            vec![
                (Axis::X, [1, 0, 0], -1.0),
                (Axis::X, [0, 0, 0], -1.0),
                (Axis::Y, [0, 0, 0], 1.0)
            ]
        );
        assert!(voltage_path([1, 1, 1], [1, 1, 1]).is_empty());
    }

    #[test]
    fn current_loop_normal_and_sign() {
        let nodes = [10, 10, 10];
        // Flat box normal to z: +z.
        let a = current_loop([3, 4, 2], [3, 4, 2], None, nodes);
        assert!(a.is_none(), "a point has no unique flat axis");
        let a = current_loop([3, 3, 2], [4, 4, 2], None, nodes).unwrap();
        let b = current_loop([3, 3, 2], [4, 4, 2], Some(csx::Axis::Z), nodes).unwrap();
        assert_eq!(a, b);
        // Reversed along the normal: opposite sign, same layer.
        let up = current_loop([3, 4, 1], [3, 4, 4], Some(csx::Axis::Z), nodes).unwrap();
        let down = current_loop([3, 4, 4], [3, 4, 1], Some(csx::Axis::Z), nodes).unwrap();
        assert_eq!(up.len(), down.len());
        for (u, d) in up.iter().zip(&down) {
            assert_eq!((u.0, u.1), (d.0, d.1));
            assert_eq!(u.2, -d.2);
        }
        // +∮H about +z around the column (3, 4): +H_y at x=3, −H_y at x=2, …
        assert!(up.contains(&(Axis::Y, [3, 4, 2], 1.0)));
        assert!(up.contains(&(Axis::X, [3, 4, 2], -1.0)));
    }
}
