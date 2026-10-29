//! Mapping of the discretised csx sources, lumped elements, probes and
//! dumps ([`ModelSources`]) onto engine operators. The semantics are
//! documented on [`Simulation`](super::Simulation).

use std::collections::HashMap;

use csx::properties::{DumpMode, ExcitationType, LumpedElementType, ProbeType};

use super::results::{ProbeKind, ProbeLocation};
use crate::Result;
use crate::axis::Axis;
use crate::compile::{ModelSources, NodeMap};
use crate::engine::boundary::PmlLayers;
use crate::engine::cylindrical;
use crate::engine::port::loop_entries;
use crate::engine::{CoeffRounding, EdgeMaterials, e_coefficients, e_edge_circuit};
use crate::engine::{
    DumpPosition, DumpQuantity, DumpTiming, EdgeDft, EdgeSource, EdgeValues, Engine, FieldSampler,
    FreqDump, LumpedElement, OperatorHandle, PeriodicPlaneWave, PlaneWave, Probe, ProbeQuantity,
    Schedule, SourceKind, TimeDump, Topology, WavePart,
};
use crate::fdtd::Fdtd;
use crate::io::openems::dump as oe;

/// The engine operator behind a model probe.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProbeOp {
    /// Voltage or current: a weighted sum of edge values.
    Scalar(OperatorHandle<Probe>),
    /// E or H field vector: the native components, one weighted edge
    /// each (x, y, z).
    Native([OperatorHandle<Probe>; 3]),
}

/// A mapped ProbeBox.
#[derive(Debug, Clone)]
pub(crate) struct ProbeEntry {
    pub name: String,
    pub id: u32,
    pub number: u32,
    pub kind: ProbeKind,
    pub op: ProbeOp,
    pub frequencies: Vec<f64>,
    pub start_time: f64,
    pub stop_time: f64,
    /// Snapped position as openEMS reports it (waveguide probes: the
    /// range of openEMS's samples; none on cylindrical grids).
    pub location: Option<ProbeLocation>,
    /// A waveguide (mode-matching) probe, which records the mode purity
    /// too.
    pub waveguide: bool,
    /// H-type probes after a run: the sample at `(N + ½)·dt` openEMS
    /// records after the last step `N` (time, values, mode purity).
    pub final_sample: Option<(f64, Vec<f64>, Option<f64>)>,
    /// The probe's own `OverSampling` ([`ProbeSpec::oversampling`]).
    pub oversampling: Option<u32>,
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
    /// openEMS files: the E edges whose circuit a parallel R/C lumped
    /// element replaced, with their capacitance (`(axis, node, C)`), for
    /// the openEMS poles on them (`Dispersive::rebase_openems`).
    pub circuits: Vec<(Axis, [usize; 3], f64)>,
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
    /// Edges inside openEMS's PML layers, taken out of `edges` (their
    /// caps stay).
    in_layer: Vec<EdgeValues>,
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

/// `A_e/l_e` of an E edge as openEMS weights it when it spreads a lumped
/// element (black box, design §5.16.1): the dual face area over the edge
/// length, with the dual extent of an outer mesh line taken as a full
/// cell (openEMS's operator gives boundary edges the capacitance of a full
/// dual cell) and, on a closed azimuth, the two stored copies of a seam
/// edge counted once with the full `Δα`.
fn openems_edge_geometry(grid: &crate::Grid, axis: Axis, node: [usize; 3]) -> f64 {
    let mut area = grid.dual_face_area(axis, node);
    let cyl = grid.is_cylindrical();
    for t in Axis::ALL {
        if t == axis {
            continue;
        }
        let n = grid.lines(t).len();
        let i = node[t.index()];
        let outer = n > 1 && (i == 0 || i + 1 == n);
        if !outer {
            continue;
        }
        if cyl && t == Axis::X {
            // ρ: the axis keeps its disc; elsewhere the clipped dual ring
            // is extended by the missing half cell.
            if i == 0 && grid.has_axis() {
                continue;
            }
            if axis == Axis::Z {
                let r = grid.lines(Axis::X);
                let (lo, hi) = if i == 0 {
                    (r[0], 0.5 * (r[0] + r[1]))
                } else {
                    (0.5 * (r[i - 1] + r[i]), r[i])
                };
                let (lo2, hi2) = if i == 0 {
                    (2.0 * lo - hi, hi)
                } else {
                    (lo, 2.0 * hi - lo)
                };
                area *= (hi2 * hi2 - lo2 * lo2) / (hi * hi - lo * lo);
                continue;
            }
        }
        area *= 2.0;
    }
    area / grid.edge_length(axis, node)
}

/// Re-spreads a csx lumped element over its edges the way openEMS does
/// (black box, design §5.16.1). A parallel R/C element without an
/// inductor acts like a homogeneous material filling its edges
/// (conductivity `κ`, permittivity `ε` uniform) rather than in equal parts
/// per edge: with `g_e` the [`openems_edge_geometry`] of edge `e` and
/// `S = Σ_k 1/Σ_{e∈k} g_e` (series positions `k` along the axis, edges `e`
/// in parallel at each), every edge gets `G_e = g_e·S/R` and
/// `C_e = g_e·C·S`, which keeps the element's total `R` and `C` (series sum
/// of parallel sums). Elements with an inductor and series elements keep
/// csx's equal parts (`R·p/s`, `L·p/s`, `C·s/p`). On a uniform mesh both
/// are the same; on graded meshes and cylindrical grids (`g_e ∝ ρ` for a
/// radial element) they differ. The element's values are recovered from
/// csx's distribution over every edge it listed (copies included); the
/// totals refer to the physical edges: openEMS counts the α = 2π copies of
/// a closed azimuth as extra parallel edges (a 20 Ω ring on 16 α cells
/// acts as 21.25 Ω there), ems keeps the nominal value.
fn spread_like_openems(g: &mut Group, grid: &crate::Grid) {
    let material = g.topology == Topology::Parallel && g.edges.iter().all(|e| e.l.is_none());
    if g.edges.is_empty() || !material {
        if grid.is_cylindrical() {
            redistribute(g);
        }
        return;
    }
    let (s0, p0) = counts(g.axis, g.raw.iter().copied());
    let a = g.axis.index();
    let geo: Vec<f64> = g
        .edges
        .iter()
        .map(|e| openems_edge_geometry(grid, g.axis, e.node))
        .collect();
    let mut per: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for (e, &w) in g.edges.iter().zip(&geo) {
        *per.entry(e.node[a]).or_insert(0.0) += w;
    }
    let sum: f64 = per.values().map(|w| 1.0 / w).sum();
    if !(sum.is_finite() && sum > 0.0) {
        return;
    }
    // csx: R_e = R·p/s, L_e = L·p/s, C_e = C·s/p.
    let first = g.edges[0];
    let r = first.r.map(|v| v * s0 / p0);
    let c = first.c.map(|v| v * p0 / s0);
    for (e, &w) in g.edges.iter_mut().zip(&geo) {
        e.r = r.map(|r| r / (w * sum));
        e.c = c.map(|c| c * w * sum);
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
/// Extra signal delay \[s\] that gives an engine source of `kind` the
/// openEMS excitation timing: the field value at time `T` (E at
/// `(n + 1)·dt`, H at `(n + ½)·dt`) receives the signal sample at
/// `T − dt` (openEMS's E excitation of step `n` uses its signal sample `n`
/// at `n·dt`, its H excitation the sample at `(n − ½)·dt`; established by
/// black-box comparison, see `tests/openems_core.rs`). The engine samples
/// at the times documented on [`SourceKind`], so the lag is `dt/2` for
/// soft sources (sampled mid-update) and `dt` for hard ones (sampled at
/// the field time).
pub(crate) fn openems_lag(kind: SourceKind, dt: f64) -> f64 {
    if kind.is_hard() { dt } else { 0.5 * dt }
}

/// The amplitude function of a model excitation with openEMS timing: the
/// waveform sampled at `τ = t − lag − D·dt` (`lag` from [`openems_lag`],
/// `D = ⌊Delay/dt⌋`: openEMS delays a signal by whole steps, rounded
/// down), and 0 before the signal's first sample (`τ < 0` on the E grid,
/// `τ < dt/2` on the H grid; openEMS has no samples before its index 0).
pub(crate) fn openems_signal(
    wf: crate::Waveform,
    delay: f64,
    lag: f64,
    dt: f64,
) -> impl Fn(f64) -> f64 + Send + 'static {
    let steps = if delay > 0.0 && dt > 0.0 {
        (delay / dt).floor()
    } else {
        0.0
    };
    let shift = lag + steps * dt;
    move |t| {
        let tau = t - shift;
        if tau < -0.25 * dt {
            0.0
        } else {
            wf.value(tau, dt)
        }
    }
}

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
    // Dirac and step: openEMS's Nyquist rate is 1 step (it prints the
    // frequency `1/(2·dt)`), whatever `f_max` says; dumps record every step.
    if matches!(
        fdtd.get_waveform(),
        Some(crate::Waveform::Dirac | crate::Waveform::Step)
    ) {
        return 1;
    }
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
/// [`oe::sample_indices`]); `None` if the box is outside the mesh. The box
/// snaps to the dual lines (cell centres) for cell interpolation and for
/// the raw values of the dual-mesh quantities H and B (`DumpMode` 0, as
/// openEMS: a box from 6 to 12 on a 1 mm mesh dumps H at 5.5 … 12.5).
fn dump_indices(
    grid: &crate::Grid,
    d: &crate::compile::ProbeSpec<csx::properties::DumpBox>,
    pos: DumpPosition,
    q: DumpQuantity,
    delta_unit: f64,
) -> Option<[Vec<usize>; 3]> {
    let dual = pos == DumpPosition::CellCenters
        || (pos == DumpPosition::Native && matches!(q, DumpQuantity::H | DumpQuantity::B));
    let s = &d.settings;
    let mut out: [Vec<usize>; 3] = Default::default();
    for a in Axis::ALL {
        let i = a.index();
        let l = grid.lines(a);
        let n = l.len();
        let cand: Vec<f64> = if dual {
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
        // Closed azimuth: openEMS dumps the N distinct α lines (no 2π
        // line) and their N dual lines (black box, design §5.15).
        let mut cand = cand;
        if a == Axis::Y && grid.is_closed() && n >= 2 {
            cand.truncate(n - 1);
        }
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
        let (mut lo, mut hi) = (d.corners[0][i], d.corners[1][i]);
        if lo > hi {
            std::mem::swap(&mut lo, &mut hi);
        }
        // Raw H and B: from the cell holding the lower corner (closed
        // above, `snap_cell`) to the cell holding the upper one (closed
        // below), as openEMS does on graded meshes (black box, design
        // §5.16.3; Cartesian grids); for a box reaching into the mesh.
        let tol = 1e-9 * (l[n - 1] - l[0]).abs();
        if dual
            && !grid.is_cylindrical()
            && pos == DumpPosition::Native
            && n >= 2
            && hi >= l[0] - tol
            && lo <= l[n - 1] + tol
        {
            let a = snap_cell(l, lo);
            let b = if hi - lo <= tol {
                a
            } else {
                l.partition_point(|&x| x <= hi + tol).saturating_sub(1)
            };
            let b = b.clamp(a, cand.len() - 1);
            lo = cand[a];
            hi = cand[b];
        }
        out[i] = oe::sample_indices(&cand, (l[0], l[n - 1]), lo, hi, stride, res)?;
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn map_sources(
    engine: &mut Engine,
    src: &ModelSources,
    node_map: &NodeMap,
    fdtd: &Fdtd,
    warnings: &mut Vec<String>,
    bloch: Option<BlochMap>,
    pml: Option<&PmlLayers>,
    medium: Option<(&EdgeMaterials, CoeffRounding)>,
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
                in_layer: Vec::new(),
            });
            groups.len() - 1
        });
        let grp = &mut groups[g];
        grp.raw.push(raw_node);
        if cylindrical && !placed.insert((g, axis, node)) {
            continue;
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
    if fdtd.get_openems_run() {
        for g in &mut groups {
            spread_like_openems(g, &cyl_grid);
        }
    } else if cylindrical {
        for g in &mut groups {
            redistribute(g);
        }
    }
    // openEMS's PML layer replaces the update of its edges: an element's
    // R and C have no effect there, with an inductor or in series openEMS
    // diverges (black box, design §5.16.2). The element keeps its values
    // per edge (spread over all its edges) and its caps.
    if let Some(layers) = pml {
        for g in &mut groups {
            let (inside, keep): (Vec<EdgeValues>, Vec<EdgeValues>) = g
                .edges
                .iter()
                .copied()
                .partition(|e| layers.contains(true, g.axis, e.node));
            if inside.is_empty() {
                continue;
            }
            let unstable = g.topology == Topology::Series || inside.iter().any(|e| e.l.is_some());
            warnings.push(format!(
                "lumped element {:?}: {} of its {} edge(s) lie inside a PML layer and are \
                 ignored there (openEMS's layer replaces their update{})",
                g.name,
                inside.len(),
                inside.len() + keep.len(),
                if unstable {
                    "; openEMS's run diverges with an inductor or a series element there"
                } else {
                    ""
                }
            ));
            g.edges = keep;
            g.in_layer = inside;
        }
        lumped_edge.clear();
        for (gi, g) in groups.iter().enumerate() {
            for (e, ev) in g.edges.iter().enumerate() {
                lumped_edge.insert((g.axis, ev.node), (gi, e));
            }
        }
    }
    // openEMS semantics (black box, design §5.16.1): an excitation over a
    // lumped element is a plain soft source on every element, and the
    // element's branch (inductor, series RLC) sees the edge voltage
    // without it; see the install below. ems semantics: a Thevenin source
    // in series with the resistor or the series branch.
    let openems = fdtd.get_openems_run();
    for g in &mut groups {
        g.can_source = openems
            || match g.topology {
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
        // openEMS timing: a field at time `T` gets the signal at `T − dt`
        // (see `openems_lag`).
        let mut op = EdgeSource::new(openems_signal(
            w.clone(),
            ex.delay,
            openems_lag(kind, dt),
            dt,
        ))
        .kind(kind);
        let mut thevenin = 0;
        let mut hard_on_lumped = 0usize;
        // Cylindrical grids: physical edges already handled (canonical copy).
        let mut seen = std::collections::HashSet::new();
        for &(edge, a) in &ex.amplitudes {
            let (axis, node) = node_map.edge(edge);
            let last = grid.nodes()[1] - 1;
            // openEMS (files): a seam ρ or z edge of a closed grid is
            // driven on the copy whose line the source lies on; the seam
            // update keeps its state on the α = 2π copy, so a source on
            // the α = 0 line only reaches that copy (and the H next to
            // it) each step (black box, design §5.16.11).
            let lo = [node[0], 0, node[2]];
            let literal = openems
                && grid.is_closed()
                && !kind.is_magnetic()
                && axis != Axis::Y
                && (node[1] == 0 || node[1] == last)
                && engine
                    .find_operator::<crate::engine::openems_cyl::OpenemsSeam>()
                    .is_some_and(|s| s.runs_e(axis, lo));
            let copies = if literal {
                if !seen.insert((axis, node)) {
                    continue;
                }
                engine.exempt_from_periodic_merge(axis, lo);
                vec![node]
            } else if grid.is_cylindrical() && !kind.is_magnetic() {
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
                // openEMS: a plain soft source, except on a parallel
                // element with a resistor, where the element's Thevenin
                // source reproduces it (see the install below).
                if !openems || (grp.topology == Topology::Parallel && grp.edges[e].r.is_some()) {
                    continue;
                }
            } else if kind == SourceKind::SoftE && lumped_edge.contains_key(&(axis, node)) {
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
            if g.in_layer.is_empty() {
                warnings.push(format!(
                    "lumped element {:?}: no usable edges; ignored",
                    g.name
                ));
            } else if g.caps {
                add_caps(engine, g.axis, &g.in_layer);
            }
            continue;
        }
        if g.caps {
            if g.in_layer.is_empty() {
                add_caps(engine, g.axis, &g.edges);
            } else {
                add_caps(engine, g.axis, &[&g.edges[..], &g.in_layer[..]].concat());
            }
        }
        // openEMS replaces the edge's medium by the element (design
        // §5.16.1) and limits the edge capacitance of a parallel element
        // with an inductor and no capacitor, and of every series element
        // by its branch impedance at the excitation's frequency (§5.16.3).
        let folded = fdtd.get_openems_run();
        if folded {
            let parallel = g.topology == Topology::Parallel;
            let no_c = g.edges.iter().all(|e| e.c.is_none());
            let omega = waveform
                .and_then(|w| w.nyquist_frequency())
                .filter(|_| !parallel || (no_c && g.edges.iter().any(|e| e.l.is_some())))
                .map(|f| 2.0 * core::f64::consts::PI * f);
            let kept = replace_medium(
                engine,
                g.axis,
                &g.edges,
                g.topology,
                medium,
                omega,
                &mut mapped.circuits,
            );
            if omega.is_some() && kept < 0.2 {
                warnings.push(if parallel {
                    format!(
                        "lumped element {:?}: openEMS's limit on the edge capacitance of a \
                         parallel R/L element keeps {:.0} % of an edge's capacitance; the \
                         explicit update may become unstable (as it does in openEMS)",
                        g.name,
                        100.0 * kept
                    )
                } else {
                    format!(
                        "lumped element {:?}: openEMS's limit on the edge capacitance of a \
                         series element keeps {:.0} % of an edge's capacitance; the update may \
                         become unstable (as it does in openEMS)",
                        g.name,
                        100.0 * kept
                    )
                });
            }
        }
        let ne = g.edges.len();
        let has_l = g.edges.first().is_some_and(|e| e.l.is_some());
        let has_r = g.edges.first().is_some_and(|e| e.r.is_some());
        // openEMS adds the excitation to the edge voltage after the update
        // (a soft source). On a parallel element with a resistor, the
        // element's Thevenin source adds `cb·g_e·v_e` per step instead, so
        // `v_e = w/(cb·g_e)` gives the same voltage per step, with
        // `cb = dt/(C + G·dt/2)` of the edge with the element folded in.
        let gain: Vec<f64> = {
            let nodes = engine.grid().nodes();
            let co = engine.coefficients();
            g.edges
                .iter()
                .map(|e| {
                    let i = (e.node[0] * nodes[1] + e.node[1]) * nodes[2] + e.node[2];
                    let ge = e.r.map_or(0.0, |r| 1.0 / r);
                    let (_, cb) = if folded {
                        (co.ca(g.axis)[i], co.cb(g.axis)[i])
                    } else {
                        crate::engine::fold_parallel(
                            co.ca(g.axis)[i],
                            co.cb(g.axis)[i],
                            dt,
                            ge,
                            e.c.unwrap_or(0.0),
                        )
                    };
                    if ge > 0.0 {
                        1.0 / (f64::from(cb) * ge)
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let branch = match g.topology {
            Topology::Parallel => has_l,
            Topology::Series => g
                .edges
                .first()
                .is_some_and(|e| e.l.is_some() || e.c.is_some()),
        };
        // openEMS updates a parallel inductor explicitly (leapfrog). That
        // is only stable while the edge's own resonance is resolved,
        // `dt²/(L_e·C_e) = dt·cb/L_e` small; beyond 1 (openEMS itself
        // diverges well before the isolated limit 4) ems keeps its
        // unconditionally stable trapezoidal branch.
        let mut explicit = openems && g.topology == Topology::Parallel && has_l;
        if explicit {
            let nodes = engine.grid().nodes();
            let co = engine.coefficients();
            let worst = g
                .edges
                .iter()
                .map(|e| {
                    let i = (e.node[0] * nodes[1] + e.node[1]) * nodes[2] + e.node[2];
                    // `folded` holds here (an openEMS parallel element).
                    let cb = co.cb(g.axis)[i];
                    dt * f64::from(cb) / e.l.unwrap_or(f64::INFINITY)
                })
                .fold(0.0f64, f64::max);
            if worst > 1.0 {
                explicit = false;
                warnings.push(format!(
                    "lumped element {:?}: inductance too small for openEMS's explicit \
                     inductor update (dt²/(L·C) = {worst:.3} per edge > 1, where openEMS \
                     becomes unstable); the implicit trapezoidal update is used",
                    g.name
                ));
            }
        }
        let mut el = LumpedElement::from_edges(g.axis, g.topology, g.edges)?
            .record_current()
            .explicit_inductor(explicit)
            .medium_folded(folded);
        for (k, weights) in g.sources {
            debug_assert_eq!(weights.len(), ne);
            let ex = &src.excitations[k];
            let wf = waveform.expect("sources only with a waveform").clone();
            // The element samples its source at `(n + ½)·dt` for the E
            // update to `(n + 1)·dt`, like a soft E source.
            let lag = openems_lag(SourceKind::SoftE, dt);
            if !openems {
                el = el.edge_source(weights, openems_signal(wf.clone(), ex.delay, lag, dt));
                continue;
            }
            // openEMS: the excitation is a plain soft source (added with
            // the other soft sources, after the element), and the element
            // is driven by the edge voltage without it. The stored `u^n`
            // holds the previous step's soft source `V^{n−1}`, so the
            // drive loses `V^{n−1}` (explicit inductor, drive `u^n`) or
            // `V^{n−1}/2` (series branch or resistor, drive `ū`): a source
            // one step later, of full or half the voltage. A parallel
            // resistor or capacitor is folded into the edge and sees the
            // whole voltage, like any medium; with a resistor the plain
            // soft source is the element's Thevenin source scaled by
            // `gain` (so the element's current is the port current).
            if has_r && g.topology == Topology::Parallel {
                let scaled = weights.iter().zip(&gain).map(|(w, k)| w * k).collect();
                el = el.edge_source(scaled, openems_signal(wf.clone(), ex.delay, lag, dt));
            }
            let signal = openems_signal(wf, ex.delay, lag + dt, dt);
            match g.topology {
                Topology::Parallel if branch => {
                    el = el.branch_drive_source(weights, signal);
                }
                Topology::Parallel => {}
                Topology::Series => {
                    let half = weights.iter().map(|w| 0.5 * w).collect();
                    el = el.edge_source(half, signal);
                }
            }
        }
        let h = el.add_to(engine)?;
        mapped.lumped.push((g.name, h));
    }
    // Soft E sources add after the lumped updates; hard sources overwrite
    // last; H sources act in post_h.
    for (name, op) in soft_e.into_iter().chain(others) {
        // openEMS's hard E source edges have no update of their own: the
        // operators acting after the update (its Mur faces) read them as
        // 0 before the source writes (black box: a Mur face edge next to
        // a hard source fits `u_1^n − k·u_0^n`, design §5.16.3). Not in
        // openEMS's PML layers, whose update ems completes itself.
        if openems && op.get_kind() == SourceKind::HardE {
            let mut co = engine.coefficients_mut();
            for (axis, node, _) in op.edges() {
                if pml.is_some_and(|l| l.contains(true, axis, node)) {
                    continue;
                }
                let i = idx(node);
                let (ca, cb) = co.e_mut(axis);
                ca[i] = 0.0;
                cb[i] = 0.0;
            }
        }
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
        // The spectral incident field of layered / dispersive backgrounds
        // needs the end of the excitation.
        let duration = w.duration().map_or(f64::INFINITY, |d| delay + d);
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
        let spec = pw.spec.clone().with_duration(duration);
        match PlaneWave::new(engine, &spec, amplitude) {
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
                if let Some(samples) = &p.mode_samples {
                    pr = pr.mode_purity(samples.clone(), weight);
                }
                // openEMS prints the α of cylindrical grids (radians) times the
                // drawing unit as well.
                let alpha_unit = if grid.is_cylindrical() {
                    src.delta_unit
                } else {
                    1.0
                };
                let at = |idx: [usize; 3]| {
                    Axis::ALL.map(|a| {
                        let v = grid.lines(a)[idx[a.index()]];
                        if a == Axis::Y { v * alpha_unit } else { v }
                    })
                };
                let location = p.mode_range.map(|(start, stop)| ProbeLocation {
                    start,
                    stop,
                    start_coords: at(start),
                    stop_coords: at(stop),
                });
                mapped.probes.push(ProbeEntry {
                    name: p.name.clone(),
                    id: p.id,
                    number: s.number,
                    kind,
                    op: ProbeOp::Scalar(engine.add_operator(pr)),
                    frequencies: s.frequencies.clone(),
                    start_time: s.start_time.value(),
                    stop_time: s.stop_time.value(),
                    location,
                    waveguide: true,
                    final_sample: None,
                    oversampling: p.oversampling,
                });
                continue;
            }
            // Reported by the compile step.
            _ => continue,
        };
        // openEMS snapping (ties to the lower line) of the raw corners.
        let prim = |c: [f64; 3]| Axis::ALL.map(|a| snap_lower(grid.lines(a), c[a.index()]));
        // Dual lines: openEMS takes the cell holding the coordinate (closed
        // at its upper end), ems natively the nearest dual line; the two
        // differ on graded meshes only.
        let dual_snap = if fdtd.get_openems_run() {
            snap_cell
        } else {
            snap_dual
        };
        let dual = |c: [f64; 3]| Axis::ALL.map(|a| dual_snap(grid.lines(a), c[a.index()]));
        let (start, stop) = (prim(p.corners[0]), prim(p.corners[1]));
        // openEMS prints the α of cylindrical grids (radians) times the
        // drawing unit as well.
        let alpha_unit = if grid.is_cylindrical() {
            src.delta_unit
        } else {
            1.0
        };
        let at = |idx: [usize; 3]| {
            Axis::ALL.map(|a| {
                let v = grid.lines(a)[idx[a.index()]];
                if a == Axis::Y { v * alpha_unit } else { v }
            })
        };
        let primary_location = ProbeLocation {
            start,
            stop,
            start_coords: at(start),
            stop_coords: at(stop),
        };
        let mut location = primary_location;
        // openEMS skips a voltage probe whose box is not a line and a
        // current probe whose box extends along its normal (or, without
        // NormDir, is not flat), judged on the raw corners (black box,
        // design §5.16.1); ems would integrate them. Mirrored for files.
        if fdtd.get_openems_run() {
            let extent: Vec<Axis> = Axis::ALL
                .into_iter()
                .filter(|a| p.corners[0][a.index()] != p.corners[1][a.index()])
                .collect();
            let skip = match kind {
                ProbeKind::Voltage if extent.len() != 1 => Some(format!(
                    "its box is not a line (dimension {})",
                    extent.len()
                )),
                ProbeKind::Current => match s.norm_dir.and_then(|d| Axis::from_index(d.index())) {
                    Some(n) if extent.contains(&n) => {
                        Some(format!("its box extends along the normal direction {n}"))
                    }
                    None if extent.len() == 3 => {
                        Some("its box is not flat and has no NormDir".to_string())
                    }
                    _ => None,
                },
                _ => None,
            };
            if let Some(why) = skip {
                warnings.push(format!(
                    "probe {:?}: {why}; ignored, as openEMS does",
                    p.name
                ));
                continue;
            }
        }
        let op = match kind {
            ProbeKind::Voltage => {
                let mut pr = Probe::new(ProbeQuantity::Voltage);
                let path = voltage_path(start, stop);
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
                // A flat box: the loop lies in the dual plane nearest to
                // the box's raw normal coordinate (openEMS).
                let normal = current_normal(start, stop, s.norm_dir);
                let flat_layer = normal
                    .filter(|n| start[n.index()] == stop[n.index()])
                    .map(|n| dual(p.corners[0])[n.index()]);
                // Transverse corners exactly half-way between two lines snap
                // inwards (openEMS: 8.5 → 9 at the lower, 11.5 → 11 at the
                // upper corner); a point-thin box whose corners then cross
                // integrates nothing.
                let mut crossed = [false; 3];
                let (mut start, mut stop) = (start, stop);
                if let Some(n) = normal {
                    for a in Axis::ALL.into_iter().filter(|&a| a != n) {
                        let d = a.index();
                        let (r0, r1) = (p.corners[0][d], p.corners[1][d]);
                        let lo = snap_upper(grid.lines(a), r0.min(r1));
                        let hi = snap_lower(grid.lines(a), r0.max(r1));
                        crossed[d] = lo > hi;
                        (start[d], stop[d]) = if r0 <= r1 { (lo, hi) } else { (hi, lo) };
                    }
                    // openEMS on a closed azimuth (files): the loop runs on
                    // the dual α lines from the one below the lower corner
                    // to the one above the upper corner, clamped to its
                    // lines α_{1/2} … α_{N−1/2}: a box from α = 0 (or to
                    // 2π) leaves out the column on the seam (black box,
                    // `cyl_current_seam`). A box over the whole ring keeps
                    // ems's closed loop (design §5.16.2).
                    let y = Axis::Y.index();
                    let last = nodes[y] - 1;
                    if grid.is_closed() && n != Axis::Y && fdtd.get_openems_run() && !crossed[y] {
                        let (a, b) = (start[y].min(stop[y]), start[y].max(stop[y]));
                        if !(a == 0 && b == last) && (a == 0 || b == last) {
                            let (a, b) = (a.max(1), b.min(last - 1));
                            crossed[y] = a > b;
                            (start[y], stop[y]) = if start[y] <= stop[y] { (a, b) } else { (b, a) };
                        }
                    }
                }
                let Some((entries, layer)) =
                    current_loop(start, stop, s.norm_dir, nodes, flat_layer)
                else {
                    warnings.push(format!(
                        "current probe {:?}: no normal direction (set NormDir or use a \
                         box that is flat along one axis); ignored",
                        p.name
                    ));
                    continue;
                };
                // Reported like openEMS: the dual lines of the loop.
                let n = normal.expect("a loop has a normal").index();
                let mut lo = [0; 3];
                let mut hi = [0; 3];
                for d in 0..3 {
                    if d == n {
                        (lo[d], hi[d]) = (layer, layer);
                    } else {
                        let (a, b) = (start[d].min(stop[d]), start[d].max(stop[d]));
                        let (a, b) = if crossed[d] { (b, a) } else { (a, b) };
                        (lo[d], hi[d]) = (a.saturating_sub(1), b.min(nodes[d].saturating_sub(2)));
                    }
                }
                location = ProbeLocation {
                    start: lo,
                    stop: hi,
                    start_coords: at(lo),
                    stop_coords: at(hi),
                };
                // openEMS files (Cartesian): openEMS's order and f32 sum.
                let oe_sum = fdtd.get_openems_run() && !grid.is_cylindrical();
                let entries = if oe_sum {
                    openems_loop_order(entries, n, start, stop)
                } else {
                    entries
                };
                let mut pr = Probe::new(ProbeQuantity::Current).f32_sum(oe_sum);
                let empty = crossed.contains(&true);
                for (axis, node, w) in entries.into_iter().filter(|_| !empty) {
                    pr = pr.entry(axis, node, w * weight);
                }
                ProbeOp::Scalar(engine.add_operator(pr))
            }
            ProbeKind::EField | ProbeKind::HField => {
                // openEMS: the native field components at the (snapped)
                // start corner: E on the three primary edges leaving the
                // node, H on the three dual edges of the dual node
                // (i + ½, j + ½, k + ½).
                let e = kind == ProbeKind::EField;
                let idx = if e { start } else { dual(p.corners[0]) };
                if !e {
                    let stop_dual = dual(p.corners[1]);
                    location = ProbeLocation {
                        start: idx,
                        stop: stop_dual,
                        start_coords: at(idx),
                        stop_coords: at(stop_dual),
                    };
                }
                let q = if e {
                    ProbeQuantity::Voltage
                } else {
                    ProbeQuantity::Current
                };
                let comps = Axis::ALL.map(|a| {
                    let mut pr = Probe::new(q);
                    let ok = Axis::ALL.iter().all(|&b| {
                        let i = idx[b.index()];
                        // E_a needs an edge along a; H_a dual edges along
                        // the two other axes.
                        if (b == a) == e {
                            i + 1 < nodes[b.index()]
                        } else {
                            true
                        }
                    });
                    if ok {
                        let len = if e {
                            grid.edge_length(a, idx)
                        } else {
                            grid.dual_edge_length(a, idx)
                        };
                        if len > 0.0 {
                            pr = pr.entry(a, idx, weight / len);
                        }
                    }
                    engine.add_operator(pr)
                });
                ProbeOp::Native(comps)
            }
        };
        mapped.probes.push(ProbeEntry {
            name: p.name.clone(),
            id: p.id,
            number: s.number,
            kind,
            op,
            frequencies: s.frequencies.clone(),
            start_time: s.start_time.value(),
            stop_time: s.stop_time.value(),
            location: Some(location),
            waveguide: false,
            final_sample: None,
            oversampling: p.oversampling,
        });
    }

    // ---- dumps (openEMS sampling rules, see `io::openems::dump`).
    let k = dump_interval(fdtd, dt);
    let nyquist = openems_nyquist(fdtd, dt);
    let oversampling = fdtd.get_oversampling().unwrap_or(4);
    for d in &src.dumps {
        let s = &d.settings;
        // A dump's own OverSampling replaces the <FDTD> one.
        let td_interval = oe::time_domain_interval(nyquist, d.oversampling.unwrap_or(oversampling));
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
        let Some(indices) = dump_indices(&grid, d, pos, q, src.delta_unit) else {
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

/// openEMS semantics ([`Fdtd::openems_run`]) of a lumped element on its
/// edges, folded once into the edge coefficients.
///
/// A parallel element's capacitance *replaces* the edge's own (medium)
/// capacitance, and its resistance the medium's conductance; values the
/// element does not set keep the medium's. (Established by black-box
/// comparison with openEMS v0.37: C on vacuum and on a lossy dielectric, R
/// on a lossy dielectric agree to ~1e-6; adding the element to the medium,
/// as the engine's [`LumpedElement`] does by itself, is ~1 % off for a
/// 1 pF element on a 1 mm vacuum cell.) A series element keeps the medium
/// capacitance and drops its conductance (`--debug-operator`: `vv = 1` on
/// lossy media); a series resistor without a branch is folded in as the
/// edge's conductance (it acts as the parallel fold, §5.16.1). With
/// `omega`, the capacitance is then limited ([`capacitance_limit`]).
///
/// The edge gets `C = f32(C_e)` or the medium's `C`, `G = f32(1/R_e)` or
/// the medium's `G`, `C = f32(C_max)` where limited (rounded per
/// `medium`'s [`CoeffRounding`], openEMS's rounding of the circuit
/// quantities, design §5.16.7), then `vv`, `vi` in `f64`
/// (`--debug-operator` dumps of 160 random R, C, R‖C, R‖L, L, series
/// elements on vacuum, dielectrics and lossy media, uniform and graded
/// meshes: every element edge bit for bit). The medium's `C`, `G` are
/// those of the edge materials of `medium` where they give the edge's
/// coefficients exactly, else they are recovered from the coefficients
/// (`C = dt·(1 + ca)/(2·cb)`, `G = (1 − ca)/cb`, exact up to their `f32`
/// rounding). The element is installed with
/// [`medium_folded`](LumpedElement::medium_folded). Returns the smallest
/// kept fraction of an edge's capacitance under the limit (1 without).
fn replace_medium(
    engine: &mut Engine,
    axis: Axis,
    edges: &[EdgeValues],
    topology: Topology,
    medium: Option<(&EdgeMaterials, CoeffRounding)>,
    omega: Option<f64>,
    circuits: &mut Vec<(Axis, [usize; 3], f64)>,
) -> f64 {
    let grid = engine.grid().clone();
    let dt = engine.dt();
    let rounding = medium.map_or(CoeffRounding::Exact, |m| m.1);
    let mut kept = 1.0f64;
    let mut co = engine.coefficients_mut();
    let (ca, cb) = co.e_mut(axis);
    for e in edges {
        let series = topology == Topology::Series;
        let branch = series && (e.l.is_some() || e.c.is_some());
        if !series && e.c.is_none() && e.r.is_none() && omega.is_none() {
            continue;
        }
        let i = grid.index(e.node);
        let (a, b) = (ca[i], cb[i]);
        if !b.is_finite() {
            continue;
        }
        let exact = medium
            .map(|(m, r)| e_edge_circuit(&grid, m, axis, e.node, r))
            .filter(|&(c, g)| e_coefficients(c, g, dt) == (a, b));
        let (mut cap, mut cond) = exact.unwrap_or_else(|| {
            let (a, b) = (f64::from(a), f64::from(b));
            (dt * (1.0 + a) / (2.0 * b), (1.0 - a) / b)
        });
        if !series && let Some(c) = e.c {
            cap = rounding.store(c);
        }
        if series {
            cond = 0.0;
        }
        if !branch && let Some(r) = e.r {
            cond = rounding.store(1.0 / r);
        }
        if let Some(omega) = omega {
            let limit = capacitance_limit(e, omega, topology);
            if cap > limit {
                kept = kept.min(limit / cap);
                cap = rounding.store(limit);
            }
        }
        (ca[i], cb[i]) = e_coefficients(cap, cond, dt);
        // openEMS folds a parallel R/C element into the edge before its
        // poles take the edge's coefficients; an element with an
        // inductor or in series comes after them (design §5.16.3).
        if !series && e.l.is_none() {
            circuits.push((axis, e.node, cap));
        }
    }
    kept
}

/// openEMS's limit on the medium capacitance of the edges of a parallel
/// lumped element with an inductor and without a capacitor (black box,
/// design §5.16.1, fitted from `--debug-operator` dumps): at the
/// excitation's frequency `ω = 2π·f` (`f0 + fc` for a Gaussian, `f0`
/// otherwise, the frequency of its Nyquist rate) the edge's capacitive
/// admittance stays below a twentieth of the element's resistive and
/// inductive ones, `C_e ≤ min(1/(20·ω·R_e), 1/(20·ω²·L_e))` with the
/// per-edge `R_e`, `L_e`. Edges above the limit get exactly the limit
/// (the medium conductance is kept). It does not depend on the time step;
/// it can leave an edge far below its medium capacitance, so a large `R`
/// or `L` makes openEMS's update unstable (it diverges for a 1 mm vacuum
/// edge with `L` ≳ 3e-8 H, or on the `lumped_graded_rl` element with
/// `R` ≳ 150 Ω). A series element (any of R, L, C) is limited by its
/// branch impedance instead, `C_e ≤ 1/(20·ω·|R_e + jωL_e + 1/(jωC_e')|)`
/// with the element's per-edge `C_e'` (wave 14, same dumps: R, RL, L, LC
/// and RLC branches, `R` = 1…1000 Ω, `L` = 1.1…50 nH, `C` = 0.5…5 pF, two
/// frequencies, 5 digits; this is the "different update" of large series
/// resistors noted before). Infinite if `omega` is not finite and
/// positive, or nothing limits.
fn capacitance_limit(e: &EdgeValues, omega: f64, topology: Topology) -> f64 {
    if !(omega.is_finite() && omega > 0.0) {
        return f64::INFINITY;
    }
    match topology {
        Topology::Parallel => {
            e.r.map(|r| 1.0 / (20.0 * omega * r))
                .into_iter()
                .chain(e.l.map(|l| 1.0 / (20.0 * omega * omega * l)))
                .fold(f64::INFINITY, f64::min)
        }
        // The series branch's admittance at ω.
        Topology::Series => {
            let r = e.r.unwrap_or(0.0);
            let x = e.l.map_or(0.0, |l| omega * l) - e.c.map_or(0.0, |c| 1.0 / (omega * c));
            let z = r.hypot(x);
            if z > 0.0 {
                1.0 / (20.0 * omega * z)
            } else {
                f64::INFINITY
            }
        }
    }
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

/// Weighted dual edges `(axis, node, weight)` of a current loop.
pub(crate) type LoopEntries = Vec<(Axis, [usize; 3], f64)>;

/// The normal axis of a current probe: `NormDir`, or the box's only flat
/// axis.
pub(crate) fn current_normal(
    start: [usize; 3],
    stop: [usize; 3],
    norm_dir: Option<csx::Axis>,
) -> Option<Axis> {
    match norm_dir {
        Some(d) => Axis::from_index(d.index()),
        None => {
            let flat: Vec<Axis> = Axis::ALL
                .into_iter()
                .filter(|a| start[a.index()] == stop[a.index()])
                .collect();
            match flat[..] {
                [a] => Some(a),
                _ => None,
            }
        }
    }
}

/// Dual edges `(axis, node, weight)` whose weighted sum is the current
/// through the probe box from `start` towards `stop` along the normal
/// (`+normal` for a flat box), and the E-edge layer `k` of the loop (the
/// loop lies in the dual plane between nodes `k` and `k + 1` along the
/// normal), or `None` without a normal direction.
///
/// The layer is `flat_layer` if given (openEMS: the dual plane nearest to
/// a flat box), else the middle layer of the box (for a flat box at node
/// `k`, the layer from `k` to `k + 1`).
pub(crate) fn current_loop(
    start: [usize; 3],
    stop: [usize; 3],
    norm_dir: Option<csx::Axis>,
    nodes: [usize; 3],
    flat_layer: Option<usize>,
) -> Option<(LoopEntries, usize)> {
    let n = current_normal(start, stop, norm_dir)?;
    let ni = n.index();
    if nodes[ni] < 2 {
        return None;
    }
    let lo = [0, 1, 2].map(|d| start[d].min(stop[d]));
    let hi = [0, 1, 2].map(|d| start[d].max(stop[d]));
    let layer = if let Some(k) = flat_layer {
        k.min(nodes[ni] - 2)
    } else if lo[ni] == hi[ni] {
        lo[ni].min(nodes[ni] - 2)
    } else {
        lo[ni] + (hi[ni] - lo[ni] - 1) / 2
    };
    let sign = if start[ni] <= stop[ni] { 1.0 } else { -1.0 };
    // loop_entries gives −∮H (right-handed about +n).
    Some((
        loop_entries(n, lo, hi, layer)
            .into_iter()
            .map(|(a, p, w)| (a, p, -w * sign))
            .collect(),
        layer,
    ))
}

/// The entries of a [`current_loop`] (normal `n`, box `start`…`stop`) in
/// the order openEMS sums them (black box, design §5.16.9): the side
/// below the box along `c` (the `b` components), the side above it along
/// `b` (the `c` components), the side above it along `c`, the side below
/// it along `b`, each with ascending node index (`(n, b, c)` cyclic).
fn openems_loop_order(
    mut entries: LoopEntries,
    n: usize,
    start: [usize; 3],
    stop: [usize; 3],
) -> LoopEntries {
    let (b, c) = ((n + 1) % 3, (n + 2) % 3);
    let lo = [0, 1, 2].map(|d| start[d].min(stop[d]));
    let hi = [0, 1, 2].map(|d| start[d].max(stop[d]));
    entries.sort_by_key(|&(a, p, _)| {
        let a = a.index();
        if a == b && p[c] < lo[c] {
            (0, p[b])
        } else if a == c && p[b] >= hi[b] {
            (1, p[c])
        } else if a == b {
            (2, p[b])
        } else {
            (3, p[c])
        }
    });
    entries
}

/// Index of the mesh line nearest to `x` (openEMS probe snapping: on a
/// tie, within `1e-9` of the local cell size, the lower line).
pub(crate) fn snap_lower(lines: &[f64], x: f64) -> usize {
    let mut best = 0;
    for (i, &l) in lines.iter().enumerate().skip(1) {
        let tol = 1e-9 * (l - lines[i - 1]).abs();
        if (l - x).abs() < (lines[best] - x).abs() - tol {
            best = i;
        }
    }
    best
}

/// Index of the line of `lines` nearest to `x`, ties to the upper one
/// (see [`snap_lower`]).
pub(crate) fn snap_upper(lines: &[f64], x: f64) -> usize {
    let mut best = 0;
    for (i, &l) in lines.iter().enumerate().skip(1) {
        let tol = 1e-9 * (l - lines[i - 1]).abs();
        if (l - x).abs() <= (lines[best] - x).abs() + tol {
            best = i;
        }
    }
    best
}

/// Index `i` of the dual line `(l[i] + l[i+1])/2` nearest to `x` (ties to
/// the lower one), `0` for a single-line axis.
pub(crate) fn snap_dual(lines: &[f64], x: f64) -> usize {
    if lines.len() < 2 {
        return 0;
    }
    let mid: Vec<f64> = lines.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    snap_lower(&mid, x)
}

/// Index `i` of the cell `(l[i], l[i+1]]` holding `x` (a coordinate on a
/// line, within `1e-9` of the local cell size, belongs to the cell below
/// it; clamped into the mesh), `0` for a single-line axis. openEMS v0.37
/// snaps H-field probes and the dual plane of current probes this way
/// (black box: point probes swept through a graded mesh, the indices in
/// the file headers; design §5.16.1): on a uniform mesh it is the nearest
/// dual line with ties down ([`snap_dual`]), on a graded one a point in a
/// cell next to a smaller one can be nearer to that cell's centre.
pub(crate) fn snap_cell(lines: &[f64], x: f64) -> usize {
    if lines.len() < 2 {
        return 0;
    }
    let mut i = 0;
    for k in 1..lines.len() - 1 {
        let tol = 1e-9 * (lines[k + 1] - lines[k - 1]).abs();
        if lines[k] < x - tol {
            i = k;
        }
    }
    i
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
    fn openems_snapping_ties_down() {
        let l = [0.0, 1.0, 2.0, 3.5];
        assert_eq!(snap_lower(&l, 0.5), 0);
        assert_eq!(snap_lower(&l, 0.51), 1);
        assert_eq!(snap_lower(&l, 1.5), 1);
        assert_eq!(snap_lower(&l, 9.0), 3);
        assert_eq!(snap_lower(&l, -1.0), 0);
        // Dual lines 0.5, 1.5, 2.75.
        assert_eq!(snap_dual(&l, 1.0), 0);
        assert_eq!(snap_dual(&l, 1.2), 1);
        assert_eq!(snap_dual(&l, 3.5), 2);
        assert_eq!(snap_dual(&[1.0], 3.0), 0);
        // openEMS's cell rule (graded z mesh of the black-box sweep).
        let z = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0];
        for (x, i) in [
            (0.0, 0),
            (0.3, 0),
            (4.85, 4),
            (4.9, 4),
            (5.0, 4),
            (5.01, 5),
            (5.5, 5),
            (5.55, 6),
            (6.05, 7),
            (8.1, 9),
            (12.0, 10),
            (13.0, 10),
        ] {
            assert_eq!(snap_cell(&z, x), i, "{x}");
        }
        assert_eq!(snap_dual(&z, 4.9), 5);
        assert_eq!(snap_cell(&[1.0], 3.0), 0);
    }

    #[test]
    fn current_loop_normal_and_sign() {
        let nodes = [10, 10, 10];
        // Flat box normal to z: +z.
        let a = current_loop([3, 4, 2], [3, 4, 2], None, nodes, None);
        assert!(a.is_none(), "a point has no unique flat axis");
        let a = current_loop([3, 3, 2], [4, 4, 2], None, nodes, None).unwrap();
        let b = current_loop([3, 3, 2], [4, 4, 2], Some(csx::Axis::Z), nodes, None).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.1, 2);
        let c = current_loop([3, 3, 2], [4, 4, 2], None, nodes, Some(1)).unwrap();
        assert_eq!(c.1, 1);
        // Reversed along the normal: opposite sign, same layer.
        let (up, _) = current_loop([3, 4, 1], [3, 4, 4], Some(csx::Axis::Z), nodes, None).unwrap();
        let (down, _) =
            current_loop([3, 4, 4], [3, 4, 1], Some(csx::Axis::Z), nodes, None).unwrap();
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
