//! Field dumps: unit conversion and interpolation on a non-uniform grid,
//! time stamps, schedules, sub-sampling, frequency-domain accumulation,
//! VTR output and determinism.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ems::engine::{
    DumpPosition, DumpQuantity, DumpRegion, Engine, FieldSampler, Fields, FreqDump, Grid, Operator,
    Schedule, Snapshot, StepContext, TimeDump, VtrSeries,
};
use ems::{Axis, Complex64, Error};

const MM: f64 = 1e-3;

fn grid() -> Grid {
    Grid::new([
        vec![0.0, 1.0, 2.5, 4.5, 5.0].map_mm(),
        vec![0.0, 0.7, 1.9, 2.4, 3.6].map_mm(),
        vec![0.0, 0.4, 1.0, 2.2].map_mm(),
    ])
    .unwrap()
}

trait MapMm {
    fn map_mm(self) -> Vec<f64>;
}
impl MapMm for Vec<f64> {
    fn map_mm(self) -> Vec<f64> {
        self.into_iter().map(|v| v * MM).collect()
    }
}

/// Analytic E (V/m) as a function of position in mm, linear per axis.
fn e_field(a: Axis, [x, y, z]: [f64; 3]) -> f64 {
    match a {
        Axis::X => 2.0 + 3.0 * x + 5.0 * y,
        Axis::Y => 1.0 - 2.0 * y + 4.0 * z,
        Axis::Z => 3.0 + x + z,
    }
}

/// Analytic H (A/m), independent of its own coordinate (a dual edge's
/// midpoint is not its node on a graded grid).
fn h_field(a: Axis, [x, y, z]: [f64; 3]) -> f64 {
    match a {
        Axis::X => 1.0 + 2.0 * y + 3.0 * z,
        Axis::Y => 2.0 - x + z,
        Axis::Z => x + 2.0 * y,
    }
}

fn mm(g: &Grid, a: Axis, i: usize) -> f64 {
    g.lines(a)[i] / MM
}
fn mid_mm(g: &Grid, a: Axis, i: usize) -> f64 {
    g.dual_line(a, i) / MM
}

/// Sets `u` and `i` to the integral form of the analytic fields.
fn set_analytic(f: &mut Fields<'_>, g: &Grid) {
    let n = g.nodes();
    for i in 0..n[0] {
        for j in 0..n[1] {
            for k in 0..n[2] {
                let node = [i, j, k];
                for a in Axis::ALL {
                    let ax = a.index();
                    // E edge: midpoint along a, node lines across.
                    if node[ax] + 1 < n[ax] {
                        let mut p = Axis::ALL.map(|b| mm(g, b, node[b.index()]));
                        p[ax] = mid_mm(g, a, node[ax]);
                        f.u.set(a, node, (e_field(a, p) * g.delta(a, node[ax])) as f32);
                    }
                    // H dual edge: node line along a, cell midpoints across.
                    let (b, c) = a.others();
                    if node[b.index()] + 1 < n[b.index()] && node[c.index()] + 1 < n[c.index()] {
                        let mut p = [0.0; 3];
                        p[ax] = mm(g, a, node[ax]);
                        p[b.index()] = mid_mm(g, b, node[b.index()]);
                        p[c.index()] = mid_mm(g, c, node[c.index()]);
                        let l = g.dual_delta(a, node[ax]);
                        f.i.set(a, node, (h_field(a, p) * l) as f32);
                    }
                }
            }
        }
    }
}

fn engine(threads: usize) -> Engine {
    Engine::builder(grid()).threads(threads).build().unwrap()
}

/// Position (mm) where the node rule actually evaluates component `a` of
/// `q` at `node`: the node itself, except on the outer nodes where the
/// one-sided rules fall back to the adjacent cell midpoint.
fn node_rule_pos(g: &Grid, q: DumpQuantity, a: Axis, node: [usize; 3]) -> [f64; 3] {
    Axis::ALL.map(|b| {
        let (i, n) = (node[b.index()], g.nodes()[b.index()]);
        let interp = match q {
            DumpQuantity::H => a != b,
            _ => a == b,
        };
        if interp && i == 0 {
            mid_mm(g, b, 0)
        } else if interp && i == n - 1 {
            mid_mm(g, b, n - 2)
        } else {
            mm(g, b, i)
        }
    })
}

fn close(got: f32, want: f64, what: &str) {
    assert!(
        (got as f64 - want).abs() <= 2e-6 * want.abs().max(1.0),
        "{what}: got {got}, want {want}"
    );
}

fn snapshot_at_step0(q: DumpQuantity, pos: DumpPosition, region: DumpRegion) -> Snapshot {
    let mut e = engine(1);
    let g = e.grid().clone();
    set_analytic(&mut e.fields_mut(), &g);
    let s = FieldSampler::new(&g, region, q, pos).unwrap();
    let direct = s.snapshot(&e);
    let h = e.add_operator(TimeDump::new(s));
    e.step();
    let snaps = e.operator(&h).sink().snapshots();
    assert_eq!(snaps.len(), 2);
    assert_eq!(snaps[0], direct);
    snaps[0].clone()
}

#[test]
fn node_values_are_converted_and_interpolated() {
    let g = grid();
    let n = g.nodes();
    for q in [DumpQuantity::E, DumpQuantity::H] {
        let snap = snapshot_at_step0(q, DumpPosition::Nodes, DumpRegion::whole(&g));
        assert_eq!(snap.step, 0);
        assert_eq!(snap.data.len(), g.num_nodes());
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let v = snap.data[g.index([i, j, k])];
                    for a in Axis::ALL {
                        let p = node_rule_pos(&g, q, a, [i, j, k]);
                        let want = match q {
                            DumpQuantity::H => h_field(a, p),
                            _ => e_field(a, p),
                        };
                        close(v[a.index()], want, &format!("{q:?}{a} at [{i},{j},{k}]"));
                    }
                }
            }
        }
    }
}

#[test]
fn cell_centre_values_are_exact_for_linear_fields() {
    let g = grid();
    let c = g.cells();
    for q in [DumpQuantity::E, DumpQuantity::H] {
        let s = FieldSampler::new(&g, DumpRegion::whole(&g), q, DumpPosition::CellCenters).unwrap();
        assert_eq!(s.dims(), c);
        assert_eq!(s.coords(Axis::X), &g.dual_lines(Axis::X)[..]);
        let snap = snapshot_at_step0(q, DumpPosition::CellCenters, DumpRegion::whole(&g));
        for i in 0..c[0] {
            for j in 0..c[1] {
                for k in 0..c[2] {
                    let v = snap.data[(i * c[1] + j) * c[2] + k];
                    let p = [
                        mid_mm(&g, Axis::X, i),
                        mid_mm(&g, Axis::Y, j),
                        mid_mm(&g, Axis::Z, k),
                    ];
                    for a in Axis::ALL {
                        let want = match q {
                            DumpQuantity::H => h_field(a, p),
                            _ => e_field(a, p),
                        };
                        close(
                            v[a.index()],
                            want,
                            &format!("{q:?}{a} in cell [{i},{j},{k}]"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn degenerate_cell_centre_plane_uses_node_rule() {
    let g = grid();
    let region = DumpRegion::new([4, 2, 0], [0, 2, 3]); // corners in any order
    assert_eq!(region.lo(), [0, 2, 0]);
    let s = FieldSampler::new(&g, region, DumpQuantity::E, DumpPosition::CellCenters).unwrap();
    assert_eq!(s.dims(), [4, 1, 3]);
    assert_eq!(s.coords(Axis::Y), &[g.lines(Axis::Y)[2]]);
    let snap = snapshot_at_step0(DumpQuantity::E, DumpPosition::CellCenters, region);
    for i in 0..4 {
        for k in 0..3 {
            let p = [
                mid_mm(&g, Axis::X, i),
                mm(&g, Axis::Y, 2),
                mid_mm(&g, Axis::Z, k),
            ];
            for a in Axis::ALL {
                close(snap.data[i * 3 + k][a.index()], e_field(a, p), "plane");
            }
        }
    }
}

#[test]
fn sub_sampling() {
    let g = grid();
    let full = snapshot_at_step0(DumpQuantity::H, DumpPosition::Nodes, DumpRegion::whole(&g));
    let region = DumpRegion::new([1, 0, 0], [4, 4, 3]).stride([2, 1, 3]);
    let s = FieldSampler::new(&g, region, DumpQuantity::H, DumpPosition::Nodes).unwrap();
    assert_eq!(s.dims(), [2, 5, 2]);
    assert_eq!(
        s.coords(Axis::X),
        &[g.lines(Axis::X)[1], g.lines(Axis::X)[3]]
    );
    assert_eq!(
        s.coords(Axis::Z),
        &[g.lines(Axis::Z)[0], g.lines(Axis::Z)[3]]
    );
    let sub = snapshot_at_step0(DumpQuantity::H, DumpPosition::Nodes, region);
    let mut idx = 0;
    for i in [1, 3] {
        for j in 0..5 {
            for k in [0, 3] {
                assert_eq!(sub.data[idx], full.data[g.index([i, j, k])]);
                idx += 1;
            }
        }
    }
    // A single point.
    let s = FieldSampler::new(
        &g,
        DumpRegion::new([2, 3, 1], [2, 3, 1]),
        DumpQuantity::E,
        DumpPosition::Nodes,
    )
    .unwrap();
    assert_eq!((s.dims(), s.len()), ([1, 1, 1], 1));
}

#[test]
fn region_must_fit() {
    let g = grid();
    let r = DumpRegion::new([0, 0, 0], [4, 5, 3]);
    assert!(matches!(
        FieldSampler::new(&g, r, DumpQuantity::E, DumpPosition::Nodes),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn time_stamps_and_schedules() {
    let mut e = engine(1);
    let dt = e.dt();
    let g = e.grid().clone();
    let r = DumpRegion::whole(&g);
    let es = FieldSampler::new(&g, r, DumpQuantity::E, DumpPosition::Nodes).unwrap();
    let hs = FieldSampler::new(&g, r, DumpQuantity::H, DumpPosition::Nodes).unwrap();
    let he = e.add_operator(TimeDump::new(es.clone()).schedule(Schedule::Every(2)));
    let hh = e.add_operator(TimeDump::new(hs).schedule(Schedule::Every(2)));
    let hl = e.add_operator(TimeDump::new(es.clone()).schedule(Schedule::Steps(vec![3, 1, 3])));
    let hi = e.add_operator(TimeDump::new(es).schedule(Schedule::Interval(2.5 * dt)));
    e.run(6);
    let steps = |s: &[Snapshot]| s.iter().map(|s| s.step).collect::<Vec<_>>();
    let ee = e.operator(&he).sink().snapshots();
    let hh = e.operator(&hh).sink().snapshots();
    assert_eq!(steps(ee), [0, 2, 4, 6]);
    assert_eq!(steps(hh), [0, 2, 4, 6]);
    for (a, b) in ee.iter().zip(hh) {
        assert_eq!(a.time, a.step as f64 * dt);
        assert_eq!(b.time, (b.step as f64 - 0.5) * dt);
    }
    assert_eq!(steps(e.operator(&hl).sink().snapshots()), [1, 3]);
    assert_eq!(steps(e.operator(&hi).sink().snapshots()), [0, 3, 5]);
    // take() drains the memory sink.
    assert_eq!(e.operator_mut(&he).sink_mut().take().len(), 4);
    assert!(e.operator(&he).sink().snapshots().is_empty());
}

#[test]
fn closure_sink_and_error_stop() {
    let mut e = engine(1);
    let g = e.grid().clone();
    let s = FieldSampler::new(
        &g,
        DumpRegion::whole(&g),
        DumpQuantity::E,
        DumpPosition::Nodes,
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let sink = move |s: &FieldSampler, snap: Snapshot| {
        assert_eq!(snap.data.len(), s.len());
        if c.fetch_add(1, Ordering::Relaxed) == 2 {
            Err(Error::Io("full".into()))
        } else {
            Ok(())
        }
    };
    let h = e.add_operator(TimeDump::with_sink(s, sink));
    e.run(10);
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert!(matches!(e.operator(&h).error(), Some(Error::Io(_))));
}

/// Overwrites `u_x` and `i_z` with spatially uniform sinusoids at the
/// exact instants of the states (E at `m·dt`, H at `(m − ½)·dt`).
struct Driver {
    w: f64,
    amp: [f64; 2],
    phase: [f64; 2],
}

impl Driver {
    fn set_e(&self, f: &mut Fields<'_>, g: &Grid, t: f64) {
        let n = g.nodes();
        let e = self.amp[0] * (self.w * t + self.phase[0]).cos();
        for i in 0..n[0] - 1 {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    f.u.set(Axis::X, [i, j, k], (e * g.delta(Axis::X, i)) as f32);
                }
            }
        }
    }
    fn set_h(&self, f: &mut Fields<'_>, g: &Grid, t: f64) {
        let n = g.nodes();
        let h = self.amp[1] * (self.w * t + self.phase[1]).cos();
        for i in 0..n[0] - 1 {
            for j in 0..n[1] - 1 {
                for k in 0..n[2] {
                    f.i.set(Axis::Z, [i, j, k], (h * g.dual_delta(Axis::Z, k)) as f32);
                }
            }
        }
    }
}

impl Operator for Driver {
    fn pre_h(&mut self, f: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if ctx.step == 0 {
            self.set_e(f, ctx.grid, 0.0);
            self.set_h(f, ctx.grid, -0.5 * ctx.dt);
        }
    }
    fn post_h(&mut self, f: &mut Fields<'_>, ctx: &StepContext<'_>) {
        self.set_h(f, ctx.grid, ctx.h_time());
    }
    fn post_e(&mut self, f: &mut Fields<'_>, ctx: &StepContext<'_>) {
        self.set_e(f, ctx.grid, ctx.e_time());
    }
}

#[test]
fn frequency_domain_of_a_driven_sinusoid() {
    let mut e = engine(2);
    let dt = e.dt();
    let g = e.grid().clone();
    let period = 20.0;
    let f0 = 1.0 / (period * dt);
    let (amp, phase) = ([3.0, 0.02], [0.7, -1.1]);
    e.add_operator(Driver {
        w: core::f64::consts::TAU * f0,
        amp,
        phase,
    });
    let r = DumpRegion::new([0, 1, 0], [4, 3, 3]);
    let mk = |q| FieldSampler::new(&g, r, q, DumpPosition::Nodes).unwrap();
    let freqs = vec![f0, 3.0 * f0];
    let he = e.add_operator(FreqDump::new(mk(DumpQuantity::E), freqs.clone()));
    let hh = e.add_operator(FreqDump::new(mk(DumpQuantity::H), freqs.clone()));
    let he2 = e.add_operator(FreqDump::new(mk(DumpQuantity::E), freqs.clone()).interval(2));
    // States 0..=399: 400 samples, exactly 20 periods.
    e.run(399);
    for (h, comp, q) in [(he, 0, 0), (hh, 2, 1), (he2, 0, 0)] {
        let d = e.operator(&h);
        let total = 400.0 * dt;
        // Σ cos(ωt + φ)·e^{-jωt}·Δt over whole periods = (T/2)·e^{jφ}, with
        // t the quantity's own sample time: a wrong half-step in the H
        // time stamp would rotate the phase by π/20.
        let want = Complex64::cis(phase[q]) * (0.5 * total * amp[q]);
        assert_eq!(d.freqs(), &freqs[..]);
        assert!((d.sample_interval() * d.samples() as f64 - total).abs() < 1e-9 * total);
        for v in d.values(0) {
            let err = (v[comp] - want).norm() / want.norm();
            assert!(
                err < 1e-5,
                "{:?}: {} vs {want}",
                d.sampler().quantity(),
                v[comp]
            );
        }
        // Orthogonal harmonic.
        for v in d.values(1) {
            assert!(v[comp].norm() < 1e-5 * want.norm());
        }
    }
    // reset() clears.
    e.operator_mut(&he).reset();
    assert_eq!(e.operator(&he).samples(), 0);
    assert!(
        e.operator(&he)
            .values(0)
            .iter()
            .all(|v| v[0] == Complex64::ZERO)
    );
}

struct TempDir(PathBuf);
impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("ems-dump-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read(p: impl AsRef<std::path::Path>) -> String {
    std::fs::read_to_string(p).unwrap()
}

#[test]
fn vtr_output() {
    let tmp = TempDir::new("vtr");
    let mut e = engine(1);
    let g = e.grid().clone();
    set_analytic(&mut e.fields_mut(), &g);
    let r = DumpRegion::new([0, 0, 1], [4, 4, 1]).stride([2, 1, 1]);
    let s = FieldSampler::new(&g, r, DumpQuantity::E, DumpPosition::Nodes).unwrap();
    let hs = e.add_operator(
        TimeDump::with_sink(s.clone(), VtrSeries::new(tmp.0.join("td"), "E"))
            .schedule(Schedule::Every(10)),
    );
    let hm = e.add_operator(TimeDump::new(s.clone()).schedule(Schedule::Every(10)));
    let hf = e.add_operator(FreqDump::new(s, vec![1e9, 2e9]));
    e.run(30);
    assert!(e.operator(&hs).error().is_none());
    let series = e.operator(&hs).sink();
    assert_eq!(series.collection().entries().len(), 4);
    let pvd = read(series.pvd_path());
    assert_eq!(pvd.matches("<DataSet ").count(), 4);
    assert!(pvd.contains(r#"file="E_00000030.vtr""#));
    let vtr = read(tmp.0.join("td/E_00000010.vtr"));
    assert!(vtr.contains(r#"<RectilinearGrid WholeExtent="0 2 0 4 0 0">"#));
    assert!(vtr.contains(r#"Name="E""#) && vtr.contains(r#"NumberOfComponents="3""#));
    assert!(vtr.contains("TimeValue"));
    assert!(vtr.trim_end().ends_with("</VTKFile>"));

    // The in-memory dump writes the same files.
    e.operator(&hm).write_vtr(tmp.0.join("mem"), "E").unwrap();
    for f in ["E_00000000.vtr", "E_00000020.vtr", "E.pvd"] {
        assert_eq!(
            read(tmp.0.join("mem").join(f)),
            read(tmp.0.join("td").join(f))
        );
    }

    e.operator(&hf).write_vtr(tmp.0.join("fd"), "Ef").unwrap();
    let pvd = read(tmp.0.join("fd/Ef.pvd"));
    assert_eq!(pvd.matches("<DataSet ").count(), 2);
    assert!(pvd.contains(r#"file="Ef_f1.vtr""#));
    let vtr = read(tmp.0.join("fd/Ef_f0.vtr"));
    assert!(vtr.contains(r#"Name="E_re""#) && vtr.contains(r#"Name="E_im""#));
    assert!(vtr.contains(r#"Name="Frequency""#));
    assert!(vtr.contains(r#"WholeExtent="0 2 0 4 0 0""#));
}

/// Pseudo-random initial state, deterministic.
fn scramble(f: &mut Fields<'_>, g: &Grid) {
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let n = g.nodes();
    for i in 0..n[0] {
        for j in 0..n[1] {
            for k in 0..n[2] {
                for a in Axis::ALL {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let v = ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 1e-3;
                    let ax = a.index();
                    let (b, c) = a.others();
                    if [i, j, k][ax] + 1 < n[ax]
                        && [i, j, k][b.index()] > 0
                        && [i, j, k][c.index()] > 0
                        && [i, j, k][b.index()] + 1 < n[b.index()]
                        && [i, j, k][c.index()] + 1 < n[c.index()]
                    {
                        f.u.set(a, [i, j, k], v);
                    }
                }
            }
        }
    }
}

#[test]
fn deterministic_across_threads() {
    let run = |threads: usize| {
        let lines = |n: usize, s: f64| -> Vec<f64> {
            (0..n)
                .map(|i| s * (i as f64 + 0.15 * (i as f64).sin()))
                .collect()
        };
        let g = Grid::new([lines(12, 1e-3), lines(10, 1.2e-3), lines(8, 0.9e-3)]).unwrap();
        let mut e = Engine::builder(g.clone()).threads(threads).build().unwrap();
        scramble(&mut e.fields_mut(), &g);
        let r = DumpRegion::whole(&g).stride([1, 2, 1]);
        let es = FieldSampler::new(&g, r, DumpQuantity::E, DumpPosition::Nodes).unwrap();
        let hs = FieldSampler::new(&g, r, DumpQuantity::H, DumpPosition::CellCenters).unwrap();
        let ht = e.add_operator(TimeDump::new(es).schedule(Schedule::Every(5)));
        let hf = e.add_operator(FreqDump::new(hs, vec![20e9, 45e9]));
        e.run(60);
        let td = e.operator(&ht).sink().snapshots().to_vec();
        let fd = [0, 1].map(|k| e.operator(&hf).values(k));
        (e.threads(), td, fd)
    };
    let (t1, td1, fd1) = run(1);
    let (t4, td4, fd4) = run(4);
    assert_eq!((t1, t4), (1, 4));
    assert_eq!(td1.len(), 13);
    assert!(td1.last().unwrap().data.iter().any(|v| v[0] != 0.0));
    assert_eq!(td1, td4);
    assert_eq!(fd1, fd4);
}
