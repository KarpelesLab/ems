//! The surface impedance beside the merged faces of conformal PEC: the
//! faces are computed in the serial hooks after each half-step, the edges
//! of the surface impedance by the engine's threads before them, and
//! every value is what it is with the edges in the serial hooks after the
//! faces (design §5.9.1, *Threads*).

use csx::primitives::{BoxShape, LinPoly, Polygon};
use csx::{ContinuousStructure, Material, Primitive, Property};
use ems::compile::{CompileOptions, ConformalOutlines, SheetModel, compile};
use ems::engine::surface::Conductor;
use ems::engine::{EdgeSource, FusedSteps, Probe};
use ems::{Axis, Boundary, Cpml, Face, Fdtd};

fn lines(lo: f64, hi: f64, h: f64) -> Vec<f64> {
    let n = ((hi - lo) / h).round() as usize;
    (0..=n).map(|q| lo + q as f64 * h).collect()
}

/// Energies, probe values and all fields of a run, as bits.
type Run = (Vec<u64>, Vec<u64>, Vec<u32>);

/// A strip 8 mm long and 1 mm wide at `deg` to the mesh on 0.5 mm of
/// `ε_r` = 4 in a closed box, cells of 0.25 mm: a conducting sheet
/// (rational model) or `solid` lossy metal 0.1 mm thick, both of 2e5 S/m.
/// With `pml` the box has absorbing layers of six cells on its x and y
/// faces and the strip runs through them (16 mm): small faces in the
/// layers are merged, and their layer corrections read sheet edges too.
/// Returns the run, the number of faces conformal PEC interpolates or
/// merges, whether the slabs ran the edges and how many of them the
/// merged faces read.
fn run(
    deg: f64,
    solid: bool,
    pml: bool,
    threads: usize,
    serial: bool,
) -> (Run, usize, bool, usize) {
    let (h, sub, len, w) = (0.25, 0.5, if pml { 16.0 } else { 8.0 }, 1.0);
    let l = lines(-6.0, 6.0, h);
    let mut lz = lines(0.0, sub, sub / 2.0);
    if solid {
        lz.push(sub + 0.1);
    }
    lz.extend([0.8, 1.2, 1.7, 2.3]);
    let t = deg.to_radians();
    let c = [0.37 * h, 0.21 * h];
    let at = |u: f64, v: f64| {
        (
            c[0] + u * t.cos() - v * t.sin(),
            c[1] + u * t.sin() + v * t.cos(),
        )
    };
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for (a, v) in [l.clone(), l.clone(), lz].into_iter().enumerate() {
        m.grid.set_lines(a, v).unwrap();
    }
    let fr4 = m.add_property(Property::material("substrate", Material::new(4.0, 0.0)));
    m.add_primitive(fr4, BoxShape::new([-6.0, -6.0, 0.0], [6.0, 6.0, sub]))
        .unwrap();
    let verts = vec![
        at(-0.5 * len, -0.5 * w),
        at(0.5 * len, -0.5 * w),
        at(0.5 * len, 0.5 * w),
        at(-0.5 * len, 0.5 * w),
    ];
    let (prop, prim) = if solid {
        (
            Property::metal("strip"),
            Primitive::new(LinPoly::new(2, sub, 0.1, verts)),
        )
    } else {
        (
            Property::conducting_sheet("strip", 2e5, 35e-6),
            Primitive::new(Polygon::new(2, sub, verts)),
        )
    };
    let id = m.add_property(prop);
    m.add_primitive(id, prim.with_priority(10)).unwrap();
    let mut fdtd = Fdtd::new().threads(threads).max_frequency(12e9);
    if pml {
        for face in [Face::XMin, Face::XMax, Face::YMin, Face::YMax] {
            fdtd = fdtd.boundary(face, Boundary::Pml(Cpml::cells(6)));
        }
    }
    let mut opts = CompileOptions::default()
        .with_conformal(true)
        .with_conformal_outlines(ConformalOutlines::Slanted)
        .with_conformal_solids(ConformalOutlines::All)
        .with_sheet_model(SheetModel::SurfaceImpedance);
    if solid {
        opts = opts.with_lossy_metal("strip", Conductor::new(2e5));
    }
    let mut cm = compile(&m, &fdtd, &opts).unwrap();
    cm.surface_serial = serial;
    let (mut e, hnd) = cm.build().unwrap();
    e.set_fused_steps(FusedSteps::Auto);
    let rep = hnd.conformal.expect("conformal PEC");
    let op = e.operator(hnd.surface.as_ref().expect("surface impedance"));
    let (slabs, held) = (op.runs_in_slabs(), op.held_back());
    assert!(op.solid_edges() + op.sheet_edges() > 100);
    // E_z under the strip, 3 mm from its centre on either side.
    let near = |x: f64| (0..l.len()).min_by(|&a, &b| (l[a] - x).abs().total_cmp(&(l[b] - x).abs()));
    let end = |s: f64| {
        let (x, y) = at(s * 3.0, 0.0);
        [near(x).unwrap(), near(y).unwrap(), 0]
    };
    // A Gaussian pulse of 60 ps.
    let pulse = |t: f64| (-((t - 180e-12) / 60e-12).powi(2)).exp();
    e.add_operator(EdgeSource::new(pulse).edge(Axis::Z, end(-1.0), 1.0));
    let p = e.add_operator(Probe::voltage(Axis::Z, end(1.0)));
    let mut w = vec![e.energy().to_bits()];
    for _ in 0..6 {
        e.run(60);
        w.push(e.energy().to_bits());
        // The energy recorded within a step, as the end criterion takes it.
        w.push(e.step_with_energy().to_bits());
    }
    let v: Vec<u64> = e
        .operator(&p)
        .values()
        .iter()
        .map(|x| x.to_bits())
        .collect();
    let fields: Vec<u32> = Axis::ALL
        .iter()
        .flat_map(|&a| {
            let mut v = e.u_component(a);
            v.extend(e.i_component(a));
            v
        })
        .map(f32::to_bits)
        .collect();
    ((w, v, fields), rep.interpolated + rep.merged, slabs, held)
}

/// Sheets and solid metal whose outlines cross the cells so that
/// conformal PEC interpolates or merges small faces: the slabs run the
/// edges, and energies, probe and fields equal those of the serial hooks
/// bit for bit, for any thread count.
#[test]
fn slabs_equal_the_serial_hooks_beside_merged_faces() {
    for (deg, solid, pml) in [
        (17.0, false, false),
        (30.0, false, false),
        (17.0, true, false),
        (30.0, true, false),
        (17.0, false, true),
        (30.0, true, true),
    ] {
        let case = format!("{deg}°, solid {solid}, layers {pml}");
        let (want, faces, slabs, held) = run(deg, solid, pml, 1, true);
        assert!(!slabs && held == 0);
        assert!(faces > 0, "{case}: no merged faces");
        assert!(want.2.iter().any(|&v| f32::from_bits(v) != 0.0));
        assert!(want.0.iter().any(|&v| f64::from_bits(v) != 0.0));
        assert_eq!(
            run(deg, solid, pml, 3, true).0,
            want,
            "{case}: serial hooks, 3 threads"
        );
        for threads in [1, 2, 3, 5] {
            let (got, _, slabs, held) = run(deg, solid, pml, threads, false);
            eprintln!("{case}, {threads} threads: {faces} faces, {held} edges held");
            assert!(slabs, "{case}: not in the slabs");
            assert!(held > 0, "{case}: the merged faces read no edge");
            assert!(got == want, "{case}, {threads} threads");
        }
    }
}
