//! Compile step: csx model + Fdtd settings → engine (materials, metal,
//! sheets, boundaries, units, time step).

use csx::primitives::BoxShape;
use csx::{ContinuousStructure, CoordSystem, Primitive, Property};
use ems::compile::{CompileOptions, compile};
use ems::constants::C0;
use ems::engine::{
    CellMaterials, Coefficients, EdgeSource, Engine, FaceCondition, Grid, Material, Probe,
    TimeStep, courant_limit,
};
use ems::{Axis, Boundaries, Boundary, Cpml, Error, Face, Fdtd, Mur, Simulation};

fn material(eps: f64, kappa: f64, mu: f64, sigma: f64) -> csx::Material {
    let mut m = csx::Material::new(eps, kappa);
    m.mue.set(mu);
    m.sigma.set(sigma);
    m
}

/// Model with `cells` uniform cells of `d` drawing units per axis.
fn uniform_model(cells: [usize; 3], d: f64, unit: f64) -> ContinuousStructure {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = unit;
    for (a, &n) in cells.iter().enumerate() {
        m.grid.set_lines(a, (0..=n).map(|i| i as f64 * d)).unwrap();
    }
    m
}

fn add_box(m: &mut ContinuousStructure, p: Property, lo: [f64; 3], hi: [f64; 3], prio: i32) {
    let id = m.add_property(p);
    m.add_primitive(
        id,
        Primitive::new(BoxShape::new(lo, hi)).with_priority(prio),
    )
    .unwrap();
}

fn serial() -> Fdtd {
    Fdtd::new().threads(1)
}

// ---------------------------------------------------------------- cavity

fn pulse(f0: f64, bw: f64) -> (impl Fn(f64) -> f64 + Send + 'static, f64) {
    let tau = 1.0 / (std::f64::consts::PI * bw);
    let t0 = 4.0 * tau;
    let f = move |t: f64| {
        let x = (t - t0) / tau;
        (-x * x).exp() * (2.0 * std::f64::consts::PI * f0 * (t - t0)).cos()
    };
    (f, 2.0 * t0)
}

fn spectrum(times: &[f64], values: &[f64], t_start: f64, f: f64) -> f64 {
    let first = times.iter().position(|&t| t >= t_start).unwrap();
    let (t, v) = (&times[first..], &values[first..]);
    let n = t.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (q, (&tt, &x)) in t.iter().zip(v).enumerate() {
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * q as f64 / (n - 1.0)).cos();
        let ph = -2.0 * std::f64::consts::PI * f * tt;
        re += w * x * ph.cos();
        im += w * x * ph.sin();
    }
    re.hypot(im)
}

fn peak(times: &[f64], values: &[f64], t_start: f64, lo: f64, hi: f64) -> f64 {
    let s = |f: f64| spectrum(times, values, t_start, f);
    let n = 200;
    let step = (hi - lo) / n as f64;
    let (best, _) = (0..=n)
        .map(|q| (q, s(lo + q as f64 * step)))
        .fold((0, f64::MIN), |a, b| if b.1 > a.1 { b } else { a });
    let (mut a, mut b) = (
        lo + (best as f64 - 1.0) * step,
        lo + (best as f64 + 1.0) * step,
    );
    let g = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..60 {
        let c = b - g * (b - a);
        let d = a + g * (b - a);
        if s(c) > s(d) {
            b = d;
        } else {
            a = c;
        }
    }
    0.5 * (a + b)
}

/// A 100 × 45 × 70 mm cavity whose walls are six zero-thickness metal
/// sheets, inside a mesh extended by 2 cells per side (PMC outer faces),
/// filled with a csx dielectric of eps_r = 2.25. The (1,0,1) and (2,0,1)
/// resonances must match the analytic values.
#[test]
fn metal_sheet_cavity_from_csx() {
    let d = 5.0; // mm
    let dims = [100.0, 45.0, 70.0];
    let pad = 2.0 * d;
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3;
    for (a, len) in dims.iter().enumerate() {
        let n = (len / d) as usize + 4;
        model
            .grid
            .set_lines(a, (0..=n).map(|i| i as f64 * d - pad))
            .unwrap();
    }
    let eps = 2.25;
    add_box(
        &mut model,
        Property::material("fill", csx::Material::new(eps, 0.0)),
        [0.0; 3],
        dims,
        0,
    );
    let metal = model.add_property(Property::metal("walls"));
    for a in 0..3 {
        for side in [0.0, dims[a]] {
            let (mut lo, mut hi) = ([0.0; 3], dims);
            lo[a] = side;
            hi[a] = side;
            model
                .add_primitive(
                    metal,
                    Primitive::new(BoxShape::new(lo, hi)).with_priority(10),
                )
                .unwrap();
        }
    }
    let fdtd = Fdtd::new()
        .boundaries(Boundaries::all(Boundary::Pmc))
        .threads(2);
    let mut sim = Simulation::new(&model, &fdtd).unwrap();
    assert!(sim.warnings().is_empty(), "{:?}", sim.warnings());
    // Node of a point (metres) inside the cavity.
    let node = |p: [f64; 3]| -> [usize; 3] {
        core::array::from_fn(|a| ((p[a] * 1e3 + pad) / d).round() as usize)
    };
    let src = node([0.3 * dims[0], 0.3 * dims[1], 0.4 * dims[2]].map(|v| v * 1e-3));
    let prb = node([0.7 * dims[0], 0.5 * dims[1], 0.6 * dims[2]].map(|v| v * 1e-3));
    let scale = 1.0 / eps.sqrt();
    let (f, t_end) = pulse(3.2e9 * scale, 2.5e9 * scale);
    let e = sim.engine_mut();
    e.add_operator(EdgeSource::new(f).edge(Axis::Y, src, 1.0));
    let h = e.add_operator(Probe::voltage(Axis::Y, prb));
    e.run(4500);
    let p = e.operator(&h);
    let dm = dims.map(|v| v * 1e-3);
    for (m, n) in [(1.0, 1.0), (2.0, 1.0)] {
        let fa = C0 / (2.0 * eps.sqrt()) * ((m / dm[0]).powi(2) + (n / dm[2]).powi(2)).sqrt();
        let f = peak(p.times(), p.values(), t_end, 0.95 * fa, 1.05 * fa);
        let err = (f - fa) / fa;
        eprintln!(
            "mode ({m},0,{n}): {f:.6e} vs {fa:.6e} ({:.3}%)",
            err * 100.0
        );
        assert!(
            err.abs() < 0.005,
            "mode ({m},0,{n}) off by {:.3}%",
            err * 100.0
        );
    }
    // Outside the cavity nothing gets through the walls.
    let outside = sim.engine().u(Axis::Y, [1, 3, 3]);
    assert!(outside.abs() < 1e-6, "{outside}");
}

// ------------------------------------------------------------- materials

/// Dielectric half space x < 3 mm in a 6-cell cube: csx edge values and the
/// coefficients against the per-cell path.
#[test]
fn dielectric_edges_match_cell_path() {
    let mut model = uniform_model([6, 6, 6], 1.0, 1e-3);
    add_box(
        &mut model,
        Property::material("diel", material(4.0, 0.5, 2.0, 0.0)),
        [0.0, 0.0, 0.0],
        [3.0, 6.0, 6.0],
        0,
    );
    let c = compile(&model, &serial(), &CompileOptions::default()).unwrap();
    let g = &c.grid;
    let m = &c.materials;
    // E_y edges: inside (x = 1), on the interface (x = 3), outside (x = 5).
    assert_eq!(m.eps_r(Axis::Y)[g.index([1, 2, 2])], 4.0);
    assert_eq!(m.eps_r(Axis::Y)[g.index([3, 2, 2])], 2.5);
    assert_eq!(m.eps_r(Axis::Y)[g.index([5, 2, 2])], 1.0);
    assert_eq!(m.kappa(Axis::Y)[g.index([3, 2, 2])], 0.25);
    // E_x edges lie in one cell along x: no averaging.
    assert_eq!(m.eps_r(Axis::X)[g.index([2, 2, 2])], 4.0);
    assert_eq!(m.eps_r(Axis::X)[g.index([3, 2, 2])], 1.0);
    // H_x through the interface plane: harmonic mean of 2 and 1.
    let mu = m.mu_r(Axis::X)[g.index([3, 2, 2])];
    assert!((mu - 4.0 / 3.0).abs() < 1e-12, "{mu}");
    assert_eq!(m.mu_r(Axis::Y)[g.index([2, 2, 2])], 2.0);

    // Same geometry through the per-cell path.
    let mut cells = CellMaterials::vacuum(g);
    cells.fill_box(
        [0..3, 0..6, 0..6],
        Material::dielectric(4.0).with_kappa(0.5).with_mu_r(2.0),
    );
    let dt = TimeStep::default().resolve(g).unwrap();
    let a = Coefficients::compute(g, &cells, &c.faces, dt).unwrap();
    let sim = Simulation::new(&model, &serial()).unwrap();
    let b = sim.engine().coefficients();
    assert_eq!(sim.engine().dt(), dt);
    for axis in Axis::ALL {
        for (x, y) in [
            (a.ca(axis), b.ca(axis)),
            (a.cb(axis), b.cb(axis)),
            (a.da(axis), b.da(axis)),
            (a.db(axis), b.db(axis)),
        ] {
            for (p, q) in x.iter().zip(y) {
                assert!((p - q).abs() <= 4.0 * f32::EPSILON * p.abs().max(q.abs()));
            }
        }
    }
}

/// Magnetic loss: csx averages sigma arithmetically over the two half
/// cells, the per-cell path combines them as series R–L branches. Equal on
/// homogeneous dual edges, different across an interface.
#[test]
fn magnetic_loss_averaging_differs_at_interfaces() {
    let mut model = uniform_model([6, 4, 4], 1.0, 1e-3);
    add_box(
        &mut model,
        Property::material("lossy", material(1.0, 0.0, 3.0, 100.0)),
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 4.0],
        0,
    );
    let c = compile(&model, &serial(), &CompileOptions::default()).unwrap();
    let g = &c.grid;
    let mut cells = CellMaterials::vacuum(g);
    cells.fill_box(
        [0..3, 0..4, 0..4],
        Material::VACUUM.with_mu_r(3.0).with_sigma_m(100.0),
    );
    let from_cells = ems::EdgeMaterials::from_cells(g, &cells).unwrap();
    let interior = g.index([1, 1, 1]);
    let iface = g.index([3, 1, 1]);
    let s = |m: &ems::EdgeMaterials, i| m.sigma_m(Axis::X)[i];
    assert_eq!(s(&c.materials, interior), 100.0);
    assert!((s(&from_cells, interior) - 100.0).abs() < 1e-9);
    // csx: arithmetic mean 50.
    assert_eq!(s(&c.materials, iface), 50.0);
    // Cell path: R = L²·Σ R_c/L_c² with L_c ∝ mu_c/half: per unit A/l*,
    // sigma_eff = mu_eff²·(1/2)·(sigma/mu²) = (1.5)²·0.5·100/9 = 12.5.
    let want = 1.5f64.powi(2) * 0.5 * 100.0 / 9.0;
    assert!((s(&from_cells, iface) - want).abs() < 1e-9);
    // Permeabilities agree (both harmonic).
    assert!((c.materials.mu_r(Axis::X)[iface] - from_cells.mu_r(Axis::X)[iface]).abs() < 1e-12);
}

// ----------------------------------------------------------------- metal

#[test]
fn metal_sheet_marks_in_plane_edges() {
    let mut model = uniform_model([6, 5, 6], 1.0, 1e-3);
    add_box(
        &mut model,
        Property::metal("sheet"),
        [1.0, 1.0, 3.0],
        [4.0, 3.0, 3.0],
        0,
    );
    let sim = Simulation::new(&model, &serial()).unwrap();
    assert!(sim.warnings().is_empty(), "{:?}", sim.warnings());
    let e = sim.engine();
    let g = e.grid();
    let n = g.nodes();
    let co = e.coefficients();
    let mut expected = 0;
    for axis in Axis::ALL {
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let p = [i, j, k];
                    if p[axis.index()] + 1 >= n[axis.index()] {
                        continue;
                    }
                    let inside = k == 3
                        && match axis {
                            Axis::X => (1..4).contains(&i) && (1..=3).contains(&j),
                            Axis::Y => (1..=4).contains(&i) && (1..3).contains(&j),
                            Axis::Z => false,
                        };
                    let on_face =
                        (0..3).any(|t| t != axis.index() && (p[t] == 0 || p[t] == n[t] - 1));
                    let cb = co.cb(axis)[g.index(p)];
                    if inside {
                        expected += 1;
                        assert_eq!(cb, 0.0, "{axis}:{p:?} should be PEC");
                    } else if !on_face {
                        assert!(cb > 0.0, "{axis}:{p:?} should not be PEC");
                    }
                }
            }
        }
    }
    assert_eq!(expected, 3 * 3 + 4 * 2);
}

#[test]
fn conducting_sheet_is_pec_with_warning() {
    let mut model = uniform_model([4, 4, 4], 1.0, 1e-3);
    add_box(
        &mut model,
        Property::conducting_sheet("cu", 5.8e7, 35e-6),
        [0.0, 0.0, 2.0],
        [4.0, 4.0, 2.0],
        0,
    );
    let c = compile(&model, &serial(), &CompileOptions::default()).unwrap();
    assert!(c.materials.is_pec_edge(Axis::X, [1, 1, 2]));
    assert!(c.materials.is_pec_edge(Axis::Y, [2, 1, 2]));
    assert!(!c.materials.is_pec_edge(Axis::Z, [1, 1, 2]));
    assert!(!c.materials.is_pec_edge(Axis::X, [1, 1, 1]));
    assert_eq!(c.sources.sheets.len(), 1);
    assert!(
        c.warnings
            .iter()
            .any(|w| w.contains("ConductingSheet") && w.contains("PEC")),
        "{:?}",
        c.warnings
    );
}

#[test]
fn material_over_metal_priority() {
    // A dielectric block with higher priority than an overlapping metal
    // block: edges inside the overlap are not shorted.
    let mut model = uniform_model([6, 6, 6], 1.0, 1e-3);
    add_box(&mut model, Property::metal("m"), [1.0; 3], [5.0; 3], 0);
    add_box(
        &mut model,
        Property::material("d", csx::Material::new(3.0, 0.0)),
        [2.0; 3],
        [4.0; 3],
        5,
    );
    let c = compile(&model, &serial(), &CompileOptions::default()).unwrap();
    assert!(c.materials.is_pec_edge(Axis::X, [1, 1, 1]));
    assert!(!c.materials.is_pec_edge(Axis::X, [2, 3, 3]));
    assert_eq!(c.materials.eps_r(Axis::X)[c.grid.index([2, 3, 3])], 3.0);
}

// ------------------------------------------------------------ boundaries

#[test]
fn boundaries_map_to_conditions_and_operators() {
    let model = uniform_model([12, 10, 12], 1.0, 1e-3);
    let fdtd = serial()
        .boundary(Face::XMin, Boundary::Mur(Mur::with_phase_velocity(2e8)))
        .boundary(Face::XMax, Boundary::Pml(Cpml::cells(4)))
        .boundary(Face::YMin, Boundary::Pmc)
        .boundary(Face::YMax, Boundary::Pec)
        .boundary(Face::ZMin, Boundary::Pml(Cpml::cells(3)))
        .boundary(Face::ZMax, Boundary::mur());
    let c = compile(&model, &fdtd, &CompileOptions::default()).unwrap();
    for f in Face::ALL {
        let want = if f == Face::YMin {
            FaceCondition::Pmc
        } else {
            FaceCondition::Pec
        };
        assert_eq!(c.faces[f], want, "{f}");
    }
    assert_eq!(c.mur, vec![(Face::XMin, 2e8), (Face::ZMax, C0)]);
    assert_eq!(
        c.pml,
        vec![(Face::XMax, Cpml::cells(4)), (Face::ZMin, Cpml::cells(3))]
    );

    let mut sim = Simulation::new(&model, &fdtd).unwrap();
    let h = sim.boundaries().clone();
    let e = sim.engine();
    let faces: Vec<Face> = e.operator(&h.cpml.unwrap()).layers().map(|l| l.0).collect();
    assert_eq!(faces, vec![Face::XMax, Face::ZMin]);
    let _mur = e.operator(&h.mur.unwrap());
    // PMC face keeps its tangential edges, PEC face does not.
    let g = e.grid();
    assert!(e.coefficients().cb(Axis::X)[g.index([5, 0, 5])] > 0.0);
    assert_eq!(e.coefficients().cb(Axis::X)[g.index([5, 10, 5])], 0.0);
    sim.engine_mut().run(5);

    // No absorbing faces: no operators.
    let sim = Simulation::new(&model, &serial()).unwrap();
    assert!(sim.boundaries().mur.is_none() && sim.boundaries().cpml.is_none());

    // A PML thicker than the mesh is rejected.
    let bad = serial().boundary(Face::XMax, Boundary::pml(20));
    assert!(Simulation::new(&model, &bad).is_err());
}

// ------------------------------------------------------- units, timestep

#[test]
fn delta_unit_scales_to_metres() {
    let model = uniform_model([4, 5, 6], 2.0, 1e-3); // 2 mm cells
    let sim = Simulation::new(&model, &serial()).unwrap();
    let g = sim.grid();
    assert_eq!(sim.delta_unit(), 1e-3);
    assert_eq!(g.nodes(), [5, 6, 7]);
    assert!((g.lines(Axis::Z)[6] - 0.012).abs() < 1e-15);
    assert!((g.min_delta(Axis::X) - 0.002).abs() < 1e-15);
    let want = Grid::uniform([4, 5, 6], [0.002; 3]).unwrap();
    assert!((sim.engine().dt() - 0.99 * courant_limit(&want)).abs() < 1e-24);

    // Time step factor and override.
    let s = Simulation::new(&model, &serial().timestep_factor(0.5)).unwrap();
    assert!((s.engine().dt() - 0.5 * courant_limit(&want)).abs() < 1e-24);
    let dt = 0.3 * courant_limit(&want);
    let s = Simulation::new(&model, &serial().timestep(dt)).unwrap();
    assert_eq!(s.engine().dt(), dt);
    let too_big = serial().timestep(2.0 * courant_limit(&want));
    assert!(Simulation::new(&model, &too_big).is_err());

    // Bad delta unit.
    let mut m2 = model.clone();
    m2.grid.delta_unit = 0.0;
    assert!(matches!(
        Simulation::new(&m2, &serial()),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn cylindrical_grid_is_unsupported() {
    let mut model = uniform_model([4, 4, 4], 1.0, 1e-3);
    model.grid.coord_system = CoordSystem::Cylindrical;
    assert!(matches!(
        Simulation::new(&model, &serial()),
        Err(Error::Unsupported(_))
    ));
}

// ------------------------------------------------------------------- run

#[test]
fn run_uses_settings_and_maps_sources() {
    let mut model = uniform_model([4, 4, 4], 1.0, 1e-3);
    add_box(
        &mut model,
        Property::excitation("src", csx::properties::Excitation::new([0.0, 0.0, 1.0])),
        [2.0, 2.0, 1.0],
        [2.0, 2.0, 3.0],
        0,
    );
    let fdtd = serial()
        .max_timesteps(37)
        .waveform(ems::Waveform::gaussian(1e9, 0.5e9));
    let mut sim = Simulation::new(&model, &fdtd).unwrap();
    // The only warning is the mesh check's: 37 steps are too few for the
    // pulse.
    assert_eq!(sim.warnings().len(), 1, "{:?}", sim.warnings());
    assert!(
        sim.warnings()[0].starts_with("mesh warning: the excitation needs"),
        "{:?}",
        sim.warnings()
    );
    let opts = CompileOptions {
        mesh_check: false,
        ..CompileOptions::default()
    };
    let quiet = Simulation::with_options(&model, &fdtd, &opts).unwrap();
    assert!(quiet.warnings().is_empty(), "{:?}", quiet.warnings());
    // Discretised and mapped onto a soft source.
    assert_eq!(sim.source("src").unwrap().edges().count(), 2);
    let ex = &sim.sources().excitations;
    assert_eq!(ex.len(), 1);
    assert_eq!(ex[0].amplitudes.len(), 2);
    let (axis, node) = sim.node_map().edge(ex[0].amplitudes[0].0);
    assert_eq!(axis, Axis::Z);
    assert_eq!([node[0], node[1]], [2, 2]);
    let lim = sim.run_limits();
    assert_eq!(lim.max_timesteps, 37);
    // MaxTime limits the step count via Fdtd::timestep_limit.
    let dt = sim.engine().dt();
    let timed = Simulation::new(&model, &fdtd.clone().max_time(10.5 * dt)).unwrap();
    assert_eq!(timed.run_limits().max_timesteps, 11);
    let long = Simulation::new(&model, &fdtd.clone().max_time(1e6 * dt)).unwrap();
    assert_eq!(long.run_limits().max_timesteps, 37);
    assert!(lim.excitation_end.unwrap() > 0.0);
    let out = sim.run().unwrap();
    assert_eq!(out.outcome().unwrap().steps, 37);
    assert_eq!(sim.engine().timestep(), 37);
    // Building directly from the compiled data gives the same engine.
    let c = compile(&model, &fdtd, &CompileOptions::default()).unwrap();
    let (e, _) = c.build().unwrap();
    assert_eq!(e.dt(), sim.engine().dt());
    let _: Engine = sim.into_engine();
}
