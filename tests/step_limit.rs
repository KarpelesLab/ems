//! The time step from the cells ([`ems::StepLimit::Cells`], design
//! §5.4.1): what it is on boards, and that nothing grows at it.
//!
//! Stability is tested on the scheme as it is assembled, not on its
//! theory: every model is compiled with all its operators (CPML, Debye
//! poles, sheets as ladders and as surface impedances, lossy metal,
//! conformal PEC, lumped ports), its edges are filled with noise, which
//! has every mode of the grid in it up to the highest, and it is stepped
//! without a source far beyond the length of a run. An unstable mode
//! grows by a constant factor per step, so the energy would leave any
//! bound; it must stay below its start.

use csx::primitives::{BoxShape, LinPoly, Polygon};
use csx::properties::{DebyeMaterial, DebyePole};
use csx::{ContinuousStructure, Primitive, Property};
use ems::compile::{ConformalOutlines, SheetModel, StepChoice, StepObstacle};
use ems::engine::surface::Conductor;
use ems::engine::{EdgeMaterials, Engine, Grid, Material, TimeStep, cell_limit, courant_limit};
use ems::port::LumpedPort;
use ems::{
    Axis, Boundaries, Boundary, CompileOptions, Cpml, Fdtd, Mur, Simulation, StepLimit, Waveform,
};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Copper {
    /// Perfect conductor without thickness.
    Pec,
    /// Conducting sheets on the ladder.
    Ladder,
    /// Conducting sheets on the rational surface impedance.
    Surface,
    /// 35 µm of lossy metal.
    Solid,
}

#[derive(Clone, Copy, Debug)]
struct Board {
    copper: Copper,
    /// Angle of the strip to the x axis, degrees (0: on the mesh).
    angle: f64,
    debye: bool,
    /// Cells of the laminate's 0.2 mm (the smallest cells of the mesh).
    layers: usize,
    /// Lateral cell, mm.
    cell: f64,
    /// Mesh lines 20 µm on each side of the strip's edges (y).
    fine_edges: bool,
    pml: bool,
    factor: f64,
    limit: StepLimit,
}

impl Board {
    fn new(copper: Copper) -> Self {
        Self {
            copper,
            angle: 0.0,
            debye: true,
            layers: 2,
            cell: 0.5,
            fine_edges: false,
            pml: true,
            factor: 0.95,
            limit: StepLimit::Cells,
        }
    }

    fn model(&self) -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        let n = (12.0 / self.cell).round() as usize;
        let lateral: Vec<f64> = (0..=n).map(|i| i as f64 * self.cell).collect();
        m.grid.set_lines(0, lateral.clone()).unwrap();
        let mut y = lateral;
        if self.fine_edges {
            y.extend([4.48, 4.52, 5.98, 6.02]);
            y.sort_by(f64::total_cmp);
            y.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        }
        m.grid.set_lines(1, y).unwrap();
        // Ground at 2.0, the laminate to 2.2, air above and below.
        let top = 2.2;
        let mut z = vec![0.0, 0.4, 0.8, 1.1, 1.4, 1.6, 1.75];
        z.extend((0..=self.layers).map(|i| 2.0 + 0.2 * i as f64 / self.layers as f64));
        if self.copper == Copper::Solid {
            z.push(top + 0.035);
        }
        z.extend([2.45, 2.7, 3.0, 3.4, 3.8, 4.2, 4.6]);
        m.grid.set_lines(2, z).unwrap();
        let laminate = if self.debye {
            Property::debye_material(
                "laminate",
                DebyeMaterial {
                    material: csx::Material::new(4.1, 1e-4),
                    poles: vec![DebyePole::new(0.12, 1.6e-11), DebyePole::new(0.1, 1.6e-10)],
                },
            )
        } else {
            Property::material("laminate", csx::Material::new(4.3, 0.0))
        };
        let id = m.add_property(laminate);
        m.add_primitive(
            id,
            Primitive::new(BoxShape::new([0.0, 0.0, 2.0], [12.0, 12.0, top])),
        )
        .unwrap();
        let copper = |name: &str| match self.copper {
            Copper::Pec | Copper::Solid => Property::metal(name),
            Copper::Ladder | Copper::Surface => Property::conducting_sheet(name, 5.8e7, 35e-6),
        };
        let ground = m.add_property(copper("ground"));
        m.add_primitive(
            ground,
            Primitive::new(BoxShape::new([0.0, 0.0, 2.0], [12.0, 12.0, 2.0])).with_priority(10),
        )
        .unwrap();
        // The strip: 6 mm × 1.5 mm about (6, 5.25), turned by `angle`.
        let (c, s) = (self.angle.to_radians().cos(), self.angle.to_radians().sin());
        let corners: Vec<(f64, f64)> = [(-3.0, -0.75), (3.0, -0.75), (3.0, 0.75), (-3.0, 0.75)]
            .into_iter()
            .map(|(x, y)| (6.0 + c * x - s * y, 5.25 + s * x + c * y))
            .collect();
        let strip = m.add_property(copper("strip"));
        let shape = if self.copper == Copper::Solid {
            Primitive::new(LinPoly::new(2, top, 0.035, corners))
        } else {
            Primitive::new(Polygon::new(2, top, corners))
        };
        m.add_primitive(strip, shape.with_priority(10)).unwrap();
        // A 50 Ω port from the strip's middle line to the ground, near
        // each end, on mesh nodes.
        for (i, t) in [(1, -2.5), (2, 2.5)] {
            let snap = |v: f64| (v / self.cell).round() * self.cell;
            let (x, y) = (snap(6.0 + c * t), snap(5.25 + s * t));
            LumpedPort::new(i, 50.0, [x, y, top], [x, y, 2.0], Axis::Z)
                .priority(30)
                .add_to(&mut m)
                .unwrap();
        }
        m
    }

    fn fdtd(&self) -> Fdtd {
        let pml = Cpml::cells(4).alpha_max(2.0 * std::f64::consts::PI * 8.854e-12 * 5e9);
        Fdtd::new()
            .waveform(Waveform::gaussian(5e9, 5e9))
            .boundaries(if self.pml {
                Boundaries::all(Boundary::Pml(pml))
            } else {
                Boundaries::all(Boundary::Pec)
            })
            .max_timesteps(10)
            .timestep_factor(self.factor)
            .step_limit(self.limit)
            .threads(1)
    }

    fn options(&self) -> CompileOptions {
        let mut o = CompileOptions::default()
            .with_conformal(true)
            .with_conformal_outlines(ConformalOutlines::Slanted)
            .with_conformal_solids(ConformalOutlines::Slanted)
            .with_surface_band(5e7, 1e10);
        match self.copper {
            Copper::Surface => o = o.with_sheet_model(SheetModel::SurfaceImpedance),
            Copper::Solid => {
                o = o
                    .with_lossy_metal("strip", Conductor::new(5.8e7))
                    .with_lossy_metal("ground", Conductor::new(5.8e7));
            }
            _ => {}
        }
        o
    }

    fn simulation(&self) -> Simulation {
        Simulation::with_options(&self.model(), &self.fdtd(), &self.options()).unwrap()
    }
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
}

/// Noise on every edge the kernel updates.
fn fill_with_noise(engine: &mut Engine, mut seed: u64) {
    let n = engine.grid().nodes();
    let grid = engine.grid().clone();
    for a in Axis::ALL {
        let cb = engine.coefficients().cb(a).to_vec();
        let mut f = engine.fields_mut();
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let v = lcg(&mut seed);
                    if cb[grid.index([i, j, k])] > 0.0 {
                        f.u.set(a, [i, j, k], v);
                    }
                }
            }
        }
    }
}

/// Steps `sim` from noise for `steps` steps and returns the energy at the
/// start and the largest and the last of the energies read every 200
/// steps after the first 200.
fn noise_run(sim: &mut Simulation, steps: u64) -> (f64, f64, f64) {
    fill_with_noise(sim.engine_mut(), 0x5eed);
    sim.step(1);
    let start = sim.engine().energy();
    assert!(start.is_finite() && start > 0.0);
    let (mut most, mut last) = (0.0f64, start);
    let mut done = 1;
    while done < steps {
        sim.step(200);
        done += 200;
        last = sim.engine().energy();
        assert!(last.is_finite(), "energy {last} after {done} steps");
        most = most.max(last);
    }
    (start, most, last)
}

fn assert_bounded(name: &str, sim: &mut Simulation, steps: u64) {
    let dt = sim.engine().dt();
    let (start, most, last) = noise_run(sim, steps);
    eprintln!(
        "{name}: dt {dt:.4e} s, energy {start:.3e} at the start, at most {most:.3e} after 200 \
         steps, {last:.3e} after {steps}"
    );
    assert!(most <= start, "{name}: {most:e} above the start {start:e}");
}

/// The step ratio to the Courant step, and what decided.
fn gain(b: &Board) -> (f64, StepChoice) {
    let sim = b.simulation();
    let courant = Board {
        limit: StepLimit::Courant,
        ..*b
    }
    .simulation();
    let report = sim_step(&b.model(), &b.fdtd(), &b.options());
    assert_eq!(report.dt, sim.engine().dt());
    (sim.engine().dt() / courant.engine().dt(), report.limit)
}

fn sim_step(
    model: &ContinuousStructure,
    fdtd: &Fdtd,
    opts: &CompileOptions,
) -> ems::compile::StepReport {
    ems::compile::compile(model, fdtd, opts)
        .unwrap()
        .step
        .expect("a Cartesian model on the automatic step")
}

/// A uniform grid filled with `ε_r` = 4: the bound is twice the Courant
/// limit, the scheme is bounded just below it and grows just above.
#[test]
fn the_bound_is_sharp_in_a_dielectric() {
    let g = Grid::uniform([16, 16, 16], [1e-3, 1.2e-3, 0.4e-3]).unwrap();
    let mats = EdgeMaterials::new(&g, Material::dielectric(4.0));
    let limit = cell_limit(&g, &mats).unwrap();
    assert!((limit / courant_limit(&g) - 2.0).abs() < 1e-12);
    let run = |factor: f64| {
        let mut e = Engine::builder(g.clone())
            .edge_materials(mats.clone())
            .timestep(TimeStep::Bound(factor * limit))
            .threads(1)
            .build()
            .unwrap();
        fill_with_noise(&mut e, 3);
        e.step();
        let start = e.energy();
        e.run(4000);
        e.energy() / start
    };
    let below = run(0.999);
    assert!(below > 0.0 && below < 1.0 + 1e-3, "{below}");
    // (The highest mode of sixteen cells between walls is 0.5 % below
    // the grid's.)
    let above = run(1.03);
    assert!(above.is_nan() || above >= 1e6, "{above}");
}

/// On a board the cells give a longer step: the thinnest cells are in the
/// laminate, and those of the air above are not the narrowest.
#[test]
fn a_board_takes_a_longer_step() {
    for copper in [Copper::Pec, Copper::Ladder, Copper::Surface] {
        let (ratio, limit) = gain(&Board::new(copper));
        eprintln!("{copper:?}: {ratio:.3} times the Courant step ({limit:?})");
        assert_eq!(limit, StepChoice::Cells, "{copper:?}");
        // 0.1 mm cells of laminate (ε∞ 4.1) under 0.25 mm cells of
        // air: between the air's cells and the laminate's own.
        assert!(ratio > 1.3 && ratio < 2.03, "{copper:?}: {ratio}");
    }
    // Lossy metal 35 µm thick: its cells are in air, the smallest of
    // every axis in one cell, which by itself has the Courant limit.
    // With the shares balanced the thin cells have most of the dual
    // lengths they share with the laminate's cells below and the air's
    // above, and their edges on the laminate have the laminate's part.
    let (ratio, limit) = gain(&Board::new(Copper::Solid));
    eprintln!("Solid: {ratio:.3} times the Courant step ({limit:?})");
    assert!((2.0..3.0).contains(&ratio), "{ratio} ({limit:?})");
    // The Courant step where it is asked for, and where a part of the
    // model is outside the bound.
    let b = Board {
        limit: StepLimit::Courant,
        ..Board::new(Copper::Pec)
    };
    let r = sim_step(&b.model(), &b.fdtd(), &b.options());
    assert_eq!((r.limit, r.dt), (StepChoice::Courant, r.courant));
    let b = Board::new(Copper::Pec);
    let mur = b
        .fdtd()
        .boundaries(Boundaries::all(Boundary::Mur(Mur::default())));
    let r = sim_step(&b.model(), &mur, &b.options());
    assert_eq!(r.limit, StepChoice::CourantKept(StepObstacle::Mur));
    assert_eq!(r.dt, r.courant);
}

/// The bound against the estimate: the power iteration on the kernel's
/// own coefficients approaches the largest eigenvalue from below, so the
/// stability figure it reads at the step of the cells is at most the
/// factor wherever the kernel holds the whole lossless scheme (perfect
/// copper), and says how much of the limit the bound takes.
#[test]
fn the_bound_is_not_above_the_estimate() {
    use ems::engine::conformal::stability_factor;
    for (copper, layers, fine_edges, debye) in [
        (Copper::Pec, 2, false, true),
        (Copper::Pec, 2, false, false),
        (Copper::Pec, 8, true, false),
        (Copper::Pec, 4, true, true),
    ] {
        let b = Board {
            layers,
            fine_edges,
            debye,
            factor: 1.0,
            ..Board::new(copper)
        };
        let (ratio, limit) = gain(&b);
        let sim = b.simulation();
        let n = sim.engine().grid().nodes();
        let s = stability_factor(&sim.engine().coefficients(), n, &[], 3000);
        eprintln!(
            "{copper:?}, {layers} layers, fine edges {fine_edges}, Debye {debye}: {ratio:.4} times \
             the Courant step ({limit:?}), the estimate reads {s:.5} of the limit"
        );
        assert!(s <= 1.0 + 1e-6, "{s}");
        // The bound takes all but a few parts in a hundred of it.
        assert!(s > 0.95, "{s}");
    }
}

/// The step of the cells is a step of the model, whatever the threads.
#[test]
fn the_step_does_not_depend_on_the_threads() {
    let b = Board::new(Copper::Surface);
    let a = b.simulation().engine().dt();
    let c = Simulation::with_options(&b.model(), &b.fdtd().threads(3), &b.options())
        .unwrap()
        .engine()
        .dt();
    assert_eq!(a.to_bits(), c.to_bits());
}

/// Every copper model on the mesh, with the Debye laminate and the CPML:
/// nothing grows in 30 000 steps from noise.
#[test]
fn boards_on_the_mesh_stay_bounded() {
    for copper in [Copper::Pec, Copper::Ladder, Copper::Surface, Copper::Solid] {
        let b = Board::new(copper);
        assert_bounded(&format!("{copper:?}"), &mut b.simulation(), 30_000);
    }
}

/// The same at the largest factor the settings take (1), and in a closed
/// box, where nothing absorbs what the lossless part keeps, a part in a
/// thousand below it. With the shares balanced the bound is the limit
/// itself on such a board (the estimate reads 1.00000 of it,
/// `the_bound_is_not_above_the_estimate`), and at the limit the highest
/// mode of a lossless box grows with the step count, as it does at the
/// Courant limit of a uniform mesh: the closed box of perfect copper
/// read 26 times its energy after 30 000 steps at factor 1.
#[test]
fn boards_at_the_limit_stay_bounded() {
    for copper in [Copper::Pec, Copper::Ladder, Copper::Surface, Copper::Solid] {
        for pml in [true, false] {
            let factor = if pml { 1.0 } else { 0.999 };
            let b = Board {
                factor,
                pml,
                debye: pml,
                ..Board::new(copper)
            };
            assert_bounded(
                &format!("{copper:?}, factor {factor}, pml {pml}"),
                &mut b.simulation(),
                30_000,
            );
        }
    }
}

/// Strips oblique to the mesh: conformal PEC's cut cells, the sheets'
/// cut edges and faces, the lossy metal's cut surfaces.
#[test]
fn slanted_strips_stay_bounded() {
    for copper in [Copper::Pec, Copper::Ladder, Copper::Surface, Copper::Solid] {
        for angle in [17.0, 45.0] {
            let b = Board {
                angle,
                ..Board::new(copper)
            };
            let (ratio, limit) = gain(&b);
            let mut sim = b.simulation();
            let cut = sim
                .boundaries()
                .conformal
                .map_or(0, |r| r.edges + r.sheet_edges);
            assert!(cut > 0, "{copper:?} at {angle}°: no cut edge");
            assert!(ratio >= 1.0);
            assert_bounded(
                &format!(
                    "{copper:?} at {angle}°, {cut} cut edges, {ratio:.3} times the Courant step \
                     ({limit:?})"
                ),
                &mut sim,
                30_000,
            );
        }
    }
}

/// Adversarial meshes: cells of 20 µm beside cells of 0.5 mm along the
/// strip's edges, a laminate of eight layers of 25 µm, both under a strip
/// at an angle.
#[test]
fn strongly_graded_meshes_stay_bounded() {
    for copper in [Copper::Pec, Copper::Surface, Copper::Solid] {
        for (angle, layers, fine_edges) in [(0.0, 8, true), (30.0, 8, false), (30.0, 2, true)] {
            let b = Board {
                angle,
                layers,
                fine_edges,
                ..Board::new(copper)
            };
            let (ratio, limit) = gain(&b);
            assert_bounded(
                &format!(
                    "{copper:?} at {angle}°, {layers} layers, fine edges {fine_edges}, \
                     {ratio:.3} times the Courant step ({limit:?})"
                ),
                &mut b.simulation(),
                40_000,
            );
        }
    }
}

/// Cut cells at the limit: a strip at 30° over a laminate of eight
/// layers, whose cut cells are in the bound as the scheme has them and
/// keep the faces they keep at the Courant step, as perfect copper and
/// as lossy metal (whose edges beside the cut cells answer for
/// themselves by the floor of their capacitance), a part in a thousand
/// below the bound in a closed box and at it with the CPML.
#[test]
fn cut_cells_at_the_limit_stay_bounded() {
    for copper in [Copper::Pec, Copper::Solid] {
        for pml in [true, false] {
            let factor = if pml { 1.0 } else { 0.999 };
            let b = Board {
                angle: 30.0,
                layers: 8,
                factor,
                pml,
                debye: pml,
                ..Board::new(copper)
            };
            let (ratio, limit) = gain(&b);
            assert!(
                matches!(limit, StepChoice::Cells | StepChoice::CutCells) && ratio > 1.3,
                "{copper:?}: {ratio} ({limit:?})"
            );
            // What conformal PEC does with its faces is what it does at
            // the Courant step.
            let faces = |limit: StepLimit| {
                let r = Board { limit, ..b }
                    .simulation()
                    .boundaries()
                    .conformal
                    .expect("a slanted strip");
                (r.faces, r.kept, r.interpolated, r.hosts, r.merged, r.groups)
            };
            assert_eq!(faces(StepLimit::Cells), faces(StepLimit::Courant));
            assert!(faces(StepLimit::Cells).1 > 0, "{copper:?}: no face kept");
            assert_bounded(
                &format!(
                    "{copper:?} at 30°, factor {factor}, pml {pml}, {ratio:.3} times the Courant \
                     step"
                ),
                &mut b.simulation(),
                40_000,
            );
        }
    }
}

/// The long form of the above: 400 000 steps each, some forty times a
/// run of such a region.
#[test]
#[ignore]
fn boards_stay_bounded_for_long() {
    for copper in [Copper::Pec, Copper::Ladder, Copper::Surface, Copper::Solid] {
        for angle in [0.0, 30.0] {
            let b = Board {
                angle,
                layers: 4,
                ..Board::new(copper)
            };
            assert_bounded(
                &format!("{copper:?} at {angle}°"),
                &mut b.simulation(),
                400_000,
            );
        }
    }
}
