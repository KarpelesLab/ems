//! Validation of absorbing sheets (csx `AbsorbingBC`, first-order Mur on an
//! interior mesh plane): normal-incidence reflection of a TEM pulse
//! compared with the outer-face Mur boundary, the outward-normal sign
//! convention, and the csx / XML mapping.

use ems::constants::C0;
use ems::engine::boundary::abc::{AbcSheet, AbsorbingSheet, SheetVelocity};
use ems::engine::boundary::mur::MurBoundary;
use ems::engine::{EdgeSource, Engine, FaceCondition, FaceConditions, Grid, Probe};
use ems::{Axis, Face};

const D: f64 = 1e-3;
const TAU: f64 = 2.8e-11;

/// Gaussian pulse of 1/e half-width `TAU`, delayed by `4·TAU`.
fn gaussian(t: f64) -> f64 {
    let x = (t - 4.0 * TAU) / TAU;
    (-x * x).exp()
}

/// What terminates the line on the `+x` side.
#[derive(Clone, Copy)]
enum End {
    /// Nothing: a much longer reference line.
    None,
    /// Mur on the `x_max` face.
    MurFace,
    /// Absorbing sheet at x plane `plane` (outward normal `+x` if
    /// `positive`), PEC face behind it.
    Sheet { plane: usize, positive: bool },
}

/// Quasi-1-D TEM line along x (1 cell across): E along y, PEC y faces, PMC
/// z faces, Mur on `x_min`. Soft source at plane `src`; returns the E_y
/// voltage at `probe` for `steps` steps.
fn line(cells: usize, src: usize, probe: usize, end: End, steps: u64) -> Vec<f64> {
    let grid = Grid::uniform([cells, 1, 1], [D; 3]).unwrap();
    let faces = FaceConditions::default()
        .with(Face::ZMin, FaceCondition::Pmc)
        .with(Face::ZMax, FaceCondition::Pmc);
    let mut e = Engine::builder(grid)
        .faces(faces)
        .threads(1)
        .build()
        .unwrap();
    let mut mur = vec![(Face::XMin, C0)];
    if matches!(end, End::MurFace) {
        mur.push((Face::XMax, C0));
    }
    e.add_operator(MurBoundary::with_faces(&e, &mur).unwrap());
    if let End::Sheet { plane, positive } = end {
        let sheet = AbcSheet {
            normal: Axis::X,
            plane,
            positive,
            start: [0; 3],
            stop: [0, 1, 1],
        };
        let op = AbsorbingSheet::new(&e, &sheet, SheetVelocity::Fixed(C0)).unwrap();
        // y edges at k = 0, 1 (z edges lie on the PEC y faces).
        assert_eq!(op.edge_count(), 2);
        e.add_operator(op);
    }
    e.add_operator(
        EdgeSource::new(gaussian)
            .edge(Axis::Y, [src, 0, 0], 1.0)
            .edge(Axis::Y, [src, 0, 1], 1.0),
    );
    let h = e.add_operator(Probe::voltage(Axis::Y, [probe, 0, 0]));
    e.run(steps);
    e.operator(&h).values().to_vec()
}

fn steps() -> u64 {
    let dt = 0.99 * D / (C0 * 3f64.sqrt());
    (8.5e-10 / dt) as u64
}

/// Peak |run − reference| relative to the reference peak, in dB.
fn db(reference: &[f64], run: &[f64]) -> f64 {
    let peak = reference.iter().fold(0.0f64, |x, v| x.max(v.abs()));
    let diff = reference
        .iter()
        .zip(run)
        .fold(0.0f64, |x, (p, q)| x.max((p - q).abs()));
    20.0 * (diff / peak).log10()
}

/// Line of 120 cells, source at 30, probe at 60; the reference extends the
/// line by 300 cells on the +x side, whose end reflection arrives after
/// the window.
fn reflection(end: End) -> f64 {
    let n = steps();
    let reference = line(420, 30, 60, End::None, n);
    db(&reference, &line(120, 30, 60, end, n))
}

#[test]
fn sheet_absorbs_like_mur_face() {
    let face = reflection(End::MurFace);
    // Interior sheet with 20 cells of line (and a PEC face) behind it.
    let sheet = reflection(End::Sheet {
        plane: 100,
        positive: true,
    });
    // Sheet on the last interior plane, one cell before the PEC face.
    let last = reflection(End::Sheet {
        plane: 119,
        positive: true,
    });
    eprintln!("reflection: Mur face {face:.1} dB, sheet {sheet:.1} dB, last plane {last:.1} dB");
    assert!(face < -40.0, "Mur face {face:.1} dB");
    // Same scheme and cell size: the sheet matches the face Mur level.
    assert!(
        sheet < face + 1.0,
        "sheet {sheet:.1} dB vs face {face:.1} dB"
    );
    assert!(
        last < face + 1.0,
        "last plane {last:.1} dB vs face {face:.1} dB"
    );
}

#[test]
fn wrong_outward_normal_reflects() {
    // Outward normal −x: the incoming +x pulse arrives from behind the
    // sheet, which does not absorb it.
    let r = reflection(End::Sheet {
        plane: 100,
        positive: false,
    });
    eprintln!("wrong sign: {r:.1} dB");
    assert!(r > -3.0, "{r:.1} dB");
}

#[test]
fn transmitted_wave_continues_behind_the_sheet() {
    // A probe behind an interior sheet sees the incident pulse pass
    // (approximately unchanged at normal incidence) before the PEC face
    // reflection comes back.
    let n = steps();
    let reference = line(420, 30, 110, End::None, n);
    let run = line(
        120,
        30,
        110,
        End::Sheet {
            plane: 100,
            positive: true,
        },
        n,
    );
    // Window: until the PEC reflection at x = 120 returns to the probe
    // (pulse peak at 4·TAU, 80 + 2·10 cells of travel, minus 4·TAU of
    // pulse front).
    let dt = 0.99 * D / (C0 * 3f64.sqrt());
    let w = ((100.0 * D / C0) / dt) as usize;
    let r = db(&reference[..w], &run[..w]);
    eprintln!("transmitted pulse error behind the sheet: {r:.1} dB");
    assert!(r < -20.0, "{r:.1} dB");
}

#[test]
fn sheet_is_deterministic_across_threads() {
    let run = |threads: usize| {
        let grid = Grid::uniform([24, 8, 8], [D; 3]).unwrap();
        let mut e = Engine::builder(grid).threads(threads).build().unwrap();
        let sheet = AbcSheet {
            normal: Axis::X,
            plane: 18,
            positive: true,
            start: [0, 2, 1],
            stop: [0, 7, 6],
        };
        e.add_operator(AbsorbingSheet::new(&e, &sheet, SheetVelocity::Fixed(C0)).unwrap());
        e.add_operator(EdgeSource::new(gaussian).edge(Axis::Z, [8, 4, 4], 1.0));
        e.run(200);
        e.u_component(Axis::Z)
    };
    assert_eq!(run(1), run(3));
}

/// csx model of the same TEM line (1 mm cells, 1 cell across) with an
/// AbsorbingBC sheet at x = `plane` mm, run through [`ems::Simulation`].
mod csx_model {
    use csx::primitives::BoxShape;
    use csx::properties::{AbcType, AbsorbingBC, Excitation, ProbeBox, ProbeType};
    use csx::{ContinuousStructure, Primitive, Property};
    use ems::{Boundaries, Boundary, Face, Fdtd, Mur, Simulation, Waveform};

    pub fn model(cells: usize, abc: Option<(f64, AbsorbingBC)>) -> ContinuousStructure {
        let mut m = ContinuousStructure::new();
        m.grid.delta_unit = 1e-3;
        m.grid.set_lines(0, (0..=cells).map(|i| i as f64)).unwrap();
        m.grid.set_lines(1, [0.0, 1.0]).unwrap();
        m.grid.set_lines(2, [0.0, 1.0]).unwrap();
        let ex = m.add_property(Property::excitation(
            "src",
            Excitation::new([0.0, 1.0, 0.0]),
        ));
        m.add_primitive(
            ex,
            Primitive::new(BoxShape::new([30.0, 0.0, 0.0], [30.0, 1.0, 1.0])),
        )
        .unwrap();
        let pr = m.add_property(Property::probe("v", ProbeBox::new(ProbeType::Voltage)));
        m.add_primitive(
            pr,
            Primitive::new(BoxShape::new([60.0, 0.0, 0.0], [60.0, 1.0, 0.0])),
        )
        .unwrap();
        if let Some((x, a)) = abc {
            let p = m.add_property(Property::absorbing_bc("abc", a));
            m.add_primitive(
                p,
                Primitive::new(BoxShape::new([x, 0.0, 0.0], [x, 1.0, 1.0])),
            )
            .unwrap();
        }
        m
    }

    pub fn fdtd(xmax: Boundary) -> Fdtd {
        Fdtd::new()
            .waveform(Waveform::gaussian(5e9, 5e9))
            .boundaries(Boundaries::all(Boundary::Pec))
            .boundary(Face::XMin, mur())
            .boundary(Face::XMax, xmax)
            .boundary(Face::ZMin, Boundary::Pmc)
            .boundary(Face::ZMax, Boundary::Pmc)
            // ~1.3 ns: the x_max reflection reaches the probe, the end of
            // the 420-cell reference line does not.
            .max_timesteps(700)
    }

    pub fn mur() -> Boundary {
        Boundary::Mur(Mur::new())
    }

    pub fn abc(positive: bool, abc_type: AbcType) -> AbsorbingBC {
        AbsorbingBC {
            normal_sign_positive: positive,
            abc_type,
            ..Default::default()
        }
    }

    pub fn probe(mut sim: Simulation) -> Vec<f64> {
        let res = sim.run().unwrap();
        res.probe("v").unwrap().scalar().unwrap().to_vec()
    }
}

#[test]
fn csx_absorbing_bc_maps_to_sheet() {
    use csx::properties::AbcType;
    use csx_model::*;
    use ems::{Boundary, Simulation};

    let m = model(120, Some((100.0, abc(true, AbcType::Mur1st))));
    let sim = Simulation::new(&m, &fdtd(Boundary::Pec)).unwrap();
    let h = &sim.boundaries().abc;
    assert_eq!(h.len(), 1);
    let s = *sim.engine().operator(&h[0]).sheet();
    assert_eq!(
        s,
        AbcSheet {
            normal: Axis::X,
            plane: 100,
            positive: true,
            start: [100, 0, 0],
            stop: [100, 1, 1],
        }
    );
    assert!(
        sim.warnings().iter().all(|w| !w.contains("AbsorbingBC")),
        "{:?}",
        sim.warnings()
    );
    let with_sheet = probe(sim);
    let face = probe(Simulation::new(&model(120, None), &fdtd(mur())).unwrap());
    let reference = probe(Simulation::new(&model(420, None), &fdtd(Boundary::Pec)).unwrap());
    let (r_sheet, r_face) = (db(&reference, &with_sheet), db(&reference, &face));
    eprintln!("csx model: sheet {r_sheet:.1} dB, Mur face {r_face:.1} dB");
    assert!(r_sheet < r_face + 1.0 && r_sheet < -40.0);
}

#[test]
fn csx_absorbing_bc_warnings() {
    use csx::primitives::BoxShape;
    use csx::properties::AbcType;
    use csx::{Primitive, Property};
    use csx_model::*;
    use ems::{Boundary, Simulation};

    let mut m = model(40, Some((20.0, abc(false, AbcType::Mur1stSa))));
    let p = m.add_property(Property::absorbing_bc("vol", abc(true, AbcType::Mur1st)));
    m.add_primitive(
        p,
        Primitive::new(BoxShape::new([5.0, 0.0, 0.0], [8.0, 1.0, 1.0])),
    )
    .unwrap();
    let sim = Simulation::new(&m, &fdtd(Boundary::Pec)).unwrap();
    let w = sim.warnings().join("\n");
    assert!(w.contains("MUR_1ST_SA not implemented"), "{w}");
    assert!(w.contains("\"vol\": primitive is a volume"), "{w}");
    let h = &sim.boundaries().abc;
    assert_eq!(h.len(), 1);
    assert!(!sim.engine().operator(&h[0]).sheet().positive);
}

#[cfg(feature = "xml")]
#[test]
fn xml_absorbing_bc_round_trip() {
    use csx::OpenEmsDocument;
    use csx::properties::AbcType;
    use csx_model::*;
    use ems::{Boundary, Simulation};

    let model = model(120, Some((100.0, abc(true, AbcType::Mur1st))));
    let fdtd = fdtd(Boundary::Pec);
    let direct = probe(Simulation::new(&model, &fdtd).unwrap());
    let mut doc = OpenEmsDocument {
        structure: model,
        ..Default::default()
    };
    doc.set_fdtd_settings(&fdtd.to_csx());
    let xml = doc.to_xml_string();
    assert!(xml.contains("<AbsorbingBC"), "{xml}");
    let sim = Simulation::from_xml_str(&xml).unwrap();
    assert_eq!(sim.boundaries().abc.len(), 1);
    assert_eq!(probe(sim), direct);
}
