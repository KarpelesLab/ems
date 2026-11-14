//! Typed warnings ([`ems::Simulation::diagnostics`]) of a real model: the
//! lines the compile step and the mesh check write get their codes and
//! subjects, and the lines of one kind about one property are one entry.

use csx::primitives::BoxShape;
use csx::properties::{ProbeBox, ProbeType};
use csx::{ContinuousStructure, Primitive, Property};
use ems::{Boundaries, Boundary, Cpml, Fdtd, Simulation, WarningCode, Waveform};

#[test]
fn diagnostics_of_a_board_region() {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for a in 0..3 {
        m.grid
            .set_lines(a, (0..=20).map(|i| i as f64 * 0.5))
            .unwrap();
    }
    // Five copper strips through the absorbing layers, one inside.
    let copper = m.add_property(Property::metal("copper"));
    for k in 0..5 {
        let y = 2.0 + k as f64;
        let strip = BoxShape::new([0.0, y, 5.0], [10.0, y + 0.5, 5.0]);
        m.add_primitive(copper, Primitive::new(strip)).unwrap();
    }
    let pad = BoxShape::new([4.0, 7.0, 5.0], [6.0, 7.5, 5.0]);
    m.add_primitive(copper, Primitive::new(pad)).unwrap();
    let sheet = m.add_property(Property::conducting_sheet("lossy", 5.8e7, 35e-6));
    let plane = BoxShape::new([0.0, 0.0, 4.0], [10.0, 10.0, 4.0]);
    m.add_primitive(sheet, Primitive::new(plane)).unwrap();
    // A probe between mesh lines.
    let probe = m.add_property(Property::probe("ut", ProbeBox::new(ProbeType::Voltage)));
    let line = BoxShape::new([5.2, 5.0, 4.0], [5.2, 5.0, 5.0]);
    m.add_primitive(probe, Primitive::new(line)).unwrap();

    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(5e9, 5e9))
        .boundaries(Boundaries::all(Boundary::Pml(Cpml::cells(4))))
        .max_timesteps(10)
        .threads(1);
    let sim = Simulation::new(&m, &fdtd).unwrap();
    let lines = sim.warnings();
    let typed = sim.diagnostics();
    let find = |code: WarningCode, subject: Option<&str>| {
        typed
            .iter()
            .find(|w| w.code() == code && w.subject() == subject)
            .unwrap_or_else(|| panic!("no {code} about {subject:?} in {lines:#?}"))
    };

    // One entry for the five strips, where the text has five lines.
    let in_pml = find(WarningCode::MeshInPml, Some("copper"));
    assert_eq!(in_pml.count(), 5);
    assert_eq!(in_pml.primitive(), None);
    assert_eq!(
        in_pml.message(),
        lines.iter().find(|l| l.contains("\"copper\"")).unwrap()
    );
    let sheet_in_pml = find(WarningCode::MeshInPml, Some("lossy"));
    assert_eq!(
        (sheet_in_pml.count(), sheet_in_pml.primitive()),
        (1, Some(0))
    );
    assert_eq!(find(WarningCode::SheetInPml, None).count(), 1);
    assert_eq!(find(WarningCode::BoxSnapped, Some("ut")).count(), 1);

    // Nothing of this model is unknown to the table, every line is counted
    // once, and a single line is its own text.
    assert!(
        typed.iter().all(|w| w.code() != WarningCode::Other),
        "{lines:#?}"
    );
    assert_eq!(typed.iter().map(|w| w.count()).sum::<usize>(), lines.len());
    for w in typed.iter().filter(|w| w.count() == 1) {
        assert!(lines.contains(&w.to_string()));
    }
}
