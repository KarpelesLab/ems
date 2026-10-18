//! The README quick-start snippet: it is kept verbatim between the
//! `README-BEGIN` / `README-END` markers below (checked against README.md),
//! and it must run without warnings.

use csx::primitives::BoxShape;
use csx::{ContinuousStructure, Primitive, Property};
use ems::prelude::*;

fn quick_start() -> Result<(Vec<String>, Vec<Complex64>)> {
    let mut warnings = Vec::new();
    // README-BEGIN
    let mut model = ContinuousStructure::new();
    model.grid.delta_unit = 1e-3; // drawing unit: mm
    model.grid.set_lines(0, (0..=40).map(f64::from))?;
    model.grid.set_lines(1, (0..=40).map(f64::from))?;
    model.grid.set_lines(2, (0..=20).map(f64::from))?;
    // 2 mm FR-4 on a ground plane (the PEC z_min face), a 12 mm square patch
    // on top; everything stays clear of the 8-cell PML on the other faces.
    let sub = model.add_property(Property::material("sub", csx::Material::new(4.3, 0.0)));
    model.add_primitive(
        sub,
        Primitive::new(BoxShape::new([10., 10., 0.], [30., 30., 2.])),
    )?;
    let patch = model.add_property(Property::metal("patch"));
    model.add_primitive(
        patch,
        Primitive::new(BoxShape::new([14., 14., 2.], [26., 26., 2.])),
    )?;
    let port = LumpedPort::new(1, 50.0, [17., 20., 0.], [17., 20., 2.], Axis::Z)
        .excite(1.0)
        .add_to(&mut model)?; // writes LumpedElement + Excitation + probes into the model

    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(2e9, 1e9))
        .boundaries(Boundaries::all(Boundary::pml(8)).with(Face::ZMin, Boundary::Pec))
        .end_criteria(EndCriteria::energy_decay_db(-40.0));
    let mut sim = Simulation::new(&model, &fdtd)?;
    for w in sim.warnings() {
        eprintln!("warning: {w}"); // includes the solver-aware mesh checks
    }
    let res = sim.run()?;
    let z_in = port.analyze(&res, &[1e9, 2e9, 3e9])?.z_in();
    // README-END
    warnings.extend(sim.warnings().iter().cloned());
    Ok((warnings, z_in))
}

/// The README's first ```rust block, with the indentation of this file.
fn readme_block() -> Vec<String> {
    // Normalise line endings (Windows checkouts may use CRLF).
    let readme = include_str!("../README.md").replace("\r\n", "\n");
    let start = readme.find("```rust\n").expect("README has a rust block") + 8;
    let end = start + readme[start..].find("```").expect("closed block");
    readme[start..end]
        .lines()
        .filter(|l| !l.starts_with("use "))
        .map(|l| l.trim_end().to_string())
        .collect::<Vec<_>>()
}

fn test_block() -> Vec<String> {
    let src = include_str!("readme.rs").replace("\r\n", "\n");
    let a = src.find("// README-BEGIN\n").unwrap() + "// README-BEGIN\n".len();
    let b = src.find("    // README-END").unwrap();
    src[a..b]
        .lines()
        .map(|l| l.strip_prefix("    ").unwrap_or(l).trim_end().to_string())
        .collect()
}

#[test]
fn readme_snippet_matches_this_test() {
    let trim = |v: Vec<String>| -> Vec<String> {
        let v: Vec<String> = v.into_iter().skip_while(|l| l.is_empty()).collect();
        let mut v = v;
        while v.last().is_some_and(|l| l.is_empty()) {
            v.pop();
        }
        v
    };
    assert_eq!(trim(readme_block()), trim(test_block()));
}

#[test]
fn readme_snippet_runs_without_warnings() {
    let (warnings, z_in) = quick_start().unwrap();
    assert!(warnings.is_empty(), "{warnings:#?}");
    assert_eq!(z_in.len(), 3);
    assert!(z_in.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
}
