//! SAR dumps on a cylindrical grid (design §5.12, §5.15): a lossy tissue
//! ring in a closed circular PEC cavity with an axis, fed by a one-cell
//! lumped port inside the tissue. Nothing radiates, so the SAR volume's
//! total power (cell volumes `Δα·Δz·(ρ_{i+1}² − ρ_i²)/2`, edge weights of
//! the cylindrical edge conductances) must equal the port's accepted power,
//! and the tissue mass is that of the annulus. Mass-averaged dumps use
//! cubes in physical space over the annular cells (`post::sar`).

use std::f64::consts::TAU;

use csx::primitives::BoxShape;
use csx::properties::{DumpBox, DumpType};
use csx::{ContinuousStructure, CoordSystem, Primitive, Property, Scalar};
use ems::compile::sar::SarKind;
use ems::post::sar::VoxelClass;
use ems::prelude::*;

const FREQS: [f64; 3] = [0.5e9, 1e9, 1.5e9];
const NA: usize = 16;

fn add_box(m: &mut ContinuousStructure, p: Property, lo: [f64; 3], hi: [f64; 3]) {
    let id = m.add_property(p);
    m.add_primitive(id, Primitive::new(BoxShape::new(lo, hi)))
        .unwrap();
}

/// Cavity ρ ≤ 20 mm, 0 ≤ z ≤ 20 mm (1 mm cells, 16 α cells); tissue
/// (ε_r 40, σ 1.5 S/m, 1000 kg/m³) for 4 ≤ ρ ≤ 16 mm, 4 ≤ z ≤ 16 mm over
/// the full circle; port along z at ρ = 8 mm, α = α_3, z 10..11.
fn model() -> (ContinuousStructure, LumpedPort) {
    let mut m = ContinuousStructure::new();
    m.coord_system = CoordSystem::Cylindrical;
    m.grid.coord_system = CoordSystem::Cylindrical;
    m.grid.delta_unit = 1e-3;
    m.grid.set_lines(0, (0..=20).map(f64::from)).unwrap();
    m.grid
        .set_lines(1, (0..=NA).map(|j| j as f64 * TAU / NA as f64))
        .unwrap();
    m.grid.set_lines(2, (0..=20).map(f64::from)).unwrap();
    let mut tissue = csx::Material::new(40.0, 1.5);
    tissue.density = Scalar::new(1000.0);
    add_box(
        &mut m,
        Property::material("tissue", tissue),
        [4.0, 0.0, 4.0],
        [16.0, TAU, 16.0],
    );
    let a3 = 3.0 * TAU / NA as f64;
    let port = LumpedPort::new(1, 50.0, [8.0, a3, 10.0], [8.0, a3, 11.0], Axis::Z)
        .excite(1.0)
        .add_to(&mut m)
        .unwrap();
    for (name, t) in [
        ("local", DumpType::SarLocal),
        ("avg1g", DumpType::Sar1g),
        ("raw", DumpType::SarRaw),
    ] {
        let mut d = DumpBox::new(t);
        d.probe.frequencies = FREQS.to_vec();
        add_box(
            &mut m,
            Property::dump(name, d),
            [0.0, 0.0, 0.0],
            [20.0, TAU, 20.0],
        );
    }
    (m, port)
}

fn run() -> &'static (Results, LumpedPort, Vec<String>) {
    static RUN: std::sync::OnceLock<(Results, LumpedPort, Vec<String>)> =
        std::sync::OnceLock::new();
    RUN.get_or_init(|| {
        let (m, port) = model();
        let fdtd = Fdtd::new()
            .waveform(Waveform::gaussian(1e9, 0.8e9))
            .end_criteria(EndCriteria::energy_decay_db(-70.0));
        let mut sim = Simulation::new(&m, &fdtd).unwrap();
        let w = sim.warnings().to_vec();
        (sim.run().unwrap(), port, w)
    })
}

#[test]
fn cylindrical_absorbed_power_equals_accepted_power() {
    let (res, port, w) = run();
    // No warnings but mesh remarks: the averaged dump is kept.
    assert!(w.iter().all(|w| w.starts_with("mesh ")), "{w:?}");
    let avg = res.sar("avg1g").unwrap();
    assert_eq!(avg.kind(), SarKind::Averaged(1e-3));
    let acc = port.analyze(res, &FREQS).unwrap().accepted_power();
    let local = res.sar("local").unwrap();
    let raw = res.sar("raw").unwrap();
    assert_eq!(local.kind(), SarKind::Local);
    let mass = 1000.0 * 0.5 * (16e-3f64.powi(2) - 4e-3f64.powi(2)) * TAU * 12e-3;
    for (k, &f) in FREQS.iter().enumerate() {
        let v = local.volume(k);
        assert!(v.is_cylindrical());
        assert_eq!(v, raw.volume(k));
        let absorbed = v.total_power();
        let rel = absorbed / acc[k] - 1.0;
        println!(
            "{f:e} Hz: accepted {:.6e}, absorbed {absorbed:.6e}, rel {rel:+.2e}",
            acc[k]
        );
        assert!(rel.abs() < 2e-4, "power balance at {f} Hz: {rel:e}");
        assert_eq!(v.tissue_power(), absorbed);
        assert!(
            (v.tissue_mass() / mass - 1.0).abs() < 1e-12,
            "{} vs {mass}",
            v.tissue_mass()
        );
        // Local SAR in the tissue only, peak next to the port gap.
        let l = local.field(k).unwrap();
        let c = l.cells();
        assert_eq!(c, [20, NA, 20]);
        for i in 0..c[0] {
            for kk in 0..c[2] {
                let idx = (i * c[1] + 5) * c[2] + kk;
                let inside = (4..16).contains(&i) && (4..16).contains(&kk);
                assert_eq!(l.classes()[idx] == VoxelClass::Local, inside);
            }
        }
        let m = l.max().unwrap();
        assert!([7, 8].contains(&m.cell[0]) && [2, 3].contains(&m.cell[1]));
        assert_eq!(m.cell[2], 10);
        // 1 g averages over physical cubes: every tissue voxel gets one,
        // each centred or face cube holds 1 g, and the peak lies between
        // the whole-tissue average and the local peak, near the port.
        let a = avg.field(k).unwrap();
        assert_eq!(a, v.averaged(1e-3).unwrap());
        for (i, (&c, &cm)) in a.classes().iter().zip(a.cube_masses()).enumerate() {
            assert_eq!(c == VoxelClass::NoTissue, v.density()[i] == 0.0);
            assert_ne!(c, VoxelClass::Unresolved);
            if matches!(c, VoxelClass::Centered | VoxelClass::Boundary) {
                assert!((cm / 1e-3 - 1.0).abs() < 1e-9, "{cm}");
            }
        }
        let am = a.max().unwrap();
        println!(
            "  1 g peak {:.4e} at {:?} (local {:.4e}, whole {:.4e})",
            am.value,
            am.cell,
            m.value,
            v.whole_average()
        );
        assert!(am.value < m.value && am.value > v.whole_average());
        assert!((6..=10).contains(&am.cell[0]) && (8..=12).contains(&am.cell[2]));
    }
}
