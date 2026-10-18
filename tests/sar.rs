//! SAR dumps through `Simulation`: a lossy tissue block in a closed PEC box,
//! fed by a one-cell lumped port inside the block. With no radiation, all
//! the power accepted by the port is absorbed in the block, so the SAR
//! volume's total power must equal the port's accepted power at every
//! frequency.
//!
//! The port is one edge long on purpose: the terminal current of a
//! multi-edge element is the mean over its series edges, so `V·I` differs
//! from the element's actual power by the non-uniformity of the edge
//! voltages (about 1 % for a 4-cell gap here).

use csx::primitives::BoxShape;
use csx::properties::{DumpBox, DumpType};
use csx::{ContinuousStructure, Primitive, Property, Scalar};
use ems::compile::sar::SarKind;
use ems::post::sar::VoxelClass;
use ems::prelude::*;

const FREQS: [f64; 3] = [0.5e9, 1e9, 1.5e9];

fn add_box(m: &mut ContinuousStructure, p: Property, lo: [f64; 3], hi: [f64; 3]) {
    let id = m.add_property(p);
    m.add_primitive(id, Primitive::new(BoxShape::new(lo, hi)))
        .unwrap();
}

/// 24 mm PEC box, 1 mm cells; tissue (ε_r 40, σ 1.5 S/m, 1000 kg/m³) in
/// cells 4..20 on every axis; port gap along z at x = 6, y = 12, z 12..13.
fn model() -> (ContinuousStructure, LumpedPort) {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for a in 0..3 {
        m.grid.set_lines(a, (0..=24).map(f64::from)).unwrap();
    }
    let mut tissue = csx::Material::new(40.0, 1.5);
    tissue.density = Scalar::new(1000.0);
    add_box(
        &mut m,
        Property::material("tissue", tissue),
        [4.0; 3],
        [20.0; 3],
    );
    let port = LumpedPort::new(1, 50.0, [6.0, 12.0, 12.0], [6.0, 12.0, 13.0], Axis::Z)
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
        add_box(&mut m, Property::dump(name, d), [0.0; 3], [24.0; 3]);
    }
    (m, port)
}

/// The simulation, run once for all tests.
fn run() -> &'static (Results, LumpedPort) {
    static RUN: std::sync::OnceLock<(Results, LumpedPort)> = std::sync::OnceLock::new();
    RUN.get_or_init(|| {
        let (m, port) = model();
        let fdtd = Fdtd::new()
            .waveform(Waveform::gaussian(1e9, 0.8e9))
            .end_criteria(EndCriteria::energy_decay_db(-70.0));
        let mut sim = Simulation::new(&m, &fdtd).unwrap();
        assert!(sim.warnings().is_empty(), "{:?}", sim.warnings());
        (sim.run().unwrap(), port)
    })
}

#[test]
fn absorbed_power_equals_accepted_power() {
    let (res, port) = run();
    let acc = port.analyze(res, &FREQS).unwrap().accepted_power();
    let local = res.sar("local").unwrap();
    assert_eq!(local.kind(), SarKind::Local);
    assert_eq!(local.frequencies(), FREQS);
    for (k, &f) in FREQS.iter().enumerate() {
        let v = local.volume(k);
        let absorbed = v.total_power();
        let rel = absorbed / acc[k] - 1.0;
        println!(
            "{f:e} Hz: accepted {:.6e}, absorbed {absorbed:.6e}, rel {rel:+.2e}",
            acc[k]
        );
        assert!(rel.abs() < 1e-4, "power balance at {f} Hz: {rel:e}");
        // Only the tissue is lossy.
        assert_eq!(v.tissue_power(), absorbed);
        assert!((v.tissue_mass() - 1000.0 * 16e-3f64.powi(3)).abs() < 1e-15);
        // Normalised to 1 W accepted.
        let norm = v.scaled(1.0 / acc[k]);
        assert!((norm.total_power() - 1.0).abs() < 1e-4);
        assert!((norm.whole_average() * norm.tissue_mass() - 1.0).abs() < 1e-4);
    }
}

#[test]
fn local_averaged_and_raw_dumps() {
    let (res, _) = run();
    assert_eq!(res.sar_dumps().len(), 3);
    let local = res.sar("local").unwrap();
    let avg = res.sar("avg1g").unwrap();
    let raw = res.sar("raw").unwrap();
    assert_eq!(avg.kind(), SarKind::Averaged(1e-3));
    assert_eq!(raw.kind(), SarKind::Raw);
    for k in 0..FREQS.len() {
        // All three see the same field.
        assert_eq!(local.volume(k), raw.volume(k));
        assert_eq!(local.volume(k), avg.volume(k));
        let l = local.field(k).unwrap();
        let a = avg.field(k).unwrap();
        assert_eq!(raw.field(k).unwrap(), l);
        assert_eq!(a.mass(), Some(1e-3));
        // Tissue cells only.
        let c = l.cells();
        assert_eq!(c, [24; 3]);
        for i in 0..c[0] {
            for j in 0..c[1] {
                for kk in 0..c[2] {
                    let idx = (i * c[1] + j) * c[2] + kk;
                    let inside = [i, j, kk].iter().all(|x| (4..20).contains(x));
                    assert_eq!(l.classes()[idx] == VoxelClass::Local, inside);
                    assert_eq!(a.classes()[idx] != VoxelClass::NoTissue, inside);
                    assert!(a.classes()[idx] != VoxelClass::Unresolved);
                }
            }
        }
        // An average never exceeds the local peak; the hottest voxel touches
        // the port gap (x = 6, y = 12, z = 12..13).
        let (lm, am) = (l.max().unwrap(), a.max().unwrap());
        println!("{k}: local max {lm:?}, 1 g max {am:?}");
        assert!(am.value <= lm.value * (1.0 + 1e-12));
        assert!(am.value > 0.0 && am.value < 0.5 * lm.value);
        assert!([5, 6].contains(&lm.cell[0]) && [11, 12].contains(&lm.cell[1]));
        assert_eq!(lm.cell[2], 12);
        // The peak 1 g cube is centred 2.5 mm behind the gap; it reaches
        // 0.5 mm out of the block (5 % air, valid).
        assert_eq!(am.cell, [8, 11, 12]);
        assert_eq!(
            a.classes()[(8 * c[1] + 11) * c[2] + 12],
            VoxelClass::Centered
        );
        // The block is 16 mm, a 1 g cube 10 mm: the block centre has a
        // valid centred cube, the corner voxels do not.
        assert_eq!(
            a.classes()[(12 * c[1] + 12) * c[2] + 12],
            VoxelClass::Centered
        );
        assert_eq!(a.classes()[(4 * c[1] + 4) * c[2] + 4], VoxelClass::Covered);
    }
}

#[test]
fn sar_field_files() {
    let (res, _) = run();
    let a = res.sar("avg1g").unwrap().field(1).unwrap();
    let dir = std::env::temp_dir().join(format!("ems-sar-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let vtr = dir.join("sar.vtr");
    a.write_vtr(&vtr).unwrap();
    let text = std::fs::read_to_string(&vtr).unwrap();
    assert!(text.contains("Name=\"SAR\"") && text.contains("Name=\"class\""));
    #[cfg(feature = "hdf5")]
    {
        let f = hdf5rs::File::from_bytes(a.h5_bytes().unwrap()).unwrap();
        let sar = f.dataset("/SAR").unwrap().read_nd::<f64>().unwrap();
        assert_eq!(sar.shape, vec![24, 24, 24]);
        assert_eq!(sar.data, a.values());
        let class: Vec<u8> = f.dataset("/class").unwrap().read().unwrap();
        assert!(class.iter().zip(a.classes()).all(|(&c, &k)| c == k as u8));
        let x: Vec<f64> = f.dataset("/mesh/x").unwrap().read().unwrap();
        assert_eq!(x, a.lines(Axis::X));
        let mass: f64 = f
            .object("/")
            .unwrap()
            .attribute("mass")
            .unwrap()
            .read_scalar()
            .unwrap();
        assert_eq!(mass, 1e-3);
        let path = dir.join("sar.h5");
        a.write_h5(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), a.h5_bytes().unwrap());
    }
    std::fs::remove_dir_all(&dir).ok();
}
