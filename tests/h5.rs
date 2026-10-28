//! HDF5 output (feature `hdf5`): dumps and probes of a small simulation are
//! written in the ems layout and read back with the `hdf5rs` reader; every
//! value must round-trip bit for bit.
#![cfg(feature = "hdf5")]

use csx::primitives::BoxShape;
use csx::properties::{DumpBox, DumpFileType, DumpType, Excitation, ProbeBox, ProbeType};
use csx::{ContinuousStructure, Primitive, Property};
use ems::io::h5::{self, H5Options};
use ems::{Axis, Fdtd, ProbeValues, Results, Simulation, Waveform};
use hdf5rs::{Complex, File};

fn add_box(m: &mut ContinuousStructure, p: Property, lo: [f64; 3], hi: [f64; 3]) {
    let id = m.add_property(p);
    m.add_primitive(id, Primitive::new(BoxShape::new(lo, hi)))
        .unwrap();
}

/// An 8 mm vacuum cube with a soft source, a time-domain H dump (with
/// HDF5 file type), a frequency-domain E dump and two probes.
fn run() -> Results {
    let mut m = ContinuousStructure::new();
    m.grid.delta_unit = 1e-3;
    for a in 0..3 {
        m.grid.set_lines(a, (0..=8).map(f64::from)).unwrap();
    }
    add_box(
        &mut m,
        Property::excitation("src", Excitation::new([0.0, 0.0, 1.0])),
        [4.0, 4.0, 3.0],
        [4.0, 4.0, 5.0],
    );
    let mut fd = DumpBox::new(DumpType::EFieldFd);
    fd.probe.frequencies = vec![2e9, 1.5e9];
    add_box(
        &mut m,
        Property::dump("fd", fd),
        [2.0, 2.0, 2.0],
        [6.0, 5.0, 4.0],
    );
    let td = DumpBox::new(DumpType::HField).with_file_type(DumpFileType::Hdf5);
    add_box(
        &mut m,
        Property::dump("td/h", td),
        [0.0, 1.0, 4.0],
        [8.0, 7.0, 4.0],
    );
    add_box(
        &mut m,
        Property::probe("v", ProbeBox::new(ProbeType::Voltage)),
        [4.0, 4.0, 3.0],
        [4.0, 4.0, 5.0],
    );
    add_box(
        &mut m,
        Property::probe(
            "e",
            ProbeBox::new(ProbeType::EField).with_frequencies([2e9]),
        ),
        [3.0, 5.0, 4.0],
        [3.0, 5.0, 4.0],
    );
    let fdtd = Fdtd::new()
        .waveform(Waveform::gaussian(2e9, 1e9))
        .max_timesteps(120)
        .threads(1);
    Simulation::new(&m, &fdtd).unwrap().run().unwrap()
}

fn attr_str(f: &File, path: &str, name: &str) -> String {
    f.object(path)
        .unwrap()
        .attribute(name)
        .unwrap()
        .read_scalar()
        .unwrap()
}

fn bits32(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn bits64(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn time_dump_round_trip() {
    let res = run();
    let d = res.dump("td/h").unwrap();
    assert_eq!(d.file_type(), DumpFileType::Hdf5);
    assert_eq!(res.dump("fd").unwrap().file_type(), DumpFileType::Vtk);
    let snaps = d.snapshots().unwrap();
    assert!(snaps.len() > 2);
    let s = d.sampler();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("h5_td.h5");
    d.write_h5(&path).unwrap();
    let f = File::open(&path).unwrap();

    assert_eq!(attr_str(&f, "/", "format"), "ems-dump");
    assert_eq!(attr_str(&f, "/", "quantity"), "H");
    assert_eq!(attr_str(&f, "/", "domain"), "time");
    assert_eq!(attr_str(&f, "/", "name"), "td/h");
    assert_eq!(attr_str(&f, "/", "unit"), "A/m");
    let root = f.object("/").unwrap();
    let version: u32 = root.attribute("version").unwrap().read_scalar().unwrap();
    assert_eq!(version, h5::LAYOUT_VERSION);
    let id: u32 = root.attribute("id").unwrap().read_scalar().unwrap();
    assert_eq!(id, d.id());

    for (axis, n) in Axis::ALL.into_iter().zip(["x", "y", "z"]) {
        let c: Vec<f64> = f.dataset(&format!("/mesh/{n}")).unwrap().read().unwrap();
        assert_eq!(bits64(&c), bits64(s.coords(axis)));
    }
    let steps: Vec<u64> = f.dataset("/step").unwrap().read().unwrap();
    let times: Vec<f64> = f.dataset("/time").unwrap().read().unwrap();
    assert_eq!(steps, snaps.iter().map(|x| x.step).collect::<Vec<_>>());
    assert_eq!(
        bits64(&times),
        bits64(&snaps.iter().map(|x| x.time).collect::<Vec<_>>())
    );

    let ds = f.dataset("/H").unwrap();
    let [nx, ny, nz] = s.dims().map(|n| n as u64);
    assert_eq!(ds.shape(), [snaps.len() as u64, nx, ny, nz, 3]);
    let all: Vec<f32> = ds.read().unwrap();
    let want: Vec<f32> = snaps
        .iter()
        .flat_map(|x| x.data.iter().flatten().copied())
        .collect();
    assert!(want.iter().any(|&x| x != 0.0));
    assert_eq!(bits32(&all), bits32(&want));
}

#[test]
fn freq_dump_round_trip() {
    let res = run();
    let d = res.dump("fd").unwrap();
    let fdump = d.freq().unwrap();
    let bytes = h5::dump_bytes(d, &H5Options::default()).unwrap();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("h5_fd.h5");
    h5::write_dump(&path, d, &H5Options::default()).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let f = File::from_bytes(bytes).unwrap();
    assert_eq!(attr_str(&f, "/", "domain"), "frequency");
    assert_eq!(attr_str(&f, "/", "quantity"), "E");
    let samples: u64 = f
        .object("/")
        .unwrap()
        .attribute("samples")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(samples, fdump.samples());
    let freqs: Vec<f64> = f.dataset("/frequency").unwrap().read().unwrap();
    assert_eq!(bits64(&freqs), bits64(fdump.freqs()));

    let ds = f.dataset("/E").unwrap();
    let [nx, ny, nz] = d.sampler().dims().map(|n| n as u64);
    assert_eq!(ds.shape(), [2, nx, ny, nz, 3]);
    let got: Vec<Complex<f64>> = ds.read().unwrap();
    let want: Vec<_> = (0..2).flat_map(|k| fdump.values(k)).flatten().collect();
    assert_eq!(got.len(), want.len());
    assert!(want.iter().any(|c| c.norm() > 0.0));
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(
            (g.re.to_bits(), g.im.to_bits()),
            (w.re.to_bits(), w.im.to_bits())
        );
    }
}

#[test]
fn probes_round_trip() {
    let res = run();
    let bytes = h5::probes_bytes(res.probes()).unwrap();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("h5_probes.h5");
    h5::write_probes(&path, res.probes()).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let f = File::from_bytes(bytes).unwrap();
    assert_eq!(attr_str(&f, "/", "format"), "ems-probes");
    for p in res.probes() {
        let g = format!("/{}", p.name());
        assert_eq!(attr_str(&f, &g, "name"), p.name());
        let t: Vec<f64> = f.dataset(&format!("{g}/time")).unwrap().read().unwrap();
        assert_eq!(bits64(&t), bits64(p.times()));
        let ds = f.dataset(&format!("{g}/values")).unwrap();
        let v: Vec<f64> = ds.read().unwrap();
        match p.values() {
            ProbeValues::Scalar(want) => {
                assert_eq!(attr_str(&f, &g, "kind"), "voltage");
                assert_eq!(ds.shape(), [want.len() as u64]);
                assert_eq!(bits64(&v), bits64(want));
            }
            ProbeValues::Vector(want) => {
                assert_eq!(attr_str(&f, &g, "kind"), "efield");
                assert_eq!(ds.shape(), [want.len() as u64, 3]);
                assert_eq!(bits64(&v), bits64(want.as_flattened()));
                let fr: Vec<f64> = f
                    .dataset(&format!("{g}/frequencies"))
                    .unwrap()
                    .read()
                    .unwrap();
                assert_eq!(fr, [2e9]);
            }
            other => panic!("unexpected probe values {other:?}"),
        }
    }
}

#[cfg(feature = "hdf5-deflate")]
#[test]
fn deflate_round_trip() {
    let res = run();
    let d = res.dump("td/h").unwrap();
    let plain = h5::dump_bytes(d, &H5Options::default()).unwrap();
    let packed = h5::dump_bytes(d, &H5Options::default().deflate(6)).unwrap();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("h5_td_deflate.h5");
    std::fs::write(path, &packed).unwrap();
    assert!(
        packed.len() < plain.len(),
        "{} vs {}",
        packed.len(),
        plain.len()
    );
    let a: Vec<f32> = File::from_bytes(plain)
        .unwrap()
        .dataset("/H")
        .unwrap()
        .read()
        .unwrap();
    let b: Vec<f32> = File::from_bytes(packed)
        .unwrap()
        .dataset("/H")
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(bits32(&a), bits32(&b));
}
