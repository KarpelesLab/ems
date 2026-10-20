//! openEMS-compatible dump files: ems runs the fixture models of
//! `tests/fixtures/openems/dump_*` (written by a black-box openEMS, see
//! `scripts/openems-oracle/gen-dumps.sh`) and writes its dumps with
//! `ems::io::openems::dump`; every file openEMS wrote must exist with the
//! same structure (groups, datasets, shapes, datatypes, attributes; VTK
//! element structure) and the same values within a float32 tolerance.
//!
//! ems applies openEMS's excitation timing to models read from files
//! (design §5.16.1), so the models run as written.
#![cfg(all(feature = "xml", feature = "hdf5"))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ems::Simulation;
use ems::io::openems::dump::{DumpFileOptions, write_dump, write_sar};
use hdf5rs::{File, Value};

fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/openems")
        .join(case)
}

/// Runs `case` with ems and writes all dumps into a fresh directory.
fn run_ems(case: &str, variant: &str, opts: &DumpFileOptions) -> PathBuf {
    let out = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("openems_dumps")
        .join(case)
        .join(variant);
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    let mut sim = Simulation::from_xml(fixture(case).join("model.xml")).unwrap();
    let res = sim.run().unwrap();
    for d in res.dumps() {
        write_dump(&out, d, opts).unwrap();
    }
    for s in res.sar_dumps() {
        write_sar(&out, s).unwrap();
    }
    out
}

// ---------------------------------------------------------------------
// HDF5 comparison.

/// A flattened HDF5 file: path → (kind/shape/datatype signature,
/// attributes, numeric values).
/// Attribute name → (type signature, values).
type Attrs = BTreeMap<String, (String, Vec<Value>)>;

#[derive(Debug, Default)]
struct H5Tree {
    objects: BTreeMap<String, (String, Attrs, Vec<f64>)>,
}

fn dtype_sig(dt: &hdf5rs::Datatype) -> String {
    format!("{:?}/{}", dt.class, dt.size)
}

fn numbers(v: &[Value]) -> Vec<f64> {
    let mut out = Vec::new();
    for x in v {
        match x {
            Value::Float(f) => out.push(*f),
            Value::Int(i) => out.push(*i as f64),
            Value::UInt(u) => out.push(*u as f64),
            Value::Complex { re, im } => out.extend([*re, *im]),
            Value::Compound(m) => {
                out.extend(numbers(&m.iter().map(|p| p.1.clone()).collect::<Vec<_>>()))
            }
            Value::Array(a) | Value::VarLen(a) => out.extend(numbers(a)),
            _ => {}
        }
    }
    out
}

fn read_tree(path: &Path) -> H5Tree {
    let f = File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut t = H5Tree::default();
    fn attrs(o: &[hdf5rs::Attribute]) -> Attrs {
        o.iter()
            .map(|a| {
                let sig = format!("{}{:?}", dtype_sig(a.datatype()), a.shape());
                (a.name().to_string(), (sig, a.values().unwrap()))
            })
            .collect()
    }
    fn walk(t: &mut H5Tree, g: &hdf5rs::Group, path: &str) {
        t.objects.insert(
            path.to_string(),
            ("group".into(), attrs(&g.attributes().unwrap()), Vec::new()),
        );
        for name in g.link_names().unwrap() {
            let p = format!("{}/{name}", path.trim_end_matches('/'));
            let o = g.object(&name).unwrap();
            if let Some(sub) = o.as_group() {
                walk(t, sub, &p);
            } else if let Some(d) = o.as_dataset() {
                let sig = format!("dataset {:?} {}", d.shape(), dtype_sig(d.datatype()));
                let vals = numbers(&d.read_values().unwrap());
                t.objects
                    .insert(p, (sig, attrs(&d.attributes().unwrap()), vals));
            }
        }
    }
    walk(&mut t, &f.root().unwrap(), "/");
    t
}

/// Compares two HDF5 files (structure asserted); returns the largest data
/// difference relative to the largest |value| of the reference file, or
/// the largest relative difference of a numeric attribute if larger. Attributes in `loose`
/// are compared by type only.
fn compare_h5(want: &Path, got: &Path, loose: &[&str]) -> f64 {
    let (a, b) = (read_tree(want), read_tree(got));
    let name = want.file_name().unwrap().to_string_lossy();
    let ka: Vec<_> = a.objects.keys().collect();
    let kb: Vec<_> = b.objects.keys().collect();
    assert_eq!(ka, kb, "{name}: objects");
    let (mut num, mut den, mut attr_err) = (0.0f64, 0.0f64, 0.0f64);
    for (p, (sa, aa, va)) in &a.objects {
        let (sb, ab, vb) = &b.objects[p];
        assert_eq!(sa, sb, "{name}:{p}: kind/shape/type");
        let an: Vec<_> = aa.keys().collect();
        let bn: Vec<_> = ab.keys().collect();
        assert_eq!(an, bn, "{name}:{p}: attribute names");
        for (k, (ta, xa)) in aa {
            let (tb, xb) = &ab[k];
            assert_eq!(ta, tb, "{name}:{p}@{k}: attribute type");
            if loose.contains(&k.as_str()) {
                continue;
            }
            let (na, nb) = (numbers(xa), numbers(xb));
            if na.is_empty() {
                assert_eq!(format!("{xa:?}"), format!("{xb:?}"), "{name}:{p}@{k}");
            } else {
                assert_eq!(na.len(), nb.len(), "{name}:{p}");
                for (x, y) in na.iter().zip(&nb) {
                    let m = x.abs().max(y.abs());
                    if m > 0.0 {
                        attr_err = attr_err.max((x - y).abs() / m);
                    }
                }
            }
        }
        assert_eq!(va.len(), vb.len(), "{name}:{p}: value count");
        for (x, y) in va.iter().zip(vb) {
            num = num.max((x - y).abs());
            den = den.max(x.abs());
        }
    }
    attr_err.max(if den > 0.0 { num / den } else { num })
}

// ---------------------------------------------------------------------
// VTK comparison.

/// The XML structure of a VTK file with the binary payloads, the array
/// pointer names and the range values masked.
fn vtk_skeleton(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| {
            let t = l.trim();
            !(t.is_empty()
                || (!t.starts_with('<')
                    && t.chars()
                        .all(|c| c.is_ascii_alphanumeric() || "+/=".contains(c))))
        })
        .filter(|l| l.trim().parse::<f64>().is_err())
        .map(|l| {
            let mut s = l.to_string();
            for key in ["RangeMin=\"", "RangeMax=\"", "Name=\"Array 0x"] {
                if let Some(i) = s.find(key) {
                    let j = s[i + key.len()..].find('"').unwrap() + i + key.len();
                    s.replace_range(i + key.len()..j, "*");
                }
            }
            s
        })
        .collect()
}

#[cfg(feature = "hdf5-deflate")]
fn b64(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            _ => 63,
        }
    };
    let mut out = Vec::new();
    for ch in s.as_bytes().chunks(4) {
        let n = ch.iter().filter(|&&c| c != b'=').count();
        let mut v = 0u32;
        for (i, &c) in ch.iter().enumerate() {
            if c != b'=' {
                v |= val(c) << (18 - 6 * i);
            }
        }
        out.extend(&v.to_be_bytes()[1..n.max(1)]);
    }
    out
}

/// Decodes the compressed binary arrays of a VTK file (f32 or f64 each).
#[cfg(feature = "hdf5-deflate")]
fn vtk_arrays(text: &str) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    let mut ty = "";
    for l in text.lines() {
        let t = l.trim();
        if t.starts_with("<DataArray") {
            ty = if t.contains("Float64") { "f64" } else { "f32" };
            continue;
        }
        if t.is_empty() || t.starts_with('<') || !t.ends_with('=') && t.len() < 8 {
            continue;
        }
        if !t
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "+/=".contains(c))
        {
            continue;
        }
        let first = b64(&t[..8]);
        let nb = u32::from_le_bytes(first[..4].try_into().unwrap()) as usize;
        let hlen = (3 + nb) * 4;
        let hb = hlen.div_ceil(3) * 4;
        let head = b64(&t[..hb]);
        let sizes: Vec<usize> = (0..nb)
            .map(|i| u32::from_le_bytes(head[12 + 4 * i..16 + 4 * i].try_into().unwrap()) as usize)
            .collect();
        let data = b64(&t[hb..]);
        let mut raw = Vec::new();
        let mut pos = 0;
        for s in sizes {
            raw.extend(
                hdf5rs::filters::deflate::zlib_decompress(&data[pos..pos + s], 1 << 15, 1 << 26)
                    .unwrap(),
            );
            pos += s;
        }
        out.push(if ty == "f64" {
            raw.chunks(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        } else {
            raw.chunks(4)
                .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap())))
                .collect()
        });
    }
    out
}

/// Compares two VTK files; returns the relative difference of the field
/// array (the first one) and asserts the coordinates agree.
fn compare_vtk(want: &Path, got: &Path) -> f64 {
    let (a, b) = (
        std::fs::read_to_string(want).unwrap(),
        std::fs::read_to_string(got).unwrap(),
    );
    let name = want.file_name().unwrap().to_string_lossy();
    assert_eq!(vtk_skeleton(&a), vtk_skeleton(&b), "{name}: VTK structure");
    #[cfg(feature = "hdf5-deflate")]
    {
        let (xa, xb) = (vtk_arrays(&a), vtk_arrays(&b));
        assert_eq!(xa.len(), xb.len(), "{name}: array count");
        for (x, y) in xa.iter().zip(&xb).skip(1) {
            for (p, q) in x.iter().zip(y) {
                assert!(
                    (p - q).abs() <= 1e-6 * p.abs().max(1e-9),
                    "{name}: coordinates"
                );
            }
        }
        let den = xa[0].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let num = xa[0]
            .iter()
            .zip(&xb[0])
            .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
        if den > 0.0 { num / den } else { num }
    }
    #[cfg(not(feature = "hdf5-deflate"))]
    0.0
}

// ---------------------------------------------------------------------
// Cases.

/// The fixture files of `dir` (HDF5 and VTK).
fn fixture_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e == "h5" || e == "vtr" || e == "vts")
        })
        .collect();
    v.sort();
    v
}

/// Compares every fixture file of `case/sub` with the ems output; returns
/// the worst relative difference per file, skipping the numbers of files
/// in `values_skip`.
fn check(
    case: &str,
    sub: &str,
    opts: DumpFileOptions,
    values_skip: &[&str],
) -> BTreeMap<String, f64> {
    let dir = if sub.is_empty() {
        fixture(case)
    } else {
        fixture(case).join(sub)
    };
    let out = run_ems(case, sub, &opts);
    let mut errs = BTreeMap::new();
    for f in fixture_files(&dir) {
        let n = f.file_name().unwrap().to_string_lossy().to_string();
        let g = out.join(&n);
        assert!(g.exists(), "{case}: ems did not write {n}");
        let e = if n.ends_with(".h5") {
            compare_h5(
                &f,
                &g,
                &[
                    "proc_time",
                    "maxSAR",
                    "maxSAR_idx",
                    "valid_cubes",
                    "used_cubes",
                    "unused_cubes",
                ],
            )
        } else {
            compare_vtk(&f, &g)
        };
        let e = if values_skip.iter().any(|s| n.starts_with(s)) {
            0.0
        } else {
            e
        };
        errs.insert(n, e);
    }
    let worst = errs.values().fold(0.0f64, |a, &b| a.max(b));
    eprintln!("{case}/{sub}: worst relative difference {worst:e}");
    errs
}

fn assert_close(case: &str, errs: &BTreeMap<String, f64>, tol: f64) {
    for (n, e) in errs {
        assert!(
            *e <= tol,
            "{case}/{n}: relative difference {e:e} > {tol:e}\n{errs:#?}"
        );
    }
}

#[test]
fn time_domain_dumps() {
    let e = check("dump_td", "", DumpFileOptions::new(), &[]);
    assert_eq!(e.len(), 18 + 12);
    assert_close("dump_td", &e, 1e-5);
}

#[test]
fn frequency_domain_dumps() {
    let e = check("dump_fd", "", DumpFileOptions::new(), &[]);
    assert_close("dump_fd", &e, 1e-4);
}

#[test]
fn legacy_hdf5_layout() {
    let e = check(
        "dump_legacy",
        "",
        DumpFileOptions::new().legacy_hdf5(true),
        &[],
    );
    assert_close("dump_legacy", &e, 1e-5);
}

#[test]
fn sampling_rules() {
    let e = check("dump_sampling", "", DumpFileOptions::new(), &[]);
    assert!(!fixture("dump_sampling").join("outside.h5").exists());
    assert_close("dump_sampling", &e, 1e-5);
}

#[test]
fn cylindrical_dumps() {
    let e = check("dump_cyl", "", DumpFileOptions::new(), &[]);
    assert_close("dump_cyl", &e, 1e-5);
    let e = check(
        "dump_cyl",
        "native",
        DumpFileOptions::new().native_field_dumps(true),
        &[],
    );
    assert_close("dump_cyl/native", &e, 1e-5);
}

/// SAR files: local and raw SAR match; the 1 g average uses openEMS's
/// `SIMPLE` rule, but openEMS solves the cube size only to ±5 % of the
/// mass (all cubes of this small block are clipped: within 2.9 % of the
/// peak, `tests/openems_sar_calc.rs`), so only its structure is compared.
#[test]
fn sar_dumps() {
    let e = check("dump_sar", "", DumpFileOptions::new(), &["sar_1g"]);
    assert_close("dump_sar", &e, 1e-5);
}

/// SAR files on a cylindrical grid (`cyl_sar`, gen-cyl.sh): openEMS's
/// cell volume is `ρ_i·Δα·Δρ·Δz` with the radius of the cell's lower ρ
/// line, and it scales the α cell widths by the drawing unit; ems writes
/// the same (its own SAR volumes use the exact annular cells). The cell
/// fields of the two codes differ by about 1e-4 of their peak on this
/// model, which shows as 3.5e-3 of the peak SAR in the weaker tissue
/// field (both codes' local SAR equal `κ|E|²/(2ρ)` of their raw files to
/// 1e-7).
#[test]
fn cylindrical_sar_dumps() {
    let e = check("cyl_sar", "", DumpFileOptions::new(), &[]);
    assert_close("cyl_sar", &e, 6e-3);
}
