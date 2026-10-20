//! Structure-and-value comparison of two HDF5 files (openEMS black-box
//! parity tests): same objects, dataset shapes and datatypes, attribute
//! names and types; returns the value differences.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::Path;

use hdf5rs::{File, Value};

/// Attribute name → (type signature, values).
pub type Attrs = BTreeMap<String, (String, Vec<Value>)>;

/// A flattened HDF5 file: path → (kind/shape/datatype signature,
/// attributes, numeric values).
#[derive(Debug, Default)]
pub struct H5Tree {
    pub objects: BTreeMap<String, (String, Attrs, Vec<f64>)>,
}

fn dtype_sig(dt: &hdf5rs::Datatype) -> String {
    format!("{:?}/{}", dt.class, dt.size)
}

/// The numbers in `v` (complex and compound values flattened).
pub fn numbers(v: &[Value]) -> Vec<f64> {
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

/// Reads every group, dataset and attribute of the file at `path`.
pub fn read_tree(path: &Path) -> H5Tree {
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

/// Differences of two files with the same structure (asserted).
#[derive(Debug, Default, Clone, Copy)]
pub struct H5Diff {
    /// Largest relative difference of a numeric attribute.
    pub attrs: f64,
    /// Largest data difference relative to the largest |value| of the
    /// same dataset in `want`, over all datasets.
    pub data: f64,
}

impl H5Diff {
    /// The larger of the two.
    pub fn worst(&self) -> f64 {
        self.attrs.max(self.data)
    }
}

/// Compares `got` with `want`: identical structure (asserted), and the
/// relative value differences. Attributes in `loose` are compared by type
/// only, datasets whose path contains one of `skip_data` by shape only.
pub fn compare_h5(want: &Path, got: &Path, loose: &[&str], skip_data: &[&str]) -> H5Diff {
    let (a, b) = (read_tree(want), read_tree(got));
    let name = want.file_name().unwrap().to_string_lossy();
    let ka: Vec<_> = a.objects.keys().collect();
    let kb: Vec<_> = b.objects.keys().collect();
    assert_eq!(ka, kb, "{name}: objects");
    let mut d = H5Diff::default();
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
                assert_eq!(na.len(), nb.len(), "{name}:{p}@{k}");
                for (x, y) in na.iter().zip(&nb) {
                    let m = x.abs().max(y.abs());
                    if m > 0.0 {
                        d.attrs = d.attrs.max((x - y).abs() / m);
                    }
                }
            }
        }
        assert_eq!(va.len(), vb.len(), "{name}:{p}: value count");
        if skip_data.iter().any(|s| p.contains(s)) {
            continue;
        }
        let den = va.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let num = va
            .iter()
            .zip(vb)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        if den > 0.0 {
            d.data = d.data.max(num / den);
        } else {
            d.data = d.data.max(num);
        }
    }
    d
}
