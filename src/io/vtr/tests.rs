use super::*;
use std::collections::HashMap;

// ---------------------------------------------------------------- helpers

/// Test-local base64 decoder (standard alphabet, padding, whitespace ignored).
fn b64_decode(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("bad base64 char {c:?}"),
        }
    };
    let chars: Vec<u8> = s.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
    assert_eq!(chars.len() % 4, 0, "base64 length");
    let mut out = Vec::new();
    for q in chars.chunks(4) {
        let pad = q.iter().filter(|&&c| c == b'=').count();
        let n = q
            .iter()
            .map(|&c| if c == b'=' { 0 } else { val(c) })
            .fold(0u32, |acc, v| (acc << 6) | v);
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&bytes[..3 - pad]);
    }
    out
}

#[derive(Debug)]
struct Arr {
    attrs: HashMap<String, String>,
    body: String,
}

fn parse_attrs(tag: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut rest = tag;
    while let Some(eq) = rest.find("=\"") {
        let key = rest[..eq].split_whitespace().last().unwrap().to_string();
        let after = &rest[eq + 2..];
        let end = after.find('"').unwrap();
        map.insert(key, after[..end].to_string());
        rest = &after[end + 1..];
    }
    map
}

/// All `<DataArray>` elements (attributes and text body) of an XML document.
fn arrays(xml: &str) -> Vec<Arr> {
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<DataArray") {
        let after = &rest[start..];
        let gt = after.find('>').unwrap();
        let attrs = parse_attrs(&after[..gt]);
        let close = after.find("</DataArray>").unwrap();
        out.push(Arr {
            attrs,
            body: after[gt + 1..close].to_string(),
        });
        rest = &after[close..];
    }
    out
}

fn find<'x>(arrs: &'x [Arr], name: &str) -> &'x Arr {
    arrs.iter()
        .find(|a| a.attrs["Name"] == name)
        .unwrap_or_else(|| panic!("array {name} not found"))
}

/// Checks that tags are balanced and properly nested; returns the element
/// names in document order.
fn check_well_formed(xml: &str) -> Vec<String> {
    let mut stack: Vec<String> = Vec::new();
    let mut names = Vec::new();
    let mut rest = xml;
    assert!(xml.starts_with("<?xml version=\"1.0\"?>\n"));
    while let Some(lt) = rest.find('<') {
        let after = &rest[lt + 1..];
        let gt = after.find('>').expect("unterminated tag");
        let tag = &after[..gt];
        assert!(!tag.contains('<'), "'<' inside tag {tag:?}");
        if let Some(name) = tag.strip_prefix('/') {
            assert_eq!(stack.pop().as_deref(), Some(name), "mismatched close");
        } else if !tag.starts_with('?') {
            let name = tag.split_whitespace().next().unwrap().trim_end_matches('/');
            names.push(name.to_string());
            if !tag.ends_with('/') {
                stack.push(name.to_string());
            }
        }
        rest = &after[gt + 1..];
    }
    assert!(stack.is_empty(), "unclosed: {stack:?}");
    assert!(rest.trim().is_empty());
    names
}

fn ascii_values(a: &Arr) -> Vec<f64> {
    a.body
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect()
}

fn binary_values(a: &Arr, header: HeaderType) -> Vec<f64> {
    let bytes = b64_decode(&a.body);
    decode_block(&bytes, header, &a.attrs["type"]).0
}

/// Decodes one header + data block; returns values and bytes consumed.
fn decode_block(bytes: &[u8], header: HeaderType, ty: &str) -> (Vec<f64>, usize) {
    let (len, h) = match header {
        HeaderType::UInt32 => (
            u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize,
            4,
        ),
        HeaderType::UInt64 => (
            u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize,
            8,
        ),
    };
    let data = &bytes[h..h + len];
    let vals = match ty {
        "Float32" => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f64::from(f32::from_le_bytes(*c)))
            .collect(),
        "Float64" => data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect(),
        t => panic!("type {t}"),
    };
    (vals, h + len)
}

fn render(w: &VtrWriter<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    w.write_to(&mut out).unwrap();
    out
}

fn text(w: &VtrWriter<'_>) -> String {
    String::from_utf8(render(w)).unwrap()
}

// A non-cubic 2×3×4 grid.
const X: [f64; 2] = [0.0, 1.0];
const Y: [f64; 3] = [0.0, 0.5, 2.0];
const Z: [f64; 4] = [-1.0, 0.0, 1.0, 3.0];

/// Tag value of node/cell (i, j, k).
fn tag(i: usize, j: usize, k: usize) -> f64 {
    (100 * i + 10 * j + k) as f64
}

/// Buffer of tags for `dims` in `layout` order.
fn tagged(dims: [usize; 3], layout: Layout) -> Vec<f64> {
    let mut v = vec![f64::NAN; dims.iter().product()];
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                v[layout.index(dims, i, j, k)] = tag(i, j, k);
            }
        }
    }
    v
}

/// Expected VTK-ordered values (x fastest) for `dims`, `ncomp` components
/// where component `c` adds `c * 1000`.
fn expected(dims: [usize; 3], ncomp: usize) -> Vec<f64> {
    let mut v = Vec::new();
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                for c in 0..ncomp {
                    v.push(tag(i, j, k) + 1000.0 * c as f64);
                }
            }
        }
    }
    v
}

// ------------------------------------------------------------------ tests

#[test]
fn layout_index() {
    let d = [2, 3, 4];
    assert_eq!(Layout::ZFastest.index(d, 1, 2, 3), (3 + 2) * 4 + 3);
    assert_eq!(Layout::XFastest.index(d, 1, 2, 3), (3 * 3 + 2) * 2 + 1);
    // Both are bijections onto 0..24.
    for l in [Layout::ZFastest, Layout::XFastest] {
        let mut seen = [false; 24];
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    seen[l.index(d, i, j, k)] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s));
    }
}

#[test]
fn structure_and_metadata() {
    let p = tagged([2, 3, 4], Layout::ZFastest);
    let c = tagged([1, 2, 3], Layout::ZFastest);
    let v: Vec<[f32; 3]> = vec![[0.0; 3]; 24];
    let freq = [2.4e9];
    let s = text(
        &VtrWriter::new(&X, &Y, &Z)
            .time(1.25e-9)
            .field("Frequency", &freq)
            .point_scalar("p", &p, Layout::ZFastest)
            .point_vector("E", &v, Layout::ZFastest)
            .cell_scalar("c", &c, Layout::ZFastest),
    );
    let names = check_well_formed(&s);
    assert_eq!(
        names,
        [
            "VTKFile",
            "RectilinearGrid",
            "FieldData",
            "DataArray",
            "DataArray",
            "Piece",
            "PointData",
            "DataArray",
            "DataArray",
            "CellData",
            "DataArray",
            "Coordinates",
            "DataArray",
            "DataArray",
            "DataArray",
        ]
    );
    assert!(s.contains(
        r#"<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">"#
    ));
    assert!(s.contains(r#"<RectilinearGrid WholeExtent="0 1 0 2 0 3">"#));
    assert!(s.contains(r#"<Piece Extent="0 1 0 2 0 3">"#));
    assert!(s.contains(r#"<PointData Scalars="p" Vectors="E">"#));
    assert!(s.contains(r#"<CellData Scalars="c">"#));
    let arrs = arrays(&s);
    let t = find(&arrs, "TimeValue");
    assert_eq!(t.attrs["NumberOfTuples"], "1");
    assert_eq!(t.attrs["format"], "ascii");
    assert_eq!(ascii_values(t), [1.25e-9]);
    assert_eq!(ascii_values(find(&arrs, "Frequency")), [2.4e9]);
    let e = find(&arrs, "E");
    assert_eq!(e.attrs["type"], "Float32");
    assert_eq!(e.attrs["NumberOfComponents"], "3");
    assert_eq!(e.attrs["format"], "binary");
    assert_eq!(find(&arrs, "p").attrs["type"], "Float64");
    for (name, coords) in [("x", &X[..]), ("y", &Y[..]), ("z", &Z[..])] {
        assert_eq!(binary_values(find(&arrs, name), HeaderType::UInt64), coords);
    }
}

#[test]
fn ascii_round_trip_and_reordering() {
    let dims = [2, 3, 4];
    let z = tagged(dims, Layout::ZFastest);
    let x = tagged(dims, Layout::XFastest);
    assert_ne!(z, x);
    let s = text(
        &VtrWriter::new(&X, &Y, &Z)
            .encoding(Encoding::Ascii)
            .point_scalar("z", &z, Layout::ZFastest)
            .point_scalar("x", &x, Layout::XFastest),
    );
    check_well_formed(&s);
    let arrs = arrays(&s);
    assert!(arrs.iter().all(|a| a.attrs["format"] == "ascii"));
    let want = expected(dims, 1);
    assert_eq!(ascii_values(find(&arrs, "z")), want);
    assert_eq!(ascii_values(find(&arrs, "x")), want);
    // Grid coordinates come back exactly.
    let coords: Vec<&Arr> = arrs
        .iter()
        .filter(|a| ["x", "y", "z"].contains(&a.attrs["Name"].as_str()))
        .skip(2)
        .collect();
    assert_eq!(ascii_values(coords[0]), X);
    assert_eq!(ascii_values(coords[1]), Y);
    assert_eq!(ascii_values(coords[2]), Z);
}

#[test]
fn ascii_exact_floats() {
    let x = [0.0, 0.1, 1.0 / 3.0];
    let vals: Vec<f64> = vec![std::f64::consts::PI, -1e-300, 5e-324];
    let vals32: Vec<f32> = vec![0.1, -3.4e38, 1.0 / 3.0];
    let s = text(
        &VtrWriter::new(&x, &[0.0], &[0.0])
            .encoding(Encoding::Ascii)
            .point_scalar("a", &vals, Layout::XFastest)
            .point_scalar("b", &vals32, Layout::XFastest),
    );
    let arrs = arrays(&s);
    assert_eq!(ascii_values(find(&arrs, "a")), vals);
    let b: Vec<f32> = find(&arrs, "b")
        .body
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect();
    assert_eq!(b, vals32);
    assert_eq!(ascii_values(find(&arrs, "x")), x);
}

#[test]
fn binary_vectors_both_headers() {
    let dims = [2, 3, 4];
    let comp = |c: usize| -> Vec<f32> {
        tagged(dims, Layout::ZFastest)
            .iter()
            .map(|v| (v + 1000.0 * c as f64) as f32)
            .collect()
    };
    let (cx, cy, cz) = (comp(0), comp(1), comp(2));
    let inter: Vec<[f32; 3]> = (0..24).map(|n| [cx[n], cy[n], cz[n]]).collect();
    for header in [HeaderType::UInt32, HeaderType::UInt64] {
        let s = text(
            &VtrWriter::new(&X, &Y, &Z)
                .header_type(header)
                .point_vector("inter", &inter, Layout::ZFastest)
                .point("split", Field::Split([&cx, &cy, &cz]), Layout::ZFastest),
        );
        check_well_formed(&s);
        assert!(s.contains(&format!(r#"header_type="{}""#, header.name())));
        let arrs = arrays(&s);
        let want = expected(dims, 3);
        for name in ["inter", "split"] {
            let a = find(&arrs, name);
            assert_eq!(a.attrs["NumberOfComponents"], "3");
            assert_eq!(binary_values(a, header), want, "{name} {header:?}");
            // The header counts data bytes only.
            let bytes = b64_decode(&a.body);
            assert_eq!(bytes.len(), header.size() + 24 * 3 * 4);
        }
        assert_eq!(binary_values(find(&arrs, "y"), header), Y);
    }
}

#[test]
fn cell_data_dims_and_planes() {
    // 2×3×4 points → 1×2×3 cells.
    let cdims = [1, 2, 3];
    let c = tagged(cdims, Layout::ZFastest);
    let s = text(
        &VtrWriter::new(&X, &Y, &Z)
            .encoding(Encoding::Ascii)
            .cell_scalar("c", &c, Layout::ZFastest),
    );
    assert_eq!(ascii_values(find(&arrays(&s), "c")), expected(cdims, 1));

    // A z-plane (3×4×1 points) has 2×3 cells.
    let x = [0.0, 1.0, 2.0];
    let y = [0.0, 1.0, 2.0, 3.0];
    let pdims = [3, 4, 1];
    let cdims = [2, 3, 1];
    let p = tagged(pdims, Layout::ZFastest);
    let cv: Vec<[f64; 3]> = tagged(cdims, Layout::ZFastest)
        .iter()
        .map(|&v| [v, v + 1000.0, v + 2000.0])
        .collect();
    let s = text(
        &VtrWriter::new(&x, &y, &[0.5])
            .encoding(Encoding::Ascii)
            .point_scalar("p", &p, Layout::ZFastest)
            .cell_vector("cv", &cv, Layout::ZFastest),
    );
    assert!(s.contains(r#"WholeExtent="0 2 0 3 0 0""#));
    let arrs = arrays(&s);
    assert_eq!(ascii_values(find(&arrs, "p")), expected(pdims, 1));
    assert_eq!(ascii_values(find(&arrs, "cv")), expected(cdims, 3));
}

#[test]
fn appended_raw() {
    let dims = [2, 3, 4];
    let p32: Vec<f32> = tagged(dims, Layout::ZFastest)
        .iter()
        .map(|&v| v as f32)
        .collect();
    let c = tagged([1, 2, 3], Layout::XFastest);
    for header in [HeaderType::UInt32, HeaderType::UInt64] {
        let out = render(
            &VtrWriter::new(&X, &Y, &Z)
                .time(3.0)
                .encoding(Encoding::Appended)
                .header_type(header)
                .point_scalar("p", &p32, Layout::ZFastest)
                .cell_scalar("c", &c, Layout::XFastest),
        );
        let marker = b"<AppendedData encoding=\"raw\">\n   _";
        let pos = out
            .windows(marker.len())
            .position(|w| w == marker)
            .expect("AppendedData block");
        let head = std::str::from_utf8(&out[..pos]).unwrap();
        let raw_start = pos + marker.len();
        let tail = b"\n  </AppendedData>\n</VTKFile>\n";
        assert!(out.ends_with(tail));
        let raw = &out[raw_start..out.len() - tail.len()];

        // The XML part (with the appended block stubbed out) is well formed.
        check_well_formed(&format!(
            "{head}<AppendedData encoding=\"raw\"></AppendedData>\n</VTKFile>\n"
        ));
        let arrs = arrays(head);
        let mut consumed = 0;
        let mut results = HashMap::new();
        for a in arrs.iter().filter(|a| a.attrs["format"] == "appended") {
            let off: usize = a.attrs["offset"].parse().unwrap();
            assert_eq!(off, consumed, "offsets are contiguous");
            let (vals, used) = decode_block(&raw[off..], header, &a.attrs["type"]);
            consumed += used;
            results.insert(a.attrs["Name"].clone(), vals);
        }
        assert_eq!(consumed, raw.len());
        assert_eq!(results["p"], expected(dims, 1));
        assert_eq!(results["c"], expected([1, 2, 3], 1));
        assert_eq!(results["x"], X);
        assert_eq!(results["z"], Z);
        // Field data stays inline ASCII.
        assert_eq!(ascii_values(find(&arrs, "TimeValue")), [3.0]);
    }
}

#[test]
fn complex_arrays() {
    let x = [0.0, 1.0, 2.0];
    let re = [3.0f64, 0.0, -1.0];
    let im = [4.0f64, 2.0, 0.0];
    let s = text(
        &VtrWriter::new(&x, &[0.0], &[0.0])
            .encoding(Encoding::Ascii)
            .point_complex(
                "V",
                Field::Scalar(&re),
                Field::Scalar(&im),
                Layout::ZFastest,
                ComplexForm::ReIm,
            )
            .point_complex(
                "V",
                Field::Scalar(&re),
                Field::Scalar(&im),
                Layout::ZFastest,
                ComplexForm::MagPhase,
            ),
    );
    let arrs = arrays(&s);
    assert_eq!(ascii_values(find(&arrs, "V_re")), re);
    assert_eq!(ascii_values(find(&arrs, "V_im")), im);
    assert_eq!(ascii_values(find(&arrs, "V_mag")), [5.0, 2.0, 1.0]);
    let ph = ascii_values(find(&arrs, "V_phase"));
    let want = [
        4f64.atan2(3.0),
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
    ];
    for (a, b) in ph.iter().zip(want) {
        assert!((a - b).abs() < 1e-15);
    }

    // Complex vector cell data, f32, split components.
    let y = [0.0, 1.0];
    let rx = [1.0f32, 2.0];
    let ry = [3.0f32, 4.0];
    let rz = [5.0f32, 6.0];
    let iv = [[-1.0f32, -3.0, -5.0], [-2.0, -4.0, -6.0]];
    // 3×2×1 points → 2×1×1 cells; im given interleaved, re split.
    let s = text(
        &VtrWriter::new(&x, &y, &[0.0])
            .encoding(Encoding::Ascii)
            .cell_complex(
                "J",
                Field::Split([&rx, &ry, &rz]),
                Field::Vector(&iv),
                Layout::ZFastest,
                ComplexForm::ReIm,
            ),
    );
    let arrs = arrays(&s);
    assert_eq!(
        ascii_values(find(&arrs, "J_re")),
        [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
    assert_eq!(
        ascii_values(find(&arrs, "J_im")),
        [-1.0, -3.0, -5.0, -2.0, -4.0, -6.0]
    );
}

fn invalid(w: VtrWriter<'_>) {
    let mut out = Vec::new();
    match w.write_to(&mut out) {
        Err(Error::Invalid(_)) => {}
        other => panic!("expected Error::Invalid, got {other:?}"),
    }
    assert!(out.is_empty(), "nothing written on error");
}

#[test]
fn validation_errors() {
    let p = vec![0.0f64; 24];
    let short = vec![0.0f64; 23];
    let cells = vec![0.0f64; 6];
    let v = vec![[0.0f32; 3]; 23];
    let s: Vec<f32> = vec![0.0; 24];
    let s_short: Vec<f32> = vec![0.0; 6];
    let g = || VtrWriter::new(&X, &Y, &Z);

    invalid(g().point_scalar("p", &short, Layout::ZFastest));
    // A cell-sized buffer is not a point buffer, and vice versa.
    invalid(g().point_scalar("p", &cells, Layout::ZFastest));
    invalid(g().cell_scalar("c", &p, Layout::ZFastest));
    invalid(g().point_vector("v", &v, Layout::ZFastest));
    invalid(g().point("v", Field::Split([&s, &s_short, &s]), Layout::ZFastest));
    invalid(g().point_complex(
        "z",
        Field::Scalar(&p),
        Field::Scalar(&short),
        Layout::ZFastest,
        ComplexForm::ReIm,
    ));
    invalid(g().point_complex(
        "z",
        Field::Scalar(&s),
        Field::Split([&s, &s, &s]),
        Layout::ZFastest,
        ComplexForm::ReIm,
    ));
    // Names.
    invalid(g().point_scalar("", &p, Layout::ZFastest));
    invalid(
        g().point_scalar("a", &p, Layout::ZFastest)
            .point_scalar("a", &s, Layout::ZFastest),
    );
    invalid(g().time(1.0).field("TimeValue", &[2.0]));
    // Grid.
    invalid(VtrWriter::new(&[], &Y, &Z));
    invalid(VtrWriter::new(&X, &[0.0, 0.0], &Z));
    invalid(VtrWriter::new(&X, &[1.0, 0.0], &Z));
    invalid(VtrWriter::new(&X, &Y, &[0.0, f64::NAN]));

    // The same name as point and as cell data is fine.
    let c = vec![0.0f64; 6];
    let mut out = Vec::new();
    g().point_scalar("a", &p, Layout::ZFastest)
        .cell_scalar("a", &c, Layout::ZFastest)
        .write_to(&mut out)
        .unwrap();
}

#[test]
fn attribute_escaping() {
    let p = [1.0f64];
    let s = text(&VtrWriter::new(&[0.0], &[0.0], &[0.0]).point_scalar(
        "a<\"&'>b",
        &p,
        Layout::ZFastest,
    ));
    check_well_formed(&s);
    assert!(s.contains(r#"Name="a&lt;&quot;&amp;&apos;&gt;b""#));
}

#[test]
fn write_to_path_matches_write_to() {
    let dir = std::env::temp_dir().join(format!("ems-vtr-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = tagged([2, 3, 4], Layout::ZFastest);
    let w = VtrWriter::new(&X, &Y, &Z).point_scalar("p", &p, Layout::ZFastest);
    let path = dir.join("dump.vtr");
    w.write(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), render(&w));

    let mut pvd = PvdCollection::new();
    pvd.push(0.0, "dump.vtr");
    let pvd_path = dir.join("series.pvd");
    pvd.write(&pvd_path).unwrap();
    assert!(
        std::fs::read_to_string(&pvd_path)
            .unwrap()
            .contains(r#"file="dump.vtr""#)
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pvd_structure() {
    let mut pvd = PvdCollection::new();
    pvd.push(0.0, "E_0.vtr")
        .push(2.5e-11, "E_1.vtr")
        .push_part(2.5e-11, 1, "H&1.vtr");
    assert_eq!(pvd.entries().len(), 3);
    let mut out = Vec::new();
    pvd.write_to(&mut out).unwrap();
    let s = String::from_utf8(out).unwrap();
    let names = check_well_formed(&s);
    assert_eq!(
        names,
        ["VTKFile", "Collection", "DataSet", "DataSet", "DataSet"]
    );
    assert!(s.contains(r#"<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">"#));
    let sets: Vec<HashMap<String, String>> = s
        .lines()
        .filter(|l| l.trim_start().starts_with("<DataSet"))
        .map(parse_attrs)
        .collect();
    let steps: Vec<f64> = sets
        .iter()
        .map(|a| a["timestep"].parse().unwrap())
        .collect();
    assert_eq!(steps, [0.0, 2.5e-11, 2.5e-11]);
    assert_eq!(sets[1]["file"], "E_1.vtr");
    assert_eq!(sets[2]["file"], "H&amp;1.vtr");
    assert_eq!(sets[2]["part"], "1");

    let mut bad = PvdCollection::new();
    bad.push(f64::NAN, "a.vtr");
    assert!(matches!(bad.write_to(Vec::new()), Err(Error::Invalid(_))));
    let mut bad = PvdCollection::new();
    bad.push(0.0, "");
    assert!(matches!(bad.write_to(Vec::new()), Err(Error::Invalid(_))));
}
