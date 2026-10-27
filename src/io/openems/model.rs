//! Reading openEMS model files with openEMS's reading rules (feature
//! `xml`).
//!
//! csx reads the CSXCAD/openEMS XML format leniently and with its own
//! defaults. Where openEMS v0.37 reads the same file differently, and the
//! difference is visible in its output (observed on the black-box oracle,
//! fixtures `tests/fixtures/openems/sweep_*`, design §5.16.3), [`read_model`]
//! and [`parse_model`] rewrite the raw XML before csx sees it, so that an
//! openEMS user's file means the same in ems:
//!
//! * **Boundary names** (`<BoundaryCond xmin=…>`): `PEC`, `PMC`, `MUR` and
//!   `PML_<n>` are case sensitive; a value starting with a digit or sign is
//!   read as an integer prefix (`1.7` is 1, `2.000000e+00` is 2, `0x1` is
//!   0): 0 PEC, 1 PMC, 2 MUR, 3 `PML_8`, anything else PEC; `PML_<n>`
//!   takes the integer prefix of `<n>` (`PML_08`, `PML_8.7` are `PML_8`),
//!   and `n ≤ 0` is PEC. Every other spelling (`pec`, `Mur`, `PML`,
//!   `PML 8`, `" PEC"`) is PEC, with openEMS's "unknown" warning. csx
//!   accepts more spellings (case-insensitive, `PML 8`) and rejects some
//!   that openEMS runs (`4`, `PML_8.0`).
//! * **`OverSampling`** below 2 is the default, 4 (csx rejects negative
//!   values).
//! * **`DumpMode`**: a field dump (`DumpType` 0–5, 10–15) without the
//!   attribute is mode 0 (no interpolation), not csx's 1 (design §8.1).
//! * **Anisotropic materials** (`Isotropy="0"`): a list with fewer than
//!   three values sets the leading components only; the others keep the
//!   default (`ε_r = μ_r = 1`, `κ = σ = 0`, weight 1). csx repeats a single
//!   value on all three axes and fills a short list with 0.
//! * **Probe and dump boxes**: a `ProbeBox` / `DumpBox` with several
//!   primitives is several probes, `<name>_0`, `<name>_1`, … (one file
//!   each), in primitive order; a primitive's `<Transformation>` is
//!   ignored (the box of its untransformed coordinates is used); a
//!   voltage probe whose primitive is not a `Box` is skipped (openEMS:
//!   "Probe primitive type or dimension not suitable").
//! * **Lumped elements**: primitives other than `Box` are skipped (openEMS:
//!   "Primitives other than boxes are not supported for lumped
//!   elements").
//!
//! * **Cylindrical meshes** without `CylinderCoords="1"` on `<FDTD>`
//!   become Cartesian meshes, and their primitives' coordinates Cartesian
//!   ones (openEMS's solver choice).
//!
//! Each rewrite that changes the meaning of the file adds a warning.

use std::path::Path;

use csx::io::dom::{Element, Node};
use csx::{ContinuousStructure, OpenEmsDocument};

use crate::{Error, Result};

/// Reads the openEMS file at `path` with openEMS's reading rules (see the
/// [module docs](self)): the document and the warnings of the rewrites.
/// Files that are not valid UTF-8 are decoded as Latin-1 (as
/// [`OpenEmsDocument::read_file`] does); the structure's `base_dir` is the
/// file's directory.
///
/// # Errors
/// I/O errors, and the parse errors of [`parse_model`].
pub fn read_model(path: impl AsRef<Path>) -> Result<(OpenEmsDocument, Vec<String>)> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let text = String::from_utf8(bytes)
        .unwrap_or_else(|e| e.as_bytes().iter().map(|&c| char::from(c)).collect());
    let (mut doc, w) = parse_model(&text)?;
    doc.structure.base_dir = path.parent().map(Path::to_path_buf);
    Ok((doc, w))
}

/// Parses an openEMS document with openEMS's reading rules (see the
/// [module docs](self)).
///
/// ```
/// let xml = r#"<openEMS><FDTD NumberOfTimesteps="10">
///   <BoundaryCond xmin="pec" xmax="PML_8.0" ymin="4" ymax="1" zmin="MUR" zmax="3"/></FDTD>
///   <ContinuousStructure><RectilinearGrid DeltaUnit="1">
///   <XLines>0,1</XLines><YLines>0,1</YLines><ZLines>0,1</ZLines></RectilinearGrid>
///   <Properties/></ContinuousStructure></openEMS>"#;
/// let (doc, warnings) = ems::io::openems::parse_model(xml)?;
/// let bc = doc.fdtd_settings()?.unwrap().boundary.unwrap();
/// assert_eq!(bc.faces.map(|f| f.to_string()), ["PEC", "PML_8", "PEC", "PMC", "MUR", "PML_8"]);
/// assert_eq!(warnings.len(), 3); // "pec" (unknown), "PML_8.0" and "4" (read as)
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Errors
/// Malformed XML, an unexpected root element and the model errors of csx
/// ([`Error::Csx`]).
pub fn parse_model(xml: &str) -> Result<(OpenEmsDocument, Vec<String>)> {
    let mut root = csx::io::dom::parse(xml)?;
    let mut warnings = Vec::new();
    let mut doc = OpenEmsDocument::default();
    match root.name.as_str() {
        "ContinuousStructure" => {
            fix_structure(&mut root, &mut warnings);
            doc.structure = ContinuousStructure::from_xml_element(&root)?;
        }
        "openEMS" => {
            let mut found = false;
            for c in &mut root.children {
                let Node::Element(c) = c else { continue };
                match c.name.as_str() {
                    "FDTD" if doc.fdtd.is_none() => {
                        fix_fdtd(c, &mut warnings);
                        doc.fdtd = Some(c.clone());
                    }
                    "ContinuousStructure" if !found => {
                        fix_structure(c, &mut warnings);
                        doc.structure = ContinuousStructure::from_xml_element(c)?;
                        found = true;
                    }
                    _ => doc.extra.push(c.clone()),
                }
            }
        }
        other => {
            return Err(Error::Csx(csx::Error::Format(format!(
                "unexpected root element <{other}>"
            ))));
        }
    }
    // openEMS runs its cylindrical solver only with `CylinderCoords="1"`
    // on `<FDTD>`; without it a cylindrical mesh runs as a Cartesian one
    // (black box: rectilinear debug dumps, the α lines as y lines in
    // drawing units; boxes of materials, excitations and probes taken at
    // their literal corners as Cartesian boxes, the probes of such a
    // model bit for bit; design §5.16.11).
    if let Some(f) = &doc.fdtd
        && doc.structure.grid.coord_system == csx::CoordSystem::Cylindrical
        && !f
            .attr("CylinderCoords")
            .and_then(int_prefix)
            .is_some_and(|v| v != 0)
    {
        doc.structure.grid.coord_system = csx::CoordSystem::Cartesian;
        doc.structure.coord_system = csx::CoordSystem::Cartesian;
        warnings.push(
            "cylindrical mesh without CylinderCoords=\"1\" on <FDTD>: run as a Cartesian \
             mesh (rho, alpha, z and the primitives' coordinates as x, y, z), as openEMS does"
                .into(),
        );
    }
    Ok((doc, warnings))
}

fn elements_mut(el: &mut Element) -> impl Iterator<Item = &mut Element> {
    el.children.iter_mut().filter_map(|n| match n {
        Node::Element(e) => Some(e),
        Node::Text(_) => None,
    })
}

fn child_mut<'a>(el: &'a mut Element, name: &str) -> Option<&'a mut Element> {
    elements_mut(el).find(|e| e.name == name)
}

/// The integer prefix of `s` as C's `atoi` reads it (leading white space,
/// an optional sign, digits); `None` without digits.
fn int_prefix(s: &str) -> Option<i64> {
    let t = s.trim_start();
    let (neg, rest) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return None;
    }
    let v = digits.parse::<i64>().unwrap_or(i64::MAX);
    Some(if neg { -v } else { v })
}

/// openEMS's reading of a boundary value: the canonical csx spelling and
/// whether openEMS warns about it ("unknown ... set to PEC").
pub(crate) fn boundary_name(v: &str) -> (String, bool) {
    match v {
        "PEC" | "PMC" | "MUR" => return (v.to_string(), false),
        _ => {}
    }
    if let Some(n) = v.strip_prefix("PML_") {
        return match int_prefix(n) {
            Some(n) if n > 0 => (format!("PML_{n}"), false),
            _ => ("PEC".into(), false),
        };
    }
    let numeric = v
        .trim_start()
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit() || c == '-' || c == '+');
    if numeric {
        let name = match int_prefix(v) {
            Some(1) => "PMC",
            Some(2) => "MUR",
            Some(3) => "PML_8",
            _ => "PEC",
        };
        return (name.into(), false);
    }
    ("PEC".into(), true)
}

fn fix_fdtd(fdtd: &mut Element, warnings: &mut Vec<String>) {
    // OverSampling below 2 is openEMS's default (4); csx rejects negative
    // values.
    if fdtd
        .attr("OverSampling")
        .and_then(|v| v.trim().parse::<f64>().ok())
        .is_some_and(|v| v < 2.0)
    {
        fdtd.attrs.retain(|(k, _)| k != "OverSampling");
    }
    let Some(bc) = child_mut(fdtd, "BoundaryCond") else {
        return;
    };
    for face in csx::fdtd::FACE_NAMES {
        let Some(v) = bc.attr(face).map(str::to_owned) else {
            continue;
        };
        let (name, unknown) = boundary_name(&v);
        if unknown {
            warnings.push(format!(
                "boundary condition {face}={v:?} is unknown to openEMS (names are case \
                 sensitive: PEC, PMC, MUR, PML_<n>); set to PEC, as openEMS does"
            ));
        } else if name != v && !matches!(v.as_str(), "0" | "1" | "2" | "3") {
            warnings.push(format!(
                "boundary condition {face}={v:?} read as {name}, as openEMS does"
            ));
        }
        bc.set_attr(face, name);
    }
}

/// Default of a material quantity (`<Property>`) and of its weight.
const MATERIAL_DEFAULTS: [(&str, &str); 4] = [
    ("Epsilon", "1"),
    ("Mue", "1"),
    ("Kappa", "0"),
    ("Sigma", "0"),
];

/// Splits a list at commas outside parentheses (as csx does).
fn split_list(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0);
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth <= 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() || !out.is_empty() {
        out.push(last);
    }
    out
}

fn fix_anisotropic(prop: &mut Element) {
    let iso = prop.attr("Isotropy").map(str::trim);
    let anisotropic = iso.is_some_and(|v| {
        v == "0" || v.eq_ignore_ascii_case("false") || v.parse::<f64>().is_ok_and(|x| x == 0.0)
    });
    if !anisotropic {
        return;
    }
    for (child, weight) in [("Property", false), ("Weight", true)] {
        let Some(c) = child_mut(prop, child) else {
            continue;
        };
        for (q, def) in MATERIAL_DEFAULTS {
            let Some(v) = c.attr(q) else { continue };
            let parts = split_list(v);
            if parts.is_empty() || parts.len() >= 3 {
                continue;
            }
            let def = if weight { "1" } else { def };
            let mut full: Vec<&str> = parts;
            while full.len() < 3 {
                full.push(def);
            }
            let joined = full.join(",");
            c.set_attr(q, joined);
        }
    }
}

/// True for the field dump types (time and frequency domain E, H, J,
/// rot H, D, B), whose `DumpMode` default matters.
fn is_field_dump(el: &Element) -> bool {
    let t = el
        .attr("DumpType")
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(0.0);
    t.fract() == 0.0 && matches!(t as i64, 0..=5 | 10..=15)
}

fn primitives_of(el: &Element) -> Vec<Element> {
    el.child("Primitives")
        .map(|p| p.elements().cloned().collect())
        .unwrap_or_default()
}

fn with_primitives(el: &Element, prims: Vec<Element>) -> Element {
    let mut out = el.clone();
    if let Some(p) = child_mut(&mut out, "Primitives") {
        p.children = prims.into_iter().map(Node::Element).collect();
    }
    out
}

/// The probe/dump rewrites; `None` drops the property.
fn fix_box(el: &Element, warnings: &mut Vec<String>) -> Vec<Element> {
    let name = el.attr("Name").unwrap_or("").to_string();
    let mut prims = primitives_of(el);
    for p in &mut prims {
        let before = p.children.len();
        p.children
            .retain(|n| !matches!(n, Node::Element(e) if e.name == "Transformation"));
        if p.children.len() != before {
            warnings.push(format!(
                "probe/dump {name:?}: the transformation of its {} primitive is ignored, \
                 as openEMS does",
                p.name
            ));
        }
    }
    let voltage = el.name == "ProbeBox"
        && el
            .attr("Type")
            .and_then(|v| v.trim().parse::<f64>().ok())
            .is_none_or(|t| t == 0.0);
    let mut el = with_primitives(el, prims.clone());
    if el.name == "DumpBox" && el.attr("DumpMode").is_none() && is_field_dump(&el) {
        el.set_attr("DumpMode", "0");
    }
    let keep = |p: &Element, n: &str, warnings: &mut Vec<String>| -> bool {
        if voltage && p.name != "Box" {
            warnings.push(format!(
                "voltage probe {n:?}: a {} primitive is not supported by openEMS (a Box is \
                 needed); the probe is skipped, as openEMS does",
                p.name
            ));
            false
        } else {
            true
        }
    };
    if prims.len() <= 1 {
        return match prims.first() {
            Some(p) if !keep(p, &name, warnings) => Vec::new(),
            _ => vec![el],
        };
    }
    warnings.push(format!(
        "probe/dump {name:?} has {} primitives: openEMS runs one per primitive, named \
         {name}_0 … {name}_{}",
        prims.len(),
        prims.len() - 1
    ));
    let mut out = Vec::new();
    for (i, p) in prims.into_iter().enumerate() {
        let n = format!("{name}_{i}");
        if !keep(&p, &n, warnings) {
            continue;
        }
        let mut e = with_primitives(&el, vec![p]);
        e.set_attr("Name", n);
        out.push(e);
    }
    out
}

fn fix_lumped(el: &mut Element, warnings: &mut Vec<String>) {
    let name = el.attr("Name").unwrap_or("").to_string();
    let Some(p) = child_mut(el, "Primitives") else {
        return;
    };
    p.children.retain(|n| match n {
        Node::Element(e) if e.name != "Box" => {
            warnings.push(format!(
                "lumped element {name:?}: a {} primitive is not supported by openEMS (boxes \
                 only); skipped, as openEMS does",
                e.name
            ));
            false
        }
        _ => true,
    });
}

fn fix_structure(cs: &mut Element, warnings: &mut Vec<String>) {
    let Some(props) = child_mut(cs, "Properties") else {
        return;
    };
    let mut children = Vec::with_capacity(props.children.len());
    for n in std::mem::take(&mut props.children) {
        match n {
            Node::Element(mut e) => match e.name.as_str() {
                "Material" | "LorentzMaterial" | "DebyeMaterial" | "DiscMaterial" => {
                    fix_anisotropic(&mut e);
                    children.push(Node::Element(e));
                }
                "ProbeBox" | "DumpBox" => {
                    children.extend(fix_box(&e, warnings).into_iter().map(Node::Element));
                }
                "LumpedElement" => {
                    fix_lumped(&mut e, warnings);
                    children.push(Node::Element(e));
                }
                _ => children.push(Node::Element(e)),
            },
            t => children.push(t),
        }
    }
    props.children = children;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_names() {
        for (v, name, unknown) in [
            ("PEC", "PEC", false),
            ("PMC", "PMC", false),
            ("MUR", "MUR", false),
            ("0", "PEC", false),
            ("1", "PMC", false),
            ("2", "MUR", false),
            ("3", "PML_8", false),
            ("4", "PEC", false),
            ("-1", "PEC", false),
            ("1.7", "PMC", false),
            ("2.000000e+00", "MUR", false),
            ("0x1", "PEC", false),
            ("PML_8", "PML_8", false),
            ("PML_08", "PML_8", false),
            ("PML_8.7", "PML_8", false),
            ("PML_1", "PML_1", false),
            ("PML_0", "PEC", false),
            ("PML_x", "PEC", false),
            ("PML_-2", "PEC", false),
            ("pec", "PEC", true),
            ("Mur", "PEC", true),
            ("PML", "PEC", true),
            ("PML 8", "PEC", true),
            ("PML-8", "PEC", true),
            (" PEC", "PEC", true),
            ("PMCX", "PEC", true),
        ] {
            assert_eq!(boundary_name(v), (name.to_string(), unknown), "{v:?}");
        }
    }

    const GRID: &str = r#"<RectilinearGrid DeltaUnit="1"><XLines>0,1,2</XLines><YLines>0,1,2</YLines><ZLines>0,1,2</ZLines></RectilinearGrid>"#;

    fn structure(props: &str) -> (ContinuousStructure, Vec<String>) {
        let xml = format!(
            "<openEMS><FDTD/><ContinuousStructure>{GRID}<Properties>{props}</Properties></ContinuousStructure></openEMS>"
        );
        let (doc, w) = parse_model(&xml).unwrap();
        (doc.structure, w)
    }

    #[test]
    fn anisotropic_defaults() {
        use csx::properties::{MaterialQuantity, PropertyKind};
        let (s, _) = structure(
            r#"<Material Name="m" Isotropy="0"><Property Epsilon="3" Kappa="0.1,0.2"/><Weight Epsilon="x"/></Material>
               <Material Name="iso"><Property Epsilon="3"/></Material>"#,
        );
        let PropertyKind::Material(m) = &s.properties()[0].kind else {
            panic!()
        };
        let eps = m.quantity(MaterialQuantity::Epsilon);
        let v: Vec<f64> = eps.value.iter().map(|x| x.value()).collect();
        assert_eq!(v, [3.0, 1.0, 1.0]);
        let k = m.quantity(MaterialQuantity::Kappa);
        let v: Vec<f64> = k.value.iter().map(|x| x.value()).collect();
        assert_eq!(v, [0.1, 0.2, 0.0]);
        let w: Vec<&str> = eps.weight.iter().map(|w| w.source()).collect();
        assert_eq!(w, ["x", "1", "1"]);
        let PropertyKind::Material(m) = &s.properties()[1].kind else {
            panic!()
        };
        let v: Vec<f64> = m
            .quantity(MaterialQuantity::Epsilon)
            .value
            .iter()
            .map(|x| x.value())
            .collect();
        assert_eq!(v, [3.0; 3]);
    }

    #[test]
    fn boxes() {
        use csx::properties::{DumpMode, PropertyKind};
        let b =
            |x: f64| format!(r#"<Box><P1 X="{x}" Y="0" Z="0"/><P2 X="{x}" Y="0" Z="1"/></Box>"#);
        let (s, w) = structure(&format!(
            r#"<ProbeBox Name="u" Type="0"><Primitives>{}{}</Primitives></ProbeBox>
               <ProbeBox Name="c"><Primitives><Cylinder Radius="0"><P1 X="0" Y="0" Z="0"/><P2 X="0" Y="0" Z="1"/></Cylinder></Primitives></ProbeBox>
               <ProbeBox Name="e" Type="2"><Primitives><Box><P1 X="0" Y="0" Z="0"/><P2 X="0" Y="0" Z="0"/><Transformation><Translate Argument="1,1,1"/></Transformation></Box></Primitives></ProbeBox>
               <DumpBox Name="d" DumpType="1"><Primitives>{}</Primitives></DumpBox>
               <DumpBox Name="s" DumpType="20"><Primitives>{}</Primitives></DumpBox>
               <LumpedElement Name="r" Direction="2" R="50"><Primitives>{}<Sphere Radius="1"><Center X="0" Y="0" Z="0"/></Sphere></Primitives></LumpedElement>"#,
            b(0.0),
            b(1.0),
            b(0.0),
            b(0.0),
            b(0.0)
        ));
        let names: Vec<&str> = s.properties().iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["u_0", "u_1", "e", "d", "s", "r"]);
        assert_eq!(s.properties()[5].primitives.len(), 1);
        assert!(s.properties()[2].primitives[0].transform.is_empty());
        let PropertyKind::DumpBox(d) = &s.properties()[3].kind else {
            panic!()
        };
        assert_eq!(d.dump_mode, DumpMode::NoInterpolation);
        let PropertyKind::DumpBox(d) = &s.properties()[4].kind else {
            panic!()
        };
        assert_eq!(d.dump_mode, DumpMode::Node);
        assert_eq!(w.len(), 4, "{w:?}");
    }
}
