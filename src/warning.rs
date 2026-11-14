//! Typed warnings: what [`Simulation::warnings`](crate::Simulation::warnings)
//! and the other warning lists of the crate say, with a stable code, the
//! model property they are about and a count.
//!
//! The warning lists of the crate ([`Simulation::warnings`](crate::Simulation::warnings),
//! [`Compiled::warnings`](crate::Compiled::warnings),
//! `io::openems::read_model`) are lines
//! of text, one per finding; a model with twenty primitives reaching into
//! an absorbing layer gets twenty lines. [`Warning`] is one finding read
//! back from its line, and [`collect`] turns a list into one entry per
//! kind of finding and subject:
//!
//! ```
//! use ems::warning::{WarningCode, collect};
//!
//! let lines = [
//!     r#"mesh warning: Metal "gnd" extends into the PML of face(s) xmin [property "gnd" primitive #0]"#,
//!     r#"mesh warning: Metal "gnd" extends into the PML of face(s) ymax [property "gnd" primitive #3]"#,
//!     r#"probe "port_ut1": box corners snapped to the nearest mesh lines"#,
//! ];
//! let warnings = collect(&lines);
//! assert_eq!(warnings.len(), 2);
//! assert_eq!(warnings[0].code(), WarningCode::MeshInPml);
//! assert_eq!(warnings[0].code().as_str(), "mesh.in_pml");
//! assert_eq!(warnings[0].subject(), Some("gnd"));
//! assert_eq!(warnings[0].count(), 2);
//! assert_eq!(warnings[1].code(), WarningCode::BoxSnapped);
//! assert_eq!(warnings[1].subject(), Some("port_ut1"));
//! assert_eq!(warnings[1].to_string(), lines[2]);
//! ```
//!
//! # How the codes are assigned
//!
//! A warning's code and subject are read from its text ([`Warning::parse`]):
//! the code from the wording of the finding (the variants of
//! [`WarningCode`] say which; the more specific wording wins), the subject from the property name the line quotes in its head
//! (`probe "name": …`, `property "name" primitive #2 (Box): …`) or in the
//! location of a mesh-check line (`[property "name" primitive #2]`). The
//! lines come from this crate and from csx's discretisation; every wording
//! they use is covered by a test of this module, and a line with a wording
//! the table does not know gets [`WarningCode::Other`] (never an error), so
//! a consumer should treat `Other` as "a warning to show", not as a
//! failure.

use core::fmt;

/// What a [`Warning`] is about: a stable identifier of the kind of
/// finding ([`as_str`](Self::as_str) gives it as text).
///
/// The findings each code stands for are listed per variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum WarningCode {
    /// `mesh.in_pml`: a primitive extends into an absorbing layer (mesh
    /// check).
    MeshInPml,
    /// `mesh.pml_grading`: the cells of an absorbing layer are strongly
    /// graded (mesh check).
    MeshPmlGrading,
    /// `mesh.grading`: neighbour cell size ratios above the limit (mesh
    /// check).
    MeshGrading,
    /// `mesh.time_step`: tiny cells (or the arcs near a cylindrical axis)
    /// dominate the time step (mesh check).
    MeshTimeStep,
    /// `mesh.resolution`: cells coarser than the wavelength rule (mesh
    /// check).
    MeshResolution,
    /// `mesh.excitation_steps`: the excitation needs more steps than the
    /// run is limited to (mesh check).
    MeshExcitationSteps,
    /// `mesh.outside`: a primitive lies outside the mesh, or on no mesh
    /// line (mesh check).
    MeshOutside,
    /// `mesh.other`: any other line of the mesh check.
    Mesh,
    /// `box.snapped`: the corners of a probe, dump or plane-wave box were
    /// moved to the nearest mesh lines.
    BoxSnapped,
    /// `box.clamped`: a box reaching outside the mesh was clamped to it.
    BoxClamped,
    /// `conformal.stability`: the stability estimate of the conformal PEC
    /// cells failed.
    ConformalStability,
    /// `sheet.in_pml`: conducting-sheet edges inside an absorbing layer.
    SheetInPml,
    /// `sheet.other`: any other finding about conducting sheets (values
    /// without effect, edges approximated as PEC, edges on the axis).
    Sheet,
    /// `metal.lossy`: a finding about lossy metal
    /// ([`CompileOptions::with_lossy_metal`](crate::CompileOptions::with_lossy_metal)):
    /// a name that is no `Metal` property, edges that stay perfect
    /// conductors (zero-thickness metal, cylindrical grids, conformal cut
    /// cells), metal inside openEMS's PML.
    LossyMetal,
    /// `dispersive.ignored`: dispersive poles that were ignored.
    DispersiveIgnored,
    /// `lumped.stability`: a lumped element whose update may be unstable,
    /// or was replaced to stay stable.
    LumpedStability,
    /// `lumped.value`: a lumped element value or type that was not used.
    LumpedValue,
    /// `lumped.edges`: edges of a lumped element that were ignored,
    /// removed or coupled (PEC, absorbing layers, sheets, another
    /// element, multigrid zones), or an element without usable edges.
    LumpedEdges,
    /// `excitation.edges`: an excitation whose edges were not driven as
    /// drawn (no edges, on a lumped element, on a sheet, off the kept
    /// lines of a multigrid).
    ExcitationEdges,
    /// `plane_wave`: a finding about a plane-wave excitation.
    PlaneWave,
    /// `wire`: a finding about thin wires.
    Wire,
    /// `sar`: a finding about a SAR dump.
    Sar,
    /// `nf2ff`: a finding about a near-field to far-field box.
    Nf2ff,
    /// `absorbing_bc`: a finding about an `AbsorbingBC` property.
    AbsorbingBc,
    /// `probe.ignored`: a probe or dump that records nothing or was
    /// ignored (no normal direction, zero mode function, outside the
    /// mesh, both ends on one node).
    ProbeIgnored,
    /// `probe.other`: any other finding about a probe or dump.
    Probe,
    /// `primitive.not_discretised`: a primitive that cannot be put on the
    /// mesh as drawn (replaced by its bounding box, staircased, not
    /// axis-aligned).
    PrimitiveNotDiscretised,
    /// `primitive.off_mesh`: a line, sheet or curve that is not on mesh
    /// lines.
    PrimitiveOffMesh,
    /// `primitive.no_edges`: a primitive or property that covers no mesh
    /// edge (empty, a point, no primitive).
    PrimitiveNoEdges,
    /// `property.ignored`: a property the solver does not use (disabled,
    /// of an unknown type, a frequency-domain dump without frequencies).
    PropertyIgnored,
    /// `cylindrical`: a finding specific to cylindrical grids (alpha
    /// ranges, multigrid radii, faces that are no boundary).
    Cylindrical,
    /// `openems.file`: how an openEMS file was read where it is unclear
    /// or unusual (`as openEMS does`).
    OpenEmsFile,
    /// `other`: a line with a wording this table does not know.
    Other,
}

/// The wordings of each code, in matching order: a line gets the first
/// code of which it contains every part.
const WORDINGS: &[(&[&str], WarningCode)] = {
    use WarningCode::*;
    &[
        (&["mesh ", "extends into the PML"], MeshInPml),
        (&["mesh ", "PML cells are strongly graded"], MeshPmlGrading),
        (&["mesh ", "neighbour cell size ratio"], MeshGrading),
        (&["mesh ", "dominate the time step"], MeshTimeStep),
        (&["mesh ", "coarser than lambda/"], MeshResolution),
        (&["mesh ", "the excitation needs"], MeshExcitationSteps),
        (&["mesh ", "lies outside the mesh"], MeshOutside),
        (&["mesh ", "is not on a mesh line"], MeshOutside),
        (&["mesh warning: "], Mesh),
        (&["mesh error: "], Mesh),
        (&["snapped to the nearest mesh lines"], BoxSnapped),
        (&["clamped to the mesh"], BoxClamped),
        (&["conformal PEC: estimated stability"], ConformalStability),
        (&["conducting-sheet edge(s) inside the PML"], SheetInPml),
        (&["lossy metal"], LossyMetal),
        (&["sheet roughness of "], Sheet),
        (&["conducting sheets on a cylindrical grid"], Sheet),
        // Lumped elements and excitations on sheets are theirs.
        (&["lumped element ", "may become unstable"], LumpedStability),
        (&["lumped element ", "inductor update"], LumpedStability),
        (&["lumped element ", "is not positive"], LumpedValue),
        (&["lumped element ", "unknown LEtype"], LumpedValue),
        (
            &["lumped element ", "primitive is not supported"],
            OpenEmsFile,
        ),
        (&["lumped element over alpha"], Cylindrical),
        (&["lumped element covers no edge"], LumpedEdges),
        (&["lumped element "], LumpedEdges),
        (&["plane wave "], PlaneWave),
        (&["plane-wave "], PlaneWave),
        (&["excitation type ", "is not discretised"], ExcitationEdges),
        (&["excitation covers no grid edge"], ExcitationEdges),
        (&["excitation "], ExcitationEdges),
        (&["conformal PEC sheets"], Cylindrical),
        (&["ConductingSheet "], Sheet),
        (&["conducting-sheet edge(s)"], Sheet),
        (&["dispersive "], DispersiveIgnored),
        (&["Debye pole(s)"], DispersiveIgnored),
        (&["SAR dump "], Sar),
        (&["NF2FF box"], Nf2ff),
        (&["AbsorbingBC "], AbsorbingBc),
        (&["absorbing boundary on "], Cylindrical),
        (&["multigrid radi"], Cylindrical),
        (&["alpha range "], Cylindrical),
        (&["on the alpha = "], Cylindrical),
        (&["cylindrical mesh without"], OpenEmsFile),
        (&["boundary condition "], OpenEmsFile),
        (&["thin wire"], Wire),
        (&["wire "], Wire),
        (&["probe/dump ", "as openEMS does"], OpenEmsFile),
        (
            &["probe/dump ", "openEMS runs one per primitive"],
            OpenEmsFile,
        ),
        (&["probe ", "is not supported by openEMS"], OpenEmsFile),
        (&["probe ", "; ignored"], ProbeIgnored),
        (&["probe ", "it reads 0"], ProbeIgnored),
        (&["dump ", "; disabled"], ProbeIgnored),
        (&["probe "], Probe),
        (&["probe/dump "], Probe),
        (&["dump "], Probe),
        (&["replaced by its bounding box"], PrimitiveNotDiscretised),
        (&["cannot be discretised"], PrimitiveNotDiscretised),
        (&["(staircased)"], PrimitiveNotDiscretised),
        (&["not on mesh lines"], PrimitiveOffMesh),
        (&["is not on a mesh plane"], PrimitiveOffMesh),
        (&["covers no grid edge"], PrimitiveNoEdges),
        (&["cover no edge"], PrimitiveNoEdges),
        (&["empty primitive"], PrimitiveNoEdges),
        (&["primitive is a single point"], PrimitiveNoEdges),
        (&["no primitive; ignored"], PrimitiveNoEdges),
        (&["only the first of ", " primitives is used"], Probe),
        (&["property ", "; ignored"], PropertyIgnored),
    ]
};

impl WarningCode {
    /// The code as text: lower-case words joined by `.` and `_`, stable
    /// across releases (`mesh.in_pml`, `box.snapped`, …; see the
    /// variants).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MeshInPml => "mesh.in_pml",
            Self::MeshPmlGrading => "mesh.pml_grading",
            Self::MeshGrading => "mesh.grading",
            Self::MeshTimeStep => "mesh.time_step",
            Self::MeshResolution => "mesh.resolution",
            Self::MeshExcitationSteps => "mesh.excitation_steps",
            Self::MeshOutside => "mesh.outside",
            Self::Mesh => "mesh.other",
            Self::BoxSnapped => "box.snapped",
            Self::BoxClamped => "box.clamped",
            Self::ConformalStability => "conformal.stability",
            Self::SheetInPml => "sheet.in_pml",
            Self::Sheet => "sheet.other",
            Self::LossyMetal => "metal.lossy",
            Self::DispersiveIgnored => "dispersive.ignored",
            Self::LumpedStability => "lumped.stability",
            Self::LumpedValue => "lumped.value",
            Self::LumpedEdges => "lumped.edges",
            Self::ExcitationEdges => "excitation.edges",
            Self::PlaneWave => "plane_wave",
            Self::Wire => "wire",
            Self::Sar => "sar",
            Self::Nf2ff => "nf2ff",
            Self::AbsorbingBc => "absorbing_bc",
            Self::ProbeIgnored => "probe.ignored",
            Self::Probe => "probe.other",
            Self::PrimitiveNotDiscretised => "primitive.not_discretised",
            Self::PrimitiveOffMesh => "primitive.off_mesh",
            Self::PrimitiveNoEdges => "primitive.no_edges",
            Self::PropertyIgnored => "property.ignored",
            Self::Cylindrical => "cylindrical",
            Self::OpenEmsFile => "openems.file",
            Self::Other => "other",
        }
    }

    /// Whether the code is one of the mesh check's (`mesh.*`).
    pub fn is_mesh(self) -> bool {
        self <= Self::Mesh
    }

    /// The code of a warning line (see the [module docs](self)).
    fn of(line: &str) -> Self {
        WORDINGS
            .iter()
            .find(|(parts, _)| {
                // The parts in order, each after the one before.
                let mut rest = line;
                parts.iter().all(|p| match rest.find(p) {
                    Some(i) => {
                        rest = &rest[i + p.len()..];
                        true
                    }
                    None => false,
                })
            })
            .map_or(Self::Other, |&(_, code)| code)
    }
}

impl fmt::Display for WarningCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One warning with its [code](WarningCode), the property it is about
/// and how many lines it stands for; see the [module docs](self).
///
/// [`Display`](fmt::Display) gives the text of the (first) line, followed
/// by `(N warnings of this kind)` when the entry stands for several.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Warning {
    code: WarningCode,
    subject: Option<String>,
    primitive: Option<usize>,
    count: usize,
    message: String,
}

impl Warning {
    /// Reads one warning line (see the [module docs](self) for how the
    /// code and the subject are found). Never fails: an unknown wording
    /// gives [`WarningCode::Other`] without a subject.
    pub fn parse(line: &str) -> Self {
        let (subject, primitive) = match subject_of(line) {
            Some((s, p)) => (Some(s), p),
            None => (None, None),
        };
        Self {
            code: WarningCode::of(line),
            subject,
            primitive,
            count: 1,
            message: line.to_string(),
        }
    }

    /// The kind of finding.
    pub fn code(&self) -> WarningCode {
        self.code
    }

    /// The name of the model property (material, port element, probe,
    /// excitation, …) the warning is about, if its line names one.
    pub fn subject(&self) -> Option<&str> {
        self.subject.as_deref()
    }

    /// The index of the primitive within the [subject](Self::subject)
    /// property, if the line names one and the entry stands for that line
    /// alone.
    pub fn primitive(&self) -> Option<usize> {
        self.primitive
    }

    /// How many warning lines this entry stands for (1 unless it came
    /// from [`collect`]).
    pub fn count(&self) -> usize {
        self.count
    }

    /// The text of the line (of the first one for an entry of several).
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)?;
        if self.count > 1 {
            write!(f, " ({} warnings of this kind)", self.count)?;
        }
        Ok(())
    }
}

/// The warnings of `lines`, one entry per [code](WarningCode) and
/// [subject](Warning::subject), in the order of their first lines: lines
/// with the same code about the same property are one entry with their
/// [count](Warning::count) and the first line's text. Lines that name no
/// property are merged only when their text is the same.
pub fn collect<S: AsRef<str>>(lines: &[S]) -> Vec<Warning> {
    let mut out: Vec<Warning> = Vec::new();
    for line in lines {
        let w = Warning::parse(line.as_ref());
        let same = |o: &&mut Warning| {
            o.code == w.code
                && o.subject == w.subject
                && (w.subject.is_some() || o.message == w.message)
        };
        match out.iter_mut().find(same) {
            Some(o) => {
                o.count += 1;
                o.primitive = None;
            }
            None => out.push(w),
        }
    }
    out
}

/// The property a line is about and the primitive index it names: the
/// location of a mesh-check line, else the name quoted in the line's
/// head.
fn subject_of(line: &str) -> Option<(String, Option<usize>)> {
    // `… [property "name" primitive #3]` (mesh check).
    if line.ends_with(']')
        && let Some(i) = line.rfind("[property \"")
        && let Some((name, rest)) = quoted(&line[i + "[property ".len()..])
    {
        return Some((name, primitive_index(rest)));
    }
    // `<what> "name"…: …`: the quoted name before the first `: `, after at
    // most three words.
    let line = line
        .strip_prefix("mesh warning: ")
        .or_else(|| line.strip_prefix("mesh error: "))
        .unwrap_or(line);
    let q = line.find('"')?;
    let head = &line[..q];
    if head.contains(": ") || head.is_empty() || !head.ends_with(' ') {
        return None;
    }
    if head.split_whitespace().count() > 3 || head.contains('=') {
        return None;
    }
    let (name, rest) = quoted(&line[q..])?;
    Some((name, primitive_index(rest)))
}

/// A leading Rust-debug-quoted string of `s` and what follows it.
fn quoted(s: &str) -> Option<(String, &str)> {
    let mut chars = s.strip_prefix('"')?.char_indices();
    let mut name = String::new();
    while let Some((i, c)) = chars.next() {
        match c {
            '"' => return Some((name, &s[i + 2..])),
            '\\' => match chars.next()?.1 {
                'n' => name.push('\n'),
                't' => name.push('\t'),
                'r' => name.push('\r'),
                '0' => name.push('\0'),
                // `\"`, `\\`, `\'`; other escapes (`\u{…}`) are kept as
                // written.
                c @ ('"' | '\\' | '\'') => name.push(c),
                c => {
                    name.push('\\');
                    name.push(c);
                }
            },
            c => name.push(c),
        }
    }
    None
}

/// The `N` of a leading ` primitive #N`.
fn primitive_index(rest: &str) -> Option<usize> {
    let digits = rest.strip_prefix(" primitive #")?;
    let end = digits
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(digits.len());
    digits[..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use WarningCode::*;

    /// Every wording the crate and csx write, with its code and subject.
    /// (The texts are those of the `warnings.push` sites; a site whose
    /// text changes must change here too.)
    const LINES: &[(&str, WarningCode, Option<&str>)] = &[
        // Mesh check.
        (
            r#"mesh warning: DebyeMaterial "fr4" extends into the PML of face(s) xmin, xmax [property "fr4" primitive #0]"#,
            MeshInPml,
            Some("fr4"),
        ),
        (
            "mesh warning: PML cells are strongly graded (sizes 1.000e-4 .. 5.000e-4 m, ratio 5.00); use roughly uniform cells in the absorbing layer [xmin face]",
            MeshPmlGrading,
            None,
        ),
        (
            "mesh warning: 4 x neighbour cell size ratio(s) above 1.5; worst 2.00 [x cell 3 (x = 1 .. 2)]",
            MeshGrading,
            None,
        ),
        (
            "mesh warning: dt is 0.1 of the typical step: tiny cells dominate the time step [z cell 4 (z = 1 .. 1.01)]",
            MeshTimeStep,
            None,
        ),
        (
            "mesh warning: dt is small: the arcs near the axis dominate the time step (fewer alpha lines, a larger rho_min or multigrid near the axis help)",
            MeshTimeStep,
            None,
        ),
        (
            "mesh warning: 3 x cell(s) coarser than lambda/15 at f_max = 1.0000e10 Hz (0 below lambda/10); worst: 2.0000 (drawing units) = lambda/12.0 with n = 1.000, needs <= 1.5000 [x cell 0 (x = 0 .. 2)]",
            MeshResolution,
            None,
        ),
        (
            "mesh error: 3 x cell(s) coarser than lambda/15 at f_max = 1.0000e10 Hz (2 below lambda/10); worst: 4.0000 (drawing units) = lambda/6.0 with n = 1.000, needs <= 1.5000 [x cell 0 (x = 0 .. 4)]",
            MeshResolution,
            None,
        ),
        (
            "mesh warning: the excitation needs 500 steps but the run is limited to 100 steps",
            MeshExcitationSteps,
            None,
        ),
        (
            r#"mesh warning: Metal Box "plate" lies outside the mesh [property "plate" primitive #1]"#,
            MeshOutside,
            Some("plate"),
        ),
        (
            r#"mesh error: Metal sheet "plate" at z = 0.5 is not on a mesh line (nearest 1) and will be ignored; add a mesh line there [property "plate" primitive #0]"#,
            MeshOutside,
            Some("plate"),
        ),
        (
            "mesh error: PML needs 8 cells but the x axis has only 4 cells [xmin face]",
            Mesh,
            None,
        ),
        // Boxes.
        (
            r#"probe "port_ut1": box corners snapped to the nearest mesh lines"#,
            BoxSnapped,
            Some("port_ut1"),
        ),
        (
            r#"dump "Et": box extends outside the mesh; clamped to the mesh"#,
            BoxClamped,
            Some("Et"),
        ),
        (
            "multigrid radius 0.5 snapped to the rho line 0.4 (index 4)",
            Cylindrical,
            None,
        ),
        // Conformal cells, sheets, poles.
        (
            "conformal PEC: estimated stability factor 1.0200 (> 1 is unstable) with merge threshold 0.100; reduce the time step factor",
            ConformalStability,
            None,
        ),
        (
            "3232 conducting-sheet edge(s) inside the PML do not get the PML correction; keep sheets out of the absorbing layers",
            SheetInPml,
            None,
        ),
        (
            r#"lossy metal "copper": no property of that name; ignored"#,
            LossyMetal,
            Some("copper"),
        ),
        (
            "318 edge(s) of lossy metal are zero-thickness metal or wires and stay perfect conductors; use a ConductingSheet for a lossy sheet",
            LossyMetal,
            None,
        ),
        (
            "lossy metal with conformal PEC: the surface impedance is on the edges wholly in the metal; the cut cells along its outline stay perfectly conducting",
            LossyMetal,
            None,
        ),
        (
            r#"sheet roughness of "strip" ignored: it needs SheetModel::SurfaceImpedance on a Cartesian grid"#,
            Sheet,
            Some("strip"),
        ),
        (
            r#"ConductingSheet "cu": conductivity 0 S/m and thickness 0 m must be > 0; the sheet has no effect"#,
            Sheet,
            Some("cu"),
        ),
        (
            r#"ConductingSheet "cu": 4 edge(s) not on a planar sheet (line, curve or volume primitive) approximated as PEC"#,
            Sheet,
            Some("cu"),
        ),
        (
            "2 conducting-sheet edge(s) on the axis are left as ordinary edges",
            Sheet,
            None,
        ),
        (
            "4 conducting-sheet edge(s) in the first mesh cell of their axis made PEC (openEMS's rule)",
            Sheet,
            None,
        ),
        (
            "conformal PEC sheets: 3 cut edge(s) inside the multigrid zones; the sheets are not modelled (keep sheets outside the zones)",
            Cylindrical,
            None,
        ),
        (
            "dispersive materials: 2 edge pole(s) with negative or non-finite parameters ignored",
            DispersiveIgnored,
            None,
        ),
        (
            "12 dispersive pole(s) on edges inside a PML layer ignored (openEMS's layer replaces their update; the static medium remains)",
            DispersiveIgnored,
            None,
        ),
        (
            "8 Debye pole(s) with a relaxation time below 3 time steps ignored (openEMS's rule; the static medium remains)",
            DispersiveIgnored,
            None,
        ),
        // Lumped elements.
        (
            r#"lumped element "R1": openEMS's limit on the edge capacitance of a parallel R/L element keeps 50 % of an edge's capacitance; the explicit update may become unstable (as it does in openEMS)"#,
            LumpedStability,
            Some("R1"),
        ),
        (
            r#"lumped element "L1": inductance too small for openEMS's explicit inductor update (dt²/(L·C) = 1.200 per edge > 1, where openEMS becomes unstable); the implicit trapezoidal update is used"#,
            LumpedStability,
            Some("L1"),
        ),
        (
            r#"lumped element "R1": R = -1 is not positive; treated as absent"#,
            LumpedValue,
            Some("R1"),
        ),
        (
            r#"lumped element "R1": unknown LEtype 7; parallel assumed"#,
            LumpedValue,
            Some("R1"),
        ),
        (
            r#"lumped element "port_resist_1": edge z:[1, 2, 3] is PEC (shorted); ignored"#,
            LumpedEdges,
            Some("port_resist_1"),
        ),
        (
            r#"lumped element "R1": 2 of its 4 edge(s) lie inside a PML layer and are ignored there (openEMS's layer replaces their update)"#,
            LumpedEdges,
            Some("R1"),
        ),
        (
            r#"lumped element "R1": no usable edges; ignored"#,
            LumpedEdges,
            Some("R1"),
        ),
        (
            r#"property "R1" primitive #0 (Box): lumped element covers no edge along axis 2"#,
            LumpedEdges,
            Some("R1"),
        ),
        (
            r#"lumped element "R1": a Cylinder primitive is not supported by openEMS (boxes only); skipped, as openEMS does"#,
            OpenEmsFile,
            Some("R1"),
        ),
        // Excitations.
        (
            r#"excitation "port_excite_1": no edges; ignored"#,
            ExcitationEdges,
            Some("port_excite_1"),
        ),
        (
            r#"excitation "src": hard E source on 3 edge(s) of a lumped element; the source overwrites the edge voltage and the element has no effect there"#,
            ExcitationEdges,
            Some("src"),
        ),
        (
            r#"property "src" primitive #2 (Box): excitation covers no grid edge"#,
            ExcitationEdges,
            Some("src"),
        ),
        (
            r#"property "src" primitive #0 (Box): excitation type 7 is not discretised"#,
            ExcitationEdges,
            Some("src"),
        ),
        (
            r#"plane wave "pw": no primitive; ignored"#,
            PlaneWave,
            Some("pw"),
        ),
        (
            "2 plane-wave excitation(s) ignored: plane waves need a Cartesian grid (openEMS ignores them on cylindrical grids too)",
            PlaneWave,
            None,
        ),
        // Wires, SAR, NF2FF, absorbing sheets.
        (
            r#"wire "w" primitive #0 segment 1: not on mesh lines; staircase without thin-wire correction"#,
            Wire,
            Some("w"),
        ),
        (
            "thin wires meet at a node (junction or radius change); the thin-wire model assumes isolated straight wires there",
            Wire,
            None,
        ),
        (
            r#"SAR dump "sar": no material with a density in its box; the SAR is zero"#,
            Sar,
            Some("sar"),
        ),
        (
            "NF2FF box reaches into the PML of XMin; keep it inside the interior",
            Nf2ff,
            None,
        ),
        (
            r#"AbsorbingBC "abc": no primitive; ignored"#,
            AbsorbingBc,
            Some("abc"),
        ),
        // Probes and dumps.
        (
            r#"voltage probe "ut": start and stop snap to the same node; it reads 0"#,
            ProbeIgnored,
            Some("ut"),
        ),
        (
            r#"current probe "it": no normal direction (set NormDir or use a box that is flat along one axis); ignored"#,
            ProbeIgnored,
            Some("it"),
        ),
        (
            r#"waveguide probe "wg": no ModeFunctionX/Y/Z attributes; it reads 0"#,
            ProbeIgnored,
            Some("wg"),
        ),
        (
            r#"waveguide probe "wg": box is not flat along its normal z; the plane at its start is used"#,
            Probe,
            Some("wg"),
        ),
        (
            r#"dump "Et": box is outside the mesh; disabled"#,
            ProbeIgnored,
            Some("Et"),
        ),
        (
            r#"probe/dump "Et": OverSampling = "x" is not a positive integer; ignored (the <FDTD> value applies)"#,
            Probe,
            Some("Et"),
        ),
        (
            r#"probe/dump "Et" has 3 primitives: openEMS runs one per primitive, named Et_0 … Et_2"#,
            OpenEmsFile,
            Some("Et"),
        ),
        (
            r#"voltage probe "ut": a Cylinder primitive is not supported by openEMS (a Box is needed); the probe is skipped, as openEMS does"#,
            OpenEmsFile,
            Some("ut"),
        ),
        // Primitives (csx's discretisation and the property checks).
        (
            r#"probe "p": Cylinder primitive replaced by its bounding box"#,
            Probe,
            Some("p"),
        ),
        (
            r#"property "cu" primitive #4 (Polygon): non axis-aligned sheet cannot be discretised"#,
            PrimitiveNotDiscretised,
            Some("cu"),
        ),
        (
            r#"property "cu" primitive #4 (Curve): curve vertices are not on mesh nodes (staircased)"#,
            PrimitiveNotDiscretised,
            Some("cu"),
        ),
        (
            r#"property "cu" primitive #1 (Box): line not on mesh lines"#,
            PrimitiveOffMesh,
            Some("cu"),
        ),
        (
            r#"property "cu" primitive #1 (Box): sheet at 0.5 is not on a mesh plane (axis 2)"#,
            PrimitiveOffMesh,
            Some("cu"),
        ),
        (
            r#"property "cu" primitive #7 (Box): primitive covers no grid edge"#,
            PrimitiveNoEdges,
            Some("cu"),
        ),
        (
            r#"property "cu" primitive #7 (Box): point primitives cover no edge"#,
            PrimitiveNoEdges,
            Some("cu"),
        ),
        (
            r#"property "cu" primitive #7 (Box): empty primitive"#,
            PrimitiveNoEdges,
            Some("cu"),
        ),
        (
            r#"property "cu" (Metal): no primitive; ignored"#,
            PrimitiveNoEdges,
            Some("cu"),
        ),
        (
            r#"property "ex" (Excitation): disabled; ignored"#,
            PropertyIgnored,
            Some("ex"),
        ),
        (
            r#"property "p" (ProbeBox): only the first of 3 primitives is used"#,
            Probe,
            Some("p"),
        ),
        // Cylindrical grids and openEMS files.
        (
            "absorbing boundary on xmin ignored: on this cylindrical grid it is the axis",
            Cylindrical,
            None,
        ),
        (
            "multigrid radii [0.1] ignored: multigrid needs a cylindrical grid",
            Cylindrical,
            None,
        ),
        (
            r#"boundary condition xmin="pec" is unknown to openEMS (names are case sensitive: PEC, PMC, MUR, PML_<n>); set to PEC, as openEMS does"#,
            OpenEmsFile,
            None,
        ),
        (
            "cylindrical mesh without CylinderCoords=\"1\" on <FDTD>: run as a Cartesian mesh (rho, alpha, z and the primitives' coordinates as x, y, z), as openEMS does",
            OpenEmsFile,
            None,
        ),
        ("something new", Other, None),
    ];

    #[test]
    fn every_wording_has_its_code_and_subject() {
        for &(line, code, subject) in LINES {
            let w = Warning::parse(line);
            assert_eq!(w.code(), code, "{line}");
            assert_eq!(w.subject(), subject, "{line}");
            assert_eq!(w.count(), 1);
            assert_eq!(w.to_string(), line);
        }
    }

    #[test]
    fn codes_are_distinct_text() {
        let mut seen = std::collections::BTreeSet::new();
        for &(_, code) in WORDINGS {
            seen.insert((code.as_str(), code));
        }
        seen.insert((Other.as_str(), Other));
        let texts: std::collections::BTreeSet<_> = seen.iter().map(|s| s.0).collect();
        assert_eq!(texts.len(), seen.len());
        assert!(MeshInPml.is_mesh() && Mesh.is_mesh() && !BoxSnapped.is_mesh());
        assert_eq!(MeshGrading.to_string(), "mesh.grading");
    }

    #[test]
    fn primitive_indices_and_escaped_names() {
        let w = Warning::parse(r#"property "a \"b\" c" primitive #12 (Box): empty primitive"#);
        assert_eq!(w.subject(), Some(r#"a "b" c"#));
        assert_eq!(w.primitive(), Some(12));
        let w = Warning::parse(
            r#"mesh warning: Metal "m" extends into the PML of face(s) zmin [property "m" primitive #5]"#,
        );
        assert_eq!((w.subject(), w.primitive()), (Some("m"), Some(5)));
        // A quoted value that is no property name.
        let w =
            Warning::parse(r#"waveguide probe "wg": invalid ModeFunctionX "sin(": bad; ignored"#);
        assert_eq!(w.subject(), Some("wg"));
        assert_eq!(Warning::parse("\"").subject(), None);
    }

    #[test]
    fn collect_merges_per_code_and_subject() {
        let lines: Vec<String> = (0..19)
            .map(|k| {
                format!(
                    "mesh warning: Metal \"cu\" extends into the PML of face(s) xmin \
                     [property \"cu\" primitive #{k}]"
                )
            })
            .chain([
                r#"mesh warning: Material "fr4" extends into the PML of face(s) xmin [property "fr4" primitive #0]"#.to_string(),
                r#"property "cu" primitive #3 (Box): empty primitive"#.to_string(),
                "thin wires meet".to_string(),
                "something else".to_string(),
                "something else".to_string(),
                "something new".to_string(),
            ])
            .collect();
        let w = collect(&lines);
        let got: Vec<_> = w
            .iter()
            .map(|w| (w.code(), w.subject(), w.primitive(), w.count()))
            .collect();
        assert_eq!(
            got,
            [
                (MeshInPml, Some("cu"), None, 19),
                (MeshInPml, Some("fr4"), Some(0), 1),
                (PrimitiveNoEdges, Some("cu"), Some(3), 1),
                (Wire, None, None, 1),
                (Other, None, None, 2),
                (Other, None, None, 1),
            ]
        );
        assert_eq!(
            w[0].to_string(),
            format!("{} (19 warnings of this kind)", lines[0])
        );
        assert_eq!(w[0].message(), lines[0]);
    }
}
