//! The debug files of openEMS's `--debug-material`, `--debug-operator`,
//! `--debug-PEC` and `--debug-boxes` options.
//!
//! Names and layouts as observed with openEMS v0.37 (black box); the
//! *values* are ems's own (its averaged edge materials, its update
//! coefficients), which is what these files are for: inspecting what the
//! solver actually runs.
//!
//! * `material_dump.vtr` ([`material_dump`]): VTK XML `RectilinearGrid` on
//!   the mesh lines (metres), `f32` 3-vector point arrays `epsilon`,
//!   `mue`, `kappa`, `sigma`. Component `a` at node `p` is the value of
//!   the edge along `a` starting at `p` (`ε_r`, `κ` of the primary edge)
//!   or of the dual edge along `a` through `p` (`μ_r`, `σ_m`), i.e. the
//!   averaged edge materials of [`EdgeMaterials`]. Edges outside the
//!   mesh (a primary edge from the last line along `a`, a dual edge past
//!   the last line of a transverse axis) report the edge one line inside,
//!   as openEMS does. openEMS's values agree for `ε`, `κ` and `μ`; ems's
//!   `σ_m` average differs at interfaces (openEMS reports 0 for a dual
//!   edge half in a magnetically lossy medium; see [`EdgeMaterials`] for
//!   ems's rule).
//! * `operator_dump.vtr` ([`operator_dump`]): the same grid with the
//!   arrays `exc_volt` (the voltage amplitude `Excite·w·Δ` \[V\] of the E
//!   sources on each edge), `vv`, `vi` (E update: `u ← vv·u + vi·Σ i`),
//!   `iv`, `ii` (H update: `i ← ii·i + iv·Σ u`), in the integral form of
//!   design §5.2 (ems's `ca`, `cb`, `db`, `da`; [`Coefficients`]). openEMS
//!   folds its lumped elements into `vv`/`vi` and zeroes the tangential
//!   E coefficients on every outer face; ems keeps lumped elements,
//!   boundaries other than PEC and the other extensions as separate
//!   operators, so its arrays show the medium only.
//! * `PEC_dump.vtp` ([`pec_dump`]): VTK XML `PolyData`, one line per PEC
//!   edge of the metal in the model (not the PEC boundaries), points in
//!   metres (`f32`), merged, in order of first use; edges visited node by
//!   node (`x` outermost, `z` innermost), each node's `x`, `y`, `z` edges
//!   in that order. Appended, base64-encoded zlib data, `Int64` cell
//!   arrays.
//! * `box_dump_<name>.vtk` ([`box_dump`]): legacy ASCII VTK `POLYDATA`,
//!   the 8 corners and 6 faces of a probe's or dump's snapped box
//!   (metres). openEMS writes one per dump box (named after the
//!   property) and one per current or waveguide probe, all named
//!   `box_dump_.vtk` (each overwrites the previous one), none for
//!   voltage and field probes.
//!
//! openEMS's console announces `PEC_dump.vtk`, `material_dump.vtk` and
//! `operator_dump.vtk` but writes the extensions above.
//!
//! # Cylindrical grids
//!
//! The same arrays and components, in `(ρ, α, z)` (black box, fixtures
//! `cli_debug_cyl`, `cli_debug_cyl_sector`):
//!
//! * the material and operator dumps are VTK XML `StructuredGrid`s
//!   (`material_dump.vts`, `operator_dump.vts`) whose points are the
//!   Cartesian positions `(ρ·cos α, ρ·sin α, z)` of the mesh nodes (`f32`
//!   of the `f64` products); on a closed grid (α span 2π) the last α
//!   line is placed at exactly `α_0 + 2π`, and openEMS adds one α line,
//!   `α_1 + 2π`, past it. ems shows the physical
//!   edge on every point: on a closed grid the `2π` line and the extra
//!   line repeat the `α = 0` and `α_1` lines, and an edge stored in
//!   several copies (the seam of a closed grid, the axis z edge) shows
//!   the merged edge (ε, κ weighted by the copies' dual-face parts; `vv`,
//!   `vi` from `1/cb = Σ 1/cb_j`, `ca/cb = Σ ca_j/cb_j`). openEMS instead
//!   evaluates materials at the literal angles (the extra line and the
//!   halves past `2π` or below 0 are background), updates the seam's ρ
//!   and z edges on the `2π` line only (`vv` = `vi` = 0 on `α = 0` and on
//!   the extra line) and leaves the axis z edge out of the dump (0);
//! * `PEC_dump.vtp` has Cartesian points as well, and visits the edges
//!   with z outermost, then ρ, α innermost; a seam edge of a closed grid
//!   is drawn once, on the `2π` line; axis z edges are not drawn (openEMS
//!   draws none for a metal rod on the axis).
//!
//! An empty PEC dump has openEMS's range `1e+299` / `-1e+299`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::dump::{
    f32_bytes, norm_range, printf_g11, vtk_binary, vtk_rectilinear, vtk_structured_cyl,
};
use crate::engine::cylindrical::e_edge_copies;
use crate::engine::{Coefficients, EdgeMaterials, Grid};
use crate::{Axis, Error, Result};

/// File name of [`material_dump`] on Cartesian grids
/// ([`material_dump_file`]).
pub const MATERIAL_DUMP_FILE: &str = "material_dump.vtr";
/// File name of [`operator_dump`] on Cartesian grids
/// ([`operator_dump_file`]).
pub const OPERATOR_DUMP_FILE: &str = "operator_dump.vtr";
/// File name of [`pec_dump`].
pub const PEC_DUMP_FILE: &str = "PEC_dump.vtp";

/// File name of [`material_dump`] for `grid`: [`MATERIAL_DUMP_FILE`], or
/// `material_dump.vts` on cylindrical grids.
pub fn material_dump_file(grid: &Grid) -> &'static str {
    if grid.is_cylindrical() {
        "material_dump.vts"
    } else {
        MATERIAL_DUMP_FILE
    }
}

/// File name of [`operator_dump`] for `grid`: [`OPERATOR_DUMP_FILE`], or
/// `operator_dump.vts` on cylindrical grids.
pub fn operator_dump_file(grid: &Grid) -> &'static str {
    if grid.is_cylindrical() {
        "operator_dump.vts"
    } else {
        OPERATOR_DUMP_FILE
    }
}

fn write(path: &Path, text: &str) -> Result<PathBuf> {
    std::fs::write(path, text).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    Ok(path.to_path_buf())
}

/// The point lattice of the material and operator dumps: the mesh lines,
/// plus, on a closed cylindrical grid, the image `α_1 + 2π` of the
/// second α line (see the [module docs](self)).
struct Layout<'g> {
    grid: &'g Grid,
    /// Closed cylindrical grids: the number of distinct α lines (the
    /// `2π` line is the `α = 0` line).
    period: Option<usize>,
    coords: [Vec<f64>; 3],
}

/// The mesh lines as openEMS places the dump points: on a closed grid
/// the last α line is exactly `α_0 + 2π`.
fn point_lines(grid: &Grid) -> [Vec<f64>; 3] {
    let mut coords = Axis::ALL.map(|a| grid.lines(a).to_vec());
    if grid.is_closed() {
        let a = &mut coords[1];
        let n = a.len();
        a[n - 1] = a[0] + core::f64::consts::TAU;
    }
    coords
}

impl<'g> Layout<'g> {
    fn new(grid: &'g Grid) -> Self {
        let mut coords = point_lines(grid);
        let period = grid.is_closed().then(|| {
            let a = &mut coords[1];
            let n = a.len();
            a.push(a[1] + core::f64::consts::TAU);
            n - 1
        });
        Self {
            grid,
            period,
            coords,
        }
    }

    fn dims(&self) -> [usize; 3] {
        self.coords.each_ref().map(Vec::len)
    }

    /// The engine node shown at dump point `p`: itself, but on closed
    /// grids the α index modulo the period (the `2π` line and the extra
    /// line show the `α = 0` and `α_1` lines).
    fn node(&self, mut p: [usize; 3]) -> [usize; 3] {
        if let Some(n) = self.period {
            p[1] %= n;
        }
        p
    }

    /// Per-point 3-vectors in VTK point order (x fastest) of a function
    /// of the axis and the engine node.
    fn vectors(&self, f: impl Fn(Axis, [usize; 3]) -> f64) -> Vec<[f32; 3]> {
        let n = self.dims();
        let mut out = Vec::with_capacity(n.iter().product());
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let q = self.node([i, j, k]);
                    out.push(Axis::ALL.map(|a| f(a, q) as f32));
                }
            }
        }
        out
    }

    fn file(&self, arrays: &[(&str, &[[f32; 3]])]) -> String {
        let c = [
            &self.coords[0][..],
            &self.coords[1][..],
            &self.coords[2][..],
        ];
        if self.grid.is_cylindrical() {
            vtk_structured_cyl(c, arrays)
        } else {
            vtk_rectilinear(c, arrays)
        }
    }

    /// The stored copies of the physical E edge along `axis` at engine
    /// node `p`: the α copies of an axis z edge and both copies of an
    /// edge on the seam of a closed grid, else the edge itself.
    fn copies(&self, axis: Axis, p: [usize; 3]) -> Vec<[usize; 3]> {
        if self.grid.is_cylindrical() {
            e_edge_copies(self.grid, axis, p)
        } else {
            vec![p]
        }
    }
}

/// The text of `material_dump.vtr` (`.vts` on cylindrical grids; see the
/// [module docs](self)).
///
/// # Errors
/// [`Error::Invalid`] if `mats` does not match `grid`.
pub fn material_dump(grid: &Grid, mats: &EdgeMaterials) -> Result<String> {
    if mats.nodes() != grid.nodes() {
        return Err(Error::Invalid(
            "edge materials do not match the grid".into(),
        ));
    }
    let lay = Layout::new(grid);
    // Edges outside the mesh report the edge one line inside (openEMS).
    let n = grid.nodes();
    let clamp = |p: [usize; 3], axes: &[usize]| -> [usize; 3] {
        let mut q = p;
        for &d in axes {
            if n[d] > 1 && q[d] + 1 >= n[d] {
                q[d] = n[d] - 2;
            }
        }
        q
    };
    let primary = |a: Axis, p: [usize; 3]| clamp(p, &[a.index()]);
    let dual = |a: Axis, p: [usize; 3]| {
        let (b, c) = a.others();
        grid.index(clamp(p, &[b.index(), c.index()]))
    };
    // A primary edge stored in several copies (cylindrical axis and seam)
    // shows the physical edge: the copies' values weighted by their parts
    // of the dual face.
    let edge = |vals: &[f64], a: Axis, p: [usize; 3]| -> f64 {
        let p = primary(a, p);
        let copies = lay.copies(a, p);
        if copies.len() == 1 {
            return vals[grid.index(p)];
        }
        let (mut num, mut den) = (0.0, 0.0);
        for c in copies {
            let w = grid.dual_face_area(a, c);
            num += w * vals[grid.index(c)];
            den += w;
        }
        if den > 0.0 {
            num / den
        } else {
            vals[grid.index(p)]
        }
    };
    let eps = lay.vectors(|a, p| edge(mats.eps_r(a), a, p));
    let mu = lay.vectors(|a, p| mats.mu_r(a)[dual(a, p)]);
    let kappa = lay.vectors(|a, p| edge(mats.kappa(a), a, p));
    let sigma = lay.vectors(|a, p| mats.sigma_m(a)[dual(a, p)]);
    Ok(lay.file(&[
        ("epsilon", &eps),
        ("mue", &mu),
        ("kappa", &kappa),
        ("sigma", &sigma),
    ]))
}

/// The text of `operator_dump.vtr` (`.vts` on cylindrical grids; see the
/// [module docs](self)): `coeffs` the engine's update coefficients,
/// `exc_volt` the E source voltage amplitudes per edge `(axis, engine
/// node, volts)` (summed where several sources share an edge).
///
/// # Errors
/// [`Error::Invalid`] if `coeffs` does not match `grid`.
pub fn operator_dump(
    grid: &Grid,
    coeffs: &Coefficients,
    exc_volt: &[(Axis, [usize; 3], f64)],
) -> Result<String> {
    if coeffs.nodes() != grid.nodes() {
        return Err(Error::Invalid("coefficients do not match the grid".into()));
    }
    let lay = Layout::new(grid);
    let mut exc = [0, 1, 2].map(|_| vec![0.0f64; grid.num_nodes()]);
    for &(a, node, v) in exc_volt {
        exc[a.index()][grid.index(node)] += v;
    }
    // The update of the physical edge of stored copies: `1/cb = Σ 1/cb_j`,
    // `ca/cb = Σ ca_j/cb_j` (the merge of design §5.6.1); 0 if a copy is
    // not updated (PEC).
    let merged = |a: Axis, p: [usize; 3]| -> (f64, f64) {
        let (ca, cb) = (coeffs.ca(a), coeffs.cb(a));
        let copies = lay.copies(a, p);
        if copies.len() == 1 {
            let i = grid.index(p);
            return (f64::from(ca[i]), f64::from(cb[i]));
        }
        let (mut w, mut wa) = (0.0, 0.0);
        for c in copies {
            let i = grid.index(c);
            let b = f64::from(cb[i]);
            if b == 0.0 {
                return (0.0, 0.0);
            }
            w += 1.0 / b;
            wa += f64::from(ca[i]) / b;
        }
        (wa / w, 1.0 / w)
    };
    let exc = lay.vectors(|a, p| exc[a.index()][grid.index(p)]);
    let vv = lay.vectors(|a, p| merged(a, p).0);
    let vi = lay.vectors(|a, p| merged(a, p).1);
    let iv = lay.vectors(|a, p| f64::from(coeffs.db(a)[grid.index(p)]));
    let ii = lay.vectors(|a, p| f64::from(coeffs.da(a)[grid.index(p)]));
    Ok(lay.file(&[
        ("exc_volt", &exc),
        ("vv", &vv),
        ("vi", &vi),
        ("iv", &iv),
        ("ii", &ii),
    ]))
}

/// `name="value"` padded like VTK's appended-format headers (value and
/// closing quote in 21 columns).
fn attr(name: &str, value: &str) -> String {
    format!(
        "{name}=\"{value}\"{}",
        " ".repeat(20usize.saturating_sub(value.len()))
    )
}

/// The text of `PEC_dump.vtp` (see the [module docs](self)).
///
/// # Errors
/// [`Error::Invalid`] if `mats` does not match `grid`.
pub fn pec_dump(grid: &Grid, mats: &EdgeMaterials) -> Result<String> {
    if mats.nodes() != grid.nodes() {
        return Err(Error::Invalid(
            "edge materials do not match the grid".into(),
        ));
    }
    let n = grid.nodes();
    let cyl = grid.is_cylindrical();
    let mut ids = std::collections::HashMap::new();
    let mut points: Vec<[f32; 3]> = Vec::new();
    let point_lines = point_lines(grid);
    let mut id = |p: [usize; 3]| -> i64 {
        *ids.entry(p).or_insert_with(|| {
            let [x, y, z] = Axis::ALL.map(|a| point_lines[a.index()][p[a.index()]]);
            points.push(if cyl {
                let (s, c) = y.sin_cos();
                [(x * c) as f32, (x * s) as f32, z as f32]
            } else {
                [x as f32, y as f32, z as f32]
            });
            (points.len() - 1) as i64
        })
    };
    // Cylindrical: a seam edge of a closed grid is drawn once, on the
    // `2π` line, PEC if any copy is; axis z edges are not drawn.
    let is_pec = |a: Axis, p: [usize; 3]| -> bool {
        if !cyl {
            return mats.pec_mask(a)[grid.index(p)];
        }
        if a == Axis::Z && p[0] == 0 && grid.has_axis() {
            return false;
        }
        if grid.is_closed() && a != Axis::Y && p[1] == 0 {
            return false;
        }
        e_edge_copies(grid, a, p)
            .iter()
            .any(|&c| mats.pec_mask(a)[grid.index(c)])
    };
    let mut conn: Vec<i64> = Vec::new();
    let mut visit = |p: [usize; 3]| {
        for a in Axis::ALL {
            let d = a.index();
            if p[d] + 1 >= n[d] || !is_pec(a, p) {
                continue;
            }
            let mut q = p;
            q[d] += 1;
            conn.push(id(p));
            conn.push(id(q));
        }
    };
    // openEMS's order: x outermost, z innermost; on cylindrical grids z
    // outermost, then ρ, α innermost.
    if cyl {
        for k in 0..n[2] {
            for i in 0..n[0] {
                for j in 0..n[1] {
                    visit([i, j, k]);
                }
            }
        }
    } else {
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    visit([i, j, k]);
                }
            }
        }
    }
    let offsets: Vec<i64> = (1..=conn.len() as i64 / 2).map(|l| 2 * l).collect();
    let i64_bytes = |v: &[i64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut data = String::new();
    let mut put = |raw: &[u8]| -> usize {
        let off = data.len();
        data.push_str(&vtk_binary(raw));
        off
    };
    let pts_off = put(&f32_bytes(&points));
    let empty = i64_bytes(&[]);
    let verts = [put(&empty), put(&empty)];
    let lines = [put(&i64_bytes(&conn)), put(&i64_bytes(&offsets))];
    let strips = [put(&empty), put(&empty)];
    let polys = [put(&empty), put(&empty)];
    let range = norm_range(&points);
    let mut s = String::new();
    s.push_str("<?xml version=\"1.0\"?>\n");
    s.push_str("<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">\n");
    s.push_str("  <PolyData>\n");
    let _ = writeln!(
        s,
        "    <Piece {} {} {} {} {}>",
        attr("NumberOfPoints", &points.len().to_string()),
        attr("NumberOfVerts", "0"),
        attr("NumberOfLines", &offsets.len().to_string()),
        attr("NumberOfStrips", "0"),
        attr("NumberOfPolys", "0"),
    );
    s.push_str("      <PointData>\n      </PointData>\n      <CellData>\n      </CellData>\n");
    s.push_str("      <Points>\n");
    let (lo, hi) = if points.is_empty() {
        // openEMS: VTK's initial range of an empty array.
        ("1e+299".to_string(), "-1e+299".to_string())
    } else {
        (printf_g11(range.0), printf_g11(range.1))
    };
    let _ = writeln!(
        s,
        "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" {} {} {}/>",
        attr("RangeMin", &lo),
        attr("RangeMax", &hi),
        attr("offset", &pts_off.to_string())
    );
    s.push_str("      </Points>\n");
    for (tag, offs) in [
        ("Verts", verts),
        ("Lines", lines),
        ("Strips", strips),
        ("Polys", polys),
    ] {
        let _ = writeln!(s, "      <{tag}>");
        for (name, off) in [("connectivity", offs[0]), ("offsets", offs[1])] {
            let _ = writeln!(
                s,
                "        <DataArray type=\"Int64\" Name=\"{name}\" format=\"appended\" {} {} {}/>",
                attr("RangeMin", ""),
                attr("RangeMax", ""),
                attr("offset", &off.to_string())
            );
        }
        let _ = writeln!(s, "      </{tag}>");
    }
    s.push_str("    </Piece>\n  </PolyData>\n  <AppendedData encoding=\"base64\">\n   _");
    s.push_str(&data);
    s.push_str("\n  </AppendedData>\n</VTKFile>\n");
    Ok(s)
}

/// The text of a `box_dump_<name>.vtk` file for the box with corners
/// `start`, `stop` \[m\] (see the [module docs](self)).
pub fn box_dump(start: [f64; 3], stop: [f64; 3]) -> String {
    let (a, b) = (start, stop);
    let corners = [
        [a[0], a[1], a[2]],
        [b[0], a[1], a[2]],
        [b[0], b[1], a[2]],
        [a[0], b[1], a[2]],
        [a[0], a[1], b[2]],
        [b[0], a[1], b[2]],
        [b[0], b[1], b[2]],
        [a[0], b[1], b[2]],
    ];
    let mut s =
        String::from("# vtk DataFile Version 2.0\n\nASCII\nDATASET POLYDATA\nPOINTS 8 float\n");
    for c in corners {
        let _ = writeln!(
            s,
            "{} {} {}",
            super::probes::fmt_g(c[0], 6),
            super::probes::fmt_g(c[1], 6),
            super::probes::fmt_g(c[2], 6)
        );
    }
    s.push_str("POLYGONS 6 30\n4 0 1 2 3\n4 4 5 6 7\n4 7 6 2 3\n4 4 5 1 0\n4 0 4 7 3\n4 5 6 2 1\n");
    s
}

/// Writes [`material_dump`] as `dir/material_dump.vtr` (`.vts` on
/// cylindrical grids).
///
/// # Errors
/// See [`material_dump`]; I/O errors.
pub fn write_material_dump(
    dir: impl AsRef<Path>,
    grid: &Grid,
    mats: &EdgeMaterials,
) -> Result<PathBuf> {
    write(
        &dir.as_ref().join(material_dump_file(grid)),
        &material_dump(grid, mats)?,
    )
}

/// Writes [`operator_dump`] as `dir/operator_dump.vtr` (`.vts` on
/// cylindrical grids).
///
/// # Errors
/// See [`operator_dump`]; I/O errors.
pub fn write_operator_dump(
    dir: impl AsRef<Path>,
    grid: &Grid,
    coeffs: &Coefficients,
    exc_volt: &[(Axis, [usize; 3], f64)],
) -> Result<PathBuf> {
    write(
        &dir.as_ref().join(operator_dump_file(grid)),
        &operator_dump(grid, coeffs, exc_volt)?,
    )
}

/// Writes [`pec_dump`] as `dir/PEC_dump.vtp`.
///
/// # Errors
/// See [`pec_dump`]; I/O errors.
pub fn write_pec_dump(dir: impl AsRef<Path>, grid: &Grid, mats: &EdgeMaterials) -> Result<PathBuf> {
    write(&dir.as_ref().join(PEC_DUMP_FILE), &pec_dump(grid, mats)?)
}

/// Writes [`box_dump`] as `dir/box_dump_<name>.vtk` (`name` may be
/// empty, as openEMS's probe boxes are).
///
/// # Errors
/// I/O errors; [`Error::Invalid`] for a name that is not a plain file
/// name part.
pub fn write_box_dump(
    dir: impl AsRef<Path>,
    name: &str,
    start: [f64; 3],
    stop: [f64; 3],
) -> Result<PathBuf> {
    if name.contains(['/', '\\']) {
        return Err(Error::Invalid(format!(
            "box name {name:?} is not a plain file name"
        )));
    }
    write(
        &dir.as_ref().join(format!("box_dump_{name}.vtk")),
        &box_dump(start, stop),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{FaceConditions, Material};

    fn grid() -> Grid {
        let l: Vec<f64> = (0..4).map(|i| f64::from(i) * 1e-3).collect();
        Grid::new([l.clone(), l.clone(), l[..3].to_vec()]).unwrap()
    }

    #[test]
    fn box_file() {
        let t = box_dump([0.0005, 0.0005, 0.0015], [0.0085, 0.0085, 0.0015]);
        let want = "# vtk DataFile Version 2.0\n\nASCII\nDATASET POLYDATA\nPOINTS 8 float\n\
0.0005 0.0005 0.0015\n0.0085 0.0005 0.0015\n0.0085 0.0085 0.0015\n0.0005 0.0085 0.0015\n\
0.0005 0.0005 0.0015\n0.0085 0.0005 0.0015\n0.0085 0.0085 0.0015\n0.0005 0.0085 0.0015\n\
POLYGONS 6 30\n4 0 1 2 3\n4 4 5 6 7\n4 7 6 2 3\n4 4 5 1 0\n4 0 4 7 3\n4 5 6 2 1\n";
        assert_eq!(t, want);
    }

    #[test]
    fn pec_lines_and_padding() {
        let g = grid();
        let mut m = EdgeMaterials::new(&g, Material::VACUUM);
        m.set_pec_edge(Axis::X, [1, 1, 1]);
        m.set_pec_edge(Axis::Y, [1, 1, 1]);
        let t = pec_dump(&g, &m).unwrap();
        assert!(t.contains(
            "<Piece NumberOfPoints=\"3\"                    NumberOfVerts=\"0\"                    NumberOfLines=\"2\"                    NumberOfStrips=\"0\"                    NumberOfPolys=\"0\"                   >"
        ));
        assert!(t.contains(
            "RangeMin=\"\"                     RangeMax=\"\"                     offset=\""
        ));
        // Empty arrays: the 12-byte block header (0 blocks) only.
        assert!(t.contains("AAAAAACAAAAAAAAA"));
    }

    #[test]
    fn material_and_operator_arrays() {
        let g = grid();
        let m = EdgeMaterials::new(&g, Material::dielectric(2.0));
        let t = material_dump(&g, &m).unwrap();
        for a in ["epsilon", "mue", "kappa", "sigma"] {
            assert!(
                t.contains(&format!("Name=\"{a}\" NumberOfComponents=\"3\"")),
                "{a}"
            );
        }
        assert!(t.contains("RangeMin=\"3.4641016151377544\""));
        let c = Coefficients::from_edges(&g, &m, &FaceConditions::default(), 1e-12).unwrap();
        let t = operator_dump(&g, &c, &[(Axis::Z, [1, 1, 0], 1e-3)]).unwrap();
        for a in ["exc_volt", "vv", "vi", "iv", "ii"] {
            assert!(
                t.contains(&format!("Name=\"{a}\" NumberOfComponents=\"3\"")),
                "{a}"
            );
        }
        assert!(t.contains("Name=\"exc_volt\" NumberOfComponents=\"3\" format=\"binary\" RangeMin=\"0\" RangeMax=\"0.0010000000474974513\""));
    }
}
