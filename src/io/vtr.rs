//! VTK XML `RectilinearGrid` (`.vtr`) writer and `.pvd` collection writer.
//!
//! The file layout follows Kitware's public "VTK File Formats" documentation
//! (XML formats section): a `<VTKFile type="RectilinearGrid">` root holding one
//! `<RectilinearGrid>` with a single `<Piece>`, whose `<PointData>`,
//! `<CellData>` and `<Coordinates>` children carry `<DataArray>` elements.
//! ParaView and VisIt read these files directly.
//!
//! # Grid and data
//!
//! The grid is given by its three coordinate arrays (`x`, `y`, `z`, in metres,
//! strictly increasing). With `nx`, `ny`, `nz` coordinates per axis:
//!
//! * a **point** array has `nx·ny·nz` tuples, one per grid node;
//! * a **cell** array has `cx·cy·cz` tuples with `cx = max(nx − 1, 1)` and so
//!   on. An axis with a single coordinate is a flat (degenerate) dimension
//!   that still counts one cell layer, as VTK does for structured data; a
//!   2-D dump plane of `nx × ny × 1` points therefore has `(nx−1)(ny−1)`
//!   cells.
//!
//! Each array is a scalar (one component) or a 3-component vector, stored as
//! `f32` or `f64` (see [`VtkFloat`]), and passed as a [`Field`].
//!
//! # Memory layout
//!
//! VTK stores tuples with **x varying fastest**: tuple `(i, j, k)` is at
//! `i + nx·(j + ny·k)`. The `ems` engine stores fields with **z varying
//! fastest**, at `(i·ny + j)·nz + k`. Every array is therefore registered with
//! a [`Layout`] describing how *your* buffer is ordered; the writer reorders
//! on the fly while writing (no copy of the data is made). Use
//! [`Layout::ZFastest`] for engine buffers and [`Layout::XFastest`] for data
//! that is already in VTK order.
//!
//! # Complex data
//!
//! Frequency-domain dumps are complex. VTK has no complex type, so a complex
//! array is written as two real arrays, chosen with [`ComplexForm`]:
//! `name_re` / `name_im` (the default) or `name_mag` / `name_phase` (phase in
//! radians, `atan2(im, re)`). For vectors the split is applied per component.
//!
//! # Encodings
//!
//! See [`Encoding`]: `ascii`, inline base64 `binary` (the default), or raw
//! `appended` data. Binary data is little-endian, preceded by a byte-count
//! header of type [`HeaderType`] (`UInt64` by default), uncompressed.
//!
//! # Time series
//!
//! [`VtrWriter::time`] stores a `TimeValue` entry in the dataset's
//! `<FieldData>`, which ParaView uses as the time of the dataset.
//! [`PvdCollection`] writes a `.pvd` file that lists a series of `.vtr` files
//! with their time step (or frequency), so the whole series opens as one
//! animated dataset.
//!
//! # Example
//!
//! ```
//! use ems::io::{Encoding, Layout, VtrWriter};
//!
//! let x = [0.0, 1e-3];
//! let y = [0.0, 1e-3, 2e-3];
//! let z = [0.0, 0.5e-3, 1e-3, 1.5e-3];
//! let n = x.len() * y.len() * z.len();
//! // An engine-ordered (z fastest) vector field and a cell scalar.
//! let e: Vec<[f32; 3]> = (0..n).map(|m| [m as f32, 0.0, 1.0]).collect();
//! let eps = vec![1.0f64; (x.len() - 1) * (y.len() - 1) * (z.len() - 1)];
//!
//! let mut out = Vec::new();
//! VtrWriter::new(&x, &y, &z)
//!     .time(1.5e-9)
//!     .point_vector("E", &e, Layout::ZFastest)
//!     .cell_scalar("eps_r", &eps, Layout::ZFastest)
//!     .encoding(Encoding::Binary)
//!     .write_to(&mut out)?;
//!
//! let text = String::from_utf8_lossy(&out);
//! assert!(text.starts_with("<?xml"));
//! assert!(text.contains(r#"Name="E""#));
//! # Ok::<(), ems::Error>(())
//! ```

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::base64::Encoder;
use crate::{Error, Result};

/// Ordering of the tuples of a data buffer on the `(i, j, k)` grid.
///
/// With `(dx, dy, dz)` the tuple counts per axis (point counts for point data,
/// cell counts for cell data):
///
/// | variant | index of tuple `(i, j, k)` | used by |
/// |---|---|---|
/// | [`ZFastest`](Layout::ZFastest) | `(i·dy + j)·dz + k` | `ems` engine buffers |
/// | [`XFastest`](Layout::XFastest) | `(k·dy + j)·dx + i` | VTK files |
///
/// The writer always emits VTK order and reorders `ZFastest` input on the fly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Layout {
    /// z varies fastest, then y, then x (C order over `[i][j][k]`); the
    /// native layout of `ems` field arrays.
    #[default]
    ZFastest,
    /// x varies fastest, then y, then z (C order over `[k][j][i]`); the
    /// native layout of VTK.
    XFastest,
}

impl Layout {
    /// Linear index of tuple `(i, j, k)` in a buffer of `dims = [dx, dy, dz]`
    /// tuples ordered with this layout.
    #[inline]
    pub fn index(self, dims: [usize; 3], i: usize, j: usize, k: usize) -> usize {
        let [dx, dy, dz] = dims;
        match self {
            Layout::ZFastest => (i * dy + j) * dz + k,
            Layout::XFastest => (k * dy + j) * dx + i,
        }
    }
}

/// How data arrays are encoded in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Encoding {
    /// Human-readable decimal text (`format="ascii"`), shortest round-trip
    /// representation in exponent notation. Large and slow; handy for
    /// debugging and tests.
    Ascii,
    /// Inline base64 (`format="binary"`): each array is the base64 encoding
    /// of a little-endian byte-count header ([`HeaderType`]) immediately
    /// followed by the little-endian values, encoded as one stream. The file
    /// stays valid XML text.
    #[default]
    Binary,
    /// Raw appended data (`format="appended"`): arrays reference byte offsets
    /// into an `<AppendedData encoding="raw">` block at the end of the file,
    /// each block being a header followed by the raw little-endian values.
    /// Most compact and fastest to read and write; the file is not valid XML
    /// text any more (it contains binary bytes), which VTK readers accept.
    Appended,
}

/// Integer type of the byte-count header that precedes binary data blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum HeaderType {
    /// 32-bit header: arrays are limited to `u32::MAX` bytes.
    UInt32,
    /// 64-bit header (the default).
    #[default]
    UInt64,
}

impl HeaderType {
    fn name(self) -> &'static str {
        match self {
            HeaderType::UInt32 => "UInt32",
            HeaderType::UInt64 => "UInt64",
        }
    }

    fn size(self) -> usize {
        match self {
            HeaderType::UInt32 => 4,
            HeaderType::UInt64 => 8,
        }
    }

    fn bytes(self, nbytes: usize) -> Result<Vec<u8>> {
        Ok(match self {
            HeaderType::UInt32 => u32::try_from(nbytes)
                .map_err(|_| {
                    Error::Invalid(format!("array of {nbytes} bytes needs a UInt64 header"))
                })?
                .to_le_bytes()
                .to_vec(),
            HeaderType::UInt64 => (nbytes as u64).to_le_bytes().to_vec(),
        })
    }
}

/// How a complex array is split into two real VTK arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum ComplexForm {
    /// `name_re` and `name_im`: real and imaginary parts (the default).
    #[default]
    ReIm,
    /// `name_mag` and `name_phase`: magnitude `|z|` and phase
    /// `atan2(im, re)` in radians.
    MagPhase,
}

/// The data of one array: a scalar or a 3-component vector.
///
/// The slices are tuples in the order given by the [`Layout`] passed alongside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Field<'a, T> {
    /// One value per tuple.
    Scalar(&'a [T]),
    /// One `[x, y, z]` vector per tuple (interleaved components).
    Vector(&'a [[T; 3]]),
    /// A vector stored as three separate component buffers `[x, y, z]`, each
    /// with one value per tuple (as the engine stores `Ex`, `Ey`, `Ez`).
    Split([&'a [T]; 3]),
}

impl<T: Copy> Field<'_, T> {
    fn ncomp(&self) -> usize {
        match self {
            Field::Scalar(_) => 1,
            Field::Vector(_) | Field::Split(_) => 3,
        }
    }

    fn check_len(&self, n: usize, what: &str) -> Result<()> {
        let bad = |len: usize| {
            Err(Error::Invalid(format!(
                "VTK array {what}: {len} tuples given, the grid needs {n}"
            )))
        };
        match self {
            Field::Scalar(s) if s.len() != n => bad(s.len()),
            Field::Vector(v) if v.len() != n => bad(v.len()),
            Field::Split(s) => match s.iter().find(|c| c.len() != n) {
                Some(c) => bad(c.len()),
                None => Ok(()),
            },
            _ => Ok(()),
        }
    }

    #[inline]
    fn get(&self, idx: usize, c: usize) -> T {
        match self {
            Field::Scalar(s) => s[idx],
            Field::Vector(v) => v[idx][c],
            Field::Split(s) => s[c][idx],
        }
    }
}

mod imp {
    use super::{ComplexForm, Field};
    use std::fmt::LowerExp;

    /// Which real quantity of an array is written.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Part {
        Real,
        Re,
        Im,
        Mag,
        Phase,
    }

    impl Part {
        pub fn split(form: ComplexForm) -> [(Part, &'static str); 2] {
            match form {
                ComplexForm::ReIm => [(Part::Re, "_re"), (Part::Im, "_im")],
                ComplexForm::MagPhase => [(Part::Mag, "_mag"), (Part::Phase, "_phase")],
            }
        }
    }

    /// Typed array data: a real field, or a complex one with the part to write.
    #[derive(Debug, Clone, Copy)]
    pub struct ArrayData<'a, T> {
        pub re: Field<'a, T>,
        pub im: Option<Field<'a, T>>,
        pub part: Part,
    }

    /// Type-erased [`ArrayData`].
    #[derive(Debug, Clone, Copy)]
    pub enum Store<'a> {
        F32(ArrayData<'a, f32>),
        F64(ArrayData<'a, f64>),
    }

    pub trait Sealed: Copy + LowerExp + 'static {
        const VTK_TYPE: &'static str;
        const SIZE: usize;
        fn extend_le(self, out: &mut Vec<u8>);
        fn mag(re: Self, im: Self) -> Self;
        fn phase(re: Self, im: Self) -> Self;
        fn store(d: ArrayData<'_, Self>) -> Store<'_>;
    }

    macro_rules! sealed {
        ($t:ty, $name:literal, $var:ident) => {
            impl Sealed for $t {
                const VTK_TYPE: &'static str = $name;
                const SIZE: usize = core::mem::size_of::<$t>();
                #[inline]
                fn extend_le(self, out: &mut Vec<u8>) {
                    out.extend_from_slice(&self.to_le_bytes());
                }
                #[inline]
                fn mag(re: Self, im: Self) -> Self {
                    re.hypot(im)
                }
                #[inline]
                fn phase(re: Self, im: Self) -> Self {
                    im.atan2(re)
                }
                fn store(d: ArrayData<'_, Self>) -> Store<'_> {
                    Store::$var(d)
                }
            }
        };
    }
    sealed!(f32, "Float32", F32);
    sealed!(f64, "Float64", F64);
}

use imp::{ArrayData, Part, Sealed, Store};

/// Floating-point element types that can be written to VTK files: `f32`
/// (`Float32`) and `f64` (`Float64`). This trait is sealed.
pub trait VtkFloat: Sealed {}
impl VtkFloat for f32 {}
impl VtkFloat for f64 {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Centering {
    Point,
    Cell,
}

#[derive(Debug, Clone)]
struct Entry<'a> {
    name: String,
    centering: Centering,
    layout: Layout,
    store: Store<'a>,
}

impl Entry<'_> {
    fn ncomp(&self) -> usize {
        match &self.store {
            Store::F32(d) => d.re.ncomp(),
            Store::F64(d) => d.re.ncomp(),
        }
    }

    fn elem_size(&self) -> usize {
        match &self.store {
            Store::F32(_) => f32::SIZE,
            Store::F64(_) => f64::SIZE,
        }
    }

    fn type_name(&self) -> &'static str {
        match &self.store {
            Store::F32(_) => f32::VTK_TYPE,
            Store::F64(_) => f64::VTK_TYPE,
        }
    }
}

/// Builder for one `.vtr` file. See the [module documentation](self).
///
/// All inputs are borrowed; nothing is copied. Inconsistent inputs (wrong
/// array lengths, duplicate names, non-increasing coordinates, …) are reported
/// as [`Error::Invalid`] by [`write`](Self::write) / [`write_to`](Self::write_to)
/// before anything is written.
#[derive(Debug, Clone)]
pub struct VtrWriter<'a> {
    coords: [&'a [f64]; 3],
    entries: Vec<Entry<'a>>,
    fields: Vec<(String, &'a [f64])>,
    time: Option<f64>,
    encoding: Encoding,
    header: HeaderType,
}

impl<'a> VtrWriter<'a> {
    /// Starts a file for the rectilinear grid with coordinates `x`, `y`, `z`
    /// (metres, each non-empty and strictly increasing).
    pub fn new(x: &'a [f64], y: &'a [f64], z: &'a [f64]) -> Self {
        VtrWriter {
            coords: [x, y, z],
            entries: Vec::new(),
            fields: Vec::new(),
            time: None,
            encoding: Encoding::default(),
            header: HeaderType::default(),
        }
    }

    /// Sets the data encoding (default [`Encoding::Binary`]).
    pub fn encoding(mut self, encoding: Encoding) -> Self {
        self.encoding = encoding;
        self
    }

    /// Sets the binary header type (default [`HeaderType::UInt64`]).
    pub fn header_type(mut self, header: HeaderType) -> Self {
        self.header = header;
        self
    }

    /// Stores `t` as the `TimeValue` field-data entry, the time ParaView
    /// assigns to this dataset. For a frequency-domain dump, pass the
    /// frequency (and consider also adding it with [`field`](Self::field)
    /// under a descriptive name).
    pub fn time(mut self, t: f64) -> Self {
        self.time = Some(t);
        self
    }

    /// Adds a `Float64` field-data array (dataset-wide metadata, not attached
    /// to points or cells), e.g. `field("Frequency", &[f])`.
    pub fn field(mut self, name: impl Into<String>, values: &'a [f64]) -> Self {
        self.fields.push((name.into(), values));
        self
    }

    fn push<T: VtkFloat>(
        mut self,
        centering: Centering,
        name: String,
        layout: Layout,
        data: ArrayData<'a, T>,
    ) -> Self {
        self.entries.push(Entry {
            name,
            centering,
            layout,
            store: T::store(data),
        });
        self
    }

    fn push_real<T: VtkFloat>(
        self,
        centering: Centering,
        name: String,
        field: Field<'a, T>,
        layout: Layout,
    ) -> Self {
        let data = ArrayData {
            re: field,
            im: None,
            part: Part::Real,
        };
        self.push(centering, name, layout, data)
    }

    fn push_complex<T: VtkFloat>(
        self,
        centering: Centering,
        name: &str,
        re: Field<'a, T>,
        im: Field<'a, T>,
        layout: Layout,
        form: ComplexForm,
    ) -> Self {
        let [(p0, s0), (p1, s1)] = Part::split(form);
        let d = |part| ArrayData {
            re,
            im: Some(im),
            part,
        };
        self.push(centering, format!("{name}{s0}"), layout, d(p0))
            .push(centering, format!("{name}{s1}"), layout, d(p1))
    }

    /// Adds a point array (one tuple per grid node).
    pub fn point<T: VtkFloat>(
        self,
        name: impl Into<String>,
        field: Field<'a, T>,
        layout: Layout,
    ) -> Self {
        self.push_real(Centering::Point, name.into(), field, layout)
    }

    /// Adds a cell array (one tuple per grid cell).
    pub fn cell<T: VtkFloat>(
        self,
        name: impl Into<String>,
        field: Field<'a, T>,
        layout: Layout,
    ) -> Self {
        self.push_real(Centering::Cell, name.into(), field, layout)
    }

    /// Adds a scalar point array; shorthand for
    /// [`point`](Self::point)`(name, Field::Scalar(data), layout)`.
    pub fn point_scalar<T: VtkFloat>(
        self,
        name: impl Into<String>,
        data: &'a [T],
        layout: Layout,
    ) -> Self {
        self.point(name, Field::Scalar(data), layout)
    }

    /// Adds an interleaved vector point array; shorthand for
    /// [`point`](Self::point)`(name, Field::Vector(data), layout)`.
    pub fn point_vector<T: VtkFloat>(
        self,
        name: impl Into<String>,
        data: &'a [[T; 3]],
        layout: Layout,
    ) -> Self {
        self.point(name, Field::Vector(data), layout)
    }

    /// Adds a scalar cell array; shorthand for
    /// [`cell`](Self::cell)`(name, Field::Scalar(data), layout)`.
    pub fn cell_scalar<T: VtkFloat>(
        self,
        name: impl Into<String>,
        data: &'a [T],
        layout: Layout,
    ) -> Self {
        self.cell(name, Field::Scalar(data), layout)
    }

    /// Adds an interleaved vector cell array; shorthand for
    /// [`cell`](Self::cell)`(name, Field::Vector(data), layout)`.
    pub fn cell_vector<T: VtkFloat>(
        self,
        name: impl Into<String>,
        data: &'a [[T; 3]],
        layout: Layout,
    ) -> Self {
        self.cell(name, Field::Vector(data), layout)
    }

    /// Adds a complex point array as two real arrays (see [`ComplexForm`]).
    /// `re` and `im` must have the same shape (both scalar or both vector).
    pub fn point_complex<T: VtkFloat>(
        self,
        name: &str,
        re: Field<'a, T>,
        im: Field<'a, T>,
        layout: Layout,
        form: ComplexForm,
    ) -> Self {
        self.push_complex(Centering::Point, name, re, im, layout, form)
    }

    /// Adds a complex cell array as two real arrays (see [`ComplexForm`]).
    /// `re` and `im` must have the same shape (both scalar or both vector).
    pub fn cell_complex<T: VtkFloat>(
        self,
        name: &str,
        re: Field<'a, T>,
        im: Field<'a, T>,
        layout: Layout,
        form: ComplexForm,
    ) -> Self {
        self.push_complex(Centering::Cell, name, re, im, layout, form)
    }

    /// Point counts per axis.
    fn point_dims(&self) -> [usize; 3] {
        self.coords.map(<[f64]>::len)
    }

    /// Cell counts per axis (a single-point axis counts one flat cell layer).
    fn cell_dims(&self) -> [usize; 3] {
        self.coords.map(|c| c.len().saturating_sub(1).max(1))
    }

    fn dims(&self, c: Centering) -> [usize; 3] {
        match c {
            Centering::Point => self.point_dims(),
            Centering::Cell => self.cell_dims(),
        }
    }

    fn tuples(&self, c: Centering) -> Result<usize> {
        let [a, b, d] = self.dims(c);
        a.checked_mul(b)
            .and_then(|ab| ab.checked_mul(d))
            .ok_or_else(|| Error::Invalid("VTK grid too large".into()))
    }

    fn validate(&self) -> Result<()> {
        for (axis, c) in ["x", "y", "z"].iter().zip(self.coords) {
            if c.is_empty() {
                return Err(Error::Invalid(format!("VTK grid: no {axis} coordinates")));
            }
            if c.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid(format!(
                    "VTK grid: non-finite {axis} coordinate"
                )));
            }
            if c.windows(2).any(|w| w[1] <= w[0]) {
                return Err(Error::Invalid(format!(
                    "VTK grid: {axis} coordinates are not strictly increasing"
                )));
            }
        }
        let binary = self.encoding != Encoding::Ascii;
        for (n, e) in self.entries.iter().enumerate() {
            if e.name.is_empty() {
                return Err(Error::Invalid("VTK array with an empty name".into()));
            }
            if self.entries[..n]
                .iter()
                .any(|o| o.centering == e.centering && o.name == e.name)
            {
                return Err(Error::Invalid(format!(
                    "duplicate VTK array name {:?}",
                    e.name
                )));
            }
            let tuples = self.tuples(e.centering)?;
            let what = format!("{:?}", e.name);
            match &e.store {
                Store::F32(d) => check_data(d, tuples, &what)?,
                Store::F64(d) => check_data(d, tuples, &what)?,
            }
            if binary {
                let nbytes = tuples
                    .checked_mul(e.ncomp() * e.elem_size())
                    .ok_or_else(|| Error::Invalid("VTK array too large".into()))?;
                self.header.bytes(nbytes)?;
            }
        }
        for (n, (name, _)) in self.fields.iter().enumerate() {
            if name.is_empty() {
                return Err(Error::Invalid("VTK field data with an empty name".into()));
            }
            if self.fields[..n].iter().any(|(o, _)| o == name)
                || (self.time.is_some() && name == "TimeValue")
            {
                return Err(Error::Invalid(format!(
                    "duplicate VTK field data name {name:?}"
                )));
            }
        }
        Ok(())
    }

    /// Writes the file to `path` (buffered).
    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let mut w = BufWriter::new(File::create(path)?);
        self.write_unchecked(&mut w)?;
        w.flush()?;
        Ok(())
    }

    /// Writes the file to `w`. The output is produced in many small writes;
    /// wrap unbuffered sinks (files, sockets) in a [`BufWriter`].
    pub fn write_to<W: Write>(&self, mut w: W) -> Result<()> {
        self.validate()?;
        self.write_unchecked(&mut w)
    }

    fn write_unchecked<W: Write>(&self, w: &mut W) -> Result<()> {
        let pd = self.point_dims();
        let ext = format!("0 {} 0 {} 0 {}", pd[0] - 1, pd[1] - 1, pd[2] - 1);
        writeln!(w, r#"<?xml version="1.0"?>"#)?;
        writeln!(
            w,
            r#"<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian" header_type="{}">"#,
            self.header.name()
        )?;
        writeln!(w, r#"  <RectilinearGrid WholeExtent="{ext}">"#)?;
        self.write_field_data(w)?;
        writeln!(w, r#"    <Piece Extent="{ext}">"#)?;

        let mut offset = 0usize;
        for (centering, tag) in [
            (Centering::Point, "PointData"),
            (Centering::Cell, "CellData"),
        ] {
            let entries: Vec<&Entry<'_>> = self
                .entries
                .iter()
                .filter(|e| e.centering == centering)
                .collect();
            if entries.is_empty() {
                continue;
            }
            let mut open = format!("      <{tag}");
            for (attr, ncomp) in [("Scalars", 1), ("Vectors", 3)] {
                if let Some(e) = entries.iter().find(|e| e.ncomp() == ncomp) {
                    write!(open, r#" {attr}="{}""#, escape(&e.name)).ok();
                }
            }
            writeln!(w, "{open}>")?;
            let dims = self.dims(centering);
            let tuples = self.tuples(centering)?;
            for e in entries {
                let count = tuples * e.ncomp();
                self.open_array(
                    w,
                    e.type_name(),
                    &e.name,
                    e.ncomp(),
                    &mut offset,
                    count * e.elem_size(),
                )?;
                if self.encoding != Encoding::Appended {
                    match &e.store {
                        Store::F32(d) => {
                            self.write_values(w, count, vtk_order(d, dims, e.layout))?
                        }
                        Store::F64(d) => {
                            self.write_values(w, count, vtk_order(d, dims, e.layout))?
                        }
                    }
                }
                self.close_array(w)?;
            }
            writeln!(w, "      </{tag}>")?;
        }

        writeln!(w, "      <Coordinates>")?;
        for (name, c) in ["x", "y", "z"].iter().zip(self.coords) {
            self.open_array(w, f64::VTK_TYPE, name, 1, &mut offset, c.len() * f64::SIZE)?;
            if self.encoding != Encoding::Appended {
                self.write_values(w, c.len(), c.iter().copied())?;
            }
            self.close_array(w)?;
        }
        writeln!(w, "      </Coordinates>")?;
        writeln!(w, "    </Piece>")?;
        writeln!(w, "  </RectilinearGrid>")?;

        if self.encoding == Encoding::Appended {
            write!(w, "  <AppendedData encoding=\"raw\">\n   _")?;
            for centering in [Centering::Point, Centering::Cell] {
                let dims = self.dims(centering);
                let tuples = self.tuples(centering)?;
                for e in self.entries.iter().filter(|e| e.centering == centering) {
                    let count = tuples * e.ncomp();
                    match &e.store {
                        Store::F32(d) => self.write_raw(w, count, vtk_order(d, dims, e.layout))?,
                        Store::F64(d) => self.write_raw(w, count, vtk_order(d, dims, e.layout))?,
                    }
                }
            }
            for c in self.coords {
                self.write_raw(w, c.len(), c.iter().copied())?;
            }
            writeln!(w, "\n  </AppendedData>")?;
        }
        writeln!(w, "</VTKFile>")?;
        Ok(())
    }

    fn write_field_data<W: Write>(&self, w: &mut W) -> Result<()> {
        if self.time.is_none() && self.fields.is_empty() {
            return Ok(());
        }
        writeln!(w, "    <FieldData>")?;
        let time = self.time.map(|t| [t]);
        let all = time
            .iter()
            .map(|t| ("TimeValue", &t[..]))
            .chain(self.fields.iter().map(|(n, v)| (n.as_str(), *v)));
        for (name, values) in all {
            writeln!(
                w,
                r#"      <DataArray type="Float64" Name="{}" NumberOfTuples="{}" format="ascii">"#,
                escape(name),
                values.len()
            )?;
            write_ascii(w, values.iter().copied())?;
            writeln!(w, "      </DataArray>")?;
        }
        writeln!(w, "    </FieldData>")?;
        Ok(())
    }

    fn open_array<W: Write>(
        &self,
        w: &mut W,
        type_name: &str,
        name: &str,
        ncomp: usize,
        offset: &mut usize,
        nbytes: usize,
    ) -> Result<()> {
        write!(
            w,
            r#"        <DataArray type="{type_name}" Name="{}" NumberOfComponents="{ncomp}""#,
            escape(name)
        )?;
        match self.encoding {
            Encoding::Ascii => writeln!(w, r#" format="ascii">"#)?,
            Encoding::Binary => writeln!(w, r#" format="binary">"#)?,
            Encoding::Appended => {
                write!(w, r#" format="appended" offset="{offset}">"#)?;
                *offset += self.header.size() + nbytes;
            }
        }
        Ok(())
    }

    fn close_array<W: Write>(&self, w: &mut W) -> Result<()> {
        match self.encoding {
            Encoding::Appended => writeln!(w, "</DataArray>")?,
            _ => writeln!(w, "        </DataArray>")?,
        }
        Ok(())
    }

    /// Writes the body of an inline (ascii or binary) data array.
    fn write_values<W: Write, T: Sealed>(
        &self,
        w: &mut W,
        count: usize,
        values: impl Iterator<Item = T>,
    ) -> Result<()> {
        match self.encoding {
            Encoding::Ascii => write_ascii(w, values),
            Encoding::Binary => {
                w.write_all(b"          ")?;
                let mut enc = Encoder::new(&mut *w);
                enc.write_bytes(&self.header.bytes(count * T::SIZE)?)?;
                let mut buf = Vec::with_capacity(CHUNK * T::SIZE);
                for v in values {
                    v.extend_le(&mut buf);
                    if buf.len() >= CHUNK * T::SIZE {
                        enc.write_bytes(&buf)?;
                        buf.clear();
                    }
                }
                enc.write_bytes(&buf)?;
                enc.finish()?;
                w.write_all(b"\n")?;
                Ok(())
            }
            Encoding::Appended => unreachable!("appended data is written separately"),
        }
    }

    /// Writes one raw appended block: header then little-endian values.
    fn write_raw<W: Write, T: Sealed>(
        &self,
        w: &mut W,
        count: usize,
        values: impl Iterator<Item = T>,
    ) -> Result<()> {
        w.write_all(&self.header.bytes(count * T::SIZE)?)?;
        let mut buf = Vec::with_capacity(CHUNK * T::SIZE);
        for v in values {
            v.extend_le(&mut buf);
            if buf.len() >= CHUNK * T::SIZE {
                w.write_all(&buf)?;
                buf.clear();
            }
        }
        w.write_all(&buf)?;
        Ok(())
    }
}

/// Values buffered per write in binary modes.
const CHUNK: usize = 4096;
/// Values per line in ASCII mode.
const ASCII_PER_LINE: usize = 6;

fn check_data<T: Sealed>(d: &ArrayData<'_, T>, tuples: usize, what: &str) -> Result<()> {
    d.re.check_len(tuples, what)?;
    if let Some(im) = &d.im {
        if im.ncomp() != d.re.ncomp() {
            return Err(Error::Invalid(format!(
                "VTK array {what}: real and imaginary parts have different shapes"
            )));
        }
        im.check_len(tuples, what)?;
    }
    Ok(())
}

/// Iterates the values of `d` in VTK order (x fastest, components innermost).
fn vtk_order<'s, T: Sealed>(
    d: &'s ArrayData<'_, T>,
    dims: [usize; 3],
    layout: Layout,
) -> impl Iterator<Item = T> + 's {
    let [dx, dy, dz] = dims;
    let ncomp = d.re.ncomp();
    (0..dz).flat_map(move |k| {
        (0..dy).flat_map(move |j| {
            (0..dx).flat_map(move |i| {
                let idx = layout.index(dims, i, j, k);
                (0..ncomp).map(move |c| value(d, idx, c))
            })
        })
    })
}

#[inline]
fn value<T: Sealed>(d: &ArrayData<'_, T>, idx: usize, c: usize) -> T {
    let re = d.re.get(idx, c);
    let im = || d.im.as_ref().map_or(re, |f| f.get(idx, c));
    match d.part {
        Part::Real | Part::Re => re,
        Part::Im => im(),
        Part::Mag => T::mag(re, im()),
        Part::Phase => T::phase(re, im()),
    }
}

fn write_ascii<W: Write, T: Sealed>(w: &mut W, values: impl Iterator<Item = T>) -> Result<()> {
    let mut line = String::new();
    for (n, v) in values.enumerate() {
        if n % ASCII_PER_LINE == 0 {
            if n > 0 {
                line.push('\n');
                w.write_all(line.as_bytes())?;
                line.clear();
            }
            line.push_str("          ");
        } else {
            line.push(' ');
        }
        write!(line, "{v:e}").ok();
    }
    if !line.is_empty() {
        line.push('\n');
        w.write_all(line.as_bytes())?;
    }
    Ok(())
}

/// Escapes a string for use inside a double-quoted XML attribute.
pub(crate) fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\n' => out.push_str("&#10;"),
            '\r' => out.push_str("&#13;"),
            '\t' => out.push_str("&#9;"),
            c => out.push(c),
        }
    }
    out
}

/// One dataset of a [`PvdCollection`].
#[derive(Debug, Clone, PartialEq)]
pub struct PvdEntry {
    /// Time step (or frequency) of the dataset.
    pub timestep: f64,
    /// Part index, for datasets split into several files per time step.
    pub part: u32,
    /// File name, relative to the directory of the `.pvd` file.
    pub file: String,
}

/// Writer for a ParaView `.pvd` collection: a list of dataset files (here
/// `.vtr` files) each tagged with a time step, so ParaView opens them as one
/// time series.
///
/// The time step can be any increasing quantity; for frequency-domain dumps
/// use the frequency. File names are written as given and are resolved by
/// readers relative to the directory containing the `.pvd` file.
///
/// ```
/// use ems::io::PvdCollection;
///
/// let mut pvd = PvdCollection::new();
/// pvd.push(0.0, "E_0000.vtr");
/// pvd.push(1e-11, "E_0001.vtr");
/// let mut out = Vec::new();
/// pvd.write_to(&mut out)?;
/// assert!(String::from_utf8(out)?.contains(r#"file="E_0001.vtr""#));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PvdCollection {
    entries: Vec<PvdEntry>,
}

impl PvdCollection {
    /// Creates an empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `file` (part 0) at time step `timestep`.
    pub fn push(&mut self, timestep: f64, file: impl Into<String>) -> &mut Self {
        self.push_part(timestep, 0, file)
    }

    /// Adds `file` as part `part` of time step `timestep`.
    pub fn push_part(&mut self, timestep: f64, part: u32, file: impl Into<String>) -> &mut Self {
        self.entries.push(PvdEntry {
            timestep,
            part,
            file: file.into(),
        });
        self
    }

    /// The datasets added so far.
    pub fn entries(&self) -> &[PvdEntry] {
        &self.entries
    }

    /// Writes the collection to `w`. Fails with [`Error::Invalid`] on a
    /// non-finite time step or an empty file name.
    pub fn write_to<W: Write>(&self, mut w: W) -> Result<()> {
        self.validate()?;
        self.write_unchecked(&mut w)
    }

    /// Writes the collection to `path` (validated before the file is
    /// created).
    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let mut w = BufWriter::new(File::create(path)?);
        self.write_unchecked(&mut w)?;
        w.flush()?;
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        for e in &self.entries {
            if !e.timestep.is_finite() {
                return Err(Error::Invalid(format!(
                    "pvd: non-finite time step for {:?}",
                    e.file
                )));
            }
            if e.file.is_empty() {
                return Err(Error::Invalid("pvd: empty file name".into()));
            }
        }
        Ok(())
    }

    fn write_unchecked<W: Write>(&self, w: &mut W) -> Result<()> {
        writeln!(w, r#"<?xml version="1.0"?>"#)?;
        writeln!(
            w,
            r#"<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">"#
        )?;
        writeln!(w, "  <Collection>")?;
        for e in &self.entries {
            writeln!(
                w,
                r#"    <DataSet timestep="{:e}" group="" part="{}" file="{}"/>"#,
                e.timestep,
                e.part,
                escape(&e.file)
            )?;
        }
        writeln!(w, "  </Collection>")?;
        writeln!(w, "</VTKFile>")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
