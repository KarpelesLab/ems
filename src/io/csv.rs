//! Delimited text writer for time series and frequency series.
//!
//! The output is a table of `f64` columns:
//!
//! ```text
//! % probe: port1 voltage
//! % dt = 1e-12
//! t,u
//! 0e0,0e0
//! 1e-12,2.5e-3
//! ```
//!
//! * optional metadata lines first, each starting with a configurable comment
//!   prefix (`%` by default, `#` is the other common choice);
//! * a header row with the column names (plain, comment-prefixed, or
//!   omitted — see [`HeaderStyle`]);
//! * one row per sample, values separated by the delimiter (`,` by default).
//!
//! Values are written in Rust's shortest round-trip exponent notation
//! (`{:e}`, e.g. `1.5e-9`), so parsing a value back with
//! `str::parse::<f64>()` returns exactly the written `f64`. Non-finite values
//! are written as `NaN`, `inf` and `-inf`.
//!
//! Column names are written as given (no quoting); avoid names containing the
//! delimiter or line breaks.
//!
//! # Example
//!
//! ```
//! use ems::io::CsvWriter;
//!
//! let t = [0.0, 1e-12, 2e-12];
//! let u = [0.0, 0.5, -0.25];
//! let mut out = Vec::new();
//! CsvWriter::new()
//!     .comment("port 1 voltage")
//!     .column("t", &t)
//!     .column("u", &u)
//!     .write_to(&mut out)?;
//! assert_eq!(
//!     String::from_utf8(out).unwrap(),
//!     "% port 1 voltage\nt,u\n0e0,0e0\n1e-12,5e-1\n2e-12,-2.5e-1\n"
//! );
//! # Ok::<(), ems::Error>(())
//! ```

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::{Error, Result};

/// How the header row (column names) is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[non_exhaustive]
pub enum HeaderStyle {
    /// A plain row of names, e.g. `t,u` (the default; what spreadsheet and
    /// CSV readers expect).
    #[default]
    Plain,
    /// The row of names behind the comment prefix, e.g. `% t,u`, so tools
    /// that skip comment lines see a purely numeric table.
    Commented,
    /// No header row.
    None,
}

/// Builder for a delimited table of `f64` columns. See the
/// [module documentation](self).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CsvWriter<'a> {
    columns: Vec<(String, &'a [f64])>,
    comments: Vec<String>,
    delimiter: String,
    comment_prefix: String,
    header: HeaderStyle,
}

impl Default for CsvWriter<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> CsvWriter<'a> {
    /// Creates an empty table with delimiter `,`, comment prefix `%` and a
    /// plain header row.
    pub fn new() -> Self {
        CsvWriter {
            columns: Vec::new(),
            comments: Vec::new(),
            delimiter: ",".into(),
            comment_prefix: "%".into(),
            header: HeaderStyle::Plain,
        }
    }

    /// Appends a column. All columns must have the same length.
    pub fn column(mut self, name: impl Into<String>, values: &'a [f64]) -> Self {
        self.columns.push((name.into(), values));
        self
    }

    /// Adds a metadata line, written before the header behind the comment
    /// prefix and a space. A text containing line breaks becomes several
    /// comment lines.
    pub fn comment(mut self, text: impl Into<String>) -> Self {
        self.comments.push(text.into());
        self
    }

    /// Sets the value delimiter (default `,`; e.g. `"\t"` or `" "`).
    pub fn delimiter(mut self, delimiter: impl Into<String>) -> Self {
        self.delimiter = delimiter.into();
        self
    }

    /// Sets the comment prefix (default `%`; e.g. `#`).
    pub fn comment_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.comment_prefix = prefix.into();
        self
    }

    /// Sets how the header row is written (default [`HeaderStyle::Plain`]).
    pub fn header(mut self, style: HeaderStyle) -> Self {
        self.header = style;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.delimiter.is_empty() {
            return Err(Error::Invalid("CSV delimiter is empty".into()));
        }
        if (!self.comments.is_empty() || self.header == HeaderStyle::Commented)
            && self.comment_prefix.is_empty()
        {
            return Err(Error::Invalid("CSV comment prefix is empty".into()));
        }
        if let Some((first, rest)) = self.columns.split_first() {
            let n = first.1.len();
            if let Some((name, v)) = rest.iter().find(|(_, v)| v.len() != n) {
                return Err(Error::Invalid(format!(
                    "CSV column {name:?} has {} rows, column {:?} has {n}",
                    v.len(),
                    first.0
                )));
            }
        }
        Ok(())
    }

    /// Writes the table to `w`. Fails with [`Error::Invalid`] if the columns
    /// differ in length or the delimiter (or a needed comment prefix) is
    /// empty; nothing is written then.
    pub fn write_to<W: Write>(&self, mut w: W) -> Result<()> {
        self.validate()?;
        let mut line = String::new();
        for text in &self.comments {
            for l in text.lines() {
                line.clear();
                line.push_str(&self.comment_prefix);
                if !l.is_empty() {
                    line.push(' ');
                    line.push_str(l);
                }
                line.push('\n');
                w.write_all(line.as_bytes())?;
            }
        }
        if self.header != HeaderStyle::None && !self.columns.is_empty() {
            line.clear();
            if self.header == HeaderStyle::Commented {
                line.push_str(&self.comment_prefix);
                line.push(' ');
            }
            for (n, (name, _)) in self.columns.iter().enumerate() {
                if n > 0 {
                    line.push_str(&self.delimiter);
                }
                line.push_str(name);
            }
            line.push('\n');
            w.write_all(line.as_bytes())?;
        }
        let rows = self.columns.first().map_or(0, |c| c.1.len());
        for r in 0..rows {
            line.clear();
            for (n, (_, v)) in self.columns.iter().enumerate() {
                if n > 0 {
                    line.push_str(&self.delimiter);
                }
                write!(line, "{:e}", v[r]).ok();
            }
            line.push('\n');
            w.write_all(line.as_bytes())?;
        }
        Ok(())
    }

    /// Writes the table to `path` (buffered).
    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let mut w = BufWriter::new(File::create(path)?);
        self.write_to(&mut w)?;
        w.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(w: &CsvWriter<'_>) -> String {
        let mut out = Vec::new();
        w.write_to(&mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn basic_layout() {
        let f = [1e9, 2e9];
        let re = [0.5, -1.0];
        let im = [0.0, 3.25e-7];
        let s = render(
            &CsvWriter::new()
                .comment_prefix("#")
                .comment("S11\nZ0 = 50")
                .delimiter("\t")
                .column("f", &f)
                .column("re", &re)
                .column("im", &im),
        );
        assert_eq!(
            s,
            "# S11\n# Z0 = 50\nf\tre\tim\n1e9\t5e-1\t0e0\n2e9\t-1e0\t3.25e-7\n"
        );
    }

    #[test]
    fn header_styles() {
        let t = [1.0];
        let base = CsvWriter::new().column("t", &t);
        assert_eq!(render(&base.clone()), "t\n1e0\n");
        assert_eq!(
            render(&base.clone().header(HeaderStyle::Commented)),
            "% t\n1e0\n"
        );
        assert_eq!(render(&base.header(HeaderStyle::None)), "1e0\n");
    }

    #[test]
    fn float_round_trip() {
        let vals = [
            0.0,
            -0.0,
            1.0,
            0.1,
            1.0 / 3.0,
            std::f64::consts::PI,
            f64::MIN_POSITIVE,
            f64::MAX,
            -f64::EPSILON,
            5e-324,
            1.234_567_890_123_456_7e-300,
            299_792_458.0,
        ];
        let s = render(
            &CsvWriter::new()
                .header(HeaderStyle::None)
                .column("v", &vals),
        );
        let back: Vec<f64> = s.lines().map(|l| l.parse().unwrap()).collect();
        assert_eq!(back.len(), vals.len());
        for (a, b) in vals.iter().zip(&back) {
            assert_eq!(a.to_bits(), b.to_bits(), "{a:e} vs {b:e}");
        }
    }

    #[test]
    fn non_finite() {
        let v = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let s = render(&CsvWriter::new().header(HeaderStyle::None).column("v", &v));
        assert_eq!(s, "NaN\ninf\n-inf\n");
        let back: Vec<f64> = s.lines().map(|l| l.parse().unwrap()).collect();
        assert!(back[0].is_nan() && back[1] == f64::INFINITY && back[2] == f64::NEG_INFINITY);
    }

    #[test]
    fn errors() {
        let a = [1.0, 2.0];
        let b = [1.0];
        let mut out = Vec::new();
        let e = CsvWriter::new()
            .column("a", &a)
            .column("b", &b)
            .write_to(&mut out);
        assert!(matches!(e, Err(Error::Invalid(_))));
        assert!(out.is_empty());
        let e = CsvWriter::new()
            .delimiter("")
            .column("a", &a)
            .write_to(&mut out);
        assert!(matches!(e, Err(Error::Invalid(_))));
        let e = CsvWriter::new()
            .comment_prefix("")
            .comment("x")
            .write_to(&mut out);
        assert!(matches!(e, Err(Error::Invalid(_))));
    }

    #[test]
    fn write_to_path() {
        let dir = std::env::temp_dir().join(format!("ems-csv-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("probe.csv");
        let t = [0.0, 1.0];
        CsvWriter::new().column("t", &t).write(&path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "t\n0e0\n1e0\n");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
