//! openEMS probe text files: writer and reader.
//!
//! openEMS writes one text file per `ProbeBox`, named after the property
//! (no extension), and with `FD_Samples` a second one, `<name>_FD`. As
//! observed with openEMS v0.37:
//!
//! ```text
//! % time-domain voltage integration by openEMS v0.37.0-19-gc3a956b @Tue Oct 20 08:53:42 2026
//! % start-coordinates: (0.012,0.01,0.009) m -> [12,10,9]
//! % stop-coordinates: (0.012,0.01,0.011) m -> [12,10,11]
//! % t/s    voltage
//! 0    0
//! 4.04424970893e-11    -8.83796715812e-08
//! ```
//!
//! (the columns are separated by tabs).
//!
//! * The first line names the domain (`time-domain` / `frequency-domain`)
//!   and the probe kind (`voltage integration`, `current integration`,
//!   `electric field probe`, `magnetic field probe`, `voltage mode
//!   matching`, `current mode matching`), then the program and a date. ems writes `by ems v<version> @<date>`; readers should not
//!   rely on that line beyond its first words.
//! * The coordinate lines give the snapped probe corners in metres and
//!   their mesh indices ([`ProbeLocation`]: primary node indices for
//!   voltage and E-field probes, dual indices for current and H-field
//!   probes).
//! * Columns are separated by a tab. Time-domain values are printed with
//!   12 significant digits (C `%.12g` style), frequency-domain values and
//!   coordinates with 6 (`%g`).
//! * Time-domain columns: `t/s` and `voltage` / `current`, or the three
//!   field components `Ex/(V/m) Ey/(V/m) Ez/(V/m)` (`Hx/(A/m)` …).
//!   Waveguide probes (`ProbeBox` types 10 and 11, "mode matching") have
//!   two: `voltage` / `current` and `mode_purity` (see
//!   [`ProbeResult::mode_purity`]); their coordinate lines give the range
//!   of the probe's sample nodes.
//!   Frequency-domain files have a second header line `%\treal\timag` and
//!   a real and an imaginary column per quantity after `f/Hz` (for
//!   waveguide probes also the transform of the purity series, as openEMS
//!   writes it), e.g.
//!
//! ```text
//! % frequency-domain voltage mode matching by openEMS v0.37.0-19-gc3a956b @Tue Oct 20 13:34:58 2026
//! % start-coordinates: (0.001,0.001,0.005) m -> [1,1,5]
//! % stop-coordinates: (0.007,0.007,0.005) m -> [7,7,5]
//! % f/Hz    voltage        mode_purity
//! %    real    imag    real    imag
//! 3e+09    6.34924e-25    2.20337e-24    -2.44807e-27    -8.49551e-27
//! ```
//!
//! # Sampling (what the values are)
//!
//! openEMS writes the time-domain series every `K_t` steps (see
//! [`Results::openems_probe_interval_of`]; a probe's own `OverSampling`
//! replaces the `<FDTD>` one), starting with the initial state:
//! E-type probes (voltage, E field) at `t = k·dt` from `k = 0`, H-type
//! probes (current, H field) at `t = (k + ½)·dt`, for `k` a multiple of
//! `K_t`. ems records every step and writes the same rows: the E-type
//! initial state (all zero) is written as the row at `t = 0`. openEMS's
//! H-type files also have the state after the last step `N`, at
//! `(N + ½)·dt` (when `N` is a multiple of `K_t`); after
//! [`Simulation::run`](crate::Simulation::run) ems evaluates that H update
//! locally (without CPML and H-source corrections, not in Bloch runs and
//! where [`Probe::lookahead`](crate::engine::Probe::lookahead) has no
//! value) and writes it as well, so the files have the same rows.
//!
//! The frequency-domain value at `f` is
//!
//! ```text
//! X(f) = 2 · Σ_k x(t_k) · e^{−j2πf·t_k} · K_n·dt        (k multiple of K_n)
//! ```
//!
//! over the samples every `K_n` steps (the Nyquist interval,
//! [`Results::openems_nyquist_interval`]), with the samples' own times
//! `t_k` (established by comparison with openEMS output; note the factor
//! 2). [`ProbeResult::dft`] is the plain, every-step DFT
//! `Σ x(t_n)·e^{−j2πf·t_n}·dt` instead.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::post::Complex64;
use crate::sim::{ProbeKind, ProbeLocation, ProbeResult, ProbeValues, Results};
use crate::{Error, Result};

/// Time-domain or frequency-domain file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeDomain {
    /// `<name>`: samples over time.
    Time,
    /// `<name>_FD`: complex values at the `FD_Samples` frequencies.
    Frequency,
}

/// What an openEMS probe file holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeFileKind {
    /// `voltage integration` \[V\].
    Voltage,
    /// `current integration` \[A\].
    Current,
    /// `electric field probe` \[V/m\], three components.
    EField,
    /// `magnetic field probe` \[A/m\], three components.
    HField,
    /// `voltage mode matching` (waveguide voltage probe): the voltage
    /// \[V\] and the mode purity.
    VoltageModeMatching,
    /// `current mode matching` (waveguide current probe): the current
    /// \[A\] and the mode purity.
    CurrentModeMatching,
}

impl ProbeFileKind {
    fn title(self) -> &'static str {
        match self {
            ProbeFileKind::Voltage => "voltage integration",
            ProbeFileKind::Current => "current integration",
            ProbeFileKind::EField => "electric field probe",
            ProbeFileKind::HField => "magnetic field probe",
            ProbeFileKind::VoltageModeMatching => "voltage mode matching",
            ProbeFileKind::CurrentModeMatching => "current mode matching",
        }
    }

    /// The file kind of a probe result (mode matching for probes with a
    /// [mode purity](ProbeResult::mode_purity)).
    pub fn of(p: &ProbeResult) -> Self {
        match (p.kind(), p.mode_purity().is_some()) {
            (ProbeKind::Voltage, true) => ProbeFileKind::VoltageModeMatching,
            (ProbeKind::Current, true) => ProbeFileKind::CurrentModeMatching,
            (k, _) => k.into(),
        }
    }

    /// Whether this is a waveguide probe file (value and mode purity).
    pub fn is_mode_matching(self) -> bool {
        matches!(
            self,
            ProbeFileKind::VoltageModeMatching | ProbeFileKind::CurrentModeMatching
        )
    }

    fn columns(self) -> &'static [&'static str] {
        match self {
            ProbeFileKind::Voltage => &["voltage"],
            ProbeFileKind::Current => &["current"],
            ProbeFileKind::EField => &["Ex/(V/m)", "Ey/(V/m)", "Ez/(V/m)"],
            ProbeFileKind::HField => &["Hx/(A/m)", "Hy/(A/m)", "Hz/(A/m)"],
            ProbeFileKind::VoltageModeMatching => &["voltage", "mode_purity"],
            ProbeFileKind::CurrentModeMatching => &["current", "mode_purity"],
        }
    }

    /// Number of quantities per row (1, 2 for mode matching, or 3).
    pub fn components(self) -> usize {
        self.columns().len()
    }

    /// Whether the quantity lives at the H times `(k + ½)·dt`.
    pub fn is_magnetic(self) -> bool {
        matches!(
            self,
            ProbeFileKind::Current | ProbeFileKind::HField | ProbeFileKind::CurrentModeMatching
        )
    }
}

impl From<ProbeKind> for ProbeFileKind {
    fn from(k: ProbeKind) -> Self {
        match k {
            ProbeKind::Voltage => ProbeFileKind::Voltage,
            ProbeKind::Current => ProbeFileKind::Current,
            ProbeKind::EField => ProbeFileKind::EField,
            ProbeKind::HField => ProbeFileKind::HField,
        }
    }
}

/// The contents of one openEMS probe file.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeFile {
    /// Time or frequency domain.
    pub domain: ProbeDomain,
    /// The quantity.
    pub kind: ProbeFileKind,
    /// The snapped corners, if the file has coordinate lines.
    pub location: Option<ProbeLocation>,
    /// The program / date part of the first line (after `by `), e.g.
    /// `openEMS v0.37.0-19-gc3a956b @Tue Oct 20 08:53:42 2026`.
    pub generator: String,
    /// Numeric rows: `t` or `f` first, then the values (time domain: one
    /// per component; frequency domain: real and imaginary part per
    /// component).
    pub rows: Vec<Vec<f64>>,
}

impl ProbeFile {
    /// Parses the text of a probe file.
    ///
    /// # Errors
    /// [`Error::Invalid`] if the first line does not name a known domain
    /// and kind, or a row is not numeric or has the wrong column count.
    pub fn parse(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        let first = lines
            .next()
            .ok_or_else(|| Error::Invalid("empty probe file".into()))?;
        let head = first.trim_start_matches('%').trim();
        let domain = if head.starts_with("time-domain") {
            ProbeDomain::Time
        } else if head.starts_with("frequency-domain") {
            ProbeDomain::Frequency
        } else {
            return Err(Error::Invalid(format!(
                "not an openEMS probe file: {first:?}"
            )));
        };
        let kind = [
            ProbeFileKind::Voltage,
            ProbeFileKind::Current,
            ProbeFileKind::EField,
            ProbeFileKind::HField,
            ProbeFileKind::VoltageModeMatching,
            ProbeFileKind::CurrentModeMatching,
        ]
        .into_iter()
        .find(|k| head.contains(k.title()))
        .ok_or_else(|| Error::Invalid(format!("unknown probe kind in {first:?}")))?;
        let generator = head
            .split_once(" by ")
            .map_or(String::new(), |(_, g)| g.to_string());
        let (mut start, mut stop) = (None, None);
        let width = 1 + kind.components() * if domain == ProbeDomain::Time { 1 } else { 2 };
        let mut rows = Vec::new();
        for (no, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(c) = line.strip_prefix('%') {
                let c = c.trim();
                if let Some(v) = c.strip_prefix("start-coordinates:") {
                    start = Some(parse_location(v)?);
                } else if let Some(v) = c.strip_prefix("stop-coordinates:") {
                    stop = Some(parse_location(v)?);
                }
                continue;
            }
            let row = line
                .split_whitespace()
                .map(|v| {
                    v.parse::<f64>().map_err(|_| {
                        Error::Invalid(format!("probe file line {}: {v:?} is not a number", no + 2))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if row.len() != width {
                return Err(Error::Invalid(format!(
                    "probe file line {}: {} columns, expected {width}",
                    no + 2,
                    row.len()
                )));
            }
            rows.push(row);
        }
        let location = match (start, stop) {
            (Some((start_coords, start)), Some((stop_coords, stop))) => Some(ProbeLocation {
                start,
                stop,
                start_coords,
                stop_coords,
            }),
            _ => None,
        };
        Ok(ProbeFile {
            domain,
            kind,
            location,
            generator,
            rows,
        })
    }

    /// The first column: times \[s\] or frequencies \[Hz\].
    pub fn abscissa(&self) -> Vec<f64> {
        self.rows.iter().map(|r| r[0]).collect()
    }

    /// Time-domain values: column `c` (0-based, after the time) of every
    /// row.
    pub fn column(&self, c: usize) -> Vec<f64> {
        self.rows.iter().map(|r| r[1 + c]).collect()
    }

    /// Time-domain values as [`ProbeValues`] (`None` for a frequency-domain
    /// file; for a waveguide probe the voltage or current, see
    /// [`mode_purity`](Self::mode_purity)).
    pub fn values(&self) -> Option<ProbeValues> {
        if self.domain != ProbeDomain::Time {
            return None;
        }
        Some(if self.kind.components() == 3 {
            ProbeValues::Vector(self.rows.iter().map(|r| [r[1], r[2], r[3]]).collect())
        } else {
            ProbeValues::Scalar(self.column(0))
        })
    }

    /// The `mode_purity` column of a time-domain waveguide probe file
    /// (`None` for other files).
    pub fn mode_purity(&self) -> Option<Vec<f64>> {
        (self.domain == ProbeDomain::Time && self.kind.is_mode_matching()).then(|| self.column(1))
    }

    /// Frequency-domain values: per row, the frequency and the complex
    /// value of each component (`None` for a time-domain file).
    pub fn spectrum(&self) -> Option<Vec<(f64, Vec<Complex64>)>> {
        if self.domain != ProbeDomain::Frequency {
            return None;
        }
        Some(
            self.rows
                .iter()
                .map(|r| {
                    let v = r[1..]
                        .chunks(2)
                        .map(|c| Complex64::new(c[0], c[1]))
                        .collect();
                    (r[0], v)
                })
                .collect(),
        )
    }

    /// The file text, formatted like openEMS (see the
    /// [module docs](self)).
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let dom = match self.domain {
            ProbeDomain::Time => "time-domain",
            ProbeDomain::Frequency => "frequency-domain",
        };
        let _ = writeln!(s, "% {dom} {} by {}", self.kind.title(), self.generator);
        if let Some(l) = &self.location {
            for (what, c, i) in [
                ("start", l.start_coords, l.start),
                ("stop", l.stop_coords, l.stop),
            ] {
                let _ = writeln!(
                    s,
                    "% {what}-coordinates: ({},{},{}) m -> [{},{},{}]",
                    fmt_g(c[0], 6),
                    fmt_g(c[1], 6),
                    fmt_g(c[2], 6),
                    i[0],
                    i[1],
                    i[2]
                );
            }
        }
        let (prec, abscissa) = match self.domain {
            ProbeDomain::Time => (12, "t/s"),
            ProbeDomain::Frequency => (6, "f/Hz"),
        };
        s.push_str("% ");
        s.push_str(abscissa);
        for c in self.kind.columns() {
            s.push('\t');
            s.push_str(c);
            if self.domain == ProbeDomain::Frequency {
                s.push('\t');
            }
        }
        s.push('\n');
        if self.domain == ProbeDomain::Frequency {
            s.push('%');
            for _ in 0..self.kind.components() {
                s.push_str("\treal\timag");
            }
            s.push('\n');
        }
        for r in &self.rows {
            for (k, v) in r.iter().enumerate() {
                if k > 0 {
                    s.push('\t');
                }
                s.push_str(&fmt_g(*v, prec));
            }
            s.push('\n');
        }
        s
    }

    /// The time-domain file of `p` with openEMS's sampling every
    /// `interval` steps (see the [module docs](self#sampling-what-the-values-are)).
    pub fn time_domain(p: &ProbeResult, interval: u64) -> Self {
        let kind = ProbeFileKind::of(p);
        let samples = samples(p);
        let interval = interval.max(1);
        let rows = samples
            .iter()
            .filter(|s| s.0 % interval == 0)
            .map(|(_, t, v)| {
                let mut r = vec![*t];
                r.extend_from_slice(v);
                r
            })
            .collect();
        ProbeFile {
            domain: ProbeDomain::Time,
            kind,
            location: p.location().copied(),
            generator: generator(),
            rows,
        }
    }

    /// The frequency-domain file of `p` at its `FD_Samples` frequencies,
    /// from the samples every `nyquist` steps (see the
    /// [module docs](self#sampling-what-the-values-are)); `None` without
    /// frequencies.
    pub fn frequency_domain(p: &ProbeResult, nyquist: u64) -> Option<Self> {
        if p.frequencies().is_empty() {
            return None;
        }
        let kind = ProbeFileKind::of(p);
        let n = nyquist.max(1);
        let picked: Vec<_> = samples(p).into_iter().filter(|s| s.0 % n == 0).collect();
        let w = 2.0 * n as f64 * p.dt();
        let rows = p
            .frequencies()
            .iter()
            .map(|&f| {
                let mut acc = vec![Complex64::new(0.0, 0.0); kind.components()];
                for (_, t, v) in &picked {
                    let e = Complex64::from_polar(w, -core::f64::consts::TAU * f * t);
                    for (a, x) in acc.iter_mut().zip(v) {
                        *a += e * *x;
                    }
                }
                let mut r = vec![f];
                for a in acc {
                    r.push(a.re);
                    r.push(a.im);
                }
                r
            })
            .collect();
        Some(ProbeFile {
            domain: ProbeDomain::Frequency,
            kind,
            location: p.location().copied(),
            generator: generator(),
            rows,
        })
    }
}

/// `(step k, time, values)` of every recorded sample of `p`, with the
/// E-type initial state at `t = 0` added when the recording starts at the
/// first step.
fn samples(p: &ProbeResult) -> Vec<(u64, f64, Vec<f64>)> {
    let kind = ProbeFileKind::of(p);
    let dt = p.dt();
    let offset = if kind.is_magnetic() { 0.5 } else { 0.0 };
    let vals: Vec<Vec<f64>> = match (p.values(), p.mode_purity()) {
        (ProbeValues::Scalar(v), Some(q)) => v.iter().zip(q).map(|(&x, &q)| vec![x, q]).collect(),
        (ProbeValues::Scalar(v), None) => v.iter().map(|&x| vec![x]).collect(),
        (ProbeValues::Vector(v), _) => v.iter().map(|x| x.to_vec()).collect(),
    };
    let mut out = Vec::with_capacity(vals.len() + 2);
    let step = |t: f64| (t / dt - offset).round().max(0.0) as u64;
    if !kind.is_magnetic() && p.times().first().is_some_and(|&t| step(t) == 1) {
        out.push((0, 0.0, vec![0.0; kind.components()]));
    }
    for (&t, v) in p.times().iter().zip(vals) {
        out.push((step(t), t, v));
    }
    if let Some((t, v, q)) = &p.final_sample
        && out.last().is_none_or(|l| step(*t) > l.0)
    {
        let mut v = v.clone();
        if kind.is_mode_matching() {
            v.push(q.unwrap_or(f64::NAN));
        }
        out.push((step(*t), *t, v));
    }
    out
}

/// `ems v<version> @<date>` (UTC).
fn generator() -> String {
    format!(
        "ems v{} @{}",
        env!("CARGO_PKG_VERSION"),
        asctime_utc(SystemTime::now())
    )
}

/// C `asctime`-style date (`Tue Oct 20 08:53:42 2026`) of `t` in UTC.
fn asctime_utc(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (H. Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    const WD: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    const MO: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    format!(
        "{} {} {:2} {:02}:{:02}:{:02} {y}",
        WD[days.rem_euclid(7) as usize],
        MO[(m - 1) as usize],
        d,
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

/// `(coordinates, indices)` of `(x,y,z) m -> [i,j,k]`.
fn parse_location(v: &str) -> Result<([f64; 3], [usize; 3])> {
    let bad = || Error::Invalid(format!("bad probe coordinates {v:?}"));
    let (c, i) = v.split_once("->").ok_or_else(bad)?;
    let inner = |s: &str, open: char, close: char| -> Option<String> {
        let a = s.find(open)?;
        let b = s.rfind(close)?;
        Some(s[a + 1..b].to_string())
    };
    let c = inner(c, '(', ')').ok_or_else(bad)?;
    let i = inner(i, '[', ']').ok_or_else(bad)?;
    let cs: Vec<f64> = c
        .split(',')
        .map(|x| x.trim().parse().map_err(|_| bad()))
        .collect::<Result<_>>()?;
    let is: Vec<usize> = i
        .split(',')
        .map(|x| x.trim().parse().map_err(|_| bad()))
        .collect::<Result<_>>()?;
    if cs.len() != 3 || is.len() != 3 {
        return Err(bad());
    }
    Ok(([cs[0], cs[1], cs[2]], [is[0], is[1], is[2]]))
}

/// C `printf("%.{prec}g")`: `prec` significant digits, trailing zeros
/// removed, exponent form for exponents `< −4` or `≥ prec` (two-digit
/// exponent with sign, e.g. `1e+09`), the number format of openEMS's text
/// files and console.
///
/// ```
/// use ems::io::openems::probes::format_g;
/// assert_eq!(format_g(1.92583319e-12, 6), "1.92583e-12");
/// assert_eq!(format_g(2.5e9, 6), "2.5e+09");
/// assert_eq!(format_g(0.012, 6), "0.012");
/// ```
pub fn format_g(v: f64, prec: usize) -> String {
    fmt_g(v, prec)
}

pub(crate) fn fmt_g(v: f64, prec: usize) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() { "-0" } else { "0" }.into();
    }
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf" } else { "inf" }.into();
    }
    let prec = prec.max(1);
    let sci = format!("{:.*e}", prec - 1, v);
    let (mant, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("integer exponent");
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if exp < -4 || exp >= prec as i32 {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip(mant), exp.abs())
    } else {
        let decimals = (prec as i32 - 1 - exp).max(0) as usize;
        strip(&format!("{v:.decimals$}"))
    }
}

/// Reads an openEMS probe file.
///
/// # Errors
/// I/O errors and those of [`ProbeFile::parse`].
pub fn read_probe_file(path: impl AsRef<Path>) -> Result<ProbeFile> {
    let path = path.as_ref();
    let text =
        std::fs::read_to_string(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    ProbeFile::parse(&text)
}

/// Writes the openEMS probe files of `p` into `dir`: `<name>` (time
/// domain, every `interval` steps) and, with `FD_Samples`, `<name>_FD`
/// (from the samples every `nyquist` steps). Returns the paths written.
///
/// # Errors
/// I/O errors; [`Error::Invalid`] for a probe name that is not a plain
/// file name.
pub fn write_probe(
    p: &ProbeResult,
    dir: impl AsRef<Path>,
    interval: u64,
    nyquist: u64,
) -> Result<Vec<PathBuf>> {
    let name = p.name();
    if name.is_empty() || name.contains(['/', '\\']) || name == "." || name == ".." {
        return Err(Error::Invalid(format!(
            "probe name {name:?} is not a plain file name"
        )));
    }
    let dir = dir.as_ref();
    let mut out = Vec::new();
    let td = dir.join(name);
    write_text(&td, &ProbeFile::time_domain(p, interval).to_text())?;
    out.push(td);
    if let Some(fd) = ProbeFile::frequency_domain(p, nyquist) {
        let path = dir.join(format!("{name}_FD"));
        write_text(&path, &fd.to_text())?;
        out.push(path);
    }
    Ok(out)
}

/// Writes the openEMS probe files of every model probe of `results` into
/// `dir` (created if needed), with openEMS's sampling intervals
/// ([`Results::openems_probe_interval_of`],
/// [`Results::openems_nyquist_interval`]); see [`write_probe`]. Lumped
/// element currents (which openEMS does not write) are skipped. Returns
/// the paths written.
///
/// ```no_run
/// # #[cfg(feature = "xml")] {
/// let mut sim = ems::Simulation::from_xml("model.xml")?;
/// let res = sim.run()?;
/// ems::io::openems::write_probes(&res, "model_results")?;
/// # }
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Errors
/// See [`write_probe`].
pub fn write_probes(results: &Results, dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).map_err(|e| Error::Io(format!("{}: {e}", dir.display())))?;
    let n = results.openems_nyquist_interval();
    let mut out = Vec::new();
    for p in results.probes() {
        let k = results.openems_probe_interval_of(p);
        out.extend(write_probe(p, dir, k, n)?);
    }
    Ok(out)
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printf_g() {
        for (v, p, want) in [
            (4.04424970893e-11, 12, "4.04424970893e-11"),
            (0.00200000009499, 12, "0.00200000009499"),
            (1.0000000475, 12, "1.0000000475"),
            (-8.83796715812e-08, 12, "-8.83796715812e-08"),
            (9.62916597364e-13, 12, "9.62916597364e-13"),
            (215.330302715, 12, "215.330302715"),
            (1e9, 6, "1e+09"),
            (2.5e9, 6, "2.5e+09"),
            (-1.7692e-14, 6, "-1.7692e-14"),
            (0.012, 6, "0.012"),
            (0.01, 6, "0.01"),
            (0.0001, 6, "0.0001"),
            (0.00001, 6, "1e-05"),
            (123456.0, 6, "123456"),
            (1234567.0, 6, "1.23457e+06"),
            (0.0, 6, "0"),
            (-3.0, 6, "-3"),
        ] {
            assert_eq!(fmt_g(v, p), want, "{v} %.{p}g");
        }
    }

    #[test]
    fn asctime() {
        // 2026-10-20 08:53:42 UTC was a Tuesday.
        let t = UNIX_EPOCH + std::time::Duration::from_secs(1_792_486_422);
        assert_eq!(asctime_utc(t), "Tue Oct 20 08:53:42 2026");
        assert_eq!(asctime_utc(UNIX_EPOCH), "Thu Jan  1 00:00:00 1970");
    }

    #[test]
    fn parse_and_format_round_trip() {
        let text = "% time-domain voltage integration by openEMS v0.37.0-19-gc3a956b @Tue Oct 20 08:53:42 2026\n\
% start-coordinates: (0.012,0.01,0.009) m -> [12,10,9]\n\
% stop-coordinates: (0.012,0.01,0.011) m -> [12,10,11]\n\
% t/s\tvoltage\n\
0\t0\n\
4.04424970893e-11\t-8.83796715812e-08\n";
        let f = ProbeFile::parse(text).unwrap();
        assert_eq!(f.domain, ProbeDomain::Time);
        assert_eq!(f.kind, ProbeFileKind::Voltage);
        assert_eq!(
            f.generator,
            "openEMS v0.37.0-19-gc3a956b @Tue Oct 20 08:53:42 2026"
        );
        let l = f.location.unwrap();
        assert_eq!((l.start, l.stop), ([12, 10, 9], [12, 10, 11]));
        assert_eq!(l.stop_coords, [0.012, 0.01, 0.011]);
        assert_eq!(f.column(0), vec![0.0, -8.83796715812e-08]);
        assert_eq!(f.to_text(), text);

        let fd = "% frequency-domain electric field probe by openEMS v0.37.0-19-gc3a956b @Tue Oct 20 08:56:57 2026\n\
% start-coordinates: (0.013,0.01,0.01) m -> [13,10,10]\n\
% stop-coordinates: (0.013,0.01,0.01) m -> [13,10,10]\n\
% f/Hz\tEx/(V/m)\t\tEy/(V/m)\t\tEz/(V/m)\t\n\
%\treal\timag\treal\timag\treal\timag\n\
1e+09\t1.87806e-19\t2.68075e-19\t-2.34372e-18\t-2.3271e-19\t-3.19582e-12\t-1.11103e-12\n";
        let f = ProbeFile::parse(fd).unwrap();
        assert_eq!(
            (f.domain, f.kind),
            (ProbeDomain::Frequency, ProbeFileKind::EField)
        );
        let s = f.spectrum().unwrap();
        assert_eq!(s[0].0, 1e9);
        assert_eq!(s[0].1[2], Complex64::new(-3.19582e-12, -1.11103e-12));
        assert_eq!(f.to_text(), fd);
        let wg = "% time-domain current mode matching by openEMS v0.37.0-19-gc3a956b @Tue Oct 20 10:05:47 2026\n\
% start-coordinates: (0.001,0.001,0.03) m -> [1,1,30]\n\
% stop-coordinates: (0.019,0.009,0.03) m -> [19,9,30]\n\
% t/s\tcurrent\tmode_purity\n\
9e-13\t0\t0\n\
4.41e-11\t-1.59924365831e-18\t0.994456340029\n";
        let f = ProbeFile::parse(wg).unwrap();
        assert_eq!(f.kind, ProbeFileKind::CurrentModeMatching);
        assert!(f.kind.is_magnetic() && f.kind.is_mode_matching());
        assert_eq!(f.mode_purity().unwrap(), vec![0.0, 0.994456340029]);
        assert_eq!(
            f.values(),
            Some(ProbeValues::Scalar(vec![0.0, -1.59924365831e-18]))
        );
        assert_eq!(f.to_text(), wg);
        let wfd = "% frequency-domain voltage mode matching by openEMS v0.37.0-19-gc3a956b @Tue Oct 20 13:34:58 2026\n\
% start-coordinates: (0.001,0.001,0.005) m -> [1,1,5]\n\
% stop-coordinates: (0.007,0.007,0.005) m -> [7,7,5]\n\
% f/Hz\tvoltage\t\tmode_purity\t\n\
%\treal\timag\treal\timag\n\
3e+09\t6.34924e-25\t2.20337e-24\t-2.44807e-27\t-8.49551e-27\n";
        let f = ProbeFile::parse(wfd).unwrap();
        assert_eq!(f.kind, ProbeFileKind::VoltageModeMatching);
        assert_eq!(f.spectrum().unwrap()[0].1.len(), 2);
        assert_eq!(f.to_text(), wfd);
        assert!(ProbeFile::parse("% hello\n").is_err());
        assert!(ProbeFile::parse(&text.replace("\t0\n", "\t0\t1\n")).is_err());
    }
}
