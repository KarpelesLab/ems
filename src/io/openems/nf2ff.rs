//! The openEMS `nf2ff` tool: near-field to far-field transform of the box
//! dumps described by an nf2ff XML file (feature `hdf5`).
//!
//! openEMS computes radiation patterns in a separate program,
//! `nf2ff <file.xml>`, that reads the near-field dumps of an NF2FF box
//! (one E and one H HDF5 dump per face) and writes the far field to an
//! HDF5 file. This module reads the same XML, the same dump files and
//! writes the same output layout; [`run_xml_file`] is what the
//! `ems-nf2ff` binary does. Everything below was established on a
//! black-box openEMS v0.37 (`nf2ff` error messages, trial XML files and
//! synthetic dump files written with h5py; no openEMS source consulted);
//! the far fields agree with it to ~3e-7 relative, 5e-6 with `Eps_r` /
//! `Mue_r` (it computes in single precision somewhere), see
//! `tests/openems_nf2ff.rs`.
//!
//! # The XML file
//!
//! ```text
//! <nf2ff freq="f1,f2,..." Outfile="out.h5" [Radius="r"] [Center="x,y,z"]
//!        [Eps_r="e or e1,e2,..."] [Mue_r="m or m1,m2,..."] [Verbose="n"] [NumThreads="n"]>
//!   <theta>t1,t2,...</theta>            polar angles [rad]
//!   <phi>p1,p2,...</phi>                azimuth angles [rad]
//!   <Planes E_Field="e.h5" H_Field="h.h5"/>     one per box face
//!   <Mirror Dir="0|1|2" Type="PEC|PMC" Pos="p"/>   optional, any number
//! </nf2ff>
//! ```
//!
//! File names are relative to the working directory (not to the XML
//! file). `Radius` defaults to 1 m, `Center` to the origin, `Eps_r` and
//! `Mue_r` to 1 (one value for all frequencies, or one per frequency).
//!
//! # Input planes
//!
//! Each `Planes` element names an E and an H field dump in the openEMS
//! HDF5 layout ([`super::dump`]): `/Mesh/{x,y,z}` in metres (the
//! `mesh_scaling` attribute is not used) with one line along the face
//! normal, and either frequency-domain data `/FieldData/FD/f%d`
//! (`{r, i}` `[3, nx, ny, nz]`, frequencies in the `frequency` attribute
//! of the group; legacy `f%d_real` / `f%d_imag` `[3, nz, ny, nx]` also
//! read) or, when a requested frequency is not among them, time-domain
//! data `/FieldData/TD/*` (`[3, nx, ny, nz]`, `time` attribute), which
//! is transformed as `F(f) = 2·Σ_n x(t_n)·e^{−j2πf·t_n}·Δt` with `Δt` the
//! sample interval (the normalisation of openEMS frequency dumps). E and H
//! must be sampled on the same mesh lines. The requested frequencies must
//! match a stored one exactly (to 1e-9 relative).
//!
//! # Transform
//!
//! Surface equivalence (Balanis, *Antenna Theory*, §12.2–12.3; see
//! [`crate::nf2ff`]): on each plane `J = n̂ × H`, `M = −n̂ × E` with `n̂`
//! the face normal pointing away from `Center` (`+` when the plane
//! coordinate is at or above the centre's), integrated with trapezoidal
//! weights over the plane's mesh nodes:
//!
//! ```text
//! N = Σ J w e^{jk r̂·(r' − c)},   L = Σ M w e^{jk r̂·(r' − c)}
//! E_θ = −jk e^{−jkr}/(4πr) (L_φ + η N_θ),   E_φ = jk e^{−jkr}/(4πr) (L_θ − η N_φ)
//! ```
//!
//! with `k = 2πf√(ε_r μ_r)/c0`, `η = Z0√(μ_r/ε_r)` and `r` = `Radius`.
//! A mirror (`Dir` axis, plane at `Pos`) adds the image of every source,
//! images of images included: a PEC plane reverses the tangential J and
//! the normal M, a PMC plane the normal J and the tangential M.
//!
//! # Output file
//!
//! ```text
//! /                    @openEMS_HDF5_version f64 0.3
//! /Mesh                @MeshType f32 [1] = 2
//! /Mesh/theta, phi     f32 the angles as given [rad]
//! /Mesh/r              f32 [1] the radius
//! /nf2ff               @Frequency f32 [nf], @Prad f64 [nf], @Dmax f64 [nf]
//!                      (@Eps_r, @Mue_r f64 as given, if given)
//! /nf2ff/E_theta/FD/f%d, /nf2ff/E_phi/FD/f%d   {r: f64, i: f64} [ntheta, nphi]
//! /nf2ff/P_rad/FD/f%d  f64 [ntheta, nphi]
//! ```
//!
//! `P_rad` is the power density `(|E_θ|² + |E_φ|²)/(2η)` at the radius,
//! `Prad` the Poynting flux `½ Re Σ (E × H*)·n̂ w` through the planes
//! (doubled per mirror) and `Dmax = 4π r² max P_rad / Prad`.

use std::path::{Path, PathBuf};

use hdf5rs::{Complex, File};

use crate::constants::{C0, Z0};
use crate::post::Complex64;
use crate::{Error, Result};

/// A mirror plane of an nf2ff job (`<Mirror Dir Type Pos/>`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nf2ffMirror {
    /// Normal axis of the mirror plane (0 = x, 1 = y, 2 = z).
    pub axis: usize,
    /// `true` for a PEC plane, `false` for PMC.
    pub pec: bool,
    /// Position of the plane along `axis` \[m\].
    pub pos: f64,
}

/// One box face: an E and an H dump file (`<Planes E_Field H_Field/>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nf2ffPlaneFiles {
    /// The E-field dump.
    pub e_file: PathBuf,
    /// The H-field dump.
    pub h_file: PathBuf,
}

/// An nf2ff job as described by an openEMS nf2ff XML file (see the
/// [module docs](self#the-xml-file)).
#[derive(Debug, Clone, PartialEq)]
pub struct Nf2ffJob {
    /// Frequencies \[Hz\].
    pub frequencies: Vec<f64>,
    /// Output HDF5 file.
    pub outfile: PathBuf,
    /// Polar angles \[rad\].
    pub theta: Vec<f64>,
    /// Azimuth angles \[rad\].
    pub phi: Vec<f64>,
    /// Far-field radius \[m\] (default 1).
    pub radius: f64,
    /// Phase centre \[m\] (default origin).
    pub center: [f64; 3],
    /// Background relative permittivity, one value or one per frequency
    /// (empty: 1, not written to the output).
    pub eps_r: Vec<f64>,
    /// Background relative permeability, like [`eps_r`](Self::eps_r).
    pub mue_r: Vec<f64>,
    /// Mirror planes.
    pub mirrors: Vec<Nf2ffMirror>,
    /// The box faces.
    pub planes: Vec<Nf2ffPlaneFiles>,
    /// Verbosity (`Verbose`).
    pub verbose: u32,
    /// Requested thread count (`NumThreads`; the transform is cheap and
    /// runs single-threaded).
    pub threads: Option<usize>,
}

fn invalid(m: impl Into<String>) -> Error {
    Error::Invalid(m.into())
}

fn h5err(e: hdf5rs::Error) -> Error {
    Error::Io(format!("HDF5: {e}"))
}

/// Splits a list of numbers separated by commas, semicolons or blanks.
fn numbers(s: &str, what: &str) -> Result<Vec<f64>> {
    s.split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f64>()
                .map_err(|_| invalid(format!("nf2ff: bad number {t:?} in {what}")))
        })
        .collect()
}

impl Nf2ffJob {
    /// Parses an nf2ff XML document.
    ///
    /// # Errors
    /// Malformed XML, a root element other than `nf2ff`, or a missing
    /// frequency list, `Outfile`, `theta` or `phi` (the order in which
    /// openEMS checks them).
    pub fn from_xml_str(xml: &str) -> Result<Self> {
        let root = csx::io::dom::parse(xml).map_err(Error::Csx)?;
        if root.name != "nf2ff" {
            return Err(invalid("nf2ff: can't read nf2ff element"));
        }
        let frequencies = match root.attr("freq") {
            Some(s) => numbers(s, "freq")?,
            None => return Err(invalid("nf2ff: can't read frequency information")),
        };
        if frequencies.is_empty() {
            return Err(invalid("nf2ff: can't read frequency information"));
        }
        let outfile = root
            .attr("Outfile")
            .ok_or_else(|| invalid("nf2ff: can't read output file name"))?
            .into();
        let list = |name: &str| -> Result<Vec<f64>> {
            let el = root
                .child(name)
                .ok_or_else(|| invalid(format!("nf2ff: can't read {name} values")))?;
            let v = numbers(&el.text(), name)?;
            if v.is_empty() {
                return Err(invalid(format!("nf2ff: can't read {name} values")));
            }
            Ok(v)
        };
        let theta = list("theta")?;
        let phi = list("phi")?;
        let radius = match root.attr("Radius") {
            Some(s) => *numbers(s, "Radius")?
                .first()
                .ok_or_else(|| invalid("nf2ff: empty Radius"))?,
            None => 1.0,
        };
        let center = match root.attr("Center") {
            Some(s) => {
                let v = numbers(s, "Center")?;
                if v.len() != 3 {
                    return Err(invalid("nf2ff: Center needs three values"));
                }
                [v[0], v[1], v[2]]
            }
            None => [0.0; 3],
        };
        let opt_list = |name: &str| -> Result<Vec<f64>> {
            root.attr(name).map_or(Ok(Vec::new()), |s| numbers(s, name))
        };
        let eps_r = opt_list("Eps_r")?;
        let mue_r = opt_list("Mue_r")?;
        for (name, v) in [("Eps_r", &eps_r), ("Mue_r", &mue_r)] {
            if v.len() > 1 && v.len() != frequencies.len() {
                return Err(invalid(format!(
                    "nf2ff: {name} needs one value or one per frequency"
                )));
            }
        }
        let verbose = root
            .attr("Verbose")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let threads = root.attr("NumThreads").and_then(|s| s.trim().parse().ok());
        let mut planes = Vec::new();
        for p in root.children_named("Planes") {
            match (p.attr("E_Field"), p.attr("H_Field")) {
                (Some(e), Some(h)) => planes.push(Nf2ffPlaneFiles {
                    e_file: e.into(),
                    h_file: h.into(),
                }),
                _ => return Err(invalid("nf2ff: Planes needs E_Field and H_Field")),
            }
        }
        let mut mirrors = Vec::new();
        for m in root.children_named("Mirror") {
            let (Some(d), Some(t), Some(p)) = (m.attr("Dir"), m.attr("Type"), m.attr("Pos")) else {
                return Err(invalid("nf2ff: Mirror needs Dir, Type and Pos"));
            };
            let axis = match d.trim() {
                "0" => 0,
                "1" => 1,
                "2" => 2,
                _ => continue,
            };
            let pec = match t.trim() {
                "PEC" => true,
                "PMC" => false,
                _ => continue,
            };
            let pos = *numbers(p, "Pos")?
                .first()
                .ok_or_else(|| invalid("nf2ff: empty mirror Pos"))?;
            mirrors.push(Nf2ffMirror { axis, pec, pos });
        }
        Ok(Self {
            frequencies,
            outfile,
            theta,
            phi,
            radius,
            center,
            eps_r,
            mue_r,
            mirrors,
            planes,
            verbose,
            threads,
        })
    }

    /// Reads and parses an nf2ff XML file.
    ///
    /// # Errors
    /// I/O errors and those of [`from_xml_str`](Self::from_xml_str).
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self> {
        let s = std::fs::read_to_string(path.as_ref())
            .map_err(|e| Error::Io(format!("{}: {e}", path.as_ref().display())))?;
        Self::from_xml_str(&s)
    }

    fn material(v: &[f64], k: usize) -> f64 {
        match v.len() {
            0 => 1.0,
            1 => v[0],
            _ => v[k],
        }
    }
}

/// The near field of one plane at one frequency: E and H at the plane's
/// mesh nodes (x fastest-varying last: index `(i·ny + j)·nz + k`).
#[derive(Debug, Clone, PartialEq)]
pub struct NearFieldPlane {
    /// Mesh lines \[m\] (one along the normal).
    pub lines: [Vec<f64>; 3],
    /// E per node.
    pub e: Vec<[Complex64; 3]>,
    /// H per node.
    pub h: Vec<[Complex64; 3]>,
}

impl NearFieldPlane {
    /// The normal axis (the one with a single line).
    ///
    /// # Errors
    /// [`Error::Invalid`] unless exactly one axis has a single line.
    pub fn normal_axis(&self) -> Result<usize> {
        let single: Vec<usize> = (0..3).filter(|&a| self.lines[a].len() == 1).collect();
        match single[..] {
            [a] => Ok(a),
            _ => Err(invalid(format!(
                "nf2ff: a plane needs exactly one single-line axis, has mesh {}x{}x{}",
                self.lines[0].len(),
                self.lines[1].len(),
                self.lines[2].len()
            ))),
        }
    }
}

/// Mesh lines of a dump and its node vectors per requested frequency.
type DumpData = ([Vec<f64>; 3], Vec<Vec<[Complex64; 3]>>);

/// Field data of one dump file at the requested frequencies.
fn read_dump(path: &Path, freqs: &[f64]) -> Result<DumpData> {
    let f = File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let lines = ["x", "y", "z"].map(|n| {
        f.dataset(&format!("/Mesh/{n}"))
            .and_then(|d| d.read::<f64>())
    });
    let [x, y, z] = lines;
    let lines = [x.map_err(h5err)?, y.map_err(h5err)?, z.map_err(h5err)?];
    let [nx, ny, nz] = [lines[0].len(), lines[1].len(), lines[2].len()];
    let n = nx * ny * nz;
    let legacy = f
        .root()
        .and_then(|r| r.attribute_names())
        .map(|a| a.iter().any(|s| s == "legacy_fmt"))
        .unwrap_or(false);
    // Index of a node in the file's [3, a, b, c] array.
    let src = |c: usize, i: usize, j: usize, k: usize| -> usize {
        if legacy {
            ((c * nz + k) * ny + j) * nx + i
        } else {
            ((c * nx + i) * ny + j) * nz + k
        }
    };
    let reorder = |raw: &dyn Fn(usize) -> Complex64| -> Vec<[Complex64; 3]> {
        let mut v = Vec::with_capacity(n);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    v.push([0, 1, 2].map(|c| raw(src(c, i, j, k))));
                }
            }
        }
        v
    };
    let fd_freqs: Vec<f64> = f
        .group("/FieldData/FD")
        .and_then(|g| g.attribute("frequency"))
        .and_then(|a| a.read::<f64>())
        .unwrap_or_default();
    let mut out = Vec::with_capacity(freqs.len());
    let mut td: Option<Vec<(f64, Vec<f64>)>> = None;
    for &fr in freqs {
        if let Some(k) = fd_freqs
            .iter()
            .position(|&g| (g - fr).abs() <= 1e-9 * fr.abs().max(g.abs()))
        {
            let vals: Vec<Complex64> = if legacy {
                let re = f
                    .dataset(&format!("/FieldData/FD/f{k}_real"))
                    .and_then(|d| d.read::<f64>())
                    .map_err(h5err)?;
                let im = f
                    .dataset(&format!("/FieldData/FD/f{k}_imag"))
                    .and_then(|d| d.read::<f64>())
                    .map_err(h5err)?;
                re.iter()
                    .zip(&im)
                    .map(|(&a, &b)| Complex64::new(a, b))
                    .collect()
            } else {
                f.dataset(&format!("/FieldData/FD/f{k}"))
                    .and_then(|d| d.read::<Complex<f64>>())
                    .map_err(h5err)?
                    .into_iter()
                    .map(|c| Complex64::new(c.re, c.im))
                    .collect()
            };
            if vals.len() != 3 * n {
                return Err(invalid(format!("{}: field size mismatch", path.display())));
            }
            out.push(reorder(&|i| vals[i]));
            continue;
        }
        // Time-domain fallback.
        if td.is_none() {
            let g = f.group("/FieldData/TD").map_err(|_| {
                invalid(format!(
                    "nf2ff: frequency {fr} not found in {} (no FD or TD data)",
                    path.display()
                ))
            })?;
            let mut names = g.link_names().map_err(h5err)?;
            names.sort();
            let mut v = Vec::with_capacity(names.len());
            for name in names {
                let d = g.dataset(&name).map_err(h5err)?;
                let t = d
                    .attribute("time")
                    .and_then(|a| a.read::<f64>())
                    .map_err(h5err)?;
                let data = d.read::<f64>().map_err(h5err)?;
                if data.len() != 3 * n {
                    return Err(invalid(format!("{}: field size mismatch", path.display())));
                }
                v.push((t.first().copied().unwrap_or(0.0), data));
            }
            if v.len() < 2 {
                return Err(invalid(format!(
                    "nf2ff: {} has fewer than two time samples",
                    path.display()
                )));
            }
            td = Some(v);
        }
        let samples = td.as_ref().expect("set above");
        let dt = samples[1].0 - samples[0].0;
        let mut acc = vec![Complex64::ZERO; 3 * n];
        for (t, data) in samples {
            let w = Complex64::cis(-2.0 * std::f64::consts::PI * fr * t) * (2.0 * dt);
            for (a, &x) in acc.iter_mut().zip(data) {
                *a += w * x;
            }
        }
        out.push(reorder(&|i| acc[i]));
    }
    Ok((lines, out))
}

/// Reads the E and H dumps of one plane at `freqs` (see the
/// [module docs](self#input-planes)).
///
/// # Errors
/// Unreadable files, a frequency that is in neither the FD nor (as TD
/// samples) in the files, or E and H meshes that differ.
pub fn read_plane(files: &Nf2ffPlaneFiles, freqs: &[f64]) -> Result<Vec<NearFieldPlane>> {
    let (le, e) = read_dump(&files.e_file, freqs)?;
    let (lh, h) = read_dump(&files.h_file, freqs)?;
    if le != lh {
        return Err(invalid(format!(
            "nf2ff: mesh lines of {} and {} don't agree",
            files.e_file.display(),
            files.h_file.display()
        )));
    }
    Ok(e.into_iter()
        .zip(h)
        .map(|(e, h)| NearFieldPlane {
            lines: le.clone(),
            e,
            h,
        })
        .collect())
}

/// Trapezoidal node weights of a line set (1 for a single line).
fn trapezoid(l: &[f64]) -> Vec<f64> {
    let mut w = vec![0.0; l.len()];
    if l.len() == 1 {
        w[0] = 1.0;
        return w;
    }
    for i in 0..l.len() - 1 {
        let d = 0.5 * (l[i + 1] - l[i]);
        w[i] += d;
        w[i + 1] += d;
    }
    w
}

fn cross(a: [Complex64; 3], b: [Complex64; 3]) -> [Complex64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The far field of an nf2ff job (one entry per frequency; angle arrays
/// `[ntheta][nphi]` flattened theta-major).
#[derive(Debug, Clone, PartialEq)]
pub struct Nf2ffFarField {
    /// Frequencies \[Hz\].
    pub frequencies: Vec<f64>,
    /// Polar angles \[rad\].
    pub theta: Vec<f64>,
    /// Azimuth angles \[rad\].
    pub phi: Vec<f64>,
    /// Radius \[m\].
    pub radius: f64,
    /// `E_θ` at the radius, per frequency.
    pub e_theta: Vec<Vec<Complex64>>,
    /// `E_φ` at the radius, per frequency.
    pub e_phi: Vec<Vec<Complex64>>,
    /// Power density `(|E_θ|² + |E_φ|²)/(2η)` \[W/m²\], per frequency.
    pub p_rad: Vec<Vec<f64>>,
    /// Poynting flux through the planes (with mirror images) \[W\].
    pub prad: Vec<f64>,
    /// `4π r² max P_rad / Prad`.
    pub dmax: Vec<f64>,
    /// `Eps_r` as given (written when non-empty).
    pub eps_r: Vec<f64>,
    /// `Mue_r` as given (written when non-empty).
    pub mue_r: Vec<f64>,
}

/// One equivalent source point: position, weight, J, M.
struct Source {
    pos: [f64; 3],
    w: f64,
    j: [Complex64; 3],
    m: [Complex64; 3],
}

/// The equivalent surface currents of a plane (and the plane's flux).
fn plane_sources(p: &NearFieldPlane, center: [f64; 3]) -> Result<(Vec<Source>, f64)> {
    let a = p.normal_axis()?;
    let sign = if p.lines[a][0] >= center[a] {
        1.0
    } else {
        -1.0
    };
    let mut nv = [Complex64::ZERO; 3];
    nv[a] = Complex64::from(sign);
    let w: [Vec<f64>; 3] = [0, 1, 2].map(|k| trapezoid(&p.lines[k]));
    let [nx, ny, nz] = [0, 1, 2].map(|k| p.lines[k].len());
    if p.e.len() != nx * ny * nz || p.h.len() != p.e.len() {
        return Err(invalid("nf2ff: plane field size mismatch"));
    }
    let mut out = Vec::with_capacity(p.e.len());
    let mut flux = 0.0;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let idx = (i * ny + j) * nz + k;
                let wt = w[0][i] * w[1][j] * w[2][k];
                let e = p.e[idx];
                let h = p.h[idx];
                let jj = cross(nv, h);
                let mm = cross(nv, e).map(|c| -c);
                let s = cross(e, h.map(Complex64::conj));
                flux += 0.5 * sign * s[a].re * wt;
                out.push(Source {
                    pos: [p.lines[0][i], p.lines[1][j], p.lines[2][k]],
                    w: wt,
                    j: jj,
                    m: mm,
                });
            }
        }
    }
    Ok((out, flux))
}

/// Adds the images of `src` in `m` (see the [module docs](self#transform)).
fn image(src: &Source, m: &Nf2ffMirror) -> Source {
    let mut pos = src.pos;
    pos[m.axis] = 2.0 * m.pos - pos[m.axis];
    let (mut j, mut mm) = (src.j, src.m);
    for c in 0..3 {
        let normal = c == m.axis;
        // PEC: tangential J and normal M reversed; PMC: the others.
        if normal != m.pec {
            j[c] = -j[c];
        } else {
            mm[c] = -mm[c];
        }
    }
    Source {
        pos,
        w: src.w,
        j,
        m: mm,
    }
}

/// Computes the far field of `planes` (per plane, one [`NearFieldPlane`]
/// per job frequency).
///
/// # Errors
/// Planes without a single normal axis or inconsistent field sizes.
pub fn far_field(job: &Nf2ffJob, planes: &[Vec<NearFieldPlane>]) -> Result<Nf2ffFarField> {
    let nf = job.frequencies.len();
    let (nt, np) = (job.theta.len(), job.phi.len());
    let mut res = Nf2ffFarField {
        frequencies: job.frequencies.clone(),
        theta: job.theta.clone(),
        phi: job.phi.clone(),
        radius: job.radius,
        e_theta: Vec::with_capacity(nf),
        e_phi: Vec::with_capacity(nf),
        p_rad: Vec::with_capacity(nf),
        prad: Vec::with_capacity(nf),
        dmax: Vec::with_capacity(nf),
        eps_r: job.eps_r.clone(),
        mue_r: job.mue_r.clone(),
    };
    for (kf, &f) in job.frequencies.iter().enumerate() {
        let eps = Nf2ffJob::material(&job.eps_r, kf);
        let mue = Nf2ffJob::material(&job.mue_r, kf);
        let k = 2.0 * std::f64::consts::PI * f * (eps * mue).sqrt() / C0;
        let eta = Z0 * (mue / eps).sqrt();
        let mut srcs = Vec::new();
        let mut flux = 0.0;
        for p in planes {
            let plane = p
                .get(kf)
                .ok_or_else(|| invalid("nf2ff: missing plane frequency"))?;
            let (s, fl) = plane_sources(plane, job.center)?;
            srcs.extend(s);
            flux += fl;
        }
        for m in &job.mirrors {
            let imgs: Vec<Source> = srcs.iter().map(|s| image(s, m)).collect();
            srcs.extend(imgs);
            flux *= 2.0;
        }
        let r = job.radius;
        let pre =
            Complex64::new(0.0, k) * Complex64::cis(-k * r) / (4.0 * std::f64::consts::PI * r);
        let mut et = Vec::with_capacity(nt * np);
        let mut ep = Vec::with_capacity(nt * np);
        let mut pr = Vec::with_capacity(nt * np);
        for &t in &job.theta {
            let (st, ct) = t.sin_cos();
            for &ph in &job.phi {
                let (sp, cp) = ph.sin_cos();
                let rh = [st * cp, st * sp, ct];
                let th = [ct * cp, ct * sp, -st];
                let phv = [-sp, cp, 0.0];
                let mut n = [Complex64::ZERO; 3];
                let mut l = [Complex64::ZERO; 3];
                for s in &srcs {
                    let arg = k
                        * (rh[0] * (s.pos[0] - job.center[0])
                            + rh[1] * (s.pos[1] - job.center[1])
                            + rh[2] * (s.pos[2] - job.center[2]));
                    let e = Complex64::cis(arg) * s.w;
                    for c in 0..3 {
                        n[c] += s.j[c] * e;
                        l[c] += s.m[c] * e;
                    }
                }
                let dot = |v: [Complex64; 3], u: [f64; 3]| v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
                let (n_t, n_p, l_t, l_p) = (dot(n, th), dot(n, phv), dot(l, th), dot(l, phv));
                let e_t = -(pre * (l_p + n_t * eta));
                let e_p = pre * (l_t - n_p * eta);
                pr.push((e_t.norm_sqr() + e_p.norm_sqr()) / (2.0 * eta));
                et.push(e_t);
                ep.push(e_p);
            }
        }
        let umax = pr.iter().copied().fold(0.0f64, f64::max) * r * r;
        res.dmax.push(4.0 * std::f64::consts::PI * umax / flux);
        res.prad.push(flux);
        res.e_theta.push(et);
        res.e_phi.push(ep);
        res.p_rad.push(pr);
    }
    Ok(res)
}

mod h5 {
    use hdf5rs::Datatype;
    use hdf5rs::format::Sizes;
    use hdf5rs::write::{EncodeContext, FileWriter, H5Write, WriteAt, WriteOptions};

    use super::*;

    /// `{r: f64, i: f64}` (h5py's `complex128`).
    struct C64(Complex64);

    impl H5Write for C64 {
        fn datatype(sizes: Sizes) -> Datatype {
            Datatype::packed_compound(vec![
                ("r".into(), f64::datatype(sizes)),
                ("i".into(), f64::datatype(sizes)),
            ])
            .expect("two f64 members form a valid compound")
        }

        fn encode(&self, out: &mut Vec<u8>, ctx: &mut EncodeContext<'_>) -> hdf5rs::Result<()> {
            self.0.re.encode(out, ctx)?;
            self.0.im.encode(out, ctx)
        }
    }

    pub(super) fn options() -> WriteOptions {
        WriteOptions {
            superblock_version: 0,
            ..WriteOptions::default()
        }
    }

    pub(super) fn fill<S: WriteAt>(
        f: &mut FileWriter<S>,
        ff: &Nf2ffFarField,
    ) -> hdf5rs::Result<()> {
        let mut root = f.root();
        root.set_attr("openEMS_HDF5_version", &0.3f64)?;
        let shape = [ff.theta.len() as u64, ff.phi.len() as u64];
        {
            let mut mesh = root.create_group("Mesh")?;
            mesh.set_attr("MeshType", &[2.0f32][..])?;
            let to32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
            mesh.new_dataset::<f32>("phi")
                .shape(&[ff.phi.len() as u64])
                .write(&to32(&ff.phi))?;
            mesh.new_dataset::<f32>("r")
                .shape(&[1])
                .write(&[ff.radius as f32])?;
            mesh.new_dataset::<f32>("theta")
                .shape(&[ff.theta.len() as u64])
                .write(&to32(&ff.theta))?;
        }
        let mut g = root.create_group("nf2ff")?;
        g.set_attr("Dmax", &ff.dmax[..])?;
        if !ff.eps_r.is_empty() {
            g.set_attr("Eps_r", &ff.eps_r[..])?;
        }
        let fr: Vec<f32> = ff.frequencies.iter().map(|&x| x as f32).collect();
        g.set_attr("Frequency", &fr[..])?;
        if !ff.mue_r.is_empty() {
            g.set_attr("Mue_r", &ff.mue_r[..])?;
        }
        g.set_attr("Prad", &ff.prad[..])?;
        for (name, data) in [("E_phi", &ff.e_phi), ("E_theta", &ff.e_theta)] {
            let mut q = g.create_group(name)?;
            let mut fd = q.create_group("FD")?;
            for (k, v) in data.iter().enumerate() {
                let v: Vec<C64> = v.iter().copied().map(C64).collect();
                fd.new_dataset::<C64>(&format!("f{k}"))
                    .shape(&shape)
                    .write(&v)?;
            }
        }
        let mut q = g.create_group("P_rad")?;
        let mut fd = q.create_group("FD")?;
        for (k, v) in ff.p_rad.iter().enumerate() {
            fd.new_dataset::<f64>(&format!("f{k}"))
                .shape(&shape)
                .write(v)?;
        }
        Ok(())
    }
}

/// The output file of [`Nf2ffFarField`] in memory (see the
/// [module docs](self#output-file)).
///
/// # Errors
/// Encoding errors ([`Error::Io`]).
pub fn far_field_hdf5_bytes(ff: &Nf2ffFarField) -> Result<Vec<u8>> {
    let mut f = hdf5rs::write::FileWriter::new(Vec::new(), h5::options()).map_err(h5err)?;
    h5::fill(&mut f, ff).map_err(h5err)?;
    f.finish().map_err(h5err)
}

/// Writes the output file of [`Nf2ffFarField`].
///
/// # Errors
/// Encoding and I/O errors.
pub fn write_far_field(path: impl AsRef<Path>, ff: &Nf2ffFarField) -> Result<()> {
    let bytes = far_field_hdf5_bytes(ff)?;
    std::fs::write(path.as_ref(), bytes)
        .map_err(|e| Error::Io(format!("{}: {e}", path.as_ref().display())))
}

/// Runs a parsed job: reads the planes, computes and writes the far field.
///
/// # Errors
/// Those of [`read_plane`], [`far_field`] and [`write_far_field`].
pub fn run_job(job: &Nf2ffJob) -> Result<Nf2ffFarField> {
    let planes = job
        .planes
        .iter()
        .map(|p| read_plane(p, &job.frequencies))
        .collect::<Result<Vec<_>>>()?;
    let ff = far_field(job, &planes)?;
    write_far_field(&job.outfile, &ff)?;
    Ok(ff)
}

/// Runs the nf2ff XML file at `path` like `nf2ff <path>` (file names
/// relative to the working directory).
///
/// # Errors
/// Those of [`Nf2ffJob::read_file`] and [`run_job`].
pub fn run_xml_file(path: impl AsRef<Path>) -> Result<Nf2ffFarField> {
    run_job(&Nf2ffJob::read_file(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_job() {
        let j = Nf2ffJob::from_xml_str(
            r#"<?xml version="1.0"?>
            <nf2ff freq="1e9,2e9" Outfile="o.h5" Radius="2" Center="1,2,3" Eps_r="2" Verbose="1">
              <theta>0,0.5, 1</theta><phi>0</phi>
              <Planes E_Field="e.h5" H_Field="h.h5"/>
              <Mirror Dir="2" Type="PMC" Pos="0.5"/>
            </nf2ff>"#,
        )
        .unwrap();
        assert_eq!(j.frequencies, [1e9, 2e9]);
        assert_eq!(j.theta, [0.0, 0.5, 1.0]);
        assert_eq!(j.radius, 2.0);
        assert_eq!(j.center, [1.0, 2.0, 3.0]);
        assert_eq!(j.eps_r, [2.0]);
        assert_eq!(
            j.mirrors,
            [Nf2ffMirror {
                axis: 2,
                pec: false,
                pos: 0.5
            }]
        );
        assert_eq!(j.planes.len(), 1);
        for (xml, msg) in [
            ("<foo/>", "nf2ff element"),
            ("<nf2ff/>", "frequency"),
            ("<nf2ff freq=\"1\"/>", "output file"),
            ("<nf2ff freq=\"1\" Outfile=\"o\"/>", "theta"),
            (
                "<nf2ff freq=\"1\" Outfile=\"o\"><theta>1</theta></nf2ff>",
                "phi",
            ),
        ] {
            let e = Nf2ffJob::from_xml_str(xml).unwrap_err().to_string();
            assert!(e.contains(msg), "{xml}: {e}");
        }
    }

    /// A uniform x-directed electric surface current on a small square
    /// (H = ŷ·(1/2) on a z = 0 face, J = ẑ × H = −x̂/2) radiates like a
    /// Hertzian dipole of moment `J·A` far away.
    #[test]
    fn small_current_sheet_is_a_dipole() {
        let a = 1e-3;
        let lines = [vec![-a / 2.0, a / 2.0], vec![-a / 2.0, a / 2.0], vec![0.0]];
        let h = [Complex64::ZERO, Complex64::new(0.5, 0.0), Complex64::ZERO];
        let plane = NearFieldPlane {
            lines,
            e: vec![[Complex64::ZERO; 3]; 4],
            h: vec![h; 4],
        };
        let job = Nf2ffJob {
            frequencies: vec![1e9],
            outfile: "x.h5".into(),
            theta: vec![std::f64::consts::FRAC_PI_2],
            phi: vec![std::f64::consts::FRAC_PI_2],
            radius: 1.0,
            center: [0.0; 3],
            eps_r: vec![],
            mue_r: vec![],
            mirrors: vec![],
            planes: vec![],
            verbose: 0,
            threads: None,
        };
        let ff = far_field(&job, &[vec![plane]]).unwrap();
        // Broadside (along y) of an x dipole of moment Il = 0.5·a²:
        // |E| = η k Il / (4π r), times cos(k a/2) for the two node rows
        // at y = ±a/2.
        let k = 2.0 * std::f64::consts::PI * 1e9 / C0;
        let want = Z0 * k * 0.5 * a * a / (4.0 * std::f64::consts::PI) * (0.5 * k * a).cos();
        let got = (ff.e_theta[0][0].norm_sqr() + ff.e_phi[0][0].norm_sqr()).sqrt();
        assert!((got / want - 1.0).abs() < 1e-12, "{got} vs {want}");
        let bytes = far_field_hdf5_bytes(&ff).unwrap();
        assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");
    }
}
