//! Near-field to far-field (NF2FF) transform: radiation patterns,
//! directivity, gain and radiated power (design §5.10).
//!
//! # Method
//!
//! A closed box around the radiator records the tangential E and H fields
//! on its six faces as running DFTs at chosen frequencies (one
//! [`FreqDump`] per face and field, so the E/H time staggering is taken
//! care of by the absolute-time DFT kernel and both fields are co-located
//! at the primary nodes of the face, see [`crate::engine::dump`]). By the
//! surface equivalence principle (Balanis, *Antenna Theory*, §12.2–12.3;
//! Taflove & Hagness, *Computational Electrodynamics*, ch. 8) the fields
//! outside the box are those radiated in free space by the surface
//! currents
//!
//! ```text
//! J = n̂ × H,    M = −n̂ × E        (n̂: outward normal)
//! ```
//!
//! In the far zone (`r → ∞`, `e^{jωt}` time convention, the one of the
//! DFTs) the radiation vectors
//!
//! ```text
//! N(θ,φ) = ∮ J(r') e^{+jk r̂·(r' − c)} dS',   L(θ,φ) = ∮ M(r') e^{+jk r̂·(r' − c)} dS'
//! ```
//!
//! (phase centre `c`, by default the box centre) give
//!
//! ```text
//! E_θ = −jk e^{−jkr}/(4πr) · (L_φ + η N_θ),   E_φ = +jk e^{−jkr}/(4πr) · (L_θ − η N_φ)
//! ```
//!
//! and the radiation intensity `U = r²(|E_θ|² + |E_φ|²)/(2η)`. The surface
//! integrals use the trapezoidal rule over the face nodes (exact for
//! fields linear between nodes, on any non-uniform mesh).
//!
//! * **Radiated power** [`radiated_power`](Nf2ffResult::radiated_power) is
//!   the Poynting flux `½ Re ∮ (E × H*)·n̂ dS` through the box. It equals
//!   the integral of `U` over the sphere
//!   ([`radiated_power_far`](Nf2ffResult::radiated_power_far)) when the box
//!   is in lossless free space.
//! * **Directivity** `D = 4π U / P_rad` (with the box flux).
//! * **Gain** `G = 4π U / P_acc` with the power accepted by the feed, e.g.
//!   [`PortResult::accepted_power`](crate::post::PortResult::accepted_power)
//!   of a port of the same run (radiation efficiency `P_rad / P_acc`).
//!
//! # Units and normalisation
//!
//! The near fields are DFTs of transients (V/m·s, A/m·s), so the far
//! fields and powers are spectral densities. Ratios (directivity, gain
//! with a port of the same run, patterns) are independent of that
//! normalisation; divide by the excitation spectrum for transfer-function
//! values.
//!
//! # Requirements
//!
//! * The box must enclose all sources and scatterers and lie in vacuum
//!   (η = Z0, k = ω/c0). Keep it a few cells inside the PML: the faces must
//!   lie on interior mesh planes (index `1 … n−2`).
//! * Accuracy is limited by the second-order co-location of E and H on
//!   the faces. Very close to a small source, where the reactive near
//!   field dominates, this costs about 1 % of radiated power; a box a
//!   few more cells out removes most of it (design §5.10).
//! * The run must last until the fields have decayed (DFT of a complete
//!   transient), e.g. an energy end criterion of −40 dB or below.
//!
//! # Ground planes and symmetry ([`Mirror`])
//!
//! A face lying on an infinite PEC or PMC plane (a PEC/PMC domain boundary
//! or symmetry plane) can be replaced by image theory: the result then sums
//! the box currents and their images. Faces on a mirror plane are skipped
//! automatically; the radiated power is that of the real half-space (or
//! quarter / octant for several mirrors), so directivities refer to it
//! (e.g. a monopole over ground has twice the directivity of the dipole).
//! The far fields are meaningful only on the real side of every mirror.
//!
//! # Usage
//!
//! With a [`Simulation`](crate::Simulation) (box corners in drawing units):
//!
//! ```no_run
//! # fn demo(model: &csx::ContinuousStructure, fdtd: &ems::Fdtd) -> ems::Result<()> {
//! use ems::nf2ff::Nf2ffBox;
//! let mut sim = ems::Simulation::new(model, fdtd)?;
//! let nf = sim.add_nf2ff(&Nf2ffBox::new([-50.0; 3], [50.0; 3], [1e9]))?;
//! sim.run()?;
//! let ff = sim.nf2ff(&nf);
//! println!("D_max = {:.2} dBi", ff.max_directivity(0).dbi());
//! # Ok(()) }
//! ```
//!
//! On a bare [`Engine`](crate::engine::Engine), register an [`Nf2ff`]
//! operator (node indices) after all sources and read
//! [`Nf2ff::result`].

use core::f64::consts::PI;

use crate::axis::Axis;
use crate::constants::{C0, Z0};
use crate::engine::{
    DumpPosition, DumpQuantity, DumpRegion, FieldSampler, Fields, FreqDump, Grid, Hooks, Operator,
    OperatorHandle, StepContext,
};
use crate::fdtd::Face;
use crate::post::Complex64;
use crate::{Error, Result};

/// Kind of an infinite mirror plane (see the [module docs](self#ground-planes-and-symmetry-mirror)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirrorKind {
    /// Perfect electric conductor (ground plane): tangential `J` and normal
    /// `M` images are reversed.
    Pec,
    /// Perfect magnetic conductor: normal `J` and tangential `M` images are
    /// reversed.
    Pmc,
}

/// An infinite PEC or PMC plane normal to `axis` at `position` \[m\].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mirror {
    /// Normal of the plane.
    pub axis: Axis,
    /// Plane coordinate \[m\].
    pub position: f64,
    /// PEC or PMC.
    pub kind: MirrorKind,
}

impl Mirror {
    /// Mirror normal to `axis` at `position` \[m\].
    pub fn new(axis: Axis, position: f64, kind: MirrorKind) -> Self {
        Self {
            axis,
            position,
            kind,
        }
    }
}

/// One recorded face: E and H DFTs over its plane.
#[derive(Debug, Clone)]
struct FaceDft {
    face: Face,
    e: FreqDump,
    h: FreqDump,
}

/// Engine operator recording the tangential fields on the faces of a box
/// of primary nodes (see the [module docs](self)).
///
/// Register it **after** the sources, lumped elements and boundaries (like
/// the dumps). Sampling runs on the stepping thread in a fixed order, so
/// the result is bit-identical for any engine thread count.
#[derive(Debug, Clone)]
pub struct Nf2ff {
    lo: [usize; 3],
    hi: [usize; 3],
    freqs: Vec<f64>,
    faces: Vec<FaceDft>,
    center: [f64; 3],
}

impl Nf2ff {
    /// Box between the nodes `a` and `b` (corners in any order) of `grid`,
    /// recording all six faces at `freqs` \[Hz\].
    ///
    /// # Errors
    /// See [`with_faces`](Self::with_faces).
    pub fn new(
        grid: &Grid,
        a: [usize; 3],
        b: [usize; 3],
        freqs: impl Into<Vec<f64>>,
    ) -> Result<Self> {
        Self::with_faces(grid, a, b, freqs, [true; 6])
    }

    /// Like [`new`](Self::new), recording only the faces `faces[f.index()]`
    /// ([`Face::index`] order: x min, x max, y min, …).
    ///
    /// # Errors
    /// [`Error::Invalid`] if the box is flat along an axis or does not fit
    /// the grid, if a recorded face lies on an outer mesh plane, if no face
    /// is recorded, or if a frequency is not positive and finite (or none
    /// is given).
    pub fn with_faces(
        grid: &Grid,
        a: [usize; 3],
        b: [usize; 3],
        freqs: impl Into<Vec<f64>>,
        faces: [bool; 6],
    ) -> Result<Self> {
        let freqs = freqs.into();
        if freqs.is_empty() || freqs.iter().any(|f| !(f.is_finite() && *f > 0.0)) {
            return Err(Error::Invalid(format!(
                "NF2FF frequencies must be positive and finite (got {freqs:?})"
            )));
        }
        let lo = [0, 1, 2].map(|i| a[i].min(b[i]));
        let hi = [0, 1, 2].map(|i| a[i].max(b[i]));
        let n = grid.nodes();
        for ax in Axis::ALL {
            let i = ax.index();
            if lo[i] >= hi[i] {
                return Err(Error::Invalid(format!(
                    "NF2FF box is flat along {ax} (nodes {} .. {})",
                    lo[i], hi[i]
                )));
            }
            if hi[i] >= n[i] {
                return Err(Error::Invalid(format!(
                    "NF2FF box node {} along {ax} outside the grid ({} nodes)",
                    hi[i], n[i]
                )));
            }
        }
        if !faces.contains(&true) {
            return Err(Error::Invalid("NF2FF box records no face".into()));
        }
        let mut out = Vec::new();
        for face in Face::ALL {
            if !faces[face.index()] {
                continue;
            }
            let ax = face.axis().index();
            let plane = if face.is_max() { hi[ax] } else { lo[ax] };
            if plane == 0 || plane + 1 >= n[ax] {
                return Err(Error::Invalid(format!(
                    "NF2FF face {face:?} lies on the outer mesh plane {plane}; \
                     move it inside the domain (and outside any PML)"
                )));
            }
            let (mut fa, mut fb) = (lo, hi);
            fa[ax] = plane;
            fb[ax] = plane;
            let region = DumpRegion::new(fa, fb);
            let dft = |q| -> Result<FreqDump> {
                let s = FieldSampler::new(grid, region, q, DumpPosition::Nodes)?;
                Ok(FreqDump::new(s, freqs.clone()))
            };
            out.push(FaceDft {
                face,
                e: dft(DumpQuantity::E)?,
                h: dft(DumpQuantity::H)?,
            });
        }
        let center = Axis::ALL.map(|ax| {
            let l = grid.lines(ax);
            0.5 * (l[lo[ax.index()]] + l[hi[ax.index()]])
        });
        Ok(Self {
            lo,
            hi,
            freqs,
            faces: out,
            center,
        })
    }

    /// Samples only every `k`-th state (`k ≥ 1`, default 1); see
    /// [`FreqDump::interval`] for the aliasing limit `1/(2k·dt)`.
    pub fn interval(mut self, k: u64) -> Self {
        for f in &mut self.faces {
            f.e = f.e.clone().interval(k);
            f.h = f.h.clone().interval(k);
        }
        self
    }

    /// Lower box corner (node indices).
    pub fn lo(&self) -> [usize; 3] {
        self.lo
    }

    /// Upper box corner (node indices).
    pub fn hi(&self) -> [usize; 3] {
        self.hi
    }

    /// The analysis frequencies \[Hz\].
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// The recorded faces.
    pub fn faces(&self) -> impl Iterator<Item = Face> + '_ {
        self.faces.iter().map(|f| f.face)
    }

    /// Number of E states accumulated so far.
    pub fn samples(&self) -> u64 {
        self.faces.first().map_or(0, |f| f.e.samples())
    }

    /// The far-field result of the fields recorded so far, with the phase
    /// centre at the box centre and no mirror.
    pub fn result(&self) -> Nf2ffResult {
        let faces = self
            .faces
            .iter()
            .map(|f| FaceData::new(f, self.freqs.len()))
            .collect();
        Nf2ffResult {
            freqs: self.freqs.clone(),
            center: self.center,
            faces,
            mirrors: Vec::new(),
        }
    }
}

impl Operator for Nf2ff {
    fn pre_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        for f in &mut self.faces {
            f.e.pre_h(fields, ctx);
            f.h.pre_h(fields, ctx);
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        for f in &mut self.faces {
            f.e.post_e(fields, ctx);
            f.h.post_e(fields, ctx);
        }
    }

    fn hooks(&self) -> Hooks {
        Hooks::PRE_H | Hooks::POST_E
    }
}

/// An NF2FF box for [`Simulation::add_nf2ff`](crate::Simulation::add_nf2ff),
/// in csx drawing units.
#[derive(Debug, Clone, PartialEq)]
pub struct Nf2ffBox {
    start: [f64; 3],
    stop: [f64; 3],
    freqs: Vec<f64>,
    faces: [bool; 6],
    mirrors: Vec<(Axis, f64, MirrorKind)>,
    interval: Option<u64>,
}

impl Nf2ffBox {
    /// Box between the corners `start` and `stop` (drawing units, snapped
    /// to the nearest mesh lines) recording all six faces at `freqs`
    /// \[Hz\].
    pub fn new(start: [f64; 3], stop: [f64; 3], freqs: impl Into<Vec<f64>>) -> Self {
        Self {
            start,
            stop,
            freqs: freqs.into(),
            faces: [true; 6],
            mirrors: Vec::new(),
            interval: None,
        }
    }

    /// Records only the faces with `faces[f.index()]` set.
    pub fn faces(mut self, faces: [bool; 6]) -> Self {
        self.faces = faces;
        self
    }

    /// Does not record `face` (e.g. one lying on a ground plane).
    pub fn without_face(mut self, face: Face) -> Self {
        self.faces[face.index()] = false;
        self
    }

    /// Adds an infinite mirror plane normal to `axis` at `position`
    /// (drawing units); see [`Mirror`]. A face on that plane is not
    /// recorded.
    pub fn mirror(mut self, axis: Axis, position: f64, kind: MirrorKind) -> Self {
        self.mirrors.push((axis, position, kind));
        self
    }

    /// Samples every `k`-th state only (see [`Nf2ff::interval`]). Default:
    /// the interval of the simulation's frequency dumps (the Nyquist
    /// interval of the settings' maximum frequency divided by the
    /// oversampling factor, see [`Simulation`](crate::Simulation)), which
    /// is exact for band-limited fields and much cheaper than sampling
    /// every step.
    pub fn interval(mut self, k: u64) -> Self {
        self.interval = Some(k.max(1));
        self
    }

    /// The corners in drawing units.
    pub fn corners(&self) -> ([f64; 3], [f64; 3]) {
        (self.start, self.stop)
    }

    /// The analysis frequencies \[Hz\].
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// Builds the engine operator for `grid` (lines in metres; drawing unit
    /// `unit` m), sampling every `default_interval` steps unless
    /// [`interval`](Self::interval) was set. Returns the operator and the
    /// mirrors in metres.
    pub(crate) fn build(
        &self,
        grid: &Grid,
        unit: f64,
        default_interval: u64,
    ) -> Result<(Nf2ff, Vec<Mirror>)> {
        let snap = |ax: Axis, x: f64| -> Result<usize> {
            if !x.is_finite() {
                return Err(Error::Invalid(format!(
                    "NF2FF box coordinate {x} along {ax}"
                )));
            }
            let xm = x * unit;
            let (i, _) = grid
                .lines(ax)
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, (v - xm).abs()))
                .fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
            Ok(i)
        };
        let mut a = [0; 3];
        let mut b = [0; 3];
        for ax in Axis::ALL {
            a[ax.index()] = snap(ax, self.start[ax.index()])?;
            b[ax.index()] = snap(ax, self.stop[ax.index()])?;
        }
        let lo = [0, 1, 2].map(|i| a[i].min(b[i]));
        let hi = [0, 1, 2].map(|i| a[i].max(b[i]));
        let mut faces = self.faces;
        let mut mirrors = Vec::new();
        for &(ax, pos, kind) in &self.mirrors {
            let p = pos * unit;
            if !p.is_finite() {
                return Err(Error::Invalid(format!("NF2FF mirror position {pos}")));
            }
            let l = grid.lines(ax);
            let i = ax.index();
            for (max, node) in [(false, lo[i]), (true, hi[i])] {
                if (l[node] - p).abs() <= 1e-6 * grid.min_delta(ax) {
                    faces[Face::new(ax, max).index()] = false;
                }
            }
            mirrors.push(Mirror::new(ax, p, kind));
        }
        let op = Nf2ff::with_faces(grid, lo, hi, self.freqs.clone(), faces)?
            .interval(self.interval.unwrap_or(default_interval));
        Ok((op, mirrors))
    }
}

/// Handle to an NF2FF box added with
/// [`Simulation::add_nf2ff`](crate::Simulation::add_nf2ff).
#[derive(Debug, Clone)]
pub struct Nf2ffHandle {
    pub(crate) op: OperatorHandle<Nf2ff>,
    pub(crate) mirrors: Vec<Mirror>,
}

impl Nf2ffHandle {
    /// The engine operator handle.
    pub fn operator(&self) -> OperatorHandle<Nf2ff> {
        self.op
    }

    /// The mirror planes (metres).
    pub fn mirrors(&self) -> &[Mirror] {
        &self.mirrors
    }
}

/// Equivalent currents of one face.
#[derive(Debug, Clone)]
struct FaceData {
    face: Face,
    /// Sample positions \[m\].
    points: Vec<[f64; 3]>,
    /// Trapezoidal area weights \[m²\].
    weights: Vec<f64>,
    /// `J = n̂ × H` per frequency and point.
    j: Vec<Vec<[Complex64; 3]>>,
    /// `M = −n̂ × E` per frequency and point.
    m: Vec<Vec<[Complex64; 3]>>,
}

/// Trapezoidal weights of the node coordinates `x`.
fn trapezoid(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let l = if i > 0 { x[i] - x[i - 1] } else { 0.0 };
            let r = if i + 1 < n { x[i + 1] - x[i] } else { 0.0 };
            0.5 * (l + r)
        })
        .collect()
}

fn cross(a: [Complex64; 3], b: [f64; 3]) -> [Complex64; 3] {
    // a × b with real b.
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normal(face: Face) -> [f64; 3] {
    let mut n = [0.0; 3];
    n[face.axis().index()] = if face.is_max() { 1.0 } else { -1.0 };
    n
}

impl FaceData {
    fn new(f: &FaceDft, nf: usize) -> Self {
        let s = f.e.sampler();
        let c = Axis::ALL.map(|a| s.coords(a));
        let w = Axis::ALL.map(|a| {
            if a == f.face.axis() {
                vec![1.0]
            } else {
                trapezoid(c[a.index()])
            }
        });
        let [dx, dy, dz] = s.dims();
        let mut points = Vec::with_capacity(s.len());
        let mut weights = Vec::with_capacity(s.len());
        for p in 0..dx {
            for q in 0..dy {
                for r in 0..dz {
                    points.push([c[0][p], c[1][q], c[2][r]]);
                    weights.push(w[0][p] * w[1][q] * w[2][r]);
                }
            }
        }
        let n = normal(f.face);
        let neg_n = n.map(|x| -x);
        let (j, m) = (0..nf)
            .map(|k| {
                let e = f.e.values(k);
                let h = f.h.values(k);
                // n̂ × H = −(H × n̂);  −n̂ × E = E × n̂.
                let j = h.iter().map(|&h| cross(h, neg_n)).collect();
                let m = e.iter().map(|&e| cross(e, n)).collect();
                (j, m)
            })
            .unzip();
        Self {
            face: f.face,
            points,
            weights,
            j,
            m,
        }
    }

    fn on_mirror(&self, mirrors: &[Mirror]) -> bool {
        let ax = self.face.axis();
        let x = self.points.first().map_or(f64::NAN, |p| p[ax.index()]);
        mirrors.iter().any(|m| {
            m.axis == ax
                && (x - m.position).abs() <= 1e-9 * x.abs().max(m.position.abs()).max(1e-30)
        })
    }
}

/// Far field in one direction: `r·E·e^{jkr}` components (V·s for a
/// transient excitation, see the [module docs](self#units-and-normalisation)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FarField {
    /// Polar angle θ \[rad\] (from +z).
    pub theta: f64,
    /// Azimuth φ \[rad\] (from +x towards +y).
    pub phi: f64,
    /// `r·E_θ·e^{jkr}`.
    pub e_theta: Complex64,
    /// `r·E_φ·e^{jkr}`.
    pub e_phi: Complex64,
}

impl FarField {
    /// Radiation intensity `U = (|rE_θ|² + |rE_φ|²)/(2η0)` (W/sr for peak
    /// phasors).
    pub fn intensity(&self) -> f64 {
        (self.e_theta.norm_sqr() + self.e_phi.norm_sqr()) / (2.0 * Z0)
    }

    /// The field `[E_θ, E_φ]` at distance `r` \[m\] for wavenumber `k`
    /// \[1/m\]: `(rE)·e^{−jkr}/r`.
    pub fn at_radius(&self, k: f64, r: f64) -> [Complex64; 2] {
        let p = Complex64::cis(-k * r) / r;
        [self.e_theta * p, self.e_phi * p]
    }
}

/// The direction of maximum directivity found by
/// [`Nf2ffResult::max_directivity`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxDirectivity {
    /// Directivity (linear).
    pub directivity: f64,
    /// θ \[rad\].
    pub theta: f64,
    /// φ \[rad\].
    pub phi: f64,
}

impl MaxDirectivity {
    /// The directivity in dBi.
    pub fn dbi(&self) -> f64 {
        10.0 * self.directivity.log10()
    }
}

/// Far-field evaluation of a recorded NF2FF box (see the
/// [module docs](self)). Frequencies are addressed by index into
/// [`freqs`](Self::freqs).
#[derive(Debug, Clone)]
pub struct Nf2ffResult {
    freqs: Vec<f64>,
    center: [f64; 3],
    faces: Vec<FaceData>,
    mirrors: Vec<Mirror>,
}

/// Unit vectors `(r̂, θ̂, φ̂)`.
fn basis(theta: f64, phi: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let (st, ct) = theta.sin_cos();
    let (sp, cp) = phi.sin_cos();
    (
        [st * cp, st * sp, ct],
        [ct * cp, ct * sp, -st],
        [-sp, cp, 0.0],
    )
}

fn dot(a: [Complex64; 3], b: [f64; 3]) -> Complex64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Nf2ffResult {
    /// The analysis frequencies \[Hz\].
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// Free-space wavenumber `2πf/c0` at frequency index `k`.
    pub fn wavenumber(&self, k: usize) -> f64 {
        2.0 * PI * self.freqs[k] / C0
    }

    /// The phase centre \[m\] (default: the box centre).
    pub fn center(&self) -> [f64; 3] {
        self.center
    }

    /// Uses `c` \[m\] as phase centre (changes only the far-field phases).
    pub fn with_center(mut self, c: [f64; 3]) -> Self {
        self.center = c;
        self
    }

    /// The mirror planes.
    pub fn mirrors(&self) -> &[Mirror] {
        &self.mirrors
    }

    /// Adds an infinite mirror plane (see [`Mirror`]); faces lying on it
    /// are ignored from then on.
    ///
    /// # Errors
    /// [`Error::Invalid`] if a mirror normal to the same axis exists, or if
    /// the plane cuts through the box.
    pub fn with_mirror(mut self, m: Mirror) -> Result<Self> {
        if self.mirrors.iter().any(|o| o.axis == m.axis) {
            return Err(Error::Invalid(format!(
                "NF2FF: a mirror normal to {} is already set",
                m.axis
            )));
        }
        let i = m.axis.index();
        let (lo, hi) = self
            .faces
            .iter()
            .flat_map(|f| f.points.iter().map(move |p| p[i]))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| {
                (a.min(x), b.max(x))
            });
        let tol = 1e-9 * lo.abs().max(hi.abs()).max(1e-30);
        if m.position > lo + tol && m.position < hi - tol {
            return Err(Error::Invalid(format!(
                "NF2FF mirror plane {} = {} m cuts the box ({lo} .. {hi} m)",
                m.axis, m.position
            )));
        }
        self.mirrors.push(m);
        Ok(self)
    }

    /// The faces taking part (not on a mirror plane).
    fn active(&self) -> impl Iterator<Item = &FaceData> {
        self.faces.iter().filter(|f| !f.on_mirror(&self.mirrors))
    }

    /// Whether direction `r̂` lies on the real side of every mirror.
    fn is_real_side(&self, r: [f64; 3]) -> bool {
        self.mirrors.iter().all(|m| {
            let i = m.axis.index();
            let side = self.center[i] - m.position;
            r[i] * side >= -1e-12
        })
    }

    /// Radiation vectors `(N, L)` at frequency index `k` towards `r̂`,
    /// including the mirror images.
    fn radiation_vectors(&self, k: usize, r: [f64; 3]) -> ([Complex64; 3], [Complex64; 3]) {
        let kw = self.wavenumber(k);
        let c = self.center;
        let mut n = [Complex64::ZERO; 3];
        let mut l = [Complex64::ZERO; 3];
        let nm = self.mirrors.len();
        for mask in 0..(1usize << nm) {
            // Position reflection and current signs of this image.
            let mut refl = [None::<f64>; 3];
            let mut sj = [1.0f64; 3];
            let mut sm = [1.0f64; 3];
            for (b, m) in self.mirrors.iter().enumerate() {
                if mask & (1 << b) == 0 {
                    continue;
                }
                let a = m.axis.index();
                refl[a] = Some(m.position);
                for t in 0..3 {
                    let tangential = t != a;
                    let (fj, fm) = match (m.kind, tangential) {
                        (MirrorKind::Pec, true) => (-1.0, 1.0),
                        (MirrorKind::Pec, false) => (1.0, -1.0),
                        (MirrorKind::Pmc, true) => (1.0, -1.0),
                        (MirrorKind::Pmc, false) => (-1.0, 1.0),
                    };
                    sj[t] *= fj;
                    sm[t] *= fm;
                }
            }
            for f in self.active() {
                let (jk, mk) = (&f.j[k], &f.m[k]);
                for (q, p) in f.points.iter().enumerate() {
                    let mut ph = 0.0;
                    for a in 0..3 {
                        let x = match refl[a] {
                            Some(x0) => 2.0 * x0 - p[a],
                            None => p[a],
                        };
                        ph += r[a] * (x - c[a]);
                    }
                    let e = Complex64::cis(kw * ph) * f.weights[q];
                    for a in 0..3 {
                        n[a] += jk[q][a] * (e * sj[a]);
                        l[a] += mk[q][a] * (e * sm[a]);
                    }
                }
            }
        }
        (n, l)
    }

    /// The far field towards `(theta, phi)` \[rad\] at frequency index `k`.
    ///
    /// # Panics
    /// If `k` is out of range.
    pub fn far_field(&self, k: usize, theta: f64, phi: f64) -> FarField {
        let (r, t, p) = basis(theta, phi);
        let (n, l) = self.radiation_vectors(k, r);
        let kw = self.wavenumber(k);
        let a = Complex64::new(0.0, kw / (4.0 * PI));
        let (nt, np) = (dot(n, t), dot(n, p));
        let (lt, lp) = (dot(l, t), dot(l, p));
        FarField {
            theta,
            phi,
            e_theta: -(a * (lp + nt * Z0)),
            e_phi: a * (lt - np * Z0),
        }
    }

    /// Far fields at the directions `dirs` (`(θ, φ)` pairs \[rad\]).
    pub fn pattern(&self, k: usize, dirs: &[(f64, f64)]) -> Vec<FarField> {
        dirs.iter().map(|&(t, p)| self.far_field(k, t, p)).collect()
    }

    /// θ cut at fixed `phi`: far fields at `thetas` \[rad\].
    pub fn cut_theta(&self, k: usize, phi: f64, thetas: &[f64]) -> Vec<FarField> {
        thetas.iter().map(|&t| self.far_field(k, t, phi)).collect()
    }

    /// φ cut at fixed `theta`: far fields at `phis` \[rad\].
    pub fn cut_phi(&self, k: usize, theta: f64, phis: &[f64]) -> Vec<FarField> {
        phis.iter().map(|&p| self.far_field(k, theta, p)).collect()
    }

    /// Radiation intensity `U(θ, φ)`.
    pub fn radiation_intensity(&self, k: usize, theta: f64, phi: f64) -> f64 {
        self.far_field(k, theta, phi).intensity()
    }

    /// Radiated power: the Poynting flux `½ Re ∮(E × H*)·n̂ dS` out of the
    /// box (of the real half-space with mirrors) at frequency index `k`.
    pub fn radiated_power(&self, k: usize) -> f64 {
        let mut p = 0.0;
        for f in self.active() {
            let n = normal(f.face);
            for (q, w) in f.weights.iter().enumerate() {
                // E_t = n̂ × M, and n̂·(E × H*) = −E_t·J*.
                let m = f.m[k][q];
                let et = cross(m, n).map(|x| -x);
                let j = f.j[k][q];
                let s: f64 = (0..3).map(|a| (et[a] * j[a].conj()).re).sum();
                p -= w * s;
            }
        }
        0.5 * p
    }

    /// Radiated power from the far field: `∮ U dΩ` over the sphere (the
    /// real half-space with mirrors), midpoint rule with `n_theta × n_phi`
    /// directions.
    pub fn radiated_power_far(&self, k: usize, n_theta: usize, n_phi: usize) -> f64 {
        let (nt, np) = (n_theta.max(1), n_phi.max(1));
        let (dt, dp) = (PI / nt as f64, 2.0 * PI / np as f64);
        let mut s = 0.0;
        for a in 0..nt {
            let t = (a as f64 + 0.5) * dt;
            for b in 0..np {
                let p = (b as f64 + 0.5) * dp;
                s += self.radiation_intensity(k, t, p) * t.sin();
            }
        }
        s * dt * dp / (1usize << self.mirrors.len()) as f64
    }

    /// Directivity `4πU/P_rad` towards `(θ, φ)`, with the box flux
    /// [`radiated_power`](Self::radiated_power).
    pub fn directivity(&self, k: usize, theta: f64, phi: f64) -> f64 {
        4.0 * PI * self.radiation_intensity(k, theta, phi) / self.radiated_power(k)
    }

    /// Gain `4πU/P_acc` towards `(θ, φ)` for the accepted power
    /// `accepted_power` (same normalisation, e.g.
    /// [`PortResult::accepted_power`](crate::post::PortResult::accepted_power)
    /// of the feeding port at the same frequency).
    pub fn gain(&self, k: usize, theta: f64, phi: f64, accepted_power: f64) -> f64 {
        4.0 * PI * self.radiation_intensity(k, theta, phi) / accepted_power
    }

    /// The maximum of the directivity over the real directions: a 5° scan
    /// refined twice (to ≈ 0.05°) around the best direction.
    pub fn max_directivity(&self, k: usize) -> MaxDirectivity {
        let p_rad = self.radiated_power(k);
        let mut best = (f64::NEG_INFINITY, 0.0, 0.0);
        let eval = |t: f64, p: f64, best: &mut (f64, f64, f64)| {
            let t = t.clamp(0.0, PI);
            if !self.is_real_side(basis(t, p).0) {
                return;
            }
            let u = self.radiation_intensity(k, t, p);
            if u > best.0 {
                *best = (u, t, p);
            }
        };
        let step = 5f64.to_radians();
        for a in 0..=36 {
            for b in 0..72 {
                eval(a as f64 * step, b as f64 * step, &mut best);
            }
        }
        for scale in [0.1, 0.01] {
            let s = step * scale;
            let (_, t0, p0) = best;
            for a in -10..=10 {
                for b in -10..=10 {
                    eval(t0 + a as f64 * s, p0 + b as f64 * s, &mut best);
                }
            }
        }
        MaxDirectivity {
            directivity: 4.0 * PI * best.0 / p_rad,
            theta: best.1,
            phi: best.2.rem_euclid(2.0 * PI),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trapezoid_weights() {
        let w = trapezoid(&[0.0, 1.0, 3.0]);
        assert_eq!(w, vec![0.5, 1.5, 1.0]);
        assert_eq!(w.iter().sum::<f64>(), 3.0);
    }

    #[test]
    fn invalid_boxes() {
        let g = Grid::uniform([10, 10, 10], [1e-3; 3]).unwrap();
        assert!(Nf2ff::new(&g, [2, 2, 2], [8, 8, 8], [1e9]).is_ok());
        assert!(Nf2ff::new(&g, [2, 2, 2], [8, 8, 2], [1e9]).is_err());
        assert!(Nf2ff::new(&g, [0, 2, 2], [8, 8, 8], [1e9]).is_err());
        assert!(Nf2ff::new(&g, [2, 2, 2], [8, 8, 10], [1e9]).is_err());
        assert!(Nf2ff::new(&g, [2, 2, 2], [8, 8, 11], [1e9]).is_err());
        assert!(Nf2ff::new(&g, [2, 2, 2], [8, 8, 8], Vec::new()).is_err());
        assert!(Nf2ff::new(&g, [2, 2, 2], [8, 8, 8], [-1.0]).is_err());
        assert!(Nf2ff::with_faces(&g, [2, 2, 2], [8, 8, 8], [1e9], [false; 6]).is_err());
        // An unrecorded face may lie on the outer plane.
        let mut f = [true; 6];
        f[Face::ZMin.index()] = false;
        assert!(Nf2ff::with_faces(&g, [2, 2, 0], [8, 8, 8], [1e9], f).is_ok());
    }

    /// Analytic Hertzian dipole currents on a box: the transform must
    /// reproduce `E_θ = jηkIl sinθ/(4π)` and `D = 1.5`.
    #[test]
    fn analytic_hertzian_dipole() {
        // Fields of a z-directed electric dipole Il = 1 at the origin
        // (Balanis eq. 4-8a..c, 4-10a..c), e^{jωt}.
        let f = 1e9;
        let kw = 2.0 * PI * f / C0;
        let fields = |p: [f64; 3]| -> ([Complex64; 3], [Complex64; 3]) {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let (st, ct) = ((p[0] * p[0] + p[1] * p[1]).sqrt() / r, p[2] / r);
            let phi = p[1].atan2(p[0]);
            let j = Complex64::new(0.0, 1.0);
            let g = Complex64::cis(-kw * r);
            // jkr as a complex number.
            let kr = Complex64::new(0.0, kw * r);
            let er = g * (Z0 * ct / (2.0 * PI * r * r)) * (1.0 + 1.0 / kr);
            let base = g * j * (Z0 * kw * st / (4.0 * PI * r));
            let et = base * (1.0 + 1.0 / kr) - base * (1.0 / (kw * r * kw * r));
            let hp = g * j * (kw * st / (4.0 * PI * r)) * (1.0 + 1.0 / kr);
            let (sp, cp) = phi.sin_cos();
            let rh = [st * cp, st * sp, ct];
            let th = [ct * cp, ct * sp, -st];
            let ph = [-sp, cp, 0.0];
            let e = [0, 1, 2].map(|a| er * rh[a] + et * th[a]);
            let h = [0, 1, 2].map(|a| hp * ph[a]);
            (e, h)
        };
        // A 0.4 m cube sampled every 10 mm (λ/30).
        let n = 41;
        let xs: Vec<f64> = (0..n)
            .map(|i| -0.2 + 0.4 * i as f64 / (n - 1) as f64)
            .collect();
        let mut faces = Vec::new();
        for face in Face::ALL {
            let ax = face.axis().index();
            let x0 = if face.is_max() { 0.2 } else { -0.2 };
            let (a1, a2) = ((ax + 1) % 3, (ax + 2) % 3);
            let wt = trapezoid(&xs);
            let mut fd = FaceData {
                face,
                points: Vec::new(),
                weights: Vec::new(),
                j: vec![Vec::new()],
                m: vec![Vec::new()],
            };
            let nn = normal(face);
            for (p, &u) in xs.iter().enumerate() {
                for (q, &v) in xs.iter().enumerate() {
                    let mut pt = [0.0; 3];
                    pt[ax] = x0;
                    pt[a1] = u;
                    pt[a2] = v;
                    let (e, h) = fields(pt);
                    fd.points.push(pt);
                    fd.weights.push(wt[p] * wt[q]);
                    fd.j[0].push(cross(h, nn.map(|x| -x)));
                    fd.m[0].push(cross(e, nn));
                }
            }
            faces.push(fd);
        }
        let res = Nf2ffResult {
            freqs: vec![f],
            center: [0.0; 3],
            faces,
            mirrors: Vec::new(),
        };
        // P_rad of the dipole: η (kIl)²/(12π).
        let p_exact = Z0 * kw * kw / (12.0 * PI);
        let p_box = res.radiated_power(0);
        assert!((p_box / p_exact - 1.0).abs() < 2e-3, "{p_box} vs {p_exact}");
        let p_far = res.radiated_power_far(0, 45, 8);
        assert!((p_far / p_exact - 1.0).abs() < 2e-3, "{p_far} vs {p_exact}");
        for &t in &[0.3, 0.9, PI / 2.0, 2.2] {
            let ff = res.far_field(0, t, 0.7);
            let want = Complex64::new(0.0, Z0 * kw * t.sin() / (4.0 * PI));
            assert!(
                (ff.e_theta - want).norm() < 2e-3 * want.norm(),
                "{t}: {:?} vs {want}",
                ff.e_theta
            );
            assert!(ff.e_phi.norm() < 1e-3 * want.norm());
        }
        let d = res.max_directivity(0);
        assert!((d.directivity - 1.5).abs() < 5e-3, "{d:?}");
        assert!((d.theta - PI / 2.0).abs() < 1e-2, "{d:?}");
    }
}
