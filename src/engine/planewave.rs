//! Plane-wave excitation by the total-field / scattered-field (TF/SF)
//! technique (Taflove & Hagness, *Computational Electrodynamics*, 3rd ed.,
//! ch. 5).
//!
//! # Regions
//!
//! A box of mesh nodes `start..=stop` (all three axes, `start < stop`)
//! splits the grid into the **total-field** region (the box, including the
//! E edges lying on its surface) and the **scattered-field** region
//! (everything else). Inside the box the fields are the total fields, the
//! incident plane wave plus whatever the structure scatters; outside they
//! are the scattered fields only, so the incident wave is invisible there.
//!
//! In engine terms an E edge `u_a(p)` is total-field when it lies in the
//! closed box (`start_a ≤ p_a < stop_a` along the edge,
//! `start ≤ p ≤ stop` across it) and an H dual edge `i_a(p)` is
//! total-field when `start_a ≤ p_a ≤ stop_a` and its two half-cell
//! coordinates lie strictly inside the box (`start ≤ p < stop` across it).
//! With these definitions every total-field H edge only reads total-field E
//! edges, so the regions are coupled only through the E edges on the box
//! surface and the H edges just outside it.
//!
//! # Consistency conditions
//!
//! The kernels update every edge from its neighbours as stored. Where a
//! stencil mixes the two regions the operator adds the missing incident
//! part afterwards (the update is linear, so this is exact):
//!
//! * after the H update, every scattered-field H edge `i_h` next to a
//!   total-field E edge `u_e` gets `+db_h·K_he·u_inc,e^n` (the kernel used
//!   the total `u_e`, the scattered value was needed);
//! * after the E update, every total-field E edge gets
//!   `+cb_e·K_he·i_inc,h^{n+½}` for each scattered-field H neighbour.
//!
//! `K_he = ±1` is the sign of `u_e` in the circulation of `i_h` (equal to
//! the sign of `i_h` in the circulation of `u_e`: the integral-form curls
//! are transposes, see the [engine docs](super#energy)), and `cb`, `db`
//! are the engine's own coefficients, so materials and PEC edges on the
//! box surface are handled consistently (a PEC edge, `cb = 0`, gets no
//! correction). Box faces lying on the domain boundary have no scattered
//! side and get no corrections.
//!
//! # Incident field
//!
//! `u_inc = E_inc·l` and `i_inc = H_inc·l*` are the incident field at the
//! edge's position times its length, with
//!
//! ```text
//! E_inc(r, t) = E0 · F_E(ξ, t),   H_inc(r, t) = (k̂ × E0)/η · F_H(ξ, t),
//! ξ = k̂·(r − r_ref),
//! ```
//!
//! where `r_ref` is the box corner the wave enters first (the corner with
//! the smallest `k̂·r`), `E0` the polarisation vector (V/m, perpendicular
//! to `k̂`) and `η = Z0·√(μ_r/ε_r)` the background wave impedance (see
//! [below](self#background-medium)).
//!
//! `F_E` and `F_H` come from an **auxiliary 1-D Yee grid** along `k̂`
//! (Taflove & Hagness ch. 5, incident-wave source condition) using the
//! engine's time step: E nodes at `ξ_m = (m − m0)·Δξ`, H nodes half way in
//! between, a hard source at node 0 and a graded matched-loss layer
//! terminated by a short at the far end. Because the 1-D grid runs on the
//! same time step with the same staggering, the incident E is available at
//! `n·dt` and the incident H at `(n + ½)·dt`, exactly when the corrections
//! need them. Values between nodes are interpolated with 4-point
//! (cubic) Lagrange interpolation.
//!
//! **Dispersion matching.** The 1-D spacing `Δξ` is chosen so that the 1-D
//! grid's numerical wavenumber equals the 3-D Yee wavenumber along `k̂` at
//! the matching frequency `f_m`:
//!
//! ```text
//! (sin(ω dt/2)/(v dt))² = Σ_a (sin(k k̂_a Δ_a/2)/Δ_a)²     (3-D, solved for k)
//!  sin(ω dt/2)/(v dt)   = sin(k Δξ/2)/Δξ                 (1-D, solved for Δξ)
//! ```
//!
//! with `Δ_a` the mean cell size of the box along each axis. Along a mesh
//! axis on a uniform mesh this gives `Δξ = Δ_a` and the 3-D plane wave *is*
//! the 1-D solution, so the incident field is reproduced to `f32` rounding
//! at every frequency. For oblique incidence the match is exact only at
//! `f_m`; elsewhere the residual dispersion mismatch, the cubic
//! interpolation and (on graded meshes) the variation of the cell size
//! leak a small part of the incident wave into the scattered-field region
//! (`tests/planewave.rs` reports the measured levels).
//!
//! The source of the 1-D grid is driven `m0` cells before `r_ref`; its
//! time function is advanced by the numerical travel time
//! `m0·Δξ·k/ω_m` so that `F_E(0, t) = a(t)` at `f_m` (exactly, for all
//! frequencies, in the axis-aligned uniform case).
//!
//! # Background medium
//!
//! The incident wave is that of the homogeneous, isotropic background
//! given by the spec (`ε_r`, `μ_r`, and the conductivities `κ`, `σ_m`):
//! the auxiliary grid runs at its speed `v = c0/√(ε_r μ_r)` and `η` is its
//! (lossless-part) impedance. [`PlaneWave::new`] checks that the update
//! coefficients of every edge taking part in a correction are those of
//! that background (to `f32` rounding) and fails otherwise, so a
//! background that differs from the spec, or is not homogeneous over the
//! box surface (a substrate crossing it, a lumped element on it), is an
//! error rather than a leaking source. [`Medium::of_box_surface`] finds
//! the background of an engine; [`crate::Simulation`] does this
//! automatically (from the compiled materials, [`crate::compile::planewave`]).
//!
//! # Lossy background
//!
//! With conductivities, the auxiliary grid uses the engine's semi-implicit
//! loss terms (`ca = (1 − s)/(1 + s)`, `cb ∝ 1/(1 + s)` with
//! `s = κ·dt/(2ε)`, and alike for H with `σ_m·dt/(2μ)`), so along a mesh
//! axis on a uniform mesh it is again the exact discrete 3-D plane wave,
//! attenuation included. The auxiliary source sits `m0` cells before the
//! reference corner; its amplitude is raised by the grid's own attenuation
//! over that distance at the matching frequency (and advanced by the
//! phase delay), so `E_inc(r_ref) = E0·a(t)` holds exactly at `f_m` and to
//! `O(Δα·m0Δξ)` elsewhere (`Δα` the variation of the attenuation constant
//! over the band; `m0` is a few cells). For oblique incidence the spacing
//! `Δξ` is matched with the lossless dispersion relation.
//!
//! # Layered and dispersive backgrounds
//!
//! A background stratified along one axis `s` ([`PlaneWaveSpec::with_layers`]:
//! dielectric or lossy layers, a substrate on a ground plane, layers
//! crossing the box surface), or one with dispersive poles on the box
//! surface ([`PlaneWaveSpec::with_dispersion`], Drude/Lorentz/Debye of a
//! [`Dispersive`](super::Dispersive) operator), gets an incident field
//! computed **in the frequency domain from the engine's own discrete
//! equations** instead of the auxiliary grid:
//!
//! * The wave has a fixed transverse slowness `p = k̂_t/v` (`v` the phase
//!   velocity of the incidence medium at the matching frequency, so the
//!   direction is `k̂` there): `x(r, t) = g(s, t − p·(r − r_ref))`. On a
//!   uniform transverse mesh `e^{−jω p·r}` is an exact eigenfunction of the
//!   transverse Yee differences, so per frequency the 3-D update equations
//!   reduce to a 1-D system along one column of edges (`u_t1, u_t2, i_s`
//!   on node levels, `u_s, i_t1, i_t2` on dual levels), with the engine's
//!   coefficients (semi-implicit losses) and the poles' discrete
//!   susceptibilities (`χ(Ω_t)` trapezoidal, `χ(Ω)` leapfrog, see
//!   [`dispersive`](super::dispersive)). Its solution is the exact discrete
//!   plane wave of the layered medium — reflections and transmissions of
//!   every interface, the mesh grading along `s`, oblique incidence,
//!   evanescent (total internal reflection) layers — for every frequency,
//!   not only at a matching frequency.
//! * The column is that of the box surface (an interior mesh line through
//!   a box edge; in a homogeneous dispersive background only its medium
//!   is used). In a layered background it spans the whole domain along
//!   `s` and ends on the domain's PEC/PMC face or, for an open end
//!   ([`LayerEnd`], [`PlaneWaveSpec::with_layer_ends`]), in a uniform
//!   extension of the end medium with a stretched-coordinate PML. A current
//!   sheet in the extension on the side the wave comes from radiates it;
//!   the same sheet in the same line filled with the incidence medium
//!   (open at both ends) gives the pure incident wave, which normalises
//!   the solution so that the incident `E_t` at `r_ref` is `E0_t·a(t)`
//!   (the component along `s` follows from the discrete equations).
//! * Per frequency bin of an FFT window (time step `dt/2`; bins above the
//!   engine's Nyquist frequency, where `a(t)` has no content, and where the
//!   incident wave does not propagate in the incidence medium are skipped)
//!   the two sheet solutions are combined for the polarisation; the
//!   column fields times the excitation spectrum are inverse-FFT'd into
//!   time series per level and component. The DC bin is singular and not
//!   solved: it is a constant per series, fixed by the field being zero
//!   once the pulse has passed. The window grows until the series have
//!   decayed to 1e-6 of their peak in its third quarter (else
//!   [`Error::Invalid`]: resonant layers or an excitation that does not
//!   end); the incident field before `t = 0` must be below 1e-5 (strong
//!   dispersion near a cutoff, e.g. a Drude medium near its plasma
//!   frequency, sends slow components that would have had to arrive
//!   before the run starts: an error asking to delay the excitation).
//! * Each correction reads its series with cubic interpolation at its
//!   transverse delay `p·(r − r_ref)` (fixed per edge). Corrections of
//!   edges whose poles the [`Dispersive`](super::Dispersive) operator
//!   integrates implicitly (trapezoidal, and all Debye poles) are added to
//!   the start value before the kernel (`pre_e` / `pre_h`, divided by
//!   `ca` / `da`), so that the implicit solve sees them.
//!
//! This needs the excitation's duration ([`PlaneWaveSpec::with_duration`]):
//! the amplitude function is sampled once at set-up. Every edge taking part
//! must have the column's medium at its level (layered) or the background
//! medium (homogeneous), poles included, else [`Error::Invalid`]. Measured
//! (`tests/planewave_layered.rs`): leakage −125 … −134 dB for a half-space,
//! a slab and a grounded substrate across the box at 0–40° (both
//! polarisations, also on a graded mesh along the layers) and for Debye,
//! Drude and Lorentz backgrounds; the field in the box equals the discrete
//! 1-D Yee response of the stack to 1.4e-7 at normal incidence.
//!
//! # Energy
//!
//! The injected wave is real field energy: inside the box the fields are
//! the total fields, so [`Engine::energy`](super::Engine::energy) includes
//! the incident wave while it crosses the box (the source is not
//! energy-conserving, like any source, and the energy look-ahead does not
//! see the `post_h` corrections of the H edges just outside the box, which
//! is an error of the order of the incident energy in one cell layer while
//! the wave is on the surface). Once the incident pulse has left the box,
//! the incident field is zero everywhere and the engine energy is exactly
//! the scattered-field energy. [`PlaneWave::transit_time`] gives that
//! time; [`crate::Simulation`] adds it (and the excitation delay) to the
//! excitation end of its [`RunLimits`](super::RunLimits), so the energy end
//! criterion only judges the scattered field, relative to the largest
//! total energy seen (normally the incident pulse inside the box).
//!
//! # Limitations
//!
//! * The background must be homogeneous and isotropic around the box
//!   surface, or stratified along one axis that the wave crosses
//!   (`|k̂_s| ≥ 1e-3`), with isotropic layers; scatterers belong strictly
//!   inside the box. The spectral incident field is exact on a uniform
//!   transverse mesh (graded meshes across the layers leak like the
//!   auxiliary grid's oblique incidence); it needs an excitation that ends.
//! * The box must stay clear of absorbing boundaries (CPML, Mur): the
//!   corrections are applied in the serial `post_h` / `post_e` hooks and
//!   assume the plain Yee update on the surface.

mod spectral;

use std::collections::HashMap;

pub use spectral::LayerEnd;

use super::Engine;
use super::dispersive::{EdgePole, Field, Pole, Scheme};
use super::excite::Amplitude;
use super::fields::Fields;
use super::medium::{CoefficientMedia, EdgeMedia, Medium};
use super::operator::{Hooks, Operator, StepContext};
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};
use crate::post::Complex64;
use crate::{Error, Result};

/// Poles per edge: `(electric, axis index, node)`.
type PoleMap = HashMap<(bool, usize, [usize; 3]), Vec<Pole>>;

fn pole_map(entries: &[EdgePole]) -> PoleMap {
    let mut m = PoleMap::new();
    for e in entries {
        m.entry((e.field == Field::Electric, e.axis.index(), e.node))
            .or_default()
            .push(e.pole);
    }
    m
}

/// What [`PlaneWave::new`] hands to the spectral set-up.
struct SpectralInput<'a> {
    k_hat: [f64; 3],
    e0: [f64; 3],
    h0: [f64; 3],
    r_ref: [f64; 3],
    f: f64,
    poles: &'a PoleMap,
    e_raw: Vec<Raw>,
    h_raw: Vec<Raw>,
}

/// Geometry and physics of a [`PlaneWave`] (see the [module docs](self)).
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneWaveSpec {
    /// First node of the total-field box.
    pub start: [usize; 3],
    /// Last node of the total-field box (`start < stop` on every axis).
    pub stop: [usize; 3],
    /// Propagation direction (any length > 0; normalised).
    pub direction: [f64; 3],
    /// Incident E vector \[V/m\] at amplitude 1 of the time function;
    /// must be perpendicular to `direction`.
    pub polarization: [f64; 3],
    /// Relative permittivity of the background around the box (default 1).
    pub eps_r: f64,
    /// Relative permeability of the background around the box (default 1).
    pub mu_r: f64,
    /// Electric conductivity of the background \[S/m\] (default 0).
    pub kappa: f64,
    /// Magnetic conductivity of the background \[Ω/m\] (default 0).
    pub sigma_m: f64,
    /// Frequency \[Hz\] at which the auxiliary grid's dispersion is matched
    /// to the 3-D grid along `direction`; `0` (default) picks 20 mean cells
    /// per wavelength. With a [layered or dispersive
    /// background](self#layered-and-dispersive-backgrounds), the frequency
    /// at which the incidence medium's phase velocity sets the transverse
    /// slowness (the direction is exact there).
    pub match_frequency: f64,
    /// Stratification axis of a [layered
    /// background](self#layered-and-dispersive-backgrounds) (`None`:
    /// homogeneous).
    pub layers: Option<Axis>,
    /// How the layered medium ends at the lower and upper domain faces
    /// along [`layers`](Self::layers) (default: open at both).
    pub layer_ends: [LayerEnd; 2],
    /// Time \[s\] after which the amplitude function is zero (needed by
    /// the spectral incident field of layered and dispersive backgrounds;
    /// default 0).
    pub duration: f64,
    /// Dispersive poles of the background near the box (see
    /// [`with_dispersion`](Self::with_dispersion)).
    pub poles: Vec<EdgePole>,
    /// Time integration of those poles (that of the engine's
    /// [`Dispersive`](super::Dispersive) operator).
    pub pole_scheme: Scheme,
}

impl PlaneWaveSpec {
    /// Box `start..=stop` (node indices), propagation `direction`, E vector
    /// `polarization` \[V/m\]; vacuum background, default matching.
    pub fn new(
        start: [usize; 3],
        stop: [usize; 3],
        direction: [f64; 3],
        polarization: [f64; 3],
    ) -> Self {
        Self {
            start,
            stop,
            direction,
            polarization,
            eps_r: 1.0,
            mu_r: 1.0,
            kappa: 0.0,
            sigma_m: 0.0,
            match_frequency: 0.0,
            layers: None,
            layer_ends: [LayerEnd::Open; 2],
            duration: 0.0,
            poles: Vec::new(),
            pole_scheme: Scheme::Trapezoidal,
        }
    }

    /// Sets the (lossless part of the) background material around the
    /// box.
    #[must_use]
    pub fn with_background(mut self, eps_r: f64, mu_r: f64) -> Self {
        self.eps_r = eps_r;
        self.mu_r = mu_r;
        self
    }

    /// Sets the background conductivities: electric `kappa` \[S/m\] and
    /// magnetic `sigma_m` \[Ω/m\] (see the
    /// [module docs](self#lossy-background)).
    #[must_use]
    pub fn with_losses(mut self, kappa: f64, sigma_m: f64) -> Self {
        self.kappa = kappa;
        self.sigma_m = sigma_m;
        self
    }

    /// Sets the whole background medium.
    #[must_use]
    pub fn with_medium(self, m: Medium) -> Self {
        self.with_background(m.eps_r, m.mu_r)
            .with_losses(m.kappa, m.sigma_m)
    }

    /// The background medium.
    pub fn medium(&self) -> Medium {
        Medium {
            eps_r: self.eps_r,
            mu_r: self.mu_r,
            kappa: self.kappa,
            sigma_m: self.sigma_m,
        }
    }

    /// Sets the dispersion-matching frequency \[Hz\] (`0`: default).
    #[must_use]
    pub fn with_match_frequency(mut self, f: f64) -> Self {
        self.match_frequency = f;
        self
    }

    /// Declares the background stratified along `axis`: the media may vary
    /// along it (also across the box surface) but not across it (see the
    /// [module docs](self#layered-and-dispersive-backgrounds)).
    #[must_use]
    pub fn with_layers(mut self, axis: Axis) -> Self {
        self.layers = Some(axis);
        self
    }

    /// Sets how the layered medium ends at the lower and upper domain
    /// faces along the stratification axis (default [`LayerEnd::Open`]).
    #[must_use]
    pub fn with_layer_ends(mut self, lower: LayerEnd, upper: LayerEnd) -> Self {
        self.layer_ends = [lower, upper];
        self
    }

    /// Sets the time \[s\] after which the amplitude function is zero.
    #[must_use]
    pub fn with_duration(mut self, d: f64) -> Self {
        self.duration = d;
        self
    }

    /// Takes the dispersive poles of the engine (e.g.
    /// [`Dispersive::entries`](super::Dispersive::entries) and
    /// [`Dispersive::scheme`](super::Dispersive::scheme)) into account: a
    /// box surface with poles gets the spectral incident field of the
    /// dispersive background. Only the entries within one node of the box
    /// on at least two axes are kept (the box surface and any column
    /// through it).
    #[must_use]
    pub fn with_dispersion(mut self, entries: &[EdgePole], scheme: Scheme) -> Self {
        let (lo, hi) = (self.start, self.stop);
        self.poles = entries
            .iter()
            .filter(|e| {
                (0..3)
                    .filter(|&a| e.node[a] + 1 >= lo[a] && e.node[a] <= hi[a] + 1)
                    .count()
                    >= 2
            })
            .copied()
            .collect();
        self.pole_scheme = scheme;
        self
    }
}

/// One correction: `field[axis][node] += coef · Σ w_q · aux[m + q]`
/// (spectral: `series[n·UP + m + q]` at step `n`).
#[derive(Debug, Clone, Copy)]
struct Correction {
    axis: Axis,
    node: [usize; 3],
    coef: f64,
    series: usize,
    m: usize,
    w: [f64; 4],
}

/// Cells of the matched-loss layer at the far end of the auxiliary grid.
const AUX_PML: usize = 64;
/// Design round-trip reflection of that layer.
const AUX_PML_R: f64 = 1e-12;
/// Cells of the taper from a lossy background to a distortionless line
/// before that layer (lossy backgrounds only).
const AUX_TAPER: usize = 256;

/// Total-field / scattered-field plane-wave source; see the
/// [module docs](self).
///
/// ```
/// use ems::engine::{Engine, Grid, PlaneWave, PlaneWaveSpec};
///
/// let grid = Grid::uniform([20, 20, 20], [1e-3; 3])?;
/// let mut e = Engine::builder(grid).threads(1).build()?;
/// let spec = PlaneWaveSpec::new([5; 3], [15; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
/// let pw = PlaneWave::new(&e, &spec, |t| (-((t - 1e-10) / 3e-11).powi(2)).exp())?;
/// assert!((pw.spacing() - 1e-3).abs() < 1e-15); // axis-aligned: the cell size
/// let h = e.add_operator(pw);
/// e.run(10);
/// # let _ = h;
/// # Ok::<(), ems::Error>(())
/// ```
pub struct PlaneWave {
    spec: PlaneWaveSpec,
    /// The spectral incident field (layered or dispersive background).
    spectral: Option<Box<Spectral>>,
    amplitude: Amplitude,
    k_hat: [f64; 3],
    e0: [f64; 3],
    h0: [f64; 3],
    r_ref: [f64; 3],
    dxi: f64,
    m0: usize,
    lead: f64,
    gain: f64,
    transit: f64,
    main: usize,
    e: Vec<f64>,
    h: Vec<f64>,
    ca: Vec<f64>,
    cb: Vec<f64>,
    da: Vec<f64>,
    db: Vec<f64>,
    e_corr: Vec<Correction>,
    h_corr: Vec<Correction>,
}

/// State of the spectral incident field.
#[derive(Debug, Clone)]
struct Spectral {
    inc: spectral::Incident,
    s: Axis,
    /// Steps done (time of the E voltages `steps·dt`).
    steps: u64,
    dt: f64,
    /// Node and dual positions along `s` of the stored levels.
    nodes: Vec<f64>,
    duals: Vec<f64>,
    /// Corrections of edges with implicitly integrated (trapezoidal)
    /// poles, applied before the kernel (`pre_e` / `pre_h`, coefficient
    /// divided by `ca` / `da`) so that the poles' implicit solve sees them.
    e_pre: Vec<Correction>,
    h_pre: Vec<Correction>,
}

impl core::fmt::Debug for PlaneWave {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PlaneWave")
            .field("spec", &self.spec)
            .field("spacing", &self.dxi)
            .field("spectral", &self.spectral.is_some())
            .field("reference_point", &self.r_ref)
            .field("e_corrections", &self.e_corr.len())
            .field("h_corrections", &self.h_corr.len())
            .finish_non_exhaustive()
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Bisection for the root of a function that is `< 0` at `lo` and `> 0`
/// at `hi` (or the reverse); 200 halvings reach `f64` resolution.
fn bisect(mut lo: f64, mut hi: f64, f: impl Fn(f64) -> f64) -> f64 {
    let s_lo = f(lo) < 0.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if (f(mid) < 0.0) == s_lo {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The auxiliary spacing `Δξ` and the matched wavenumber `k` (see the
/// [module docs](self#incident-field)).
fn match_spacing(dt: f64, v: f64, f: f64, dir: [f64; 3], cell: [f64; 3]) -> Result<(f64, f64)> {
    let s = (core::f64::consts::PI * f * dt).sin() / (v * dt);
    // 3-D: Σ (sin(k d_a Δ_a/2)/Δ_a)² = s², increasing in k up to the first
    // axis reaching k d_a Δ_a = π.
    let k_hi = (0..3)
        .filter(|&a| dir[a].abs() > 0.0)
        .map(|a| core::f64::consts::PI / (dir[a].abs() * cell[a]))
        .fold(f64::INFINITY, f64::min);
    let rhs = |k: f64| {
        (0..3)
            .map(|a| {
                let x = (0.5 * k * dir[a] * cell[a]).sin() / cell[a];
                x * x
            })
            .sum::<f64>()
            - s * s
    };
    if !(s.is_finite() && s > 0.0 && rhs(k_hi) > 0.0) {
        return Err(Error::Invalid(format!(
            "plane wave: matching frequency {f} Hz is beyond the resolution of the mesh"
        )));
    }
    let k = bisect(0.0, k_hi, rhs);
    // Along a mesh axis the 1-D grid with that axis' spacing is the exact
    // match (the 3-D relation reduces to the 1-D one).
    let axes: Vec<usize> = (0..3).filter(|&a| dir[a] != 0.0).collect();
    if let &[a] = axes.as_slice() {
        return Ok((cell[a], k));
    }
    // 1-D: sin(k Δξ/2)/Δξ = s, decreasing in Δξ on [v dt, π/k].
    let g = |d: f64| (0.5 * k * d).sin() / d - s;
    let lo = v * dt;
    if g(lo) <= 0.0 {
        // The 3-D grid is (numerically) dispersion-free along k̂: the
        // magic-time-step 1-D grid matches.
        return Ok((lo, k));
    }
    let hi = core::f64::consts::PI / k;
    if g(hi) > 0.0 {
        return Err(Error::Invalid(format!(
            "plane wave: cannot match the auxiliary grid dispersion at {f} Hz"
        )));
    }
    Ok((bisect(lo, hi, g), k))
}

/// Propagation constant `γ·Δξ` of the (lossy) auxiliary grid at
/// `ω·dt = wdt`, for per-step loss numbers `se`, `sm` and Courant number
/// `r = v·dt/Δξ`: with `A = 2j·sin(ωdt/2) + 2s·cos(ωdt/2)` for either
/// field, `sinh(γΔξ/2) = √(A_e·A_h)/(2r)`; the root with `Re γ ≥ 0`
/// (decaying downstream).
fn aux_gamma(wdt: f64, se: f64, sm: f64, r: f64) -> Complex64 {
    let (sn, cs) = (0.5 * wdt).sin_cos();
    let a = |s: f64| Complex64::new(2.0 * s * cs, 2.0 * sn);
    let z = (a(se) * a(sm)).sqrt() / (2.0 * r);
    // asinh z = ln(z + √(z² + 1)).
    let g = (z + (z * z + 1.0).sqrt()).ln() * 2.0;
    if g.re < 0.0 { -g } else { g }
}

/// The medium an edge's coefficients correspond to, for error messages.
fn describe(
    engine: &Engine,
    co: &super::Coefficients,
    what: &str,
    axis: Axis,
    node: [usize; 3],
    e: bool,
) -> String {
    let m = CoefficientMedia {
        grid: engine.grid(),
        co,
        dt: engine.dt(),
    };
    let v = if e {
        m.e_edge(axis, node)
    } else {
        m.h_edge(axis, node)
    };
    let (name, loss) = if e {
        ("eps_r", "kappa")
    } else {
        ("mu_r", "sigma_m")
    };
    match v {
        Some((x, l)) => format!(
            "the {what} edge along {axis} at {node:?} on the box surface has {name} = {x:.6}, \
             {loss} = {l:.6e}"
        ),
        None => format!("the {what} edge along {axis} at {node:?} on the box surface is PEC"),
    }
}

/// 4-point Lagrange stencil for fractional node position `s ≥ 1`: first
/// node and weights.
fn stencil(s: f64) -> (usize, [f64; 4]) {
    let m = s.floor();
    let t = s - m;
    let w = [
        -t * (t - 1.0) * (t - 2.0) / 6.0,
        (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0,
        -(t + 1.0) * t * (t - 2.0) / 2.0,
        (t + 1.0) * t * (t - 1.0) / 6.0,
    ];
    (m as usize - 1, w)
}

fn interp(v: &[f64], m: usize, w: &[f64; 4]) -> f64 {
    w[0] * v[m] + w[1] * v[m + 1] + w[2] * v[m + 2] + w[3] * v[m + 3]
}

/// A correction before its stencil is known: the corrected edge, the
/// coefficient without the incident field's polarisation, and the edge
/// whose incident value it needs (`comp` at `cnode`, at `pos`).
struct Raw {
    axis: Axis,
    node: [usize; 3],
    coef: f64,
    comp: Axis,
    cnode: [usize; 3],
    pos: [f64; 3],
    xi: f64,
}

impl PlaneWave {
    /// Sets up the source for `engine`'s grid, time step and coefficients
    /// (take them as final: build after any operator that changes
    /// coefficients on the box surface). `amplitude(t)` is the time
    /// function of the incident wave at the reference corner.
    ///
    /// # Errors
    /// [`Error::Invalid`] for a box outside the mesh or without volume, a
    /// zero or non-finite direction, a polarisation that is zero or not
    /// perpendicular to the direction, an invalid background or one that
    /// does not match the engine's coefficients on the box surface (see
    /// the [module docs](self#background-medium)), or a matching frequency
    /// the mesh cannot resolve;
    /// [`Error::Unsupported`] on a cylindrical grid.
    pub fn new(
        engine: &Engine,
        spec: &PlaneWaveSpec,
        amplitude: impl Fn(f64) -> f64 + Send + 'static,
    ) -> Result<Self> {
        let grid = engine.grid();
        if grid.is_cylindrical() {
            return Err(Error::Unsupported(
                "plane waves need a Cartesian grid".into(),
            ));
        }
        let n = grid.nodes();
        let (lo, hi) = (spec.start, spec.stop);
        if (0..3).any(|a| lo[a] >= hi[a] || hi[a] >= n[a]) {
            return Err(Error::Invalid(format!(
                "plane wave: box {lo:?}..={hi:?} must have start < stop within the {n:?} nodes"
            )));
        }
        let len = dot(spec.direction, spec.direction).sqrt();
        if !(len.is_finite() && len > 0.0) {
            return Err(Error::Invalid(format!(
                "plane wave: invalid propagation direction {:?}",
                spec.direction
            )));
        }
        let k_hat = spec.direction.map(|d| d / len);
        let e0 = spec.polarization;
        let e_len = dot(e0, e0).sqrt();
        if !(e_len.is_finite() && e_len > 0.0) || dot(e0, k_hat).abs() > 1e-9 * e_len {
            return Err(Error::Invalid(format!(
                "plane wave: polarisation {e0:?} must be non-zero and perpendicular to the \
                 direction {k_hat:?}"
            )));
        }
        let medium = spec.medium();
        if medium.validate().is_err() {
            return Err(Error::Invalid(format!(
                "plane wave: invalid background eps_r = {}, mu_r = {}, kappa = {}, sigma_m = {}",
                spec.eps_r, spec.mu_r, spec.kappa, spec.sigma_m
            )));
        }
        let (er, mr) = (spec.eps_r, spec.mu_r);
        let v = medium.velocity();
        let eta = medium.impedance();
        let h0 = cross(k_hat, e0).map(|x| x / eta);
        let dt = engine.dt();
        // Per-step loss numbers of the background, `G·dt/(2C)`, `R·dt/(2L)`.
        let se = spec.kappa * dt / (2.0 * EPS0 * er);
        let sm = spec.sigma_m * dt / (2.0 * MU0 * mr);
        // Coefficients the background gives an edge of geometry factor
        // `geo`, as the coefficient builder computes them.
        let expect = |geo: f64, c: f64, g: f64| {
            let (cap, cond) = (c * geo, g * geo);
            let den = cap + 0.5 * cond * dt;
            (((cap - 0.5 * cond * dt) / den) as f32, (dt / den) as f32)
        };
        let matches = |(a, b): (f32, f32), (ea, eb): (f32, f32)| {
            (a - ea).abs() <= 1e-5 && (b - eb).abs() <= 1e-5 * eb.abs()
        };
        let mut mismatch: Option<String> = None;
        let cell = Axis::ALL.map(|ax| {
            let a = ax.index();
            let l = grid.lines(ax);
            (l[hi[a]] - l[lo[a]]) / (hi[a] - lo[a]) as f64
        });
        let mean = (cell[0] + cell[1] + cell[2]) / 3.0;
        let f = if spec.match_frequency > 0.0 {
            spec.match_frequency
        } else {
            v / (20.0 * mean)
        };
        if !f.is_finite() {
            return Err(Error::Invalid(format!(
                "plane wave: invalid matching frequency {f}"
            )));
        }
        let (dxi, k) = match_spacing(dt, v, f, k_hat, cell)?;
        let r_ref = Axis::ALL.map(|ax| {
            let a = ax.index();
            let l = grid.lines(ax);
            if k_hat[a] >= 0.0 { l[lo[a]] } else { l[hi[a]] }
        });
        let xi = |r: [f64; 3]| dot(k_hat, [r[0] - r_ref[0], r[1] - r_ref[1], r[2] - r_ref[2]]);

        let tf_h = |d: Axis, q: [usize; 3]| {
            let (e, f) = d.others();
            let (di, ei, fi) = (d.index(), e.index(), f.index());
            (lo[di]..=hi[di]).contains(&q[di])
                && (lo[ei]..hi[ei]).contains(&q[ei])
                && (lo[fi]..hi[fi]).contains(&q[fi])
        };
        let h_valid = |d: Axis, q: [usize; 3]| {
            let (e, f) = d.others();
            (0..3).all(|a| q[a] < n[a])
                && q[e.index()] + 1 < n[e.index()]
                && q[f.index()] + 1 < n[f.index()]
        };
        let e_pos = |a: Axis, p: [usize; 3]| {
            let (b, c) = a.others();
            let mut r = [0.0; 3];
            r[a.index()] = grid.dual_line(a, p[a.index()]);
            r[b.index()] = grid.lines(b)[p[b.index()]];
            r[c.index()] = grid.lines(c)[p[c.index()]];
            r
        };
        let h_pos = |d: Axis, q: [usize; 3]| {
            let (e, f) = d.others();
            let mut r = [0.0; 3];
            r[d.index()] = grid.lines(d)[q[d.index()]];
            r[e.index()] = grid.dual_line(e, q[e.index()]);
            r[f.index()] = grid.dual_line(f, q[f.index()]);
            r
        };

        let co = engine.coefficients();
        let (mut e_raw, mut h_raw) = (Vec::new(), Vec::new());
        for a in Axis::ALL {
            let (b, c) = a.others();
            let (ai, bi, ci) = (a.index(), b.index(), c.index());
            let cb_a = co.cb(a);
            for pa in lo[ai]..hi[ai] {
                for pb in lo[bi]..=hi[bi] {
                    for pc in lo[ci]..=hi[ci] {
                        let on_surface =
                            pb == lo[bi] || pb == hi[bi] || pc == lo[ci] || pc == hi[ci];
                        if !on_surface {
                            continue;
                        }
                        let mut p = [0; 3];
                        (p[ai], p[bi], p[ci]) = (pa, pb, pc);
                        let cbv = cb_a[grid.index(p)] as f64;
                        let mut e_checked = false;
                        // The H neighbours in the circulation of u_a(p),
                        // with their signs (see the E kernel).
                        let minus = |t: usize| {
                            let mut q = p;
                            q[t] = q[t].wrapping_sub(1);
                            q
                        };
                        let terms = [
                            (c, p, 1.0),
                            (c, minus(bi), -1.0),
                            (b, p, -1.0),
                            (b, minus(ci), 1.0),
                        ];
                        for (d, q, s) in terms {
                            if !h_valid(d, q) || tf_h(d, q) {
                                continue;
                            }
                            let di = d.index();
                            if mismatch.is_none() && cbv != 0.0 && !e_checked {
                                e_checked = true;
                                let geo = grid.dual_delta(b, pb) * grid.dual_delta(c, pc)
                                    / grid.delta(a, pa);
                                let got = (co.ca(a)[grid.index(p)], co.cb(a)[grid.index(p)]);
                                if !matches(got, expect(geo, EPS0 * er, spec.kappa)) {
                                    mismatch = Some(describe(engine, &co, "E", a, p, true));
                                }
                            }
                            let dbv = co.db(d)[grid.index(q)] as f64;
                            if mismatch.is_none() && dbv != 0.0 {
                                let (de, df) = d.others();
                                let geo = grid.delta(de, q[de.index()])
                                    * grid.delta(df, q[df.index()])
                                    / grid.dual_delta(d, q[di]);
                                let got = (co.da(d)[grid.index(q)], co.db(d)[grid.index(q)]);
                                if !matches(got, expect(geo, MU0 * mr, spec.sigma_m)) {
                                    mismatch = Some(describe(engine, &co, "H", d, q, false));
                                }
                            }
                            if cbv != 0.0 {
                                let pos = h_pos(d, q);
                                e_raw.push(Raw {
                                    axis: a,
                                    node: p,
                                    coef: cbv * s * grid.dual_delta(d, q[di]),
                                    comp: d,
                                    cnode: q,
                                    pos,
                                    xi: xi(pos),
                                });
                            }
                            if dbv != 0.0 {
                                let pos = e_pos(a, p);
                                h_raw.push(Raw {
                                    axis: d,
                                    node: q,
                                    coef: dbv * s * grid.delta(a, pa),
                                    comp: a,
                                    cnode: p,
                                    pos,
                                    xi: xi(pos),
                                });
                            }
                        }
                    }
                }
            }
        }
        drop(co);
        // Layered or dispersive background: the spectral incident field.
        let poles = pole_map(&spec.poles);
        let touches =
            |e: bool, axis: Axis, node: [usize; 3]| poles.contains_key(&(e, axis.index(), node));
        if spec.layers.is_some()
            || e_raw
                .iter()
                .any(|r| touches(true, r.axis, r.node) || touches(false, r.comp, r.cnode))
            || h_raw
                .iter()
                .any(|r| touches(false, r.axis, r.node) || touches(true, r.comp, r.cnode))
        {
            return Self::new_spectral(
                engine,
                spec,
                Box::new(amplitude),
                SpectralInput {
                    k_hat,
                    e0,
                    h0,
                    r_ref,
                    f,
                    poles: &poles,
                    e_raw,
                    h_raw,
                },
            );
        }
        // Homogeneous: only the components of the incident polarisation.
        e_raw.retain(|r| h0[r.comp.index()] != 0.0);
        h_raw.retain(|r| e0[r.comp.index()] != 0.0);
        for r in &mut e_raw {
            r.coef *= h0[r.comp.index()];
        }
        for r in &mut h_raw {
            r.coef *= e0[r.comp.index()];
        }
        if let Some(m) = mismatch {
            return Err(Error::Invalid(format!(
                "plane wave: {m}, not the background {medium:?} of the source; the medium \
                 around the total-field box must be homogeneous and match \
                 PlaneWaveSpec::with_background / with_losses"
            )));
        }

        // Auxiliary grid: E node m at ξ = (m − m0)·Δξ, H node m at
        // ξ = (m + ½ − m0)·Δξ.
        let (xi_min, xi_max) = e_raw
            .iter()
            .chain(&h_raw)
            .fold((0.0f64, 0.0f64), |(a, b), r| (a.min(r.xi), b.max(r.xi)));
        // Downstream the grid covers the whole domain, so that
        // [`incident_e`](Self::incident_e) works everywhere past the
        // reference corner; upstream only what the corrections need (the
        // source runs `m0` cells ahead of the reference).
        let xi_dom = (0..8).fold(xi_max, |m, c: usize| {
            let r = Axis::ALL.map(|ax| {
                let l = grid.lines(ax);
                if c >> ax.index() & 1 == 1 {
                    l[l.len() - 1]
                } else {
                    l[0]
                }
            });
            m.max(xi(r))
        });
        let m0 = (-xi_min / dxi).ceil().max(0.0) as usize + 3;
        let main = (xi_dom / dxi).ceil() as usize + m0 + 4;
        // The matched layer has equal (normalised) E and H losses, which is
        // reflectionless only on a line with equal background losses (a
        // distortionless line, impedance η at all frequencies). In a lossy
        // background the smaller of the two losses is first raised to the
        // larger one by a smooth (C²) taper.
        let taper = if se == sm { 0 } else { AUX_TAPER };
        let s_top = se.max(sm);
        let pml0 = main + taper;
        let nodes = pml0 + AUX_PML + 1;
        let r = v * dt / dxi;
        let smax = -AUX_PML_R.ln() * r / AUX_PML as f64;
        let loss = |x: f64, bg: f64| {
            let t = ((x - main as f64) / taper.max(1) as f64).clamp(0.0, 1.0);
            let g = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t);
            let pml = x - pml0 as f64;
            let p = if pml > 0.0 {
                smax * (pml / AUX_PML as f64).powi(3)
            } else {
                0.0
            };
            bg + (s_top - bg) * if taper == 0 { 0.0 } else { g } + p
        };
        let (mut ca, mut cb) = (vec![1.0; nodes], vec![0.0; nodes]);
        for m in 1..nodes - 1 {
            let s = loss(m as f64, se);
            ca[m] = (1.0 - s) / (1.0 + s);
            cb[m] = r / (1.0 + s);
        }
        let (mut da, mut db) = (vec![1.0; nodes - 1], vec![0.0; nodes - 1]);
        for m in 0..nodes - 1 {
            let s = loss(m as f64 + 0.5, sm);
            da[m] = (1.0 - s) / (1.0 + s);
            db[m] = r / (1.0 + s);
        }
        let finish = |raw: Vec<Raw>, half: f64| -> Vec<Correction> {
            raw.into_iter()
                .map(|c| {
                    let (m, w) = stencil(c.xi / dxi + m0 as f64 - half);
                    Correction {
                        axis: c.axis,
                        node: c.node,
                        coef: c.coef,
                        series: 0,
                        m,
                        w,
                    }
                })
                .collect()
        };
        let omega = 2.0 * core::f64::consts::PI * f;
        // The source runs m0 cells ahead: advance it by the travel time and,
        // in a lossy background, raise it by the attenuation over that
        // distance, both from the auxiliary grid's propagation constant at
        // the matching frequency (exact there).
        let (lead, gain) = if se == 0.0 && sm == 0.0 {
            (m0 as f64 * dxi * k / omega, 1.0)
        } else {
            let g = aux_gamma(omega * dt, se, sm, r) / dxi;
            let x = m0 as f64 * dxi;
            (g.im * x / omega, (g.re * x).exp())
        };
        // Time for the wave to cross the box: the far corner's ξ at the
        // background speed, plus 10 % for the numerical dispersion.
        let far = (0..8).fold(0.0f64, |m, c: usize| {
            let r = Axis::ALL.map(|ax| {
                let l = grid.lines(ax);
                let a = ax.index();
                l[if c >> a & 1 == 1 { hi[a] } else { lo[a] }]
            });
            m.max(xi(r))
        });
        let transit = 1.1 * far / v;
        Ok(Self {
            spec: spec.clone(),
            spectral: None,
            amplitude: Box::new(amplitude),
            k_hat,
            e0,
            h0,
            r_ref,
            dxi,
            m0,
            lead,
            gain,
            transit,
            main,
            e: vec![0.0; nodes],
            h: vec![0.0; nodes - 1],
            ca,
            cb,
            da,
            db,
            e_corr: finish(e_raw, 0.5),
            h_corr: finish(h_raw, 0.0),
        })
    }

    /// The spectral variant of [`new`](Self::new) (see the [module
    /// docs](self#layered-and-dispersive-backgrounds)).
    fn new_spectral(
        engine: &Engine,
        spec: &PlaneWaveSpec,
        amplitude: Amplitude,
        inp: SpectralInput<'_>,
    ) -> Result<Self> {
        let grid = engine.grid();
        let n = grid.nodes();
        let (lo, hi) = (spec.start, spec.stop);
        let k_hat = inp.k_hat;
        let s = spec.layers.unwrap_or_else(|| {
            Axis::ALL.into_iter().fold(Axis::X, |b, a| {
                if k_hat[a.index()].abs() > k_hat[b.index()].abs() {
                    a
                } else {
                    b
                }
            })
        });
        let si = s.index();
        // The column: on the box surface where possible (an interior mesh
        // line), else through the box.
        let mut col = [0usize; 3];
        for t in Axis::ALL {
            let ti = t.index();
            col[ti] = if t == s {
                0
            } else if lo[ti] > 0 {
                lo[ti]
            } else if hi[ti] + 1 < n[ti] {
                hi[ti]
            } else {
                (lo[ti] + hi[ti]) / 2
            };
        }
        let co = engine.coefficients();
        let dt = engine.dt();
        let med = |e: bool, axis: Axis, node: [usize; 3]| -> spectral::EdgeMed {
            let i = grid.index(node);
            let (b, c) = axis.others();
            let (ca, cb, geo) = if e {
                (
                    co.ca(axis)[i] as f64,
                    co.cb(axis)[i] as f64,
                    grid.dual_delta(b, node[b.index()]) * grid.dual_delta(c, node[c.index()])
                        / grid.delta(axis, node[axis.index()]),
                )
            } else {
                (
                    co.da(axis)[i] as f64,
                    co.db(axis)[i] as f64,
                    grid.delta(b, node[b.index()]) * grid.delta(c, node[c.index()])
                        / grid.dual_delta(axis, node[axis.index()]),
                )
            };
            spectral::EdgeMed {
                e,
                ca,
                cb,
                geo,
                poles: inp
                    .poles
                    .get(&(e, axis.index(), node))
                    .cloned()
                    .unwrap_or_default(),
            }
        };
        let slot_axis = |k: usize| -> (bool, Axis) {
            let (t1, t2) = s.others();
            [
                (true, t1),
                (true, t2),
                (false, s),
                (true, s),
                (false, t1),
                (false, t2),
            ][k]
        };
        let mut meds = Vec::new();
        // Material table: layered, per slot and engine level along s;
        // homogeneous, one E and one H medium from the box surface.
        let mut table = vec![[None::<usize>; 6]; n[si]];
        if spec.layers.is_some() {
            for (lev, row) in table.iter_mut().enumerate() {
                for (k, slot) in row.iter_mut().enumerate() {
                    if k >= 3 && lev + 1 >= n[si] {
                        continue;
                    }
                    let (e, axis) = slot_axis(k);
                    let mut node = col;
                    node[si] = lev;
                    let m = med(e, axis, node);
                    if m.cb != 0.0 {
                        meds.push(m);
                        *slot = Some(meds.len() - 1);
                    }
                }
            }
        } else {
            let e = inp.e_raw.first().map(|r| med(true, r.axis, r.node));
            let h = inp.h_raw.first().map(|r| med(false, r.axis, r.node));
            let (Some(e), Some(h)) = (e, h) else {
                return Err(Error::Invalid(
                    "plane wave: no edge of the box surface takes part".into(),
                ));
            };
            meds.push(e);
            meds.push(h);
        }
        let layered = spec.layers.is_some();
        let index = |k: usize, lev: usize| -> Option<usize> {
            if layered {
                table.get(lev).and_then(|r| r[k])
            } else {
                Some(usize::from(!slot_axis(k).0))
            }
        };
        // Every edge taking part must have the column's medium at its level.
        let wm = 2.0 * core::f64::consts::PI * inp.f;
        let check = |e: bool, axis: Axis, node: [usize; 3]| -> Result<()> {
            let m = med(e, axis, node);
            let Some(y) = m.y_at(wm, dt, spec.pole_scheme) else {
                return Ok(());
            };
            let k = spectral::slot_of(s, e, axis);
            let want = index(k, node[si]).and_then(|i| meds[i].y_at(wm, dt, spec.pole_scheme));
            match want {
                Some(w) if (y - w).norm() <= 1e-5 * w.norm() => Ok(()),
                _ => Err(Error::Invalid(format!(
                    "plane wave: {}, not the {} background along {s} at level {} of the \
                     source (column {col:?})",
                    describe(engine, &co, if e { "E" } else { "H" }, axis, node, e),
                    if layered { "layered" } else { "homogeneous" },
                    node[si],
                ))),
            }
        };
        for r in &inp.e_raw {
            check(true, r.axis, r.node)?;
            check(false, r.comp, r.cnode)?;
        }
        for r in &inp.h_raw {
            check(false, r.axis, r.node)?;
            check(true, r.comp, r.cnode)?;
        }
        drop(co);
        let setup = spectral::Setup {
            grid,
            dt,
            s,
            col,
            layered,
            ends: spec.layer_ends,
            lo,
            hi,
            meds,
            med_index: Box::new(index),
            scheme: spec.pole_scheme,
            k_hat,
            e0: inp.e0,
            match_frequency: inp.f,
            duration: spec.duration,
        };
        let inc = spectral::synthesise(&setup, &|t| amplitude(t))?;
        let delta = dt / spectral::UP as f64;
        let p = inc.p;
        let finish = |raw: &[Raw], e: bool, half: f64| -> Result<Vec<Correction>> {
            raw.iter()
                .map(|r| {
                    let id = inc
                        .id(s, e, r.comp, r.cnode[si])
                        .ok_or_else(|| Error::Invalid("plane wave: level out of range".into()))?;
                    let tau = dot(
                        p,
                        [
                            r.pos[0] - inp.r_ref[0],
                            r.pos[1] - inp.r_ref[1],
                            r.pos[2] - inp.r_ref[2],
                        ],
                    );
                    // Sample position at step 0 (negative: the wave has not
                    // arrived yet), biased by BIAS samples.
                    let c0 = half * spectral::UP as f64 + inc.pre as f64 - tau / delta;
                    let fl = c0.floor();
                    let (_, w) = stencil(1.0 + (c0 - fl));
                    let m = fl as isize - 1 + BIAS as isize;
                    if m < 0 {
                        return Err(Error::Invalid(
                            "plane wave: the total-field box is too large for the spectral \
                             incident field"
                                .into(),
                        ));
                    }
                    Ok(Correction {
                        axis: r.axis,
                        node: r.node,
                        coef: r.coef,
                        series: id,
                        m: m as usize,
                        w,
                    })
                })
                .collect()
        };
        // E corrections read the incident H at (n + ½)·dt, H corrections
        // the incident E at n·dt.
        let mut e_corr = finish(&inp.e_raw, false, 0.5)?;
        let mut h_corr = finish(&inp.h_raw, true, 0.0)?;
        // Edges whose poles the Dispersive operator solves implicitly with
        // the kernel result (slab hooks between the kernel and the serial
        // post hooks): their corrections must enter the kernel result,
        // i.e. be added to the start value divided by ca (da).
        let implicit = |e: bool, c: &Correction| {
            inp.poles
                .get(&(e, c.axis.index(), c.node))
                .is_some_and(|ps| {
                    ps.iter().any(|p| {
                        spec.pole_scheme == Scheme::Trapezoidal || matches!(p, Pole::Debye { .. })
                    })
                })
        };
        let (mut e_pre, mut h_pre) = (Vec::new(), Vec::new());
        {
            let co = engine.coefficients();
            let split = |list: &mut Vec<Correction>, pre: &mut Vec<Correction>, e: bool| {
                list.retain(|c| {
                    if !implicit(e, c) {
                        return true;
                    }
                    let i = grid.index(c.node);
                    let a = if e {
                        co.ca(c.axis)[i]
                    } else {
                        co.da(c.axis)[i]
                    } as f64;
                    if a.abs() < 1e-3 {
                        return true;
                    }
                    pre.push(Correction {
                        coef: c.coef / a,
                        ..*c
                    });
                    false
                });
            };
            split(&mut e_corr, &mut e_pre, true);
            split(&mut h_corr, &mut h_pre, false);
        }
        let v = spec.medium().velocity();
        let far = (0..8).fold(0.0f64, |m, c: usize| {
            let r = Axis::ALL.map(|ax| {
                let l = grid.lines(ax);
                let a = ax.index();
                l[if c >> a & 1 == 1 { hi[a] } else { lo[a] }]
            });
            m.max(dot(
                k_hat,
                [
                    r[0] - inp.r_ref[0],
                    r[1] - inp.r_ref[1],
                    r[2] - inp.r_ref[2],
                ],
            ))
        });
        let transit = (1.1 * far / v).max(inc.t_last - spec.duration);
        let ls = grid.lines(s);
        let nodes = ls[inc.k0..=inc.k1].to_vec();
        let duals = (inc.k0..inc.k1).map(|k| grid.dual_line(s, k)).collect();
        Ok(Self {
            spec: spec.clone(),
            spectral: Some(Box::new(Spectral {
                inc,
                s,
                steps: engine.timestep(),
                dt,
                nodes,
                duals,
                e_pre,
                h_pre,
            })),
            amplitude,
            k_hat,
            e0: inp.e0,
            h0: inp.h0,
            r_ref: inp.r_ref,
            dxi: 0.0,
            m0: 0,
            lead: 0.0,
            gain: 1.0,
            transit,
            main: 0,
            e: Vec::new(),
            h: Vec::new(),
            ca: Vec::new(),
            cb: Vec::new(),
            da: Vec::new(),
            db: Vec::new(),
            e_corr,
            h_corr,
        })
    }

    /// Whether the incident field is the spectral one of a [layered or
    /// dispersive background](self#layered-and-dispersive-backgrounds)
    /// (else the auxiliary 1-D grid).
    pub fn is_spectral(&self) -> bool {
        self.spectral.is_some()
    }

    /// The settings.
    pub fn spec(&self) -> &PlaneWaveSpec {
        &self.spec
    }

    /// Unit propagation vector `k̂`.
    pub fn direction(&self) -> [f64; 3] {
        self.k_hat
    }

    /// Spacing `Δξ` of the auxiliary 1-D grid \[m\] (0 for the spectral incident
    /// field of a layered or dispersive background, which has none).
    pub fn spacing(&self) -> f64 {
        self.dxi
    }

    /// The reference point `r_ref` \[m\]: the box corner the wave enters
    /// first, where the incident E is `E0·a(t)`.
    pub fn reference_point(&self) -> [f64; 3] {
        self.r_ref
    }

    /// Time \[s\] the incident wave takes to cross the total-field box,
    /// from the reference corner to the corner it leaves last, at the
    /// background speed plus 10 % for the numerical dispersion. A pulse
    /// that is over at the reference corner at time `t` has left the box
    /// by `t + transit_time()`; from then on the fields everywhere are
    /// scattered fields only (see the [module docs](self#energy)).
    pub fn transit_time(&self) -> f64 {
        self.transit
    }

    /// Number of E and H edge corrections applied per step.
    pub fn corrections(&self) -> (usize, usize) {
        (self.e_corr.len(), self.h_corr.len())
    }

    /// The spectral incident E (`e`) or H at `r`, linear between the
    /// stored levels along the stratification axis.
    fn spectral_at(&self, sp: &Spectral, r: [f64; 3], e: bool) -> Option<[f64; 3]> {
        let inc = &sp.inc;
        let si = sp.s.index();
        let t = if e {
            sp.steps as f64 * sp.dt
        } else {
            (sp.steps as f64 - 0.5) * sp.dt
        };
        let tau = dot(
            inc.p,
            [
                r[0] - self.r_ref[0],
                r[1] - self.r_ref[1],
                r[2] - self.r_ref[2],
            ],
        );
        let mut out = [0.0; 3];
        for a in Axis::ALL {
            let node = e != (a == sp.s);
            let pos = if node { &sp.nodes } else { &sp.duals };
            let x = r[si];
            let j = pos.windows(2).position(|w| w[0] <= x && x <= w[1])?;
            let f = (x - pos[j]) / (pos[j + 1] - pos[j]);
            let val = |k: usize| {
                inc.id(sp.s, e, a, inc.k0 + k)
                    .map_or(0.0, |id| inc.value(id, t - tau, sp.dt))
            };
            out[a.index()] = (1.0 - f) * val(j) + f * val(j + 1);
        }
        Some(out)
    }

    /// Fractional auxiliary position of `r` for nodes offset by `half`, if
    /// the stencil fits.
    fn aux_at(&self, r: [f64; 3], half: f64) -> Option<(usize, [f64; 4])> {
        let d = [
            r[0] - self.r_ref[0],
            r[1] - self.r_ref[1],
            r[2] - self.r_ref[2],
        ];
        let s = dot(self.k_hat, d) / self.dxi + self.m0 as f64 - half;
        (s >= 1.0 && s + 3.0 < self.main as f64).then(|| stencil(s))
    }

    /// Incident E \[V/m\] at `r` \[m\] at the time of the engine's E
    /// voltages (`n·dt` after `n` steps), or `None` if `r` lies before
    /// the auxiliary grid's start or beyond its end.
    pub fn incident_e(&self, r: [f64; 3]) -> Option<[f64; 3]> {
        if let Some(sp) = &self.spectral {
            return self.spectral_at(sp, r, true);
        }
        let (m, w) = self.aux_at(r, 0.0)?;
        let f = interp(&self.e, m, &w);
        Some(self.e0.map(|x| x * f))
    }

    /// Incident H \[A/m\] at `r` \[m\] at the time of the engine's H
    /// currents (`(n − ½)·dt` after `n` steps), or `None` as for
    /// [`incident_e`](Self::incident_e).
    pub fn incident_h(&self, r: [f64; 3]) -> Option<[f64; 3]> {
        if let Some(sp) = &self.spectral {
            return self.spectral_at(sp, r, false);
        }
        let (m, w) = self.aux_at(r, 0.5)?;
        let f = interp(&self.h, m, &w);
        Some(self.h0.map(|x| x * f))
    }
}

/// Bias of the spectral corrections' sample offsets (samples before the
/// series start, where the incident field is zero).
const BIAS: usize = 1 << 24;

/// `Σ_q w_q·v[base − BIAS + q]` of a correction (zero outside the series).
fn series_value(v: &[f32], base: usize, w: &[f64; 4]) -> f64 {
    let Some(base) = base.checked_sub(BIAS) else {
        return 0.0;
    };
    match v.get(base..base + 4) {
        Some(x) => {
            w[0] * x[0] as f64 + w[1] * x[1] as f64 + w[2] * x[2] as f64 + w[3] * x[3] as f64
        }
        None => 0.0,
    }
}

impl Operator for PlaneWave {
    fn hooks(&self) -> Hooks {
        match &self.spectral {
            Some(sp) if !sp.e_pre.is_empty() || !sp.h_pre.is_empty() => Hooks::ALL,
            _ => Hooks::POST_H | Hooks::POST_E,
        }
    }

    fn pre_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(sp) = &self.spectral {
            let n = ctx.step as usize * spectral::UP;
            for c in &sp.h_pre {
                let v = series_value(&sp.inc.series[c.series], n + c.m, &c.w);
                *fields.i.get_mut(c.axis, c.node) += (c.coef * v) as f32;
            }
        }
    }

    fn pre_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(sp) = &self.spectral {
            let n = ctx.step as usize * spectral::UP;
            for c in &sp.e_pre {
                let v = series_value(&sp.inc.series[c.series], n + c.m, &c.w);
                *fields.u.get_mut(c.axis, c.node) += (c.coef * v) as f32;
            }
        }
    }

    fn post_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(sp) = &self.spectral {
            // The incident E at n·dt − τ.
            let n = ctx.step as usize * spectral::UP;
            for c in &self.h_corr {
                let v = series_value(&sp.inc.series[c.series], n + c.m, &c.w);
                *fields.i.get_mut(c.axis, c.node) += (c.coef * v) as f32;
            }
            return;
        }
        // The H kernel read u^n; the auxiliary E is at n·dt.
        for c in &self.h_corr {
            *fields.i.get_mut(c.axis, c.node) += (c.coef * interp(&self.e, c.m, &c.w)) as f32;
        }
        let e = &self.e;
        for (m, h) in self.h.iter_mut().enumerate() {
            *h = self.da[m] * *h - self.db[m] * (e[m + 1] - e[m]);
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if let Some(sp) = &mut self.spectral {
            // The incident H at (n + ½)·dt − τ.
            let n = ctx.step as usize * spectral::UP;
            for c in &self.e_corr {
                let v = series_value(&sp.inc.series[c.series], n + c.m, &c.w);
                *fields.u.get_mut(c.axis, c.node) += (c.coef * v) as f32;
            }
            sp.steps = ctx.step + 1;
            return;
        }
        // The E kernel read i^{n+½}; so does the auxiliary H now.
        for c in &self.e_corr {
            *fields.u.get_mut(c.axis, c.node) += (c.coef * interp(&self.h, c.m, &c.w)) as f32;
        }
        let last = self.e.len() - 1;
        for m in 1..last {
            self.e[m] = self.ca[m] * self.e[m] - self.cb[m] * (self.h[m] - self.h[m - 1]);
        }
        self.e[0] = self.gain * (self.amplitude)(ctx.e_time() + self.lead);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::C0;

    /// The lossy auxiliary-grid propagation constant satisfies the
    /// semi-implicit Yee relation `4·sin²(K/2) = Ω²με·Δ² − jΩμκ·cos(ωdt/2)·Δ²`
    /// (`K = −jγΔ`, `Ω = (2/dt)·sin(ωdt/2)`), and decays downstream.
    #[test]
    fn aux_gamma_solves_the_dispersion_relation() {
        let dt = 0.99e-3 / (C0 * 3f64.sqrt());
        let (er, kappa, f) = (2.0f64, 0.13, 8e9 / 2f64.sqrt());
        let se = kappa * dt / (2.0 * EPS0 * er);
        let r = C0 / er.sqrt() * dt / 1e-3;
        let g = aux_gamma(2.0 * core::f64::consts::PI * f * dt, se, 0.0, r);
        let om = 2.0 / dt * (core::f64::consts::PI * f * dt).sin();
        let c = (core::f64::consts::PI * f * dt).cos();
        let x = Complex64::new(
            om * om * MU0 * EPS0 * er * 1e-6,
            -om * MU0 * kappa * c * 1e-6,
        );
        let sh = ((g * 0.5).exp() - (g * -0.5).exp()) * 0.5;
        // sinh²(γΔ/2) = −sin²(K/2).
        assert!((sh * sh + x * 0.25).norm() < 1e-12 * x.norm(), "{g:?}");
        assert!(g.re > 0.0 && g.im > 0.0, "{g:?}");
    }

    #[test]
    fn stencil_reproduces_cubics() {
        let f = |x: f64| 0.3 - x + 0.2 * x * x - 0.05 * x * x * x;
        let v: Vec<f64> = (0..8).map(|m| f(m as f64)).collect();
        for s in [1.0, 1.25, 2.5, 3.999, 4.0] {
            let (m, w) = stencil(s);
            assert!((interp(&v, m, &w) - f(s)).abs() < 1e-12, "{s}");
        }
    }

    #[test]
    fn axis_aligned_match_is_the_mesh_spacing() {
        let dt = 0.99 * 1e-3 / (C0 * 3f64.sqrt());
        let (d, _) = match_spacing(dt, C0, 10e9, [1.0, 0.0, 0.0], [1e-3; 3]).unwrap();
        assert_eq!(d, 1e-3);
        // Oblique: matched spacing lies between v·dt and the cell size.
        let s = 0.5f64.sqrt();
        let (d, _) = match_spacing(dt, C0, 10e9, [s, s, 0.0], [1e-3; 3]).unwrap();
        assert!(d > C0 * dt && d < 1e-3, "{d}");
    }
}
