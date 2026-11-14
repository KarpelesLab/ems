//! Surface impedance of lossy conductors: a rational fit of the skin-effect
//! impedance, integrated in the time domain on the faces of solid metal and
//! on the two faces of conducting sheets, with a roughness factor.
//!
//! # Physical model
//!
//! At a good conductor the tangential electric field is the surface current
//! times the surface impedance, `E_tan = Z_s·(n × H)` (Leontovich boundary
//! condition; Ramo, Whinnery and Van Duzer, *Fields and Waves in
//! Communication Electronics*, 3rd ed., §3.16–3.17). The conductor's
//! interior is not meshed:
//!
//! * **Solid metal** (deeper than a few skin depths): each face is the half
//!   space `Z_s = √(jωμ0/(σ + jωε0)) ≈ (1 + j)·R_s`, `R_s = 1/(σδ)`.
//! * **A sheet** of thickness `t` is the two-port of [`super::sheet`],
//!   `E± = Z11·H± + Z12·H∓`. It is symmetric, so the sum and the difference
//!   of the two faces are two independent one-ports,
//!
//!   ```text
//!   E+ + E− = Z_e·(H+ + H−),   Z_e = Z11 + Z12 = Z_c·coth(γt/2)
//!   E+ − E− = Z_o·(H+ − H−),   Z_o = Z11 − Z12 = Z_c·tanh(γt/2)
//!   ```
//!
//!   (the bisection of a symmetric two-port into its even and odd halves).
//!   For `t ≪ δ` this is the resistive sheet (`Z_e → 2/(σt)`, `Z_o → 0`:
//!   both faces at one voltage), for `t ≫ δ` two independent half spaces
//!   (`Z_e = Z_o = Z_c`), and in between it keeps the shielding
//!   (`Z12 = (Z_e − Z_o)/2`).
//!
//! **Roughness** multiplies the resistance (the dissipated power) by a
//! factor `K(δ)` of the skin depth ([`Roughness`]): Hammerstad and Jensen's
//! `1 + (2/π)·atan(1.4·(Δ/δ)²)` (E. Hammerstad, O. Jensen, "Accurate models
//! for microstrip computer-aided design", IEEE MTT-S Digest, 1980) or
//! Huray's "snowball" factor `1 + (3/2)·SR/(1 + δ/a + δ²/(2a²))`
//! (P. G. Huray et al., "Impact of copper surface texture on loss: a model
//! that works", DesignCon 2010; Huray, *The Foundations of Signal
//! Integrity*, 2009). The reactance stays the smooth conductor's.
//!
//! # Rational fit
//!
//! `Z(ω)` goes like `√(jω)`, which has no finite recursion in time. Over
//! the band of a run it is replaced by a series of a resistor and parallel
//! R–L sections ([`ImpedanceFit`]),
//!
//! ```text
//! Z(s) ≈ r0 + Σ_p r_p·s/(s + a_p),      r0, r_p ≥ 0
//! ```
//!
//! the first Foster form of an R–L impedance, which every one of the
//! impedances above is (they are diffusion lines). The corner frequencies
//! `a_p` are fixed, three per two decades from a decade below to a decade
//! above the band (plus one far above it, which acts as an inductance; for
//! a sheet also the first poles of its own partial fractions), and the
//! `r` are a non-negative least-squares fit of the resistance and the
//! reactance in relative terms (Lawson and Hanson's active-set NNLS,
//! *Solving Least Squares Problems*, 1974, ch. 23). Non-negative terms are
//! passive whatever the fit error, so the time stepping stays stable. A
//! rough surface's `R_s·K + j·R_s` is not a causal pair; there the
//! reactance gets a small weight and takes the value that goes with the
//! resistance. This is the approach of the surface-impedance boundary
//! conditions of J. G. Maloney and G. S. Smith (IEEE Trans. AP-40, 1992),
//! J. H. Beggs, R. J. Luebbers, K. S. Yee and K. S. Kunz (IEEE Trans.
//! AP-40, 1992) and K. S. Oh and J. E. Schutt-Ainé (IEEE Trans. AP-43,
//! 1995), with the circuit form of the fit.
//!
//! The fit of a half space over two decades is within 0.5 % in resistance
//! and reactance ([`half_space_impedance`]); it costs one current and
//! seven section voltages per face, against the 48-node ladder of
//! [`super::sheet`] for 35 µm of copper.
//!
//! # Discretisation
//!
//! *Solid metal.* The edges on the surface of the metal are PEC in the
//! engine. Each gets a face voltage `u` instead of zero. The faces of the
//! mesh whose four edges are all PEC lie in the metal or on its surface
//! and stay without field (their `db` is set to 0); the others around the
//! edge form its dual loop outside the metal, whose circulation `S` is the
//! current flowing along the edge on the conductor surface inside the dual
//! face. That surface has the width `w`: for every such face, half a cell
//! along the face's normal on each side where the neighbouring parallel
//! edge is PEC too (the whole dual length on a flat face, half of each of
//! the two faces meeting at a convex edge). With `g = w/l`,
//!
//! ```text
//! C·du/dt + G·u + h = S,      u = (1/g)·Z_s ∗ h
//! ```
//!
//! with `C`, `G` of the part of the dual face outside the metal, and `C`
//! at least that of the quarters of the dual face next to a face with
//! field, in vacuum (the explicit coupling to those faces needs it: an
//! edge without capacitance is a resistor in their loops), and beside a
//! cell that conformal PEC cuts at least `4·dt·Σ n·dt/L` over the faces
//! around it, `n` the edges of the face that carry a surface (conformal
//! PEC took the edge for a perfect conductor when it decided which small
//! faces the time step allows; at the highest frequencies of the mesh
//! the trapezoidal admittance vanishes and the edge is its capacitance
//! alone). Edges
//! inside the metal (no face left) and re-entrant edges keep `u = 0`.
//! Edges of zero-thickness metal (faces left on two opposite sides) are
//! not solid metal and stay PEC ([`SurfaceImpedance::thin`]); use a sheet.
//! (Two opposite faces that the metal's outline both crosses are the two
//! rims of a conductor one row of edges wide, which is solid.)
//!
//! *Oblique surfaces.* With conformal PEC a surface of the metal crosses
//! the cells. The edges wholly in the metal beside it carry it (the
//! covered parts of the edges it cuts stay perfect conductors), and the
//! count of half cells above is then a staircase's: every edge beside a
//! wall at 45° would carry one cell of it where `√2` cells cross its dual
//! face, and the wall would lose 41 % too much. The width of the surface
//! that an edge along `d` carries through a face with field comes from
//! the surface itself ([`CutSurface`], which the compile step takes from
//! the points where the surface crosses the mesh edges). Let the face's
//! dual edge run along `c`, of length `ℓ`:
//!
//! ```text
//! w = ℓ·cos γ / sin φ
//! ```
//!
//! with `γ` the angle in the edge's dual plane between the surface's
//! trace and the dual edge, and `φ` the angle between the edge and the
//! surface's normal. The first factor is Ampère's law on the dual loop:
//! next to a good conductor `H` is the surface current turned by a right
//! angle, so the dual edge collects the current of the piece of surface
//! it projects on, and the dual edges of the edges along a wall project
//! on all of it, once. The second is the Leontovich condition: the edge
//! has the component `sin φ` of the tangential field that drives the
//! current through its dual face, so `u = E_t·l·sin φ` for `h = K·ℓ·cos
//! γ`. A surface turned in the dual plane (`γ ≠ 0`) goes on to both sides
//! of the edge whatever lies on the mesh lines there; one along the dual
//! edge is counted by half cells as above. Where another surface of the
//! metal stands on the edge (a floor edge running under a wall), the edge
//! lies in its own surface over the part `a` of its length that the face
//! leaves open, and `g = w/(a·l)`.
//!
//! *Rims.* A flat surface in a mesh plane whose outline crosses the
//! cells: the edges its outline cuts keep a perfectly conducting covered
//! part and a displacement current only, so the dual loop of the last
//! edge wholly in the metal encloses the current of all the metal from
//! its own inner half cell out to the rim, a patch of width `P` (its
//! reach to the rim, [`CutSurface::reach`], measured by the compile step
//! along the mesh lines through its nodes, plus the half cell on the
//! other side). The current of a patch that ends on a rim is not spread
//! evenly, and how the patch is lumped decides its loss: with rims on
//! mesh lines the scheme has an edge on the rim that carries half a cell
//! `λ` and far more current than a row inside (measured on a strip of
//! 35 µm copper, 1 mm wide over 0.5 mm of `ε_r` = 4: the two rim rows
//! carry 0.37, 0.28 and 0.23 of the current each at two, four and eight
//! cells across, a row inside 0.27, 0.15 and 0.08). An edge beside the
//! rim is given the loss those rows would have with its patch: the half
//! cell at the rim with the share `φ = (λ/P)^0.28` of the patch's
//! current, the rest of the patch with the rest,
//!
//! ```text
//! 1/w = φ²/λ + (1 − φ)²/(P − λ)
//! ```
//!
//! (`λ` taken along the mesh line across the edge, `1/cos` of the
//! outline's angle to the edge times the half cell); an edge with a rim on
//! both sides, a conductor one row of edges wide, `φ = ½·(2λ/P)^0.4` in
//! each of two half cells. The exponents are fitted, not derived: 0.28
//! makes the loss of a strip independent of where its rims lie in the
//! cells (below), and the staircase's own rim share falls with `λ^0.29`
//! between four and eight cells across. Counting the metal by area
//! (`w = P`) halves the loss of such a strip; counting half cells where
//! the neighbouring edge is covered, as before, read 26 % more loss with
//! the rims between mesh lines or oblique to them than with the rims on
//! mesh lines.
//!
//! *Sheets.* As in [`super::sheet`] the edge is split into its two face
//! voltages `u+` (the engine's value) and `u−`, each with its share of the
//! edge's capacitance and of the dual loop, and the H updates on the `−`
//! side are corrected for `u− − u+`. The two faces are joined by `Z_e` and
//! `Z_o` instead of the ladder. The width `w` of the edge's strip of the
//! sheet is, as for solid metal, half a cell on each side where the sheet
//! goes on (a parallel sheet edge or metal): an edge on the border of a
//! sheet carries half a cell of conductor, not the whole dual length the
//! ladder gives it. A voltage that another operator (a source) adds to a
//! sheet edge is added to both faces. An edge the outline of its sheet
//! cuts ([`SurfaceImpedance::cut_sheet`]) is its open part in series with
//! the covered part, and the edges around a cut face are loaded and
//! stored as in
//! [the ladder](super::sheet#outlines-between-mesh-lines): the two
//! models take the same cuts.
//!
//! *Time stepping.* Face voltages, currents and sections are integrated
//! with the trapezoidal rule centred at `(n + ½)·dt`, driven by the H
//! currents at that time. A section advances as
//! `v̄ = (2·v + 2·r·(h̄ − h))/(2 + a·dt)` in the means over the step, so a
//! one-port is `x̄ = Z∞·h̄ + e` with `Z∞ = r0 + Σ 2r/(2 + a·dt)` and `e`
//! from the state: the face equations are solved in closed form (one
//! division per solid edge, a 2 × 2 system per sheet edge). The
//! trapezoidal rule keeps each section passive for any `dt`, and the
//! explicit coupling to H has the Courant bound of an ordinary edge.
//!
//! # Absorbing layers
//!
//! Inside a CPML ([`CpmlBoundary`]) the circulations of these edges are
//! stretched like the kernel's: the operator keeps its own `ψ` for every
//! derivative term along a layer normal, on the E side
//! (`S_p/κ + ψ`, `ψ ← b·ψ + a·S_p`) and for the H corrections of sheets,
//! with the layer's coefficients at the same positions. The CPML is linear
//! in the derivative terms, so this is the stretched update of the split
//! faces, and a conductor running into the layer is absorbed with the
//! wave it guides. Register the CPML before [`SurfaceImpedance::add_to`].
//!
//! # Threads
//!
//! The edges are run by the engine's threads, each on its x-slab, right
//! after the kernel of the half-step ([`SlabHooks`]): every edge by the
//! same expressions from the same values whichever slab it is in, with
//! its state in its slab, so results are bit-identical for any thread
//! count, and steps can run fused. Two things cross a slab boundary. An
//! edge's circulation reads the currents of the plane before its own,
//! which the E half-step may read. And the H correction of a dual edge in
//! the last plane of a slab needs the face voltages of a sheet edge in
//! the first plane of the next one: that slab leaves them, after its E
//! update, where the other reads them in the next H half-step, together
//! with the array value, from which both compute what a source has added
//! since (the same expression on the same values).
//!
//! Slab hooks run before every serial hook. Where an operator registered
//! earlier has serial hooks after a half-step (the merged faces of
//! conformal PEC, a ladder sheet), the edges stay in the serial hooks, in
//! registration order as before ([`SurfaceImpedance::runs_in_slabs`]).
//! Register the operator after the others that work in the slabs (the
//! CPML, dispersive media, Mur faces), as the compile step does: one
//! registered later now acts on these edges after their update, not
//! before it.
//!
//! # Validity
//!
//! * Cartesian grids.
//! * The displacement current inside the conductor is left out
//!   (`σ ≫ ωε0`; with it the impedances are no R–L functions), so
//!   `γ = √(jωμ0σ)`, `Z_c = √(jωμ0/σ)` here.
//! * Solid metal must be thicker than a few skin depths at the lowest
//!   frequency of interest (the faces are independent half spaces); the
//!   resistance does not go to the DC value of a thin conductor.
//! * The fit holds over its band; outside it stays passive, with the
//!   resistance levelling off.
//! * With conformal PEC the covered part of an edge that solid metal cuts
//!   stays perfectly conducting; the whole edges next to it carry its
//!   share of the surface ([`SurfaceImpedance::cut_edge`]), with the
//!   width the surface has across them
//!   ([`SurfaceImpedance::cut_surface`]). Sheets take
//!   the cuts of their outline as the ladder does
//!   ([`SurfaceImpedance::cut_sheet`], [`SurfaceImpedance::cut_face`]).
//!
//! # Energy
//!
//! [`Operator::energy`] reports the capacitive energy of the face voltages
//! (less the part the engine counts on sheet edges) and the magnetic
//! energy of the sections, `½·(r/a)·(h − v/r)²/g`, plus for sheets the
//! look-ahead term of [`super::sheet`]; outside absorbing layers the
//! engine total only decreases.

use core::ops::Range;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use super::Engine;
use super::boundary::cpml::CpmlBoundary;
use super::fields::Fields;
use super::kernel::flush;
use super::operator::{Hooks, Operator, OperatorHandle, StepContext};
use super::sheet::{CUT_FACE_LOADING, SheetEdge, skin_depth};
use super::slab::{Fusion, SlabContext, SlabHooks, SlabMut, SlabPartition, SlabRef};
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};
use crate::post::Complex64;
use crate::{Error, Result};

/// Roughness of a conductor surface: the factor `K ≥ 1` on its surface
/// resistance as a function of the skin depth (module docs).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum Roughness {
    /// A smooth surface, `K = 1`.
    #[default]
    Smooth,
    /// Hammerstad and Jensen: `K = 1 + (2/π)·atan(1.4·(rms/δ)²)`,
    /// saturating at 2.
    HammerstadJensen {
        /// RMS roughness \[m\].
        rms: f64,
    },
    /// Huray's snowball model:
    /// `K = 1 + (3/2)·ratio/(1 + δ/radius + δ²/(2·radius²))`.
    Huray {
        /// Radius of the spheres \[m\].
        radius: f64,
        /// Surface of the spheres per flat area, `N·4πa²/A` (the
        /// Hall–Huray surface ratio).
        ratio: f64,
    },
}

impl Roughness {
    /// The factor on the surface resistance at skin depth `delta` \[m\].
    ///
    /// ```
    /// use ems::engine::surface::Roughness;
    /// let r = Roughness::HammerstadJensen { rms: 1e-6 };
    /// assert!((r.factor(1e-3) - 1.0).abs() < 1e-5);
    /// assert!((r.factor(1e-9) - 2.0).abs() < 1e-5);
    /// ```
    pub fn factor(&self, delta: f64) -> f64 {
        match *self {
            Self::Smooth => 1.0,
            Self::HammerstadJensen { rms } => {
                1.0 + core::f64::consts::FRAC_2_PI * (1.4 * (rms / delta).powi(2)).atan()
            }
            Self::Huray { radius, ratio } => {
                1.0 + 1.5 * ratio / (1.0 + delta / radius + delta * delta / (2.0 * radius * radius))
            }
        }
    }

    fn validate(&self) -> Result<()> {
        let ok = |x: f64| x.is_finite() && x >= 0.0;
        let good = match *self {
            Self::Smooth => true,
            Self::HammerstadJensen { rms } => ok(rms),
            Self::Huray { radius, ratio } => ok(radius) && radius > 0.0 && ok(ratio),
        };
        if good {
            Ok(())
        } else {
            Err(Error::Invalid(format!(
                "roughness {self:?}: parameters must be finite and not negative"
            )))
        }
    }
}

/// A lossy conductor: conductivity and surface roughness.
///
/// ```
/// use ems::engine::surface::{Conductor, Roughness};
/// let cu = Conductor::new(5.8e7).with_roughness(Roughness::HammerstadJensen { rms: 1e-6 });
/// assert_eq!(cu.conductivity, 5.8e7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Conductor {
    /// Conductivity \[S/m\].
    pub conductivity: f64,
    /// Surface roughness (default smooth).
    pub roughness: Roughness,
}

impl Conductor {
    /// A smooth conductor of conductivity `sigma` \[S/m\].
    pub fn new(sigma: f64) -> Self {
        Self {
            conductivity: sigma,
            roughness: Roughness::Smooth,
        }
    }

    /// Returns `self` with the surface roughness `r`.
    #[must_use]
    pub fn with_roughness(mut self, r: Roughness) -> Self {
        self.roughness = r;
        self
    }

    fn validate(&self) -> Result<()> {
        if !(self.conductivity.is_finite() && self.conductivity > 0.0) {
            return Err(Error::Invalid(format!(
                "conductor: conductivity {} S/m must be finite and > 0",
                self.conductivity
            )));
        }
        self.roughness.validate()
    }
}

fn csqrt(z: Complex64) -> Complex64 {
    let (r, a) = z.to_polar();
    Complex64::from_polar(r.sqrt(), 0.5 * a)
}

/// `K·Re z + j·Im z` with the roughness factor of `c` at `f`.
fn roughen(c: &Conductor, f: f64, z: Complex64) -> Complex64 {
    let k = c.roughness.factor(skin_depth(c.conductivity, f));
    Complex64::new(z.re * k, z.im)
}

/// Surface impedance \[Ω per square\] of a half space of `c` at `f`
/// \[Hz\]: `√(jωμ0/σ)`, its resistance times the roughness factor.
pub fn half_space_impedance(c: &Conductor, f: f64) -> Complex64 {
    let w = core::f64::consts::TAU * f;
    let z = csqrt(Complex64::new(0.0, w * MU0 / c.conductivity));
    roughen(c, f, z)
}

/// Even and odd impedances `(Z_e, Z_o)` \[Ω per square\] of a sheet of `c`
/// with thickness `t` \[m\] at `f` \[Hz\] (module docs):
/// `Z_c·coth(γt/2)` and `Z_c·tanh(γt/2)`, their resistances times the
/// roughness factor.
pub fn sheet_mode_impedances(c: &Conductor, t: f64, f: f64) -> (Complex64, Complex64) {
    let w = core::f64::consts::TAU * f;
    let jwm = Complex64::new(0.0, w * MU0);
    // x² = (γt/2)², imaginary.
    let x2 = jwm * (c.conductivity * t * t / 4.0);
    let (ze, zo) = if x2.norm() < 1e-3 {
        // coth x = 1/x + x/3 − x³/45, tanh x = x − x³/3 + 2x⁵/15, with
        // Z_c/x = 2/(σt) and Z_c·x = jωμ0·t/2 (no cancellation).
        (
            (Complex64::ONE + x2 / 3.0 - x2 * x2 / 45.0) * (2.0 / (c.conductivity * t)),
            jwm * (0.5 * t) * (Complex64::ONE - x2 / 3.0 + x2 * x2 * (2.0 / 15.0)),
        )
    } else {
        let (gamma, zc) = (csqrt(jwm * c.conductivity), csqrt(jwm / c.conductivity));
        // tanh(x) for Re x ≥ 0, without overflow.
        let e2 = (gamma * (-t)).exp();
        let th = (Complex64::ONE - e2) / (Complex64::ONE + e2);
        (zc / th, zc * th)
    };
    (roughen(c, f, ze), roughen(c, f, zo))
}

/// Least squares `min |A·x − b|` on the columns `cols` (Householder QR;
/// a column dependent on the previous ones gets 0).
fn lstsq(cols: &[&[f64]], b: &[f64]) -> Vec<f64> {
    let k = cols.len();
    let mut q: Vec<Vec<f64>> = cols.iter().map(|c| c.to_vec()).collect();
    let mut y = b.to_vec();
    let mut diag = vec![0.0; k];
    for j in 0..k {
        let norm = q[j][j..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if q[j][j] > 0.0 { -norm } else { norm };
        let mut v = q[j][j..].to_vec();
        v[0] -= alpha;
        let vn: f64 = v.iter().map(|x| x * x).sum();
        diag[j] = alpha;
        if vn == 0.0 {
            continue;
        }
        let reflect = |col: &mut [f64]| {
            let d = 2.0 * v.iter().zip(&col[j..]).map(|(a, b)| a * b).sum::<f64>() / vn;
            for (c, vi) in col[j..].iter_mut().zip(&v) {
                *c -= d * vi;
            }
        };
        for col in q.iter_mut().skip(j + 1) {
            reflect(col);
        }
        reflect(&mut y);
    }
    let scale = diag.iter().fold(0.0f64, |a, d| a.max(d.abs()));
    let mut x = vec![0.0; k];
    for j in (0..k).rev() {
        if diag[j].abs() <= 1e-12 * scale {
            continue;
        }
        let mut s = y[j];
        for i in j + 1..k {
            s -= q[i][j] * x[i];
        }
        x[j] = s / diag[j];
    }
    x
}

/// Non-negative least squares `min |A·x − b|`, `x ≥ 0`, for the columns
/// `a` (Lawson and Hanson's active-set method; columns scaled to unit
/// norm internally).
fn nnls(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let norms: Vec<f64> = a
        .iter()
        .map(|c| c.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect();
    let a: Vec<Vec<f64>> = a
        .iter()
        .zip(&norms)
        .map(|(c, &s)| {
            c.iter()
                .map(|v| if s > 0.0 { v / s } else { 0.0 })
                .collect()
        })
        .collect();
    let mut x = vec![0.0; n];
    let mut passive = vec![false; n];
    let tol = 1e-12 * (b.len() as f64).sqrt();
    for _ in 0..3 * n + 10 {
        let mut r = b.to_vec();
        for (c, &xi) in a.iter().zip(&x) {
            if xi != 0.0 {
                for (ri, ci) in r.iter_mut().zip(c) {
                    *ri -= ci * xi;
                }
            }
        }
        let w: Vec<f64> = a
            .iter()
            .map(|c| c.iter().zip(&r).map(|(a, b)| a * b).sum())
            .collect();
        let Some(t) = (0..n)
            .filter(|&j| !passive[j] && w[j] > tol)
            .max_by(|&i, &j| w[i].total_cmp(&w[j]))
        else {
            break;
        };
        passive[t] = true;
        loop {
            let set: Vec<usize> = (0..n).filter(|&j| passive[j]).collect();
            let cols: Vec<&[f64]> = set.iter().map(|&j| &a[j][..]).collect();
            let zp = lstsq(&cols, b);
            let mut z = vec![0.0; n];
            for (&c, &v) in set.iter().zip(&zp) {
                z[c] = v;
            }
            if set.iter().all(|&c| z[c] > 0.0) {
                x = z;
                break;
            }
            let alpha = set
                .iter()
                .filter(|&&c| z[c] <= 0.0)
                .map(|&c| x[c] / (x[c] - z[c]))
                .fold(f64::INFINITY, f64::min);
            for j in 0..n {
                x[j] += alpha * (z[j] - x[j]);
                if passive[j] && x[j] <= 1e-15 {
                    passive[j] = false;
                    x[j] = 0.0;
                }
            }
            if !passive.iter().any(|&p| p) {
                break;
            }
        }
    }
    x.iter()
        .zip(&norms)
        .map(|(v, &s)| if s > 0.0 { v / s } else { 0.0 })
        .collect()
}

/// Capacitance of a solid edge beside a cut cell, in units of
/// `dt·Σ n·dt/L` of the faces around it, `n` the surface edges of the
/// face (see `install`). With `(Σ u)² ≤ (s + Σ_n u)² ≤ s²/(1 − θ) +
/// n·Σ_n u²/θ` for the circulation of a face, `s` over its ordinary
/// edges, the cells stay stable with these edges in them if they were
/// stable at `(1 − θ)` of the limit without and `C ≥ dt²·Σ n/L / (4θ)`:
/// 4 is `θ = 1/16`, a stability factor of 0.97 before. Measured: a copper
/// strip at 17° on cells of 0.125 × 0.125 × 0.1 mm grew at 0.5 and not at
/// 1; a box cavity with walls of 2e4 S/m at 45° on 1 mm cells and at 30°
/// on 0.5 mm cells (`tests/validation_lossy_walls.rs`) grew at 2 and not
/// at 4, its Q 0.06 % lower per unit beyond.
const FLOOR_MARGIN: f64 = 4.0;

/// Frequencies sampled by [`ImpedanceFit::fit`].
const FIT_SAMPLES: usize = 64;
/// Corner frequencies per decade (1.5: a half space within 0.5 %; 2:
/// 0.07 % with half as many sections again; 1: 4.5 %).
const CORNERS_PER_DECADE: f64 = 1.5;
/// Most poles of a sheet's own partial fractions offered to its fit.
const SHEET_POLES: usize = 6;
/// Weight of the reactance against the resistance for a rough surface.
const ROUGH_REACTANCE_WEIGHT: f64 = 0.02;

/// A passive rational impedance `Z(s) = r0 + Σ r·s/(s + a)` \[Ω\]: a
/// resistor in series with parallel R–L sections (module docs).
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct ImpedanceFit {
    /// Series resistance `r0` \[Ω\].
    pub r0: f64,
    /// The sections `(r, a)`: resistance \[Ω\] and corner `a = R/L`
    /// \[1/s\], all positive.
    pub sections: Vec<(f64, f64)>,
}

impl ImpedanceFit {
    /// The impedance at `f` \[Hz\].
    pub fn eval(&self, f: f64) -> Complex64 {
        let w = core::f64::consts::TAU * f;
        let mut z = Complex64::new(self.r0, 0.0);
        for &(r, a) in &self.sections {
            // s/(s + a) = (ω² + jωa)/(ω² + a²).
            let d = w * w + a * a;
            z += Complex64::new(r * w * w / d, r * w * a / d);
        }
        z
    }

    /// Fits `target(f)` over `f_lo..f_hi` \[Hz\] (module docs): the
    /// resistance in relative terms at 64 log-spaced frequencies, the
    /// reactance likewise with the weight `reactance_weight`.
    ///
    /// # Errors
    /// [`Error::Invalid`] unless `0 < f_lo < f_hi` are finite and the
    /// target is finite with a real part that is not negative over the
    /// band.
    pub fn fit(
        target: impl Fn(f64) -> Complex64,
        f_lo: f64,
        f_hi: f64,
        reactance_weight: f64,
    ) -> Result<Self> {
        Self::fit_with(target, f_lo, f_hi, reactance_weight, &[])
    }

    /// [`fit`](Self::fit) with the further corners `extra` \[1/s\].
    fn fit_with(
        target: impl Fn(f64) -> Complex64,
        f_lo: f64,
        f_hi: f64,
        reactance_weight: f64,
        extra: &[f64],
    ) -> Result<Self> {
        if !(f_lo.is_finite() && f_hi.is_finite() && f_lo > 0.0 && f_hi > f_lo) {
            return Err(Error::Invalid(format!(
                "impedance fit: band {f_lo}..{f_hi} Hz must be finite with 0 < f_lo < f_hi"
            )));
        }
        let log_space = |lo: f64, hi: f64, count: usize| -> Vec<f64> {
            let (a, b) = (lo.ln(), hi.ln());
            (0..count)
                .map(|i| (a + (b - a) * i as f64 / (count - 1) as f64).exp())
                .collect()
        };
        let freqs = log_space(f_lo, f_hi, FIT_SAMPLES);
        let (c_lo, c_hi) = (f_lo / 10.0, f_hi * 10.0);
        let count = ((c_hi / c_lo).log10() * CORNERS_PER_DECADE).ceil() as usize + 1;
        let mut corners = log_space(c_lo, c_hi, count.max(2));
        // Far above the band: an inductance.
        corners.push(f_hi * 1e3);
        let mut corners: Vec<f64> = corners.iter().map(|f| core::f64::consts::TAU * f).collect();
        corners.extend_from_slice(extra);
        corners.sort_by(f64::total_cmp);
        let rows = 2 * freqs.len();
        let mut a = vec![vec![0.0; rows]; corners.len() + 1];
        let mut b = vec![0.0; rows];
        for (i, &f) in freqs.iter().enumerate() {
            let z = target(f);
            let mag = z.norm();
            if !(mag.is_finite() && mag > 0.0 && z.re >= -1e-6 * mag) {
                return Err(Error::Invalid(format!(
                    "impedance fit: target {z} at {f} Hz must be finite, with a real part ≥ 0"
                )));
            }
            let w = core::f64::consts::TAU * f;
            // Relative weights; a part that is (numerically) nothing next
            // to the other is weighted against the magnitude.
            let re = z.re.max(0.0);
            let (wr, wi) = (
                1.0 / re.max(1e-6 * mag),
                reactance_weight / z.im.abs().max(1e-6 * mag),
            );
            b[2 * i] = re * wr;
            b[2 * i + 1] = z.im * wi;
            a[0][2 * i] = wr;
            for (p, &c) in corners.iter().enumerate() {
                let d = w * w + c * c;
                a[1 + p][2 * i] = wr * w * w / d;
                a[1 + p][2 * i + 1] = wi * w * c / d;
            }
        }
        let x = nnls(&a, &b);
        Ok(Self {
            r0: x[0],
            sections: x[1..]
                .iter()
                .zip(&corners)
                .filter(|(r, _)| **r > 0.0)
                .map(|(&r, &c)| (r, c))
                .collect(),
        })
    }

    /// The fit of a half space of `c` over `f_lo..f_hi`
    /// ([`half_space_impedance`]).
    ///
    /// # Errors
    /// As [`fit`](Self::fit), or an invalid conductor.
    pub fn half_space(c: &Conductor, f_lo: f64, f_hi: f64) -> Result<Self> {
        c.validate()?;
        Self::fit(|f| half_space_impedance(c, f), f_lo, f_hi, weight(c))
    }

    /// The fits `(Z_e, Z_o)` of a sheet of `c` with thickness `t` \[m\]
    /// over `f_lo..f_hi` ([`sheet_mode_impedances`]).
    ///
    /// # Errors
    /// As [`fit`](Self::fit), or an invalid conductor or thickness.
    pub fn sheet(c: &Conductor, t: f64, f_lo: f64, f_hi: f64) -> Result<(Self, Self)> {
        c.validate()?;
        if !(t.is_finite() && t > 0.0) {
            return Err(Error::Invalid(format!(
                "sheet thickness {t} m must be finite and > 0"
            )));
        }
        let w = weight(c);
        // The poles of the two impedances themselves (their partial
        // fractions: `Z_c·tanh(γt/2) = Σ (4/σt)·s/(s + a_m)` over odd `m`,
        // `Z_c·coth(γt/2) = 2/σt +` the same over even `m`, with
        // `a_m = m²π²/(μ0σt²)`), as far as they lie in the fitted range;
        // the fit's own corners stand for the rest.
        let a1 = core::f64::consts::PI.powi(2) / (MU0 * c.conductivity * t * t);
        let top = core::f64::consts::TAU * f_hi * 10.0;
        let poles = |first: usize| -> Vec<f64> {
            (0..SHEET_POLES)
                .map(|k| ((first + 2 * k) as f64).powi(2) * a1)
                .filter(|&a| a < top)
                .collect()
        };
        Ok((
            Self::fit_with(
                |f| sheet_mode_impedances(c, t, f).0,
                f_lo,
                f_hi,
                w,
                &poles(2),
            )?,
            Self::fit_with(
                |f| sheet_mode_impedances(c, t, f).1,
                f_lo,
                f_hi,
                w,
                &poles(1),
            )?,
        ))
    }
}

fn weight(c: &Conductor) -> f64 {
    if c.roughness == Roughness::Smooth {
        1.0
    } else {
        ROUGH_REACTANCE_WEIGHT
    }
}

/// An edge on the surface of solid lossy metal (module docs). The edge
/// must be PEC in the engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidEdge {
    /// Direction of the edge.
    pub axis: Axis,
    /// Start node of the edge.
    pub node: [usize; 3],
    /// The metal.
    pub conductor: Conductor,
    /// Capacitance \[F\] and conductance \[S\] of the edge for the part of
    /// its dual face outside the metal (`ε·A/l`, `κ·A/l`); `None`: vacuum
    /// on the quarters of the dual face next to a face that carries field.
    pub outside: Option<(f64, f64)>,
}

/// The conductor surface that a solid edge carries through one of the
/// faces around it, where that surface crosses the cells
/// ([`SurfaceImpedance::cut_surface`]; module docs, *Oblique surfaces*).
///
/// The face (normal `c`, the direction of its dual edge) contains the
/// edge (along `d`) and extends from it along the third axis `a`. Two
/// slopes describe the surface next to the edge: `along`, of its trace in
/// the plane of the face against the edge (`|Δa/Δd|`; 0: parallel to the
/// edge), and `across`, of its trace in the edge's dual plane against the
/// dual edge of the face (`|Δa/Δc|`; 0: parallel to it). A surface along
/// the mesh has both 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutSurface {
    /// The solid edge: axis and start node.
    pub edge: (Axis, [usize; 3]),
    /// The face: normal and lowest node.
    pub face: (Axis, [usize; 3]),
    /// Slope of the surface's trace in the face against the edge.
    pub along: f64,
    /// Slope of the surface's trace in the dual plane of the edge against
    /// the dual edge of the face.
    pub across: f64,
    /// The part of the edge's length that lies in the surface, in
    /// `(0, 1]`: less than 1 where another surface of the metal rises
    /// from the edge through the face (a floor edge that runs under a
    /// wall).
    pub exposed: f64,
    /// How far the metal reaches from the edge in the plane of the face
    /// \[m\], the mean over the edge's length (0: not given). Where
    /// that plane holds a surface of the metal whose outline crosses the
    /// face, this is the width the edge carries of it on this side: its
    /// own half cell and what lies under the cut edges beyond, whose
    /// current its dual loop encloses.
    pub reach: f64,
}

/// Least part of an edge taken as lying in its surface
/// ([`CutSurface::exposed`]).
const MIN_EXPOSED: f64 = 1e-3;

/// Largest slope taken from an [`CutSurface`] along the edge: an edge
/// at 84° to its surface carries it with a tenth of its voltage.
const MAX_SLOPE: f64 = 10.0;

/// Exponent of the share of a rim patch's current that the half cell at
/// the rim carries, `(λ/P)^0.28` for a patch of width `P` (module docs,
/// *Rims*).
const RIM_SHARE: f64 = 0.28;
/// The same for a patch with a rim on both sides, `½·(2λ/P)^0.4` in each
/// of the two half cells.
const STRIP_SHARE: f64 = 0.4;

/// Width of the surface that an edge beside the rim of a flat surface
/// carries on its dual edge: `reach` of metal from the edge to the rim,
/// in a cell whose half is `lump`, and `inner` on the other side (module
/// docs, *Rims*).
fn rim_width(reach: f64, lump: f64, inner: f64) -> f64 {
    let p = reach + inner;
    if !(lump > 0.0 && p > lump) {
        return p;
    }
    let f = (lump / p).powf(RIM_SHARE);
    1.0 / (f * f / lump + (1.0 - f) * (1.0 - f) / (p - lump))
}

/// [`rim_width`] of an edge with a rim on both sides, `total` of metal
/// between them.
fn strip_width(total: f64, lump: f64) -> f64 {
    if !(lump > 0.0 && total > 2.0 * lump) {
        return total;
    }
    let f = 0.5 * (2.0 * lump / total).powf(STRIP_SHARE);
    1.0 / (2.0 * f * f / lump + (1.0 - 2.0 * f) * (1.0 - 2.0 * f) / (total - 2.0 * lump))
}

/// The discrete one-port of an [`ImpedanceFit`] for one time step
/// (module docs, *Time stepping*).
#[derive(Debug, Clone)]
struct Port {
    /// `Z∞ = r0 + Σ β`.
    z_inf: f64,
    /// `Σ β`.
    b_sum: f64,
    /// Per section: `2/(2 + a·dt)`, `(2 − a·dt)/(2 + a·dt)`, `2·β`, and
    /// for the energy `r`, `½·r/a`.
    sec: Vec<[f64; 5]>,
}

impl Port {
    fn new(fit: &ImpedanceFit, dt: f64) -> Self {
        let sec: Vec<[f64; 5]> = fit
            .sections
            .iter()
            .map(|&(r, a)| {
                let d = 2.0 + a * dt;
                [2.0 / d, (2.0 - a * dt) / d, 4.0 * r / d, r, 0.5 * r / a]
            })
            .collect();
        let b_sum: f64 = sec.iter().map(|s| 0.5 * s[2]).sum();
        // A floor keeps the face equations regular for a degenerate fit.
        let z_inf = (fit.r0 + b_sum).max(1e-30);
        Self { z_inf, b_sum, sec }
    }

    /// Values per port in the state: the current, then the sections.
    fn stride(&self) -> usize {
        1 + self.sec.len()
    }

    /// `e` of `x̄ = Z∞·h̄ + e` from the state `[h, v…]`.
    #[inline]
    fn history(&self, st: &[f64]) -> f64 {
        let mut e = -self.b_sum * st[0];
        for (s, v) in self.sec.iter().zip(&st[1..]) {
            e += s[0] * v;
        }
        e
    }

    /// Advances the state to the end of the step, given the mean current.
    #[inline]
    fn advance(&self, st: &mut [f64], h_mean: f64) {
        let dh = h_mean - st[0];
        for (s, v) in self.sec.iter().zip(&mut st[1..]) {
            *v = s[1] * *v + s[2] * dh;
        }
        st[0] += 2.0 * dh;
    }

    /// Magnetic energy of the sections per unit `1/g`.
    fn energy(&self, st: &[f64]) -> f64 {
        self.sec
            .iter()
            .zip(&st[1..])
            .map(|(s, v)| {
                let il = st[0] - v / s[3];
                s[4] * il * il
            })
            .sum()
    }
}

/// A group of couplings: one derivative term of a face's circulation.
#[derive(Debug, Clone, Copy, Default)]
struct Group {
    c0: u32,
    nc: u32,
    /// Index of its CPML stretch, or `u32::MAX`.
    stretch: u32,
}

/// CPML stretch of one derivative term: `1/κ`, `a`, `b` and `ψ`.
#[derive(Debug, Clone, Copy)]
struct Stretch {
    ik: f64,
    a: f64,
    b: f64,
    psi: f64,
}

/// Where a field value lies. As installed: the x-plane and the offset
/// `j·nz + k` in it. In the serial hooks: the slab and the offset in its
/// block. In a slab: 0 and the offset in the slab's block, or 1 and the
/// offset in the one plane of the neighbour slab the half-step may read.
type Loc = (u32, u32);

/// A dual edge in a face's circulation: location and weight.
#[derive(Debug, Clone, Copy)]
struct Coupling {
    axis: Axis,
    loc: Loc,
    w: f64,
}

/// The H correction of a sheet's `−` side on one dual edge:
/// `i −= db·(t·(u− − u+)/κ + ψ)`, as the compile step collects it.
#[derive(Debug, Clone, Copy)]
struct RawCorrection {
    axis: Axis,
    loc: Loc,
    /// `db·t`, `t` the weight of the edge in the dual edge's curl.
    dbt: f64,
    /// `db` and the stretch, inside a CPML (else `u32::MAX`).
    db: f64,
    stretch: u32,
    /// `½·L·corr` for the energy (as [`super::sheet`]).
    ecorr: f64,
}

/// A [`RawCorrection`] where it is applied: with the dual edge, in the
/// H half-step.
#[derive(Debug, Clone, Copy)]
struct Correction {
    axis: Axis,
    loc: Loc,
    /// x-plane of the dual edge.
    plane: u32,
    dbt: f64,
    db: f64,
    stretch: u32,
    /// The sheet edge: its index among the sheets of this part, or with
    /// [`REMOTE`] set the index of its slot in the next slab.
    edge: u32,
}

/// Flag of [`Correction::edge`]: the sheet edge lies in the first plane
/// of the next slab.
const REMOTE: u32 = 1 << 31;

/// The energy term of a [`RawCorrection`], with its sheet edge.
#[derive(Debug, Clone, Copy)]
struct EnergyTerm {
    axis: Axis,
    loc: Loc,
    ecorr: f64,
}

#[derive(Debug, Clone)]
struct SolidState {
    axis: Axis,
    /// x-plane of the edge.
    plane: u32,
    loc: Loc,
    groups: [Group; 2],
    /// `2C/dt`, `1/(2C/dt + G + g/Z∞)`, `g`.
    two_c: f64,
    inv: f64,
    g: f64,
    port: u32,
    st: u32,
    u: f64,
}

#[derive(Debug, Clone)]
struct SheetState {
    axis: Axis,
    node: [usize; 3],
    loc: Loc,
    /// Groups of the `+` face, then of the `−` face.
    groups: [Group; 4],
    /// `2C±/dt`.
    two_c: [f64; 2],
    /// Inverse of the 2 × 2 face matrix.
    m: [f64; 3],
    g: f64,
    c_edge: f64,
    /// Even and odd ports and their states.
    port: [u32; 2],
    st: [u32; 2],
    /// Its corrections: raw ones until the operator is registered, then
    /// their energy terms.
    k0: u32,
    nk: u32,
    u: [f64; 2],
    /// Index of its open part in `gaps` (edges partly covered), or
    /// `u32::MAX`.
    gap: u32,
    /// Share of `u−` in the face voltage the array holds,
    /// `u+ + mix·(u− − u+)` (0: `u+`).
    mix: f64,
    /// The array value written by the last E update (edges with a gap or
    /// a mix, whose `u+` is not read back from the array).
    written: f32,
    /// Its term of the look-ahead energy after the last E update.
    wc: f64,
    /// Its slot among those its slab publishes for the slab before it,
    /// or `u32::MAX`.
    slot: u32,
}

/// How a sheet edge takes what another operator added to its array value
/// since the last E update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// The array holds `u+`: both faces move with it.
    Plain,
    /// The array holds a mix of the faces: both move by the difference.
    Mix,
    /// The edge has an open part, which takes the difference.
    Gap,
}

impl SheetState {
    fn mode(&self) -> Mode {
        if self.gap != u32::MAX {
            Mode::Gap
        } else if self.mix != 0.0 {
            Mode::Mix
        } else {
            Mode::Plain
        }
    }
}

/// The face voltages `u`, the array value `written` by the last E update
/// and what goes to the open part, once the array holds `up`: what other
/// operators (sources) added to the edge since the last E update acts on
/// both faces (or on the open part of an edge partly covered). Applying
/// it twice changes nothing.
#[inline]
fn synced(mode: Mode, u: [f64; 2], written: f32, up: f32) -> ([f64; 2], f32, f64) {
    match mode {
        Mode::Plain if up != u[0] as f32 => ([up as f64, u[1] + (up as f64 - u[0])], written, 0.0),
        Mode::Mix if up != written => {
            let d = up as f64 - written as f64;
            ([u[0] + d, u[1] + d], up, 0.0)
        }
        Mode::Gap if up != written => (u, up, up as f64 - written as f64),
        _ => (u, written, 0.0),
    }
}

/// The open part of a sheet edge the sheet covers partly: an ordinary
/// edge in series with the covered part (as in [`super::sheet`]).
#[derive(Debug, Clone, Copy)]
struct Gap {
    /// Kernel coefficients of the edge (`cb` scaled to the open part).
    ca: f64,
    cb: f64,
    /// Voltage over the open part.
    u: f64,
}

/// A sheet edge in the first plane of a slab, as the slab before it sees
/// it in the H half-step: its face voltages and array value after the
/// last E update. Written by its own slab in the E half-step, read by
/// the other one in the H half-step (never at the same time).
#[derive(Debug, Default)]
struct Slot {
    u: [AtomicU64; 2],
    written: AtomicU32,
}

/// A sheet edge of the next slab that a correction here needs: where its
/// array value is, and how it takes what was added to it.
#[derive(Debug, Clone, Copy)]
struct Remote {
    axis: Axis,
    loc: Loc,
    mode: Mode,
}

/// The edges of one slab (or all of them, run in the serial hooks) with
/// their state.
#[derive(Debug, Default)]
struct Part {
    solids: Vec<SolidState>,
    sheets: Vec<SheetState>,
    couplings: Vec<Coupling>,
    /// Stretches of the circulations (E half-step).
    stretches: Vec<Stretch>,
    /// Currents and section voltages of the ports.
    state: Vec<f64>,
    gaps: Vec<Gap>,
    terms: Vec<EnergyTerm>,
    /// The H corrections on the dual edges of this part, by x-plane (in
    /// the order of their sheet edges within a plane).
    corrections: Vec<Correction>,
    /// Stretches of the corrections (H half-step).
    h_stretches: Vec<Stretch>,
    remote: Vec<Remote>,
}

/// Field access of one half-step: the serial hooks' or a slab's.
trait Access {
    fn i(&self, axis: Axis, loc: Loc) -> f32;
    fn u(&self, axis: Axis, loc: Loc) -> f32;
    fn set_i(&mut self, axis: Axis, loc: Loc, v: f32);
    fn set_u(&mut self, axis: Axis, loc: Loc, v: f32);
}

struct Serial<'a, 'b>(&'a mut Fields<'b>);

impl Access for Serial<'_, '_> {
    #[inline]
    fn i(&self, axis: Axis, loc: Loc) -> f32 {
        self.0.i.at(axis, (loc.0 as usize, loc.1 as usize))
    }
    #[inline]
    fn u(&self, axis: Axis, loc: Loc) -> f32 {
        self.0.u.at(axis, (loc.0 as usize, loc.1 as usize))
    }
    #[inline]
    fn set_i(&mut self, axis: Axis, loc: Loc, v: f32) {
        self.0.i.set_at(axis, (loc.0 as usize, loc.1 as usize), v);
    }
    #[inline]
    fn set_u(&mut self, axis: Axis, loc: Loc, v: f32) {
        self.0.u.set_at(axis, (loc.0 as usize, loc.1 as usize), v);
    }
}

/// A slab in its H half-step: `i` written, `u` read.
struct SlabH<'a, 'b, 'c> {
    i: &'a mut SlabMut<'b>,
    u: &'a SlabRef<'c>,
}

/// The value at `loc` of a slab's readable field.
#[inline]
fn read(f: &SlabRef<'_>, axis: Axis, loc: Loc) -> f32 {
    if loc.0 == 0 {
        f.block[axis.index()][loc.1 as usize]
    } else {
        let (_, block, plane) = f.neighbour.expect("the neighbour plane");
        block[axis.index()][plane * f.n[1] * f.n[2] + loc.1 as usize]
    }
}

impl Access for SlabH<'_, '_, '_> {
    #[inline]
    fn i(&self, axis: Axis, loc: Loc) -> f32 {
        self.i.block[axis.index()][loc.1 as usize]
    }
    #[inline]
    fn u(&self, axis: Axis, loc: Loc) -> f32 {
        read(self.u, axis, loc)
    }
    #[inline]
    fn set_i(&mut self, axis: Axis, loc: Loc, v: f32) {
        self.i.block[axis.index()][loc.1 as usize] = v;
    }
    fn set_u(&mut self, _: Axis, _: Loc, _: f32) {
        unreachable!("the H half-step does not write voltages")
    }
}

/// A slab in its E half-step: `u` written, `i` read.
struct SlabE<'a, 'b, 'c> {
    u: &'a mut SlabMut<'b>,
    i: &'a SlabRef<'c>,
}

impl Access for SlabE<'_, '_, '_> {
    #[inline]
    fn i(&self, axis: Axis, loc: Loc) -> f32 {
        read(self.i, axis, loc)
    }
    #[inline]
    fn u(&self, axis: Axis, loc: Loc) -> f32 {
        self.u.block[axis.index()][loc.1 as usize]
    }
    fn set_i(&mut self, _: Axis, _: Loc, _: f32) {
        unreachable!("the E half-step does not write currents")
    }
    #[inline]
    fn set_u(&mut self, axis: Axis, loc: Loc, v: f32) {
        self.u.block[axis.index()][loc.1 as usize] = v;
    }
}

/// The stretched sum of a group of couplings.
#[inline]
fn circulation<A: Access>(
    g: &Group,
    couplings: &[Coupling],
    stretches: &mut [Stretch],
    f: &A,
) -> f64 {
    let mut s = 0.0;
    for c in &couplings[g.c0 as usize..(g.c0 + g.nc) as usize] {
        s += c.w * f.i(c.axis, c.loc) as f64;
    }
    if g.stretch != u32::MAX {
        let q = &mut stretches[g.stretch as usize];
        q.psi = q.b * q.psi + q.a * s;
        s = s * q.ik + q.psi;
    }
    s
}

impl Part {
    /// The solid edges in the x-planes `planes` (edges are in plane
    /// order).
    fn solids_in(&self, planes: &Range<usize>) -> Range<usize> {
        let lo = self
            .solids
            .partition_point(|e| (e.plane as usize) < planes.start);
        let hi = self
            .solids
            .partition_point(|e| (e.plane as usize) < planes.end);
        lo..hi
    }

    /// The sheet edges in the x-planes `planes`.
    fn sheets_in(&self, planes: &Range<usize>) -> Range<usize> {
        let lo = self.sheets.partition_point(|e| e.node[0] < planes.start);
        let hi = self.sheets.partition_point(|e| e.node[0] < planes.end);
        lo..hi
    }

    /// The corrections on the dual edges in the x-planes `planes`.
    fn corrections_in(&self, planes: &Range<usize>) -> Range<usize> {
        let lo = self
            .corrections
            .partition_point(|c| (c.plane as usize) < planes.start);
        let hi = self
            .corrections
            .partition_point(|c| (c.plane as usize) < planes.end);
        lo..hi
    }

    /// The H half-step: the sheet edges `sheets` take what was added to
    /// their array values, and the corrections `corrections` are applied;
    /// `next` holds the slots of the next slab.
    fn step_h<A: Access>(
        &mut self,
        f: &mut A,
        sheets: Range<usize>,
        corrections: Range<usize>,
        next: Option<&[Slot]>,
    ) {
        for e in &mut self.sheets[sheets] {
            let up = f.u(e.axis, e.loc);
            let (u, written, open) = synced(e.mode(), e.u, e.written, up);
            e.u = u;
            e.written = written;
            if e.gap != u32::MAX {
                self.gaps[e.gap as usize].u += open;
            }
        }
        for c in &self.corrections[corrections] {
            // The face voltages of the edge as its own slab has them
            // after this half-step (it may not have got there yet).
            let (mode, u, written, up) = if c.edge & REMOTE == 0 {
                let e = &self.sheets[c.edge as usize];
                (e.mode(), e.u, e.written, f.u(e.axis, e.loc))
            } else {
                let k = (c.edge & !REMOTE) as usize;
                let (r, slot) = (&self.remote[k], &next.expect("slots of the next slab")[k]);
                (
                    r.mode,
                    [
                        f64::from_bits(slot.u[0].load(Ordering::Relaxed)),
                        f64::from_bits(slot.u[1].load(Ordering::Relaxed)),
                    ],
                    f32::from_bits(slot.written.load(Ordering::Relaxed)),
                    f.u(r.axis, r.loc),
                )
            };
            let (u, _, _) = synced(mode, u, written, up);
            let mut d = c.dbt * (u[1] - u[0]);
            if c.stretch != u32::MAX {
                let q = &mut self.h_stretches[c.stretch as usize];
                let t = d / c.db;
                q.psi = q.b * q.psi + q.a * t;
                d = c.db * (t * q.ik + q.psi);
            }
            let v = f.i(c.axis, c.loc);
            f.set_i(c.axis, c.loc, flush(v - d as f32));
        }
    }

    /// The E half-step of the solid edges `solids` and the sheet edges
    /// `sheets`; `slots` are the slots of this slab.
    fn step_e<A: Access>(
        &mut self,
        ports: &[Port],
        f: &mut A,
        solids: Range<usize>,
        sheets: Range<usize>,
        slots: &[Slot],
    ) {
        let Self {
            solids: all_solids,
            sheets: all_sheets,
            couplings,
            stretches,
            state,
            gaps,
            terms,
            ..
        } = self;
        for e in &mut all_solids[solids] {
            let s = circulation(&e.groups[0], couplings, stretches, f)
                + circulation(&e.groups[1], couplings, stretches, f);
            let port = &ports[e.port as usize];
            let st = &mut state[e.st as usize..e.st as usize + port.stride()];
            let hist = port.history(st);
            let um = (s + e.two_c * e.u + hist / port.z_inf) * e.inv;
            port.advance(st, (e.g * um - hist) / port.z_inf);
            e.u = 2.0 * um - e.u;
            f.set_u(e.axis, e.loc, flush(e.u as f32));
        }
        for e in &mut all_sheets[sheets] {
            let mut s = [0.0; 2];
            for (q, g) in e.groups.iter().enumerate() {
                if g.nc != 0 {
                    s[q / 2] += circulation(g, couplings, stretches, f);
                }
            }
            let (pe, po) = (&ports[e.port[0] as usize], &ports[e.port[1] as usize]);
            let (he, ho) = (
                pe.history(&state[e.st[0] as usize..e.st[0] as usize + pe.stride()]),
                po.history(&state[e.st[1] as usize..e.st[1] as usize + po.stride()]),
            );
            let (je, jo) = (0.5 * he / pe.z_inf, 0.5 * ho / po.z_inf);
            let r = [
                s[0] + e.two_c[0] * e.u[0] + je + jo,
                s[1] + e.two_c[1] * e.u[1] + je - jo,
            ];
            let um = [e.m[0] * r[0] + e.m[1] * r[1], e.m[1] * r[0] + e.m[2] * r[1]];
            pe.advance(
                &mut state[e.st[0] as usize..e.st[0] as usize + pe.stride()],
                (e.g * (um[0] + um[1]) - he) / pe.z_inf,
            );
            po.advance(
                &mut state[e.st[1] as usize..e.st[1] as usize + po.stride()],
                (e.g * (um[0] - um[1]) - ho) / po.z_inf,
            );
            e.u = [2.0 * um[0] - e.u[0], 2.0 * um[1] - e.u[1]];
            // The face voltage in the array value, and the open part: the
            // kernel's update of an ordinary edge, with the currents of
            // all its dual edges.
            let face = e.u[0] + e.mix * (e.u[1] - e.u[0]);
            let plain = e.gap == u32::MAX && e.mix == 0.0;
            let up = if e.gap == u32::MAX {
                flush(face as f32)
            } else {
                let gap = &mut gaps[e.gap as usize];
                gap.u = gap.ca * gap.u + gap.cb * (s[0] + s[1]);
                flush((gap.u + face) as f32)
            };
            e.written = up;
            f.set_u(e.axis, e.loc, up);
            let mut s_e = 0.0;
            for c in &terms[e.k0 as usize..(e.k0 + e.nk) as usize] {
                s_e += c.ecorr * f.i(c.axis, c.loc) as f64;
            }
            e.wc = if plain {
                s_e * (e.u[1] - up as f64)
            } else {
                s_e * (e.u[1] - e.u[0])
            };
            if e.slot != u32::MAX {
                let slot = &slots[e.slot as usize];
                slot.u[0].store(e.u[0].to_bits(), Ordering::Relaxed);
                slot.u[1].store(e.u[1].to_bits(), Ordering::Relaxed);
                slot.written.store(up.to_bits(), Ordering::Relaxed);
            }
        }
    }
}

/// The installed edges: one part per slab, each run by the thread of its
/// slab, or one part run in the serial hooks.
#[derive(Debug)]
struct Parts {
    parts: Vec<Mutex<Part>>,
    ports: Arc<[Port]>,
    /// Per slab, the sheet edges of its first plane that the slab before
    /// it corrects dual edges for.
    slots: Vec<Vec<Slot>>,
}

impl Parts {
    fn part(&self, s: usize) -> std::sync::MutexGuard<'_, Part> {
        self.parts[s].lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl SlabHooks for Parts {
    fn post_h_slab(&self, i: &mut SlabMut<'_>, u: &SlabRef<'_>, ctx: &SlabContext<'_>) {
        self.post_h_planes(i, u, ctx, 0..usize::MAX);
    }

    fn post_e_slab(&self, u: &mut SlabMut<'_>, i: &SlabRef<'_>, ctx: &SlabContext<'_>) {
        self.post_e_planes(u, i, ctx, 0..usize::MAX);
    }

    fn fusion(&self) -> Fusion {
        Fusion::Planes
    }

    fn post_h_planes(
        &self,
        i: &mut SlabMut<'_>,
        u: &SlabRef<'_>,
        ctx: &SlabContext<'_>,
        planes: Range<usize>,
    ) {
        let mut p = self.part(ctx.slab);
        let (sheets, corrections) = (p.sheets_in(&planes), p.corrections_in(&planes));
        if sheets.is_empty() && corrections.is_empty() {
            return;
        }
        let next = self.slots.get(ctx.slab + 1).map(|s| &s[..]);
        p.step_h(&mut SlabH { i, u }, sheets, corrections, next);
    }

    fn post_e_planes(
        &self,
        u: &mut SlabMut<'_>,
        i: &SlabRef<'_>,
        ctx: &SlabContext<'_>,
        planes: Range<usize>,
    ) {
        let mut p = self.part(ctx.slab);
        let (solids, sheets) = (p.solids_in(&planes), p.sheets_in(&planes));
        if solids.is_empty() && sheets.is_empty() {
            return;
        }
        p.step_e(
            &self.ports,
            &mut SlabE { u, i },
            solids,
            sheets,
            &self.slots[ctx.slab],
        );
    }
}

/// Surface impedance on the faces of solid lossy metal and on conducting
/// sheets, with the rational skin-effect model of the
/// [module documentation](self).
///
/// Collect edges with [`push_solid`](Self::push_solid),
/// [`add_box`](Self::add_box) and [`push_sheet`](Self::push_sheet), then
/// install with [`add_to`](Self::add_to), after the engine's
/// [`CpmlBoundary`] if it has one.
///
/// ```
/// use ems::engine::surface::{Conductor, SurfaceImpedance};
/// use ems::engine::{EdgeMaterials, Engine, Grid};
/// use ems::Axis;
///
/// let g = Grid::uniform([8, 8, 8], [1e-3; 3]).unwrap();
/// // A metal block on the nodes [2, 6] × [2, 6] × [3, 4].
/// let mut m = EdgeMaterials::vacuum(&g);
/// for (axis, node) in ems::engine::surface::box_edges([2, 2, 3], [6, 6, 4]) {
///     m.set_pec_edge(axis, node);
/// }
/// let mut e = Engine::builder(g).edge_materials(m).threads(1).build().unwrap();
/// let mut s = SurfaceImpedance::new(Some(10e9));
/// s.add_box([2, 2, 3], [6, 6, 4], Conductor::new(5.8e7));
/// let h = s.add_to(&mut e).unwrap();
/// e.run(10);
/// // Every edge of the block is on its surface.
/// assert_eq!(e.operator(&h).solid_edges(), 2 * 2 * 4 * 5 + 5 * 5 - 9);
/// ```
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct SurfaceImpedance {
    f_max: Option<f64>,
    band: Option<(f64, f64)>,
    solid_entries: Vec<SolidEdge>,
    sheet_entries: Vec<(SheetEdge, Roughness)>,
    /// Open fractions of sheet edges partly covered
    /// ([`cut_sheet`](Self::cut_sheet)).
    sheet_cuts: Vec<(Axis, [usize; 3], f64)>,
    /// Faces partly covered by a sheet ([`cut_face`](Self::cut_face)).
    cut_faces: Vec<(Axis, [usize; 3])>,
    /// Open fractions of the edges solid metal covers in part
    /// ([`cut_edge`](Self::cut_edge)).
    cut_edges: Vec<(Axis, [usize; 3], f64)>,
    /// Surfaces crossing the cells beside solid edges
    /// ([`cut_surface`](Self::cut_surface)).
    oblique: Vec<CutSurface>,
    gaps: Vec<Gap>,
    installed: bool,
    skipped: usize,
    interior: usize,
    thin: usize,
    dt: f64,
    nodes: [usize; 3],
    /// Whether the edges can be run by the slabs (see [`Operator::slab_hooks`]).
    slab_ok: bool,
    /// The edges as installed, until the operator is registered.
    ports: Vec<Port>,
    solids: Vec<SolidState>,
    sheets: Vec<SheetState>,
    couplings: Vec<Coupling>,
    corrections: Vec<RawCorrection>,
    stretches: Vec<Stretch>,
    state: Vec<f64>,
    /// Counts of the installed edges.
    counts: Counts,
    /// The edges of a registered operator, and whether the slabs run
    /// them.
    parts: Option<Arc<Parts>>,
    slabs: bool,
}

/// Counts of the installed edges (kept when the lists go to the parts).
#[derive(Debug, Clone, Copy, Default)]
struct Counts {
    solids: usize,
    sheets: usize,
    gaps: usize,
    stretches: usize,
    state: usize,
}

/// The edges of the node box `start..=stop`: every edge between two of
/// its nodes, as `(axis, start node)`.
pub fn box_edges(start: [usize; 3], stop: [usize; 3]) -> Vec<(Axis, [usize; 3])> {
    let mut out = Vec::new();
    for axis in Axis::ALL {
        let a = axis.index();
        for i in start[0]..=stop[0] {
            for j in start[1]..=stop[1] {
                for k in start[2]..=stop[2] {
                    let node = [i, j, k];
                    if node[a] < stop[a] {
                        out.push((axis, node));
                    }
                }
            }
        }
    }
    out
}

/// The CPML coefficients `(1/κ, a, b)` of the layers normal to `p` at the
/// line (`e_side`) or cell index `k`, if a layer stretches there.
fn stretch_at(
    cpml: Option<&CpmlBoundary>,
    n: [usize; 3],
    p: Axis,
    k: usize,
    e_side: bool,
) -> Option<(f64, f64, f64)> {
    let cpml = cpml?;
    let np = n[p.index()];
    for (face, params) in cpml.layers().filter(|(f, _)| f.axis() == p) {
        let c = params.cells;
        let prof = if e_side {
            cpml.e_profile(face)
        } else {
            cpml.h_profile(face)
        }?;
        let idx = if face.is_max() {
            let inner = np - 1 - c;
            // E: the lines inner + 1..np; H: the cells inner..np − 1.
            let first = if e_side { inner + 1 } else { inner };
            (k >= first).then(|| k - first)
        } else {
            (k < c).then_some(k)
        };
        if let Some(q) = idx.and_then(|i| prof.get(i)) {
            return (q.a != 0.0 || q.kappa != 1.0).then_some((1.0 / q.kappa, q.a, q.b));
        }
    }
    None
}

impl SurfaceImpedance {
    /// An empty set for a run up to `f_max` \[Hz\]; `None`: `1/(10·dt)` of
    /// the engine. The impedances are fitted from `f_max/100` to `f_max`
    /// unless [`with_band`](Self::with_band) says otherwise.
    pub fn new(f_max: Option<f64>) -> Self {
        Self {
            f_max,
            ..Self::default()
        }
    }

    /// Returns `self` fitting the impedances over `f_lo..f_hi` \[Hz\].
    #[must_use]
    pub fn with_band(mut self, f_lo: f64, f_hi: f64) -> Self {
        self.band = Some((f_lo, f_hi));
        self
    }

    /// Adds an edge of solid metal. Edges inside the metal may be given
    /// too: they are left alone ([`interior`](Self::interior)).
    pub fn push_solid(&mut self, e: SolidEdge) {
        self.solid_entries.push(e);
    }

    /// Adds every edge of the node box `start..=stop` as solid metal `c`
    /// (the edges must be PEC in the engine).
    pub fn add_box(&mut self, start: [usize; 3], stop: [usize; 3], c: Conductor) {
        for (axis, node) in box_edges(start, stop) {
            self.push_solid(SolidEdge {
                axis,
                node,
                conductor: c,
                outside: None,
            });
        }
    }

    /// Adds a sheet edge (as for
    /// [`ConductingSheet`](super::ConductingSheet), which integrates the
    /// same sheet with a ladder) with the surface roughness `roughness`
    /// on both faces.
    pub fn push_sheet(&mut self, e: SheetEdge, roughness: Roughness) {
        self.sheet_entries.push((e, roughness));
    }

    /// Declares the sheet edge along `axis` at `node` as covered by the
    /// sheet over the fraction `1 − open` of its length only (`open` in
    /// `(0, 1)`; other values and edges that are not pushed are ignored),
    /// as [`ConductingSheet::cut`](super::ConductingSheet::cut) does for
    /// the ladder: the engine's coefficient of the edge must be that of
    /// its open part when the operator is installed (as
    /// [`ConformalPec::apply`](super::ConformalPec::apply) leaves a cut
    /// edge). The covered part carries the two face voltages, with
    /// `g = w/((1 − open)·l)` and the capacitance of its part of the dual
    /// face, in series with the open part; see
    /// [Outlines between mesh lines](super::sheet#outlines-between-mesh-lines).
    pub fn cut_sheet(&mut self, axis: Axis, node: [usize; 3], open: f64) {
        if open > 0.0 && open < 1.0 {
            self.sheet_cuts.push((axis, node, open));
        }
    }

    /// Declares the cell face of normal `axis` with lowest corner `node`,
    /// lying in the plane of a sheet, as covered by it in part (a cut
    /// face of conformal PEC), as
    /// [`ConductingSheet::cut_face`](super::ConductingSheet::cut_face):
    /// the sheet edges around it get the capacitance loading
    /// ([`CUT_FACE_LOADING`]) and hold the voltage the crossing dual
    /// edges need in the engine's array.
    pub fn cut_face(&mut self, axis: Axis, node: [usize; 3]) {
        self.cut_faces.push((axis, node));
    }

    /// Declares the edge along `axis` at `node` as covered by solid metal
    /// over the fraction `1 − open` of its length (a cut edge of
    /// conformal PEC; `open` in `(0, 1)`, other values are ignored). Its
    /// covered part stays perfectly conducting, but it counts as that
    /// fraction of a conductor edge for the width of the surface its
    /// neighbours of solid lossy metal carry: next to an outline oblique
    /// to the mesh those would otherwise carry half a cell where the
    /// metal goes on (a copper strip of four cells at 45° had a Q of 392
    /// where the same strip along the mesh had 639; with the cut edges
    /// counted, 602).
    pub fn cut_edge(&mut self, axis: Axis, node: [usize; 3], open: f64) {
        if open > 0.0 && open < 1.0 {
            self.cut_edges.push((axis, node, open));
        }
    }

    /// Declares how the conductor surface which the solid edge `s.edge`
    /// carries through its face `s.face` lies in the cells
    /// ([`CutSurface`]): the width of that surface in the edge's dual
    /// face is then the face's dual length projected on the surface, not
    /// the staircase's count of half cells (module docs, *Oblique
    /// surfaces*). Entries for edges or faces that are not installed are
    /// ignored, as are those with slopes that are not finite or no part
    /// of the edge in the surface; the last one given for an edge and
    /// face counts.
    pub fn cut_surface(&mut self, s: CutSurface) {
        if s.along.is_finite() && s.across.is_finite() && s.exposed > 0.0 {
            self.oblique.push(s);
        }
    }

    /// Number of installed sheet edges the sheet covers in part
    /// ([`cut_sheet`](Self::cut_sheet)).
    pub fn cut_sheet_edges(&self) -> usize {
        self.counts.gaps
    }

    /// Number of installed edges of solid metal.
    pub fn solid_edges(&self) -> usize {
        self.counts.solids
    }

    /// Number of installed sheet edges.
    pub fn sheet_edges(&self) -> usize {
        self.counts.sheets
    }

    /// Number of requested edges skipped at installation: duplicates,
    /// solid edges that are not PEC and sheet edges that are.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Number of requested solid edges inside the metal or on a
    /// re-entrant edge (no face with field around them), which stay at 0.
    pub fn interior(&self) -> usize {
        self.interior
    }

    /// Number of requested solid edges that are not on solid metal (faces
    /// with field on two opposite sides: zero-thickness metal, wires) and
    /// stay PEC.
    pub fn thin(&self) -> usize {
        self.thin
    }

    /// Number of derivative terms stretched by a CPML.
    pub fn stretched(&self) -> usize {
        self.counts.stretches
    }

    /// Number of state values (currents and section voltages).
    pub fn state_len(&self) -> usize {
        self.counts.state
    }

    /// Voltage of the `−` face of an installed sheet edge (the `+` face
    /// voltage is the engine's `u`).
    pub fn lower_voltage(&self, axis: Axis, node: [usize; 3]) -> Option<f64> {
        let found = |sheets: &[SheetState]| {
            sheets
                .iter()
                .find(|e| e.axis == axis && e.node == node)
                .map(|e| e.u[1])
        };
        match &self.parts {
            Some(p) => (0..p.parts.len()).find_map(|s| found(&p.part(s).sheets)),
            None => found(&self.sheets),
        }
    }

    /// Whether the edges are run by the engine's threads, each on its
    /// slab ([`Operator::slab_hooks`]); `false` before registration and
    /// where an operator registered earlier acts in the serial hooks after
    /// a half-step (the order of the two would change).
    pub fn runs_in_slabs(&self) -> bool {
        self.slabs
    }

    fn band(&self, dt: f64) -> Result<(f64, f64)> {
        let (lo, hi) = match (self.band, self.f_max) {
            (Some(b), _) => b,
            (None, Some(f)) => (f / 100.0, f),
            (None, None) => (1e-3 / dt, 0.1 / dt),
        };
        if lo.is_finite() && hi.is_finite() && lo > 0.0 && hi > lo {
            Ok((lo, hi))
        } else {
            Err(Error::Invalid(format!(
                "surface impedance: band {lo}..{hi} Hz must be finite with 0 < f_lo < f_hi"
            )))
        }
    }

    /// Adds the port of `fit`, returning its index and state offset
    /// factory.
    fn add_port(&mut self, fit: &ImpedanceFit) -> u32 {
        self.ports.push(Port::new(fit, self.dt));
        self.ports.len() as u32 - 1
    }

    fn alloc(&mut self, port: u32) -> u32 {
        let at = self.state.len();
        self.state
            .resize(at + self.ports[port as usize].stride(), 0.0);
        at as u32
    }

    /// Adds a group of couplings with its stretch along `p` at line `k`.
    fn group(
        &mut self,
        items: &[Coupling],
        cpml: Option<&CpmlBoundary>,
        n: [usize; 3],
        p: Axis,
        k: usize,
    ) -> Group {
        if items.is_empty() {
            return Group {
                stretch: u32::MAX,
                ..Group::default()
            };
        }
        let c0 = self.couplings.len() as u32;
        self.couplings.extend_from_slice(items);
        let stretch = match stretch_at(cpml, n, p, k, true) {
            Some((ik, a, b)) => {
                self.stretches.push(Stretch { ik, a, b, psi: 0.0 });
                self.stretches.len() as u32 - 1
            }
            None => u32::MAX,
        };
        Group {
            c0,
            nc: items.len() as u32,
            stretch,
        }
    }

    fn install(&mut self, engine: &mut Engine) -> Result<()> {
        if self.installed {
            return Err(Error::Invalid("surface impedance already installed".into()));
        }
        let grid = engine.grid().clone();
        if grid.is_cylindrical() {
            return Err(Error::Invalid(
                "surface impedance: Cartesian grids only".into(),
            ));
        }
        let n = grid.nodes();
        let exists = |axis: Axis, p: [usize; 3]| {
            (0..3).all(|t| p[t] < n[t]) && p[axis.index()] + 1 < n[axis.index()]
        };
        for e in &self.solid_entries {
            e.conductor.validate()?;
            if !exists(e.axis, e.node) {
                return Err(Error::Invalid(format!(
                    "surface impedance: {e:?} is not an edge of the grid ({n:?} nodes)"
                )));
            }
            if let Some((c, g)) = e.outside
                && !(c.is_finite() && g.is_finite() && c >= 0.0 && g >= 0.0)
            {
                return Err(Error::Invalid(format!(
                    "surface impedance: {e:?}: capacitance and conductance must be ≥ 0"
                )));
            }
        }
        for (e, r) in &self.sheet_entries {
            r.validate()?;
            let bad = |m: &str| Err(Error::Invalid(format!("surface impedance: {e:?}: {m}")));
            if e.axis == e.normal {
                return bad("the edge is not tangential to the sheet");
            }
            if !(e.conductivity.is_finite() && e.conductivity > 0.0) {
                return bad("conductivity must be finite and > 0");
            }
            if !(e.thickness.is_finite() && e.thickness > 0.0) {
                return bad("thickness must be finite and > 0");
            }
            if let Some((c, g)) = e.upper_share
                && !((0.0..=1.0).contains(&c) && (0.0..=1.0).contains(&g))
            {
                return bad("shares must be in [0, 1]");
            }
            if !exists(e.axis, e.node) {
                return bad("not an edge of the grid");
            }
        }
        let dt = engine.dt();
        self.dt = dt;
        let (f_lo, f_hi) = self.band(dt)?;
        self.nodes = n;
        // Slab hooks run before every serial hook: only when no operator
        // registered before this one acts serially after a half-step
        // (their order would change), as for the Mur faces.
        self.slab_ok = engine.periodic.is_none()
            && !engine
                .ops
                .iter()
                .any(|r| r.hooks.intersects(Hooks::POST_H | Hooks::POST_E));
        // (x-plane, offset in the plane.)
        let locate = |p: [usize; 3]| (p[0] as u32, (p[1] * n[2] + p[2]) as u32);
        let cpml: Option<CpmlBoundary> = engine.find_operator::<CpmlBoundary>().cloned();
        let cpml = cpml.as_ref().filter(|c| !c.is_cylindrical_rho_max());
        // PEC edges and the faces without field, from the coefficients as
        // they are now.
        let (pec, db): ([Vec<bool>; 3], [Vec<f32>; 3]) = {
            let co = engine.coefficients();
            (
                Axis::ALL.map(|a| co.cb(a).iter().map(|&c| c == 0.0).collect()),
                Axis::ALL.map(|a| co.db(a).to_vec()),
            )
        };
        let is_pec =
            |axis: Axis, p: [usize; 3]| exists(axis, p) && pec[axis.index()][grid.index(p)];
        // The face normal to `ax` at `m` (if it exists): whether all four
        // of its edges are PEC.
        let face = |ax: Axis, m: [usize; 3]| -> Option<bool> {
            let (p, q) = ax.others();
            if (0..3).any(|t| m[t] >= n[t])
                || m[p.index()] + 1 >= n[p.index()]
                || m[q.index()] + 1 >= n[q.index()]
            {
                return None;
            }
            let (mut mp, mut mq) = (m, m);
            mp[p.index()] += 1;
            mq[q.index()] += 1;
            Some(is_pec(p, m) && is_pec(p, mq) && is_pec(q, m) && is_pec(q, mp))
        };
        let half = |ax: Axis, k: usize, up: bool| -> f64 {
            if up {
                if k + 1 < n[ax.index()] {
                    0.5 * grid.delta(ax, k)
                } else {
                    0.0
                }
            } else if k > 0 {
                0.5 * grid.delta(ax, k - 1)
            } else {
                0.0
            }
        };
        // The four dual edges around an edge along `d` at `node`: the
        // dual edge's axis, its node, the side it lies on (axis, towards
        // higher index) and its sign in the E update.
        let around = |d: Axis, node: [usize; 3]| {
            let (b, c) = d.others();
            [
                (c, None, (b, true), 1.0),
                (c, Some(b), (b, false), -1.0),
                (b, None, (c, true), -1.0),
                (b, Some(c), (c, false), 1.0),
            ]
            .map(|(ax, off, side, s)| {
                let mut m = node;
                let ok = match off {
                    Some(o) if m[o.index()] == 0 => false,
                    Some(o) => {
                        m[o.index()] -= 1;
                        true
                    }
                    None => true,
                };
                (ax, ok.then_some(m), side, s)
            })
        };

        // (Looked up only: the order of the map does not matter.)
        let cut_edges: std::collections::HashMap<(usize, [usize; 3]), f64> = self
            .cut_edges
            .iter()
            .map(|&(a, p, f)| ((a.index(), p), 1.0 - f))
            .collect();
        // The part of the parallel edge at `p` that is a conductor.
        let cover = |axis: Axis, p: [usize; 3]| {
            if is_pec(axis, p) {
                1.0
            } else {
                cut_edges.get(&(axis.index(), p)).copied().unwrap_or(0.0)
            }
        };

        // (Looked up only: the order of the map does not matter.)
        type FaceKey = (usize, [usize; 3], usize, [usize; 3]);
        let oblique: std::collections::HashMap<FaceKey, (f64, f64, f64)> = self
            .oblique
            .iter()
            .filter(|o| o.along != 0.0 || o.across != 0.0 || o.exposed != 1.0)
            .map(|o| {
                (
                    (o.edge.0.index(), o.edge.1, o.face.0.index(), o.face.1),
                    (
                        o.along.abs().min(MAX_SLOPE),
                        o.across.abs(),
                        o.exposed.clamp(MIN_EXPOSED, 1.0),
                    ),
                )
            })
            .collect();

        // (edge, the axis across it, towards higher index) → reach.
        type SideKey = (usize, [usize; 3], usize, bool);
        let mut reach: std::collections::HashMap<SideKey, (f64, f64)> =
            std::collections::HashMap::new();
        for o in &self.oblique {
            if o.reach > 0.0 && o.reach.is_finite() {
                let a = 3 - o.edge.0.index() - o.face.0.index();
                let up = o.face.1[a] == o.edge.1[a];
                // (The half cell at the rim, along the mesh line across
                // the edge: `1/cos` of the outline's angle to the edge.)
                let slant = (1.0 + o.along.abs().min(MAX_SLOPE).powi(2)).sqrt();
                reach.insert((o.edge.0.index(), o.edge.1, a, up), (o.reach, slant));
            }
        }

        // `Σ dt/L` of the faces beside the requested edges that the
        // merged faces of conformal PEC drive in the kernel's place.
        // (Looked up only: the order of the map does not matter.)
        let requested: std::collections::HashSet<(usize, usize)> = if self.cut_edges.is_empty() {
            std::collections::HashSet::new()
        } else {
            self.solid_entries
                .iter()
                .map(|e| (e.axis.index(), grid.index(e.node)))
                .collect()
        };
        let merged: std::collections::HashMap<(usize, usize), f64> = if self.cut_edges.is_empty() {
            std::collections::HashMap::new()
        } else {
            engine
                .find_operator::<super::conformal::MergedFaces>()
                .map(|m| m.edge_diagonal(n, |a, k| requested.contains(&(a, k))))
                .unwrap_or_default()
        };
        // The edges of the face normal to `ax` at `m` that may get a
        // face voltage here.
        let surface_edges = |ax: Axis, m: [usize; 3]| {
            let (p, q) = ax.others();
            let (mut mp, mut mq) = (m, m);
            mp[p.index()] += 1;
            mq[q.index()] += 1;
            [(p, m), (p, mq), (q, m), (q, mp)]
                .iter()
                .filter(|&&(a, at)| {
                    is_pec(a, at) && requested.contains(&(a.index(), grid.index(at)))
                })
                .count()
        };
        // Whether the face normal to `ax` at `m` has an edge that solid
        // metal cuts.
        let by_cut = |ax: Axis, m: [usize; 3]| {
            let (p, q) = ax.others();
            let (mut mp, mut mq) = (m, m);
            mp[p.index()] += 1;
            mq[q.index()] += 1;
            [(p, m), (p, mq), (q, m), (q, mp)]
                .iter()
                .any(|&(a, at)| cut_edges.contains_key(&(a.index(), at)))
        };

        // ---- Solid metal ----
        let mut fits: Vec<(Conductor, u32)> = Vec::new();
        let mut order: Vec<usize> = (0..self.solid_entries.len()).collect();
        order.sort_by_key(|&i| {
            (
                self.solid_entries[i].node,
                self.solid_entries[i].axis.index(),
            )
        });
        let mut dead: Vec<(Axis, usize)> = Vec::new();
        let mut last = None;
        for i in order {
            let e = self.solid_entries[i];
            if last == Some((e.node, e.axis)) || !is_pec(e.axis, e.node) {
                self.skipped += 1;
                continue;
            }
            last = Some((e.node, e.axis));
            let d = e.axis;
            let (b, c) = d.others();
            // Faces with field around the edge.
            let mut live = Vec::new();
            let mut sides = [[false; 2]; 2]; // [b, c][−, +]
            let mut cut_sides = [[false; 2]; 2];
            for (ax, m, (side, up), s) in around(d, e.node) {
                let Some(m) = m else { continue };
                match face(ax, m) {
                    Some(false) => {
                        live.push((ax, m, s));
                        sides[usize::from(side == c)][usize::from(up)] = true;
                        cut_sides[usize::from(side == c)][usize::from(up)] = by_cut(ax, m);
                    }
                    Some(true) => dead.push((ax, grid.index(m))),
                    None => {}
                }
            }
            if live.is_empty() {
                self.interior += 1;
                continue;
            }
            // Field on two opposite sides: metal without thickness in
            // the mesh. Not so where the metal's outline crosses both
            // faces: a conductor one row of edges wide, with its metal
            // in both (and below or above the edge).
            if (0..2).any(|a| sides[a][0] && sides[a][1] && !(cut_sides[a][0] && cut_sides[a][1])) {
                self.thin += 1;
                continue;
            }
            // Width of the conductor surface inside the dual face.
            // Through each face with field: the dual length of the face
            // as far as the conductor goes on beside the edge. Where the
            // surface is oblique (`oblique`), the dual edge runs at an
            // angle to it and the surface at an angle to the edge:
            // `ℓ·cos γ / sin φ` (module docs).
            // `w` is the width over the part of the edge's length that
            // lies in the surface: a face through which another surface
            // rises from the edge leaves `exposed` of it.
            let mut w = 0.0;
            for &(ax, m, _) in &live {
                let trace = oblique.get(&(d.index(), e.node, ax.index(), m)).copied();
                let (factor, turned, exposed) = match trace {
                    Some((along, across, exposed)) => {
                        let c2 = 1.0 + across * across;
                        ((c2 + along * along).sqrt() / c2, across > 1e-9, exposed)
                    }
                    None => (1.0, false, 1.0),
                };
                // The two halves of the dual edge: (length, conductor
                // beside it, measured reach of a flat surface's rim).
                let mut halves = [(0.0, 0.0, None); 2];
                for (h, up) in [false, true].into_iter().enumerate() {
                    let mut q = e.node;
                    let k = q[ax.index()];
                    if up {
                        q[ax.index()] += 1;
                    } else if k == 0 {
                        continue;
                    } else {
                        q[ax.index()] -= 1;
                    }
                    // A surface turned in the dual plane goes on to
                    // both sides, whatever lies on the mesh line there.
                    let on = if turned { 1.0 } else { cover(d, q) };
                    let len = half(ax, k, up);
                    let rim = if trace.is_none() && on < 1.0 && len > 0.0 {
                        reach.get(&(d.index(), e.node, ax.index(), up)).copied()
                    } else {
                        None
                    };
                    halves[h] = (len, on, rim);
                }
                w += match (trace, halves[0].2, halves[1].2) {
                    (Some(_), ..) | (None, None, None) => {
                        (halves[0].0 * halves[0].1 + halves[1].0 * halves[1].1) * factor / exposed
                    }
                    (None, Some(a), None) => {
                        rim_width(a.0, halves[0].0 * a.1, halves[1].0 * halves[1].1)
                    }
                    (None, None, Some(a)) => {
                        rim_width(a.0, halves[1].0 * a.1, halves[0].0 * halves[0].1)
                    }
                    (None, Some(a), Some(b)) => {
                        strip_width(a.0 + b.0, halves[0].0.min(halves[1].0) * a.1.max(b.1))
                    }
                };
            }
            if w <= 0.0 {
                self.interior += 1;
                continue;
            }
            let l = grid.edge_length(d, e.node);
            let g = w / l;
            // Quarter (sb, sc) of the dual face: outside if a face with
            // field borders it. In vacuum that is the least capacitance
            // the edge can have, and what the explicit coupling to its
            // faces needs: values given for the edge (sampled on the
            // drawn metal, which reaches past the edges of its staircase)
            // are raised to it. An edge without capacitance is a resistor
            // `Z∞/g` in its faces' loops, integrated explicitly: a strip
            // of 2e5 S/m grew from the corners of its end.
            let mut area = 0.0;
            for sb in [false, true] {
                for sc in [false, true] {
                    if sides[0][usize::from(sb)] || sides[1][usize::from(sc)] {
                        area += half(b, e.node[b.index()], sb) * half(c, e.node[c.index()], sc);
                    }
                }
            }
            let mut floor = EPS0 * area / l;
            // Beside a cut cell the edge needs more. The trapezoidal
            // rule's admittance `g/Z` goes to zero at the highest
            // frequency of the mesh (it is `Y·cos(ω·dt/2)` in the edge's
            // update), where the edge is left with its capacitance alone:
            // an ordinary edge, where conformal PEC took a perfect
            // conductor when it decided which small faces the time step
            // allows. A face of a tenth of its area next to it has ten
            // times the `dt/L`. With `C = dt·Σ dt/L` over the faces around
            // it (those the merged faces drive included) the edge has the
            // diagonal `dt²·(C⁻¹·Kᵀ·L⁻¹·K)_ee = 1` (an edge of cubic
            // cells at the Courant limit has 4/3); with
            // [`FLOOR_MARGIN`] times that it is out of the way of those
            // modes. Next to `g/Z` this capacitance is nothing.
            if live.iter().any(|&(ax, m, _)| by_cut(ax, m)) {
                let mut sum = merged
                    .get(&(d.index(), grid.index(e.node)))
                    .copied()
                    .unwrap_or(0.0);
                for &(ax, m, _) in &live {
                    // (A face's circulation squared is at most the
                    // squares of its edges times their number: each of
                    // the face's surface edges answers for all of them.)
                    sum +=
                        surface_edges(ax, m).max(1) as f64 * db[ax.index()][grid.index(m)] as f64;
                }
                floor = floor.max(FLOOR_MARGIN * dt * sum);
            }
            let (cap, cond) = match e.outside {
                Some((cap, cond)) => (cap.max(floor), cond),
                None => (floor, 0.0),
            };
            let port = match fits.iter().find(|f| f.0 == e.conductor) {
                Some(f) => f.1,
                None => {
                    let fit = ImpedanceFit::half_space(&e.conductor, f_lo, f_hi)?;
                    let p = self.add_port(&fit);
                    fits.push((e.conductor, p));
                    p
                }
            };
            // The couplings by derivative axis: a dual edge along `ax`
            // belongs to the derivative along the third axis.
            let mut groups = [Group::default(); 2];
            for (slot, p) in [b, c].into_iter().enumerate() {
                let items: Vec<Coupling> = live
                    .iter()
                    .filter(|x| x.0 != p)
                    .map(|&(ax, m, s)| Coupling {
                        axis: ax,
                        loc: locate(m),
                        w: s,
                    })
                    .collect();
                groups[slot] = self.group(&items, cpml, n, p, e.node[p.index()]);
            }
            let two_c = 2.0 * cap / dt;
            let st = self.alloc(port);
            self.solids.push(SolidState {
                axis: d,
                plane: e.node[0] as u32,
                loc: locate(e.node),
                groups,
                two_c,
                inv: 1.0 / (two_c + cond + g / self.ports[port as usize].z_inf),
                g,
                port,
                st,
                u: 0.0,
            });
        }
        if !dead.is_empty() {
            let mut co = engine.coefficients_mut();
            for (ax, idx) in dead {
                co.h_mut(ax).1[idx] = 0.0;
            }
        }

        // ---- Sheets ----
        type SheetKey = (f64, f64, Roughness);
        let mut sheet_fits: Vec<(SheetKey, [u32; 2])> = Vec::new();
        // (Looked up only: the order of the map does not matter.)
        let sheet_normals: std::collections::HashMap<(usize, [usize; 3]), usize> = self
            .sheet_entries
            .iter()
            .map(|(e, _)| ((e.axis.index(), e.node), e.normal.index()))
            .collect();
        let cuts: std::collections::HashMap<(usize, [usize; 3]), f64> = self
            .sheet_cuts
            .iter()
            .map(|&(a, p, f)| ((a.index(), p), f))
            .collect();
        let cut_faces: std::collections::HashSet<(usize, [usize; 3])> = self
            .cut_faces
            .iter()
            .map(|&(a, p)| (a.index(), p))
            .collect();
        let outlines = !cut_faces.is_empty();
        let mut order: Vec<usize> = (0..self.sheet_entries.len()).collect();
        order.sort_by_key(|&i| {
            let e = &self.sheet_entries[i].0;
            (e.node, e.axis.index())
        });
        let mut last = None;
        for i in order {
            let (e, rough) = self.sheet_entries[i];
            if last == Some((e.node, e.axis)) || is_pec(e.axis, e.node) {
                self.skipped += 1;
                continue;
            }
            last = Some((e.node, e.axis));
            let (d, nrm) = (e.axis, e.normal);
            let t_ax = Axis::ALL[3 - nrm.index() - d.index()];
            let idx = grid.index(e.node);
            let (ca, cb) = {
                let co = engine.coefficients();
                (co.ca(d)[idx] as f64, co.cb(d)[idx] as f64)
            };
            let (c_edge, g_edge) = (dt * (1.0 + ca) / (2.0 * cb), (1.0 - ca) / cb);
            // An edge partly covered: the kernel's coefficients are those
            // of the open part (C/f, G/f); the covered part has
            // C/(1 − f), G/(1 − f) and the conductor of its length.
            let open = cuts.get(&(d.index(), e.node)).copied();
            let part = open.map_or(1.0, |f| f / (1.0 - f));
            let (c_cov, g_cov) = (part * c_edge, part * g_edge);
            let k = e.node[nrm.index()];
            let (up, dn) = (half(nrm, k, true), half(nrm, k, false));
            let f_up = up / (up + dn);
            // Width of the edge's strip of the sheet: half a cell on each
            // side where the sheet goes on (a parallel edge of the same
            // sheet plane, or metal); a single line of edges takes the
            // dual length.
            let mut w = 0.0;
            for up in [false, true] {
                let mut q = e.node;
                let kt = q[t_ax.index()];
                if up {
                    q[t_ax.index()] += 1;
                } else if kt == 0 {
                    continue;
                } else {
                    q[t_ax.index()] -= 1;
                }
                if sheet_normals.get(&(d.index(), q)) == Some(&nrm.index()) || is_pec(d, q) {
                    w += half(t_ax, kt, up);
                }
            }
            if w <= 0.0 {
                w = grid.dual_delta(t_ax, e.node[t_ax.index()]);
            }
            let g = w / grid.edge_length(d, e.node);
            let g = open.map_or(g, |f| g / (1.0 - f));
            let (sc, sg) = e.upper_share.unwrap_or((f_up, f_up));
            // Around a cut face: loaded (as the ladder's faces).
            let by_cut_face = {
                let mut lo = e.node;
                let below = lo[t_ax.index()] > 0;
                lo[t_ax.index()] = lo[t_ax.index()].saturating_sub(1);
                cut_faces.contains(&(nrm.index(), e.node))
                    || (below && cut_faces.contains(&(nrm.index(), lo)))
            };
            let load = if by_cut_face { CUT_FACE_LOADING } else { 1.0 };
            let key = (e.conductivity, e.thickness, rough);
            let port = match sheet_fits.iter().find(|f| f.0 == key) {
                Some(f) => f.1,
                None => {
                    let c = Conductor::new(e.conductivity).with_roughness(rough);
                    let (ze, zo) = ImpedanceFit::sheet(&c, e.thickness, f_lo, f_hi)?;
                    let p = [self.add_port(&ze), self.add_port(&zo)];
                    sheet_fits.push((key, p));
                    p
                }
            };
            // Couplings of the two faces and H corrections.
            let mut items: [[Vec<Coupling>; 2]; 2] = Default::default(); // [face][group]
            let k0 = self.corrections.len() as u32;
            // (dual edge axis, its node, shares of the two faces, sign)
            let mut dual = Vec::with_capacity(4);
            // Share of `u−` in the stored voltage: a crossing dual edge
            // that conformal PEC computes itself (`db = 0` in the kernel)
            // reads the array and cannot be corrected.
            let mut mix = 0.0;
            for (ax, m, (side, up), s) in around(d, e.node) {
                let Some(m) = m else { continue };
                if face(ax, m).is_none() {
                    continue;
                }
                let (w_up, w_dn) = if ax == nrm {
                    (f_up, 1.0 - f_up)
                } else if side == nrm && !up {
                    (0.0, 1.0)
                } else {
                    (1.0, 0.0)
                };
                if ax == nrm && outlines && db[ax.index()][grid.index(m)] == 0.0 {
                    mix = w_dn;
                }
                dual.push((ax, m, (w_up, w_dn), s));
            }
            for (ax, m, (w_up, w_dn), s) in dual {
                // Derivative along the third axis.
                let p = Axis::ALL[3 - d.index() - ax.index()];
                let slot = usize::from(p != d.others().0);
                let loc = locate(m);
                if w_up != 0.0 {
                    items[0][slot].push(Coupling {
                        axis: ax,
                        loc,
                        w: s * w_up,
                    });
                }
                if w_dn != 0.0 {
                    items[1][slot].push(Coupling {
                        axis: ax,
                        loc,
                        w: s * w_dn,
                    });
                }
                // The dual edge read the stored voltage, with the share
                // `mix` of `u−`; its own share is `w_dn`.
                let wc = w_dn - mix;
                let dbv = db[ax.index()][grid.index(m)] as f64;
                if wc != 0.0 && dbv != 0.0 {
                    let da = engine.coefficients().da(ax)[grid.index(m)] as f64;
                    let stretch = match stretch_at(cpml, n, p, m[p.index()], false) {
                        Some((ik, a, b)) => {
                            self.stretches.push(Stretch { ik, a, b, psi: 0.0 });
                            self.stretches.len() as u32 - 1
                        }
                        None => u32::MAX,
                    };
                    self.corrections.push(RawCorrection {
                        axis: ax,
                        loc,
                        dbt: dbv * s * wc,
                        db: dbv,
                        stretch,
                        ecorr: -0.25 * dt * (1.0 + da) * s * wc,
                    });
                }
            }
            let (b_ax, c_ax) = d.others();
            let mut groups = [Group::default(); 4];
            for f in 0..2 {
                for (slot, p) in [b_ax, c_ax].into_iter().enumerate() {
                    groups[2 * f + slot] =
                        self.group(&items[f][slot], cpml, n, p, e.node[p.index()]);
                }
            }
            let two_c = [
                2.0 * load * sc * c_cov / dt,
                2.0 * load * (1.0 - sc) * c_cov / dt,
            ];
            let (ye, yo) = (
                0.5 * g / self.ports[port[0] as usize].z_inf,
                0.5 * g / self.ports[port[1] as usize].z_inf,
            );
            let m11 = two_c[0] + sg * g_cov + ye + yo;
            let m22 = two_c[1] + (1.0 - sg) * g_cov + ye + yo;
            let m12 = ye - yo;
            let det = m11 * m22 - m12 * m12;
            let st = [self.alloc(port[0]), self.alloc(port[1])];
            let gap = match open {
                Some(_) => {
                    self.gaps.push(Gap { ca, cb, u: 0.0 });
                    self.gaps.len() as u32 - 1
                }
                None => u32::MAX,
            };
            self.sheets.push(SheetState {
                axis: d,
                node: e.node,
                loc: locate(e.node),
                groups,
                two_c,
                m: [m22 / det, -m12 / det, m11 / det],
                g,
                c_edge,
                port,
                st,
                k0,
                nk: self.corrections.len() as u32 - k0,
                u: [0.0; 2],
                gap,
                mix,
                written: 0.0,
                wc: 0.0,
                slot: u32::MAX,
            });
        }
        self.counts = Counts {
            solids: self.solids.len(),
            sheets: self.sheets.len(),
            gaps: self.gaps.len(),
            stretches: self.stretches.len(),
            state: self.state.len(),
        };
        self.installed = true;
        Ok(())
    }

    /// Validates the edges, fits the impedances for `engine` and registers
    /// the operator. The faces inside solid metal get `db = 0`; no other
    /// coefficient changes.
    ///
    /// # Errors
    /// [`Error::Invalid`] for an invalid edge, conductor or band, a
    /// cylindrical grid or a second installation.
    pub fn add_to(mut self, engine: &mut Engine) -> Result<OperatorHandle<SurfaceImpedance>> {
        self.install(engine)?;
        Ok(engine.add_operator(self))
    }

    /// The installed edges as parts: one per slab of `starts` (the first
    /// x-plane of each slab, and the number of planes), or a single part
    /// for the serial hooks, its locations as slab and offset.
    fn take_parts(&mut self, starts: &[usize], slabs: bool) -> Parts {
        let n = self.nodes;
        let count = if slabs { starts.len() - 1 } else { 1 };
        let slab_of = |plane: usize| {
            if slabs {
                starts.partition_point(|&s| s <= plane) - 1
            } else {
                0
            }
        };
        let ports: Arc<[Port]> = std::mem::take(&mut self.ports).into();
        let mut parts: Vec<Part> = (0..count).map(|_| Part::default()).collect();
        let mut slots: Vec<usize> = vec![0; count];
        let couplings = std::mem::take(&mut self.couplings);
        let stretches = std::mem::take(&mut self.stretches);
        let raw = std::mem::take(&mut self.corrections);
        let gaps = std::mem::take(&mut self.gaps);
        self.state = Vec::new();
        // A group with its couplings and its stretch, in `p`.
        let group = |p: &mut Part, g: Group| -> Group {
            if g.nc == 0 {
                return g;
            }
            let c0 = p.couplings.len() as u32;
            p.couplings
                .extend_from_slice(&couplings[g.c0 as usize..(g.c0 + g.nc) as usize]);
            let stretch = if g.stretch == u32::MAX {
                u32::MAX
            } else {
                p.stretches.push(stretches[g.stretch as usize]);
                p.stretches.len() as u32 - 1
            };
            Group {
                c0,
                nc: g.nc,
                stretch,
            }
        };
        let alloc = |p: &mut Part, port: u32| -> u32 {
            let at = p.state.len();
            p.state.resize(at + ports[port as usize].stride(), 0.0);
            at as u32
        };
        for mut e in std::mem::take(&mut self.solids) {
            let p = &mut parts[slab_of(e.plane as usize)];
            e.groups = e.groups.map(|g| group(p, g));
            e.st = alloc(p, e.port);
            p.solids.push(e);
        }
        for mut e in std::mem::take(&mut self.sheets) {
            let s = slab_of(e.node[0]);
            let mode = e.mode();
            let local = parts[s].sheets.len() as u32;
            for c in &raw[e.k0 as usize..(e.k0 + e.nk) as usize] {
                // The dual edge lies in the plane of the edge or in the
                // one before it: in this slab or in the one before.
                let t = slab_of(c.loc.0 as usize);
                let edge = if t == s {
                    local
                } else {
                    debug_assert_eq!(t + 1, s);
                    if e.slot == u32::MAX {
                        e.slot = slots[s] as u32;
                        slots[s] += 1;
                        parts[t].remote.push(Remote {
                            axis: e.axis,
                            loc: e.loc,
                            mode,
                        });
                    }
                    e.slot | REMOTE
                };
                let p = &mut parts[t];
                let stretch = if c.stretch == u32::MAX {
                    u32::MAX
                } else {
                    p.h_stretches.push(stretches[c.stretch as usize]);
                    p.h_stretches.len() as u32 - 1
                };
                p.corrections.push(Correction {
                    axis: c.axis,
                    loc: c.loc,
                    plane: c.loc.0,
                    dbt: c.dbt,
                    db: c.db,
                    stretch,
                    edge,
                });
            }
            let p = &mut parts[s];
            let k0 = p.terms.len() as u32;
            p.terms.extend(
                raw[e.k0 as usize..(e.k0 + e.nk) as usize]
                    .iter()
                    .map(|c| EnergyTerm {
                        axis: c.axis,
                        loc: c.loc,
                        ecorr: c.ecorr,
                    }),
            );
            e.k0 = k0;
            e.groups = e.groups.map(|g| group(p, g));
            e.st = [alloc(p, e.port[0]), alloc(p, e.port[1])];
            if e.gap != u32::MAX {
                p.gaps.push(gaps[e.gap as usize]);
                e.gap = p.gaps.len() as u32 - 1;
            }
            p.sheets.push(e);
        }
        let plane = (n[1] * n[2]) as u32;
        for (s, p) in parts.iter_mut().enumerate() {
            if slabs {
                // By plane; the corrections of a dual edge keep the order
                // of their sheet edges.
                p.corrections.sort_by_key(|c| c.plane);
                // The offset in the slab's block, or in the plane of the
                // neighbour slab that the half-step reads (the plane
                // before for the currents of a circulation, the plane
                // after for the voltage of a sheet edge).
                let (lo, hi) = (starts[s] as u32, starts[s + 1] as u32);
                let conv = |l: &mut Loc| {
                    *l = if (lo..hi).contains(&l.0) {
                        (0, (l.0 - lo) * plane + l.1)
                    } else {
                        (1, l.1)
                    };
                };
                p.solids.iter_mut().for_each(|e| conv(&mut e.loc));
                p.sheets.iter_mut().for_each(|e| conv(&mut e.loc));
                p.couplings.iter_mut().for_each(|c| {
                    debug_assert!((lo..hi).contains(&c.loc.0) || c.loc.0 + 1 == lo);
                    conv(&mut c.loc);
                });
                p.terms.iter_mut().for_each(|c| conv(&mut c.loc));
                p.corrections.iter_mut().for_each(|c| conv(&mut c.loc));
                p.remote.iter_mut().for_each(|r| {
                    debug_assert_eq!(r.loc.0, hi);
                    conv(&mut r.loc);
                });
            } else {
                // Slab and offset in its block.
                let conv = |l: &mut Loc| {
                    let s = starts.partition_point(|&s| s <= l.0 as usize) - 1;
                    *l = (s as u32, (l.0 - starts[s] as u32) * plane + l.1);
                };
                p.solids.iter_mut().for_each(|e| conv(&mut e.loc));
                p.sheets.iter_mut().for_each(|e| conv(&mut e.loc));
                p.couplings.iter_mut().for_each(|c| conv(&mut c.loc));
                p.terms.iter_mut().for_each(|c| conv(&mut c.loc));
                p.corrections.iter_mut().for_each(|c| conv(&mut c.loc));
            }
            p.solids.shrink_to_fit();
            p.sheets.shrink_to_fit();
            p.couplings.shrink_to_fit();
            p.stretches.shrink_to_fit();
            p.state.shrink_to_fit();
            p.gaps.shrink_to_fit();
            p.terms.shrink_to_fit();
            p.corrections.shrink_to_fit();
            p.h_stretches.shrink_to_fit();
            p.remote.shrink_to_fit();
        }
        Parts {
            parts: parts.into_iter().map(Mutex::new).collect(),
            ports,
            slots: slots
                .into_iter()
                .map(|k| (0..k).map(|_| Slot::default()).collect())
                .collect(),
        }
    }
}

impl Operator for SurfaceImpedance {
    fn hooks(&self) -> Hooks {
        let mut h = Hooks::NONE;
        if self.slabs {
            return h;
        }
        if self.counts.sheets != 0 {
            h |= Hooks::POST_H;
        }
        if self.counts.sheets + self.counts.solids != 0 {
            h |= Hooks::POST_E;
        }
        h
    }

    /// The edges go to the slabs of the engine, each run by its thread
    /// right after the kernel in the same half-step (bit-identical for
    /// any thread count: every edge is computed by the same expressions
    /// from the same values, and what a slab needs of the next one's
    /// edges is what that slab left after its last E update). Where an
    /// operator registered earlier has serial hooks after a half-step,
    /// they stay in the serial hooks, in registration order as before.
    fn slab_hooks(&mut self, part: &SlabPartition<'_>) -> Option<Arc<dyn SlabHooks>> {
        if !self.installed || self.parts.is_some() {
            // Registered without `add_to` (nothing to do) or twice.
            return None;
        }
        assert_eq!(
            part.grid().nodes(),
            self.nodes,
            "surface impedance installed for another grid"
        );
        let starts: Vec<usize> = (0..part.slabs())
            .map(|s| part.planes(s).start)
            .chain([self.nodes[0]])
            .collect();
        self.slabs = self.slab_ok && self.counts.sheets + self.counts.solids != 0;
        let parts = Arc::new(self.take_parts(&starts, self.slabs));
        self.parts = Some(Arc::clone(&parts));
        if self.slabs { Some(parts) } else { None }
    }

    fn post_h(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        let Some(parts) = &self.parts else { return };
        let mut p = parts.part(0);
        let (sheets, corrections) = (0..p.sheets.len(), 0..p.corrections.len());
        p.step_h(&mut Serial(f), sheets, corrections, None);
    }

    fn post_e(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        let Some(parts) = &self.parts else { return };
        let mut p = parts.part(0);
        let (solids, sheets) = (0..p.solids.len(), 0..p.sheets.len());
        p.step_e(&parts.ports, &mut Serial(f), solids, sheets, &[]);
    }

    fn energy(&self) -> f64 {
        let Some(parts) = &self.parts else { return 0.0 };
        let ports = &parts.ports;
        let held: Vec<_> = (0..parts.parts.len()).map(|s| parts.part(s)).collect();
        // (One running sum over all parts, in edge order: independent of
        // the partition.)
        let mut w = 0.0;
        let dt = self.dt;
        for p in &held {
            for e in &p.solids {
                let port = &ports[e.port as usize];
                let st = &p.state[e.st as usize..e.st as usize + port.stride()];
                w += 0.25 * e.two_c * dt * e.u * e.u + port.energy(st) / e.g;
            }
        }
        let mut w_corr = 0.0;
        for p in &held {
            for e in &p.sheets {
                for q in 0..2 {
                    w += 0.25 * e.two_c[q] * dt * e.u[q] * e.u[q];
                    let port = &ports[e.port[q] as usize];
                    let st = &p.state[e.st[q] as usize..e.st[q] as usize + port.stride()];
                    w += 0.5 * port.energy(st) / e.g;
                }
                // The engine's field sum has ½·C·(array value)².
                let face = e.u[0] + e.mix * (e.u[1] - e.u[0]);
                let stored = if e.gap == u32::MAX {
                    face
                } else {
                    let uo = p.gaps[e.gap as usize].u;
                    w += 0.5 * e.c_edge * uo * uo;
                    uo + face
                };
                w -= 0.5 * e.c_edge * stored * stored;
                w_corr += e.wc;
            }
        }
        w + w_corr
    }

    /// The magnetic energy of the sections: the field inside the
    /// conductor. A section's current closes through its own resistor,
    /// and the face voltage `r·i_L` it leaves drives the fields outside
    /// through `r` against their impedance, so it decays with the
    /// section's `1/a` whatever the fields do (the lowest corners lie a
    /// decade below the band).
    fn energy_held(&self) -> f64 {
        let Some(parts) = &self.parts else { return 0.0 };
        let ports = &parts.ports;
        let held: Vec<_> = (0..parts.parts.len()).map(|s| parts.part(s)).collect();
        // (In edge order, as `energy`.)
        let mut w = 0.0;
        for p in &held {
            for e in &p.solids {
                let port = &ports[e.port as usize];
                let st = &p.state[e.st as usize..e.st as usize + port.stride()];
                w += port.energy(st) / e.g;
            }
        }
        for p in &held {
            for e in &p.sheets {
                for q in 0..2 {
                    let port = &ports[e.port[q] as usize];
                    let st = &p.state[e.st[q] as usize..e.st[q] as usize + port.stride()];
                    w += 0.5 * port.energy(st) / e.g;
                }
            }
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CU: f64 = 5.8e7;

    /// The width of a rim patch: the half cell of a rim on the mesh line,
    /// growing with the reach, below the patch's own width (its current
    /// crowds at the rim), and continuous where the next edge comes in.
    #[test]
    fn rim_widths() {
        let l = 0.25e-3;
        assert_eq!(rim_width(0.0, l, l), l);
        let mut last = l;
        for q in 1..=40 {
            let reach = q as f64 * 0.05 * l;
            let w = rim_width(reach, l, l);
            assert!(w > last && w < reach + l, "{reach}: {w}");
            last = w;
        }
        // One cell of reach: the rim's half cell with (1/3)^0.28 = 0.735
        // of the current and a cell with the rest, 0.87 cells.
        let w = rim_width(2.0 * l, l, l);
        assert!((w / (2.0 * l) - 0.869).abs() < 2e-3, "{}", w / (2.0 * l));
        // A patch narrower than the half cell is the patch.
        assert_eq!(rim_width(0.1 * l, l, 0.0), 0.1 * l);
        // Two rims: two half cells for a cell, less than the patch beyond.
        assert_eq!(strip_width(2.0 * l, l), 2.0 * l);
        let w = strip_width(4.0 * l, l);
        assert!(w > 2.0 * l && w < 4.0 * l, "{w}");
    }

    #[test]
    fn nnls_small() {
        // The unconstrained optimum (1, −1) is clipped to (0.5, 0).
        let a = vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
        let x = nnls(&a, &[1.0, -1.0, 0.0]);
        assert!(x[1] == 0.0 && (x[0] - 0.5).abs() < 1e-9, "{x:?}");
    }

    #[test]
    fn roughness_factors() {
        let d = skin_depth(CU, 1e9);
        let hj = Roughness::HammerstadJensen { rms: 1e-6 };
        let want = 1.0 + 2.0 / core::f64::consts::PI * (1.4 * (1e-6 / d).powi(2)).atan();
        assert!((hj.factor(d) - want).abs() < 1e-12);
        let hu = Roughness::Huray {
            radius: 0.5e-6,
            ratio: 2.0,
        };
        // Low frequency: 1; high frequency: 1 + 1.5·ratio.
        assert!((hu.factor(1.0) - 1.0).abs() < 1e-9);
        assert!((hu.factor(1e-12) - 4.0).abs() < 1e-4);
        assert_eq!(Roughness::Smooth.factor(d), 1.0);
    }

    fn worst(fit: &ImpedanceFit, z: impl Fn(f64) -> Complex64, lo: f64, hi: f64) -> (f64, f64) {
        let (mut er, mut ex) = (0.0f64, 0.0f64);
        for i in 0..=200 {
            let f = lo * (hi / lo).powf(i as f64 / 200.0);
            let (a, b) = (fit.eval(f), z(f));
            er = er.max((a.re / b.re - 1.0).abs());
            ex = ex.max((a.im / b.im - 1.0).abs());
        }
        (er, ex)
    }

    #[test]
    fn half_space_fit() {
        for (lo, hi) in [(1e8, 1e10), (6e7, 6e9), (1e7, 2e10)] {
            let c = Conductor::new(CU);
            let fit = ImpedanceFit::half_space(&c, lo, hi).unwrap();
            let (er, ex) = worst(&fit, |f| half_space_impedance(&c, f), lo, hi);
            eprintln!(
                "{lo:e}..{hi:e}: {} sections, R {er:.4}, X {ex:.4}",
                fit.sections.len()
            );
            assert!(er < 0.006 && ex < 0.006, "{lo} {hi}: {er} {ex}");
            assert!(fit.sections.len() <= 9, "{}", fit.sections.len());
            assert!(fit.r0 >= 0.0 && fit.sections.iter().all(|s| s.0 > 0.0 && s.1 > 0.0));
        }
    }

    #[test]
    fn rough_fit_keeps_the_resistance() {
        for r in [
            Roughness::HammerstadJensen { rms: 1e-6 },
            Roughness::Huray {
                radius: 0.5e-6,
                ratio: 2.0,
            },
        ] {
            let c = Conductor::new(CU).with_roughness(r);
            let fit = ImpedanceFit::half_space(&c, 1e8, 1e10).unwrap();
            let (er, _) = worst(&fit, |f| half_space_impedance(&c, f), 1e8, 1e10);
            eprintln!("{r:?}: R {er:.4}");
            assert!(er < 0.03, "{r:?}: {er}");
            // The reactance is the causal one: at least the smooth one.
            for f in [1e8, 1e9, 1e10] {
                let (a, b) = (fit.eval(f), half_space_impedance(&c, f));
                assert!(a.im > 0.9 * b.im && a.im < 2.5 * b.im, "{r:?} {f}: {a} {b}");
            }
        }
    }

    #[test]
    fn sheet_fit_limits() {
        // Thin: Z_e = 2/(σt); thick: both the half space.
        let c = Conductor::new(1e4);
        let (ze, zo) = ImpedanceFit::sheet(&c, 1e-6, 1e8, 1e10).unwrap();
        let z = ze.eval(1e9);
        assert!((z.re / 200.0 - 1.0).abs() < 1e-3, "{z}");
        assert!(zo.eval(1e9).norm() < 1e-2);
        let c = Conductor::new(CU);
        let (ze, zo) = ImpedanceFit::sheet(&c, 35e-6, 1e8, 1e10).unwrap();
        for f in [1e8, 1e9, 1e10] {
            let (e, o) = sheet_mode_impedances(&c, 35e-6, f);
            for (a, b) in [(ze.eval(f), e), (zo.eval(f), o)] {
                assert!((a / b - 1.0).norm() < 0.02, "{f}: {a} {b}");
            }
        }
        // In between, both within 2 %.
        let c = Conductor::new(1e6);
        let (ze, zo) = ImpedanceFit::sheet(&c, 20e-6, 1.5e8, 1.5e10).unwrap();
        let e = worst(
            &ze,
            |f| sheet_mode_impedances(&c, 20e-6, f).0,
            1.5e8,
            1.5e10,
        );
        let o = worst(
            &zo,
            |f| sheet_mode_impedances(&c, 20e-6, f).1,
            1.5e8,
            1.5e10,
        );
        eprintln!(
            "sheet 1e6, 20 µm: even {e:?} ({} sections), odd {o:?} ({})",
            ze.sections.len(),
            zo.sections.len()
        );
        let (ze, zo) = ImpedanceFit::sheet(&Conductor::new(CU), 35e-6, 6e7, 6e9).unwrap();
        eprintln!(
            "copper 35 µm: {} and {} sections",
            ze.sections.len(),
            zo.sections.len()
        );
        assert!(e.0 < 0.02 && e.1 < 0.03, "{e:?}");
        assert!(o.0 < 0.03 && o.1 < 0.02, "{o:?}");
    }

    #[test]
    fn port_matches_the_fit() {
        // The trapezoidal one-port driven by a sine: its impedance at a
        // frequency well below 1/dt is the fit's.
        let c = Conductor::new(CU);
        let fit = ImpedanceFit::half_space(&c, 1e8, 1e10).unwrap();
        let dt = 1e-13;
        let port = Port::new(&fit, dt);
        let mut st = vec![0.0; port.stride()];
        let f = 2e9;
        let w = core::f64::consts::TAU * f;
        let (mut acc_x, mut acc_h) = (Complex64::ZERO, Complex64::ZERO);
        let steps = 40_000usize;
        for n in 0..steps {
            // Mean current over the step and the mean voltage it gives.
            let tm = (n as f64 + 0.5) * dt;
            let h0 = (w * n as f64 * dt).sin();
            let h1 = (w * (n as f64 + 1.0) * dt).sin();
            st[0] = h0;
            let hm = 0.5 * (h0 + h1);
            let x = port.z_inf * hm + port.history(&st);
            port.advance(&mut st, hm);
            if n >= steps / 2 {
                let ph = Complex64::from_polar(1.0, -w * tm);
                acc_x += ph * x;
                acc_h += ph * hm;
            }
        }
        let z = acc_x / acc_h;
        let want = fit.eval(f);
        assert!((z / want - 1.0).norm() < 2e-3, "{z} vs {want}");
    }
}
