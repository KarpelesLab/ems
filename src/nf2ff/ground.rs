//! Far fields over a planar stratified ground (design §5.10): the
//! asymptotic (stationary-phase) form of the layered-medium Green's
//! function.
//!
//! The equivalent currents on the box radiate in the presence of the
//! ground. Decomposing their field into plane waves, each wave meets the
//! stratified medium as a plane wave and is reflected or transmitted with
//! the transmission-line coefficients of its transverse wavenumber `k_t`
//! (TE: `Z = ωμ/k_z`, TM: `Z = k_z/(ωε)`, tangential E as voltage;
//! W. C. Chew, *Waves and Fields in Inhomogeneous Media*, ch. 2; L. B.
//! Felsen and N. Marcuvitz, *Radiation and Scattering of Waves*, ch. 5).
//! In the far zone only the wave travelling towards the observer survives
//! (stationary phase), which gives:
//!
//! * **On the box side** (`r̂·n̂ ≥ 0`, `n̂` the ground's normal towards the
//!   box), the direct field plus the specular reflection of the field
//!   radiated towards the mirror direction `r̂' = r̂ − 2(r̂·n̂)n̂`:
//!
//!   ```text
//!   E(r̂) = E_0(r̂) + e^{−2jk₁ cosθ h}·[Γ_TE (E'·ê) ê + Γ_TM M(E' − (E'·ê) ê)],
//!   E' = E_0(r̂'),   ê = n̂ × r̂ / |n̂ × r̂|,   M(v) = v − 2(v·n̂)n̂,
//!   ```
//!
//!   with `E_0` the free-space far field of the box currents (phase centre
//!   `c`), `h = (c − p₀)·n̂` the height of `c` over the interface and
//!   `Γ` the tangential-E reflection coefficients of the stack at
//!   `k_t = k₁ sin θ`. A PEC ground (`Γ = −1`) gives image theory.
//! * **Inside a lossless bottom half-space** (`r̂·n̂ < 0`, wavenumber
//!   `k_N`), the transmitted wave with `k_t = k_N·(r̂ − (r̂·n̂)n̂)`, which
//!   left the box medium along the (complex, beyond the critical angle)
//!   direction `d̂ = (k_t − k_z1 n̂)/k₁`, `k_z1 = √(k₁² − |k_t|²)`:
//!
//!   ```text
//!   E·ê   = (k_zN/k_z1)·T_TE·e^{φ}·(E_0(d̂)·ê)
//!   E·p̂_N = (k_N/k₁)·T_TM·e^{φ}·(E_0(d̂)·p̂_1),   p̂ = ê × (direction),
//!   φ = −j k_z1 h + j k_zN (h + D),
//!   ```
//!
//!   with `T` the tangential-E transmission through the stack (total
//!   thickness `D`). `E_0` at a complex direction is the analytic
//!   continuation of the radiation integrals (the evanescent part of the
//!   box currents' plane-wave spectrum, which a source close to the
//!   interface sends into a denser half-space beyond the critical angle).
//!
//! Both follow from the same plane-wave spectrum: the far field in either
//! medium is `2πj·k_z·Ê(k_t)·e^{−jkr}/r` of its own spectral amplitude.

use crate::axis::Axis;
use crate::constants::{C0, EPS0, MU0};
use crate::engine::Medium;
use crate::post::Complex64;
use crate::{Error, Result};

/// One layer of a [`Ground`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundLayer {
    /// Thickness \[m\].
    pub thickness: f64,
    /// Material (losses allowed).
    pub medium: Medium,
}

/// What lies beyond the last layer of a [`Ground`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GroundBase {
    /// A half-space of this medium (lossless: the far field is computed
    /// in it too; lossy: no far field there).
    HalfSpace(Medium),
    /// A perfect electric conductor.
    Pec,
    /// A perfect magnetic conductor.
    Pmc,
}

/// A planar stratified medium filling one side of the plane `axis =
/// position` (the side away from the NF2FF box): layers from the plane
/// outwards, then a [`GroundBase`]. See [`Nf2ffResult::with_ground`](super::Nf2ffResult::with_ground).
#[derive(Debug, Clone, PartialEq)]
pub struct Ground {
    /// Normal of the interface.
    pub axis: Axis,
    /// Interface coordinate \[m\] (top of the first layer).
    pub position: f64,
    /// Layers from the interface away from the box.
    pub layers: Vec<GroundLayer>,
    /// Beyond the layers.
    pub base: GroundBase,
}

impl Ground {
    /// A ground of `base` directly at `position` (no layers).
    pub fn new(axis: Axis, position: f64, base: GroundBase) -> Self {
        Self {
            axis,
            position,
            layers: Vec::new(),
            base,
        }
    }

    /// A homogeneous half-space of `medium` beyond `position`.
    pub fn half_space(axis: Axis, position: f64, medium: Medium) -> Self {
        Self::new(axis, position, GroundBase::HalfSpace(medium))
    }

    /// A dielectric substrate of `thickness` on a PEC ground plane.
    pub fn grounded_slab(axis: Axis, position: f64, thickness: f64, medium: Medium) -> Self {
        Self::new(axis, position, GroundBase::Pec).layer(thickness, medium)
    }

    /// Appends a layer (after the existing ones, away from the box).
    #[must_use]
    pub fn layer(mut self, thickness: f64, medium: Medium) -> Self {
        self.layers.push(GroundLayer { thickness, medium });
        self
    }

    /// Checks thicknesses and media.
    ///
    /// # Errors
    /// [`Error::Invalid`] for a non-finite position or a thickness `≤ 0`
    /// or an invalid medium.
    pub fn validate(&self) -> Result<()> {
        if !self.position.is_finite() {
            return Err(Error::Invalid(format!(
                "NF2FF ground position {}",
                self.position
            )));
        }
        for l in &self.layers {
            if !(l.thickness.is_finite() && l.thickness > 0.0) {
                return Err(Error::Invalid(format!(
                    "NF2FF ground layer thickness {}",
                    l.thickness
                )));
            }
            l.medium.validate()?;
        }
        if let GroundBase::HalfSpace(m) = self.base {
            m.validate()?;
        }
        Ok(())
    }

    /// Total thickness of the layers \[m\].
    pub fn thickness(&self) -> f64 {
        self.layers.iter().map(|l| l.thickness).sum()
    }

    /// The lossless bottom half-space, if the far field exists there.
    pub(super) fn far_base(&self) -> Option<Medium> {
        match self.base {
            GroundBase::HalfSpace(m) if m.is_lossless() => Some(m),
            _ => None,
        }
    }
}

/// Complex `ε`, `μ` of a medium at `ω`.
fn eps_mu(m: &Medium, w: f64) -> (Complex64, Complex64) {
    (
        Complex64::new(EPS0 * m.eps_r, -m.kappa / w),
        Complex64::new(MU0 * m.mu_r, -m.sigma_m / w),
    )
}

/// `√(k² − k_t²)` with `Im ≤ 0` (decaying or outgoing away from the box).
fn kz(k2: Complex64, kt: f64) -> Complex64 {
    let mut z = (k2 - kt * kt).sqrt();
    if z.im > 0.0 || (z.im == 0.0 && z.re < 0.0) {
        z = -z;
    }
    z
}

/// Reflection and transmission of the stack for one polarisation.
#[derive(Debug, Clone, Copy)]
pub(super) struct Coefficients {
    /// Tangential-E reflection at the interface.
    pub gamma: Complex64,
    /// Tangential-E transmission to the bottom of the stack (only with a
    /// half-space base).
    pub t: Complex64,
}

/// `[TE, TM]` coefficients of `g` at angular frequency `w` and transverse
/// wavenumber `kt`, seen from the box medium `m1`.
pub(super) fn coefficients(g: &Ground, m1: &Medium, w: f64, kt: f64) -> [Coefficients; 2] {
    let j = Complex64::I;
    let imp = |m: &Medium, te: bool| -> (Complex64, Complex64) {
        let (e, u) = eps_mu(m, w);
        let k2 = e * u * (w * w);
        let z = kz(k2, kt);
        (if te { u * w / z } else { z / (e * w) }, z)
    };
    [true, false].map(|te| {
        let (z1, _) = imp(m1, te);
        // Load impedance below the last layer (None: open, PMC).
        let mut zl: Option<Complex64> = match g.base {
            GroundBase::HalfSpace(m) => Some(imp(&m, te).0),
            GroundBase::Pec => Some(Complex64::ZERO),
            GroundBase::Pmc => None,
        };
        // V_bottom / V_top of each layer, bottom up.
        let mut ratio = Complex64::ONE;
        for l in g.layers.iter().rev() {
            let (zi, kzi) = imp(&l.medium, te);
            let x = kzi * l.thickness;
            // e^{−2jx} decays (Im x ≤ 0): stable tan and 1/cos.
            let q = (-(j * x) * 2.0).exp();
            let tan = -j * (1.0 - q) / (1.0 + q);
            let inv_cos = (-(j * x)).exp() * 2.0 / (1.0 + q);
            let (zin, vr) = match zl {
                // Short (PEC): Z_in = j·Z_i·tan, no voltage at the bottom.
                Some(zl) if zl == Complex64::ZERO => (j * zi * tan, Complex64::ZERO),
                Some(zl) => (
                    zi * (zl + j * zi * tan) / (zi + j * zl * tan),
                    inv_cos / (1.0 + j * (zi / zl) * tan),
                ),
                // Open load: Z_in = Z_i/(j tan), V_bot/V_top = 1/cos.
                None => (zi / (j * tan), inv_cos),
            };
            ratio *= vr;
            zl = Some(zin);
        }
        let gamma = match zl {
            Some(z) => (z - z1) / (z + z1),
            None => Complex64::ONE,
        };
        Coefficients {
            gamma,
            t: (1.0 + gamma) * ratio,
        }
    })
}

/// Wavenumber of a lossless medium at `w`.
pub(super) fn wavenumber(m: &Medium, w: f64) -> f64 {
    w * (m.eps_r * m.mu_r).sqrt() / C0
}

/// `k_z` in the box medium for transverse wavenumber `kt` (real or
/// `−j·α` beyond `k₁`).
pub(super) fn kz1(k1: f64, kt: f64) -> Complex64 {
    kz(Complex64::new(k1 * k1, 0.0), kt)
}
