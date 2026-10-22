//! Dispersive (Drude, Lorentz, Debye) media by the auxiliary differential
//! equation (ADE) method, as an engine operator.
//!
//! # Material model
//!
//! A dispersive edge carries, in addition to the static (high-frequency)
//! permittivity `ε∞` of its [`Coefficients`](super::Coefficients), a sum
//! of polarisation poles: with the engineering time convention `e^{jωt}`
//! (the one of [`crate::post::dft()`]: phasors are `∫x(t)·e^{−jωt}dt`),
//!
//! ```text
//! ε(ω) = ε0·( ε∞ + Σ_p χ_p(ω) )
//! ```
//!
//! with the susceptibilities of [`Pole`] (all frequencies in rad/s):
//!
//! ```text
//! Lorentz:  χ(ω) = ω_p² / (ω_0² − ω² + jωγ)
//! Drude:    χ(ω) = −ω_p² / (ω² − jωγ)          (Lorentz with ω_0 = 0)
//! Debye:    χ(ω) = Δε / (1 + jωτ)
//! ```
//!
//! Losses are `Im χ < 0` in this convention. Magnetic poles (`μ(ω)`) are
//! the same with `μ0`, `μ∞` (see [`Field::Magnetic`]). These are the forms
//! of Taflove & Hagness, *Computational Electrodynamics*, 3rd ed., ch. 9
//! (Debye, Lorentz and Drude media) and Gedney, *Introduction to the FDTD
//! Method*, ch. 8, written for `e^{jωt}`.
//!
//! # Polarisation current as a circuit branch (integral form)
//!
//! The engine works with edge voltages `u = E·l` and dual-edge currents
//! `i = H·l*` (engine module docs, design §5.2). Let `P_p` be the
//! polarisation of pole `p` and define its **edge charge and current**
//! `q_p = P_p·A*`, `j_p = dq_p/dt = J_p·A*` (`A*` the dual face pierced
//! by the edge, `l` its length). Ampère's law around the dual face becomes
//!
//! ```text
//! C·du/dt + G·u + Σ_p j_p = ∮i,        C = ε0·ε∞·A*/l
//! ```
//!
//! and each pole's ADE, multiplied by `A*`, is with `g = A*/l`,
//! `m = ε0·g`:
//!
//! ```text
//! Lorentz/Drude:  d²q/dt² + γ·dq/dt + ω_0²·q = m·ω_p²·u
//!   ⇔  L·dj/dt + R·j + q/C_p = u,   L = 1/(m·ω_p²), R = γ·L, 1/C_p = ω_0²·L
//! Debye:          τ·dq/dt + q = m·Δε·u
//!   ⇔  R·j + q/C_p = u,             C_p = m·Δε, R = τ/C_p, L = 0
//! ```
//!
//! so every pole is a **series R–L–C branch** across the edge, in parallel
//! with the edge capacitor `C` (a Drude pole has no capacitor, a Debye pole
//! no inductor). Its admittance `1/(jωL + R + 1/(jωC_p))` is exactly
//! `jω·ε0·χ_p(ω)·g`. This is the equivalent-circuit reading of the ADE
//! (the lumped-element FDTD of Piket-May, Taflove and Baron, IEEE MTT
//! 42(8), 1994, applied to polarisation currents).
//!
//! # Time discretisation (semi-implicit, trapezoidal)
//!
//! Each branch is integrated with the trapezoidal rule centred at
//! `t = (n + ½)·dt` — second-order accurate, the same centring as the
//! Yee E update — and *coupled implicitly* with the edge voltage, as in
//! the semi-implicit multi-term scheme of M. Okoniewski, M. Mrozowski and
//! M. A. Stuchly, "Simple treatment of multi-term dispersion in FDTD",
//! IEEE Microwave and Guided Wave Letters 7(5), 121–123, 1997 (there the
//! polarisation current is driven by `(E^{n+1} + E^n)/2`, too). With
//! `j̄ = (j^{n+1} + j^n)/2`:
//!
//! ```text
//! C·(u^{n+1} − u^n)/dt + G·(u^{n+1} + u^n)/2 + Σ_p j̄_p = (∮i)^{n+½}
//! L·(j^{n+1} − j^n)/dt + R·j̄ + (q^{n+1} + q^n)/(2C_p) = (u^{n+1} + u^n)/2
//! q^{n+1} = q^n + dt·j̄
//! ```
//!
//! The first line is `u^{n+1} = u* − cb·Σ_p j̄_p` with `u*` the plain
//! kernel result. Eliminating `j^{n+1}` and `q^{n+1}`, each branch gives
//! `j̄_p = (k_p + ½·u^{n+1})/z_p` with
//!
//! ```text
//! z_p = 2L/dt + R + dt/(2C_p),     k_p = (2L/dt)·j_p^n − q_p^n/C_p + ½·u^n,
//! ```
//!
//! so the coupled system is linear in `u^{n+1}` and is solved in closed
//! form per edge, for any number of poles:
//!
//! ```text
//! u^{n+1} = (u* − cb·Σ_p k_p/z_p) / (1 + ½·cb·Σ_p 1/z_p)
//! ```
//!
//! then `j̄_p`, `j_p^{n+1} = 2·j̄_p − j_p^n`, `q_p^{n+1} = q_p^n + dt·j̄_p`.
//! A Debye pole with `τ = 0` (`R = L = 0`) is a plain capacitor, a pole
//! with `γ = 0` is lossless; both are handled by the same formulas.
//! Compared with the fully explicit central-difference ADE of Taflove &
//! Hagness ch. 9 (polarisation at integer steps, `P^{n+1}` from
//! `P^n, P^{n−1}, E^n`), this scheme needs no extra stability condition
//! (below).
//!
//! **Discrete permittivity.** With `z = e^{jω·dt}` the branch equations
//! give `J = U/(jΩ_t·L + R + 1/(jΩ_t·C_p))` for the integer-time phasors,
//! `Ω_t = (2/dt)·tan(ω·dt/2)` (the bilinear map), and the E update reads
//! `jΩ·C·U + cos(ω·dt/2)·J = (∮I)` with the Yee operator
//! `Ω = (2/dt)·sin(ω·dt/2)`. Since `cos(ω·dt/2)/Ω = 1/Ω_t`, the scheme is
//! exactly the Yee scheme in a medium of permittivity
//!
//! ```text
//! ε_d(ω) = ε0·( ε∞ + Σ_p χ_p(Ω_t) )
//! ```
//!
//! i.e. the continuum `χ` evaluated at the warped frequency
//! `Ω_t = ω·(1 + (ω·dt)²/12 + …)` ([`Pole::discrete_susceptibility`]). The
//! validation tests compare against this exactly.
//!
//! **Stability and energy.** With the field invariant `W^n` (engine module
//! docs), the E update gives `W^{n+1} − W^n = −dt·Σ ū·j̄` per edge
//! (lossless background, `ū = (u^{n+1} + u^n)/2`), and the trapezoidal
//! branch update gives `W_p^{n+1} − W_p^n = dt·j̄·ū − dt·R·j̄²` for the
//! branch energy
//!
//! ```text
//! W_p^n = ½·L·(j_p^n)² + ½·(q_p^n)²/C_p     (≥ 0)
//! ```
//!
//! So `W + Σ W_p` is non-increasing (exactly conserved for lossless poles,
//! up to `f32` rounding of the fields): the poles add no stability limit
//! beyond the Courant limit of the `ε∞` medium. [`Operator::energy`]
//! reports `Σ W_p` of the electric poles at `t = n·dt` (plus the magnetic
//! poles, below), so [`Engine::energy`](super::Engine::energy) is the
//! conserved total.
//!
//! # Magnetic poles
//!
//! Dually, a magnetic pole adds a magnetic polarisation voltage `k_p`
//! (`= μ0·A·dM_p/dt`, `A` the primary face) to Faraday's law,
//! `L_H·di/dt + R_H·i + Σ_p k_p = −∮u`, with `k_p` obeying the same
//! branch equation driven by the dual-edge current `i`
//! (`m = μ0·A/l*`). The update is identical with `(u, ca, cb)` replaced by
//! `(i, da, db)`, centred at integer times (`i` from `(n − ½)·dt` to
//! `(n + ½)·dt`), and gives `μ_d(ω) = μ0·(μ∞ + Σ χ_p(Ω_t))`.
//!
//! **Magnetic energy.** The magnetic branch state lives at half-integer
//! times (`B^{n−½} = Σ ½·L·k² + ½·q²/C_p` after the H update that ends at
//! `(n − ½)·dt`). Write `k̄^n` for the branch average of the H step from
//! `(n − ½)·dt` to `(n + ½)·dt`, which lowers the dual-edge current by
//! `db·k̄^n` relative to the plain kernel result `i*^{n+½}`. The field
//! invariant with the corrected `i^{n+½}` changes by
//! `−½·dt·i^{n+½}·(k̄^n + k̄^{n+1})` per step, so the conserved total is
//! `W^n + B^{n+½} − ½·dt·k̄^n·i^{n+½}` (check by differencing with the
//! trapezoidal branch update). By the branch update this equals
//! `W^n + B^{n−½} + ½·dt·k̄^n·i^{n−½}`, and since
//! `½·L·i^{n−½}·(i*^{n+½} − i^{n+½}) = ½·dt·i^{n−½}·k̄^n` (lossless
//! dual edge, `db = dt/L`), it is exactly
//!
//! ```text
//! W*^n + B^{n−½},     W*^n = ½·Σ C·(u^n)² + ½·Σ L·i^{n−½}·i*^{n+½}
//! ```
//!
//! `W*^n` is what [`Engine::energy`](super::Engine::energy) and
//! [`Engine::step_with_energy`](super::Engine::step_with_energy) evaluate:
//! both take `i^{n+½}` from the H kernel *before* the slab hooks, i.e.
//! without the magnetic correction. So [`Operator::energy`] reports the
//! electric poles at `n·dt` plus the magnetic poles at `(n − ½)·dt`
//! ([`magnetic_energy`](Dispersive::magnetic_energy)), and the engine
//! total is conserved for lossless magnetic poles too (for a lossy
//! background `da < 1` the identity holds up to a term of the order of
//! the loss, which only dissipates).

//! # Implementation
//!
//! The operator runs as [`SlabHooks`] on the worker threads: edges are
//! partitioned by x-plane like the fields, each slab keeps its own state
//! behind an (uncontended) mutex, and every edge is updated by the same
//! expressions whatever the partition, so results are bit-identical for
//! any thread count. Per step:
//!
//! * `post_h_slab`: correct the magnetic-pole dual edges (`i^{n+½}`), then
//!   save `u^n` of the electric-pole edges (the voltage the E kernel
//!   starts from, after all `post_e` / `pre_h` operators);
//! * `post_e_slab`: correct the electric-pole edges (`u^{n+1}`), then save
//!   `i^{n+½}` of the magnetic-pole dual edges.
//!
//! The initial `i^{−½}` of magnetic-pole edges is taken as 0 (it is only
//! saved from the first E half-step on). Edge coefficients `cb` / `db` are
//! read every step, so lumped elements folded into the edge are honoured.
//!
//! **Ordering with other operators.** The correction treats the field as
//! it finds it after the kernel and the slab hooks registered *before* it
//! as `u*`. Register it after the CPML (as [`crate::compile`] does), so
//! that a dispersive medium extending into the PML is coupled implicitly
//! with the CPML term too. Serial `post_e` operators (sources, lumped
//! branches) act after it and are not part of the implicit solve.
//!
//! Edges with `cb = 0` (PEC, padding, PEC outer faces) or, for magnetic
//! poles, `db = 0` at installation are skipped (counted by
//! [`skipped`](Dispersive::skipped)).

use core::ops::Range;

use std::sync::{Arc, Mutex, PoisonError};

use super::Engine;
use super::grid::Grid;
use super::operator::{Hooks, Operator, OperatorHandle};
use super::slab::{Fusion, SlabContext, SlabHooks, SlabMut, SlabPartition, SlabRef};
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};
use crate::post::Complex64;
use crate::{Error, Result};

/// One polarisation pole of a dispersive medium (see the
/// [module docs](self) for the model and time convention). Frequencies in
/// rad/s, times in s; the susceptibility is relative (multiplies `ε0` or
/// `μ0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pole {
    /// `χ(ω) = ω_p² / (ω_0² − ω² + jωγ)`.
    Lorentz {
        /// Plasma frequency `ω_p` \[rad/s\] (pole strength
        /// `Δε = ω_p²/ω_0²`).
        omega_p: f64,
        /// Resonance frequency `ω_0` \[rad/s\].
        omega_0: f64,
        /// Damping `γ` \[1/s\] (full width of the absorption line).
        gamma: f64,
    },
    /// `χ(ω) = −ω_p² / (ω² − jωγ)`: a Lorentz pole with `ω_0 = 0`.
    Drude {
        /// Plasma frequency `ω_p` \[rad/s\].
        omega_p: f64,
        /// Collision frequency `γ` \[1/s\] (DC conductivity
        /// `ε0·ω_p²/γ`).
        gamma: f64,
    },
    /// `χ(ω) = Δε / (1 + jωτ)`.
    Debye {
        /// Permittivity step `Δε = ε_s − ε∞`.
        delta_eps: f64,
        /// Relaxation time `τ` \[s\].
        tau: f64,
    },
}

impl Pole {
    /// Lorentz pole `ω_p² / (ω_0² − ω² + jωγ)`.
    pub const fn lorentz(omega_p: f64, omega_0: f64, gamma: f64) -> Self {
        Pole::Lorentz {
            omega_p,
            omega_0,
            gamma,
        }
    }

    /// Drude pole `−ω_p² / (ω² − jωγ)`.
    pub const fn drude(omega_p: f64, gamma: f64) -> Self {
        Pole::Drude { omega_p, gamma }
    }

    /// Debye pole `Δε / (1 + jωτ)`.
    pub const fn debye(delta_eps: f64, tau: f64) -> Self {
        Pole::Debye { delta_eps, tau }
    }

    /// Relative susceptibility `χ(ω)` at angular frequency `omega`
    /// (`e^{jωt}` convention: losses have `Im χ < 0`).
    pub fn susceptibility(&self, omega: f64) -> Complex64 {
        let j = Complex64::I;
        match *self {
            Pole::Lorentz {
                omega_p,
                omega_0,
                gamma,
            } => omega_p * omega_p / (omega_0 * omega_0 - omega * omega + j * (omega * gamma)),
            Pole::Drude { omega_p, gamma } => {
                -(omega_p * omega_p) / (omega * omega - j * (omega * gamma))
            }
            Pole::Debye { delta_eps, tau } => delta_eps / (1.0 + j * (omega * tau)),
        }
    }

    /// The susceptibility the time-stepping scheme actually realises at
    /// frequency `f` \[Hz\] with time step `dt`: `χ(Ω_t)`,
    /// `Ω_t = (2/dt)·tan(π·f·dt)` (see [Discrete permittivity](self#time-discretisation-semi-implicit-trapezoidal)).
    /// Together with the Yee operator `Ω = (2/dt)·sin(π·f·dt)` in place of
    /// `ω`, the scheme is exactly a Yee scheme with `ε∞ + Σ χ(Ω_t)`.
    pub fn discrete_susceptibility(&self, f: f64, dt: f64) -> Complex64 {
        self.susceptibility(2.0 / dt * (core::f64::consts::PI * f * dt).tan())
    }

    /// The pole with its susceptibility multiplied by `s ≥ 0` (`ω_p` by
    /// `√s`, `Δε` by `s`): e.g. the share of a dual face filled with the
    /// medium.
    pub fn scaled(self, s: f64) -> Self {
        match self {
            Pole::Lorentz {
                omega_p,
                omega_0,
                gamma,
            } => Pole::Lorentz {
                omega_p: omega_p * s.sqrt(),
                omega_0,
                gamma,
            },
            Pole::Drude { omega_p, gamma } => Pole::Drude {
                omega_p: omega_p * s.sqrt(),
                gamma,
            },
            Pole::Debye { delta_eps, tau } => Pole::Debye {
                delta_eps: delta_eps * s,
                tau,
            },
        }
    }

    /// Whether the pole has zero strength (`ω_p = 0` or `Δε = 0`) and so
    /// does nothing.
    pub fn is_null(&self) -> bool {
        match *self {
            Pole::Lorentz { omega_p, .. } | Pole::Drude { omega_p, .. } => omega_p == 0.0,
            Pole::Debye { delta_eps, .. } => delta_eps == 0.0,
        }
    }

    /// Checks that all parameters are finite and `≥ 0` (a passive pole).
    pub fn validate(&self) -> Result<()> {
        let vals: &[(&str, f64)] = match self {
            Pole::Lorentz {
                omega_p,
                omega_0,
                gamma,
            } => &[("ω_p", *omega_p), ("ω_0", *omega_0), ("γ", *gamma)],
            Pole::Drude { omega_p, gamma } => &[("ω_p", *omega_p), ("γ", *gamma)],
            Pole::Debye { delta_eps, tau } => &[("Δε", *delta_eps), ("τ", *tau)],
        };
        for &(name, v) in vals {
            if !(v.is_finite() && v >= 0.0) {
                return Err(Error::Invalid(format!(
                    "dispersive pole {self:?}: {name} = {v} must be finite and >= 0"
                )));
            }
        }
        Ok(())
    }

    /// Branch `(2L/dt, 1/C_p, z)` for `m = X0·g` (module docs).
    fn branch(&self, m: f64, dt: f64) -> (f64, f64, f64) {
        let (l, r, inv_c) = match *self {
            Pole::Lorentz {
                omega_p,
                omega_0,
                gamma,
            } => {
                let l = 1.0 / (m * omega_p * omega_p);
                (l, gamma * l, omega_0 * omega_0 * l)
            }
            Pole::Drude { omega_p, gamma } => {
                let l = 1.0 / (m * omega_p * omega_p);
                (l, gamma * l, 0.0)
            }
            Pole::Debye { delta_eps, tau } => {
                let c = m * delta_eps;
                (0.0, tau / c, 1.0 / c)
            }
        };
        let a = 2.0 * l / dt;
        (a, inv_c, a + r + 0.5 * dt * inv_c)
    }

    /// Branch of the explicit leapfrog scheme ([`Scheme::Leapfrog`]) for
    /// `m = X0·g`: `(k_a, k_b, 1/C_p, ½·L)` with
    /// `j^{n+½} = k_a·j^{n−½} + k_b·(u^n − q^n/C_p)`. `None` for a pole
    /// without inductance (Debye), which keeps the trapezoidal update.
    fn leapfrog(&self, m: f64, dt: f64) -> Option<(f64, f64, f64, f64)> {
        let (l, r, inv_c) = match *self {
            Pole::Lorentz {
                omega_p,
                omega_0,
                gamma,
            } => {
                let l = 1.0 / (m * omega_p * omega_p);
                (l, gamma * l, omega_0 * omega_0 * l)
            }
            Pole::Drude { omega_p, gamma } => {
                let l = 1.0 / (m * omega_p * omega_p);
                (l, gamma * l, 0.0)
            }
            Pole::Debye { .. } => return None,
        };
        let d = l / dt + 0.5 * r;
        Some(((l / dt - 0.5 * r) / d, 1.0 / d, inv_c, 0.5 * l))
    }

    /// The susceptibility the explicit leapfrog scheme
    /// ([`Scheme::Leapfrog`]) realises at frequency `f` \[Hz\] with time
    /// step `dt`, for a Lorentz or Drude pole (Debye poles are always
    /// integrated trapezoidally, see [`discrete_susceptibility`](Self::discrete_susceptibility)).
    ///
    /// With `z = e^{jω·dt}`, the branch recursion gives the admittance
    /// `1/(jΩ·L + cos(ω·dt/2)·R + 1/(jΩ·C_p))` between the half-step
    /// current and the integer-time voltage, so (as for
    /// [`discrete_susceptibility`](Self::discrete_susceptibility)) the
    /// scheme is the Yee scheme with
    /// `χ_d = ω_p² / (ω_0² − Ω² + j·Ω·γ·cos(ω·dt/2))`,
    /// `Ω = (2/dt)·sin(ω·dt/2)`: the continuum `χ` at the Yee frequency
    /// `Ω`, which differs from `χ(ω)` by `O((ω·dt)²)` (and from the
    /// trapezoidal `χ(Ω_t)` by about twice that).
    pub fn leapfrog_susceptibility(&self, f: f64, dt: f64) -> Complex64 {
        let w = 2.0 * core::f64::consts::PI * f;
        let c = (0.5 * w * dt).cos();
        let om = 2.0 / dt * (0.5 * w * dt).sin();
        let j = Complex64::I;
        match *self {
            Pole::Lorentz {
                omega_p,
                omega_0,
                gamma,
            } => omega_p * omega_p / (omega_0 * omega_0 - om * om + j * (om * gamma * c)),
            Pole::Drude { omega_p, gamma } => {
                omega_p * omega_p / (-(om * om) + j * (om * gamma * c))
            }
            Pole::Debye { .. } => self.discrete_susceptibility(f, dt),
        }
    }
}

/// Time integration of the Lorentz and Drude poles (Debye poles are always
/// integrated with the trapezoidal rule, which is also what openEMS
/// does: identical probe signals to `1e-6`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scheme {
    /// Semi-implicit trapezoidal rule coupled with the E update (module
    /// docs): unconditionally stable, exact discrete energy balance,
    /// `χ(Ω_t)` with the bilinear frequency map (default of the engine).
    #[default]
    Trapezoidal,
    /// Explicit leapfrog, the scheme openEMS uses (identified from its
    /// probe output to `1e-6`, see design §5.16): the branch current lives
    /// at half steps and is driven by the start voltage,
    ///
    /// ```text
    /// L·(j^{n+½} − j^{n−½})/dt + R·(j^{n+½} + j^{n−½})/2 + q^n/C_p = u^n
    /// q^{n+1} = q^n + dt·j^{n+½},      u^{n+1} = u* − cb·Σ_p j_p^{n+½}
    /// ```
    ///
    /// (for magnetic poles the same one half step earlier: the branch
    /// voltage at `n·dt` is driven by `i^{n−½}`). This is the
    /// polarisation-current ADE of T. Kashiwa and I. Fukai, *Microwave
    /// Opt. Technol. Lett.* 3(6), 203–205, 1990 (see also Taflove &
    /// Hagness ch. 9) written for edge quantities. Its susceptibility is
    /// [`Pole::leapfrog_susceptibility`]. Being explicit, it adds a
    /// stability limit (roughly `ω_0·dt < 2` and `ω_p·dt/√ε∞` well below
    /// 2, combined with the Courant limit), and the reported pole energy
    /// `½·L·j² + ½·q²/C_p` mixes time levels, so the engine energy is not
    /// an exact invariant with these poles.
    Leapfrog,
}

/// Which field a pole acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Field {
    /// Electric pole on a primary (E) edge: `ε(ω)`.
    Electric,
    /// Magnetic pole on a dual (H) edge: `μ(ω)`; the edge of axis `a` at
    /// node `n` is `i[a]` at `n` (engine layout).
    Magnetic,
}

/// One pole on one edge, as collected by [`Dispersive::push`] (and
/// produced by the compile step from csx dispersive materials).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgePole {
    /// Electric (E edge) or magnetic (H dual edge) pole.
    pub field: Field,
    /// Direction of the edge.
    pub axis: Axis,
    /// Start node of the E edge / node of the dual edge (engine layout).
    pub node: [usize; 3],
    /// The pole, with the edge's effective strength.
    pub pole: Pole,
}

/// Branch state of all edges of one field type in one slab (structure of
/// arrays, edges in global order, poles in CSR layout).
#[derive(Debug, Default, Clone)]
struct Side {
    /// Axis index of each edge.
    axis: Vec<u8>,
    /// Global x-plane of each edge.
    plane: Vec<u32>,
    /// Offset `j·nz + k` in the plane.
    off: Vec<u32>,
    /// Saved field value at the start of the update (`u^n` / `i^{n−½}`).
    prev: Vec<f64>,
    /// `Σ_p 1/z_p` per edge (trapezoidal poles only).
    sum_inv_z: Vec<f64>,
    /// End of each edge's poles in the pole arrays.
    pole_end: Vec<u32>,
    /// `2L/dt` per pole (leapfrog poles: `k_a`).
    a: Vec<f64>,
    /// `1/C_p` per pole.
    inv_c: Vec<f64>,
    /// `1/z_p` per pole (leapfrog poles: `k_b`).
    inv_z: Vec<f64>,
    /// `½·L` per pole (energy).
    half_l: Vec<f64>,
    /// Whether the pole uses the explicit leapfrog update.
    explicit: Vec<bool>,
    /// Branch current `j^n` (magnetic: voltage) per pole.
    j: Vec<f64>,
    /// Branch charge `q^n` per pole.
    q: Vec<f64>,
}

impl Side {
    fn edges(&self) -> usize {
        self.axis.len()
    }

    /// The edges in the x-planes `planes` (edges are in plane order).
    fn edges_in(&self, planes: &Range<usize>) -> Range<usize> {
        let lo = self.plane.partition_point(|&p| (p as usize) < planes.start);
        let hi = self.plane.partition_point(|&p| (p as usize) < planes.end);
        lo..hi
    }

    /// Saves the current field values of the edges `edges`.
    fn capture(&mut self, f: &SlabRef<'_>, edges: Range<usize>) {
        for e in edges {
            let axis = Axis::ALL[self.axis[e] as usize];
            self.prev[e] = f.plane(axis, self.plane[e] as usize)[self.off[e] as usize] as f64;
        }
    }

    /// The implicit correction of every edge (module docs); `coef` is
    /// `cb` (electric) or `db` (magnetic) per axis; only the edges
    /// `edges`.
    fn correct(&mut self, f: &mut SlabMut<'_>, coef: [&[f32]; 3], dt: f64, edges: Range<usize>) {
        let p = f.nodes()[1] * f.nodes()[2];
        let mut p0 = edges
            .start
            .checked_sub(1)
            .map_or(0, |e| self.pole_end[e] as usize);
        for e in edges {
            let p1 = self.pole_end[e] as usize;
            let ax = self.axis[e] as usize;
            let gi = self.plane[e] as usize;
            let off = self.off[e] as usize;
            let cb = coef[ax][gi * p + off] as f64;
            let half_prev = 0.5 * self.prev[e];
            let plane = f.plane_mut(Axis::ALL[ax], gi);
            let u_star = plane[off] as f64;
            let mut s = 0.0;
            for q in p0..p1 {
                if self.explicit[q] {
                    // Leapfrog: j^{n+½} from u^n, then q^{n+1}.
                    let j = self.a[q] * self.j[q]
                        + self.inv_z[q] * (self.prev[e] - self.q[q] * self.inv_c[q]);
                    self.j[q] = j;
                    self.q[q] += dt * j;
                    s += j;
                } else {
                    s += (self.a[q] * self.j[q] - self.q[q] * self.inv_c[q] + half_prev)
                        * self.inv_z[q];
                }
            }
            let u_new = ((u_star - cb * s) / (1.0 + 0.5 * cb * self.sum_inv_z[e])) as f32;
            plane[off] = u_new;
            let half_new = 0.5 * u_new as f64;
            for q in p0..p1 {
                if self.explicit[q] {
                    continue;
                }
                let k = self.a[q] * self.j[q] - self.q[q] * self.inv_c[q] + half_prev;
                let jb = (k + half_new) * self.inv_z[q];
                self.j[q] = 2.0 * jb - self.j[q];
                self.q[q] += dt * jb;
            }
            p0 = p1;
        }
    }

    /// Adds `Σ ½·L·j² + ½·q²/C_p` to `w`, in pole order (one running
    /// sum over all slabs keeps the order independent of the partition).
    fn add_energy(&self, w: &mut f64) {
        for q in 0..self.a.len() {
            *w += self.half_l[q] * self.j[q] * self.j[q]
                + 0.5 * self.inv_c[q] * self.q[q] * self.q[q];
        }
    }
}

/// State of one slab.
#[derive(Debug, Default, Clone)]
struct SlabState {
    e: Side,
    h: Side,
}

/// One edge with its branches, before partitioning.
#[derive(Debug, Clone)]
struct EdgeBranches {
    field: Field,
    axis: Axis,
    node: [usize; 3],
    /// Per pole: `(2L/dt, 1/C_p, 1/z, ½·L, false)` (trapezoidal) or
    /// `(k_a, 1/C_p, k_b, ½·L, true)` (leapfrog).
    poles: Vec<(f64, f64, f64, f64, bool)>,
}

/// The slab hooks: per-slab state.
#[derive(Debug)]
struct Parts {
    dt: f64,
    slabs: Vec<Mutex<SlabState>>,
}

impl Parts {
    fn slab(&self, s: usize) -> std::sync::MutexGuard<'_, SlabState> {
        self.slabs[s].lock().unwrap_or_else(PoisonError::into_inner)
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
        let mut st = self.slab(ctx.slab);
        let co = ctx.coefficients;
        let r = st.h.edges_in(&planes);
        if !r.is_empty() {
            st.h.correct(
                i,
                [co.db(Axis::X), co.db(Axis::Y), co.db(Axis::Z)],
                self.dt,
                r,
            );
        }
        let r = st.e.edges_in(&planes);
        st.e.capture(u, r);
    }

    fn post_e_planes(
        &self,
        u: &mut SlabMut<'_>,
        i: &SlabRef<'_>,
        ctx: &SlabContext<'_>,
        planes: Range<usize>,
    ) {
        let mut st = self.slab(ctx.slab);
        let co = ctx.coefficients;
        let r = st.e.edges_in(&planes);
        if !r.is_empty() {
            st.e.correct(
                u,
                [co.cb(Axis::X), co.cb(Axis::Y), co.cb(Axis::Z)],
                self.dt,
                r,
            );
        }
        let r = st.h.edges_in(&planes);
        st.h.capture(i, r);
    }
}

/// A dispersive medium: Drude / Lorentz / Debye poles on a set of E edges
/// (and optionally H dual edges), updated by the ADE method; see the
/// [module documentation](self) for the model, the scheme and the
/// conventions.
///
/// Collect poles with [`push`](Self::push) or
/// [`fill_cells`](Self::fill_cells), then install with
/// [`add_to`](Self::add_to). The static part `ε∞` (`μ∞`) belongs in the
/// engine's material map; the poles only add `χ(ω)`.
///
/// ```
/// use ems::engine::{dispersive::{Dispersive, Field, Pole}, CellMaterials, Engine, Grid, Material};
///
/// let g = Grid::uniform([8, 8, 8], [1e-3; 3]).unwrap();
/// let mut m = CellMaterials::new(&g, Material::VACUUM);
/// m.fill_box([2..6, 2..6, 2..6], Material::dielectric(2.0)); // ε∞
/// let mut e = Engine::builder(g.clone()).materials(m).threads(1).build().unwrap();
/// let mut d = Dispersive::new();
/// d.fill_cells(&g, [2..6, 2..6, 2..6], Field::Electric, &[Pole::debye(3.0, 20e-12)]).unwrap();
/// let h = d.add_to(&mut e).unwrap();
/// e.run(10);
/// assert!(e.operator(&h).edges() > 0);
/// ```
#[derive(Debug, Default)]
pub struct Dispersive {
    entries: Vec<EdgePole>,
    scheme: Scheme,
    /// Built at installation, partitioned at registration.
    built: Vec<EdgeBranches>,
    nodes: [usize; 3],
    dt: f64,
    skipped: usize,
    installed: bool,
    parts: Option<Arc<Parts>>,
}

impl Dispersive {
    /// An empty medium (trapezoidal scheme, [`Scheme::Trapezoidal`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the time integration of the Lorentz and Drude poles (default
    /// [`Scheme::Trapezoidal`]; Debye poles are always trapezoidal).
    #[must_use]
    pub fn with_scheme(mut self, scheme: Scheme) -> Self {
        self.scheme = scheme;
        self
    }

    /// The time integration of the Lorentz and Drude poles.
    pub fn scheme(&self) -> Scheme {
        self.scheme
    }

    /// Adds `pole` to the edge of `field` type along `axis` at `node`
    /// (E edge `u[axis]` / dual edge `i[axis]` at that node). The pole's
    /// strength is the effective one of the edge (average over its dual
    /// face, resp. dual edge; see [`Pole::scaled`]). Several poles on the
    /// same edge add up (solved together implicitly). Null poles
    /// ([`Pole::is_null`]) are ignored.
    pub fn push(&mut self, field: Field, axis: Axis, node: [usize; 3], pole: Pole) {
        if !pole.is_null() {
            self.entries.push(EdgePole {
                field,
                axis,
                node,
                pole,
            });
        }
    }

    /// Fills the cell box `cells` (cell index ranges per axis) with
    /// `poles` acting on `field` along all three axes (isotropic).
    ///
    /// Every edge touching the box gets the poles scaled by the share of
    /// its averaging volume inside the box, with the same weights the
    /// engine uses for `ε` / `μ` ([`EdgeMaterials`](super::EdgeMaterials)):
    /// E edges by the area of the (in-domain) quarter cells of their dual
    /// face, H dual edges by the length of their (in-domain) half cells
    /// (for H this linear average is exact only on homogeneous dual edges;
    /// the static `μ` uses the series combination). So a box filled with
    /// `ε∞` in [`CellMaterials`](super::CellMaterials) and with the poles
    /// here is the medium `ε∞ + Σχ` with consistent interface averaging.
    ///
    /// # Errors
    /// [`Error::Invalid`] for an invalid pole or a box outside the grid.
    pub fn fill_cells(
        &mut self,
        grid: &Grid,
        cells: [core::ops::Range<usize>; 3],
        field: Field,
        poles: &[Pole],
    ) -> Result<()> {
        for p in poles {
            p.validate()?;
        }
        let nc = grid.cells();
        if (0..3).any(|a| cells[a].end > nc[a] || cells[a].start > cells[a].end) {
            return Err(Error::Invalid(format!(
                "dispersive cell box {cells:?} does not fit the grid ({nc:?} cells)"
            )));
        }
        if cells.iter().any(|r| r.is_empty()) {
            return Ok(());
        }
        let n = grid.nodes();
        let inside = |a: usize, c: usize| cells[a].contains(&c);
        for axis in Axis::ALL {
            let (ab, ac) = axis.others();
            let (a, b, c) = (axis.index(), ab.index(), ac.index());
            // Node ranges that can touch the box.
            let range = |t: usize, along: bool| {
                if along {
                    cells[t].start..cells[t].end
                } else {
                    cells[t].start..(cells[t].end + 1).min(n[t])
                }
            };
            let (ra, rb, rc) = match field {
                Field::Electric => (range(a, true), range(b, false), range(c, false)),
                Field::Magnetic => (range(a, false), range(b, true), range(c, true)),
            };
            for pa in ra {
                for pb in rb.clone() {
                    for pc in rc.clone() {
                        let mut node = [0; 3];
                        node[a] = pa;
                        node[b] = pb;
                        node[c] = pc;
                        let share = match field {
                            Field::Electric => {
                                // Quarter cells (cb, cc) of the dual face.
                                let (mut w_in, mut w) = (0.0, 0.0);
                                for cb in [pb.wrapping_sub(1), pb] {
                                    if cb >= n[b] - 1 {
                                        continue;
                                    }
                                    for cc in [pc.wrapping_sub(1), pc] {
                                        if cc >= n[c] - 1 {
                                            continue;
                                        }
                                        let area = grid.delta(ab, cb) * grid.delta(ac, cc);
                                        w += area;
                                        if inside(b, cb) && inside(c, cc) {
                                            w_in += area;
                                        }
                                    }
                                }
                                if w > 0.0 { w_in / w } else { 0.0 }
                            }
                            Field::Magnetic => {
                                let (mut w_in, mut w) = (0.0, 0.0);
                                for ca in [pa.wrapping_sub(1), pa] {
                                    if ca >= n[a] - 1 {
                                        continue;
                                    }
                                    let len = grid.delta(axis, ca);
                                    w += len;
                                    if inside(a, ca) {
                                        w_in += len;
                                    }
                                }
                                if w > 0.0 { w_in / w } else { 0.0 }
                            }
                        };
                        if share > 0.0 {
                            for &p in poles {
                                self.push(field, axis, node, p.scaled(share));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Adds a pole entry (see [`push`](Self::push)).
    pub fn push_entry(&mut self, e: EdgePole) {
        self.push(e.field, e.axis, e.node, e.pole);
    }

    /// The requested (non-null) pole entries, in insertion order.
    pub fn entries(&self) -> &[EdgePole] {
        &self.entries
    }

    /// Number of requested (non-null) pole entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no pole was requested.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of distinct edges carrying poles after installation.
    pub fn edges(&self) -> usize {
        match &self.parts {
            Some(p) => (0..p.slabs.len())
                .map(|s| {
                    let st = p.slab(s);
                    st.e.edges() + st.h.edges()
                })
                .sum(),
            None => self.built.len(),
        }
    }

    /// Number of requested edges skipped at installation because they are
    /// PEC / padding (`cb = 0`, resp. `db = 0`).
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Computes the branch parameters for `engine`'s grid and time step;
    /// the engine is not changed.
    fn install(&mut self, engine: &Engine) -> Result<()> {
        if self.installed {
            return Err(Error::Invalid("dispersive medium already installed".into()));
        }
        let grid = engine.grid();
        let n = grid.nodes();
        let dt = engine.dt();
        for e in &self.entries {
            e.pole.validate()?;
            let a = e.axis.index();
            let (b, c) = e.axis.others();
            let (b, c) = (b.index(), c.index());
            let exists = (0..3).all(|t| e.node[t] < n[t])
                && match e.field {
                    Field::Electric => e.node[a] + 1 < n[a],
                    Field::Magnetic => e.node[b] + 1 < n[b] && e.node[c] + 1 < n[c],
                };
            if !exists {
                return Err(Error::Invalid(format!(
                    "dispersive {:?} edge {}:{:?} is not in the grid ({n:?} nodes)",
                    e.field, e.axis, e.node
                )));
            }
        }
        let mut entries = self.entries.clone();
        let key = |e: &EdgePole| (e.node[0], e.field, e.axis.index(), e.node[1], e.node[2]);
        entries.sort_by_key(key);
        let co = engine.coefficients();
        let mut built: Vec<EdgeBranches> = Vec::new();
        let mut skipped = 0;
        let mut last = None;
        for e in &entries {
            let k = key(e);
            let idx = grid.index(e.node);
            let (ab, ac) = e.axis.others();
            // `m = X0·g`, `g` the dual face area over the edge length
            // (cylindrical grids: the physical lengths and areas).
            let (coef, m) = match e.field {
                Field::Electric if grid.is_cylindrical() => (
                    co.cb(e.axis)[idx],
                    EPS0 * grid.dual_face_area(e.axis, e.node) / grid.edge_length(e.axis, e.node),
                ),
                Field::Electric => (
                    co.cb(e.axis)[idx],
                    EPS0 * grid.dual_delta(ab, e.node[ab.index()])
                        * grid.dual_delta(ac, e.node[ac.index()])
                        / grid.delta(e.axis, e.node[e.axis.index()]),
                ),
                Field::Magnetic if grid.is_cylindrical() => (
                    co.db(e.axis)[idx],
                    MU0 * grid.face_area(e.axis, e.node) / grid.dual_edge_length(e.axis, e.node),
                ),
                Field::Magnetic => (
                    co.db(e.axis)[idx],
                    MU0 * grid.delta(ab, e.node[ab.index()]) * grid.delta(ac, e.node[ac.index()])
                        / grid.dual_delta(e.axis, e.node[e.axis.index()]),
                ),
            };
            let new_edge = last != Some(k);
            last = Some(k);
            if coef == 0.0 {
                if new_edge {
                    skipped += 1;
                }
                continue;
            }
            if new_edge {
                built.push(EdgeBranches {
                    field: e.field,
                    axis: e.axis,
                    node: e.node,
                    poles: Vec::new(),
                });
            }
            let leap = match self.scheme {
                Scheme::Leapfrog => e.pole.leapfrog(m, dt),
                Scheme::Trapezoidal => None,
            };
            let branch = match leap {
                Some((ka, kb, inv_c, half_l)) => (ka, inv_c, kb, half_l, true),
                None => {
                    let (a, inv_c, z) = e.pole.branch(m, dt);
                    (a, inv_c, 1.0 / z, 0.25 * a * dt, false)
                }
            };
            built
                .last_mut()
                .expect("edge just pushed")
                .poles
                .push(branch);
        }
        drop(co);
        self.built = built;
        self.skipped = skipped;
        self.nodes = n;
        self.dt = dt;
        self.installed = true;
        Ok(())
    }

    /// Validates the poles and edges, computes the branch parameters for
    /// `engine` and registers the operator (no coefficient changes).
    ///
    /// Register it after the CPML, if any (see the
    /// [module docs](self#implementation)).
    ///
    /// # Errors
    /// [`Error::Invalid`] for an invalid pole, an edge outside the grid or
    /// a second installation; the engine is unchanged then.
    pub fn add_to(mut self, engine: &mut Engine) -> Result<OperatorHandle<Dispersive>> {
        self.install(engine)?;
        Ok(engine.add_operator(self))
    }

    fn sum_energy(&self, field: Field) -> f64 {
        let Some(p) = &self.parts else { return 0.0 };
        let mut w = 0.0;
        for s in 0..p.slabs.len() {
            let st = p.slab(s);
            match field {
                Field::Electric => st.e.add_energy(&mut w),
                Field::Magnetic => st.h.add_energy(&mut w),
            }
        }
        w
    }

    /// Energy stored in the electric poles at `t = n·dt`,
    /// `Σ ½·L·j² + ½·q²/C_p` \[J\] (module docs).
    pub fn electric_energy(&self) -> f64 {
        self.sum_energy(Field::Electric)
    }

    /// Energy stored in the magnetic poles at `t = (n − ½)·dt` \[J\]
    /// (see [Magnetic poles](self#magnetic-poles) for why this is the
    /// right time level for the engine's energy invariant).
    pub fn magnetic_energy(&self) -> f64 {
        self.sum_energy(Field::Magnetic)
    }
}

impl Operator for Dispersive {
    fn hooks(&self) -> Hooks {
        Hooks::NONE
    }

    fn slab_hooks(&mut self, part: &SlabPartition<'_>) -> Option<Arc<dyn SlabHooks>> {
        if !self.installed || self.parts.is_some() {
            // Registered without `add_to` (nothing to do) or twice.
            return None;
        }
        assert_eq!(
            part.grid().nodes(),
            self.nodes,
            "dispersive medium built for another grid"
        );
        assert_eq!(part.dt(), self.dt, "dispersive medium built for another dt");
        let nz = self.nodes[2];
        let mut slabs: Vec<SlabState> = vec![SlabState::default(); part.slabs()];
        let mut s = 0;
        for eb in std::mem::take(&mut self.built) {
            while !part.planes(s).contains(&eb.node[0]) {
                s += 1;
            }
            let side = match eb.field {
                Field::Electric => &mut slabs[s].e,
                Field::Magnetic => &mut slabs[s].h,
            };
            side.axis.push(eb.axis.index() as u8);
            side.plane.push(eb.node[0] as u32);
            side.off.push((eb.node[1] * nz + eb.node[2]) as u32);
            side.prev.push(0.0);
            let mut sum = 0.0;
            for (a, inv_c, inv_z, half_l, explicit) in eb.poles {
                side.a.push(a);
                side.inv_c.push(inv_c);
                side.inv_z.push(inv_z);
                side.half_l.push(half_l);
                side.explicit.push(explicit);
                side.j.push(0.0);
                side.q.push(0.0);
                if !explicit {
                    sum += inv_z;
                }
            }
            side.sum_inv_z.push(sum);
            side.pole_end.push(side.a.len() as u32);
        }
        let parts = Arc::new(Parts {
            dt: self.dt,
            slabs: slabs.into_iter().map(Mutex::new).collect(),
        });
        self.parts = Some(parts.clone());
        Some(parts)
    }

    /// [`electric_energy`](Self::electric_energy) plus
    /// [`magnetic_energy`](Self::magnetic_energy) (see
    /// [Magnetic poles](self#magnetic-poles)).
    fn energy(&self) -> f64 {
        self.electric_energy() + self.magnetic_energy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn susceptibility_forms() {
        let w = 2e10;
        let d = Pole::drude(3e10, 1e9).susceptibility(w);
        let l = Pole::lorentz(3e10, 0.0, 1e9).susceptibility(w);
        assert!((d - l).norm() < 1e-12 * d.norm());
        // Passive: Im χ < 0 (e^{jωt}).
        assert!(d.im < 0.0);
        let db = Pole::debye(2.0, 1e-11).susceptibility(w);
        assert!(
            db.im < 0.0 && (Pole::debye(2.0, 1e-11).susceptibility(0.0).re - 2.0).abs() < 1e-15
        );
        assert!(Pole::lorentz(1.0, f64::NAN, 0.0).validate().is_err());
        assert!(Pole::debye(-1.0, 0.0).validate().is_err());
        let s = Pole::lorentz(3e10, 1e10, 1e9).scaled(0.25);
        let r = s.susceptibility(w) / Pole::lorentz(3e10, 1e10, 1e9).susceptibility(w);
        assert!((r - 0.25).norm() < 1e-12);
    }

    /// The branch admittance `1/(jωL + R + 1/(jωC))` is `jω·ε0·χ·g`.
    #[test]
    fn branch_admittance_matches_susceptibility() {
        let dt = 1e-12;
        let m = EPS0 * 2.5e-3;
        let w = 3e10;
        let j = Complex64::I;
        for p in [
            Pole::lorentz(4e10, 2e10, 3e9),
            Pole::drude(4e10, 3e9),
            Pole::debye(3.0, 2e-11),
        ] {
            let (a, inv_c, z) = p.branch(m, dt);
            let l = 0.5 * a * dt;
            let r = z - a - 0.5 * dt * inv_c;
            let zb = j * (w * l) + r + inv_c / (j * w);
            let y = 1.0 / zb;
            let want = j * (w * EPS0 * 2.5e-3) * p.susceptibility(w);
            assert!((y - want).norm() < 1e-9 * want.norm(), "{p:?}");
        }
    }
}
