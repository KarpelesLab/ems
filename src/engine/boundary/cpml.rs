//! Convolutional perfectly matched layer (CPML) as an engine operator.
//!
//! Recursive-convolution CPML of J. A. Roden and S. D. Gedney,
//! "Convolutional PML (CPML): An efficient FDTD implementation of the
//! CFS-PML for arbitrary media", *Microwave and Optical Technology Letters*
//! 27(5), 334–339, 2000; see also Taflove & Hagness, *Computational
//! Electrodynamics*, 3rd ed., ch. 7 (stretched-coordinate PML, polynomial
//! grading, CFS-PML) and Gedney, *Introduction to the FDTD Method*, ch. 11.
//! The PML concept itself is Berenger (1994).
//!
//! # Continuous formulation
//!
//! In a PML normal to axis `p`, the derivative `∂/∂p` is replaced by
//! `(1/s_p)·∂/∂p` with the complex-frequency-shifted stretching factor
//!
//! ```text
//! s_p = κ_p + σ_p / (α_p + jω·ε0)
//! ```
//!
//! Its inverse in the time domain is `δ(t)/κ_p + ζ_p(t)` with
//! `ζ_p(t) = −σ_p/(ε0·κ_p²)·exp(−(σ_p/κ_p + α_p)·t/ε0)·u(t)`. Assuming the
//! field derivative constant over each time step, the convolution `ζ_p ∗ ∂F`
//! becomes the recursion (Roden & Gedney, eqs. 14–17)
//!
//! ```text
//! ψ^{n} = b·ψ^{n−1} + a·(∂F/∂p)^{n}
//! b = exp(−(σ/κ + α)·dt/ε0)
//! a = σ·(b − 1) / (σ·κ + κ²·α)          (a = 0 where σ = 0)
//! ```
//!
//! so that e.g. for `E_x` in a layer normal to `y`
//!
//! ```text
//! ε·∂E_x/∂t + κ_e·E_x = (1/κ_y)·∂H_z/∂y + ψ_{E_x,y} − ∂H_y/∂z
//! ```
//!
//! Only the term differentiating along the layer normal changes; the other
//! curl term is untouched. Because the stretching is a change of
//! coordinates, it is independent of the medium: `ε`, `μ`, `κ_e` and `σ_m`
//! of the background enter only through the ordinary update coefficients.
//! The same `s_p` (with `ε0`) is used for E and H, which is the stretched-
//! coordinate form of the matching condition `σ/ε0 = σ*/μ0`.
//!
//! # Integral (voltage/current) form
//!
//! The engine stores `u = E·l` on primary edges and `i = H·l*` on dual
//! edges, and the kernel computes (engine module docs)
//!
//! ```text
//! u_a ← ca·u_a + cb·(S_next + S_prev)       cb = dt/(C + G·dt/2), C = ε·A*/l
//! i_a ← da·i_a − db·(T_next + T_prev)       db = dt/(L + R·dt/2), L = μ·A/l*
//! ```
//!
//! where the contour sum around the dual (primary) face splits into two
//! *derivative terms*, one per transverse axis. For a component `a`, with
//! `(n, q) = a.others()` the next and previous axes:
//!
//! ```text
//! S_n = i_q[node] − i_q[node − e_n],   S_q = −(i_n[node] − i_n[node − e_q])
//! T_n = u_q[node + e_n] − u_q[node],   T_q = −(u_n[node + e_q] − u_n[node])
//! ```
//!
//! Each is an exact area-weighted derivative. Taking `S_y` of an `E_x` edge
//! (dual face `A* = l*_y·l*_z`): the two `i_z` dual edges have the same
//! length `l*_z`, so
//!
//! ```text
//! S_y = l*_z·(H_z[j] − H_z[j−1]) = A*·(H_z[j] − H_z[j−1])/l*_y = A*·D_y H_z
//! ```
//!
//! with `D_y` the centred difference at the edge's own y position (a primary
//! node). Multiplying the stretched E_x equation by `l` and using
//! `cb = dt·l/(A*·(ε + κ_e·dt/2))` gives
//!
//! ```text
//! u ← ca·u + cb·(S_y/κ_y + Ψ + S_z),   Ψ = A*·ψ,   Ψ ← b·Ψ + a·S_y
//! ```
//!
//! `A*` is constant in time, so the recursion for the scaled `Ψ` (units of
//! current) has the same `a, b` as the point form, and **no length or area
//! factor appears explicitly**: everything is carried by `cb` and the
//! stored edge integrals. The same holds for H with `T_p = A·D_p E` and
//! `db`. Hence, since the kernel already applied the unstretched term, the
//! operator adds the correction
//!
//! ```text
//! post_e:  Ψ_E ← b_E·Ψ_E + a_E·S_p;   u ← u + cb·((1/κ_E − 1)·S_p + Ψ_E)
//! post_h:  Ψ_H ← b_H·Ψ_H + a_H·T_p;   i ← i − db·((1/κ_H − 1)·T_p + Ψ_H)
//! ```
//!
//! for every component `a ≠ p` (two E and two H components per face). This
//! is exactly the update of Roden & Gedney with `ψ` updated first and then
//! used (`ψ^{n+½}` for `E^{n+1}`, `ψ^{n}` for `H^{n+½}`): the correction is
//! linear in quantities the kernel does not modify (`i` during the E update,
//! `u` during the H update), so applying it afterwards is identical.
//!
//! `κ` cannot be folded into `cb`: the kernel's single `cb` multiplies both
//! derivative terms, and `κ_p` scales only one of them. Keeping the whole
//! correction in the operator also leaves the coefficient arrays untouched,
//! so materials, PEC edges and lumped elements inside the layer keep working
//! (a PEC edge has `cb = 0` and receives no correction).
//!
//! # Positions and grading
//!
//! The layer of a face occupies its outermost `n = Cpml::cells` mesh cells,
//! from the outer face (PEC, see below) to the interface plane. With `ρ` the
//! normalised depth (0 at the interface, 1 at the outer face), the profile
//! is (Taflove & Hagness ch. 7, polynomial grading; Roden & Gedney for `α`)
//!
//! ```text
//! σ(ρ) = σ_max(Δ)·ρ^m,   κ(ρ) = 1 + (κ_max − 1)·ρ^m,   α(ρ) = α_max·(1 − ρ)
//! ```
//!
//! evaluated **where each derivative is taken**:
//!
//! * the `S_p` of the E update is centred on the edge's primary node, so
//!   E-side parameters use `ρ` of the mesh line `k`: `ρ = |k − k_i|/n`;
//! * the `T_p` of the H update is centred on the dual node, the midpoint of
//!   cell `k`, so H-side parameters use `ρ = |k + ½ − k_i|/n`,
//!
//! with `k_i` the index of the interface line.
//!
//! `σ_max = Cpml::sigma_max(Z0, Δ)` uses the **local** cell size `Δ` normal
//! to the layer: the cell itself for H positions, the mean of the (one or
//! two) adjacent cells for E positions.
//!
//! **Graded PML meshes.** `ρ` is measured in *cells* (mesh index), not in
//! metres. On a uniform mesh the two are identical. On a graded mesh this
//! choice, together with the local `σ_max ∝ 1/Δ`, makes the attenuation per
//! cell `σ·Δ` follow exactly the profile of a uniform layer, which is what
//! governs the discretisation reflection. Measured at normal incidence
//! (8 cells, `m = 3`, cells growing outwards by 1.1× / 1.25×), index depth
//! gives −57 / −49 dB against −51 / −39 dB for physical depth (where the
//! large outer cells take most of the depth and the inner cells barely
//! attenuate). The remaining level is insensitive to `σ_max`, i.e. it is
//! dominated by the mesh grading itself.
//!
//! At the interface (`ρ = 0`) `σ = 0`, so `a = 0` and the E edges on the
//! interface plane need no correction; they are skipped.
//!
//! # Outer face, edges and corners
//!
//! Build the engine with [`FaceCondition::Pec`](crate::engine::FaceCondition::Pec)
//! on every CPML face (design §5.6): the layer is PEC-backed. The PEC face
//! edges have `cb = 0`, so the correction is inert there; a PMC-backed layer
//! would work the same way.
//!
//! Every face's slab spans the full transverse extent of the mesh. Where two
//! or three layers overlap (domain edges and corners), each face's `ψ`
//! corrects only the derivative along its own normal and the corrections
//! add, which is exactly the multi-axis stretched update — correct by
//! construction. Opposite faces must not overlap.
//!
//! # Background medium
//!
//! The update is exact for any background `ε`, `μ`, `κ_e`, `σ_m` (see
//! above): the stretching uses `ε0`, so it is a pure change of
//! coordinates. Only the *choice* of `σ_max` depends on the medium: a wave
//! of refractive index `n = √(ε_r·μ_r)` has the wavenumber `n·ω/c`, so the
//! layer attenuates it by `exp(−n·Z0·∫σ dp)`, `n` times more per cell than
//! in vacuum. The discretisation reflection is governed by the attenuation
//! per cell, so the optimum `σ_max` of a medium is the vacuum optimum
//! divided by `n` (in Taflove & Hagness's ch. 7 form, where `σ` is
//! referred to the medium's own `ε`, this is the familiar
//! `σ_opt ∝ 1/(η·Δ)` with the medium's `η`).
//!
//! [`CpmlBoundary::new`] therefore measures, per face, the mean refractive
//! index of the layer, `n = √(ε̄_r·μ̄_r)`, from the engine's coefficients
//! (`ε̄_r` the volume-weighted mean over the layer's tangential, non-PEC E
//! edges, `μ̄_r` likewise over its tangential dual edges; values within
//! 1e-6 of 1 are taken as exactly 1, so vacuum layers are unchanged), and
//! uses `σ_max = sigma_factor · 0.8·(m + 1)/(n·Z0·Δ)`
//! ([`CpmlBoundary::refractive_index`]). One `n` per face keeps the
//! stretching a function of the normal coordinate alone, which is what
//! makes the layer reflectionless at transverse material interfaces inside
//! it (a substrate crossing the layer); for such mixed layers the mean is
//! a compromise (the optimum is broad: a factor of 2 in `σ_max` costs
//! about 5 dB). [`CpmlBoundary::with_grid`], without materials, assumes
//! vacuum.
//!
//! Measured at normal incidence (8 cells, `m = 3`, homogeneous background
//! including the layer): −83.6 dB in vacuum, −85.7 dB for `ε_r` = 4 and 12
//! (−80.5 / −73.4 dB with the vacuum `σ_max`).
//!
//! # Cylindrical grids
//!
//! On a cylindrical grid the z faces (and the α faces of a sector) use
//! the layers above unchanged. The `ρ_max` face uses the cylindrical PML
//! of [`cyl_pml`](super::cyl_pml) instead, a stretch of the radius itself,
//! which also enters the α metric. It takes over every component in its
//! x-planes, including the corners with the z / α layers, which are cut at
//! its interface. [`CpmlBoundary::plane_rho_max`] restores the plain
//! Cartesian stretch there, for comparison only (it reflects at the order
//! of `1/(kρ)`).
//!
//! # Parallel execution and cost
//!
//! The correction runs as [slab hooks](crate::engine::SlabHooks): inside
//! each half-step, every worker thread corrects the part of every layer
//! that lies in its own x-slab, right after its kernel update, with no
//! extra barrier. At registration
//! ([`Operator::slab_hooks`]) each
//! layer is cut along x at the engine's slab boundaries: an x-normal layer
//! (whole x-planes) falls into the slab(s) holding those planes, a y- or
//! z-normal layer (spanning all x) is split across all slabs. Each piece
//! owns its own `ψ` values, and every value is computed by the same
//! expression as in a single piece, so results are bit-identical for any
//! thread count. The operator declares no serial hooks
//! ([`Hooks::NONE`](crate::engine::Hooks::NONE)), so an engine with only
//! CPML keeps one barrier per half-step.
//!
//! Per step the pass touches four field components per face layer and
//! stores four `f32` `ψ` values per layer node.

use std::sync::{Arc, Mutex, PoisonError};

use super::cyl_pml::CylLayer;
use crate::axis::Axis;
use crate::constants::{EPS0, MU0, Z0};
use crate::engine::Coefficients;
use crate::engine::Engine;
use crate::engine::grid::Grid;
use crate::engine::kernel::flush;
use crate::engine::operator::{Hooks, Operator};
use crate::engine::slab::{SlabContext, SlabHooks, SlabMut, SlabPartition, SlabRef};
use crate::fdtd::{Cpml, Face};
use crate::{Error, Result};

/// Recursive-convolution coefficients at one depth position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpmlCoefficients {
    /// Normalised depth `ρ ∈ [0, 1]` (0 at the interface).
    pub rho: f64,
    /// Conductivity `σ(ρ)` \[S/m\].
    pub sigma: f64,
    /// Stretching `κ(ρ) ≥ 1`.
    pub kappa: f64,
    /// Frequency shift `α(ρ)` \[S/m\].
    pub alpha: f64,
    /// `b = exp(−(σ/κ + α)·dt/ε0)`.
    pub b: f64,
    /// `a = σ·(b − 1)/(σ·κ + κ²·α)` (0 where `σ = 0`).
    pub a: f64,
}

impl CpmlCoefficients {
    /// Grading of `p` at depth `rho` for local cell size `delta` \[m\] and
    /// time step `dt` \[s\] (see the [module docs](self)), for a vacuum
    /// layer.
    pub fn new(p: &Cpml, rho: f64, delta: f64, dt: f64) -> Self {
        Self::in_medium(p, rho, delta, dt, 1.0)
    }

    /// Like [`new`](Self::new) for a layer of refractive index `n`
    /// (`σ_max` divided by `n`, see
    /// [Background medium](self#background-medium)).
    pub fn in_medium(p: &Cpml, rho: f64, delta: f64, dt: f64, n: f64) -> Self {
        let g = rho.powf(p.grading_order);
        let sigma = p.sigma_max(n * Z0, delta) * g;
        let kappa = 1.0 + (p.kappa_max - 1.0) * g;
        let alpha = p.alpha_max * (1.0 - rho);
        let b = (-(sigma / kappa + alpha) * dt / EPS0).exp();
        let den = sigma * kappa + kappa * kappa * alpha;
        let a = if sigma > 0.0 && den > 0.0 {
            sigma * (b - 1.0) / den
        } else {
            0.0
        };
        Self {
            rho,
            sigma,
            kappa,
            alpha,
            b,
            a,
        }
    }
}

/// `ψ` storage and coefficients for one field component in one face layer.
#[derive(Debug, Clone)]
struct Slab {
    /// Updated component.
    comp: Axis,
    /// Component whose derivative along `normal` is corrected.
    other: Axis,
    /// Layer normal.
    normal: Axis,
    /// `+1` if `normal == comp.next()`, else `−1` (curl sign).
    sign: f32,
    /// Node box `lo..hi` covered by the slab.
    lo: [usize; 3],
    hi: [usize; 3],
    /// Per depth index `node[normal] − lo[normal]`.
    b: Vec<f32>,
    a: Vec<f32>,
    /// `1/κ − 1`.
    k: Vec<f32>,
    /// One value per node of the box, in `(i, j, k)` order (empty in the
    /// unpartitioned templates of a [`Layer`]).
    psi: Vec<f32>,
}

impl Slab {
    fn new(
        comp: Axis,
        normal: Axis,
        lo: [usize; 3],
        hi: [usize; 3],
        co: &[CpmlCoefficients],
    ) -> Self {
        let (sign, other) = if normal == comp.next() {
            (1.0, comp.prev())
        } else {
            (-1.0, comp.next())
        };
        Self {
            comp,
            other,
            normal,
            sign,
            lo,
            hi,
            b: co.iter().map(|c| c.b as f32).collect(),
            a: co.iter().map(|c| c.a as f32).collect(),
            k: co.iter().map(|c| (1.0 / c.kappa - 1.0) as f32).collect(),
            psi: Vec::new(),
        }
    }

    /// The part of the slab in the x-planes `x0..x1`, with zeroed `ψ`, or
    /// `None` if it is empty. The per-depth coefficients of an x-normal
    /// slab are cut accordingly, so every node keeps its values.
    fn clip(&self, x0: usize, x1: usize) -> Option<Self> {
        let (lo0, hi0) = (self.lo[0].max(x0), self.hi[0].min(x1));
        if lo0 >= hi0 || (0..3).any(|d| self.lo[d] >= self.hi[d]) {
            return None;
        }
        let mut s = Self {
            psi: Vec::new(),
            ..self.clone()
        };
        if self.normal == Axis::X {
            let r = lo0 - self.lo[0]..hi0 - self.lo[0];
            s.b = self.b[r.clone()].to_vec();
            s.a = self.a[r.clone()].to_vec();
            s.k = self.k[r].to_vec();
        }
        (s.lo[0], s.hi[0]) = (lo0, hi0);
        let len = (0..3).map(|d| s.hi[d] - s.lo[d]).product();
        s.psi = vec![0.0; len];
        Some(s)
    }

    /// Applies the correction. `e_side`: target is `u` (derivative of `i`
    /// backwards), else target is `i` (derivative of `u` forwards).
    /// `coef` is `cb` (E) or `db` (H) of `comp` in global layout; `zero`
    /// is an all-zero x-plane.
    fn apply(
        &mut self,
        tgt: &mut SlabMut<'_>,
        src: &SlabRef<'_>,
        coef: &[f32],
        zero: &[f32],
        e_side: bool,
    ) {
        let n = src.nodes();
        let nz = n[2];
        let pl = n[1] * nz;
        // E: u += cb·(…); H: i −= db·(…).
        let scale = if e_side { 1.0 } else { -1.0 };
        let (c, sign, normal) = (self.other, self.sign, self.normal);
        let (lo, hi) = (self.lo, self.hi);
        let len = hi[2] - lo[2];
        let mut q = 0;
        for i in lo[0]..hi[0] {
            let cur = src.plane(c, i);
            // The neighbouring x-plane for an x-normal layer (possibly
            // the neighbour plane of the next / previous slab).
            let other: &[f32] = match (normal, e_side) {
                (Axis::X, true) if i == 0 => zero, // outside the domain
                (Axis::X, true) => src.plane(c, i - 1),
                (Axis::X, false) => src.plane(c, i + 1),
                _ => cur,
            };
            let cf = &coef[i * pl..(i + 1) * pl];
            let t = tgt.plane_mut(self.comp, i);
            for j in lo[1]..hi[1] {
                let o = j * nz + lo[2];
                let psi = &mut self.psi[q..q + len];
                q += len;
                let (t, cf) = (&mut t[o..o + len], &cf[o..o + len]);
                // Derivative term S_p / T_p = sign·(h − l) over the row.
                match normal {
                    Axis::X | Axis::Y => {
                        let z = if normal == Axis::X { i } else { j } - lo[normal.index()];
                        let (h, l) = match (normal, e_side) {
                            (Axis::X, true) => (&cur[o..o + len], &other[o..o + len]),
                            (Axis::X, false) => (&other[o..o + len], &cur[o..o + len]),
                            (_, true) if j == 0 => (&cur[o..o + len], &zero[..len]),
                            (_, true) => (&cur[o..o + len], &cur[o - nz..o - nz + len]),
                            (_, false) => (&cur[o + nz..o + nz + len], &cur[o..o + len]),
                        };
                        let co = [self.b[z], self.a[z], self.k[z]];
                        row_uniform(t, psi, cf, h, l, co, sign, scale);
                    }
                    Axis::Z => {
                        let (b, a, k) = (&self.b[..], &self.a[..], &self.k[..]);
                        if !e_side {
                            let (h, l) = (&cur[o + 1..o + 1 + len], &cur[o..o + len]);
                            row_graded(t, psi, cf, h, l, [b, a, k], sign, scale);
                        } else if lo[2] > 0 {
                            let (h, l) = (&cur[o..o + len], &cur[o - 1..o - 1 + len]);
                            row_graded(t, psi, cf, h, l, [b, a, k], sign, scale);
                        } else if len > 0 {
                            // k = 0: the k − 1 neighbour is outside the domain.
                            let (t0, t1) = t.split_at_mut(1);
                            let (p0, p1) = psi.split_at_mut(1);
                            let co0 = [&b[..1], &a[..1], &k[..1]];
                            row_graded(
                                t0,
                                p0,
                                &cf[..1],
                                &cur[o..o + 1],
                                &zero[..1],
                                co0,
                                sign,
                                scale,
                            );
                            let co1 = [&b[1..], &a[1..], &k[1..]];
                            let (h, l) = (&cur[o + 1..o + len], &cur[o..o + len - 1]);
                            row_graded(t1, p1, &cf[1..], h, l, co1, sign, scale);
                        }
                    }
                }
            }
        }
    }
}

/// One row of a slab whose depth is constant along the row:
/// `s = sign·(h − l)`, `ψ ← b·ψ + a·s`, `t ← t + scale·cf·((1/κ − 1)·s + ψ)`
/// with `co = [b, a, 1/κ − 1]`.
#[inline]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn row_uniform(
    t: &mut [f32],
    psi: &mut [f32],
    cf: &[f32],
    h: &[f32],
    l: &[f32],
    co: [f32; 3],
    sign: f32,
    scale: f32,
) {
    let [b, a, k] = co;
    let n = t.len();
    let (psi, cf, h, l) = (&mut psi[..n], &cf[..n], &h[..n], &l[..n]);
    for q in 0..n {
        let s = sign * (h[q] - l[q]);
        psi[q] = flush(b * psi[q] + a * s);
        t[q] = flush(t[q] + scale * cf[q] * (k * s + psi[q]));
    }
}

/// Like [`row_uniform`], with the depth (and so `co = [b, a, 1/κ − 1]`)
/// varying along the row.
#[inline]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn row_graded(
    t: &mut [f32],
    psi: &mut [f32],
    cf: &[f32],
    h: &[f32],
    l: &[f32],
    co: [&[f32]; 3],
    sign: f32,
    scale: f32,
) {
    let [b, a, k] = co;
    let n = t.len();
    let (psi, cf, h, l) = (&mut psi[..n], &cf[..n], &h[..n], &l[..n]);
    let (b, a, k) = (&b[..n], &a[..n], &k[..n]);
    for q in 0..n {
        let s = sign * (h[q] - l[q]);
        psi[q] = flush(b[q] * psi[q] + a[q] * s);
        t[q] = flush(t[q] + scale * cf[q] * (k[q] * s + psi[q]));
    }
}

/// One face layer: its settings and four slabs.
#[derive(Debug, Clone)]
struct Layer {
    face: Face,
    params: Cpml,
    /// Refractive index the grading is tuned for.
    n: f64,
    /// The two E slabs and two H slabs.
    e: [Slab; 2],
    h: [Slab; 2],
    e_coeffs: Vec<CpmlCoefficients>,
    h_coeffs: Vec<CpmlCoefficients>,
}

/// CPML absorbing layers on any set of faces, as one engine
/// [`Operator`].
///
/// See the [module docs](self) for the formulation. Build the engine with
/// PEC on the CPML faces, then register the operator. Its correction runs
/// inside the half-steps, in parallel ([module docs](self#parallel-execution-and-cost)),
/// so serial `post_h` / `post_e` hooks of any operator (probes, dumps)
/// always see corrected values, whatever the registration order:
///
/// ```
/// use ems::engine::boundary::cpml::CpmlBoundary;
/// use ems::engine::{Engine, Grid};
/// use ems::{Cpml, Face};
///
/// let grid = Grid::uniform([30, 30, 30], [1e-3; 3])?;
/// let mut engine = Engine::builder(grid).threads(1).build()?;
/// let pml = CpmlBoundary::new(&engine, Face::ALL.map(|f| (f, Cpml::cells(8))))?;
/// engine.add_operator(pml);
/// engine.run(10);
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Panics
/// Registering it ([`Engine::add_operator`]) panics if the engine's grid
/// or time step differ from those it was built for.
#[derive(Debug, Clone)]
pub struct CpmlBoundary {
    nodes: [usize; 3],
    dt: f64,
    layers: Vec<Layer>,
    /// The `ρ_max` layer of a cylindrical grid uses the cylindrical PML
    /// ([`cyl_pml`](super::cyl_pml)) and these are its grid and interface
    /// line.
    cyl: Option<(Grid, usize)>,
    /// The state partitioned over the engine's slabs, once registered.
    parts: Option<Arc<CpmlParts>>,
}

/// The layers' pieces in each x-slab of an engine, run by the workers.
#[derive(Debug)]
struct CpmlParts {
    slabs: Vec<Mutex<Pieces>>,
    /// An all-zero x-plane (out-of-domain neighbours).
    zero: Vec<f32>,
}

/// The pieces of one x-slab, in layer order.
#[derive(Debug, Default)]
struct Pieces {
    e: Vec<Slab>,
    h: Vec<Slab>,
    cyl: Option<CylLayer>,
}

impl CpmlParts {
    fn pieces(&self, slab: usize) -> std::sync::MutexGuard<'_, Pieces> {
        self.slabs[slab]
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl SlabHooks for CpmlParts {
    fn post_h_slab(&self, i: &mut SlabMut<'_>, u: &SlabRef<'_>, ctx: &SlabContext<'_>) {
        let mut p = self.pieces(ctx.slab);
        for s in &mut p.h {
            s.apply(i, u, ctx.coefficients.db(s.comp), &self.zero, false);
        }
        if let Some(c) = &mut p.cyl {
            c.post_h(i, u, |a| ctx.coefficients.db(a), &self.zero);
        }
    }

    fn post_e_slab(&self, u: &mut SlabMut<'_>, i: &SlabRef<'_>, ctx: &SlabContext<'_>) {
        let mut p = self.pieces(ctx.slab);
        for s in &mut p.e {
            s.apply(u, i, ctx.coefficients.cb(s.comp), &self.zero, true);
        }
        if let Some(c) = &mut p.cyl {
            c.post_e(u, i, |a| ctx.coefficients.cb(a), &self.zero);
        }
    }
}

impl CpmlBoundary {
    /// CPML layers for `engine` (its grid and time step), one per
    /// `(face, parameters)` entry, with `σ_max` tuned to the mean
    /// refractive index of each layer's medium (see
    /// [Background medium](self#background-medium)).
    ///
    /// # Errors
    /// As [`with_grid`](Self::with_grid).
    pub fn new(engine: &Engine, layers: impl IntoIterator<Item = (Face, Cpml)>) -> Result<Self> {
        let grid = engine.grid();
        let co = engine.coefficients();
        let dt = engine.dt();
        Self::build(grid, dt, layers, |face, cells| {
            layer_index(grid, &co, dt, face, cells)
        })
    }

    /// CPML layers for a mesh `grid` and time step `dt` \[s\].
    ///
    /// # Errors
    /// [`Error::Invalid`] if parameters are invalid ([`Cpml::validate`]),
    /// a face is listed twice, a layer has more cells than the mesh, or two
    /// opposite layers overlap.
    /// The layers are tuned for vacuum (no material information).
    pub fn with_grid(
        grid: &Grid,
        dt: f64,
        layers: impl IntoIterator<Item = (Face, Cpml)>,
    ) -> Result<Self> {
        Self::build(grid, dt, layers, |_, _| 1.0)
    }

    fn build(
        grid: &Grid,
        dt: f64,
        layers: impl IntoIterator<Item = (Face, Cpml)>,
        mut index: impl FnMut(Face, usize) -> f64,
    ) -> Result<Self> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::Invalid(format!("CPML: invalid time step {dt}")));
        }
        let n = grid.nodes();
        let mut cells = [0usize; 6];
        let mut out = Vec::new();
        let mut cyl = None;
        for (face, params) in layers {
            params.validate()?;
            if cells[face.index()] != 0 {
                return Err(Error::Invalid(format!("CPML: face {face} given twice")));
            }
            let p = face.axis();
            let total = n[p.index()] - 1;
            if params.cells > total {
                return Err(Error::Invalid(format!(
                    "CPML on {face}: {} cells but the mesh has only {total} along {p}",
                    params.cells
                )));
            }
            cells[face.index()] = params.cells;
            if grid.is_cylindrical() && face == Face::XMax {
                let inner = total - params.cells;
                if grid.lines(Axis::X)[inner] <= 0.0 {
                    return Err(Error::Invalid(format!(
                        "PML on {face}: the layer of {} cells reaches the axis",
                        params.cells
                    )));
                }
                cyl = Some((grid.clone(), inner));
            }
            let n_face = index(face, params.cells);
            out.push(Layer::new(grid, dt, face, params, n_face));
        }
        for p in Axis::ALL {
            let (lo, hi) = (cells[2 * p.index()], cells[2 * p.index() + 1]);
            if lo + hi > n[p.index()] - 1 {
                return Err(Error::Invalid(format!(
                    "CPML: opposite layers along {p} ({lo} + {hi} cells) overlap"
                )));
            }
        }
        Ok(Self {
            nodes: n,
            dt,
            layers: out,
            cyl,
            parts: None,
        })
    }

    /// The faces and their parameters, in the order given.
    pub fn layers(&self) -> impl Iterator<Item = (Face, &Cpml)> + '_ {
        self.layers.iter().map(|l| (l.face, &l.params))
    }

    /// Refractive index the grading of `face` is tuned for (1 for vacuum
    /// and for layers built with [`with_grid`](Self::with_grid)), or
    /// `None` if `face` has no layer.
    pub fn refractive_index(&self, face: Face) -> Option<f64> {
        self.layer(face).map(|l| l.n)
    }

    /// Grading at the E-side positions of `face`: the layer's mesh lines
    /// (outer face included, interface line excluded) in increasing index
    /// order, or `None` if `face` has no layer.
    pub fn e_profile(&self, face: Face) -> Option<&[CpmlCoefficients]> {
        self.layer(face).map(|l| &l.e_coeffs[..])
    }

    /// Grading at the H-side positions of `face`: the midpoints of the
    /// layer's cells in increasing index order, or `None` if `face` has no
    /// layer.
    pub fn h_profile(&self, face: Face) -> Option<&[CpmlCoefficients]> {
        self.layer(face).map(|l| &l.h_coeffs[..])
    }

    /// Whether the `ρ_max` face of a cylindrical grid uses the cylindrical
    /// PML ([`cyl_pml`](super::cyl_pml), the default there).
    pub fn is_cylindrical_rho_max(&self) -> bool {
        self.cyl.is_some()
    }

    /// Uses the Cartesian (plane-wave) stretch on the `ρ_max` face of a
    /// cylindrical grid instead of the cylindrical PML. It reflects an
    /// outgoing cylindrical wave at the order of `1/(kρ)`; this exists for
    /// comparison only.
    pub fn plane_rho_max(mut self) -> Self {
        self.cyl = None;
        self
    }

    /// The Cartesian stretch `[β, w, 1/κ − 1]` of the layers normal to
    /// `normal` for the `comp` correction (E or H side), per node index
    /// along `normal`, or `None` if `normal` has no layer.
    fn transverse(&self, e_side: bool, comp: Axis, normal: Axis) -> Option<[Vec<f32>; 3]> {
        let len = self.nodes[normal.index()];
        let mut out: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; len]);
        let mut any = false;
        for l in self.layers.iter().filter(|l| l.face.axis() == normal) {
            let slabs = if e_side { &l.e } else { &l.h };
            for s in slabs.iter().filter(|s| s.comp == comp) {
                let r = s.lo[normal.index()]..s.hi[normal.index()];
                out[0][r.clone()].copy_from_slice(&s.b);
                out[1][r.clone()].copy_from_slice(&s.a);
                out[2][r].copy_from_slice(&s.k);
                any = true;
            }
        }
        any.then_some(out)
    }

    /// The cylindrical `ρ_max` layer as the operator builds it (whole
    /// grid, no state), if this boundary uses one; for operators that
    /// update faces of the layer themselves
    /// ([`ConformalPec::set_cpml`](crate::engine::ConformalPec::set_cpml)).
    pub(crate) fn cyl_layer(&self) -> Option<CylLayer> {
        self.cyl.as_ref().map(|(grid, _)| {
            let l = self.layer(Face::XMax).expect("ρ_max layer");
            CylLayer::new(grid, self.dt, &l.params, l.n, |e, c, d| {
                self.transverse(e, c, d)
            })
        })
    }

    fn layer(&self, face: Face) -> Option<&Layer> {
        self.layers.iter().find(|l| l.face == face)
    }
}

/// Mean refractive index `√(ε̄_r·μ̄_r)` of the `cells`-cell layer on `face`
/// (module docs, *Background medium*), from the update coefficients.
fn layer_index(grid: &Grid, co: &Coefficients, dt: f64, face: Face, cells: usize) -> f64 {
    let p = face.axis();
    let n = grid.nodes();
    let np = n[p.index()];
    let inner = if face.is_max() { np - 1 - cells } else { cells };
    // Lines of the layer (interface included) and its cells.
    let lines = if face.is_max() {
        inner..np
    } else {
        0..inner + 1
    };
    let cell_r = if face.is_max() {
        inner..np - 1
    } else {
        0..inner
    };
    let (mut we, mut se, mut wh, mut sh) = (0.0, 0.0, 0.0, 0.0);
    let (t1, t2) = p.others();
    for comp in [t1, t2] {
        let (ab, ac) = comp.others();
        let (a, b, c) = (comp.index(), ab.index(), ac.index());
        let (ca, cb) = (co.ca(comp), co.cb(comp));
        let (da, db) = (co.da(comp), co.db(comp));
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let node = [i, j, k];
                    let idx = grid.index(node);
                    let along = node[p.index()];
                    if lines.contains(&along) && node[a] + 1 < n[a] && cb[idx] != 0.0 {
                        // Physical geometry (also cylindrical; the same
                        // products as `dual_delta`·`dual_delta`, `delta` on
                        // Cartesian grids).
                        let area = grid.dual_face_area(comp, node);
                        let l = grid.edge_length(comp, node);
                        let cap = dt * (1.0 + ca[idx] as f64) / (2.0 * cb[idx] as f64);
                        let vol = area * l;
                        we += vol;
                        se += vol * cap * l / (EPS0 * area);
                    }
                    if cell_r.contains(&along)
                        && node[b] + 1 < n[b]
                        && node[c] + 1 < n[c]
                        && db[idx] != 0.0
                    {
                        let area = grid.face_area(comp, node);
                        let l = grid.dual_edge_length(comp, node);
                        let ind = dt * (1.0 + da[idx] as f64) / (2.0 * db[idx] as f64);
                        let vol = area * l;
                        wh += vol;
                        sh += vol * ind * l / (MU0 * area);
                    }
                }
            }
        }
    }
    let eps = if we > 0.0 { se / we } else { 1.0 };
    let mu = if wh > 0.0 { sh / wh } else { 1.0 };
    let n_r = (eps * mu).sqrt();
    if (n_r - 1.0).abs() < 1e-6 || !n_r.is_finite() || n_r <= 0.0 {
        1.0
    } else {
        n_r
    }
}

impl Layer {
    fn new(grid: &Grid, dt: f64, face: Face, params: Cpml, n_face: f64) -> Self {
        let p = face.axis();
        let n = grid.nodes();
        let np = n[p.index()];
        let c = params.cells;
        // Interface line index.
        let inner = if face.is_max() { np - 1 - c } else { c };
        // Normalised depth of a (half-)integer line index, counted in cells.
        let rho = |x: f64| ((x - inner as f64).abs() / c as f64).clamp(0.0, 1.0);
        // E positions: lines strictly inside the layer plus the outer line.
        let e_idx: Vec<usize> = if face.is_max() {
            (inner + 1..np).collect()
        } else {
            (0..inner).collect()
        };
        // H positions: the layer's cells.
        let h_idx: Vec<usize> = if face.is_max() {
            (inner..np - 1).collect()
        } else {
            (0..inner).collect()
        };
        // The grading needs the physical cell size: on the α faces of a
        // cylindrical sector the arc `ρ·Δα` at the middle of the grid's ρ
        // range (the layer's attenuation scales with ρ, so it is optimal
        // there; `Δα` alone, in radians, made it the arc at ρ = 1 m).
        let scale = if grid.is_cylindrical() && p == Axis::Y {
            let r = grid.lines(Axis::X);
            0.5 * (r[0] + r[r.len() - 1])
        } else {
            1.0
        };
        let node_delta = |k: usize| {
            let lo = (k > 0).then(|| grid.delta(p, k - 1));
            let hi = (k + 1 < np).then(|| grid.delta(p, k));
            scale
                * match (lo, hi) {
                    (Some(a), Some(b)) => 0.5 * (a + b),
                    (Some(a), None) | (None, Some(a)) => a,
                    (None, None) => unreachable!("grids have two lines per axis"),
                }
        };
        let e_coeffs: Vec<_> = e_idx
            .iter()
            .map(|&k| {
                CpmlCoefficients::in_medium(&params, rho(k as f64), node_delta(k), dt, n_face)
            })
            .collect();
        let h_coeffs: Vec<_> = h_idx
            .iter()
            .map(|&k| {
                let d = scale * grid.delta(p, k);
                CpmlCoefficients::in_medium(&params, rho(k as f64 + 0.5), d, dt, n_face)
            })
            .collect();
        let range = |idx: &[usize]| {
            (
                idx.first().copied().unwrap_or(0),
                idx.last().map_or(0, |&l| l + 1),
            )
        };
        let (a1, a2) = p.others();
        let e_slab = |comp: Axis| {
            let (mut lo, mut hi) = ([0; 3], n);
            hi[comp.index()] = n[comp.index()] - 1; // no padding
            (lo[p.index()], hi[p.index()]) = range(&e_idx);
            Slab::new(comp, p, lo, hi, &e_coeffs)
        };
        let h_slab = |comp: Axis| {
            let (mut lo, mut hi) = ([0; 3], n);
            let o = if comp == a1 { a2 } else { a1 };
            hi[o.index()] = n[o.index()] - 1; // no padding
            (lo[p.index()], hi[p.index()]) = range(&h_idx);
            Slab::new(comp, p, lo, hi, &h_coeffs)
        };
        Self {
            face,
            params,
            n: n_face,
            e: [e_slab(a1), e_slab(a2)],
            h: [h_slab(a1), h_slab(a2)],
            e_coeffs,
            h_coeffs,
        }
    }
}

impl Operator for CpmlBoundary {
    fn hooks(&self) -> Hooks {
        Hooks::NONE
    }

    fn slab_hooks(&mut self, part: &SlabPartition<'_>) -> Option<Arc<dyn SlabHooks>> {
        assert_eq!(
            part.grid().nodes(),
            self.nodes,
            "CPML built for another grid"
        );
        assert_eq!(part.dt(), self.dt, "CPML built for another time step");
        // The cylindrical ρ_max layer takes over all of its x-planes,
        // including the overlaps with the other faces' layers.
        let cyl = self.cyl.as_ref().map(|(grid, inner)| {
            let l = self.layer(Face::XMax).expect("ρ_max layer");
            let layer = CylLayer::new(grid, self.dt, &l.params, l.n, |e, c, d| {
                self.transverse(e, c, d)
            });
            (layer, *inner)
        });
        let x_end = cyl.as_ref().map_or(self.nodes[0], |c| c.1);
        let slabs = (0..part.slabs())
            .map(|q| {
                let r = part.planes(q);
                let mut p = Pieces::default();
                for l in &self.layers {
                    if cyl.is_some() && l.face == Face::XMax {
                        continue;
                    }
                    let end = r.end.min(x_end);
                    p.e.extend(l.e.iter().filter_map(|s| s.clip(r.start, end)));
                    p.h.extend(l.h.iter().filter_map(|s| s.clip(r.start, end)));
                }
                p.cyl = cyl.as_ref().map(|c| c.0.clip(r.start, r.end));
                Mutex::new(p)
            })
            .collect();
        let parts = Arc::new(CpmlParts {
            slabs,
            zero: vec![0.0; self.nodes[1] * self.nodes[2]],
        });
        self.parts = Some(Arc::clone(&parts));
        Some(parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grading_positions() {
        let g = Grid::uniform([20, 4, 4], [1e-3; 3]).unwrap();
        let dt = 1e-12;
        let p = Cpml::cells(4);
        let b = CpmlBoundary::with_grid(&g, dt, [(Face::XMin, p), (Face::XMax, p)]).unwrap();
        for face in [Face::XMin, Face::XMax] {
            let e = b.e_profile(face).unwrap();
            let h = b.h_profile(face).unwrap();
            assert_eq!((e.len(), h.len()), (4, 4));
            // Outermost first for XMin, innermost first for XMax.
            let er: Vec<f64> = e.iter().map(|c| c.rho).collect();
            let hr: Vec<f64> = h.iter().map(|c| c.rho).collect();
            let (mut ew, mut hw) = (vec![1.0, 0.75, 0.5, 0.25], vec![0.875, 0.625, 0.375, 0.125]);
            if face.is_max() {
                ew.reverse();
                hw.reverse();
            }
            for (a, b) in er.iter().zip(&ew).chain(hr.iter().zip(&hw)) {
                assert!((a - b).abs() < 1e-12, "{er:?} {hr:?}");
            }
        }
        let c = b.e_profile(Face::XMin).unwrap()[0];
        let smax = p.sigma_max(Z0, 1e-3);
        assert!((c.sigma / smax - 1.0).abs() < 1e-12);
        assert!((c.b - (-smax * dt / EPS0).exp()).abs() < 1e-15);
        assert!((c.a - (c.b - 1.0)).abs() < 1e-15); // κ = 1, α = 0
        // CFS terms.
        let q = Cpml::cells(4).kappa_max(5.0).alpha_max(0.05);
        let c = CpmlCoefficients::new(&q, 0.0, 1e-3, dt);
        assert_eq!((c.sigma, c.kappa, c.a), (0.0, 1.0, 0.0));
        assert!((c.alpha - 0.05).abs() < 1e-15);
        let c = CpmlCoefficients::new(&q, 1.0, 1e-3, dt);
        assert_eq!((c.kappa, c.alpha), (5.0, 0.0));
    }

    #[test]
    fn graded_local_delta() {
        // Cells growing outwards on y_max: 1, 1, 2, 3 mm.
        let y = vec![0.0, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3];
        let g = Grid::new([vec![0.0, 1e-3], y, vec![0.0, 1e-3]]).unwrap();
        let p = Cpml::cells(3);
        let b = CpmlBoundary::with_grid(&g, 1e-12, [(Face::YMax, p)]).unwrap();
        let h = b.h_profile(Face::YMax).unwrap();
        // Interface at line 2. Cell 4 (3 mm): ρ = 2.5/3, local Δ = 3 mm.
        assert!((h[2].rho - 2.5 / 3.0).abs() < 1e-12);
        let want = p.sigma_max(Z0, 3e-3) * (2.5f64 / 3.0).powi(3);
        assert!((h[2].sigma / want - 1.0).abs() < 1e-12);
        let e = b.e_profile(Face::YMax).unwrap();
        // Line 4: mean of the 2 and 3 mm cells, ρ = 2/3; line 5: single
        // 3 mm cell, ρ = 1.
        let want = p.sigma_max(Z0, 2.5e-3) * (2.0f64 / 3.0).powi(3);
        assert!((e[1].sigma / want - 1.0).abs() < 1e-12);
        assert!((e[2].sigma / p.sigma_max(Z0, 3e-3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn layer_refractive_index() {
        use crate::engine::{CellMaterials, Material};
        let g = Grid::uniform([12, 6, 6], [1e-3; 3]).unwrap();
        let build = |m: Material| {
            let mats = CellMaterials::new(&g, m);
            let e = Engine::builder(g.clone())
                .materials(mats)
                .threads(1)
                .build()
                .unwrap();
            let p = Cpml::cells(4);
            CpmlBoundary::new(&e, [(Face::XMin, p), (Face::ZMax, p)]).unwrap()
        };
        let b = build(Material::VACUUM);
        assert_eq!(b.refractive_index(Face::XMin), Some(1.0));
        assert_eq!(b.refractive_index(Face::YMin), None);
        let vac = CpmlBoundary::with_grid(&g, 1e-12, [(Face::XMin, Cpml::cells(4))]).unwrap();
        assert_eq!(vac.refractive_index(Face::XMin), Some(1.0));
        let b = build(Material::dielectric(4.0));
        let n = b.refractive_index(Face::ZMax).unwrap();
        assert!((n - 2.0).abs() < 1e-6, "{n}");
        // σ_max is the vacuum optimum divided by n.
        let e = b.e_profile(Face::ZMax).unwrap().last().unwrap();
        let want = Cpml::cells(4).sigma_max(Z0, 1e-3) / n;
        assert!((e.sigma / want - 1.0).abs() < 1e-12);
        let m = Material {
            eps_r: [2.0; 3],
            mu_r: [3.0; 3],
            ..Material::VACUUM
        };
        let n = build(m).refractive_index(Face::XMin).unwrap();
        assert!((n - 6f64.sqrt()).abs() < 1e-6, "{n}");
    }

    #[test]
    fn validation() {
        let g = Grid::uniform([10, 4, 4], [1e-3; 3]).unwrap();
        let ok = |l: Vec<(Face, Cpml)>| CpmlBoundary::with_grid(&g, 1e-12, l).is_ok();
        assert!(ok(vec![
            (Face::XMin, Cpml::cells(5)),
            (Face::XMax, Cpml::cells(5))
        ]));
        assert!(!ok(vec![
            (Face::XMin, Cpml::cells(6)),
            (Face::XMax, Cpml::cells(5))
        ]));
        assert!(!ok(vec![(Face::YMin, Cpml::cells(5))]));
        assert!(!ok(vec![
            (Face::XMin, Cpml::cells(2)),
            (Face::XMin, Cpml::cells(2))
        ]));
        assert!(!ok(vec![(Face::XMin, Cpml::cells(0))]));
        assert!(CpmlBoundary::with_grid(&g, 0.0, []).is_err());
    }
}
