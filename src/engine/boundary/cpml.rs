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
//! above). Only the *choice* of `σ_max` assumes vacuum (`η = Z0`): in a
//! homogeneous dielectric of relative permittivity `ε_r` the attenuation per
//! cell is `√ε_r` times larger, and Taflove & Hagness (ch. 7) recommend
//! scaling the optimum by `1/√ε_r`; set
//! [`Cpml::sigma_factor`](crate::Cpml::sigma_factor) accordingly.
//!
//! # Cost
//!
//! The operator runs serially on the calling thread between the half-steps
//! (the workers are parked), touching four field components per face slab
//! and storing four `f32` `ψ` values per slab node. A model dominated by its
//! PML therefore loses much of the parallel speed-up; a parallel PML pass is
//! future work.

use crate::axis::Axis;
use crate::constants::{EPS0, Z0};
use crate::engine::Engine;
use crate::engine::fields::{FieldArrays, Fields};
use crate::engine::grid::Grid;
use crate::engine::kernel::flush;
use crate::engine::operator::{Operator, StepContext};
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
    /// time step `dt` \[s\] (see the [module docs](self)).
    pub fn new(p: &Cpml, rho: f64, delta: f64, dt: f64) -> Self {
        let g = rho.powf(p.grading_order);
        let sigma = p.sigma_max(Z0, delta) * g;
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
    /// One value per node of the box, in `(i, j, k)` order.
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
        let len = (0..3).map(|d| hi[d] - lo[d]).product();
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
            psi: vec![0.0; len],
        }
    }

    /// Applies the correction. `e_side`: target is `u` (derivative of `i`
    /// backwards), else target is `i` (derivative of `u` forwards).
    /// `coef` is `cb` (E) or `db` (H) of `comp` in global layout; `zero`
    /// is an all-zero x-plane.
    fn apply(
        &mut self,
        tgt: &mut FieldArrays<'_>,
        src: &FieldArrays<'_>,
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
            // The neighbouring x-plane for an x-normal layer.
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
/// PEC on the CPML faces, then register the operator **before** any
/// operator that reads fields inside the layer in `post_h` / `post_e`
/// (probes, dumps), so they see corrected values:
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
#[derive(Debug, Clone)]
pub struct CpmlBoundary {
    nodes: [usize; 3],
    dt: f64,
    layers: Vec<Layer>,
    /// An all-zero x-plane (out-of-domain neighbours).
    zero: Vec<f32>,
}

impl CpmlBoundary {
    /// CPML layers for `engine` (its grid and time step), one per
    /// `(face, parameters)` entry.
    pub fn new(engine: &Engine, layers: impl IntoIterator<Item = (Face, Cpml)>) -> Result<Self> {
        Self::with_grid(engine.grid(), engine.dt(), layers)
    }

    /// CPML layers for a mesh `grid` and time step `dt` \[s\].
    ///
    /// # Errors
    /// [`Error::Invalid`] if parameters are invalid ([`Cpml::validate`]),
    /// a face is listed twice, a layer has more cells than the mesh, or two
    /// opposite layers overlap.
    pub fn with_grid(
        grid: &Grid,
        dt: f64,
        layers: impl IntoIterator<Item = (Face, Cpml)>,
    ) -> Result<Self> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::Invalid(format!("CPML: invalid time step {dt}")));
        }
        let n = grid.nodes();
        let mut cells = [0usize; 6];
        let mut out = Vec::new();
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
            out.push(Layer::new(grid, dt, face, params));
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
            zero: vec![0.0; n[1] * n[2]],
        })
    }

    /// The faces and their parameters, in the order given.
    pub fn layers(&self) -> impl Iterator<Item = (Face, &Cpml)> + '_ {
        self.layers.iter().map(|l| (l.face, &l.params))
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

    fn layer(&self, face: Face) -> Option<&Layer> {
        self.layers.iter().find(|l| l.face == face)
    }

    fn check(&self, ctx: &StepContext<'_>) {
        assert_eq!(ctx.grid.nodes(), self.nodes, "CPML built for another grid");
        assert_eq!(ctx.dt, self.dt, "CPML built for another time step");
    }
}

impl Layer {
    fn new(grid: &Grid, dt: f64, face: Face, params: Cpml) -> Self {
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
        let node_delta = |k: usize| {
            let lo = (k > 0).then(|| grid.delta(p, k - 1));
            let hi = (k + 1 < np).then(|| grid.delta(p, k));
            match (lo, hi) {
                (Some(a), Some(b)) => 0.5 * (a + b),
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => unreachable!("grids have two lines per axis"),
            }
        };
        let e_coeffs: Vec<_> = e_idx
            .iter()
            .map(|&k| CpmlCoefficients::new(&params, rho(k as f64), node_delta(k), dt))
            .collect();
        let h_coeffs: Vec<_> = h_idx
            .iter()
            .map(|&k| CpmlCoefficients::new(&params, rho(k as f64 + 0.5), grid.delta(p, k), dt))
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
            e: [e_slab(a1), e_slab(a2)],
            h: [h_slab(a1), h_slab(a2)],
            e_coeffs,
            h_coeffs,
        }
    }
}

impl Operator for CpmlBoundary {
    fn post_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        self.check(ctx);
        let Fields { u, i } = fields;
        for l in &mut self.layers {
            for s in &mut l.h {
                s.apply(i, u, ctx.coefficients.db(s.comp), &self.zero, false);
            }
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        self.check(ctx);
        let Fields { u, i } = fields;
        for l in &mut self.layers {
            for s in &mut l.e {
                s.apply(u, i, ctx.coefficients.cb(s.comp), &self.zero, true);
            }
        }
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
