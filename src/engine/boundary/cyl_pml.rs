//! Cylindrical PML on the `ρ_max` face of a cylindrical grid (design
//! §5.15.2).
//!
//! The Cartesian CPML stretches `∂/∂ρ` only. An outgoing cylindrical wave
//! also sees the metric `ρ` of the α direction, so a plane-wave stretch in
//! ρ is not a change of coordinates of the cylindrical Maxwell equations
//! and reflects at the order of `1/(kρ)`. The matched layer is the complex
//! coordinate stretch of the radius itself (F. L. Teixeira and W. C. Chew,
//! "PML-FDTD in cylindrical and spherical grids", *IEEE Microwave and
//! Guided Wave Letters* 7(9), 1997; W. C. Chew and W. H. Weedon, "A 3D
//! perfectly matched medium from modified Maxwell's equations with
//! stretched coordinates", *Microwave Opt. Technol. Lett.* 7(13), 1994):
//!
//! ```text
//! ρ̃(ρ) = ρ0 + ∫_{ρ0}^{ρ} s(ρ') dρ',   s = κ + σ/(α + jω·ε0)
//! ```
//!
//! which enters both the radial derivative (`dρ̃ = s·dρ`) and the α metric
//! (`ρ̃` instead of `ρ`).
//!
//! # Integral form: a change of edge lengths
//!
//! In the integral formulation (design §5.2) all geometry sits in the
//! circuit values `C = ε·A*/l` and `L = μ·A/l*`, and the circulations
//! (`Σi`, `Σu`) are topological. Writing Maxwell's equations in the
//! stretched coordinates `(ρ̃, α, z)` and taking the edge integrals over the
//! *stretched* edges, the update keeps its circulations and only the
//! lengths and areas become complex, frequency-dependent:
//!
//! ```text
//! (jω·C + G)·g(ω)·u = Σi,     (jω·L + R)·h(ω)·i = −Σu
//! ```
//!
//! with `g = (Ã*/A*)/(l̃/l)` and `h = (Ã/A)/(l̃*/l*)`. In the cells of the
//! layer (positions as in the table of §5.15):
//!
//! | edge | `g` or `h` | at |
//! |---|---|---|
//! | E_ρ (cell `i`) | `s̄/s` | `s` over cell `i`, `s̄` at `ρ_{i+½}` |
//! | E_α (line `i`) | `s/s̄` | `s` over the dual cell, `s̄` at `ρ_i` |
//! | E_z (line `i`) | `s·s̄₂` | over the dual cell |
//! | H_ρ (line `i`) | `s̄/s` | `s` over the dual cell, `s̄` at `ρ_i` |
//! | H_α (cell `i`) | `s/s̄` | `s` over cell `i`, `s̄` at `ρ_{i+½}` |
//! | H_z (cell `i`) | `s·s̄₂` | over cell `i` |
//!
//! where, for an interval `[ρ_a, ρ_b]`, `s = (ρ̃_b − ρ̃_a)/(ρ_b − ρ_a)` is the
//! mean stretch, `s̄ = ρ̃/ρ` the metric ratio at a point and
//! `s̄₂ = (ρ̃_a + ρ̃_b)/(ρ_a + ρ_b)` (the annular areas give
//! `(ρ̃_b² − ρ̃_a²)/(ρ_b² − ρ_a²) = s·s̄₂`). This is the discrete counterpart
//! of the anisotropic medium `Λ = diag(s̄/s, s/s̄, s·s̄)` of Teixeira & Chew.
//! Outside the layer every factor is exactly 1 and `u`, `i` are the
//! physical edge integrals. Each of `ρ̃` and the interval stretches is of the
//! form `a + b/p` with `p = jω·ε0 + α` (`α` constant over the layer, so the
//! stretch is an exact integral; see [Grading](self#grading)), so every
//! `g⁻¹` and `h⁻¹` is a ratio or product of first-order rational functions
//! of `p`.
//!
//! # Time domain
//!
//! The kernel advances `u` with the plain circulation `S = Σi`:
//! `u ← ca·u + cb·S`. With `(jω·C + G)·u = g⁻¹·S`, the operator adds
//!
//! ```text
//! u ← u + cb·(Y − S),   Y = g⁻¹ ∗ S
//! i ← i − db·(Y − T),   Y = h⁻¹ ∗ T   (T = Σu)
//! ```
//!
//! after the kernel, exactly as the Cartesian CPML does with its single
//! derivative term. `g⁻¹` is applied as a cascade of at most two
//! first-order stages (`s̄/s`, `s/s̄`, or `1/s̄₂` then `1/s`), each
//!
//! ```text
//! F(p) = (A·p + B)/(C·p + D) = A/C + r/(jω + γ),
//! γ = (α + D/C)/ε0,   r = (B − A·D/C)/(C·ε0)
//! ```
//!
//! discretised with the bilinear (trapezoidal) rule
//! `jω → (2/dt)·(1 − z⁻¹)/(1 + z⁻¹)`:
//!
//! ```text
//! y^n = (A/C)·x^n + ψ^n,   ψ^n = β·ψ^{n−1} + w·(x^n + x^{n−1})
//! β = (2 − γ·dt)/(2 + γ·dt),   w = r·dt/(2 + γ·dt)
//! ```
//!
//! with `x^n` the input at the half-step being updated. Every stage is
//! then exact at the warped frequency `Ω = (2/dt)·tan(ω·dt/2)`; all stages
//! share that warp, so the semi-discrete layer is still an exact stretch
//! (with `s(Ω)`) and adds no time-discretisation reflection. The
//! first-order recursive convolution of the Cartesian CPML (Roden &
//! Gedney) costs one state less but was measured 20 dB worse at
//! `ρ = 26 mm`, improving by 4.5 dB per halving of `dt`. All poles are at
//! `γ ≥ α/ε0 > 0`, so `|β| < 1`.
//!
//! # Late-time stability
//!
//! With `α = 0` the stretch `s` is infinite at DC, and some factors carry
//! `s` in the numerator (on the interface line `H_ρ` sees `s_dual/s̄ =
//! s_dual`, a pure integrator). The static fields left by a source with
//! DC content then grow slowly (quadratically in energy), the weak
//! instability known from unsplit PMLs. The layer therefore always uses a
//! complex frequency shift, `α = max(α_max, ALPHA_FLOOR·σ_max)` with
//! [`ALPHA_FLOOR`] = 10⁻³ and `σ_max` of the first layer cell. Its corner
//! frequency `α/(2π·ε0)` is about 1 % of the frequency at which the cells
//! are λ/20, far below any band the mesh resolves, so the absorption there
//! is unchanged.
//!
//! # Grading
//!
//! `σ`, `κ` follow the Cartesian CPML profile in mesh-index depth
//! `t = (x − x0)/n` (`x` the fractional ρ index, `x0` the interface line,
//! `n` the cells): `σ = σ_max(Δ)·t^m`, `κ = 1 + (κ_max − 1)·t^m`, with the
//! layer's refractive index in `σ_max`. Within each cell `ρ` is linear in
//! `x`, so the integrals are exact: `∫σ dρ = Σ_tot·t^{m+1}` with
//! `Σ_tot = σ_max(Δ)·Δ·n/(m + 1)` (independent of `Δ`), and `∫(κ − 1) dρ`
//! summed cell by cell. The frequency shift is a constant `α` over the
//! layer (a graded `α` would make `ρ̃` non-rational in `p`; see
//! [Late-time stability](self#late-time-stability) for its value).
//!
//! # Other faces
//!
//! The layer handles every component in the x-planes `x0..` (the whole
//! layer). Where a Cartesian CPML of the α or z faces overlaps it (domain
//! edges and corners), the stretch of that face is a stretch of `∂/∂α` or
//! `∂/∂z` in the same coordinates: the derivative term along that axis is
//! replaced by `(1/κ_q)·D + ψ_q` (the face's own recursion and grading)
//! *before* `g⁻¹` is applied to the sum, which is the exact two-axis
//! stretch. The Cartesian layers are cut off at `x0` there
//! ([`CpmlBoundary`](super::CpmlBoundary) does that). Simply adding the
//! two corrections instead was measured to be unstable.
//!
//! # Parallel execution
//!
//! The layer's x-planes are split at the engine's slab boundaries like
//! the CPML's; each node's state and arithmetic are the same in any
//! piece, so results are bit-identical for any thread count. Per step it
//! recomputes both circulation terms of every component in its `n + 1`
//! planes and stores six `f32` values per node and component.

use crate::axis::Axis;
use crate::constants::{EPS0, Z0};
use crate::engine::grid::Grid;
use crate::engine::kernel::flush;
use crate::engine::slab::{SlabMut, SlabRef};
use crate::fdtd::Cpml;

/// A complex length or ratio `a + b/p`, `p = jω·ε0 + α`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Len {
    a: f64,
    b: f64,
}

impl Len {
    fn add(self, o: Len) -> Len {
        Len {
            a: self.a + o.a,
            b: self.b + o.b,
        }
    }
    fn sub(self, o: Len) -> Len {
        Len {
            a: self.a - o.a,
            b: self.b - o.b,
        }
    }
    fn scale(self, f: f64) -> Len {
        Len {
            a: self.a * f,
            b: self.b * f,
        }
    }
}

const ONE: Len = Len { a: 1.0, b: 0.0 };

/// One first-order stage `y = k·x + ψ`, `ψ ← β·ψ + w·x`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Stage {
    k: f32,
    beta: f32,
    w: f32,
}

impl Stage {
    const IDENTITY: Stage = Stage {
        k: 1.0,
        beta: 0.0,
        w: 0.0,
    };

    /// The stage `F = num/den`, bilinear form (see the
    /// [module docs](self#time-domain)).
    fn ratio(num: Len, den: Len, alpha: f64, dt: f64) -> Stage {
        let (a, b, c, d) = (num.a, num.b, den.a, den.b);
        let k = a / c;
        let gamma = (alpha + d / c) / EPS0;
        let r = (b - a * d / c) / (c * EPS0);
        let x = gamma * dt;
        let (beta, w) = ((2.0 - x) / (2.0 + x), r * dt / (2.0 + x));
        Stage {
            k: k as f32,
            beta: beta as f32,
            w: w as f32,
        }
    }
}

/// The stretched radius of a layer: `ρ̃(x) = ρ(x) + K(x) + Σ(x)/p` at
/// fractional ρ index `x`.
struct Profile<'a> {
    rho: &'a [f64],
    /// Interface line `x0` and number of cells `n`.
    inner: usize,
    cells: f64,
    m: f64,
    /// `Σ_tot = ∫σ dρ` over the whole layer.
    sigma_tot: f64,
    /// `K(x)` at each line (0 up to the interface).
    kappa_cum: Vec<f64>,
    kappa_max: f64,
}

impl<'a> Profile<'a> {
    fn new(rho: &'a [f64], inner: usize, p: &Cpml, n_r: f64) -> Self {
        let cells = p.cells as f64;
        let m = p.grading_order;
        let sigma_tot = p.sigma_max(n_r * Z0, 1.0) * cells / (m + 1.0);
        let mut s = Self {
            rho,
            inner,
            cells,
            m,
            sigma_tot,
            kappa_cum: vec![0.0; rho.len()],
            kappa_max: p.kappa_max,
        };
        for c in 0..rho.len() - 1 {
            let dk = s.kappa_in_cell(c, c as f64 + 1.0);
            s.kappa_cum[c + 1] = s.kappa_cum[c] + dk;
        }
        s
    }

    /// Normalised depth `t ∈ [0, 1]`.
    fn t(&self, x: f64) -> f64 {
        ((x - self.inner as f64) / self.cells).clamp(0.0, 1.0)
    }

    /// `∫(κ − 1) dρ` from line `c` to `x ∈ [c, c + 1]`.
    fn kappa_in_cell(&self, c: usize, x: f64) -> f64 {
        let d = self.rho[c + 1] - self.rho[c];
        let e = self.m + 1.0;
        d * (self.kappa_max - 1.0) * self.cells / e * (self.t(x).powf(e) - self.t(c as f64).powf(e))
    }

    /// Cell containing `x` (the last cell for the outer line).
    fn cell(&self, x: f64) -> usize {
        (x.floor() as usize).min(self.rho.len() - 2)
    }

    fn rho(&self, x: f64) -> f64 {
        let c = self.cell(x);
        self.rho[c] + (x - c as f64) * (self.rho[c + 1] - self.rho[c])
    }

    fn tilde(&self, x: f64) -> Len {
        let c = self.cell(x);
        let k = self.kappa_cum[c] + self.kappa_in_cell(c, x);
        Len {
            a: self.rho(x) + k,
            b: self.sigma_tot * self.t(x).powf(self.m + 1.0),
        }
    }

    /// Metric ratio `s̄ = ρ̃/ρ` at `x`.
    fn sbar(&self, x: f64) -> Len {
        self.tilde(x).scale(1.0 / self.rho(x))
    }

    /// Mean stretch `s` over `[xa, xb]`.
    fn s(&self, xa: f64, xb: f64) -> Len {
        self.tilde(xb)
            .sub(self.tilde(xa))
            .scale(1.0 / (self.rho(xb) - self.rho(xa)))
    }

    /// `s̄₂ = (ρ̃_a + ρ̃_b)/(ρ_a + ρ_b)`.
    fn sbar2(&self, xa: f64, xb: f64) -> Len {
        self.tilde(xa)
            .add(self.tilde(xb))
            .scale(1.0 / (self.rho(xa) + self.rho(xb)))
    }
}

/// The stages of `g⁻¹` (E, `e_side`) or `h⁻¹` (H) of `comp` at ρ index `i`.
fn stages(
    prof: &Profile<'_>,
    comp: Axis,
    e_side: bool,
    i: usize,
    alpha: f64,
    dt: f64,
) -> [Stage; 2] {
    let last = (prof.rho.len() - 1) as f64;
    let x = i as f64;
    // Cell i and the dual cell of line i (clamped to the domain).
    let cell = (x, x + 1.0);
    let dual = ((x - 0.5).max(0.0), (x + 0.5).min(last));
    let r = |n: Len, d: Len| Stage::ratio(n, d, alpha, dt);
    // On a cell (E_ρ, H_α, H_z) or on a line (E_α, E_z, H_ρ).
    match (comp, e_side) {
        // g = s̄/s (E_ρ): g⁻¹ = s_cell / s̄(i + ½).
        (Axis::X, true) => [
            r(prof.s(cell.0, cell.1), prof.sbar(x + 0.5)),
            Stage::IDENTITY,
        ],
        // h = s̄/s (H_ρ): h⁻¹ = s_dual / s̄(i).
        (Axis::X, false) => [r(prof.s(dual.0, dual.1), prof.sbar(x)), Stage::IDENTITY],
        // g = s/s̄ (E_α): g⁻¹ = s̄(i) / s_dual.
        (Axis::Y, true) => [r(prof.sbar(x), prof.s(dual.0, dual.1)), Stage::IDENTITY],
        // h = s/s̄ (H_α): h⁻¹ = s̄(i + ½) / s_cell.
        (Axis::Y, false) => [
            r(prof.sbar(x + 0.5), prof.s(cell.0, cell.1)),
            Stage::IDENTITY,
        ],
        // g = s·s̄₂ over the dual cell (E_z), h likewise over the cell (H_z).
        (Axis::Z, _) => {
            let (a, b) = if e_side { dual } else { cell };
            [r(ONE, prof.sbar2(a, b)), r(ONE, prof.s(a, b))]
        }
    }
}

/// Ratio of the CFS floor of the frequency shift to `σ_max` (see the
/// [module docs](self#late-time-stability)).
pub const ALPHA_FLOOR: f64 = 1e-3;

/// The constant frequency shift `α` \[S/m\] of the ρ_max layer `p`
/// (refractive index `n_r`) on `grid`: `max(α_max, ALPHA_FLOOR·σ_max)`
/// with `σ_max` of the first layer cell.
pub(crate) fn alpha(grid: &Grid, p: &Cpml, n_r: f64) -> f64 {
    let n = grid.nodes()[0];
    let d = grid.delta(Axis::X, n - 1 - p.cells);
    p.alpha_max.max(ALPHA_FLOOR * p.sigma_max(n_r * Z0, d))
}

/// The recursion of a Cartesian CPML stretch along one transverse axis,
/// per node index along that axis: `[β, w, 1/κ − 1]` (all 0 outside the
/// layers).
pub(crate) type Transverse = [Vec<f32>; 3];

/// One field component of the layer (or of its part in one x-slab).
#[derive(Debug, Clone, PartialEq)]
struct Comp {
    comp: Axis,
    e_side: bool,
    /// x-planes `x_lo..x_hi`, and the j, k ranges updated by the kernel.
    x: (usize, usize),
    j: (usize, usize),
    k: (usize, usize),
    /// Per x-plane (from `x.0`): the two stages.
    st: Vec<[Stage; 2]>,
    /// Transverse stretches of the two derivative terms (`D_next` of
    /// `prev` and `D_prev` of `next`), if that axis has CPML faces.
    tr: [Option<Transverse>; 2],
    /// `[ψ_stage0, x_stage0, ψ_stage1, x_stage1, ψ_tr0, ψ_tr1]` per node
    /// of the box (`x` the previous stage input).
    psi: Vec<[f32; 6]>,
}

/// The cylindrical PML of the `ρ_max` face (see the [module docs](self)).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CylLayer {
    e: Vec<Comp>,
    h: Vec<Comp>,
}

impl CylLayer {
    /// The layer of `p` (refractive index `n_r`) on the `ρ_max` face of
    /// `grid` at time step `dt`. `transverse(e_side, comp, axis)` gives the
    /// Cartesian CPML stretch of the α or z faces, if any. The grid must
    /// be cylindrical with the interface radius `> 0` (checked by the
    /// caller).
    pub(crate) fn new(
        grid: &Grid,
        dt: f64,
        p: &Cpml,
        n_r: f64,
        mut transverse: impl FnMut(bool, Axis, Axis) -> Option<Transverse>,
    ) -> Self {
        let n = grid.nodes();
        let rho = grid.lines(Axis::X);
        let inner = n[0] - 1 - p.cells;
        let prof = Profile::new(rho, inner, p, n_r);
        let alpha = alpha(grid, p, n_r);
        let mut comp = |a: Axis, e_side: bool| {
            // The kernel's ranges (no padding entries).
            let full = |d: Axis| {
                let pad = if e_side { d == a } else { d != a };
                (0, n[d.index()] - usize::from(pad))
            };
            let mut x = full(Axis::X);
            x.0 = inner;
            let st = (x.0..x.1)
                .map(|i| stages(&prof, a, e_side, i, alpha, dt))
                .collect();
            let (nx, pv) = (a.next(), a.prev());
            let tr = [nx, pv].map(|d| {
                if d == Axis::X {
                    None
                } else {
                    transverse(e_side, a, d)
                }
            });
            Comp {
                comp: a,
                e_side,
                x,
                j: full(Axis::Y),
                k: full(Axis::Z),
                st,
                tr,
                psi: Vec::new(),
            }
        };
        let e = Axis::ALL.map(|a| comp(a, true)).to_vec();
        let h = Axis::ALL.map(|a| comp(a, false)).to_vec();
        Self { e, h }
    }

    /// The H-side stretch of the face normal to `a` at node `p`, if it is
    /// in the layer: what the layer applies to that face's circulation,
    /// for operators that update a face themselves (conformal merge
    /// groups, whose kernel `db` is 0 so the layer's correction vanishes).
    pub(crate) fn h_face(&self, a: Axis, p: [usize; 3]) -> Option<FaceStretch> {
        Self::stretch(&self.h, a, p)
    }

    /// The E-side stretch of the edge along `a` at node `p`, if it is in
    /// the layer ([`h_face`](Self::h_face) for E edges: conformal sheet
    /// pieces, whose kernel `cb` is 0). Its
    /// [`correction`](FaceStretch::correction) of the edge's derivative
    /// terms `x1` (`D_next` of `i_prev`) and `x2` (`D_prev` of `i_next`) is
    /// what the layer adds: `u ← u + cb·(Y − S)`.
    pub(crate) fn e_edge(&self, a: Axis, p: [usize; 3]) -> Option<FaceStretch> {
        Self::stretch(&self.e, a, p)
    }

    fn stretch(comps: &[Comp], a: Axis, p: [usize; 3]) -> Option<FaceStretch> {
        let c = comps.iter().find(|c| c.comp == a)?;
        let inside = |r: (usize, usize), x: usize| (r.0..r.1).contains(&x);
        if !(inside(c.x, p[0]) && inside(c.j, p[1]) && inside(c.k, p[2])) {
            return None;
        }
        let tco = |which: usize, d: Axis| -> [f32; 3] {
            match &c.tr[which] {
                None => [0.0; 3],
                Some(v) => {
                    let ix = p[d.index()];
                    [v[0][ix], v[1][ix], v[2][ix]]
                }
            }
        };
        Some(FaceStretch {
            st: c.st[p[0] - c.x.0],
            tr: [tco(0, a.next()), tco(1, a.prev())],
        })
    }

    /// The part in the x-planes `x0..x1` (with zeroed state).
    pub(crate) fn clip(&self, x0: usize, x1: usize) -> Self {
        let clip = |c: &Comp| {
            let (lo, hi) = (c.x.0.max(x0), c.x.1.min(x1));
            (lo < hi).then(|| {
                let len = (hi - lo) * (c.j.1 - c.j.0) * (c.k.1 - c.k.0);
                Comp {
                    x: (lo, hi),
                    st: c.st[lo - c.x.0..hi - c.x.0].to_vec(),
                    psi: vec![[0.0; 6]; len],
                    ..c.clone()
                }
            })
        };
        Self {
            e: self.e.iter().filter_map(clip).collect(),
            h: self.h.iter().filter_map(clip).collect(),
        }
    }

    /// The correction after the E kernel (`coef(axis)` = `cb`).
    pub(crate) fn post_e<'c>(
        &mut self,
        u: &mut SlabMut<'_>,
        i: &SlabRef<'_>,
        coef: impl Fn(Axis) -> &'c [f32],
        zero: &[f32],
    ) {
        for c in &mut self.e {
            c.apply(u, i, coef(c.comp), zero);
        }
    }

    /// The correction after the H kernel (`coef(axis)` = `db`).
    pub(crate) fn post_h<'c>(
        &mut self,
        i: &mut SlabMut<'_>,
        u: &SlabRef<'_>,
        coef: impl Fn(Axis) -> &'c [f32],
        zero: &[f32],
    ) {
        for c in &mut self.h {
            c.apply(i, u, coef(c.comp), zero);
        }
    }
}

/// The layer's H-side stretch of one face ([`CylLayer::h_face`]): the two
/// stages of `h⁻¹` and the transverse CPML recursions `[β, w, 1/κ − 1]` of
/// its two derivative terms (zero without a transverse layer).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FaceStretch {
    st: [Stage; 2],
    tr: [[f32; 3]; 2],
}

impl FaceStretch {
    /// `Y − T` for the derivative terms `x1` (`D_next` of the `prev`
    /// component) and `x2` (`D_prev` of `next`), `T = x1 − x2`, with the
    /// per-face state `psi`; the same arithmetic as the layer's own
    /// update, so `i ← da·i − db·(T + (Y − T))` is the stretched update.
    pub(crate) fn correction(&self, psi: &mut [f32; 6], x1: f32, x2: f32) -> f32 {
        let [s0, s1] = self.st;
        let s = x1 - x2;
        let [b1, w1, k1] = self.tr[0];
        psi[4] = flush(b1 * psi[4] + w1 * x1);
        let y1 = x1 + k1 * x1 + psi[4];
        let [b2, w2, k2] = self.tr[1];
        psi[5] = flush(b2 * psi[5] + w2 * x2);
        let y2 = x2 + k2 * x2 + psi[5];
        let sp = y1 - y2;
        psi[0] = flush(s0.beta * psi[0] + s0.w * (sp + psi[1]));
        psi[1] = sp;
        let y = s0.k * sp + psi[0];
        psi[2] = flush(s1.beta * psi[2] + s1.w * (y + psi[3]));
        psi[3] = y;
        let y = s1.k * y + psi[2];
        y - s
    }
}

/// The derivative of component `c` along `d` at x-plane `i`, row `j`, for
/// `k` in `k0..k0 + out.len()`: backward (`f[p] − f[p − e_d]`, zero outside
/// the domain) on the E side, forward on the H side.
#[allow(clippy::too_many_arguments)]
fn deriv(
    out: &mut [f32],
    src: &SlabRef<'_>,
    zero: &[f32],
    c: Axis,
    d: Axis,
    i: usize,
    j: usize,
    k0: usize,
    e_side: bool,
) {
    let nz = src.nodes()[2];
    let o = j * nz + k0;
    let len = out.len();
    let cur = src.plane(c, i);
    match d {
        Axis::X => {
            if e_side {
                let prev = if i == 0 { zero } else { src.plane(c, i - 1) };
                for (q, v) in out.iter_mut().enumerate() {
                    *v = cur[o + q] - prev[o + q];
                }
            } else {
                let next = src.plane(c, i + 1);
                for (q, v) in out.iter_mut().enumerate() {
                    *v = next[o + q] - cur[o + q];
                }
            }
        }
        Axis::Y => {
            if !e_side {
                let (h, l) = (&cur[o + nz..o + nz + len], &cur[o..o + len]);
                for (v, (a, b)) in out.iter_mut().zip(h.iter().zip(l)) {
                    *v = a - b;
                }
            } else if j == 0 {
                out.copy_from_slice(&cur[o..o + len]);
            } else {
                let (h, l) = (&cur[o..o + len], &cur[o - nz..o - nz + len]);
                for (v, (a, b)) in out.iter_mut().zip(h.iter().zip(l)) {
                    *v = a - b;
                }
            }
        }
        Axis::Z => {
            for (q, v) in out.iter_mut().enumerate() {
                let p = o + q;
                *v = if !e_side {
                    cur[p + 1] - cur[p]
                } else if k0 + q == 0 {
                    cur[p]
                } else {
                    cur[p] - cur[p - 1]
                };
            }
        }
    }
}

impl Comp {
    fn apply(&mut self, tgt: &mut SlabMut<'_>, src: &SlabRef<'_>, coef: &[f32], zero: &[f32]) {
        let n = src.nodes();
        let (nz, pl) = (n[2], n[1] * n[2]);
        let (a, e_side) = (self.comp, self.e_side);
        let (dn, dp) = (a.next(), a.prev());
        // E: u += cb·(Y − S); H: i −= db·(Y − T).
        let scale: f32 = if e_side { 1.0 } else { -1.0 };
        let len = self.k.1 - self.k.0;
        let (mut d1, mut d2) = (vec![0.0f32; len], vec![0.0f32; len]);
        let mut q = 0;
        for i in self.x.0..self.x.1 {
            let [s0, s1] = self.st[i - self.x.0];
            let cf = &coef[i * pl..(i + 1) * pl];
            for j in self.j.0..self.j.1 {
                // Term 1: D_next of comp prev; term 2: D_prev of comp next.
                deriv(&mut d1, src, zero, dp, dn, i, j, self.k.0, e_side);
                deriv(&mut d2, src, zero, dn, dp, i, j, self.k.0, e_side);
                let o = j * nz + self.k.0;
                let t = &mut tgt.plane_mut(a, i)[o..o + len];
                let cf = &cf[o..o + len];
                let psi = &mut self.psi[q..q + len];
                q += len;
                // Transverse coefficients: per j (Y) or per k (Z).
                let tco = |which: usize, d: Axis, kk: usize| -> [f32; 3] {
                    match &self.tr[which] {
                        None => [0.0; 3],
                        Some(v) => {
                            let ix = if d == Axis::Y { j } else { self.k.0 + kk };
                            [v[0][ix], v[1][ix], v[2][ix]]
                        }
                    }
                };
                for kk in 0..len {
                    let (x1, x2) = (d1[kk], d2[kk]);
                    let s = x1 - x2;
                    let ps = &mut psi[kk];
                    let [b1, w1, k1] = tco(0, dn, kk);
                    ps[4] = flush(b1 * ps[4] + w1 * x1);
                    let y1 = x1 + k1 * x1 + ps[4];
                    let [b2, w2, k2] = tco(1, dp, kk);
                    ps[5] = flush(b2 * ps[5] + w2 * x2);
                    let y2 = x2 + k2 * x2 + ps[5];
                    let sp = y1 - y2;
                    ps[0] = flush(s0.beta * ps[0] + s0.w * (sp + ps[1]));
                    ps[1] = sp;
                    let y = s0.k * sp + ps[0];
                    ps[2] = flush(s1.beta * ps[2] + s1.w * (y + ps[3]));
                    ps[3] = y;
                    let y = s1.k * y + ps[2];
                    t[kk] = flush(t[kk] + scale * cf[kk] * (y - s));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    fn mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    fn div(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        let d = b.0 * b.0 + b.1 * b.1;
        ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
    }

    #[test]
    fn stage_is_exact_at_the_warped_frequency() {
        // F = num/den in p = jΩ·ε0 + α against the recursion's transfer
        // function k + w·(1 + z⁻¹)/(1 − β·z⁻¹) at z = e^{jω·dt}.
        let (dt, alpha) = (1e-12, 0.05);
        let cases = [
            (ONE, Len { a: 2.0, b: 3.0 }),
            (Len { a: 1.5, b: 4.0 }, Len { a: 1.1, b: 0.2 }),
            (Len { a: 1.0, b: 2.0 }, ONE),
        ];
        for (num, den) in cases {
            let st = Stage::ratio(num, den, alpha, dt);
            let (k, beta, w) = (st.k as f64, st.beta as f64, st.w as f64);
            for f in [1e8, 3e9, 3e10, 1e11] {
                let wdt = TAU * f * dt;
                let zi = ((-wdt).cos(), (-wdt).sin());
                let h = div(
                    mul((w, 0.0), (1.0 + zi.0, zi.1)),
                    (1.0 - beta * zi.0, -beta * zi.1),
                );
                let h = (k + h.0, h.1);
                let p = (alpha, 2.0 / dt * (wdt / 2.0).tan() * EPS0);
                let want = div(
                    (num.a * p.0 + num.b, num.a * p.1),
                    (den.a * p.0 + den.b, den.a * p.1),
                );
                let err = (h.0 - want.0).hypot(h.1 - want.1) / want.0.hypot(want.1);
                assert!(err < 1e-5, "{num:?}/{den:?} at {f:e}: {h:?} vs {want:?}");
            }
        }
    }

    #[test]
    fn profile_integrals() {
        let rho: Vec<f64> = (0..=20).map(|i| 0.01 + i as f64 * 1e-3).collect();
        let p = Cpml::cells(8).kappa_max(3.0);
        let prof = Profile::new(&rho, 12, &p, 1.0);
        // Nothing before the interface.
        assert_eq!(prof.tilde(12.0), Len { a: rho[12], b: 0.0 });
        assert_eq!(prof.sbar(5.5), ONE);
        // Total σ and κ integrals.
        let t = prof.tilde(20.0);
        let smax = p.sigma_max(Z0, 1e-3);
        let want_s = smax * 8e-3 / 4.0;
        assert!((t.b / want_s - 1.0).abs() < 1e-12);
        let want_k = rho[20] + 2.0 * 8e-3 / 4.0;
        assert!((t.a / want_k - 1.0).abs() < 1e-12);
        // Mean stretch of the outer cell approaches the outer σ, κ.
        let s = prof.s(19.0, 20.0);
        let mean_t3: f64 = (8f64.powi(4) - 7f64.powi(4)) / 4.0 / 8f64.powi(3);
        assert!((s.b / (smax * mean_t3) - 1.0).abs() < 1e-12);
    }
}
