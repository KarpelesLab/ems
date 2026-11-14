//! Conducting sheets: thin lossy conductors of finite thickness lying in a
//! mesh plane, as an engine operator.
//!
//! # Physical model
//!
//! A conducting sheet is a slab of conductivity `σ` and thickness `t` that
//! is much thinner than the mesh cells, so it is drawn with zero thickness
//! in a mesh plane. Inside a good conductor the displacement current is
//! negligible next to the conduction current and the tangential fields
//! obey the one-dimensional diffusion (skin-effect) equations across the
//! thickness (`z` normal to the sheet, `e^{jωt}`):
//!
//! ```text
//! dE/dz = −jωμ0·H,      dH/dz = −(σ + jωε0)·E
//! ```
//!
//! so per unit square the slab is a two-port, a uniform line of
//! propagation constant `γ = √(jωμ0·(σ + jωε0))` and wave impedance
//! `Z_c = √(jωμ0/(σ + jωε0))` (Ramo, Whinnery and Van Duzer, *Fields and
//! Waves in Communication Electronics*, 3rd ed., §3.16–3.17 on skin effect
//! and shielding; Schelkunoff's transmission-line view of shields). With
//! `E±`, `H±` the tangential fields at its two faces (`H` the current
//! flowing into the face):
//!
//! ```text
//! E+ = Z11·H+ + Z12·H−,    Z11 = Z_c·coth(γt),   Z12 = Z_c·csch(γt)
//! ```
//!
//! This covers the whole range of thickness against skin depth
//! `δ = √(2/(ωμ0σ))`:
//!
//! * `t ≪ δ`: `Z11 ≈ Z12 ≈ 1/(σt)`, a **resistive sheet** of sheet
//!   conductance `σ·t`; a plane wave is transmitted with
//!   `T = 2/(2 + η0·σ·t)`;
//! * `t ≫ δ`: each face is a **good-conductor surface impedance**
//!   `Z11 → Z_c ≈ (1 + j)/(σδ)` (skin-effect loss `R_s = 1/(σδ)` on the
//!   side that carries the current), and the transmission falls like
//!   `e^{−t/δ}`.
//!
//! A single shunt admittance on the sheet edge (`u` common to both faces)
//! can only reproduce one of these limits: the shielding needs
//! `1/Z12`, the surface loss `1/Z11`, and the two differ by `cosh(γt)`.
//! The model therefore keeps both faces.
//!
//! # Discretisation: a 1D sub-grid across the thickness
//!
//! Each sheet edge `u` (direction `d`, sheet normal `n`, in-plane
//! transverse direction `t`) is split into the voltages of its two faces,
//! `u+` (side of higher index along `n`) and `u−`, joined by a ladder
//! that discretises the slab across its thickness: `N` layers of
//! thickness `Δz_k`, nodes at the layer boundaries carrying the shunt
//! conductance `G_k = σ·V_k·g` and capacitance `C_k = ε0·V_k·g`
//! (`V_k` the half layers around node `k`, `g = w/l` with `w` the dual
//! length of the edge along `t` and `l` its length), and series
//! inductances `L_k = μ0·Δz_k/g` carrying the current `h_k` through each
//! layer. This is the finite-volume (circuit) form of the diffusion
//! equations above — the same contour-path discretisation as the Yee
//! scheme itself, refined in the normal direction inside the conductor —
//! so it converges to the exact two-port with `O((Δz/δ)²)`.
//!
//! The FDTD dual face of the edge is split at the sheet into its two
//! halves. The `+` half holds the share `C+` of the edge capacitance and
//! `G+` of its conductance (by default the half-cell lengths along `n`;
//! the compile step passes material-aware shares) and is bounded by the
//! dual edge on the `+` side (`H_t` above the sheet) and the `+` parts of
//! the two dual edges along `n` that cross the sheet (weights `f± =
//! Δ±/(Δ+ + Δ−)`, the share of their length on each side). Ampère's law on
//! each half closes on the ladder current at the face. Dually, the H
//! updates see `u−` for the dual edge on the `−` side and `f+·u+ + f−·u−`
//! for the crossing dual edges. The weights are the same in both laws, so
//! the coupled system keeps the symmetric (energy-conserving) structure of
//! the Yee scheme.
//!
//! The layers are graded from both faces: the outer layer is
//! [`SURFACE_LAYER`]`·δ_min` thick (`δ_min` the skin depth at the run's
//! highest frequency `f_max`), each further one [`GROWTH`] times thicker,
//! up to the middle. Since at depth `z` only frequencies with
//! `δ(f) ≳ z` still carry field, geometric grading keeps `Δz/δ` small for
//! all frequencies at a cost of `O(log(t/δ_min))` layers. A sheet thinner
//! than two surface layers gets uniform layers (one layer when
//! `t ≤ SURFACE_LAYER·δ_min`). With the defaults the face impedance `Z11`
//! is within about 1 % of the exact one from DC to `f_max`
//! ([`model_abcd`] vs [`slab_abcd`]); e.g. 35 µm of copper with
//! `f_max = 10 GHz` uses 48 layers, a 1 µm sheet 8.
//!
//! # Cylindrical grids
//!
//! On a cylindrical grid (design §5.15) the sheet lies on a `ρ = const`,
//! `α = const` or `z = const` surface and the ladder uses physical lengths:
//! `l` is [`Grid::edge_length`](super::Grid::edge_length) and `w` the width
//! of the edge's strip of the sheet, the arc `ρ·Δα*` at the sheet for an α
//! width (`ρ_{i+½}` for a ρ edge, `ρ_i` for a z edge on a ρ sheet). The
//! default capacitance share of a z edge on a `ρ = const` sheet is the
//! share of its annular dual face, `(ρ_hi² − ρ²)/(ρ_hi² − ρ_lo²)`. An edge
//! on the α seam of a closed grid exists in two stored copies
//! ([`e_edge_copies`](super::cylindrical::e_edge_copies)); it gets one
//! ladder, fed by the dual edges around both copies, and its face voltage
//! is written to both. For a sheet crossing the seam the copies hold the
//! two halves of the strip (`g` summed); for a sheet *on* the seam (normal
//! α) they are its two sides, so the `+` share of the capacitance and
//! conductance is that of the `α = 0` copy (its kernel coefficients,
//! materials included). Copies of an axis z edge (the dual face is the whole
//! disc around the axis) are skipped.
//!
//! # Outlines between mesh lines
//!
//! Where the outline of a sheet crosses the cells
//! ([`ConductingSheet::cut`], [`ConductingSheet::cut_face`]; the compile
//! step declares both for slanted outlines with conformal PEC, design
//! §5.13.2), an edge the sheet covers partly is two parts in series
//! carrying the same current: the open part, of length `f·l`, is an
//! ordinary edge with the capacitance `C/f` (the Dey–Mittra edge of
//! [`crate::engine::conformal`], whose scaled coefficient the kernel
//! holds), and the covered part, of length `(1 − f)·l`, is the ladder
//! with `g = w/((1 − f)·l)` between the two face voltages, which take the
//! capacitance `C/(1 − f)` of that part of the dual face. Both are driven
//! by the same dual edges, and those see the sum of the two voltages, so
//! the system stays symmetric. A perfectly conducting sheet then gives the
//! conformal PEC edge, `f → 0` the whole sheet edge.
//!
//! A face of the sheet's plane that the sheet covers in part (a *cut
//! face*) keeps the sheet edges in its contour: the current that wraps
//! around the sheet's rim is the `H` of the open part of that face, and
//! it flows through the ladders of the edges along the rim, which is
//! where a strip's loss is largest. Two things follow:
//!
//! * *The array value.* The small faces and the hosts of conformal PEC
//!   are computed by its operator (`db = 0` in the kernel), which reads
//!   the array and cannot be corrected afterwards. A sheet edge next to
//!   such a face stores what crossing dual edges need,
//!   `f+·u+ + f−·u−`, instead of `u+`, and the corrections of its other
//!   dual edges are taken against that. (With `u+` in the array such a
//!   face drove the two faces by `f±` and read `u+` alone, which is not
//!   passive: the run diverged.)
//! * *Stability.* A cut face has less inductance than a whole one (its
//!   open area, or the part its deviation current keeps), and the open
//!   parts of its edges have more capacitance, in proportion; the face
//!   voltages of the ladders do not, and resonated with the small faces
//!   above the Courant limit (a 45° strip diverged within 600 steps).
//!   The capacitances of the two faces of every sheet edge around a cut
//!   face are therefore multiplied by [`CUT_FACE_LOADING`] (measured on
//!   strips at six angles: 1 diverges on most, 2 on one, 4 and 8 on none,
//!   at a stability factor of 0.987). On a good conductor these voltages
//!   are `Z_s/η` of the fields around, so the energy this adds is
//!   negligible (4 and 8 give the same frequency and Q to four digits);
//!   on a resistive film it is a capacitance along the rim.
//!
//! # Time stepping
//!
//! The ladder and the two face voltages are integrated with the
//! trapezoidal rule centred at `(n + ½)·dt`, like the semi-implicit loss
//! term of the Yee E update (and the dispersive and lumped operators),
//! with the H currents at `(n + ½)·dt` as the drive. Per edge and step this
//! is one symmetric, diagonally dominant tridiagonal system in the mean
//! node voltages `ū = (u^{n+1} + u^n)/2`; its factorisation is computed
//! once at installation. The scheme is unconditionally stable for the
//! ladder (any `σ`, `t`, `Δz`) and adds no time-step limit: the explicit
//! coupling to the H currents has the same Courant bound as an ordinary
//! edge (split capacitances with split coupling weights).
//!
//! # Validity
//!
//! * Valid for any `t/δ`, provided the sheet is thin compared to the cells
//!   (it occupies no volume in the mesh) and `σ ≫ ωε0` is not required
//!   (`ε0` is kept), but the conductor's permittivity and permeability are
//!   those of vacuum.
//! * Lateral current flow inside the sheet is carried by the ordinary
//!   in-plane mesh (each edge has its own ladder); the crossing dual edges
//!   see the thickness-averaged face voltage.
//! * Accuracy above `f_max` degrades gracefully (the ladder remains
//!   passive).
//!
//! # Implementation
//!
//! The operator uses the serial `post_h` / `post_e` hooks, so it acts
//! after every slab hook (CPML, dispersive media) and before serial
//! operators registered later (sources, lumped elements). `post_h`
//! corrects the H currents on the `−` side for `u− − u+` (the H kernel
//! used the array value, which holds `u+`); `post_e` recomputes the face
//! and ladder voltages from scratch (the kernel result on the sheet edges
//! is discarded) and writes `u+` back. Results do not depend on the thread
//! count. Sheet edges inside a CPML lose the CPML correction (the compile
//! step warns about it).
//!
//! # Energy
//!
//! The coupled scheme conserves (lossless ladder) or dissipates
//!
//! ```text
//! W^n = ½·Σ C·u² (other edges) + Σ_sheet edges (½·Σ_k C_k·u_k² + ½·Σ_k L_k·h_k²)
//!       + ½·Σ L·i^{n−½}·i^{n+½}
//! ```
//!
//! with the face capacitances `C_k` including the split edge capacitance
//! and `i^{n+½}` the H currents *after* the `post_h` correction. The
//! engine's field sum ([`Engine::energy`](super::Engine::energy),
//! [`Engine::step_with_energy`](super::Engine::step_with_energy)) uses
//! `C·u+²` on the sheet edge and the H kernel result `i*^{n+½}` before the
//! correction `i^{n+½} = i*^{n+½} + corr·(u− − u+)^n`. [`Operator::energy`]
//! therefore reports
//!
//! ```text
//! Σ_k ½·C_k·u_k² − ½·C·u+² + Σ_k ½·L_k·h_k²  +  Σ_c ½·L_c·corr_c·i_c^{n−½}·(u− − u+)^n
//! ```
//!
//! (the last sum over the corrected dual edges, evaluated at the end of
//! `post_e`, when the arrays hold `i^{n−½}` and `u^n` of the next step), so
//! the engine total is the invariant above: exactly conserved for a
//! lossless ladder up to `f32` rounding, non-increasing otherwise. Before
//! the first step the last term is taken as 0. Operators that change the
//! sheet edges' `u` or the corrected `i` after this one are not accounted
//! for.

use super::Engine;
use super::fields::Fields;
use super::kernel::flush;
use super::operator::{Hooks, Operator, OperatorHandle, StepContext};
use crate::axis::Axis;
use crate::constants::{EPS0, MU0};
use crate::post::Complex64;
use crate::{Error, Result};

/// Thickness of the outermost layers of the ladder, in skin depths at the
/// highest frequency (see the [module docs](self#discretisation-a-1d-sub-grid-across-the-thickness)).
pub const SURFACE_LAYER: f64 = 0.15;

/// Factor on the face capacitances of the sheet edges around a face the
/// sheet covers in part ([`ConductingSheet::cut_face`]; see the
/// [module docs](self#outlines-between-mesh-lines)).
pub const CUT_FACE_LOADING: f64 = 8.0;

/// Growth ratio of consecutive layers from the faces to the middle.
pub const GROWTH: f64 = 1.15;

/// Skin depth `√(2/(ωμ0σ))` \[m\] of conductivity `sigma` \[S/m\] at `f`
/// \[Hz\].
pub fn skin_depth(sigma: f64, f: f64) -> f64 {
    (2.0 / (2.0 * core::f64::consts::PI * f * MU0 * sigma)).sqrt()
}

/// Layer thicknesses \[m\] used for a sheet of conductivity `sigma`
/// \[S/m\] and thickness `thickness` \[m\] in a run up to `f_max` \[Hz\]
/// (graded from both faces, see the module docs). Sums to `thickness`.
pub fn layer_profile(sigma: f64, thickness: f64, f_max: f64) -> Vec<f64> {
    let s = SURFACE_LAYER * skin_depth(sigma, f_max);
    if !(s.is_finite() && s > 0.0) || thickness <= 2.0 * s {
        let n = if s.is_finite() && s > 0.0 {
            ((thickness / s).ceil() as usize).max(1)
        } else {
            1
        };
        return vec![thickness / n as f64; n];
    }
    let half_t = 0.5 * thickness;
    let mut half = Vec::new();
    let (mut acc, mut dz) = (0.0, s);
    while acc + dz < half_t {
        half.push(dz);
        acc += dz;
        dz *= GROWTH;
    }
    let rem = half_t - acc;
    match half.last_mut() {
        Some(last) if rem < 0.5 * *last => *last += rem,
        _ => half.push(rem),
    }
    let mut out = half.clone();
    out.extend(half.iter().rev());
    out
}

/// `coth(x)` and `csch(x)` for `Re x ≥ 0`, without overflow.
fn coth_csch(x: Complex64) -> (Complex64, Complex64) {
    let e1 = (-x).exp();
    let e2 = e1 * e1;
    let den = Complex64::ONE - e2;
    ((Complex64::ONE + e2) / den, e1 * 2.0 / den)
}

fn csqrt(z: Complex64) -> Complex64 {
    let (r, a) = z.to_polar();
    Complex64::from_polar(r.sqrt(), 0.5 * a)
}

/// Impedance matrix `[Z11, Z12]` \[Ω per square\] of the exact slab
/// (conductivity `sigma`, thickness `t`, vacuum `ε`, `μ`) at `f` \[Hz\]:
/// `E± = Z11·H± + Z12·H∓` with `H±` flowing into the faces (module docs).
pub fn slab_impedance(sigma: f64, t: f64, f: f64) -> (Complex64, Complex64) {
    let w = 2.0 * core::f64::consts::PI * f;
    let y = Complex64::new(sigma, w * EPS0);
    let jwm = Complex64::new(0.0, w * MU0);
    let gamma = csqrt(jwm * y);
    let zc = csqrt(jwm / y);
    let (coth, csch) = coth_csch(gamma * t);
    (zc * coth, zc * csch)
}

/// Chain (ABCD) matrix `[A, B, C, D]` per square of the exact slab at `f`
/// (`(E1, H1) = M·(E2, H2)`, `H2` leaving the second face).
pub fn slab_abcd(sigma: f64, t: f64, f: f64) -> [Complex64; 4] {
    let (z11, z12) = slab_impedance(sigma, t, f);
    z_to_abcd(z11, z12)
}

fn z_to_abcd(z11: Complex64, z12: Complex64) -> [Complex64; 4] {
    let a = z11 / z12;
    [a, (z11 * z11 - z12 * z12) / z12, z12.inv(), a]
}

/// Chain matrix per square of the discretised ladder (continuous time) the
/// operator uses for a sheet `(sigma, t)` in a run up to `f_max`,
/// evaluated at `f` (compare with [`slab_abcd`]).
pub fn model_abcd(sigma: f64, t: f64, f_max: f64, f: f64) -> [Complex64; 4] {
    let w = 2.0 * core::f64::consts::PI * f;
    let dz = layer_profile(sigma, t, f_max);
    let mul = |m: [Complex64; 4], n: [Complex64; 4]| {
        [
            m[0] * n[0] + m[1] * n[2],
            m[0] * n[1] + m[1] * n[3],
            m[2] * n[0] + m[3] * n[2],
            m[2] * n[1] + m[3] * n[3],
        ]
    };
    let shunt = |v: f64| {
        let y = Complex64::new(sigma * v, w * EPS0 * v);
        [Complex64::ONE, Complex64::ZERO, y, Complex64::ONE]
    };
    let series = |l: f64| {
        let z = Complex64::new(0.0, w * MU0 * l);
        [Complex64::ONE, z, Complex64::ZERO, Complex64::ONE]
    };
    let mut m = shunt(0.5 * dz[0]);
    for (k, &d) in dz.iter().enumerate() {
        m = mul(m, series(d));
        let v = 0.5 * (d + dz.get(k + 1).copied().unwrap_or(0.0));
        m = mul(m, shunt(v));
    }
    m
}

/// Geometry of the sheet edge along `d` at `node` on a sheet normal to
/// `nrm`: `(g, f_up, s_c)` with `g = w/l` (`l` the physical edge length,
/// `w` the physical width of its strip of the sheet: the dual length
/// across, along the third axis, at the sheet), `f_up` the share of the
/// crossing dual edges (along `nrm`) on the `+` side and `s_c` the default
/// share of the edge capacitance on the `+` side: the share of the dual
/// face area. On Cartesian grids `s_c = f_up`; on cylindrical grids the
/// dual face of a z edge on a `ρ = const` sheet is an annulus, so `s_c`
/// is the area share `(ρ_hi² − ρ²)/(ρ_hi² − ρ_lo²)`, and an α width is
/// the arc `ρ·Δα*` at the sheet (`ρ_{i+½}` for a ρ edge on a `z` sheet,
/// `ρ_i` for a z edge on a `ρ` sheet).
fn sheet_geometry(grid: &super::Grid, d: Axis, nrm: Axis, node: [usize; 3]) -> (f64, f64, f64) {
    let n = grid.nodes();
    let t_ax = Axis::ALL[3 - nrm.index() - d.index()];
    let k = node[nrm.index()];
    let up = if k + 1 < n[nrm.index()] {
        grid.delta(nrm, k)
    } else {
        0.0
    };
    let dn = if k > 0 { grid.delta(nrm, k - 1) } else { 0.0 };
    let f_up = up / (up + dn);
    let cyl = grid.is_cylindrical();
    let w = if cyl && t_ax == Axis::Y {
        // ρ_{i+½}·Δα* (multigrid aware); a z edge sits at ρ_i.
        let arc = grid.dual_edge_length(Axis::Y, node);
        if d == Axis::Z {
            let r = grid.lines(Axis::X);
            let i = node[0];
            let rd = if i + 1 < r.len() {
                0.5 * (r[i] + r[i + 1])
            } else {
                r[i]
            };
            if rd > 0.0 { arc * r[i] / rd } else { 0.0 }
        } else {
            arc
        }
    } else {
        grid.dual_delta(t_ax, node[t_ax.index()])
    };
    let l = grid.edge_length(d, node);
    let s_c = if cyl && d == Axis::Z && nrm == Axis::X {
        let r = grid.lines(Axis::X);
        let ri = r[k];
        let hi = if k + 1 < r.len() {
            0.5 * (ri + r[k + 1])
        } else {
            ri
        };
        let lo = if k > 0 { 0.5 * (ri + r[k - 1]) } else { ri };
        let (a_up, a_dn) = (hi * hi - ri * ri, ri * ri - lo * lo);
        a_up / (a_up + a_dn)
    } else {
        f_up
    };
    (w / l, f_up, s_c)
}

/// One sheet edge, as collected by [`ConductingSheet::push`] (and produced
/// by the compile step from csx `ConductingSheet` properties).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SheetEdge {
    /// Direction of the E edge (tangential to the sheet).
    pub axis: Axis,
    /// Start node of the edge (engine layout).
    pub node: [usize; 3],
    /// Normal of the sheet (≠ `axis`); the sheet lies in the node plane
    /// `node[normal]`.
    pub normal: Axis,
    /// Conductivity \[S/m\].
    pub conductivity: f64,
    /// Thickness \[m\].
    pub thickness: f64,
    /// Shares of the edge capacitance and conductance on the `+` side of
    /// the sheet (towards higher index along `normal`), each in `[0, 1]`;
    /// `None`: the share of the half-cell lengths (right for the same
    /// medium on both sides).
    pub upper_share: Option<(f64, f64)>,
}

/// A dual edge bordering the dual face of a sheet edge.
#[derive(Debug, Clone, Copy)]
struct Coupling {
    axis: Axis,
    node: [usize; 3],
    /// Sign of `i` in the E update of the edge.
    sign: f64,
    /// Weight of `i` in Ampère's law of the `+` / `−` face (sign included).
    w_up: f64,
    w_dn: f64,
    /// H correction factor: `i += corr·(u− − u+)`.
    corr: f64,
    /// Energy-invariant correction factor `½·L·corr` of this dual edge
    /// (`L = dt·(1 + da)/(2·db)`, as in the engine's energy sum).
    ecorr: f64,
}

/// One installed sheet edge.
#[derive(Debug, Clone)]
struct EdgeState {
    axis: Axis,
    /// The stored copies of the edge (engine nodes), canonical first.
    copies: Vec<[usize; 3]>,
    /// Edge capacitance of the kernel (`C`, for the energy correction).
    c_edge: f64,
    /// First node / layer of the constants in their arrays (shared by
    /// the edges with the same ladder).
    k0: usize,
    ka: usize,
    /// First node / layer in the flat arrays; nodes `n0..n0 + nn`.
    n0: usize,
    nn: usize,
    l0: usize,
    c0: usize,
    nc: usize,
    /// Index of its open part in `gaps` (edges partly covered).
    gap: Option<usize>,
    /// Share of `u−` in the face voltage the array holds,
    /// `u+ + mix·(u− − u+)` (0: `u+`).
    mix: f64,
    /// The array value written by the last `post_e` (edges with a gap or
    /// a mix, whose `u+` is not read back from the array).
    written: f32,
}

/// The open part of an edge the sheet covers partly: an ordinary edge in
/// series with the ladder (module docs).
#[derive(Debug, Clone, Copy)]
struct Gap {
    /// Kernel coefficients of the edge (`cb` scaled to the open part).
    ca: f64,
    cb: f64,
    /// Voltage over the open part.
    u: f64,
}

/// Conducting sheets: thin lossy conductors on mesh planes, with the
/// skin-effect model of the [module documentation](self).
///
/// Collect edges with [`push`](Self::push) or [`add_rect`](Self::add_rect),
/// then install with [`add_to`](Self::add_to). The sheet edges must not be
/// PEC in the engine's material map (PEC edges are skipped).
///
/// ```
/// use ems::engine::{ConductingSheet, Engine, Grid};
/// use ems::Axis;
///
/// let g = Grid::uniform([8, 8, 8], [1e-3; 3]).unwrap();
/// let mut e = Engine::builder(g).threads(1).build().unwrap();
/// let mut s = ConductingSheet::new(Some(10e9));
/// // 35 µm of copper on the plane z = 4, x, y ∈ [2, 6] (node indices).
/// s.add_rect(Axis::Z, 4, [2, 2, 0], [6, 6, 0], 5.8e7, 35e-6);
/// let h = s.add_to(&mut e).unwrap();
/// e.run(10);
/// assert_eq!(e.operator(&h).edges(), 2 * 4 * 5);
/// ```
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct ConductingSheet {
    f_max: Option<f64>,
    entries: Vec<SheetEdge>,
    /// Open fractions of edges partly covered ([`cut`](Self::cut)).
    cuts: Vec<(Axis, [usize; 3], f64)>,
    /// Faces partly covered ([`cut_face`](Self::cut_face)).
    cut_faces: Vec<(Axis, [usize; 3])>,
    gaps: Vec<Gap>,
    installed: bool,
    skipped: usize,
    dt: f64,
    edges: Vec<EdgeState>,
    couplings: Vec<Coupling>,
    /// Per node of the distinct ladders: `2C_k/dt`, Thomas `c'_k` and
    /// `1/den_k`; per node of every edge: voltage `u_k`.
    two_c: Vec<f64>,
    cp: Vec<f64>,
    inv_den: Vec<f64>,
    u: Vec<f64>,
    /// Per layer of the distinct ladders: `dt/(2L_k)`; per layer of every
    /// edge: current `h_k`.
    a: Vec<f64>,
    h: Vec<f64>,
    scratch: Vec<f64>,
    /// `Σ ½·L·corr·i^{n−½}·(u− − u+)^n` of the last `post_e` (module docs,
    /// *Energy*).
    w_corr: f64,
}

impl ConductingSheet {
    /// An empty set of sheets for a run up to `f_max` \[Hz\] (the layer
    /// grading of thick sheets resolves the skin depth at `f_max`); `None`:
    /// `1/(10·dt)` of the engine.
    pub fn new(f_max: Option<f64>) -> Self {
        Self {
            f_max,
            ..Self::default()
        }
    }

    /// Adds a sheet edge.
    pub fn push(&mut self, e: SheetEdge) {
        self.entries.push(e);
    }

    /// Declares the sheet edge along `axis` at `node` as covered by the
    /// sheet over the fraction `1 − open` of its length only (`open` in
    /// `(0, 1)`, the open length fraction; other values and edges that
    /// are not pushed are ignored). The engine's coefficient of the edge
    /// must be that of its open part when the sheets are installed (`cb`
    /// scaled by `open`, as [`ConformalPec::apply`](super::ConformalPec::apply)
    /// leaves a cut edge); see the
    /// [module docs](self#outlines-between-mesh-lines). Cartesian grids.
    pub fn cut(&mut self, axis: Axis, node: [usize; 3], open: f64) {
        if open > 0.0 && open < 1.0 {
            self.cuts.push((axis, node, open));
        }
    }

    /// Declares the cell face of normal `axis` with lowest corner `node`,
    /// lying in the plane of a sheet, as covered by it in part (a cut
    /// face of conformal PEC, with less inductance than a whole face):
    /// the sheet edges around it get the capacitance loading and the
    /// array value of the
    /// [module docs](self#outlines-between-mesh-lines). Cartesian grids.
    pub fn cut_face(&mut self, axis: Axis, node: [usize; 3]) {
        self.cut_faces.push((axis, node));
    }

    /// Adds every tangential edge of the node rectangle `start..=stop` on
    /// the node plane `plane` normal to `normal` (the `normal` components
    /// of `start` and `stop` are ignored), with conductivity `sigma`
    /// \[S/m\] and thickness `t` \[m\].
    pub fn add_rect(
        &mut self,
        normal: Axis,
        plane: usize,
        start: [usize; 3],
        stop: [usize; 3],
        sigma: f64,
        t: f64,
    ) {
        let (p, q) = normal.others();
        for (d, o) in [(p, q), (q, p)] {
            let (di, oi) = (d.index(), o.index());
            for a in start[di]..stop[di] {
                for b in start[oi]..=stop[oi] {
                    let mut node = [0; 3];
                    node[normal.index()] = plane;
                    node[di] = a;
                    node[oi] = b;
                    self.push(SheetEdge {
                        axis: d,
                        node,
                        normal,
                        conductivity: sigma,
                        thickness: t,
                        upper_share: None,
                    });
                }
            }
        }
    }

    /// The requested edges, in insertion order.
    pub fn entries(&self) -> &[SheetEdge] {
        &self.entries
    }

    /// Number of installed sheet edges.
    pub fn edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of requested edges skipped at installation because they are
    /// PEC (`cb = 0`) or duplicates.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Total number of ladder nodes (two per edge at least).
    pub fn ladder_nodes(&self) -> usize {
        self.u.len()
    }

    /// Voltage of the `−` face of an installed edge, `None` if `(axis,
    /// node)` is not an installed sheet edge (the `+` face voltage is the
    /// engine's `u`).
    pub fn lower_voltage(&self, axis: Axis, node: [usize; 3]) -> Option<f64> {
        self.edges
            .iter()
            .find(|e| e.axis == axis && e.copies.contains(&node))
            .map(|e| self.u[e.n0 + e.nn - 1])
    }

    fn validate(e: &SheetEdge, n: [usize; 3]) -> Result<()> {
        let bad = |m: String| Err(Error::Invalid(format!("conducting sheet edge {e:?}: {m}")));
        if e.axis == e.normal {
            return bad("the edge is not tangential to the sheet".into());
        }
        if !(e.conductivity.is_finite() && e.conductivity > 0.0) {
            return bad("conductivity must be finite and > 0".into());
        }
        if !(e.thickness.is_finite() && e.thickness > 0.0) {
            return bad("thickness must be finite and > 0".into());
        }
        if let Some((c, g)) = e.upper_share
            && !((0.0..=1.0).contains(&c) && (0.0..=1.0).contains(&g))
        {
            return bad("shares must be in [0, 1]".into());
        }
        let a = e.axis.index();
        if (0..3).any(|t| e.node[t] >= n[t]) || e.node[a] + 1 >= n[a] {
            return bad(format!("not an edge of the grid ({n:?} nodes)"));
        }
        Ok(())
    }

    fn install(&mut self, engine: &Engine) -> Result<()> {
        if self.installed {
            return Err(Error::Invalid("conducting sheet already installed".into()));
        }
        let grid = engine.grid();
        let n = grid.nodes();
        for e in &self.entries {
            Self::validate(e, n)?;
        }
        let dt = engine.dt();
        let f_max = match self.f_max {
            Some(f) if f.is_finite() && f > 0.0 => f,
            Some(f) => {
                return Err(Error::Invalid(format!(
                    "conducting sheet: f_max {f} must be finite and > 0"
                )));
            }
            None => 0.1 / dt,
        };
        let mut order: Vec<usize> = (0..self.entries.len()).collect();
        let key = |e: &SheetEdge| {
            (
                super::cylindrical::e_edge_copies(grid, e.axis, e.node)[0],
                e.axis.index(),
            )
        };
        order.sort_by_key(|&i| key(&self.entries[i]));
        let co = engine.coefficients();
        let cuts: std::collections::HashMap<(usize, [usize; 3]), f64> = self
            .cuts
            .iter()
            .map(|&(a, p, f)| ((a.index(), p), f))
            .collect();
        let cut_faces: std::collections::HashSet<(usize, [usize; 3])> = self
            .cut_faces
            .iter()
            .map(|&(a, p)| (a.index(), p))
            .collect();
        let outlines = !cut_faces.is_empty();
        // The distinct ladders: constants → their place in the arrays.
        // (Looked up only: the order of the map does not matter.)
        let mut ladders: std::collections::HashMap<Vec<u64>, (usize, usize)> =
            std::collections::HashMap::new();
        let mut last = None;
        for &i in &order {
            let e = self.entries[i];
            // The stored copies of the physical edge (both copies of an
            // edge on the α seam of a closed cylindrical grid): one ladder
            // for the physical edge, fed by the dual edges of every copy.
            let copies = super::cylindrical::e_edge_copies(grid, e.axis, e.node);
            let canon = (copies[0], e.axis.index());
            if last == Some(canon) || copies.len() > 2 {
                // Duplicates, and z edges on the axis (their dual face is
                // the whole disc around the axis; not supported).
                self.skipped += 1;
                continue;
            }
            last = Some(canon);
            // Per copy: capacitance and conductance of its (part of the)
            // dual face.
            let mut cg = Vec::with_capacity(copies.len());
            for &c in &copies {
                let idx = grid.index(c);
                let (ca, cb) = (co.ca(e.axis)[idx] as f64, co.cb(e.axis)[idx] as f64);
                if cb == 0.0 {
                    break;
                }
                cg.push((dt * (1.0 + ca) / (2.0 * cb), (1.0 - ca) / cb));
            }
            if cg.len() < copies.len() {
                // PEC.
                self.skipped += 1;
                continue;
            }
            let c_edge: f64 = cg.iter().map(|x| x.0).sum();
            let g_edge: f64 = cg.iter().map(|x| x.1).sum();
            // An edge partly covered: the kernel's coefficients are those
            // of the open part (C/f, G/f); the covered part has
            // C/(1 − f), G/(1 − f) and a ladder of its length.
            let open = (copies.len() == 1)
                .then(|| cuts.get(&(e.axis.index(), e.node)).copied())
                .flatten();
            let part = open.map_or(1.0, |f| f / (1.0 - f));
            let (c_cov, g_cov) = (part * c_edge, part * g_edge);
            let (nrm, d) = (e.normal, e.axis);
            let geom = |node: [usize; 3]| sheet_geometry(grid, d, nrm, node);
            // A sheet on the α seam of a closed grid (normal α): the two
            // copies are its two sides (the j = 0 copy holds the `+` half
            // of the dual face, the other one the `−` half, each with its
            // own kernel capacitance and conductance, materials
            // included), and they share one strip of the sheet.
            let seam_normal = copies.len() == 2 && nrm == Axis::Y;
            let (geo, sc, sg) = if seam_normal {
                let up = |x: f64, y: f64| if y > 0.0 { x / y } else { 0.5 };
                let sc = up(cg[0].0, c_edge);
                (
                    geom(copies[0]).0,
                    sc,
                    if g_edge > 0.0 {
                        up(cg[0].1, g_edge)
                    } else {
                        sc
                    },
                )
            } else {
                let s = geom(e.node).2;
                let (sc, sg) = e.upper_share.unwrap_or((s, s));
                (copies.iter().map(|&c| geom(c).0).sum(), sc, sg)
            };
            let geo: f64 = open.map_or(geo, |f| geo / (1.0 - f));

            // Ladder.
            let dz = layer_profile(e.conductivity, e.thickness, f_max);
            let nn = dz.len() + 1;
            let (n0, l0) = (self.u.len(), self.h.len());
            let mut a: Vec<f64> = Vec::with_capacity(nn - 1);
            let mut g_node = vec![0.0; nn];
            let mut c_node = vec![0.0; nn];
            for (j, &z) in dz.iter().enumerate() {
                for kk in [j, j + 1] {
                    g_node[kk] += 0.5 * e.conductivity * z * geo;
                    c_node[kk] += 0.5 * EPS0 * z * geo;
                }
                a.push(dt * geo / (2.0 * MU0 * z));
                self.h.push(0.0);
            }
            // Around a cut face: loaded (module docs).
            let (b, c) = d.others();
            let by_cut_face = copies.len() == 1 && {
                let mut lo = e.node;
                let o = if b == nrm { c } else { b };
                let below = lo[o.index()] > 0;
                lo[o.index()] = lo[o.index()].saturating_sub(1);
                cut_faces.contains(&(nrm.index(), e.node))
                    || (below && cut_faces.contains(&(nrm.index(), lo)))
            };
            let load = if by_cut_face { CUT_FACE_LOADING } else { 1.0 };
            c_node[0] += load * sc * c_cov;
            g_node[0] += sg * g_cov;
            c_node[nn - 1] += load * (1.0 - sc) * c_cov;
            g_node[nn - 1] += (1.0 - sg) * g_cov;
            // Thomas factorisation of the constant matrix.
            let mut consts: Vec<[f64; 3]> = Vec::with_capacity(nn);
            let mut prev_cp = 0.0;
            for kk in 0..nn {
                let a_up = if kk + 1 < nn { a[kk] } else { 0.0 };
                let a_dn = if kk > 0 { a[kk - 1] } else { 0.0 };
                let two_c = 2.0 * c_node[kk] / dt;
                let diag = two_c + g_node[kk] + a_up + a_dn;
                let den = diag + a_dn * prev_cp;
                let cp = -a_up / den;
                consts.push([two_c, cp, 1.0 / den]);
                self.u.push(0.0);
                prev_cp = cp;
            }
            // The constants of the ladder are kept once for all edges
            // that have the same ones (the edges of a sheet on cells of
            // one size, in one medium): the key is their bit patterns.
            let key: Vec<u64> = a
                .iter()
                .chain(consts.iter().flatten())
                .map(|v| v.to_bits())
                .collect();
            let (k0, ka) = *ladders.entry(key).or_insert_with(|| {
                let at = (self.two_c.len(), self.a.len());
                for k in &consts {
                    self.two_c.push(k[0]);
                    self.cp.push(k[1]);
                    self.inv_den.push(k[2]);
                }
                self.a.extend_from_slice(&a);
                at
            });

            // Couplings: the four dual edges around the dual face of every
            // copy, with their sign in the E update (engine kernel
            // convention).
            let c0 = self.couplings.len();
            // (axis, node, sign, shares of the two faces, da, db)
            let mut around = Vec::with_capacity(4 * copies.len());
            // Share of `u−` in the stored voltage (module docs, *Outlines
            // between mesh lines*).
            let mut mix = 0.0;
            for (node, ax, off, s) in copies.iter().flat_map(|&node| {
                [
                    (node, c, None, 1.0),
                    (node, c, Some(b), -1.0),
                    (node, b, None, -1.0),
                    (node, b, Some(c), 1.0),
                ]
            }) {
                let mut m = node;

                if let Some(o) = off {
                    if m[o.index()] == 0 {
                        continue;
                    }
                    m[o.index()] -= 1;
                }
                let (p, q) = ax.others();
                if m[p.index()] + 1 >= n[p.index()] || m[q.index()] + 1 >= n[q.index()] {
                    continue;
                }
                let (w_up, w_dn) = if ax == nrm {
                    let f_up = geom(node).1;
                    (f_up, 1.0 - f_up)
                } else if off == Some(nrm) {
                    (0.0, 1.0)
                } else {
                    (1.0, 0.0)
                };
                let mi = grid.index(m);
                let (da, db) = (co.da(ax)[mi] as f64, co.db(ax)[mi] as f64);
                if ax == nrm && db == 0.0 && outlines && copies.len() == 1 {
                    mix = w_dn;
                }
                around.push((ax, m, s, (w_up, w_dn), (da, db)));
            }
            for (ax, m, s, (w_up, w_dn), (da, db)) in around {
                // (The dual edge read the stored voltage, with the share
                // `mix` of `u−`; its own share is `w_dn`.)
                let corr = -db * s * (w_dn - mix);
                let ecorr = if db == 0.0 {
                    0.0
                } else {
                    0.5 * dt * (1.0 + da) / (2.0 * db) * corr
                };
                self.couplings.push(Coupling {
                    axis: ax,
                    node: m,
                    sign: s,
                    w_up: s * w_up,
                    w_dn: s * w_dn,
                    corr,
                    ecorr,
                });
            }
            let gap = open.map(|_| {
                let idx = grid.index(copies[0]);
                self.gaps.push(Gap {
                    ca: co.ca(d)[idx] as f64,
                    cb: co.cb(d)[idx] as f64,
                    u: 0.0,
                });
                self.gaps.len() - 1
            });
            self.edges.push(EdgeState {
                axis: d,
                copies,
                c_edge,
                n0,
                nn,
                l0,
                k0,
                ka,
                c0,
                nc: self.couplings.len() - c0,
                gap,
                mix,
                written: 0.0,
            });
        }
        drop(co);
        let max_nn = self.edges.iter().map(|e| e.nn).max().unwrap_or(0);
        self.scratch = vec![0.0; max_nn];
        self.dt = dt;
        self.installed = true;
        Ok(())
    }

    /// Validates the edges, builds the ladders for `engine` and registers
    /// the operator (no coefficient changes).
    ///
    /// # Errors
    /// [`Error::Invalid`] for an invalid edge (not tangential, outside the
    /// grid, non-positive conductivity or thickness), an invalid `f_max` or
    /// a second installation; the engine is unchanged then.
    pub fn add_to(mut self, engine: &mut Engine) -> Result<OperatorHandle<ConductingSheet>> {
        self.install(engine)?;
        Ok(engine.add_operator(self))
    }
}

impl Operator for ConductingSheet {
    fn hooks(&self) -> Hooks {
        if self.edges.is_empty() {
            Hooks::NONE
        } else {
            Hooks::POST_H | Hooks::POST_E
        }
    }

    fn post_h(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        for e in &self.edges {
            let stored = f.u.get(e.axis, e.copies[0]);
            if e.gap.is_none() && e.mix == 0.0 {
                self.u[e.n0] = stored as f64;
            } else if stored != e.written {
                // What another operator added since `post_e` (a source):
                // to the open part, or to `u+` as on a plain sheet edge.
                let d = stored as f64 - e.written as f64;
                match e.gap {
                    Some(g) => self.gaps[g].u += d,
                    None => self.u[e.n0] += d,
                }
            }
            let diff = self.u[e.n0 + e.nn - 1] - self.u[e.n0];
            for c in &self.couplings[e.c0..e.c0 + e.nc] {
                if c.corr != 0.0 {
                    let v = f.i.get_mut(c.axis, c.node);
                    *v = flush(*v + (c.corr * diff) as f32);
                }
            }
        }
    }

    fn post_e(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        let x = &mut self.scratch;
        let mut w_corr = 0.0;
        for e in &mut self.edges {
            // (`s_all`: the E update's sum, for the open part.)
            let (mut s_up, mut s_dn, mut s_e, mut s_all) = (0.0, 0.0, 0.0, 0.0);
            for c in &self.couplings[e.c0..e.c0 + e.nc] {
                let i = f.i.get(c.axis, c.node) as f64;
                s_up += c.w_up * i;
                s_dn += c.w_dn * i;
                s_e += c.ecorr * i;
                s_all += c.sign * i;
            }
            let (n0, nn, l0) = (e.n0, e.nn, e.l0);
            let u = &mut self.u[n0..n0 + nn];
            let (two_c, cp, inv_den) = (
                &self.two_c[e.k0..e.k0 + nn],
                &self.cp[e.k0..e.k0 + nn],
                &self.inv_den[e.k0..e.k0 + nn],
            );
            let a = &self.a[e.ka..e.ka + nn - 1];
            let h = &mut self.h[l0..l0 + nn - 1];
            // Forward sweep: x holds d'_k.
            for k in 0..nn {
                let mut r = two_c[k] * u[k];
                if k + 1 < nn {
                    r -= h[k];
                }
                if k > 0 {
                    r += h[k - 1] + a[k - 1] * x[k - 1];
                }
                if k == 0 {
                    r += s_up;
                }
                if k + 1 == nn {
                    r += s_dn;
                }
                x[k] = r * inv_den[k];
            }
            // Back substitution: x holds the mean voltages ū_k.
            for k in (0..nn - 1).rev() {
                x[k] -= cp[k] * x[k + 1];
            }
            for j in 0..nn - 1 {
                h[j] += 2.0 * a[j] * (x[j] - x[j + 1]);
            }
            for k in 0..nn {
                u[k] = 2.0 * x[k] - u[k];
            }
            // The face voltage in the array value.
            let face = if e.mix == 0.0 {
                u[0]
            } else {
                u[0] + e.mix * (u[nn - 1] - u[0])
            };
            let up = match e.gap {
                None => flush(face as f32),
                Some(g) => {
                    // The open part: the kernel's update of an ordinary
                    // edge, with the currents of all its dual edges.
                    let gap = &mut self.gaps[g];
                    gap.u = gap.ca * gap.u + gap.cb * s_all;
                    flush((gap.u + face) as f32)
                }
            };
            e.written = up;
            for &c in &e.copies {
                f.u.set(e.axis, c, up);
            }
            w_corr += if e.gap.is_some() || e.mix != 0.0 {
                s_e * (u[nn - 1] - u[0])
            } else {
                s_e * (u[nn - 1] - up as f64)
            };
        }
        self.w_corr = w_corr;
    }

    fn energy(&self) -> f64 {
        let mut w = 0.0;
        for e in &self.edges {
            let (n0, nn, l0) = (e.n0, e.nn, e.l0);
            for k in 0..nn {
                w += 0.25 * self.two_c[k + e.k0] * self.dt * self.u[n0 + k] * self.u[n0 + k];
            }
            // The engine's field sum has ½·C·(array value)².
            let face = if e.mix == 0.0 {
                self.u[n0]
            } else {
                self.u[n0] + e.mix * (self.u[n0 + nn - 1] - self.u[n0])
            };
            let stored = match e.gap {
                None => face,
                Some(g) => {
                    let uo = self.gaps[g].u;
                    w += 0.5 * e.c_edge * uo * uo;
                    uo + face
                }
            };
            w -= 0.5 * e.c_edge * stored * stored;
            for j in 0..nn - 1 {
                let l = self.dt / (2.0 * self.a[e.ka + j]);
                w += 0.5 * l * self.h[l0 + j] * self.h[l0 + j];
            }
        }
        w + self.w_corr
    }

    /// The magnetic energy of the ladder's inductances: the field inside
    /// the conductor, which diffuses into its own resistance once the
    /// field outside is gone.
    fn energy_held(&self) -> f64 {
        let mut w = 0.0;
        for e in &self.edges {
            for j in 0..e.nn - 1 {
                let l = self.dt / (2.0 * self.a[e.ka + j]);
                w += 0.5 * l * self.h[e.l0 + j] * self.h[e.l0 + j];
            }
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_sums_and_grades() {
        for (s, t, f) in [(5.8e7, 35e-6, 10e9), (5.8e7, 35e-9, 1e9), (1e5, 1e-3, 1e10)] {
            let p = layer_profile(s, t, f);
            let sum: f64 = p.iter().sum();
            assert!((sum - t).abs() < 1e-12 * t, "{s} {t}: {sum}");
            assert!(p.iter().all(|&d| d > 0.0));
        }
        assert_eq!(layer_profile(5.8e7, 35e-9, 1e9).len(), 1);
        assert_eq!(layer_profile(5.8e7, 35e-6, 10e9).len(), 48);
    }

    #[test]
    fn thin_and_thick_limits() {
        // t ≪ δ: Z11 ≈ Z12 ≈ 1/(σt).
        let (z11, z12) = slab_impedance(1e4, 1e-6, 1e9);
        let r = 1.0 / (1e4 * 1e-6);
        assert!((z11.re / r - 1.0).abs() < 1e-3 && (z12.re / r - 1.0).abs() < 1e-3);
        // t ≫ δ: Z11 → (1 + j)/(σδ), Z12 → 0.
        let (s, f) = (5.8e7, 10e9);
        let (z11, z12) = slab_impedance(s, 1e-3, f);
        let rs = 1.0 / (s * skin_depth(s, f));
        assert!((z11 - Complex64::new(rs, rs)).norm() < 1e-6 * rs);
        assert!(z12.norm() < 1e-100);
    }

    #[test]
    fn ladder_matches_slab() {
        for (s, t, fmax) in [
            (5.8e7, 35e-6, 10e9),
            (5.8e7, 1e-6, 10e9),
            (5.8e7, 5e-6, 10e9),
            (1e5, 1e-3, 10e9),
            (1e3, 1e-4, 1e10),
        ] {
            let mut worst: f64 = 0.0;
            for i in 1..=40 {
                let f = fmax * i as f64 / 40.0;
                let m = model_abcd(s, t, fmax, f);
                let x = slab_abcd(s, t, f);
                // Z11 = A/C.
                let za = m[0] / m[2];
                let ze = x[0] / x[2];
                worst = worst.max((za / ze - 1.0).norm());
            }
            assert!(worst < 0.012, "{s} {t}: Z11 error {worst}");
        }
    }
}
