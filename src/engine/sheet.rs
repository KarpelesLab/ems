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
    node: [usize; 3],
    /// Edge capacitance of the kernel (`C`, for the energy correction).
    c_edge: f64,
    /// First node / layer in the flat arrays; nodes `n0..n0 + nn`.
    n0: usize,
    nn: usize,
    l0: usize,
    c0: usize,
    nc: usize,
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
pub struct ConductingSheet {
    f_max: Option<f64>,
    entries: Vec<SheetEdge>,
    installed: bool,
    skipped: usize,
    dt: f64,
    edges: Vec<EdgeState>,
    couplings: Vec<Coupling>,
    /// Per node: `2C_k/dt`, Thomas `c'_k` and `1/den_k`, voltage `u_k`.
    two_c: Vec<f64>,
    cp: Vec<f64>,
    inv_den: Vec<f64>,
    u: Vec<f64>,
    /// Per layer: `dt/(2L_k)`, current `h_k`.
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
            .find(|e| e.axis == axis && e.node == node)
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
        let key = |e: &SheetEdge| (e.node, e.axis.index());
        order.sort_by_key(|&i| key(&self.entries[i]));
        let co = engine.coefficients();
        let mut last = None;
        for &i in &order {
            let e = self.entries[i];
            if last == Some(key(&e)) {
                self.skipped += 1;
                continue;
            }
            last = Some(key(&e));
            let idx = grid.index(e.node);
            let (ca, cb) = (co.ca(e.axis)[idx], co.cb(e.axis)[idx]);
            if cb == 0.0 {
                self.skipped += 1;
                continue;
            }
            let c_edge = dt * (1.0 + ca as f64) / (2.0 * cb as f64);
            let g_edge = (1.0 - ca as f64) / cb as f64;
            let (nrm, d) = (e.normal, e.axis);
            let t_ax = Axis::ALL[3 - nrm.index() - d.index()];
            let k = e.node[nrm.index()];
            let up = if k + 1 < n[nrm.index()] {
                grid.delta(nrm, k)
            } else {
                0.0
            };
            let dn = if k > 0 { grid.delta(nrm, k - 1) } else { 0.0 };
            let f_up = up / (up + dn);
            let (sc, sg) = e.upper_share.unwrap_or((f_up, f_up));
            let geo =
                grid.dual_delta(t_ax, e.node[t_ax.index()]) / grid.delta(d, e.node[d.index()]);

            // Ladder.
            let dz = layer_profile(e.conductivity, e.thickness, f_max);
            let nn = dz.len() + 1;
            let (n0, l0) = (self.u.len(), self.a.len());
            let mut g_node = vec![0.0; nn];
            let mut c_node = vec![0.0; nn];
            for (j, &z) in dz.iter().enumerate() {
                for kk in [j, j + 1] {
                    g_node[kk] += 0.5 * e.conductivity * z * geo;
                    c_node[kk] += 0.5 * EPS0 * z * geo;
                }
                self.a.push(dt * geo / (2.0 * MU0 * z));
                self.h.push(0.0);
            }
            c_node[0] += sc * c_edge;
            g_node[0] += sg * g_edge;
            c_node[nn - 1] += (1.0 - sc) * c_edge;
            g_node[nn - 1] += (1.0 - sg) * g_edge;
            // Thomas factorisation of the constant matrix.
            let a = &self.a[l0..];
            let mut prev_cp = 0.0;
            for kk in 0..nn {
                let a_up = if kk + 1 < nn { a[kk] } else { 0.0 };
                let a_dn = if kk > 0 { a[kk - 1] } else { 0.0 };
                let two_c = 2.0 * c_node[kk] / dt;
                let diag = two_c + g_node[kk] + a_up + a_dn;
                let den = diag + a_dn * prev_cp;
                let cp = -a_up / den;
                self.two_c.push(two_c);
                self.cp.push(cp);
                self.inv_den.push(1.0 / den);
                self.u.push(0.0);
                prev_cp = cp;
            }

            // Couplings: the four dual edges around the dual face, with
            // their sign in the E update (engine kernel convention).
            let (b, c) = d.others();
            let c0 = self.couplings.len();
            for (ax, off, s) in [
                (c, None, 1.0),
                (c, Some(b), -1.0),
                (b, None, -1.0),
                (b, Some(c), 1.0),
            ] {
                let mut m = e.node;
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
                    (f_up, 1.0 - f_up)
                } else if off == Some(nrm) {
                    (0.0, 1.0)
                } else {
                    (1.0, 0.0)
                };
                let mi = grid.index(m);
                let (da, db) = (co.da(ax)[mi] as f64, co.db(ax)[mi] as f64);
                let corr = -db * s * w_dn;
                let ecorr = if db == 0.0 {
                    0.0
                } else {
                    0.5 * dt * (1.0 + da) / (2.0 * db) * corr
                };
                self.couplings.push(Coupling {
                    axis: ax,
                    node: m,
                    w_up: s * w_up,
                    w_dn: s * w_dn,
                    corr,
                    ecorr,
                });
            }
            self.edges.push(EdgeState {
                axis: d,
                node: e.node,
                c_edge,
                n0,
                nn,
                l0,
                c0,
                nc: self.couplings.len() - c0,
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
            let up = f.u.get(e.axis, e.node) as f64;
            self.u[e.n0] = up;
            let diff = self.u[e.n0 + e.nn - 1] - up;
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
        for e in &self.edges {
            let (mut s_up, mut s_dn, mut s_e) = (0.0, 0.0, 0.0);
            for c in &self.couplings[e.c0..e.c0 + e.nc] {
                let i = f.i.get(c.axis, c.node) as f64;
                s_up += c.w_up * i;
                s_dn += c.w_dn * i;
                s_e += c.ecorr * i;
            }
            let (n0, nn, l0) = (e.n0, e.nn, e.l0);
            let u = &mut self.u[n0..n0 + nn];
            let (two_c, cp, inv_den) = (
                &self.two_c[n0..n0 + nn],
                &self.cp[n0..n0 + nn],
                &self.inv_den[n0..n0 + nn],
            );
            let a = &self.a[l0..l0 + nn - 1];
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
            let up = flush(u[0] as f32);
            f.u.set(e.axis, e.node, up);
            w_corr += s_e * (u[nn - 1] - up as f64);
        }
        self.w_corr = w_corr;
    }

    fn energy(&self) -> f64 {
        let mut w = 0.0;
        for e in &self.edges {
            let (n0, nn, l0) = (e.n0, e.nn, e.l0);
            for k in 0..nn {
                w += 0.25 * self.two_c[k + n0] * self.dt * self.u[n0 + k] * self.u[n0 + k];
            }
            w -= 0.5 * e.c_edge * self.u[n0] * self.u[n0];
            for j in 0..nn - 1 {
                let l = self.dt / (2.0 * self.a[l0 + j]);
                w += 0.5 * l * self.h[l0 + j] * self.h[l0 + j];
            }
        }
        w + self.w_corr
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
