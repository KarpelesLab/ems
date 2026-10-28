//! Plane-wave injection into a periodic unit cell at fixed transverse
//! wavevector (design §5.6.2): the oblique-incidence source of the Bloch
//! (sine–cosine) method.
//!
//! # Geometry
//!
//! The unit cell is periodic (or Bloch-periodic) along the two transverse
//! axes and open along the **normal** axis `n`. The total-field region is
//! the slab of node planes `start ..= stop` along `n`, spanning the whole
//! cell transversally; the wave enters through the plane `start` (forward,
//! `+n`) or `stop` (backward, `−n`). The other plane removes the incident
//! wave again (scattered-field region beyond it), unless it lies on the
//! domain boundary (a plane on the boundary gets no corrections, so the
//! total-field region then reaches through the absorbing layer).
//!
//! The corrections are those of the total-field / scattered-field method
//! ([`crate::engine::planewave`], consistency conditions): the tangential
//! E edges on the two planes get `+cb·K·i_inc` for their scattered-field H
//! neighbours, and those H edges get `+db·K·u_inc`.
//!
//! # Incident field at fixed transverse wavevector
//!
//! The incident field is `F(r, t) = F̂(ζ, t)·e^{−j k_t·(r − r_0)}` with
//! `ζ` the coordinate along `n`, `r_0` the first mesh line of every axis
//! and a fixed transverse wavevector `k_t` (`k_t·n̂ = 0`). Inserting it
//! into Maxwell's equations replaces every transverse derivative by
//! `−j k̃_t`, which leaves a **complex 1-D problem** for the six
//! components `F̂(ζ, t)`. On the Yee grid the transverse difference of two
//! neighbouring samples of `e^{−jk x}` is exactly `−j k̃·(sample between
//! them)` with the discrete wavenumber
//!
//! ```text
//! k̃ = 2·sin(k·Δ/2)/Δ
//! ```
//!
//! for a uniform transverse spacing `Δ`, so the 1-D problem *is* the 3-D
//! Yee scheme restricted to Bloch-phased fields. It is integrated on an
//! auxiliary grid that reuses the 3-D mesh lines along `n` (so its update
//! is the 3-D update of the incident field, edge for edge) with the
//! engine's time step. The incident field therefore leaks into the
//! scattered-field region only at `f32` rounding on a uniform transverse
//! mesh. On a graded transverse mesh `k̃` is taken with the mean spacing
//! and the incident field is approximate.
//!
//! The auxiliary grid is driven by a **hard source** on its tangential E
//! one or two cells upstream of the entry plane, `Ê_t = p·a(t)` with the
//! tangential polarisation `p` (V/m at amplitude 1 of the time function),
//! and ends in a graded CPML along `ζ` (reflectionless for any `k_t`, in
//! the continuum). Its field at the entry plane
//! is recorded each step ([`incident`](PeriodicPlaneWave::incident)): it
//! is `p·a(t)` filtered by the short upstream path, the reference against
//! which reflection and transmission are measured.
//!
//! # Constant transverse wavevector
//!
//! With `k_t` fixed, each frequency travels at its own angle,
//! `sin θ(f) = |k_t|·v/(2πf)` (`v` the background phase velocity), and
//! frequencies below `f_c = |k_t|·v/(2π)` are evanescent (grazing at
//! `f_c`). This is inherent to the sine–cosine method. The tangential E of
//! the incident wave is `p` at every frequency; its normal component
//! follows from the field equations.
//!
//! # Real and imaginary part
//!
//! The Bloch method carries the complex field in two real engines. One
//! operator per engine is needed, each running the same auxiliary grid
//! (cheap: 1-D) and injecting its own part
//! ([`WavePart::Real`]: `Re(F̂·e^{−jφ})`, [`WavePart::Imag`]:
//! `Im(F̂·e^{−jφ})`). With `k_t = 0` the imaginary part is zero.

use super::Engine;
use super::excite::Amplitude;
use super::fields::Fields;
use super::operator::{Hooks, Operator, StepContext};
use crate::axis::Axis;
use crate::constants::{C0, EPS0, MU0};
use crate::post::Complex64;
use crate::{Error, Result};

/// Which part of the complex incident field an operator injects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavePart {
    /// `Re(F̂·e^{−jφ})` (the real engine of a Bloch pair, or a plain
    /// periodic engine at `k_t = 0`).
    Real,
    /// `Im(F̂·e^{−jφ})` (the imaginary engine of a Bloch pair).
    Imag,
}

/// Geometry and physics of a [`PeriodicPlaneWave`] (see the
/// [module docs](self)).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PeriodicPlaneWaveSpec {
    /// Normal axis of the injection planes.
    pub axis: Axis,
    /// First node plane of the total-field region along `axis`.
    pub start: usize,
    /// Last node plane of the total-field region (`start < stop`).
    pub stop: usize,
    /// Propagation towards `+axis` (enters at `start`) or `−axis` (enters
    /// at `stop`).
    pub forward: bool,
    /// Transverse wavevector \[rad/m\] (component along `axis` must be 0).
    pub k_t: [f64; 3],
    /// Tangential incident E \[V/m\] at amplitude 1 of the time function
    /// (component along `axis` must be 0).
    pub polarization: [f64; 3],
    /// Relative permittivity of the background at the injection planes.
    pub eps_r: f64,
    /// Relative permeability of the background at the injection planes.
    pub mu_r: f64,
}

impl PeriodicPlaneWaveSpec {
    /// Total-field slab `start..=stop` along `axis`, direction, transverse
    /// wavevector and tangential polarisation; vacuum background.
    pub fn new(
        axis: Axis,
        start: usize,
        stop: usize,
        forward: bool,
        k_t: [f64; 3],
        polarization: [f64; 3],
    ) -> Self {
        Self {
            axis,
            start,
            stop,
            forward,
            k_t,
            polarization,
            eps_r: 1.0,
            mu_r: 1.0,
        }
    }

    /// Sets the background material at the injection planes.
    #[must_use]
    pub fn with_background(mut self, eps_r: f64, mu_r: f64) -> Self {
        self.eps_r = eps_r;
        self.mu_r = mu_r;
        self
    }

    /// The incidence angle \[rad\] at frequency `f` \[Hz\] (`None` below the
    /// cut-off `|k_t|·v/(2π)`, where the wave is evanescent).
    pub fn angle(&self, f: f64) -> Option<f64> {
        let kt = norm(self.k_t);
        let k = 2.0 * core::f64::consts::PI * f * (self.eps_r * self.mu_r).sqrt() / C0;
        (kt <= k).then(|| (kt / k).asin())
    }
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// One correction: `field[axis][node] += coef·(wr·re + wi·im)` of the
/// auxiliary component `comp` at index `m`.
#[derive(Debug, Clone, Copy)]
struct Correction {
    axis: Axis,
    node: [usize; 3],
    comp: usize,
    m: usize,
    wr: f64,
    wi: f64,
}

/// Cells of the CPML at the far end of the auxiliary grid.
const AUX_PML: usize = 64;
/// Polynomial grading order of that layer.
const AUX_PML_M: f64 = 3.0;

/// The complex 1-D grid (see the [module docs](self)).
#[derive(Debug, Clone)]
struct Aux {
    n: usize,
    /// Discrete transverse wavenumbers `k̃` (0 along `n`).
    kt: [f64; 3],
    /// Primary spacings `Δ_j` (between nodes `j` and `j + 1`).
    d: Vec<f64>,
    /// Dual spacings at nodes.
    dd: Vec<f64>,
    /// `dt/ε` and `dt/μ` of the background.
    cb: f64,
    db: f64,
    /// CPML decay factors `b = e^{−σ·dt/ε0}` of the `ζ` derivatives at
    /// nodes (for tangential E) and duals (for tangential H); 1 outside the
    /// layer.
    b_node: Vec<f64>,
    b_dual: Vec<f64>,
    /// Field components (complex): tangential E at nodes, normal E at
    /// duals; tangential H at duals, normal H at nodes.
    e: [Vec<Complex64>; 3],
    h: [Vec<Complex64>; 3],
    /// CPML convolution terms of the `ζ` derivatives of the tangential
    /// components (E at nodes, H at duals).
    psi_e: [Vec<Complex64>; 3],
    psi_h: [Vec<Complex64>; 3],
    /// Hard source node and the node of the PEC end.
    src: usize,
    end: usize,
}

/// `−j·k·x`.
fn mj(k: f64, x: Complex64) -> Complex64 {
    Complex64::new(k * x.im, -k * x.re)
}

/// The stretched `ζ` derivative `∂ζ + ψ` with the CPML recursion
/// `ψ ← b·ψ + (b − 1)·∂ζ` (κ = 1, α = 0).
fn stretched(diff: Complex64, b: f64, psi: &mut Complex64) -> Complex64 {
    if b == 1.0 {
        return diff;
    }
    *psi = *psi * b + diff * (b - 1.0);
    diff + *psi
}

impl Aux {
    fn update_h(&mut self) {
        let n = self.n;
        let nodes = self.d.len() + 1;
        for a in 0..3 {
            let (b, c) = ((a + 1) % 3, (a + 2) % 3);
            if a == n {
                // H_n at node j: T_b E_c − T_c E_b (tangential E at node j).
                for j in 0..nodes {
                    let curl = mj(self.kt[b], self.e[c][j]) - mj(self.kt[c], self.e[b][j]);
                    self.h[a][j] -= curl * self.db;
                }
            } else {
                // The ζ-differentiated tangential E component and its sign.
                let (t, s) = if b == n { (c, 1.0) } else { (b, -1.0) };
                let other = if b == n { c } else { b };
                for j in 0..nodes - 1 {
                    let diff = (self.e[t][j + 1] - self.e[t][j]) * (1.0 / self.d[j]);
                    let dz = stretched(diff, self.b_dual[j], &mut self.psi_h[a][j]);
                    let tr = mj(self.kt[other], self.e[n][j]);
                    // b == n: ∂ζE_c − T_c E_n; c == n: T_b E_n − ∂ζE_b.
                    let curl = dz * s - tr * s;
                    self.h[a][j] -= curl * self.db;
                }
            }
        }
    }

    fn update_e(&mut self) {
        let n = self.n;
        let nodes = self.d.len() + 1;
        for a in 0..3 {
            let (b, c) = ((a + 1) % 3, (a + 2) % 3);
            if a == n {
                // E_n at dual j: T_b H_c − T_c H_b (tangential H at dual j).
                for j in 0..nodes - 1 {
                    let curl = mj(self.kt[b], self.h[c][j]) - mj(self.kt[c], self.h[b][j]);
                    self.e[a][j] += curl * self.cb;
                }
            } else {
                let (t, s) = if b == n { (c, 1.0) } else { (b, -1.0) };
                let other = if b == n { c } else { b };
                for j in 1..nodes - 1 {
                    if j == self.src || j == self.end {
                        continue;
                    }
                    let diff = (self.h[t][j] - self.h[t][j - 1]) * (1.0 / self.dd[j]);
                    let dz = stretched(diff, self.b_node[j], &mut self.psi_e[a][j]);
                    let tr = mj(self.kt[other], self.h[n][j]);
                    // b == n: ∂ζH_c − T_c H_n; c == n: T_b H_n − ∂ζH_b.
                    let curl = dz * s - tr * s;
                    self.e[a][j] += curl * self.cb;
                }
            }
        }
    }
}

/// Plane-wave source for a periodic unit cell at fixed transverse
/// wavevector; see the [module docs](self).
///
/// ```
/// use ems::engine::{Engine, FaceCondition, FaceConditions, Grid};
/// use ems::engine::bloch_wave::{PeriodicPlaneWave, PeriodicPlaneWaveSpec, WavePart};
/// use ems::{Axis, Face};
///
/// let g = Grid::uniform([4, 4, 40], [1e-3; 3])?;
/// let faces = FaceConditions::default()
///     .with(Face::XMin, FaceCondition::Periodic)
///     .with(Face::XMax, FaceCondition::Periodic)
///     .with(Face::YMin, FaceCondition::Periodic)
///     .with(Face::YMax, FaceCondition::Periodic);
/// let mut e = Engine::builder(g).faces(faces).threads(1).build()?;
/// // Normal incidence along +z, E along x, total field from plane 10 on.
/// let spec = PeriodicPlaneWaveSpec::new(Axis::Z, 10, 39, true, [0.0; 3], [1.0, 0.0, 0.0]);
/// let pw = PeriodicPlaneWave::new(&e, &spec, WavePart::Real, |t| (-((t - 5e-11) / 1.5e-11).powi(2)).exp())?;
/// let h = e.add_operator(pw);
/// e.run(20);
/// assert_eq!(e.operator(&h).incident().len(), 20);
/// # Ok::<(), ems::Error>(())
/// ```
#[non_exhaustive]
pub struct PeriodicPlaneWave {
    spec: PeriodicPlaneWaveSpec,
    part: WavePart,
    amplitude: Amplitude,
    aux: Aux,
    /// Auxiliary node of the entry plane.
    entry: usize,
    e_corr: Vec<Correction>,
    h_corr: Vec<Correction>,
    incident: Vec<[Complex64; 3]>,
}

impl core::fmt::Debug for PeriodicPlaneWave {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PeriodicPlaneWave")
            .field("spec", &self.spec)
            .field("part", &self.part)
            .field("e_corrections", &self.e_corr.len())
            .field("h_corrections", &self.h_corr.len())
            .finish_non_exhaustive()
    }
}

/// The discrete transverse wavenumber of `k` on the lines of `axis`.
fn discrete_k(lines: &[f64], k: f64) -> f64 {
    if k == 0.0 || lines.len() < 2 {
        return k;
    }
    let d = (lines[lines.len() - 1] - lines[0]) / (lines.len() - 1) as f64;
    2.0 * (0.5 * k * d).sin() / d
}

impl PeriodicPlaneWave {
    /// Sets up the source for `engine`'s grid, time step and coefficients
    /// (take them as final: build after any operator that changes
    /// coefficients on the injection planes), injecting `part` of the
    /// complex incident field. `amplitude(t)` drives the auxiliary grid's
    /// hard source.
    ///
    /// # Errors
    /// [`Error::Invalid`] for planes outside the mesh or `start ≥ stop`,
    /// an entry plane on the domain boundary (no room for the source), a
    /// non-finite or normal `k_t`, a polarisation that is zero, non-finite
    /// or has a normal component, or an invalid background.
    pub fn new(
        engine: &Engine,
        spec: &PeriodicPlaneWaveSpec,
        part: WavePart,
        amplitude: impl Fn(f64) -> f64 + Send + 'static,
    ) -> Result<Self> {
        let grid = engine.grid();
        let nn = grid.nodes();
        let ax = spec.axis;
        let n = ax.index();
        let (lo, hi) = (spec.start, spec.stop);
        if lo >= hi || hi >= nn[n] {
            return Err(Error::Invalid(format!(
                "periodic plane wave: planes {lo}..={hi} must have start < stop within the \
                 {} nodes along {ax}",
                nn[n]
            )));
        }
        let entry_3d = if spec.forward { lo } else { hi };
        if entry_3d == 0 || entry_3d == nn[n] - 1 {
            return Err(Error::Invalid(format!(
                "periodic plane wave: the entry plane {entry_3d} lies on the domain boundary"
            )));
        }
        if spec.k_t.iter().any(|k| !k.is_finite()) || spec.k_t[n] != 0.0 {
            return Err(Error::Invalid(format!(
                "periodic plane wave: transverse wavevector {:?} must be finite with no \
                 component along {ax}",
                spec.k_t
            )));
        }
        let p = spec.polarization;
        let pl = norm(p);
        if !(pl.is_finite() && pl > 0.0) || p[n] != 0.0 {
            return Err(Error::Invalid(format!(
                "periodic plane wave: polarisation {p:?} must be non-zero, finite and \
                 tangential to the planes"
            )));
        }
        let (er, mr) = (spec.eps_r, spec.mu_r);
        if !(er.is_finite() && er > 0.0 && mr.is_finite() && mr > 0.0) {
            return Err(Error::Invalid(format!(
                "periodic plane wave: invalid background eps_r = {er}, mu_r = {mr}"
            )));
        }
        let dt = engine.dt();
        let lines = grid.lines(ax);

        // Auxiliary nodes: 3-D lines [first ..= last] plus the loss layer
        // downstream; `off` = 3-D node of auxiliary node 0.
        // Source cells upstream of the entry plane (at least 1, see the
        // boundary check above).
        let m0 = if spec.forward {
            lo.min(2)
        } else {
            (nn[n] - 1 - hi).min(2)
        };
        let (first, last) = if spec.forward {
            (lo - m0, (hi + 2).min(nn[n] - 1))
        } else {
            (lo.saturating_sub(2), hi + m0)
        };
        let mut z: Vec<f64> = lines[first..=last].to_vec();
        let off = if spec.forward {
            let d = z[z.len() - 1] - z[z.len() - 2];
            for _ in 0..AUX_PML + 1 {
                let x = z[z.len() - 1] + d;
                z.push(x);
            }
            first as isize
        } else {
            let d = z[1] - z[0];
            for _ in 0..AUX_PML + 1 {
                z.insert(0, z[0] - d);
            }
            first as isize - (AUX_PML + 1) as isize
        };
        let nodes = z.len();
        let d: Vec<f64> = z.windows(2).map(|w| w[1] - w[0]).collect();
        let dd: Vec<f64> = (0..nodes)
            .map(|j| {
                let a = if j > 0 { d[j - 1] } else { 0.0 };
                let b = if j + 1 < nodes { d[j] } else { 0.0 };
                0.5 * (a + b)
            })
            .collect();
        let (src, end) = if spec.forward {
            (0, nodes - 1)
        } else {
            (nodes - 1, 0)
        };
        // CPML profile: depth into the layer (cells) of a position given as
        // a fractional node index; σ_max = 0.8·(m + 1)/(η·Δ).
        let (eps, mu) = (EPS0 * er, MU0 * mr);
        let eta = (mu / eps).sqrt();
        let dl = if spec.forward { d[d.len() - 1] } else { d[0] };
        let smax = 0.8 * (AUX_PML_M + 1.0) / (eta * dl);
        let depth = |pos: f64| -> f64 {
            let x = if spec.forward {
                pos - (nodes - 1 - AUX_PML - 1) as f64
            } else {
                (AUX_PML + 1) as f64 - pos
            };
            x.clamp(0.0, AUX_PML as f64)
        };
        let bfac = |pos: f64| {
            let s = smax * (depth(pos) / AUX_PML as f64).powf(AUX_PML_M);
            (-s * dt / eps).exp()
        };
        let b_node = (0..nodes).map(|j| bfac(j as f64)).collect();
        let b_dual = (0..nodes - 1).map(|j| bfac(j as f64 + 0.5)).collect();
        let kt = Axis::ALL.map(|a| {
            if a == ax {
                0.0
            } else {
                discrete_k(grid.lines(a), spec.k_t[a.index()])
            }
        });
        let alloc = |at_node: bool| vec![Complex64::ZERO; if at_node { nodes } else { nodes - 1 }];
        let aux = Aux {
            n,
            kt,
            d,
            dd,
            cb: dt / eps,
            db: dt / mu,
            b_node,
            b_dual,
            e: [0, 1, 2].map(|a| alloc(a != n)),
            h: [0, 1, 2].map(|a| alloc(a == n)),
            psi_e: [0, 1, 2].map(|a| alloc(a != n)),
            psi_h: [0, 1, 2].map(|a| alloc(a == n)),
            src,
            end,
        };
        let aux_of = |node3: usize| -> usize { (node3 as isize - off) as usize };
        let entry = aux_of(entry_3d);

        // Phase reference: the first mesh line of every axis.
        let r0 = Axis::ALL.map(|a| grid.lines(a)[0]);
        let phase = |r: [f64; 3]| (0..3).map(|a| spec.k_t[a] * (r[a] - r0[a])).sum::<f64>();
        let weights = |phi: f64| {
            let (s, c) = phi.sin_cos();
            match part {
                WavePart::Real => (c, s),
                WavePart::Imag => (-s, c),
            }
        };
        let e_pos = |a: Axis, q: [usize; 3]| {
            let (b, c) = a.others();
            let mut r = [0.0; 3];
            r[a.index()] = grid.dual_line(a, q[a.index()]);
            r[b.index()] = grid.lines(b)[q[b.index()]];
            r[c.index()] = grid.lines(c)[q[c.index()]];
            r
        };
        let h_pos = |dax: Axis, q: [usize; 3]| {
            let (e, f) = dax.others();
            let mut r = [0.0; 3];
            r[dax.index()] = grid.lines(dax)[q[dax.index()]];
            r[e.index()] = grid.dual_line(e, q[e.index()]);
            r[f.index()] = grid.dual_line(f, q[f.index()]);
            r
        };
        // Total-field H: tangential H at duals lo ≤ q < hi, normal H at
        // nodes lo ≤ q ≤ hi.
        let tf_h = |dax: Axis, q: [usize; 3]| {
            if dax == ax {
                (lo..=hi).contains(&q[n])
            } else {
                (lo..hi).contains(&q[n])
            }
        };
        let h_valid = |dax: Axis, q: [usize; 3]| {
            let (e, f) = dax.others();
            (0..3).all(|a| q[a] < nn[a])
                && q[e.index()] + 1 < nn[e.index()]
                && q[f.index()] + 1 < nn[f.index()]
        };

        let co = engine.coefficients();
        let (mut e_corr, mut h_corr) = (Vec::new(), Vec::new());
        for plane in [lo, hi] {
            for a in Axis::ALL.into_iter().filter(|&a| a != ax) {
                let (b, c) = a.others();
                let t = if b == ax { c } else { b };
                let (ai, ti) = (a.index(), t.index());
                for pa in 0..nn[ai] - 1 {
                    for pt in 0..nn[ti] {
                        let mut pnode = [0; 3];
                        (pnode[ai], pnode[ti], pnode[n]) = (pa, pt, plane);
                        let cbv = co.cb(a)[grid.index(pnode)] as f64;
                        let minus = |s: usize| {
                            let mut q = pnode;
                            q[s] = q[s].wrapping_sub(1);
                            q
                        };
                        let terms = [
                            (c, pnode, 1.0),
                            (c, minus(b.index()), -1.0),
                            (b, pnode, -1.0),
                            (b, minus(c.index()), 1.0),
                        ];
                        for (dax, q, s) in terms {
                            if !h_valid(dax, q) || tf_h(dax, q) {
                                continue;
                            }
                            let di = dax.index();
                            // Tangential H at dual q[n] (scattered side).
                            debug_assert_ne!(dax, ax);
                            if cbv != 0.0 {
                                let (wr, wi) = weights(phase(h_pos(dax, q)));
                                let coef = cbv * s * grid.dual_delta(dax, q[di]);
                                e_corr.push(Correction {
                                    axis: a,
                                    node: pnode,
                                    comp: di,
                                    m: aux_of(q[n]),
                                    wr: coef * wr,
                                    wi: coef * wi,
                                });
                            }
                            let dbv = co.db(dax)[grid.index(q)] as f64;
                            if dbv != 0.0 {
                                let (wr, wi) = weights(phase(e_pos(a, pnode)));
                                let coef = dbv * s * grid.delta(a, pa);
                                h_corr.push(Correction {
                                    axis: dax,
                                    node: q,
                                    comp: ai,
                                    m: aux_of(plane),
                                    wr: coef * wr,
                                    wi: coef * wi,
                                });
                            }
                        }
                    }
                }
            }
        }
        drop(co);
        Ok(Self {
            spec: spec.clone(),
            part,
            amplitude: Box::new(amplitude),
            aux,
            entry,
            e_corr,
            h_corr,
            incident: Vec::new(),
        })
    }

    /// The settings.
    pub fn spec(&self) -> &PeriodicPlaneWaveSpec {
        &self.spec
    }

    /// The part of the complex field this operator injects.
    pub fn part(&self) -> WavePart {
        self.part
    }

    /// Number of E and H edge corrections applied per step.
    pub fn corrections(&self) -> (usize, usize) {
        (self.e_corr.len(), self.h_corr.len())
    }

    /// The discrete transverse wavenumbers `k̃` used by the auxiliary grid
    /// \[rad/m\].
    pub fn discrete_wavevector(&self) -> [f64; 3] {
        self.aux.kt
    }

    /// The complex incident E \[V/m\] on the entry plane at the phase
    /// reference (the first mesh line of every transverse axis), one value
    /// per step at the E times `(n + 1)·dt`: the tangential components as
    /// computed, the normal component averaged from the two half cells
    /// around the plane.
    pub fn incident(&self) -> &[[Complex64; 3]] {
        &self.incident
    }
}

impl Operator for PeriodicPlaneWave {
    fn hooks(&self) -> Hooks {
        Hooks::POST_H | Hooks::POST_E
    }

    fn post_h(&mut self, fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        // The H kernel read u^n; the auxiliary E is at n·dt.
        for c in &self.h_corr {
            let v = self.aux.e[c.comp][c.m];
            *fields.i.get_mut(c.axis, c.node) += (c.wr * v.re + c.wi * v.im) as f32;
        }
        self.aux.update_h();
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        // The E kernel read i^{n+½}; so does the auxiliary H now.
        for c in &self.e_corr {
            let v = self.aux.h[c.comp][c.m];
            *fields.u.get_mut(c.axis, c.node) += (c.wr * v.re + c.wi * v.im) as f32;
        }
        self.aux.update_e();
        let a = (self.amplitude)(ctx.e_time());
        let n = self.aux.n;
        for t in (0..3).filter(|&t| t != n) {
            self.aux.e[t][self.aux.src] = Complex64::new(self.spec.polarization[t] * a, 0.0);
            self.aux.e[t][self.aux.end] = Complex64::ZERO;
        }
        let j = self.entry;
        let mut inc = [Complex64::ZERO; 3];
        for (t, v) in inc.iter_mut().enumerate() {
            *v = if t == n {
                (self.aux.e[n][j - 1] + self.aux.e[n][j]) * 0.5
            } else {
                self.aux.e[t][j]
            };
        }
        self.incident.push(inc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_wavenumber() {
        let l: Vec<f64> = (0..11).map(|i| i as f64 * 1e-3).collect();
        let k = 300.0;
        let kt = discrete_k(&l, k);
        assert!((kt - 2.0 * (0.5 * k * 1e-3).sin() / 1e-3).abs() < 1e-12);
        assert!(kt < k);
        assert_eq!(discrete_k(&l, 0.0), 0.0);
    }

    #[test]
    fn angle_and_cutoff() {
        let s = PeriodicPlaneWaveSpec::new(Axis::Z, 2, 9, true, [10.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let fc = 10.0 * C0 / (2.0 * core::f64::consts::PI);
        assert!(s.angle(0.99 * fc).is_none());
        let th = s.angle(2.0 * fc).unwrap();
        assert!((th - 30f64.to_radians()).abs() < 1e-12);
    }
}
