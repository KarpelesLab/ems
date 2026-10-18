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
//! to `k̂`) and `η = Z0·√(μ_r/ε_r)` the background wave impedance.
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
//! # Limitations
//!
//! * The background (`ε_r`, `μ_r`, lossless) must be homogeneous around the
//!   box surface; scatterers belong strictly inside the box.
//! * The box must stay clear of absorbing boundaries (CPML, Mur): the
//!   corrections are applied in the serial `post_h` / `post_e` hooks and
//!   assume the plain Yee update on the surface.
//! * The energy reported by [`Engine::energy`](super::Engine::energy) does
//!   not account for the injected wave.

use super::Engine;
use super::excite::Amplitude;
use super::fields::Fields;
use super::operator::{Hooks, Operator, StepContext};
use crate::axis::Axis;
use crate::constants::{C0, Z0};
use crate::{Error, Result};

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
    /// Frequency \[Hz\] at which the auxiliary grid's dispersion is matched
    /// to the 3-D grid along `direction`; `0` (default) picks 20 mean cells
    /// per wavelength.
    pub match_frequency: f64,
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
            match_frequency: 0.0,
        }
    }

    /// Sets the background material around the box.
    #[must_use]
    pub fn with_background(mut self, eps_r: f64, mu_r: f64) -> Self {
        self.eps_r = eps_r;
        self.mu_r = mu_r;
        self
    }

    /// Sets the dispersion-matching frequency \[Hz\] (`0`: default).
    #[must_use]
    pub fn with_match_frequency(mut self, f: f64) -> Self {
        self.match_frequency = f;
        self
    }
}

/// One correction: `field[axis][node] += coef · Σ w_q · aux[m + q]`.
#[derive(Debug, Clone, Copy)]
struct Correction {
    axis: Axis,
    node: [usize; 3],
    coef: f64,
    m: usize,
    w: [f64; 4],
}

/// Cells of the matched-loss layer at the far end of the auxiliary grid.
const AUX_PML: usize = 64;
/// Design round-trip reflection of that layer.
const AUX_PML_R: f64 = 1e-12;

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
    amplitude: Amplitude,
    k_hat: [f64; 3],
    e0: [f64; 3],
    h0: [f64; 3],
    r_ref: [f64; 3],
    dxi: f64,
    m0: usize,
    lead: f64,
    e: Vec<f64>,
    h: Vec<f64>,
    ca: Vec<f64>,
    cb: Vec<f64>,
    da: Vec<f64>,
    db: Vec<f64>,
    e_corr: Vec<Correction>,
    h_corr: Vec<Correction>,
}

impl core::fmt::Debug for PlaneWave {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PlaneWave")
            .field("spec", &self.spec)
            .field("spacing", &self.dxi)
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

/// A correction before its stencil is known.
struct Raw {
    axis: Axis,
    node: [usize; 3],
    coef: f64,
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
    /// perpendicular to the direction, an invalid background, or a
    /// matching frequency the mesh cannot resolve;
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
        let (er, mr) = (spec.eps_r, spec.mu_r);
        if !(er.is_finite() && er > 0.0 && mr.is_finite() && mr > 0.0) {
            return Err(Error::Invalid(format!(
                "plane wave: invalid background eps_r = {er}, mu_r = {mr}"
            )));
        }
        let v = C0 / (er * mr).sqrt();
        let eta = Z0 * (mr / er).sqrt();
        let h0 = cross(k_hat, e0).map(|x| x / eta);
        let dt = engine.dt();
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
            let (cb_a, ea) = (co.cb(a), e0[ai]);
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
                            if cbv != 0.0 && h0[di] != 0.0 {
                                e_raw.push(Raw {
                                    axis: a,
                                    node: p,
                                    coef: cbv * s * h0[di] * grid.dual_delta(d, q[di]),
                                    xi: xi(h_pos(d, q)),
                                });
                            }
                            let dbv = co.db(d)[grid.index(q)] as f64;
                            if dbv != 0.0 && ea != 0.0 {
                                h_raw.push(Raw {
                                    axis: d,
                                    node: q,
                                    coef: dbv * s * ea * grid.delta(a, pa),
                                    xi: xi(e_pos(a, p)),
                                });
                            }
                        }
                    }
                }
            }
        }
        drop(co);

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
        let nodes = main + AUX_PML + 1;
        let r = v * dt / dxi;
        let smax = -AUX_PML_R.ln() * r / AUX_PML as f64;
        let loss = |x: f64| {
            if x > 0.0 {
                smax * (x / AUX_PML as f64).powi(3)
            } else {
                0.0
            }
        };
        let (mut ca, mut cb) = (vec![1.0; nodes], vec![0.0; nodes]);
        for m in 1..nodes - 1 {
            let s = loss(m as f64 - main as f64);
            ca[m] = (1.0 - s) / (1.0 + s);
            cb[m] = r / (1.0 + s);
        }
        let (mut da, mut db) = (vec![1.0; nodes - 1], vec![0.0; nodes - 1]);
        for m in 0..nodes - 1 {
            let s = loss(m as f64 + 0.5 - main as f64);
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
                        m,
                        w,
                    }
                })
                .collect()
        };
        let omega = 2.0 * core::f64::consts::PI * f;
        Ok(Self {
            spec: spec.clone(),
            amplitude: Box::new(amplitude),
            k_hat,
            e0,
            h0,
            r_ref,
            dxi,
            m0,
            lead: m0 as f64 * dxi * k / omega,
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

    /// The settings.
    pub fn spec(&self) -> &PlaneWaveSpec {
        &self.spec
    }

    /// Unit propagation vector `k̂`.
    pub fn direction(&self) -> [f64; 3] {
        self.k_hat
    }

    /// Spacing `Δξ` of the auxiliary 1-D grid \[m\].
    pub fn spacing(&self) -> f64 {
        self.dxi
    }

    /// The reference point `r_ref` \[m\]: the box corner the wave enters
    /// first, where the incident E is `E0·a(t)`.
    pub fn reference_point(&self) -> [f64; 3] {
        self.r_ref
    }

    /// Number of E and H edge corrections applied per step.
    pub fn corrections(&self) -> (usize, usize) {
        (self.e_corr.len(), self.h_corr.len())
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
        (s >= 1.0 && s + 3.0 < (self.e.len() - AUX_PML - 1) as f64).then(|| stencil(s))
    }

    /// Incident E \[V/m\] at `r` \[m\] at the time of the engine's E
    /// voltages (`n·dt` after `n` steps), or `None` if `r` lies before
    /// the auxiliary grid's start or beyond its end.
    pub fn incident_e(&self, r: [f64; 3]) -> Option<[f64; 3]> {
        let (m, w) = self.aux_at(r, 0.0)?;
        let f = interp(&self.e, m, &w);
        Some(self.e0.map(|x| x * f))
    }

    /// Incident H \[A/m\] at `r` \[m\] at the time of the engine's H
    /// currents (`(n − ½)·dt` after `n` steps), or `None` as for
    /// [`incident_e`](Self::incident_e).
    pub fn incident_h(&self, r: [f64; 3]) -> Option<[f64; 3]> {
        let (m, w) = self.aux_at(r, 0.5)?;
        let f = interp(&self.h, m, &w);
        Some(self.h0.map(|x| x * f))
    }
}

impl Operator for PlaneWave {
    fn hooks(&self) -> Hooks {
        Hooks::POST_H | Hooks::POST_E
    }

    fn post_h(&mut self, fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {
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
        // The E kernel read i^{n+½}; so does the auxiliary H now.
        for c in &self.e_corr {
            *fields.u.get_mut(c.axis, c.node) += (c.coef * interp(&self.h, c.m, &c.w)) as f32;
        }
        let last = self.e.len() - 1;
        for m in 1..last {
            self.e[m] = self.ca[m] * self.e[m] - self.cb[m] * (self.h[m] - self.h[m - 1]);
        }
        self.e[0] = (self.amplitude)(ctx.e_time() + self.lead);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
