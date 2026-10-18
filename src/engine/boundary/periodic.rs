//! Periodic boundaries (zero phase shift) on pairs of opposite faces
//! (design §5.6.1).
//!
//! # Model
//!
//! An axis `a` with `n` nodes is periodic with period
//! `L = x_{n−1} − x_0` when node `n − 1` is identified with node `0`. The
//! grid then has `n − 1` distinct node planes along `a`; the two outer
//! planes are two **copies** of the same plane:
//!
//! * the E edges tangential to the face (the two components other than
//!   `a`) exist on both copies;
//! * the H dual edges normal to the face (component `a`) exist on both
//!   copies, each with the half dual length inside the domain;
//! * everything else is interior or padding.
//!
//! Both faces are built like [`FaceCondition::Pmc`]
//! faces ([`FaceCondition::Periodic`]):
//! each copy of a tangential E edge is updated over its **half** dual face
//! (the half inside the domain), with the out-of-domain H read as zero.
//! The two halves together are the full dual face of the periodic edge:
//! with `w = 1/cb` (`= C/dt + G/2` of the copy's half face, see
//! [`Coefficients`]), the update of the merged edge,
//!
//! ```text
//! (C₀ + C₁)·(u' − u)/dt + (G₀ + G₁)·(u' + u)/2 = ∮₀ i + ∮₁ i,
//! ```
//!
//! is exactly the `1/cb`-weighted mean of the two half updates
//! `u'_k = ca_k·u + cb_k·∮_k i`:
//!
//! ```text
//! u' = (w₀·u'₀ + w₁·u'₁) / (w₀ + w₁).
//! ```
//!
//! After every E update the engine replaces both copies by that mean
//! (in the serial `post_e` gap), so the two copies always hold the same value. A
//! copy with `cb = 0` (PEC) makes the edge PEC, unless both copies have
//! `cb = 0`: the kernel then does not update the edge at all, and the
//! values written by an operator (e.g. a Mur face crossing the periodic
//! face) are kept (both copies set to their mean). Edges on the intersection of
//! two periodic faces (four copies) are merged axis by axis with the
//! accumulated weights, which gives the weighted mean over all four.
//!
//! The H copies need no merging: both integrate the same equation (the
//! circulation of the merged E around the same primary face), each scaled by
//! its own half dual length, so `i₀/l₀ = i₁/l₁ = H` holds up to rounding,
//! and their sum is the full dual edge current seen by the merged E edges.
//! (With magnetic loss differing between the first and last cell along `a`
//! the two halves would decay at slightly different rates; the materials at
//! a periodic face should match, as they do for any physically periodic
//! structure.)
//!
//! # Energy
//!
//! The merged scheme is the Yee scheme on the periodic (toroidal) grid, so
//! the leapfrog invariant of [`Engine::energy`](crate::engine::Engine::energy)
//! is conserved: the field sums count both copies, `C₀u² + C₁u² =
//! (C₀ + C₁)u²`, which is the capacitance of the merged edge.
//!
//! # Operators
//!
//! The merge runs at the start of the `post_e` gap, before any operator's
//! `post_e` hook (so probes, dumps and boundary operators see merged
//! values), and once more after the hooks when some operator has a
//! `post_e` hook, so that corrections an operator applies to one copy are
//! folded in. Slab hooks (CPML) run before both. A correction proportional
//! to `cb` (a current injected into the half face, e.g. dispersive or
//! lumped branches) is thereby combined exactly; an additive voltage
//! applied to only one copy (a soft source covering one face plane but not
//! the other) is scaled by that copy's weight share, about ½. Sources and
//! probes on a periodic face should cover both copies (e.g. span the whole
//! period, both ends included).

use crate::axis::Axis;
use crate::engine::coeffs::{Coefficients, FaceCondition, FaceConditions};
use crate::engine::fields::Fields;
use crate::engine::run::{Stepper, drive};
use crate::engine::{Engine, EngineBuilder, Grid, Progress, RunLimits, RunOutcome};
use crate::fdtd::Face;
use crate::{Error, Result};

/// The merge weights of a pair of copies with `1/cb` weights `wl`, `wh`
/// (infinite for `cb = 0`): `None` if exactly one side is PEC (the merged
/// edge is PEC), equal weights if both are (the kernel does not update the
/// edge at all; its values belong to an operator such as a Mur face that
/// crosses the periodic face, so they are kept, averaged), else the
/// weights themselves.
fn pair_weights(wl: f64, wh: f64) -> Option<(f64, f64)> {
    match (wl.is_infinite(), wh.is_infinite()) {
        (true, true) => Some((1.0, 1.0)),
        (false, false) => Some((wl, wh)),
        _ => None,
    }
}

/// The periodic axes of an engine and the face merge; built by the engine
/// from its [`FaceConditions`].
#[derive(Debug, Clone)]
pub(crate) struct Periodic {
    axes: [bool; 3],
    n: [usize; 3],
}

impl Periodic {
    /// The periodic axes of `faces` (`None` if there are none).
    ///
    /// # Errors
    /// [`Error::Invalid`] if only one face of an axis is periodic.
    pub(crate) fn from_faces(faces: &FaceConditions, n: [usize; 3]) -> Result<Option<Self>> {
        let mut axes = [false; 3];
        for a in Axis::ALL {
            let lo = faces[Face::new(a, false)] == FaceCondition::Periodic;
            let hi = faces[Face::new(a, true)] == FaceCondition::Periodic;
            if lo != hi {
                return Err(Error::Invalid(format!(
                    "periodic boundary on {} needs {} periodic too",
                    Face::new(a, !lo),
                    Face::new(a, lo)
                )));
            }
            axes[a.index()] = lo;
        }
        Ok(axes.iter().any(|&p| p).then_some(Self { axes, n }))
    }

    fn index(&self, [i, j, k]: [usize; 3]) -> usize {
        (i * self.n[1] + j) * self.n[2] + k
    }

    /// `1/cb` of component `c` at node `p` summed over the copies already
    /// merged by the passes of the periodic axes before `a` (infinite if
    /// any copy is PEC).
    fn weight(&self, cb: &[f32], c: usize, a: usize, p: [usize; 3]) -> f64 {
        let w = |p: [usize; 3]| {
            let b = cb[self.index(p)];
            if b == 0.0 {
                f64::INFINITY
            } else {
                1.0 / b as f64
            }
        };
        let mut sum = w(p);
        // Only one earlier axis can be tangential to component `c` too.
        for b in (0..a).filter(|&b| b != c && self.axes[b]) {
            let last = self.n[b] - 1;
            if p[b] == 0 || p[b] == last {
                let mut q = p;
                q[b] = last - p[b];
                sum += w(q);
            }
        }
        sum
    }

    /// Calls `pair(a, axis, lo, hi, w_lo, w_hi)` for every pair of copies
    /// of a tangential E edge: pass by pass over the periodic axes `a` in
    /// x, y, z order, with the accumulated weights of [`weight`](Self::weight).
    fn pairs(
        &self,
        co: &Coefficients,
        mut pair: impl FnMut(usize, Axis, [usize; 3], [usize; 3], f64, f64),
    ) {
        let n = self.n;
        for a in (0..3).filter(|&a| self.axes[a]) {
            let (b, c) = match a {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            for comp in [b, c] {
                let axis = Axis::ALL[comp];
                let cb = co.cb(axis);
                let other = if comp == b { c } else { b };
                // Nodes of the low copy: p[a] = 0, p[comp] < n[comp] − 1
                // (the last one is padding), p[other] free.
                for x in 0..n[comp] - 1 {
                    for y in 0..n[other] {
                        let mut lo = [0; 3];
                        lo[comp] = x;
                        lo[other] = y;
                        let mut hi = lo;
                        hi[a] = n[a] - 1;
                        let (wl, wh) = (self.weight(cb, comp, a, lo), self.weight(cb, comp, a, hi));
                        pair(a, axis, lo, hi, wl, wh);
                    }
                }
            }
        }
    }

    /// Replaces both copies of every tangential E edge on the periodic
    /// faces by their `1/cb`-weighted mean (see the
    /// [module docs](self)). Deterministic: the same arithmetic in a fixed
    /// order, independent of the thread count.
    pub(crate) fn merge(&self, f: &mut Fields<'_>, co: &Coefficients) {
        self.pairs(co, |_, axis, lo, hi, wl, wh| {
            let v = match pair_weights(wl, wh) {
                None => 0.0,
                Some((wl, wh)) => {
                    let (ul, uh) = (f.u.get(axis, lo) as f64, f.u.get(axis, hi) as f64);
                    ((wl * ul + wh * uh) / (wl + wh)) as f32
                }
            };
            f.u.set(axis, lo, v);
            f.u.set(axis, hi, v);
        });
    }

    /// [`merge`](Self::merge) on `f64` arrays in field layout (`u[a]` the
    /// E component `a`), for the stability estimate of
    /// [`cylindrical`](crate::engine::cylindrical).
    pub(crate) fn merge_f64(&self, u: &mut [Vec<f64>; 3], co: &Coefficients) {
        self.pairs(co, |_, axis, lo, hi, wl, wh| {
            let a = axis.index();
            let (il, ih) = (self.index(lo), self.index(hi));
            let v = match pair_weights(wl, wh) {
                None => 0.0,
                Some((wl, wh)) => (wl * u[a][il] + wh * u[a][ih]) / (wl + wh),
            };
            u[a][il] = v;
            u[a][ih] = v;
        });
    }

    /// The Bloch merge of [`BlochPair`]: the field is `re + j·im`, and the
    /// high copy of axis `a` is the low one times `e^{−jψ_a}`. Both copies
    /// are replaced by the weighted mean taken at the low copy,
    /// `U = (w_lo·u_lo + w_hi·u_hi·e^{jψ}) / (w_lo + w_hi)`, i.e.
    /// `u_lo = U`, `u_hi = U·e^{−jψ}`. With `ψ = 0` and `im = 0` this is
    /// [`merge`](Self::merge) bit for bit.
    fn merge_bloch(
        &self,
        re: &mut Fields<'_>,
        im: &mut Fields<'_>,
        co: &Coefficients,
        phase: &[(f64, f64); 3],
    ) {
        self.pairs(co, |a, axis, lo, hi, wl, wh| {
            let (c, s) = phase[a];
            let (vr, vi) = if let Some((wl, wh)) = pair_weights(wl, wh) {
                let (lr, li) = (re.u.get(axis, lo) as f64, im.u.get(axis, lo) as f64);
                let (hr, hi) = (re.u.get(axis, hi) as f64, im.u.get(axis, hi) as f64);
                // u_hi·e^{jψ} = (hr·c − hi·s) + j(hr·s + hi·c).
                let w = wl + wh;
                (
                    (wl * lr + wh * (hr * c - hi * s)) / w,
                    (wl * li + wh * (hr * s + hi * c)) / w,
                )
            } else {
                (0.0, 0.0)
            };
            re.u.set(axis, lo, vr as f32);
            im.u.set(axis, lo, vi as f32);
            // U·e^{−jψ} = (vr·c + vi·s) + j(vi·c − vr·s).
            re.u.set(axis, hi, (vr * c + vi * s) as f32);
            im.u.set(axis, hi, (vi * c - vr * s) as f32);
        });
    }
}

/// Bloch (Floquet) periodic boundaries by the sine–cosine method
/// (design §5.6.1): two real engines carry the real and imaginary parts of
/// a complex field `u = u_re + j·u_im`, coupled only at the periodic
/// faces, where the field obeys `F(x + L) = F(x)·e^{−jψ}` with the phase
/// shift `ψ = k_t·L` per period.
///
/// The pair is built from one [`EngineBuilder`]
/// whose Bloch axes have [`FaceCondition::Periodic`] on both faces; both
/// engines get the same grid, materials and time step, with those faces
/// built as PMC (the coefficient layout of a periodic face) and without
/// the engines' own zero-phase merge. After every step of both engines the
/// pair merges the face copies with the phase factor (the complex form of
/// the merge in the [module docs](self)).
///
/// Operators are added to the two engines separately
/// ([`real_mut`](Self::real_mut), [`imag_mut`](Self::imag_mut)); a real
/// source in the real engine alone excites the complex field. Operators
/// that change coefficients (lumped elements, sheets) or act on the
/// fields must be added identically to both engines, and should stay off
/// the periodic face planes: the merge runs after all hooks of a step, so
/// `post_e` operators there see the copies before the merge. Slab-parallel
/// boundaries (CPML on the other faces) are fine.
///
/// The energy [`energy`](Self::energy) is the sum of both engines' energy
/// invariants, which the Bloch-coupled scheme conserves (the `e^{−jψ}`
/// coupling is unitary).
///
/// [`Simulation`](crate::Simulation) drives such a pair itself when the
/// settings ask for Bloch boundaries ([`Fdtd::bloch_wavevector`](crate::Fdtd::bloch_wavevector)
/// or an oblique plane wave in a periodic cell, see the Bloch mode of
/// [`Simulation`](crate::Simulation#bloch-mode)).
///
/// ```
/// use ems::engine::boundary::periodic::BlochPair;
/// use ems::engine::{Engine, FaceCondition, FaceConditions, Grid, Probe};
/// use ems::{Axis, Face};
///
/// let g = Grid::uniform([8, 6, 6], [1e-3; 3])?;
/// let faces = FaceConditions::default()
///     .with(Face::XMin, FaceCondition::Periodic)
///     .with(Face::XMax, FaceCondition::Periodic);
/// let psi = BlochPair::phase_for(&g, [500.0, 0.0, 0.0]); // k_x = 500 rad/m
/// let mut pair = BlochPair::new(Engine::builder(g).faces(faces), psi)?;
/// let h = pair.real_mut().add_operator(Probe::voltage(Axis::Y, [4, 2, 3]));
/// pair.run(10);
/// assert_eq!(pair.real().operator(&h).values().len(), 10);
/// # Ok::<(), ems::Error>(())
/// ```
pub struct BlochPair {
    re: Engine,
    im: Engine,
    coupling: BlochCoupling,
}

impl core::fmt::Debug for BlochPair {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BlochPair")
            .field("phase", &self.coupling.phase)
            .field("re", &self.re)
            .field("im", &self.im)
            .finish_non_exhaustive()
    }
}

/// The face coupling of a Bloch pair: the periodic axes and the phase
/// factors of the merge. Shared by [`BlochPair`] and the Bloch mode of
/// [`Simulation`](crate::Simulation).
#[derive(Debug, Clone)]
pub(crate) struct BlochCoupling {
    periodic: Periodic,
    phase: [f64; 3],
    trig: [(f64, f64); 3],
}

impl BlochCoupling {
    /// Validates `phase` against the periodic faces of `builder` and
    /// returns the coupling plus the builder for both engines (periodic
    /// faces built as PMC, without the engines' own zero-phase merge).
    ///
    /// # Errors
    /// See [`BlochPair::new`].
    pub(crate) fn new(builder: EngineBuilder, phase: [f64; 3]) -> Result<(Self, EngineBuilder)> {
        let periodic =
            Periodic::from_faces(&builder.faces, builder.grid.nodes())?.ok_or_else(|| {
                Error::Invalid("Bloch pair: no face pair is FaceCondition::Periodic".into())
            })?;
        for a in Axis::ALL {
            let p = phase[a.index()];
            if !p.is_finite() || (p != 0.0 && !periodic.axes[a.index()]) {
                return Err(Error::Invalid(format!(
                    "Bloch phase {p} on axis {a} (must be finite, and 0 unless the axis is periodic)"
                )));
            }
        }
        let mut b = builder;
        for face in Face::ALL {
            if b.faces[face] == FaceCondition::Periodic {
                b.faces[face] = FaceCondition::Pmc;
            }
        }
        let coupling = Self {
            periodic,
            phase,
            trig: phase.map(|p| (p.cos(), p.sin())),
        };
        Ok((coupling, b))
    }

    /// The phase shifts per period \[rad\].
    pub(crate) fn phase(&self) -> [f64; 3] {
        self.phase
    }

    /// The Bloch merge of the face copies of `re` and `im`, after both
    /// engines finished a step.
    fn merge(&self, re: &Engine, im: &Engine) {
        let co = re.coefficients();
        let mut fr = re.shared.store.lock();
        let mut fi = im.shared.store.lock();
        self.periodic.merge_bloch(&mut fr, &mut fi, &co, &self.trig);
    }

    /// One step of both engines plus the merge; with `energy`, returns the
    /// summed energy invariant of the state before the step (0 otherwise).
    pub(crate) fn step(&self, re: &mut Engine, im: &mut Engine, energy: bool) -> f64 {
        let w = if energy {
            re.step_with_energy() + im.step_with_energy()
        } else {
            re.step();
            im.step();
            0.0
        };
        self.merge(re, im);
        w
    }
}

/// [`Stepper`] over the two engines of a Bloch pair.
pub(crate) struct BlochStepper<'a> {
    pub(crate) re: &'a mut Engine,
    pub(crate) im: &'a mut Engine,
    pub(crate) coupling: &'a BlochCoupling,
}

impl Stepper for BlochStepper<'_> {
    fn timestep(&self) -> u64 {
        self.re.timestep()
    }
    fn dt(&self) -> f64 {
        self.re.dt()
    }
    fn time(&self) -> f64 {
        self.re.time()
    }
    fn cells(&self) -> f64 {
        2.0 * self.re.grid().num_cells() as f64
    }
    fn step_with_energy(&mut self) -> f64 {
        self.coupling.step(self.re, self.im, true)
    }
    fn run(&mut self, n: u64) {
        for _ in 0..n {
            self.coupling.step(self.re, self.im, false);
        }
    }
}

impl BlochPair {
    /// Builds the two engines from `builder`; `phase[a]` is the phase shift
    /// `ψ_a` per period \[rad\] of the periodic axis `a` (0 for the others).
    ///
    /// # Errors
    /// [`Error::Invalid`] if no axis is periodic, a phase is not finite or
    /// is non-zero on a non-periodic axis, and the errors of
    /// [`EngineBuilder::build`](crate::engine::EngineBuilder::build).
    pub fn new(builder: EngineBuilder, phase: [f64; 3]) -> Result<Self> {
        let (coupling, b) = BlochCoupling::new(builder, phase)?;
        let re = b.clone().build()?;
        let im = b.build()?;
        Ok(Self { re, im, coupling })
    }

    /// The phase shifts `ψ = k·L` per period for a transverse wavevector
    /// `k` \[rad/m\] on `grid` (`L` = first to last mesh line of each
    /// axis).
    pub fn phase_for(grid: &Grid, k: [f64; 3]) -> [f64; 3] {
        Axis::ALL.map(|a| {
            let l = grid.lines(a);
            k[a.index()] * (l[l.len() - 1] - l[0])
        })
    }

    /// The phase shifts per period \[rad\].
    pub fn phase(&self) -> [f64; 3] {
        self.coupling.phase
    }

    /// The engine of the real part.
    pub fn real(&self) -> &Engine {
        &self.re
    }

    /// Mutable access to the engine of the real part (to add operators).
    pub fn real_mut(&mut self) -> &mut Engine {
        &mut self.re
    }

    /// The engine of the imaginary part.
    pub fn imag(&self) -> &Engine {
        &self.im
    }

    /// Mutable access to the engine of the imaginary part.
    pub fn imag_mut(&mut self) -> &mut Engine {
        &mut self.im
    }

    /// Mutable access to both engines at once, `(real, imag)`.
    pub fn engines_mut(&mut self) -> (&mut Engine, &mut Engine) {
        (&mut self.re, &mut self.im)
    }

    /// Advances both engines one step and applies the Bloch merge.
    pub fn step(&mut self) {
        self.coupling.step(&mut self.re, &mut self.im, false);
    }

    /// Advances one step like [`step`](Self::step) and returns the energy
    /// invariant of the state before it: the sum of both engines'
    /// [`Engine::step_with_energy`].
    pub fn step_with_energy(&mut self) -> f64 {
        self.coupling.step(&mut self.re, &mut self.im, true)
    }

    /// Advances `n` steps.
    pub fn run(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Runs both engines until the end criteria are met (with the energy
    /// summed over both) or `limits.max_timesteps` steps have run; the
    /// algorithm of [`Engine::run_until_with`].
    ///
    /// # Errors
    /// See [`Engine::run_until_with`].
    pub fn run_until(&mut self, limits: &RunLimits) -> Result<RunOutcome> {
        self.run_until_with(limits, |_| core::ops::ControlFlow::Continue(()))
    }

    /// Like [`run_until`](Self::run_until), calling `progress` after every
    /// energy check (see [`Engine::run_until_with`]).
    ///
    /// # Errors
    /// See [`Engine::run_until_with`].
    pub fn run_until_with<F>(&mut self, limits: &RunLimits, progress: F) -> Result<RunOutcome>
    where
        F: FnMut(&Progress) -> core::ops::ControlFlow<()>,
    {
        let mut s = BlochStepper {
            re: &mut self.re,
            im: &mut self.im,
            coupling: &self.coupling,
        };
        drive(&mut s, limits, progress)
    }

    /// The energy invariant of the complex field \[J\]: the sum of both
    /// engines' [`Engine::energy`].
    pub fn energy(&self) -> f64 {
        self.re.energy() + self.im.energy()
    }
}
