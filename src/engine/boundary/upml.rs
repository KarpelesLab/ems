//! openEMS-compatible PML: the uniaxial PML that runs `PML_<n>` faces of
//! openEMS files.
//!
//! ems's own absorbing layer is the CPML of [`cpml`](super::cpml)
//! (−84 dB at normal incidence with 8 cells). openEMS's `PML_8` reflects
//! about −58 dB, with the opposite sign of the residual, so models read
//! from openEMS files that end in PML layers differed from openEMS by
//! 1e-3…1e-2. For those files ([`Fdtd::openems_run`](crate::Fdtd::openems_run))
//! the faces are run by this operator instead, which reproduces openEMS's
//! layer bit for bit on Cartesian grids (design §5.16.2, §5.16.8).
//!
//! # Formulation
//!
//! The update is the uniaxial PML of S. D. Gedney, "An anisotropic
//! perfectly matched layer-absorbing medium for the truncation of FDTD
//! lattices", *IEEE Trans. Antennas Propag.* 44(12), 1630–1639, 1996
//! (also Taflove & Hagness, *Computational Electrodynamics*, 3rd ed.,
//! §7.8–7.9): the medium of the layer is the diagonal tensor
//! `ε·diag(s_y s_z/s_x, s_z s_x/s_y, s_x s_y/s_z)` (likewise for `μ`)
//! with `s_w = 1 + σ_w/(jω ε0)`. For a component `a` with `(p, q)` the
//! next and previous axes (cyclic, `p = a + 1`, `q = a + 2`) it is
//! integrated in two stages through an auxiliary flux `F`:
//!
//! ```text
//! jω·s_p·F_a = (∇×H)_a              (F: the flux density, D·l)
//! s_a·F_a    = ε·s_q·E_a
//! ```
//!
//! which in the integral form of the engine (`u = E·l`, `C = ε·A*/l`),
//! with `a_w = σ_w·dt/(2ε0)` and centred differences, becomes
//!
//! ```text
//! F' = (1 − a_p)/(1 + a_p)·F + dt/((A*/l)·(1 + a_p))·∮i
//! u' = ((1 − a_q)/(1 + a_q)·u − (1 − a_a)/((1 + a_q)·ε)·F) + (1 + a_a)/((1 + a_q)·ε)·F'
//! ```
//!
//! and the same for H with `μ`, `A/l*` and `−∮u`. Every edge of a layer
//! carries its flux `F` (also where the two stages would collapse into one
//! lossy update in exact arithmetic), and the `f32` operations run in the
//! order written: this is openEMS's arithmetic, fitted bit for bit from
//! its per-step field dumps (see [Where this comes from](self#where-this-comes-from)).
//! Each coefficient is evaluated in `f64` and rounded once to `f32`; the
//! `ε` and `μ` of the second stage are `ε_r·ε0'` and `μ_r·μ0'` with
//! openEMS's constants `ε0' = 1/(4π·10⁻⁷·c0²)` and `μ0' = 1.256637062·10⁻⁶`
//! (the grading `a_w` uses ems's, which give the same values). Each `a_w` is
//! taken where the edge lies along `w`: the edge midpoint along its own
//! direction and its start node (a primary line) along the others for E;
//! the node along its own direction and the dual lines (cell midpoints)
//! along the others for H. Overlapping layers (edges and corners of the
//! domain) simply have several non-zero `a_w`; opposite layers on the same
//! axis add their `a`.
//!
//! The layer's medium keeps its `ε` and `μ` (the capacitance and
//! inductance of the engine coefficients), but its electric and magnetic
//! conductivities are dropped, as openEMS does: inside the layer the update
//! above replaces the lossy one. Edges on the interface plane (where every
//! `a_w` is 0) keep their ordinary update, losses included.
//!
//! # Grading
//!
//! For a layer of `n` cells and physical width `W` (from the interface
//! line to the outer face), with `Δ = W/n` the mean cell width and `d > 0`
//! the physical depth below the interface,
//!
//! ```text
//! σ(d) = ln(10⁶)·ln(2.5) / (2·Δ·Z0·(2.5ⁿ − 1)) · 2.5^(d/Δ)
//! ```
//!
//! (an exponential profile whose total `∫σ` gives a round-trip attenuation
//! of 10⁻⁶ at normal incidence), and `σ = 0` at and outside the interface.
//! The same `σ` applies to E and H (`σ*/μ0 = σ/ε0`), in any medium (`Z0`,
//! `ε0`: the stretch does not depend on the material).
//!
//! # Where this comes from
//!
//! Black-box measurements of openEMS v0.37 (design §5.16.2, §5.16.8): its
//! `--debug-operator` dump shows, inside a `PML_n` layer, the coefficients
//! of the flux stage above (`vv = (1 − a_p)/(1 + a_p)`,
//! `vi = dt/((A/l)·(1 + a_p))`, every edge of a graded three-face model);
//! per-step dumps of all E and H components of small models with 1 m
//! cells (one to five faces, overlaps, `ε_r`, `μ_r` and lossy media in the
//! layers, both face sides, 18 models) are reproduced bit for bit at every
//! step by the update above. openEMS ignores `PML_Grading` (§5.16.1); the
//! profile above is the only one it runs.
//!
//! # Engine coupling
//!
//! Every layer edge gets `ca = 0` and `cb = dt/((A/l)·(1 + a_p))`, so that
//! the kernel writes the flux increment `t` into `u` (with no fused
//! multiply-add, `0·u + cb·∮i` is `cb·∮i` exactly, and `vv·F + t` then has
//! the bits of openEMS's flux update); a
//! [slab hook](crate::engine::SlabHooks) then forms `F'` and `u'` from its
//! stored `F` and `u` (two `f32` per edge). The stored `u` is taken back
//! from the fields in the other half-step, after the serial hooks, so that
//! sources and the periodic / axis merges acting on a layer edge are seen
//! as openEMS sees them: a source adds to the field, not to the flux. The
//! hook runs inside the half-step on the thread owning the slab; every
//! value is computed by the same expression whatever the partition, so
//! results are bit-identical for any thread count.
//!
//! The kernel's read-only H look-ahead sees the flux increment instead of
//! the current on the H edges of the layers.
//! The operator corrects openEMS's energy measure
//! ([`Engine::energy_openems`](crate::engine::Engine::energy_openems),
//! through [`SlabHooks::energy_openems_slab`]), so that it counts the
//! whole domain with the true fields as openEMS's does (the stop step and
//! level of a patch with `PML_6` faces are reproduced), and the last H
//! sample of probes and dumps ([`Operator::lookahead_current`]). The
//! energy invariant ([`Engine::energy`]) is not meaningful with this
//! layer.
//!
//! PEC edges inside a layer stay PEC. On cylindrical grids the same layer
//! is used on every face (openEMS does the same: its operator dump on a
//! `ρ_max` layer shows the same profile along `ρ`).

use core::ops::Range;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use crate::axis::Axis;
use crate::constants::{EPS0, MU0, Z0};
use crate::engine::grid::Grid;
use crate::engine::kernel::flush;
use crate::engine::operator::{Hooks, Operator};
use crate::engine::slab::{Fusion, SlabContext, SlabHooks, SlabMut, SlabPartition, SlabRef};
use crate::engine::{Coefficients, EdgeMaterials, Engine};
use crate::fdtd::Face;
use crate::{Error, Result};

/// The permittivity of vacuum \[F/m\] in the second stage of openEMS's
/// layer update: `1/(μ0·c0²)` with the former exact `μ0 = 4π·10⁻⁷`
/// (black box, design §5.16.8; ems's [`EPS0`] is CODATA 2022's).
const OPENEMS_PML_EPS0: f64 = 8.854_187_817_620_389e-12;

/// The permeability of vacuum \[H/m\] in the second stage of openEMS's
/// layer update: `1.256637062·10⁻⁶`, ten digits (black box, design §5.16.8).
const OPENEMS_PML_MU0: f64 = 1.256_637_062e-6;

/// Conductivity `σ(d)` \[S/m\] of openEMS's `PML_n` layer at physical depth
/// `depth` \[m\] below the interface, for a layer of `cells` cells and
/// physical width `width` \[m\] (see [Grading](self#grading)); 0 for
/// `depth ≤ 0`.
pub fn openems_pml_sigma(depth: f64, width: f64, cells: usize) -> f64 {
    if depth <= 0.0 || cells == 0 || width <= 0.0 {
        return 0.0;
    }
    let dl = width / cells as f64;
    let g = 2.5f64;
    (1e6f64).ln() * g.ln() / (2.0 * dl * Z0 * (g.powi(cells as i32) - 1.0)) * g.powf(depth / dl)
}

/// Per-axis `a = σ·dt/(2ε0)` at the mesh lines and at the cell midpoints.
#[derive(Debug, Clone)]
struct Profile {
    /// `[axis][line]`.
    node: [Vec<f64>; 3],
    /// `[axis][cell]` (cell `k` between lines `k` and `k + 1`; one extra
    /// 0 entry at the end).
    mid: [Vec<f64>; 3],
}

impl Profile {
    fn new(grid: &Grid, dt: f64, faces: &[(Face, usize)]) -> Self {
        let mut node: [Vec<f64>; 3] = core::array::from_fn(|a| vec![0.0; grid.nodes()[a]]);
        let mut mid: [Vec<f64>; 3] = core::array::from_fn(|a| vec![0.0; grid.nodes()[a]]);
        for &(face, cells) in faces {
            let ax = face.axis();
            let l = grid.lines(ax);
            let last = l.len() - 1;
            let (inner, outer) = if face.is_max() {
                (l[last - cells], l[last])
            } else {
                (l[cells], l[0])
            };
            let width = (outer - inner).abs();
            let depth = |x: f64| if face.is_max() { x - inner } else { inner - x };
            let a = |x: f64| openems_pml_sigma(depth(x), width, cells) * dt / (2.0 * EPS0);
            for k in 0..=last {
                node[ax.index()][k] += a(l[k]);
                if k < last {
                    mid[ax.index()][k] += a(0.5 * (l[k] + l[k + 1]));
                }
            }
        }
        Self { node, mid }
    }

    /// `[a_a, a_p, a_q]` of the E edge (`e_side`) or H dual edge along
    /// `comp` at `node`.
    fn at(&self, e_side: bool, comp: Axis, node: [usize; 3]) -> [f64; 3] {
        let (p, q) = comp.others();
        let (own, other) = if e_side {
            (&self.mid, &self.node)
        } else {
            (&self.node, &self.mid)
        };
        [
            own[comp.index()][node[comp.index()]],
            other[p.index()][node[p.index()]],
            other[q.index()][node[q.index()]],
        ]
    }
}

/// The layer update of one E edge (`e_side`) or H dual edge of `grid`
/// inside the `faces` layers (time step `dt`), for a caller that runs it
/// itself (openEMS's seam of a closed cylindrical grid,
/// [`OpenemsSeam`](crate::engine::openems_cyl::OpenemsSeam)): the flux
/// increment coefficient `dt/((A/l)·(1 + a_p))` and the four update
/// coefficients of [`Run::apply`], from the edge's geometry `geo = A/l`
/// and relative permittivity (permeability) `rel`; `None` outside the
/// layers.
pub(crate) fn layer_update(
    profile: &PmlProfile,
    dt: f64,
    e_side: bool,
    comp: Axis,
    node: [usize; 3],
    geo: f64,
    rel: f64,
) -> Option<(f32, [f32; 4])> {
    let [aa, ap, aq] = profile.0.at(e_side, comp, node);
    if aa == 0.0 && ap == 0.0 && aq == 0.0 {
        return None;
    }
    let m = if e_side {
        OPENEMS_PML_EPS0
    } else {
        OPENEMS_PML_MU0
    } * rel;
    Some((
        (dt / ((1.0 + ap) * geo)) as f32,
        [
            ((1.0 - ap) / (1.0 + ap)) as f32,
            ((1.0 - aq) / (1.0 + aq)) as f32,
            ((1.0 + aa) / ((1.0 + aq) * m)) as f32,
            ((1.0 - aa) / ((1.0 + aq) * m)) as f32,
        ],
    ))
}

/// One step of the layer update of [`Run::apply`]: the flux `f` and the
/// value `v` the update starts from, the kernel's flux increment `t`;
/// returns the new value (and updates `f`).
pub(crate) fn layer_step(co: [f32; 4], f: &mut f32, v: f32, t: f32) -> f32 {
    let [ap, aq, bp, bm] = co;
    let f0 = *f;
    let f1 = flush(flush(ap * f0) + t);
    *f = f1;
    flush(flush(flush(aq * v) - flush(bm * f0)) + flush(bp * f1))
}

/// The grading of the layers on `faces` (see [`layer_update`]).
#[derive(Debug, Clone)]
pub(crate) struct PmlProfile(Profile);

impl PmlProfile {
    /// The profile of `faces` (face, cells) on `grid` for time step `dt`.
    pub(crate) fn new(grid: &Grid, dt: f64, faces: &[(Face, usize)]) -> Self {
        Self(Profile::new(grid, dt, faces))
    }
}

/// Which edges lie inside the layers of an [`OpenEmsPml`] (those whose
/// update the layer replaces; the interface planes are outside), from
/// [`OpenEmsPml::layers`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PmlLayers(Profile);

impl PmlLayers {
    /// Whether the E edge (`e_side`) or H dual edge along `comp` at `node`
    /// lies inside a layer.
    pub fn contains(&self, e_side: bool, comp: Axis, node: [usize; 3]) -> bool {
        self.0.at(e_side, comp, node).iter().any(|&a| a != 0.0)
    }
}

/// The edges with state of one component in one x-plane, in contiguous
/// segments along z. The coefficients of a row depend on `k` and on the
/// row's position, and most rows of a layer share them, so they are kept
/// once per distinct row ("pattern") and the update streams only the
/// field and the state.
#[derive(Debug, Clone)]
struct Run {
    comp: Axis,
    plane: usize,
    nz: usize,
    /// Segments `(j·nz + k, length, pattern)` of consecutive edges.
    seg: Vec<(u32, u32, u32)>,
    /// Distinct coefficient rows (each `nz` long, indexed by `k`):
    /// `(1 − a_p)/(1 + a_p)`, `(1 − a_q)/(1 + a_q)`, `(1 + a_a)/(1 + a_q)`,
    /// `(1 − a_a)/(1 + a_q)`.
    pats: Vec<[Vec<f32>; 4]>,
    /// Flux `F` per edge, in segment order.
    f: Vec<f32>,
    /// The value the update starts from (`u` or `i`) per edge: the last
    /// one written, refreshed from the fields in the other half-step.
    v: Vec<f32>,
}

impl Run {
    fn new(comp: Axis, plane: usize, nz: usize) -> Self {
        Self {
            comp,
            plane,
            nz,
            seg: Vec::new(),
            pats: Vec::new(),
            f: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Adds row `j`: the edges at the `k` where `mask` is set, with the
    /// coefficient row `co` (zero elsewhere); `known` maps the bits of the
    /// rows seen so far to their pattern.
    fn push_row(
        &mut self,
        j: usize,
        mask: &[bool],
        co: [Vec<f32>; 4],
        known: &mut HashMap<Vec<u32>, usize>,
    ) {
        if !mask.iter().any(|&m| m) {
            return;
        }
        let key: Vec<u32> = co.iter().flatten().map(|x| x.to_bits()).collect();
        let pat = *known.entry(key).or_insert_with(|| {
            self.pats.push(co);
            self.pats.len() - 1
        });
        let mut k = 0;
        while k < self.nz {
            if !mask[k] {
                k += 1;
                continue;
            }
            let k0 = k;
            while k < self.nz && mask[k] {
                k += 1;
            }
            self.seg
                .push(((j * self.nz + k0) as u32, (k - k0) as u32, pat as u32));
            self.f.extend(core::iter::repeat_n(0.0, k - k0));
        }
        self.v.resize(self.f.len(), 0.0);
    }

    fn len(&self) -> usize {
        self.f.len()
    }

    /// The state index and coefficients of the edge at in-plane offset
    /// `off`, if it is one of the run's.
    fn find(&self, off: usize) -> Option<(usize, [f32; 4])> {
        let mut q = 0;
        for &(o, n, p) in &self.seg {
            let (o, n) = (o as usize, n as usize);
            if (o..o + n).contains(&off) {
                let k = off % self.nz;
                let p = &self.pats[p as usize];
                return Some((q + off - o, [p[0][k], p[1][k], p[2][k], p[3][k]]));
            }
            q += n;
        }
        None
    }

    /// In-plane offset and coefficients of every edge, in order.
    fn edges(&self) -> impl Iterator<Item = (usize, [f32; 4])> + '_ {
        self.seg.iter().flat_map(move |&(o, n, p)| {
            let p = &self.pats[p as usize];
            (o as usize..(o + n) as usize).map(move |o| {
                let k = o % self.nz;
                (o, [p[0][k], p[1][k], p[2][k], p[3][k]])
            })
        })
    }

    /// Takes the current values of the edges from `src` as the values the
    /// next update starts from (operators may have changed them after
    /// [`apply`](Self::apply): sources, the periodic and axis merges).
    fn refresh(&mut self, src: &SlabRef<'_>) {
        let t = src.plane(self.comp, self.plane);
        let mut q = 0;
        for &(o, n, _) in &self.seg {
            let (o, n) = (o as usize, n as usize);
            self.v[q..q + n].copy_from_slice(&t[o..o + n]);
            q += n;
        }
    }

    /// Completes the update: the kernel left the flux increment in the
    /// target array.
    fn apply(&mut self, tgt: &mut SlabMut<'_>) {
        let t = tgt.plane_mut(self.comp, self.plane);
        let mut q = 0;
        for &(o, n, p) in &self.seg {
            let (o, n) = (o as usize, n as usize);
            let k0 = o % self.nz;
            let t = &mut t[o..o + n];
            let [ap, aq, bp, bm] = &self.pats[p as usize];
            let (ap, aq) = (&ap[k0..k0 + n], &aq[k0..k0 + n]);
            let (bp, bm) = (&bp[k0..k0 + n], &bm[k0..k0 + n]);
            let (f, v) = (&mut self.f[q..q + n], &mut self.v[q..q + n]);
            q += n;
            for k in 0..n {
                let f0 = f[k];
                let f1 = flush(flush(ap[k] * f0) + t[k]);
                let x = flush(flush(flush(aq[k] * v[k]) - flush(bm[k] * f0)) + flush(bp[k] * f1));
                f[k] = f1;
                v[k] = x;
                t[k] = x;
            }
        }
    }
}

/// The runs of one x-slab.
#[derive(Debug, Default)]
struct Pieces {
    e: Vec<Run>,
    h: Vec<Run>,
}

/// The state partitioned over the engine's slabs.
#[derive(Debug)]
struct Parts {
    slabs: Vec<Mutex<Pieces>>,
    /// The slab of each x-plane.
    slab_of: Vec<usize>,
}

impl Parts {
    fn pieces(&self, slab: usize) -> std::sync::MutexGuard<'_, Pieces> {
        self.slabs[slab]
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl SlabHooks for Parts {
    fn post_h_slab(&self, i: &mut SlabMut<'_>, u: &SlabRef<'_>, ctx: &SlabContext<'_>) {
        let mut p = self.pieces(ctx.slab);
        // `u` is final now (sources, merges and other serial hooks of the
        // previous step included): the E update of this step starts from it.
        for r in &mut p.e {
            r.refresh(u);
        }
        for r in &mut p.h {
            r.apply(i);
        }
    }

    fn post_e_slab(&self, u: &mut SlabMut<'_>, i: &SlabRef<'_>, ctx: &SlabContext<'_>) {
        let mut p = self.pieces(ctx.slab);
        for r in &mut p.h {
            r.refresh(i);
        }
        for r in &mut p.e {
            r.apply(u);
        }
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
        let mut p = self.pieces(ctx.slab);
        for r in p.e.iter_mut().filter(|r| planes.contains(&r.plane)) {
            r.refresh(u);
        }
        for r in p.h.iter_mut().filter(|r| planes.contains(&r.plane)) {
            r.apply(i);
        }
    }

    fn post_e_planes(
        &self,
        u: &mut SlabMut<'_>,
        i: &SlabRef<'_>,
        ctx: &SlabContext<'_>,
        planes: Range<usize>,
    ) {
        let mut p = self.pieces(ctx.slab);
        for r in p.h.iter_mut().filter(|r| planes.contains(&r.plane)) {
            r.refresh(i);
        }
        for r in p.e.iter_mut().filter(|r| planes.contains(&r.plane)) {
            r.apply(u);
        }
    }

    /// The H edges with state: the kernel's look-ahead yields their flux
    /// increment; the true `i^{n+½}` follows from it and the state as in
    /// the layer update.
    fn energy_openems_slab(
        &self,
        slab: usize,
        u: &SlabRef<'_>,
        coefficients: &Coefficients,
        add: &mut dyn FnMut(usize, f64),
    ) {
        let n = u.nodes();
        let nz = n[2];
        for r in &self.pieces(slab).h {
            let i = r.plane;
            let pl = n[1] * nz;
            let db = &coefficients.db(r.comp)[i * pl..(i + 1) * pl];
            let (ux, uy, uz) = (
                u.plane(Axis::X, i),
                u.plane(Axis::Y, i),
                u.plane(Axis::Z, i),
            );
            // The next plane (H_y, H_z dual edges exist below the last one).
            let next = |a: Axis| u.try_plane(a, i + 1).unwrap_or(&[]);
            let (uyn, uzn) = match r.comp {
                Axis::X => (&[][..], &[][..]),
                _ => (next(Axis::Y), next(Axis::Z)),
            };
            let mut sum = 0.0f64;
            for (q, (o, [ap, aq, bp, bm])) in r.edges().enumerate() {
                // The kernel's circulation (engine::kernel::h_slab).
                let t = match r.comp {
                    Axis::X => (uz[o + nz] - uz[o]) - (uy[o + 1] - uy[o]),
                    Axis::Y => (ux[o + 1] - ux[o]) - (uzn[o] - uz[o]),
                    Axis::Z => (uyn[o] - uy[o]) - (ux[o + nz] - ux[o]),
                };
                let inc = flush(0.0 * r.v[q] - db[o] * t);
                let f0 = r.f[q];
                let f1 = flush(flush(ap * f0) + inc);
                let v = flush(flush(flush(aq * r.v[q]) - flush(bm * f0)) + flush(bp * f1));
                sum += v as f64 * v as f64 - inc as f64 * inc as f64;
            }
            add(i, MU0 * sum);
        }
    }
}

/// openEMS's `PML_<n>` layers on any set of faces, as one engine
/// [`Operator`] (see the [module docs](self)).
///
/// Build the engine with PEC on the layer faces, then create the operator
/// (which rewrites the coefficients of the layer edges) and register it:
///
/// ```
/// use ems::engine::boundary::upml::OpenEmsPml;
/// use ems::engine::{Engine, Grid};
/// use ems::Face;
///
/// let grid = Grid::uniform([30, 30, 30], [1e-3; 3])?;
/// let mut engine = Engine::builder(grid).threads(1).build()?;
/// let pml = OpenEmsPml::new(&mut engine, Face::ALL.map(|f| (f, 8)))?;
/// engine.add_operator(pml);
/// engine.run(10);
/// # Ok::<(), ems::Error>(())
/// ```
///
/// # Panics
/// Registering it ([`Engine::add_operator`]) panics if the engine's grid
/// or time step differ from those it was built for.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OpenEmsPml {
    nodes: [usize; 3],
    dt: f64,
    faces: Vec<(Face, usize)>,
    /// Edges with state, E then H, in index order (unpartitioned).
    e: Vec<Run>,
    h: Vec<Run>,
    /// Edges whose coefficients were rewritten (E, H).
    edges: [usize; 2],
    parts: Option<Arc<Parts>>,
    layers: PmlLayers,
}

impl OpenEmsPml {
    /// Layers of `cells` cells on each listed face of `engine` (its grid,
    /// time step and coefficients): rewrites the coefficients of the layer
    /// edges and prepares their state (see the [module docs](self)).
    /// Create it after every change of the medium coefficients. Lumped
    /// elements must not act on layer edges ([`layers`](Self::layers)):
    /// openEMS ignores their R and C there.
    ///
    /// # Errors
    /// [`Error::Invalid`] if a face is listed twice, has no cells or more
    /// cells than the mesh along its axis.
    pub fn new(
        engine: &mut Engine,
        faces: impl IntoIterator<Item = (Face, usize)>,
    ) -> Result<Self> {
        Self::build(engine, faces, None)
    }

    /// Like [`new`](Self::new), with the edge materials the engine was
    /// built from: the second stage of the update divides by the edge's
    /// `ε0·ε_r` (`μ0·μ_r`), as openEMS does. [`new`](Self::new) takes the
    /// capacitance (inductance) from the engine coefficients instead,
    /// which differs in the last bits.
    ///
    /// # Errors
    /// As [`new`](Self::new), and if the materials do not match the grid.
    pub fn with_materials(
        engine: &mut Engine,
        faces: impl IntoIterator<Item = (Face, usize)>,
        materials: &EdgeMaterials,
    ) -> Result<Self> {
        if materials.nodes() != engine.grid().nodes() {
            return Err(Error::Invalid(
                "PML: the edge materials do not match the grid".into(),
            ));
        }
        Self::build(engine, faces, Some(materials))
    }

    fn build(
        engine: &mut Engine,
        faces: impl IntoIterator<Item = (Face, usize)>,
        materials: Option<&EdgeMaterials>,
    ) -> Result<Self> {
        let grid = engine.grid().clone();
        let dt = engine.dt();
        let n = grid.nodes();
        let mut seen = [false; 6];
        let mut list = Vec::new();
        for (face, cells) in faces {
            if std::mem::replace(&mut seen[face.index()], true) {
                return Err(Error::Invalid(format!("PML: face {face} given twice")));
            }
            let total = n[face.axis().index()] - 1;
            if cells == 0 || cells > total {
                return Err(Error::Invalid(format!(
                    "PML on {face}: {cells} cells, the mesh has {total} along {}",
                    face.axis()
                )));
            }
            list.push((face, cells));
        }
        let prof = Profile::new(&grid, dt, &list);
        let cyl = grid.is_cylindrical();
        let mut co = engine.coefficients_mut();
        let (mut e_runs, mut h_runs) = (Vec::new(), Vec::new());
        let mut edges = [0usize; 2];
        for (side, e_side) in [(0usize, true), (1, false)] {
            for comp in Axis::ALL {
                let (p, q) = comp.others();
                let (ca, cb) = if e_side {
                    co.e_mut(comp)
                } else {
                    co.h_mut(comp)
                };
                for i in 0..n[0] {
                    let mut run = Run::new(comp, i, n[2]);
                    let mut known = HashMap::new();
                    for j in 0..n[1] {
                        let mut mask = vec![false; n[2]];
                        let mut row: [Vec<f32>; 4] = core::array::from_fn(|_| vec![0.0; n[2]]);
                        for k in 0..n[2] {
                            let node = [i, j, k];
                            let idx = grid.index(node);
                            if cb[idx] == 0.0 {
                                continue; // PEC, padding or non-existing
                            }
                            let [aa, ap, aq] = prof.at(e_side, comp, node);
                            if aa == 0.0 && ap == 0.0 && aq == 0.0 {
                                continue;
                            }
                            // A/l of the edge (E) or dual edge (H), as the
                            // engine coefficients take it.
                            let geo = match (e_side, cyl) {
                                (true, true) => {
                                    grid.dual_face_area(comp, node) / grid.edge_length(comp, node)
                                }
                                (true, false) => {
                                    grid.dual_delta(p, node[p.index()])
                                        * grid.dual_delta(q, node[q.index()])
                                        / grid.delta(comp, node[comp.index()])
                                }
                                (false, true) => {
                                    grid.face_area(comp, node) / grid.dual_edge_length(comp, node)
                                }
                                (false, false) => {
                                    grid.delta(p, node[p.index()]) * grid.delta(q, node[q.index()])
                                        / grid.dual_delta(comp, node[comp.index()])
                                }
                            };
                            // ε (μ) of the edge's medium.
                            let m = match materials {
                                Some(mt) if e_side => OPENEMS_PML_EPS0 * mt.eps_r(comp)[idx],
                                Some(mt) => OPENEMS_PML_MU0 * mt.mu_r(comp)[idx],
                                None => Coefficients::capacitance(ca[idx], cb[idx], dt) / geo,
                            };
                            edges[side] += 1;
                            // Flux stage: the kernel writes the increment
                            // dt·∮/(A/l·(1 + a_p)).
                            ca[idx] = 0.0;
                            cb[idx] = (dt / ((1.0 + ap) * geo)) as f32;
                            mask[k] = true;
                            row[0][k] = ((1.0 - ap) / (1.0 + ap)) as f32;
                            row[1][k] = ((1.0 - aq) / (1.0 + aq)) as f32;
                            row[2][k] = ((1.0 + aa) / ((1.0 + aq) * m)) as f32;
                            row[3][k] = ((1.0 - aa) / ((1.0 + aq) * m)) as f32;
                        }
                        run.push_row(j, &mask, row, &mut known);
                    }
                    if run.len() > 0 {
                        if e_side {
                            e_runs.push(run);
                        } else {
                            h_runs.push(run);
                        }
                    }
                }
            }
        }
        drop(co);
        Ok(Self {
            nodes: n,
            dt,
            faces: list,
            e: e_runs,
            h: h_runs,
            edges,
            parts: None,
            layers: PmlLayers(prof),
        })
    }

    /// The edges inside the layers.
    pub fn layers(&self) -> &PmlLayers {
        &self.layers
    }

    /// The faces and their cell counts, in the order given.
    pub fn faces(&self) -> &[(Face, usize)] {
        &self.faces
    }

    /// Number of E edges and H dual edges inside the layers (those whose
    /// update was changed).
    pub fn edge_counts(&self) -> [usize; 2] {
        self.edges
    }

    /// Number of E edges and H dual edges that carry state (flux and last
    /// value): every edge inside the layers, as in openEMS.
    pub fn state_counts(&self) -> [usize; 2] {
        let count = |runs: &[Run]| runs.iter().map(Run::len).sum();
        [count(&self.e), count(&self.h)]
    }
}

impl Operator for OpenEmsPml {
    fn hooks(&self) -> Hooks {
        Hooks::NONE
    }

    fn has_lookahead_currents(&self) -> bool {
        !self.h.is_empty()
    }

    /// The H edges with state: the kernel's value is their flux
    /// increment, completed as in the layer update.
    fn lookahead_current(&self, axis: Axis, node: [usize; 3], kernel: f32) -> Option<f32> {
        let parts = self.parts.as_ref()?;
        let p = parts.pieces(*parts.slab_of.get(node[0])?);
        let r = p.h.iter().find(|r| r.comp == axis && r.plane == node[0])?;
        let (q, [ap, aq, bp, bm]) = r.find(node[1] * self.nodes[2] + node[2])?;
        let f0 = r.f[q];
        let f1 = flush(flush(ap * f0) + kernel);
        Some(flush(
            flush(flush(aq * r.v[q]) - flush(bm * f0)) + flush(bp * f1),
        ))
    }

    fn slab_hooks(&mut self, part: &SlabPartition<'_>) -> Option<Arc<dyn SlabHooks>> {
        assert_eq!(
            part.grid().nodes(),
            self.nodes,
            "PML built for another grid"
        );
        assert_eq!(part.dt(), self.dt, "PML built for another time step");
        if self.e.is_empty() && self.h.is_empty() {
            return None;
        }
        let slabs = (0..part.slabs())
            .map(|s| {
                let r = part.planes(s);
                let take = |runs: &[Run]| {
                    runs.iter()
                        .filter(|run| r.contains(&run.plane))
                        .cloned()
                        .collect()
                };
                Mutex::new(Pieces {
                    e: take(&self.e),
                    h: take(&self.h),
                })
            })
            .collect();
        let mut slab_of = vec![0; self.nodes[0]];
        for s in 0..part.slabs() {
            for x in part.planes(s) {
                slab_of[x] = s;
            }
        }
        let parts = Arc::new(Parts { slabs, slab_of });
        self.parts = Some(parts.clone());
        Some(parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `vv` of openEMS's operator dump at depth 1 mm in a uniform 1 mm
    /// `PML_4` (`(1 − a)/(1 + a)`, f32), dt = 1.925833270632893e-12 s.
    #[test]
    fn profile_matches_openems_dump() {
        let dt = 1.925833270632893e-12;
        let v = |d: f64| {
            let a = openems_pml_sigma(d * 1e-3, 4e-3, 4) * dt / (2.0 * EPS0);
            ((1.0 - a) / (1.0 + a)) as f32
        };
        for (d, want) in [
            (0.5, 0.8589056f32),
            (1.0, 0.7856962),
            (1.5, 0.68101895),
            (2.0, 0.53842795),
            (2.5, 0.35649642),
            (3.0, 0.14281084),
            (3.5, -0.08507279),
        ] {
            assert!((v(d) - want).abs() < 1e-7, "{d}: {} vs {want}", v(d));
        }
        assert_eq!(openems_pml_sigma(0.0, 4e-3, 4), 0.0);
    }

    #[test]
    fn rejects_bad_faces() {
        let grid = Grid::uniform([6, 6, 6], [1e-3; 3]).unwrap();
        let mut e = Engine::builder(grid).threads(1).build().unwrap();
        assert!(OpenEmsPml::new(&mut e, [(Face::XMin, 7)]).is_err());
        assert!(OpenEmsPml::new(&mut e, [(Face::XMin, 0)]).is_err());
        assert!(OpenEmsPml::new(&mut e, [(Face::XMin, 2), (Face::XMin, 2)]).is_err());
    }
}
