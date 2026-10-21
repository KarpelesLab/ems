//! Conformal (sub-cell) PEC: curved metal surfaces on the Cartesian grid
//! without staircasing (design §5.13).
//!
//! # Method
//!
//! The Dey–Mittra conformal PEC scheme (S. Dey, R. Mittra, "A locally
//! conformal finite-difference time-domain (FDTD) algorithm for modeling
//! three-dimensional perfectly conducting objects", IEEE Microwave Guided
//! Wave Lett. 7(9), 1997) evaluates Faraday's law on the part of each cell
//! face that lies outside the metal,
//!
//! ```text
//! μ·A_open·dH/dt = −∮_open E·dl = −Σ_e E_e·l_e,open ,
//! ```
//!
//! while Ampère's law is left as it is. In the integral form of the engine
//! (design §5.2) this is a pure change of coefficients once the stored
//! edge voltage of a cut edge is read as the voltage over its *open part*,
//! `u = E·l_open = f_e·E·l` (`f_e`: open length fraction of the edge):
//!
//! * the E update `E ← E + (dt/ε)·curl H` becomes
//!   `u ← u + f_e·cb·Kᵀi`: **`cb → f_e·cb`** (edge capacitance `C/f_e`;
//!   `ca` is unchanged because `C` and `G` scale alike);
//! * the H update becomes `i ← i − (db/f_A)·K·u` with the ordinary curl
//!   `K`: **`db → db/f_A`** (`f_A`: open area fraction of the face; the
//!   inductance becomes `f_A·L`, `da` is unchanged).
//!
//! The scheme keeps the symmetric form `C·du/dt = Kᵀi`, `L·di/dt = −K·u`
//! of the uncut grid (it is the finite-integration reading of
//! Dey–Mittra), so the engine's energy invariant (module docs of
//! [`engine`](super)) stays exact. Edges with `f_e = 0` are PEC.
//!
//! # Small cut faces: interpolation
//!
//! A face that keeps a long open edge but little area has a small `f_A·L`
//! and a local resonance far above the Courant limit of the uncut grid:
//! plain Dey–Mittra needs a time step `∝ √f_A,min`. Raising `f_A` to a
//! minimum cures that only for minima around 0.3–0.5, and then costs most
//! of the accuracy gain (first-order convergence again). Every face with
//! `f_A <` [`ConformalOptions::merge_below`] therefore loses its own
//! current: its current is a fixed combination of the currents of
//! neighbouring faces, `i = P·g` with `g` the remaining (reduced)
//! currents, and the reduced system is the Galerkin (Ritz) restriction of
//! Dey–Mittra to that subspace, in the spirit of the uniformly stable
//! conformal scheme (I. Zagorodnov, R. Schuhmann, T. Weiland, "A uniformly
//! stable conformal FDTD-method in Cartesian grids", Int. J. Numer. Model.
//! 16, 2003):
//!
//! ```text
//! C·du/dt = Kᵀ·P·g,     (PᵀLP)·dg/dt + (PᵀRP)·g = −Pᵀ·K·u ,
//! ```
//!
//! symmetric, so the energy `½uᵀCu + ½gᵀ(PᵀLP)g` is conserved, and
//! `PᵀLP ≥` the reduced faces' own inductances, so no small mass is left.
//!
//! * **Interpolation** (the default): the small face's `H` is the affine
//!   interpolation, at the centroid of its open area, of three faces of
//!   the same plane with their own current: the *host* across its most
//!   open edge and one neighbour of the host along each in-plane axis
//!   (preferably away from the small face; barycentric weights, at most
//!   2.5 in magnitude). Reproducing affine fields makes the restriction
//!   second-order accurate for every field component, including `E`
//!   tangential to the wall (TE modes). The hosts sharing small faces are
//!   coupled through `PᵀLP`: they form blocks solved each step with an
//!   envelope Cholesky factor (reverse Cuthill–McKee order), cheap because
//!   a block is a band along the wall. On the α faces of a cylindrical
//!   grid, whose dual edge `ρ·Δα*` varies across the face, the small
//!   face's current and inductance are those of the dual edge through its
//!   centroid (`ρ_c·Δα*`, not the face centre's): measured at the centre
//!   the interpolated `H` converged at first order there (TM010 of a
//!   coaxial shell, +0.21 / +0.081 / +0.038 % at 1 / 0.5 / 0.25 mm).
//! * **Merging** (fallback, and the scheme of faces given without a
//!   centroid, [`ConformalPec::push_face`]): `P` is 0/1, the small face
//!   shares the current of the neighbour across its most-open edge (a group
//!   still below the threshold merges again), with `L_g = Σ f_A·L` and the
//!   circulation of the merged contour, in the spirit of the enlarged-cell
//!   technique (T. Xiao, Q. H. Liu, IEEE MWCL 14(12), 2004). Constant
//!   extrapolation of `H` over a cell is first order: measured on TE11 of
//!   a circular cavity, +0.58 / +0.26 % at 10 / 20 cells per radius, against
//!   +0.076 / +0.018 % interpolated. It remains for small faces without a
//!   valid stencil (no host, a degenerate or too extrapolating triangle) and
//!   inside absorbing layers.
//!
//! The reduced currents are updated in the serial `post_h` hook of the
//! [`MergedFaces`] operator, which writes every face they drive; those
//! faces get `da = 1`, `db = 0`, so the kernel leaves them alone, and the
//! operator reports `½·g^{n−½}ᵀ·(PᵀLP)·g^{n+½}` ([`Operator::energy`]), so
//! the invariant remains exact. A small face without an open in-plane
//! neighbour keeps its own current with `f_A` raised to the threshold
//! ([`ConformalReport::floored`]).
//!
//! # Deviation currents: no spurious static fields
//!
//! Removing a small face's own current enlarges the null space of the
//! reduced curl `PᵀK`: an `E` field circulating around the small face
//! (on its cut edges) that the weighted host circulations do not see is
//! static in the reduced scheme although its curl is not zero. In a
//! cross-section these are extra zero-frequency modes; along a wall
//! (z-invariant cut cells) they travel as a TEM-like wave at about `c`,
//! guided by the cut cells: reflected where a wall ends in open space
//! (guide walls ending one cell before a CPML: energy stuck at −26 dB),
//! trapped between PEC plates, and next to a CPML (hard merge groups in
//! the layer, interpolated faces outside) growing exponentially. Plain
//! Dey–Mittra (no reduction, small time step) has none of this.
//!
//! So every interpolated face, and every small member of a merge group,
//! also keeps a **deviation current** `d`: `i_s = Σ w·g + d_s` (merge:
//! `i_m = g + d_m`). `P` is then square and invertible (no extra null
//! space) and the Ritz system gets an extra inductance on the deviations
//! only, `M = PᵀLP + diag(a·L)` with `a =`
//! [`ConformalOptions::deviation_area`]: `M_dd = (f_A + a)·L`. The Schur
//! complement of each deviation is at least `a·L` (the hosts' mass
//! contains `f_A·L·w·wᵀ`), so its local resonance is that of a face of
//! open fraction `a`, below the Courant limit like the area floors, and
//! the time step is unchanged. For smooth fields the deviation is the
//! interpolation error, `O(h²)`, so the extra inductance costs nothing in
//! accuracy (measured: equal or smaller errors on every cavity of design
//! §5.13; second order on the sphere's modes, which were pre-asymptotic
//! before). The symmetric form and the energy invariant are unchanged;
//! merge groups with deviations are solved as blocks like the hosts.
//!
//! # Absorbing layers
//!
//! Inside CPML layers ([`ConformalPec::set_cpml`]) small faces are merged,
//! not interpolated (interpolation across the graded stretching grew
//! exponentially), and no stencil reaches into a layer. The CPML corrects
//! only faces with a kernel `db`, so the merge groups apply its H-side
//! correction to their members themselves (`Ψ ← b·Ψ + a·T`, circulation
//! `+ (1/κ − 1)·T + Ψ`, with the CPML's grading); without it the field at
//! the conductors was not absorbed (a conformal coaxial line in CPML read
//! a 30 % impedance error from the reflection). The correction of a
//! member goes to every reduced current driving it (the group and the
//! member's deviation). The cylindrical PML of a `ρ_max` face is applied
//! the same way: its filter `h⁻¹` of the member's circulation (with the
//! z / α layers' stretch inside it in the corners), see
//! [`set_cpml`](ConformalPec::set_cpml); unstretched, the merged faces in
//! that layer made a run diverge. Interpolating in the layers instead and
//! area floors there were measured worse (floors: +0.51 % instead of
//! +0.07 % on the coaxial impedance, from the mismatch at the layer).

//! # Stability check
//!
//! The lossless scheme is stable iff `λ_max(S) < 4` for the symmetric
//! `S = √cb·(PK)ᵀ·dt·(PᵀLP)⁻¹·(PK)·√cb` (`cb·db = dt²/(L·C)`).
//! [`ConformalPec::apply`] estimates `s = √λ_max(S)/2` by a deterministic
//! power iteration ([`stability_factor`]) started on the cut cells, where
//! an unstable mode would be localised (`s < 1`: stable; an uncut grid at
//! Courant factor `S_c` gives `s ≈ S_c`). If `s >`
//! [`ConformalOptions::max_stability`], the threshold is raised (×1.25, up
//! to 1) and the construction repeated, at most
//! [`ConformalOptions::max_rounds`] times; [`ConformalReport::verified`]
//! tells whether it passed. Power iteration approaches `λ_max` from below,
//! so `s` is an estimate, not a bound; the long-run tests check
//! boundedness directly.
//!
//! # Thin PEC sheets
//!
//! Zero-thickness PEC sheets cutting cells ([`SheetCutEdge`],
//! [`SheetCutFace`]) are modelled by splitting the cut edges into pieces
//! and the cut faces into regions on either side of the sheet, each with
//! its own voltage or current, and the region contours running along the
//! sheet; the split system is again `C·du/dt = K'ᵀi`, `L·di/dt = −K'u`, so
//! the energy invariant stays exact. Small regions are interpolated (or,
//! as the fallback, merged) like small cut faces, from regions on their
//! own side of the sheet, and the stability check above runs on the
//! combined system
//! (the pieces as extra unknowns). The [`ConformalSheets`] operator holds
//! the split unknowns; see its module docs (`conformal/sheets.rs`) for
//! the coupling with the kernel. Faces owned by the sheet system are never
//! used by the solid-metal scheme.

use std::collections::{HashMap, HashSet};

use super::coeffs::Coefficients;
use super::fields::Fields;
use super::kernel::flush;
use super::operator::{Hooks, Operator, StepContext};
use super::{Engine, Grid};
use crate::Face;
use crate::axis::Axis;
use crate::engine::boundary::cpml::{CpmlBoundary, CpmlCoefficients};
use crate::engine::boundary::cyl_pml::{CylLayer, FaceStretch};
use crate::{Error, Result};

mod sheets;

pub use sheets::{ConformalSheets, SheetCutEdge, SheetCutFace, SheetRegion};

/// An E edge partly inside metal: open length fraction `fraction` in
/// `[0, 1)` (0: PEC).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalEdge {
    /// Edge direction.
    pub axis: Axis,
    /// Start node.
    pub node: [usize; 3],
    /// Open (non-metal) fraction of the edge length.
    pub fraction: f64,
}

/// A cell face (H dual edge) partly inside metal: open area fraction
/// `fraction` in `[0, 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalFace {
    /// Face normal (the H component).
    pub axis: Axis,
    /// Lowest corner node.
    pub node: [usize; 3],
    /// Open (non-metal) fraction of the face area.
    pub fraction: f64,
}

/// Options of [`ConformalPec::apply`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalOptions {
    /// Faces with an open area fraction below this lose their own current:
    /// interpolated from neighbours, or merged with one (default 0.5; see
    /// the [module docs](self#small-cut-faces-interpolation)).
    pub merge_below: f64,
    /// Accepted stability factor `s = √λ_max/2` (default 0.999).
    pub max_stability: f64,
    /// Power iterations per check (default 300; 0 skips the check).
    pub iterations: usize,
    /// Rounds of threshold raising (default 4).
    pub max_rounds: usize,
    /// Area floor of the deviation currents of interpolated faces (default
    /// 0.5; 0: no deviation currents, the plain restriction): an
    /// interpolated face's current is the interpolation plus its own
    /// deviation, whose inductance is that of a face with at least this
    /// open fraction (see the
    /// [module docs](self#deviation-currents-no-spurious-static-fields)).
    pub deviation_area: f64,
}

impl Default for ConformalOptions {
    fn default() -> Self {
        Self {
            merge_below: 0.5,
            max_stability: 0.999,
            iterations: 300,
            max_rounds: 4,
            deviation_area: 0.5,
        }
    }
}

/// What [`ConformalPec::apply`] did. `merged` and `groups` count the
/// merge fallback, `interpolated` and `hosts` the interpolation (see the
/// [module docs](self#small-cut-faces-interpolation)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalReport {
    /// Cut edges scaled (including fully covered ones made PEC).
    pub edges: usize,
    /// Cut faces with their own current (`db → db/f_A`).
    pub faces: usize,
    /// Small cut faces whose current is interpolated from neighbours.
    pub interpolated: usize,
    /// Faces whose currents the interpolated faces read (each keeps its
    /// own current, updated by the operator).
    pub hosts: usize,
    /// Faces merged into groups (small faces and their merge partners).
    pub merged: usize,
    /// Merge groups (one current each).
    pub groups: usize,
    /// Small faces without an open neighbour, `f_A` raised to the
    /// threshold instead.
    pub floored: usize,
    /// Final merge threshold.
    pub merge_below: f64,
    /// Estimated stability factor `s` (NaN if not checked).
    pub stability: f64,
    /// Whether `s ≤ max_stability` (true if not checked).
    pub verified: bool,
    /// Edges cut by thin PEC sheets (split into pieces).
    pub sheet_edges: usize,
    /// Faces cut by thin PEC sheets.
    pub sheet_faces: usize,
    /// Region currents of the sheet system after merging and
    /// interpolation (one per group, host or other unmerged region).
    pub sheet_currents: usize,
    /// Small sheet regions whose current is interpolated from regions on
    /// the same side of the sheet.
    pub sheet_interpolated: usize,
    /// Sheet regions (or absorbed uncut faces) read by the interpolated
    /// ones.
    pub sheet_hosts: usize,
    /// Sheet regions merged into groups (including absorbed uncut faces).
    pub sheet_merged: usize,
    /// Small sheet regions without an open neighbour (area raised to the
    /// threshold).
    pub sheet_floored: usize,
}

impl ConformalReport {
    /// A warning text if the stability check failed.
    pub fn warning(&self) -> Option<String> {
        (!self.verified).then(|| {
            format!(
                "conformal PEC: estimated stability factor {:.4} (> 1 is unstable) with \
                 merge threshold {:.3}; reduce the time step factor",
                self.stability, self.merge_below
            )
        })
    }
}

/// Conformal PEC geometry (cut edges and faces) to be applied to an
/// engine; see the [module docs](self).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConformalPec {
    edges: Vec<ConformalEdge>,
    faces: Vec<ConformalFace>,
    /// Open-area centroids of the faces (parallel to `faces`), in units of
    /// the face sides: `[t_b, t_c]` in `[0, 1]²` with `(b, c) =
    /// axis.others()`.
    centroids: Vec<Option<[f64; 2]>>,
    /// Cells of the absorbing layer on each face (by [`Face::index`]),
    /// where small faces are merged, not interpolated.
    absorbing: [usize; 6],
    /// H-side CPML grading of the Cartesian-type layers
    /// ([`set_cpml`](Self::set_cpml)), per cell of the layer in increasing
    /// index order.
    cpml_h: Vec<(Face, usize, Vec<CpmlCoefficients>)>,
    /// The cylindrical `ρ_max` layer ([`set_cpml`](Self::set_cpml)), whose
    /// H-side stretch merge groups apply to their members in it.
    cyl_h: Option<CylLayer>,
    sheet_edges: Vec<SheetCutEdge>,
    sheet_faces: Vec<SheetCutFace>,
    options: ConformalOptions,
}

/// Face key: `(normal axis index, engine node index)`.
type Key = (usize, usize);

/// Interpolation stencil of a small face: `(key, node, weight on the
/// current)` of the faces it reads.
type Stencil = Vec<(Key, [usize; 3], f64)>;

/// A circulation term at a located field position: `(axis, (slab,
/// offset), weight)`.
type FlatTerm = (Axis, (usize, usize), f64);

impl ConformalPec {
    /// Empty geometry with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the options.
    #[must_use]
    pub fn with_options(mut self, options: ConformalOptions) -> Self {
        self.options = options;
        self
    }

    /// The options.
    pub fn options(&self) -> &ConformalOptions {
        &self.options
    }

    /// Adds a cut edge.
    pub fn push_edge(&mut self, e: ConformalEdge) {
        self.edges.push(e);
    }

    /// Adds a cut face without its open-area centroid. A small face
    /// (`f_A <` [`ConformalOptions::merge_below`]) added this way cannot be
    /// interpolated and is merged with a neighbour instead (first-order
    /// accurate, see the [module docs](self#small-cut-faces-interpolation));
    /// prefer [`push_face_with_centroid`](Self::push_face_with_centroid).
    pub fn push_face(&mut self, f: ConformalFace) {
        self.faces.push(f);
        self.centroids.push(None);
    }

    /// Adds a cut face with the centroid of its open area, `[t_b, t_c]`
    /// in units of the face sides along `(b, c) = f.axis.others()`
    /// (`[0.5, 0.5]` for an uncut face; on a cylindrical grid the centroid
    /// of the open polygon in the `(ρ, α, z)` parameters).
    pub fn push_face_with_centroid(&mut self, f: ConformalFace, centroid: [f64; 2]) {
        self.faces.push(f);
        self.centroids.push(Some(centroid));
    }

    /// Declares absorbing layers (CPML): `(face, cells)`. Small faces in
    /// them, or whose interpolation stencil would reach into them, are
    /// merged instead of interpolated (see the
    /// [module docs](self#absorbing-layers)).
    pub fn set_absorbing_layers(&mut self, layers: impl IntoIterator<Item = (Face, usize)>) {
        for (f, n) in layers {
            self.absorbing[f.index()] = self.absorbing[f.index()].max(n);
        }
    }

    /// Declares the CPML layers of `cpml` as absorbing layers
    /// ([`set_absorbing_layers`](Self::set_absorbing_layers)) and keeps
    /// their H-side grading: the faces of merge groups inside a layer get
    /// the stretched-coordinate correction of the CPML in the group update
    /// (the [`CpmlBoundary`] skips them, their kernel `db` being 0; see the
    /// [module docs](self#absorbing-layers)). Call before
    /// [`apply`](Self::apply). The cylindrical PML of a `ρ_max` face
    /// ([`CpmlBoundary::is_cylindrical_rho_max`]) is applied likewise: its
    /// stretch of the radius (and of the z or α faces' layers in the
    /// corners) filters each merged member's circulation.
    pub fn set_cpml(&mut self, cpml: &CpmlBoundary) {
        self.cyl_h = cpml.cyl_layer();
        for (face, params) in cpml.layers() {
            self.set_absorbing_layers([(face, params.cells)]);
            if cpml.is_cylindrical_rho_max() && face == Face::XMax {
                continue;
            }
            if let Some(h) = cpml.h_profile(face) {
                self.cpml_h.retain(|l| l.0 != face);
                self.cpml_h.push((face, params.cells, h.to_vec()));
            }
        }
    }

    /// Whether the face normal to `a` at `p` touches an absorbing layer
    /// (including its interface plane).
    fn in_absorbing(&self, a: Axis, p: [usize; 3], n: [usize; 3]) -> bool {
        Face::ALL.into_iter().any(|f| {
            let cells = self.absorbing[f.index()];
            if cells == 0 {
                return false;
            }
            let d = f.axis().index();
            let last = n[d] - 1;
            let normal = d == a.index();
            if f.is_max() {
                // Cells last − cells .. last.
                p[d] + usize::from(!normal) > last.saturating_sub(cells)
                    || (normal && p[d] >= last.saturating_sub(cells))
            } else {
                p[d] < cells || (normal && p[d] <= cells)
            }
        })
    }

    /// The cut edges.
    pub fn edges(&self) -> &[ConformalEdge] {
        &self.edges
    }

    /// The cut faces.
    pub fn faces(&self) -> &[ConformalFace] {
        &self.faces
    }

    /// Adds an edge cut by thin PEC sheets (see the [module docs](crate::engine::conformal#thin-pec-sheets)).
    pub fn push_sheet_edge(&mut self, e: SheetCutEdge) {
        self.sheet_edges.push(e);
    }

    /// Adds a face cut by thin PEC sheets (see the [module docs](crate::engine::conformal#thin-pec-sheets)).
    pub fn push_sheet_face(&mut self, f: SheetCutFace) {
        self.sheet_faces.push(f);
    }

    /// The edges cut by thin sheets.
    pub fn sheet_edges(&self) -> &[SheetCutEdge] {
        &self.sheet_edges
    }

    /// The faces cut by thin sheets.
    pub fn sheet_faces(&self) -> &[SheetCutFace] {
        &self.sheet_faces
    }

    /// Whether there is nothing to apply.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
            && self.faces.is_empty()
            && self.sheet_edges.is_empty()
            && self.sheet_faces.is_empty()
    }

    /// Applies the scheme to `engine`: scales the coefficients of cut edges
    /// and faces, builds the merge groups and registers their
    /// [`MergedFaces`] operator (if any), after the stability check of the
    /// [module docs](self#stability-check). Call once, before operators
    /// that read or modify coefficients are added and before the first
    /// step.
    ///
    /// # Errors
    /// [`Error::Invalid`] for fractions outside `[0, 1]` or edges / faces
    /// that do not exist on the grid.
    pub fn apply(&self, engine: &mut Engine) -> Result<ConformalReport> {
        let grid = engine.grid().clone();
        let n = grid.nodes();
        let cyl = grid.is_cylindrical();
        let copy_grid = grid.clone();
        let copies = move |a: Axis, p: [usize; 3]| {
            if cyl {
                crate::engine::cylindrical::e_edge_copies(&copy_grid, a, p)
            } else {
                vec![p]
            }
        };
        self.validate(n, &copies)?;
        let dt = engine.dt();
        let mut co = engine.coefficients().clone();
        for e in &self.edges {
            let (_, cb) = co.e_mut(e.axis);
            let k = index(n, e.node);
            cb[k] = (cb[k] as f64 * e.fraction) as f32;
        }
        let mut seeds: Vec<(usize, usize)> = Vec::new();
        for f in &self.faces {
            for (a, p) in face_edges(f.axis, f.node) {
                seeds.push((a.index(), index(n, p)));
            }
        }
        let sheet = (!self.sheet_faces.is_empty()).then(|| {
            sheets::Prepared::new(&self.sheet_edges, &self.sheet_faces, &co, n, dt, &copies)
        });
        if let Some(p) = &sheet {
            seeds.extend(p.seeds());
        }
        let volume_faces: HashSet<Key> = self
            .faces
            .iter()
            .map(|f| (f.axis.index(), index(n, f.node)))
            .collect();
        let mut tau = self.options.merge_below.clamp(0.0, 1.0);
        let rounds = self.options.max_rounds.max(1);
        let absorbing = |a: Axis, p: [usize; 3]| self.in_absorbing(a, p, n);
        for round in 0..rounds {
            let built = sheet.as_ref().map(|p| {
                p.construct(
                    &co,
                    &grid,
                    dt,
                    tau,
                    &volume_faces,
                    &absorbing,
                    &copies,
                    self.options.deviation_area.clamp(0.0, 1.0),
                )
            });
            let empty = HashSet::new();
            let owned = built.as_ref().map_or(&empty, |b| &b.owned);
            let (mut coeffs, merged, mut report) = self.construct(&co, &grid, dt, tau, owned);
            let mut gterms = group_terms(&merged.groups[..merged.scalar], n);
            let mut bterms = merged.block_terms(n);
            let mut extra = Vec::new();
            if let Some(b) = &built {
                b.apply_coefficients(&mut coeffs, n);
                let (g, bt, sq) = b.stability_terms(n);
                gterms.extend(g);
                bterms.extend(bt);
                extra = sq;
                report.sheet_edges = b.report.edges;
                report.sheet_faces = b.report.faces;
                report.sheet_currents = b.report.groups;
                report.sheet_merged = b.report.merged;
                report.sheet_floored = b.report.floored;
                report.sheet_interpolated = b.report.interpolated;
                report.sheet_hosts = b.report.hosts;
            }
            let last = round + 1 == rounds || tau >= 1.0;
            if self.options.iterations > 0 {
                let s = stability(
                    &coeffs,
                    n,
                    &gterms,
                    &bterms,
                    &extra,
                    &seeds,
                    self.options.iterations,
                );
                report.stability = s;
                report.verified = s <= self.options.max_stability;
            }
            if report.verified || last {
                *engine.coefficients_mut() = coeffs;
                // Before the merge groups, whose energy look-ahead reads
                // the corrected edge voltages.
                if let Some(b) = built {
                    engine.add_operator(b.op);
                }
                if !merged.is_empty() {
                    engine.add_operator(merged);
                }
                return Ok(report);
            }
            tau = (tau * 1.25).min(1.0);
        }
        unreachable!("the last round returns")
    }

    fn validate(&self, n: [usize; 3], copies: &sheets::Copies<'_>) -> Result<()> {
        for e in &self.edges {
            let a = e.axis.index();
            if !(0.0..=1.0).contains(&e.fraction)
                || (0..3).any(|d| e.node[d] >= n[d])
                || e.node[a] + 1 >= n[a]
            {
                return Err(Error::Invalid(format!("invalid conformal edge {e:?}")));
            }
        }
        for (f, c) in self.faces.iter().zip(&self.centroids) {
            let bad_centroid = c.is_some_and(|c| !c.iter().all(|t| (0.0..=1.0).contains(t)));
            if !(0.0..=1.0).contains(&f.fraction) || !face_exists(f.axis, f.node, n) || bad_centroid
            {
                return Err(Error::Invalid(format!("invalid conformal face {f:?}")));
            }
        }
        sheets::validate(&self.sheet_edges, &self.sheet_faces, n, copies)
    }

    /// Coefficients (edge scaling already in `co`), the operator holding
    /// the interpolated and merged faces, and counts, for merge threshold
    /// `tau`; the faces in `owned` belong to the thin-sheet system and are
    /// left alone.
    fn construct(
        &self,
        co: &Coefficients,
        grid: &Grid,
        dt: f64,
        tau: f64,
        owned: &HashSet<Key>,
    ) -> (Coefficients, MergedFaces, ConformalReport) {
        let n = grid.nodes();
        let idx = |p: [usize; 3]| index(n, p);
        let mut out = co.clone();
        let mut report = ConformalReport {
            edges: self.edges.len(),
            faces: 0,
            interpolated: 0,
            hosts: 0,
            merged: 0,
            groups: 0,
            floored: 0,
            merge_below: tau,
            stability: f64::NAN,
            verified: true,
            sheet_edges: 0,
            sheet_faces: 0,
            sheet_currents: 0,
            sheet_merged: 0,
            sheet_floored: 0,
            sheet_interpolated: 0,
            sheet_hosts: 0,
        };
        let face_info: HashMap<Key, (f64, Option<[f64; 2]>)> = self
            .faces
            .iter()
            .zip(&self.centroids)
            .map(|(f, &c)| ((f.axis.index(), idx(f.node)), (f.fraction, c)))
            .collect();
        let edge_frac: HashMap<Key, f64> = self
            .edges
            .iter()
            .map(|e| ((e.axis.index(), idx(e.node)), e.fraction))
            .collect();
        let fa_of =
            |a: Axis, p: [usize; 3]| face_info.get(&(a.index(), idx(p))).map_or(1.0, |x| x.0);
        let open = |a: Axis, p: [usize; 3]| co.cb(a)[idx(p)] > 0.0;
        let fe_of = |a: Axis, p: [usize; 3]| {
            if open(a, p) {
                *edge_frac.get(&(a.index(), idx(p))).unwrap_or(&1.0)
            } else {
                0.0
            }
        };
        let live = |a: Axis, p: [usize; 3]| {
            !owned.contains(&(a.index(), idx(p)))
                && face_edges(a, p).iter().any(|&(e, q)| open(e, q))
        };
        // Inductance and resistance of the full face, from (da, db).
        let lr_full = |a: Axis, k: usize| -> (f64, f64) {
            let (da, db) = (co.da(a)[k] as f64, co.db(a)[k] as f64);
            if db > 0.0 {
                (dt * (1.0 + da) / (2.0 * db), (1.0 - da) / db)
            } else {
                (0.0, 0.0)
            }
        };
        // Position of the (open part of the) face in its plane, in grid
        // coordinates along (b, c).
        let pos = |a: Axis, p: [usize; 3]| -> [f64; 2] {
            let t = face_info
                .get(&(a.index(), idx(p)))
                .and_then(|x| x.1)
                .unwrap_or([0.5, 0.5]);
            let (b, c) = a.others();
            let at = |d: Axis, t: f64| {
                let l = grid.lines(d);
                let i = p[d.index()];
                l[i] + t * (l[i + 1] - l[i])
            };
            [at(b, t[0]), at(c, t[1])]
        };

        let small: Vec<&ConformalFace> = self
            .faces
            .iter()
            .filter(|f| f.fraction < tau && live(f.axis, f.node))
            .collect();

        // Interpolation stencil of a small face (see `interpolation_weights`):
        // its host, the in-plane neighbour across its most open edge (as for
        // merging), and one neighbour of the host along each in-plane axis,
        // preferably on the side away from the small face; all with their
        // own current (f_A ≥ τ, not merged, outside absorbing layers) and
        // reached through open edges. Returns `(key, node, weight on the
        // current)`.
        let stencil = |f: &ConformalFace, uf: &Groups| -> Option<Stencil> {
            let a = f.axis;
            let k0 = (a.index(), idx(f.node));
            face_info.get(&k0)?.1?;

            if self.in_absorbing(a, f.node, n) {
                return None;
            }
            let free = |p: [usize; 3]| {
                let k = (a.index(), idx(p));
                live(a, p)
                    && fa_of(a, p) >= tau
                    && co.db(a)[k.1] > 0.0
                    && !uf.contains(k)
                    && !self.in_absorbing(a, p, n)
            };
            let mut host: Option<((f64, f64), [usize; 3])> = None;
            for (e, q, nb) in in_plane_neighbours(a, f.node, n) {
                let fe = fe_of(e, q);
                if fe <= 0.0 || !free(nb) {
                    continue;
                }
                let score = (fe, fa_of(a, nb));
                if host.is_none_or(|h| score > h.0) {
                    host = Some((score, nb));
                }
            }
            let host = host?.1;
            let (b, c) = a.others();
            let scale = [
                grid.delta(b, f.node[b.index()]),
                grid.delta(c, f.node[c.index()]),
            ];
            // Offsets from the small face's centroid, in its cell sides.
            let xs = pos(a, f.node);
            let rel = |p: [usize; 3]| {
                let x = pos(a, p);
                [(x[0] - xs[0]) / scale[0], (x[1] - xs[1]) / scale[1]]
            };
            let dh = rel(host);
            let mut pts = vec![(host, dh)];
            for (q, d) in [b, c].into_iter().enumerate() {
                let mut pick: Option<(bool, [usize; 3], [f64; 2])> = None;
                for (e, en, nb) in in_plane_neighbours(a, host, n) {
                    if nb[d.index()] == host[d.index()]
                        || nb == f.node
                        || fe_of(e, en) <= 0.0
                        || !free(nb)
                    {
                        continue;
                    }
                    let r = rel(nb);
                    // Away from the small face: farther from it along d.
                    let toward = r[q].abs() < dh[q].abs();
                    if pick.is_none_or(|p| p.0 && !toward) {
                        pick = Some((toward, nb, r));
                    }
                }
                match pick {
                    Some((_, nb, r)) => pts.push((nb, r)),
                    // Without a neighbour along d the stencil is exact only
                    // if the centroid has no offset along d.
                    None if dh[q].abs() > 1e-9 => return None,
                    None => {}
                }
            }
            let d: Vec<[f64; 2]> = pts.iter().map(|p| p.1).collect();
            let w = interpolation_weights(&d)?;
            let ls = centroid_dual_length(grid, a, f.node, face_info.get(&k0)?.1?);
            Some(
                pts.iter()
                    .zip(w)
                    .map(|(p, w)| {
                        let k = (a.index(), idx(p.0));
                        (k, p.0, w * ls / grid.dual_edge_length(a, p.0))
                    })
                    .collect(),
            )
        };

        // Small faces without a stencil are merged with a neighbour
        // (union-find over faces; each root keeps its member list, sorted,
        // and total open area); faces in merge groups no longer serve as
        // stencil faces, so the stencils are rebuilt until no new face
        // fails.
        let mut uf = Groups::default();
        let mut merge_seeds: Vec<&ConformalFace> = Vec::new();
        let mut stencils: Vec<(Key, Axis, [usize; 3], Stencil)>;
        loop {
            stencils = Vec::new();
            let mut failing = Vec::new();
            for f in &small {
                let k = (f.axis.index(), idx(f.node));
                if uf.contains(k) {
                    continue;
                }
                match stencil(f, &uf) {
                    Some(s) => stencils.push((k, f.axis, f.node, s)),
                    None => failing.push(*f),
                }
            }
            if failing.is_empty() {
                break;
            }
            for f in &failing {
                uf.add((f.axis.index(), idx(f.node)), f.node, f.fraction);
            }
            merge_seeds.extend(failing);
            for _pass in 0..merge_seeds.len().max(1) {
                let mut changed = false;
                for f in &merge_seeds {
                    let k0 = (f.axis.index(), idx(f.node));
                    let root = uf.find(k0);
                    // Once per group and pass: by its first member.
                    if uf.area[&root] >= tau || uf.members[&root][0] != k0 {
                        continue;
                    }
                    let mut best: Option<((f64, f64), Key, [usize; 3])> = None;
                    for m in uf.members[&root].clone() {
                        let p = uf.node[&m];
                        for (e, q, nb) in in_plane_neighbours(f.axis, p, n) {
                            let fe = fe_of(e, q);
                            if fe <= 0.0 || !live(f.axis, nb) {
                                continue;
                            }
                            let kn = (f.axis.index(), idx(nb));
                            if uf.contains(kn) && uf.find(kn) == root {
                                continue;
                            }
                            let score = (fe, fa_of(f.axis, nb));
                            if best.is_none_or(|b| score > b.0) {
                                best = Some((score, kn, nb));
                            }
                        }
                    }
                    let Some(((_, fa_nb), kn, nb)) = best else {
                        continue;
                    };
                    if !uf.contains(kn) {
                        uf.add(kn, nb, fa_nb);
                    }
                    uf.union(root, kn);
                    changed = true;
                }
                if !changed {
                    break;
                }
            }
        }

        let mut grouped: HashSet<Key> = HashSet::new();
        let mut groups = Vec::new();
        let mut driven: Vec<Driven> = Vec::new();
        let mut freeze = |out: &mut Coefficients, k: Key| {
            let a = Axis::from_index(k.0).expect("axis in 0..3");
            let (da_m, db_m) = out.h_mut(a);
            da_m[k.1] = 1.0;
            db_m[k.1] = 0.0;
            grouped.insert(k);
        };
        // With deviation currents a merge group's small members keep a
        // deviation each (soft merging, see the module docs): such groups
        // join the blocks, after the hosts.
        let deviation = self.options.deviation_area.clamp(0.0, 1.0);
        let mut soft: Vec<SoftGroup> = Vec::new();
        for (_, members) in uf.groups() {
            let (mut l, mut r) = (0.0, 0.0);
            let mut terms: HashMap<Key, (Axis, [usize; 3], f64)> = HashMap::new();
            let mut faces = Vec::with_capacity(members.len());
            for &k in &members {
                let a = Axis::from_index(k.0).expect("axis in 0..3");
                let p = uf.node[&k];
                let (lf, rf) = lr_full(a, k.1);
                let fa = fa_of(a, p);
                l += fa * lf;
                r += fa * rf;
                add_terms(&mut terms, a, p, 1.0, n);
                faces.push((a, p, fa, lf, rf));
                freeze(&mut out, k);
            }
            report.merged += members.len();
            if l <= 0.0 {
                continue;
            }
            if deviation > 0.0 {
                soft.push(SoftGroup {
                    terms: sorted_terms(terms, n),
                    l,
                    r,
                    faces,
                });
                continue;
            }
            let g = groups.len();
            groups.push(Group::new(sorted_terms(terms, n), l, r, dt));
            driven.extend(faces.into_iter().map(|(a, p, ..)| (a, p, vec![(g, 1.0)])));
        }
        report.groups = groups.len() + soft.len();
        // Interpolation: every stencil face becomes a reduced current
        // ("host") with its own circulation plus the weighted circulations
        // of the small faces reading it (`PᵀK`), and the hosts sharing small
        // faces form blocks with the consistent inductance `PᵀLP` (and
        // `PᵀRP`), see `Block`.
        struct Host {
            axis: Axis,
            node: [usize; 3],
            terms: HashMap<Key, (Axis, [usize; 3], f64)>,
        }
        let mut hosts: std::collections::BTreeMap<Key, Host> = Default::default();
        for (_, a, p, st) in &stencils {
            for &(kh, ph, w) in st {
                let h = hosts.entry(kh).or_insert_with(|| {
                    let mut terms = HashMap::new();
                    add_terms(&mut terms, *a, ph, 1.0, n);
                    Host {
                        axis: *a,
                        node: ph,
                        terms,
                    }
                });
                add_terms(&mut h.terms, *a, *p, w, n);
            }
        }
        // Host number (in key order) and the mass / loss entries.
        let host_keys: Vec<Key> = hosts.keys().copied().collect();
        let local: HashMap<Key, usize> =
            host_keys.iter().enumerate().map(|(q, &k)| (k, q)).collect();
        let mut ml: Vec<HashMap<usize, (f64, f64)>> = vec![HashMap::new(); host_keys.len()];
        for (q, &kh) in host_keys.iter().enumerate() {
            let h = &hosts[&kh];
            let (lh, rh) = lr_full(h.axis, kh.1);
            let fh = fa_of(h.axis, h.node);
            ml[q].insert(q, (fh * lh, fh * rh));
        }
        let mut link = Groups::default();
        for (q, _) in host_keys.iter().enumerate() {
            link.add((0, q), [0; 3], 0.0);
        }
        // Deviation currents (local numbers after the hosts, in stencil
        // order): `i_s = Σ w·g + d_s`, mass `max(f_A, deviation_area)·L`
        // on `d_s`, the consistent `f_A·L·w` couplings to the hosts.
        let nh = host_keys.len();
        for (ns, (k, a, p, st)) in stencils.iter().enumerate() {
            let (lf, rf) = lr_full(*a, k.1);
            // (Inductance with the current measured at the centroid, see
            // `centroid_dual_length`.)
            let c = face_info.get(k).and_then(|x| x.1).unwrap_or([0.5; 2]);
            let fa = fa_of(*a, *p) * grid.dual_edge_length(*a, *p)
                / centroid_dual_length(grid, *a, *p, c);
            for &(kj, _, wj) in st {
                let j = local[&kj];
                for &(kk, _, wk) in st {
                    let kq = local[&kk];
                    let e = ml[j].entry(kq).or_insert((0.0, 0.0));
                    e.0 += wj * wk * fa * lf;
                    e.1 += wj * wk * fa * rf;
                }
                link.union((0, local[&st[0].0]), (0, j));
            }
            if deviation > 0.0 {
                let qd = nh + ns;
                let mut row = HashMap::from([(qd, ((fa + deviation) * lf, fa * rf))]);
                for &(kj, _, wj) in st {
                    let j = local[&kj];
                    let cross = (wj * fa * lf, wj * fa * rf);
                    let e = row.entry(j).or_insert((0.0, 0.0));
                    e.0 += cross.0;
                    e.1 += cross.1;
                    let e = ml[j].entry(qd).or_insert((0.0, 0.0));
                    e.0 += cross.0;
                    e.1 += cross.1;
                }
                ml.push(row);
                link.add((0, qd), [0; 3], 0.0);
                link.union((0, local[&st[0].0]), (0, qd));
            }
        }
        let first_host = groups.len();
        for (kh, h) in hosts {
            let q = local[&kh];
            let g = first_host + q;
            let (l, r) = ml[q][&q];
            groups.push(Group::new(sorted_terms(h.terms, n), l, r, dt));
            driven.push((h.axis, h.node, vec![(g, 1.0)]));
            freeze(&mut out, kh);
        }
        // The deviations: the small faces' own circulations.
        let with_deviation = ml.len() > nh;
        if with_deviation {
            for (ns, (_, a, p, _)) in stencils.iter().enumerate() {
                let mut terms = HashMap::new();
                add_terms(&mut terms, *a, *p, 1.0, n);
                let (l, r) = ml[nh + ns][&(nh + ns)];
                groups.push(Group::new(sorted_terms(terms, n), l, r, dt));
            }
        }
        // Soft merge groups: the group current `g` (mass `Σ f_A·L`) and a
        // deviation per small member (`i_m = g + d_m`), after the hosts.
        for sg in soft {
            let qg = ml.len();
            let g = first_host + qg;
            ml.push(HashMap::from([(qg, (sg.l, sg.r))]));
            link.add((0, qg), [0; 3], 0.0);
            groups.push(Group::new(sg.terms, sg.l, sg.r, dt));
            for (a, p, fa, lf, rf) in sg.faces {
                let mut w = vec![(g, 1.0)];
                if fa < tau {
                    let qd = ml.len();
                    let cross = (fa * lf, fa * rf);
                    ml.push(HashMap::from([
                        (qd, ((fa + deviation) * lf, fa * rf)),
                        (qg, cross),
                    ]));
                    ml[qg].insert(qd, cross);
                    link.add((0, qd), [0; 3], 0.0);
                    link.union((0, qg), (0, qd));
                    let mut terms = HashMap::new();
                    add_terms(&mut terms, a, p, 1.0, n);
                    let (l, r) = ml[qd][&qd];
                    groups.push(Group::new(sorted_terms(terms, n), l, r, dt));
                    w.push((first_host + qd, 1.0));
                }
                driven.push((a, p, w));
            }
        }
        // Blocks: connected hosts (and deviations), components in order of
        // their first host.
        let mut comps: Vec<Vec<usize>> = Vec::new();
        let mut seen_root: HashMap<Key, usize> = HashMap::new();
        for q in 0..ml.len() {
            let r = link.find((0, q));
            let c = *seen_root.entry(r).or_insert_with(|| {
                comps.push(Vec::new());
                comps.len() - 1
            });
            comps[c].push(q);
        }
        let blocks: Vec<Block> = comps
            .iter()
            .map(|comp| Block::build(comp, &ml, first_host, dt))
            .collect();
        report.hosts = host_keys.len();
        report.interpolated = stencils.len();
        for (ns, (k, a, p, st)) in stencils.into_iter().enumerate() {
            let mut w: Vec<(usize, f64)> =
                st.iter().map(|s| (first_host + local[&s.0], s.2)).collect();
            if with_deviation {
                w.push((first_host + nh + ns, 1.0));
            }
            driven.push((a, p, w));
            freeze(&mut out, k);
        }

        // CPML stretching of the merged faces inside the layers, on every
        // reduced current driving the face (the cylindrical ρ_max layer
        // handles its corners with the z / α layers itself).
        let mut pml = Vec::new();
        let mut cyl_terms = Vec::new();
        for (a, p, targets) in &driven {
            let (a, p) = (*a, *p);
            if !self.in_absorbing(a, p, n) {
                continue;
            }
            let g = targets.clone();
            if let Some(st) = self.cyl_h.as_ref().and_then(|c| c.h_face(a, p)) {
                cyl_terms.push(CylTerm::new(g, a, p, st));
                continue;
            }
            for (face, cells, h) in &self.cpml_h {
                let d = face.axis();
                if d == a {
                    continue;
                }
                let (np, k) = (n[d.index()], p[d.index()]);
                let pos = if face.is_max() {
                    let inner = np - 1 - cells;
                    (k >= inner).then(|| k - inner)
                } else {
                    (k < *cells).then_some(k)
                };
                let Some(c) = pos.and_then(|q| h.get(q)) else {
                    continue;
                };
                // T_d = sign·(u_o(p + e_d) − u_o(p)), the part of the curl
                // of face a differentiating along d (CPML module docs).
                let (sign, other) = if d == a.next() {
                    (1.0, a.prev())
                } else {
                    (-1.0, a.next())
                };
                let mut hi = p;
                hi[d.index()] += 1;
                pml.push(PmlTerm {
                    targets: g.clone(),
                    other,
                    nodes: [hi, p],
                    at: None,
                    sign,
                    b: c.b,
                    a: c.a,
                    k: 1.0 / c.kappa - 1.0,
                    psi: 0.0,
                });
            }
        }

        // Remaining cut faces keep their own current with db/f_A (raised
        // to the threshold if small and neither interpolated nor merged).
        for f in &self.faces {
            let k = idx(f.node);
            if grouped.contains(&(f.axis.index(), k)) || !live(f.axis, f.node) {
                // (Faces owned by the sheet system are not live.)
                continue;
            }
            let mut fa = f.fraction;
            if fa < tau {
                fa = tau.max(1e-6);
                report.floored += 1;
            }
            if fa < 1.0 {
                let (_, db) = out.h_mut(f.axis);
                db[k] = (db[k] as f64 / fa) as f32;
            }
            report.faces += 1;
        }
        (
            out,
            MergedFaces::new(groups, first_host, blocks, driven, pml, cyl_terms),
            report,
        )
    }
}

/// A merge group with deviation currents, before it joins the blocks:
/// merged circulation, `Σ f_A·L`, `Σ f_A·R`, and its members `(axis, node,
/// f_A, L, R)`.
struct SoftGroup {
    terms: Vec<(Axis, [usize; 3], f64)>,
    l: f64,
    r: f64,
    faces: Vec<(Axis, [usize; 3], f64, f64, f64)>,
}

/// Largest accepted interpolation weight magnitude.
const MAX_WEIGHT: f64 = 2.5;

/// Weights `w_q` with `Σ w_q = 1` and `Σ w_q·d_q = 0` for the stencil
/// points `d_q` (offsets from the small face's centroid in units of its
/// cell sides): the barycentric coordinates of the centroid in the
/// triangle of three points, the linear ones on the line of two, or `[1]`
/// for a single point. Exact for affine fields. `None` for a degenerate
/// stencil, a centroid off the line or point of a shorter one, or a
/// weight above [`MAX_WEIGHT`].
fn interpolation_weights(d: &[[f64; 2]]) -> Option<Vec<f64>> {
    // Offsets below this (in cell sides) count as on the point / line.
    const ON: f64 = 1e-6;
    let w = match d {
        [p] if p[0].hypot(p[1]) < ON => vec![1.0],
        [_] => return None,
        [p, q] => {
            // p + t·(q − p) = 0 on the line through both.
            let e = [q[0] - p[0], q[1] - p[1]];
            let len2 = e[0] * e[0] + e[1] * e[1];
            if len2 < 1e-6 {
                return None;
            }
            let t = -(p[0] * e[0] + p[1] * e[1]) / len2;
            if (p[0] + t * e[0]).hypot(p[1] + t * e[1]) > ON {
                return None;
            }
            vec![1.0 - t, t]
        }
        [p, q, r] => {
            let det = (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]);
            if det.abs() < 1e-3 {
                return None;
            }
            // Barycentric coordinates of the origin.
            let wq = ((-p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (-p[1])) / det;
            let wr = ((q[0] - p[0]) * (-p[1]) - (-p[0]) * (q[1] - p[1])) / det;
            vec![1.0 - wq - wr, wq, wr]
        }
        _ => return None,
    };
    w.iter().all(|w| w.abs() <= MAX_WEIGHT).then_some(w)
}

/// Dual edge length at the open-area centroid `c` (face-side units) of
/// the face `(a, p)`: the length over which an interpolated small face's
/// current `i = H·l` is measured. It differs from
/// [`Grid::dual_edge_length`] (at the face centre) only on the α faces of
/// a cylindrical grid (`ρ_c·Δα*`, the dual edge running along α).
fn centroid_dual_length(grid: &Grid, a: Axis, p: [usize; 3], c: [f64; 2]) -> f64 {
    let l = grid.dual_edge_length(a, p);
    if !grid.is_cylindrical() || a != Axis::Y {
        return l;
    }
    // (b, c) = (z, ρ).
    let r = grid.lines(Axis::X);
    let i = p[0];
    let mid = 0.5 * (r[i] + r[i + 1]);
    if mid <= 0.0 {
        return l;
    }
    l * (r[i] + c[1] * (r[i + 1] - r[i])) / mid
}

/// Adds `w` times the curl of face `(a, p)` to a circulation.
fn add_terms(
    terms: &mut HashMap<Key, (Axis, [usize; 3], f64)>,
    a: Axis,
    p: [usize; 3],
    w: f64,
    n: [usize; 3],
) {
    for ((e, q), sign) in face_edges(a, p).into_iter().zip(SIGNS) {
        terms
            .entry((e.index(), index(n, q)))
            .and_modify(|t| t.2 += w * sign as f64)
            .or_insert((e, q, w * sign as f64));
    }
}

/// The non-zero terms of a circulation in index order.
fn sorted_terms(
    terms: HashMap<Key, (Axis, [usize; 3], f64)>,
    n: [usize; 3],
) -> Vec<(Axis, [usize; 3], f64)> {
    let mut t: Vec<(Axis, [usize; 3], f64)> =
        terms.into_values().filter(|t| t.2.abs() > 1e-12).collect();
    t.sort_by_key(|t| (t.0.index(), index(n, t.1)));
    t
}

/// Hosts coupled by the small faces they interpolate: the update
/// `A·g^{n+½} = B·g^{n−½} − dt·PᵀK·u^n` with `A = M + (dt/2)·R`,
/// `B = M − (dt/2)·R`, `M = PᵀLP`, `R = PᵀRP` (rows in the block's
/// elimination order, local indices).
#[derive(Debug, Clone, PartialEq)]
struct Block {
    /// Group indices, in elimination order.
    members: Vec<usize>,
    /// Envelope Cholesky factor of `A`.
    factor: Envelope,
    /// Factor of `M` if it differs from `A` (lossy), for the stability
    /// check.
    mfactor: Option<Envelope>,
    m: Vec<Vec<(usize, f64)>>,
    b: Vec<Vec<(usize, f64)>>,
    dt: f64,
}

/// Mass and loss entries `(PᵀLP, PᵀRP)` of coupled reduced currents, by
/// local current index: row `q` maps column `k` to `(L, R)`.
type MassRows = [HashMap<usize, (f64, f64)>];

impl Block {
    /// The block of the connected local currents `comp` of `ml` (group
    /// index = `first` + local index), in reverse Cuthill–McKee order.
    fn build(comp: &[usize], ml: &MassRows, first: usize, dt: f64) -> Self {
        let adj = |q: usize| -> Vec<usize> {
            let mut v: Vec<usize> = ml[q].keys().copied().filter(|&k| k != q).collect();
            v.sort_unstable();
            v
        };
        let order = rcm_order(comp, adj);
        let pos_of: HashMap<usize, usize> =
            order.iter().enumerate().map(|(i, &q)| (q, i)).collect();
        let rows = |f: &dyn Fn(f64, f64) -> f64| -> Vec<Vec<(usize, f64)>> {
            order
                .iter()
                .map(|&q| {
                    let mut row: Vec<(usize, f64)> = ml[q]
                        .iter()
                        .map(|(&k, &(l, r))| (pos_of[&k], f(l, r)))
                        .collect();
                    row.sort_by_key(|e| e.0);
                    row
                })
                .collect()
        };
        let m = rows(&|l, _| l);
        let a = rows(&|l, r| l + 0.5 * dt * r);
        let b = rows(&|l, r| l - 0.5 * dt * r);
        let factor = Envelope::factor(&a).expect("PᵀLP is positive definite");
        let mfactor = if a == m {
            None
        } else {
            Some(Envelope::factor(&m).expect("PᵀLP is positive definite"))
        };
        Block {
            members: order.iter().map(|&q| first + q).collect(),
            factor,
            mfactor,
            m,
            b,
            dt,
        }
    }

    /// Writes `g^{n+½}` of the members into `out` (group-indexed), from
    /// the currents `g^{n−½}` (`cur`, by group index) and the circulations
    /// `c` (group-indexed); `x` is scratch.
    fn advance(&self, cur: &dyn Fn(usize) -> f64, c: &[f64], x: &mut Vec<f64>, out: &mut [f64]) {
        x.clear();
        x.extend(self.b.iter().zip(&self.members).map(|(row, &q)| {
            row.iter()
                .map(|&(k, v)| v * cur(self.members[k]))
                .sum::<f64>()
                - self.dt * c[q]
        }));
        self.factor.solve(x);
        for (&q, &v) in self.members.iter().zip(x.iter()) {
            out[q] = flush(v as f32) as f64;
        }
    }
}

/// Lower-triangular Cholesky factor stored by rows from the first non-zero
/// column (envelope / profile storage; e.g. A. George, J. W. Liu,
/// *Computer Solution of Large Sparse Positive Definite Systems*, 1981).
#[derive(Debug, Clone, PartialEq)]
struct Envelope {
    first: Vec<usize>,
    /// Row `i`: entries for columns `first[i]..=i`.
    rows: Vec<Vec<f64>>,
}

impl Envelope {
    /// Factors the symmetric matrix given by its rows (`(column, value)`,
    /// both triangles); `None` if it is not positive definite.
    fn factor(a: &[Vec<(usize, f64)>]) -> Option<Self> {
        let n = a.len();
        let first: Vec<usize> = a
            .iter()
            .enumerate()
            .map(|(i, r)| r.iter().map(|e| e.0).min().unwrap_or(i).min(i))
            .collect();
        let mut rows: Vec<Vec<f64>> = (0..n).map(|i| vec![0.0; i - first[i] + 1]).collect();
        for (i, r) in a.iter().enumerate() {
            for &(j, v) in r {
                if j <= i {
                    rows[i][j - first[i]] += v;
                }
            }
        }
        for i in 0..n {
            for j in first[i]..=i {
                let lo = first[i].max(first[j]);
                let mut s = rows[i][j - first[i]];
                for k in lo..j {
                    s -= rows[i][k - first[i]] * rows[j][k - first[j]];
                }
                if j == i {
                    if s.is_nan() || s <= 0.0 {
                        return None;
                    }
                    rows[i][i - first[i]] = s.sqrt();
                } else {
                    rows[i][j - first[i]] = s / rows[j][j - first[j]];
                }
            }
        }
        Some(Self { first, rows })
    }

    /// Solves `L·Lᵀ·x = b` in place.
    fn solve(&self, x: &mut [f64]) {
        let n = x.len();
        for i in 0..n {
            let f = self.first[i];
            let r = &self.rows[i];
            let mut s = x[i];
            for j in f..i {
                s -= r[j - f] * x[j];
            }
            x[i] = s / r[i - f];
        }
        for i in (0..n).rev() {
            let f = self.first[i];
            let r = &self.rows[i];
            x[i] /= r[i - f];
            let xi = x[i];
            for j in f..i {
                x[j] -= r[j - f] * xi;
            }
        }
    }
}

/// Reverse Cuthill–McKee order of the nodes `comp` (one connected
/// component) with neighbours `adj` (E. Cuthill, J. McKee, 1969; reversed
/// as proposed by A. George), which keeps the envelope narrow; starts from
/// the first node of minimum degree, deterministic.
fn rcm_order(comp: &[usize], adj: impl Fn(usize) -> Vec<usize>) -> Vec<usize> {
    let deg: HashMap<usize, usize> = comp.iter().map(|&q| (q, adj(q).len())).collect();
    let mut done: HashSet<usize> = HashSet::new();
    let mut order = Vec::with_capacity(comp.len());
    let mut starts: Vec<usize> = comp.to_vec();
    starts.sort_by_key(|q| (deg[q], *q));
    for s in starts {
        if !done.insert(s) {
            continue;
        }
        let mut queue = std::collections::VecDeque::from([s]);
        while let Some(q) = queue.pop_front() {
            order.push(q);
            let mut nb: Vec<usize> = adj(q).into_iter().filter(|k| !done.contains(k)).collect();
            nb.sort_by_key(|k| (deg.get(k).copied().unwrap_or(0), *k));
            for k in nb {
                if done.insert(k) {
                    queue.push_back(k);
                }
            }
        }
    }
    order.reverse();
    order
}

/// Engine flat index of node `p`.
fn index(n: [usize; 3], p: [usize; 3]) -> usize {
    (p[0] * n[1] + p[1]) * n[2] + p[2]
}

/// Signs of the [`face_edges`] in the kernel's curl
/// (`+u_b(p) + u_c(p+e_b) − u_b(p+e_c) − u_c(p)`).
const SIGNS: [i32; 4] = [1, 1, -1, -1];

fn face_exists(axis: Axis, p: [usize; 3], n: [usize; 3]) -> bool {
    let (b, c) = axis.others();
    (0..3).all(|d| p[d] < n[d])
        && p[b.index()] + 1 < n[b.index()]
        && p[c.index()] + 1 < n[c.index()]
}

/// The four E edges `(axis, start node)` bounding the face normal to
/// `axis` with lowest corner `node`, in perimeter order: `b` at `p`, `c`
/// at `p + e_b`, `b` at `p + e_c`, `c` at `p` (`(b, c) = axis.others()`).
fn face_edges(axis: Axis, node: [usize; 3]) -> [(Axis, [usize; 3]); 4] {
    let (b, c) = axis.others();
    let mut pb = node;
    pb[b.index()] += 1;
    let mut pc = node;
    pc[c.index()] += 1;
    [(b, node), (c, pb), (b, pc), (c, node)]
}

/// In-plane neighbours of the face `(axis, p)` that exist: `(shared edge
/// axis, shared edge node, neighbour face node)`.
fn in_plane_neighbours(
    axis: Axis,
    p: [usize; 3],
    n: [usize; 3],
) -> Vec<(Axis, [usize; 3], [usize; 3])> {
    let (b, c) = axis.others();
    let mut v = Vec::with_capacity(4);
    // (shared edge axis, offset axis): the b edges border the neighbours
    // along c and vice versa.
    for (edge, off) in [(b, c), (c, b)] {
        let o = off.index();
        if p[o] > 0 {
            let mut q = p;
            q[o] -= 1;
            v.push((edge, p, q));
        }
        let mut q = p;
        q[o] += 1;
        if face_exists(axis, q, n) {
            v.push((edge, q, q));
        }
    }
    v
}

/// Union-find over face keys with member lists and areas per root.
#[derive(Default)]
struct Groups {
    parent: HashMap<Key, Key>,
    node: HashMap<Key, [usize; 3]>,
    members: HashMap<Key, Vec<Key>>,
    area: HashMap<Key, f64>,
}

impl Groups {
    fn add(&mut self, k: Key, node: [usize; 3], area: f64) {
        self.parent.insert(k, k);
        self.node.insert(k, node);
        self.members.insert(k, vec![k]);
        self.area.insert(k, area);
    }

    fn contains(&self, k: Key) -> bool {
        self.parent.contains_key(&k)
    }

    fn find(&mut self, k: Key) -> Key {
        let mut r = k;
        while self.parent[&r] != r {
            r = self.parent[&r];
        }
        let mut c = k;
        while self.parent[&c] != r {
            let next = self.parent[&c];
            self.parent.insert(c, r);
            c = next;
        }
        r
    }

    /// Joins the groups of `a` and `b`; the smaller key stays the root.
    fn union(&mut self, a: Key, b: Key) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let (keep, gone) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.parent.insert(gone, keep);
        let a = self.area.remove(&gone).unwrap_or(0.0);
        *self.area.get_mut(&keep).expect("root area") += a;
        let mv = self.members.remove(&gone).unwrap_or_default();
        let list = self.members.get_mut(&keep).expect("root members");
        list.extend(mv);
        list.sort_unstable();
    }

    /// Groups with more than one face, sorted by root.
    fn groups(&self) -> Vec<(Key, Vec<Key>)> {
        let mut g: Vec<(Key, Vec<Key>)> = self
            .members
            .iter()
            .filter(|(_, m)| m.len() > 1)
            .map(|(&r, m)| (r, m.clone()))
            .collect();
        g.sort_unstable();
        g
    }
}

/// One reduced current: a merge group or an interpolation host.
#[derive(Debug, Clone, PartialEq)]
struct Group {
    /// Circulation terms `(edge axis, edge node, weight)`.
    terms: Vec<(Axis, [usize; 3], f64)>,
    da: f64,
    db: f64,
    /// Inductance \[H\].
    l: f64,
    /// `dt/L` (lossless update factor, for the stability check).
    dt_over_l: f64,
    /// Current (after the last H update).
    i: f64,
}

impl Group {
    fn new(terms: Vec<(Axis, [usize; 3], f64)>, l: f64, r: f64, dt: f64) -> Self {
        let den = l + 0.5 * r * dt;
        Self {
            terms,
            da: (l - 0.5 * r * dt) / den,
            db: dt / den,
            l,
            dt_over_l: dt / l,
            i: 0.0,
        }
    }
}

/// A face whose current the operator writes: `Σ w·i_g` over
/// `(group, weight)`.
type Driven = (Axis, [usize; 3], Vec<(usize, f64)>);

/// The reduced currents of a [`ConformalPec`] (merge groups and
/// interpolation hosts), registered by [`ConformalPec::apply`]: updates
/// each after the H kernel, writes the faces they drive (members of a
/// group, hosts, interpolated small faces) and reports their energy; see
/// the [module docs](self#small-cut-faces-interpolation).
#[derive(Debug, Clone, PartialEq)]
pub struct MergedFaces {
    /// Merge groups (`..scalar`, own `da`, `db`) and hosts (in `blocks`).
    groups: Vec<Group>,
    scalar: usize,
    blocks: Vec<Block>,
    faces: Vec<Driven>,
    /// Circulations of the groups (scratch of `post_h`; after `post_e`,
    /// those of `u^{n+1}` for the energy look-ahead).
    circ: Vec<f64>,
    next: Vec<f64>,
    scratch: Vec<f64>,
    /// Field positions of the circulation terms (per group: axis, slab,
    /// offset, weight) and of the driven faces, filled on first use.
    flat_terms: Vec<Vec<FlatTerm>>,
    flat_faces: Vec<(Axis, (usize, usize))>,
    /// CPML corrections of merged faces in absorbing layers.
    pml: Vec<PmlTerm>,
    /// Cylindrical `ρ_max` PML corrections of merged faces.
    cyl: Vec<CylTerm>,
}

/// The cylindrical `ρ_max` PML correction of one merged face (design
/// §5.15.2): circulation `+= Y − T` with `Y = h⁻¹ ∗ T` the layer's filter
/// of the face's circulation `T = x1 − x2` (`x1 = u_c(p + e_b) − u_c(p)`,
/// `x2 = u_b(p + e_c) − u_b(p)`, `(b, c) = axis.others()`), exactly as
/// the layer itself corrects faces with a kernel `db`.
#[derive(Debug, Clone, PartialEq)]
struct CylTerm {
    /// Reduced currents driving the face, with their weights.
    targets: Vec<(usize, f64)>,
    /// Edges `[(c, p + e_b), (c, p), (b, p + e_c), (b, p)]`.
    edges: [(Axis, [usize; 3]); 4],
    /// Located `edges` (filled on first use).
    at: Option<[(usize, usize); 4]>,
    stretch: FaceStretch,
    psi: [f32; 6],
}

impl CylTerm {
    fn new(targets: Vec<(usize, f64)>, a: Axis, p: [usize; 3], stretch: FaceStretch) -> Self {
        let (b, c) = a.others();
        let (mut pb, mut pc) = (p, p);
        pb[b.index()] += 1;
        pc[c.index()] += 1;
        Self {
            targets,
            edges: [(c, pb), (c, p), (b, pc), (b, p)],
            at: None,
            stretch,
            psi: [0.0; 6],
        }
    }
}

/// The CPML correction of one merged face in a layer (H side, one
/// derivative direction): `Ψ ← b·Ψ + a·T`, circulation `+= (1/κ − 1)·T + Ψ`
/// with `T = sign·(u_other(nodes[0]) − u_other(nodes[1]))`; see
/// [`crate::engine::boundary::cpml`].
#[derive(Debug, Clone, PartialEq)]
struct PmlTerm {
    /// Reduced currents driving the face, with their weights.
    targets: Vec<(usize, f64)>,
    other: Axis,
    nodes: [[usize; 3]; 2],
    /// Located `nodes` (filled on first use).
    at: Option<[(usize, usize); 2]>,
    sign: f64,
    b: f64,
    a: f64,
    /// `1/κ − 1`.
    k: f64,
    psi: f64,
}

impl MergedFaces {
    fn new(
        groups: Vec<Group>,
        scalar: usize,
        blocks: Vec<Block>,
        faces: Vec<Driven>,
        pml: Vec<PmlTerm>,
        cyl: Vec<CylTerm>,
    ) -> Self {
        let n = groups.len();
        Self {
            pml,
            cyl,
            groups,
            scalar,
            blocks,
            faces,
            circ: vec![0.0; n],
            next: vec![0.0; n],
            scratch: Vec::new(),
            flat_terms: Vec::new(),
            flat_faces: Vec::new(),
        }
    }

    fn locate(&mut self, f: &Fields<'_>) {
        if self.flat_terms.len() == self.groups.len() {
            return;
        }
        self.flat_terms = self
            .groups
            .iter()
            .map(|g| {
                g.terms
                    .iter()
                    .map(|&(a, p, w)| (a, f.u.layout().locate(p), w))
                    .collect()
            })
            .collect();
        self.flat_faces = self
            .faces
            .iter()
            .map(|(a, p, _)| (*a, f.i.layout().locate(*p)))
            .collect();
    }

    fn circulations(&mut self, f: &Fields<'_>) {
        self.locate(f);
        for (c, t) in self.circ.iter_mut().zip(&self.flat_terms) {
            *c = t.iter().map(|&(a, at, w)| w * f.u.at(a, at) as f64).sum();
        }
    }

    /// `g^{n+½}` of every reduced current from the currents and the
    /// circulations `c`, into `out`.
    fn advance(&self, c: &[f64], x: &mut Vec<f64>, out: &mut [f64]) {
        for (q, g) in self.groups[..self.scalar].iter().enumerate() {
            out[q] = flush((g.da * g.i - g.db * c[q]) as f32) as f64;
        }
        for b in &self.blocks {
            b.advance(&|q| self.groups[q].i, c, x, out);
        }
    }

    /// Number of reduced currents.
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Whether there are no reduced currents.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

impl Operator for MergedFaces {
    fn hooks(&self) -> Hooks {
        Hooks::POST_H | Hooks::POST_E
    }

    fn post_h(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        self.circulations(f);
        for t in &mut self.pml {
            let at = *t.at.get_or_insert_with(|| {
                [
                    f.u.layout().locate(t.nodes[0]),
                    f.u.layout().locate(t.nodes[1]),
                ]
            });
            let o = t.other;
            let d = t.sign * (f.u.at(o, at[0]) as f64 - f.u.at(o, at[1]) as f64);
            t.psi = t.b * t.psi + t.a * d;
            for &(g, w) in &t.targets {
                self.circ[g] += w * (t.k * d + t.psi);
            }
        }
        for t in &mut self.cyl {
            let at = *t
                .at
                .get_or_insert_with(|| t.edges.map(|(_, p)| f.u.layout().locate(p)));
            let v = |q: usize| f.u.at(t.edges[q].0, at[q]);
            let (x1, x2) = (v(0) - v(1), v(2) - v(3));
            let c = t.stretch.correction(&mut t.psi, x1, x2) as f64;
            for &(g, w) in &t.targets {
                self.circ[g] += w * c;
            }
        }
        let (mut next, mut x) = (
            core::mem::take(&mut self.next),
            core::mem::take(&mut self.scratch),
        );
        self.advance(&self.circ, &mut x, &mut next);
        for (g, &i) in self.groups.iter_mut().zip(&next) {
            g.i = i;
        }
        (self.next, self.scratch) = (next, x);
        for ((_, _, w), &(a, at)) in self.faces.iter().zip(&self.flat_faces) {
            let i: f64 = w.iter().map(|&(g, w)| w * self.groups[g].i).sum();
            f.i.set_at(a, at, i as f32);
        }
    }

    fn post_e(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        // For the energy at t = (n + 1)·dt (`energy`).
        self.circulations(f);
    }

    /// `½·i^{n+½}ᵀ·M·i^{n+3/2}` at `t = (n + 1)·dt`, with `i^{n+3/2}` the
    /// expression the next `post_h` evaluates (from the circulations of
    /// `post_e`; solved here, on demand).
    fn energy(&self) -> f64 {
        let mut next = vec![0.0; self.groups.len()];
        self.advance(&self.circ, &mut Vec::new(), &mut next);
        let mut w = 0.0;
        for (g, n) in self.groups[..self.scalar].iter().zip(&next) {
            w += 0.5 * g.l * g.i * n;
        }
        for b in &self.blocks {
            for (row, &q) in b.m.iter().zip(&b.members) {
                let mn: f64 = row.iter().map(|&(k, v)| v * next[b.members[k]]).sum();
                w += 0.5 * self.groups[q].i * mn;
            }
        }
        w
    }
}

/// Estimated stability factor `s = √λ_max(S)/2` of the lossless leapfrog
/// scheme with coefficients `co` on a grid with `nodes` nodes (no merge
/// groups), `S = √cb·Kᵀ·db·K·√cb` (`s < 1`: stable; see the
/// [module docs](self#stability-check)).
///
/// Deterministic power iteration (`iterations` steps, `f64`, single
/// thread) from a start vector concentrated on the E edges listed in
/// `seeds` (`(axis index, engine node index)`), plus a small fixed
/// pseudo-random component on every edge; returns the final Rayleigh
/// quotient, which approaches `λ_max` from below.
pub fn stability_factor(
    co: &Coefficients,
    nodes: [usize; 3],
    seeds: &[(usize, usize)],
    iterations: usize,
) -> f64 {
    stability(co, nodes, &[], &[], &[], seeds, iterations)
}

/// A block of coupled hosts for [`apply_s`]: the circulation terms of each
/// member (as in [`GroupTerms`], local order), the factor of `M` and `dt`.
type BlockTerms<'a> = (Vec<Vec<(usize, usize, f64)>>, &'a Envelope, f64);

impl MergedFaces {
    fn block_terms(&self, n: [usize; 3]) -> Vec<BlockTerms<'_>> {
        self.blocks
            .iter()
            .map(|b| {
                let terms = b
                    .members
                    .iter()
                    .map(|&q| {
                        self.groups[q]
                            .terms
                            .iter()
                            .map(|&(a, p, s)| (a.index(), index(n, p), s))
                            .collect()
                    })
                    .collect();
                (terms, b.mfactor.as_ref().unwrap_or(&b.factor), b.dt)
            })
            .collect()
    }
}

/// A merge group for [`apply_s`]: `dt/L_g` and the circulation terms
/// `(array, index, sign)`; arrays 0..3 are the E components (engine node
/// index), array 3 the extra unknowns (thin-sheet edge pieces).
type GroupTerms = (f64, Vec<(usize, usize, f64)>);

fn group_terms(groups: &[Group], n: [usize; 3]) -> Vec<GroupTerms> {
    groups
        .iter()
        .map(|g| {
            (
                g.dt_over_l,
                g.terms
                    .iter()
                    .map(|&(a, p, s)| (a.index(), index(n, p), s))
                    .collect(),
            )
        })
        .collect()
}

/// `extra` holds `√cb` of the extra unknowns (array 3 of the group terms).
fn stability(
    co: &Coefficients,
    n: [usize; 3],
    gterms: &[GroupTerms],
    bterms: &[BlockTerms<'_>],
    extra: &[f64],
    seeds: &[(usize, usize)],
    iterations: usize,
) -> f64 {
    let len = n[0] * n[1] * n[2];
    let sq: [Vec<f64>; 4] = core::array::from_fn(|a| {
        if a < 3 {
            co.data[3 + a].iter().map(|&c| (c as f64).sqrt()).collect()
        } else {
            extra.to_vec()
        }
    });
    let db: [&[f32]; 3] = core::array::from_fn(|a| &co.data[9 + a][..]);
    let mut v: [Vec<f64>; 4] = core::array::from_fn(|a| {
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ (a as u64 + 1);
        (0..sq[a].len())
            .map(|k| {
                // xorshift: a fixed, thread-independent sequence.
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let r = (state >> 11) as f64 / (1u64 << 53) as f64;
                if sq[a][k] > 0.0 {
                    1e-3 * (r - 0.5)
                } else {
                    0.0
                }
            })
            .collect()
    });
    for &(a, k) in seeds {
        if sq[a][k] > 0.0 {
            v[a][k] += 1.0;
        }
    }
    let mut h: [Vec<f64>; 3] = core::array::from_fn(|_| vec![0.0; len]);
    let mut w: [Vec<f64>; 4] = core::array::from_fn(|a| vec![0.0; sq[a].len()]);
    let mut lambda = 0.0;
    for _ in 0..iterations.max(1) {
        let norm = v.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        for x in v.iter_mut().flatten() {
            *x /= norm;
        }
        apply_s(n, &sq, &db, (gterms, bterms), &v, &mut h, &mut w);
        lambda = v
            .iter()
            .flatten()
            .zip(w.iter().flatten())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        core::mem::swap(&mut v, &mut w);
    }
    lambda.max(0.0).sqrt() / 2.0
}

/// `w = S·v` with `S = √cb·(PK)ᵀ·(dt/L)·(PK)·√cb`, using the kernel's curl
/// convention (face `a` at `p`: `u_c(p+e_b) − u_c(p) − u_b(p+e_c) + u_b(p)`
/// with `(b, c) = a.others()`); merged faces have `db = 0` and are covered
/// by `groups` (`(dt/L_g, terms)`); `h` is scratch.
fn apply_s(
    n: [usize; 3],
    sq: &[Vec<f64>; 4],
    db: &[&[f32]; 3],
    (groups, blocks): (&[GroupTerms], &[BlockTerms<'_>]),
    v: &[Vec<f64>; 4],
    h: &mut [Vec<f64>; 3],
    w: &mut [Vec<f64>; 4],
) {
    let stride = [n[1] * n[2], n[2], 1];
    let x: [Vec<f64>; 4] =
        core::array::from_fn(|a| v[a].iter().zip(&sq[a]).map(|(v, s)| v * s).collect());
    for wa in w.iter_mut() {
        wa.fill(0.0);
    }
    for axis in Axis::ALL {
        let a = axis.index();
        let (bb, cc) = axis.others();
        let (b, c) = (bb.index(), cc.index());
        let ha = &mut h[a];
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let p = [i, j, k];
                    let q = (i * n[1] + j) * n[2] + k;
                    let d = db[a][q] as f64;
                    if d == 0.0 || p[b] + 1 >= n[b] || p[c] + 1 >= n[c] {
                        ha[q] = 0.0;
                        continue;
                    }
                    let curl = x[c][q + stride[b]] - x[c][q] - x[b][q + stride[c]] + x[b][q];
                    ha[q] = d * curl;
                }
            }
        }
        // Kᵀ: scatter each face value to its four edges.
        for q in 0..ha.len() {
            let f = ha[q];
            if f == 0.0 {
                continue;
            }
            w[c][q + stride[b]] += f;
            w[c][q] -= f;
            w[b][q + stride[c]] -= f;
            w[b][q] += f;
        }
    }
    for (d, terms) in groups {
        let curl: f64 = terms.iter().map(|&(a, k, s)| s * x[a][k]).sum();
        let f = d * curl;
        for &(a, k, s) in terms {
            w[a][k] += s * f;
        }
    }
    for (terms, m, dt) in blocks {
        let mut y: Vec<f64> = terms
            .iter()
            .map(|t| t.iter().map(|&(a, k, s)| s * x[a][k]).sum())
            .collect();
        m.solve(&mut y);
        for (t, y) in terms.iter().zip(y) {
            for &(a, k, s) in t {
                w[a][k] += s * dt * y;
            }
        }
    }
    for a in 0..4 {
        for (wv, s) in w[a].iter_mut().zip(&sq[a]) {
            *wv *= s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{CellMaterials, Grid, TimeStep};

    fn engine(s: f64) -> Engine {
        let g = Grid::uniform([12, 12, 12], [1e-3; 3]).unwrap();
        Engine::builder(g.clone())
            .materials(CellMaterials::vacuum(&g))
            .timestep(TimeStep::Courant(s))
            .threads(1)
            .build()
            .unwrap()
    }

    #[test]
    fn uncut_grid_stability_is_courant_factor() {
        let e = engine(0.9);
        let n = e.grid().nodes();
        let s = stability_factor(&e.coefficients(), n, &[], 400);
        // PEC walls: the highest mode is slightly below the infinite-grid
        // limit.
        assert!(s < 0.9 && s > 0.85, "{s}");
    }

    /// A z-normal face keeping its full x edge at y = 5 but only 2 % of
    /// its area (the y edges 2 % open, the far x edge closed).
    fn sliver() -> ConformalPec {
        let mut c = ConformalPec::new();
        c.push_face(ConformalFace {
            axis: Axis::Z,
            node: [5, 5, 5],
            fraction: 0.02,
        });
        for q in [[5, 5, 5], [6, 5, 5]] {
            c.push_edge(ConformalEdge {
                axis: Axis::Y,
                node: q,
                fraction: 0.02,
            });
        }
        c.push_edge(ConformalEdge {
            axis: Axis::X,
            node: [5, 6, 5],
            fraction: 0.0,
        });
        c
    }

    #[test]
    fn sliver_face_is_merged_and_stable() {
        // Plain Dey–Mittra (threshold 0: no merging) is unstable.
        let raw = sliver().with_options(ConformalOptions {
            merge_below: 0.0,
            max_rounds: 1,
            ..ConformalOptions::default()
        });
        let mut e = engine(0.99);
        let r = raw.apply(&mut e).unwrap();
        assert!(!r.verified && r.stability > 1.5, "{r:?}");
        assert!(r.warning().is_some());
        assert_eq!(r.merged, 0);
        let mut e = engine(0.99);
        let r = sliver().apply(&mut e).unwrap();
        assert!(r.verified, "{r:?}");
        assert_eq!((r.merged, r.groups, r.floored), (2, 1, 0), "{r:?}");
        assert!(r.stability < 0.999 && r.stability > 0.9, "{r:?}");
        // Edge scaling cb·f_e; the merged faces are left to the operator.
        let n = e.grid().nodes();
        let base = engine(0.99);
        let k = index(n, [5, 5, 5]);
        let want = base.coefficients().cb(Axis::Y)[k] as f64 * 0.02;
        assert!((e.coefficients().cb(Axis::Y)[k] as f64 / want - 1.0).abs() < 1e-6);
        assert_eq!(e.coefficients().cb(Axis::X)[index(n, [5, 6, 5])], 0.0);
        assert_eq!(e.coefficients().db(Axis::Z)[k], 0.0);
        // Merged into the face across the full x edge (y − 1).
        assert_eq!(e.coefficients().db(Axis::Z)[index(n, [5, 4, 5])], 0.0);
    }

    #[test]
    fn merged_scheme_conserves_energy() {
        let mut e = engine(0.99);
        let r = sliver().apply(&mut e).unwrap();
        assert!(r.verified);
        {
            let mut f = e.fields_mut();
            f.u.set(Axis::X, [5, 5, 5], 1.0);
            f.u.set(Axis::Y, [3, 7, 4], -0.5);
        }
        e.run(1);
        let w0 = e.energy();
        let mut dev: f64 = 0.0;
        for _ in 0..50 {
            e.run(100);
            dev = dev.max((e.energy() / w0 - 1.0).abs());
        }
        assert!(dev < 1e-5, "{dev}");
        // The in-step energy agrees with the look-ahead.
        let a = e.energy();
        let b = e.step_with_energy();
        assert_eq!(a, b);
    }

    /// [`sliver`] with the centroids of the open areas: the sliver (2 % of
    /// the face along its full x edge at y = 5) and its neighbours.
    fn sliver_with_centroids() -> ConformalPec {
        let mut c = ConformalPec::new();
        c.push_face_with_centroid(
            ConformalFace {
                axis: Axis::Z,
                node: [5, 5, 5],
                fraction: 0.02,
            },
            [0.5, 0.01],
        );
        for e in sliver().edges() {
            c.push_edge(*e);
        }
        c
    }

    #[test]
    fn sliver_face_is_interpolated_and_stable() {
        let mut e = engine(0.99);
        let r = sliver_with_centroids().apply(&mut e).unwrap();
        assert!(r.verified, "{r:?}");
        // Host across the full x edge (y − 1) and its neighbours along x
        // and y (away from the sliver).
        assert_eq!((r.interpolated, r.hosts, r.merged), (1, 3, 0), "{r:?}");
        assert!(r.stability < 0.999 && r.stability > 0.9, "{r:?}");
        let n = e.grid().nodes();
        for p in [[5, 5, 5], [5, 4, 5], [5, 3, 5]] {
            assert_eq!(e.coefficients().db(Axis::Z)[index(n, p)], 0.0, "{p:?}");
        }
    }

    #[test]
    fn interpolated_scheme_conserves_energy() {
        let mut e = engine(0.99);
        let r = sliver_with_centroids().apply(&mut e).unwrap();
        assert!(r.verified);
        {
            let mut f = e.fields_mut();
            f.u.set(Axis::X, [5, 5, 5], 1.0);
            f.u.set(Axis::Y, [4, 4, 4], -0.5);
            f.u.set(Axis::Y, [3, 7, 4], 0.25);
        }
        e.run(1);
        let w0 = e.energy();
        let mut dev: f64 = 0.0;
        for _ in 0..50 {
            e.run(100);
            dev = dev.max((e.energy() / w0 - 1.0).abs());
        }
        assert!(dev < 1e-5, "{dev}");
        let a = e.energy();
        let b = e.step_with_energy();
        assert_eq!(a, b);
    }

    #[test]
    fn weights_are_affine_exact() {
        for d in [
            vec![[0.0, 0.0]],
            vec![[0.6, 0.0], [1.6, 0.0]],
            vec![[0.6, 0.2], [1.6, 0.25], [0.55, -0.8]],
        ] {
            let w = interpolation_weights(&d).unwrap();
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "{w:?}");
            for ax in 0..2 {
                let m: f64 = w.iter().zip(&d).map(|(w, d)| w * d[ax]).sum();
                assert!(m.abs() < 1e-12, "{d:?} {w:?}");
            }
        }
        // Off the point or line, too far an extrapolation, degenerate
        // triangles.
        assert!(interpolation_weights(&[[0.7, 0.1]]).is_none());
        assert!(interpolation_weights(&[[0.6, 0.1], [1.6, 0.0]]).is_none());
        assert!(interpolation_weights(&[[5.0, 0.0], [6.0, 0.0]]).is_none());
        assert!(interpolation_weights(&[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]).is_none());
    }

    #[test]
    fn envelope_cholesky_solves() {
        // A banded SPD matrix with one long-range entry (ring closure).
        let n = 7;
        let mut a: Vec<Vec<(usize, f64)>> = (0..n).map(|i| vec![(i, 4.0 + i as f64)]).collect();
        let mut link = |i: usize, j: usize, v: f64| {
            a[i].push((j, v));
            a[j].push((i, v));
        };
        for i in 0..n - 1 {
            link(i, i + 1, -1.0);
        }
        link(0, n - 1, 0.5);
        let f = Envelope::factor(&a).unwrap();
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin() + 0.3).collect();
        let mut b: Vec<f64> = a
            .iter()
            .map(|row| row.iter().map(|&(j, v)| v * x[j]).sum())
            .collect();
        f.solve(&mut b);
        for (p, q) in b.iter().zip(&x) {
            assert!((p - q).abs() < 1e-12, "{b:?} {x:?}");
        }
        // Not positive definite.
        assert!(Envelope::factor(&[vec![(0, 1.0), (1, 2.0)], vec![(0, 2.0), (1, 1.0)]]).is_none());
    }

    #[test]
    fn rejects_bad_input() {
        let mut e = engine(0.9);
        let mut c = ConformalPec::new();
        c.push_edge(ConformalEdge {
            axis: Axis::X,
            node: [12, 0, 0],
            fraction: 0.5,
        });
        assert!(c.apply(&mut e).is_err());
        let mut c = ConformalPec::new();
        c.push_face(ConformalFace {
            axis: Axis::X,
            node: [0, 0, 0],
            fraction: 1.5,
        });
        assert!(c.apply(&mut e).is_err());
    }
}
