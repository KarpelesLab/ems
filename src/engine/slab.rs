//! Slab-parallel operator work (see [`SlabHooks`]).

use core::ops::Range;

use super::coeffs::Coefficients;
use super::fields::Block;
use super::grid::Grid;
use crate::axis::Axis;

/// The x-slab partition of an engine, passed to
/// [`Operator::slab_hooks`](super::Operator::slab_hooks) at registration.
#[derive(Debug, Clone, Copy)]
pub struct SlabPartition<'a> {
    pub(crate) grid: &'a Grid,
    pub(crate) dt: f64,
    /// First x-plane of each slab, plus `nx`.
    pub(crate) starts: &'a [usize],
}

impl SlabPartition<'_> {
    /// The mesh.
    pub fn grid(&self) -> &Grid {
        self.grid
    }

    /// Time step \[s\].
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Number of slabs (threads).
    pub fn slabs(&self) -> usize {
        self.starts.len() - 1
    }

    /// The x-planes of slab `s`.
    pub fn planes(&self, s: usize) -> Range<usize> {
        self.starts[s]..self.starts[s + 1]
    }
}

/// Information passed to the [`SlabHooks`] half-step calls.
#[derive(Debug, Clone, Copy)]
pub struct SlabContext<'a> {
    /// Index of the slab (in `0..`[`SlabPartition::slabs`]).
    pub slab: usize,
    /// Index `n` of the step in progress (as
    /// [`StepContext::step`](super::StepContext::step)).
    pub step: u64,
    /// Time step \[s\].
    pub dt: f64,
    /// The update coefficients.
    pub coefficients: &'a Coefficients,
}

/// Write access to one field type (all three components) in the x-planes
/// of one slab.
#[derive(Debug)]
pub struct SlabMut<'a> {
    pub(crate) n: [usize; 3],
    pub(crate) planes: Range<usize>,
    pub(crate) block: &'a mut Block,
}

impl SlabMut<'_> {
    /// Node counts `[nx, ny, nz]` of the whole grid.
    pub fn nodes(&self) -> [usize; 3] {
        self.n
    }

    /// The global x-planes of this slab.
    pub fn planes(&self) -> Range<usize> {
        self.planes.clone()
    }

    fn offset(&self, i: usize) -> usize {
        assert!(
            self.planes.contains(&i),
            "x-plane {i} is not in slab {:?}",
            self.planes
        );
        (i - self.planes.start) * self.n[1] * self.n[2]
    }

    /// The x-plane `i` of component `axis` (`ny·nz` values, index
    /// `j·nz + k`).
    ///
    /// # Panics
    /// If `i` is not one of this slab's planes.
    pub fn plane(&self, axis: Axis, i: usize) -> &[f32] {
        let o = self.offset(i);
        &self.block[axis.index()][o..o + self.n[1] * self.n[2]]
    }

    /// Mutable x-plane `i` of component `axis`.
    ///
    /// # Panics
    /// If `i` is not one of this slab's planes.
    pub fn plane_mut(&mut self, axis: Axis, i: usize) -> &mut [f32] {
        let o = self.offset(i);
        let p = self.n[1] * self.n[2];
        &mut self.block[axis.index()][o..o + p]
    }
}

/// Read access to one field type in the x-planes of one slab plus one
/// neighbour plane: the first plane of the next slab during the H
/// half-step (for `u`), the last plane of the previous slab during the E
/// half-step (for `i`).
#[derive(Debug)]
pub struct SlabRef<'a> {
    pub(crate) n: [usize; 3],
    pub(crate) planes: Range<usize>,
    pub(crate) block: &'a Block,
    /// Global index and block-local plane of the neighbour plane.
    pub(crate) neighbour: Option<(usize, &'a Block, usize)>,
}

impl SlabRef<'_> {
    /// Node counts `[nx, ny, nz]` of the whole grid.
    pub fn nodes(&self) -> [usize; 3] {
        self.n
    }

    /// The global x-planes of this slab (not including the neighbour
    /// plane).
    pub fn planes(&self) -> Range<usize> {
        self.planes.clone()
    }

    /// The x-plane `i` of component `axis`, if it is one of this slab's
    /// planes or the neighbour plane; `None` otherwise (including planes
    /// outside the grid).
    pub fn try_plane(&self, axis: Axis, i: usize) -> Option<&[f32]> {
        let p = self.n[1] * self.n[2];
        let (b, l) = if self.planes.contains(&i) {
            (self.block, i - self.planes.start)
        } else {
            match self.neighbour {
                Some((g, b, l)) if g == i => (b, l),
                _ => return None,
            }
        };
        Some(&b[axis.index()][l * p..(l + 1) * p])
    }

    /// Like [`try_plane`](Self::try_plane).
    ///
    /// # Panics
    /// If the plane is not available.
    pub fn plane(&self, axis: Axis, i: usize) -> &[f32] {
        self.try_plane(axis, i)
            .unwrap_or_else(|| panic!("x-plane {i} is not readable from slab {:?}", self.planes))
    }
}

/// How the [`SlabHooks`] of an operator can run in a fused step
/// ([`SlabHooks::fusion`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fusion {
    /// Not at all: steps run unfused while the operator is registered.
    None,
    /// Plane by plane, through [`SlabHooks::post_h_planes`] and
    /// [`SlabHooks::post_e_planes`].
    Planes,
    /// Whole slabs, for hooks that act on the voltages only:
    /// [`post_h_slab`](SlabHooks::post_h_slab) reads `u` and nothing else
    /// (no access to `i`), so the engine may call it before the slab's H
    /// update, which leaves `u` unchanged; and
    /// [`post_e_slab`](SlabHooks::post_e_slab) reads and writes `u` only,
    /// and is called after the E update of the whole slab and the
    /// plane-wise hooks of all operators. Steps are fused only if every
    /// such operator is registered after all [`Planes`](Self::Planes)
    /// ones, so the hooks still run in registration order where they
    /// interact.
    Voltages,
}

/// The slab-parallel part of an [`Operator`](super::Operator): bulk
/// corrections (CPML) run by the worker pool inside the half-step, right
/// after the kernel on each x-slab.
///
/// An operator that touches whole regions of the grid can hand the engine
/// a `SlabHooks` object from
/// [`Operator::slab_hooks`](super::Operator::slab_hooks). The engine calls
/// it once per x-slab and half-step, on the thread that owns the slab,
/// after that slab's kernel update and before the serial `post_h` /
/// `post_e` hooks of any operator. The call sees exactly what the kernel of
/// that slab sees: write access to the updated field type in the slab's own
/// x-planes ([`SlabMut`]), read access to the other field type in its own
/// planes plus one neighbour plane ([`SlabRef`]: the next plane for H, the
/// previous one for E). No barrier is added.
///
/// **Determinism.** The slab partition must not change the arithmetic:
/// work for a node must be computed by the same expression whichever slab
/// it falls in, and state (e.g. `ψ` arrays) is kept per slab
/// ([`SlabPartition`] is passed at registration for that purpose). Then
/// results are bit-identical for any thread count.
///
/// Methods take `&self` because the slabs run concurrently: keep any
/// mutable state per slab, e.g. one `Mutex` per slab (each is only ever
/// locked by the thread of its slab, so it is uncontended). Both methods
/// default to doing nothing.
pub trait SlabHooks: Send + Sync + 'static {
    /// After the H kernel on slab `ctx.slab`: `i` (written, own planes)
    /// is at `(n + ½)·dt`, `u` (read, own planes plus the first plane of the
    /// next slab) at `n·dt`.
    fn post_h_slab(&self, _i: &mut SlabMut<'_>, _u: &SlabRef<'_>, _ctx: &SlabContext<'_>) {}

    /// After the E kernel on slab `ctx.slab`: `u` (written, own planes) is
    /// at `(n + 1)·dt`, `i` (read, own planes plus the last plane of the
    /// previous slab) at `(n + ½)·dt`.
    fn post_e_slab(&self, _u: &mut SlabMut<'_>, _i: &SlabRef<'_>, _ctx: &SlabContext<'_>) {}

    /// How the hooks can run in a fused step (default [`Fusion::None`]).
    ///
    /// When the hooks of every registered operator allow it (see
    /// [`Fusion`]), the engine runs the steps that have no serial hook
    /// between their H and E updates *fused*: each worker updates H and
    /// then E x-plane by x-plane, so the fields of a plane are streamed
    /// from memory once per step instead of twice. Queried once, at
    /// registration.
    fn fusion(&self) -> Fusion {
        Fusion::None
    }

    /// The part of [`post_h_slab`](Self::post_h_slab) that belongs to the
    /// dual edges in the x-planes `planes` (a sub-range of the slab's
    /// planes), with the same arguments; called instead of `post_h_slab`
    /// in fused steps if [`fusion`](Self::fusion) is [`Fusion::Planes`].
    ///
    /// Within one half-step the engine calls it once for each plane of the
    /// slab, in no particular order, so the work for a plane must not
    /// depend on the other planes' calls. It may write `i` in `planes`
    /// only, and read `i` in `planes` and `u` in `planes` and the plane
    /// after them: those hold their values as in `post_h_slab`. Other
    /// planes of `u` may already hold `u^{n+1}` (fused steps).
    fn post_h_planes(
        &self,
        _i: &mut SlabMut<'_>,
        _u: &SlabRef<'_>,
        _ctx: &SlabContext<'_>,
        _planes: Range<usize>,
    ) {
    }

    /// The part of [`post_e_slab`](Self::post_e_slab) that belongs to the
    /// E edges in the x-planes `planes`; called instead of `post_e_slab`
    /// in fused steps if [`fusion`](Self::fusion) is [`Fusion::Planes`]. As
    /// [`post_h_planes`](Self::post_h_planes): it may write `u` in
    /// `planes` only, and read `u` in `planes` and `i` in `planes` and the
    /// plane before them. Other planes of `u` may still hold `u^n` and
    /// other planes of `i` may still hold `i^{n−½}` (fused steps).
    fn post_e_planes(
        &self,
        _u: &mut SlabMut<'_>,
        _i: &SlabRef<'_>,
        _ctx: &SlabContext<'_>,
        _planes: Range<usize>,
    ) {
    }

    /// Correction \[J/m\] to openEMS's energy measure
    /// ([`Engine::energy_openems`](super::Engine::energy_openems)) for the
    /// dual edges of slab `slab` whose `i^{n+½}` the kernel's read-only H
    /// look-ahead does not produce, because [`post_h_slab`](Self::post_h_slab)
    /// completes their update: call `add(gi, w)` with the correction
    /// `μ0·Σ (i_true² − i_kernel²)` of each x-plane `gi` (summed in a fixed
    /// order, so the measure stays independent of the partition). `u` is
    /// the state after `n` steps (own planes plus the first plane of the
    /// next slab). Called on the thread of the slab, with the workers
    /// evaluating the energy.
    fn energy_openems_slab(
        &self,
        _slab: usize,
        _u: &SlabRef<'_>,
        _coefficients: &Coefficients,
        _add: &mut dyn FnMut(usize, f64),
    ) {
    }
}
