//! Persistent worker pool (std only).
//!
//! `threads − 1` long-lived workers plus the calling thread each own one
//! x-slab. A run of whole steps is started by publishing a [`Plan`] and
//! passing the barrier; workers then lock their own slab for writing and
//! the neighbour slab(s) for reading (see [`fields`](super::fields)) for
//! the kernel and for the operators' [slab hooks](super::SlabHooks).
//! Between the half-steps one barrier separates H from E. Where the
//! operators allow it, a step runs *fused* instead ([`Shared::fused_a`],
//! [`Shared::fused_b`]): each slab updates the H of its last x-plane,
//! then (once the previous slab has done the same, a wait on that
//! neighbour only) H and E plane by plane, with one barrier per step.
//! Only where
//! registered operators need serial hooks at a step ([`GAP_A`], from
//! [`Operator::hooks_at`](super::Operator::hooks_at)) does the caller stop
//! the workers at an extra barrier and run those hooks while they wait.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;

use super::coeffs::Coefficients;
use super::fields::{Block, FieldStore, read, write};
use super::kernel::{
    block_planes, block_planes_mut, energy_h_slab, planes_around, update_e_plane, update_e_slab,
    update_h_plane, update_h_slab,
};
use super::rows::KernelCoeffs;
use super::run::EnergyMeasure;
use super::slab::{Fusion, SlabContext, SlabHooks, SlabMut, SlabRef};

/// The slab hooks of all registered operators, in registration order.
#[derive(Clone)]
pub(crate) struct SlabOps {
    pub(crate) hooks: Vec<Arc<dyn SlabHooks>>,
    /// [`SlabHooks::fusion`] of each.
    modes: Vec<Fusion>,
    /// Whether steps can run fused with these hooks.
    fusable: bool,
}

impl SlabOps {
    /// Appends `h`.
    pub(crate) fn push(&mut self, h: Arc<dyn SlabHooks>) {
        let m = h.fusion();
        self.fusable &= match m {
            Fusion::None => false,
            // In registration order after the voltage-only hooks.
            Fusion::Planes => !self.modes.contains(&Fusion::Voltages),
            Fusion::Voltages => true,
        };
        self.modes.push(m);
        self.hooks.push(h);
    }

    /// The hooks of mode `m`, in registration order.
    fn of(&self, m: Fusion) -> impl Iterator<Item = &Arc<dyn SlabHooks>> {
        self.hooks
            .iter()
            .zip(&self.modes)
            .filter_map(move |(h, &mode)| (mode == m).then_some(h))
    }
}

impl Default for SlabOps {
    fn default() -> Self {
        Self {
            hooks: Vec::new(),
            modes: Vec::new(),
            fusable: true,
        }
    }
}

impl core::fmt::Debug for SlabOps {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "SlabOps({})", self.hooks.len())
    }
}

/// State shared between the engine and its workers.
#[derive(Debug)]
pub(crate) struct Shared {
    pub(crate) store: FieldStore,
    pub(crate) coeffs: RwLock<Coefficients>,
    /// Kernel view of `coeffs`; stale while `kernel_stale` is set.
    kernel: RwLock<KernelCoeffs>,
    /// Set when `coeffs` may have been modified since `kernel` was built.
    kernel_stale: AtomicBool,
    /// Incremented whenever `coeffs` may have been modified (caches keyed
    /// on the coefficients, e.g. the periodic merge table).
    coeff_gen: AtomicU64,
    threads: usize,
    /// Zero plane (`ny·nz`) for out-of-domain neighbours.
    pub(crate) zero: Vec<f32>,
    /// Per slab, a copy of the currents of its last x-plane, made in the
    /// first phase of a fused step for the E update of the next slab
    /// (see [`Shared::fused_a`]).
    halo: Vec<RwLock<Block>>,
    /// Per slab, the number of first phases of fused steps it has
    /// completed (the next slab waits for it instead of a barrier).
    halo_ready: Vec<Flag>,
    /// Slab hooks; written only while the workers are idle.
    pub(crate) slab_ops: RwLock<SlabOps>,
    /// Per-x-plane energy partial sums (`f64` bits) written by an
    /// energy-recording H update; each plane is written by exactly one
    /// thread and read after the run's completion barrier.
    plane_energy: Vec<AtomicU64>,
    barrier: SpinBarrier,
    cmd: AtomicU8,
    /// The published [`Plan`].
    steps: AtomicU64,
    step0: AtomicU64,
    dt_bits: AtomicU64,
    energy_first: AtomicBool,
    /// The gap bits of each step of the published plan.
    sched: RwLock<Vec<u8>>,
    /// Measure of a [`CMD_ENERGY`] phase (`true`: openEMS).
    energy_oe: AtomicBool,
    /// Cartesian grid: the openEMS measure counts the outer dual edges
    /// with full cells (see [`EnergyMeasure::OpenEms`]).
    cartesian: bool,
    /// Whether steps run fused where the operators allow it
    /// ([`FusedSteps`]; set while the workers are idle).
    fuse: AtomicBool,
}

/// When the engine runs steps fused, i.e. H and E x-plane by x-plane in
/// one pass over the fields
/// ([`Engine::set_fused_steps`](super::Engine::set_fused_steps)). Results
/// are bit-identical either way; this only affects speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FusedSteps {
    /// Where the operators allow it and the fields of a slab exceed
    /// [`FUSE_MIN_BYTES`] (the default).
    #[default]
    Auto,
    /// Where the operators allow it, whatever the grid size.
    Always,
    /// Never: every step runs its H and E updates as separate passes.
    Never,
}

impl FusedSteps {
    fn resolve(self, n: [usize; 3], slabs: usize) -> bool {
        match self {
            Self::Auto => 24 * n.iter().product::<usize>() / slabs.max(1) >= FUSE_MIN_BYTES,
            Self::Always => true,
            Self::Never => false,
        }
    }
}

/// [`FusedSteps::Auto`] fuses steps only when the fields of a slab (24
/// bytes per node) exceed this size. Below it a slab's fields stay in the core's share of
/// the caches between the half-steps, so fusing saves no memory traffic;
/// measured on the development machine (design §6), the fused schedule
/// was then up to 20 % slower at 16–32 threads (100³ and 128³ boxes),
/// while it is faster from about this size on (×1.3–1.6 at 200³–256³).
pub const FUSE_MIN_BYTES: usize = 4 << 20;

/// Where the caller stops the workers to run serial hooks, per step of a
/// [`Plan`]: bit [`GAP_A`] between the E update and the next H update
/// (`post_e`, then `pre_h` of the next step), bit [`GAP_B`] between H and
/// E (`post_h`, then `pre_e`). Steps whose operators need no serial hook
/// run without the extra barriers.
pub(crate) const GAP_A: u8 = 1;
/// See [`GAP_A`].
pub(crate) const GAP_B: u8 = 2;
/// Not a gap: the movable `post_h` and `pre_e` hooks of the step
/// ([`Operator::movable_hooks`](super::Operator::movable_hooks)) run in
/// the step's [`GAP_A`] and the one before it (both set), with no
/// [`GAP_B`].
pub(crate) const GAP_M: u8 = 4;

/// `steps` whole steps starting at engine step `step0`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Plan {
    pub(crate) steps: u64,
    pub(crate) step0: u64,
    pub(crate) dt: f64,
    /// Record the energy invariant in the first H update.
    pub(crate) energy_first: bool,
}

/// A serial hook point, passed to the caller's callback with the step
/// offset `t` within the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gap {
    /// `pre_h` of step `t`.
    PreH(u64),
    /// `post_h` and `pre_e` of step `t`.
    Mid(u64),
    /// `post_e` of step `t`.
    PostE(u64),
}

const CMD_EXIT: u8 = 0;
const CMD_RUN: u8 = 1;
/// Evaluate the energy of the current state on every slab (read-only).
const CMD_ENERGY: u8 = 2;

impl Shared {
    pub(crate) fn new(
        store: FieldStore,
        coeffs: Coefficients,
        threads: usize,
        cartesian: bool,
    ) -> Self {
        let zero = vec![0.0; store.layout.plane];
        let plane_energy = (0..store.layout.n[0]).map(|_| AtomicU64::new(0)).collect();
        let kernel = KernelCoeffs::new(&coeffs, threads);
        let (store_n, slabs) = (store.layout.n, store.layout.slabs());
        let halo = (0..store.layout.slabs())
            .map(|_| RwLock::new(core::array::from_fn(|_| vec![0.0; store.layout.plane])))
            .collect();
        Self {
            halo,
            halo_ready: (0..store.layout.slabs()).map(|_| Flag::default()).collect(),
            store,
            coeffs: RwLock::new(coeffs),
            kernel: RwLock::new(kernel),
            kernel_stale: AtomicBool::new(false),
            coeff_gen: AtomicU64::new(0),
            threads,
            zero,
            slab_ops: RwLock::new(SlabOps::default()),
            plane_energy,
            barrier: SpinBarrier::new(threads),
            cmd: AtomicU8::new(CMD_EXIT),
            steps: AtomicU64::new(0),
            step0: AtomicU64::new(0),
            dt_bits: AtomicU64::new(0),
            energy_first: AtomicBool::new(false),
            sched: RwLock::new(Vec::new()),
            energy_oe: AtomicBool::new(false),
            cartesian,
            fuse: AtomicBool::new(FusedSteps::Auto.resolve(store_n, slabs)),
        }
    }

    /// Sets the fused-step policy.
    pub(crate) fn set_fused(&self, mode: FusedSteps) {
        let on = mode.resolve(self.store.layout.n, self.store.layout.slabs());
        self.fuse.store(on, Ordering::Relaxed);
    }

    /// Marks the kernel view stale (call when handing out mutable access
    /// to `coeffs`).
    pub(crate) fn coeffs_changed(&self) {
        self.kernel_stale.store(true, Ordering::Relaxed);
        self.coeff_gen.fetch_add(1, Ordering::Relaxed);
    }

    /// The coefficient generation (see `coeff_gen`).
    pub(crate) fn coeff_gen(&self) -> u64 {
        self.coeff_gen.load(Ordering::Relaxed)
    }

    /// Rebuilds the kernel view if stale (workers idle).
    pub(crate) fn refresh_kernel(&self) {
        if self.kernel_stale.swap(false, Ordering::Relaxed) {
            let k = KernelCoeffs::new(&read(&self.coeffs), self.threads);
            *write(&self.kernel) = k;
        }
    }

    fn publish(&self, p: &Plan, sched: &[u8]) {
        self.steps.store(p.steps, Ordering::Relaxed);
        self.step0.store(p.step0, Ordering::Relaxed);
        self.dt_bits.store(p.dt.to_bits(), Ordering::Relaxed);
        self.energy_first.store(p.energy_first, Ordering::Relaxed);
        {
            let mut s = write(&self.sched);
            s.clear();
            s.extend_from_slice(sched);
        }
        self.cmd.store(CMD_RUN, Ordering::Release);
    }

    fn plan(&self) -> Plan {
        Plan {
            steps: self.steps.load(Ordering::Relaxed),
            step0: self.step0.load(Ordering::Relaxed),
            dt: f64::from_bits(self.dt_bits.load(Ordering::Relaxed)),
            energy_first: self.energy_first.load(Ordering::Relaxed),
        }
    }

    /// H half-step of slab `s` (kernel, optionally recording energy, then
    /// the slab hooks).
    fn h(&self, s: usize, energy: bool, ops: &SlabOps, step: u64, dt: f64) {
        let co = read(&self.coeffs);
        let kc = read(&self.kernel);
        let rows = kc.view(&co);
        if energy {
            energy_h_slab::<true>(
                &self.store,
                &rows,
                s,
                EnergyMeasure::Invariant,
                self.cartesian,
                &mut |gi, w| {
                    self.plane_energy[gi].store(w.to_bits(), Ordering::Relaxed);
                },
            );
        } else {
            update_h_slab(&self.store, &rows, s);
        }
        if ops.hooks.is_empty() {
            return;
        }
        let st = &self.store;
        let (p0, p1) = st.layout.planes(s);
        let mut iw = write(&st.i[s]);
        let uo = read(&st.u[s]);
        let un = st.u.get(s + 1).map(read);
        let mut tgt = SlabMut {
            n: st.layout.n,
            planes: p0..p1,
            block: &mut iw,
        };
        let src = SlabRef {
            n: st.layout.n,
            planes: p0..p1,
            block: &uo,
            neighbour: un.as_deref().map(|b| (p1, b, 0)),
        };
        let ctx = SlabContext {
            slab: s,
            step,
            dt,
            coefficients: &co,
        };
        for op in &ops.hooks {
            op.post_h_slab(&mut tgt, &src, &ctx);
        }
    }

    /// E half-step of slab `s` (kernel, then the slab hooks).
    fn e(&self, s: usize, ops: &SlabOps, step: u64, dt: f64) {
        let co = read(&self.coeffs);
        update_e_slab(&self.store, &read(&self.kernel).view(&co), &self.zero, s);
        if ops.hooks.is_empty() {
            return;
        }
        let st = &self.store;
        let (p0, p1) = st.layout.planes(s);
        let mut uw = write(&st.u[s]);
        let io = read(&st.i[s]);
        let ip = s.checked_sub(1).map(|q| read(&st.i[q]));
        let mut tgt = SlabMut {
            n: st.layout.n,
            planes: p0..p1,
            block: &mut uw,
        };
        let src = SlabRef {
            n: st.layout.n,
            planes: p0..p1,
            block: &io,
            neighbour: ip.as_deref().map(|b| {
                let last = b[0].len() / st.layout.plane - 1;
                (p0 - 1, b, last)
            }),
        };
        let ctx = SlabContext {
            slab: s,
            step,
            dt,
            coefficients: &co,
        };
        for op in &ops.hooks {
            op.post_e_slab(&mut tgt, &src, &ctx);
        }
    }

    /// First phase of a fused step on slab `s`: the H update of the
    /// slab's last x-plane, which needs the voltages of the next slab's
    /// first plane before that slab updates them. Its currents are copied
    /// to `halo[s]` for the E update of the next slab's first plane in
    /// [`fused_b`](Self::fused_b). Voltage-only slab hooks
    /// ([`Fusion::Voltages`]) run their H part first.
    fn fused_a(&self, s: usize, ops: &SlabOps, step: u64, dt: f64) {
        let co = read(&self.coeffs);
        let kc = read(&self.kernel);
        let rows = kc.view(&co);
        let st = &self.store;
        let n = st.layout.n;
        let p = st.layout.plane;
        let (p0, p1) = st.layout.planes(s);
        let (g, l) = (p1 - 1, p1 - 1 - p0);
        let mut iw = write(&st.i[s]);
        let uo = read(&st.u[s]);
        let un = st.u.get(s + 1).map(read);
        let ctx = SlabContext {
            slab: s,
            step,
            dt,
            coefficients: &co,
        };
        let src = SlabRef {
            n,
            planes: p0..p1,
            block: &uo,
            neighbour: un.as_deref().map(|b| (p1, b, 0)),
        };
        // Voltage-only H hooks: `u` is the same before and after the H
        // update.
        for op in ops.of(Fusion::Voltages) {
            let mut tgt = SlabMut {
                n,
                planes: p0..p1,
                block: &mut iw,
            };
            op.post_h_slab(&mut tgt, &src, &ctx);
        }
        let next = un.as_deref().map(|b| [&b[1][..p], &b[2][..p]]);
        update_h_plane(
            &rows,
            n,
            g,
            block_planes_mut(&mut iw, l, p),
            block_planes(&uo, l, p),
            next,
        );
        let mut tgt = SlabMut {
            n,
            planes: p0..p1,
            block: &mut iw,
        };
        for op in ops.of(Fusion::Planes) {
            op.post_h_planes(&mut tgt, &src, &ctx, g..p1);
        }
        if s + 1 < st.layout.slabs() {
            // The E kernel reads the y and z currents; slab hooks may read
            // all three.
            let first = if ops.hooks.is_empty() { 1 } else { 0 };
            let mut h = write(&self.halo[s]);
            for (d, c) in h.iter_mut().zip(block_planes(&iw, l, p)).skip(first) {
                d.copy_from_slice(c);
            }
        }
        // Release the next slab (its voltages are no longer read here).
        drop((un, uo, iw));
        self.halo_ready[s].0.fetch_add(1, Ordering::Release);
    }

    /// Second phase of a fused step on slab `s`: plane by plane, the H
    /// update of plane `g` (all but the last, done by
    /// [`fused_a`](Self::fused_a)) and then the E update of plane `g`,
    /// whose inputs (the currents of planes `g` and `g − 1`) are final by
    /// then, while the voltages of plane `g` are no longer needed by any H
    /// update. With plane-wise slab hooks each update is followed by the
    /// hooks of its plane ([`SlabHooks::post_h_planes`],
    /// [`SlabHooks::post_e_planes`]). Every value is computed by the same
    /// expression from the same inputs as in the unfused half-steps, so
    /// results are bit-identical; the fields of a plane are read from
    /// memory once per step instead of twice. Voltage-only slab hooks
    /// ([`Fusion::Voltages`]) run after the whole slab.
    fn fused_b(&self, s: usize, ops: &SlabOps, step: u64, dt: f64) {
        let co = read(&self.coeffs);
        let kc = read(&self.kernel);
        let rows = kc.view(&co);
        let st = &self.store;
        let n = st.layout.n;
        let p = st.layout.plane;
        let (p0, p1) = st.layout.planes(s);
        // The previous slab must have finished its first phase: it reads
        // our first plane of voltages and writes the halo we read.
        if s > 0 {
            let own = self.halo_ready[s].0.load(Ordering::Relaxed);
            wait_until(|| self.halo_ready[s - 1].0.load(Ordering::Acquire) >= own);
        }
        let mut iw = write(&st.i[s]);
        let mut uw = write(&st.u[s]);
        let hp = s.checked_sub(1).map(|q| read(&self.halo[q]));
        let zero = &self.zero[..p];
        let ctx = SlabContext {
            slab: s,
            step,
            dt,
            coefficients: &co,
        };
        let hooks = ops.of(Fusion::Planes).next().is_some();
        for g in p0..p1 {
            let l = g - p0;
            if g + 1 < p1 {
                let (_, ucur, unext) = planes_around(&mut uw, l, p);
                let [x, y, z] = ucur;
                let unext = unext.map(|[_, y, z]| [y, z]);
                update_h_plane(
                    &rows,
                    n,
                    g,
                    block_planes_mut(&mut iw, l, p),
                    [x, y, z],
                    unext,
                );
                if hooks {
                    let mut tgt = SlabMut {
                        n,
                        planes: p0..p1,
                        block: &mut iw,
                    };
                    let src = SlabRef {
                        n,
                        planes: p0..p1,
                        block: &uw,
                        neighbour: None,
                    };
                    for op in ops.of(Fusion::Planes) {
                        op.post_h_planes(&mut tgt, &src, &ctx, g..g + 1);
                    }
                }
            }
            let (ipl, icur, _) = planes_around(&mut iw, l, p);
            let prev = match ipl {
                Some([_, y, z]) => [y, z],
                None if g == 0 => [zero, zero],
                None => {
                    let h = hp.as_deref().expect("previous slab");
                    [&h[1][..], &h[2][..]]
                }
            };
            let [x, y, z] = icur;
            update_e_plane(
                &rows,
                n,
                g,
                block_planes_mut(&mut uw, l, p),
                [x, y, z],
                prev,
                zero,
            );
            if hooks {
                let mut tgt = SlabMut {
                    n,
                    planes: p0..p1,
                    block: &mut uw,
                };
                let src = SlabRef {
                    n,
                    planes: p0..p1,
                    block: &iw,
                    neighbour: hp.as_deref().map(|b| (p0 - 1, b, 0)),
                };
                for op in ops.of(Fusion::Planes) {
                    op.post_e_planes(&mut tgt, &src, &ctx, g..g + 1);
                }
            }
        }
        // Voltage-only E hooks: after the whole slab, as in an unfused
        // step (they come last in registration order).
        let src = SlabRef {
            n,
            planes: p0..p1,
            block: &iw,
            neighbour: hp.as_deref().map(|b| (p0 - 1, b, 0)),
        };
        for op in ops.of(Fusion::Voltages) {
            let mut tgt = SlabMut {
                n,
                planes: p0..p1,
                block: &mut uw,
            };
            op.post_e_slab(&mut tgt, &src, &ctx);
        }
    }

    /// Energy recorded by the last energy-recording H update: the
    /// per-plane sums combined in plane order (independent of the
    /// partition), times `dt/4`.
    pub(crate) fn recorded_energy(&self, dt: f64) -> f64 {
        self.plane_sum(dt, EnergyMeasure::Invariant)
    }

    /// The energy of the current state without changing it, in `measure`
    /// (calling thread only; the workers must be idle).
    pub(crate) fn current_energy(&self, dt: f64, measure: EnergyMeasure) -> f64 {
        self.refresh_kernel();
        let co = read(&self.coeffs);
        let kc = read(&self.kernel);
        let rows = kc.view(&co);
        let mut planes = vec![0.0f64; self.store.layout.n[0]];
        for s in 0..self.store.layout.slabs() {
            energy_h_slab::<false>(
                &self.store,
                &rows,
                s,
                measure,
                self.cartesian,
                &mut |gi, w| planes[gi] = w,
            );
            if measure == EnergyMeasure::OpenEms {
                self.openems_correction(s, &co, &mut |gi, w| planes[gi] += w);
            }
        }
        scale_energy(planes.iter().sum::<f64>(), dt, measure)
    }

    /// The slab hooks' corrections to openEMS's measure for slab `s`
    /// ([`SlabHooks::energy_openems_slab`]), in registration order.
    fn openems_correction(&self, s: usize, co: &Coefficients, add: &mut dyn FnMut(usize, f64)) {
        let ops = read(&self.slab_ops).clone();
        if ops.hooks.is_empty() {
            return;
        }
        let st = &self.store;
        let (p0, p1) = st.layout.planes(s);
        let uo = read(&st.u[s]);
        let un = st.u.get(s + 1).map(read);
        let src = SlabRef {
            n: st.layout.n,
            planes: p0..p1,
            block: &uo,
            neighbour: un.as_deref().map(|b| (p1, b, 0)),
        };
        for op in &ops.hooks {
            op.energy_openems_slab(s, &src, co, add);
        }
    }

    /// Records the energy share of slab `s` of the current state in
    /// `plane_energy` (read-only pass).
    fn energy_slab(&self, s: usize, measure: EnergyMeasure) {
        let co = read(&self.coeffs);
        let kc = read(&self.kernel);
        let rows = kc.view(&co);
        let (p0, p1) = self.store.layout.planes(s);
        let mut planes = vec![0.0f64; p1 - p0];
        energy_h_slab::<false>(
            &self.store,
            &rows,
            s,
            measure,
            self.cartesian,
            &mut |gi, w| {
                planes[gi - p0] = w;
            },
        );
        if measure == EnergyMeasure::OpenEms {
            self.openems_correction(s, &co, &mut |gi, w| planes[gi - p0] += w);
        }
        for (l, w) in planes.into_iter().enumerate() {
            self.plane_energy[p0 + l].store(w.to_bits(), Ordering::Relaxed);
        }
    }

    /// The per-plane sums of `plane_energy` in plane order, scaled for
    /// `measure`.
    fn plane_sum(&self, dt: f64, measure: EnergyMeasure) -> f64 {
        let s: f64 = self
            .plane_energy
            .iter()
            .map(|w| f64::from_bits(w.load(Ordering::Relaxed)))
            .sum();
        scale_energy(s, dt, measure)
    }
}

/// The kernel's plane sums are in units of `dt/4` for the invariant and
/// final for openEMS's measure.
fn scale_energy(sum: f64, dt: f64, measure: EnergyMeasure) -> f64 {
    match measure {
        EnergyMeasure::Invariant => 0.25 * dt * sum,
        EnergyMeasure::OpenEms => sum,
    }
}

/// [`Shared::current_energy`] evaluated by all threads, each on its own
/// slab (the same per-plane sums in the same order: bit-identical to the
/// serial evaluation).
pub(crate) fn energy(pool: Option<&mut Pool>, sh: &Shared, dt: f64, measure: EnergyMeasure) -> f64 {
    let Some(p) = pool else {
        return sh.current_energy(dt, measure);
    };
    sh.refresh_kernel();
    p.busy = true;
    sh.energy_oe
        .store(measure == EnergyMeasure::OpenEms, Ordering::Relaxed);
    sh.cmd.store(CMD_ENERGY, Ordering::Release);
    sh.barrier.wait();
    sh.energy_slab(0, measure);
    sh.barrier.wait();
    p.busy = false;
    sh.plane_sum(dt, measure)
}

/// A counter on its own cache lines (spun on by another thread).
#[derive(Debug, Default)]
#[repr(align(128))]
struct Flag(AtomicU64);

/// Spins, then yields, until `ready()` (like [`SpinBarrier::wait`], for
/// the short waits on a neighbour slab).
fn wait_until(ready: impl Fn() -> bool) {
    for _ in 0..SPIN {
        if ready() {
            return;
        }
        std::hint::spin_loop();
    }
    while !ready() {
        std::thread::yield_now();
    }
}

/// Sense-free generation barrier: spins briefly (cheap for the short
/// phases of small models), then yields its time slice for a while (so a
/// thread of the same run that was preempted by other load on the machine
/// can finish its slab instead of competing with spinners), then blocks on
/// a condition variable so idle workers do not burn CPU between `run`
/// calls.
#[derive(Debug)]
struct SpinBarrier {
    n: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
}

/// Busy-wait iterations before yielding.
const SPIN: u32 = 1 << 10;
/// `yield_now` iterations before sleeping.
const YIELD: u32 = 1 << 12;

impl SpinBarrier {
    fn new(n: usize) -> Self {
        Self {
            n,
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        }
    }

    fn wait(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            self.count.store(0, Ordering::Relaxed);
            let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.generation
                .store(generation.wrapping_add(1), Ordering::Release);
            self.cv.notify_all();
            return;
        }
        for _ in 0..SPIN {
            if self.generation.load(Ordering::Acquire) != generation {
                return;
            }
            std::hint::spin_loop();
        }
        for _ in 0..YIELD {
            if self.generation.load(Ordering::Acquire) != generation {
                return;
            }
            std::thread::yield_now();
        }
        let mut g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        while self.generation.load(Ordering::Acquire) == generation {
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Handle to the worker threads (absent when running single-threaded).
#[derive(Debug)]
pub(crate) struct Pool {
    workers: Vec<JoinHandle<()>>,
    /// Set while a phase is in flight; if the engine is dropped during a
    /// panic in that state the workers cannot be joined safely.
    busy: bool,
}

/// One-shot start signal: workers wait here before joining the barrier,
/// so that a failed spawn can release the ones already started.
#[derive(Debug, Default)]
struct StartGate {
    /// `None` until decided; `Some(true)` = run, `Some(false)` = exit.
    state: Mutex<Option<bool>>,
    cv: Condvar,
}

impl StartGate {
    fn open(&self, run: bool) {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = Some(run);
        self.cv.notify_all();
    }

    fn wait(&self) -> bool {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(run) = *g {
                return run;
            }
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }
}

fn worker(sh: Arc<Shared>, s: usize) {
    loop {
        sh.barrier.wait();
        match sh.cmd.load(Ordering::Acquire) {
            CMD_RUN => {}
            CMD_ENERGY => {
                let m = if sh.energy_oe.load(Ordering::Relaxed) {
                    EnergyMeasure::OpenEms
                } else {
                    EnergyMeasure::Invariant
                };
                sh.energy_slab(s, m);
                sh.barrier.wait();
                continue;
            }
            _ => return,
        }
        let p = sh.plan();
        let ops = read(&sh.slab_ops).clone();
        let sched = read(&sh.sched).clone();
        for t in 0..p.steps {
            let step = p.step0 + t;
            let g = sched[t as usize];
            if fusable(&sh, &ops, g, p.energy_first && t == 0) {
                sh.fused_a(s, &ops, step, p.dt);
                sh.fused_b(s, &ops, step, p.dt);
                sh.barrier.wait();
            } else {
                sh.h(s, p.energy_first && t == 0, &ops, step, p.dt);
                sh.barrier.wait();
                if g & GAP_B != 0 {
                    sh.barrier.wait(); // caller runs post_h / pre_e
                }
                sh.e(s, &ops, step, p.dt);
                sh.barrier.wait();
            }
            if g & GAP_A != 0 && t + 1 < p.steps {
                sh.barrier.wait(); // caller runs post_e / pre_h
            }
        }
        // The last wait was the completion barrier.
    }
}

/// Whether a step with gap bits `g` runs fused ([`Shared::fused_a`],
/// [`Shared::fused_b`]): no serial hooks between its H and E updates, no
/// energy recording, and slab hooks that allow it ([`SlabHooks::fusion`]).
fn fusable(sh: &Shared, ops: &SlabOps, g: u8, energy: bool) -> bool {
    sh.fuse.load(Ordering::Relaxed) && g & GAP_B == 0 && !energy && ops.fusable
}

/// Runs `plan` on all slabs: slab 0 on the calling thread, the others on
/// the workers of `pool` (none when single-threaded). `hook` is called on
/// the calling thread at every serial hook point of `sched` (the gap bits
/// of each step, see [`GAP_A`]; `pre_h0`: the `pre_h` hooks before the
/// first step), with the workers parked (one extra barrier per gap and
/// step).
pub(crate) fn drive(
    mut pool: Option<&mut Pool>,
    sh: &Shared,
    plan: &Plan,
    sched: &[u8],
    pre_h0: bool,
    hook: &mut dyn FnMut(Gap),
) {
    if plan.steps == 0 {
        return;
    }
    assert_eq!(sched.len() as u64, plan.steps, "one gap entry per step");
    sh.refresh_kernel();
    let threaded = pool.is_some();
    let wait = || {
        if threaded {
            sh.barrier.wait();
        }
    };
    if pre_h0 {
        hook(Gap::PreH(0));
    }
    let ops = read(&sh.slab_ops).clone();
    if let Some(p) = pool.as_deref_mut() {
        p.busy = true;
        sh.publish(plan, sched);
        sh.barrier.wait();
    }
    for t in 0..plan.steps {
        let step = plan.step0 + t;
        let g = sched[t as usize];
        if fusable(sh, &ops, g, plan.energy_first && t == 0) {
            sh.fused_a(0, &ops, step, plan.dt);
            sh.fused_b(0, &ops, step, plan.dt);
            wait();
        } else {
            sh.h(0, plan.energy_first && t == 0, &ops, step, plan.dt);
            wait();
            if g & GAP_B != 0 {
                hook(Gap::Mid(t));
                wait();
            }
            sh.e(0, &ops, step, plan.dt);
            wait();
        }
        if g & GAP_A != 0 {
            hook(Gap::PostE(t));
            if t + 1 < plan.steps {
                hook(Gap::PreH(t + 1));
                wait();
            }
        }
    }
    if let Some(p) = pool {
        p.busy = false;
    }
}

impl Pool {
    /// Spawns one worker per slab except slab 0, which the caller runs.
    pub(crate) fn spawn(sh: &Arc<Shared>) -> std::io::Result<Self> {
        let slabs = sh.store.layout.slabs();
        let gate = Arc::new(StartGate::default());
        let mut workers = Vec::with_capacity(slabs.saturating_sub(1));
        for s in 1..slabs {
            let sh = Arc::clone(sh);
            let g = Arc::clone(&gate);
            let h = std::thread::Builder::new()
                .name(format!("ems-worker-{s}"))
                .spawn(move || {
                    if g.wait() {
                        worker(sh, s);
                    }
                });
            match h {
                Ok(h) => workers.push(h),
                Err(e) => {
                    // The barrier can no longer be completed: tell the
                    // workers already started to exit before entering it,
                    // and join them.
                    gate.open(false);
                    for w in workers {
                        let _ = w.join();
                    }
                    return Err(e);
                }
            }
        }
        gate.open(true);
        Ok(Self {
            workers,
            busy: false,
        })
    }

    pub(crate) fn shutdown(&mut self, sh: &Shared) {
        if self.busy || self.workers.is_empty() {
            self.workers.clear(); // detach
            return;
        }
        sh.cmd.store(CMD_EXIT, Ordering::Release);
        sh.barrier.wait();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}
