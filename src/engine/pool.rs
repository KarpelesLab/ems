//! Persistent worker pool (std only).
//!
//! `threads − 1` long-lived workers plus the calling thread each own one
//! x-slab. A run of whole steps is started by publishing a [`Plan`] and
//! passing the barrier; workers then lock their own slab for writing and
//! the neighbour slab(s) for reading (see [`fields`](super::fields)) for
//! the kernel and for the operators' [slab hooks](super::SlabHooks).
//! Between the half-steps one barrier separates H from E. Only where
//! registered operators have serial hooks ([`Gaps`]) does the caller stop
//! the workers at an extra barrier and run those hooks while they wait.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;

use super::coeffs::Coefficients;
use super::fields::{FieldStore, read, write};
use super::kernel::{energy_h_slab, update_e_slab, update_h_slab};
use super::slab::{SlabContext, SlabHooks, SlabMut, SlabRef};

/// The slab hooks of all registered operators, in registration order.
#[derive(Default, Clone)]
pub(crate) struct SlabOps(pub(crate) Vec<Arc<dyn SlabHooks>>);

impl core::fmt::Debug for SlabOps {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "SlabOps({})", self.0.len())
    }
}

/// State shared between the engine and its workers.
#[derive(Debug)]
pub(crate) struct Shared {
    pub(crate) store: FieldStore,
    pub(crate) coeffs: RwLock<Coefficients>,
    /// Zero plane (`ny·nz`) for out-of-domain neighbours.
    pub(crate) zero: Vec<f32>,
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
    gap_a: AtomicBool,
    gap_b: AtomicBool,
}

/// Where the caller must stop the workers to run serial hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Gaps {
    /// Between E and the next H (`post_e`, then `pre_h`).
    pub(crate) a: bool,
    /// Between H and E (`post_h`, then `pre_e`).
    pub(crate) b: bool,
}

/// `steps` whole steps starting at engine step `step0`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Plan {
    pub(crate) steps: u64,
    pub(crate) step0: u64,
    pub(crate) dt: f64,
    /// Record the energy invariant in the first H update.
    pub(crate) energy_first: bool,
    pub(crate) gaps: Gaps,
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

impl Shared {
    pub(crate) fn new(store: FieldStore, coeffs: Coefficients, threads: usize) -> Self {
        let zero = vec![0.0; store.layout.plane];
        let plane_energy = (0..store.layout.n[0]).map(|_| AtomicU64::new(0)).collect();
        Self {
            store,
            coeffs: RwLock::new(coeffs),
            zero,
            slab_ops: RwLock::new(SlabOps::default()),
            plane_energy,
            barrier: SpinBarrier::new(threads),
            cmd: AtomicU8::new(CMD_EXIT),
            steps: AtomicU64::new(0),
            step0: AtomicU64::new(0),
            dt_bits: AtomicU64::new(0),
            energy_first: AtomicBool::new(false),
            gap_a: AtomicBool::new(false),
            gap_b: AtomicBool::new(false),
        }
    }

    fn publish(&self, p: &Plan) {
        self.steps.store(p.steps, Ordering::Relaxed);
        self.step0.store(p.step0, Ordering::Relaxed);
        self.dt_bits.store(p.dt.to_bits(), Ordering::Relaxed);
        self.energy_first.store(p.energy_first, Ordering::Relaxed);
        self.gap_a.store(p.gaps.a, Ordering::Relaxed);
        self.gap_b.store(p.gaps.b, Ordering::Relaxed);
        self.cmd.store(CMD_RUN, Ordering::Release);
    }

    fn plan(&self) -> Plan {
        Plan {
            steps: self.steps.load(Ordering::Relaxed),
            step0: self.step0.load(Ordering::Relaxed),
            dt: f64::from_bits(self.dt_bits.load(Ordering::Relaxed)),
            energy_first: self.energy_first.load(Ordering::Relaxed),
            gaps: Gaps {
                a: self.gap_a.load(Ordering::Relaxed),
                b: self.gap_b.load(Ordering::Relaxed),
            },
        }
    }

    /// H half-step of slab `s` (kernel, optionally recording energy, then
    /// the slab hooks).
    fn h(&self, s: usize, energy: bool, ops: &SlabOps, step: u64, dt: f64) {
        let co = read(&self.coeffs);
        if energy {
            energy_h_slab::<true>(&self.store, &co, s, &mut |gi, w| {
                self.plane_energy[gi].store(w.to_bits(), Ordering::Relaxed);
            });
        } else {
            update_h_slab(&self.store, &co, s);
        }
        if ops.0.is_empty() {
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
        for op in &ops.0 {
            op.post_h_slab(&mut tgt, &src, &ctx);
        }
    }

    /// E half-step of slab `s` (kernel, then the slab hooks).
    fn e(&self, s: usize, ops: &SlabOps, step: u64, dt: f64) {
        let co = read(&self.coeffs);
        update_e_slab(&self.store, &co, &self.zero, s);
        if ops.0.is_empty() {
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
        for op in &ops.0 {
            op.post_e_slab(&mut tgt, &src, &ctx);
        }
    }

    /// Energy recorded by the last energy-recording H update: the
    /// per-plane sums combined in plane order (independent of the
    /// partition), times `dt/4`.
    pub(crate) fn recorded_energy(&self, dt: f64) -> f64 {
        let s: f64 = self
            .plane_energy
            .iter()
            .map(|w| f64::from_bits(w.load(Ordering::Relaxed)))
            .sum();
        0.25 * dt * s
    }

    /// The energy invariant of the current state without changing it
    /// (calling thread only; the workers must be idle).
    pub(crate) fn current_energy(&self, dt: f64) -> f64 {
        let co = read(&self.coeffs);
        let mut planes = vec![0.0f64; self.store.layout.n[0]];
        for s in 0..self.store.layout.slabs() {
            energy_h_slab::<false>(&self.store, &co, s, &mut |gi, w| planes[gi] = w);
        }
        0.25 * dt * planes.iter().sum::<f64>()
    }
}

/// Sense-free generation barrier: spins briefly (cheap for the short
/// phases of small models), then blocks on a condition variable so idle
/// workers do not burn CPU between `run` calls.
#[derive(Debug)]
struct SpinBarrier {
    n: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
}

const SPIN: u32 = 1 << 14;

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
        if sh.cmd.load(Ordering::Acquire) != CMD_RUN {
            return;
        }
        let p = sh.plan();
        let ops = read(&sh.slab_ops).clone();
        for t in 0..p.steps {
            let step = p.step0 + t;
            sh.h(s, p.energy_first && t == 0, &ops, step, p.dt);
            sh.barrier.wait();
            if p.gaps.b {
                sh.barrier.wait(); // caller runs post_h / pre_e
            }
            sh.e(s, &ops, step, p.dt);
            sh.barrier.wait();
            if p.gaps.a && t + 1 < p.steps {
                sh.barrier.wait(); // caller runs post_e / pre_h
            }
        }
        // The last wait was the completion barrier.
    }
}

/// Runs `plan` on all slabs: slab 0 on the calling thread, the others on
/// the workers of `pool` (none when single-threaded). `hook` is called on
/// the calling thread at every serial hook point of `plan.gaps`, with the
/// workers parked (one extra barrier per gap and step).
pub(crate) fn drive(
    mut pool: Option<&mut Pool>,
    sh: &Shared,
    plan: &Plan,
    hook: &mut dyn FnMut(Gap),
) {
    if plan.steps == 0 {
        return;
    }
    let threaded = pool.is_some();
    let wait = || {
        if threaded {
            sh.barrier.wait();
        }
    };
    if plan.gaps.a {
        hook(Gap::PreH(0));
    }
    let ops = read(&sh.slab_ops).clone();
    if let Some(p) = pool.as_deref_mut() {
        p.busy = true;
        sh.publish(plan);
        sh.barrier.wait();
    }
    for t in 0..plan.steps {
        let step = plan.step0 + t;
        sh.h(0, plan.energy_first && t == 0, &ops, step, plan.dt);
        wait();
        if plan.gaps.b {
            hook(Gap::Mid(t));
            wait();
        }
        sh.e(0, &ops, step, plan.dt);
        wait();
        if plan.gaps.a {
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
