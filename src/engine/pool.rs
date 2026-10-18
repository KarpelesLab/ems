//! Persistent worker pool (std only).
//!
//! `threads − 1` long-lived workers plus the calling thread each own one
//! x-slab. A phase (H half-step, E half-step, or a run of whole steps with
//! no operators) is started by publishing a command and passing the
//! barrier; workers then lock their own slab for writing and the neighbour
//! slab(s) for reading (see [`fields`](super::fields)). Operators run on the
//! calling thread while the workers wait at the barrier.

use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;

use super::coeffs::Coefficients;
use super::fields::{FieldStore, read};
use super::kernel::{update_e_slab, update_h_slab};

/// State shared between the engine and its workers.
#[derive(Debug)]
pub(crate) struct Shared {
    pub(crate) store: FieldStore,
    pub(crate) coeffs: RwLock<Coefficients>,
    /// Zero plane (`ny·nz`) for out-of-domain neighbours.
    pub(crate) zero: Vec<f32>,
    barrier: SpinBarrier,
    cmd: AtomicU8,
    steps: AtomicU64,
}

const CMD_EXIT: u8 = 0;
const CMD_H: u8 = 1;
const CMD_E: u8 = 2;
const CMD_RUN: u8 = 3;

impl Shared {
    pub(crate) fn new(store: FieldStore, coeffs: Coefficients, threads: usize) -> Self {
        let zero = vec![0.0; store.layout.plane];
        Self {
            store,
            coeffs: RwLock::new(coeffs),
            zero,
            barrier: SpinBarrier::new(threads),
            cmd: AtomicU8::new(CMD_EXIT),
            steps: AtomicU64::new(0),
        }
    }

    pub(crate) fn h(&self, s: usize) {
        update_h_slab(&self.store, &read(&self.coeffs), s);
    }

    pub(crate) fn e(&self, s: usize) {
        update_e_slab(&self.store, &read(&self.coeffs), &self.zero, s);
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
        match sh.cmd.load(Ordering::Acquire) {
            CMD_H => sh.h(s),
            CMD_E => sh.e(s),
            CMD_RUN => {
                let n = sh.steps.load(Ordering::Acquire);
                for _ in 0..n {
                    sh.h(s);
                    sh.barrier.wait();
                    sh.e(s);
                    sh.barrier.wait();
                }
                continue; // the last wait was the completion barrier
            }
            _ => return,
        }
        sh.barrier.wait();
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

    /// One H or E half-step over all slabs.
    pub(crate) fn half(&mut self, sh: &Shared, e: bool) {
        self.busy = true;
        sh.cmd
            .store(if e { CMD_E } else { CMD_H }, Ordering::Release);
        sh.barrier.wait();
        if e {
            sh.e(0)
        } else {
            sh.h(0)
        }
        sh.barrier.wait();
        self.busy = false;
    }

    /// `n` whole steps with no operator hooks in between (one barrier per
    /// half-step).
    pub(crate) fn run(&mut self, sh: &Shared, n: u64) {
        if n == 0 {
            return;
        }
        self.busy = true;
        sh.steps.store(n, Ordering::Release);
        sh.cmd.store(CMD_RUN, Ordering::Release);
        sh.barrier.wait();
        for _ in 0..n {
            sh.h(0);
            sh.barrier.wait();
            sh.e(0);
            sh.barrier.wait();
        }
        self.busy = false;
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
