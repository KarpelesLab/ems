//! Fused steps (design §6): the engine updates H and E x-plane by x-plane
//! in one pass where the operators allow it, and moves read-only serial
//! hooks (current probes, lumped-element voltage capture) out of the gap
//! between the half-steps. Both must leave results unchanged bit for bit.
//!
//! Each fixture model runs twice with the same thread count: with fused
//! steps forced on ([`FusedSteps::Always`]; the models are too small for
//! the default to fuse them), and with fused steps off plus an extra
//! operator that keeps the serial hooks in their original gaps (a serial
//! `post_h` needed at every step, which no hook can be moved around, and
//! slab hooks that cannot run fused). The complete final field states
//! must be identical.
#![cfg(feature = "xml")]

use std::path::Path;
use std::sync::Arc;

use ems::compile::CompileOptions;
use ems::engine::{Fields, FusedSteps, Hooks, Operator, SlabHooks, SlabPartition, StepContext};
use ems::io::openems::read_model;
use ems::{Axis, Simulation};

/// Forces unfused steps with the serial hooks in their original gaps.
struct Unfuse;

struct NoFusion;

impl SlabHooks for NoFusion {}

impl Operator for Unfuse {
    fn hooks(&self) -> Hooks {
        Hooks::POST_H
    }

    fn post_h(&mut self, _fields: &mut Fields<'_>, _ctx: &StepContext<'_>) {}

    fn slab_hooks(&mut self, _partition: &SlabPartition<'_>) -> Option<Arc<dyn SlabHooks>> {
        Some(Arc::new(NoFusion))
    }
}

/// The final fields (all six components) and the steps run.
fn state(case: &str, threads: usize, openems: bool, unfused: bool) -> (u64, Vec<Vec<u32>>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/openems")
        .join(case)
        .join("model.xml");
    let (doc, _) = read_model(path).unwrap();
    let mut sim = Simulation::from_document_with(&doc, &CompileOptions::default(), |f| {
        f.threads(threads).openems_run(openems)
    })
    .unwrap();
    if unfused {
        sim.engine_mut().set_fused_steps(FusedSteps::Never);
        sim.engine_mut().add_operator(Unfuse);
    } else {
        sim.engine_mut().set_fused_steps(FusedSteps::Always);
    }
    sim.run().unwrap();
    let e = sim.engine();
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let mut out = Vec::new();
    for a in Axis::ALL {
        out.push(bits(e.u_component(a)));
        out.push(bits(e.i_component(a)));
    }
    (e.timestep(), out)
}

fn check(case: &str, threads: &[usize]) {
    for &t in threads {
        for openems in [true, false] {
            let fused = state(case, t, openems, false);
            let plain = state(case, t, openems, true);
            assert_eq!(
                fused.0, plain.0,
                "{case}, {t} threads, openems {openems}: steps"
            );
            assert!(
                fused.1 == plain.1,
                "{case}, {t} threads, openems {openems}: fields differ"
            );
        }
    }
}

/// PML (openEMS's layers for files, CPML natively), soft source, probes
/// of voltage and current, energy end criterion.
#[test]
fn pml_and_probes() {
    check("pml", &[1, 3, 4]);
    check("current_probes", &[1, 2]);
}

/// Dispersive slab in PML, plane wave.
#[test]
fn dispersive() {
    check("adv_drude_slab", &[1, 3]);
}

/// Lumped port (moved `pre_e` capture).
#[test]
fn lumped() {
    check("lumped_port", &[1, 3]);
    check("lumped_rc", &[2]);
}

/// Mur faces (voltage-only slab hooks) in a dielectric, and with PEC/PMC.
#[test]
fn mur() {
    check("adv_mur_dielectric", &[1, 3]);
    check("pmc_mur", &[2]);
}

/// Cylindrical grid, coax.
#[test]
fn cylindrical() {
    check("adv_cyl_coax", &[1, 3]);
}

/// Thin conducting sheet and a graded waveguide.
#[test]
fn sheet_and_waveguide() {
    check("adv_sheet_copper", &[1, 3]);
    check("adv_wg_graded", &[3]);
}
