//! Kernel throughput on an N³-cell vacuum box with PEC walls.
//!
//! ```text
//! cargo run --release --example bench_vacuum -- [N=128] [threads=0 (default)] [steps=200] [K=0] [pml=0]
//! ```
//!
//! Prints the update rate in Mcells/s (cells × steps / wall time; one cell
//! update = all six field components). With `K > 0` the steps run through
//! `Engine::run_until` with an energy evaluation every `K` steps (and an
//! end criterion that never fires), to measure that overhead. With
//! `pml > 0` a CPML of `pml` cells is added on all six faces.

use std::time::Instant;

use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::{Engine, Grid, RunLimits};
use ems::{Axis, Cpml, EndCriteria, Face};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric argument"))
        .collect();
    let n = args.first().copied().unwrap_or(128);
    let threads = args.get(1).copied().unwrap_or(0);
    let steps = args.get(2).copied().unwrap_or(200) as u64;
    let k = args.get(3).copied().unwrap_or(0) as u64;
    let pml = args.get(4).copied().unwrap_or(0);

    let grid = Grid::uniform([n, n, n], [1e-3; 3]).expect("grid");
    let cells = grid.num_cells() as f64;
    let mut e = Engine::builder(grid)
        .threads(threads)
        .build()
        .expect("engine");
    if pml > 0 {
        let b = CpmlBoundary::new(&e, Face::ALL.map(|f| (f, Cpml::cells(pml)))).expect("cpml");
        e.add_operator(b);
    }
    {
        let mut f = e.fields_mut();
        f.u.set(Axis::Z, [n / 3, n / 2, n / 2], 1.0);
    }
    e.run(10); // warm up
    let t = Instant::now();
    if k == 0 {
        e.run(steps);
    } else {
        let limits =
            RunLimits::new(steps).end_criteria(EndCriteria::energy_ratio(1e-30).check_interval(k));
        e.run_until(&limits).expect("run");
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "N={n} ({cells:.0} cells), threads={}, steps={steps}, K={k}, pml={pml}: {:.1} Mcells/s ({:.3} ms/step)",
        e.threads(),
        cells * steps as f64 / s / 1e6,
        s / steps as f64 * 1e3
    );
}
