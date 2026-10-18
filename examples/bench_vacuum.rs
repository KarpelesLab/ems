//! Kernel throughput on an N³-cell vacuum box with PEC walls.
//!
//! ```text
//! cargo run --release --example bench_vacuum -- [N=128] [threads=0 (all)] [steps=200]
//! ```
//!
//! Prints the update rate in Mcells/s (cells × steps / wall time; one cell
//! update = all six field components).

use std::time::Instant;

use ems::engine::{Engine, Grid};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric argument"))
        .collect();
    let n = args.first().copied().unwrap_or(128);
    let threads = args.get(1).copied().unwrap_or(0);
    let steps = args.get(2).copied().unwrap_or(200) as u64;

    let grid = Grid::uniform([n, n, n], [1e-3; 3]).expect("grid");
    let cells = grid.num_cells() as f64;
    let mut e = Engine::builder(grid)
        .threads(threads)
        .build()
        .expect("engine");
    {
        let mut f = e.fields_mut();
        f.u.set(2, [n / 3, n / 2, n / 2], 1.0);
    }
    e.run(10); // warm up
    let t = Instant::now();
    e.run(steps);
    let s = t.elapsed().as_secs_f64();
    println!(
        "N={n} ({cells:.0} cells), threads={}, steps={steps}: {:.1} Mcells/s ({:.3} ms/step)",
        e.threads(),
        cells * steps as f64 / s / 1e6,
        s / steps as f64 * 1e3
    );
}
