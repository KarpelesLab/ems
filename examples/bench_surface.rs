//! Cost per edge of the conductor and laminate models on a board-like
//! grid: a conducting sheet (rational model) over the whole plane `k = 2`,
//! a solid lossy plate of two cells below it, or a Debye laminate in the
//! three cell layers at the bottom.
//!
//! ```text
//! cargo run --release --example bench_surface -- [model=sheet] [threads=1] [steps=300] [nx=224] [ny=160] [nz=16] [pml=0]
//! ```
//!
//! `model`: `none`, `sheet`, `solid`, `debye` (eight poles per edge).
//! Prints the update rate, the time per step and edge of the model above
//! the grid without it (run `none` for that), and a hash of the final
//! fields and the energy: the same for any thread count, and what a
//! change that claims to keep results has to print again.

use std::time::Instant;

use ems::engine::boundary::cpml::CpmlBoundary;
use ems::engine::dispersive::{Dispersive, Field, Pole};
use ems::engine::sheet::SheetEdge;
use ems::engine::surface::{Conductor, Roughness, SurfaceImpedance, box_edges};
use ems::engine::{EdgeMaterials, Engine, Grid};
use ems::{Axis, Cpml, Face};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args.first().map_or("sheet", |s| s.as_str());
    let num = |k: usize, d: usize| args.get(k).map_or(d, |a| a.parse().expect("number"));
    let (threads, steps) = (num(1, 1), num(2, 300) as u64);
    let n = [num(3, 224), num(4, 160), num(5, 16)];
    let pml = num(6, 0);

    let grid = Grid::uniform(n, [0.25e-3, 0.25e-3, 0.2e-3]).expect("grid");
    let cells = grid.num_cells() as f64;
    let mut m = EdgeMaterials::vacuum(&grid);
    let plate = ([0, 0, 1], [n[0], n[1], 3]);
    if model == "solid" {
        for (axis, node) in box_edges(plate.0, plate.1) {
            m.set_pec_edge(axis, node);
        }
    }
    let mut e = Engine::builder(grid)
        .edge_materials(m)
        .threads(threads)
        .build()
        .expect("engine");
    if pml > 0 {
        let faces = [Face::XMin, Face::XMax, Face::YMin, Face::YMax, Face::ZMax];
        let b = CpmlBoundary::new(&e, faces.map(|f| (f, Cpml::cells(pml)))).expect("cpml");
        e.add_operator(b);
    }
    let mut edges = 0;
    match model {
        "none" => {}
        "sheet" => {
            let mut s = SurfaceImpedance::new(Some(10e9));
            for (axis, other) in [(Axis::X, Axis::Y), (Axis::Y, Axis::X)] {
                for a in 0..n[axis.index()] {
                    for b in 1..n[other.index()] {
                        let mut node = [0, 0, 2];
                        node[axis.index()] = a;
                        node[other.index()] = b;
                        s.push_sheet(
                            SheetEdge {
                                axis,
                                node,
                                normal: Axis::Z,
                                conductivity: 5.8e7,
                                thickness: 35e-6,
                                upper_share: None,
                            },
                            Roughness::Smooth,
                        );
                    }
                }
            }
            let h = s.add_to(&mut e).expect("sheet");
            edges = e.operator(&h).sheet_edges();
        }
        "solid" => {
            let mut s = SurfaceImpedance::new(Some(10e9));
            s.add_box(plate.0, plate.1, Conductor::new(5.8e7));
            let h = s.add_to(&mut e).expect("plate");
            edges = e.operator(&h).solid_edges();
        }
        "debye" => {
            let mut d = Dispersive::new();
            for axis in Axis::ALL {
                for i in 1..n[0] {
                    for j in 1..n[1] {
                        for k in 1..3 {
                            for p in 0..8 {
                                let tau = 1e-9 / 10f64.powf(0.5 * p as f64);
                                d.push(Field::Electric, axis, [i, j, k], Pole::debye(0.05, tau));
                            }
                        }
                    }
                }
            }
            let h = d.add_to(&mut e).expect("laminate");
            edges = e.operator(&h).edges();
        }
        other => panic!("unknown model {other}"),
    }
    {
        let mut f = e.fields_mut();
        f.u.set(Axis::Z, [n[0] / 3, n[1] / 2, 3], 1.0);
        f.u.set(Axis::Z, [n[0] / 2, n[1] / 3, 1], -0.6);
        f.u.set(Axis::X, [2 * n[0] / 3, n[1] / 2, 4], 0.8);
    }
    e.run(20); // warm up
    let t = Instant::now();
    e.run(steps);
    let s = t.elapsed().as_secs_f64();
    // FNV-1a over the bits of every field value and of the energy.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            hash = (hash ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for axis in Axis::ALL {
        e.u_component(axis)
            .iter()
            .for_each(|v| eat(v.to_bits() as u64));
        e.i_component(axis)
            .iter()
            .for_each(|v| eat(v.to_bits() as u64));
    }
    eat(e.energy().to_bits());
    println!(
        "{model}: {cells:.0} cells, {edges} edges, threads={}, steps={steps}, pml={pml}: {:.1} Mcells/s, {:.3} ms/step, {:.1} ns per edge and step (all of the step), hash {hash:016x}",
        e.threads(),
        cells * steps as f64 / s / 1e6,
        s / steps as f64 * 1e3,
        if edges == 0 {
            0.0
        } else {
            s / steps as f64 / edges as f64 * 1e9
        },
    );
}
