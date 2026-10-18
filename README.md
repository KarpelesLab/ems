# ems

[![CI](https://github.com/KarpelesLab/ems/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/ems/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ems.svg)](https://crates.io/crates/ems)
[![docs.rs](https://img.shields.io/docsrs/ems)](https://docs.rs/ems)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure Rust FDTD (finite-difference time-domain) electromagnetic field solver,
intended as a replacement for [openEMS](https://openems.de).

* Clean-room implementation: no openEMS or CSXCAD code was consulted. See
  [CLEANROOM.md](CLEANROOM.md). MIT licensed.
* Models come from [`csx`](https://crates.io/crates/csx), the companion
  CSXCAD-compatible geometry and material crate. `ems` adds the solver: mesh
  compilation, time stepping, boundaries, sources, probes, ports, and
  post-processing.
* Goal: run existing openEMS XML models unchanged.
* Edition 2024, MSRV 1.95.

Status: early development (milestone 1 mostly done). The API will still
change. See [docs/DESIGN.md](docs/DESIGN.md) for the architecture and
roadmap.

## Quick start

Build a model with `csx`, describe the solver settings with `ems::Fdtd`,
run it, and analyse the port:

```rust
use csx::primitives::BoxShape;
use csx::{ContinuousStructure, Primitive, Property};
use ems::prelude::*;

let mut model = ContinuousStructure::new();
model.grid.delta_unit = 1e-3; // drawing unit: mm
for a in 0..3 {
    model.grid.set_lines(a, (0..=40).map(f64::from))?;
}
let sub = model.add_property(Property::material("sub", csx::Material::new(4.3, 0.0)));
model.add_primitive(sub, Primitive::new(BoxShape::new([0., 0., 0.], [40., 40., 2.])))?;
let port = LumpedPort::new(1, 50.0, [20., 20., 0.], [20., 20., 2.], Axis::Z)
    .excite(1.0)
    .add_to(&mut model)?; // writes LumpedElement + Excitation + probes into the model

let fdtd = Fdtd::new()
    .waveform(Waveform::gaussian(2e9, 1e9))
    .boundaries(Boundaries::all(Boundary::pml(8)))
    .end_criteria(EndCriteria::energy_decay_db(-40.0));
let mut sim = Simulation::new(&model, &fdtd)?;
for w in sim.warnings() {
    eprintln!("warning: {w}"); // includes the solver-aware mesh checks
}
let res = sim.run()?;
let z_in = port.analyze(&res, &[1e9, 2e9, 3e9])?.z_in();
```

Run an existing openEMS XML file (`Simulation::from_xml` in the API):

```text
cargo run --release --example run_xml -- model.xml [--threads N] [--out DIR]
```

It prints warnings, progress and the outcome, and writes every probe as
`DIR/<name>.csv` and every dump as `.vtr` files. More examples:
`cavity`, `microstrip`, `patch_antenna`, `dipole` (`cargo run --release
--example <name>`).

## Status

| area | implemented | not yet |
|---|---|---|
| mesh | Cartesian rectilinear (non-uniform), solver-aware checks (`ems::mesh`) | cylindrical (M4) |
| materials | ε, μ, κ, σ, diagonal anisotropy, weighting, sub-cell averaging (via `csx`); Drude, Lorentz, Debye (electric and magnetic) | |
| metal | PEC volumes, sheets and lines; priorities | conducting-sheet model (approximated as PEC), conformal (M4) |
| boundaries | PEC, PMC, Mur, CPML (`PML_n`) | periodic (M3) |
| sources | E/H soft/hard excitations with weighting and delay; Gaussian, sine, Dirac, step, custom waveforms | plane waves |
| lumped elements | R, L, C, parallel and series | |
| probes | voltage, current, E/H field; DFT | waveguide probes |
| dumps | E/H time and frequency domain, VTK `.vtr` | HDF5, openEMS file layout (M2) |
| ports | lumped port helper, S-parameters, Z_in | MSL / waveguide ports (M2), NF2FF (M3) |
| openEMS XML | `Simulation::from_xml`, `<FDTD>` settings, bit-identical round trip | |
| engine | multithreaded, deterministic for any thread count, energy end criteria | GPU |

## Features

| feature | default | description |
|---|---|---|
| `xml` | yes | run openEMS XML files (via `csx/xml`) |
| `hdf5` | no | write field dumps and probes as HDF5 (`ems::io::h5`, ems-native layout), via the pure-Rust [`hdf5rs`](https://crates.io/crates/hdf5rs) |
| `hdf5-deflate` | no | `hdf5` plus gzip compression of field datasets |

## Documentation

| document | contents |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | architecture, numerical formulation, milestones |
| [CLEANROOM.md](CLEANROOM.md) | rules for contributors (mandatory reading) |

## License

MIT. See [LICENSE](LICENSE).
