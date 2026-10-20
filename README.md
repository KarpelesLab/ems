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

Status: early development (milestones 1–4 done; openEMS parity work
ongoing, checked against a black-box openEMS, see
[scripts/openems-oracle](scripts/openems-oracle/README.md)). The API will still change. See [docs/DESIGN.md](docs/DESIGN.md) for the architecture and
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
model.grid.set_lines(0, (0..=40).map(f64::from))?;
model.grid.set_lines(1, (0..=40).map(f64::from))?;
model.grid.set_lines(2, (0..=20).map(f64::from))?;
// 2 mm FR-4 on a ground plane (the PEC z_min face), a 12 mm square patch
// on top; everything stays clear of the 8-cell PML on the other faces.
let sub = model.add_property(Property::material("sub", csx::Material::new(4.3, 0.0)));
model.add_primitive(
    sub,
    Primitive::new(BoxShape::new([10., 10., 0.], [30., 30., 2.])),
)?;
let patch = model.add_property(Property::metal("patch"));
model.add_primitive(
    patch,
    Primitive::new(BoxShape::new([14., 14., 2.], [26., 26., 2.])),
)?;
let port = LumpedPort::new(1, 50.0, [17., 20., 0.], [17., 20., 2.], Axis::Z)
    .excite(1.0)
    .add_to(&mut model)?; // writes LumpedElement + Excitation + probes into the model

let fdtd = Fdtd::new()
    .waveform(Waveform::gaussian(2e9, 1e9))
    .boundaries(Boundaries::all(Boundary::pml(8)).with(Face::ZMin, Boundary::Pec))
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
`cavity`, `microstrip`, `waveguide`, `patch_antenna`, `dipole` (`cargo run --release
--example <name>`).

## Status

| area | implemented | not yet |
|---|---|---|
| mesh | Cartesian and cylindrical (ρ, α, z, with axis, closed azimuth and azimuthal multigrid near the axis) rectilinear, non-uniform; solver-aware checks (`ems::mesh`) | |
| materials | ε, μ, κ, σ, diagonal anisotropy, weighting, sub-cell averaging (via `csx`); Drude, Lorentz, Debye (electric and magnetic) | |
| metal | PEC volumes, sheets and lines; priorities; ConductingSheet (skin-effect thin conductor); conformal PEC volumes and zero-thickness sheets (`CompileOptions::with_conformal`); thin wires (radius below the cell size), axis-aligned or oblique | conformal PEC on cylindrical grids |
| boundaries | PEC, PMC, Mur, CPML (`PML_n`, medium-tuned); AbsorbingBC sheets; periodic; Bloch / oblique incidence (`Fdtd::bloch_wavevector` or an oblique plane wave); stretched-radius cylindrical PML on ρ_max | |
| sources | E/H soft/hard excitations with weighting and delay; plane waves (TF/SF, homogeneous possibly lossy background); Gaussian, sine, Dirac, step, custom waveforms | |
| lumped elements | R, L, C, parallel and series | |
| probes | voltage, current, E/H field, waveguide mode voltage/current; DFT; openEMS probe text files (`ems::io::openems`) | openEMS mode-matching probe file format |
| dumps | E, H, J, rot H, D, B in time and frequency domain, node/cell/native sampling, sub-sampling, OptResolution; ems VTK/HDF5 and the openEMS file layouts (VTK `.vtr`/`.vts`, HDF5 current and legacy) | |
| ports | lumped, curve, microstrip, coplanar waveguide, stripline, coaxial, rectangular and circular waveguide helpers; S-parameters, Z_in | |
| antennas | NF2FF (`ems::nf2ff`): patterns, directivity, gain, radiated power; PEC/PMC mirror planes; homogeneous lossless background | openEMS nf2ff file format |
| SAR | local, 1 g / 10 g averaged (`ems::post::sar`), from csx SAR dumps; openEMS SAR dump files | openEMS's in-run averaging; `sar_calc` |
| openEMS XML | `Simulation::from_xml`, `<FDTD>` settings, bit-identical round trip; openEMS semantics (signals, timing, time step rules, probe snapping, end criteria) matched to ~1e-6 on the oracle fixtures | openEMS command-line drop-in |
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
