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

Status: early development. Nothing is usable yet. See
[docs/DESIGN.md](docs/DESIGN.md) for the architecture and roadmap.

## Features

| feature | default | description |
|---|---|---|
| `xml` | yes | run openEMS XML files (via `csx/xml`) |

## Documentation

| document | contents |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | architecture, numerical formulation, milestones |
| [CLEANROOM.md](CLEANROOM.md) | rules for contributors (mandatory reading) |

## License

MIT. See [LICENSE](LICENSE).
