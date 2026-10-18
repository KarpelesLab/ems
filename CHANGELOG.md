# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1](https://github.com/KarpelesLab/ems/compare/v0.0.0...v0.0.1) - 2026-10-18

### Other

- Depend on csx 0.0.2 from crates.io; drop the git pin
- Record M1 implementation status in design doc
- Drop stale OverSampling warning; list dispersive media in README
- Add dispersive media (Drude/Lorentz/Debye ADE operator) and csx mapping
- Ignore mesh-check warnings in the port probe-placement test
- quick start, run_xml usage and status table
- Add microstrip, patch antenna and dipole examples
- Add Simulation::run_with and the run_xml and cavity examples
- Run openEMS XML files; mesh checks and MaxTime in Simulation
- Second-order lumped-port reference plane; probe/dump polish
- Map csx excitations, lumped elements, probes and dumps onto the engine
- Add analytic physics validation suite (design §9)
- Add solver-aware mesh checks (ems::mesh)
- Record csx integration feedback in design doc
- Ignore agent build directories
- Run CPML slab-parallel, skip empty hook phases, add operator energy
- Add csx compile step, per-edge materials and Simulation
- Add engine field dump operators (time and frequency domain)
- Build against csx master until csx 0.0.2 is released
- Convert FDTD settings and waveforms from/to csx::fdtd
- Add CPML absorbing boundary operator
- Document implemented lumped element model
- Add leapfrog energy invariant and run_until end criteria
- Add lumped elements, lumped ports and port analysis
- Add first-order Mur absorbing boundary operator
- Use Axis and Face in the engine API
- Update design doc with engine core decisions
- Add engine validation tests and vacuum benchmark example
- Add FDTD engine core: grid, materials, coefficients, kernels, thread pool
- Add io module: VTK .vtr/.pvd writers and CSV series writer
- Ignore local agent worktrees
- Add waveforms, DFT and solver settings
