# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2](https://github.com/KarpelesLab/ems/compare/v0.0.1...v0.0.2) - 2026-10-19

### Other

- Update README and design status after wave 9
- Oblique thin wires: 1-D wire network coupled to the Yee lattice
- Conformal zero-thickness PEC sheets (split cells, merged small regions)
- Cylindrical PML on the rho_max face (complex-stretched radius)
- Azimuthal multigrid near the cylindrical axis
- Plane waves and NF2FF in non-vacuum backgrounds; plane-wave end criterion
- Use f64::consts::EULER_GAMMA (clippy approx_constant on Rust 1.99)
- Update README and design status after wave 8 (M4 done)
- Fix conformal/Bloch merge: pass the engine through; no conformal on cylindrical grids
- Cylindrical grids through compile and Simulation; lumped copies; docs
- Cylindrical (rho, alpha, z) engine: geometry, axis, closed azimuth, stability
- Bloch mode in Simulation: oblique plane waves on periodic structures
- Add thin-wire model for csx Wire primitives thinner than the staircase
- Add conformal PEC (Dey-Mittra in integral form with cell merging)
- normalise CRLF so the snippet check passes on Windows
- Update README and design status after wave 7 (M3 done)
- avoid hdf5-only intra-doc links in builds without the feature
- Add SAR: local and 1 g / 10 g mass-averaged, from csx SAR dumps
- Add periodic boundaries and Bloch (sine-cosine) engine pairs
- Add plane-wave excitation (TF/SF) and csx PlaneWave mapping
- Lumped ports: document hand analysis; run_xml writes element currents
- README quick start: warning-free model, kept in sync by a test
- tune sigma_max to the layer's refractive index
- Make the engine energy exact with conducting sheets
- Include magnetic dispersive poles in the engine energy invariant
- Add NF2FF transform: radiation patterns, directivity, gain, power
- Update README and design status after wave 6 (M2 mostly done)
- Add ConductingSheet thin-conductor model (skin effect) and csx mapping
- Fix rustdoc link to private constant in default_threads docs
- branch-free, 4-lane energy accumulation loops
- yield in the worker barrier before sleeping
- default to one thread per physical core, fewer on small grids
- read coefficients through a row-deduplicated kernel view
- Add microstrip-line port helper (MslPort) with line-parameter extraction
- Document waveguide probes in the Simulation docs
- Add rectangular waveguide port (TE_mn) and waveguide mode probes
- Design doc: M1 complete; hdf5rs feedback
- Add optional HDF5 output (feature `hdf5`, pure-Rust hdf5rs)
- Add AbsorbingBC sheets (first-order Mur on interior mesh planes)
- Design doc: mark the 64-thread slowdown as unexplained
- Record idle-machine benchmark in design doc

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
