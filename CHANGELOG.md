# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.5](https://github.com/KarpelesLab/ems/compare/v0.0.4...v0.0.5) - 2026-10-29

### Other

- Design status: wave 19 (poles on axis grids, closed cylindrical grids, random sweep 547 of 560)
- openEMS files: the H normal to an upper PEC face stays 0; random sweep status (547 of 560)
- openEMS files: each edge goes to one excitation by priority; hard E sources on sheet edges add like soft ones
- openEMS files: lumped elements on sheet edges are spread over all edges, caps leave the sheet alone
- openEMS files: dispersive poles take the edge coefficients with parallel lumped elements folded in
- openEMS files: a conducting sheet on the lines an absorbing face claims is PEC
- openEMS files: conducting-sheet priorities against materials, metal, poles and lumped elements
- openEMS files: lumped elements keep their edges against PMC walls
- openEMS files: a Mur face carrying an excitation is held through sample L + 10
- openEMS files: lumped elements win their edges over metal at any priority
- Design status: closed cylindrical grids, alpha index ranges, graded alpha meshes
- openEMS files: seam sources on the copy they are drawn on, look-ahead H of the multigrid
- openEMS files: alpha index ranges of cylindrical boxes, graded alpha widths, the alpha = 0 seam line
- openEMS's arithmetic for several Debye poles and for magnetic poles on the seam
- Lorentz/Drude poles on cylindrical grids with the axis

## [0.0.4](https://github.com/KarpelesLab/ems/compare/v0.0.3...v0.0.4) - 2026-10-28

### Other

- Mark public structs and enums #[non_exhaustive]

## [0.0.3](https://github.com/KarpelesLab/ems/compare/v0.0.2...v0.0.3) - 2026-10-28

### Other

- Design status: openEMS sheet circuit, sweep 150 of 169 exact
- openEMS files: ConductingSheet circuit in openEMS's single-precision arithmetic
- openEMS files: sheet table bin from openEMS's single-precision (t/delta)^2
- series lumped elements against the Debye update (5.16.13)
- second round of series lumped element probes (5.16.13)
- structure probes of openEMS's series lumped elements (5.16.13)
- Design status: openEMS Debye update, sweep 149 of 169 exact
- openEMS's Debye arithmetic (5.16.12), sweep 149 of 169 exact
- openEMS files: openEMS's f32 arithmetic of Debye poles, also on the seam and axis
- structure probes of openEMS's Debye update (5.16.12)
- Design status: openEMS dispersive poles (5.16.12), sweep 148 of 169 exact
- openEMS's seam with dispersive poles (5.16.12)
- openEMS files: dispersive poles on cylindrical grids and openEMS's seam with poles
- openEMS's arithmetic of dispersive poles (5.16.12)
- openEMS files: Debye poles with a relaxation time below 3 dt are dropped
- openEMS files: openEMS's f32 arithmetic of Lorentz and Drude poles
- Oracle workflow: skip man-db triggers, apt retries and lock timeouts
- Design status: openEMS sheet branch update (5.16.13)
- openEMS files: openEMS's single-precision sheet branch update, refined sheet capacitance
- Design status: openEMS cylindrical arithmetic (5.16.11)
- openEMS's cylindrical arithmetic (5.16.11), README rows
- openEMS files: the seam inside PML layers, per-operation flush of the layer update
- openEMS files: openEMS's alpha weights of z-edge materials on cylindrical grids
- Cylindrical fixtures: tolerances of the models now identical to openEMS
- openEMS files: a cylindrical mesh without CylinderCoords runs as a Cartesian one
- openEMS files: openEMS's cylindrical multigrid (OpenemsMultigrid)
- Dispersive poles on cylindrical grids: candidate edges from the (rho, z) box
- openEMS files: cylindrical curl order, per-operation flush, seam and axis update
- Design status: feature sweep 146 of 169 exact after wave 16
- Merge wave 16 time step (5.16.10) with J's constants and lumped work (5.16.9); status
- openEMS files: openEMS's computed time step bit for bit
- openEMS round-off of lumped elements, current probes, vacuum constants (5.16.8)
- openEMS files: openEMS's vacuum constants in the coefficients
- openEMS files: OverSampling on ProbeBox and DumpBox
- openEMS files: openEMS's current-probe summation (f32, side order)
- openEMS files: lumped elements folded once with openEMS's coefficient rounding
- Design status: typo
- Design status: openEMS PML and Mur arithmetic
- openEMS's PML and Mur arithmetic (5.16.8)
- openEMS PML: openEMS's layer arithmetic, bit for bit on Cartesian grids
- Mur faces and AbsorbingBC sheets: openEMS's update arithmetic for files
- Design status: openEMS coefficient rounding, remaining sweep items
- openEMS files: openEMS's coefficient rounding (CoeffRounding)
- Design status: wave 15, openEMS summation order; prim_sphshell_metal exact
- openEMS files: openEMS's summation order of the curl (CurlOrder)
- exact subnormal flush (select), normal values unchanged
- Design status: oracle at openEMS-Project be30220
- DESIGN, README: openEMS e89286f -> be30220, measured (5.16.5, 7.2, 8.1 item 17)
- Follow openEMS-Project be30220: VTK layout, --gpu-device, exit status, inside tolerance
- Oracle workflow: time out hung package installs
- Oracle workflow: pinned openEMS revision on pushes, upstream HEAD weekly as a drift report
- status after wave 14
- timestep_cyl tests: clippy
- openEMS multigrid structure as measured; README: cyl_current_seam
- Current boxes touching the alpha seam of closed cylindrical grids loop like openEMS (files)
- Cylindrical fixtures run on openEMS's own time step; dumps sample the last step on closed and axis grids
- openEMS time step on cylindrical grids (files)
- Random differential sweep fixtures rs_* and DESIGN 5.16: sweep results, Mur and lumped rules
- openEMS holds a Mur face carrying an E excitation while it runs; upper PML face normal H stays 0
- fixes from a randomized differential sweep against openEMS
- lumped elements act only outside openEMS's PML layers
- openEMS energy measure: leave out the E on the three upper faces, not on both x faces
- material samples in metal see the material under it; f32 CFL step; openEMS's H-probe cell rule
- DESIGN 5.13: sheet deviation currents, sheets in absorbing layers, PML look-ahead, junctions
- Conformal sheets: junctions of several sheets and sheets touching solid metal
- Conformal sheets: deviation currents on merged regions, stretch in absorbing layers, PML in the energy look-ahead
- fuse steps only on large slabs, neighbour wait instead of mid barrier
- fused H/E steps plane by plane, movable read-only hooks
- Design and README: layered/dispersive plane waves and NF2FF ground (wave 14 C)
- NF2FF over a stratified ground: reflected and transmitted far fields
- Plane waves in layered and dispersive backgrounds: exact spectral incident field
- Sectoral waveguide port with fractional-order Bessel modes; DESIGN for the cylindrical ports
- CPW and stripline ports on cylindrical grids along z or alpha
- openEMS sheet model: interior/border width rule, cylindrical grids
- Triangle sheet tests run the 45-degree plate rotated exactly and by rotate_axis
- Conformal sheets: exact hits of nodes and edge midpoints do not depend on rounding
- Design and README: status after wave 13
- DESIGN 5.13: sheet interpolation, deviation currents, cylindrical PML on merged faces
- Conformal PEC: deviation currents, soft merging, cylindrical PML on merged faces
- Conformal sheets: interpolate small regions from their own side of the sheet
- Dumps of Dirac and step excitations record every step, as openEMS does
- openEMS end-criterion energy: full-cell outer duals, x-face rule, half-length check
- Current probes: half-way corners snap inwards, zero-length cylinders are points
- Rennings time step reproduced on doubly graded meshes and layered media
- Sweep test: no recorded differences left
- sigma_m rule, openEMS sheet model, polygon and polyhedron ties
- openEMS files: polygon vertex ties of metal Polygon and LinPoly
- openEMS files: emulate openEMS's ConductingSheet model
- openEMS files: dual-edge magnetic conductivity rule
- Field dumps on closed cylindrical grids as openEMS writes them
- openEMS's axis E_z update on closed cylindrical grids (file path)
- Microstrip port on cylindrical grids along z or alpha; sector CPML grading
- openEMS PMC faces: no update of the face's tangential E, skip the axis
- Radial-line port on cylindrical grids
- Debug material, operator and PEC dumps on cylindrical grids
- openEMS's radial material weights on cylindrical grids (file path)
- Lumped elements: openEMS's capacitance limit for parallel R/L without C
- SAR averaging: keep a half-size start within the tolerance
- SAR averaging: openEMS's weights, solver safeguards and validity on edge cases
- Design and README: openEMS PML emulation (wave 13 A)
- PML fixtures against openEMS; tighter PML tolerances
- openEMS's PML for the PML_n faces of openEMS files
- Slab-hook corrections to openEMS's energy and the look-ahead H
- DESIGN status after wave 12: conformal accuracy
- Design and README: end-to-end parity, openEMS benchmark, wave 12 status
- Probes read precomputed field locations
- Serial Mur faces read and write precomputed field locations
- Periodic face merge from a precomputed table
- a clone of a registered operator takes its edges back
- Default thread count: at most one thread per five x-planes
- Write the VTK files of time-domain dumps on several threads
- openEMS time-step criteria from precomputed edge tables
- Serial gaps only at the steps that need them
- End-to-end openEMS parity fixtures on realistic models
- Mur faces run as slab hooks
- Probe file headers scale the cylindrical alpha by the drawing unit
- Mur faces of XML runs follow openEMS at interfaces and corners
- End criterion of XML runs uses openEMS's energy measure
- DESIGN 5.13: interpolated small faces, absorbing layers, convergence tables
- Conformal PEC: TE/TM/TEM convergence tests and the CPML guide regression
- Conformal PEC: interpolate small cut faces; fix conformal walls in CPML
- Cylindrical SAR files: openEMS's exact averaging on the warped cells (sar_calc within 5e-8)
- refuse a cylindrical-grid x/y mirror that cuts the box at build time
- DESIGN 5.15 and README: cylindrical gaps closed (wave 12)
- Oblique thin wires on cylindrical grids
- Thin conformal PEC sheets on cylindrical grids
- Port helpers on cylindrical sector grids
- mirrors normal to x and y on cylindrical grids
- Cylindrical grids: periodic selection of sources and lumped elements; alpha conventions of openEMS
- CellConstantMaterial on cylindrical grids
- Mass-averaged SAR on cylindrical grids
- openEMS SAR averaging rules (DESIGN 5.12, 5.16, 7.1), README status
- openEMS averaging, face-cube types, -a auto range
- :sar: openEMS-compatible mass averaging (sar_calc methods)
- 0-step runs write the t = 0 row of E-type probe files, as openEMS
- DESIGN 5.16.3: XML feature-coverage matrix; csx feedback 17-21
- Lumped elements on non-Box primitives are skipped, as in openEMS
- XML feature-coverage sweep: 52 openEMS oracle fixtures and tests
- FDTD settings openEMS runs: odd TimeStepFactor, endCriteria, OverSampling
- NormalSignPositive is the inward normal, as in openEMS
- Dump sampling: OptResolution round-off, raw H/B boxes on dual lines
- Metal spheres, spherical shells and wires: open surface for files
- Read openEMS files with openEMS's reading rules
- Update README and design status after wave 11 (CLI, nf2ff/sar_calc tools, cylindrical gaps)
- require openEMS's row count for H probe files too
- DESIGN 5.16: numeric parity findings; tighten the PML parity test
- openEMS ignores it too
- End criterion of XML runs counts the Mur face edges like openEMS
- XML runs skip the probe boxes openEMS skips
- Lumped elements: openEMS update and spreading for XML models
- byte-exact fixtures on Windows, gen scripts under bash; per-dataset HDF5 dump comparison
- nf2ff tool, openEMS SAR methods and sar_calc (design 5.10, 5.12, 5.16, 7.1)
- openEMS averaging methods, in-run SIMPLE, sar_calc tool (ems-sar-calc)
- :openems::nf2ff: the openEMS nf2ff tool (ems-nf2ff)
- DESIGN 5.11: plane waves are Cartesian only (openEMS ignores them on cylindrical grids)
- Record that openEMS ignores plane waves on cylindrical grids
- Conformal PEC (solid metal) on cylindrical grids
- NF2FF on cylindrical grids
- SAR dumps on cylindrical grids; cylindrical oracle fixtures
- Conducting sheets on cylindrical grids
- Waveguide probes: mode purity, openEMS mode-matching files, final H sample
- :openems: excitation signal, statistics and debug files
- ems binary: a drop-in replacement for the openEMS executable
- Update README and design status after wave 10 (openEMS parity, M2 done)
- Optimise test builds (opt-level 2): the suite runs full simulations
- Reconcile core parity with dump and advanced-physics work
- :openems: gate the write_probes doctest on the xml feature
- openEMS core parity fixtures and tests
- :openems: openEMS probe text files (writer and reader)
- Follow openEMS semantics for XML models (black-box parity)
- Revert "Sample soft E/H sources at the start of their update (openEMS timing)"
- Fix a private intra-doc link in compile::planewave
- Oracle README: describe gen-adv.sh
- openEMS fixtures: graded microstrip; gate the parity tests on xml
- Mur faces tuned to the local medium; record the PML gap
- Cylindrical grids: openEMS fixtures, dispersive poles, CylinderCoords
- Waveguide mode probes: openEMS sampling and normalisation
- Plane waves: openEMS phase reference; record the ConductingSheet model gap
- Dispersive materials: openEMS pole assignment and leapfrog ADE
- Sample soft E/H sources at the start of their update (openEMS timing)
- openEMS dump fixtures and comparison tests
- openEMS-compatible field dump and SAR files (io::openems::dump)
- Dump quantities J, rot H, D, B; raw (native) samples; openEMS dump timing
- Add the new port helpers to the prelude
- Validate the circular waveguide port (cylindrical and Cartesian grids)
- widen the port boxes to the enclosing mesh lines
- Circular waveguide port (CircWaveguidePort, TE_nm/TM_nm)
- coaxial-line port on Cartesian and cylindrical grids
- coplanar-waveguide port with Wen/Simons quasi-static reference
- symmetric strip-line port with Cohn's exact impedance
- a lumped port joined to its far terminal by a PEC polyline
- Waveguide mode probes on cylindrical grids (z-normal planes)
- Share the transmission-line port extraction (port::line, LinePortResult)
- openEMS oracle workflow (rebuild, regenerate fixtures, test)
- Add the black-box openEMS oracle recipe for reference fixtures

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
