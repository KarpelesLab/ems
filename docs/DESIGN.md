# `ems` — design document (draft for review, rev. 2)

`ems` is a Rust FDTD (finite-difference time-domain) electromagnetic field solver,
intended as a clean-room replacement for openEMS.

`ems` is the solver half of a two-crate pair:

| crate | replaces | responsibility |
|---|---|---|
| [`csx`](https://crates.io/crates/csx) | CSXCAD | geometry, primitives, materials and other properties, parameters and expressions, rectilinear grid, CSXCAD/openEMS XML |
| `ems` (this crate) | openEMS | turning a `csx` model into an FDTD engine, time stepping, boundaries, sources, probes, ports, post-processing |

Crate metadata, matching `csx` conventions:
- Edition 2024, `rust-version = "1.95"`, version `0.0.0`, license MIT.
- `missing_docs = "warn"`.
- The same CI layout as csx: fmt, clippy `-D warnings` on all/no-default
  features, tests on Linux/macOS/Windows, docs, an MSRV job pinned to 1.95, and
  release-plz.

---

## 1. Clean-room policy

Same rules as `csx`. No openEMS, CSXCAD, or AppCSXCAD source code is read or
copied, by humans or by AI agents, not even "to check how they did it".

**Allowed sources**
- Published literature:
  - Yee (1966)
  - Taflove & Hagness, *Computational Electrodynamics*
  - Gedney, *Introduction to the FDTD Method*
  - Roden & Gedney (2000), CPML
  - Mur (1981)
  - Luebbers / Piket-May on lumped elements
  - Pozar for ports and S-parameters
- openEMS's public user documentation (wiki, tutorials, published API reference).
  Use it to learn which features exist and what they mean to users.
- The **observable openEMS XML file format**, for interoperability. Element and
  attribute names are an interface; where the docs are silent, we choose the
  behaviour deliberately and document it in rustdoc.
- **Black-box observation of a compiled openEMS** (the "oracle",
  `scripts/openems-oracle/`): we run it on XML we wrote and read its console
  output and the files it writes (probe text files, HDF5, VTK, statistics),
  to match output formats and to compare numbers. Its sources, headers and
  bundled scripts are deleted after the build and are never read.
  Fixtures live in `tests/fixtures/openems/`.

**Process**
- Every non-trivial algorithm has a doc comment citing its literature source.
- `CLEANROOM.md` (shared wording with csx) is affirmed in the PR template.

**Consequence:** we are not bound to openEMS's internal formulation. openEMS
uses an equivalent-circuit FDTD. We use the standard Yee scheme in integral form
(§5), which is textbook material and comes out equivalent on rectilinear meshes.

## 2. What lives where

**`csx` owns the model.** `ems` must not duplicate any of it:
- Primitives and inside tests, including cylindrical-coordinate primitives and
  transforms.
- Materials, including per-direction ε/μ/κ/σ and spatial weighting expressions.
- Dispersive material parameters (Lorentz/Debye).
- Metal, ConductingSheet, LumpedElement, Excitation, ProbeBox, DumpBox,
  AbsorbingBC.
- Priorities and the background material.
- `RectilinearGrid`: lines, delta unit, snapping, and smooth-mesh helpers.
- Parameters and expressions, re-evaluated by `update()`.
- XML read/write, including the openEMS document wrapper.
- The file-level `<FDTD>` settings, typed as `csx::fdtd::FdtdSettings`
  (all fields optional for a faithful round trip; excitation signal,
  boundary conditions, timestep settings). See §11 Q1.

**`ems` owns the solver:**
- **Resolved FDTD settings** (`ems::Fdtd`, `ems::Waveform`): the validated
  solver configuration, including what the file format cannot express
  (CPML grading parameters, thread count, end-criteria check interval,
  analytic waveform spectrum). `Fdtd::from_csx` converts csx's
  `FdtdSettings` (returning warnings for ignored settings) and
  `Fdtd::to_csx` converts back; `Waveform::from_csx` / `to_csx` do the
  same for the excitation signal.
- **Compile step.** `csx::ContinuousStructure` + settings → `Engine`. This
  rasterises materials onto edges, maps Excitation/ProbeBox/DumpBox/LumpedElement
  primitives to engine operators, and computes the time step.
- **Engine.** Fields, update kernels, boundaries, sources, lumped elements,
  probes, dumps, and end criteria.
- **High-level helpers** such as lumped ports (and later MSL/waveguide ports and
  NF2FF boxes). These **emit csx properties** (LumpedElement + Excitation +
  voltage/current ProbeBoxes), the same way an openEMS script does. A port built
  in Rust therefore round-trips through XML, and a port read from XML runs
  unchanged.
- **Post-processing:** DFT, port voltages/currents, S-parameters, impedance,
  later far-field.

Because the engine consumes csx properties rather than its own ad-hoc source and
probe types, **running existing openEMS XML files comes almost for free**. csx
parses the file; `ems` only needs to interpret the `<FDTD>` section. This is now
part of M1 (§3).

## 3. Scope

### Milestone 1 ("solid core")
- 3D Cartesian Yee FDTD on a non-uniform rectilinear mesh (`csx` grid).
- Materials:
  - ε, μ, electric conductivity κ, magnetic loss.
  - **Diagonal anisotropy** (x/y/z values from csx). This is free in the
    per-edge formulation.
  - Spatial weighting expressions (evaluated via csx).
- Metal (PEC), including **zero-thickness sheets**, and PMC boundaries.
- Boundaries per face: PEC, PMC, Mur (1st order), CPML.
- Waveforms: Gaussian-modulated pulse, sinusoid, Dirac, step, custom
  `Fn(f64) -> f64`.
- Sources from csx `Excitation` (soft/hard E-field over a primitive, with
  vector weighting and delay).
- Lumped elements from csx `LumpedElement`. R is in M1; L/C come later in M1 if
  time allows, otherwise M2.
- Lumped port helper and S-parameters.
- Probes from csx `ProbeBox`:
  - voltage: line integral of E;
  - current: loop integral of H;
  - E/H field probes;
  - time-domain series, plus running DFTs at the probe's frequencies.
- Dumps from csx `DumpBox`: time-domain snapshots and frequency-domain DFTs.
  `.vtr` (VTK) writer in-tree; HDF5 behind a feature flag.
- End criteria: max timesteps plus energy decay.
- Multithreaded and deterministic regardless of thread count.
- **openEMS XML**: `ems::Simulation::from_xml(path)`, built on csx's reader.
  Results are exposed through the Rust API; on-disk output compatibility is M2.

### Later milestones
| Milestone | Feature | csx side |
|---|---|---|
| M2 | Dispersive materials via ADE (Lorentz, Drude as zero-ω₀ Lorentz, Debye) — **done** (`engine::dispersive`, see §6) | Lorentz/Debye properties |
| M2 | ConductingSheet (thin lossy conductor) surface model — **done** (`engine::sheet`, §5.9) | ConductingSheet |
| M2 | MSL and rectangular waveguide port helpers — **done** (`port::MslPort`, `port::RectWaveguidePort`, §5.5) | probes, excitations with weighting |
| M2 | openEMS-compatible probe/dump output files (paths, text layout, HDF5 layout) | — |
| M2 | AbsorbingBC sheets — **done** (`engine::boundary::abc`) | AbsorbingBC |
| M3 | NF2FF, radiation patterns, gain/directivity — **done** (`ems::nf2ff`, §5.10) | DumpBox |
| M3 | Periodic / Bloch boundaries — **done** (`Boundary::Periodic`, engine-level `BlochPair`, §5.6.1) | — |
| M4 | Cylindrical-coordinate mesh — **done** (§5.15); multigrid near the axis not yet | cylindrical `RectilinearGrid` |
| M4 | Conformal / sub-cell PEC, thin wires — **done** (§5.13, §5.14) | Wire/Curve primitives |
| M4 | SAR — **done** (`ems::post::sar`, §5.12) | DumpBox |
| later | Python bindings (pyo3, probably a separate `ems-py` crate covering both), GPU backend | — |

Cylindrical-coordinate *primitives* on a Cartesian mesh already work in M1:
inside tests belong to csx.

## 4. Public API sketch

The csx names below are illustrative. ems follows the csx API as it lands.

```rust
use csx::{ContinuousStructure, Material, Metal, primitives::Box as CsxBox};
use ems::prelude::*;

// 1. Model (csx): geometry and materials in drawing units, scaled by the grid's delta unit.
let mut model = ContinuousStructure::new();
model.grid_mut().set_delta_unit(1e-3);                     // mm
let fr4 = model.add_property(Material::new("FR4").epsilon(4.3).kappa(0.01));
let cu  = model.add_property(Metal::new("copper"));
model.add_primitive(fr4, CsxBox::new([-20., -15., 0.], [20., 15., 1.6]), 0);
model.add_primitive(cu,  CsxBox::new([-5., -5., 1.6], [5., 5., 1.6]), 10);     // sheet
model.add_primitive(cu,  CsxBox::new([-20., -15., 0.], [20., 15., 0.]), 10);   // ground

// 2. Solver settings (ems): the openEMS <FDTD> section.
let fdtd = Fdtd::new()
    .waveform(Waveform::gaussian(f0, fc))
    .boundaries(Boundaries::all(Boundary::Pml(Cpml::cells(8))))
    .end_criteria(EndCriteria::energy_decay_db(-40.0))
    .max_timesteps(200_000);

// 3. Helpers write csx properties into the model.
let port = LumpedPort::new(1, 50.0, [0., 0., 0.], [0., 0., 1.6], Axis::Z)
    .excite(1.0)
    .add_to(&mut model)?;

// Meshing uses csx grid helpers; ems adds solver-aware checks
// (λ/20 at f_max in the densest material, grading ratio, PML room).
ems::mesh::check(&model, &fdtd)?;

// 4. Compile and run (or `prepare()` for an Engine you can step yourself).
let res = Simulation::new(&model, &fdtd)?.run()?;
let s11 = port.analyze(&res, &freqs).s11();

// Alternatively, run an existing openEMS file:
let res = Simulation::from_xml("patch.xml")?.run()?;
```

Design rules:
- **Units.** Geometry stays in csx drawing units. The compile step multiplies by
  the grid's delta unit, and everything inside ems is SI (metres, seconds).
- **Expressions.** The compile step borrows the model immutably and calls
  csx's `update()` on a clone if expressions are dirty. ems never mutates the
  caller's model, except through explicit helper calls such as `add_to`.
- **Handles.** Results are looked up by the csx property they came from
  (typed id, or name for XML-sourced models).
- **Stepping.** `Simulation::prepare()` returns an `Engine` with `step()`,
  `run_until()`, and read access to the fields.
- **Errors.** A hand-written `ems::Error` enum (no `thiserror`, like csx) that
  wraps `csx::Error`. Compiling is fallible; stepping is not.
- **I/O.** No hidden I/O; writing files is always an explicit call.

## 5. Numerical formulation

### 5.1 Grid
- Primary mesh: the csx `RectilinearGrid` lines, times the delta unit.
- E lives on **primary edges**. H lives on **dual edges**, which pierce
  primary-face centres.
- Dual nodes are the primary-cell midpoints. Dual cells at the domain boundary
  are half cells.
- Cartesian `(x, y, z)` or cylindrical `(ρ, α, z)` (§5.13): `ρ`, `z` scaled
  by the delta unit, `α` in radians.

### 5.2 Integral (voltage/current) form
We store edge integrals, not field values:
- `u = E·l`, on a primary edge of length `l`;
- `i = H·l*`, on a dual edge of length `l*`.

Per face:

```
C · du/dt + G · u = Σ i   (around the dual face pierced by the primary edge)
L · di/dt + R · i = −Σ u  (around the primary face pierced by the dual edge)
```

The circuit quantities, with `A*` the dual face area and `A` the primary face area:
- `C = ε·A*/l` and `G = κ·A*/l`
- `L = μ·A/l*` and `R = σ_m·A/l*`

Leapfrog in time, with the loss term averaged semi-implicitly:

```
u^{n+1}   = ca·u^n     + cb·(Σi)^{n+1/2}
i^{n+3/2} = da·i^{n+1/2} − db·(Σu)^{n+1}
ca = (C − G·dt/2)/(C + G·dt/2),   cb = dt/(C + G·dt/2)
```

This is the standard Yee scheme (Taflove, chapters on non-uniform grids and
lumped elements). It has two advantages:
- Lumped R/L/C and ports are just extra `G`/`C` (or an auxiliary state) added
  to an edge.
- Cell sizes are folded into the coefficients, so the kernel contains no
  geometry.

### 5.3 Rasterising the csx model
The compile step turns csx properties into per-edge coefficients.

**Implemented (`ems::compile`):** the sampling itself is done by
`csx::discretize`, which returns per-E-edge ε/κ (quarter-cell averages),
per-H-face μ (harmonic mean of the two half cells) and σ_m (arithmetic
mean), a PEC mask and conducting-sheet edges. The compile step reorders
these into `engine::EdgeMaterials` (z-fastest layout) and the engine builds
`C`, `G`, `L`, `R` from them. The per-cell `CellMaterials` path is converted
to `EdgeMaterials` too, so both share one coefficient builder; they differ
only in how σ_m is combined across an interface. Conducting sheets get the
thin-conductor model of §5.9. Models read from openEMS files take
openEMS's σ_m rule and sheet model instead (§5.16.2, §5.16.3).

**Volume materials** (Material, later dispersive): sampled at **primary cell
centres**, using the csx priority rules.
- Per-direction values feed the edges of that direction: x-edges use ε_x, κ_x.
- `C`/`G` of an edge is the area-weighted parallel sum over the up to 4 cells
  that share its dual face.
- `L`/`R` combines the 2 half-cells along the dual edge in series.
- Weighting expressions are evaluated by csx at the sample point.

**Metal / PEC** is evaluated on **edges**, not cells. An E-edge whose segment lies
inside a metal primitive (closed bounds, using csx's tolerance) gets
`ca = cb = 0`.
- 2D primitives (csx `dimension() == 2`, e.g. a zero-thickness box) mark the
  in-plane edges they cover.
- This requires a mesh line in the sheet plane, which `ems::mesh::check` verifies.

**Priority between metal and material** follows csx priorities. A metal
primitive below a higher-priority material does not short the edge.

**Speed.** We do not query `property_at` for every cell. The compile step iterates
the primitives:
1. Map each primitive's bounding box to an index range on the grid.
2. Inside-test only those cells/edges.
3. Keep a per-cell "winning priority" buffer, so the highest priority wins.
   Equal priority goes to the later primitive, matching csx's documented
   tie-break.

The x-slabs are processed in parallel. This needs csx primitives/properties to
be `Sync` (§8).

### 5.4 Time step
- `dt = S / (c·sqrt(1/Δx_min² + 1/Δy_min² + 1/Δz_min²))`, with `S = 0.99`.
  Here `c` is the vacuum speed of light. Materials with ε_r, μ_r ≥ 1 only make
  the bound more conservative.
- Users may override `dt`, but the override is validated.
- The openEMS `<FDTD>` time-step factor maps onto `S`.
- openEMS files (`Fdtd::from_csx`) use openEMS's own criteria instead
  (`TimeStepMethod`, `engine::timestep`, §5.16): Rennings (default) or a
  per-edge Gershgorin bound (method 1), on the operator with materials,
  factor 1 by default, and shortened for a sinusoid so that its period is
  an integer number of steps; evaluated with openEMS's own arithmetic and
  constants, so that the step is openEMS's bit for bit (§5.16.10).

### 5.5 Lumped elements and ports
**LumpedElement (R)** over a box with `M` series edges along its direction and
`N` parallel columns:
- Per-edge conductance `g = M/(N·R)`, added to `G`.
- A parallel C is folded into the edge's `C` (`C_e = M·C/N`); R and C are pure
  coefficient changes, composed with the material's existing C/G recovered
  from `ca`/`cb`.
- Inductors and series branches (any of R+L+C in series, or L in parallel)
  use one auxiliary branch state per edge, updated trapezoidally and solved
  implicitly with the edge voltage. This adds no stability limit.
  Series vs. parallel follows the csx property's documented semantics.
- Branch energy (L/C states) is not yet included in `Engine::energy()`.
- openEMS files (§5.16.1): parallel inductors use an explicit (leapfrog)
  update instead (`LumpedElement::explicit_inductor`), parallel R/C
  elements are spread like a material over edges of different sizes, an
  excitation over an element is openEMS's plain soft source with the
  branch driven without it (`LumpedElement::branch_drive_source`), and a
  parallel element with an inductor but no capacitor limits its edges'
  capacitance to `min(1/(20·ω·R_e), 1/(20·ω²·L_e))`.

**Lumped port** = LumpedElement(R) + Excitation + voltage ProbeBox + current
ProbeBox, emitted into csx by the helper.
- The engine treats the excitation on a resistive edge as a Thevenin source:
  `u^{n+1} = ca·u^n + cb·(Σi + g·V_s^{n+1/2}/M)`.
- This is exactly what you get when an E-field excitation overlaps a lumped
  resistor, so XML-sourced ports behave identically.
- **Implemented:** `Simulation` maps csx `LumpedElement`s to edge-list
  `engine::LumpedElement`s (per-edge values from `csx::discretize`) and turns
  a soft E excitation on a resistive lumped edge into that element's
  Thevenin source (`V_e = E·l` per edge). `ems::port::LumpedPort` emits the
  four csx properties; `tests/sim_operators.rs` checks it against the
  engine-level `engine::LumpedPort`.

**Port analysis** (pure post-processing over probe results):
- V and I DFTs; the current is corrected for its half-step time offset.
- Reference plane: `I` is the Thevenin branch current (the port
  resistor's terminal current, recorded by the lumped element), not the H
  loop around the port edges. Both V and I then sit at the port edges and
  the port cell's own capacitance belongs to the structure: a matched TEM
  line reads `Z/cos(K/2)`, `|S11| = tan²(K/4)` (second order), instead of
  `Z·e^{jK/2}`, `|S11| = tan(K/4)` with the loop current
  (`tests/validation_port.rs`). The `port_it<n>` ProbeBox stays an H-loop
  probe (openEMS semantics), so a hand analysis of written results must use
  the element current instead: the `run_xml` example writes it as
  `port_resist_<n>.csv` (`I = −i`), next to `port_ut<n>.csv`.
- Power waves with reference `Z_ref` (Pozar) give `S_ij`, `Z_in`, and
  accepted/incident power. `Z_ref` may vary with frequency
  (`PortResult::new_per_frequency`), for transmission-line ports whose
  reference is the extracted line impedance.

**Microstrip port** (`ems::port::MslPort`, M2): a csx-emitting helper
like `LumpedPort`. It writes the strip over the port length, an optional
feed at the start (a soft E excitation from strip to ground, optionally
over a feed resistor, which makes it a Thevenin source), three voltage
probes at equally spaced mesh lines and two current probes (H loops around
the strip) on the dual planes between them. With the line written as
`V(s) = A·e^{−γs} + B·e^{γs}`, `I(s) = (A·e^{−γs} − B·e^{γs})/Z_L`
(propagation coordinate `s`, spacing `d`):

```text
sinh²(γd/2) = (V1 + V3 − 2·V2) / (4·V2)
Z_L         = −(V3 − V1) / (2·sinh(γd/2)·(Ia + Ib))
I(s2)       = (Ia + Ib) / (2·cosh(γd/2))
```

then `A`, `B` at the middle probe are moved to the reference plane with
`e^{∓γΔs}`. These relations are exact for any exponential mode sampled at
those points, so on a uniform mesh section they reproduce the discrete Yee
line exactly (its `γ` includes the numerical dispersion, its `Z_L` is the
discrete wave impedance): `tests/validation_msl.rs` recovers `Z = η·h/w`
and the 1-D Yee wavenumber of a PMC-walled parallel-plate line to the f32
round-off of the fields (which enters `β` as `~1e-8/(βd)²`). Design
choices and limits:
- The mesh must exist before `add_to` (probe positions are mesh lines);
  three **equally spaced** lines are required (non-uniform spacing breaks
  the exactness), and `add_to` fails otherwise.
- Strip edges need not be mesh lines. With lines on the edges a
  zero-thickness strip acts wider (Z ≈ −8 % for w/h ≈ 2 at w/6 cells); with
  the thirds rule the microstrip test is within 0.7 % (Z) and 0.1 %
  (ε_eff) of Hammerstad.
- The feed's near field (radiation and surface waves in an open structure,
  below-cutoff box modes in a shielded one) decays slowly: probes 5·h from
  the feed read ε_eff 5 % high, at ≈ 11·h within 1 %. The port should be
  long (≥ 10 substrate heights between feed and probes).
- `β` is ill-conditioned where `V2` has a standing-wave node.

**Waveguide ports** (`ems::port::RectWaveguidePort`, TE_mn of a rectangular
guide). The helper emits only csx properties, so the port round-trips
through XML:
- A soft E `Excitation` over the excitation plane, with the transverse
  mode function as its csx weighting functions.
- A `WaveguideVoltage` and a `WaveguideCurrent` `ProbeBox` over the
  reference plane, with the mode functions `e` and `h = d̂ × e` as the
  csx user attributes `ModeFunctionX/Y/Z` (expressions in drawing-unit
  coordinates, values in 1/m, written as numbers: openEMS's parser has
  no `pi`), plus `emsModeSampling="exact"`, which selects ems's exact
  overlap integrals; without it a waveguide probe samples like openEMS
  (§5.16, `compile/mode.rs`), which openEMS itself always does.
- The mode functions are normalised to `∫|e|² dA = 1`, so a single wave
  has `V = ∫E_t·e dA`, `I = ∫H_t·h dA` and `V/I = Z_TE = ωμ/β`.

The compile step turns each waveguide probe into a weighted sum of edge
values (`src/compile/mode.rs`). Tangential E and H of a transverse Yee
plane share their transverse positions, so V/I of one mode has no
transverse sampling error. H lies half a cell off the plane and is
interpolated linearly from the two adjacent dual planes, which adds an
`O((βΔ)²)` error.

The analysis uses `PortResult::with_z_refs` with `Z_ref(f) = Z_TE(f)`.
The probe and the soft-source plane must differ: the source's sheet
current makes H discontinuous at its own plane. The guide behind the
source must be absorbing.

In `tests/validation_waveguide.rs` (1 mm cells, 20 × 10 mm guide):
- A matched guide gives `|S11|` of −54 to −63 dB and `|S21| = 1 ± 6e-4`
  over 9–12 GHz.
- β from `arg S21` agrees with the exact Yee dispersion relation to about
  1e-4. The remaining 0.1% from the analytic value is the scheme's
  dispersion.
- A short gives `|S11| = 1 ± 1e-3` with the right round-trip phase.

**Curve port** (`ems::port::CurvePort`): a lumped port whose terminals do
not share a mesh line, joined by a polyline (the parity item for openEMS's
"curve" port helper, designed here from what such a port must do):
- The vertices are snapped to the nearest mesh nodes. The **feed edge** is
  the first cell of the curve (from the first node, one cell along the
  axis on which the first segment advances most). On it the port is
  exactly a `LumpedPort` (resistor, Thevenin excitation, voltage probe,
  element current, same names); the rest of the polyline is a PEC csx
  `Curve` (staircased by csx through the nearest nodes) or, with
  `wire_radius`, a csx `Wire` (thin-wire / oblique-wire model, §5.14).
- The curve carries no voltage, so `V` along the whole curve equals the
  feed-edge voltage and the current is the series current through it; the
  reference plane is the feed edge and the curve's inductance belongs to
  the structure. A resistor spread over the staircase was rejected: csx
  lumped elements act on one direction per box, and the series
  connection is equivalent at the terminals while keeping the
  second-order `LumpedPort` reference plane.
- Validation (`tests/validation_curve.rs`, 1 mm cells, PEC box 16 × 8 ×
  8 mm): a loop over the ground into a 100 Ω load reads `Re Z_in` = 100.07
  / 100.27 / 101.07 Ω at 0.1 / 0.2 / 0.4 GHz (the rise is the loop's stray
  capacitance, `≈ 2ω²LC`) and a constant `L` = 5.61 nH (straight runs),
  5.06 nH (with a staircased diagonal). As an oblique thin wire of
  `r0` = Δ/10 and Δ/20 the loop gains 2.10 / 4.04 nH, vs the wire-over-
  ground estimate `(μ0/2π)·ln(r_eq/r0)·ℓ` = 1.71 / 3.44 nH (+23 / +17 %;
  ratio 1.92 vs 2.01). The port is bit-identical to a hand-built
  `LumpedPort` plus `Curve`.
- Limit: an all-axis-aligned polyline with `wire_radius` becomes separate
  straight thin wires meeting at the bends (warned); their end corrections
  overestimate a bent wire's inductance (9.3 nH instead of 5.6 nH at
  `r0 ≈ r_eq`), so the plain `Curve` is preferable there.

**Coplanar waveguide port** (`ems::port::CpwPort`). A csx-emitting line
port with the analysis of the microstrip port (shared in `port::line`):
three equally spaced strip-to-ground voltages, two H loops, `γ` and `Z_L`
from the exact exponential-mode relations, waves moved to the reference
plane. Geometry and design choices:
- The box spans the whole metal along the width axis. The centre strip `S`
  is centred, the gaps `W` sit on both sides, and the grounds fill the
  rest. All three are zero-thickness boxes on the metal plane
  (`stop[height]`) over the port length. `start[height]` is the far side
  of the substrate. The substrate and an optional backing ground are part
  of the model.
- Feed: two line excitations across the gaps at the feed plane, directed
  from the strip to each ground (`amplitude/W` V/m): `port_excite_<n>A`
  (`+width` gap) and `B`. With `feed_resistance(R)` there are two `2R`
  line resistors `port_resist_<n>A/B` across the gaps, i.e. a Thevenin
  source of `amplitude` behind `R`. A line primitive covers the edges
  whose midpoints lie in the gap, which are exactly the edges the metal
  boxes leave free, so the feed matches the metal with or without lines on
  the edges.
- Voltages run from the strip centre line to the first line on the
  `+width` ground. The loops go around the centre strip only.
- Reference (`CpwPort::quasi_static`, `quasi_static_of`): Wen (1969) and
  Simons (2001) conformal mapping, zero thickness, infinite grounds,
  `K(k)` by the AGM (`port::elliptic`). Conventional CPW:
  `k1 = sinh(πS/4h)/sinh(π(S+2W)/4h)`,
  `ε_eff = 1 + (ε_r−1)/2·K(k1)K(k0')/(K(k1')K(k0))`,
  `Z0 = (η0/4)/√ε_eff·K(k0')/K(k0)`. Conductor-backed:
  `k3 = tanh(πS/4h)/tanh(π(S+2W)/4h)`, `q = K(k3)K(k0')/(K(k3')K(k0))`,
  `ε_eff = (1+ε_r·q)/(1+q)`,
  `Z0 = (η0/2)/√ε_eff/(K(k0)/K(k0') + K(k3)/K(k3'))`.
- Limits. The feed's near field (the cut-off modes of the shield or box
  around the line) decays with the lowest cutoff of the cross section. In
  a 6 mm wide box it decays with `α ≈ 0.57/mm`: probes 7.5 / 10.5 /
  12.5 / 15 mm from the feed read `β` +0.9 / +0.17 / +0.05 / +0.02 %
  high. A CPML 4 mm above an open CPW (graded cells) gave ±1 % ripples
  in `Z_L` and `−35 dB` S11, from the line's near field reaching the
  layer; at 8 mm they are gone (−66 dB). The port is not available on
  cylindrical grids.

`tests/validation_cpw.rs` (`S` = 1 mm, `W` = 0.5 mm, grounds to PEC side
walls at `|y|` = 3 mm, thirds rule at all four edges with `dy` = 0.214 mm,
probes 15 mm from the feed, through line between two ports):

| case | result |
|---|---|
| air, PEC covers at `z = ±1` mm (exact TEM; reference `(η0/4)·K(k3')/K(k3)`), 1 mm cells along the line | `Z_L` = 92.74 Ω vs 93.22 Ω (−0.51 %), `β` = 1-D Yee wavenumber within 1.5e-4, `|S11|` ≤ −68 dB; the passive port's `Z_L` is constant to 1e-6 (debug 16 s) |
| conductor-backed, ε_r = 4.4, h = 0.5 mm, CPML 8 mm above (ignored) | `Z_L` −0.47 %, `ε_eff` −1.0 % vs Simons (1–3 GHz), `|S11|` ≤ −66 dB |
| conventional, same substrate, CPML 8 mm above and below (ignored) | `Z_L` −1.0 %, `ε_eff` +0.7 % vs Wen/Simons, `|S11|` ≤ −60 dB |

**Stripline port** (`ems::port::StriplinePort`). It works like the
microstrip port, but the grounds are at `start[height]` and
`stop[height]` (part of the model) and the strip is on the mid-plane,
which must be a mesh line.
- Feed: two boxes from the strip to each ground, with excitations
  `port_excite_<n>A/B` (`amplitude/(b/2)` V/m, pointing away from the
  strip) and optional `2R` resistors `port_resist_<n>A/B` (a source `R`).
- Voltage from the strip to the lower ground. The loops go around the
  strip.
- Reference (`StriplinePort::z0`, `z0_of`): Cohn's (1954) exact
  zero-thickness result `Z0 = η0/(4√ε_r)·K(k)/K(k')`, `k = sech(πw/2b)`,
  `k' = tanh(πw/2b)` (both computed directly, and the ratio as
  `AGM(1,k)/AGM(1,k')`). For a strip of finite thickness, Wheeler's
  formula is the usual closed form. The port's strip is a sheet, so it is
  not used.

`tests/validation_stripline.rs`:

| case | result |
|---|---|
| strip over the full width between PMC side walls (two parallel-plate halves, `Z = η0·b/(4w√ε_r)` exactly) | `Z` and `β` (1-D Yee) to round-off (≤ 0.36 of the `1e-6 + 5e-8/(βd)²` tolerance), `|S11|` −85 dB |
| `w/b` = 1, ε_r = 2.2, `b` = 2 mm, half model (PMC through the strip), thirds rule, `Δ` = 0.5 / 0.25 / 0.125 mm across the strip (`dz` = 0.25 / 0.25 / 0.125) | `Z` +0.18 / −1.18 / −0.57 % vs Cohn (44.06 Ω), `ε_eff` = 2.201, `|S11|` ≤ −66 dB. The 0.5 mm value benefits from a cancellation; the refined meshes converge from below (fast test: 0.5 mm, 2 s in debug) |

**Coaxial port** (`ems::port::CoaxPort`, inner radius `a`, outer radius
`b`, TEM `Z0 = η0/(2π√ε_r)·ln(b/a)`, Pozar §2.2). Same line analysis as
the MSL port (shared code in `src/port/line.rs`): three radial voltage
probes at equally spaced mesh lines, two H loops around the inner
conductor on the dual planes between them, `γ` and `Z_L` extracted,
waves moved to the reference plane. Design choices:
- The port writes the conductors over its length (plus an optional
  `feed_extension` behind the feed), optionally a shield (`b` to
  `r_shield`) and a dielectric filling. The feed is a soft E excitation
  over the annulus with the TEM profile `E_ρ = V/(ρ·ln(b/a))` (csx
  weighting functions masked with `if()` to `a < ρ < b`); without a feed
  resistor the line behind the feed must end in a PML.
- **Cylindrical grids** (`dir = z`, axis at `ρ = 0`): conductors are
  `(ρ, α, z)` boxes, or the `ρ_min`/`ρ_max` PEC faces; a radial feed
  resistor over the full circle is available (Thevenin source with the
  excitation). The current loop box runs from the first `ρ` line to the
  last line `≤ a` over **all** `α` lines: on a closed azimuth the `H_α`
  edges on `α = 0` and `α = 2π` are two stored copies, each with half of
  the dual length (`Grid::dual_delta` is halved at the ends), so both are
  needed (leaving one out read `Z_L` 6.7 % high, i.e. `1/(2N)` of the
  current missing). The loop's radial sides do not cancel at the seam,
  but they lie in or on the inner conductor, where `H_ρ = 0`.
- **Cartesian grids** (any `dir`): the voltage path runs from the node
  nearest the axis to the first line at `ρ ≥ b` (or the middle of the
  shield); the current loop is the smallest box of lines containing every
  node within `a` of the axis, and must fit inside `b`. The conductors are
  written as **whole mesh cells** (cells whose centre is inside), not as
  csx `Cylinder`s: csx marks an edge PEC when its centre is in metal, so a
  curved primitive leaves nodes at the conductor potential with a free
  longitudinal edge, which is incompatible with a TEM field. With a
  `Cylinder` the measured line had `ε_eff` 12 % high and `Z` 3 % high
  (frequency independent); with whole cells it is exactly TEM.
- Limits: the staircase acts like conductor surfaces about `0.2·Δ`
  outside the nominal radii (`Z_L` −5.9 % at 6 cells per inner radius,
  −3.4 % at 12; first order). The feed's near field decays over a few
  `b − a`: `β` off by 7e-4 with the middle probe 7 mm (5.4·(b−a)) past the
  feed, 4e-5 at 10 mm; keep about `8·(b − a)`. Sectors (open azimuth)
  measure the sector's current (the impedance of the sector line).

`tests/validation_coax.rs` (ε_r = 2, `a` = 1 mm, `b` = 2.3 mm, through
line of two ports, PML ends, 2–8 GHz):

| case | result |
|---|---|
| cylindrical, `Δρ` = 0.1 mm, 8 α cells, PEC `ρ` faces as conductors | `Z_L` within 7.2e-5 of the discrete TEM value `η/(2π)·Σ Δρ/ρ_{i+½}` (4.4e-4 of `Z0` = 35.31 Ω), `β` within 3.7e-5 of the 1-D Yee wavenumber, `|S11|` ≤ −80.6 dB, `|S21 − e^{−jβL}|` ≤ 8e-5 |
| cylindrical, port-written inner conductor and shield, 35 Ω Thevenin feed resistor | `Z_L` within 1.1e-3 of discrete, `|S11|` ≤ −59.9 dB |
| Cartesian staircase, `a` = 1.5 mm, `b` = 3.45 mm, 0.25 mm cells, shield to 4 mm | `ε_eff` = 2.0002–2.0022 (Yee dispersion), `Z_L` = 33.21 Ω vs `Z0` = 35.31 Ω (−5.9 %), `|S11|` ≤ −86 dB |

**Circular waveguide port** (`ems::port::CircWaveguidePort`, TE_nm or
TM_nm, `n` azimuthal, `m` radial). Same structure as the rectangular port
(soft mode-shaped E excitation, `WaveguideVoltage` / `WaveguideCurrent`
probes, `Z_ref(f)` = the wave impedance), so it round-trips through XML.
- Mode functions (Pozar §3.4, Marcuvitz §2.3), `ρ` from the guide axis,
  `φ0` a polarisation angle:
  `TE: e = N·∇ψ × ẑ`, `ψ = J_n(k_cρ)·sin(n(φ−φ0))` (`J_0(k_cρ)` for
  `n = 0`); `TM: e = N·∇ψ`, `ψ = J_n(k_cρ)·cos(n(φ−φ0))`; `h = d̂ × e`.
  `φ0 = 0` puts TE11's centre field along the first transverse axis.
  `k_c = p'_nm/a` (TE) or `p_nm/a` (TM).
- Normalisation `∫|e|² dA = 1` in closed form: `∫|∇ψ|² = k_c²∫ψ²`
  (Green, with the Neumann/Dirichlet wall condition) and
  `∫₀^a J_n²ρ dρ = a²/2·(1 − n²/p'²)·J_n(p')²` (TE), `a²/2·J_n'(p)²`
  (TM); `π` or `2π` (`n = 0`) for the azimuth. The unit test integrates
  the written expressions numerically (to 1e-3, the quadrature).
- Bessel functions: `J_n` from Bessel's integral by the trapezoidal rule
  (exponentially convergent for the periodic integrand), the zeros of
  `J_n` and `J_n'` by scan and bisection (`port/bessel.rs`, checked
  against A&S tables 9.5/9.7 to 1e-9 and against csx's `jn`). The csx
  expressions use `jn(n, x)` and `J_n' = (J_{n−1} − J_{n+1})/2`; the
  `J_n/ρ` term has its limit written at `ρ = 0`.
- Cartesian grids: any propagation axis, expressions in the offsets from
  the centre, masked to `ρ ≤ a` with `if()`; the wall is part of the
  model (PEC, staircased or conformal). Cylindrical grids: the guide on
  the axis along `z`; the expressions use `rho` and `a` and give the
  `(ρ, α)` components, the probe weights carry the `ρ·Δα` metric
  (`compile/mode.rs`); the wall can be the `ρ_max` PEC face. The azimuth
  is closed, or a sector whose α faces are symmetry planes of the mode
  (a half or quarter model): on each face the mode's tangential E
  vanishes (`cos(n(α − φ0)) = 0`, or TE0m: a PEC wall) or its tangential
  H does (`sin(n(α − φ0)) = 0`, or TM0m: a PMC wall), reported by
  `sector_walls`; the mode is normalised over the sector, so V, I and the
  power are the modelled part's and the S-parameters the full guide's
  (TE11 half guide with PEC walls, `φ0 = π/2`: the full 8 × 12 guide's
  S-parameters to 1.4e-8).
- Limits: the reference impedance and `β` are the analytic ones, so a
  discretised cross section whose cutoff differs from `p/a` (azimuthal
  sampling, staircased wall) is seen as a mismatch: a cutoff error `δ`
  gives `|S11| ≈ |δ|·k_c²/(2β²)`. The `radius` argument can be tuned to
  the effective radius of a staircased wall. Inside an azimuthal
  multigrid zone the mode probes are not valid (§5.15.1; warning).

Validation (`tests/validation_circ_waveguide.rs`; `a` = 10 mm, TE11
`f_c` = 8.785 GHz, Δz = 1 mm, ports 32–40 mm apart, CPML(8) ends). The
effective cutoff `k_c,eff` is taken from the measured `β` with the Yee
z dispersion, `sin²(ωΔt/2)/(cΔt/2)² − sin²(βΔz/2)/(Δz/2)²`; its spread
over frequency measures the port, its offset the cross section.

| case (10–13 GHz unless noted) | worst `|S11|` | `|S21| − 1` | `k_c,eff/k_c − 1` |
|---|---|---|---|
| TE11, cylindrical 8 ρ × 12 α cells (fast test, 17 s debug) | −37.8 dB | ≤ 1.4e-3 | −0.79 % ± 0.01 % |
| TE11, cylindrical 10 × 16 | −46.1 dB | ≤ 1.4e-3 | −0.44 % ± 0.004 % |
| TE11, cylindrical 20 × 32 | −49.2 dB | ≤ 5e-4 | −0.11 % ± 0.002 % |
| TM01, cylindrical 20 × 16, 13–16 GHz | −48.6 dB | ≤ 1e-4 | −0.071 % ± 0.000 % |
| TE01, cylindrical 20 × 16, 20–23 GHz | −47.3 dB | ≤ 4e-4 | −0.15 % ± 0.001 % |
| TE11 short, cylindrical 20 × 32 | `|S11| = 1 ± 2e-4` | | β within 0.25–0.41 % of analytic (the cutoff offset) |
| TE11, Cartesian 0.5 mm, staircased wall | −28.3 dB | ≤ 1e-3 | −3.6 % ± 0.48 % |
| same, port radius tuned to `a·k_c/k_c,eff` = 10.372 mm | −42.1 dB | ≤ 1e-3 | +0.001 % ± 0.50 % (vs the tuned `k_c`) |

On the cylindrical grid the TE11 cutoff error halves with each halving of
the cells (azimuthal sampling, `(2/Δα)·sin(Δα/2)`, and the radial cells),
and the spread of `k_c,eff` is ≤ 1e-4: the port extracts a single mode
cleanly. On the staircased Cartesian wall the discrete cutoff is 3.6 % low
and the analytic `Z_TE` is then a 5 % mismatch near cutoff, which is the
`|S11|` reported; a port radius tuned to the measured cutoff brings it to
−42 dB. The measured `β` is the same in both runs: the ±0.5 % drift of
`k_c,eff` over the band belongs to the staircased guide, not to the port.
The conformal wall (§5.13) on the same guide (PEC block, vacuum cylinder,
CPML ends through the cut cells) first did not decay (87 % of the peak
after 30 000 steps, ~20× slower): the CPML read the cut edges' `C/f_e` as
a huge refractive index and switched itself off, and open edges on the
wall held a local resonance. Both are fixed (§5.13, *Absorbing layers*);
a coarse version is the regression test `circular_guide_with_pml_decays`
(`tests/conformal_modes.rs`).

**Radial-line port** (`ems::port::RadialLinePort`, cylindrical grids):
a parallel-plate radial waveguide between two `z` planes `h` apart, the
TM₀₀ mode (`E_z` uniform across the gap and in α, `H_α`) as a
non-uniform line, `dV/dρ = −jωμ·h/(Δα·ρ)·I`, `dI/dρ = −jωε·Δα·ρ/h·V`,
solved by `V = A·H₀⁽²⁾(kρ) + B·H₀⁽¹⁾(kρ)`, `I = −j(Δα·ρ/(ηh))·(A·H₁⁽²⁾ +
B·H₁⁽¹⁾)` (Marcuvitz §2.6; `Δα` the grid's azimuth span). Design:
- Feed ring at `r_feed` (soft `E_z` excitation and/or a `z` resistor
  over every α column), three voltage lines at equally spaced ρ lines
  past the feed, two current rings on the dual radii between them:
  flat current boxes normal to z from the first ρ line over the whole α
  range (the total current through the disc, which by Ampère is the
  radial current in the `+` plate), a quarter cell into the cell nearest
  the middle of the gap, which ems (layer of the nearest node) and
  openEMS (nearest dual plane) read alike. Outward ports
  (`r_end > r_feed`, incident `H⁽²⁾`) and inward ones (incident `H⁽¹⁾`).
- Extraction: `A`, `B` fitted to the five probe values (least squares,
  currents scaled by `η·h/(Δα·ρ)`) with the discrete wavenumber along
  the probe spacing `d`, `k = (2/d)·asin(Ω·√(με)·d/2)`, `Ω` the
  time-discrete frequency; the fit residual is reported (TM₀₀ purity).
  Reported at the reference radius: `V`, `I`, `V_inc`, `V_ref`, the
  travelling-wave `Γ = V_ref/V_inc`, the incident wave impedance
  `Z₊ = j·η·h/(Δα·ρ)·H₀⁽²⁾/H₁⁽²⁾` (complex: Kurokawa power waves with it
  would not vanish for a matched line), and the pseudo-waves
  `a, b = V_inc, V_ref·√Re(1/Z₊)` with `½(|a|² − |b|²) = ½·Re(V·I*)`
  exactly (the two cylindrical waves carry no cross power), so
  `b/a = Γ`. A real reference impedance gives an ordinary `PortResult`.
- `Y₀`, `Y₁` (new in `port/bessel.rs`): A&S 9.1.22 integral, composite
  16-point Gauss–Legendre; A&S table 9.1 to 1e-12, Wronskian to 1e-11
  over 0.01…120.
- Sector grids: PMC α walls (TM₀₀'s `E_z` lives on them). The wall
  columns have half-width dual cells, so a feed resistor spread equally
  over the columns drives the wall columns twice as hard: the feed is
  then not uniform in α, and its azimuthal near field (`∝ (r_f/ρ)^ν`)
  reached the probes at −45 dB (fit residual 1e-3). The port therefore
  splits the resistor: wall columns `2N·R` each, the inner ones
  `N/(N−1)·R` (uniform α cells required). (`CoaxPort`'s radial feed
  resistor on sectors has the same column weighting; not changed.)

`tests/validation_radial.rs` (plates = PEC z faces 2 mm apart, two
cells; `Δρ` = 0.5 mm; 4–16 GHz; analytic Hankel functions from the
ascending series, independent of the port's quadrature):

| case | result |
|---|---|
| matched: outward from 2 mm (PMC `ρ_min` = open end at the feed), reference 20 mm, cylindrical PML on `ρ_max` | `|Γ|` ≤ −59.4 dB (−71…−77 dB above 4 GHz), `V/I` within 2.1e-3 of the analytic `Z₊`, fit residual ≤ 3.9e-5 |
| short at 30 mm (PEC `ρ_max`): `Γ = −H₀⁽²⁾(kb)H₀⁽¹⁾(kr)/(H₀⁽¹⁾(kb)H₀⁽²⁾(kr))` | `||Γ| − 1|` ≤ 4.6e-3, phase within 2.8e-3 rad |
| quarter sector, PMC α walls, split feed resistor | `|Γ|` ≤ −59.4 dB (−45 dB with one resistor over all columns) |
| inward from a ring at 30 mm to 12 mm, grid with the axis: `Γ = H₀⁽²⁾(kr)/H₀⁽¹⁾(kr)` | `||Γ| − 1|` ≤ 3.2e-4, phase within 1.7e-3 rad |

openEMS runs the port's properties (black box, `cyl_radial_port`: the
quarter-sector short with the split resistor, read from the file):
probes within 7.8e-7…1.4e-6 of ems. It needed one more piece of
openEMS's PMC (§5.15): openEMS also never updates the tangential E on a
PMC face (its operator dump has `vv = vi = 0` there), so its current
rings, which include the wall lines, read 3/4 of ems's (0.33 relative)
until the file path did the same.

**Microstrip on cylindrical grids** (`MslPort`). openEMS's engine has no
port concept: its scripts' MSL, CPW and stripline ports are boxes
(excitation, lumped element, probes) that on a cylindrical grid become
`(ρ, α, z)` boxes and run as such (as the radial port's do). The line
analysis needs a line uniform along its propagation axis, so `MslPort`
accepts on cylindrical grids the strip along `z` above a ρ ground (on a
cylindrical substrate, width an α range), along α above a ρ ground
(width a z range) or along α above a z ground (width a ρ range), and
refuses lines along ρ (not uniform) and grounds normal to α. Along α,
positions are angles and `γ` is per metre of arc at the middle of the
strip's cross-section (`Z_L` and the waves do not depend on it).
`tests/validation_msl_cyl.rs`: along z with the strip around the whole
azimuth (a coax: ground = PEC `ρ_min` face, strip on the `ρ_max` face,
ε_r = 2) `Z_L` within 1.0e-5 of the discrete coax value, `β` within
1.0e-5 of the Yee wavenumber, `|S11|` ≤ −85.5 dB; along α at
`R` = 200 mm (full-width strip, h = 1 mm, w = 2 mm between PMC ρ walls,
CPML α faces) `Z_L` within 1.4e-4 of `η0·h/w`, `β` within 4.8e-5,
`|S11|` ≤ −85.7 dB.

**CPW and strip line on cylindrical grids** (wave 14; `CpwPort`,
`StriplinePort`, shared check `port::line::cylindrical_line`). The same
uniform lines as `MslPort`: along z with the height along ρ (metal on a
`ρ` surface, widths and gaps angles: arcs at the metal or strip radius,
which also scale the CPW gap excitation `amplitude/W`), along α with the
height along ρ (width along z) or along z (width along ρ); lines along ρ
and heights along α are refused. Along α the line coordinate is the arc
at the metal radius (height along ρ) or at the centre of the box's ρ
range (height along z). The quasi-static references (`quasi_static`,
`z0`) are the flat ones with the arc lengths. Validation
(`tests/validation_lines_cyl.rs`), against discrete references:

- *strip line along z between two coaxial grounds*, the strip a full
  cylinder at the mid radius: two discrete coaxes in parallel,
  `Z = Z_in·Z_out/(Z_in + Z_out)` with `η/(2π)·Σ Δρ/ρ_{i+½}`: `Z_L`
  within 7e-6, `β` within 7e-6 of the Yee wavenumber, `|S11|` ≤ −85 dB;
- *CPW along z on a cylinder* (`ρ` = 1.3 mm, strip 30° wide, 2.2-cell
  gaps, a shield `ρ` 1 … 1.6 mm with a PEC fin on the seam joining the
  grounds to both shield walls, so that one TEM mode exists): the TEM
  mode of a z-uniform Yee line is the discrete electrostatic solution
  of its cross section (edge capacitances `ε·(dual length)/(edge
  length)` in `(ρ, α)`), so the reference is `Z = 1/(c·C')` from a 2-D
  Laplace solve on the same mesh (conjugate gradients in the test):
  `Z_L` within 2.2e-5 of 53.53 Ω (the flat Wen formula with the arcs:
  117.5 Ω), `β` within 1.9e-5, `|S11|` ≤ −84 dB;
- *strip line along α* between two z planes at `R` = 200 mm (full width
  between PMC ρ walls): `η·b/(4w)` within 1.4e-4, `β` 6.6e-5;
- *CPW along α* on a z plane at `R` = 200 mm (a 5 × 2.5 mm PEC box): the
  flat discrete line of the same `(ρ, z)` cross section within 2.7e-4
  (curvature, `O((w/R)²)`), `β` 1.3e-5, `|S11|` ≤ −85 dB.

Two cautions found on the way, both native and not cylindrical: (1) the
probes must be past the decay of the feed's cut-off modes in the shield
(the first CPW attempt, a 10 mm box with probes 7.5 mm from the feed,
read 6 % off at the excited port and exact at the far one); (2) a
zero-thickness metal edge more than half a cell beyond the last mesh
line inside the metal makes the transverse mesh edge it cuts PEC (its
midpoint is in the metal) while the line beyond keeps its free edges
along the line: the metal border is no equipotential and the CPW is a
slow-wave line (`ε_eff` = 1.40 in air for 2-cell gaps, Cartesian and
cylindrical alike; `Z_L` 18 % high). `CpwPort`'s mesh requirements now
say so; the thirds rule is safe.

**Sectoral waveguide port** (wave 14; `SectorWaveguidePort`,
`port/sector_waveguide.rs`). A rectangular-waveguide port has no
uniform counterpart on a cylindrical grid except the guide whose cross
section is a circular sector (`ρ ≤ b`, the grid's α span `φ_s < 2π`,
PEC α faces and `ρ = b`), whose modes are those of a wedge
(Marcuvitz §2.5, Harrington §5-8): `H_z ∝ J_ν(k_c ρ)·cos(να)` (TE_mn) or
`E_z ∝ J_ν(k_c ρ)·sin(να)` (TM_mn) with the fractional order
`ν = mπ/φ_s` and `k_c·b` the n-th zero of `J_ν'` or `J_ν`. ems now
evaluates `J_ν` of real order (power series, A&S 9.1.10, with a Lanczos
`Γ`; the zeros by bracketing), writes the mode functions as csx
expressions (Horner polynomials in `(k_c ρ/2)²` times
`(k_c ρ/2)^{ν−1}`, regular on the axis for `ν ≥ 1`), normalises them
with the Lommel integrals over the sector and analyses the port like
`CircWaveguidePort`. Orders `0 < ν < 1` (openings wider than `mπ`, a
field singular at the axis) and closed azimuths (use
`CircWaveguidePort`) are refused. Validation
(`tests/validation_sector_waveguide.rs`, matched guide, CPML ends): a
120° sector, TE11 (`ν = 3/2`, `k_c·b = j'_{3/2,1}` = 2.460536, cutoff
11.75 GHz) on 8 × 8 cells: `|S11|` ≤ −42 dB, `|S21|` within 1.1e-3 of 1,
the effective cutoff of the discrete mode `−0.46 %` from `j'_{3/2,1}/b`
with a spread of ±0.012 % over 13–17 GHz (the port itself is clean; the
offset is the cross section's discretisation), `−0.12 %` on 16 × 16
cells (second order, ratio 3.7); a 72° sector, TM11 (`ν = 5/2`,
`j_{5/2,1}` = 5.763459, over-moded: TE11 and TE01 propagate) on 12 × 8
cells: `|S11|` ≤ −29 dB, `k_c,eff` −0.65 % ± 0.011 %. A curved
(bent) rectangular guide along α has modes of non-integer order in the
propagation constant (not a fixed cross-section problem) and is not
provided.

### 5.6 Boundaries
`Boundary::{Pec, Pmc, Mur{v_ph}, Pml(Cpml), Periodic}` per face, mirroring the
openEMS `<BoundaryCond>` choices (including `PML_n`); `Periodic` is ems-only.

- **PEC**: tangential E on the face is 0.
- **PMC**: tangential H is 0. In the integral form this needs no operator:
  the kernel updates the boundary-face tangential E with its half dual face,
  and H outside the domain reads as zero (implemented, `FaceCondition::Pmc`).
- **Mur 1st order**: one-way wave equation on the tangential E of the face.
  The engine builds that face as PEC; the operator saves values in `pre_e` and
  overwrites the face's tangential E in `post_e`. Without a phase velocity
  each edge is tuned to its local medium `c0/√(ε_r μ_r)` (openEMS's
  behaviour, §5.16; a vacuum-tuned face in `ε_r = 4` reflects 34–49 %).
- **CPML** (Roden & Gedney 2000):
  - Occupies the **outermost `n` cells of the mesh**. This matches openEMS
    `PML_n` semantics, so XML models mesh identically.
  - `ems::mesh::check` warns if structure primitives intrude into the PML.
  - Grading `m = 3` and `σ_max = 0.8·(m+1)/(n·Z0·Δ_local)`; κ_max and α_max
    are configurable. `n = √(ε̄_r·μ̄_r)` is the mean refractive index of each
    face's layer, measured from the engine coefficients (the stretching uses
    `ε0`, so the attenuation per cell is `n` times the vacuum one; one `n`
    per face keeps the stretch a function of the normal coordinate only).
    8 cells at normal incidence: −83.6 dB in vacuum, −85.7 dB in ε_r = 4 or
    12 (−80.5 / −73.4 dB with the vacuum `σ_max`).
  - The depth `ρ` is counted in cells (mesh index, half-integer for H), so on a
    graded PML mesh the per-cell attenuation `σ·Δ` keeps the uniform profile.
  - One operator (`engine::boundary::cpml::CpmlBoundary`) handles all CPML
    faces; it applies `cb·((1/κ − 1)·S + ψ)` corrections after the kernels and
    runs serially between the half-steps for now.
  - ψ arrays are stored only for the slabs; each ψ term handles one derivative
    direction, so edges and corners are correct.
  - On the `ρ_max` face of a cylindrical grid the same operator runs the
    cylindrical PML instead (§5.15.2).
- **openEMS's PML** (openEMS files only, `Fdtd::openems_run`;
  `engine::boundary::upml::OpenEmsPml`, §5.16.2): the `PML_n` faces of
  a file run the layer openEMS runs, Gedney's uniaxial PML with
  openEMS's fixed exponential grading, instead of the CPML, so that
  results match openEMS (−58 dB at normal incidence with 8 cells, against
  the CPML's −84 dB; openEMS's arithmetic bit for bit, §5.16.8). The native API keeps the CPML; with conformal PEC
  the CPML is used for files too (the conformal step couples to it).
- **AbsorbingBC sheets** (csx `AbsorbingBC`, implemented,
  `engine::boundary::abc::AbsorbingSheet`): the same first-order Mur update
  on the tangential E of a rectangle on an interior mesh plane, against the
  neighbour one node behind the outward normal. The file's
  `NormalSignPositive` is the sign of the opposite (inward) normal: `1`
  absorbs waves travelling along `−a`, as in openEMS, with which the
  probes then agree to 1e-6 (§5.16.3; MUR_1ST_SA gives openEMS's output
  of MUR_1ST bit for bit at normal incidence).
  - One-way: waves travelling along the outward normal are absorbed (−58 dB
    at normal incidence, the same as the Mur face on the same mesh). The
    region behind the sheet still exists: the sheet drives it like a hard
    source, so the transmitted wave continues there, and a wave arriving
    from behind sees a hard wall and is reflected.
  - `PhaseVelocity = 0` uses the local `c0/√(ε_r μ_r)` per edge (c0 in
    vacuum, csx's documented meaning). `MUR_1ST_SA` and unknown types fall
    back to plain first-order Mur with a warning.
  - Each primitive must snap to a flat rectangle; volumes, lines and points
    are ignored with a warning. Sheets are registered last, after dispersive
    media and the Mur faces.
- **Periodic and Bloch**: see §5.6.1 (implemented).

#### 5.6.1 Periodic and Bloch boundaries

`Boundary::Periodic` on both faces of an axis (`FaceCondition::Periodic` in
the engine; one-sided use is rejected). The period is the domain length
along the axis, first to last mesh line, so node `n − 1` is identified with
node `0`.

- **No kernel change.** The two outer node planes are two *copies* of one
  plane. Each copy of a tangential E edge is updated by the unchanged
  kernel over its half dual face, exactly as on a PMC face (out-of-domain H
  reads as zero). The two half faces together are the full dual face. With
  `w = 1/cb = C/dt + G/2` of each half, the update of the merged edge,
  `(C₀ + C₁)(u' − u)/dt + (G₀ + G₁)(u' + u)/2 = ∮₀ i + ∮₁ i`, is exactly
  the weighted mean `u' = (w₀u'₀ + w₁u'₁)/(w₀ + w₁)` of the two half
  updates. The engine writes that mean to both copies after every E
  update. A PEC copy (`cb = 0`) makes the edge PEC. Edges where two
  periodic faces meet (four copies) are merged axis by axis with
  accumulated weights.
- **H copies** (dual edges normal to the face, half length each) need no
  merge. Both integrate the same Faraday equation of one primary face,
  `L_k di_k/dt = −∮u`, so `i_k/l_k` is the same H, and their sum is the full
  dual-edge current the merged E edges see.
- **Where it runs.** In the serial `post_e` gap (with x periodicity the two
  copies sit in different slabs), before the operators' `post_e` hooks so
  that probes, dumps and Mur faces see merged values. It runs once more
  after those hooks when any exist, to fold in their corrections. A
  correction proportional to `cb` (dispersive and lumped branch currents)
  is combined exactly. Sources on a face plane should cover both copies.
  CPML on the other faces runs in slab hooks before the merge and is
  linear in the half-face quantities, so periodic plus CPML (an open
  periodic structure) is exact.
- **Energy.** The scheme is the Yee scheme on the toroidal grid. The
  field sums count both copies, `C₀u² + C₁u² = (C₀ + C₁)u²`, so the
  invariant `W` is conserved (closed periodic box: `|W/W₀ − 1| < 3e-6` over
  10⁴ steps). Results are bit-identical for any thread count, and both
  copies are bit-equal.
- **Mesh check.** The neighbour-ratio check also compares the last cell
  with the first.
- **openEMS.** The XML format has no periodic boundary. `Fdtd::to_csx`
  writes such faces as PEC, and `Fdtd::to_csx_checked` fails instead.
- **Bloch / Floquet** (`engine::BlochPair`, engine level): the sine–cosine
  method (Taflove & Hagness, ch. 13). Two real engines carry the real and
  imaginary parts of a complex field with `F(x + L) = F(x)·e^{−jψ}`,
  `ψ = k_t·L`. They are built from one `EngineBuilder` with the periodic
  faces as PMC, each runs one step, and then the pair merges the copies
  with the phase: `U = (w_lo u_lo + w_hi u_hi e^{jψ})/(w_lo + w_hi)`,
  `u_lo = U`, `u_hi = U e^{−jψ}`. With `ψ = 0` this is the periodic merge
  bit for bit. The coupling is unitary, so `W_re + W_im` is conserved.
  Because the merge runs after all hooks of a step, `post_e` operators on
  the face planes see unmerged copies; keep them off the faces, and add
  coefficient-changing operators to both engines. `Simulation` drives a
  pair in its Bloch mode (§5.6.2).
- **Validation** (`tests/periodic.rs`):

  | Check | Result |
  |---|---|
  | Uniform sheet source, box periodic in x and y with CPML in z (graded mesh): spread across the plane | 1.6e-6 of peak |
  | Same case against the 1-D line | 1.7e-6 of peak |
  | Dielectric slab: \|Γ\| against the exact discrete line | 4e-5 |
  | Inhomogeneous cell, one period against two periods | 1e-6 of peak |
  | Periodic cavity resonances (k_x = 2πm/a) against the Yee dispersion relation | ≤ 5e-6 |
  | Same, graded mesh (Δ₀ ≠ Δ_last), against the continuum | ≤ 0.21% |
  | Bloch cavity, ψ = π/2: resonances at k_x = ψ/a and (ψ − 2π)/a against Yee | ≤ 6e-8 |
  | Bloch closed box: complex energy drift over 5000 steps | 1e-6 |

#### 5.6.2 Oblique incidence on periodic structures

The high-level form of the Bloch pair: a plane wave at a fixed transverse
wavevector `k_t` in a unit cell that is periodic along the transverse axes
and open (CPML) along the normal `n`.

- **Constant `k_t`.** The sine–cosine method fixes `k_t`, not the angle:
  frequency `f` travels at `sin θ(f) = |k_t|·v/(2πf)`, and frequencies
  below `f_c = |k_t|·v/(2π)` are evanescent. A user who wants angle `θ₀` at
  `f_d` sets `k_t = (2π f_d/v) sin θ₀` and reads each frequency at its own
  angle. Sweeping angles at one frequency takes one run per `k_t`.
- **Settings.** `Fdtd::bloch_wavevector(k)` (explicit), or implicitly an
  oblique csx `PlaneWave` excitation whose box spans the periodic cell (every
  periodic axis and all but one other axis completely). Then
  `k_t = (2π f_d/c)·k̂_t`, with `f_d` = `Fdtd::bloch_frequency` or else the
  plane wave's dispersion-matching frequency (the waveform centre). Several
  oblique plane waves must agree. An explicit `k_t` overrides the
  direction, with a warning when they differ. The openEMS format has no
  representation (`to_csx_checked` fails).
- **Two engines in `Simulation`.**
  - Both engines come from one `Compiled::builder()`, and both get
    `add_boundary_operators`, the lumped elements, probes and dumps.
  - Excitations and Thevenin sources drive the real engine only. The
    imaginary engine is excited through the faces and by its part of the
    plane wave.
  - The run loop (`engine::run::drive`, shared with `Engine::run_until`)
    steps both and merges, and the end criterion uses `W_re + W_im`.
  - Probe results carry `imag`, and their DFTs return `DFT[re] + j·DFT[im]`.
  - Frequency dumps are combined in place (`FreqDump::add_imaginary`).
    Time dumps keep the imaginary snapshots separately.
  - SAR dumps are skipped (warning) and NF2FF boxes refused.
  - Without Bloch settings nothing changes: one engine, bit-identical to
    before.
- **Source** (`engine::PeriodicPlaneWave`).
  - **Injection.** It is TF/SF on the two node planes `start`/`stop` of
    the box along `n`. The corrections are those of §5.11, restricted to the
    tangential E on the planes and the tangential H half a cell outside.
    A plane on the domain boundary gets none.
  - **Incident field.** It is `F̂(ζ, t)·e^{−j k_t·(r − r₀)}`. Every
    transverse derivative becomes `−j k̃` with the discrete wavenumber
    `k̃ = 2 sin(kΔ/2)/Δ` (exact for a uniform transverse mesh). That leaves
    a complex 1-D problem for all six components, which is the 3-D Yee
    scheme restricted to Bloch-phased fields.
  - **Auxiliary grid.** It reuses the 3-D mesh lines along `n` and the
    engine's `dt`, so the incident field is the discrete 3-D solution and
    leaks only at rounding level.
  - **Driving.** The auxiliary grid has a hard tangential-E source
    (`p·a(t)`) two cells upstream, and a 64-cell CPML along `ζ` at the far
    end. A plain matched-loss layer, the first attempt, reflected the
    oblique waves back into the incident field. That cost 1e-2 in |R| near
    cut-off and leaked −25 dB.
  - **Two engines.** Each engine runs its own copy of the (cheap) auxiliary
    grid and injects `Re` or `Im` of the phased field. The real engine's
    copy records the complex incident E on the entry plane
    (`Results::incident`), the reference for R and T.
- **Validation** (`tests/bloch.rs`; 4 × 2-cell cell, 0.5 mm cells, slab
  ε_r = 4 of 10 mm, 30° at 10 GHz, evaluated at 7–14 GHz, 45.6°–20.9°):

  | Check | Result |
  |---|---|
  | TE slab: \|R\|, \|T\| against the exact discrete (Yee) slab response | ≤ 5e-5 |
  | TM slab: same | ≤ 1.3e-4 |
  | TE / TM slab against Fresnel at the per-frequency angle (≤ 12.5 GHz; above, mesh dispersion) | ≤ 6.7e-3 |
  | Lossless slab: \|R\|² + \|T\|² − 1 | ≤ 3e-4 |
  | Empty cell: leakage into the scattered-field region (±z, k_t along x, y, oblique) | −92 dB |
  | Empty cell: \|T\| − 1 | ≤ 6e-5 |
  | k_t = 0: pair against a plain periodic engine | bit-identical, imaginary part 0 |
  | Normal incidence through `Simulation`: Bloch mode (k_t = 0) against the plain periodic TF/SF run | 6e-7 |
  | TM slab through `Simulation` (k_t derived from the plane wave) against discrete / Fresnel | 1.3e-4 / 6.7e-3 |
  | FSS (PEC patches, 3 mm period, TE 30°, below the first grating lobe): \|R\|² + \|T\|² − 1 | ≤ 3e-4 |
  | 1 vs 3 threads (engine and `Simulation`) | bit-identical |

  The exact discrete reference marches the 1-D recurrence of the
  Bloch-phased Yee equations through the slab (tangential edges on the
  slab faces carry the mean of the two cells).
- **Limits.** The background at the injection planes must be homogeneous
  and lossless. The phase reference is the first mesh line of every axis.
  On graded transverse meshes `k̃` uses the mean spacing, so the incident
  field is approximate. Measure R and T far enough from a patterned
  surface for the evanescent Floquet orders to have decayed, and below the
  first grating lobe if a single probe is to see only the specular order.

### 5.7 Waveforms
`Waveform` is an enum: Gaussian, Sinusoid, Dirac, Step, and
`Custom(Arc<dyn Fn(f64) -> f64 + Send + Sync>)`.
- **Gaussian**: centre `f0`, 20 dB half-bandwidth `fc`, delayed so it starts
  below 1e-6 of the peak.
- **Custom**: an expression string from XML becomes `Waveform::Expression`,
  evaluated by csx's `ExcitationSignal::evaluator` (variable `t`, plus `f0`
  and `f_max`), so values match csx exactly and the source text round-trips.
- **Sampling**: engine sources (`EdgeSource`, Rust models) are sampled
  at the centre of their update (soft E at `t^{n+1/2}`, see `SourceKind`);
  model (csx) excitations use openEMS's timing instead (§5.16.1): a soft
  E source added after the E update `u^n → u^{n+1}` gets the signal at
  `n·dt`, a soft H source added after `i^{n−½} → i^{n+½}` at `(n − ½)·dt`
  (confirmed against openEMS output to 1e-6). The update centres shift
  every signal by half a step, a 3–7 % L2 difference for a 5 GHz pulse at
  1 mm cells.
- **openEMS Gaussian**: `Waveform::OpenEmsGaussian` (`τ = 3/(2π fc)`,
  `T = 3τ`, cut at `2T`), what a file's Gaussian becomes (§5.16.1).
- **Spectrum**: `Waveform::spectrum(f)` is used to normalise frequency-domain
  results.

### 5.8 End criteria
- Energy is computed every K steps. The naive `W = ½ΣCu² + ½ΣLi²` (with u and
  i half a step apart) ripples by tens of percent over an oscillation, so the
  end criterion uses the staggered invariant `½ΣCu² + ½ΣL·i^{n−½}·i^{n+½}`
  (exactly conserved by lossless leapfrog; derivation in the `engine`
  module docs). It is computed inside the H half-step of every K-th step
  (`Engine::step_with_energy`), as per-x-plane partial sums combined in
  plane order, so it is bit-identical for any thread count. Measured drift
  in a lossless cavity: ~1e-6 relative over 1e5 steps; overhead ~1% at
  K = 50.
- The run stops when the excitation is over and `W/W_max <` threshold, or at
  `max_timesteps`.
- **Plane waves.** The TF/SF source injects real field energy: inside the
  box the fields are total fields, so `W` includes the incident wave while
  it crosses the box (the look-ahead misses the `post_h` corrections of the
  H edges just outside the box, an error of the order of the incident
  energy in one cell layer while the wave is on the surface). Once the
  pulse has left the box the incident field is zero everywhere and `W` is
  exactly the scattered-field energy. `Simulation::run_limits` therefore
  moves the excitation end to `Delay + duration + PlaneWave::transit_time`
  (far-corner `ξ` at the background speed, +10 % for numerical
  dispersion), so the criterion judges only the scattered field, relative
  to the largest total energy seen (normally the incident pulse in the
  box). Measured: an empty box in ε_r = 4 stops at the first check after
  the transit with `W/W_max` = −133 dB.
- `Results` records the reason it stopped (`Engine::run_until` returns a
  `RunOutcome` with a `StopReason`: energy decay, max timesteps, aborted
  by the progress callback, or diverged).

### 5.9 Conducting sheets
A csx `ConductingSheet` (conductivity `σ`, thickness `t`) is a slab much
thinner than the cells, drawn with zero thickness on a mesh plane
(`engine::sheet::ConductingSheet`, mapping in `compile::sheet`).

**Physics.** Inside a conductor the tangential fields diffuse across the
thickness (`dE/dz = −jωμ0·H`, `dH/dz = −(σ + jωε0)·E`), so per unit square
the slab is a uniform line of `γ = √(jωμ0(σ + jωε0))`,
`Z_c = √(jωμ0/(σ + jωε0))`, a two-port with `Z11 = Z_c·coth(γt)`,
`Z12 = Z_c·csch(γt)` (skin effect and shielding, Ramo–Whinnery–Van Duzer).
For `t ≪ δ` it is a resistive sheet of conductance `σt`
(`T = 2/(2 + η0σt)`); for `t ≫ δ` each face is a surface impedance
`(1 + j)/(σδ)` and the transmission falls like `e^{−t/δ}`. A single shunt
admittance on the edge cannot do both (shielding needs `1/Z12`, surface loss
`1/Z11`; they differ by `cosh γt`), so the model keeps both faces.

**Discretisation.** Each sheet edge is split into its two face voltages
`u+`, `u−`, joined by a ladder that is the finite-volume (circuit) form of
the diffusion equations: node conductances `σ·V_k·g` and capacitances
`ε0·V_k·g`, series inductances `μ0·Δz_k/g` (`g = w/l` of the edge). In other
words, a 1-D sub-grid across the thickness, error `O((Δz/δ)²)`. The layers
are graded geometrically from both faces: first layer `0.15·δ(f_max)`,
growth 1.15 (35 µm of copper at 10 GHz: 48 layers; `t ≤ 0.15·δ`: one
layer), which keeps `Z11` within ~1 % of the exact value from DC to `f_max`.
The edge's dual face is split at the sheet: each half has its share of the
edge capacitance and conductance (material-aware, sampled by the compile
step) and its half of the crossing dual edges, and the H updates see `u−`
below the sheet and the length-weighted mean of `u±` on the crossing dual
edges. The weights are the same in Ampère's and Faraday's laws, which keeps
the symmetric (energy-conserving) structure of the Yee scheme. The ladder
is integrated trapezoidally at `(n + ½)·dt`, coupled implicitly with the
face voltages (one prefactored tridiagonal solve per edge and step): no
extra stability limit, passive for any `σ`, `t`. The operator's energy is
the ladder energy, minus the edge capacitor the engine already counts,
plus the term `½·L·corr·i^{n−½}·(u− − u+)^n` of the H correction the
engine's look-ahead does not see; with it the engine energy is the exact
invariant (drift 2.8e-7 over 2000 steps for a lossless sheet in a 3-D
cavity, against 1.4e-3 without the correction term).

**Compile mapping.** The sheet normal comes from the property's planar
primitives; edges of line, curve or volume primitives (no normal) fall back
to PEC with a warning. PEC wins over a sheet. Sheets with `σ ≤ 0` or
`t ≤ 0` have no effect (warning). Sheet edges inside a CPML lose the CPML
correction (warning).

**Cylindrical grids** (§5.15). Sheets on `ρ = const`, `α = const` and
`z = const` surfaces use the same ladder with physical lengths: `g = w/l`
with `l` the edge length (`ρ_i·Δα` for α edges) and `w` the width of the
edge's strip of the sheet (`ρ_{i+½}·Δα*` for a ρ edge on a z sheet,
`ρ_i·Δα*` for a z edge on a ρ sheet, else the dual cell length); the
default capacitance share of a z edge on a ρ sheet is the share of the
annular dual face, `(ρ_hi² − ρ²)/(ρ_hi² − ρ_lo²)`, and the compile step
weights the material samples of a z edge's dual face by `ρ`. An edge on the
α seam of a closed grid is one physical edge stored twice: it gets one
ladder fed by the dual edges of both copies (the summed strip of a sheet
crossing the seam), and for a sheet *on* the seam (normal α) the two
copies are its two sides, each with its own kernel capacitance and
conductance (materials included). z edges on the axis stay ordinary
(warning); multigrid zones couple only the face voltage (warning). csx
selects sheet edges in an α range normalised to `[−π, π)`, so the compile
step re-selects planar sheets on the grid's α lines (§8.1). Validation
(`tests/cylindrical_sheet.rs`, coax `a = 2 mm`, `b = 6 mm`, 16 × 8 cells
across): a resistive annulus (`η0σt = 3.77`) across a matched line
transmits within 2e-4…1.7e-3 of `2/(2 + η0σt)` (2–8 GHz); a half-wave
coaxial cavity whose inner conductor is a lossy tube (`σ = 1e5 S/m`,
`t = 100 µm`, `t/δ = 4.4`) rings down with `Q = 195.6` against
`(ωL' + X')/R' = 195.9` from the exact `Z11`; with a resistive fin
(`σt = 1e-4 S`, an α sheet from `a` to `b`) `Q = 57.73` against
`ωC'/G' = 57.65`, identical on the seam and at `α = π/2`; results are
thread-count independent.

**Validation** (`tests/validation_sheet_conducting.rs`, TEM line,
2–12 GHz): thin sheet (`η0σt = 3.77`) transmission within 1.3e-4 of
`2/(2 + η0σt)`; thick-ish sheet (`σ = 1e6 S/m`, `t = 20 µm`,
`t/δ = 1.8…4.4`, 16 dB below the thin-sheet value) within 0.27 dB / 1.2° of
the exact slab, and within 1e-3 of the discrete ladder itself (the
remainder is the layer discretisation); thick sheet (`t/δ = 4.4…11`)
reflection loss `1 − |Γ| ≈ 2R_s/η0` within 0.3 %; passive and
thread-count independent in a 3-D cavity.

### 5.10 Near-field to far-field transform

`ems::nf2ff` (M3) computes radiation patterns from a closed box of
recorded near fields, by the surface equivalence principle (Balanis,
*Antenna Theory*, §12.2–12.3; Taflove & Hagness, ch. 8).

- **Recording.** The engine operator `Nf2ff` holds one `FreqDump` per box
  face and field (E and H), sampled at the primary nodes of the face. The
  dumps' interpolation co-locates the two fields: E is averaged along its
  own axis, and H bilinearly from the four surrounding dual edges, which is
  across the face normal for the tangential components. The DFT kernel
  uses each quantity's absolute sample time, (m − ½)·dt for H. That is
  the half-step phase correction, so no explicit `e^{jωdt/2}` factor is
  needed. Sampling runs serially on the stepping thread and is
  bit-identical for any thread count.
- **Sampling interval.** `Simulation::add_nf2ff` samples every k-th state,
  with the same k as the frequency dumps: the Nyquist interval of the
  settings' f_max divided by the oversampling factor (default 4). The
  fields are band-limited, so this barely changes the results (patch
  example: the same directivity to 0.01 dB) and costs much less (the
  whole patch run goes from 40 s to 3.8 s).
- **Equivalent currents.** On each face with outward normal n̂:
  J = n̂ × H and M = −n̂ × E. The face integrals use trapezoidal node
  weights, so they work on non-uniform meshes.
- **Far field.** With the e^{jωt} convention of the DFTs and phase
  centre c (the box centre by default):
  N = ∮ J e^{jk r̂·(r'−c)} dS' and L = ∮ M e^{jk r̂·(r'−c)} dS'. Then
  rE_θ e^{jkr} = −jk/(4π)·(L_φ + η N_θ),
  rE_φ e^{jkr} = +jk/(4π)·(L_θ − η N_φ), and
  U = (|rE_θ|² + |rE_φ|²)/(2η), with k = ω√(ε_r μ_r)/c0 and
  η = η0·√(μ_r/ε_r) of the background.
- **Background.** The box must lie in one homogeneous, isotropic,
  lossless medium (`Nf2ff::with_background`, `Nf2ffResult::with_background`,
  `Nf2ffBox::background`), possibly over a stratified ground (below). `Simulation::add_nf2ff` detects it from the
  engine coefficients of the recorded faces (`engine::Medium::of_box_surface`
  sampling: tangential E on the face, crossing H, tangential H half a cell
  either side; `C = (1 + ca)·dt/(2cb)`, `G = (1 − ca)/cb`, alike for H;
  PEC edges skipped; relative tolerance 1e-4). Lossy → `Unsupported`
  (no far field in a lossy medium; an explicit background uses the lossless
  part); inhomogeneous → warning and vacuum.
- **Power, directivity and gain.**
  - P_rad is the box Poynting flux ½ Re ∮ n̂·(E × H*) dS = −½ Re ∮ E_t·J* dS.
    `radiated_power_far` integrates U over the sphere instead, as a check.
  - D = 4πU/P_rad.
  - G = 4πU/P_acc, with P_acc the port's `accepted_power` from the same
    run. Both come from DFTs of the same transient, so the normalisation
    cancels.
- **Mirrors.** For an infinite PEC or PMC plane, image currents are
  added. A PEC plane reverses tangential J and normal M; a PMC plane
  reverses normal J and tangential M. Faces on the plane are skipped, and
  P_rad covers only the real half-space, so a monopole over ground gets
  twice the dipole directivity. The box must not cross a mirror plane,
  and at most one mirror per axis is allowed.
- **Accuracy** (`tests/nf2ff.rs`, `src/nf2ff.rs` unit test):

  | Case | Result |
  |---|---|
  | Analytic Hertzian-dipole fields on a 0.4 m cube (transform only) | P_rad within 2e-3, D = 1.5 within 5e-3 |
  | Short dipole (FDTD, λ/30 cells, box 6 cells out) | D = 1.490 (−0.7%) |
  | Same, box 9 cells out | D = 1.502 |
  | Short dipole: P_far/P_box | 0.993 (box 6 cells) → 0.9997 (box 9 cells) |
  | Monopole over a PEC boundary with a PEC mirror | D = 2.987 = 2 × dipole |
  | Half-wave dipole (L = 0.475 λ) | D = 1.650, P_far/P_box = 1.0001, P_box/P_acc = 1.0013 |
  | Patch antenna example | broadside D = 6.70 dBi |
  | Short dipole in ε_r = 4 (5 mm cells, λ_med/30) | D = 1.490, P_box/P_Hertz = 0.997, \|rE_θ\|/Hertz = 0.995 |
  | Same vs vacuum at equal moment Il | P ratio 2.002 (√ε_r), \|rE\| ratio 1.000 |

  (P_Hertz = η k² \|Il\|²/(12π) with the measured port current and the
  gap length; R_rad = 2P/\|I\|² = 0.438 Ω against (2π/3)·η·(l/λ)² = 0.438 Ω.)

  The small short-dipole error comes from the second-order co-location
  of the strong reactive near field close to the source. It shrinks when
  the box is moved further out; the transform itself is exact for any
  closed surface.
- **Cylindrical grids** (§5.15). The box is a node box in `(ρ, α, z)`:
  the `ρ = const` side surfaces, the `z = const` caps and, for an annular
  sector, the `α = const` faces; the inner side is left out when the box
  reaches the axis, the α faces when it spans the full circle of a closed
  grid (the seam is interior, so a sector may also start or end on it).
  Fields are sampled at the face nodes as on Cartesian grids, rotated to
  Cartesian components at each node, with the local outward normal
  (`±ρ̂(α)`, `±α̂(α)`, `±ẑ`) and trapezoidal weights with the metric
  (`ρ·Δα·Δz` sides, `ρ·Δρ·Δα` caps, `Δρ·Δz` α faces). Phase centre: centre
  of the Cartesian bounding box (on the axis for a full cylinder). The
  background is detected from the coefficients with the physical lengths
  and areas. Mirrors are Cartesian planes: normal to `z`, or normal to
  `x` / `y` if they do not cut the box; α faces lying in such a plane
  (a sector `α ∈ [0, π]` and `y = 0`, i.e. the PEC/PMC α walls of a
  half-space model) are skipped like any face on a mirror plane (a
  geometric test on the face's sample points). Validation
  (`src/nf2ff.rs` unit test, `tests/cylindrical_nf2ff.rs`): analytic
  Hertzian-dipole fields given in `(ρ, α, z)` components on an annular
  sector around an off-axis dipole (all six faces): `P_rad` within 2e-3,
  `|rE_θ|` within 2e-3, `D = 1.5` within 5e-3; FDTD short dipole on the
  axis (10 mm cells, 16 α cells, box `ρ ≤ 60 mm`, `|z| ≤ 60 mm`,
  cylindrical PML on `ρ_max`): `D = 1.5016 / 1.4990` at 0.7 / 1 GHz,
  `P_far/P_box = 1.0007 / 0.9987`, `P_box/P_Hertz = 1.002 / 1.005`, `sin²θ`
  pattern without φ dependence to 1 %; monopole over a PEC `z` plane with
  a `z` mirror `D = 3.005`; analytic z dipole and image over `y = 0`
  through a half-cylinder with a `y` mirror: far field within 3e-3, box
  power = the pair's half-space power within 3e-3; FDTD z dipole 50 mm
  over the PEC α walls of an `α ∈ [0, π]` sector grid: `D = 6.41`
  against 6.50 for the pair, pattern within 8e-3 of the peak,
  `P_far/P_box = 0.996`. openEMS does write frequency-domain field
  dumps on cylindrical grids (`dump_cyl`); its `nf2ff` tool (below) was
  not compared on them.
- **Layered ground** (wave 14; `nf2ff::Ground`,
  `Nf2ffResult::with_ground`, `src/nf2ff/ground.rs`). A box over a planar
  stratified medium that extends to infinity (a dielectric half-space,
  layers on it, a substrate on PEC/PMC) cannot enclose it; the box
  currents are instead radiated with the layered-medium Green's function
  in its far-zone (stationary-phase) form. Every plane wave of the
  currents' spectrum meets the stack with the transmission-line
  coefficients of its `k_t` (TE `Z = ωμ/k_z`, TM `Z = k_z/(ωε)`, tangential
  E as voltage, layers by `Z_in` recursion; Chew, *Waves and Fields in
  Inhomogeneous Media*, ch. 2), and the far field in either medium is
  `2πj·k_z·Ê(k_t)e^{−jkr}/r` of its own spectral amplitude:
  - box side (`r̂·n̂ ≥ 0`): `E = E_0(r̂) + e^{−2jk cosθ h}[Γ_TE (E'·ê)ê +
    Γ_TM M(E'_TM)]` with `E' = E_0(r̂')` the free far field towards the
    mirror direction, `M` the reflection about the interface, `h` the
    phase centre's height (PEC: `Γ = −1`, image theory);
  - inside a lossless bottom half-space: the transmitted wave,
    `E·ê = (k_zN/k_z1)T_TE e^{φ}(E_0(d̂)·ê)`, `E·p̂_N = (k_N/k_1)T_TM
    e^{φ}(E_0(d̂)·p̂_1)`, `φ = −jk_z1 h + jk_zN(h + D)`, with `d̂ = (k_t −
    k_z1 n̂)/k_1` complex beyond the critical angle (the radiation
    integrals continued analytically: the evanescent spectrum of a
    source near a denser half-space); `η`, `k` of that half-space for `U`.

  Lossy layers are allowed (no far field inside a lossy bottom
  half-space); surface waves (a grounded slab's TM0) carry box power that
  reaches no far-field direction. Mirrors normal to other axes combine.
  Validation:

  | Case | Result |
  |---|---|
  | PEC / PMC ground vs the PEC / PMC mirror (analytic dipoles, vertical and horizontal) | 1e-9 |
  | Analytic dipoles 0.3 m over `ε_r = 4`: `E_θ` (vertical) / `E_φ` (horizontal) vs Balanis §4.8 `R_v` / `R_h` | ≤ 3e-3 of the free field |
  | Vanishing slab on PEC vs PEC; vacuum layer on PEC vs deeper PEC; quarter-wave slab on PEC | 1e-5, 1e-9, `Γ = +1` |
  | FDTD dipole (10 mm cells, 1 GHz) 45 mm over `ε_r = 4`, box from 10 mm above the interface: upper pattern vs `sinθ·|1 + R_v e^{−2jkh cosθ}|` | 3.3e-4 / 6.0e-4 at 0.8 / 1 GHz |
  | Same: `P_far` (upper + transmitted, 360 θ steps) / `P_box`; share into the ground | 0.9888 / 0.9900; 77.7 % / 75.6 %, `D_max` 29 / 33 at θ = 150° (critical angle) |
  | FDTD dipole over a 40 mm `ε_r = 3` grounded slab: two boxes (10 mm vs 30 mm over the slab, ±60 vs ±80 mm) | far fields within 6.4e-3 of the peak; space wave 5.7 % of `P_box` (TM0 surface wave) |
  | Vacuum "slab" on the PEC face (no surface wave): `P_far/P_box` | 0.9885 |

  (The remaining 1 % of the power balance is the near-source co-location
  error of the short dipole, the same as without ground, §5.10 table.)
- **Not done.** Lossy backgrounds around the box; frequency-dependent
  phase centres; the far field inside a lossy bottom half-space;
  surface-wave power (the box flux includes it, the far field does not).
- **openEMS's `nf2ff` tool** (`io::openems::nf2ff`, binary `ems-nf2ff`,
  feature `hdf5`). openEMS transforms its NF2FF box dumps with a separate
  program, `nf2ff <job.xml>`, that reads E/H dumps named in an XML file
  and writes a far-field HDF5 file. Its XML schema and output layout were
  found by black-box experiments only (error messages, trial XML files,
  synthetic h5py dumps; no openEMS source or scripts consulted):
  - XML: `<nf2ff freq Outfile [Radius Center Eps_r Mue_r Verbose
    NumThreads]>` with `<theta>`/`<phi>` lists in radians, one
    `<Planes E_Field H_Field/>` per face and optional
    `<Mirror Dir="0|1|2" Type="PEC|PMC" Pos/>` (images of images
    included). File names are relative to the working directory.
  - Input: openEMS dumps on one plane each (one line along the normal),
    frequency domain (also legacy layout) or, if the frequency is not
    stored, time domain with `F = 2Σx(t)e^{−jωt}Δt`; E and H on the same
    lines.
  - Transform: the formulas above with trapezoidal node weights, the
    normal pointing away from `Center`, `k`/`η` from `Eps_r`/`Mue_r`.
    Output: `/Mesh/{theta,phi,r}` (f32), `/nf2ff/{E_theta,E_phi}/FD/f%d`
    (complex), `P_rad` = `|E|²/(2η)` at `Radius`, attributes `Prad` (box
    Poynting flux, doubled per mirror), `Dmax = 4πr² max P_rad/Prad`,
    `Frequency` (f32).
  - Agreement (`tests/openems_nf2ff.rs`, `gen-nf2ff.sh`): same file
    structure; on the same synthetic inputs ≤ 4.5e-6 relative (openEMS
    computes in single precision somewhere; ~3e-7 without materials). A
    short dipole run by both solvers with the same NF2FF box dumps
    (DumpType 10/11, DumpMode 1, FileType 1 on the six faces): ems's
    face dumps are within 5e-4 of openEMS's, the far fields within 1e-5,
    the box flux `Prad` within 1 % (the near field of a small box is
    strongly reactive), `Dmax` 1.49 vs 1.50.

### 5.11 Plane waves (total-field / scattered-field)

csx `PlaneWave` excitations (type 10) are injected with the TF/SF technique
(Taflove & Hagness ch. 5) by `engine::PlaneWave`; the compile mapping is in
`compile::planewave`.

- **Regions.** A node box `start..=stop` is the total-field region: the E
  edges in the closed box and the H edges whose half-cell coordinates lie
  strictly inside. Then total-field H edges only read total-field E edges,
  and the two regions touch only through the E edges on the box surface and
  the H edges just outside.
- **Consistency conditions.** After the H update every scattered-field H
  edge next to a surface E edge gets `+db·K·u_inc^n`; after the E update
  every surface E edge gets `+cb·K·i_inc^{n+½}` per scattered-field H
  neighbour (`K = ±1` the shared curl sign; the integral-form curls are
  transposes, so one sign serves both). Using the engine's own `cb`, `db`
  keeps materials and PEC on the surface consistent. Box faces on the domain
  boundary get no corrections.
- **Incident field** from an auxiliary 1-D Yee grid along `k̂` on the
  engine's time step (so E and H come with the right staggering), hard
  source a few cells before the entry corner, a 64-cell graded matched-loss
  layer at the far end, and 4-point Lagrange interpolation at the projected
  edge positions `ξ = k̂·(r − r_ref)`. Its spacing is chosen so that its
  numerical wavenumber equals the 3-D Yee wavenumber along `k̂` at a matching
  frequency (the waveform centre); along a mesh axis this is the cell size
  and the match is exact at every frequency, and so it is along the face
  and body diagonals of a cubic mesh (all samples then sit half-way between
  auxiliary nodes, and the interpolation is a uniform filter of an exact
  discrete wave).
- **Reference.** `E_inc = Excite·a(t − t_d)` at the box corner the wave
  enters first; the 1-D source runs ahead by the numerical travel time.
  Through `Simulation` (XML/csx models), `t_d` is openEMS's lag, one time
  step plus the travel time across the first cell, and `Delay` is ignored
  like openEMS does (§5.16); the engine operator takes any `t_d`.
- **Background.** Homogeneous, isotropic, possibly lossy (`ε_r`, `μ_r`,
  `κ`, `σ_m`). The compile step takes it from the effective edge materials
  of the box faces not on the domain boundary (`compile::planewave`,
  same sampling as §5.10; inhomogeneous → compile error naming the plane
  wave). `PlaneWave::new` checks that the coefficients of every edge
  taking part in a correction match the spec's background (to `f32`
  rounding) and fails otherwise. The 1-D grid uses the background speed,
  impedance and the engine's semi-implicit loss terms, so along a mesh
  axis it is the exact discrete 3-D wave, attenuation included. In a lossy
  background the source amplitude is raised by the grid's attenuation over
  the `m0` cells to the reference corner (exact at the matching
  frequency; propagation constant from
  `sinh(γΔξ/2) = √(A_e A_h)/(2r)`, `A = 2j·sin(ωdt/2) + 2s·cos(ωdt/2)`),
  and the matched layer is preceded by a 256-cell C² taper to a
  distortionless line (equal E and H loss), on which the equal-loss layer is
  reflectionless (without it the layer's reflection put a 3e-3 backward
  wave into the incident field).
- **Measured** (`tests/planewave.rs`, 1 mm cubic mesh, 8–16 GHz pulse):
  scattered-field leakage −130 dB at normal incidence and along the
  diagonals, −77 dB at 30°, about −55 dB on a mesh graded ±20 %. A PEC
  plate and a dielectric slab inside the box give the exact discrete (Yee)
  reflection to 5e-5 and 4e-5. In an ε_r = 4 background (4 GHz, the same
  cells per wavelength): leakage −131 dB normal, −130 dB (1,1,1), −77 dB
  at 30°; inside the box E(x2)/E(x1) equals the discrete propagation factor
  to 1e-7 and |E/H| the discrete impedance η0/2·cos(K/2) to 1e-7. Lossy
  background (ε_r = 2, κ = 0.13 S/m, tan δ ≈ 0.2): leakage −130 dB,
  propagation factor (attenuation included) to 2e-5, E_inc(r_ref)/a at the
  matching frequency 1 to 2e-6. A csx model in ε_r = 4: background
  detected exactly, probe E within 1e-3 of a(t − ξ/v).
- **Layered and dispersive backgrounds** (wave 14;
  `engine::planewave` module docs, `src/engine/planewave/spectral.rs`). A
  box crossing planar layers (`PlaneWaveSpec::with_layers`, ends
  `LayerEnd::{Open, Pec, Pmc}`) or with dispersive poles on its surface
  (`with_dispersion`) gets the exact discrete incident field computed
  per frequency instead of the auxiliary grid:
  - the wave has a fixed transverse slowness `p = k̂_t/v_inc(f_m)`; on a
    uniform transverse mesh `e^{−jωp·r}` diagonalises the transverse Yee
    differences, so the 3-D equations reduce to a 1-D system along one
    column of edges (six unknowns per level, `i_s`/`u_s` eliminated
    locally, banded LU with partial pivoting), built from the engine's
    own `ca, cb, da, db` and the poles' discrete `χ` (trapezoidal
    `χ(Ω_t)`, leapfrog `χ(Ω)`);
  - the column spans the domain along `s` (layered) or the box ± 2 nodes
    (homogeneous dispersive, its medium only); open ends get a uniform
    extension of the end medium with a frequency-domain stretched-
    coordinate PML (cubic, design round trip 1e-10), closed ends the
    domain's PEC/PMC planes; a current sheet in the incidence-side
    extension radiates the wave; the same line filled with the incidence
    medium (opened at a closed far end) gives the pure incident wave, which
    normalises `E_t(r_ref) = E0_t·a(t)` (two sheets, `u_t1` and `u_t2`,
    combined for the polarisation);
  - FFT window at `dt/2` (radix 2), bins above the engine Nyquist
    frequency or where the incident wave is evanescent in the incidence
    medium skipped (else the layers can trap it: a real-axis pole); the
    DC bin is not solved (singular) — a constant per series fixed by the
    field being zero after the pulse; the window doubles until the
    series' third quarter is below 1e-6 of the peak (error otherwise:
    resonant layers, or an excitation that does not end); the incident
    field before `t = 0` must stay below 1e-5 (error: a Drude medium near
    its plasma frequency sends slow components that would have had to
    arrive before the run);
  - corrections read their series with cubic interpolation at their fixed
    transverse delay; those of edges whose poles the `Dispersive`
    operator solves implicitly (trapezoidal, all Debye) are added to the
    start value before the kernel (`pre_e`/`pre_h`, divided by `ca`/`da`):
    added after the implicit solve they leaked −35…−41 dB;
  - every edge taking part must have the column's medium at its level
    (layered) or the background's (homogeneous), poles included;
  - `Simulation` (native, not openEMS files): a box surface whose medium
    varies along one axis the wave crosses (`|k̂_s| ≥ 1e-3`) is layered,
    ends from the domain boundaries (PEC, PMC, else open); compiled poles
    are handed over; the duration is the waveform's (non-terminating
    waveforms: the plane wave is dropped with a warning).

  Measured (`tests/planewave_layered.rs`, 1 mm cubic mesh, 36³ with CPML,
  box 13…23, 8 ± 4 GHz pulse):

  | Case | Leakage |
  |---|---|
  | `ε_r = 4` half-space across the box: normal, 30° TE / TM, (1,1,2) mixed, −z 30° TM | −132, −130, −127, −126, −130 dB |
  | 40°: `ε_r = 6` slab across the bottom face; grounded `ε_r = 3` substrate (PEC z_min, from above) TE / TM; half-space on a z mesh graded 0.7…1.3 mm | −126, −130 / −125, −133 dB |
  | Homogeneous Debye, Drude, Lorentz (trapezoidal and leapfrog), normal and 30° (10 ± 2 GHz) | −127 … −135 dB |
  | Drude layer (4 cells) across the box, 25° TM | −130 dB |
  | `Simulation`, csx substrate across the box, 30° TM | −132 dB |
  | Vacuum, 30° (layered mode; the auxiliary grid gives −77 dB) | −130 dB |

  Physics: at normal incidence the field in the box equals the
  independent discrete 1-D Yee response of an `ε_r = 5` slab to 1.4e-7;
  at 35° the field 3 cells into an `ε_r = 4` half-space is the continuum
  Fresnel `t` (TE / TM) to 4.9e-3 / 4.6e-3 (second-order mesh error); in
  dispersive backgrounds `E(x2)/E(x1)` is the discrete factor of
  `ε∞ + χ_d` to 2.2e-7. Set-up 0.2–0.4 s for these boxes.
- **Energy** see §5.8: the end criterion waits for the transit of the
  incident pulse through the box.
- **Limitations.** Homogeneous isotropic background around the box, or
  layered along one axis the wave crosses, or dispersive (spectral
  incident field: uniform transverse mesh for exactness, an excitation
  that ends, no slow components before `t = 0`); lossy backgrounds not
  supported by the periodic-cell variant, which also ignores layers and
  poles; box clear of CPML/Mur. Cartesian grids only:
  on a cylindrical grid the excitation is dropped with a warning, as
  openEMS ignores it there (§5.16.2); a TF/SF box in `(ρ, α, z)` would
  need the incident field evaluated analytically at the edge positions
  and was not built for lack of a reference to compare with.

### 5.12 SAR

`post::sar`, fed by csx `DumpBox` types `SARLocal` / `SAR1g` / `SAR10g` /
`SARRaw` (or directly from an engine `EdgeDft`).

**Input.** An `EdgeDft` operator transforms the raw voltages of every E
edge of the box's cells (no interpolation: SAR needs the mean of `|E|²`,
not the square of a mean field). The compile step samples, per cell, the
conductivity `κ` (per axis) and the csx `Density` with `n³` points (`n` =
subsampling); metal and background have `ρ = 0` and are not tissue.

**Power.** With peak phasors, per cell
`p = ½·Σ_a σ_a·¼·Σ_{4 edges ∥ a}|E_e|²`, `E_e = U_e/l_e·cos(πf·dt)`. The
cosine is the transform of the time-averaged voltage `(u^{n+1}+u^n)/2` that
the conduction term of the update uses, so `Σ p·V` is exactly the loss of
the discrete scheme: it equals `Σ_e ½·G_e·|Ū_e|²` for the area-weighted edge
conductances, and a time-domain integral of `G_e·ū_e²·dt` agrees with the
Parseval sum of the DFTs to 1e-9. Local SAR is `p/ρ`. Values from transient
DFTs are spectral densities; normalise by the accepted port power. Only `κ`
contributes: dispersive-pole losses are not included.

**Mass averaging.** After the cube procedure of IEEE Std 1528 /
IEC/IEEE 62704-1 as publicly described:

1. per tissue voxel, an axis-aligned cube centred on it is grown until its
   tissue mass is `m` (voxels cut by the cube count with their volume
   fraction, solved to 1e-10 of the mass, via 3-D prefix sums); valid if at
   most 10 % of its volume is non-tissue (outside the domain counts as air);
2. voxels whose centred cube is invalid take the largest average of the
   valid cubes containing their centre;
3. voxels in no valid cube take the cube with the voxel at the centre of a
   face, of the six orientations the one with least air.

Deviations: no rotated/tilted cubes; step 3 simplifies the standards'
surface treatment; non-biological lossy material counts as air.

**Cylindrical grids** (§5.15). The voxels are the `(ρ, α, z)` cells, of
volume `Δα·Δz·(ρ_{i+1}² − ρ_i²)/2` (`SarVolume::cylindrical`); the edge
fields use the physical edge lengths and each edge's `|E|²` is weighted
with the share of the cell volume its dual cell has there, as the edge
conductances `κ·A*/l` count it (¼ for ρ edges, `ρ_e/(4ρ_{i+½})` for α
edges, the half-annulus shares for z edges), so `Σ p·V` is again the
discrete conduction loss. The compile step samples `κ` and density in
`(ρ, α, z)` with `ρ`-weighted subsamples.

*Mass averaging* (`post::sar` submodule `cyl`, src/post/sar/cyl.rs) uses
cubes in physical space, axis-aligned with Cartesian `x, y, z`, centred
on the voxel centre `(ρ_{i+½}, α_{j+½}, z_{k+½})`, with the same three
steps and air rule as on Cartesian cells (face cubes put the voxel
centre at the centre of a cube face, six orientations). The overlap of
a cube with each annular cell is exact: the z overlap is an interval,
and the area of the cube's square section inside the sector is
`|Q ∩ W ∩ D(ρ_1)| − |Q ∩ W ∩ D(ρ_0)|` (`W` the cell's wedge, two
half-planes, cells wider than π/2 split; `Q ∩ W` a convex polygon by
Sutherland–Hodgman; polygon ∩ disc by summing, over its edges, the
triangle (origin, edge) inside the disc: straight pieces inside,
circular sectors outside). The cube mass is then a continuous increasing
function of the side, solved to 1e-12 like the Cartesian cubes.
Validation (`post::sar` unit tests): sector areas exact to 1e-14 (a
square cut into 28 annular cells sums to its area to 1e-12); a constant
power density averages to the local value on every tissue voxel of a
tissue cylinder in air (centred, covered and face cubes, all cubes 1e-10
of the mass); a power density constant per z layer averages exactly
(1e-10) to the layer mean over the cube height; a linear `p(x, z)`
averages to the centre value within 4.6e-3 (the cells' piecewise
constant `p`); thread independent.

Validation of the power (`tests/cylindrical_sar.rs`): a tissue ring in a
closed circular PEC cavity with an axis (16 α cells), fed by a one-cell port in
the tissue: absorbed power within 1.5e-5…2.2e-5 of the accepted power
(0.5–1.5 GHz), tissue mass exact; the 1 g average has a voxel cube of
1 g everywhere and its peak next to the port. openEMS-compatible files
(`io::openems::dump`): openEMS writes local, raw and averaged SAR on
cylindrical grids (`Mesh` `rho`/`alpha`/`z`, `.vts` for VTK); observed
(`cyl_sar`): its cell volume is `ρ_i·Δα·Δρ·Δz` with the radius of the
cell's *lower* line (so its `mass` attribute is first-order low: 3.223 g
against 3.393 g for the exact sector), and it scales the α `CellWidth`s
by the drawing unit. ems writes the same files with the same rules (local
SAR within 3.5e-3 of openEMS's peak; the remainder is a 1e-4 difference of
the two codes' cell fields on this model, both codes' local SAR being
`κ|E|²/(2ρ)` of their raw files to 1e-7).

*openEMS's averaging on cylindrical grids* (black box, `cyl_sar` with a
1 g dump): its in-run 1 g file equals `sar_calc` on its raw dump bit for
bit, and also `sar_calc` on the same raw file with the axes renamed to
`x, y, z` and `@mesh_type` 0. So openEMS averages the `(ρ, α, z)` cells
as if they were Cartesian cells of the raw dump's `CellWidth`s (`Δρ`,
`Δα·u` with the drawing unit `u`, `Δz`), each weighing with its `Volume`
`ρ_i·Δα·Δρ·Δz`: its "cubes" are boxes of `s/(Δα·u)` α cells, neither
cubes nor of constant physical size (at `u` = 1 mm and ρ = 20 mm, a
1 g cube spans about 20 α cells of 0.2 mm "width"). Its `sar_calc` output
of a cylindrical dump has a Cartesian `/Mesh` (`x, y, z` = the raw `rho`,
`alpha`, `z`, `@mesh_type` 0). ems reproduces this for openEMS files
(the widths stacked from 0 as Cartesian lines, `SarVolume::with_cell_volumes`
with openEMS's volumes, then openEMS's averaging `averaged_openems`; used
by the in-run dumps of `io::openems::dump` and by `ems-sar-calc` on
cylindrical raw dumps): `ems-sar-calc` on openEMS's raw file equals
openEMS's 1 g file to 5e-8 of the peak; ems's own in-run 1 g dump is
within 4.5e-3, the field difference of the two codes on that model (as
its local SAR). Rust-built models
(`SarResult::field`) use ems's physical cubes.

**Output.** `SarField`: per-voxel values plus how each was obtained
(`VoxelClass`), the peak and its location, `.vtr` (cell data) and, with
`hdf5`, an `ems-sar` HDF5 layout.

**Validation** (`tests/sar.rs`, `post::sar` unit tests): lossy tissue block
in a closed PEC box fed by a one-cell lumped port inside it: absorbed power
equals the port's accepted power within 3e-6 / 1.6e-5 / 3.9e-5 at
0.5 / 1 / 1.5 GHz (a multi-edge port is ~1 % off because its terminal
current is the mean over its series edges). Averaging: a constant field
averages to the local value everywhere (incl. boundary voxels and graded
meshes), a linear gradient to the centre value (1e-12), the air rule and
boundary cubes on a half-space and a one-voxel slab, results independent of
the thread count.

**openEMS methods** (`SarVolume::averaged_openems`, `post::sar::openems`,
`post::sar::AveragingMethod`; black-box openEMS v0.37, no source
consulted). openEMS's in-run averaged dumps (DumpType 21/22) are identical
to `sar_calc --method SIMPLE` on the raw dump (no DumpBox attribute
selecting another method was found). Identified with synthetic raw files
fed to `sar_calc -e -v` (the `-e` cube mass and volume give each voxel's
final cube side; a power gradient along one axis gives the cube's extent
along it; the `-v` counters "invalid cubes case 1/2" separate the
validity rules):

- *Weights.* A voxel inside a cube counts with the product of its
  per-axis overlap fractions times its volume — in `sar_calc` the raw
  file's `Volume` dataset (doubling it changes the cubes; its float32
  rounding, 2.8e-8 low for 1 mm and 2 mm cells, makes a cube whose face
  should lie exactly on a tissue boundary a little larger, which decides
  that face's validity: `dump_sar`'s 18 such cubes are invalid). Outside
  the domain counts as nothing, air (`ρ = 0`) as no mass and no power
  (lossy air does not count). Quirk: a cell holding *both* ends of the
  cube along an axis (touching a cell line counts) weighs
  `|(2u − 1)(2v − 1)|` along it, `u` the fraction of the cell above the
  lower end, `v` the fraction below the upper end: `(s/d)²` for a cube
  centred in the cell (the earlier "squared fraction", seen in the
  reported volumes of cubes thinner than a 3 mm cell), `|2s/d − 1|` for a
  face cube shorter than its voxel along its normal (zero at half a cell,
  rising again below). Found by perturbing the power of single cells:
  the change of a voxel's average is the cell's weight over the cube mass,
  which gives every cell's weight in the chosen cube (fixed cube, since
  the power changes neither the cube nor, in C95.3, its choice).
- *Cube solver.* `s₀ = (m/ρ_v)^{1/3}`, `a = s₀/2`; `ρ̄ = M(a)/a³` (the
  full geometric volume, also when clipped); `b = max((m/ρ̄)^{1/3},
  2·min(Δx, Δy, Δz))` for a centred cube, `(m/ρ̄)^{1/3}` for a face cube
  (the minimum made 10 % of the face cubes of graded meshes end
  elsewhere); then secant steps through the last two points
  from `(a, M(a))`, `(b, M(b))` until `|M − m| ≤ tol·m`, `tol` = 5 %
  (`SIMPLE`, `IEEE_C95_3`, inclusive, e.g. a first point at −4.55 % is
  kept and the switch is at exactly −5 %) or 1e-6 (`IEEE_62704`). How it
  was found: clipped uniform cubes give rational sides (`120/19` mm for a
  6 mm cube 2.5 mm from the domain edge = the secant through the masses at
  3 and 6 mm); a lone voxel of density `q` in uniform tissue gives a
  first point accepted without iteration with `ρ̄ − 1 ∝ q(q−1)/8` (the
  mean density of the half-size cube); thin lines of tissue (mass linear
  in `s`) make every secant exact and expose the start points; the
  anisotropic and sub-2-cell cases fixed the minimum (`2·min Δ`, the
  squared fractions in the half-size cube). Two safeguards (wave 13): a
  step leaving the bracket (largest side seen below the mass, smallest
  above it) bisects the bracket, and a step to a side `≤ 0` doubles the
  last side (equal masses give an infinite step that these catch). The
  plain secant stalls below a cell, where `M ∝ s⁶`: a sweep of cubes of
  0.2…1.25 cells in uniform tissue gave openEMS's sides for all 22 sizes
  with the bisection and for only those ≥ 0.85 cell without it (where the
  path stays in the bracket); the doubling came from face-cube paths
  through the non-monotonic `|2s/d − 1|` region (a scan of the third
  point reproduced openEMS's side at `2b`). A half-size start already
  within the tolerance is kept (never on Cartesian files, where `M(a) ≈
  m/8`; on cylindrical raw files, whose `Volume` is 10…30 times the
  widths' product, SIMPLE took it: `cyl_sar` at 0.02 and 0.1 g differed
  by 1.4e-5 and 1.1e-3 of the peak before, now ≤ 1e-7 for every method
  from 1 mg to 1 g; `sar_calc_edge` replays 0.1 g with SIMPLE and C95.3).
- *SIMPLE*: each tissue voxel gets the average (tissue power over tissue
  mass) of its centred cube; no cube → 0 (counted unused).
- *IEEE_C95_3 / IEEE_62704 validity* of a centred cube: inside the
  domain, and no face entirely in air (the voxel layer containing the
  face must hold a tissue voxel over the face's extent; 2 % of a face in
  tissue suffices, an air pocket inside or a corner in air is fine), also
  on every iterate of the solver lighter than the mass: a lighter iterate
  with a face in air makes the cube invalid even when the final cube's
  faces all touch tissue (heavier iterates and iterates leaving the
  domain do not count). This was the "openEMS rejects ~10 % more cubes
  of about two cells on graded meshes": on 11 random graded cases (blobs
  in air, salt-and-pepper tissue, 0.0004…0.01 g, 2600 cubes) the rule
  separates openEMS's verdicts without an exception, where face-area
  thresholds, face point samples, cell-centre layers or air fractions do
  not; checking the half-size start alone fits the heavy masses only.
  `IEEE_62704` also rejects more than 10 % air (by volume, `1 −
  V_tissue/V` with the weighted volume `V`, which is `s³` unless a cell
  holds both ends; random-density blobs separate cleanly at 0.0991 /
  0.1006; with `s³` every cube below a cell failed).
  Invalid voxels take the maximum average of the valid cubes that
  *overlap* them with positive volume (C95.3) or *contain* them entirely
  (62704; touching counts). The rest (62704's "unused", also C95.3 where
  no valid cube overlaps, e.g. thin slabs) take a face cube: the cube
  extending from one of the voxel's six faces (`-e` type 1…6: +x from the
  lower x face, −x from the upper one, …), centred across, solved with
  the method's tolerance; 62704 takes, among those whose volume within the
  domain is at most 1.05 × the smallest, the largest average (4064 voxels
  of random blocks: 0 mismatches; air fraction, side or air volume do not
  separate); C95.3 the smallest volume. A face cube shorter than its
  voxel along its normal weighs the same with either face (`|2s/d − 1|`
  is symmetric), so the `+` and `−` cubes along that axis coincide and
  openEMS's choice between them (its `-e` type) is decided by rounding.
- *Counts* (`@valid/used/unused/air_cubes`): own valid cube / maximum of
  others / face cube or none / not tissue. `maxSAR_idx`: the f32 maximum,
  first centred cube among equal values.
- *Auto range* (`-a N`): the output holds only the cells of the bounding
  box of the tissue cells with local SAR ≥ `10^{−N/10}` × the peak local
  SAR (at any frequency), widened by `(m/ρ_min)^{1/3}/2` (lightest
  tissue) and extended to the cells holding the widened faces (Mesh and
  datasets cropped, counts and peak over the box; FD `@autorange`,
  `@autorange_padding`, `f%d@autorange_threshold`); the averages see the
  whole domain (equal to the cropped full computation, also the IEEE
  coverage). A threshold above the averaged peak prints openEMS's
  warning.
- *Agreement* (`tests/openems_sar_calc.rs` replays every `sar_calc`
  command of the fixtures, `gen-sar.sh`): every file equal to float
  precision (data ≤ 1e-6 of the peak, 2e-6 for 62704 whose cube masses
  agree to its 1e-6 tolerance) on `dump_sar` (2 mm cells, 1 g, all cubes
  clipped, touching faces; before: SIMPLE 2.9 %, IEEE up to 78 % of the
  peak), the uniform block (before: 1 %), `sar_calc_synth` (graded
  mesh, random density and irregular tissue with each method, light and
  heavy masses, `-a`; a thin slab of face cubes; 1×1×3 mm cells) and
  `sar_calc_edge` (wave 13: 1 mm cells with cubes of 0.3…1 cell, a
  graded mesh of 0.4…1.6 mm cells with a random-density ball in air and
  the same mesh filled with tissue, 0.0003…0.003 g, each method, cube
  statistics where no two cubes tie). Before wave 13 the edge files
  differed by up to 100 % of the peak (SIMPLE cubes of 0.3 cell: 94 of
  210 tissue voxels, mostly without a cube), 32 % (C95.3 validity on the graded ball,
  45 voxels), 23 % (62704 face cubes of the filled mesh, 299 voxels);
  now ≤ 1.3e-7 of the peak. A random sweep (216 runs: blobs, filled and
  salt-and-pepper tissue, uniform and graded meshes, uniform and random
  density, anisotropic cells, 0.0001…0.008 g, each method) left
  differences only in ties and in 1 % of the C95.3 face cubes below
  about 0.6 cell. Known differences, all decided by openEMS's rounding:
  face cubes that tie (equal volumes for C95.3, equal averages for
  62704, common in uniform tissue: 16 of 294 voxels of `dump_sar`; for
  62704 only the `-e` type differs, the cubes being the same); cubes of
  a whole number of cells in uniform tissue (a face on a cell line at
  the 1e-8 level of the single-precision volumes decides the validity,
  e.g. 1 mg in 1 mm cells of 1000 kg/m³); a face cube whose half-size
  start is exactly half a cell (its mass is zero, openEMS's start then
  hinges on rounding; ems starts from the whole domain). Not reproduced:
  about 1 % of the C95.3 face cubes below about 0.6 cell, whose long,
  non-monotonic iterations end on another side within the 5 %.

`io::openems::sar_calc` and the binary `ems-sar-calc` (feature `hdf5`) take
`sar_calc`'s command line (`-i -o -m --method -a -n -v -e
--legacyHDF5Dumps`) and write its output layout:
root `@mass`, `@proc_time`; `/Mesh` copied; `/FieldData/FD` with
`@frequency` (f32), the cube counts and `f%d` (f32, `@maxSAR`,
`@maxSAR_idx`, `@power`); `-e` adds `f0_Cube{Mass,Type,Vol}`; `-a` the
auto-range attributes; the legacy layout is `[nz, ny, nx]` with
`@legacy_fmt`.

### 5.13 Conformal PEC

Opt-in (`CompileOptions::conformal`, `run_xml --conformal`); off by default,
so existing results are unchanged. Geometry in `compile::conformal`, scheme
in `engine::conformal`.

**Geometry.** "In metal" = the highest-priority material/metal at the point
is a `Metal` property with a solid (3-D) primitive; curves, wires and
sheets on mesh planes keep csx's PEC mask, sheets off the mesh planes are
split (§5.13.1). Nodes and edge midpoints in the bounding box
of the solid metal are queried; an edge whose three samples differ is
sampled at 9 points and each transition bisected (2⁻³⁰ of the edge), giving
the open fraction `f_e`; transitions within 1e-6 of an edge end are moved
onto it. A face's open area `f_A` is the shoelace area of the polygon of
its open corners and transition points in perimeter order (straight chords
across the metal: exact for planar walls; a corner is open if either edge
meeting there is open at it), its centroid the point the scheme
interpolates small faces at. Edges with `f_e = 0` become PEC, cut edges
non-PEC (unless their PEC came from a sheet/wire), and so do the edges of
faces without open area: their open part lies on the metal surface. E
excitations on cut edges are scaled by `f_e`.

**Scheme (Dey–Mittra in integral form).** With `u = E·l_open` on cut edges,
Dey–Mittra is a coefficient change: `cb → f_e·cb`, `db → db/f_A`. The
system stays `C·du/dt = Kᵀi`, `L·di/dt = −K·u`, so the energy invariant of
§6 remains exact.

**Small faces: stability and accuracy.** Slivers (small `f_A`, long open
edge) resonate above the Courant limit: unmodified Dey–Mittra gave an
estimated `s = ωdt/2` up to 8 (sphere) and diverged. Area floors
(`f_A ≥ τ`) only became stable at τ ≈ 0.3–0.5 and then converged at first
order again. Faces with `f_A < τ = 0.5` therefore lose their own current:
`i = P·g` with `g` the remaining currents, and the scheme is the Ritz
restriction of Dey–Mittra to that subspace (in the spirit of the uniformly
stable conformal scheme of Zagorodnov, Schuhmann & Weiland, 2003):
`C·du/dt = KᵀP·g`, `(PᵀLP)·dg/dt = −PᵀK·u`. It is symmetric (energy
invariant exact) and `PᵀLP` is at least the inductance of the faces that
keep their current, so the time step is unchanged.

* *Merging* (`P` 0/1, the previous scheme, now the fallback): the small
  face shares the current of the neighbour across its most-open edge
  (enlarged cell, Xiao & Liu 2004), `L_g = Σ f_A·L`, merged contour. This
  is stable, but it extrapolates `H` as a constant over a cell, an `O(h)`
  error wherever `H` varies across the merge. For TM modes of z-invariant
  cavities the merges of the x/y faces run along z, where nothing varies,
  hence second order; TE modes carry `H_z` on the cut z faces, merged
  in-plane: first order (TE11 of a circle +0.58 / +0.26 % at 10 / 20 cells
  per radius). Pure Dey–Mittra (no merging, Courant factor 0.05) gives
  −0.015 / −0.003 % there; the error grew with the merge threshold.
* *Interpolation* (default): the small face's `H` is the affine
  interpolation, at its open-area centroid, of three faces with their own
  current: the host across its most open edge and one host neighbour per
  in-plane axis (away from the small face), barycentric weights (≤ 2.5 in
  magnitude; typical 1.5–2). Exact for affine fields, so second order.
  Hosts sharing small faces are coupled by `PᵀLP`: per block (a band along
  the wall) an envelope Cholesky factor in reverse Cuthill–McKee order,
  solved in the `post_h` operator (`MergedFaces`). Two variants failed:
  row-sum lumping of `PᵀLP` (diagonal, but negative weights lower the
  hosts' masses: `s` = 1.16 at Courant factor 0.99), and a weighted
  least-squares stencil over ~10 faces (it spread weight over unrelated
  faces and supported a spurious wall-guided mode).
  On the α faces of a cylindrical grid the dual edge `ρ·Δα*` varies
  across the face: the interpolated face's current and inductance are
  those of the dual edge through its centroid (`ρ_c·Δα*`). At the face
  centre the interpolation converged at first order there (coaxial shell
  TM010 with the small side at 0.3 of a cell, §5.13.1: +0.21 / +0.081 /
  +0.038 % at 1 / 0.5 / 0.25 mm, now −0.25 / −0.066 / −0.017 %).

**Deviation currents (wave 13).** Removing a face's current enlarges the
null space of `PᵀK`: an `E` field circulating on the small face's cut
edges that no reduced current sees is static in the reduced scheme
although its curl is not zero. In a cross section these are extra
zero-frequency modes (invisible in 2-D resonances); along a wall they
travel as a TEM-like wave at ~`c` guided by the cut cells. This was the
"weak wall-guided wave" of wave 12, and it is strong when excited
directly: a point source in a circular guide whose walls end one cell
before the CPML left the energy at −3 dB after 20 000 steps, standing
along z with `λ/2` = 12 mm at 12.5 GHz (staircase −92 dB), the TE11 port
run settled at −26 dB after 30 000 steps (staircase −50 dB in 2901).
Merging and interpolation alike; plain Dey–Mittra (`τ` = 0, Courant
factor 0.1) decays to −86 dB. Next to a CPML (plates running into it,
hard merge groups in the layer, interpolated faces outside) the same
fields grew exponentially after ~7500 steps (~6 dB per 500 steps), on the
Cartesian and the cylindrical grid alike (a wave-12 defect; a CFS shift
of 0.5 S/m hid it).

So every interpolated face and every small member of a merge group keeps
a deviation current: `i_s = Σ w·g + d_s` (merge: `i_m = g + d_m`). `P` is
square and invertible again, and the deviations get an extra inductance
`a·L` (`ConformalOptions::deviation_area` = 0.5): `M = PᵀLP + diag(a·L)`,
`M_dd = (f_A + a)·L`. The Schur complement of a deviation is at least
`a·L` (the hosts' mass contains `f_A·L·w·wᵀ`), so its local resonance is
that of a face with open fraction `a`, like the area floors that were
stable, and the time step is unchanged (`s` up to 0.997 on the plates,
0.97–0.99 elsewhere). For smooth fields the deviation is the
interpolation error, so accuracy is not lost: every cavity below has
equal or smaller errors, and the sphere's modes, pre-asymptotic before,
converge at second order. (A first try with `M_dd = max(f_A, a)·L` left
merged members with `f_A` near `a` a Schur mass near 0: `s` = 4.6 on the
coarse sphere.) Merge groups with deviations are solved in blocks like the
hosts. Measured (`tests/conformal_modes.rs`,
`guide_walls_ending_before_pml_decay`, `plates_into_cpml_long_run_is_stable`;
`tests/cylindrical_conformal_pml.rs`): walls ending one cell before the
CPML reach −50 dB in 2651 steps (staircase 2901; point source −92 dB
like staircase); the plates settle at −123 dB (staircase −125 dB) with no
growth; 12-cell z wedge with z and `ρ_max` layers −74 dB after 15 000
steps (staircase −77 dB, before: exponential growth).

Small faces without a valid stencil are merged (with deviation currents;
tens on a sphere, none on the 2-D cavities); a small face that cannot
merge either keeps its
current with `f_A` raised to `τ`. A power iteration on
`S = √cb·(PK)ᵀ·dt·(PᵀLP)⁻¹·(PK)·√cb`, started on the cut cells, checks
`s < 0.999` before the first step; if not, `τ` is raised ×1.25 (up to 4
rounds) and a warning is issued if it still fails. The power estimate is
from below (not a proof); the long-run energy tests are the direct check.

**Absorbing layers.** A circular guide with conformal walls running into
CPML ends never decayed (87 % of the peak energy after 30 000 steps,
~20× slower than staircase). Causes and fixes:

1. The CPML tunes `σ_max` to the layer's mean refractive index read from
   the coefficients (§5.6); a cut edge carries `C/f_e`, so slivers made the
   layer read as `ε_r ~ 1/f_e` and the absorption vanished: the guide was
   effectively closed and its energy constant. The CPML is now built from
   the coefficients before the conformal scaling.
2. With the wall through mesh nodes (`x = ±a`, `y = 0`, and the 6-8-10
   points of `a = 10` at 1 mm cells) the edges on the surface were open
   (`f_e = 1`) between zero-area faces: an undamped local resonance at
   13 GHz holding −18 dB of the energy. Snapping and PEC surface edges
   (above) remove it.
3. Interpolation inside the stretched layer grew exponentially; small faces
   in a CPML layer are merged and no stencil reaches into one. The CPML only
   corrects faces with a kernel `db`, so the merge groups apply its H-side
   correction (`Ψ ← bΨ + aT`, `+ (1/κ − 1)·T + Ψ` in the group's
   circulation) to their members themselves (`ConformalPec::set_cpml`);
   without it a conformal coax read a 30 % impedance error, from the
   reflection of the unstretched faces at its conductors. With deviation
   currents the correction of a member goes to every reduced current
   driving it (group and deviation).
4. The cylindrical PML of `ρ_max` (§5.15.2) filters each face's whole
   circulation by `h⁻¹`; it skips merged faces (`db = 0`) as the CPML
   does. Merge groups now apply it themselves (wave 13): the layer's two
   stages and, in the corners, the z / α layers' stretch of each
   derivative term inside them (`CylLayer::h_face`, the same arithmetic as
   the layer), added to the group's (and deviation's) circulation as
   `Y − T`. Measured (`tests/cylindrical_conformal_pml.rs`): a radial wedge
   guide (walls on radial planes between α lines, PMC z plates, TEM
   `E_α`) into PML(8) at 50 mm reflects −89.4 dB against a 560-cell longer
   reference (staircase −90.3 dB); without the stretch the run diverged.
   Rejected: area floors instead of merging in the layers (stable, but
   the floored faces' wrong mass mismatches the layer: coax impedance
   +0.51 / +0.25 % at 10 / 20 cells instead of +0.03 / +0.002 %).

Measured (`tests/conformal_modes.rs`, guide `a` = 10 mm, 1 mm cells, 50
cells along z, CPML(8), TE11 port, −50 dB criterion): staircase 6401
steps, conformal 6701 steps (9101 before the deviation currents) (2.2 s; 0.25 ms per step against 0.075 for
staircase, the serial operator covering ~22 % of this small grid's faces).
Energy look-ahead (wave 14): the merge groups' share of the invariant,
`½·g^{n+½}ᵀ·M·g^{n+3/2}`, evaluates `g^{n+3/2}` in `post_e` from the
circulations of `u^{n+1}`; it now adds the CPML and cylindrical-PML
corrections there too, without advancing their state, so it is bit for
bit the current the next `post_h` computes (unit test
`energy_lookahead_includes_the_pml`; before, the groups in a layer
evaluated an unstretched current). The kernel's faces still use their
pre-correction `i^{n+½}` (§5.8), unchanged. The step counts of the guide
runs (6701, 2651) did not change at the criterion's 100-step resolution.

**Validation** (`tests/conformal_modes.rs`, `cavity_table` / `coax_table`,
release, `--ignored`; PEC block with a vacuum region carved out, centres off
the nodes; 2-D cross sections with 2 cells along z between PMC (TE) or PEC
(TM) faces; resonance = windowed-spectrum peak, Courant factor 0.99).
Degenerate modes are measured with source = probe on orthogonal directions
whose signals are summed, so that by reciprocity every orientation of the
mesh-split pair or triple has equal weight and the peak is their mean.
"Merging" is the scheme before wave 12, "wave 12" the interpolation
without deviation currents, on the same models:

| mode | cells | staircase | merging | wave 12 | now (deviation currents) | order |
|---|---|---|---|---|---|---|
| circle TE11 (`r` = 10 mm) | 10 / 20 / 40 per radius | −6.74 / −3.27 / −1.66 % | +0.567 / +0.260 / +0.124 % | +0.0757 / +0.0182 / +0.0054 % | +0.0308 / +0.0086 / +0.0023 % | 1.8, 1.9 |
| circle TE21 | 10 / 20 / 40 | −8.72 / −4.20 / −2.11 % | −0.371 / +0.413 / +0.210 % | +0.0299 / +0.0052 / +0.0024 % | −0.0635 / −0.0146 / −0.0036 % | 2.1, 2.0 |
| circle TM010 | 10 / 20 / 40 | −3.76 / −1.91 / −0.91 % | −0.081 / −0.020 / −0.0046 % | −0.0759 / −0.0163 / −0.0039 % | −0.0618 / −0.0151 / −0.0038 % | 2.0, 2.0 |
| circle TM110 | 10 / 20 / 40 | −3.90 / −1.95 / −0.92 % | −0.236 / −0.059 / −0.0144 % | −0.210 / −0.059 / −0.0138 % | −0.225 / −0.0545 / −0.0136 % | 2.0, 2.0 |
| rectangle 20 × 13 mm rotated 17°, TE10 cutoff | 1 / 2 / 4 cells per mm | −8.28 / −4.60 / −2.28 % | +0.520 / +0.211 / +0.106 % | +0.189 / +0.041 / +0.018 % | +0.130 / +0.024 / +0.010 % | 2.5, 1.3 |
| same, TE01 cutoff | 1 / 2 / 4 | −9.86 / −5.18 / −2.52 % | +0.326 / +0.134 / +0.068 % | +0.145 / +0.025 / +0.012 % | +0.050 / +0.0053 / +0.0054 % | 3.2, – |
| same, TM11 | 1 / 2 / 4 | −4.89 / −2.46 / −1.27 % | −0.156 / −0.036 / −0.0083 % | −0.143 / −0.032 / −0.0078 % | −0.126 / −0.031 / −0.0078 % | 2.0, 2.0 |
| sphere lowest TM (`r` = 10 mm) | 10 / 20 / 40 | −5.28 / −2.65 / −1.33 % | +0.341 / +0.145 / – % | +0.069 / +0.088 / +0.0031 % | +0.0584 / +0.0160 / +0.0041 % | 1.9, 2.0 |
| sphere lowest TE | 10 / 20 / 40 | −3.73 / −1.84 / −0.91 % | −0.036 / −0.012 / – % | +0.011 / −0.0004 / +0.0030 % | −0.121 / −0.0254 / −0.0054 % | 2.3, 2.2 |
| coax `b/a` = 5/2 mm, `Z_TEM` (2–6 GHz) | 5 / 10 / 20 per inner radius | +4.13 / +1.93 / +1.03 % | +27.4 / +15.2 / +8.05 % | +0.527 / +0.069 / +0.017 % | +0.174 / +0.033 / +0.0024 % | 2.4, 3.8 |

Stability estimates 0.90–0.99 (the TE runs 0.983–0.988, set by the bulk
at Courant factor 0.99), energy constant after the source (drift ≤ 1e-6
over 20 000 steps on the coarse sphere), bit-identical for 1 and 3
threads. The rotated rectangle's TE rows flatten at ~1e-4 where its
corners (two walls in one cell, merged faces) dominate; the TE11/TE21
rows reach a few 1e-5, the level of the bulk Yee dispersion difference
between the split partners. Wave 12's sphere TM row was pre-asymptotic
(a hump at 20 cells) and its TE row changed sign: both now converge at
second order, so the hump was the spurious static fields, not the time
step. The merging column's coax also had both CPML defects above.

**Cylindrical grids** (§5.15). Solid metal works with the same scheme on
the `(ρ, α, z)` edges and faces; the metal test is made at their
`(ρ, α, z)` points. Edge fractions are the same in parameter and physical
length (α edges are arcs of constant radius), and so are the face
fractions of ρ and α faces; a cut z face (area element `ρ·dρ·dα`) is
measured by `∮ ρ²/2 dα` over its open polygon instead of the shoelace
formula. Thin conformal sheets: §5.13.1 (cylindrical grids). Validation
(`tests/cylindrical_conformal.rs`): a circular PEC cavity (`a = 10 mm`)
whose axis is 30 mm off the grid axis (its wall is no coordinate
surface), α cells of the radial cell size at 30 mm, 2 cells in z:

| mode | cells per radius | staircase | conformal |
|---|---|---|---|
| TM010 | 10 / 20 | −3.69 / −1.91 % | −0.087 / −0.0215 % (order 2.0; wave 12: −0.122 / −0.024 %; merging: −0.117 / −0.029 %) |
| TE11 (PMC z walls) | 10 / 20 | −7.00 / −3.50 % | +0.0113 / +0.0034 % (order 1.7; wave 12: +0.096 / +0.019 %; merging: +0.586 / +0.279 %) |

stability estimates 0.84–0.94, energy constant to 3e-8 after the source,
bit-identical for 1 and 3 threads. With merging TE11 converged at first
order here as on a Cartesian mesh: a property of the merged sliver faces
carrying `H_z`, not of the metric; for these walls the annular z-face
measure itself changes the frequencies by under 0.02 %. The interpolation
works in the `(ρ, α, z)` parameters (affine there; the centroid of a cut z
face is that of its polygon in `(ρ, α)`, `O(h²/ρ)` from the area-weighted
one).

**Limitations.** Solid metal here, thin sheets in §5.13.1; metal
intruding into a face without crossing its edges and features thinner than
half a cell between the edge samples are not seen. Voltage probes on cut
edges read `E·l_open`. The power-iteration check costs a few hundred
matrix–vector products on the whole grid at setup (single thread); the
operator is serial (0.25 vs 0.075 ms per step on the 25 × 25 × 50 guide
above; the deviation currents add one unknown per small face to the
blocks).

#### 5.13.1 Thin PEC sheets

Same switch (`CompileOptions::conformal`). Applies to `Metal` primitives of
dimension 2 that csx cannot put on its PEC mask (it warns "non axis-aligned
sheet" / "not on a mesh plane" and drops them; with the option these
warnings are removed): zero-thickness boxes (any transformation), polygons
and zero-length extrusions off or oblique to the mesh planes, zero-width
cylindrical and spherical shells. Geometry in `compile::conformal::sheets`,
scheme in `engine::conformal::sheets` (`ConformalSheets` operator).

**Why splitting.** A sheet has PEC on both faces but the two sides are
independent fields. Dey–Mittra's coefficient change (§5.13) needs one
voltage per edge and one current per face, which cannot hold the field on
both sides of a sheet crossing the cell. Thickening the sheet to the
smaller side of each cell would make it a staircase again on the side that
loses its part (first order).

**Geometry.** Each sheet has a side function `φ` (plane: normal distance;
cylinder/sphere: squared distance to axis/centre minus `r²`), evaluated at
the nodes, so every node is on a definite side and the parity of the
crossings is consistent around every face. Crossings of an edge are the
roots of `φ` along it (linear/quadratic, bisection if rounding breaks the
parity); a crossing on the sheet's extent (polygon, rectangle, cylinder
length) is a **cut**, kept ≥ 1e-6 of the edge from the ends. Faces around
cut edges are partitioned with the crossings of one sheet: the perimeter
arcs between crossings alternate sides; the negative side (half-space,
cylinder or sphere interior) is convex, so all its arcs form one region,
and each positive arc is a region closed by the trace between its end
crossings. A trace not fully on the sheet (plate edge or hole in the face)
does not separate: its arc joins the negative region (a slit). Region
areas are the polygons of corners and crossings with chords for the traces
(exact for planes, second order for curves). Pieces (edge parts between
cuts) belong to the region of the arc containing their midpoint.

*Exact hits* (wave 13). A sheet through mesh nodes or edge midpoints
must not depend on how the transformation rounds: the 45° plate of the
triangle tests below (`rotate_axis(2, π/4)`) misses the far corner node
and the midpoints of the y edges on `x = a/2` by an ulp with this
platform's libm (`sin(π/4)`, `cos(π/4)` one ulp apart) and hits them
exactly on Windows (and with the exact matrix of entries `1/√2`). The
TE10 error was −0.0532 % / −0.358 % at n = 20 (−0.0160 / −0.0942 % at
40, −0.0042 / −0.0241 % at 80), the CI failure. The geometry of the
split was the same; the cause was csx's point query, which puts an
edge on the PEC mask when its midpoint lies on a sheet (within its
rounding): the three y edges through `(a/2, a/2)` became PEC, so the
cut edge lost both pieces and the two half faces their contour across
the plate. Fixes in `compile::conformal::sheets`: (1) cut edges are
taken off csx's PEC mask unless another primitive may have put them
there (solid metal containing the midpoint, or any other metal or
conducting-sheet primitive near the edge, conservatively); (2) edges
lying in a sheet (both ends and the midpoint on the surface, the
midpoint on its extent) are put on it, so a sheet along an edge is PEC
there whether csx saw it or not; (3) the side function is zero within
`1e-12` of the coordinate scale (`(s + r)²·1e-12` for the quadratic
ones), so a node on the sheet is on the positive side for either
rounding, and crossings come from the side changes between the roots
(roots within `1e-9` of an edge end move onto it), so an edge leaving
such a node to the negative side is cut at its start in both cases.
Now both give −0.0532 / −0.0160 / −0.0042 % (TM21 −0.169 / −0.0435 %,
unchanged: `E_y` carries no TM field). The only remaining difference is
the merge threshold: the half faces on `x = a/2` have `f_r = 0.5 ± ulp`
and are interpolated (182 regions) or keep their current (180), which
leaves the result the same to the digits shown. `tests/conformal_sheet.rs`
runs both rotations, so the tie is tested on every platform.

**Scheme.** Each piece `p` of a cut edge has its own voltage
`u_p = E·l_p` (`cb_p = f_p·cb`, `f_p = l_p/l`); each region `r` has its own
current with inductance `f_r·L` and the contour of its pieces and uncut
edges (the sheet contributes `E_tan = 0`). Ampère on a piece reads the
regions containing it, the transpose. The system is again
`C·du/dt = K'ᵀi`, `L·di/dt = −K'u` (`K'`: incidence of the split grid) and
the §6 invariant (plus the operator's own `½C_p u_p² + ½L_r i_r i_r'`) is
exact. Coupling with the kernel: cut edges are `ca = cb = 0` in the kernel
and updated by the operator (their stored `u` is the sum of the pieces);
owned faces are `da = 1, db = 0`, the operator writes the current of the
largest region there; an uncut edge of a cut face on another region is
corrected after the E kernel by `cb·s·(i_r − i_primary)`. Every face
around a cut edge is owned, so the kernel never reads a split value.

**Stability.** For the lossless leapfrog with positive `C`, `L` the
condition is `s = (dt/2)·√λ_max(C^{-½}K'ᵀL⁻¹K'C^{-½}) < 1`, and then the
conserved `W` is positive definite, so the fields are bounded. A region's
row of that matrix is `∝ (Σ_p f_p)/f_r` (Gershgorin): splitting adds
nothing dangerous through the pieces (small `f_p` only lowers `cb_p`) but a
region with little area and a long open contour resonates far above the
Courant limit: without remedy `s` = 13–37 on spherical shells at 2.5–5
cells per radius.
As for solid metal (§5.13), regions with `f_r < 0.5` lose their own
current. Since wave 13 they are **interpolated** like small faces: the
region's `H` is the affine interpolation, at its centroid (computed by
the compile step with the region areas: `SheetRegion::centroid`), of the
host across its most open piece and one host neighbour per in-plane axis,
all regions (or uncut faces, then taken over by the operator) **on the
same side of the sheet**: candidates are reached across pieces and uncut
edges only, which never cross the sheet. Hosts sharing regions form
blocks with `PᵀLP`, and each interpolated region keeps a deviation
current (§5.13). Where the host stencil fails (the host has no free
neighbour along an axis, as in the 45° wedge where the plate meets the
box), the best triangle of free same-side candidates within three steps
(the eight nearest to the centroid, smallest largest weight) is tried
(wave 14: 26 → 8 merged regions in the triangle tests, 17 → 9 on the
coarse sphere). Regions without a valid stencil are merged, as before,
with the
region **on the same side** of the in-plane neighbour across their
most-open piece (that neighbour can be an uncut face, which is then owned
by the operator), until the group reaches the threshold: one current with
`L_g = Σ f_r·L` and the merged contour, `K' → P·K'`, still symmetric. A
group has at least half a face of area but no more than its members'
perimeter, so its row stays near that of an uncut face. Since wave 14
each small member keeps a deviation current as for solid metal
(`i_m = g + d_m`, `M_dd = (f_r + a)·L`, soft groups solved in blocks).
Merging never
crosses the sheet (pieces belong to one side), so the two sides stay
decoupled. The power iteration of §5.13 runs on the combined system
(pieces as extra unknowns, seeded on the cut cells), with the same
threshold raising if it fails. Measured: `s` = 0.89–0.985 at Courant factor
0.99 for every case tried (the tests below and 48 random spheres, plates,
tilted cylinders and pairs of crossing plates at 8–17 cells), no floored
regions, energy drift ≤ 1e-6 over 10 000 steps.

**Validation** (`tests/conformal_sheet.rs`; probe spectrum as in §5.13;
the staircase column is the equivalent solid metal without the option,
since csx drops oblique sheets):

| cavity | cells | staircase | merged regions (wave 12) | conformal sheet now |
|---|---|---|---|---|
| cylindrical shell TE11 (r = 10 mm, PMC z faces: `H_z` on the cut z faces) | 5 / 10 / 20 per radius | −13.9 / −6.74 / −3.27 % | +1.41 / +0.567 / +0.260 % (order 1.2) | +0.093 / +0.0308 / +0.0086 % (order 1.6, 1.85) |
| right isosceles triangle (legs 20 mm) behind a 45° plate, lowest TE (`cos πx/a + cos πy/a`) | 20 / 40 / 80 | −4.95 / −2.51 / −1.26 % | +2.29 / +1.19 / – % (order 0.9) | −0.0660 / −0.0168 / −0.0042 % (order 1.97, 2.0; wave 13: −0.0532 / −0.0160 / −0.0042 %) |
| cylindrical shell TM010 | 5 / 10 / 20 per radius | −6.84 / −3.76 / −1.91 % | −0.323 / −0.081 / −0.0197 % | −0.279 / −0.0618 / −0.0151 % (order 2.2, 2.0) |
| same triangle, TM21 | 20×21 / 40×41 | −2.54 / −1.26 % | −0.160 / −0.042 % | −0.169 / −0.0435 % (order 1.96) |
| spherical shell lowest TM (r = 10 mm) | 5 per radius | −10.4 % | +0.31 % | +0.106 % |

Wave 13 interpolation without the deviation currents gave TE11
+0.389 / +0.0763 / +0.0185 % (order 2.2) and the triangle TE
−0.0315 / −0.0131 / −0.0038 % at 20 / 40 / 80 (first-order merging at
the plate's two ends, where it meets the box, makes 20 pre-asymptotic;
the order at 80 → 160 was 1.9). With them the TE errors drop further
(TE11 now equals the solid-metal circle of §5.13 to the digits shown).
`s` = 0.89–0.988, energy drift ≤ 5e-7 over 20 000 steps.

The field behind the plate stays below 1e-5 of the driven side (f32
rounding of the corrections). Energy drift over 20 000 steps after the
source: 3.0e-7 (coarse spherical shell), 2.2e-7 (finite square plate tilted
about two axes, with plate edges inside cut faces). Bit-identical results
for 1 and 4 threads. Without sheets the solid-metal scheme is bit-identical
to before (probe signals compared).

*Wave 14, merged regions.* With the deviation currents and the wider
stencil the triangle TE row reads −0.0660 / −0.0168 / −0.0042 % (order
1.97, 2.0; before −0.0532 / −0.0160 / −0.0042 %, order 1.7, 1.9): the
coarse level was lower by a cancellation of the first-order corner
merges, the finest is unchanged. The result does not depend on `a` in
0.25–3 (−0.0659 … −0.0667 %; `a` = 30 puts the deviations' local
resonance near the mode, −0.088 %). TM21 unchanged (−0.1688 /
−0.0435 %), spherical shell +0.113 % (+0.106 %), cylindrical-grid shells
unchanged to the digits of the table below.

**Absorbing layers** (wave 14; `tests/conformal_sheet_pml.rs`,
`tests/cylindrical_conformal_pml.rs`). The kernel's CPML skips the sheet
operator's unknowns (cut edges `cb = 0`, owned faces `db = 0`), so cut
cells in a layer were not stretched: two oblique plates running into a
CPML diverged (+358 dB), the radial wedge of §5.13 with sheet walls into
the `ρ_max` PML too (+668 dB). The operator now applies the layers
itself, with their grading and arithmetic: the CPML recursion of each
piece's derivative term across the layer (from the regions of the faces
of the right normal), of each region's circulation term (its pieces and
edges across the layer, added to every reduced current driving it), and
of the corrections of uncut edges (the region's current minus the
primary's, whose term the CPML already stretched); the cylindrical layer
likewise (`CylLayer::e_edge`, `h_face`, the z / α layers inside it in
the corners). Small regions are interpolated inside the layers too:
merged there (as small volume faces are), the level plates below held a
field that grew from −74 dB after 4000 steps (~1 dB per 1000 steps), and
with hard merges (no deviation currents) the tilted plates decayed 40 %
slower and settled 6 dB higher. Measured (1 mm cells, plates 4.2 mm apart
tilted 0.35 rad about x, or level 0.03 and 0.17 of a cell off the mesh
planes, CPML(8), against a domain 160 cells longer): reflection
−82.2 dB tilted, −78.0 dB level (plates on the mesh planes, csx's mask:
−78.5 dB); 36 000 steps: both settle at the static level the source
leaves (−89.5 / −92.7 dB; mesh-plane plates −92.5 dB) and stay. Radial
sheet walls into `ρ_max` PML(8): −89.3 dB (solid conformal walls
−89.4 dB, staircase −90.3 dB); with z PML in the corners 30 000 steps
end at −77.8 dB without growth (solid walls −76.5 dB). 1 and 3 threads
bit-identical.

**Junctions** (wave 14; `compile::conformal::sheets::junction`,
`tests/conformal_sheet_junction.rs`). One sheet used to partition each
face: where a second sheet crossed it, or a sheet ended on another sheet
or on solid metal inside the face, the ending sheet's trace did not
separate and its two sides shared the face's current. Faces cut by two
or more sheets, or touched by solid metal, are now divided by the planar
arrangement of all traces: the face sides (split at cuts and metal
transitions), each sheet's chord restricted to its extent (16 samples,
bisected), the metal's open-polygon chords. A segment end within `1e-6`
of the face size from another segment is moved onto it; a sheet whose
extent ends on another sheet's surface or on the metal (tested on the
surfaces) is extended along its trace to that surface's chord (at most a
quarter of the face: chords of curved traces are off by their sagitta).
Segments are split at intersections, dangling ends pruned (plate edges,
which do not separate), the bounded faces of the graph are the regions,
those in the metal dropped (point next to the longest side against the
open polygon); each piece goes to the region with the longest open part
of it, and pieces of edges partly in metal get their open lengths
(`SheetCutEdge::open`, so `cb_p = cb·l_open/l` instead of the edge
fraction times the piece fraction). Faces cut by one sheet without metal
keep the arc partition (unit test: the arrangement reproduces it). A
second defect showed up where a septum touches a curved solid wall: the
sheet system took over, as hosts, the uncut faces the solid-metal scheme
needed for its small faces, which then merged among themselves below the
threshold (`s` = 1.046 whatever the threshold; TE run diverged). The
solid-metal scheme is now built first against the sheets' cut faces, and
the faces it drives are excluded from the sheet system.
Measured (2-D cross sections, PMC z faces for TE; leak = largest |E| on
edges in the undriven part / outside a cell from the walls, against the
probe's peak):

| cavity | mode | before (h = 1 / 0.5 mm) | now (h = 1 / 0.5 mm) |
|---|---|---|---|
| box of four plates (20 × 13 mm, 17°), septum ending on two of them (T), plates ending at the corners (L); 8 × 13 part | TE01 | +3.61 / +3.09 %, leak 0.91 / 1.19 | −0.134 / −0.032 %, leak 4e-7 |
| same, all plates 1 mm past each other (X) | TE01 | +2.08 %, leak 0.62 | −0.134 / −0.032 %, leak 4e-7 |
| same (T, L) | TM11 | −0.356 % | −0.355 / −0.082 % |
| circle of solid metal (r = 10 mm) with a septum sheet along a diameter at 23°, ending on the wall or 1 mm into it; half disc | TE11 | diverged (`s` = 1.046) / −0.100 %, leak 2.4 | +0.032 / +0.0086 %, leak 1e-6 |
| same | TM11 | −0.235 / −0.056 % | −0.235 / −0.056 % |
| cylindrical grid: shell (r = 6 mm, 12 mm off the axis) with a septum plate ending on it | TE11 | – | −0.380 / −0.026 %, leak 4e-7 |

(TM fields have no `H_z` on the shared z-face regions, so the old gap
did not show in TM.) Bit-identical for 1 and 3 threads.

**Limitations.** Open polyhedra, boxes given in cylindrical coordinates
and conducting (`ConductingSheet`) oblique sheets are not handled. A
trace that bulges into a face without crossing its edges is not seen,
nor a sheet ending inside a face more than a quarter of the face short
of another surface (it stays a slit). Sources and lumped elements on cut
edges are not supported (the operator owns those voltages). Voltage
probes on a cut edge read the sum of its pieces. Small volume-cut faces
remain merged inside CPML layers (§5.13); for sheets, merging there was
measured unstable, so the remaining merges in layers are the stencil
fallback only.

**Cylindrical grids** (§5.15; `compile::conformal` submodule
`sheets::cyl`). The same Cartesian sheet surfaces are cut against the
`(ρ, α, z)` edges mapped to Cartesian points: ρ and z edges are straight
segments as before, α edges arcs, whose crossings are the sign changes
of `φ` at 16 samples along the arc, bisected. Faces are split in their
parameter rectangle with straight chords (exact for a coaxial shell on a
z face, whose trace is `ρ = r`; second order otherwise); ρ and α faces
have a constant metric, z faces measure their polygons with `ρ·dρ·dα`
(`∮ ρ²/2 dα`, closed form along straight parameter segments). Sheets on
a mesh surface of the cylindrical grid keep csx's mask. The stored
copies of a physical edge (seam of a closed azimuth, axis) share one set
of pieces in the engine operator (merged coefficients `1/cb = Σ 1/cb_j`,
`ca/cb = Σ ca_j/cb_j`); the kernel zeroes every copy and the operator
writes all of them, so the periodic and axis merges keep the value; H
faces stay per copy as on the uncut grid. Sheets cutting edges inside
multigrid zones are dropped with a warning. Validation
(`tests/cylindrical_conformal_sheet.rs`, TM010 against `j01`): a coaxial
shell `r` = 10.3 mm between ρ lines (16 α cells, axis; it cuts the seam
copies) −0.254 / −0.037 % at 1 / 0.5 mm (merged regions: −0.110 /
−0.037 %; the small inner side at 1 mm, none at 0.5 mm); the same shell at
`r = 10 + 0.3·h` (the inner side at 0.3 of a cell at every step)
−0.254 / −0.066 / −0.017 % at 1 / 0.5 / 0.25 mm (order 1.95, 1.97; merged:
−0.110 / −0.034 / −0.0095 %; interpolated with the face-centre α dual
edge: +0.21 / +0.081 / +0.038 %, first order, see §5.13); an off-axis
shell (radius 6 mm, centre 12 mm off the axis, across the seam, annular
grid with α cells of `h` at 12 mm) −0.346 / −0.086 / −0.0206 % at
1 / 0.5 / 0.25 mm (order 2.0, 2.1; merged: −0.327 / −0.087 %); stability
factor 0.88–0.98, energy constant to 1e-7 after the source, 1 vs 3
threads bit-identical. Three refinements: `cylindrical_sheet_table`
(`--ignored`).

### 5.14 Thin wires

A line of PEC edges (csx's staircase of a `Curve` or thin `Wire`) acts like
a round wire of the *lattice radius* `r_eq` of its node, whatever the real
radius: the potential of one node of the infinite 2-D lattice relative to
far points is `(1/2π)·ln(ρ/r_eq)` per unit conductivity. From the lattice
Green's function, `r_eq = e^(−γ)/(2√2)·Δ = 0.1985·Δ` for square cells
(`engine::wire::lattice_radius`, any aspect ratio). Antenna impedances depend
on `ln(Δ/r0)`, so this matters (a dipole of `Δ/100` and one of `Δ/5` would
otherwise be the same). `engine::wire` (`ThinWire`) models straight,
axis-aligned wires with `0 < r0 < r_eq` by changing only edge materials,
which keeps the scheme the symmetric Yee scheme (exact energy invariant,
every operator unchanged):

- Per unit length, the transverse E field is a resistor network (edge
  conductance `ε·l*/l`) and the transverse H field the same network for the
  flux function (`l*/(μ·l)` across each face). Dividing the four
  conductances from the wire node to its neighbours by `g` adds
  `(g − 1)/Σc` in series (exact on square cells by symmetry), so
  `g = 1 + Σc/(2π)·ln(r_eq/r0)` (square cells: `1 + (2/π)·ln(r_eq/r0)`) turns
  the node into a wire of radius `r0`: radial E edges `ε/g`, the four faces
  around each wire edge `μ·g`. `L'C' = με` is unchanged, so the wave speed
  along the wire is too. (The classic contour-path model of Umashankar and
  Taflove assumes the `1/r` profile across the first cell, `g = ln(Δ/r0)/2`,
  i.e. `r_eq = 0.135·Δ`, and corrects only the H update, so it leaves the
  capacitance and the TEM impedance unchanged.)
- The four axial faces around each wire node also get `μ·g`: with them no
  local loop of modified elements is stiffer than the unmodified lattice, so
  the Courant limit still holds (without them a run at `g ≈ 2.5` diverged).
- **Open ends.** A full cell of wire at the end node puts the open end half a
  cell beyond it (as for any staircase), and the lattice capacitance of the
  E edge continuing the wire (the cap edge) loads the end like a top hat
  that is far too large for a thin wire (a dipole on 10 mm cells resonated
  8 % low). Open ends get half a cell at the end node (radial edges
  `ε/(2g)`, axial and end-segment faces `μ·2g`) and the cap edge scaled like
  a lattice whose staircase radius is `r0` (`ε·r0/r_eq`, its faces
  `μ·max(2g, r_eq/r0)` for stability). The open end is then at the end node.
- Compile (`compile::wire`): every polyline segment of a metal `Wire` that
  runs along an axis on mesh lines becomes a run between the nearest mesh
  lines; lumped edges on the same line touching a run (dipole feed gaps,
  monopole bases) join it; contiguous runs of one radius form one
  `ThinWire` (`Compiled::thin_wires`). A wire with any oblique segment
  goes to the oblique model of §5.14.1 as a whole. Off-mesh segments, radii
  at or above `r_eq` (below half a cell) and cylindrical structures keep the
  plain staircase with a warning; wires sharing a node warn. The mesh check
  looks past the correction (`ThinWire::remove`) when it estimates the
  refractive index.

Limits: wires closer than about three cells to each other or to other
conductors, bends and junctions are outside the isolated straight-wire
assumption; on rectangular or graded cells `(g − 1)/Σc` is approximate
(0.4 % of `Z0` on 1 × 1.5 mm cells); fields polarised across the wire see a
one-cell rod of `ε/g`, `μ·g` that a real thin wire does not present.

**Validation** (`tests/thin_wire.rs`, `tests/thin_wire_dipole.rs`):

| case | result |
|---|---|
| staircase wire in a 10 mm square tube, `Z0` vs closed form `η/(2π)·ln(1.0787·D/d)` at `r_eq` | 197.72 vs 197.99 Ω (−0.14 %), confirms `r_eq` |
| thin wire, `r_eq/r0` = 5, 50, 1000: `Z0 − Z0(staircase)` vs `η/(2π)·ln(r_eq/r0)` | within 6.4·10⁻⁵ of `Z0`; within 0.1 % of the closed form |
| same, phase constant vs the 1-D Yee line | within 9.3·10⁻⁵ |
| 1 × 1.5 mm cells, `r_eq/r0` = 10, 100 | 0.39 %, 0.36 % |
| dipole (170 mm, 10 mm gap), `a` = 0.496 mm on 5 / 10 mm cells vs staircase on 2.5 mm | `f_res` −0.53 / +0.31 %, `R_in` −0.22 / −0.28 %, `dX/df` +1.8 / +0.1 % |
| `a` = 0.248 mm on 2.5 / 5 / 10 mm vs staircase on 1.25 mm | `f_res` −0.32 / −0.22 / +0.71 %, `R_in` −0.13 / −0.23 / −0.74 % |
| closed tube, `r_eq/r0` = 1000 with a feed gap, Courant 0.999, 2·10⁴ steps | energy within 2·10⁻⁶, bounded; 1 vs 4 threads bit-identical |

The plain staircase on 10 mm cells (`a ≈ 2 mm`) gives a dipole slope of
661 Ω/GHz; the thin wire of 0.496 mm on the same mesh 974 Ω/GHz (reference
973), and 0.198 mm 1160 Ω/GHz with a 1.4 % higher resonance.

#### 5.14.1 Oblique thin wires

The material model above needs the wire on a line of mesh edges. Wires of
any orientation and position (Cartesian grids) use a separate model
(`engine::oblique_wire`, operator `ObliqueWires`; compile rules in
`compile::wire`): the wire carries its own unknowns on a 1-D polyline
grid that is independent of the mesh, as in the thin-wire models of
Holland and Simpson (1981) and the hybrid wire models of Edelvik et al.
(2003), coupled to the Yee lattice by interpolation and spreading with
one and the same matrix. We chose this over a staircase with corrections:
a staircase of an oblique wire is a meander √2…√3 times longer than the
wire, whose inductance and resonance no per-edge material change can
repair.

**Unknowns.** The polyline is cut into segments of at most half the
local cell size (polyline vertices are nodes; coincident end points of
different wires are one junction node). Per segment `s` a current `I_s`,
per node `j` a correction potential `V_j`.

**Coupling.** `P_{e,s} = ∫_s w_e·dl` with the lowest-order Whitney edge
functions `w_e` (constant along edge `e`, bilinear across: the usual
trilinear interpolation of the Yee components; exact per piece of the
segment between mesh planes with two Gauss points). The wire sees
`(Pᵀu)_s = ∫_s E·dl`; the grid gets `−P·I` in Ampère's law
(`C·du/dt = Kᵀi − P·I`). Since `∇λ_n` (node hat) is the sum of the Whitney
functions of the edges at `n`, the deposit is charge conserving: the grid
charge is exactly the wire node charges spread with the trilinear hats.

**Wire equations.** `L_s·dI_s/dt + R_s·I_s = (Pᵀu)_s − (V_b − V_a) +
V_src,s` and `C_j·dV_j/dt = (current into j)`; `R_s`, `V_src,s` only on a
feed segment (`WireFeed`: delta gap with internal resistance; port
voltage `V_src − R·Ī`, current `Ī`). Ends on a conductor (PEC edge,
lumped-element edge, PEC outer face: any edge at a node end point, the
edge for an end on an edge, all four edges for an end inside a face) have
`V = 0`; other ends are open, at the end node.

**Time stepping and stability.** `I`, `V` live at integer times and are
advanced with the trapezoidal rule, implicitly coupled with the E update
(the lumped-branch scheme of §5.5 for a whole network). With the kernel
result `u*` and `Ī = (Iⁿ + I^{n+1})/2`:

```
u^{n+1} = u* − cb·P·Ī
[2L/dt + R + ½·PᵀD_cb·P + (dt/2)·G·C⁻¹·Gᵀ]·Ī = (2L/dt)·Iⁿ + Pᵀ(u* + uⁿ)/2 − G·Vⁿ + V_src
I^{n+1} = 2Ī − Iⁿ,   V^{n+1} = Vⁿ + dt·C⁻¹·Gᵀ·Ī
```

(`G`: node-to-segment incidence, `D_cb` the edges' `cb`). The matrix is
symmetric positive definite, constant and banded along the wire; it is
factored once (envelope Cholesky) at the first step. Multiplying the E
update by `ū = (uⁿ + u^{n+1})/2` and the wire equations by `Ī` and
`V̄` gives

```
(W^{n+1} + W_w^{n+1}) − (Wⁿ + W_wⁿ) = dt·Ī·V_src − dt·R·Ī² (lossless grid),
W_w = ½·Σ L_s·I_s² + ½·Σ C_j·V_j²,
```

with `W` the leapfrog invariant of §6: the coupling terms cancel exactly.
`W` is positive definite under the ordinary Courant condition and `W_w`
always, so **the time step stays the Yee one**, for any radius, segment
length and orientation, and the scheme is energy conserving (the
operator reports `W_w`). An explicit leapfrog of `I` and `V` would add
a local stiffness `(ω·dt/2)² = dt²/(4·L_s·C_e)` on the coupled edges,
about `0.5·(Δ/ℓ)/ln(r_eq/r0)` on cubic cells at `S = 0.99` (segment
length `ℓ`), on top of a lattice already at 0.98 of its bound; it would
need a reduced time step. The implicit coupling avoids any reduction.

**In-cell corrections.** The grid already holds part of the wire's
field: that of the deposited (smeared) current and charge. `L_s` and
`C_j` add the rest, so that the total is the reduced-kernel thin wire of
radius `r0`. For a uniform charge density and current on the wire within
a window of five local cell sizes:

- `k(x) = (1/4π)∫ ds/√(|x(s) − x|² + r0²) − φ_lat(x)` is the potential per
  unit line charge (times `ε`) that the lattice misses at `x`; `φ_lat`
  is the potential of the node charges spread with the hats, propagated
  with the lattice Green's function of the discrete Laplacian
  (`−Δ_h G = δ/V`, the static Yee lattice for the charge; computed from
  the Bessel-function integral `G(n) = (1/V)∫₀^∞ Π_d e^(−2t/h_d²)·
  I_{n_d}(2t/h_d²) dt`, Watson's `0.2527310/h` at the origin for cubic
  cells), sampled with the same hats. `C_j = ε·∫ ds/k(s)` over the node's
  half segments (3 Gauss points each).
- `L_s = μ·[(1/4π)∫_s∫ t̂·t̂'/√(|x − x'|² + r0²) − Λ_lat,s]`, `Λ_lat` the
  flux of the Whitney current deposit through the lattice (`−Δ_h A = μ·J`
  per component in the Coulomb gauge) seen by the segment's own weights.

The lattice is taken uniform with the spacings of the cell containing the
node (segment midpoint): exact on uniform meshes, an approximation on
graded ones. For a straight wire on a mesh line the corrections give back
the lattice radius of §5.14 (`r_e` 0.2003 vs 0.1985·Δ, the window
truncation), at a cell centre 0.717·Δ (2-D estimate from the bilinear
spreading, `r_eq·exp(2π·(1/8 + 1/(4π)))` = 0.718·Δ); an oblique wire
passes through all positions in between, hence corrections per element.
Both corrections must be positive (roughly `r0 < 0.15·Δ`); thicker
wires keep the csx staircase with a warning.

Why `∫ds/k` over the node's half segments rather than `k` at the node:
at an open end `k` drops to about half (the near field is one-sided), and
using the end-point value made the end node's capacitance that of a
whole interior node, i.e. half a segment of extra wire: a dipole's
resonance was then first order in the segment length (−6.0 / −3.4 / −1.9
/ −1.2 % for 10 / 5 / 2.5 / 1.25 mm segments on 10 mm cells). With the
integral it is converged at half a cell (below).

**Compile.** A metal csx `Wire` (radius > 0, Cartesian) with at least
one segment not along a mesh axis goes to this model as a whole, if the
corrections are positive; it is removed from the model before csx's
discretisation (no staircase; csx's warnings are renumbered to the
original primitive indices). Axis-aligned wires keep the material model
unchanged. Ends at lumped-element edges (a feed gap given as a csx lumped
port) and on metal are connected. There is no csx form of a wire feed;
`WireFeed` is engine-level.

**Validation** (`tests/oblique_wire.rs`, `tests/oblique_wire_dipole.rs`
`--ignored` in release). Dipole 170 mm, radius `a` = 0.496 mm, 10 mm gap,
50 Ω, on uniform 10 mm cells; reference: the material model on the same
cells (0.8381 GHz, 72.37 Ω, 974 Ω/GHz; the 2.5 mm staircase gives
0.8355 GHz, 72.58 Ω, 973 Ω/GHz):

| oblique model, wire | `f_res` | `R_in` | `dX/df` |
|---|---|---|---|
| along z, on a mesh line | 0.8294 GHz (−1.05 %) | 72.08 Ω (−0.40 %) | 990 (+1.7 %) |
| along z, cell centre | 0.8334 GHz (−0.57 %) | 71.25 Ω (−1.56 %) | 986 (+1.2 %) |
| 30° in xz | 0.8311 GHz (−0.84 %) | 71.60 Ω (−1.07 %) | 987 (+1.3 %) |
| 45° in xz | 0.8323 GHz (−0.70 %) | 71.56 Ω (−1.13 %) | 985 (+1.1 %) |
| 30° in xz, off-mesh | 0.8334 GHz (−0.56 %) | 71.39 Ω (−1.35 %) | 987 (+1.3 %) |
| along (1, 2, 3), off-mesh | 0.8333 GHz (−0.57 %) | 71.46 Ω (−1.26 %) | 985 (+1.1 %) |

Spread of the resonance over orientations and positions 0.49 %; all
within −0.7…−0.25 % of the fine staircase. Segment length 10 / 5 / 2.5 /
1.25 mm (along z): 0.8256 / 0.8298 / 0.8304 / 0.8306 GHz. Mesh 5 / 10 /
20 mm: on a mesh line 0.8310 / 0.8294 / 0.8245 GHz, at a cell centre
0.8338 / 0.8334 / 0.8337 GHz (the dipole is 8.5 cells long on 20 mm).
Fast test (debug, 20 mm cells): 45° and (1, 2, 3) against the same wire
along z within 0.9 % (`f_res`), 2.2 % (`R_in`), 1.1 % (slope). Energy:
closed PEC box, three wires (bend, junction, an end standing on the
floor, a lossless feed, `r0` = Δ/1000 and Δ/20), Courant 0.999, 15 000
steps after the source: drift within 3.4·10⁻⁸ with 17 % of the energy in
the wire state; 1 vs 4 threads bit-identical. Open 45° dipole with Mur
boundaries, 50 000 steps: the energy decays to the static residue of the
pulse (3.7·10⁻⁸ of the peak) and stays constant. Through csx with the gap
as a lumped port and the arms tilted 30° from its terminals: +2.6 %
`f_res`, +6.2 % `R_in`, −6.3 % slope (the gap is then the lattice's
lumped edge, a 10 mm piece of `r_eq ≈ 2 mm` wire, and the arms are offset
by the gap).

**Limits.** Remaining position dependence (on a mesh line the resonance
is 0.4 % low on 10 mm cells, 1.1 % on 20 mm) comes from the local
(diagonal) correction near the ends and the feed, where the charge is
far from uniform over the correction's range (about a cell); a full
correction matrix would remove it at the cost of a wider band. Graded
meshes use the local cell's spacings; media are assumed homogeneous
around the wire (`ε_r`, `μ_r` averaged over the coupled edges); wires
closer than a few cells to each other or to other conductors and wire
ends in the PML (the correction line is not absorbed) are not covered;
wire energy is reported, wire losses (finite conductivity) are not
modelled.

**Cylindrical grids** (`engine::oblique_wire` submodule `cyl`). The wire
stays a polyline of straight Cartesian segments. The coupling is the
parameter-space Whitney 1-form of the `(ρ, α, z)` cells integrated along
each segment's image, `P_{e,s} = ∫_s W_e dq_d/Δq_e` (`q_d` ∈ ρ, α, z,
`Δq_e` the parameter length): the voltages are circulations, so
`(Pᵀu)_s = ∫_s E·dl` exactly as before, and charge conservation is the
same topological identity. Segments are split at ρ cylinders, α
half-planes and z planes and integrated with 5-point Gauss–Legendre.
Seam and axis edge copies are coupled as one physical edge (canonical
copy, merged `cb`; the update is written into every copy). The in-cell
corrections take the lattice locally uniform with the physical spacings
`(Δρ, ρ_c·Δα, Δz)` in the unrolled frame `(ρ, ρ_c·α, z)`, whose hats are
exactly the parameter hats; the kernels use the true geometry. Wires
must stay out of the first radial cell around the axis. On cylindrical
grids every csx metal `Wire` of radius > 0 goes to this model (radial
and z-parallel ones too: the material model of §5.14 is Cartesian).
Validation (`tests/cylindrical_oblique_wire.rs`): a closed annular
cavity with wires across the seam (bend, junction, grounded end) keeps
its energy to 8e-8 over 10 000 steps at Courant 0.999 (1 vs 3 threads
bit-identical); the 170 mm dipole on 20 mm `(ρ, α, z)` cells at
`R0` = 600 mm (`R0·Δα` = 20 mm) along z, at 45° in the ρz plane and along
(1, 2, 3) agrees with the same wire on 20 mm Cartesian cells within
0.09 % (resonance), 0.7 % (`R_in`) and 0.7 % (slope); 1200 mm alike
(0.09 / 0.9 / 1.4 %). With strong curvature the agreement degrades: at
`R0` = 300 mm (the domain reaches ρ = 60 mm) along z +0.5 % / −1.7 %
(resonance / `R_in`, the curved environment and the absorbing faces of
the sector), along (1, 2, 3) +0.2 % / −3.9 % (the correction window,
5 cells, is a third of `R0`).

### 5.15 Cylindrical coordinates

A csx grid with `CoordSystem` 1 becomes an engine `Grid::cylindrical`: axes
`(ρ, α, z)` (`Axis::X`, `Y`, `Z`), `ρ` and `z` in metres, `α` in radians
(`ρ ≥ 0`, azimuth span `≤ 2π`). Code: `engine::cylindrical`,
`compile::cylindrical`; tests: `tests/cylindrical.rs`,
`tests/cylindrical_sim.rs`.

**Geometry, not kernels.** The integral form (§5.2) only sees circulations
of `u = ∫E·dl` and `i = ∫H·dl`; all geometry is in the coefficients
`C = ε·A*/l`, `L = μ·A/l*`. The grid supplies the physical lengths and
areas (`Grid::edge_length`, `dual_edge_length`, `dual_face_area`,
`face_area`):

| edge | primary length `l` | dual face `A*` (E) / primary face `A` (H) | dual length `l*` |
|---|---|---|---|
| ρ | `Δρ` | `ρ_{i+½}·Δα*·Δz*` | `Δρ*` |
| α | `ρ_i·Δα` | `Δρ*·Δz*` | `ρ_{i+½}·Δα*` |
| z | `Δz` | `Δα*·(ρ_hi² − ρ_lo²)/2` (dual lines around `ρ_i`) | `Δz*` |
| ρ face (H_ρ) | | `ρ_i·Δα·Δz` | |
| α face (H_α) | | `Δρ·Δz` | |
| z face (H_z) | | `Δα·(ρ_{i+1}² − ρ_i²)/2` | |

csx's discretisation already uses the Jacobian in its volume weights; the
per-cell engine path (`CellMaterials`) weights the quarter cells around a z
edge by their annular areas. The Cartesian path is untouched (bit-identical).

**Closed azimuth.** An `α` span of 2π makes the α faces periodic (§5.6.1),
whatever the boundary settings say; absorbing boundaries configured there are
dropped with a warning.

**Axis.** If `ρ_0 = 0`, the `ρ_min` face is the z axis (its boundary
setting is ignored, built like PMC). α edges on the axis have zero length and
H_ρ faces zero area: neither exists (`ca = cb = 0`, `da = db = 0`). The axis
z edge is one edge stored once per α index. Ampère's law over the dual disc
`ρ ≤ ρ_½` gives `C·du/dt + G·u = Σ_j i_α(ρ_½, α_j)` with `C = Σ_j C_j`,
which is exactly the `1/cb`-weighted mean of the copies' kernel results (the
derivation of the periodic merge). The engine applies that merge after every
E update, after the periodic merge; the field energy sums the copies
(`Σ C_j u² = C u²`), so the energy invariant holds (measured drift
`2.5e-8` over 5000 steps).

**Copies of an edge.** The axis z edge and the edges on the α faces of a
closed grid exist in several stored copies (`engine::cylindrical::e_edge_copies`).
Sources and elements act on the physical edge: the simulation mapping puts
an additive source on every copy, a lumped element on the canonical copy
only. The element works with the merged edge's `cb` and writes its result to
all copies, so the merge after its hook keeps it. csx spreads an element over
the edges it lists counting every copy as a parallel edge; the per-edge
values are rescaled to the edges kept, which preserves the element's totals.

**Periodic merge fix.** A pair of copies that both have `cb = 0` (e.g. a Mur
face crossing a periodic face) is not updated by the kernel; the merge now
keeps the operator's values (mean) instead of forcing 0, which had made Mur
reflect along the seam.

**Time step.** No per-axis formula bounds the operator near the axis, where
the α arcs `ρ_½·Δα` are short. `courant_limit` computes the limit of the
discrete operator: leapfrog is stable for `dt < 2/√λ_max`, with `λ_max` of
`A = C⁻¹KᵀL⁻¹K`, estimated by Lanczos in the `C` inner product on the vacuum
operator with all non-periodic faces open (PMC — PEC faces and materials only
lower `λ_max`), including the merges, plus 0.1 % margin; cached in the grid.
On a uniform Cartesian box it reproduces the CFL formula to 1e-6; runs at
Courant factor 1.0 stay stable. Without multigrid the time step is set by
the innermost arcs (the mesh check warns); multigrid (§5.15.1) coarsens
them.

**Dumps and probes.** Components are `(ρ, α, z)`; `E_α = u_α/(ρ_i·Δα)`,
`H_α = i_α/(ρ_{i+½}·Δα*)` (the α edges next to the axis are interpolated
from the outer edge only). Dump coordinates are `(ρ, α, z)`, so VTK files
show the parameter box, not the physical cylinder.

**Field dumps of openEMS files on a closed azimuth** (black box,
per-step HDF5 dumps of a 6-cell closed grid with the axis): openEMS
writes the `N` distinct α lines (no `2π` line) and their `N` dual lines
(no dual line past `2π`), in HDF5 and VTK, and interpolates node values
(`DumpMode` 1) across the seam: `E_α` and the transverse H at `α_0`
average the cells `N − 1` and 0. ems did both one-sided and wrote `N + 1`
lines; it now follows openEMS for files (`sim::map::dump_indices`), and
interpolates periodically natively too (the seam is no boundary). A cell
centre of the first ring (`DumpMode` 2) averages `E_α` with the
non-existing axis edge as 0 in openEMS (half the outer edge; ems natively
takes the outer edge alone), mirrored for files. With openEMS's axis
update (§5.16.2) every E and H dump of that model agrees to 1.7e-6 of
its peak (fixture `dump_cyl_closed`, `tests/openems_dumps.rs`; before: a
seventh α line, 1.5e-2…8e-2 near the axis and at the seam, 0.44 for the
cell-centre `E_α`).

**Debug dumps** (`ems --debug-material`, `--debug-operator`,
`--debug-PEC`; `io::openems::debug`; fixtures `cli_debug_cyl` (closed
azimuth with the axis, a lossy dielectric, a metal block on the seam, a
lumped resistor) and `cli_debug_cyl_sector` (a sector on a graded radial
mesh), `tests/cli.rs`). openEMS (black box) writes the material and
operator dumps as VTK `StructuredGrid`s, `material_dump.vts` and
`operator_dump.vts`, with the arrays and component layout of the
Cartesian files and Cartesian `f32` points `(ρ·cos α, ρ·sin α, z)`; on a
closed grid the last α line is placed at exactly `α_0 + 2π` and one more
line, `α_1 + 2π`, follows it (26 points for 25 lines). `PEC_dump.vtp`
keeps its PolyData layout with Cartesian points, visiting the edges
with z outermost, then ρ, α innermost (Cartesian: x outermost, z
innermost); seam edges are drawn on the `2π` line, axis z edges not at
all (a metal rod on the axis gives an empty dump). ems writes the same
files with its own values: every point shows the physical edge (on a
closed grid the `2π` and the extra line repeat the `α = 0` and `α_1`
lines; edges stored in several copies, the seam and the axis z edge,
show the merged edge: ε, κ weighted by the copies' dual-face parts,
`1/vi = Σ 1/cb_j`, `vv/vi = Σ ca_j/cb_j`). Agreement (both runs with
`TimeStep` 4e-13 s): points and PEC lines identical; every material
value (with openEMS's radial weights on the file path, below) and every
update coefficient equal to float precision (`vi` 1.5e-5 of 230), except
where the codes describe different things: openEMS evaluates materials
at the literal angles (its `2π` line sees background past `2π`, its
extra line is all background), updates the seam's ρ and z edges on the
`2π` line only (`vv` = `vi` = 0 at `α = 0` and on the extra line),
leaves the axis z edge out of its dumped operator (0), and its `iv` of H
dual edges crossing an outer face uses the whole dual length (twice
ems's half inside the domain; these H components are fixed by the face
condition). (An empty PEC dump carried VTK's range `1e+299` /
`-1e+299` before openEMS be30220, whose own writer has no ranges.)

**Boundaries.** CPML and Mur work on the z faces as on Cartesian grids
(coax TEM reflection −86 dB with CPML(8)). On the α faces of a sector the
CPML grades `σ` with the physical arc `ρ_m·Δα` at the middle `ρ_m` of the
grid's ρ range (its attenuation scales with ρ, so it is optimal there);
it used `Δα` in radians as a length, i.e. the arc at ρ = 1 m, and a
microstrip along α at `R` = 200 mm saw `|S11|` = −23.6 dB from it
(−85.7 dB now, §5.5). openEMS PMC faces of files (the magnetic wall half
a cell inside, §5.16.1) also keep the tangential E and the normal H on
the face itself at 0, as openEMS does (`cyl_radial_port`; no other
fixture changes), and skip the axis and the α faces of a closed grid,
which are not boundaries (before, a file's `xmin="PMC"` on a grid with
the axis zeroed the `H_α` ring at `ρ_½`). A PML on the `ρ_max` face is the
cylindrical PML of §5.15.2 (−94 dB at 58 mm with 8 cells). Mur there uses
the plane-wave one-way equation, which does not match an outgoing
cylindrical wave: the reflection is of order `1/(kρ)`, measured −31 dB at
`ρ = 26 mm`, −37 dB at 56 mm, −43 dB at 116 mm (5–25 GHz; the
Cartesian-style CPML layer, still available as `plane_rho_max()` for
comparison, is alike).

**Available on cylindrical grids**: lumped ports, lumped elements,
voltage, current and field probes, waveguide mode probes on a
`z`-normal plane (the exact weights of `compile/mode.rs` with the
`ρ·Δα` metric), field dumps (VTK and HDF5), dispersive materials
(§5.16), conducting sheets on ρ, α and z surfaces (§5.9), local, raw and
mass-averaged SAR (physical cubes, and openEMS's warped cubes for openEMS
files, §5.12), solid conformal metal and thin conformal sheets (§5.13,
§5.13.1), oblique thin wires (§5.14.1), `CellConstantMaterial` (cell
midpoints, below), NF2FF boxes (cylinders and annular sectors, mirrors
normal to z, and to x / y when they do not cut the box, §5.10), the
coaxial and circular-waveguide port helpers (`CoaxPort`,
`CircWaveguidePort`, along `z` on the axis, also on sectors with
symmetry walls, §5.5) and `LumpedPort` / `CurvePort`.

Also the radial-line port (`RadialLinePort`, §5.5), `MslPort`,
`CpwPort` and `StriplinePort` on the uniform lines along z or α, and the
sectoral waveguide port (`SectorWaveguidePort`) on sector grids (§5.5).

**Not available** (dropped with a warning or refused): plane waves
(openEMS ignores them there too: no field at all, no message; §5.16.2),
the line ports (`MslPort`, `CpwPort`, `StriplinePort`) along ρ or with
the height along α, and the rectangular-waveguide port helper
(`Error::Unsupported`): its geometry and analysis assume a planar
Cartesian cross section (the sectoral port above is its cylindrical
counterpart). The
material model of axis-aligned thin wires (§5.14) is Cartesian; on
cylindrical grids thin wires use the oblique model.

**CellConstantMaterial** (`compile/cellconst.rs`): each cell takes the
material at its parameter midpoint `(ρ_{i+½}, α_{j+½}, z_{k+½})`
(weighting functions at its Cartesian position) and
`EdgeMaterials::from_cells` averages the cells onto the edges with the
annular areas. openEMS (black box, `cyl_cellconst`) samples the same
point: its probes with `CellConstantMaterial="1"` are bit-identical to
the box snapped to the cells whose midpoints it holds, for a radial box
face between a cell's midpoint and its centroid. The codes' edge
averages (openEMS's material dump against ems's) agree on every ρ and α
edge; on z edges at a radial interface openEMS weighs the two radial
halves of the dual face by `Δρ_h` times the radius of the dual line
bounding each half instead of their annular areas `Δρ_h·ρ̄_h` (first
order in `Δρ/ρ`; ems keeps the exact areas natively). Models read from
openEMS files use openEMS's weight (`EdgeMaterials::from_cells_openems`):
the `cyl_cellconst` probes agree to 5e-7…8e-6 (before: 1.2e-4…1.7e-3).

**Radial averaging of openEMS files** (`compile::cylavg`, file path
only, `CompileOptions::subsampling` 1). Without `CellConstantMaterial`
both codes sample one point per quarter cell around an edge, its
centre (measured: a box face swept through a cell flips each average
exactly when it crosses a quarter's centre), but weigh the radial
halves differently (`--debug-material`, `cli_debug_cyl_sector`):

| average | csx (ems native) | openEMS |
|---|---|---|
| ε, κ of a z edge on `ρ_i` | annular area `Δρ_h·(ρ_i + ρ_b)/2` (exact) | `Δρ_h·ρ_b`, `ρ_b` the dual line bounding the half; in α the dual widths of lines `j` and `j − 1` instead of the halves' widths (§5.16.11) |
| ε, κ of an α edge | `Δρ_h·ρ_h` (`ρ_h` the half's centre) | `Δρ_h` |
| μ of a ρ dual edge (H_ρ face), harmonic | `Δρ_h·ρ_h` | `Δρ_h` |

(times the half widths along the other axis, which carry no `ρ`; all
other averages have no radial weight in either code.) The exact
parallel capacitance of an α edge's halves would weigh them by
`Δρ_h/ρ_h`, so neither code is exact there. `cylavg::apply` recomputes
these three averages with openEMS's weights for models read from files
(metal samples left out and identical samples kept exact, as in csx;
`σ_m` left to csx). The material dumps then agree on every edge (both
fixtures), and the probes of `cyl_radial` (a dielectric, magnetic,
lossy box with its radial faces on mesh lines, its α and z faces
inside cells) agree to 1.4e-7…3.2e-6 (csx's weights: 3.2e-4…4.4e-3).

**Azimuth conventions of boxes** (`compile::cylindrical`). csx selects
the edges of a primitive within its bounding box, whose α range it
normalises to `[−π, π)` (§8.1): on a `[0, 2π]` grid sources and lumped
elements at `α > π` got only part of their edges, or none (a lumped ring
over `[0, 2π]` only its `α ≤ π` half). ems re-selects the edges of
excitations and lumped elements on cylindrical grids with every α line
as a candidate, testing the primitive at `α` and `α ± 2π` (csx's rules
otherwise; sheets already were, `wrapped_sheets`), so angles are
periodic: a box means the same region whatever multiple of 2π its α
corners are given in. openEMS (black box, `cyl_wrap_*`, operator and
PEC dumps):

- materials, metal, sheets and excitations take the α range literally
  against the grid's α lines: a dielectric over `[−π, 0]` on a `[0, 2π]`
  grid is absent (but for the α = 0 dual faces), one over `[π, 3π]`
  covers only `[π, 2π]`, metal over `[−π, 0]` is absent;
- a lumped element has each α bound (after sorting) reduced modulo 2π
  into `[0, 2π)`: `[0, 2π]` becomes the α = 0 line, `[π, 2π]` and
  `[3π/2, 5π/2]` are empty, `[−π, π]` becomes the α = π line;
- what lies on the α = 0 line of a closed grid only (a source, a lumped
  element, a conducting sheet, a metal fin) has no effect: openEMS
  updates the α = 2π copy of that line (a source or a metal fin on
  α = 2π works; the fin then agrees with ems to 1e-7). So a lumped ring
  over exactly `[0, 2π]` vanishes: openEMS's probes equal ems's without
  the element to 1e-7, while `[0, 2π − 6e-15]` is the whole ring;
- within the grid's α range both codes agree: a half-ring resistor over
  `[π, 2π)` to 1e-7, a source over `[π, 3π/2]` to 1e-6 (a source on the
  seam line itself differs by 5 %).

No openEMS documentation describes these as conventions; they make the
same physical box behave differently depending on how its angles are
written, so ems keeps the periodic interpretation and warns, for models
read from files, about boxes whose α range leaves the grid's, lumped
elements whose bounds openEMS would reduce, and sources / elements /
sheets on the α = 0 line of a closed grid only.

**Validation.**

| case | result |
|---|---|
| circular PEC cavity `a = 10 mm`, `h = 15 mm`, 20 × 24 × 15 cells, with axis: TM010, TE111, TM011, TM110, TE011 | within 0.19 % of the Bessel-zero frequencies |
| same, TE211 (m = 2) | −0.65 %: the discrete azimuthal wavenumber `(2/Δα)·sin(mΔα/2)` of 24 cells accounts for −0.61 % |
| same cavity built in csx (`Simulation`), TM010 / TE111 / TM011 / TM110 | −0.07 / −0.15 / −0.41 / −0.19 % |
| lossless cavity energy, 5000 steps | drift 2.5e-8 |
| random fields at Courant factor 1.0, 20 000 steps | energy constant to 1.3e-6 |
| coax `b/a = 3`, closed azimuth, Mur ends | `|Z|` within 2e-4 of the discrete TEM value (0.1 % of `η/2π·ln(b/a)`), phase velocity 0.9999 c |
| coax with radial lumped port and 100 Ω load spanning the circle | `Z_in` within 0.25 % (0.25 GHz) … 1.3 % (2 GHz) of the transmission-line model (csx had given both only their `α ≤ π` half, which read 0.1–0.7 %) |
| one-column port on the periodic α = 0 face vs at α = π/2 | equal within 3e-5 |
| axis port listed on one or all α copies | equal to 1e-12 (the rounding of the per-copy element values) |
| openEMS XML (`CoordSystem="1"`) round trip | bit-identical probes |
| 1 vs 3 threads | bit-identical |

#### 5.15.1 Azimuthal multigrid near the axis

**API.** `Fdtd::multigrid(radii)` (openEMS `MultiGrid`, comma-separated
radii in drawing units; `Fdtd::from_csx` / `to_csx` map it) or, on the
engine, `Grid::with_multigrid(&[ρ line indices])`. Each radius snaps to
the nearest ρ line `I` (warning if moved). Inside it, cells `i < I`, the
α resolution is halved once more: the coarsening factor of cell `i` is
`f_i = 2^ℓ` with `ℓ` the number of zone boundaries above it. Rules: a
cylindrical grid, `1 ≤ I ≤ nρ − 2`, distinct lines, α cells a multiple of
`2^L`, and at least 3 coarse α cells on a closed grid. With no radii,
the code paths are unchanged (the existing tests are bit-identical).
Code: `engine::multigrid`, the multigrid branches of `Grid`'s geometry,
`compile::cylindrical::multigrid`. Tests: `tests/cylindrical_multigrid.rs`.

**Embedding, not a second grid.** The fields keep the fine layout; the
coarse Yee grid is a subspace of it. Write `f` for a cell's factor, and
call a line `j` *kept* in a cell if `j % f = 0`.

| fine entry | coarse zone |
|---|---|
| E_ρ, H_α in cell `i`, and E_z on a line whose two cells share `f` | exist on kept lines only (others: `ca = cb = 0` / `da = db = 0`); kept ones take the coarse dual geometry: `Δα*` becomes `(α_{j+f} − α_{j−f})/2` in `A*` (E) and in `l*` (H_α, so `i_α` holds `H·l*_coarse`) |
| E_α on ring `i` (`f` = factor of the outer cell) | the `f` fine edges of a coarse arc form one edge: `u_e = s_e·U`, `s_e = Δα_e/Σ Δα` (E_α constant along the arc) |
| H_ρ on ring `i`, H_z in cell `i` | the `f` fine faces of a coarse face carry one value (copies) |
| E_z on a zone boundary `ρ_I` (coarse `2f` inside, `f` outside) | lines kept inside: *masters*; lines kept only outside: *slaves*, `u_s = (1 − t)·U_a + t·U_b` from the neighbouring masters (`t` the α fraction). A master's dual face has the coarse width inside and the fine width outside; a slave's has only its outer half (its inner half lies in the coarse cell, covered by the master's arc) |

The axis z edge is unchanged (all copies kept; its disc is bounded by
the coarse H_α arcs). Materials are sampled on the fine dual faces (the
csx averages); the coarse edges reuse their fine edge's average.

**Coupling = Galerkin restriction.** Let `S` (E) and `T` (H) be these
embeddings, `C`, `L` the edge capacitances and inductances, `K` the face
circulation (§6 energy). The scheme run is the fine Yee scheme
restricted to `range S × range T`:

```text
SᵀCS·dU/dt + SᵀGS·U = SᵀKᵀT·I,   TᵀLT·dI/dt + TᵀRT·I = −TᵀKS·U.
```

With `C̃ = SᵀCS`, `L̃ = TᵀLT` and the curls `K̃ = TᵀKS`, `K̃ᵀ = SᵀKᵀT`, this
is a leapfrog scheme of the §6 form with symmetric positive mass
matrices, so (i) `½UᵀC̃U + ½IᵀL̃I'` is exactly conserved (lossless) or
non-increasing, and (ii) it is stable for `dt < 2/√λ_max(C̃⁻¹K̃ᵀL̃⁻¹K̃)`.
Both hold whatever the zones, including the interfaces; there is no
interpolation-induced instability because the restriction (`Sᵀ`, `Tᵀ`)
is the transpose of the prolongation.

In the fine arrays it is the fine kernel followed by a projection. From
the kernel `u* = ca·u + cb·Kᵀi`, i.e. `W·u* = W⁻·u + Kᵀi` with
`W = 1/cb = C/dt + G/2`, `W⁻ = ca/cb`, and `u = S·U`:

```text
SᵀWS·U' = SᵀW⁻S·U + SᵀKᵀi = SᵀW·u*   ⇒   u' = S·(SᵀWS)⁻¹·SᵀW·u*,
```

the `W`-orthogonal projection of the kernel result (no old values
needed), and likewise `i' = T·(TᵀVT)⁻¹·TᵀV·i*` with `V = 1/db`. On a coarse
arc this is `U' = Σ s_e W_e u*_e / Σ s_e² W_e`; on a coarse face the
`1/db`-weighted mean of the copies (the periodic-merge formula, §5.6.1);
on a zone boundary a symmetric tridiagonal system per z index (cyclic on
a closed grid, solved by Sherman–Morrison; diagonally dominant), with
PEC masters fixed at 0 and PEC slaves dropped from the interpolation.
The coarse geometry is exactly what makes `C̃ = SᵀCS` and `L̃ = TᵀLT` the
coarse Yee values: `Σ s_e² C_e = εA*/Σ l_e` for an arc, `Σ L_f = μ Σ A_f/l*`
for a face, the coarse `A*`, `l*` for the single kept edges; for the
masters and slaves `SᵀCS` is the consistent (non-lumped) mass of the
linear interpolation, row sum = the area of the coarse dual face.

The engine runs the H projection in the `post_h` gap (before and after
operator hooks there) and the E projection after the periodic and axis
merges (and again after `post_e` hooks), serially in a fixed order: the
results are bit-identical for any thread count. The energy look-ahead
uses the unprojected H kernel result, which is exact because the copies
of `i^{n−½}` are equal (`Σ L_f ĩ i*_f = ĩ·L̃ ĩ'`). `courant_limit` applies
both projections inside the Lanczos operator (still self-adjoint in the
`C̃` inner product), so the reported limit is the coupled scheme's.

**Accuracy.** In the coarse zone the scheme is the coarse Yee grid
(second order). On a zone boundary the z field is coupled to second order
(linear interpolation of the slaves), but the fine α edges on `ρ_I` see
the coarse H_z (and the coarse cell sees their sum), a piecewise-constant
coupling with a first-order local error ∝ `ρ_I·Δα`. Its effect on
resonances grows with the zone radius and pushes frequencies up; the
time step only needs the innermost cells coarsened, so small radii are the
right choice (the mesh check still reports the limiting arcs).

**Validation** (`tests/cylindrical_multigrid.rs`; the cavity of §5.15,
20 × 24 × 15 cells, `Δρ` = 0.5 mm):

| case | dt vs single grid | result |
|---|---|---|
| zones at ρ = 1.5 / 0.5 mm (lines 3, 1; 12 then 6 α cells) | ×3.65 | TM010 −0.057 %, TE111 −0.133 %, TM011 −0.096 %, TE211 −0.615 %, TM110 −0.016 %, TE011 −0.036 % (single grid −0.069 / −0.188 / −0.117 / −0.648 / −0.186 / −0.183 %) |
| three levels, lines 4, 2, 1 (3 α cells at the axis) | ×5.94 | −0.035 / −0.017 / −0.057 / −0.548 / +0.444 / +0.348 % |
| large zones, lines 8, 4 | ×3.66 | TE111 +0.55 %, TM110 +0.42 %: no extra gain, larger interface error |
| same cavity in csx, `Fdtd::multigrid([1.5, 0.5])` | ×3.65 | TM010 / TE111 / TM011 / TM110 −0.045 / −0.124 / −0.309 / −0.015 % |
| lossless cavity energy, 5000 steps after the pulse | | drift 5.7e-7 |
| random fields at Courant factor 1.0, 20 000 steps: closed grid with axis (3 levels), PEC sector, annulus without axis | ×4.90, ×3.90, ×1.67 | energy constant to 4.4e-7, 1.7e-7, 6.0e-7 |
| 1 vs 3 threads | | bit-identical |

**openEMS files** with one zone boundary on a closed grid with the axis
(vacuum inside the zone) run openEMS's own scheme instead, a coarse
operator coupled by interpolation (`OpenemsMultigrid`, §5.16.11); its
results equal openEMS's to the printed digits.

**Limits.** Removed edges hold 0, so field dumps and probes inside the
zones see the stored representation: removed lines read 0 and H_α holds
`H·l*_coarse` (dumps divide by the fine length, so it reads about `f`
times the field). Excitations on removed edges have no effect and lumped elements
on them fail (PEC edge); elements on coarse arcs or slaves act through
the projection; the compile step warns about both. Absorbing α faces of
a sector are not combined with multigrid (the projection would treat
their `cb = 0` edges as PEC).

#### 5.15.2 Cylindrical PML on `ρ_max`

Code: `engine::boundary::cyl_pml`, used by `CpmlBoundary` for the `XMax`
face of every cylindrical grid (so `Simulation` and openEMS XML models
get it automatically); tests: `tests/cyl_pml.rs`.

**Why the Cartesian layer reflects.** The CPML replaces `∂/∂ρ` by
`(1/s)·∂/∂ρ`. In cylindrical coordinates the radius also appears in the α
metric (`ρ·dα`), so stretching only the derivative is not a change of
coordinates and the layer is not matched to `H_ν(kρ)`; the mismatch is of
order `1/(kρ)`. The matched layer stretches the radius itself
(Teixeira & Chew 1997; Chew & Weedon 1994):

```
ρ̃ = ρ0 + ∫_{ρ0}^{ρ} s dρ',   s = κ + σ/(α + jω·ε0)
```

which enters both `dρ̃ = s·dρ` and the metric `ρ̃·dα`.

**Integral form.** The circulations of §5.2 are topological. In the
stretched coordinates only the edge lengths and face areas change, and
they become complex, so the PML is a per-edge frequency-dependent `C` and
`L`. With `u`, `i` the integrals over the stretched edges:

```
(jω·C + G)·g·u = Σi,   g = (Ã*/A*)/(l̃/l)
(jω·L + R)·h·i = −Σu,  h = (Ã/A)/(l̃*/l*)
```

From the table of §5.15, with `s` the mean stretch over an interval
(`(ρ̃_b − ρ̃_a)/(ρ_b − ρ_a)`), `s̄ = ρ̃/ρ` at a point and
`s̄₂ = (ρ̃_a + ρ̃_b)/(ρ_a + ρ_b)`:

| component | `g` / `h` | intervals |
|---|---|---|
| E_ρ (cell i) | `s̄/s` | `s` over cell `i`, `s̄` at `ρ_{i+½}` |
| E_α (line i) | `s/s̄` | `s` over the dual cell, `s̄` at `ρ_i` |
| E_z (line i) | `s·s̄₂` | the dual cell (annular area `(ρ̃_b² − ρ̃_a²)/(ρ_b² − ρ_a²)`) |
| H_ρ (line i) | `s̄/s` | `s` over the dual cell, `s̄` at `ρ_i` |
| H_α (cell i) | `s/s̄` | `s` over cell `i`, `s̄` at `ρ_{i+½}` |
| H_z (cell i) | `s·s̄₂` | cell `i` |

This is the discrete anisotropic medium `diag(s̄/s, s/s̄, s·s̄)` of
Teixeira & Chew. Outside the layer every factor is exactly 1, so there `u`
and `i` are the physical integrals. The stretched radius at any
fractional index is `ρ̃ = (ρ + K) + Σ/p`, with `p = jω·ε0 + α`,
`K = ∫(κ − 1)dρ` and `Σ = ∫σ dρ`. `α` is constant over the layer: a
graded `α` would make `ρ̃` non-rational in `p`. Every factor is then a
ratio or product of first-order functions `(A·p + B)/(C·p + D)` of `p`.

**Time stepping.** The kernel applies the plain circulation `S`. The
operator adds `cb·(Y − S)` (E) or `−db·(Y − T)` (H), with `Y = g⁻¹∗S`
evaluated as a cascade of at most two first-order stages
(`s/s̄`, `s̄/s`, or `1/s̄₂` then `1/s`). This is the same "correct after the
kernel" structure as the CPML, but `g⁻¹` multiplies the **whole**
circulation, not one derivative term. Each stage `A/C + r/(jω + γ)`
(`γ = (α + D/C)/ε0`) is discretised with the bilinear (trapezoidal) rule:

```
y^n = (A/C)·x^n + ψ^n,   ψ^n = β·ψ^{n−1} + w·(x^n + x^{n−1}),
β = (2 − γ·dt)/(2 + γ·dt),   w = r·dt/(2 + γ·dt)
```

This realises every stage exactly at the warped frequency
`Ω = (2/dt)·tan(ω·dt/2)`. All stages share that warp, so the semi-discrete
layer is still an exact stretch, `s(Ω)`. The first-order recursive
convolution of the CPML (`ψ ← e^{−γdt}ψ + …`) was tried first. It gave
−63 dB at 26 mm and improved by 4.5 dB per halving of `dt`, so it was a
time-discretisation error. The bilinear form gives −82 dB, independent of
`dt`. State: two `ψ` and two previous inputs per edge.

**Grading.** As the CPML: index depth `t`, `σ = σ_max(Δ)·t^m`,
`κ = 1 + (κ_max − 1)·t^m`, medium-tuned `σ_max`. The integrals are exact
(`ρ` is linear in the index within a cell), and `Σ = Σ_tot·t^{m+1}` with
`Σ_tot = σ_max(Δ)·Δ·n/(m + 1)`.

**Late-time stability.** With `α = 0` the stretch is infinite at DC, and
`s` appears in the numerator of some factors (a pure integrator for `H_ρ`
on the interface line, where `s̄ = 1`). Static fields left by a source with
DC content then made the energy grow slowly, quadratically in time
(+0.8 % over 100 000 steps), the known weak instability of unsplit PMLs.
The layer therefore uses the CFS shift `α = max(α_max, 10⁻³·σ_max)`.
That moves every pole to `γ ≥ α/ε0 > 0`, and the CFS corner
`α/(2π·ε0)` sits near 1 % of the frequency at which the cells are λ/20,
so absorption in band is unchanged. With it, static remainders settle (a
damped ripple of ~1e-5) and never grow.

**Corners.** The layer handles every component in its x-planes, including
where z-face (or α-face, for sectors) CPML layers overlap it. Those
Cartesian layers are cut at the interface, and inside the ρ layer the
derivative term along that axis is stretched with the face's own CPML
recursion, `(1/κ_q)·D + ψ_q`, before `g⁻¹` is applied to the sum. That is
the exact two-axis stretch. Simply adding the two corrections (z-CPML on
its derivative term, `g⁻¹` on the plain sum) is **unstable** (measured).

**Restrictions.** The interface radius must be `> 0` (the layer must not
reach the axis). A `ρ_min` PML (annulus) keeps the Cartesian stretch.
Conformal merge groups in the layer (their kernel `db` is 0) apply the
layer's filter to their members themselves (§5.13 "Absorbing layers").
Conformal thin sheets in the layer are not stretched.

**Validation** (`tests/cyl_pml.rs`). Measured as the time-gated
difference to a reference domain 560 cells larger; 0.5 mm cells,
5–25 GHz; annulus from 2 mm with 16 α cells (all resolved azimuthal
orders excited). TM between PEC plates, TE between PMC plates:

| case | cylindrical PML | Cartesian-style layer |
|---|---|---|
| TM, layer at 58 mm, PML(8) | −93.8 dB | −38.2 dB |
| TE, same | −90.5 dB | −37.0 dB |
| TM, PML(4) / (6) / (12) / (16) | −50.8 / −73.7 / −101.2 / −105.8 dB | −38.1 … −39.5 dB |
| TE, PML(4) / (6) / (12) / (16) | −51.4 / −75.1 / −98.0 / −102.2 dB | −35.6 … −38.1 dB |
| TM / TE, PML(8), m = 2 | −86.5 / −86.3 dB | −38.2 / −36.9 dB |
| TM / TE, PML(8), m = 4 | −91.3 / −88.4 dB | −38.1 / −37.0 dB |
| TM / TE, PML(8), sigma_factor 0.5 / 2 | −53.2 / −90.7, −53.7 / −89.9 dB | −38 dB |
| TM / TE, PML(8), κ_max = 3 | −99.5 / −89.8 dB | −38.4 / −37.2 dB |
| TM / TE, PML(8), α_max = 0.05 S/m | −94.9 / −89.5 dB | −38.2 / −37.1 dB |
| TM / TE, layer at 26 mm | −82.4 / −80.3 dB | −31.6 / −30.8 dB |
| TM / TE, layer at 114 mm | −99.3 / −97.4 dB | −45.4 / −42.7 dB |
| 3-D, 1 mm cells, PML(8) on ρ_max and both z faces (corners) | −75.0 dB | −33.0 dB |
| zero-mean pulse, 16 000 steps | energy −113 … −134 dB of peak, no late growth | |
| DC source remainder, 12 000 steps | constant within 2e-4 | |
| 1, 2, 3, 5 threads (3-D, corners) | bit-identical | |

Thin layers are limited by the round trip through the PEC-backed layer,
`exp(−1.6·n·sigma_factor)`: −56 dB for 4 cells or for
`sigma_factor = 0.5`. 6 cells already reach −74 dB, and 8 cells the
discretisation floor of about −90 dB, which falls slowly with the radius
(−81 dB at 26 mm, −98 dB at 114 mm; Δ/ρ effects). The 3-D number also
contains the grazing incidence on the Cartesian z layers.

### 5.16 openEMS parity (black-box fixtures)

What openEMS v0.37 does was established by running the compiled oracle
(`scripts/openems-oracle`, no openEMS source consulted; since wave 14
openEMS-Project be30220, openEMS v0.37.0-32, CSXCAD v0.7.0-8, §5.16.5)
on hand-written
models and comparing its output files with ems, as a black box. The
fixtures live in `tests/fixtures/openems/`, each set with its own
regeneration script and test file:

- core XML semantics (§5.16.1): `gen-core.sh`, `tests/openems_core.rs`;
- advanced physics (§5.16.2, `adv_*`): `gen-adv.sh`, `tests/openems_adv.rs`;
- field dumps and SAR files (`dump_*`, `io::openems::dump`): `gen-dumps.sh`,
  `tests/openems_dumps.rs`;
- cylindrical-grid features (`cyl_*`: conducting sheets, SAR files incl.
  1 g, plane waves, `CellConstantMaterial`, α ranges): `gen-cyl.sh`,
  `tests/openems_cyl.rs`, `tests/openems_dumps.rs`;
- the command-line program (`cli_*`, §7.2): `gen-cli.sh`, `tests/cli.rs`;
- the `nf2ff` tool (`nf2ff_*`, §5.10): `gen-nf2ff.sh`,
  `tests/openems_nf2ff.rs`;
- the `sar_calc` tool (`sar_calc_*`: openEMS's raw dump, synthetic raw
  files of `sar-synth.py`, §5.12): `gen-sar.sh`,
  `tests/openems_sar_calc.rs`;
- the XML feature-coverage sweep (§5.16.3, `sweep_*`): `gen-sweep.sh`,
  `tests/openems_sweep.rs`;
- end-to-end models (`e2e_*`, §5.16.4): `gen-e2e.sh`,
  `tests/openems_e2e.rs`;
- `PML_n` layers (`pml_*`, §5.16.2): `gen-pml.sh`, `tests/openems_pml.rs`;
- models of the randomized differential sweep (`rs_*`, §5.16.3):
  `gen-rsweep.sh`, `tests/openems_rsweep.rs`.

ems follows openEMS for models read from files; Rust-built models keep the
ems conventions where noted (`Fdtd::openems_run`, `Fdtd::file_equivalent`).
Files are read with openEMS's reading rules (`io::openems::read_model`,
§5.16.3) where they differ from csx's.
"Agreement" is the relative L2 difference of the probe signals.

#### 5.16.1 Core semantics

- **Signals.** Gaussian `cos(2πf0(t−T))·exp(−((t−T)/τ)²)` with
  `τ = 3/(2π·fc)`, `T = 3τ`, samples `k = 1 … ⌊2T/dt⌋` (sample 0 is 0);
  sine `sin(2πf0 t)` with sample 0 zero and one period repeated (so H
  samples at multiples of the period are 0 too), `dt` shortened to
  `T/(⌊T/dt⌋ + 1)`; Dirac `[0, 1]`; step 1 from sample 0; custom: the
  fparser expression of `t` only (its Nyquist frequency is `f0`). E
  samples are at `k·dt`, H samples at `(k + ½)·dt`.
- **Excitation timing.** The field value at time `T` (E at `(n+1)·dt`,
  H at `(n+½)·dt`) gets the signal at `T − dt − ⌊Delay/dt⌋·dt` (all four
  kinds; soft adds, hard overwrites): a soft E source added after the E
  update `n → n+1` uses the signal at `n·dt`, a soft H source after
  `n−½ → n+½` at `(n−½)·dt` (§5.7). With these, vacuum and dielectric
  lines agree to 1e-6. Soft E sources add `Excite·w·l` (no `dt/ε`
  scaling). ems applies this as a lag on the model's signal
  (`sim::map::openems_signal`); the engine sources keep their own sample
  times.
- **Probes.** Voltage: primary-line snapping, ties down; lines only.
  Current: transverse extent of the snapped box (a corner exactly half-way
  between two lines snaps inwards: the lower corner up, the upper one down,
  whatever the corner order; a point-thin box whose corners then cross
  reads 0; a zero-length Cylinder or CylindricalShell is the point box of
  its axis; fixture `current_probe_ties`), loop in the dual plane
  nearest to a flat box's normal coordinate (ties down); a normal extent
  disables the probe. E/H field probes: native components at the start
  corner (E: edges leaving the node; H: dual edges of the dual node),
  no interpolation. Time series every `max(1, ⌊N/OS⌋)` steps (`N`
  Nyquist interval `⌊1/(2 f_max dt)⌋`, `OS` OverSampling, default 4,
  values < 2 → 4; Dirac/step: `N = 1`), E rows from `t = 0`; FD values
  `2·Σ x(t_k) e^{−j2πft_k}·N·dt` over every `N`-th step;
  StartTime/StopTime keep one extra sample (§ `ProbeResult`). The
  FDTD `f_max` attribute is ignored by openEMS. See §5.16.2 (current
  probes) for the graded microstrip that exposed the dual-plane rule.
  H-type files (current, H field) end with the H after the last step,
  `(N + ½)·dt`; ems adds that row for files from a look-ahead H update
  (`Probe::lookahead_sample`, the machinery of the final dump step; not on
  periodic, multigrid or axis/closed cylindrical grids), so the row counts
  agree (`tests/openems_core.rs` now requires it). openEMS skips, with a
  console error, a voltage probe whose raw box is not a line (a point, a
  flat or solid box, also a thin box that snaps to one line) and a
  current probe whose raw box extends along `NormDir` (or is solid
  without one); ems does too for files, with a warning (fixture
  `probe_skip`). The dual line of an H-field probe and the dual plane of
  a flat current probe are those of the cell holding the coordinate,
  closed at its upper end (a coordinate on a line takes the cell below;
  wave 14, point probes swept through a graded mesh): on uniform meshes
  that is the nearest dual line with ties down, which ems used; on
  graded ones a point in a large cell next to a small one is nearer to
  the small cell's centre (z = 4.9 between cells 4…5 and 5…5.5: openEMS
  4.5, ems 5.25). Files follow it (`sim::map::snap_cell`). `ems::port::LumpedPort` writes its current probe as a
  flat box on the line above the port's middle dual plane, which both
  codes read as that dual plane. `FD_Samples`: commas only (§8.1).
- **Time step.** openEMS's time step depends on the materials, with both
  rules it offers (`TimeStep::OpenEms`, `engine::timestep`; ems's own CFL
  bound of §5.4 uses the vacuum speed). Observed with `--no-simulation`
  (6 printed digits):
  - `TimeStepMethod="1"` reproduces exactly (every test geometry, uniform
    and graded meshes, ε and μ inclusions):
    `1/dt² = max_e (1/C_e)·Σ_{f∋e} 1/L_f` (the 4 dual edges `f` around
    the edge, `C`, `L` the circuit values of §5.2 with the averaged
    materials), i.e. the inverse square root of the diagonal of the
    discrete curl-curl operator. On a uniform vacuum cube that is
    `Δ/(2c)`, 0.866 of the CFL limit. Any other unknown id behaves as 1.
  - The default (`3`, reported as "Rennings_2", asymmetric in the mesh
    grading) gives the exact CFL value on uniform vacuum meshes of any
    aspect ratio and `dt ∝ √ε` in a filled medium. It is a per-edge rule
    (`engine::timestep`): for the E edge along `a` at node `n` and its
    transverse axes `b`, `c`,
    `ω² = T_b + T_c + max(O_c + |D_b|, O_b + |D_c|)`, where `T_b` couples
    the edge to its two faces along `b` (`¼·Σ_f (1/L_f)·(1/C_a + 1/C_a')`
    over the face's two `a` edges), `O_b ± D_b = (1/L_c(n−e_b) or
    1/L_c(n))·½(1/C_b(n) + 1/C_b(n+e_a))` (the `b` edges at both ends of
    the edge, with the dual edge below resp. at the node), and, when
    `(a, b)` runs against the cyclic order, the face below along `b`
    takes its `ε` and `μ` one node lower along `c` (geometry unchanged).
    Fitted from black-box runs in three steps (wave 13): meshes with a
    fine x cell next to y cells that shrink upwards isolated the `|D|`
    term (openEMS's `O` factor was `(3 − r)/2` instead of `(1 + r)/2` for
    a cell ratio `r < 1`; cells shrinking along both transverse axes fixed
    the `max`); then the coefficient of `1 − 1/ε_avg` (resp. `1 − 1/μ`)
    in `1/dt²` for layers below an air edge (it is linear, and it
    matched the face below along `x` of a `y` edge one node lower)
    located the shifted materials. Agreement to the printed 6 digits:
    ~1 000 random vacuum meshes graded along one to three axes (cell
    ratios up to 30; the earlier form, `O_b + O_c − L_c(n)·Q_b` for the
    lower `b`, was off on 73 of 420 doubly graded ones, by up to 8 %,
    where cells shrink along a transverse axis), ~240 meshes with
    grid-aligned ε and μ layers and boxes, and 560 runs of the `ems`
    binary against openEMS with grid-aligned and off-grid ε, μ, κ,
    anisotropic and metal boxes. Full precision (wave 14; the probe files'
    12-digit time column, the HDF5 `time` attribute being an `f32`): the
    Rennings step is a double within 4e-8 of ems's (14 random graded
    vacuum meshes; openEMS's single-precision round-off inside the
    criterion, not reproduced: rounding the circuit values `C`, `L`, the
    lengths and areas, the mesh deltas or lines to `f32` in eleven
    combinations never gave its value). The method-1 step is always an
    exact `f32`: ems now evaluates it from `f32` circuit values with an
    `f32` square root and division, bit for bit on 23 of 27 vacuum and 18
    of 20 dielectric random meshes (rounding only the result: 15 of 27),
    one `f32` step (≤ 1.2e-7) apart on the others. Wave 16: both steps
    bit for bit (openEMS's constants and single-precision couplings,
    §5.16.10). The `e2e_patch` and `e2e_msl_filter` fixtures (were 0.5
    and 2 % off) now run on openEMS's own time step (no `TimeStep`
    attribute; regenerated; filter ports 6e-7…1.1e-6, were 2e-7…5e-7 with
    the same rounded step), and every Cartesian
    fixture without a fixed step agrees. A metal box sharing a face with a
    higher-priority material (3 of 60 random cases, up to 1.4 %, wave 13)
    was not the time step but the edge materials: csx leaves material
    samples in metal out of an edge's average, openEMS takes the material
    *under* the metal there (the highest-priority non-metal property, or
    the background; `--debug-material`: face edges `(3·1 + 4)/4 = 1.75`
    where ems had `(1 + 4)/2`… on vacuum, `7.75` inside an `ε_r = 9`
    block), and gives every PEC edge inside the metal that material too,
    which its time step counts like any edge: a metal block inside an
    `ε_r = 9` block stepped at 2.07e-12 s in ems, 5.73e-12 s in openEMS.
    Files now discretise the materials with the metal primitives removed
    (`compile::openems::without_metal`; PEC mask, sheets and sources from
    the full model): 300 random metal/material face ties (graded meshes,
    a lower-priority block under the metal, off-grid metal in 30 %) all
    agree to the printed digits (2 of 100 were 1.3–1.9e-3 off before) and
    the material dumps are identical; fixture `metal_material_tie` (5e-7…
    9e-7). Cylindrical grids: next item.
- **Time step on cylindrical grids** (wave 14, `engine::timestep_cyl`,
  files and `TimeStep::OpenEms`; ems had used its operator limit there
  and the cylindrical fixtures fixed `TimeStep`). openEMS evaluates the
  same two criteria with the physical cylindrical circuit values and
  these conventions, fitted from `--no-simulation -vvv` (step and
  limiting edge of every grid) on ~560 random meshes (sectors and closed
  grids, with and without the axis, graded ρ, α, z, ρ_min from 0.3 to
  50 mm) and the operator dump:
  - geometry outside the mesh comes from lines extrapolated with the
    outermost cell (the radius of the cell beyond `ρ_max` enters; the
    dual line below `ρ_min` is `ρ_min − Δρ_0/2`, clipped at 0), while
    indices outside clamp as on Cartesian meshes; Gershgorin takes full
    boundary duals and counts the faces beyond the boundary (the
    Cartesian half-cell rule when cells are mirror images);
  - z components swap their α widths: the dual face of E_z at `α_j` is
    `Δα_j·(ρ_hi² − ρ_lo²)/2` with the *primary* `Δα_j`, the face of H_z
    in cell `j` takes the *dual* width (the operator dump has the same
    coefficients; equal on uniform α);
  - Rennings shifts the geometry, not only the materials, of the face
    below (cancels on Cartesian meshes), and evaluates the own-axis term
    at each end `m` of the edge, `½Σ_m max(P_c(m) + |D_b(m)|, P_b(m) +
    |D_c(m)|)` (`P_b(m) = L̄_c/C_b(m)`; the Cartesian formula when both
    ends pick the same branch, which they do on Cartesian meshes). An E_ρ
    edge of a uniform mesh gets `½(ρ_{i+1}/ρ_{i+½} + ρ_{i+½}/ρ_i)/Δρ²`;
    the end-averaged form had `½(ρ_{i+½}/ρ_i + ρ_{i+½}/ρ_{i+1})/Δρ²`, up to
    2 % off (14 of 160 sectors, 6 of 80 closed grids);
  - a closed azimuth gets openEMS's extra line `α_1 + 2π` and is treated
    as a sector;
  - with the axis openEMS uses the Gershgorin criterion whatever
    `TimeStepMethod` says (both methods print the same step on all 160
    axis meshes), without the axis z edge (updated outside the
    operator) and with the H_ρ faces of zero area left out;
  - `MultiGrid`: one grid per zone; a zone boundary on line `I` makes an
    inner grid of the lines `ρ_0 … ρ_I` and every second α line, and the
    outer grid's step is set by its edges with ρ index `> I`; the step
    is the smallest ("split position @I+1" in openEMS's console; openEMS
    rejects some splits, "split invalid", which ems accepts).
  Agreement to the printed 6 digits: all 160 sector, 80 closed, 160
  axis meshes (vacuum, both methods) and 251 runs of the `ems` binary
  with random ε, μ, κ boxes, closed and multigrid grids, except 4 of 60
  closed grids with a material ending between `α_{N−1}` and 2π or
  starting inside the first cell: openEMS reads the materials of its
  `2π` and extra lines neither periodically nor at the literal angles
  (a box over `[−1, 6.2]` rad leaves the extra line in vacuum and the
  `2π` line in the material; one over `[0, 2π]` fills both), up to 5 %;
  ems takes the periodic images. Full precision (probe time columns): 1e-8
  … 6e-8 on the fixtures (wave 16: bit for bit, §5.16.10). The cylindrical fixtures now run on openEMS's
  own step (regenerated: `adv_cyl_*`, `cyl_*`, `dump_cyl*`, `pml_cyl`,
  `e2e_coax_monopole`), except the debug dumps (`cli_debug_cyl*`), the
  conducting-sheet cavity (`cyl_sheet_*`: openEMS's thin fin model grows
  without bound at its own, 11 % longer step) and `adv_cyl_multigrid`
  (§5.16.2). The closed dump `dump_cyl_closed` then records the state
  after the last step, which ems now samples on cylindrical grids with
  the axis or a closed azimuth too (its look-ahead H update; only E is
  merged there).
- **Dumps of a Dirac or step excitation** record every step (openEMS's
  Nyquist rate is then 1 step whatever `f_max` says; ems used `f_max`
  and wrote every 12th step of `dump_dirac`, wave 13; agreement 5e-7).
- **Run control.** `endCriteria` absent or 0 → 1e-6; checked every Nyquist
  interval on the state after the interval (stops at a multiple of it);
  `MaxTime` → `⌊MaxTime/dt⌋` steps; no `NumberOfTimesteps` → 0 steps.
  The energy openEMS compares is `W = ε0·Σ u² + μ0·Σ i²` over every edge,
  with `u` at `n·dt` and `i` at `(n + ½)·dt`, no material or cell-size
  weighting and no lumped or pole state (fitted from full E/H dumps of a
  graded WR-90 guide with a dielectric post, every check step: the
  printed ratios follow it with a constant 0.005 dB offset; its printed
  `Energy` is the same `W` in J/m). On uniform vacuum meshes it is
  proportional to the ems invariant, which is why PEC boxes and the
  `end_criteria` fixture agreed to 1e-3 dB before; on graded meshes and
  in media the two differ by up to 0.2 dB, and slowly decaying runs
  stopped 80–300 steps apart (Mur guides: 6237/6258/8694/11193 against
  openEMS's 6153/6174/8715/10899). Files now use it
  (`EnergyMeasure::OpenEms`, evaluated on the state by all engine
  threads, `Engine::energy_in`): the same stop steps in all those cases.
  PML layers count with their true fields (§5.16.2; the operator corrects
  the look-ahead of the edges it completes itself, `SlabHooks::energy_openems_slab`).
  The former 0.005 dB offset (wave 13) came from three openEMS details,
  fitted from per-step native E/H dumps of a uniform Mur box (symmetric
  and off-centre sources) against the ratio openEMS prints at its stop
  step, scanning `endCriteria` to stop it at every check: (1) the dual
  edges on the outer planes normal to their own axis (H_x at x = 0 and
  x = max, …) count with openEMS's full-cell dual length, i.e. `4·i²` in
  terms of ems's half-cell currents (the same full boundary cells as in
  its time step); (2) the E on the three upper faces (x, y or z at its
  last line) is not counted, the lower faces' is (on PEC faces it is 0
  anyway; wave 13 had read this as "both x faces", which the symmetric
  boxes could not tell apart); (3) besides every
  Nyquist interval openEMS checks once more after half the Gaussian
  excitation's length, `⌊L/2⌋` steps (496 of 992, 298 of 596, 323 of 647,
  354 of 709: a threshold between the regular checks stops the run
  there; `RunLimits::extra_check`). ems does all three for files
  (Cartesian grids for (1), (2)). Printed ratios at the stop step, before
  → after: −1.2620 dB (openEMS) vs −1.2675 → −1.2621; −8.5441 vs −8.5569
  → −8.5438; −27.8626 vs −27.8825 → −27.8626; −61.6809 vs −62.6582 →
  −61.6801; `e2e_waveguide_iris` −40.0719 vs −40.0999 → −40.0721; a
  −10 dB criterion stopped at 684 instead of 496 → 496 (fixtures
  `end_criteria_half`, `end_criteria_mur`; ≤ 3e-4 dB down to −32 dB,
  2.5e-3 dB at −61 dB). `e2e_drude_slab` (Drude/Lorentz poles, plane
  wave) went from 0.41 to 0.027 dB at −50 dB in wave 13. Wave 14 split
  that model (end-criterion scans: the ratio openEMS prints at the stop
  step for thresholds every few dB): with a soft source instead of the
  plane wave the Drude and Lorentz slabs agree to 2e-4 dB (no pole state
  in the measure, as assumed), and the vacuum line with a soft source was
  still 0.03–0.99 dB off: a TEM pulse leaving through the Mur z faces.
  A least-squares fit of per-plane weights to 19 printed ratios of that
  line (raw E/H dumps every step, openEMS's own fields) gave weight 0 for
  the E on the z = max plane and 1 elsewhere (max. error 1e-4 dB), and
  an asymmetric all-Mur box (source vector (1, 0.7, 0.4)) then fixed rule
  (2): upper faces left out, 1e-4 dB on 8 checks, where "both x faces"
  left 0.028 dB and "upper faces without the x-min face" 0.0277 dB. Now
  (`end_criteria_tem`, new): the line 0.0000…0.0004 dB on every check
  (was up to 0.99 dB, one stop 18 steps late), `end_criteria_mur`
  −60.7507 against −60.7506 (was −60.7531), `end_criteria` −32.2832
  against −32.2833, `e2e_waveguide_iris` −40.0719 exact, and
  `e2e_drude_slab` 0.0127 dB at −50 dB (was 0.027) with the same stop
  step at every threshold from −10 to −60 dB (two of eight were 17 steps
  apart). What is left there is a constant 0.013 dB from the first check
  on, i.e. the maximum energy while the pulse is in the total-field box:
  openEMS's plane-wave injection error (§5.16.2; fields 1e-3 apart). With
  the plane wave in vacuum the late ratios differ by 0.1–0.3 dB, the
  energy of openEMS's −37 dB leakage into the scattered-field region,
  which ems does not have (not emulated).
- **Materials.** Edge averaging agrees with csx's (anisotropic, weighted,
  lossy, μ, σ: ~1e-6); `CellConstantMaterial` samples cell centres.
- **Boundaries.** PEC and Mur (with phase velocity) agree to ~1e-6; Mur
  without it, see §5.16.2. Two Mur details found on the end-to-end models
  (per-edge least-squares fits of openEMS's per-step dumps, residuals
  4e-8): an edge on two Mur faces takes the update of the face of the
  lower axis (x before y before z) alone, with its interior neighbour at
  `n + 1` as 0, `u^{n+1} = u_1^n − k_1·u^n` (`MurBoundary::openems_corners`;
  an all-Mur box went from 5e-3 to 1e-6; wave 13 had fitted the average of
  both faces on a uniform box, where the two are the same; wave 14's box
  with a different last cell on every face tells them apart, residual
  1e-7 on all twelve domain edges), and the H component normal to an
  upper Mur face (wave 14: also an upper PML face) is never updated in
  the face plane (it feeds no E update; ems holds it at 0 for files, which
  matters for the energy and for dumps on the face). PMC: the wall is the outermost dual layer
  (tangential H there held at 0), half a cell inside. PML: ems runs
  openEMS's own layer for files (§5.16.2, ~1e-6; the CPML differed by
  0.2–0.4 % on near-field probes behind an 8-cell layer). openEMS ignores `PML_Grading`: its operator dump and
  probe files are bit-identical with any expression (also with unknown
  variables, on `<BoundaryCond>` or `<FDTD>`, under a dozen plausible
  attribute names; fixture `adv_pml_grading`); ems ignores it as well
  (warning).
- **Lumped elements** (identified from `--debug-operator` dumps, which
  hold `vv`, `vi`, `iv`, `ii` per edge, and from full E/H dumps every
  step, by regressing each edge's update; ems follows all of it for files):
  - A parallel element's C replaces the edge's medium capacitance and its
    R the medium conductance (`vv`, `vi` of the edge). Without an
    inductor it is spread like a homogeneous material: `G_e = κ·A_e/l_e`,
    `C_e = ε_L·A_e/l_e` (`A_e` the dual face with full cells at the mesh
    boundary, `l_e` the edge), `κ`, `ε_L` uniform and set by the totals
    (series sum of parallel sums). Measured on a graded mesh (`G_e`
    over `A_e/l_e` constant to 6 digits, total 50.000 Ω) and on a coax
    ring (`R_e ∝ 1/ρ`). Elements with an inductor and series elements are
    spread in equal parts (csx's `R·p/s`; uniform meshes cannot tell).
  - A parallel inductor is updated explicitly: `i_L^{n+½} = i_L^{n−½} +
    (dt/L_e)·u^n`, `u^{n+1} = vv·u^n + vi·(Σi − i_L^{n+½})` (ems's
    trapezoidal branch was 0.4–1.6 % off; `explicit_inductor`). It is
    unstable once `dt²/(L_e·C_e)` approaches 4 (openEMS diverges at
    `L` = 0.1 nH on a 1 mm vacuum edge); ems falls back to its
    trapezoidal branch above 1 (warning).
  - A series element is the trapezoidal (bilinear) discretisation of its
    admittance, driven by the edge voltage `ū`, which is what ems's
    branch is; a pure series R behaves as the parallel fold.
  - An excitation over any element is a plain soft source added after
    the update, and the element (inductor current, series branch) is
    driven by the edge voltage *without* it: `u^n − V^{n−1}` resp.
    `ū − V^{n−1}/2` in terms of the stored voltages (fit residuals
    4e-7). A series element with an excitation was ~30 % off with ems's
    Thevenin source (`branch_drive_source`).
  - Agreement now (`lumped_exc`, `lumped_graded`, `lumped_l`,
    `lumped_rc`, `lumped_port`): 1e-7…5e-5 with an excitation over the
    element or a well-conditioned voltage; voltages over a nearly
    shorted element stay at 4e-4 (`lumped_l`, the inductor voltage is
    2e-5 of the field around it) and 1.3e-3 (`lumped_rc` series RLC):
    the f32 round-off of the neighbouring fields, not a formulation
    difference (ems's and openEMS's single-edge updates fit the same
    coefficients). The "series elements on dielectrics 2–5 %" of the
    earlier notes did not reproduce (7.6e-5 on ε_r = 4).
  - A parallel element with an inductor and *without* a capacitor gets
    its edge capacitance limited (wave 13; `--debug-operator` dumps,
    `vv`/`vi` turned back into each edge's `C` and `G`): at the
    excitation's frequency `ω = 2π·f` (`f0 + fc` for a Gaussian, `f0` for
    a sinusoid and for a custom signal, whose `f_max` does not count) the
    edge's capacitive admittance stays below a twentieth of the element's
    resistive and inductive ones, `C_e ≤ min(1/(20·ω·R_e),
    1/(20·ω²·L_e))` with the per-edge (equal-part) `R_e`, `L_e`; edges
    above the limit get it exactly, the conductance is untouched. Fitted
    over `R` = 50…1000 Ω, `L` = 0.2…10 nH, two time steps (5e-13 and
    9e-13 s: the limit does not depend on `dt`), `fc` = 1 and 2 GHz, a
    sinusoid and a custom signal: the limit `C_max·R_e` equals
    `1/(20·ω)` and `C_max·L_e` equals `1/(20·ω²)` to 5 digits. No limit
    with a capacitor (R‖L‖C), on R, C or R‖C elements, or on series
    elements. This was the "R‖L with an excitation over it on a graded
    mesh differs by 0.6–3 %" (the larger the `R`, the more edges are
    limited): ems now applies the limit on the file path
    (`sim::map`, `limit_medium_capacitance`), and `lumped_graded_rl`
    (R‖L with an excitation, R‖L‖C, an R‖L next to it) went from 3.0e-2
    / 3.3e-6 / 1.2e-1 to 4.0e-7 / 4.2e-7 / 1.3e-5 (the last is the
    passive element's small voltage, f32 round-off). It also explains
    the large-value behaviour noted before: a 1 mm vacuum edge's limit
    `dt²/(L_e·C_e) ≥ 20·ω²·dt² = 0.0732` (5 GHz, 1.93 ps), and the
    divergence for large `L` or `R` (the limited capacitance breaks the
    edge's stability: `R` = 150 Ω diverges on `lumped_graded_rl`, where
    15 % of an edge's capacitance is left); ems warns when less than a
    fifth is left.
  - Series elements (wave 14) get their edge capacitance limited too, by
    the branch impedance at the same `ω`: `C_e ≤ 1/(20·ω·|R_e + jωL_e +
    1/(jωC_e')|)` (`--debug-operator` dumps of R, L, RL, LC and RLC
    branches, `R` = 1…1000 Ω, `L` = 1.1…50 nH, `C` = 0.5…5 pF, two
    frequencies, 5 digits). This is the "different, unstable update" of a
    series R above ~190 Ω per 1 mm vacuum edge noted before (NaN within
    16 steps at 2 kΩ: the limited capacitance), and it made series
    elements over several columns, or with an excitation, differ by
    1e-3…1 (random sweep below); ems applies it for files.

#### 5.16.2 Advanced physics

The `adv_*` fixtures fix `TimeStep` in the XML, so both codes step with
the same `dt` and the probe samples coincide.

**Dispersive materials** (§5.3, `compile::dispersive`,
`engine::dispersive`). Agreement 0.5–3e-6 for Debye, Drude, Lorentz (lossy,
resonance in band), magnetic Lorentz and two-pole materials, on slabs with
faces on the mesh and off it, and on a coarse filled line:
- Parameters: frequencies in Hz (`ω = 2π·f`), `EpsilonRelaxTime` a
  relaxation time (`γ = 1/τ`), `ω_p² = (2π·f_plasma)²·ε`. A 1 % change of
  `f_plasma` or 1.5× in `τ` changes the signals by 10–40 %.
- Poles are not averaged: an edge gets the full poles of the property at
  its midpoint (closed bounds, priority), and the `ε` in `ω_p²` is the
  edge's averaged static `ε` (on a vacuum interface `(1 + ε∞)/2`).
  Volume-fraction averaging (ems's earlier rule) was 5–17 % off on slabs.
- Lorentz and Drude poles use an explicit leapfrog ADE (branch current at
  half steps driven by `E^n`, Kashiwa–Fukai polarisation current;
  `engine::dispersive::Scheme::Leapfrog`, now the compile default);
  Debye poles the trapezoidal rule ems already had. ems's semi-implicit
  trapezoidal Lorentz scheme (`χ(Ω_t)`, unconditionally stable, exact
  energy balance) stays available; it is `O((ω·dt)²)` away from the
  leapfrog `χ(Ω)` and differs from openEMS by 1–7 % at 1 mm / 1.8 ps and
  5 % on a 4 mm / 7 ps line. The explicit scheme adds a stability limit
  (`ω_0·dt < 2`, `ω_p·dt` small) that ems does not check.

**Plane waves** (§5.11, `compile::planewave`). `Excite` is the incident
E vector and `PropDir` the propagation direction in both codes (same
signs, amplitude to 1e-4). openEMS's incident field lags ems's former
reference (`a(t)` at the entry corner) by `dt + Σ_a |k̂_a|·Δ_a/v`: one
time step plus the travel time across the box's first cell (measured for
`+x`, `−y`, 30° in the xy plane with both polarisations, `(1,1,1)`, on
uniform 1 mm and 0.5 mm meshes, a different `dt`, and a graded mesh with
0.5 mm cells at the entry face). ems now adds this lag, and ignores
`Delay` (with a warning): openEMS's output does not change with it.
Remaining agreement 4e-4…1.5e-3 a few cells into 3-D boxes, 2e-2 after 20
cells of 1 mm on a line (4e-3 at 0.5 mm): openEMS's own injection error,
which also leaks −37 dB (1 mm) / −53 dB (0.5 mm) into its scattered-field
region; ems's leakage stays below −120 dB, so scattered-field probes
cannot be compared. In an ε_r = 4 background openEMS's plane wave is not
usable (the scattered region holds a wave as large as the incident one);
ems's is (§5.11), so that case is not matched. The lag's travel time uses
the background speed, which only matters there. Boxes crossing layers
(wave 14, the 1-D line of `adv_pw_line_slab` with the `ε_r = 4` slab
across the exit or the entry face; scratch runs, not a fixture): openEMS
runs them and injects the same vacuum wave, so its scattered-field region
holds a wave of 0.75–0.9 of the incident amplitude (in the homogeneous
line 1.4 %). ems keeps the file path as it was — the layered surface is
a compile error naming the plane wave, dispersion on the surface is not
seen — and handles layered and dispersive backgrounds natively only
(§5.11).

**Conducting sheets** (§5.9). openEMS's `ConductingSheet` (a
"behavioural" model, per its documentation) is a different model from
ems's slab. Before, the sweep sheets differed by 90–190 % (resistive sheet,
sheet across the whole box) and 3e-4…1e-3 (copper; both nearly PEC); a
copper sheet across a two-cell TEM line lost 0.48–0.52 of the exact
slab's `≈ 2R_s/η0` with no shielding (−80…−90 dB through it), a resistive
sheet (`σt = 0.01 S`) transmitted −14…−34 dB below `2/(2 + η0σt)`. The
model was measured on the black box (operator dumps for the static
coefficients, per-step E/H dumps of a TEM line for the update, a least
squares fit of the sheet-edge recursion, residual at the `f32` level) and
ems now reproduces it for files (`compile/openems_sheet.rs`; native ems
keeps the slab):

- **Update.** A sheet edge is a capacitor `C` and conductance `G` in
  parallel with two series R–L branches, `C·du/dt + G·u + Σ i_b = ∮i`,
  `L·di_b/dt + R·i_b = u`, the branches integrated like openEMS's Drude
  poles (branch current at half steps driven by `u^n`, `k_a =
  (L/dt − R/2)/(L/dt + R/2)`, `k_b = 1/(L/dt + R/2)`; ems puts them on the
  edge as Drude poles of `engine::dispersive`'s leapfrog scheme). The
  cell's own capacitance and the medium around the sheet do not enter.
- **Scaling.** Per square, `G = g·σt`, `R_b = r_b/(σt)`, `L_b = l_b·μ0·t`
  and `C = dt²/(c·μ0·t)`: the capacitance scales with `dt²` (measured at
  1.0, 1.2 and 1.8 ps; it is a stability device, not a sheet
  capacitance). An edge of length `l` with sheet width `w` gets `w/l`
  squares, `w` the half widths of its two neighbouring sheet cells
  (measured on graded meshes and on the borders of partial sheets).
- **Table.** `(g, c, l_1, r_1, l_2, r_2)` depend only on the bin
  `k(ν)` of `ν = (t/δ)²/80` at the excitation's highest frequency (`f0 +
  fc` of a Gaussian, `f0` of a sine, the `<FDTD f_max>` attribute for
  custom, Dirac and step signals; a Gaussian ignores `f_max`): `k = 0`
  for `ν ≤ 1` (the bin edge measured at `t/δ = √80` within 1e-5 for
  three `σ`, `t`), `k = ⌈log_1.5 ν⌉` above, constant from `k = 29` on.
  All 30 bins were measured (σ = 1e6 S/m) and reproduce with other `σ`,
  `t`, `dt` and positions in the bin to 7 digits. Bin 0 conducts
  `0.9988·σt` at DC; from bin 26 on `r_2` is clamped at 1000.
- **Cells** (wave 14; the wave-13 rule, half widths of the counted
  cells, was right on uniform meshes only). An edge whose midpoint is in
  the sheet (the sheet wins the priority there; ties go to the sheet) is
  a sheet edge; a material or metal of higher priority at the midpoint
  removes it. Its width is the dual edge across it in the sheet plane if
  the edge is *interior*, half of it on a *border* edge, with no other
  values: interior means that the sheet contains the two points at the
  edge's start node and the centres of its two neighbouring cells (where
  the dual edges through the start node cross the cells). A neighbouring
  cell with its lower corner on the first mesh line of the axis across
  the edge never counts (so a sheet across a two-cell line carries half
  its current: the "half loss" above), and an edge in the first cell of
  its own axis is PEC. Measured on `--debug-operator` dumps: sheets
  starting on, below and between mesh lines on graded meshes (a step
  sweep of the border through a cell: the width switches at the cell
  centre across the edge and at the edge's start node along it, never at
  its end node), the corner of two off-mesh borders (half, not a
  quarter), and the edges along oblique polygon sides, where the
  start-node sample point decides (the cell-centre points at the edge
  midpoint, ems's former rule, missed 7 of the 60 border edges of a
  triangle); then 34 random polygons (3–5 oblique sides, every normal,
  uniform and graded meshes, copper and resistive) and the fixture
  `cli_debug_cyl_sheet`/`cli_debug_sheet` models: every E coefficient
  identical (`tests/cli.rs`). openEMS's time step ignores the sheet; ems
  resolves it before the sheet capacitances are set and pins it.

Agreement now (probe files): copper across the TEM line 3e-7 (reflected)
and 1.4e-6 (transmitted), a partial-width sheet on a six-cell line
4e-7…3e-6, the sweep boxes (`sweep_sheet_*`: copper, resistive, a sheet
across the whole box, a graded mesh with a dielectric and a metal block)
3e-6…7e-5 (graded 7e-6…1.8e-5, was 1.8e-5…5e-5), triangles with three
oblique sides (`sweep_sheet_triangle`, copper, 1.9e-5…4.4e-5, was
2.2e-5…5.4e-5; `sweep_sheet_triangle_r`, `σt = 0.1 S`, 2.2e-5…4.3e-5, was
4.0e-3…7.4e-3), the ring-down level of a lossy closed box; the operator
dumps are identical. Not reproduced: two sheets meeting on a line (openEMS
gives the line edges to the first sheet but the sample points to the
other; ems: csx's last sheet). `tests/openems_adv.rs` checks the copper
line and the `≈ 0.5` loss ratio of both codes.

**Waveguide mode probes** (§5.5, `compile/mode.rs`; resolves the open
convention of §10). Measured on a 20 × 10 mm guide (TE10, uniform 1 mm
mesh; TE11-like field on a graded mesh), with mode functions `sin`, `1`,
`x`, indicators of single nodes or cells, full and partial probe boxes:
- `ModeFunctionX/Y/Z` are read from the `<Attributes>` child of the
  `ProbeBox` (as csx stores user attributes); as attributes of the
  element itself openEMS reports a parse error. Its parser does not know
  `pi`. Coordinates are in **drawing units** (like excitation weights,
  also confirmed: an E source weighted `sin(πx/20)` in a 1 mm mesh
  matches to 1e-6).
- openEMS **normalises** the mode function: `V = Σ E_t·f·A/√(Σ |f_t|²·A)`
  (scale-invariant, in volts; `f_z` ignored), positive for `E` along `f`.
  The sums run over the box's nodes (voltage) or cells (current), never
  the outermost mesh lines; voltage: `f` at the node, dual area, `E`
  linearly interpolated from the two edge midpoints; current: `f` at the
  cell centre, cell area, `H` the plain mean of the dual edges at the
  cell's lower corner node. A third file column is the **mode purity**
  `w·(Σ E_t·f·A)²/(Σ|E_t|²A·Σ|f_t|²A)` over the same samples (`H_t` for the
  current; scaled by the ProbeBox weight `w` like the value): 1 for a field
  of the mode's shape, less otherwise (0.8497 for TE10
  against `f = 1` over 19 interior nodes, 0.6536 against `x`; the
  current's half-cell offset gives 0.9945 for TE10 against `sin`); weights
  3 and −2 scale it by 3 and −2. The frequency-domain file transforms the
  purity series like the value. The position lines report the sample
  range (box without the outermost lines; current: cell corners). ems
  records it (`ProbeResult::mode_purity`; agreement 2e-7…1e-5 absolute on
  `adv_wg_*`) and `io::openems` writes and reads the format.
- ems implements this as the default sampling (agreement 3e-7 uniform,
  4e-6 graded; a single-node indicator in a TE11 null 5e-4 from
  cancellation). The quirk of the current (`f` half a cell off `H`)
  makes `V/I` of a matched TE10 mode 0.28 % high at 20 cells, so
  `RectWaveguidePort` keeps ems's exact overlap integrals
  (`emsModeSampling="exact"`), with which a matched guide reads
  −54…−63 dB instead of −49…−63 dB.

**Cylindrical grids** (§5.15). openEMS selects its cylindrical solver
with `CylinderCoords="1"` on `<FDTD>` (a `CoordSystem` attribute there is
ignored by openEMS, which then runs the `(ρ, α, z)` lines as a Cartesian
mesh); `α` lines are radians. ems follows the grid's `CoordSystem` and
accepts `CylinderCoords` silently. Both codes step with openEMS's own
time step (§5.16.1; it is smaller than ems's operator bound near the
axis, 1.54e-13 vs 2e-13 s on a 0.5 mm / 15° mesh); numbers after the
regeneration on it in wave 14:
- without the axis (coax, annular cavity, closed azimuth): 1.6e-6…3.9e-5
  (`adv_cyl_coax`, `adv_cyl_noaxis`; their α probe 1.6e-4), also static
  dielectrics (with a fixed 2e-13 or 5e-13 s step: 1e-6);
- with the axis (closed azimuth): 6.5e-7…3.6e-6 (`adv_cyl_axis`; the
  coax-fed monopole `e2e_coax_monopole` 6.2e-7…3.3e-6, Z_in 4.7e-7) with
  openEMS's axis update on the file path (5.5e-7…4.4e-6 and
  4.6e-7…1.8e-6 at the fixed steps; 4.4–5.1e-3 and 1.7–2.2e-4 before the
  axis update).
  openEMS's operator dump gives exactly ems's coefficients for every
  edge next to the axis; the axis z edge is updated outside it. Per-step
  E and H dumps (`DumpMode` 0, every step, a 6 × 6 × 4 cavity with two
  asymmetric sources) fit `ΔE_z(axis) = dt·Δα/(ε0·π·ρ_½)·(2·H_α(α_0) +
  Σ_{j=1}^{N−1} H_α(α_j))` to a residual of 1.8e-6, regressed on the
  axis value, every `H_α` of the ring and every `H_ρ` at the axis
  (coefficients 30.12 for `α_0` and 15.06 ± 0.003 for the others, the
  exact disc coefficient being 15.058; `E_z` and `H_ρ` 0): openEMS sums
  the ring over all `N + 1` α lines, so the seam line counts twice (a
  uniform sum leaves 17 %). ems adds the missing seam term for files
  (`engine::cylindrical::OpenemsAxisSeam`); natively it keeps the exact,
  energy-conserving Ampère disc. Both TM010 frequencies converge at
  second order to the Bessel value (11.47425 GHz): −6.5e-4 (openEMS) and
  −7.0e-4 (ems) at `Δρ` = 0.5 mm, −1.6e-4 and −1.7e-4 at 0.25 mm. Sectors
  with the axis were not measured (openEMS keeps the E on PMC α faces at
  0, which includes the axis edge);
- `MultiGrid`: 3.0e-3…2.0e-2 (1–3e-2 before the axis update);
  openEMS's multigrid changes its own solution by 0.6–1 % against 0.1 %
  for ems's embedding (§5.15.1), so ems keeps its embedding (accuracy
  rule of §5.16; not fitted). Wave 14 established openEMS's structure
  from its time step (§5.16.1) and its console: a zone boundary on line
  `I` runs a separate coarse operator on the lines `ρ_0 … ρ_I` with every
  second α line (its own Gershgorin step, "split position @I+1") next to
  the full-mesh operator, which updates the edges with ρ index `> I`;
  it rejects splits that leave the outer grid few lines ("split
  invalid"), and its multigrid engine is always the multithreaded one.
  Its field dumps of the inner region are interpolated from the coarse
  lines (DumpMode 0 and `--nativeFieldDumps` alike; the odd α lines
  of the inner rings are not plain interpolants), so the per-step
  regression used for the axis update does not apply directly; the
  coupling of the two operators (which grid owns the ring `ρ_I`, how the
  fine H at `ρ_{I+½}` sees the coarse E) is not fitted, and
  `adv_cyl_multigrid` keeps its fixed `TimeStep` (wave 16: fitted, and
  run for files, §5.16.11);
- dispersive poles work in openEMS on cylindrical grids, and now in ems
  (midpoint rule in `(ρ, α, z)`, physical areas): 1.7–3.8e-3 on a
  Drude-filled coax at openEMS's step of 1.8e-12 s (0.5–1e-3 at a fixed
  5e-13 s; static `ε` alone 1e-6);
- plane waves (`cyl_planewave`): openEMS ignores a `PlaneWave` excitation
  on a cylindrical grid (fields identically zero, no message, for
  propagation along z and along ρ; the same box on a Cartesian mesh runs);
  ems drops it with a warning;
- SAR dumps run there in openEMS, including averaged ones (its warped
  cubes, reproduced to ~1 %); see §5.12 (`cyl_sar`);
- `CellConstantMaterial` samples the cell midpoints in both codes;
  openEMS's radial weights at radial interfaces (z edges `Δρ_h·ρ_b`, α
  edges and H_ρ faces `Δρ_h`) are reproduced on the file path
  (`cyl_cellconst` 5e-7…8e-6, `cyl_radial` 1.4e-7…3.2e-6; before
  1.2e-4…1.7e-3 and 3.2e-4…4.4e-3, §5.15); boxes past α = π and the
  α = 2π collapse of lumped elements: §5.15 (`cyl_wrap_*`);
- the debug dumps (`--debug-material`, `--debug-operator`,
  `--debug-PEC`): `.vts` structured grids with Cartesian points, the
  same values wherever both codes describe the same edge (§5.15);
- conducting sheets (`cyl_sheet_*`, half-wave coaxial cavity of §5.9,
  `Q` from the ring-down): openEMS's sheet model behaves as on Cartesian
  grids. A thick lossy inner tube (`σ = 1e5 S/m`, `t = 100 µm`) gives
  `Q = 403` (ems natively 195.4, analytic 195.9): half the surface loss,
  as for copper on a Cartesian line. A thin resistive fin
  (`σt = 1e-4 S`) at `α = π/2` detunes the mode by −13 % (`Q = 139`; ems
  natively: no detuning, `Q = 57.7`, analytic 57.65): nearly reactive.
  The same fin on the α = 0 seam has no effect in openEMS. Since wave 14
  ems runs openEMS's model on cylindrical grids too (files, no azimuthal
  multigrid; `compile/openems_sheet.rs`), measured on operator dumps of
  about 70 cylindrical models (tubes, discs and fins, graded radial
  meshes, sectors through and around the seam): the Cartesian rules with
  physical lengths (an α edge `ρ_i·Δα`, a dual width along α `ρ_i·Δα*`
  at the edge's start node `ρ_i`, e.g. a disc's ρ edges). The seam of a
  closed azimuth: openEMS's operator holds the ρ and z edges at `α = 0`
  as zeros and updates the copy at `α = 2π` (line `N`), whose
  neighbouring cells along α are `N − 1` and `N`, a copy of cell 0 that
  is not subject to the first-cell rule (a full tube's seam edge is
  interior, the edge at `α_1` a border edge); the α edges of cell 0 are
  PEC by the first-cell rule (a probe on one reads 0). openEMS tests box
  angles as written: a fin drawn at `α = 0` is not on the `2π` copy (no
  effect, the fixture's result), one drawn at `2π` is; a point past 2π
  (cell `N`) also matches as `α − 2π` in a box of nonzero α extent; a box
  `[0, a]` contains the seam edge from `a ≥ α_{N−1}` on (tested
  `a` = 0.8…2π; a box `[−1, 1]` is the one case measured that this
  does not describe: its seam edge is not a sheet edge). Agreement
  (`tests/openems_cyl.rs`): the four cavities 8e-7 (empty, seam fin),
  2.2e-5 (fin), 2.7e-5 (tube), against 2.1 (`Q` ratio) and a 13 %
  frequency shift before; operator dumps identical
  (`cli_debug_cyl_sheet`). Residual: next to the seam a sheet ρ/z edge
  differs by 5e-3 (nearby probes 2e-4) when the field there is driven
  directly (a source on the adjacent α edge); C, G and the branches of
  that edge were each varied (±0.5 % on `C` gives 1e-3…1e-2), so it is in
  openEMS's handling of the seam copies, not resolved.

**Current probes on a closed azimuth.** A current box over the full
azimuth (α from 0 to 2π, the usual coax port current) reads `(N − 1)/N`
of the enclosed current in openEMS for `N` α cells (7/8, 3/4 measured):
the box's upper α snaps to the last line before 2π and the loop covers
α_0 … α_{N−1}; ems closes the ring (the full current, the rule of
§5.16) and keeps that difference (`e2e_coax_monopole`). Sector boxes
touching the seam (wave 14, fixture `cyl_current_seam`: z-normal boxes
on a closed coax excited over a quarter of the ring, so that H_α varies
with α; `--showProbeDiscretization`): openEMS runs the loop on the dual α
lines from the one below the lower corner to the one above the upper
corner, clamped to its dual lines `α_{1/2} … α_{N−1/2}`, so `[0, π]`
encloses the E_z columns 1…4 like `[π/4, π]`, `[π, 2π]` the columns
4…7, `[−π/2, π/2]` the columns 1…2, and `[0, π/4]` collapses to a line
(openEMS disables it: "not a surface"). ems had included the seam column
with half its dual edge (4.5/8 for `[0, π]` on a uniform field; 1.7–7.4
relative L2 here). Files now clamp the α columns of a loop that touches
the seam on one side to `1 … N − 1` (`sim::map`, the current-probe
branch; a box over the whole ring keeps ems's closed loop, 0.11 apart
here, and `[0, π/4]` is dropped with a warning): 2.5e-4…1.4e-3, the
level of a box away from the seam on this model (`[π/4, π]` 2.5e-4).
openEMS's probe file headers
print α times the drawing unit like the other coordinates (ems does now),
and it rejects α lines spanning more than 2π by a rounding error
(6.28319 written with 6 digits: "Alpha Max-Min must not be larger than
2*PI"), which ems accepts.

**Box snapping of excitations.** openEMS assigns no edge to a flat
excitation box `4e-8` (drawing units, 3e-9 relative) off a mesh line
("Unused primitive" warning at engine setup, the field stays zero); ems
snaps within `1e-6` of the local cell (`CompileOptions::snap_tolerance`)
and keeps that tolerance.

**Lumped elements** (see also §5.16.1). On a PMC-walled line with the
element across the full width both codes give every edge `R·N/M` (the
operator's boundary dual cells are full), so the probes agree to 6e-7:
the "75 Ω for 50 Ω" is the TEM wave seeing the half-width boundary
columns, in both codes. On a closed-azimuth coax ring openEMS spreads `R`
like a conductivity (`R_e ∝ 1/ρ`, which ems now does) but normalises it
over 17 α columns for 16 α cells, counting the α = 2π copy of each edge
as a parallel column: the ring acts as 21.25 Ω for 20 Ω. ems keeps the
nominal value (a genuine error of openEMS's, rule of §5.16): 1.5–4.5e-2
difference; with `R = 20·17/16` in ems the codes agree to 1e-6
(`adv_cyl_lumped_ring`).

**Mur and PML** (§5.6). A Mur face without `MUR_PhaseVelocity` is tuned
per edge to the local medium in openEMS: in an ε_r = 4 line it absorbs
to 0.5 %, where ems's former vacuum tuning reflected 34–49 %. ems now
does the same (`MurBoundary::with_faces_in`): 1e-6 agreement in ε_r = 4
and in vacuum. On layered faces (a microstrip substrate meeting the
face) the per-edge medium is not the averaged edge material: per-edge
fits of the Mur coefficient from openEMS's per-step dumps give the
material at one point, the edge's start node in the face's transverse
coordinates and the middle of the boundary cell along the normal (closed
bounds): an `E_z` edge starting on the substrate's top face takes the
substrate's `ε`, the interface `E_y` edges `ε` instead of the average,
and a substrate ending on the last interior line does not count, one
ending between it and the face does. ems samples the same point for
files (`MurMedia`, `compile::mur_point_media`): microstrip lines into a
Mur face went from 3e-2 to 3e-7, the `e2e_msl_filter` ports agree to
4e-7.

**openEMS's PML** (`PML_n`; `engine::boundary::upml`, fixtures
`pml_*`, `gen-pml.sh`, `tests/openems_pml.rs`). openEMS's `PML_8`
reflects −58 dB on a vacuum line at normal incidence (1 mm cells, 5 GHz
pulse; opposite sign to ems's residual), ems's CPML −86 dB, so runs with
PML differed by about 2e-3 (vacuum) to 4e-3 (ε_r = 4), 6e-4 on the patch
antenna probes, 3e-3 on the microstrip filter with PML line ends, 0.8–13 %
in the sweep's PML models and 2.4e-2 at the source-side port of a WR-90
iris (`PML_8` ends, 7–13 GHz pulse, 6.6 GHz cutoff; a reference with the
guide extended by 0.8 m put ems's CPML closer to the truth on every port).
Its grading cannot be changed from the file (`PML_Grading` has no
effect, §5.16.1), so for files ems now runs openEMS's layer, measured as
a black box:

- *What it is.* With `-v 3` openEMS lists one "Uniaxial PML Extension"
  per disjoint box of layer nodes (faces first, then the remaining parts
  of the overlapping ones). Its `--debug-operator` dump shows, inside a
  layer, `vi = dt·l/A` (no `ε`) and `vv = (1 − a)/(1 + a)` on the
  components whose first (flux) stage is stretched: for an x layer
  `E_z`, `H_z` (graded) and `E_x`, `E_y` (`vv = 1`); for a y layer `E_x`,
  `H_x`: the cyclic assignment of Gedney's UPML (component `a`'s flux
  is stretched along `a + 1`, the field from the flux along `a + 2` and
  `a`). Per-edge least-squares fits of per-step E/H dumps (raw HDF5, a
  Dirac source, every step) of small models (one face; xmin + graded
  xmax + ymax with corners; ε_r = 4, μ_r = 2, κ, σ_m filling the layer;
  both face sides) reproduce every edge of the layers with the two-stage
  update of `engine::boundary::upml` to 2–4e-7 of each edge's peak (f32
  round-off), with `a_w = σ_w·dt/(2ε0)` evaluated at the edge's midpoint
  along its own direction and its nodes (E) or dual lines (H) across, and
  the faces' `a` adding where layers overlap.
- *Grading.* `σ(d) = ln(10⁶)·ln(2.5)/(2Δ·Z0·(2.5ⁿ − 1))·2.5^(d/Δ)` at
  physical depth `d > 0` below the interface line, `Δ = W/n` the mean
  cell of the layer (graded layers: physical depth over the mean cell,
  not the local one), 0 at the interface: the dumped `vv` agree to 3e-8
  (f32). The same `σ` (with `Z0`, `ε0`) in any medium.
- *Media.* The layer keeps the edge's `ε` and `μ` (averaged as
  elsewhere) but drops its electric and magnetic conductivity; the
  interface-plane edges keep their ordinary lossy update.
- *Sources in a layer.* A soft source on a layer edge adds to the field,
  not to the flux: its contribution stays undamped in E (a Dirac source
  on tangential layer edges keeps a static field there). ems keeps the
  two-stage update with flux for the driven edges (`with_driven`) and
  takes the field value back after the serial hooks (sources, axis and
  periodic merges).
- *Energy.* openEMS's end-criterion energy counts the layers with their
  true fields (a patch with `PML_6` faces stops at the same step, 1956, at
  −50.509 dB against openEMS's printed −50.518, its usual offset; with the
  layer's normal H missing ems stopped at 1908).
- *Cylindrical grids.* openEMS runs the same layer on `ρ_max` (the
  profile along `ρ`) and on the z faces; ems too, for files (instead of
  the stretched-radius PML of §5.15.2).

Every layer edge gets `ca = 0` and the flux-stage `cb`, so the kernel
writes the flux increment, and a slab hook completes the update from two
stored `f32` (flux, start value), in openEMS's arithmetic since wave 16
(§5.16.8; before, the tangential components of a single layer were
collapsed into one lossy update, 1e-6…2e-5 from openEMS). Coefficient
rows are kept once per distinct row.
The kernel's look-ahead (end-criterion energy, the last H sample of
probes and dumps) gets the true H through `SlabHooks::energy_openems_slab`
and `Operator::lookahead_current`. Cost: 100³ with `PML_8` on all faces
runs at the CPML's speed (8 threads 1.8–2.1 vs 1.8–2.0 GCells/s, 16
threads 2.9–3.0 vs 2.4–2.7; a serial energy correction had cost 30 %). Wave 16's state on every layer edge costs
20–30 % of that (§5.16.8).

Agreement with openEMS (relative L2 of the probes): the vacuum line
`adv_pml_vacuum` 6e-7 (the same −58 dB reflection), `pml_face` (one
layer, oblique incidence, probes at depths 2 and 5) ≤ 3e-6,
`pml_corner` (overlaps, graded layer, lossy block, metal plate and a
source in layers) ≤ 5e-6 with raw line dumps through two corners at
2e-6 (E) and 5e-5 (H, the round-off of the static field the source
leaves in the layer), `pml_medium` ≤ 1.5e-6, `pml_end` 3e-6 with the
same stop step, `pml_cyl` (ρ_max and z layers, closed azimuth, no axis)
≤ 7e-6, the sweep's `pml_open` 4e-6 (8e-3) and a PEC guide below cutoff
with `PML_8` ends 1e-6 (2.6–4.7e-2), the patch antenna 1.4e-7…2.7e-6
(6e-5…6e-4), the WR-90 iris with `PML_8` ends (`pml_wg_iris`) 3e-7…7e-7
with the same stop step (2.4e-2), the microstrip filter with `PML_8`
line ends (`pml_msl_filter`) 2e-7…5e-7 on the ports, 1e-5 on the stub
field (3e-3).

*Lumped elements in a layer* (wave 14; probe files of a vacuum box with
`PML_6` on x_max, elements across and inside the layer, compared with the
same box without them): openEMS's R and C have no effect on layer edges
(probe files bit-identical with and without a 50 Ω resistor, a 1 pF
capacitor or an R‖C on lines inside the layer; the operator dump does not
change either), they act on the edges outside it (an element across the
interface plane x = 14 changes the fields through its outer columns
only, with the per-edge values of the whole element) and on the interface
plane; the element's caps (PEC) and an excitation over it act inside the
layer too. With an inductor (R‖L) or a series element (LEtype 1) inside
the layer openEMS diverges (NaN within 800 steps). ems had folded the R
and C into the layer's rewritten coefficients and diverged (R‖C) or ran
a different update (R: 0.25–0.99 relative difference). Files now leave
an element's edges inside a layer out after spreading its values over
all its edges (`sim::map`, `upml::PmlLayers`; caps kept, the excitation
a plain soft source, a warning), inductors and series elements included
(openEMS's divergence is not reproduced, the warning says so). Fixture
`pml_lumped` (a resistor across the interface, an R‖C box with caps
and a 50 Ω port inside the layer, Mur on the other faces): 4.7e-7…1.1e-5
on the voltages and currents, 2.5e-5 / 6e-5 on the H / E probe three
cells deep in the layer (was ~1e7: diverged); in a closed PEC box with
the same layer the codes differ by 5e-5…3e-4 with and without the
elements (the ring-down's round-off).

*Sources in a layer, round-off* (wave 14): the `pml_corner` H line dump
(5e-5 of its own largest value) is 1.8e-6 of the H scale of its E field
(`E_max/Z0`; the line runs through a weak-H region of the layer): f32
round-off of the static field the source leaves in the layer, not a
formulation difference (with the source moved out of the layer the same
dumps agree to 3.6e-6 and 4e-7). Not emulated (it would need openEMS's
operation order in single precision).

Not reproduced: on cylindrical grids with the axis both codes differ
near the axis also without PML (`pml_cyl` has none).

**Current probes** (§5.16.1). On a graded microstrip mesh the voltages
agree to 6e-7; the current probes (flat boxes on primary x planes around
the strip) differed by 33 % with ems's former loop placement. openEMS's
file header shows the indices it uses: the box's lower corner moves down
by one line on every axis, the upper stays. So its H loop runs on the
dual lines just outside the box in the transverse directions (as ems's
does), but on the dual plane *before* a flat box's primary plane `k`
(`k − ½`), where ems used `k + ½`. For a box coordinate between mesh
lines openEMS takes the dual plane nearest to it (a box at a dual
position, e.g. z = 6.5 on a 1 mm mesh, uses that plane), so the rule
needs the unsnapped coordinate (`ProbeSpec::corners`). ems now follows
it, and the currents agree to 6e-7 as well. Where the box touches the
outer mesh lines openEMS's loop stays inside the domain (a box over the
whole cross-section of a PMC/PEC line reads about half the TEM current,
ems 0).

#### 5.16.3 Feature coverage

A sweep over everything csx's XML reader understands (primitives,
transformations, priorities, properties, FDTD attributes and boundary
variants, parameters), one small hand-written model per feature, run in
both codes (openEMS single-threaded `--engine=basic`, ems through
`read_model`) and compared file by file: probe and dump files, and the
`--debug-PEC` / `--debug-material` dumps, which isolate the discretisation
(identical PEC edge sets and edge materials pin a difference on the
solver, a different edge set on the geometry). Most models are a 20³-cell
PEC box of 1 mm cells, a soft Gaussian line source, five E/H/voltage
probes and the feature under test in the middle, 600 steps of 1.5 ps;
"agreement" is the relative L2 difference over all probe columns, ~1e-5
(`f32` round-off over the ring-down of a closed box) where the
formulations agree; since wave 15 vacuum and metal models agree bit for
bit and dielectric ones to 2e-5…1.5e-4 (§5.16.6). 62 of the models are
fixtures (`sweep_*`,
`scripts/openems-oracle/gen-sweep.sh`, `tests/openems_sweep.rs`, 0.4 s).

Matrix (175 models; 165 agree to ≲ 1e-4, 7 differ, 3 are rejected by
openEMS — with openEMS be30220 the six polyhedron and rotated-cylinder
ties below agree too, 171 agree, §5.16.5; plus 30 boundary spellings and 30 odd `<FDTD>` attributes checked
on the set-up; wave 13 added the σ_m, sheet and LinPoly-vertex models and
closed their differences; wave 14 replaced the polygon sheet model by two
committed triangles, `sweep_sheet_triangle{,_r}`, and closed it):

| area | models | agree | differ (magnitude) |
|---|---|---|---|
| primitives: Box (on/off grid, reversed, metal volume/sheet/line), MultiBox, Sphere, SphericalShell, Cylinder (axis, oblique, thin), CylindricalShell, Polygon (z, x), LinPoly (z, y), RotPoly (full, half), Polyhedron, PolyhedronReader (STL, PLY), Curve, Wire, Point, UserDefined, Box in cylindrical coordinates; surfaces through edge midpoints and sample points (`tie_*`) | 47 | 40 (45 with be30220) | Polyhedron surfaces on sample points 0.5 % (material) – 2 % (metal; 34 % for a metal cube whose faces lie on mesh lines), with openEMS e89286f; UserDefined rejected by openEMS |
| transformations: Translate, Rotate_X/Y/Z, Rotate_Origin, Scale, Scale3, Matrix, order, on spheres, cylinders and sheets | 12 | 11 (12 with be30220) | a rotated cylinder with sample points within 1e-13 of its radius: 4.6 % with e89286f, 4.7e-5 with be30220 |
| priorities: overlaps, reversed, metal in material and under it, two metals | 6 | 6 | |
| materials: ε, κ, μ, σ, `Isotropy` with 1/2/3 values and weights, weights of x, y, z, ρ and per axis, Density, background, Lorentz/Drude/Debye (with weights, with κ), σ on a graded mesh with weights and anisotropy | 20 | 20 | |
| parameters, mesh: ParameterSet (Const, Linear) in coordinates, values and weights, `term:`, unsorted/duplicate lines, DeltaUnit 1, graded | 9 | 7 | `term:` prefixes and expressions in mesh lines rejected by openEMS (ems reads them) |
| excitations: soft/hard E and H, vector, weights (with parameters), delay, Enabled, Number, on metal, off-grid, on Sphere/Curve, 3-D box, transformed, two sources | 16 | 16 | |
| probes: weights, solid/off-grid/point E and H boxes, diagonal and reversed voltage boxes, current NormDir variants, FD samples of all types, Start/StopTime, two primitives, transformed, Cylinder/Curve/Point/Polygon primitives, ten shapes for field probes and dumps | 17 | 17 | (wave 13: current-probe corners half-way between lines snap inwards, zero-length Cylinder/CylindricalShell = point box; fixture `current_probe_ties`, 2e-8…2e-6, was 4.6 % resp. skipped) |
| dumps: VTK/HDF5, SubSampling, OptResolution (modes 0–2), FD, without DumpMode, two primitives, transformed | 8 | 8 | |
| boundaries: names, numbers, case, Mur phase velocity, mixed, PML_n (3, 4, 8, `PML`), AbsorbingBC (both signs and types, in a box) | 17 | 17 | (PML: 0.8 % in an open box to 4–13 % in a PEC guide below cut-off with the CPML; ems runs openEMS's layer for files since wave 13, §5.16.2: `pml_open` 4e-6, a re-built guide 1e-6) |
| lumped elements, conducting sheets: Caps, C only, series RLC, on a Cylinder; copper, resistive, whole-box, graded sheets and triangles with oblique sides (copper, resistive) | 10 | 10 | (wave 14: oblique sides were 2.5e-3…7e-3, §5.16.2) |
| FDTD: OverSampling, f_max, TimeStepFactor, TimeStepMethod 1 and 3 (graded), MaxTime, CellConstantMaterial, endCriteria, CoordSystem, 0 steps, odd values | 11 | 11 | |
| other properties: unknown property element, properties without primitives | 2 | 2 | |

Fixed in ems by the sweep (each was 2–270 % off, or did not run):

- **Reading rules** (`ems::io::openems::read_model` / `parse_model`, used
  by `Simulation::from_xml`, `from_xml_str` and the `ems` binary; they
  rewrite the raw XML before csx reads it):
  - boundary values: `PEC`, `PMC`, `MUR`, `PML_<n>` are case sensitive;
    a value starting with a digit or sign is its integer prefix (`1.7` →
    PMC, `2.000000e+00` → MUR, `0x1` → PEC; 3 → `PML_8`, other numbers
    PEC); `PML_<n>` takes the integer prefix of `n` (`PML_08`, `PML_8.7`),
    `n ≤ 0` or no digits is PEC; any other spelling (`pec`, `Mur`, `PML`,
    `PML 8`, `" PEC"`) is PEC with openEMS's "unknown" warning. Measured
    by matching the probe signal of each spelling against the four
    boundary kinds (30 spellings);
  - a field dump without `DumpMode` is mode 0 (§8.1 item 13);
  - `Isotropy="0"` with fewer than three values (or weights) sets the
    leading components only, the others keep the default (ε_r = μ_r = 1,
    κ = σ = 0, weight 1; `--debug-material`): csx copied a single value to
    all axes (60 % off) and filled a short list with 0 (no run);
  - a `ProbeBox`/`DumpBox` with N primitives is N probes `<name>_<i>`
    (files `u2_0`, `E2_0.h5`, …); a primitive's `<Transformation>` is
    ignored for probe and dump boxes (the raw P1/P2 box is used; excitations
    and materials do apply it); a voltage probe on anything but a `Box`
    (MultiBox, Wire, Curve, Cylinder) is skipped, field probes and dumps
    take the bounding box of any primitive (as ems did); lumped elements
    on non-Box primitives are skipped;
  - `OverSampling` < 2 is the default 4 (also negative, which csx
    rejected); `TimeStepFactor` outside (0, 1] is ignored; a negative
    `endCriteria` runs every step (`Fdtd::from_csx`);
  - a run of 0 steps writes the t = 0 row of the E-type probe files.
- **Metal surfaces** (`compile::openems`, files only): openEMS
  e89286f's metal Sphere, SphericalShell and Wire were open (an edge
  whose midpoint lies on the surface is not PEC), Box, Cylinder and
  CylindricalShell closed, as csx is everywhere; ems cleared those edges
  (identical `--debug-PEC` dumps on every metal case of the sweep).
  openEMS be30220 is closed for every primitive, with a tolerance
  (§5.16.5); ems dropped the rule.
- **AbsorbingBC**: `NormalSignPositive="1"` absorbs waves travelling
  along `−a` (the normal points into the terminated region); ems had the
  opposite sign. Now 1e-6 on a terminated TEM line; MUR_1ST_SA gives
  openEMS's MUR_1ST output bit for bit there.
- **Magnetic conductivity** (`compile::openems`, files only): openEMS
  takes the length-weighted harmonic mean of the two half-edge samples
  (the same samples as csx's: the face centre transversally, the middle
  of each half of the dual edge), like `μ`, but a zero upper half gives
  0 and a zero lower half is left out of the denominator:
  `(l_− + l_+)·σ_+/l_+`, the "`2σ` on the entry face, 0 on the exit
  face". Measured with `--debug-material` on slabs on uniform and graded
  meshes (`2σ`; 6000 for 2000 S/m on a 0.5/0.25 mm split), all three
  axes, pairs of lossy media (1000|3000 → 1500, 500|3000 → 857.14),
  weight functions and anisotropic values; `μ` follows the harmonic rule
  on the same samples (as in csx), and metal does not enter (the samples
  see the material under it). Identical material dumps;
  `sweep_mat_mue_sigma` 6e-3 → 2.3e-5, `sweep_mat_sigma_graded`
  9e-2 → 8.8e-5.
- **Conducting sheets** (files, Cartesian grids): openEMS's model
  (§5.16.2); the sweep sheets 0.9–1.9 → 3e-6…7e-5.
- **Polygon vertices** (`compile::openems`, metal, files only; openEMS
  e89286f, dropped with be30220, §5.16.5): the sides of a Polygon or LinPoly are closed, a point on a vertex gets a
  winding-number test with half-open rules (upward edge `a → b`: `+1`
  for `a_v < q_v ≤ b_v` with `q` on or left of it; downward: `−1` for
  `b_v < q_v ≤ a_v` with `q` strictly right of it; outside if the sum is
  0; in the polygon's `(X1, X2)`). It reproduces all 33 vertices of 10
  measured polygons (convex and concave, both orientations, `NormDir` 1
  and 2): a counter-clockwise polygon is open at its vertices of
  minimal `X2`, a clockwise rectangle at all four corners. ems clears the
  edges through such vertices (identical `--debug-PEC` dumps;
  `sweep_tie_linpoly_metal` 8–11 % → 5–8e-5). Material samples on
  vertices (vertices on quarter-cell positions) keep csx's closed test.
- **Dump sampling**: `OptResolution` compared metre distances without
  tolerance (a line exactly one resolution away was taken: 2, 4, 6, 8,
  11, … instead of every second line); raw H/B dumps snap their box to
  the dual lines (openEMS: H at 5.5 … 12.5 for a box 6 … 12).

**Randomized differential sweep** (wave 14; `scratchpad` generator, not
committed: per seed a random graded mesh of 10–16 cells per axis, each
face PEC, PMC, Mur or `PML_3…6`, up to three material boxes (ε, μ, κ,
σ_m, Drude, Lorentz), metal boxes and sheets on and off the mesh,
conducting sheets, up to two lumped elements (R, C, R‖C, R‖L, series
RLC, with or without caps and an excitation), a soft or hard E or a soft
H source, three to six probes (some off-grid, some with FD samples), a
raw line dump, Gaussian pulses, energy end criteria; plus raw E and H
dumps of the whole domain, the global comparison). A probe counts as a
disagreement above 1e-4 relative L2 unless its difference is below
1e-5 of the dump field scale at its geometry (f32 round-off of a weak
signal: a vacuum PEC box showed 1e-3 on probes at 1e-6 of the field
maximum while the whole field agreed to 1e-6); runs where both codes
diverge count as agreeing. 230 models: before 77 of the first 100 and,
after the fixes of the first 130, 90 of 100 fresh ones agreed to 1e-4;
now 212 of 230 (19 of them unstable in both codes, from R‖L elements with
openEMS's capacitance limit). Found and fixed (files; each with a
fixture `rs_<seed>`, `gen-rsweep.sh`, `tests/openems_rsweep.rs`):

- **Thin metal under materials**: a material of higher priority (or
  equal, defined first) at a metal sheet's or line's edge midpoint wins
  the edge (`compile::openems::thin_metal_priority`; csx makes sheets PEC
  regardless; `rs_3` 0.97 → 6.5e-6, `rs_29` 0.4 → 6.3e-5).
- **Dispersive poles in openEMS's PML** have no effect (the static ε
  remains; bit-identical probe files with and without the poles inside a
  layer): ems leaves them out (`BoundaryHandles::pml_poles_ignored`,
  warning; `rs_0` 0.34 → 5e-6).
- **Mur and dispersive media**: openEMS's Mur faces read their interior
  neighbours before the dispersive correction of their update (a Mur face
  next to a Lorentz medium fits no 1-D update of the corrected values);
  ems registers the Mur operator before the poles for files (`rs_1` 0.4
  → 2e-5).
- **Mur domain edges**: the lower axis's face alone (above; `rs_11` dump
  0.64 → 5e-7).
- **Hard E sources** have no update of their own in openEMS: a Mur face
  next to one reads 0 for it (`u_0^{n+1} = u_1^n − k·u_0^n`); ems zeroes
  the coefficients of hard E source edges outside PML layers (`rs_12`
  1.5 → 3e-5).
- **A Mur face carrying an E excitation** is not updated while the
  excitation runs: the face stays 0 except the source edges, which hold
  just the signal, up to step `L + 11` (`L` the excitation length; four
  lengths and two time steps; the other Mur faces are unaffected)
  (`MurBoundary::hold_faces`; sweep case 134 12 → 2e-6, 162 12 → 3e-2:
  the H probe beside the face is still off).
- **The look-ahead H** of the last dump and probe sample includes H
  sources (`Operator::lookahead_sources`; a soft H source's dump was 0.4
  off at its last step).
- **Raw H dumps** snap to cells like H probes (the cell holding the lower
  corner, closed above, to the cell holding the upper one, closed below;
  graded meshes; `rs_0`, `rs_29`).
- **Series lumped elements**: the capacitance limit (§5.16.1; `rs_109`
  0.3 → 3e-4, the weak E probes).
- **Conducting sheets**: a border edge's width is half the dual length
  (not half the border cell; `rs_99` 74 → 2.4e-4, the H probe beside the
  sheet corner), and the lines an absorbing boundary claims behave like
  the first mesh line: on a lower face lines 0…1 (Mur) or 0…n (`PML_n`),
  on an upper one the last 2 or `n + 1` lines; a sheet cell whose lower
  corner lies on one does not count and a sheet edge starting on one is
  PEC (`--debug-PEC`/`--debug-operator` dumps of sheets starting and
  ending at every line near Mur, `PML_3` and PMC faces; `rs_116` was
  unstable).
- the H normal to an upper PML face (above).

Not resolved (18 of 230): series elements (9 with an excitation over a
csheet-sided box, 3.6; 61, 101, 160, 190, 219, 225 with or without an
excitation, 2e-4…3.6e-2), the H probe next to a held Mur face (162,
3e-2), conducting sheets with dispersive or lossy media near Mur/PML
faces (56, 63, 71, 77, 152: 2e-4…5e-2 in the H dumps), a Lorentz layer
with a hard source and a port (185, 0.99) and a Drude/PMC model with an H
source (157, 2.0) that look like a slow growth in one code, an R‖L
element where openEMS keeps a few % of an edge's capacitance (55, both
growing at different rates), and 51 and 85 (6e-4, 3e-3; not analysed).
ems's `--debug-PEC` dump does not list lumped caps (openEMS's does).

Not reproduced (documented differences):

- **Polyhedron surfaces** (openEMS e89286f; be30220 is closed and
  symmetric, §5.16.5). A point exactly on a face of a Polyhedron or
  STL mesh is inside or outside depending on the face *and on where the
  polyhedron is*, like a ray-casting test whose ray depends on the
  primitive: a cube's faces on mesh lines are inside on +x/+y/+z and
  outside on −x/−y/−z for the cubes `[3, 9]³` and `[−3, 3]³`, all inside
  for `[−9, −3]³`, and only −y/−z inside for `[−7, −5]³`; the same prism
  face (outward normal `(2, −1, 0)`) is outside at `x = 2…6` and inside
  at `x = 12…16`; on the STL tetrahedron of `sweep_prim_stl` one face is
  inside and three outside. No fixed direction (fitted from 20 prism
  faces it would have to lie within `g_y/g_x ∈ (2.5, 3)`,
  `g_z/g_y ∈ (10, 15)`, which the tetrahedron contradicts) and no fixed
  target point explains it; the points along a cube's edges are mixed
  even within one row. Not emulated (csx's test is closed:
  `sweep_prim_stl` 5e-3).
- **Round-off at ties** (openEMS e89286f; be30220 has a tolerance of
  `1e-3·h_min`, emulated, §5.16.5). openEMS compared without tolerance; csx's
  inside tests accept points within ~1e-9 of the size outside, and its
  transformations round differently from openEMS's (a rotated cylinder
  through sample points: 4.6 %; the points lie within 1e-13 of the
  radius, on either side). Exact comparisons on the ems side would need
  csx's inside tests without tolerance and openEMS's transformation
  arithmetic bit for bit: not feasible in ems (the tests run inside
  `csx::discretize`), §8.1 item 17.
- **PML**: see §5.16.2.
- openEMS rejects `term:` prefixes, expressions in mesh lines and the
  `UserDefined` primitive as written by csx (ems reads them); ems
  rejects `endCriteria` > 1, a Gaussian with `fc = 0` and a sinusoid with
  `f0 = 0` (openEMS runs the first two with an empty signal and crashes
  on the third).

#### 5.16.4 End-to-end models

Five hand-written models that use many features together run in both
codes (`gen-e2e.sh`, `tests/openems_e2e.rs`; the full-size versions were
compared the same way during development):

| model | features | agreement (relative L2 unless noted) |
|---|---|---|
| `e2e_patch` | 2.4 GHz probe-fed patch on lossy FR4, lumped 50 Ω port with excitation, PML_8, FD probe, NF2FF box (nf2ff job) | probes 3e-7…2.7e-6 (own time step; 1.4e-7… with the rounded one), Zin 7.6e-7, resonance identical (2.29 GHz, −15.7 dB), far field 1.5e-6, Dmax and Prad < 1e-5 (with the CPML: probes 6e-5…6e-4, Zin 9e-4, far field 1e-3) |
| `e2e_msl_filter` | microstrip open-stub low-pass, soft feed, 3 V + 2 I probes per port, Mur on every open face, FD dump | ports 6e-7…1.1e-6 (own time step, 7e-8 apart), `U2/U1` 9e-6 dB, dump 1.3e-3 of max |
| `e2e_waveguide_iris` | WR-90, inductive iris + dielectric post, weighted TE10 source, mode-matching V/I probes, Mur ends, energy end criterion | 4e-7, purity 1e-6, same stop step, `U2/U1` 7e-7 |
| `e2e_coax_monopole` | coax-fed λ/4 monopole, cylindrical grid (closed azimuth, axis in metal), 1/ρ weighted source, Mur faces | V and fields 7.4e-7…1.8e-6 (2e-4 before openEMS's axis update, §5.16.2), I after the `(N−1)/N` factor 4.6e-7 |
| `e2e_drude_slab` | plane wave on a Drude slab + lossy Lorentz layer, E/H probes with FD samples, FD dump | 7e-4…1e-3 (openEMS's plane-wave injection), R/T 9e-4 |

At full size (2–5× the cells, the same physics; before wave 13, with the
CPML): patch 2e-4 with the far field 1e-3 and the same stop step;
microstrip filter with `PML_8` line ends 3e-3; WR-90 with `PML_8`
2.4e-2; the Drude slab with `PML_8` 1e-3. The PML variants of the filter
and the iris are now fixtures (`pml_msl_filter`, `pml_wg_iris`, 2e-7…7e-7,
§5.16.2). They led to the energy measure (§5.16.1),
the Mur corner and point-medium rules (§5.16.1, §5.16.2), the α header
scaling, the Rennings time step on layered doubly graded meshes
(§5.16.1), and the open items above (α current loops). The patch fixture
fixes a step count (the PML end-criterion energies then differed; they
agree since wave 13, `pml_end`).

#### 5.16.5 openEMS versions: e89286f → be30220

The fixtures were made with openEMS-Project e89286f (2026-10-18; openEMS
v0.37.0-19-gc3a956b, CSXCAD v0.7.0-1) until wave 14, and since then with
be30220 (2026-10-21; openEMS v0.37.0-32-gec4d2db, CSXCAD v0.7.0-8; commit
titles "dependencies: no CGAL, VTK only for AppCSXCAD, zlib", "update
submodules", "Windows package: both GPU plugins"). Both were built side by
side and run on every fixture model, the 169 feature-sweep models of the
scratch generator and the 230 models of the randomized sweep (seeds
0…229), all single-threaded `--engine=basic`; the outputs were compared
file by file (black box only: `--help`, console, files).

| | e89286f | be30220 |
|---|---|---|
| probe files, HDF5 dumps, `et`/`ht`, nf2ff, `sar_calc` | | bit-identical (all fixtures but 8 sweep models with metal or transformed primitives, below; all 230 random models; headers differ in the version string only) |
| time steps, end criteria, PML, Mur, lumped elements, dispersive media, cylindrical grids | | identical |
| VTK XML (`.vtr`, `.vts`, `.vtp`) | VTK library writer: `RangeMin`/`RangeMax` on every array, an `L2_NORM_RANGE` information key on vectors, coordinate arrays named after their address (`Array 0x5555…`); `PEC_dump.vtp` in the appended layout (`offset` attributes, padded `Piece` attributes, empty `Verts`/`Strips`/`Polys`) | own writer: none of these attributes, unnamed coordinate arrays, `PEC_dump.vtp` inline (points and `Lines` only); the encoded payloads (zlib blocks, base64) are byte-identical in all 43 VTK fixtures |
| legacy `.vtk` box dumps | | identical |
| `--help` | | lists `--gpu-device arg (=-1)`; the banner lists zlib instead of VTK |
| command-line errors (unknown, ambiguous option, invalid or missing value, value for a flag) | uncaught exception, `SIGABRT`, status 134 | `openEMS: Error: …` on stderr, status 1 |
| engines | | `gpu` needs a GPU plugin (status 2 without one), `gpu-reference` runs on the CPU and gives the `basic` engine's probe files bit for bit |
| inside tests | exact comparisons; Sphere, SphericalShell, Wire surfaces open; polygon vertices half-open; polyhedron faces inside or outside depending on the primitive's position (§5.16.3) | closed for every primitive, and a point within `t = 1e-3·h_min` of a primitive is inside |

The tolerance was measured by bisection of the face position of metal
primitives (`--debug-PEC`), material primitives (`--debug-material`),
excitations and conducting sheets (`--debug-operator`): `t` is
`1e-3 ×` the smallest cell width of the whole mesh over all three axes,
in drawing units (1 mm cells: 0.000995…0.001002; 0.5 mm cells anywhere
in any axis, even far from the primitive: 5e-4; independent of
`DeltaUnit`, the coordinate offset, the box size; on cylindrical grids
the α widths count in radians, and an α face of a box is `t/ρ` away), a
Euclidean distance for spheres, cylinders, shells and wires in every
direction tried (axis, diagonal, 3-4-5), per face for boxes (also at
box edges and corners: Chebyshev), polygon and extruded-polygon sides
and ends, polyhedron faces; for a `Scale3` it is `t/σ_min` in the
primitive's own coordinates (`4,1,1` and `4,2,1`: `t`; `4,2,2`, `2,2,2`:
`t/2`). Lumped elements, probes and dumps snap to the mesh and do not
change.

Which is more accurate. Everything numeric is bit-identical, so the
comparison is about geometry only:

- **Polyhedra.** e89286f's faces depended on where the polyhedron is: a
  metal cube `[a, b]³` with faces on mesh lines got 659, 669, 360 and 28
  of its closed 882, 882, 882 and 54 edges for `[3, 9]³`, `[−3, 3]³`,
  `[−9, −3]³`, `[−7, −5]³`, with different counts on opposite faces
  (`x+` 64 vs `x−` 13 edges for `[3, 9]³`); be30220 gives the closed,
  mirror-symmetric set (84 edges on each face) at every position — the
  discretisation of a symmetric object is symmetric again, and equals
  csx's (ems's native) closed test. Sweep: `prim_polyhedron_metal`,
  `prim_polyhedron_tet`, `prim_stl`, `prim_ply`, `tie_box_metal_cube`
  (2e-2…0.34 between the versions) now agree with ems to 5e-5…2.6e-4
  (were 5e-3…0.31).
- **Round-off at ties.** The tolerance makes ties robust to the
  arithmetic of transformations: the rotated cylinder whose sample points
  lie within 1e-13 of its radius (`tr_cylinder`, 4.6 % between ems and
  e89286f) agrees to 4.7e-5. A geometry shift of at most 1e-3 of the
  smallest cell is far below the discretisation error of any FDTD model.
- **Curved metal surfaces and polygon vertices.** Open spheres and wires
  and the orientation-dependent vertex rule (a clockwise rectangle open
  at all four corners, a counter-clockwise polygon at its lowest
  vertices) made the result depend on the primitive type and the vertex
  order; be30220 treats every primitive alike (closed), as csx does.
  ems's emulation of the old rule (`sweep_prim_sphshell_metal`,
  `prim_wire_thick`, `tie_sphere_metal`, `tie_wire_z`,
  `tie_linpoly_metal`, `tie_polygon_vertex`) was dropped: the PEC edge
  sets of all these models are now identical to be30220's.
- **Nothing is less accurate** in be30220 on any model tried; the
  random-sweep models, the e2e, PML, cylindrical, SAR, NF2FF and core
  fixtures do not change at all.

Decision: the fixtures follow be30220 (`REVISION_PIN`). ems's `ems`
binary and openEMS writers produce the new layouts (the old VTK
attributes are not kept: every VTK reader reads both, and the old
layout carried no data), accept and report `--gpu-device` (ems has no
GPU engine; `gpu` and `gpu-reference` run the CPU engine), exit with
status 1 on command-line errors, and on the file path grow the volume
primitives of metal, materials, excitations and conducting sheets by
`t` (`compile::openems::dilated`; boxes, multi-boxes, spheres,
cylinders, their shells, wires of at least half the smallest cell and
extruded polygons; not polyhedra, rotational polygons, thin wires,
sheets and lines, whose exact ties are closed in both codes, §8.1 item
17). The open-surface and polygon-vertex rules of e89286f are gone.
Feature sweep with be30220 (169 models): every discretisation that
differed between the versions is identical (`--debug-PEC`,
`--debug-material`); `tr_sphere` 4.7e-2 → 4.0e-5 and
`tr_rotate_origin` 4.3e-3 → 2.9e-5 with the tolerance; unchanged
differences: `tie_cyl_metal` 1.1e-2, `tie_cylshell_metal` 8.6e-3 (the
same in both versions, identical PEC and material dumps),
`prim_sphshell_metal` 1.5e-3 (weak field inside the shell). Fixture
size: 8.71 → 8.68 MB (VTK files 146 → 114 kB). (Wave 15: these three
were `f32` round-off, closed by openEMS's summation order, §5.16.6.)

#### 5.16.6 Round-off: summation order of the curl

`tie_cyl_metal` (1.1e-2) and `tie_cylshell_metal` (8.6e-3) had identical
`--debug-PEC`, `--debug-material` and `--debug-operator` dumps (every
interior coefficient bit for bit) and full-domain field dumps that agreed
to 1e-6 of their norm; the probes that differed (`e1`, `u1`, the H probe
in the hollow of the shell) lie in the shadow of the metal, where the
field is ~1e-4 of the source region's (largely the static charge of the
Gaussian's DC content). openEMS's own result there is round-off: the
same model mirrored in x gives `e1` 6.3e-3 and `u1` 3.4e-3 away from the
unmirrored one in openEMS, the size of the ems difference, and openEMS's
fields of the z-symmetric model are not z-symmetric (components that
vanish by symmetry reach ~1e-7 of the field near the source after 10
steps), while ems's are mirror-exact. The difference was the `f32`
rounding of the update, which accumulates over the ring-down of the
closed box and dominates where the signal is weak.

The rule was fitted from per-step `Et`/`Ht` dumps (`OverSampling="100"`)
of a vacuum box with 1 m cells (`DeltaUnit 1`: the dumped E and H are the
engine's voltages and currents exactly) and `--debug-operator` (the
`f32` coefficients): of 14 candidate forms for E and 48 for H, one
each reproduces every one of 47 160 E and 34 875 H updates bit for bit
(the second best 91 % and 80 %):

- E: `u' = vv·u + vi·(((a − b) − c) + d)`,
- H: `i' = ii·i − iv·(((a − b) + d) − c)`,

with ems's circulation `(a − b) − (c − d)` (for `E_x`: `a, b` the `H_z`
at `j`, `j − 1`, `c, d` the `H_y` at `k`, `k − 1`; for `H_x`: `a, b` the
`E_z` at `j + 1`, `j`, `c, d` the `E_y` at `k + 1`, `k`; cyclic), no
fused multiply-add, and the products and sums of a soft source after the
update as ems does. The second difference was ems's subnormal flush:
`(x + 2⁻¹⁰²) − 2⁻¹⁰²` also rounded the last bits of normal values below
~2⁻⁷⁷ (the leading tail of a wave front); openEMS keeps them. The flush
is now a select (`|x| < 2⁻¹²⁶ → 0`, every normal value unchanged), for
the native path too (strictly more accurate; `bench_vacuum` unchanged
within the noise of a loaded machine, 1 and 8 threads). With both, the
1 m vacuum box (40 steps) and `tie_cyl_metal` scaled to 1 m cells (200
steps) give openEMS's dumps bit for bit at every step, including the
last H dump (the
look-ahead H of `engine::dump` uses the same order).

ems keeps its mirror-symmetric order natively (`CurlOrder::Paired`) and
uses openEMS's (`CurlOrder::OpenEms`, `Engine::set_curl_order`,
`Compiled::curl_order`) for openEMS files on Cartesian grids. Only the
main kernel follows it; CPML/UPML, conformal and multigrid keep their
expressions. (Wave 16: cylindrical grids follow too, §5.16.11; the
order had seemed to make `dump_cyl` worse, 9.7e-6 → 1.5e-5, because the
seam, the axis and the subnormal flush were still ems's.) Thread counts
stay bit-identical (both orders are per-edge expressions).

Feature sweep (169 models, openEMS be30220, before → after): 9 → 83
models agree to the printed digits (≤ 1e-8), 136 → 73 to ≤ 1e-4, 24 → 13
above (2 of them rejected by openEMS); `tie_cyl_metal` 1.1e-2 → 0,
`tie_cylshell_metal` 8.6e-3 → 0, `prim_sphshell_metal` 1.5e-3 → 0,
`tie_wire_z` 6.5e-4 → 0, `prim_box_metal` 2.4e-4 → 9e-14, the vacuum
and metal models 6.5e-5 → 9e-10, `probe_i_dirs` 1.0 → 8.4e-3 (an
all-round-off current column). The models with dielectrics stay at
2e-5…1.5e-4 (`prim_box_mat` 8.4e-5 → 1.0e-4, the same noise in another
pattern): there openEMS's `vi` differs by one `f32` ulp on the edges of
the material boundary (72 edges of `prim_box_mat`, e.g. 96.806496 vs
96.806488 at the box corners, ε averaged 1.75), a value the correctly
rounded `f64` formula cannot give (it lies 4e-9 below the rounding
midpoint), so openEMS computes some of its operator in `f32` or in
another way — not fitted yet; the same happens for some time steps on
1 m cells (`dt` 1.3 ns, 1.0 ns: all `vi`; 1.5 ns: all `iv`). Fixtures:
`sweep_tie_cyl_metal`, `sweep_tie_cylshell_metal` (exact to the printed
digits, tolerance 1e-6; 40 kB).

#### 5.16.7 Round-off: coefficient rounding

The one-ulp `vi`/`iv` differences of §5.16.6 come from openEMS keeping
the circuit quantities of an edge in `f32`. Fitted on `--debug-operator`
dumps of the black box (hand-written vacuum boxes at 1 m and 1 mm cells
with fixed `TimeStep` 1.0…1.9 ns resp. ps, the feature sweep's 169
models and the 64 sweep fixtures; with a time step given explicitly so
that it is the same in both codes): openEMS forms

```text
C = ε0·ε_r·A/l,  G = κ·A/l,  L = μ0·μ_r·A'/l',  R = σ_m·A'/l'   (f64, rounded to f32)
vv = (C − G·dt/2)/(C + G·dt/2),  vi = dt/(C + G·dt/2)            (f64 from the f32 values, rounded to f32)
ii, iv likewise from L, R
```

i.e. `vi = f32(dt / f32(C))` for a lossless edge. The correctly rounded
`f64` quotient `f32(dt/C)` (ems's native) is one `f32` step off where
`dt/C` lies near a rounding midpoint: for the vacuum boxes in 16 of the
40 coefficients (`vi` at 1.0, 1.1, 1.3, 1.5, 1.7, 1.9 ns on 1 m cells
and 1.0, 1.1 ps on 1 mm cells, `iv` at 1.1, 1.5, 1.6, 1.8, 1.9 ns and
1.1, 1.6, 1.9 ps), the `f32` rule matches all 40; on material-boundary edges (`ε_r` 1.75: f64
96.80649, openEMS 96.806496) and lossy ones (`mat_sigma_graded`, a 50 Ω
lumped `G = f32(0.02)`, `vv` −0.25764078 vs −0.2576408) likewise. Other
candidates (all-`f32` arithmetic, `f32` `dt`, `f32` `ε_r`, other
constants, other operation orders) each fail some of these.

ems: `CoeffRounding::OpenEms` (`Coefficients::from_edges_rounded`,
`EngineBuilder::coefficient_rounding`, `Compiled::coefficient_rounding`)
for openEMS files, on Cartesian and cylindrical grids;
`CoeffRounding::Exact` stays the native default (one rounding, the
closest `f32`). Operator dumps of the 64 sweep fixtures and 169 sweep
models are now identical to openEMS's in `vv`, `vi`, `ii`, `iv` except
for: the PML layer (dumped differently), conducting sheets (their own
`C`, `G` from `compile/openems_sheet`; `sheet_graded`, `sheet_triangle_r`,
`sheet_thin_resistive`: 1–3 ulp), the 4 edges of a lumped resistor
(`lumped_caps`, `vv`: ems folds the element into the `f32` coefficients,
§5.16.1), 4 of 21660 edges of `mat_weight_coords` (a weight function of
`sin`, `cos`: `ε_r` itself differs in its last `f64` bits), and the
models whose time step openEMS computes (`fdtd_odd`, `fdtd_tsm3_graded`:
openEMS's Rennings `dt` is 3.5e-9 resp. 4.9e-9 relative below ems's;
with ems's `dt` given as `TimeStep` the graded mesh's dumps are
identical — the time step formula, not the coefficients; rounding the
inverse `C`, `L` to `f32` inside the criterion gives −1.2e-8 and
−9e-9, not fitted; both now identical, §5.16.10: openEMS's vacuum
constants and its time-step arithmetic). On cylindrical grids the rule halves the mismatched
edges (`adv_cyl_drude` 2900 → 0, `adv_cyl_multigrid` 2816 → 360,
`adv_cyl_coax` 4820 → 2394; the rest is cylindrical geometry) and
changes the probe agreement of the 31 cylindrical fixtures within its
noise (largest change 1.9e-4 → 1.8e-4).

Feature sweep (169 models, openEMS be30220, `compare.py` of the sweep
tools, before → after): 78 → 123 exact (worst probe difference 0),
13 → 5 above 1e-4 (`param_term` and `prim_userdefined`, rejected by
openEMS; `probe_i_dirs` 8.4e-3, an all-round-off current column;
`sheet_thin_resistive` 1.5e-4; `fdtd_odd` 1.3e-4, the time step); no
model worse beyond noise. `prim_box_mat`, `prim_box_reversed`,
`grid_unsorted`, `param_linear`, `prim_polyhedron_cube` 1.0e-4 → 0,
`mat_aniso_weight1` 4.1e-5 → 0, `tie_wire_mat` 6.5e-5 → 0. Remaining
~1e-5: sheets, lumped elements (8.6e-5), dispersive media (2e-5…6e-5),
Mur and PML faces, computed time steps. Sweep fixtures
(`tests/openems_sweep.rs`): 47 of 64 exact; their tolerances are 1e-6
(were 1e-4…3e-3; `prim_sphshell_metal` 3e-3, `tie_wire_z` and
`probe_fd` 1e-3, `mat_sigma_graded`, `grid_deltaunit_m` 3e-4).

#### 5.16.8 Round-off: PML layers and Mur faces

Wave 16 fitted openEMS's arithmetic of its `PML_n` layer (§5.16.2) and
of the first-order Mur faces and `AbsorbingBC` sheets, as §5.16.6 did
for the curl, from per-step `Et`/`Ht` dumps (`OverSampling` ≥ 100, raw
HDF5, `DumpMode 0`) of hand-written models with 1 m cells
(`DeltaUnit 1`: the dumped fields are the engine's `u` and `i`), a fixed
`TimeStep` and `--debug-operator` / `--debug-material`.

**Mur faces.** Of six orders of the 1-D update (§5.6) one reproduces
every face edge of a box with Mur on two faces (all 9357 updates; the
domain edges follow §5.16.3):

```text
u_b^{n+1} = (u_1^n − k·u_b^n) + k·u_1^{n+1}       (f32, k = f32((v·dt − Δ)/(v·dt + Δ)))
```

— the part before the neighbour's new value is what openEMS's domain
edge keeps alone (§5.16.3, `u_1^n − k·u_c^n`). ems's own order is
`u_1^n + k·(u_1^{n+1} − u_b^n)`. With it, an all-Mur box with graded
boundary cells and media reaching the faces (local phase velocity, 80
steps) and a PMC/PEC/Mur mix are identical to openEMS's at every step.
`MurBoundary::openems_order` (files).

**AbsorbingBC sheets** use the same order, but `k` is formed with
`v·dt` rounded to `f32`: `k = f32((f32(v·dt) − Δ)/(f32(v·dt) + Δ))`.
A 1 mm sheet needed `k` one `f32` step away from the correctly rounded
value; of 13 formulas (an `f32` `Δ`, `c0`, `dt`, other groupings) only
this one gives the needed value for 1 m, 1 mm and 0.7 mm cells, and the
sheets then match bit for bit (`AbsorbingSheet::openems_order`, files).
Two semantic differences found on the way are *not* emulated (ems keeps
its documented behaviour, rule of §5.16): with `NormalSignPositive="0"`
openEMS takes `Δ` from the cell on the far side of the sheet (between
the sheet and the next line up, while the neighbour it reads is the line
below; graded meshes only), and `PhaseVelocity="0"` runs the vacuum `c0`
also inside a dielectric (ε_r = 2.2: openEMS `k = −0.29902`, the vacuum
value; ems the local `−0.46647`).

**PML.** openEMS runs every edge of a layer (any `a_w ≠ 0`) in two
stages with its own flux `F`; the interface plane keeps the ordinary
update. The flux stage is the main engine with the coefficients its
`--debug-operator` dump shows there:

```text
F' = vv·F + vi·∮i,   vv = f32((1 − a_p)/(1 + a_p)),   vi = f32(dt/((A/l)·(1 + a_p)))
```

(`A/l` the edge's geometry factor as in the coefficients, no `ε`: every
coefficient of all 8489 layer edges of a three-face model with graded
cells and an `ε_r`/`μ_r`/κ block, bit for bit), the curl in openEMS's
order (§5.16.6). The second stage fits

```text
u' = (k·u − b₋·F) + b₊·F'        (f32, in this order)
k = f32((1 − a_q)/(1 + a_q)),  b± = f32((1 ± a_a)/((1 + a_q)·ε))
```

and the same for H with `μ` (the only one of five orders tried that
matches every update). `F` is in units of `ε·u`
(`D·l`). Fitted constants: `ε = ε_r·ε0'` with `ε0' = 1/(4π·10⁻⁷·c0²)` =
8.854187817620389e-12, `μ = μ_r·μ0'` with `μ0' = 1.256637062e-6` (ten
digits): with ems's CODATA 2022 constants about 0.5 % of the coefficients
(those within 0.003 `f32` ulp of a rounding midpoint) were one step off.
The permittivity was pinned to ±2e-11 relative and the permeability to
`4π·10⁻⁷·(1 + [3, 5.1]·10⁻¹⁰)` by intersecting, over the edges of three
models, the constants each edge's exact solution set allows; other
candidates (CODATA 2018/2022 `μ0`, `4π·10⁻⁷`, `f32` media, `f32` `a`,
other groupings) leave whole models off. The grading `a_w` uses ems's
constants (only `Z0·ε0 = 1/c0` enters it). Validated on 8 further random
models (two to five faces, PMC/PEC mixes, `ε_r` 1.3–6, `μ_r` 1.1–2.8
with averaged interfaces in the layers, `dt` 0.85–1.15 ns): every E and
H value at every step identical.

ems (`OpenEmsPml`, files): every layer edge keeps `F` and its start value
(the single-stage collapse of the tangential components and the special
treatment of driven edges are gone: openEMS's source adds to the field
in any case); the kernel writes `vi·∮i` (`ca = 0`; without fused
multiply-add `0·u + vi·∮i` is exact and `vv·F + t` has openEMS's bits),
the slab hook forms `F'` and `u'` in the order above.
`OpenEmsPml::with_materials` takes `ε_r`, `μ_r` from the edge materials
(`new` derives them from the coefficients, last bits off). Public API:
`OpenEmsPml::with_driven` removed. Cost: every layer edge streams two
`f32` of state and its coefficient row — 100³ with `PML_8` on all faces
(40 % of the cells in layers) 170–190 vs 230–250 MCells/s on 1 thread,
1.0–1.7 vs 1.7–2.1 GCells/s on 8 and 1.4–1.7 vs 2.1 GCells/s on 16
(a loaded shared machine; PEC 480–530 MCells/s on 1 thread).

Agreement. Feature sweep (169 models, fresh be30220 runs, before →
after): 123 → 136 exact, no model worse; `pml_open` 4.5e-6 → 0,
`pml_big_3/4/8/plain` 1.2e-5…1.9e-5 → 0, the Mur models
`abc_line_ref_mur`, `bc_mur_pv`, `bc_numbers`, `bc_names_lower`,
`bc_spellings`, `fdtd_endcrit` 7e-7…3e-5 → 0, `abc_line_s0_t0/t1` 1.1e-6
→ 0, `bc_mixed` 4.0e-6 → 1.8e-13 (one printed digit of an interpolated
E probe). New fixture `pml_exact` (three overlapping layers, one graded,
a lossy dielectric-magnetic block in all of them, fixed time step, 96 kB):
every probe exact. The other `pml_*` fixtures stay at 4e-7…1e-5: their
time step is openEMS's computed one, which differs from ems's in the
last bits (§5.16.7, area of the time-step criterion); with openEMS's
printed `dt` given as `TimeStep` (dumps of the first 120 steps)
`pml_face`, `pml_corner`, `pml_medium` and `pml_end` give identical
probes, `pml_wg_iris` 6e-9 (a difference at its source plane that also
appears with PEC ends), `pml_lumped` 1.3e-7 on the port current (lumped
elements, §5.16.7). Not reproduced: cylindrical grids (`pml_cyl`
unchanged at 1e-6…9e-6: openEMS's cylindrical curl order, §5.16.6).
Found on the way, not PML: `E` dumps of edges of length 0.8 m (lines
0.9, 1.7) differ in the last bit in some steps without any PML, the
fields themselves are identical (the dump's division, not fitted), and a
boundary `H` dual edge of a graded corner cell has `iv` one `f32` step
off (`--debug-operator`, 5 edges of a mesh with a 0.4 m first cell).

#### 5.16.9 Round-off: vacuum constants, lumped elements, current probes

Wave 16 (area J), black box only; the fits below use hand-written
models, `--debug-operator` dumps and per-step `DumpMode="0"` dumps
(`OverSampling` on the dump box, see the end of this section).

**Vacuum constants.** The `mat_weight_coords` edges of §5.16.7 were not
the weight function: csx's `ε_r` there is openEMS's (5.585000000000001
for the worst edge, the `f32` material dumps agree), and the `f64`
capacitance `ε0·ε_r·A/l` lies near an `f32` rounding midpoint, where
ems's `ε0` (CODATA 2022, `1/(μ0·c0²)` = 8.8541878188e-12) and
openEMS's round apart. On 24 random graded meshes with dielectric and
magnetic boxes (`ε_r` 1, 2.5; `μ_r` 1.5…7.1; weights) ems had 4…118
`vi` and 2…260 `iv` mismatches per model, all one `f32` step. Every
vacuum edge constrains the constant (`vi = f32(dt/f32(ε0·A/l))`): the
99 438 E edges need `ε0 ∈ [8.8541878174e-12, 8.8541878178e-12]`, the
87 300 H edges `μ0 ∈ [1.25663706194e-6, 1.25663706205e-6]`. openEMS's
operator uses `ε0 = 8.85418781762e-12` (CODATA 2006) and
`μ0 = 1.256637062e-6` (ten digits; not `4π·10⁻⁷`, not CODATA 2018's
1.25663706212e-6, not ems's 1.25663706127e-6); 8.8541878128e-12 and
8.854187817e-12 each fail tens of edges. With them
(`OPENEMS_EPS0`, `OPENEMS_MU0`, used by `CoeffRounding::OpenEms` for
`C` and `L`) the 24 models, 160 lumped-element models, 12 weight-function
variants and the feature sweep's 1 mm Cartesian models have every
`vv`, `vi`, `ii`, `iv` of openEMS (apart from the items below and the
PML layer). The native constants are unchanged (CODATA 2022, more
accurate). `c0 = 1/√(ε0·μ0)` of openEMS's constants is 2.2e-10 below
299 792 458 m/s: not the 3.5e-9 of openEMS's computed time step
(§5.16.7). Other uses of ε0/μ0 on the file path (time step, Mur, PML,
dispersive poles, sheet branches) keep ems's constants.

**Lumped elements.** Measured on 160 random models (R, C, L 2…10 nH,
R‖C, R‖L, series R, L, C, RL, RLC; `ε_r` 1…4.4, `κ` 0…1.7 S/m; 1–3
columns × 1–3 series edges; uniform and graded meshes; `dt` 1.0…1.9
ps): openEMS builds the element edge like a material edge, each circuit
quantity rounded to `f32` (§5.16.7):

- parallel: `C = f32(C_e)` if the element has a capacitor, else the
  medium's `f32(ε0·ε_r·A/l)`; `G = f32(1/R_e)` if it has a resistor,
  else the medium's `f32(κ·A/l)`; an R/L element without capacitor gets
  `C = f32(C_max)` where the medium exceeds the limit of §5.16.1;
- series: the medium's `C`, limited to `f32(C_max)` (§5.16.1), and
  `G = 0` — the medium's conductance is dropped (`vv = 1` on lossy
  media; ems kept it before);
- then `vv`, `vi` in `f64` from these values.

ems formed the same values by recovering `C`, `G` from the `f32`
coefficients twice (`replace_medium`, then `fold_parallel`), one `f32`
step off in `vv` on the 4 edges of `lumped_caps` and on 1–9 edges of a
third of the random models. It now folds once from the exact medium
values (`e_edge_circuit` from the edge materials, used where they give
the edge's coefficients exactly, else recovered) and installs the
element with `medium_folded`. All element edges of the 160 models are
openEMS's except series resistors without L or C: openEMS keeps
`vv = 1` there and updates the resistor separately; ems folds it as the
edge conductance. Per-step dumps (1 m cells) give the same physics,
`j = (u^n + u^{n+1})/(2R)`, but none of ~30 `f32`/`f64` operation orders
of the corrected update reproduces openEMS's bits (≤ 28 % of the
steps), nor does any one-step form for L, C, RL, RLC branches (they
carry hidden branch state): series elements stay at the round-off level
(`lumped_series_rlc` 8.8e-5; series R 9.3e-5, on a lossy medium
1.1e-4, series LC on a lossy medium 8.6e-5).

**Current probes.** openEMS sums the loop currents in `f32`, rounding
after every term, in a fixed order: for a loop with normal `n` and
`(n, b, c)` cyclic, the `b` components on the side below the box along
`c`, then the `c` components on the side above it along `b`, the `b`
components above along `c`, the `c` components below along `b`, each
side with ascending node index. Fitted on `probe_i_dirs` scaled to 1 m
cells (the noInterpolation H dump is the engine current there) over 24
side orders × 16 directions × `f32`/`f64` accumulation: only this one
reproduces all 61 samples of the round-off column `ix` and of `iz`; on
the 1 mm original the same sum of the engine's currents (taken from
ems, whose fields are openEMS's) reproduces all samples, while `H·Δl`
forms do not. The box direction does not matter (`izr` = `iz`). ems
uses it for openEMS files on Cartesian grids (`Probe::f32_sum`,
`map::openems_loop_order`; also for the sample after the last step);
native probes keep the `f64` sum.

**OverSampling of probes and dumps.** openEMS honours an `OverSampling`
attribute on `<ProbeBox>` and `<DumpBox>`: the time-domain interval is
`max(1, ⌊N/OS⌋)` with the property's value, replacing the `<FDTD>` one
(10 variants, `N` = 41: OS 10, 3, 2, 1, 50 give 4, 13, 20, 41, 1 steps;
1 is honoured, unlike `<FDTD OverSampling="1">`, which means 4; 0
crashes openEMS). csx keeps the attribute among the property's
unrecognised ones (§8.1 item 22); ems reads it from there
(`ProbeSpec::oversampling`, `ProbeResult::oversampling`,
`Results::openems_probe_interval_of`) and warns for values that are not
positive integers. Fixture `sweep_prop_oversampling` (exact).

**Not reproduced.** Conducting sheets (`sheet_thin_resistive` 1.5e-4;
the sheet fixtures 2e-5…6e-5): openEMS's sheet edge capacitance is not
`f32(K·w/l)` for any constant `K` — with a negligible conductance
(`σ` = 1e-6 S/m, `vv = 1`) and 7200 interior edges of 16 random graded
meshes no `K` reproduces more than 69 % of the `vi` (also with `f32`
widths, lengths, `dt`, per-square values, half-cell sums or another
`μ0`); the best `c` is 1.4e-7 below the table's 0.022224575. Not
fitted.

Feature sweep (169 models, openEMS be30220, before → after): 123 → 130
exact, 5 → 4 above 1e-4 (`param_term`, `prim_userdefined` rejected by
openEMS, `sheet_thin_resistive`, `fdtd_odd`): `lumped_caps` 8.6e-5 → 0,
`probe_i_dirs` 8.4e-3 → 0, `probe_i_snap`, `probe_fd`,
`probe_polygon_i`, `probe_ui_shapes` 8.8e-8 → 0, `mat_weight_coords`
4.1e-5 → 0. Sweep fixtures: `probe_polygon_i` and `mat_weight_coords`
exact (tolerance 1e-6).

#### 5.16.10 Round-off: the computed time step and openEMS's constants

openEMS's own time step (no `TimeStep` in the file) was 3.5e-9…4.9e-9
(relative) below ems's on `fdtd_odd` and `fdtd_tsm3_graded`, within
4e-8 on random meshes (§5.16.1), and the method-1 step one `f32` step off
on 15 % of them. Measured on the black box, never from its source: a probe
at the excitation of a tiny PEC box (`OverSampling` 100000: one row per
step) records `t = n·dt` with 12 digits for 3 000 steps; intersecting the
intervals `[t_n ± ½ unit]/n` of all rows pins `dt` to 1e-15 (a few `f64`
ulps), and `-vvv` prints the position of the limiting edge. Hand-made
meshes isolated the terms: uniform cubes of 0.1 mm…1 m (the error depends
on the mantissa of `Δ` only: single-precision rounding), one graded cell
(only the coupling of the limiting edge to one face changes), then random
graded meshes, with the edge's terms decomposed into their `f32` ulps.

- **Constants.** openEMS computes with `ε0 = 8.85418781762e-12` and
  `μ0 = 1.256637062e-6` (`OPENEMS_EPS0`, `OPENEMS_MU0`), not ems's CODATA
  2022 values (`ε0` 1.3e-10 lower relative, `μ0` 5.8e-10 higher; `μ0` is
  not `1/(ε0·c0²)`): the time steps of 720 random meshes fit only within
  ±1e-10 of these two (scanned), and so do the operator's coefficients:
  all 96 509 interior `vi` and 124 457 `iv` of 600 random graded meshes
  (`--debug-operator`, fixed `TimeStep`), where ems's constants rounded
  10 of 9 023 `vi` and 66 of 11 708 `iv` (60 meshes) one `f32` step
  differently. The last 9 `iv` that the 2018 CODATA `μ0` still missed
  fixed its value to 10 digits. `CoeffRounding::OpenEms` now forms `C`
  and `L` with them (Cartesian and cylindrical); the native path keeps
  CODATA 2022.
- **Rennings.** `C`, `L` are `f32` (with full dual lengths at the
  boundary, as before); each coupling `1/(C·L)` is computed in single
  precision, `X = f32(1/f32(C·L))` (uniform cubes: one rounding per
  coupling, 15 of 15 sizes, none with `f64` products; one graded cell:
  the `f32` product, 13 of 16, at most 10 for other placements); the two couplings of one face are summed in
  `f32`, `T_b = ¼(face below + face above)` in `f64` (sums in `f64`: 11 of
  31 one-graded-cell meshes; all four in `f32`: 2); the own-axis term is
  evaluated at both ends of the edge, from `f64` half sums and half
  differences of the couplings `X(C_b(m), L_c)`, and averaged (the
  cylindrical form of §5.16.1, equal in exact arithmetic); the face below
  for `(a, b)` against the cyclic order takes geometry *and* material one
  node lower along `c` (the cylindrical rule; exact arithmetic cancels
  the shift, the rounding does not: 117 → 133 of 136); `ω²` and
  `dt = 1/√ω²` in `f64`. With the constants (694 → 720 of 720) every one of
  736 random vacuum meshes, graded along one to three axes, and 15
  uniform cubes is openEMS's to the last bit; ems's `f64` formula was
  up to 4e-8 off.
- **Gershgorin** (method 1): the faces beyond the boundary count with
  full dual lengths (mirrored; the half-cell rule of §5.16.1 in exact
  arithmetic, 285 → 300 of 300), `Σ 1/L` is an `f32` sum of
  `f32(1/L)` in the order (c upper, c lower, b upper, b lower) of the
  edge's faces (24 orders tried), `ω² = Σ/C` and `dt = 1/√ω²` in `f32`.
- **Cylindrical grids** (`engine::timestep_cyl`): the same arithmetic on
  the cylindrical circuit values (`C = ε0·εr·(A/l)` with openEMS's
  constants). Bit for bit on 60 random sectors, 60 closed grids and 60
  grids with the axis, both methods (before: up to 1.3e-7 off), and 12
  multigrids. On a multigrid openEMS's inner grids take Rennings whatever
  `TimeStepMethod` says (`-vvv`: `CalcTimestep_Var3` for the inner,
  `Var1` for the outer grid); with method 1 and no axis ems had taken
  Gershgorin everywhere, 2–5 % too small on 3 of 12 meshes.

Through the `ems` binary (probe time columns compared row for row): 410
random Cartesian meshes (60 vacuum, 200 with ε, μ, κ and anisotropic
boxes on and off the grid, 150 with metal boxes), both methods,
and on the vacuum ones `TimeStepFactor` 0.61 and 0.73, a sinusoid (the
period adjustment) and an explicit `TimeStep` with a factor: every time
column identical. Unit
tests: `engine::timestep::tests::bit_exact_steps`,
`engine::timestep_cyl::tests::bit_exact_steps` (openEMS's intervals).

Feature sweep (169 models, fresh be30220 runs, before → after): 123 → 127
exact (`fdtd_odd` 1.3e-4, `fdtd_tsm3_graded` 4.5e-5,
`fdtd_timestepfactor` 3.3e-5: the time step; `mat_weight_coords` 4.1e-5:
the constants, not its weight function — its 4 edges were `ε0`'s); no
other model changed. Sweep fixtures `fdtd_odd`, `mat_weight_coords`
exact (tolerance 1e-6). The cylindrical fixtures run on openEMS's step
bit for bit; `tests/openems_cyl.rs` matches the last row, whose printed
time now lies an ulp above ems's, within 1e-6 `dt`.

Not reproduced: cylindrical meshes with material boxes (4 of 60 sectors
up to 1.5e-3, 3 of 60 closed grids up to 0.2 with Rennings): the edge
materials differ (`--debug-material`: z edges at a box face in a sector,
ε 1.46 vs 2.06; the seam of closed grids, §5.16.1), not the criterion.
Seen on the way: openEMS runs a `CoordSystem="1"` grid as Cartesian
without `CylinderCoords="1"` on `<FDTD>` (ems as cylindrical), and stops
a run with `endCriteria="0"` and a sinusoid after `NumberOfTimesteps`
where ems stops earlier.

#### 5.16.11 Round-off: cylindrical grids

Fitted on per-step `Et`/`Ht` dumps (`DumpMode` 0, `OverSampling` ≥ 100,
so every step is written) and `--debug-operator` of small hand-written
cylindrical models with 1 m cells and a fixed `TimeStep`, compared bit
for bit (field dumps inverted to the engine's `u`, `i`; a dump entry
counts as a state difference only if no `f32` value explains both
codes' dumps). Tooling and models were scratch; the rules below are on
the file path only (`Fdtd::openems_run`), ems's native cylindrical
scheme is unchanged.

- **Curl order.** openEMS's cylindrical engine sums the circulation in
  its Cartesian order (§5.16.6): on a sector (6 × 6 × 6 cells, three
  source components) every H update and every E update but the source
  edges match `((a − b) − c) + d` / `((a − b) + d) − c` (the second best
  forms 82–92 %); with it the sector's dumps are identical at every one
  of 40 steps. `CurlOrder::OpenEms` now applies to cylindrical files.
- **Flush to zero of every operation.** openEMS runs with the CPU's
  flush-to-zero mode, which flushes each operation's subnormal result,
  not only the final one: with mm cells the leading tail of a wave front
  reaches the subnormal range (`i ~ 1e-38`) after 10 steps, and only the
  per-operation flush (`kernel::oe_e`, `oe_h`: `a − b`, the sums, both
  products, the result) reproduces openEMS's zeros there (a single final
  flush, ems's native one, differs from step 10, no flush at all from
  step 8). The same holds for the PML layer update (§5.16.8:
  `upml::layer_step`, `Run::apply`; `pml_cyl` needed it, the Cartesian
  sweep is unchanged). Both on the openEMS path only.
- **The seam** (`engine::openems_cyl::OpenemsSeam`). openEMS updates
  each seam E edge (ρ and z edges on `α = 0`) once, with the
  coefficients of its whole dual face and the neighbours across the seam
  (`H_z`, `H_ρ` of the last α cell, the whole `H_α` dual edge), and each
  seam `H_α` dual edge once with its whole dual length; ems's copies with
  half faces and the `1/cb`-weighted periodic merge are exact but round
  differently. The operator recomputes these edges after the kernels
  (`H_α` post-H, both copies holding half the current, the whole value
  kept so that a subnormal half loses nothing; E post-E, before the other
  operators, the periodic merge's result replaced). A closed 4 × 6 × 4
  grid with two asymmetric sources: identical dumps at every step of 40
  (before: from step 4). Edges with dispersive poles, conducting sheets
  or changed coefficients (lumped elements) keep the merge; edges inside
  openEMS's PML run the layer update of §5.16.8 on the whole edge.
- **The axis.** openEMS's axis z edge (§5.16.2: the ring sum over the
  `N + 1` α lines, the seam's `H_α` twice) adds the terms one by one,
  `u = vv·u; u += vi·i_α(α_0); … ; u += vi·i_α(α_{N−1}); u += vi·i_α(α_0)`
  (74 of 74 updates; summing the ring first: 35 of 74), with `vv`, `vi`
  of the disc `C = ε0·ε·πρ_½²/Δz` **not** rounded to `f32` first (unlike
  every other edge, §5.16.7: `vi` 96.84546 where the rounded `C` gives
  96.84545, 8 of 8 against 5 of 8). Two axis models (one with a lossy
  dielectric and a radial material face) are identical at every step.
- **Multigrid** (`OpenemsMultigrid`). openEMS's scheme (one zone
  boundary `ρ_I`, a closed grid with the axis) reproduced bit for bit by
  an `f32` model and then in ems: a coarse operator (every second α
  line, its own coefficients of the coarse cell geometry, rounded as
  §5.16.7; the axis by the ring rule over the coarse lines) updates the
  cells and lines inside `ρ_I`; the fine operator updates everything on
  and outside `ρ_I` and reads the fields inside from the fine arrays,
  into which openEMS writes the coarse values after each half step. On α
  lines (`E_ρ`, `E_z`, `H_α`): the coarse value on even lines, the mean
  of the neighbours on odd ones (`H_α` currents halved with the dual
  length); on α cells (`E_α`, `H_ρ`, `H_z`): linear interpolation between
  the centres of the coarse cell and its neighbour (3/4 and 1/4), `E_α`
  arc voltages halved, except that the first `E_α` cell extrapolates from
  the last coarse cell's unwrapped centre (13/12 and −1/12; `H_ρ` and
  `H_z` wrap). The coarse H on `ρ_{I−½}` reads `E_z` of the even fine
  lines on `ρ_I` and, for `E_α`, the sum of the coarse arc's two fine
  edges. Exact on the fit model for 38 steps but the last H dump (ems's
  look-ahead H does not see the operator) and on `adv_cyl_multigrid`
  (3.0e-3…2.0e-2 with ems's embedding). Used for files with one zone
  boundary, a closed grid with the axis, uniform α, vacuum inside the
  zone and no PML or Mur faces; other multigrid files keep ems's
  embedding (§5.15.1). `Compiled::openems_multigrid` (new field) holds
  the zone line; the grid is then the plain fine grid, the time step the
  one resolved on the zoned grid.
- **Materials of z edges** (`compile::cylavg`). A box face swept through
  the lower α quarter of a z edge's dual face: openEMS weighs the upper
  α half by the dual width of the edge's line and the lower half by the
  dual width of the line below (a whole cell at the first line), not by
  the halves' own widths (`(α_j − α_{j−2})/(α_{j+1} − α_{j−1})` measured
  to 6 digits on 15 configurations, also on the `ρ_min` and `ρ_max`
  lines). Random sectors with one or two material boxes: 58 of 60
  material dumps identical (11 of 30 before; the two left differ on PEC
  boundary edges, where openEMS samples the out-of-domain quarter).
  Closed grids still differ on some edges near α faces of boxes (not
  resolved; the dump's α index is shifted against the lines there, which
  made the cases hard to read).
- **Dispersive poles on cylindrical grids** were all missing for a
  primitive at negative x (the candidate edges came from its Cartesian
  bounding box), native path too; fixed (every α line a candidate,
  `tests/validation_dispersive.rs`). On the seam openEMS updates the two
  copies apart (the `α = 0` copy of a Drude-filled ring steps as the
  static medium at first, then gets half the pole's correction): not
  reproduced, `adv_cyl_drude` stays at 3.8e-3.
- **`CylinderCoords`.** Without `CylinderCoords="1"` on `<FDTD>` openEMS
  runs a `CoordSystem="1"` mesh as a Cartesian one (rectilinear debug
  dumps, the α lines as y lines in drawing units) and takes the boxes at
  their literal corners; files now do the same, with a warning
  (`io::openems::parse_model`; fixture `cyl_no_cylinder_coords`, a
  lossy box and a source, identical to the printed digits). Not
  checked: other primitives (a `Cylinder` there differs from both
  readings).

Dump formula (not changed): openEMS's raw dumps divide by the edge
length computed from the drawing-unit deltas times the unit
(`f32(u/(Δ·unit))`), ems multiplies by `1/Δ` of the metre lines; for
lengths ≤ 1 both give the same `f32`, otherwise an ulp differs now and
then (the remaining 1e-15…1e-7 of the cylindrical HDF5 dumps).

Cylindrical fixtures (24 with probes or dumps, openEMS be30220; "exact"
= every printed probe digit equal), wave 15 → now: 2 → 14 exact
(`adv_cyl_axis`, `adv_cyl_coax`, `adv_cyl_noaxis`, `adv_cyl_multigrid`,
`cyl_cellconst`, `cyl_radial`, `cyl_wrap_exc`, `cyl_sheet_none`,
`cyl_sheet_seam`, `pml_cyl`, the new `cyl_no_cylinder_coords`, as
before `cyl_planewave`, `sweep_prim_cs_cyl_box`; with I's time step,
§5.16.10, which they all need), `dump_cyl_closed` to 9e-16 of its norm
(was 3e-6), `dump_cyl` 9e-8, `cyl_sar` 1.7e-7. Left: conventions ems
keeps on purpose (`cyl_wrap_ring` 0.89, `adv_cyl_lumped_ring` 4.5e-2:
openEMS's lumped elements on the α = 0 line only; `cyl_current_seam`
0.11 and `e2e_coax_monopole`'s `port_it` 0.33: its `(N − 1)/N` ring
current, §5.16.2), `adv_cyl_drude` 3.8e-3 (above), the conducting
sheets (`cyl_sheet_fin` 2.4e-5, `cyl_sheet_tube` 2.8e-5, as on
Cartesian grids), `cyl_wrap_lumped` 1.6e-5 (its weak `uz_c`; the same
resistor on a sector is exact), `cyl_radial_port` 1.4e-6 and the other
`e2e_coax_monopole` probes 3e-7…8e-5 (lumped ports, Mur faces and
material boxes leaving the α range there). Tolerances of the exact
fixtures are now 1e-6 (`tests/openems_adv.rs`, `openems_cyl.rs`,
`openems_pml.rs`, `openems_dumps.rs`). Feature sweep (169 models):
146 exact, unchanged (its only cylindrical model was exact before).

## 6. Engine architecture

```
csx::ContinuousStructure ─┐
ems::Fdtd (settings) ─────┼─► compile ─► Engine { fields, coeffs, operators[] } ─► Results
                          ┘   (rasterise → C,G,L,R → ca,cb,da,db; map Excitation/
                               LumpedElement/ProbeBox/DumpBox → operators; dt)
```

- **Storage**: six `Vec<f32>` field arrays plus twelve `f32` coefficient
  arrays, all node-shaped.
  - Index is `(i·ny + j)·nz + k`; z is contiguous for autovectorisation.
  - Padding entries (e.g. `u_x` at `i = nx−1`) have zero coefficients and stay 0.
  - Memory: about 72 bytes/cell. Large grids are memory-bandwidth bound
    (~120 B moved per cell and step, at the ~70 GB/s DRAM limit of the
    development machine), so the kernels read the coefficients through a
    row-deduplicated view (`engine::rows`): an array whose z-rows are mostly
    bitwise repeats (`ca`, `da` = 1 in lossless media; `cb`, `db` on uniformly
    meshed regions) is read from a small cached pool of distinct rows. The
    values are the same, so results are bit-identical.
- **Staggering**: after n steps, u is at `n·dt` and i at `(n−½)·dt`. A step runs
  `pre_h → H kernel → post_h → pre_e → E kernel → post_e`.
- **Precision**: `f32` fields and coefficients. `f64` for geometry, time, DFT
  accumulators and energy.
- **Threads**: std only, no rayon (keeps the dependency footprint like csx's).
  - A persistent worker pool (long-lived threads, joined on drop) splits
    x-slabs. Each slab's components sit behind their own `RwLock`: the H update
    writes `i` and reads `u` of its slab and the next one; the E update does the
    reverse. Everything is safe Rust.
  - Each half-step costs one barrier. A serial operator gap (`post_h`/`pre_e`,
    or `post_e`/`pre_h`) adds one barrier at the steps where some registered
    operator needs a hook in it (`Operator::hooks() -> Hooks`, default all,
    refined per step by `Operator::hooks_at`: probes and openEMS-timed
    dumps at their sampling steps, soft sources while their signal is not
    zero), scheduled per run of steps. Spawning per step would be too slow
    for small models.
  - The energy end criterion evaluates the state on all threads
    (`Engine::energy_in`); it used to be a serial pass on the calling
    thread every check interval, which at 16 threads and a 26-step
    Nyquist interval cost as much as the steps themselves (×2–2.5 on the
    XML vacuum boxes of §10).
  - Subnormals are flushed in the kernel arithmetic (a select,
    `|x| < 2⁻¹²⁶ → 0`, normal values unchanged; it was `+ 2⁻¹⁰² − 2⁻¹⁰²`
    until wave 15, which also rounded small normal values, §5.16.6).
    Hardware flush-to-zero would need `unsafe`, and this way is still
    deterministic.
  - Thread count is configurable and defaults to one thread per physical
    core, at most one per 4096 cells and per five x-planes
    (`engine::default_threads_for`; measured optimum on the boxes and
    end-to-end models of §10).
- **Kernel**: `update_h()` then `update_e()`, written with slice iterators so
  LLVM vectorises the k-loop.
- **Operators** (`Operator` trait): CPML, openEMS's PML (files), Mur, sources, lumped elements,
  probes, dumps, dispersive media (ADE), oblique thin wires (§5.14.1,
  serial `pre_e`/`post_e`), and later NF2FF.
  - Dispersive media (`engine::dispersive::Dispersive`): each Drude /
    Lorentz / Debye pole is a series R–L–C branch per edge (polarisation
    current `j = J·A*`), integrated with the trapezoidal rule and coupled
    implicitly with the E update (semi-implicit, Okoniewski et al. 1997);
    closed-form per-edge solve for any number of poles, no extra stability
    limit, exact discrete permittivity `ε∞ + χ(Ω_t)` with
    `Ω_t = (2/dt)·tan(ω·dt/2)`, pole energy reported through
    `Operator::energy`. Magnetic poles use the dual update on H; their
    branch energy at `(n − ½)·dt` closes the engine invariant exactly,
    because the energy look-ahead uses the uncorrected H kernel result
    (derivation in the `engine::dispersive` docs). Runs as
    slab hooks (bit-identical for any thread count); registered after the
    CPML. The compile step samples the csx dispersive materials per edge
    over the same quarter / half cells as the static average.
  - Each has `pre/post_e` and `pre/post_h` serial hooks (calling thread,
    exclusive field access, registration order) for sparse work.
  - Bulk work (CPML) uses the optional `Operator::slab_hooks`: at
    registration the operator gets the x-slab partition and returns an
    `Arc<dyn SlabHooks>`, which each worker calls on its own slab right after
    its kernel pass in the same half-step (`post_h_slab` / `post_e_slab`), with
    the kernel's own lock set: write the updated field in the slab, read the
    other one plus the neighbour plane. No extra barrier. Slab work runs
    before all serial `post_*` hooks. State is kept per slab (CPML cuts each
    layer at the slab boundaries: x-face layers are whole x-planes, y/z-face
    layers are split along x), so results are bit-identical for any thread
    count. We chose this over fusing CPML into the kernel: it keeps the kernel
    simple and the Operator trait the single extension point, at the cost of
    a second pass over the layer cells only. Mur faces run the same way
    when every face edge's neighbours lie in its own slab (≥ 2 x-planes per
    slab), the grid merges no face copies and no earlier operator has
    serial E hooks: `u_b^n`, `u_1^n` are saved in `post_h_slab`, the Mur
    values written in `post_e_slab` (bit-identical to the serial update;
    ×4 on the Mur microstrip filter at 4 threads). The Mur operator is
    registered after the dispersive poles so the order of the slab
    corrections is unchanged.
  - Fused steps and movable hooks (wave 14, below): large grids update
    H and E x-plane by x-plane in one pass; read-only `post_h` / `pre_e`
    hooks (current probes, lumped elements) move out of the mid-step
    gap.
  - `Operator::energy()` (default 0) reports energy held in operator state
    (lumped L/C branches); `Engine::energy()` adds it, in registration order,
    so lossless LC-loaded cavities conserve the total.
  - Their order is fixed and documented.
- **Determinism**: no cross-thread floating-point reductions in updates.
  Energy and probe sums are combined in slab order, so output is bit-identical
  for any thread count.
- **`unsafe`**: `forbid` like csx in M1. If a benchmark shows that a
  bounds-check-free kernel is worth it, we relax this to `deny` with a single
  audited `#[allow]` module. That would be a reviewed decision, not a default.

### 6.1 Fused steps (wave 14)

An unfused step streams the fields twice: the H pass reads `u`, `i` and
writes `i`, the E pass reads them again and writes `u` (72 bytes per
cell, coefficients pooled). On grids that do not fit the caches that is
the limit (§10 benchmark: ~61 GB/s of the machine's ~70). A **fused
step** (`engine::pool`, `Shared::fused_a` / `fused_b`) does both in one
pass, 48 bytes per cell:

1. each slab updates the H of its *last* x-plane (it needs the next
   slab's first plane of `u^n`, before that slab overwrites it) and
   copies those currents to a per-slab halo plane;
2. once the previous slab has done step 1 (a wait on that neighbour's
   counter only, no barrier), the slab runs, plane by plane, the H update
   of plane `g` (from `u^n` of planes `g`, `g+1`, not yet updated) and
   then the E update of plane `g` (from the final `i^{n+½}` of planes
   `g`, `g−1`; plane `p0−1` from the halo). One barrier ends the step.

Every value is computed by the same expression from the same inputs as
in the unfused half-steps (the kernels are split per plane / row, not
rewritten), so results are bit-identical to master and for any thread
count. Verified by hashing the complete final field state of all 147
fixture models (openEMS file path at 1, 3 and 8 threads, native path at
2, 4 and 5 threads, fused steps forced on) against master `13f8608`, and by
`tests/fused.rs` (PML, CPML, probes, dispersive, lumped, Mur,
cylindrical, sheets: fused vs. the old schedule, bit for bit). The H
kernel of a fused plane goes row by row with all three components per
row (`h_rows`), which streamed better here than component by component;
interleaving H and E row by row inside a plane was slower (more
concurrent streams) and was dropped.

What a step needs to run fused:

- **No serial hook between H and E.** `Operator::movable_hooks` (default
  none) declares a `post_h` that only reads `i` or a `pre_e` that only
  reads `u`. At a step where every needed mid-step hook is movable, the
  engine calls the `pre_e`s after the `pre_h` hooks and the `post_h`s
  first in the `post_e` gap (both read values that the other half-step
  does not change), so the step has no `GAP_B`. Movable: current probes,
  probes and openEMS-timed dumps of H/J (`from_currents`), lumped
  elements and ports (voltage capture). This also removes one barrier
  and serial phase per step on unfused steps (the WR-90 iris with its
  current mode probes: ×1.1–2.5).
- **Slab hooks that allow it** (`SlabHooks::fusion`): `Fusion::Planes`
  hooks have plane-wise forms `post_h_planes` / `post_e_planes` (CPML
  without the cylindrical `ρ_max` layer, openEMS's PML, dispersive
  poles); `Fusion::Voltages` hooks act on `u` only and run whole-slab
  before the H update / after the slab's E update (Mur), which requires
  them to be registered after all plane-wise ones (they are). Anything
  else (`Fusion::None`, the default) keeps steps unfused.
- No energy recording in the step (the energy-check step of a run stays
  unfused, one in ~26 for XML runs), no multigrid.
- **Big enough slabs** (`FusedSteps::Auto`, `Engine::set_fused_steps`):
  only when a slab's fields exceed 4 MiB (`FUSE_MIN_BYTES`). Below that
  they stay in the core's cache share between the half-steps, fusing
  saves no DRAM traffic, and the fused schedule measured up to 20 %
  slower at 16–32 threads (100³ and 128³ boxes; not explained, the
  single-thread kernels are equally fast in cache, the extra halo copy
  is ~5–10 % of a 4-plane slab's traffic). The small end-to-end models
  of §5.16.4 therefore keep the unfused passes (forced fusion measured
  within noise there).

Measured on the Threadripper 9970X **under heavy load from other jobs**
(load average 37–75 on 64 hardware threads throughout; single numbers
vary by ±20–30 % and the 32–64-thread columns most), best of 3–5
interleaved runs, master `13f8608` → this wave, Mcells/s:

| `bench_vacuum` | 1 T | 8 T | 16 T | 32 T | 64 T |
|---|---|---|---|---|---|
| 128³ PEC | 323 → 489 | 3485 → 3555 | 5114 → 5702 ¹ | 5145 → 5172 ¹ | 2590 → 1886 ¹ |
| 200³ PEC | | 536 → 796 | 607 → 848 | 696 → 812 | |
| 256³ PEC | 323 → 353 | 560 → 726 | 518 → 707 | 690 → 712 | 559 → 584 |
| 128³ CPML 8 | 145 → 197 | 687 → 660 | 1410 → 1274 ¹ | 1864 → 2591 ¹ | 678 → 659 ¹ |
| 256³ CPML 8 | 131 → 185 | 366 → 592 | 373 → 635 | 505 → 561 | 517 → 563 |

¹ unfused (slab below 4 MiB): differences are load noise.

XML boxes on the openEMS file path (`N³` cells of 1 mm, soft dipole,
voltage probe, energy end criterion, openEMS PML_8 layers or PEC),
`Simulation::run` only:

| box | 1 T | 8 T | 16 T | 32 T |
|---|---|---|---|---|
| 200³ PEC | 313 → 409 | 771 → 1051 | 808 → 1010 | 918 → 909 |
| 200³ PML_8 | 200 → 312 | 435 → 703 | 416 → 648 | 444 → 588 |
| 100³ PML_8 ¹ | 168 → 254 | 2009 → 1881 | 3416 → 2734 | 2485 → 2563 |

(¹ fused only at 1 T.) The large boxes gain ×1.3–1.6 where they are
bandwidth bound; with the machine this loaded, the absolute rates stay
below the idle-machine figures of §10. The end-to-end models (patch,
microstrip filter, coax, Drude slab) are unchanged within noise except
the waveguide iris (movable mode probes).

**The 64-thread drop at 128³.** It is not slab granularity or barrier
latency: a 32³ box (no work to speak of) costs 0.07–0.08 ms per step at
32–64 threads, while 128³ goes from 0.42 ms per step at 32 threads to
1.1 ms at 48 and 1.65 ms at 64. With one thread per hardware thread,
(a) the two SMT siblings of a core share its execution units, L1/L2 and
bandwidth, so 64 threads do the work of 32 at half speed each, and (b)
every step needs *all* 64 hardware threads at once: any other runnable
task (other jobs here, daemons on an idle machine) preempts one of them
and the whole step waits for it at the barrier, for up to a time slice.
At 32 threads the scheduler has idle siblings to put the other tasks
on. Hence the default of one thread per physical core
(`default_threads`), which stays; explicit requests above the core count
are honoured but not recommended (`--numThreads` like openEMS).

## 7. Crate layout and dependencies

```
src/
  lib.rs          prelude, re-exports (incl. `pub use csx`)
  constants.rs    C0, MU0, EPS0, Z0
  error.rs
  fdtd.rs         Fdtd settings, Boundary, Cpml, EndCriteria
  waveform.rs
  mesh.rs         solver-aware mesh checks / suggestions on top of csx grid helpers
  compile/        rasterise materials & metal, map csx properties to operators, dt
  engine/         Fields, coefficients, kernels, thread pool, operator trait
    boundary/     pec, pmc, mur, cpml
    excite.rs     Excitation operators
    lumped.rs     LumpedElement operators
    probe.rs      ProbeBox operators, DFT accumulators
    dump.rs       DumpBox operators
  port/           LumpedPort (M1), MslPort / WaveguidePort (M2): emit csx props + analyze
  post/           dft, complex (in-tree), sparams, impedance; nf2ff in M3
  io/             vtr, csv; h5 (feature "hdf5"); openEMS <FDTD> section ↔ Fdtd (feature "xml")
tests/            analytic validation (§9)
benches/          Mcells/s (in-tree harness, no criterion)
examples/         rectangular cavity, dipole, microstrip line, patch antenna, run-xml
```

**Dependencies**
- Runtime: `csx` only by default.
- Features:
  - `xml` (default): enables `csx/xml` and the `<FDTD>` mapping.
  - `hdf5` (off by default): adds `hdf5rs` (pure Rust, MIT, no C library,
    no `unsafe`; KarpelesLab). `hdf5-deflate` also enables its gzip
    filter (via `compcol`).
- No `rayon`, `thiserror`, or `num-complex`; a small in-tree `Complex64` covers
  the need.
- Dev: none required; `approx`-style helpers live in-tree.

### 7.1 HDF5 output (feature `hdf5`)

`ems::io::h5` writes dumps (one file each) and probe series (one file, a
group per probe) in an **ems-native layout**, versioned by the root
attributes `format` (`"ems-dump"` / `"ems-probes"`) and `version` (1). The
module docs hold the full description; in short:

- `/mesh/{x,y,z}`: sample coordinates in metres.
- Time dumps: `/step`, `/time` and `/E` or `/H` as f32
  `[nt, nx, ny, nz, 3]`.
- Frequency dumps: `/frequency` and `/E` or `/H` as complex f64
  `[nf, nx, ny, nz, 3]`, stored as the compound `{r, i}` that h5py reads as
  `complex128`.
- Arrays are row-major in the engine's own order (z fastest), so nothing is
  reordered. Field datasets are chunked one snapshot or frequency per chunk,
  optionally deflate-compressed.

`DumpResult::file_type()` carries the DumpBox `FileType`; `run_xml` writes
`.h5` for HDF5 dumps when the feature is on, and `.vtr` with a warning when
it is off. Files written by the tests were checked by hand with h5py 3.16.

**openEMS layout.** `ems::io::openems::dump` writes the files openEMS
writes, structure for structure (layouts observed on a black-box openEMS
v0.37, fixtures in `tests/fixtures/openems/dump_*`, regenerated by
`scripts/openems-oracle/gen-dumps.sh`): HDF5 `/Mesh/{x,y,z}` (or
`rho/alpha/z`) plus `/FieldData/TD/%08d` f32 `[3, nx, ny, nz]` (`@time`,
`@d_order "NXYZ"`) or `/FieldData/FD/f%d` complex64 `{r, i}`, root
attributes `openEMS_HDF5_version`, `dump_type`, `dual_mesh`; the legacy
layout (`[3, nz, ny, nx]`, `f%d_real/_imag`, `legacy_fmt`); VTK XML
`RectilinearGrid` (`.vtr`, cylindrical: `StructuredGrid` `.vts`) with
zlib-compressed binary arrays, time steps `_%010d`, frequency files
`_f=<f>_{abs,arg,p=%03d}`; SAR files (local, averaged, raw). The
`Simulation` samples model dumps with the openEMS rules (box snapping,
`DumpMode` 0/1/2, sub-sampling, `OptResolution`, step timing E `n·dt` /
H `(n+½)·dt`, intervals `⌊N/OverSampling⌋` and `min(N, ⌊1/(2f·dt)⌋)`,
FD normalisation `2·Σ x·e^{−jωt}·K·dt`, start/stop window, the step after
the last one); dump types J, rot H, D and B are implemented. ems output
agrees with the fixtures to ≲ 1e-6 relative (float32), mass-averaged SAR
included (openEMS's `SIMPLE` cube solver, §5.12).
The module docs hold the full description.

### 7.2 The `ems` command-line program (openEMS drop-in)

`src/bin/ems/` (binary `ems`, feature `xml`; HDF5 dumps with `hdf5`)
replaces the `openEMS` executable: `ems <FDTD_XML_FILE> [<options>...]`
with the options of `openEMS --help` (v0.37.0-32, openEMS-Project
be30220, §5.16.5) and its syntaxes
(`--opt=value`, `--opt value`, unique prefixes such as `--no-sim`, `-v`,
`-vv`, `-vvv`, `-v2`, `--verbose[=N]`), writing openEMS's files to the
current directory. Everything below was observed on the black-box oracle
(fixtures `tests/fixtures/openems/cli_*`, `scripts/openems-oracle/gen-cli.sh`,
`tests/cli.rs`, which runs the built binary):

| option | openEMS | ems |
|---|---|---|
| (none) | probe files, dumps, `et`/`ht` | same (`io::openems::{probes, dump, signal}`) |
| `--disable-dumps` | no field dumps (SAR too); their boxes still reported; v0.37 then fails at the end with an internal error (status 1) | dumps not compiled, boxes still reported, status 0 |
| `--debug-material` | `material_dump.vtr`: `epsilon`, `mue`, `kappa`, `sigma` per edge; cylindrical grids `material_dump.vts` | same layout, ems's averaged edge materials (`io::openems::debug`; for files openEMS's `σ_m` rule and the medium under conducting sheets, as openEMS) |
| `--debug-operator` | `operator_dump.vtr`: `exc_volt`, `vv`, `vi`, `iv`, `ii`; cylindrical grids `operator_dump.vts` | same arrays, ems's coefficients (lumped elements and boundaries other than PEC are separate operators in ems) |
| `--debug-PEC` | `PEC_dump.vtp`: one line per PEC edge (inline binary points and `Lines`) | identical points and lines (also cylindrical) |
| `--debug-boxes` | `box_dump_<dump>.vtk`; current and waveguide probes all as `box_dump_.vtk` | identical files |
| `--debug-CSX` | `debugCSX.xml` (`<ContinuousStructure>`) | csx's XML writer |
| `--engine=…` | `fastest`, `basic`, `sse`, `sse-compressed`, `multithreaded`, `gpu` (status 2 without a GPU plugin), `gpu-reference` (CPU, `basic`'s results bit for bit); unknown names ignored | accepted; one engine: `basic`/`sse`/`sse-compressed` on one thread |
| `--gpu-device=N` | GPU device of `--engine=gpu` (default −1: the best); an integer, otherwise a command-line error | accepted and reported, no effect |
| `--numThreads=N` | engine threads | same (0: automatic) |
| `--no-simulation` | set-up only (`et`/`ht`, debug files), status 1 | same |
| `--dump-statistics` | `openEMS_run_stats.txt` (row per progress report), `openEMS_stats.txt` | same layout (`io::openems::stats`) |
| `--exact-endcriteria` | end criteria every Nyquist interval for the whole run | accepted: ems always does |
| `--showProbeDiscretization` | `<name>: snapped primary/dual coords: (…) -> (…)   [i,j,k] -> [i,j,k]` | same lines |
| `--nativeFieldDumps`, `--legacyHDF5Dumps` | dump variants | `DumpFileOptions` |
| `-v`…`-vvv` | more output | operator summary, probe sampling, properties |

- **Exit status:** 0 after a run; 1 for a model that cannot be read or set
  up and with `--no-simulation`; 255 without arguments; 0 for `--help`;
  1 for command-line errors (unknown or ambiguous option, invalid or
  missing value, value for a flag), reported on stderr (openEMS
  e89286f aborted there on an uncaught exception, `SIGABRT`, status 134).
- **VTK files** (dumps, SAR, debug dumps): openEMS be30220's own writer's
  layout — no `RangeMin`/`RangeMax`, no information keys, unnamed
  coordinate arrays, inline binary arrays everywhere (§5.16.5); the
  layout of the earlier VTK-library writer is not written any more. ems takes the first non-option argument as the
  model; openEMS takes the first argument, whatever it is.
- **Console:** the key lines keep openEMS's form (`FDTD simulation size:
  11x7x13 --> 1001 FDTD cells` — openEMS counts mesh nodes as cells, also
  in its statistics —, `FDTD timestep is: … s; Nyquist rate: N timesteps
  @f Hz` with `f = 1/(2N·dt)`, `Excitation signal length is: L timesteps
  (…s)` or `… period is: P timesteps` for a sinusoid, `Max. number of
  timesteps: M ( --> M/L * Excitation signal length)`, progress reports
  every 4 s, `Time for … iterations with … cells : … sec`, `Speed: …
  MCells/s`); banner and remaining wording are ems's own; warnings go to
  stderr.
- **Signal length:** Gaussian `⌊2T/dt⌋ + 1` samples (cut to `M`),
  sinusoid one period `P` (two written), Dirac and step 2, custom `M`.
- **Probe files after the run:** openEMS's H-type files (current, H field,
  waveguide current) end with the state after the last step `N`, at
  `(N + ½)·dt`; ems evaluates that H update locally after
  `Simulation::run` (`Probe::lookahead`, the rule of the dumps' final
  sample) and writes it, so all files have openEMS's rows. Agreement on
  `cli_run`: 1e-8…2e-6 relative L2.
- **Gaps:** probe files are written at the end, not during the run.
  (A `DumpBox` without `DumpMode`, mode 0 in openEMS and 1 in csx, is
  read as mode 0 by `read_model`, §5.16.3.)

## 8. What ems needs from csx

csx is a work in progress. These are the interface points ems will rely on, so
they can be planned on the csx side.

| Need | csx status (as of 0.0.0 + current tree) |
|---|---|
| `Vec3`, `BoundingBox`, `CoordSystem`, `Transform`, `ParameterSet`/`Scalar`, `expr` evaluation with variables (`t` for custom waveforms) | present |
| Iterate properties → primitives with priority, Cartesian bounding box, `dimension()`, and `is_inside(point, cs)` with closed-bound tolerance | primitives in progress |
| Material getters per direction (ε, μ, κ, σ) at a point, including weighting | planned (properties) |
| Typed Excitation / ProbeBox / DumpBox / LumpedElement / Metal / ConductingSheet fields (type, vector, direction, R/L/C, weights, frequency lists, dump type/mode, sub-sampling) | planned (properties) |
| Background material | planned (structure) |
| `RectilinearGrid` lines + delta unit + coordinate system; smooth-mesh helpers | planned (grid) |
| Model types are `Send + Sync` (no `Rc`/`RefCell`), so the compile step can rasterise in parallel | to confirm |
| Cheap `Clone` of a structure (for `update()` without mutating the caller's model) | to confirm |
| XML: access to the openEMS `<FDTD>` element, as a typed struct or a raw element tree that ems maps itself (§11 Q1) | `csx::fdtd::FdtdSettings` + `OpenEmsDocument::fdtd_settings` (csx tree, after 0.0.1) |

Anything solver-specific stays out of csx, so csx remains useful to other
solvers and tools.

### 8.1 Feedback for csx (found while integrating)

Open items found by ems work against csx, re-checked against csx 0.0.2.
These are reported here, not fixed in csx from this repo.

1. **Release.** *Resolved:* csx 0.0.2 ships `discretize`, `fdtd` and `mesh`;
   ems depends on it from crates.io (no git pin).
2. **Gaussian docs.** `ExcitationSignal::Gaussian` and the crate docs say
   "Gaussian-modulated sine", but the code (correctly, and matching ems) uses a
   cosine carrier.
3. **`endCriteria=0`.** `FdtdSettings::validate` rejects `end_criteria <= 0`,
   but 0 is a valid "no end criteria" value.
4. **Signal `f_max`.** The XML writer drops the `f_max` stored inside
   Dirac/step/custom signals; only the `<FDTD f_max>` attribute survives.
5. **No change tracking.** `ContinuousStructure` has no "needs update" flag, so
   ems clones the model and calls `update()` on every compile.
6. **Disabled excitations.** `discretize` does not appear to check
   `Excitation::enabled`.
7. **Dispersive materials.** `discretize` only reports the static values;
   ems samples the poles itself with the priority query (`compile::dispersive`).
   A per-edge dispersive-material output (property id + volume fraction per
   edge) from `discretize` would avoid duplicating its sampling. Also, the
   `LorentzPole` docs give `ε(ω) = ε∞ − ω_p²/(ω² − ω_L² − iω/τ)`, while the
   openEMS documentation multiplies the whole bracket by `ε∞`
   (`ε∞·(1 − Σ f_p²/(f² − f_L² − jf/(2πτ)))`); ems follows the openEMS
   documentation. The docs should also say that `τ = 0` means lossless and
   that frequencies are in Hz.
8. **Bare axis index.** `discretize::Edge::dir` is a `usize` rather than an
   axis type.
9. **Amplitude units.** Document explicitly that excitation amplitudes are field
   values (V/m), not edge voltages.
10. **Averaging σ_m.** csx combines magnetic σ arithmetically across the two half
    cells, while μ is combined harmonically (in series). ems's per-cell path
    combines σ_m in series. Worth deciding on one convention. openEMS
    uses the harmonic mean of the same two samples with a one-sided zero
    rule (§5.16.3), which ems re-samples for files; csx could offer it
    (or expose the half-cell samples).
11. **Thirds rule at coinciding edges.** `MeshOptions::with_thirds` still adds
    lines at edge−2d/3 and edge+d/3 when a metal edge coincides with a material
    edge whose line it keeps; in `examples/microstrip.rs` this put a line
    outside the domain and graded the PML (worked around with
    `with_thirds([NaN, 0.25, NaN])`).
12. **Thin wires in `MeshBuilder`.** Mesh lines are placed on a thin `Wire`'s
    bounding box (its surface) rather than on its axis, so models with
    sub-cell wires have to be meshed with zero-radius stand-ins. For a
    tilted wire this matters more (§5.14.1): a 170 mm dipole of radius
    0.5 mm at 45° in xz with a 10 mm target gets lines at the box faces
    `±(60.1 ± 0.5)` mm in x and z and `±0.5` mm in y, graded down to
    0.5 mm cells (69 instead of about 35 lines per axis, a 20× smaller
    time step), and the cells there are then too small for the wire, so
    the oblique model rejects it. The oblique model needs no line on the
    wire at all, only a locally uniform mesh around it. Suggestion: treat
    `Wire`s thinner than the target cell as lines (no surface lines), and
    oblique ones (and `Curve`s) as contributing no fixed lines, perhaps
    with an option to keep the mesh uniform within a few cells of them.
13. **Default `DumpMode`.** The XML reader gives a `DumpBox` without a
    `DumpMode` attribute mode 1 (node interpolation); openEMS v0.37 uses
    mode 0 (no interpolation: an `HField` dump without the attribute is
    written on the dual mesh, `dual_mesh = 1`, observed on the oracle).
    ems works around it in `io::openems::read_model` (§5.16.3).
14. **Planar sheets on cylindrical grids.** `discretize` selects the edges
    of a planar metal or conducting-sheet primitive within its bounding
    box, whose α range `Primitive::bounding_box_in` normalises to
    `[−π, π)`. On a grid with α lines in `[0, 2π]` a sheet over the full
    circle only gets the edges with `α ≤ π`, and one at `α > π` none ("not
    on a mesh plane"). ems re-selects them on the grid's α range
    (`compile::cylindrical::wrapped_sheets`). The same bounding box limits
    the edges of excitations and lumped elements (a lumped ring over
    `[0, 2π]` got only `α ≤ π`, a source at `3π/2` none); ems re-selects
    those too (`wrapped_sources`). csx should test the primitive
    periodically in α when the grid is cylindrical.
15. **`FD_Samples` separators.** openEMS reads only comma-separated lists
    (`1e9,2e9`, also with spaces after the commas and a trailing comma);
    with spaces or semicolons as separators it writes no `_FD` file (and
    runs). csx accepts spaces and semicolons too (lenient, harmless), but
    rejects a Matlab-style range `1e9:1e9:3e9` with a hard parse error
    where openEMS runs without FD output; csx should warn instead.
16. **Boxes ending at α = 2π.** *Measured* (§5.15, azimuth conventions):
    only lumped elements collapse (CSXCAD reduces each α bound of a
    lumped element modulo 2π, and the resulting α = 0 column has no effect
    in openEMS); materials, metal, sheets and excitations over `[0, 2π]`
    are the full ring in both codes. ems keeps the drawn element (openEMS
    is wrong here) and warns for models read from files; csx needs no
    change for this, beyond item 14.

17. **Inside tolerance.** csx's inside tests accept points up to
    1e-12 (relative) outside a primitive. openEMS since be30220 accepts
    points within `t = 1e-3·h_min` (the smallest cell of the mesh, in
    drawing units; `t/σ_min` in the coordinates of a transformed
    primitive), closed for every primitive (§5.16.5; openEMS e89286f
    compared exactly, had open spheres and wires, a vertex rule for
    polygons and position-dependent polyhedron faces). ems grows the
    volume primitives of a file model by `t` before csx discretises it
    (`compile::openems::dilated`), which cannot be done for polyhedra,
    rotational polygons and thin wires. An absolute inside tolerance
    option in `DiscretizeOptions` (distance from the closed primitive)
    would make that exact for every shape.
18. **Short anisotropic lists.** With `Isotropy="0"`, openEMS keeps the
    default for the components a short list leaves out (`Epsilon="3"` is
    `3,1,1`, weights default to 1); csx copies a single value to all
    axes and fills missing ones with 0 (ε_r = 0: the model does not
    run). ems rewrites the lists before csx reads them (§5.16.3).
19. **Boundary spellings.** `Boundary::from_str` is case-insensitive and
    accepts `PML 8`/`PML-8`; openEMS reads exactly `PEC`, `PMC`, `MUR`,
    `PML_<n>` and integer prefixes (`2.000000e+00` is MUR, `PML_8.0` is
    `PML_8`, `4` is PEC) and treats everything else as PEC; csx rejects
    `4`, `-1`, `PML_8.0` and `PML_x`, which openEMS runs. ems normalises
    the attribute values before csx parses them (§5.16.3).
20. **Probe and dump boxes.** openEMS ignores a primitive's
    `<Transformation>` for `ProbeBox`/`DumpBox` (not for materials or
    excitations), runs one probe per primitive (`<name>_<i>`) and needs a
    `Box` for voltage probes and lumped elements. csx keeps all primitives
    and transformations (correct as a model); a documented helper for
    "the box of a probe primitive" would avoid every solver redoing this.
21. **OverSampling.** `FdtdSettings` stores `OverSampling` as `u32`, so a
    negative value is a parse error; openEMS reads any value below 2 as
    its default (4).
22. **Per-property OverSampling.** openEMS honours an `OverSampling`
    attribute on a `<ProbeBox>` or `<DumpBox>` (time-domain sampling
    every `max(1, ⌊N/OverSampling⌋)` steps, 1 allowed, 0 crashes it,
    §5.16.9); csx's `ProbeBox`/`DumpBox` have no field for it and keep it
    among `extra_attributes`, where ems reads it
    (`compile::ProbeSpec::oversampling`). A typed
    `Option<u32>` on `ProbeBox` would also write it back.

### 8.2 Feedback for hdf5rs

Found while adding the `hdf5` feature (hdf5rs 0.0.1):

1. No `H5Write` for `hdf5rs::Complex<T>` (reading works); ems defines its own
   `{r, i}` compound.
2. `.deflate()` on a dataset builder compiles without the `deflate` feature;
   the behaviour should be a compile-time gate or a documented error.
3. `Datatype::packed_compound` returns `Result`, so fixed compounds need an
   `expect`; an infallible complex constructor would help.
4. `Sizes` is only reachable as `hdf5rs::format::Sizes`.

Found while writing the openEMS-compatible dumps (hdf5rs 0.0.1): everything
openEMS's files use is supported (superblock 0, variable-length UTF-8
string attributes, `{r, i}` f32 compounds, 1-element array attributes,
nested groups), and the reader handled every openEMS file. Minor points:

5. The zlib codec (`filters::deflate::zlib_compress`) is useful outside
   HDF5 (ems uses it for VTK arrays); a stable, documented re-export would
   help.

## 9. Validation strategy

Tests check against physics, not openEMS output. Each test is small and fast
(seconds in release mode); long ones are `#[ignore]` and run nightly.

| Test | Expectation |
|---|---|
| PEC cavity TE101/TM110 (uniform and graded mesh) | < 0.5 % from analytic |
| Lossless closed cavity, 1e5 steps | energy bounded |
| Plane wave in vacuum | phase velocity matches the Yee dispersion relation |
| CPML vs. large-domain reference | < −40 dB, normal and 45° incidence |
| Mur, normal incidence | < −30 dB |
| Dielectric slab, and an anisotropic slab per polarisation | within 1 % of Fresnel |
| Conductive medium | attenuation within 1 % of analytic |
| Lumped port into a matched parallel-plate line | \|S11\| < −30 dB |
| Zero-thickness metal sheet | transmission < −60 dB |
| Priority: material over metal, metal over material | edge state as specified |
| XML round trip: build in Rust → write XML via csx → `from_xml` → run | bit-identical results |
| Thread-count determinism | bit-identical, 1 vs. N threads |
| Waveform | Gaussian spectrum matches the analytic transform |

Plus unit tests for rasterisation (edge averaging, sheet detection), probe paths,
and the `<FDTD>` mapping.

## 10. Implementation order (M1)

csx milestones are noted where ems is blocked on them.

1. Scaffold: Cargo.toml, CI copied from csx, LICENSE, README, CLEANROOM.md,
   with a `csx` path/git dependency during development.
2. `constants`, `error`, `waveform`, `fdtd` settings, in-tree `Complex64` and DFT.
   *No csx dependency beyond `expr`.*
3. Engine core on a hand-built grid, before csx's grid lands: PEC walls, soft
   point source, point probe, thread pool.
   → cavity, stability, and determinism tests.
4. Mur, then CPML. → reflection tests.
5. Compile step from csx: grid, materials, metal sheets, priorities.
   *Needs csx primitives + properties + structure + grid (csx M5–M8).*
   → Fresnel, conductor, anisotropy, sheet tests.
6. Excitation / LumpedElement / ProbeBox operators; LumpedPort helper; S-parameters.
   → port tests.
7. DumpBox operators, `.vtr` writer, end criteria, energy.
8. `<FDTD>` mapping and `Simulation::from_xml`. *Needs csx XML (csx M9).*
   → round-trip test.
9. Benchmarks, examples, docs; optional `hdf5` feature.

Steps 2–4 are independent of csx progress, so the two crates can advance in
parallel.

### Status (2026-10-25, after wave 15)

| Step | State |
|---|---|
| 1–8 | done |
| 9 | examples (cavity, microstrip, patch antenna, dipole, run_xml), README, idle-machine benchmark and optional `hdf5` output (via `hdf5rs`) done |

Pulled forward from M2:
- Dispersive media (Drude/Lorentz/Debye, electric and magnetic) via ADE.
- Second-order lumped-port reference plane.

M1 is complete. M2 done: ConductingSheet, MSL and rectangular waveguide
ports, AbsorbingBC sheets, and (wave 10) openEMS-compatible output files,
observed with the black-box oracle. M3 done: NF2FF, periodic boundaries, Bloch pairs (engine level).
Also done: plane-wave TF/SF sources (§5.11), SAR (M4, §5.12), energy
invariant with magnetic poles and conducting sheets, medium-tuned CPML.

Wave 8 (M4 done): cylindrical grids (§5.15), conformal PEC (§5.13), thin
wires (§5.14), Bloch / oblique incidence through `Simulation` (§5.6.2).

Wave 9: azimuthal multigrid near the cylindrical axis (§5.15.1, ×3.65
time step with two zones); stretched-radius cylindrical PML on ρ_max
(§5.15.2, −94 dB with 8 cells vs −38 dB before); oblique thin wires
(§5.14.1, tilted dipoles within 0.5 % of each other); conformal
zero-thickness PEC sheets (§5.13.1, second order); plane waves and NF2FF in
homogeneous non-vacuum backgrounds, plane-wave-aware energy end criterion
(§5.10, §5.11).

Wave 10 (openEMS parity, with the black-box oracle of
`scripts/openems-oracle`, fixtures in `tests/fixtures/openems`, re-run by
the `openEMS oracle` CI workflow): openEMS XML semantics (§5.16.1:
signals, source timing, time-step rules, probe snapping, run control,
CellConstantMaterial), advanced-physics conventions (§5.16.2: dispersive
poles, plane-wave phase, waveguide probes, Mur in media, cylindrical),
openEMS probe text files and field-dump files (§7.1: VTK, HDF5 current and
legacy layouts, SAR), and coax, CPW, stripline, circular-waveguide and
curve ports (§5.5). M2 is complete.

Wave 11: the `ems` binary, a drop-in for the `openEMS` executable (§7.2:
all its options, statistics and debug files, exit codes), mode-matching
probe files with mode purity; openEMS's `nf2ff` and `sar_calc` tools
(`ems-nf2ff`, `ems-sar-calc`, §5.10, §5.12) and its in-run SIMPLE SAR
averaging; openEMS's lumped-element update and spreading, Mur energy in
the end criterion, skipped probe boxes, PML_Grading (ignored by openEMS
too; §5.16); conducting sheets, SAR, NF2FF and conformal PEC volumes on
cylindrical grids (§5.15; openEMS ignores plane waves there).

Wave 12 (XML feature-coverage sweep, §5.16.3): 169 models over every
primitive, transformation, property, probe/dump variant and FDTD
attribute csx reads, 149 agree with openEMS to ≲ 1e-4; fixed: openEMS's
reading rules (`io::openems::read_model`: boundary spellings, DumpMode
default, short anisotropic lists, probe/dump boxes with several
primitives or transformations, non-Box voltage probes and lumped
elements, odd OverSampling/TimeStepFactor/endCriteria), open metal
spheres and wires, the AbsorbingBC normal sign, OptResolution round-off
and raw H dump snapping.

Wave 12 (cylindrical gaps, §5.15): mass-averaged SAR on cylindrical
grids (exact physical cubes; openEMS's warped cubes for its files),
CellConstantMaterial, thin conformal sheets, oblique thin wires, NF2FF
mirrors normal to x / y, sector circular-waveguide and coax ports;
periodic edge selection of sources and lumped elements past α = π;
openEMS's azimuth conventions measured (§5.15).

Wave 12 (end-to-end and speed): end-to-end parity on five realistic models (§5.16.4: patch
antenna with NF2FF, microstrip filter, waveguide iris, coax-fed monopole
on a cylindrical grid, dispersive slab under a plane wave), which led to
openEMS's end-criterion energy measure, its Mur corner update and
point-sampled Mur media, the α scaling of cylindrical probe headers; and
engine speed against openEMS (benchmark below): the energy check on all
threads, Mur faces as slab hooks, serial gaps only at the steps that need
them, the time-step criteria from precomputed tables, VTK dump files
written in parallel, a default thread count bounded by the x-planes.

Wave 12 (conformal accuracy, §5.13): small cut faces are interpolated
from neighbouring faces instead of merged, so TE modes converge at second
order too (circular TE11 +0.076/+0.018/+0.0054 % at 10/20/40 cells per
radius, was +0.57/+0.26/+0.12 %), time step unchanged; conformal walls in
a CPML: the CPML is tuned before the conformal scaling, wall-node
crossings snap to the node, merged faces in the layer apply the
stretching (a conformal circular guide now decays to −50 dB; coax Z
+0.53/+0.069/+0.017 %, was +27/+15/+8 %).

Wave 13 (openEMS parity on the file path, §5.16): openEMS's own PML
(Gedney's uniaxial layer with its fixed grading, `engine::boundary::upml`;
PML runs went from 1e-3…2.4e-2 to ~1e-6, the WR-90 iris and filter ports
to 3e-7…1e-6); the Rennings time step on doubly graded and layered meshes
(exact on ~1 600 random meshes; the e2e fixtures no longer fix
`TimeStep`); the end-criterion energy offset (outer dual edges, x faces,
the half-length check: ≤ 3e-4 dB to −32 dB); current probes half-way
between lines; Dirac/step dumps every step; openEMS's σ_m dual-edge rule,
its ConductingSheet model (per-square table measured from its outputs,
first row/column of cells left out) and polygon vertex ties; on
cylindrical grids the debug material/operator/PEC dumps (`.vts`), the
radial material weights, the axis E_z update and closed-grid dump layout;
SAR averaging edge cases (cell weights, solver safeguards, IEEE validity:
≤ 1.3e-7 on the new fixtures, cylindrical ≤ 1e-7); the capacitance limit
of parallel R‖L without C. The sweep stands at 163 of 174 models agreeing
to ≲ 1e-4 (8 differ, 3 rejected by openEMS).

Wave 13 (native): a radial-line port and microstrip ports along z or α on
cylindrical grids (§5.5, §5.15); sector CPML α faces graded by arc length
(was tuned as if at ρ = 1 m); conformal thin sheets interpolate small
regions from their own side (TE at sheets now second order), deviation
currents on interpolated and merged faces (remove the wall-guided wave
and a late-time growth of cut cells in a CPML that predated the wave),
cylindrical ρ_max PML on merged faces (§5.13).

Wave 14 (openEMS parity on the file path, §5.15, §5.16): openEMS's time
step on cylindrical grids (`engine::timestep_cyl`, its printed digits on
~650 meshes and runs; the cylindrical fixtures run on it) and α seam
current loops; materials averaged as if metal were absent (time-step ties
up to 1.4 %), the f32 CFL step, H probes snapped to cells, the energy
measure without E on the three upper faces (Mur TEM lines 0.99 dB →
4e-4 dB), lumped R/C ignored inside its PML, a randomized differential
sweep of 230 combined-feature models (212 agree; 11 fixes: sheet and
line priorities, poles in the PML, Mur/pole order, Mur corners, hard E
sources, the series-element capacitance limit, held Mur faces, …);
openEMS's ConductingSheet width rule (also cylindrical, oblique polygon
sides).

Wave 14 (native): plane waves in layered and dispersive backgrounds from
the exact discrete spectral incident field (−125…−135 dB leakage, §5.11);
NF2FF over a stratified ground (half-space, layers, grounded slab;
`Nf2ffResult::with_ground`, §5.10); CPW, stripline and a sectoral
waveguide port on cylindrical grids (§5.5); conformal sheets with
deviation currents on merged regions, stretched in CPML and ρ_max PML
layers, junctions of several sheets and sheets touching solid metal
(§5.13); fused H/E engine steps on large slabs (×1.3–1.7 at 200³–256³,
bit-identical, §6.1).

Wave 15 (openEMS round-off, §5.16.6): openEMS's summation order of the
curl terms on Cartesian grids, fitted bit for bit from field dumps, and a
subnormal flush that leaves normal values unchanged; vacuum and metal
models now match openEMS to the printed digits (feature sweep: 83 of 169
models exact, was 9; `tie_cyl_metal`, `tie_cylshell_metal`,
`prim_sphshell_metal` 1.1e-2…1.5e-3 → 0); openEMS's
coefficient rounding (C, G, L, R rounded to f32 first, §5.16.7): 123 of
169 exact. Wave 16: openEMS's PML, Mur and AbsorbingBC arithmetic
(§5.16.8: 123 → 136 of 169 exact); lumped elements, current-probe sums,
per-property `OverSampling` and openEMS's vacuum constants (§5.16.9:
123 → 130); openEMS's computed time step bit for bit on every grid kind
(§5.16.10: 123 → 127); together 146 of 169 exact and 162 to
the printed digits (≤ 4e-13); left: `sheet_thin_resistive` 1.5e-4,
`lumped_series_rlc` 8.8e-5, the dispersive `debye_weight`,
`drude_kappa`, `lorentz_weight` 2e-5…6e-5, and `param_term`,
`prim_userdefined`, which openEMS rejects.

Remaining:
- openEMS's azimuthal multigrid coupling (3e-3…2e-2), closed-grid time
  steps with material ending at 2π (4 of 60, ≤ 5 %), the Drude fixture on
  cylindrical grids (3.8e-3), sheets next to the α seam (5e-3 on one
  edge), two sheets meeting on a line.
- From the random sweep (§5.16.3): series elements over several columns,
  H probes next to held Mur faces, conducting sheets near Mur/PML faces in
  lossy or dispersive media, two slow-growth cases.
- From the feature sweep: dispersive-pole (ADE) arithmetic; conducting-sheet capacitances (1–3 f32 steps,
  `sheet_thin_resistive` 1.5e-4) and series lumped elements (~1e-4, the
  branch state of L and C), §5.16.9; the PML, the curl order and edge
  materials of boxes in sectors and at the seam on cylindrical grids
  (§5.16.10); `CoordSystem="1"` without `CylinderCoords` runs Cartesian
  in openEMS; `endCriteria="0"` with a sinusoid runs all steps there.
- SAR averaging: ties decided by openEMS's rounding, whole-cell cubes,
  ~1 % of small C95.3 face cubes (§5.12).
- Native: small solid conformal faces still merged in CPML layers, sheets
  ending short of a surface (slits), sources on sheet-cut edges; lossy
  backgrounds for NF2FF and surface-wave power; plane waves on non-uniform
  meshes across layers and the Bloch variant in layers; fused steps for
  energy steps, multigrid, plane waves and conformal operators; temporal
  blocking; the idle-machine benchmark.

Benchmark (`bench_vacuum`, vacuum box with PEC walls, idle AMD Threadripper
9970X, 32 cores / 64 threads, master `fa06fca`), in Mcells/s:

| grid | 1 T | 8 T | 16 T | 32 T | 64 T |
|---|---|---|---|---|---|
| 128³ | 322 | 1582 | 1680 | 2338–2463 | 1149–1352 |
| 256³ | 321 | 445 | 560 | 577 | 531 |

With 8-cell CPML on all faces, 128³ at 64 T runs at 969 Mcells/s; energy
checks every 10 steps give 1113 Mcells/s. 256³ is limited by memory
bandwidth at about 50 GB/s, well below what the machine can sustain. The
64-thread drop at 128³ is not yet explained (suspects: slab granularity, SMT).

After the wave-6 engine work (row-deduplicated coefficients, physical-core
default thread count, yielding barrier), same machine, lightly loaded:
256³ at 32 T 919 Mcells/s (×1.6), 128³ at 32 T 4468 Mcells/s, 1 T
428 Mcells/s. The machine's measured stream bandwidth is about 70 GB/s; the
old kernel was at that limit. The `patch_antenna` example went from ~25 s to
2.4 s (mostly the thread-count default), with bit-identical output.

**Against openEMS (wave 12).** Vacuum boxes run as openEMS XML files
(`N³` cells of 1 mm, a soft z dipole, one voltage probe, end criterion
1e-30, `--disable-dumps`; 2000 steps for 50³ and 100³, 740 for 150³, 312
for 200³), the `Speed:` line of each program in MCells/s (both count mesh
nodes), best of two runs, on the same Threadripper 9970X shared with
other jobs (load average 4–17 during the runs, so single numbers vary by
±20 %). openEMS v0.37 engines: `basic`, `sse`, `sse-compressed`
(single-threaded) and `multithreaded` with `--numThreads`; ems before is
master `0638148`, after is this wave:

| box | openEMS basic / sse / sse-compressed (1 T) | openEMS multithreaded 1 / 4 / 8 / 16 / 32 T | ems before 1 / 4 / 8 / 16 / 32 T | ems after 1 / 4 / 8 / 16 / 32 T | best ems / best openEMS |
|---|---|---|---|---|---|
| 50³ PEC | 205 / 555 / 700 | 678 / 1075 / 938 / 368 / 139 | 573 / 1204 / 1437 / 1271 / 952 | 628 / 1861 / 2231 / 2275 / 1791 | 2.1× (before 1.3×) |
| 50³ PML_8 | 110 / 137 / 147 | 138 / 240 / 162 / 62 / 26 | 271 / 755 / 983 / 1028 / 865 | 290 / 945 / 1360 / 1556 / 1080 | 6.5× (before 4.3×) |
| 100³ PEC | 207 / 396 / 534 | 523 / 1421 / 1710 / 1536 / 870 | 531 / 1330 / 1718 / 1963 / 1834 | 439 / 2151 / 3416 / 5109 / 3795 | 3.0× (before 1.1×) |
| 100³ PML_8 | 118 / 120 / 135 | 133 / 345 / 364 / 306 / 179 | 236 / 805 / 1307 / 1560 / 1609 | 234 / 1130 / 2088 / 3424 / 2930 | 9.4× (before 4.4×) |
| 150³ PEC | 204 / 292 / 416 | 404 / 990 / 1519 / 1947 / 1703 | 355 / 751 / 1001 / 1687 / 2083 | 372 / 1015 / 1695 / 3932 / 5103 | 2.6× (before 1.1×) |
| 150³ PML_8 | 116 / 120 / 124 | 123 / 254 / 317 / 346 / 315 | 240 / 527 / 801 / 1072 / 1515 | 254 / 595 / 947 / 1770 / 3663 | 10.6× (before 4.4×) |
| 200³ PEC | 190 / 300 / 379 | 379 / 633 / 707 / 822 / 837 | 387 / 584 / 690 / 825 / 796 | 383 / 706 / 940 / 1071 / 1275 | 1.5× (before 1.0×) |
| 200³ PML_8 | 111 / 113 / 128 | 132 / 263 / 288 / 337 / 333 | 238 / 420 / 500 / 591 / 694 | 279 / 497 / 606 / 778 / 902 | 2.7× (before 2.1×) |

The large gain of "after" at 8–32 threads is the energy check of XML runs,
which ran a serial look-ahead pass every Nyquist interval (26 steps here);
it is now evaluated by all threads. openEMS's multithreaded engine peaks at
8–16 threads and slows down past that (50³ PML: 240 MCells/s at 4 threads,
26 at 32). 200³ boxes are bound by memory bandwidth in both codes (ems
reads and writes at least the 48 bytes of fields per cell and step:
1275 MCells/s is 61 GB/s, near the machine's ~70 GB/s).

The end-to-end models of §5.16.4 at full size (wall time of the whole
run including setup and file output, best of two; openEMS with its
default engine and threads, ems with its default thread count and with
master `0638148`):

| model | cells | openEMS | ems master | ems now |
|---|---|---|---|---|
| patch antenna, PML, NF2FF dumps | 467 k | 18.7 s (275 MC/s) | 3.5 s | 2.37 s (2390 MC/s) |
| microstrip filter, PML, 512 VTK dumps | 174 k | 5.7 s (270 MC/s) | 7.7 s | 1.03 s (1690 MC/s) |
| WR-90 iris, mode probes, PML | 48 k | 0.63 s (271 MC/s) | 0.36 s | 0.31 s (667 MC/s) |
| coax monopole, cylindrical, Mur | 19 k | 0.63 s (441 MC/s) | 2.4 s | 0.58 s (388 MC/s) |
| Drude/Lorentz slab, plane wave, PML | 15 k | 0.12 s (232 MC/s) | 0.17 s | 0.08 s (298 MC/s) |

(openEMS's own time step on the cylindrical coax is 27 % smaller, so it
runs 12122 steps where ems runs 8866; ems's step there is still 12 %
slower than openEMS's, with its periodic and axis merges serial every
step.) Profiles (callgrind, single thread) found the serial periodic
merge (half of the coax run), the serial Mur faces of cylindrical grids
and the probe sampling locating every node per sample, the time-step
criteria (a third of a short guide run), and the CPML corrections (46 %
of the patch, whose grid is 74 % PML: per layer cell about 1.6 times a
kernel component update; a fixed-length variant of the short z-layer
rows was slower and was not kept).

## 11. Open questions for review

1. **Who types the `<FDTD>` section?** *Decided:* csx types it
   (`csx::fdtd::FdtdSettings`, `ExcitationSignal`, `BoundaryConditions`), as
   file-level data with every field optional. ems keeps its resolved,
   validated `Fdtd` / `Waveform` and converts both ways
   (`Fdtd::from_csx` → `(Fdtd, warnings)`, `Fdtd::to_csx`). The Gaussian
   pulse definitions of both crates are identical (cosine carrier,
   `τ = √ln10/(π·fc)`, delay `τ·√(6 ln 10)`); ems additionally truncates it
   to `[0, 2T]`.
2. **Precision.** Is `f32`-only acceptable for M1, or do you want `f64` behind a
   generic from day one?
3. **No rayon.** OK with a std-only thread pool, to match csx's minimal-dependency
   stance?
4. **`unsafe` policy.** `forbid` like csx, with a possible later audited
   exception for the kernel if benchmarks justify it?
5. **openEMS output-file compatibility.** M2 as proposed, or do you need
   existing Python/Matlab post-processing to read our files sooner?
6. **HDF5.** *Decided:* optional `hdf5` feature on the pure-Rust `hdf5rs`
   crate (no C dependency), ems-native layout; openEMS layout later (§7.1).
