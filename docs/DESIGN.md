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
thin-conductor model of §5.9.

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
  the axis along `z`, closed azimuth required; the expressions use `rho`
  and `a` and give the `(ρ, α)` components, the probe weights carry the
  `ρ·Δα` metric (`compile/mode.rs`); the wall can be the `ρ_max` PEC face.
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
The conformal wall (§5.13) was tried on the same guide (PEC block, vacuum
cylinder, CPML ends through the cut cells): its energy did not decay
(87 % of the peak after 30 000 steps) and it ran ~20× slower, so it is
not in the tests; not investigated further here.

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
- **AbsorbingBC sheets** (csx `AbsorbingBC`, implemented,
  `engine::boundary::abc::AbsorbingSheet`): the same first-order Mur update
  on the tangential E of a rectangle on an interior mesh plane, against the
  neighbour one node behind the outward normal (`NormalSignPositive`).
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
- **Sampling** (openEMS timing, confirmed against openEMS output to 1e-6):
  a soft E source added after the E update `u^n → u^{n+1}` is sampled at
  `n·dt`, a soft H source added after `i^{n−½} → i^{n+½}` at `(n − ½)·dt`
  (both the start of their update). Sampling at the update centres
  instead (`(n + ½)·dt`, resp. `n·dt`, the earlier choice) shifts every
  signal by half a step, a 3–7 % L2 difference for a 5 GHz pulse at
  1 mm cells.
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
  `Nf2ffBox::background`). `Simulation::add_nf2ff` detects it from the
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
- **Not done.** Lossy and inhomogeneous (layered) backgrounds.
  Frequency-dependent phase centres and openEMS's nf2ff file format are
  out of scope: openEMS's nf2ff code is GPL and was not consulted.

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
- **Energy** see §5.8: the end criterion waits for the transit of the
  incident pulse through the box.
- **Limitations.** Homogeneous isotropic background around the box
  (dispersive backgrounds not detected; lossy backgrounds not supported by
  the periodic-cell variant), box clear of CPML/Mur.

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
the open fraction `f_e`. A face's open area `f_A` is the shoelace area of
the polygon of its open corners and transition points in perimeter order
(straight chords across the metal: exact for planar walls). Edges with
`f_e = 0` become PEC, cut edges non-PEC (unless their PEC came from a
sheet/wire).

**Scheme (Dey–Mittra in integral form).** With `u = E·l_open` on cut edges,
Dey–Mittra is a coefficient change: `cb → f_e·cb`, `db → db/f_A`. The
system stays `C·du/dt = Kᵀi`, `L·di/dt = −K·u`, so the energy invariant of
§6 remains exact.

**Stability: merging instead of thresholds.** Slivers (small `f_A`, long
open edge) resonate above the Courant limit: unmodified Dey–Mittra gave an
estimated `s = ωdt/2` up to 8 (sphere) and diverged. Area floors
(`f_A ≥ τ`) only became stable at τ ≈ 0.3–0.5 and then converged at first
order again. Instead, faces with `f_A < 0.5` are merged with the in-plane
neighbour across their most-open edge (enlarged-cell / USC idea; groups
still below the threshold merge on): one current per group,
`L_g = Σ f_A·L`, circulation of the merged contour, read back by Ampère on
every member face, i.e. `K → P·K` with a 0/1 merge matrix `P`, which keeps
the scheme symmetric. Groups run in a `post_h` operator (`MergedFaces`; the
member faces get `da = 1`, `db = 0` and the operator reports the group
energy). A power iteration on `S = √cb·(PK)ᵀ·(dt/L)·(PK)·√cb`, started on the
cut cells, checks `s < 0.999` before the first step; if not, the threshold
is raised ×1.25 (up to 4 rounds) and a warning is issued if it still fails.
The power estimate is from below (not a proof); the long-run energy test is
the direct check.

**Validation** (`tests/conformal.rs`, PEC block with a vacuum cavity carved
out, centres off the nodes; resonance from the windowed spectrum of a probe):

| cavity | cells | staircase | conformal |
|---|---|---|---|
| cylinder TM010 (r = 10 mm) | 5 / 10 / 20 per radius | −6.84 / −3.76 / −1.91 % | −0.32 / −0.081 / −0.020 % (order 2.02) |
| sphere lowest TM (r = 10 mm) | 5 / 8 per radius | −10.4 / −6.5 % | +0.31 / +0.41 % |
| rectangle 20×13 mm rotated 17°, TM110 | h = 1 / 0.5 mm | −4.89 / −2.46 % | −0.156 / −0.036 % |

Stability estimates 0.89–0.98 (Courant factor 0.99), no floored faces;
coarse sphere: energy drift 1.3e-6 over 20 000 steps after the source;
bit-identical results for 1 and 3 threads. The sphere's lowest mode is
triply degenerate and split by the mesh, which limits the frequency
estimate of a single probe at these resolutions.

**Limitations.** Solid metal here, thin sheets in §5.13.1; metal intruding
into a face without crossing its edges and features thinner than half a
cell between the edge samples are not seen. Voltage probes on cut edges
read `E·l_open`. The power-iteration check costs a few hundred
matrix–vector products on the whole grid at setup (single thread).

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
As for solid metal (§5.13), regions with `f_r < 0.5` are merged with the
region **on the same side** of the in-plane neighbour across their
most-open piece (that neighbour can be an uncut face, which is then owned
by the operator), until the group reaches the threshold: one current with
`L_g = Σ f_r·L` and the merged contour, `K' → P·K'`, still symmetric. A
group has at least half a face of area but no more than its members'
perimeter, so its row stays near that of an uncut face. Merging never
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

| cavity | cells | staircase | conformal sheet |
|---|---|---|---|
| cylindrical shell TM010 (r = 10 mm) | 5 / 10 / 20 per radius | −6.84 / −3.76 / −1.91 % | −0.323 / −0.081 / −0.0197 % (order 2.02) |
| right isosceles triangle (legs 20 mm) cut off a square by a 45° plate, TM21 | 20×21 / 40×41 | −2.54 / −1.26 % | −0.160 / −0.042 % (order 1.93) |
| spherical shell lowest TM (r = 10 mm) | 5 per radius | −10.4 % | +0.31 % |

The field behind the plate stays below 1e-5 of the driven side (f32
rounding of the corrections). Energy drift over 20 000 steps after the
source: 3.0e-7 (coarse spherical shell), 2.2e-7 (finite square plate tilted
about two axes, with plate edges inside cut faces). Bit-identical results
for 1 and 4 threads. Without sheets the solid-metal scheme is bit-identical
to before (probe signals compared).

**Limitations.** One sheet partitions each face: faces cut by two sheets
(junctions, crossing plates) use the partition of the sheet with more cuts
there, which leaves a gap of under a cell at the junction (warning); closed
boxes of oblique plates therefore leak at their corners. Sheets touching
solid conformal metal in the same cells are combined approximately (piece
`cb` scaled by both fractions; owned faces drop their solid-metal `f_A`).
Open polyhedra, cylindrical-coordinate boxes and conducting
(`ConductingSheet`) oblique sheets are not handled. A trace that bulges
into a face without crossing its edges is not seen. Sources, lumped
elements and CPML on cut edges are not supported (the operator owns those
voltages). Voltage probes on a cut edge read the sum of its pieces.

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
closer than a few cells to each other or to other conductors, wire ends
in the PML (the correction line is not absorbed) and cylindrical grids
are not covered; wire energy is reported, wire losses (finite
conductivity) are not modelled.

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

**Boundaries.** CPML and Mur work on the z faces as on Cartesian grids
(coax TEM reflection −86 dB with CPML(8)). A PML on the `ρ_max` face is the
cylindrical PML of §5.15.2 (−94 dB at 58 mm with 8 cells). Mur there uses
the plane-wave one-way equation, which does not match an outgoing
cylindrical wave: the reflection is of order `1/(kρ)`, measured −31 dB at
`ρ = 26 mm`, −37 dB at 56 mm, −43 dB at 116 mm (5–25 GHz; the
Cartesian-style CPML layer, still available as `plane_rho_max()` for
comparison, is alike).

**Not available on cylindrical grids** (dropped with a warning or refused):
conducting sheets (PEC instead; openEMS does run its sheet model there),
plane waves, SAR dumps, NF2FF boxes, the MSL, CPW, stripline and
rectangular-waveguide port helpers (`Error::Unsupported`). Lumped ports,
lumped elements, voltage, current and field probes, waveguide mode probes on
a `z`-normal plane (the exact weights of `compile/mode.rs` with the `ρ·Δα`
metric), field dumps (VTK and HDF5), dispersive materials (§5.16), the
coaxial and circular-waveguide port helpers (`CoaxPort`,
`CircWaveguidePort`, along `z` on the axis, §5.5) and `LumpedPort` /
`CurvePort` work.

**Validation.**

| case | result |
|---|---|
| circular PEC cavity `a = 10 mm`, `h = 15 mm`, 20 × 24 × 15 cells, with axis: TM010, TE111, TM011, TM110, TE011 | within 0.19 % of the Bessel-zero frequencies |
| same, TE211 (m = 2) | −0.65 %: the discrete azimuthal wavenumber `(2/Δα)·sin(mΔα/2)` of 24 cells accounts for −0.61 % |
| same cavity built in csx (`Simulation`), TM010 / TE111 / TM011 / TM110 | −0.07 / −0.15 / −0.41 / −0.19 % |
| lossless cavity energy, 5000 steps | drift 2.5e-8 |
| random fields at Courant factor 1.0, 20 000 steps | energy constant to 1.3e-6 |
| coax `b/a = 3`, closed azimuth, Mur ends | `|Z|` within 2e-4 of the discrete TEM value (0.1 % of `η/2π·ln(b/a)`), phase velocity 0.9999 c |
| coax with radial lumped port and 100 Ω load spanning the circle | `Z_in` within 0.7 % of the transmission-line model (0.25–2 GHz) |
| one-column port on the periodic α = 0 face vs at α = π/2 | equal within 3e-5 |
| axis port listed on one or all α copies | bit-identical |
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

### 5.16 openEMS parity: advanced physics (black-box fixtures)

Conventions confirmed by running openEMS v0.37 as a black box on
hand-written models (`tests/fixtures/openems/adv_*`, regenerated by
`scripts/openems-oracle/gen-adv.sh`; tests in `tests/openems_adv.rs`).
Each fixture fixes `TimeStep` in the XML, so both codes step with the same
`dt` and the probe samples coincide; "agreement" is the relative L2
difference of the probe signals.

**Time step.** openEMS's time step depends on the materials, with both
rules it offers; ems's CFL bound uses the vacuum speed (§5.4), so models
with dielectrics run with a different `dt` unless `TimeStep` is given.
Observed (`--no-simulation`, 6 printed digits):
- `TimeStepMethod="1"` reproduces exactly (every test geometry, uniform
  and graded meshes, ε and μ inclusions):
  `dt = min over E edges of √(C_e / Σ_h 1/L_h)` (the 4 dual edges `h`
  around the edge, `C`, `L` the circuit values of §5.2 with the averaged
  materials), i.e. the inverse square root of the diagonal of the
  discrete curl-curl operator. On a uniform vacuum cube that is
  `Δ/(2c)`, 0.866 of the CFL limit.
- The default (`3`, reported as "Rennings_2") gives the exact CFL value
  on uniform vacuum meshes of any aspect ratio, `dt ∝ √ε` in a filled
  medium, and on layered media the largest per-node sum of directional
  terms `max_h (1/L_h)·½(1/C_e1 + 1/C_e2)`; but on graded meshes and 3-D
  inclusions it is not that rule (within 2–17 %), and it is not
  reproduced in ems.

**Source timing.** A soft E source added after the E update `n → n+1`
uses the waveform at `n·dt`, a soft H source after `n−½ → n+½` at
`(n−½)·dt` (§5.7). With these, vacuum and dielectric lines agree to 1e-6.

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
the background speed, which only matters there.

**Conducting sheets** (§5.9). openEMS's `ConductingSheet` (a
"behavioural" model fitted to the loss of good conductors, per its
documentation) is a different model, not reproduced:
- copper (35 µm, 1–9 GHz, TEM line): its reflection loss `1 − |Γ|` is
  0.48–0.52 of the exact slab value `≈ 2R_s/η0` that ems reproduces to
  1 %, i.e. it behaves like a shunt impedance `≈ Z_s/2` (the two faces in
  parallel: 0.80–0.94·Z_s/2, phase −5…−19°), with no shielding (it lets
  −80…−90 dB through where the slab passes −230 dB);
- `σ = 1e6 S/m`, `t = 20 µm` (`t/δ = 1.8…4.4`): loss 0.47–0.84 of the slab,
  transmission +1…+20 dB above it;
- a resistive sheet (`σ·t = 0.01 S`, `t ≪ δ`): openEMS transmits
  −14…−34 dB below `2/(2 + η0σt)`, its effective impedance being nearly
  capacitive; ems is within 0.02 dB.
ems keeps the slab model; `tests/openems_adv.rs` records the copper ratio.


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
  cell's lower corner node. A third file column is the mode purity.
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
accepts `CylinderCoords` silently. With equal `dt` (openEMS's own rule
gives a smaller step than ems's operator bound near the axis, 1.54e-13 vs
2e-13 s on a 0.5 mm / 15° mesh; it warns but runs):
- without the axis (coax, annular cavity, closed azimuth): 1e-6, also
  static dielectrics;
- with the axis: 4.4–5.1e-3, a 0.45 % amplitude and −2.7e-4 TM010
  frequency difference, halving with the first radial cell: the axis
  cell is discretised differently (`O(Δρ_0)`);
- `MultiGrid`: 1–3e-2; openEMS's multigrid changes its own solution by
  0.6–1 % against 0.1 % for ems's embedding (§5.15.1);
- dispersive poles work in openEMS on cylindrical grids, and now in ems
  (midpoint rule in `(ρ, α, z)`, physical areas): 0.5–1e-3 on a
  Drude-filled coax (static `ε` alone 1e-6);
- conducting sheets run there in openEMS (its shunt model); ems still
  uses PEC.

**Lumped elements** (not changed here; shared with the Cartesian
mapping). openEMS distributes a `LumpedElement`'s `R` differently from
ems on edges whose dual faces are clipped or uneven: in a PMC-walled line
with the element across the full width, its effective resistance is
`R·N/Σ(w_e/Δ)` (N edges per row, `w_e` their in-domain dual widths), i.e.
75 Ω for 50 Ω over 2 cells and 62.4 Ω over 4, where ems gives 50 Ω; on
a closed-azimuth coax 21.5 Ω for 20 Ω (also with the α = 2π copy left
out). A conductivity-like spreading over the dual areas would explain the
Cartesian numbers.


**Mur and PML** (§5.6). A Mur face without `MUR_PhaseVelocity` is tuned
per edge to the local medium in openEMS: in an ε_r = 4 line it absorbs
to 0.5 %, where ems's former vacuum tuning reflected 34–49 %. ems now
does the same (`MurBoundary::with_faces_in`): 1e-6 agreement in ε_r = 4
and in vacuum. openEMS's `PML_8` reflects −58 dB on a vacuum line at
normal incidence (1 mm cells, 5 GHz pulse; opposite sign to ems's
residual), ems's CPML −86 dB; ems keeps its CPML, so runs with PML differ
by about 2e-3 (vacuum) to 4e-3 (ε_r = 4).


**Current probes** (not changed here; shared with the Cartesian probe
mapping). On a graded microstrip mesh the voltages agree to 6e-7, the
current probes (flat boxes on primary x planes around the strip) by
33 %. openEMS's file header shows the indices it uses: the box's lower
corner moves down by one line on every axis, the upper stays. So its H
loop runs on the dual lines just outside the box in the transverse
directions (as ems's does), but on the dual plane *before* a flat box's
primary plane `k` (`k − ½`), where ems uses `k + ½`; with `k − ½` ems
agrees to 6e-7. For a box coordinate between mesh lines openEMS takes the
dual plane nearest to it (a box at a dual position, e.g. z = 6.5 on a
1 mm mesh, uses that plane, ems's current choice there), so the rule
needs the unsnapped coordinate. Where the box touches the outer mesh
lines openEMS's loop stays inside the domain (a box over the whole
cross-section of a PMC/PEC line reads about half the TEM current, ems 0).


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
    or `post_e`/`pre_h`) adds one barrier per step, only if some registered
    operator declares a hook in it (`Operator::hooks() -> Hooks`, default all).
    Spawning per step would be too slow for small models.
  - Subnormals are flushed in the kernel arithmetic (`+ 2⁻¹⁰² − 2⁻¹⁰²`).
    Hardware flush-to-zero would need `unsafe`, and this way is still
    deterministic.
  - Thread count is configurable and defaults to `available_parallelism()`.
- **Kernel**: `update_h()` then `update_e()`, written with slice iterators so
  LLVM vectorises the k-loop.
- **Operators** (`Operator` trait): CPML, Mur, sources, lumped elements,
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
    a second pass over the layer cells only.
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
agrees with the fixtures to ≲ 1e-6 relative (float32), except mass-averaged
SAR at tissue surfaces (openEMS's in-run `SIMPLE` rule is not reproduced).
The module docs hold the full description.

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
    combines σ_m in series. Worth deciding on one convention.
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

### Status (2026-10-19, after wave 9)

| Step | State |
|---|---|
| 1–8 | done |
| 9 | examples (cavity, microstrip, patch antenna, dipole, run_xml), README, idle-machine benchmark and optional `hdf5` output (via `hdf5rs`) done |

Pulled forward from M2:
- Dispersive media (Drude/Lorentz/Debye, electric and magnetic) via ADE.
- Second-order lumped-port reference plane.

M1 is complete. M2 done: ConductingSheet, MSL and rectangular waveguide
ports, AbsorbingBC sheets. M2 remaining: openEMS-compatible output files,
blocked on reference files (the layout cannot be taken from openEMS's GPL
scripts). M3 done: NF2FF, periodic boundaries, Bloch pairs (engine level).
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

Remaining:
- openEMS output layout (M2, needs reference files).
- Conformal PEC on cylindrical grids; NF2FF, plane waves, MSL/waveguide
  ports, sheets and dispersive media on cylindrical grids; oblique wires on
  cylindrical grids.
- Junctions of several oblique sheets in one cell (sub-cell gap); a csx
  way to express a feed on an oblique wire; layered/dispersive backgrounds
  for plane waves and NF2FF.

Open conventions to confirm against real openEMS files:
- Waveguide probe `ModeFunction*` coordinates are taken in drawing units
  (like excitation weights).

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
