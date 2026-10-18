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
| M4 | Cylindrical-coordinate mesh (incl. multigrid near the axis) | cylindrical `RectilinearGrid` |
| M4 | Conformal / sub-cell PEC, thin wires | Wire/Curve primitives |
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
- M1 requires a Cartesian grid. A cylindrical grid is rejected with a clear
  error until M4.

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
  coordinates, values in 1/m).
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

### 5.6 Boundaries
`Boundary::{Pec, Pmc, Mur{v_ph}, Pml(Cpml), Periodic}` per face, mirroring the
openEMS `<BoundaryCond>` choices (including `PML_n`); `Periodic` is ems-only.

- **PEC**: tangential E on the face is 0.
- **PMC**: tangential H is 0. In the integral form this needs no operator:
  the kernel updates the boundary-face tangential E with its half dual face,
  and H outside the domain reads as zero (implemented, `FaceCondition::Pmc`).
- **Mur 1st order**: one-way wave equation on the tangential E of the face.
  The engine builds that face as PEC; the operator saves values in `pre_e` and
  overwrites the face's tangential E in `post_e`.
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
  coefficient-changing operators to both engines. `Simulation` does not
  use the pair yet: oblique incidence also needs a plane-wave source with
  the matching `k_t`.
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

### 5.7 Waveforms
`Waveform` is an enum: Gaussian, Sinusoid, Dirac, Step, and
`Custom(Arc<dyn Fn(f64) -> f64 + Send + Sync>)`.
- **Gaussian**: centre `f0`, 20 dB half-bandwidth `fc`, delayed so it starts
  below 1e-6 of the peak.
- **Custom**: an expression string from XML becomes `Waveform::Expression`,
  evaluated by csx's `ExcitationSignal::evaluator` (variable `t`, plus `f0`
  and `f_max`), so values match csx exactly and the source text round-trips.
- **Sampling**: E sources are sampled at `t^{n+1/2}`.
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
  rE_θ e^{jkr} = −jk/(4π)·(L_φ + η0 N_θ),
  rE_φ e^{jkr} = +jk/(4π)·(L_θ − η0 N_φ), and
  U = (|rE_θ|² + |rE_φ|²)/(2η0).
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

  The small short-dipole error comes from the second-order co-location
  of the strong reactive near field close to the source. It shrinks when
  the box is moved further out; the transform itself is exact for any
  closed surface.
- **Not done.** The background medium is vacuum only (η0, k0).
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
- **Reference.** `E_inc = Excite·a(t − Delay)` at the box corner the wave
  enters first; the 1-D source runs ahead by the numerical travel time.
- **Measured** (`tests/planewave.rs`, 1 mm cubic mesh, 8–16 GHz pulse):
  scattered-field leakage −130 dB at normal incidence and along the
  diagonals, −77 dB at 30°, about −55 dB on a mesh graded ±20 %. A PEC
  plate and a dielectric slab inside the box give the exact discrete (Yee)
  reflection to 5e-5 and 4e-5.
- **Limitations.** Homogeneous lossless background around the box (vacuum
  from csx; `PlaneWaveSpec::with_background` in the engine API), box clear of
  CPML/Mur, injected energy not tracked by `Engine::energy`.

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
is a `Metal` property with a solid (3-D) primitive; sheets, curves and wires
keep csx's staircase PEC mask. Nodes and edge midpoints in the bounding box
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

**Limitations.** Only solid metal; metal intruding into a face without
crossing its edges and features thinner than half a cell between the edge
samples are not seen. Voltage probes on cut edges read `E·l_open`. The
power-iteration check costs a few hundred matrix–vector products on the
whole grid at setup (single thread).

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
  `ThinWire` (`Compiled::thin_wires`). Oblique or off-mesh segments, radii
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
  probes, dumps, dispersive media (ADE), and later NF2FF.
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
it is off. The openEMS HDF5 layout (group names, attributes, axis order) is
not implemented: it needs reference files from a real openEMS run, and
openEMS's GPL reader scripts may not be consulted (§1). Files written by the
tests were checked by hand with h5py 3.16.

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

### 8.2 Feedback for hdf5rs

Found while adding the `hdf5` feature (hdf5rs 0.0.1):

1. No `H5Write` for `hdf5rs::Complex<T>` (reading works); ems defines its own
   `{r, i}` compound.
2. `.deflate()` on a dataset builder compiles without the `deflate` feature;
   the behaviour should be a compile-time gate or a documented error.
3. `Datatype::packed_compound` returns `Result`, so fixed compounds need an
   `expect`; an infallible complex constructor would help.
4. `Sizes` is only reachable as `hdf5rs::format::Sizes`.

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

### Status (2026-10-19, after wave 7)

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

Remaining:
- openEMS output layout (M2, needs reference files).
- Bloch boundaries through `Simulation`/`Fdtd` (needs a matching oblique
  plane wave; `BlochPair` drives two engines).
- M4: cylindrical mesh, conformal / sub-cell PEC, thin wires.

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
