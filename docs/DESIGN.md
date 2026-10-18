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
| M2 | ConductingSheet (thin lossy conductor) surface model | ConductingSheet |
| M2 | MSL and rectangular waveguide port helpers | probes, excitations with weighting |
| M2 | openEMS-compatible probe/dump output files (paths, text layout, HDF5 layout) | — |
| M2 | AbsorbingBC sheets | AbsorbingBC |
| M3 | NF2FF, radiation patterns, gain/directivity | DumpBox |
| M3 | Periodic / Bloch boundaries | — |
| M4 | Cylindrical-coordinate mesh (incl. multigrid near the axis) | cylindrical `RectilinearGrid` |
| M4 | Conformal / sub-cell PEC, thin wires | Wire/Curve primitives |
| M4 | SAR | DumpBox |
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
only in how σ_m is combined across an interface. Conducting sheets are
approximated as PEC (with a warning) until the M2 sheet model.

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
  (`tests/validation_port.rs`).
- Power waves with reference `Z_ref` (Pozar) give `S_ij`, `Z_in`, and
  accepted/incident power.

### 5.6 Boundaries
`Boundary::{Pec, Pmc, Mur{v_ph}, Pml(Cpml)}` per face, mirroring the openEMS
`<BoundaryCond>` choices (including `PML_n`).

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
  - Grading `m = 3` and `σ_max = 0.8·(m+1)/(η·Δ_local)`; κ_max and α_max are
    configurable.
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
- **Periodic/Bloch**: M3. Neighbour indexing goes through the boundary layer, so
  wrap-around fits without a rewrite.

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
  - Memory: about 72 bytes/cell. A material-ID lookup table would cut this to
    about 28 B/cell; it is only worth doing if a benchmark shows a gain.
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
    `Operator::energy`. Magnetic poles use the dual update on H. Runs as
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

### Status (2026-10-18)

| Step | State |
|---|---|
| 1–8 | done |
| 9 | examples (cavity, microstrip, patch antenna, dipole, run_xml), README and idle-machine benchmark done; `hdf5` in progress |

Pulled forward from M2:
- Dispersive media (Drude/Lorentz/Debye, electric and magnetic) via ADE.
- Second-order lumped-port reference plane.

Remaining M1 items:
- The optional `hdf5` feature.

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
