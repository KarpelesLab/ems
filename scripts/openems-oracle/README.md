# openEMS oracle (fixture generator)

ems is a clean-room implementation. To match openEMS behaviour and file
formats we treat a compiled openEMS as a **black box**: we feed it XML
models, run it, and look at what comes out. This directory has the recipe
for building that black box without root and without docker.

```sh
scripts/openems-oracle/setup.sh            # → ~/.cache/ems-openems
~/.cache/ems-openems/openems model.xml     # openEMS (any of its CLI options)
~/.cache/ems-openems/openems nf2ff x.xml   # the nf2ff tool
~/.cache/ems-openems/openems sar_calc ...  # the sar_calc tool
~/.cache/ems-openems/venv/bin/python       # numpy + h5py, for inspecting output
```

The setup script builds openEMS from upstream inside a throw-away Debian
root (user namespaces), then deletes the sources, headers and the bundled
Matlab/Python scripts, keeping only the compiled tools and libraries.

## Rules (clean room)

Allowed:
- running the tools on XML we wrote (by hand, or via csx/ems);
- reading their `--help` text, console output and any files they write
  (probe text files, HDF5, VTK, statistics);
- inspecting HDF5 output structure with h5py (BSD) or hdf5rs;
- comparing numbers between ems and openEMS.

Not allowed:
- reading openEMS / CSXCAD source code, headers, or their Matlab/Python
  scripts (the setup script deletes them; do not re-download them to read);
- copying anything but observed *output formats* and *numbers*.

## CI

`.github/workflows/openems-oracle.yml` builds the oracle natively on Ubuntu
(`build-native.sh`, cached per upstream revision), reruns every
`gen-*.sh` script, and runs the tests against the regenerated fixtures:
weekly, on demand, and when fixtures or these scripts change. The `gen-*.sh`
scripts call `${OPENEMS:-~/.cache/ems-openems/openems}` and
`${OEPY:-~/.cache/ems-openems/venv/bin/python}` from the repo root.

Fixtures generated with the oracle live in `tests/fixtures/openems/`, each
case with its input XML, the openEMS outputs needed by the tests, and a
note of the openEMS revision (`~/.cache/ems-openems/REVISION`). Tests read
the committed fixtures only; CI does not need the oracle.

`REVISION_PIN` names the openEMS-Project revision the committed fixtures
were made with. The `openEMS oracle` workflow builds that revision on
pushes and pull requests (so a regenerated fixture set must reproduce the
committed one), and upstream HEAD on its weekly run or a manual run with
`upstream` set; that run only reports drift (it may fail without
blocking). When the fixtures are regenerated with a newer oracle, update
`REVISION_PIN` with them. The fixtures moved from e89286f (2026-10-18)
to be30220 (2026-10-21) in wave 14: be30220 writes VTK XML without the
range attributes and array names of the VTK library (`PEC_dump.vtp`
inline instead of appended), lists `--gpu-device` in `--help`, exits
with status 1 instead of aborting on command-line errors, and has a
closed inside test with a `1e-3·h_min` tolerance for every primitive;
everything else is bit-identical (design §5.16.5).

`setup.sh` and `build-native.sh` take `OPENEMS_REV` and a target
directory, so a second revision can be built next to the oracle (e.g.
`OPENEMS_REV=<rev> setup.sh ~/.cache/ems-openems-<rev>`); every
`gen-*.sh` script then runs against it with
`OPENEMS=~/.cache/ems-openems-<rev>/openems`. `setup.sh` records the
submodule revisions in `root/opt/REVISION`; the scripts read the short
form in `REVISION` next to the `openems` wrapper (write it by hand
for a setup.sh build).

The `gen-*.sh` scripts regenerate the fixture sets; run them from the
repository root (`OPENEMS` overrides the oracle path, `OEPY` its python).
Each runs
openEMS deterministically (single-threaded `--engine=basic`) on the
hand-written `model.xml` of every case and records the oracle revision and
command line in the case's `meta.txt`:

- `gen-core.sh [CASE...]` regenerates the core-semantics cases (those whose
  `model.xml` carries the "ems core parity fixture" comment: signals,
  excitation kinds and delay, probe semantics and skipped probe boxes,
  FD probes, time-step methods, materials, lumped elements (also with
  excitations over them and on graded meshes) and ports, boundaries, end
  criteria;
  `tests/openems_core.rs`). It keeps every file openEMS writes (probe
  files, the excitation signals `et`/`ht`) and the console output
  without banner and timing lines (`console.txt`).
- `gen-adv.sh [CASE...]` regenerates the `adv_*` cases (dispersive media,
  plane waves, waveguide probes, cylindrical grids and lumped rings,
  boundaries, the ignored `PML_Grading`;
  `tests/openems_adv.rs`), copying back the files listed on the `keep:`
  line of the case's `meta.txt`.
- `gen-cyl.sh [CASE...]` regenerates the `cyl_*` cases (cylindrical-grid
  features: conducting sheets on ρ, α and z surfaces, SAR dumps incl.
  1 g averaging, plane waves, `CellConstantMaterial`, radial material
  interfaces (`cyl_radial`), the radial-line port's properties
  (`cyl_radial_port`), sources and lumped
  elements at α > π, a lumped ring ending at α = 2π and current boxes at
  the α = 0 seam (`cyl_current_seam`);
  `tests/openems_cyl.rs`, `tests/openems_dumps.rs`), copying back the
  files listed on the `keep:` line of the case's `meta.txt` (shell
  patterns) and listing the structure of kept HDF5 files
  (`structure.txt`).
- `gen-dumps.sh [CASE...]` regenerates the `dump_*` cases (field dumps and SAR files,
  also on a closed cylindrical grid with the axis,
  in VTK and HDF5; `tests/openems_dumps.rs`): all HDF5 files, a subset of
  the VTK files, and `structure.txt`, the HDF5 structure listing of
  `h5ls.py`.
- `gen-cli.sh [CASE...]` regenerates the `cli_*` cases (the `ems`
  command-line program as a drop-in `openEMS`; `tests/cli.rs`): it runs
  openEMS with the options on the `args:` line of the case's `meta.txt`,
  copies back the files on its `keep:` line, and records the exit status
  and the names of all files written (`exit:`, `files:`), the console
  output without wall-clock lines (`console.txt`) and the HDF5 structure
  (`structure.txt`); `cli_debug_cyl` and `cli_debug_cyl_sector` hold the
  debug dumps of cylindrical grids, `cli_debug_sheet` and
  `cli_debug_cyl_sheet` the operator dumps of conducting sheets (openEMS's
  sheet model, Cartesian and cylindrical); `cli_exit` holds openEMS's exit status for
  command-line errors and special invocations (`codes.txt`) and the
  options its `--help` lists (`options.txt`).
- `gen-nf2ff.sh` regenerates the `nf2ff_*` cases of the `nf2ff` tool
  (`tests/openems_nf2ff.rs`): openEMS runs a dipole with NF2FF box dumps
  and `nf2ff` transforms them (`nf2ff_dipole`), and `nf2ff` runs on the
  synthetic planes and job files written by `nf2ff-synth.py`
  (`nf2ff_synth`).
- `gen-sweep.sh [CASE...]` regenerates the `sweep_*` cases of the XML
  feature-coverage sweep (primitives, transformations, priorities,
  materials, parameters, excitations, probe and dump boxes, boundary
  spellings, absorbing sheets, conducting sheets, surface ties, FDTD
  settings; `tests/openems_sweep.rs`,
  design §5.16.3): it copies back the files on the `keep:` line of the
  case's `meta.txt` and records the exit status, the time step, the names
  of all probe and dump files openEMS wrote (`files:`) and its warnings
  (`console:`).
- `gen-sar.sh [CASE...]` regenerates the `sar_calc_*` cases of the
  `sar_calc` tool (`tests/openems_sar_calc.rs`; cases `dump`, `block`,
  `synth`, `edge`), from the raw SAR dump of `dump_sar` and
  from the synthetic raw files of `sar-synth.py` (a tissue block; a
  graded-mesh blob, a thin slab and anisotropic cells; cubes smaller
  than a cell, a graded ball in air and a filled graded mesh), with each
  averaging method, cube statistics (`-e`) and the auto range (`-a`); the
  test replays every `command:` line of the cases' `meta.txt`.
- `gen-e2e.sh [CASE...]` regenerates the `e2e_*` cases (realistic models
  using many features together: patch antenna with NF2FF box, microstrip
  filter, waveguide iris, coax-fed monopole on a cylindrical grid,
  dispersive slab under a plane wave; `tests/openems_e2e.rs`, which runs
  the `ems` binary on them): it runs openEMS (and the `nf2ff` tool when
  the case has an `nf2ff.xml`), copies back the files on the `keep:` line
  of `meta.txt` and records the time step and the number of steps run.
- `gen-rsweep.sh [CASE...]` regenerates the `rs_*` cases (models of the
  randomized differential sweep, each exposing one openEMS behaviour;
  `tests/openems_rsweep.rs`, which runs the `ems` binary on them) like
  `gen-pml.sh` (the `keep:` line of `meta.txt`).
- `gen-pml.sh [CASE...]` regenerates the `pml_*` cases (openEMS's
  `PML_<n>` layers: one face, overlapping layers with a graded layer,
  media, metal and a source inside them and raw line dumps through the
  corners, a lossy magnetic medium, lumped elements reaching into a
  layer, the energy end criterion, openEMS's layer arithmetic with a
  fixed time step (bit for bit), a
  cylindrical grid, and PML variants of the waveguide-iris and
  microstrip-filter models; `tests/openems_pml.rs`, which runs the `ems`
  binary on them): it copies back the files on the `keep:` line of
  `meta.txt` and records the time step, the number of steps run and the
  end-criterion line of the console.
