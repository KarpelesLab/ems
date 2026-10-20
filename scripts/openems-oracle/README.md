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

Four scripts regenerate the fixture sets; run them from the repository
root (`OPENEMS` overrides the oracle path, `OEPY` its python). Each runs
openEMS deterministically (single-threaded `--engine=basic`) on the
hand-written `model.xml` of every case and records the oracle revision and
command line in the case's `meta.txt`:

- `gen-core.sh [CASE...]` regenerates the core-semantics cases (those whose
  `model.xml` carries the "ems core parity fixture" comment: signals,
  excitation kinds and delay, probe semantics, FD probes, time-step
  methods, materials, lumped elements and ports, boundaries, end criteria;
  `tests/openems_core.rs`). It keeps every file openEMS writes (probe
  files, the excitation signals `et`/`ht`) and the console output
  without banner and timing lines (`console.txt`).
- `gen-adv.sh [CASE...]` regenerates the `adv_*` cases (dispersive media,
  plane waves, waveguide probes, cylindrical grids, boundaries;
  `tests/openems_adv.rs`), copying back the files listed on the `keep:`
  line of the case's `meta.txt`.
- `gen-cyl.sh [CASE...]` regenerates the `cyl_*` cases (cylindrical-grid
  features: conducting sheets on ρ, α and z surfaces, SAR dumps, plane
  waves;
  `tests/openems_cyl.rs`, `tests/openems_dumps.rs`), copying back the
  files listed on the `keep:` line of the case's `meta.txt` (shell
  patterns) and listing the structure of kept HDF5 files
  (`structure.txt`).
- `gen-dumps.sh` regenerates the `dump_*` cases (field dumps and SAR files
  in VTK and HDF5; `tests/openems_dumps.rs`): all HDF5 files, a subset of
  the VTK files, and `structure.txt`, the HDF5 structure listing of
  `h5ls.py`.
- `gen-cli.sh [CASE...]` regenerates the `cli_*` cases (the `ems`
  command-line program as a drop-in `openEMS`; `tests/cli.rs`): it runs
  openEMS with the options on the `args:` line of the case's `meta.txt`,
  copies back the files on its `keep:` line, and records the exit status
  and the names of all files written (`exit:`, `files:`), the console
  output without wall-clock lines (`console.txt`) and the HDF5 structure
  (`structure.txt`); `cli_exit` holds openEMS's exit status for
  command-line errors and special invocations (`codes.txt`) and the
  options its `--help` lists (`options.txt`).
- `gen-nf2ff.sh` regenerates the `nf2ff_*` cases of the `nf2ff` tool
  (`tests/openems_nf2ff.rs`): openEMS runs a dipole with NF2FF box dumps
  and `nf2ff` transforms them (`nf2ff_dipole`), and `nf2ff` runs on the
  synthetic planes and job files written by `nf2ff-synth.py`
  (`nf2ff_synth`).
- `gen-sar.sh` regenerates the `sar_calc_*` cases of the `sar_calc` tool
  (`tests/openems_sar_calc.rs`), from the raw SAR dump of `dump_sar` and
  from the synthetic tissue block of `sar-synth.py`.
