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

Fixtures generated with the oracle live in `tests/fixtures/openems/`, each
case with its input XML, the openEMS outputs needed by the tests, and a
note of the openEMS revision (`~/.cache/ems-openems/REVISION`). Tests read
the committed fixtures only; CI does not need the oracle.
