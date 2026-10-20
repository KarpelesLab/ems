#!/bin/sh
# Regenerates the openEMS reference outputs of the cylindrical-grid feature
# fixtures, tests/fixtures/openems/cyl_*/ (see README.md in this directory;
# tests/openems_cyl.rs and the cylindrical cases of tests/openems_dumps.rs).
# Not run by the test suite: the tests read the committed files only.
#
# Each case directory holds a hand-written model.xml (input, never
# rewritten here) and a meta.txt whose `keep:` line lists the openEMS
# output files (shell patterns) the tests use; `args:` lists extra openEMS
# options. This script runs openEMS on model.xml in a scratch directory,
# copies the kept files back, lists the structure of kept HDF5 files
# (structure.txt, see h5ls.py) and rewrites meta.txt (openEMS revision,
# command line, time step).
#
# Usage (from the repository root): scripts/openems-oracle/gen-cyl.sh [CASE...]
# Environment: OPENEMS (default ~/.cache/ems-openems/openems), OEPY (its
# python with numpy + h5py, default ~/.cache/ems-openems/venv/bin/python).
set -eu
OPENEMS=${OPENEMS:-$HOME/.cache/ems-openems/openems}
OEPY=${OEPY:-$HOME/.cache/ems-openems/venv/bin/python}
OEDIR=$(dirname "$OPENEMS")
FIX=tests/fixtures/openems
H5LS="$(pwd)/scripts/openems-oracle/h5ls.py"
[ -d "$FIX" ] || { echo "run from the repository root" >&2; exit 1; }
if [ $# -eq 0 ]; then
  set -- $(cd "$FIX" && ls -d cyl_*)
fi
for c in "$@"; do
  d=$FIX/$c
  [ -f "$d/model.xml" ] || { echo "$d/model.xml missing" >&2; exit 1; }
  keep=$(sed -n 's/^keep: *//p' "$d/meta.txt")
  args=$(sed -n 's/^args: *//p' "$d/meta.txt")
  tmp=$(mktemp -d)
  cp "$d/model.xml" "$tmp/"
  # The basic engine is single-threaded: deterministic output.
  (cd "$tmp" && "$OPENEMS" model.xml --engine=basic $args > console.txt 2>&1)
  h5=""
  for pat in $keep; do
    for f in "$tmp"/$pat; do
      [ -e "$f" ] || { echo "$c: openEMS wrote no $pat" >&2; exit 1; }
      cp "$f" "$d/"
      case "$f" in *.h5) h5="$h5 $(basename "$f")" ;; esac
    done
  done
  if [ -n "$h5" ]; then
    (cd "$d" && "$OEPY" -I "$H5LS" $h5 > structure.txt)
  fi
  dt=$(sed -n 's/^FDTD timestep is: \([^ ]*\) s.*/\1/p' "$tmp/console.txt")
  {
    echo "# openEMS reference outputs for ems tests; regenerate with"
    echo "# scripts/openems-oracle/gen-cyl.sh $c"
    if [ -f "$OEDIR/REVISION" ]; then
      sed 's/^/revision: /' "$OEDIR/REVISION"
    fi
    echo "command: openEMS model.xml --engine=basic $args"
    echo "dt: $dt"
    echo "keep: $keep"
    echo "args: $args"
  } > "$d/meta.txt"
  rm -rf "$tmp"
  echo "$c: dt $dt, kept $keep"
done
