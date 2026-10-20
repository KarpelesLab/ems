#!/bin/sh
# Regenerates the openEMS reference outputs of the "advanced physics"
# fixtures, tests/fixtures/openems/adv_*/ (see README.md in this
# directory). Not run by the test suite: the tests read the committed
# files only.
#
# Each case directory holds a hand-written model.xml (input, never
# rewritten here) and a meta.txt whose `keep:` line lists the openEMS
# output files the tests use; `args:` lists extra openEMS options. This
# script runs openEMS on model.xml in a scratch directory, copies the kept
# files back and rewrites meta.txt (openEMS revision, command line, time
# step).
#
# Usage (from the repository root): scripts/openems-oracle/gen-adv.sh [CASE...]
# Environment: OPENEMS (default ~/.cache/ems-openems/openems).
set -eu
OPENEMS=${OPENEMS:-$HOME/.cache/ems-openems/openems}
OEDIR=$(dirname "$OPENEMS")
FIX=tests/fixtures/openems
[ -d "$FIX" ] || { echo "run from the repository root" >&2; exit 1; }
if [ $# -eq 0 ]; then
  set -- $(cd "$FIX" && ls -d adv_*)
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
  for f in $keep; do
    cp "$tmp/$f" "$d/$f"
  done
  dt=$(sed -n 's/^FDTD timestep is: \([^ ]*\) s.*/\1/p' "$tmp/console.txt")
  {
    echo "# openEMS reference outputs for ems tests; regenerate with"
    echo "# scripts/openems-oracle/gen-adv.sh $c"
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
