#!/bin/bash
# Regenerates the random differential sweep fixtures under
# tests/fixtures/openems/rs_* (small models generated at random, combining
# materials, dispersive media, metal boxes and sheets, conducting sheets,
# lumped elements, E/H sources, probes, line dumps, PEC/PMC/Mur/PML faces
# and graded meshes; design §5.16.3; tests/openems_rsweep.rs) with the
# black-box openEMS oracle (see README.md). Not run by the tests: they read
# the committed outputs only.
#
#   scripts/openems-oracle/gen-rsweep.sh [case...]      (from the repo root)
#
# Each case directory keeps its hand-written model.xml and a meta.txt
# whose `keep:` line lists the files the tests use and `args:` extra
# openEMS options. openEMS runs single-threaded (--engine=basic:
# deterministic); the kept files are copied back and meta.txt is
# rewritten (oracle revision, command line, time step, steps run, the
# end-criterion line of the console if any).
set -eu
OPENEMS="${OPENEMS:-$HOME/.cache/ems-openems/openems}"
REV="${OPENEMS_REVISION:-$(dirname "$OPENEMS")/REVISION}"
ROOT="tests/fixtures/openems"
[ -d "$ROOT" ] || { echo "run from the repository root" >&2; exit 1; }

cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
  for d in "$ROOT"/rs_*/; do
    cases+=("$(basename "$d")")
  done
fi

BASE="--numThreads=1 --engine=basic"
for c in "${cases[@]}"; do
  dir="$ROOT/$c"
  [ -f "$dir/model.xml" ] || { echo "$dir/model.xml missing" >&2; exit 1; }
  keep=$(sed -n 's/^keep: *//p' "$dir/meta.txt")
  args=$(sed -n 's/^args: *//p' "$dir/meta.txt")
  tmp=$(mktemp -d)
  cp "$dir/model.xml" "$tmp/"
  # shellcheck disable=SC2086 # word splitting of the option lists
  (cd "$tmp" && "$OPENEMS" model.xml $BASE $args > console.raw 2>&1)
  # Drop the old outputs, keep the inputs.
  find "$dir" -mindepth 1 -maxdepth 1 ! -name model.xml ! -name meta.txt -delete
  for f in $keep; do
    cp "$tmp/$f" "$dir/$f"
  done
  dt=$(sed -n 's/^FDTD timestep is: \([^ ]*\) s.*/\1/p' "$tmp/console.raw")
  steps=$(sed -n 's/^Time for \([0-9]*\) iterations.*/\1/p' "$tmp/console.raw")
  ec=$(sed -n 's/^RunFDTD: \(end-criteria .*\)$/\1/p' "$tmp/console.raw")
  {
    echo "# openEMS reference outputs for tests/openems_rsweep.rs; regenerate with"
    echo "# scripts/openems-oracle/gen-rsweep.sh $c"
    echo "oracle revision:"
    sed 's/^/  /' "$REV" 2>/dev/null || echo "  unknown"
    echo "command: openEMS model.xml $BASE $args"
    echo "dt: $dt"
    echo "steps: $steps"
    echo "end: $ec"
    echo "keep: $keep"
    echo "args: $args"
  } > "$dir/meta.txt"
  rm -rf "$tmp"
  echo "$c: dt $dt, $steps steps, kept $keep"
done
