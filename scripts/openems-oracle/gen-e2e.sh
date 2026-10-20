#!/bin/bash
# Regenerates the end-to-end fixtures under tests/fixtures/openems/e2e_*
# (realistic models exercising many features together: patch antenna with
# NF2FF box, microstrip filter, waveguide iris with mode probes, coax-fed
# monopole on a cylindrical grid, dispersive slab under a plane wave;
# tests/openems_e2e.rs) with the black-box openEMS oracle (see README.md).
# Not run by the tests: they read the committed outputs only.
#
#   scripts/openems-oracle/gen-e2e.sh [case...]      (from the repo root)
#
# Each case directory keeps its hand-written model.xml (and nf2ff.xml, a
# job for the nf2ff tool run on the dumps afterwards) and a meta.txt whose
# `keep:` line lists the files the tests use and `args:` extra openEMS
# options. openEMS runs single-threaded (--engine=basic: deterministic);
# the kept files are copied back, with console.txt (the console output
# from the operator setup on, without wall-clock lines), and meta.txt is
# rewritten (oracle revision, command line, time step, steps run).
set -eu
OPENEMS="${OPENEMS:-$HOME/.cache/ems-openems/openems}"
OEPY="${OEPY:-$HOME/.cache/ems-openems/venv/bin/python}"
REV="${OPENEMS_REVISION:-$(dirname "$OPENEMS")/REVISION}"
: "$OEPY" # not needed by this script
ROOT="tests/fixtures/openems"
[ -d "$ROOT" ] || { echo "run from the repository root" >&2; exit 1; }

cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
  for d in "$ROOT"/e2e_*/; do
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
  if [ -f "$dir/nf2ff.xml" ]; then
    cp "$dir/nf2ff.xml" "$tmp/"
  fi
  # shellcheck disable=SC2086 # word splitting of the option lists
  (cd "$tmp" && "$OPENEMS" model.xml $BASE $args > console.raw 2>&1)
  if [ -f "$tmp/nf2ff.xml" ]; then
    (cd "$tmp" && "$OPENEMS" nf2ff nf2ff.xml > nf2ff.log 2>&1)
  fi
  # Drop the old outputs, keep the inputs.
  find "$dir" -mindepth 1 -maxdepth 1 ! -name model.xml ! -name nf2ff.xml ! -name meta.txt -delete
  for f in $keep; do
    cp "$tmp/$f" "$dir/$f"
  done
  sed -n '/^Create .*FDTD operator/,$p' "$tmp/console.raw" \
    | { grep -v -E '^(Time for|Speed:|Multithreaded|Creation time|\[@)' || true; } > "$dir/console.txt"
  dt=$(sed -n 's/^FDTD timestep is: \([^ ]*\) s.*/\1/p' "$tmp/console.raw")
  steps=$(sed -n 's/^Time for \([0-9]*\) iterations.*/\1/p' "$tmp/console.raw")
  {
    echo "# openEMS reference outputs for tests/openems_e2e.rs; regenerate with"
    echo "# scripts/openems-oracle/gen-e2e.sh $c"
    echo "oracle revision:"
    sed 's/^/  /' "$REV" 2>/dev/null || echo "  unknown"
    echo "command: openEMS model.xml $BASE $args"
    if [ -f "$tmp/nf2ff.xml" ]; then
      echo "command: openEMS nf2ff nf2ff.xml"
    fi
    echo "dt: $dt"
    echo "steps: $steps"
    echo "keep: $keep"
    echo "args: $args"
  } > "$dir/meta.txt"
  rm -rf "$tmp"
  echo "$c: dt $dt, $steps steps, kept $keep"
done
