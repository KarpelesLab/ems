#!/bin/bash
# Regenerates the openEMS reference outputs of the XML feature-coverage
# sweep, tests/fixtures/openems/sweep_*/ (see README.md in this directory
# and docs/DESIGN.md §5.16.3). Not run by the test suite: the tests
# (tests/openems_sweep.rs) read the committed files only.
#
#   scripts/openems-oracle/gen-sweep.sh [CASE...]      (from the repo root)
#
# Each case directory holds a hand-written model.xml (input, never
# rewritten here; with any file it reads, e.g. an STL mesh) and a meta.txt
# whose `keep:` line lists the openEMS output files the tests compare and
# whose `args:` line lists extra openEMS options. This script runs openEMS
# deterministically (single-threaded basic engine) on model.xml in a
# scratch directory, copies the kept files back and rewrites meta.txt:
# oracle revision, command line, time step, the names of all probe and
# dump files openEMS wrote (`files:`; the tests require ems to write the
# same set), and openEMS's warnings and errors (`console:` lines).
set -eu
OPENEMS="${OPENEMS:-$HOME/.cache/ems-openems/openems}"
OEPY="${OEPY:-$HOME/.cache/ems-openems/venv/bin/python}"
REV="${OPENEMS_REVISION:-$(dirname "$OPENEMS")/REVISION}"
: "$OEPY" # not needed by this script
ROOT="tests/fixtures/openems"
[ -d "$ROOT" ] || { echo "run from the repository root" >&2; exit 1; }

cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
  for d in "$ROOT"/sweep_*/; do
    cases+=("$(basename "$d")")
  done
fi

ARGS="--numThreads=1 --engine=basic"
for c in "${cases[@]}"; do
  dir="$ROOT/$c"
  [ -f "$dir/model.xml" ] || { echo "$dir/model.xml missing" >&2; exit 1; }
  keep=$(sed -n 's/^keep: *//p' "$dir/meta.txt")
  args=$(sed -n 's/^args: *//p' "$dir/meta.txt")
  tmp=$(mktemp -d)
  # Inputs: the model and the files it reads (everything but our outputs).
  inputs=()
  for f in "$dir"/*; do
    b=$(basename "$f")
    case " meta.txt $keep " in
      *" $b "*) ;;
      *) cp "$f" "$tmp/"; inputs+=("$b") ;;
    esac
  done
  status=0
  (cd "$tmp" && "$OPENEMS" model.xml $ARGS $args > console.raw 2>&1) || status=$?
  for f in $keep; do
    cp "$tmp/$f" "$dir/$f"
  done
  dt=$(sed -n 's/^FDTD timestep is: \([^ ]*\) s.*/\1/p' "$tmp/console.raw")
  files=$(cd "$tmp" && ls | grep -v -x -e console.raw -e et -e ht $(printf -- '-e %s ' "${inputs[@]}") | tr '\n' ' ')
  {
    echo "# openEMS reference outputs for tests/openems_sweep.rs; regenerate with"
    echo "# scripts/openems-oracle/gen-sweep.sh $c"
    echo "oracle revision:"
    sed 's/^/  /' "$REV" 2>/dev/null || echo "  unknown"
    echo "command: openEMS model.xml $ARGS $args"
    echo "exit: $status"
    echo "dt: $dt"
    echo "keep: $keep"
    echo "args: $args"
    echo "files: $files"
    grep -i -E 'warning|error|unknown|skipping' "$tmp/console.raw" \
      | grep -v -E 'three times|choose a higher|end-criteria' \
      | sed 's/^/console: /' || true
  } > "$dir/meta.txt"
  rm -rf "$tmp"
  echo "$c: dt $dt, exit $status, kept $keep"
done
