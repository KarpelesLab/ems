#!/bin/bash
# Regenerates the core-semantics fixtures under tests/fixtures/openems/
# (the cases whose model.xml has the "ems core parity fixture" comment)
# with the black-box openEMS oracle (see README.md). Not run by the tests:
# they read the committed outputs only.
#
#   scripts/openems-oracle/gen-core.sh [case...]      (from the repo root)
#
# Each case directory keeps its hand-written model.xml and gets every file
# openEMS writes there (probe files, the excitation signals et/ht),
# console.txt (the openEMS console output without the banner and timing
# lines) and meta.txt (oracle revision, command line).
set -eu
OPENEMS="${OPENEMS:-$HOME/.cache/ems-openems/openems}"
OEPY="${OEPY:-$HOME/.cache/ems-openems/venv/bin/python}"
REV="${OPENEMS_REVISION:-$(dirname "$OPENEMS")/REVISION}"
: "$OEPY" # not needed by this script
ROOT="tests/fixtures/openems"
[ -d "$ROOT" ] || { echo "run from the repository root" >&2; exit 1; }

cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
  for d in "$ROOT"/*/; do
    if grep -q "ems core parity fixture" "$d/model.xml" 2>/dev/null; then
      cases+=("$(basename "$d")")
    fi
  done
fi

ARGS="--numThreads=1 --engine=basic --exact-endcriteria"
for c in "${cases[@]}"; do
  dir="$ROOT/$c"
  tmp=$(mktemp -d)
  cp "$dir/model.xml" "$tmp/model.xml"
  (cd "$tmp" && "$OPENEMS" model.xml $ARGS > console.raw 2>&1)
  # Drop the old outputs, keep the model.
  find "$dir" -mindepth 1 -maxdepth 1 ! -name model.xml -delete
  for f in "$tmp"/*; do
    case "$(basename "$f")" in
      model.xml|console.raw) ;;
      *) cp "$f" "$dir/" ;;
    esac
  done
  # Console: from the operator setup on, without wall-clock figures.
  sed -n '/^Create FDTD operator/,$p' "$tmp/console.raw" \
    | grep -v -E '^(Time for|Speed:|Multithreaded|Creation time)' > "$dir/console.txt"
  {
    echo "case: $c"
    echo "generator: scripts/openems-oracle/gen-core.sh"
    echo "command: openems model.xml $ARGS"
    echo "oracle revision:"
    sed 's/^/  /' "$REV" 2>/dev/null || echo "  unknown"
  } > "$dir/meta.txt"
  rm -rf "$tmp"
  echo "$c: $(ls "$dir" | tr '\n' ' ')"
done
