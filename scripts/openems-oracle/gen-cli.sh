#!/bin/bash
# Regenerates the openEMS command-line fixtures, tests/fixtures/openems/cli_*/
# (see README.md in this directory), for tests/cli.rs, which runs the `ems`
# binary the same way and compares. Not run by the test suite: the tests
# read the committed files only.
#
#   scripts/openems-oracle/gen-cli.sh [CASE...]      (from the repo root)
#
# Environment: OPENEMS (default ~/.cache/ems-openems/openems), OEPY (its
# python with h5py, for the HDF5 structure listings).
#
# Run cases (cli_run, cli_debug, cli_nodumps): a hand-written model.xml
# and a meta.txt whose `args:` line holds the openEMS options and whose
# `keep:` line lists the output files to copy back (shell patterns). The
# script runs openEMS on model.xml in a scratch directory (the basic
# engine is single-threaded: deterministic output) and rewrites meta.txt
# with the oracle revision, the command line, the exit status (`exit:`)
# and the names of all files openEMS wrote (`files:`); console.txt is the
# console output from the operator setup on, without the wall-clock lines
# (progress reports, time, speed); structure.txt lists the kept HDF5
# files (h5ls.py).
#
# cli_exit records the exit status of openEMS for command-line errors and
# special invocations (codes.txt: label, status, arguments) and the long
# options its --help lists (options.txt).
set -eu
OPENEMS="${OPENEMS:-$HOME/.cache/ems-openems/openems}"
OEPY="${OEPY:-$HOME/.cache/ems-openems/venv/bin/python}"
REV="$(dirname "$OPENEMS")/REVISION"
FIX=tests/fixtures/openems
H5LS="$(pwd)/scripts/openems-oracle/h5ls.py"
[ -d "$FIX" ] || { echo "run from the repository root" >&2; exit 1; }

cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
  cases=(cli_run cli_debug cli_debug_cyl cli_debug_cyl_sector cli_debug_sheet cli_debug_cyl_sheet cli_nodumps cli_exit)
fi

revision() {
  if [ -f "$REV" ]; then sed 's/^/revision: /' "$REV"; else echo "revision: unknown"; fi
}

gen_exit() {
  local d=$FIX/cli_exit tmp
  tmp=$(mktemp -d)
  cp "$FIX/cli_run/model.xml" "$tmp/model.xml"
  {
    echo "# label	exit status	openEMS arguments (model.xml: the cli_run model)"
    while IFS='|' read -r label args; do
      set +e
      # shellcheck disable=SC2086
      (cd "$tmp" && "$OPENEMS" $args > out.txt 2>&1)
      code=$?
      set -e
      printf '%s\t%s\t%s\n' "$label" "$code" "$args"
    done <<'EOF'
no-arguments|
help|--help
help-with-file|model.xml -h
missing-file|missing.xml
unknown-option|model.xml --bogus
invalid-value|model.xml --numThreads=abc
value-for-flag|model.xml --disable-dumps=1
missing-value|model.xml --engine
no-simulation|model.xml --no-simulation
abbreviated-option|model.xml --no-sim
EOF
  } > "$d/codes.txt"
  (cd "$tmp" && "$OPENEMS" --help 2>&1) | grep -o -- '--[A-Za-z][A-Za-z0-9-]*' | sort -u > "$d/options.txt"
  {
    echo "# openEMS command-line behaviour for ems tests; regenerate with"
    echo "# scripts/openems-oracle/gen-cli.sh cli_exit"
    revision
  } > "$d/meta.txt"
  rm -rf "$tmp"
  echo "cli_exit: $(wc -l < "$d/codes.txt") invocations, $(wc -l < "$d/options.txt") options"
}

gen_run() {
  local c=$1 d=$FIX/$1 tmp args keep code files
  [ -f "$d/model.xml" ] || { echo "$d/model.xml missing" >&2; exit 1; }
  args=$(sed -n 's/^args: *//p' "$d/meta.txt")
  keep=$(sed -n 's/^keep: *//p' "$d/meta.txt")
  tmp=$(mktemp -d)
  cp "$d/model.xml" "$tmp/"
  set +e
  # shellcheck disable=SC2086
  (cd "$tmp" && "$OPENEMS" model.xml $args > console.raw 2>&1)
  code=$?
  set -e
  files=$(cd "$tmp" && ls | grep -v -x -e model.xml -e console.raw | tr '\n' ' ' | sed 's/ $//')
  # Drop the old outputs, keep the inputs.
  find "$d" -mindepth 1 -maxdepth 1 ! -name model.xml ! -name meta.txt -delete
  for p in $keep; do
    for f in "$tmp"/$p; do
      [ -e "$f" ] && cp "$f" "$d/"
    done
  done
  if ls "$d"/*.h5 > /dev/null 2>&1; then
    (cd "$d" && "$OEPY" -I "$H5LS" *.h5 > structure.txt)
  fi
  sed -n '/^Create FDTD operator/,$p' "$tmp/console.raw" \
    | grep -v -E '^(Time for|Speed:|\[@|Multithreaded|Creation time)' > "$d/console.txt" || true
  {
    echo "# openEMS command-line fixture for ems tests; regenerate with"
    echo "# scripts/openems-oracle/gen-cli.sh $c"
    revision
    echo "command: openEMS model.xml $args"
    echo "args: $args"
    echo "keep: $keep"
    echo "exit: $code"
    echo "files: $files"
  } > "$d/meta.txt"
  rm -rf "$tmp"
  echo "$c: exit $code, wrote $files"
}

for c in "${cases[@]}"; do
  case "$c" in
    cli_exit) gen_exit ;;
    *) gen_run "$c" ;;
  esac
done
