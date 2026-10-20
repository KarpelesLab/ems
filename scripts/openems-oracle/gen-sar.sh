#!/bin/sh
# Regenerates the openEMS sar_calc fixtures in tests/fixtures/openems/sar_*
# with the black-box openEMS of setup.sh (tests/openems_sar_calc.rs). Run
# from the repository root:
#
#   scripts/openems-oracle/gen-sar.sh
#
# OPENEMS and OEPY override the oracle and its python (numpy + h5py).
#
# * sar_calc_dump: sar_calc on the raw SAR dump openEMS wrote for the
#   dump_sar fixture (../dump_sar/sar_raw.h5, see gen-dumps.sh), 1 g, with
#   each averaging method, the cube statistics (-e) and the legacy layout.
# * sar_calc_block: sar-synth.py writes raw.h5 (a synthetic tissue block,
#   two frequencies); sar_calc averages it over 0.05 g (SIMPLE and
#   IEEE_62704).
#
# sar_calc runs single-threaded (-n 1); its output is deterministic but for
# the @proc_time attribute.
set -eu

OPENEMS="${OPENEMS:-$HOME/.cache/ems-openems/openems}"
OEPY="${OEPY:-$HOME/.cache/ems-openems/venv/bin/python}"
REVISION="$(dirname "$OPENEMS")/REVISION"
FIX=tests/fixtures/openems
H5LS="$(pwd)/scripts/openems-oracle/h5ls.py"
[ -d "$FIX" ] || { echo "run from the repository root" >&2; exit 1; }

# sarcalc DIR INPUT OUTPUT ARGS...
sarcalc() {
    dir=$1
    in=$2
    out=$3
    shift 3
    (cd "$dir" && "$OPENEMS" sar_calc -n 1 -i "$in" -o "$out" "$@" > /dev/null 2>&1) || true
    [ -f "$dir/$out" ] || { echo "sar_calc did not write $dir/$out" >&2; exit 1; }
    echo "command: sar_calc -n 1 -i $in -o $out $*" >> "$dir/meta.txt"
}

header() {
    echo "openEMS oracle:"
    sed 's/^/  /' "$REVISION" 2>/dev/null || echo "  (no REVISION file)"
}

# --- sar_calc_dump
d=$FIX/sar_calc_dump
mkdir -p "$d"
rm -f "$d"/*.h5
header > "$d/meta.txt"
raw=../dump_sar/sar_raw.h5
sarcalc "$d" $raw simple.h5 -m 1 --method SIMPLE
sarcalc "$d" $raw simple_e.h5 -m 1 -e
sarcalc "$d" $raw legacy.h5 -m 1 --legacyHDF5Dumps
sarcalc "$d" $raw ieee62704.h5 -m 1 --method IEEE_62704
sarcalc "$d" $raw c95.h5 -m 1 --method IEEE_C95_3
(cd "$d" && "$OEPY" -I "$H5LS" simple_e.h5 legacy.h5 > structure.txt)

# --- sar_calc_block
d=$FIX/sar_calc_block
mkdir -p "$d"
rm -f "$d"/*.h5
"$OEPY" -I scripts/openems-oracle/sar-synth.py "$d/raw.h5"
header > "$d/meta.txt"
echo "input: sar-synth.py raw.h5" >> "$d/meta.txt"
sarcalc "$d" raw.h5 simple.h5 -m 0.05
sarcalc "$d" raw.h5 ieee62704.h5 -m 0.05 --method IEEE_62704
(cd "$d" && "$OEPY" -I "$H5LS" simple.h5 > structure.txt)
echo "fixtures written to $FIX/sar_calc_*"
