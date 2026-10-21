#!/bin/sh
# Regenerates the openEMS sar_calc fixtures in tests/fixtures/openems/sar_*
# with the black-box openEMS of setup.sh (tests/openems_sar_calc.rs). Run
# from the repository root:
#
#   scripts/openems-oracle/gen-sar.sh [CASE...]
#
# (all cases by default, or the named ones: dump, block, synth, edge).
# OPENEMS and OEPY override the oracle and its python (numpy + h5py).
#
# * sar_calc_dump: sar_calc on the raw SAR dump openEMS wrote for the
#   dump_sar fixture (../dump_sar/sar_raw.h5, see gen-dumps.sh), 1 g, with
#   each averaging method, the cube statistics (-e) and the legacy layout.
# * sar_calc_block: sar-synth.py writes raw.h5 (a synthetic tissue block,
#   two frequencies); sar_calc averages it over 0.05 g (SIMPLE and
#   IEEE_62704).
# * sar_calc_synth: sar-synth.py writes blob.h5 (graded mesh, random
#   density, irregular tissue), slab.h5 (a thin slab: face cubes only) and
#   aniso.h5 (1 x 1 x 3 mm cells); sar_calc averages them with each
#   method, cube statistics, light and heavy masses and the -a auto range.
# * sar_calc_edge: sar-synth.py writes tiny.h5 (1 mm cells: cubes smaller
#   than a cell), graded.h5 (cell widths 0.4 to 1.6 mm, tissue in air: the
#   IEEE validity of cubes of about two cells) and full.h5 (the graded mesh
#   filled with tissue: face cubes shorter than their voxel); cube
#   statistics only where no two candidate cubes tie (openEMS breaks ties
#   by its rounding).
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

CASES="${*:-dump block synth edge}"
want() { case " $CASES " in *" $1 "*) return 0 ;; esac; return 1; }

# --- sar_calc_dump
if want dump; then
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
fi

# --- sar_calc_block
if want block; then
d=$FIX/sar_calc_block
mkdir -p "$d"
rm -f "$d"/*.h5
"$OEPY" -I scripts/openems-oracle/sar-synth.py "$d/raw.h5"
header > "$d/meta.txt"
echo "input: sar-synth.py raw.h5" >> "$d/meta.txt"
sarcalc "$d" raw.h5 simple.h5 -m 0.05
sarcalc "$d" raw.h5 ieee62704.h5 -m 0.05 --method IEEE_62704
(cd "$d" && "$OEPY" -I "$H5LS" simple.h5 > structure.txt)
fi

# --- sar_calc_synth
if want synth; then
d=$FIX/sar_calc_synth
mkdir -p "$d"
rm -f "$d"/*.h5
header > "$d/meta.txt"
for c in blob slab aniso; do
    "$OEPY" -I scripts/openems-oracle/sar-synth.py $c "$d/$c.h5"
    echo "input: sar-synth.py $c $c.h5" >> "$d/meta.txt"
done
sarcalc "$d" blob.h5 blob_simple.h5 -m 0.05 -e
sarcalc "$d" blob.h5 blob_c95.h5 -m 0.05 --method IEEE_C95_3 -e
sarcalc "$d" blob.h5 blob_62704.h5 -m 0.05 --method IEEE_62704 -e
sarcalc "$d" blob.h5 blob_light.h5 -m 0.005
sarcalc "$d" blob.h5 blob_heavy.h5 -m 0.2 --method IEEE_C95_3
sarcalc "$d" blob.h5 blob_auto.h5 -m 0.05 -a 1
sarcalc "$d" blob.h5 blob_auto62704.h5 -m 0.05 --method IEEE_62704 -a 1.5 -e
sarcalc "$d" slab.h5 slab_c95.h5 -m 0.05 --method IEEE_C95_3 -e
sarcalc "$d" slab.h5 slab_62704.h5 -m 0.05 --method IEEE_62704 -e
sarcalc "$d" aniso.h5 aniso_simple.h5 -m 0.064 -e
sarcalc "$d" aniso.h5 aniso_small.h5 -m 0.0034 -e
(cd "$d" && "$OEPY" -I "$H5LS" blob_auto62704.h5 > structure.txt)
fi

# --- sar_calc_edge
if want edge; then
d=$FIX/sar_calc_edge
mkdir -p "$d"
rm -f "$d"/*.h5
header > "$d/meta.txt"
for c in tiny graded full; do
    "$OEPY" -I scripts/openems-oracle/sar-synth.py $c "$d/$c.h5"
    echo "input: sar-synth.py $c $c.h5" >> "$d/meta.txt"
done
sarcalc "$d" tiny.h5 tiny_simple.h5 -m 0.00003 -e
sarcalc "$d" tiny.h5 tiny_62704.h5 -m 0.0003 --method IEEE_62704 -e
sarcalc "$d" tiny.h5 tiny_c95_face.h5 -m 0.001 --method IEEE_C95_3 -e
sarcalc "$d" tiny.h5 tiny_62704_face.h5 -m 0.001 --method IEEE_62704 -e
sarcalc "$d" graded.h5 graded_c95.h5 -m 0.001 --method IEEE_C95_3 -e
sarcalc "$d" graded.h5 graded_62704.h5 -m 0.001 --method IEEE_62704 -e
sarcalc "$d" graded.h5 graded_c95_small.h5 -m 0.0006 --method IEEE_C95_3 -e
sarcalc "$d" graded.h5 graded_62704_small.h5 -m 0.0003 --method IEEE_62704
sarcalc "$d" full.h5 full_c95.h5 -m 0.003 --method IEEE_C95_3 -e
sarcalc "$d" full.h5 full_c95_small.h5 -m 0.0003 --method IEEE_C95_3
sarcalc "$d" full.h5 full_62704.h5 -m 0.001 --method IEEE_62704
fi
echo "fixtures written to $FIX/sar_calc_*"
