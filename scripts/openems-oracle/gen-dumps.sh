#!/bin/sh
# Regenerates the openEMS field-dump fixtures in tests/fixtures/openems/dump_*
# from their model.xml (hand-written XML), with the black-box openEMS of
# setup.sh. Run from the repository root:
#
#   scripts/openems-oracle/gen-dumps.sh [CASE...]
#
# (all cases without arguments).
# OPENEMS and OEPY override the oracle and its python (numpy + h5py).
# Per case it writes the dump files openEMS produced (all HDF5 files, a
# subset of the VTK files), structure.txt (HDF5 structure listing, see
# h5ls.py) and meta.txt (openEMS revision and command line). The single-
# threaded basic engine keeps the output deterministic.
set -eu

OPENEMS="${OPENEMS:-$HOME/.cache/ems-openems/openems}"
OEPY="${OEPY:-$HOME/.cache/ems-openems/venv/bin/python}"
REVISION="$(dirname "$OPENEMS")/REVISION"
FIX=tests/fixtures/openems
H5LS="$(pwd)/scripts/openems-oracle/h5ls.py"

CASES="$*"

# run CASE SUBDIR VTK_KEEP_REGEX [openEMS options...]
run() {
    if [ -n "$CASES" ] && ! echo " $CASES " | grep -q " $1 "; then
        return 0
    fi
    case=$1
    sub=$2
    keep=$3
    shift 3
    dir="$FIX/$case"
    out="$dir${sub:+/$sub}"
    tmp=$(mktemp -d)
    cp "$dir/model.xml" "$tmp/"
    (cd "$tmp" && "$OPENEMS" model.xml --engine=basic "$@" > openems.log 2>&1)
    mkdir -p "$out"
    rm -f "$out"/*.h5 "$out"/*.vtr "$out"/*.vts
    for f in "$tmp"/*.h5; do
        [ -e "$f" ] && cp "$f" "$out/"
    done
    for f in "$tmp"/*.vtr "$tmp"/*.vts; do
        [ -e "$f" ] || continue
        if basename "$f" | grep -Eq "$keep"; then
            cp "$f" "$out/"
        fi
    done
    if ls "$out"/*.h5 > /dev/null 2>&1; then
        (cd "$out" && "$OEPY" -I "$H5LS" *.h5 > structure.txt)
    fi
    {
        echo "openEMS oracle:"
        sed 's/^/  /' "$REVISION" 2>/dev/null || echo "  (no REVISION file)"
        echo "command: openEMS model.xml --engine=basic $*"
        echo "VTK files kept: $keep"
        grep "^Time for" "$tmp/openems.log" | sed "s/ : .*//" || true
    } > "$out/meta.txt"
    rm -rf "$tmp"
}

ALL='.'
FD_VTK='_(abs|arg|p=000|p=119|p=239)\.vtr$'

run dump_td "" "$ALL"
run dump_fd "" "$FD_VTK"
run dump_legacy "" "$ALL" --legacyHDF5Dumps
run dump_sampling "" "$ALL"
run dump_cyl "" '_00000000(06|12)\.vts$'
run dump_cyl native '_00000000(06|12)\.vts$' --nativeFieldDumps
rm -f "$FIX/dump_cyl/native/"*.h5 "$FIX/dump_cyl/native/structure.txt" 2> /dev/null || true
run dump_sar "" "$ALL"
run dump_cyl_closed "" '_00000000(31|62)\.vts$'
echo "fixtures written to $FIX"
