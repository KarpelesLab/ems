#!/bin/sh
# Regenerates the openEMS nf2ff fixtures in tests/fixtures/openems/nf2ff_*
# with the black-box openEMS of setup.sh (tests/openems_nf2ff.rs). Run
# from the repository root:
#
#   scripts/openems-oracle/gen-nf2ff.sh
#
# OPENEMS and OEPY override the oracle and its python (numpy + h5py).
#
# * nf2ff_dipole: openEMS runs the hand-written model.xml (single-threaded
#   basic engine: deterministic), which dumps E and H on the six faces of
#   an NF2FF box; the nf2ff tool then transforms them with the
#   hand-written nf2ff.xml. Kept: farfield.h5 and the structure of one
#   face dump (structure.txt); the dumps themselves are not kept.
# * nf2ff_synth: nf2ff-synth.py writes small synthetic near-field planes
#   (fixed seed; FD, legacy FD and TD layouts) and the job files
#   jobs/*.xml; the nf2ff tool writes out_<job>.h5 for each.
set -eu

OPENEMS="${OPENEMS:-$HOME/.cache/ems-openems/openems}"
OEPY="${OEPY:-$HOME/.cache/ems-openems/venv/bin/python}"
REVISION="$(dirname "$OPENEMS")/REVISION"
FIX=tests/fixtures/openems
H5LS="$(pwd)/scripts/openems-oracle/h5ls.py"
[ -d "$FIX" ] || { echo "run from the repository root" >&2; exit 1; }

meta() {
    echo "openEMS oracle:"
    sed 's/^/  /' "$REVISION" 2>/dev/null || echo "  (no REVISION file)"
    for c in "$@"; do
        echo "command: $c"
    done
}

# --- nf2ff_dipole
d=$FIX/nf2ff_dipole
tmp=$(mktemp -d)
cp "$d/model.xml" "$d/nf2ff.xml" "$tmp/"
(cd "$tmp" && "$OPENEMS" model.xml --engine=basic > openems.log 2>&1)
(cd "$tmp" && "$OPENEMS" nf2ff nf2ff.xml > nf2ff.log 2>&1)
rm -f "$d"/*.h5
cp "$tmp/farfield.h5" "$d/"
(cd "$tmp" && "$OEPY" -I "$H5LS" farfield.h5 nf2ff_E_xn.h5 nf2ff_H_xn.h5) > "$d/structure.txt"
meta "openEMS model.xml --engine=basic" "nf2ff nf2ff.xml" > "$d/meta.txt"
grep "^Time for" "$tmp/openems.log" | sed "s/ : .*//" >> "$d/meta.txt" || true
rm -rf "$tmp"

# --- nf2ff_synth
d=$FIX/nf2ff_synth
rm -rf "$d"
"$OEPY" -I scripts/openems-oracle/nf2ff-synth.py "$d"
for j in "$d"/jobs/*.xml; do
    (cd "$d" && "$OPENEMS" nf2ff "jobs/$(basename "$j")" > /dev/null 2>&1)
done
(cd "$d" && "$OEPY" -I "$H5LS" out_options.h5 > structure.txt)
meta "$OEPY nf2ff-synth.py $d" "nf2ff jobs/<job>.xml (from $d)" > "$d/meta.txt"
echo "fixtures written to $FIX/nf2ff_*"
