#!/bin/sh
# CI variant of setup.sh: builds the black-box openEMS directly on a Debian or
# Ubuntu host (dependencies installed by the caller, see
# .github/workflows/openems-oracle.yml), then deletes the sources, headers and
# bundled scripts. Same clean-room rules as setup.sh (README.md).
#
# Usage: build-native.sh [DIR] (default ~/.cache/ems-openems)
#   OPENEMS_REV: openEMS-Project revision (default master)
set -eu
D=${1:-$HOME/.cache/ems-openems}
REV=${OPENEMS_REV:-master}
P=$D/install
W=$(mktemp -d)
mkdir -p "$D"
if [ ! -x "$P/bin/openEMS" ]; then
  git clone -q --recursive https://github.com/thliebig/openEMS-Project.git "$W/src"
  (cd "$W/src" && git checkout -q "$REV" && git submodule update -q --init --recursive &&
    { git log -1 --format='openEMS-Project %H %cd'; git submodule status; } > "$D/REVISION")
  for d in fparser CSXCAD openEMS; do
    mkdir -p "$W/build/$d"
    (cd "$W/build/$d" &&
      cmake -DCMAKE_INSTALL_PREFIX="$P" -DFPARSER_ROOT_DIR="$P" -DCSXCAD_ROOT_DIR="$P" \
        -DCMAKE_BUILD_TYPE=Release "$W/src/$d" >/dev/null 2>&1 &&
      make -j"$(nproc)" install >/dev/null 2>&1)
    echo "built $d"
  done
  # Clean room: keep only the compiled tools and libraries.
  rm -rf "$P/include" "$P/share" "$P/bin/openEMS.sh"
fi
rm -rf "$W"
cat > "$D/openems" <<EOF
#!/bin/sh
# Usage: openems [openEMS|nf2ff|sar_calc] ARGS...  (default: openEMS)
tool=openEMS
case "\${1:-}" in openEMS|nf2ff|sar_calc) tool=\$1; shift;; esac
LD_LIBRARY_PATH=$P/lib exec $P/bin/\$tool "\$@"
EOF
chmod +x "$D/openems"
[ -d "$D/venv" ] || { python3 -m venv "$D/venv" && "$D/venv/bin/pip" install -q numpy h5py; }
cat "$D/REVISION"
