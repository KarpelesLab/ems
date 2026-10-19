#!/bin/sh
# Builds a black-box openEMS installation used only to generate reference
# fixtures (tests/fixtures/openems). See scripts/openems-oracle/README.md.
#
# Clean-room note: this compiles openEMS from its upstream sources inside a
# throw-away Debian root, then DELETES the sources, headers and bundled
# Matlab/Python scripts. Nobody working on ems reads them; only the
# compiled tools are kept and run.
#
# Needs: curl, python3, unshare with user namespaces and /etc/subuid
# (no root, no docker). Usage: setup.sh [DIR] (default ~/.cache/ems-openems)
set -eu
D=${1:-$HOME/.cache/ems-openems}
REV=${OPENEMS_REV:-master}
mkdir -p "$D/dl" "$D/root"
cd "$D/dl"

if [ ! -x "$D/root/bin/sh" ]; then
  reg=https://registry-1.docker.io/v2/library/debian
  tok=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/debian:pull" |
    python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
  auth="Authorization: Bearer $tok"
  dg=$(curl -s -H "$auth" -H "Accept: application/vnd.oci.image.index.v1+json" \
    -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" "$reg/manifests/trixie" |
    python3 -c 'import sys,json;m=json.load(sys.stdin);print([x["digest"] for x in m["manifests"] if x["platform"]["architecture"]=="amd64"][0])')
  ly=$(curl -s -H "$auth" -H "Accept: application/vnd.oci.image.manifest.v1+json" \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json" "$reg/manifests/$dg" |
    python3 -c 'import sys,json;print(json.load(sys.stdin)["layers"][0]["digest"])')
  curl -sL -H "$auth" "$reg/blobs/$ly" -o rootfs.tar.gz
  unshare --map-auto --map-root-user tar -xzf rootfs.tar.gz -C "$D/root"
  cp /etc/resolv.conf "$D/root/etc/resolv.conf"
fi

cat > "$D/inner.sh" <<'EOF'
#!/bin/sh
# Inside the user+mount namespace: mount /proc and /dev, then chroot.
R="$1"; shift
mount -t proc proc "$R/proc" 2>/dev/null || mount --rbind /proc "$R/proc"
mount --rbind /dev "$R/dev"
exec chroot "$R" /usr/bin/env -i HOME=/root PATH=/usr/sbin:/usr/bin:/sbin:/bin LANG=C.UTF-8 "$@"
EOF

cat > "$D/root/build.sh" <<EOF
#!/bin/sh
set -e
echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/99sandbox
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake git \
  libhdf5-dev libvtk9-dev libboost-all-dev libcgal-dev libtinyxml-dev ca-certificates >/dev/null
mkdir -p /opt/src && cd /opt/src
git clone -q --recursive https://github.com/thliebig/openEMS-Project.git
cd openEMS-Project && git checkout -q $REV && git submodule update -q --init --recursive
git log -1 --format='openEMS-Project %H %cd' > /opt/REVISION
git submodule status >> /opt/REVISION
P=/opt/openEMS
for d in fparser CSXCAD openEMS; do
  mkdir -p /opt/build/\$d && cd /opt/build/\$d
  cmake -DCMAKE_INSTALL_PREFIX=\$P -DFPARSER_ROOT_DIR=\$P -DCSXCAD_ROOT_DIR=\$P \
    -DCMAKE_BUILD_TYPE=Release /opt/src/openEMS-Project/\$d >/dev/null 2>&1
  make -j"\$(nproc)" install >/dev/null 2>&1
  echo "built \$d"
done
# Clean room: keep only the compiled tools and libraries.
rm -rf /opt/src /opt/build \$P/include \$P/share \$P/bin/openEMS.sh
EOF
unshare --map-auto --map-root-user --mount --pid --fork sh "$D/inner.sh" "$D/root" sh /build.sh

cat > "$D/openems" <<EOF
#!/bin/sh
# Runs the black-box openEMS build through the Debian root's loader.
# Usage: openems [openEMS|nf2ff|sar_calc] ARGS...  (default: openEMS)
RT=$D/root
tool=openEMS
case "\${1:-}" in openEMS|nf2ff|sar_calc) tool=\$1; shift;; esac
exec \$RT/lib64/ld-linux-x86-64.so.2 \\
  --library-path \$RT/opt/openEMS/lib:\$RT/usr/lib/x86_64-linux-gnu:\$RT/lib/x86_64-linux-gnu \\
  \$RT/opt/openEMS/bin/\$tool "\$@"
EOF
chmod +x "$D/openems"
[ -d "$D/venv" ] || { python3 -m venv "$D/venv" && "$D/venv/bin/pip" install -q numpy h5py; }
cat "$D/root/opt/REVISION"
echo "ready: $D/openems   (python with h5py: $D/venv/bin/python)"
