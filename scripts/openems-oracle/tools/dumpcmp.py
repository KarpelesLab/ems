#!/usr/bin/env python3
"""Compares the debug dumps of a cylindrical or Cartesian model bit for bit.

Usage: dumpcmp.py [-v] DIR    (DIR/model.xml; runs both.py with the debug options)

Runs both codes with `--no-simulation --debug-material --debug-operator` and
prints, per array and component, the values that differ where both dumps
describe the same edge; -v adds the first differing nodes. Black box: reads
files the two programs write. Run with the oracle's venv python. Also a module
(`run`), used by rsweep_cyl.py. The ems binary is $EMS, as for both.py.
"""
import argparse
import base64
import os
import re
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import both  # noqa: E402

DEBUG = ["--no-simulation", "--debug-material", "--debug-operator"]
FILES = ("material_dump", "operator_dump")


def _inflate(text):
    """Decodes one vtkZLib-compressed binary DataArray payload (UInt32 header)."""
    head = base64.b64decode(text[:24])
    blocks, _, _ = struct.unpack("<III", head[:12])
    hlen = 12 + 4 * blocks
    b64 = 4 * ((hlen + 2) // 3)
    head = base64.b64decode(text[:b64])
    sizes = struct.unpack(f"<{blocks}I", head[12:hlen])
    # The compressed blocks are one base64 run after the header's.
    data = base64.b64decode(text[b64:])
    out, pos = [], 0
    for s in sizes:
        out.append(zlib.decompress(data[pos : pos + s]))
        pos += s
    return b"".join(out)


def read_vts(path):
    """The arrays of a compressed `.vts` or `.vtr`: `{name: array[nx, ny, nz, comp]}`."""
    text = open(path, encoding="ascii", errors="replace").read()
    m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', text)
    e = [int(x) for x in m.groups()]
    n = (e[1] - e[0] + 1, e[3] - e[2] + 1, e[5] - e[4] + 1)
    out = {}
    for m in re.finditer(
        r'<DataArray type="(\w+)" Name="([^"]+)" NumberOfComponents="(\d+)"[^>]*>\s*([^<\s]*)\s*</DataArray>',
        text,
    ):
        ty, name, comps, payload = m.groups()
        dt = {"Float32": "<f4", "Float64": "<f8"}.get(ty)
        if dt is None or not payload or int(comps) != 3:
            continue
        a = np.frombuffer(_inflate(payload), dtype=dt)
        a = a.reshape(n[2], n[1], n[0], int(comps)).transpose(2, 1, 0, 3)
        out[name] = a
    return out


def cyl_mask(name, shape, closed, axis):
    """Where the dumps of a cylindrical grid describe the same edge (the
    conventions of `src/io/openems/debug.rs`, as tests/cli.rs masks them):
    True = compare. `closed`: drop the extra `α_1 + 2π` line and the `α = 0`
    copies of ρ and z edges; `axis`: the axis z edge; E coefficients of
    tangential edges on outer faces (openEMS zeroes them) and H dual edges
    crossing an outer face."""
    m = np.ones(shape, dtype=bool)
    if name == "Points":
        return m
    nx, ny, nz = shape[:3]
    if closed:
        m[:, ny - 1:] = False
        # On the `2π` line only what the seam updates there: ρ and z edges,
        # the H_α dual edge; cells past it do not exist.
        if name in ("vv", "vi", "epsilon", "kappa"):
            m[:, ny - 2, :, 1] = False
        elif name in ("iv", "ii", "mue", "sigma"):
            m[:, ny - 2, :, 0] = False
            m[:, ny - 2, :, 2] = False
        if name in ("vv", "vi"):
            m[:, 0, :, 0] = False
            m[:, 0, :, 2] = False
    if axis and name in ("vv", "vi"):
        m[0, :, :, 2] = False
    if name in ("vv", "vi"):
        last = (nx - 1, ny - 1 - (1 if closed else 0), nz - 1)
        for c in range(3):
            for o in range(3):
                if o == c:
                    continue
                if not (closed and o == 1):
                    sl = [slice(None)] * 4
                    sl[3] = c
                    for v in (0, last[o]):
                        sl[o] = v
                        m[tuple(sl)] = False
                    sl[o] = slice(None)
    if name in ("iv", "ii"):
        last = (nx - 1, ny - 1 - (1 if closed else 0), nz - 1)
        for c in range(3):
            sl = [slice(None)] * 4
            sl[3] = c
            if not (closed and c == 1):
                sl[c] = 0
                m[tuple(sl)] = False
                sl[c] = last[c]
                m[tuple(sl)] = False
                sl[c] = slice(None)
            for o in range(3):
                if o != c and not (closed and o == 1):
                    sl[o] = last[o]
                    m[tuple(sl)] = False
                    sl[o] = slice(None)
    return m


def is_cyl_flags(model):
    """`(cylindrical, closed, axis)` of a model's mesh, from its text."""
    text = open(model, errors="replace").read()
    cyl = 'CylinderCoords="1"' in text
    if not cyl:
        return False, False, False
    def lines(tag):
        m = re.search(rf"<{tag}>([^<]*)</{tag}>", text)
        return [float(x) for x in m.group(1).split(",")] if m else []
    x, y = lines("XLines"), lines("YLines")
    closed = bool(y) and abs(y[-1] - y[0] - 2 * np.pi) < 1e-6
    return True, closed, bool(x) and x[0] == 0.0


def diff_vts(a, b, flags=(False, False, False)):
    """`{array name: (differing values per component, first nodes)}`; NaN == NaN."""
    out = {}
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            # ems leaves out an all-zero `exc_volt`.
            if name != "exc_volt" or (a.get(name, b.get(name)) == 0).all() is np.False_:
                out[name] = ("missing", [])
            continue
        x, y = a[name], b[name]
        if x.shape != y.shape:
            out[name] = (f"shape {x.shape} vs {y.shape}", [])
            continue
        d = (x != y) & ~(np.isnan(x) & np.isnan(y))
        if flags[0]:
            d &= cyl_mask(name, x.shape, flags[1], flags[2])
            if flags[2] and name in ("epsilon", "kappa"):
                # The axis z edge: ems shows the disc, openEMS the lines.
                d[0, :, :, 2] = False
            if flags[1] and name in ("vv", "vi", "iv", "ii"):
                # The seam's update runs in engine::openems_cyl; ems's dump
                # shows the merged copies, which may round an ulp apart (and
                # has no own `α = 0` H_α dual edge). The probes check it.
                j = x.shape[1] - 2
                if name in ("iv", "ii"):
                    d[:, 0, :, 1] = False
                near = np.abs(x[:, j] - y[:, j]) <= 1e-6 * np.abs(x[:, j])
                d[:, j] &= ~near
        if d.any():
            where = np.argwhere(d)
            first = [
                (tuple(int(q) for q in w), float(x[tuple(w)]), float(y[tuple(w)]))
                for w in where[:12]
            ]
            out[name] = ([int(d[..., c].sum()) for c in range(x.shape[-1])], first)
    return out


def run_both(d, args):
    """both.py with one thread in each code (openEMS: both.py's own option)."""
    both.run(d, args, which=("oe",))
    both.run(d, ["--numThreads=1"] + list(args), which=("em",))


def run(d):
    """Debug dumps of `d/model.xml` by both codes: `({file: diff_vts}, failure)`."""
    run_both(d, DEBUG)
    flags = is_cyl_flags(os.path.join(d, "model.xml"))
    out = {}
    for f in FILES:
        f += ".vts" if flags[0] else ".vtr"
        pa, pb = os.path.join(d, "oe", f), os.path.join(d, "em", f)
        if not os.path.exists(pa) or not os.path.exists(pb):
            return out, f"{f}: openEMS {os.path.exists(pa)}, ems {os.path.exists(pb)}"
        diff = diff_vts(read_vts(pa), read_vts(pb), flags)
        if diff:
            out[f] = diff
    return out, None


def count(diffs):
    """The number of differing values of `run`'s first result."""
    n = 0
    for dump in diffs.values():
        for v in dump.values():
            n += sum(v[0]) if isinstance(v[0], list) else 1
    return n


def report(diffs, verbose=False):
    for dump, d in diffs.items():
        for name, (cnt, first) in d.items():
            print(f"  {dump} {name}: {cnt}")
            if verbose:
                for w, x, y in first:
                    print(f"    {w}: openEMS {x!r} ems {y!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", action="store_true")
    ap.add_argument("dir")
    a = ap.parse_args()
    diffs, failed = run(a.dir)
    print("identical" if not diffs and not failed else f"{count(diffs)} values differ {failed or ''}")
    report(diffs, a.v)
