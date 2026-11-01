#!/usr/bin/env python3
"""Finds the first field values that differ between openEMS and ems at each step.

Usage: EMS=target/release/ems firstdiff.py DIR [STEPS]

Adds whole-domain per-step E and H dumps (`Et`, `Ht`: raw fields, one dump per
step, HDF5) to DIR/model.xml, limits the run to STEPS (default 60), runs both
codes on one thread and prints, for each field, the first step with a
difference above the dumps' rounding and the entries `(component, i, j, k)`
there, with both values. Run
with the oracle's venv python. Black box: reads the dumps the programs write.
"""
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpcmp  # noqa: E402


def frames(path):
    """`[(step number, array)]` of a dump's `/FieldData/TD`, by step."""
    with h5py.File(path, "r") as f:
        td = f["FieldData"]["TD"]
        names = sorted(td.keys(), key=int)
        return [(int(n), np.array(td[n])) for n in names]


def main():
    d = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    text = open(os.path.join(d, "model.xml")).read()
    lines = {
        a: [float(v) for v in re.search(rf"<{a}Lines>([^<]*)", text).group(1).split(",")]
        for a in "XYZ"
    }
    lo = " ".join(f'{a}="{lines[a][0]!r}"' for a in "XYZ")
    hi = " ".join(f'{a}="{lines[a][-1]!r}"' for a in "XYZ")
    dumps = "".join(
        f'<DumpBox Name="{n}" DumpType="{t}" DumpMode="0" FileType="1" OverSampling="1000">'
        f'<Primitives><Box Priority="0"><P1 {lo}/><P2 {hi}/></Box></Primitives></DumpBox>'
        for n, t in (("Et", 0), ("Ht", 1))
    )
    text = text.replace("</Properties>", dumps + "</Properties>")
    text = re.sub(r'NumberOfTimesteps="\d+"', f'NumberOfTimesteps="{steps}"', text)
    text = re.sub(r'endCriteria="[^"]*"', 'endCriteria="0"', text)
    w = os.path.join(d, "fd")
    os.makedirs(w, exist_ok=True)
    with open(os.path.join(w, "model.xml"), "w") as out:
        out.write(text)
    dumpcmp.run_both(w, [])
    for name in ("Et", "Ht"):
        a = frames(os.path.join(w, "oe", name + ".h5"))
        b = frames(os.path.join(w, "em", name + ".h5"))
        print(name, "frames", len(a), len(b))
        for (na, x), (_, y) in zip(a, b):
            if x.shape != y.shape:
                print("  shapes", x.shape, y.shape)
                break
            # An `f32` dump division can differ by an ulp (design §5.16.11):
            # beyond that the state differs.
            where = np.argwhere(np.abs(x - y) > 4e-7 * np.abs(x))
            if len(where):
                print(f"  first at step {na}, {len(where)} values, shape {x.shape}")
                for w_ in where[:12]:
                    t = tuple(int(q) for q in w_)
                    print(f"    {t}: openEMS {float(x[t])!r} ems {float(y[t])!r}")
                break
        else:
            print("  identical")


if __name__ == "__main__":
    main()
