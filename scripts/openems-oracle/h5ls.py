"""Prints the structure of HDF5 files: groups, datasets (shape, dtype,
compound members, layout) and attributes (dtype, shape, value), for the
structure listings of the openEMS fixtures.

Usage: python -I h5ls.py FILE...

Floating-point attribute values are printed with 6 significant digits;
`proc_time` (a wall-clock duration) is not printed.
"""
import sys

import h5py
import numpy as np


def fmt(k, v):
    if k == "proc_time":
        return "<wall-clock time>"
    a = np.asarray(v)
    if a.dtype.kind == "f":
        vals = [f"{x:.6g}" for x in a.ravel()[:8]]
    elif a.dtype.kind in "iu":
        vals = [str(int(x)) for x in a.ravel()[:8]]
    else:
        vals = [repr(x.decode() if isinstance(x, bytes) else x) for x in a.ravel()[:8]]
    more = f" ... ({a.size} values)" if a.size > 8 else ""
    if a.shape == ():
        return vals[0]
    return "[" + ", ".join(vals) + "]" + more


def attrs(obj, ind):
    for k in sorted(obj.attrs):
        a = obj.attrs.get_id(k)
        t = a.get_type()
        dt = "vlen-str" if isinstance(t, h5py.h5t.TypeStringID) and t.is_variable_str() else str(a.dtype)
        print(f"{ind}@{k}: {dt} {a.shape} = {fmt(k, obj.attrs[k])}")


def walk(name, obj, ind=""):
    if isinstance(obj, h5py.Dataset):
        t = obj.id.get_type()
        dt = str(obj.dtype)
        if isinstance(t, h5py.h5t.TypeCompoundID):
            m = [(t.get_member_name(i).decode(), str(t.get_member_type(i).dtype)) for i in range(t.get_nmembers())]
            dt = "compound{" + ", ".join(f"{n}: {d}" for n, d in m) + "}"
        print(f"{ind}{name}: dataset {obj.shape} {dt}")
        attrs(obj, ind + "  ")
    else:
        print(f"{ind}{name}/")
        attrs(obj, ind + "  ")
        for k in sorted(obj):
            walk(k, obj[k], ind + "  ")


for p in sys.argv[1:]:
    with h5py.File(p, "r") as f:
        print(f"# {p.rsplit('/', 1)[-1]}")
        walk("/", f)
