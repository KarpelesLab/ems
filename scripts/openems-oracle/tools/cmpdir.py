"""cmpdir.py A B : relative L2 of every probe text file present in both dirs."""
import os, sys
import numpy as np

SKIP = {"model.xml", "console.txt", "et", "ht"}


def load(p):
    rows = []
    for l in open(p):
        if l.startswith("%") or not l.strip():
            continue
        try:
            rows.append([float(v) for v in l.split()])
        except ValueError:
            return None
    return np.array(rows) if rows else None


def compare(a, b, quiet=False):
    out = {}
    for f in sorted(os.listdir(a)):
        pa, pb = os.path.join(a, f), os.path.join(b, f)
        if f in SKIP or f.endswith((".h5", ".vtr", ".vtp", ".vts", ".vtk", ".txt", ".xml")) or not os.path.isfile(pa):
            continue
        if not os.path.exists(pb):
            out[f] = ("missing", None)
            continue
        x, y = load(pa), load(pb)
        if x is None or y is None:
            continue
        if x.shape != y.shape:
            out[f] = ("shape", f"{x.shape} vs {y.shape}")
            continue
        if f.endswith("_FD"):
            v = x[:, 1:]; w = y[:, 1:]
        else:
            v = x[:, 1:]; w = y[:, 1:]
            dt = np.max(np.abs(x[:, 0] - y[:, 0])) / max(1e-300, np.max(np.abs(x[:, 0])))
            if dt > 1e-5:
                out[f] = ("time", f"{dt:.1e}")
                continue
        n = np.sqrt(np.sum(v ** 2))
        r = np.sqrt(np.sum((v - w) ** 2)) / n if n > 0 else (0.0 if np.sum(w ** 2) == 0 else np.inf)
        out[f] = ("ok", r)
    return out


if __name__ == "__main__":
    for f, (k, v) in compare(sys.argv[1], sys.argv[2]).items():
        print(f, k, f"{v:.2e}" if isinstance(v, float) else v)
