"""Randomized differential sweep: random small openEMS models, both codes, compare outputs.

usage: rsweep.py OUTDIR SEED0 SEED1 [--only-gen]
"""
import os, re, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import both, cmpdir


def fmt(x):
    return f"{x:.6g}"


class Model:
    def __init__(self, seed):
        self.rng = rng = random.Random(seed)
        self.seed = seed
        self.props = []
        self.tags = []
        # ---- mesh
        self.lines = []
        for a in range(3):
            n = rng.randrange(10, 17)
            base = rng.choice([0.5, 1.0, 1.0, 1.5])
            xs = [0.0]
            graded = rng.random() < 0.5
            for i in range(n):
                d = base
                if graded:
                    d = base * rng.choice([0.5, 0.75, 1, 1, 1.25, 1.5])
                xs.append(round(xs[-1] + d, 4))
            self.lines.append(xs)
        if any(len(set(np.round(np.diff(l), 6))) > 1 for l in self.lines):
            self.tags.append("graded")
        # ---- boundaries
        self.bc = []
        for a in range(3):
            for side in range(2):
                r = rng.random()
                ncell = len(self.lines[a]) - 1
                if r < 0.2:
                    b = "PEC"
                elif r < 0.3:
                    b = "PMC"
                elif r < 0.65:
                    b = "MUR"
                else:
                    b = f"PML_{rng.randrange(3, min(6, ncell // 3) + 1)}"
                self.bc.append(b)
        self.tags += sorted({b.split("_")[0] for b in self.bc})
        self.pml = [int(b.split("_")[1]) if b.startswith("PML") else 0 for b in self.bc]
        # interior index range per axis (outside PML layers, one cell margin)
        self.inner = []
        for a in range(3):
            lo = self.pml[2 * a] + 1
            hi = len(self.lines[a]) - 1 - self.pml[2 * a + 1] - 1
            if hi - lo < 3:
                lo, hi = 1, len(self.lines[a]) - 2
            self.inner.append((lo, hi))
        # ---- excitation
        self.f0 = rng.uniform(3e9, 8e9)
        self.fc = self.f0 * rng.uniform(0.4, 0.6)
        self.steps = rng.choice([600, 800, 1200])
        self.ec = rng.choice(["1e-4", "1e-30", "1e-30"])
        self.tsm = rng.choice(["", "", "", ' TimeStepMethod="1"'])
        self.build()

    def node(self, a, inner=True):
        lo, hi = self.inner[a] if inner else (0, len(self.lines[a]) - 1)
        return self.rng.randrange(lo, hi + 1)

    def coord(self, a, i):
        return self.lines[a][i]

    def box_idx(self, size=(1, 3), inner=True, flat=None, line=None):
        """random box in index space."""
        out = []
        for a in range(3):
            lo_, hi_ = self.inner[a] if inner else (0, len(self.lines[a]) - 1)
            if flat == a or (line is not None and a != line):
                i = self.rng.randrange(lo_, hi_ + 1)
                out.append((i, i))
            else:
                i = self.rng.randrange(lo_, max(lo_, hi_ - size[0]) + 1)
                j = min(hi_, i + self.rng.randrange(size[0], size[1] + 1))
                if j == i:
                    j = min(len(self.lines[a]) - 1, i + 1)
                out.append((i, j))
        return out

    def box_xml(self, b, prio=0, offgrid=False):
        p1 = [self.coord(a, b[a][0]) for a in range(3)]
        p2 = [self.coord(a, b[a][1]) for a in range(3)]
        if offgrid:
            for a in range(3):
                if b[a][0] != b[a][1]:
                    p1[a] += self.rng.choice([0, 0.3]) * (self.lines[a][b[a][0] + 1] - self.lines[a][b[a][0]])
                    p2[a] -= self.rng.choice([0, 0.2]) * (self.lines[a][b[a][1]] - self.lines[a][b[a][1] - 1])
        return (f'<Box Priority="{prio}"><P1 X="{fmt(p1[0])}" Y="{fmt(p1[1])}" Z="{fmt(p1[2])}"/>'
                f'<P2 X="{fmt(p2[0])}" Y="{fmt(p2[1])}" Z="{fmt(p2[2])}"/></Box>')

    def build(self):
        rng = self.rng
        P = self.props
        # materials
        for m in range(rng.randrange(0, 4)):
            b = self.box_idx((2, 6), inner=rng.random() < 0.6)
            kind = rng.random()
            prio = rng.randrange(0, 5)
            if kind < 0.7:
                eps = fmt(rng.uniform(1, 6)); mue = fmt(rng.choice([1, 1, 1, rng.uniform(1, 3)]))
                kap = fmt(rng.choice([0, 0, rng.uniform(0, 0.05)]))
                sig = fmt(rng.choice([0, 0, 0, rng.uniform(0, 300)]))
                P.append(f'<Material Name="mat{m}"><Property Epsilon="{eps}" Mue="{mue}" Kappa="{kap}" Sigma="{sig}"/><Primitives>{self.box_xml(b, prio, rng.random() < 0.3)}</Primitives></Material>')
                self.tags.append("material")
            elif kind < 0.85:
                fp = fmt(rng.uniform(2e9, 7e9)); tau = fmt(rng.uniform(2e-10, 3e-9))
                P.append(f'<LorentzMaterial Name="drude{m}"><Property Epsilon="{fmt(rng.uniform(1, 3))}" EpsilonPlasmaFrequency="{fp}" EpsilonRelaxTime="{tau}"/><Primitives>{self.box_xml(b, prio)}</Primitives></LorentzMaterial>')
                self.tags.append("drude")
            else:
                fp = fmt(rng.uniform(2e9, 5e9)); f0 = fmt(rng.uniform(4e9, 12e9))
                P.append(f'<LorentzMaterial Name="lor{m}"><Property Epsilon="{fmt(rng.uniform(1, 3))}" EpsilonPlasmaFrequency="{fp}" EpsilonLorPoleFrequency="{f0}" EpsilonRelaxTime="{fmt(rng.uniform(3e-10, 2e-9))}"/><Primitives>{self.box_xml(b, prio)}</Primitives></LorentzMaterial>')
                self.tags.append("lorentz")
        # metals
        for m in range(rng.randrange(0, 3)):
            if rng.random() < 0.5:
                b = self.box_idx((1, 4))
                self.tags.append("metal")
            else:
                b = self.box_idx((2, 5), flat=rng.randrange(3))
                self.tags.append("metalsheet")
            P.append(f'<Metal Name="pec{m}"><Primitives>{self.box_xml(b, rng.randrange(0, 6), rng.random() < 0.3)}</Primitives></Metal>')
        # conducting sheet
        if rng.random() < 0.25:
            b = self.box_idx((2, 5), flat=rng.randrange(3))
            sig = fmt(rng.choice([1e3, 1e5, 5.8e7])); t = fmt(rng.choice([1e-5, 5e-5, 1e-4]))
            P.append(f'<ConductingSheet Name="cs" Conductivity="{sig}" Thickness="{t}"><Primitives>{self.box_xml(b, rng.randrange(0, 6))}</Primitives></ConductingSheet>')
            self.tags.append("csheet")
        # lumped elements
        for m in range(rng.randrange(0, 3)):
            d = rng.randrange(3)
            if rng.random() < 0.6:
                b = self.box_idx((1, 2), line=d)
            else:
                b = self.box_idx((1, 2))
            if b[d][0] == b[d][1]:
                b[d] = (b[d][0], min(len(self.lines[d]) - 1, b[d][0] + 1))
                if b[d][0] == b[d][1]:
                    b[d] = (b[d][0] - 1, b[d][1])
            t = rng.random()
            if t < 0.35:
                attrs = f'R="{fmt(rng.uniform(25, 200))}"'; tag = "R"
            elif t < 0.5:
                attrs = f'C="{fmt(rng.uniform(1e-13, 2e-12))}"'; tag = "C"
            elif t < 0.65:
                attrs = f'R="{fmt(rng.uniform(25, 200))}" C="{fmt(rng.uniform(1e-13, 1e-12))}"'; tag = "RC"
            elif t < 0.8:
                attrs = f'R="{fmt(rng.uniform(5, 30))}" L="{fmt(rng.uniform(5e-9, 2e-8))}"'; tag = "RL"
            else:
                attrs = f'R="{fmt(rng.uniform(5, 40))}" L="{fmt(rng.uniform(1e-9, 5e-9))}" C="{fmt(rng.uniform(5e-13, 2e-12))}" LEtype="1"'; tag = "series"
            caps = rng.choice([0, 1])
            P.append(f'<LumpedElement Name="le{m}" Direction="{d}" Caps="{caps}" {attrs}><Primitives>{self.box_xml(b, 5)}</Primitives></LumpedElement>')
            self.tags.append("lumped" + tag)
            if rng.random() < 0.3:
                ex = [0, 0, 0]; ex[d] = rng.choice([1, -1])
                P.append(f'<Excitation Name="lex{m}" Type="0" Excite="{ex[0]},{ex[1]},{ex[2]}"><Primitives>{self.box_xml(b, 5)}</Primitives></Excitation>')
                self.tags.append("port")
        # main source
        d = rng.randrange(3)
        typ = rng.choice([0, 0, 0, 1, 2])
        if typ == 2:
            b = self.box_idx((1, 1), inner=True)
        else:
            b = self.box_idx((1, 2), line=d, inner=rng.random() < 0.9)
        if b[d][0] == b[d][1]:
            b[d] = (max(0, b[d][0] - 1), b[d][0]) if b[d][0] > 0 else (0, 1)
        ex = [fmt(rng.choice([0, 0, rng.uniform(-1, 1)])) for _ in range(3)]
        ex[d] = "1"
        if typ == 2:
            self.tags.append("hsource")
        if typ == 1:
            self.tags.append("hardsource")
        P.append(f'<Excitation Name="src" Type="{typ}" Excite="{",".join(ex)}"><Primitives>{self.box_xml(b, 0)}</Primitives></Excitation>')
        # probes
        for k in range(rng.randrange(3, 7)):
            t = rng.random()
            fd = f'<FD_Samples>{fmt(self.f0)},{fmt(self.f0 * 1.2)}</FD_Samples>' if rng.random() < 0.3 else ""
            if t < 0.3:
                a = rng.randrange(3)
                b = self.box_idx((1, 3), line=a, inner=rng.random() < 0.8)
                if b[a][0] == b[a][1]:
                    continue
                P.append(f'<ProbeBox Name="u{k}" Type="0">{fd}<Primitives>{self.box_xml(b)}</Primitives></ProbeBox>')
            elif t < 0.5:
                a = rng.randrange(3)
                b = self.box_idx((1, 3), flat=a)
                if any(b[x][0] == b[x][1] for x in range(3) if x != a):
                    continue
                P.append(f'<ProbeBox Name="i{k}" Type="1" NormDir="{a}">{fd}<Primitives>{self.box_xml(b)}</Primitives></ProbeBox>')
            else:
                b = self.box_idx((0, 0), inner=rng.random() < 0.8)
                b = [(x[0], x[0]) for x in b]
                pt = self.box_xml(b)
                if rng.random() < 0.3:  # off-grid point
                    pt = re.sub(r'X="([^"]*)"', lambda m_: f'X="{fmt(float(m_.group(1)) + 0.2)}"', pt)
                typ = 2 if t < 0.75 else 3
                P.append(f'<ProbeBox Name="{"e" if typ == 2 else "h"}{k}" Type="{typ}">{fd}<Primitives>{pt}</Primitives></ProbeBox>')
        # full raw E and H dumps (the global comparison)
        full = [(0, len(self.lines[a]) - 1) for a in range(3)]
        P.append(f'<DumpBox Name="FE" DumpType="0" DumpMode="0" FileType="1"><Primitives>{self.box_xml(full)}</Primitives></DumpBox>')
        P.append(f'<DumpBox Name="FH" DumpType="1" DumpMode="0" FileType="1"><Primitives>{self.box_xml(full)}</Primitives></DumpBox>')
        # a raw dump (line) sometimes
        if rng.random() < 0.3:
            a = rng.randrange(3)
            b = self.box_idx((3, 8), line=a)
            q = rng.choice([0, 1])
            P.append(f'<DumpBox Name="D{q}" DumpType="{q}" DumpMode="0" FileType="1"><Primitives>{self.box_xml(b)}</Primitives></DumpBox>')
            self.tags.append("dump")

    def xml(self):
        bc = " ".join(f'{n}="{v}"' for n, v in zip(["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"], self.bc))
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<!-- random differential sweep model, seed {self.seed}; tags: {" ".join(sorted(set(self.tags)))} -->
<openEMS>
  <FDTD NumberOfTimesteps="{self.steps}" endCriteria="{self.ec}"{self.tsm}>
    <Excitation Type="0" f0="{fmt(self.f0)}" fc="{fmt(self.fc)}"/>
    <BoundaryCond {bc}/>
  </FDTD>
  <ContinuousStructure CoordSystem="0">
    <Properties>
      {chr(10).join("      " + p for p in self.props).strip()}
    </Properties>
    <RectilinearGrid DeltaUnit="0.001" CoordSystem="0">
      <XLines>{",".join(fmt(x) for x in self.lines[0])}</XLines>
      <YLines>{",".join(fmt(x) for x in self.lines[1])}</YLines>
      <ZLines>{",".join(fmt(x) for x in self.lines[2])}</ZLines>
    </RectilinearGrid>
  </ContinuousStructure>
</openEMS>
'''


def console_facts(p):
    c = open(p, errors="replace").read()
    dt = re.search(r"FDTD timestep is: (\S+)", c)
    st = re.search(r"Time for (\d+) iterations", c)
    return (float(dt.group(1)) if dt else None, int(st.group(1)) if st else None, c)


def h5cmp(a, b):
    import h5py
    try:
        fa, fb = h5py.File(a), h5py.File(b)
        ga, gb = fa["FieldData/TD"], fb["FieldData/TD"]
        if sorted(ga) != sorted(gb):
            return ("keys", f"{len(ga)} vs {len(gb)}")
        m = max(np.abs(np.asarray(ga[k])).max() for k in ga) or 1e-300
        d = max(np.abs(np.asarray(ga[k]).astype(float) - np.asarray(gb[k]).astype(float)).max() for k in ga)
        return ("ok", d / m)
    except Exception as e:  # noqa
        return ("err", str(e)[:60])


def evaluate(d):
    to = console_facts(f"{d}/oe/console.txt"); te = console_facts(f"{d}/em/console.txt")
    res = {"dt_oe": to[0], "dt_em": te[0], "steps_oe": to[1], "steps_em": te[1]}
    res["files"] = {f: (s, v) for f, (s, v) in cmpdir.compare(f"{d}/oe", f"{d}/em").items()}
    for f in os.listdir(f"{d}/oe"):
        if f.endswith(".h5") and os.path.exists(f"{d}/em/{f}"):
            res["files"][f] = h5cmp(f"{d}/oe/{f}", f"{d}/em/{f}")
    # probe peaks: a probe below 1e-3 of the strongest probe of its kind is weak
    peaks = {}
    for f in res["files"]:
        if f.endswith(".h5"):
            continue
        try:
            a = cmpdir.load(f"{d}/oe/{f}")
            peaks[f] = np.abs(a[:, 1:]).max() if a is not None else 0.0
        except Exception:  # noqa
            peaks[f] = 0.0
    kind = lambda f: f.rstrip("_FD")[0]
    top = {}
    for f, p in peaks.items():
        top[kind(f)] = max(top.get(kind(f), 0.0), p)
    res["weak"] = [f for f, p in peaks.items() if p < 1e-3 * top[kind(f)]]
    # global scale from the full dumps: a probe whose difference is below
    # 1e-5 of the field scale at its geometry is f32 round-off of a weak signal
    import h5py
    scale = {}
    for q, fn in (("E", "FE.h5"), ("H", "FH.h5")):
        try:
            g = h5py.File(f"{d}/oe/{fn}")["FieldData/TD"]
            scale[q] = max(np.abs(np.asarray(g[k])).max() for k in g)
        except Exception:  # noqa
            pass
    xml = open(f"{d}/model.xml").read()
    geo = {}
    for m_ in re.finditer(r'<ProbeBox Name="([^"]+)" Type="(\d)"[^>]*>.*?<P1 X="([^"]+)" Y="([^"]+)" Z="([^"]+)"/><P2 X="([^"]+)" Y="([^"]+)" Z="([^"]+)"/>', xml):
        p1 = np.array([float(x) for x in m_.group(3, 4, 5)]) * 1e-3
        p2 = np.array([float(x) for x in m_.group(6, 7, 8)]) * 1e-3
        ext = np.abs(p2 - p1)
        t = int(m_.group(2))
        sc = {0: scale.get("E", 0) * ext.max(), 1: scale.get("H", 0) * 2 * (ext.sum()), 2: scale.get("E", 0), 3: scale.get("H", 0)}.get(t, 0)
        geo[m_.group(1)] = sc
    res["absnorm"] = {}
    for f in list(res["files"]):
        base = f[:-3] if f.endswith("_FD") else f
        if f.endswith(".h5") or base not in geo or not geo[base] or f.endswith("_FD"):
            continue
        a = cmpdir.load(f"{d}/oe/{f}"); b = cmpdir.load(f"{d}/em/{f}")
        if a is None or b is None or a.shape != b.shape:
            continue
        an = np.abs(a[:, 1:] - b[:, 1:]).max() / geo[base]
        res["absnorm"][f] = an
        if an < 1e-5 and f not in res["weak"]:
            res["weak"].append(f)
            if f + "_FD" in res["files"]:
                res["weak"].append(f + "_FD")
    worst = 0.0; wf = ""
    for f, (s, v) in res["files"].items():
        if f in res["weak"]:
            continue
        x = v if s == "ok" and isinstance(v, float) else float("inf")
        if not np.isfinite(x) or x > worst:
            worst, wf = x, f
    if res["dt_oe"] and res["dt_em"] and abs(res["dt_em"] / res["dt_oe"] - 1) > 6e-6:
        worst, wf = float("inf"), "dt"
    if res["steps_oe"] != res["steps_em"]:
        wf = (wf + " steps").strip()
        worst = max(worst, 1.0)
    res["worst"] = worst; res["worst_file"] = wf
    oe_nan = "nan" in to[2].lower()
    for f in res["files"]:
        if not f.endswith(".h5") and not oe_nan:
            try:
                a = cmpdir.load(f"{d}/oe/{f}")
                oe_nan = a is not None and not np.isfinite(a).all()
            except Exception:  # noqa
                pass
    em_div = "diverged" in te[2]
    if oe_nan and em_div:
        res["worst"] = 0.0; res["worst_file"] = "UNSTABLE(both)"
    elif oe_nan or em_div:
        res["worst_file"] = ("UNSTABLE(oe)" if oe_nan else "UNSTABLE(em)") + " " + wf
    return res


if __name__ == "__main__":
    out = sys.argv[1]; s0, s1 = int(sys.argv[2]), int(sys.argv[3])
    only = "--only-gen" in sys.argv
    for s in range(s0, s1):
        d = f"{out}/{s}"
        os.makedirs(d, exist_ok=True)
        m = Model(s)
        open(f"{d}/model.xml", "w").write(m.xml())
        if only:
            continue
        both.run(d, [])
        r = evaluate(d)
        json.dump(r, open(f"{d}/result.json", "w"))
        flag = "BAD" if r["worst"] > 1e-4 else "ok "
        print(f"{flag} {s:4d} worst {r['worst']:.2e} {r['worst_file']:12s} steps {r['steps_oe']}/{r['steps_em']} tags {' '.join(sorted(set(m.tags)))}", flush=True)
