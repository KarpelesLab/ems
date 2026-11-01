#!/usr/bin/env python3
"""Random cylindrical differential sweep: openEMS oracle against the ems binary.

Usage: EMS=target/release/ems rsweep_cyl.py [--family NAME[,NAME]] [--jobs N] [--gen-only] WORK FIRST LAST

Writes one random model per seed in FIRST..LAST (inclusive) under WORK and
compares them: debug dumps bit for bit (dumpcmp.py), then probes (both.py,
cmpdir.py: relative L2 per file, "exact" below TOL). The families stress what an α index or the
seam can change: `metal` (boxes crossing α = 0 or leaving the lines), `pole`
(Lorentz, Drude and Debye boxes), `sheet` (conducting sheets), `prim` (other
primitives across the seam and the transformed boxes), `mix` (all of them).
A seed and a family name the same model on every machine (Python's `random`).

Run with the oracle's venv python, `~/.cache/ems-openems/venv/bin/python`.
"""
import argparse
import concurrent.futures as cf
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmpdir  # noqa: E402
import dumpcmp  # noqa: E402

TAU = 2 * math.pi
FAMILIES = ("metal", "pole", "sheet", "prim", "mix")
TOL = 1e-9


def f(x):
    return repr(round(x, 9)) if isinstance(x, float) else str(x)


def box(p1, p2, prio, cs=""):
    a = " ".join(f'{k}="{f(v)}"' for k, v in zip("XYZ", p1))
    b = " ".join(f'{k}="{f(v)}"' for k, v in zip("XYZ", p2))
    return f'<Box Priority="{prio}"{cs}><P1 {a}/><P2 {b}/></Box>'


class Mesh:
    def __init__(self, r):
        self.axis = r.random() < 0.5
        self.closed = r.random() < 0.8
        rmin = 0.0 if self.axis else r.choice([0.5, 1.0, 2.0])
        self.rho = [rmin + 0.5 * i * r.choice([1, 1, 2]) + 0.0 for i in range(0)]
        rho = [rmin]
        for _ in range(r.randint(5, 8)):
            rho.append(rho[-1] + r.choice([0.5, 0.75, 1.0]))
        self.rho = rho
        n = r.choice([6, 8, 10, 12])
        if self.closed:
            if r.random() < 0.7:
                self.al = [TAU * j / n for j in range(n + 1)]
            else:
                w = [r.uniform(0.6, 1.4) for _ in range(n)]
                s = sum(w)
                c, self.al = 0.0, [0.0]
                for x in w[:-1]:
                    c += x * TAU / s
                    self.al.append(round(c, 6))
                self.al.append(TAU)
            if r.random() < 0.2:
                self.al = [a - math.pi for a in self.al]
        else:
            a0 = r.choice([0.0, 0.0, -0.8, 0.5])
            self.al = [a0 + 0.3 * j for j in range(n + 1)]
        self.z = [0.5 * k * 2 for k in range(r.randint(5, 8))]

    def alpha_range(self, r):
        """A random `(a0, a1)`: crossing α = 0, past the lines, on or off lines."""
        lo, hi = self.al[0], self.al[-1]
        span = hi - lo
        kind = r.randrange(8)
        d = r.uniform(0.1, 0.45) * span
        if kind == 0:
            return lo - r.uniform(0.05, 0.3) * span, lo + d
        if kind == 1:
            return hi - d, hi + r.uniform(0.05, 0.3) * span
        if kind == 2:
            return lo, lo + d
        if kind == 3:
            return hi - d, hi
        if kind == 4:
            return lo + d, lo + d + span
        if kind == 5:
            return r.choice(self.al), r.choice(self.al)
        if kind == 6:
            return lo - 0.1, hi + 0.1
        a = lo + r.uniform(0.0, 0.7) * span
        return a, a + d

    def rho_range(self, r):
        a, b = sorted(r.sample(range(len(self.rho)), 2))
        return self.rho[a], self.rho[b]

    def z_range(self, r):
        a, b = sorted(r.sample(range(len(self.z)), 2))
        return self.z[a], self.z[b]

    def xml(self):
        j = lambda v: ",".join(f(x) for x in v)
        return (
            '<RectilinearGrid DeltaUnit="0.001" CoordSystem="1">'
            f"<XLines>{j(self.rho)}</XLines><YLines>{j(self.al)}</YLines>"
            f"<ZLines>{j(self.z)}</ZLines></RectilinearGrid>"
        )


def metal(r, m, name, prio):
    a = m.alpha_range(r)
    rr = m.rho_range(r)
    if m.axis and r.random() < 0.2:
        rr = (0.0, rr[1])
    zz = m.z_range(r)
    return (
        f'<Metal Name="{name}"><Primitives>'
        + box((rr[0], a[0], zz[0]), (rr[1], a[1], zz[1]), prio)
        + "</Primitives></Metal>"
    )


def pole(r, m, name, prio):
    a = m.alpha_range(r)
    rr = m.rho_range(r)
    zz = m.z_range(r)
    b = box((rr[0], a[0], zz[0]), (rr[1], a[1], zz[1]), prio)
    e = round(r.uniform(1.0, 4.0), 3)
    kind = r.randrange(4)
    if kind == 0:
        tag, p = "LorentzMaterial", f'EpsilonPlasmaFrequency="{r.uniform(1e9, 8e9):.5g}" EpsilonRelaxTime="{r.uniform(2e-10, 2e-9):.5g}"'
    elif kind == 1:
        tag, p = "LorentzMaterial", (
            f'EpsilonPlasmaFrequency="{r.uniform(1e9, 8e9):.5g}" '
            f'EpsilonLorPoleFrequency="{r.uniform(2e9, 9e9):.5g}" EpsilonRelaxTime="{r.uniform(2e-10, 2e-9):.5g}"'
        )
    elif kind == 2:
        tag, p = "DebyeMaterial", f'EpsilonDelta="{r.uniform(0.5, 4.0):.5g}" EpsilonRelaxTime="{r.uniform(5e-11, 5e-10):.5g}"'
    else:
        tag, p = "LorentzMaterial", (
            f'MuePlasmaFrequency="{r.uniform(1e9, 4e9):.5g}" '
            f'MueLorPoleFrequency="{r.uniform(4e9, 9e9):.5g}" MueRelaxTime="{r.uniform(2e-10, 2e-9):.5g}"'
        )
    return f'<{tag} Name="{name}"><Property Epsilon="{e}" {p}/><Primitives>{b}</Primitives></{tag}>'


def sheet(r, m, name, prio):
    kind = r.randrange(3)
    i = lambda v: r.randrange(1, len(v) - 1)
    a = m.alpha_range(r)
    rr = m.rho_range(r)
    zz = m.z_range(r)
    if kind == 0:  # ρ = const tube
        q = m.rho[i(m.rho)]
        b = box((q, a[0], zz[0]), (q, a[1], zz[1]), prio)
    elif kind == 1:  # z = const annulus
        q = m.z[i(m.z)]
        b = box((rr[0], a[0], q), (rr[1], a[1], q), prio)
    else:  # α = const fin (also on the seam)
        q = r.choice(m.al[:-1]) if m.closed else m.al[i(m.al)]
        b = box((rr[0], q, zz[0]), (rr[1], q, zz[1]), prio)
    cond = r.choice([1e2, 1e4, 1e6])
    thick = r.choice([1e-5, 1e-4])
    return f'<ConductingSheet Name="{name}" Conductivity="{cond:g}" Thickness="{thick:g}"><Primitives>{b}</Primitives></ConductingSheet>'


def primitive(r, m, prio):
    """A non-box shape in Cartesian or cylindrical coordinates near the seam."""
    rmax = m.rho[-1]
    z0, z1 = m.z_range(r)
    kind = r.randrange(8)
    if kind == 0:  # z cylinder off the axis, across α = 0
        c = r.uniform(0.3, 0.8) * rmax
        rad = r.uniform(0.2, 0.5) * rmax
        return f'<Cylinder Priority="{prio}" Radius="{f(rad)}"><P1 X="{f(c)}" Y="0" Z="{f(z0)}"/><P2 X="{f(c)}" Y="0" Z="{f(z1)}"/></Cylinder>'
    if kind == 1:  # shell off the axis
        c = r.uniform(0.3, 0.8) * rmax
        rad = r.uniform(0.3, 0.5) * rmax
        return f'<CylindricalShell Priority="{prio}" Radius="{f(rad)}" ShellWidth="{f(0.3 * rad)}"><P1 X="{f(c)}" Y="0" Z="{f(z0)}"/><P2 X="{f(c)}" Y="0" Z="{f(z1)}"/></CylindricalShell>'
    if kind == 2:  # sphere
        c = r.uniform(0.3, 0.8) * rmax
        rad = r.uniform(0.25, 0.5) * rmax
        return f'<Sphere Priority="{prio}" Radius="{f(rad)}"><Center X="{f(c)}" Y="{f(r.uniform(-0.2, 0.2))}" Z="{f((z0 + z1) / 2)}"/></Sphere>'
    if kind == 3:  # spherical shell
        c = r.uniform(0.3, 0.8) * rmax
        rad = r.uniform(0.3, 0.5) * rmax
        return f'<SphericalShell Priority="{prio}" Radius="{f(rad)}" ShellWidth="{f(0.3 * rad)}"><Center X="{f(c)}" Y="0" Z="{f((z0 + z1) / 2)}"/></SphericalShell>'
    if kind == 4:  # a z-extruded polygon over the seam (Cartesian vertices)
        c = r.uniform(0.3, 0.8) * rmax
        d = r.uniform(0.2, 0.4) * rmax
        vs = [(c - d, -d), (c + d, -d), (c + d, d), (c - d, d)]
        v = "".join(f'<Vertex X1="{f(x)}" X2="{f(y)}"/>' for x, y in vs)
        return f'<LinPoly Priority="{prio}" Elevation="{f(z0)}" Length="{f(z1 - z0)}" NormDir="2" QtyVertices="4">{v}</LinPoly>'
    if kind == 5:  # a flat polygon
        c = r.uniform(0.3, 0.8) * rmax
        d = r.uniform(0.2, 0.4) * rmax
        vs = [(c - d, -d), (c + d, -d), (c, d)]
        v = "".join(f'<Vertex X1="{f(x)}" X2="{f(y)}"/>' for x, y in vs)
        return f'<Polygon Priority="{prio}" Elevation="{f(m.z[len(m.z) // 2])}" NormDir="2" QtyVertices="3">{v}</Polygon>'
    if kind == 6:  # a box rotated about z (the α range is not read)
        a = m.alpha_range(r)
        rr = m.rho_range(r)
        arg = r.uniform(-1.5, 1.5)
        return (
            f'<Box Priority="{prio}"><P1 X="{f(rr[0])}" Y="{f(a[0])}" Z="{f(z0)}"/><P2 X="{f(rr[1])}" Y="{f(a[1])}" Z="{f(z1)}"/>'
            f'<Transformation><Rotate_Z Argument="{f(arg)}"/></Transformation></Box>'
        )
    # A Cartesian box across α = 0.
    rr = m.rho_range(r)
    d = r.uniform(0.2, 0.5) * rmax
    return (
        f'<Box Priority="{prio}" CoordSystem="0"><P1 X="{f(rr[0])}" Y="{f(-d)}" Z="{f(z0)}"/>'
        f'<P2 X="{f(rr[1])}" Y="{f(d)}" Z="{f(z1)}"/></Box>'
    )


def prim_prop(r, m, name, prio):
    p = primitive(r, m, prio)
    if "<Polygon" not in p and r.random() < 0.35:
        return f'<Metal Name="{name}"><Primitives>{p}</Primitives></Metal>'
    e, k, u = round(r.uniform(1.5, 5), 3), r.choice([0, 0, 0.02]), r.choice([1, 1, 2.0])
    return f'<Material Name="{name}"><Property Epsilon="{e}" Kappa="{k}" Mue="{u}"/><Primitives>{p}</Primitives></Material>'


def model(seed, family):
    r = random.Random(f"{family}:{seed}")
    m = Mesh(r)
    makers = {"metal": [metal], "pole": [pole], "sheet": [sheet], "prim": [prim_prop]}.get(
        family, [metal, pole, sheet, prim_prop]
    )
    props = []
    for q in range(r.randint(1, 3)):
        props.append(r.choice(makers)(r, m, f"p{q}", r.randint(1, 5)))
    # A soft source on interior lines, probes on and next to the seam.
    ri = len(m.rho) // 2
    zi = len(m.z) // 2
    a_s = m.al[len(m.al) // 2]
    src = box((m.rho[ri], a_s, m.z[1]), (m.rho[ri], a_s, m.z[-2]), 0)
    props.append(f'<Excitation Name="src" Type="0" Excite="0,0,1"><Primitives>{src}</Primitives></Excitation>')
    probes = [
        ("e0", 2, m.rho[ri], m.al[0], m.z[zi]),
        ("e1", 2, m.rho[ri], m.al[1], m.z[zi]),
        ("e2", 2, m.rho[-2], m.al[-2], m.z[zi]),
        ("h0", 3, m.rho[ri] + 0.2, m.al[0], m.z[zi] + 0.2),
        ("h1", 3, m.rho[1] + 0.1, m.al[-2] + 0.05, m.z[1] + 0.2),
    ]
    for name, ty, x, y, z in probes:
        b = box((x, y, z), (x, y, z), 0)
        props.append(f'<ProbeBox Name="{name}" Type="{ty}"><Primitives>{b}</Primitives></ProbeBox>')
    bc = 'xmin="PEC" xmax="PEC" ymin="PEC" ymax="PEC" zmin="PEC" zmax="PEC"'
    body = "\n      ".join(props)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- ems random cylindrical sweep ({family}, seed {seed}); scripts/openems-oracle/tools/rsweep_cyl.py -->
<openEMS>
  <FDTD NumberOfTimesteps="250" endCriteria="1e-30" CylinderCoords="1">
    <Excitation Type="0" f0="5e9" fc="5e9"/>
    <BoundaryCond {bc}/>
  </FDTD>
  <ContinuousStructure CoordSystem="1">
    <Properties>
      {body}
    </Properties>
    {m.xml()}
  </ContinuousStructure>
</openEMS>
"""


def one(job):
    """Runs one seed: `(family, seed, debug diffs, failure, {probe: error})`."""
    family, seed, base, gen_only = job
    d = os.path.join(base, f"{family}_{seed}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "model.xml"), "w") as out:
        out.write(model(seed, family))
    if gen_only:
        return family, seed, {}, None, {}
    diffs, failed = dumpcmp.run(d)
    dumpcmp.run_both(d, [])
    probes = {}
    for name, (kind, v) in cmpdir.compare(os.path.join(d, "oe"), os.path.join(d, "em")).items():
        probes[name] = v if kind == "ok" else float("inf")
    return family, seed, diffs, failed, probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mix")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("work")
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    a = ap.parse_args()
    jobs = [
        (fam, s, a.work, a.gen_only)
        for fam in a.family.split(",")
        for s in range(a.first, a.last + 1)
    ]
    dumps = sims = total = 0
    with cf.ThreadPoolExecutor(a.jobs) as pool:
        for fam, seed, diffs, failed, probes in pool.map(one, jobs):
            total += 1
            worst = max(probes.values(), default=0.0)
            dumps += not diffs and not failed
            sims += bool(probes) and worst <= TOL
            if diffs or failed or worst > TOL or not probes:
                print(f"{fam}_{seed}: dump values {dumpcmp.count(diffs)} probes {worst:.3g} {failed or ''}", flush=True)
                dumpcmp.report(diffs)
                for name, err in sorted(probes.items()):
                    if err > TOL:
                        print(f"  {name}: {err:.3g}")
    print(f"{total} models: {dumps} dumps identical, {sims} with all probes within {TOL:g}")


if __name__ == "__main__":
    main()
