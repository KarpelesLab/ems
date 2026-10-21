"""Writes the synthetic raw SAR dumps of the sar_calc fixtures.

Usage: python sar-synth.py [CASE] OUT.h5

Raw SAR dumps in the layout openEMS writes them (DumpType 29), with fixed
seeds. Hand-written for the ems sar_calc fixtures (gen-sar.sh); they are
only the inputs, the outputs come from the black-box sar_calc tool.

CASE (default block):

* block: a 12 x 12 x 10 mesh of 1 mm cells holding a 10 x 10 x 8 tissue
  block (density 1000 kg/m^3, pseudo-random conductivity) in air, with a
  smooth pseudo-random cell E field at two frequencies (sar_calc_block).
* blob: a graded 14 x 12 x 10 mesh (cell widths 0.7 to 1.3 mm), an
  ellipsoid of tissue with pseudo-random density (600 to 1400 kg/m^3) and
  conductivity, one frequency (sar_calc_synth).
* slab: a 3-cell tissue slab (1 mm cells, pseudo-random density) in air:
  no valid IEEE cube, every voxel takes a face cube (sar_calc_synth).
* aniso: uniform tissue of 1 x 1 x 3 mm cells next to an air layer, for
  cubes smaller than two of the widest cell widths (sar_calc_synth).
* tiny: 1 mm cells, pseudo-random density everywhere but one air layer,
  for cubes smaller than a cell (sar_calc_edge).
* graded: a graded 10 x 9 x 8 mesh (cell widths 0.4 to 1.6 mm) with an
  ellipsoid of tissue of pseudo-random density in air: IEEE validity of
  cubes of about two cells, face cubes (sar_calc_edge).
* full: the graded mesh of `graded` filled with tissue of pseudo-random
  density: face cubes of the domain boundary shorter than their voxel
  (sar_calc_edge).

Every density is pseudo-random, so no two cubes tie: openEMS breaks ties
by its rounding.
"""
import sys

import h5py
import numpy as np


def field(rng, n, centres, nf):
    X, Y, Z = np.meshgrid(*centres, indexing='ij')
    out = []
    for _ in range(nf):
        e = np.zeros((3,) + tuple(n), complex)
        for c in range(3):
            a = rng.normal(size=4) + 1j * rng.normal(size=4)
            e[c] = (a[0] + a[1] * np.cos(400 * X + 1.0) + a[2] * np.sin(300 * Y) + a[3] * np.cos(500 * Z + 2 * X * 100))
        out.append(e)
    return out


def block():
    rng = np.random.default_rng(62704)
    n = (12, 10 + 2, 8 + 2)
    w = 1e-3
    rho = np.zeros(n)
    rho[1:-1, 1:-1, 1:-1] = 1000.0
    sigma = np.where(rho > 0, rng.uniform(0.3, 1.2, n), 0.0)
    widths = [np.full(k, w) for k in n]
    centres = [np.arange(k) * w + w / 2 for k in n]
    return widths, centres, rho, sigma, (1e9, 2e9), field(rng, n, centres, 2)


def blob():
    rng = np.random.default_rng(1528)
    n = (14, 12, 10)
    widths = [1e-3 * (1 + 0.3 * rng.uniform(-1, 1, k)) for k in n]
    centres = [np.cumsum(w) - w / 2 for w in widths]
    X, Y, Z = np.meshgrid(*[np.arange(k) for k in n], indexing='ij')
    c = [k / 2 + rng.uniform(-1, 1) for k in n]
    r = ((X - c[0]) / (n[0] * 0.38)) ** 2 + ((Y - c[1]) / (n[1] * 0.4)) ** 2 + ((Z - c[2]) / (n[2] * 0.42)) ** 2
    tissue = r < 1
    rho = np.where(tissue, 1000 * (1 + 0.4 * rng.uniform(-1, 1, n)), 0.0)
    sigma = np.where(tissue, rng.uniform(0.3, 1.5, n), 0.0)
    return widths, centres, rho, sigma, (1e9,), field(rng, n, centres, 1)


def slab():
    rng = np.random.default_rng(953)
    n = (9, 10, 8)
    widths = [np.full(k, 1e-3) for k in n]
    centres = [np.cumsum(w) - w / 2 for w in widths]
    X, Y, Z = np.meshgrid(*[np.arange(k) for k in n], indexing='ij')
    tissue = (X >= 3) & (X < 6) & (Y >= 1) & (Y < 9) & (Z >= 1) & (Z < 7)
    rho = np.where(tissue, 1000 * (1 + 0.3 * rng.uniform(-1, 1, n)), 0.0)
    sigma = np.where(tissue, rng.uniform(0.3, 1.5, n), 0.0)
    return widths, centres, rho, sigma, (1e9,), field(rng, n, centres, 1)


def aniso():
    rng = np.random.default_rng(1529)
    n = (12, 10, 6)
    widths = [np.full(n[0], 1e-3), np.full(n[1], 1e-3), np.full(n[2], 3e-3)]
    centres = [np.cumsum(w) - w / 2 for w in widths]
    rho = np.full(n, 1000.0)
    rho[:2] = 0.0
    sigma = np.where(rho > 0, rng.uniform(0.3, 1.5, n), 0.0)
    return widths, centres, rho, sigma, (1e9,), field(rng, n, centres, 1)


def tiny():
    rng = np.random.default_rng(857)
    n = (7, 6, 6)
    widths = [np.full(k, 1e-3) for k in n]
    centres = [np.cumsum(w) - w / 2 for w in widths]
    rho = 1000 * (1 + 0.3 * rng.uniform(-1, 1, n))
    rho[:, :, -1] = 0.0
    sigma = np.where(rho > 0, rng.uniform(0.3, 1.5, n), 0.0)
    return widths, centres, rho, sigma, (1e9,), field(rng, n, centres, 1)


def graded_mesh(rng):
    n = (10, 9, 8)
    widths = [1e-3 * (1 + 0.6 * rng.uniform(-1, 1, k)) for k in n]
    return n, widths, [np.cumsum(w) - w / 2 for w in widths]


def graded():
    rng = np.random.default_rng(2)
    n, widths, centres = graded_mesh(rng)
    X, Y, Z = np.meshgrid(*[np.arange(k) for k in n], indexing='ij')
    c = [k / 2 + rng.uniform(-1, 1) for k in n]
    r = ((X - c[0]) / (n[0] * 0.4)) ** 2 + ((Y - c[1]) / (n[1] * 0.42)) ** 2 + ((Z - c[2]) / (n[2] * 0.45)) ** 2
    rho = np.where(r < 1, 1000 * (1 + 0.4 * rng.uniform(-1, 1, n)), 0.0)
    sigma = np.where(rho > 0, rng.uniform(0.3, 1.5, n), 0.0)
    return widths, centres, rho, sigma, (1e9,), field(rng, n, centres, 1)


def full():
    rng = np.random.default_rng(3)
    n, widths, centres = graded_mesh(rng)
    rho = 1000 * (1 + 0.4 * rng.uniform(-1, 1, n))
    sigma = rng.uniform(0.3, 1.5, n)
    return widths, centres, rho, sigma, (1e9,), field(rng, n, centres, 1)


def main():
    case, out = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('block', sys.argv[1])
    cases = {'block': block, 'blob': blob, 'slab': slab, 'aniso': aniso, 'tiny': tiny, 'graded': graded, 'full': full}
    widths, centres, rho, sigma, freqs, fields = cases[case]()
    vol = np.einsum('i,j,k->ijk', *widths)
    with h5py.File(out, 'w') as f:
        f.attrs['dual_mesh'] = np.uint8(1)
        f.attrs['dump_type'] = np.int32(29)
        f.attrs['openEMS_HDF5_version'] = 0.3
        cd = f.create_group('CellData')
        for name, a in (('Conductivity', sigma), ('Density', rho), ('Volume', vol)):
            d = cd.create_dataset(name, data=a.astype(np.float32))
            d.attrs['d_order'] = 'XYZ'
        cw = f.create_group('CellWidth')
        cw.attrs['mesh_scaling'] = 1.0
        cw.attrs['mesh_type'] = np.int32(0)
        m = f.create_group('Mesh')
        m.attrs['mesh_scaling'] = 0.001
        m.attrs['mesh_type'] = np.int32(0)
        for name, w, c in zip('xyz', widths, centres):
            cw.create_dataset(name, data=w)
            m.create_dataset(name, data=c)
        g = f.create_group('FieldData/FD')
        g.attrs['frequency'] = np.array(freqs)
        for k, e in enumerate(fields):
            d = g.create_dataset('f%d' % k, data=e.astype(np.complex64))
            d.attrs['d_order'] = 'NXYZ'


main()
