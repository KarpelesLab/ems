"""Writes the synthetic raw SAR dump of the sar_calc_block fixture.

Usage: python sar-synth.py OUT.h5

A 12 x 12 x 10 mesh of 1 mm cells holding a 10 x 10 x 8 tissue block
(density 1000 kg/m^3, pseudo-random conductivity) in air, with a smooth
pseudo-random cell E field at two frequencies (fixed seed), in the layout
of openEMS raw SAR dumps (DumpType 29). Hand-written for the ems sar_calc
fixtures (gen-sar.sh); it writes only the input, the outputs come from the
black-box sar_calc tool.
"""
import sys

import h5py
import numpy as np

rng = np.random.default_rng(62704)
n = (12, 10 + 2, 8 + 2)
w = 1e-3
shape = n
rho = np.zeros(shape)
rho[1:-1, 1:-1, 1:-1] = 1000.0
sigma = np.where(rho > 0, rng.uniform(0.3, 1.2, shape), 0.0)
vol = np.full(shape, w ** 3)
x, y, z = (np.arange(k) * w + w / 2 for k in n)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
fields = []
for f in (1e9, 2e9):
    e = np.zeros((3,) + shape, complex)
    for c in range(3):
        a = rng.normal(size=4) + 1j * rng.normal(size=4)
        e[c] = (a[0] + a[1] * np.cos(400 * X + 1.0) + a[2] * np.sin(300 * Y) + a[3] * np.cos(500 * Z + 2 * X * 100))
    fields.append(e)

with h5py.File(sys.argv[1], 'w') as f:
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
    for name, c in zip('xyz', (x, y, z)):
        cw.create_dataset(name, data=np.full(len(c), w))
        m.create_dataset(name, data=c)
    g = f.create_group('FieldData/FD')
    g.attrs['frequency'] = np.array([1e9, 2e9])
    for k, e in enumerate(fields):
        d = g.create_dataset('f%d' % k, data=e.astype(np.complex64))
        d.attrs['d_order'] = 'NXYZ'
