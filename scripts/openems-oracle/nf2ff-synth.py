"""Writes the synthetic inputs of the nf2ff_synth fixture (gen-nf2ff.sh).

Usage: python nf2ff-synth.py DIR

Small near-field planes with pseudo-random fields (fixed seed) in the
openEMS dump layouts nf2ff reads: frequency-domain (current and legacy
layout) and time-domain dumps, plus the nf2ff job files (jobs/*.xml) that
exercise the XML options (radius, centre, materials, mirrors, several
planes, several frequencies). Hand-written for the ems nf2ff fixtures; it
writes only inputs, the outputs come from the black-box nf2ff tool.
"""
import os
import sys

import h5py
import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20261020)
FREQS = [1e9, 1.5e9]


def header(f, x, y, z, dump_type, legacy=False):
    f.attrs['openEMS_HDF5_version'] = 0.3
    f.attrs['dump_type'] = np.int32(dump_type)
    f.attrs['dual_mesh'] = np.uint8(0)
    if legacy:
        f.attrs['legacy_fmt'] = np.uint8(1)
    m = f.create_group('Mesh')
    m.attrs['mesh_scaling'] = 0.001
    m.attrs['mesh_type'] = np.int32(0)
    for n, a in zip('xyz', (x, y, z)):
        m.create_dataset(n, data=np.asarray(a, float))


def fd(path, x, y, z, fields, dump_type, legacy=False):
    with h5py.File(path, 'w') as f:
        header(f, x, y, z, dump_type, legacy)
        g = f.create_group('FieldData/FD')
        g.attrs['frequency'] = np.asarray(FREQS, float)
        for k, F in enumerate(fields):
            if legacy:
                a = np.transpose(F, (0, 3, 2, 1))
                g.create_dataset('f%d_real' % k, data=a.real.astype(np.float32))
                g.create_dataset('f%d_imag' % k, data=a.imag.astype(np.float32))
            else:
                d = g.create_dataset('f%d' % k, data=F.astype(np.complex64))
                d.attrs['d_order'] = 'NXYZ'
                d.attrs['frequency'] = FREQS[k]


def td(path, x, y, z, snaps, times, dump_type):
    with h5py.File(path, 'w') as f:
        header(f, x, y, z, dump_type)
        g = f.create_group('FieldData/TD')
        for n, (s, t) in enumerate(zip(snaps, times)):
            d = g.create_dataset('%08d' % (5 * n), data=s.astype(np.float32))
            d.attrs['d_order'] = 'NXYZ'
            d.attrs['time'] = np.array([t], np.float32)


def rnd(shape, scale=1.0):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * scale


planes = {
    # name: (x, y, z) lines [m]; one line along the normal.
    'zp': (np.array([-0.03, -0.01, 0.0, 0.02, 0.035]), np.array([-0.02, 0.0, 0.025]), np.array([0.03])),
    'xn': (np.array([-0.04]), np.array([-0.02, 0.01, 0.025]), np.array([-0.015, 0.0, 0.01, 0.03])),
    'yp': (np.array([0.0, 0.02, 0.05]), np.array([0.02]), np.array([0.005, 0.015, 0.03])),
}
for name, (x, y, z) in planes.items():
    shape = (3, len(x), len(y), len(z))
    fd(os.path.join(out, 'E_%s.h5' % name), x, y, z, [rnd(shape), rnd(shape)], 10)
    fd(os.path.join(out, 'H_%s.h5' % name), x, y, z, [rnd(shape, 1 / 377), rnd(shape, 1 / 377)], 11)
# Legacy layout copy of one plane.
x, y, z = planes['xn']
shape = (3, len(x), len(y), len(z))
fd(os.path.join(out, 'E_leg.h5'), x, y, z, [rnd(shape), rnd(shape)], 10, legacy=True)
fd(os.path.join(out, 'H_leg.h5'), x, y, z, [rnd(shape, 1 / 377), rnd(shape, 1 / 377)], 11, legacy=True)
# Time-domain plane: two random spatial patterns in quadrature at 1 GHz.
x, y, z = planes['zp']
shape = (3, len(x), len(y), len(z))
ea, eb, ha, hb = (rng.normal(size=shape) for _ in range(4))
dt = 5e-11
tE = np.arange(20) * dt
tH = tE + 1e-11
w = 2 * np.pi * 1e9
td(os.path.join(out, 'E_td.h5'), x, y, z, [ea * np.cos(w * t) + eb * np.sin(w * t) for t in tE], tE, 0)
td(os.path.join(out, 'H_td.h5'), x, y, z, [(ha * np.cos(w * t) + hb * np.sin(w * t)) / 377 for t in tH], tH, 1)

th = ','.join('%.4f' % t for t in np.linspace(0.05, 3.1, 9))
ph = ','.join('%.4f' % p for p in np.linspace(0.0, 5.5, 6))
jobs = {
    'basic': ('freq="1e9,1.5e9"', ['zp', 'xn', 'yp'], ''),
    'options': ('freq="1e9,1.5e9" Radius="2.5" Center="0.005,-0.01,0.02" Eps_r="2.2,3" Mue_r="1.3"',
                ['zp', 'xn'], ''),
    'mirrors': ('freq="1e9"', ['zp', 'yp'],
                '<Mirror Dir="0" Type="PEC" Pos="-0.05"/>\n  <Mirror Dir="2" Type="PMC" Pos="-0.02"/>'),
    # Legacy FD layout and a time-domain plane (DFT of the samples).
    'layouts': ('freq="1e9"', ['leg', 'td'], '<Mirror Dir="1" Type="PEC" Pos="-0.03"/>'),
}
os.makedirs(os.path.join(out, 'jobs'), exist_ok=True)
for job, (attrs, pl, extra) in jobs.items():
    planes_xml = '\n'.join('  <Planes E_Field="E_%s.h5" H_Field="H_%s.h5"/>' % (p, p) for p in pl)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<nf2ff %s Outfile="out_%s.h5">\n  <theta>%s</theta>\n  <phi>%s</phi>\n%s\n  %s\n</nf2ff>\n'
           % (attrs, job, th, ph, planes_xml, extra))
    with open(os.path.join(out, 'jobs', job + '.xml'), 'w') as f:
        f.write(xml)
