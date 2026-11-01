"""usage: compare.py OE_DIR EMS_DIR CASE_DIR  compares two runs of a sweep case; last line SUMMARY <case> worst <d>"""
import sys, os, re
import numpy as np
oe, em, case = sys.argv[1], sys.argv[2], sys.argv[3]
src = set(os.listdir(case)) | {'console.txt'}


def load(p):
    rows = []
    for l in open(p):
        if l.startswith('%') or not l.strip():
            continue
        rows.append([float(x) for x in l.split()])
    return np.array(rows)


def lines(p):
    return [l.rstrip() for l in open(p, errors='replace')]


def grepw(p):
    return [l for l in lines(p) if re.search(r'(?i)warn|error|ignor|unsupport|not supported|skip|unknown', l)]


co, ce = os.path.join(oe, 'console.txt'), os.path.join(em, 'console.txt')
print('  oe', [l for l in lines(co) if l.startswith('exit')][-1], '| ems', [l for l in lines(ce) if l.startswith('exit')][-1])
for l in grepw(co)[:10]:
    print('  oe>', l[:220])
for l in grepw(ce)[:10]:
    print('  ems>', l[:220])
for tag, p in (('oe ', co), ('ems', ce)):
    for l in lines(p):
        if 'timestep is' in l or 'simulation size' in l:
            print('  ', tag, l.strip()[:150])
fo = sorted(f for f in os.listdir(oe) if f not in src and os.path.isfile(os.path.join(oe, f)))
fe = sorted(f for f in os.listdir(em) if f not in src and os.path.isfile(os.path.join(em, f)))
only_o = [f for f in fo if f not in fe]
only_e = [f for f in fe if f not in fo]
if only_o:
    print('  only openEMS:', only_o[:20])
if only_e:
    print('  only ems:', only_e[:20])


def h5cmp(a, b, name):
    import h5py
    fa, fb = h5py.File(a, 'r'), h5py.File(b, 'r')
    ds = []
    fa.visititems(lambda n, o: ds.append(n) if isinstance(o, h5py.Dataset) else None)
    worst = (0, '')
    for n in ds:
        if n not in fb:
            print(f'  {name}:{n} missing in ems')
            continue
        x, y = fa[n][()], fb[n][()]
        if x.dtype.names:
            x = x['r'] + 1j * x['i']
        if y.dtype.names:
            y = y['r'] + 1j * y['i']
        if x.shape != y.shape:
            print(f'  {name}:{n} shape {x.shape} vs {y.shape}')
            continue
        d = np.linalg.norm((x - y).ravel()) / max(np.linalg.norm(x.ravel()), 1e-300)
        if d > worst[0]:
            worst = (d, n)
    dsb = []
    fb.visititems(lambda n, o: dsb.append(n) if isinstance(o, h5py.Dataset) else None)
    print(f'  {name}: h5 {len(ds)} vs {len(dsb)} datasets, worst rel {worst[0]:.2e} {worst[1]}')


worst = []
for f in fo:
    if f not in fe:
        continue
    a, b = os.path.join(oe, f), os.path.join(em, f)
    if f.endswith('.h5'):
        try:
            h5cmp(a, b, f)
        except Exception as e:
            print('  ', f, 'h5 err', e)
        continue
    if f.endswith(('.vtr', '.vtk', '.vts', '.vtp', '.txt', '.xml')):
        continue
    try:
        x, y = load(a), load(b)
    except Exception as e:
        print('  ', f, 'unparsed', e)
        continue
    if x.shape[0] == 0:
        print(f'  {f}: empty oe ({y.shape})')
        continue
    if y.shape[0] == 0:
        print(f'  {f}: empty ems')
        continue
    if f in ('et', 'ht'):
        n = min(len(x), len(y))
        d = np.linalg.norm(x[:n, 1] - y[:n, 1]) / max(np.linalg.norm(x[:n, 1]), 1e-300)
        print(f'  {f}: rows {len(x)} vs {len(y)}, rel {d:.2e}')
        if len(x) != len(y) or d > 1e-5:
            worst.append((max(d, 1e-5 if len(x) != len(y) else 0), f))
        continue
    n = min(len(x), len(y))
    tdiff = np.max(np.abs(x[:n, 0] - y[:n, 0])) / max(np.max(np.abs(x[:n, 0])), 1e-300)
    cols = []
    for k in range(1, x.shape[1]):
        if k >= y.shape[1]:
            cols.append('missing')
            continue
        nx = np.linalg.norm(x[:n, k])
        d = np.linalg.norm(x[:n, k] - y[:n, k]) / max(nx, 1e-300)
        cols.append(f'{d:.2e}' + ('(0)' if nx == 0 else ''))
    fro = np.linalg.norm(x[:n, 1:] - y[:n, 1:]) / max(np.linalg.norm(x[:n, 1:]), 1e-300)
    if np.linalg.norm(x[:n, 1:]) == 0 and np.linalg.norm(y[:n, 1:]) == 0:
        fro = 0.0
    worst.append((fro, f))
    hdr_o = [l for l in lines(a) if l.startswith('% start') or l.startswith('% stop')]
    hdr_e = [l for l in lines(b) if l.startswith('% start') or l.startswith('% stop')]
    hd = '' if hdr_o == hdr_e else ' HDR-DIFF'
    print(f'  {f}: rows {len(x)}/{len(y)} cols {x.shape[1]}/{y.shape[1]} t {tdiff:.0e} | ' + ' '.join(cols) + f' | max {np.max(np.abs(x[:, 1:])):.2e}{hd}')
    if hd and os.environ.get('HDR'):
        print('     ', hdr_o, hdr_e)

ex_o = [l for l in lines(co) if l.startswith('exit')][-1]
ex_e = [l for l in lines(ce) if l.startswith('exit')][-1]
w = max(worst) if worst else (0.0, '-')
flags = []
if ex_o != ex_e: flags.append(f'exit oe:{ex_o[5:]} ems:{ex_e[5:]}')
if only_o: flags.append('only-oe:' + ','.join(only_o[:4]))
if only_e: flags.append('only-ems:' + ','.join(only_e[:4]))
print(f'SUMMARY {os.path.basename(case.rstrip("/")):28s} worst {w[0]:.2e} ({w[1]}) {" ".join(flags)}')
