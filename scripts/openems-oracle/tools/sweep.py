"""Runs the feature sweep (features.py) in parallel and prints one SUMMARY line per case.

usage: sweep.py WORKDIR OUT.txt EMS_BINARY [ems_only]
openEMS's outputs are cached in WORKDIR/oefresh; ems_only reuses them. J sets
the parallel jobs (default 12), ONLY a comma-separated case list. A model is
exact when its line reads `worst 0.00e+00`.
"""
import os, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

TOOLS = os.path.dirname(os.path.abspath(__file__))
OE = os.path.expanduser("~/.cache/ems-openems/openems")
PY = os.path.expanduser("~/.cache/ems-openems/venv/bin/python")
work, out, ems = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
ems_only = len(sys.argv) > 4 and sys.argv[4] == "ems_only"
cases = sorted(os.listdir(os.path.join(work, "cases")))
if os.environ.get("ONLY"):
    cases = [c for c in cases if c in os.environ["ONLY"].split(",")]


def run(cmd, cwd, extra):
    with open(os.path.join(cwd, "console.txt"), "w") as f:
        try:
            r = subprocess.run(cmd + ["model.xml", "--engine=basic", "--numThreads=1"] + extra, cwd=cwd,
                               stdout=f, stderr=subprocess.STDOUT, timeout=300).returncode
        except subprocess.TimeoutExpired:
            r = "timeout"
    with open(os.path.join(cwd, "console.txt"), "a") as f:
        f.write(f"exit {r}\n")


def job(c):
    src = os.path.join(work, "cases", c)
    o = os.path.join(work, "out_" + os.path.basename(out), c)
    shutil.rmtree(o, ignore_errors=True)
    cache = os.path.join(work, "oefresh", c)
    em = os.path.join(o, "ems")
    shutil.copytree(src, em)
    m = re.search(r"^<!-- args: (.*) -->$", open(os.path.join(src, "model.xml")).read(), re.M)
    extra = m.group(1).split() if m else []
    if not ems_only:
        shutil.rmtree(cache, ignore_errors=True)
        shutil.copytree(src, cache)
        run([OE, "openEMS"], cache, extra)
    run([ems], em, extra)
    r = subprocess.run([PY, "-I", os.path.join(TOOLS, "compare.py"), cache, em, src], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("SUMMARY")]
    shutil.rmtree(o, ignore_errors=True)
    return lines[-1] if lines else f"SUMMARY {c} ERROR {r.stderr[-200:]}"


with ThreadPoolExecutor(int(os.environ.get("J", "12"))) as ex:
    res = list(ex.map(job, cases))
open(out, "w").write("\n".join(res) + "\n")
print(sum("worst 0.00e+00" in r for r in res), "of", len(res), "exact")
