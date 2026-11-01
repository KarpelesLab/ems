"""usage: both.py DIR [extra args...]  runs openEMS in DIR/oe and ems in DIR/em on DIR/model.xml

The ems binary is $EMS (default target/release/ems, made absolute).
"""
import os, shutil, subprocess, sys
EMS = os.environ.get("EMS", "target/release/ems")
OE = os.path.expanduser("~/.cache/ems-openems/openems")


def run(d, args, which=("oe", "em"), model="model.xml"):
    for w in which:
        p = os.path.join(d, w)
        shutil.rmtree(p, ignore_errors=True)
        os.makedirs(p)
        shutil.copy(os.path.join(d, model), os.path.join(p, "model.xml"))
        cmd = ([OE, "openEMS", "model.xml", "--engine=basic", "--numThreads=1"] if w == "oe"
               else [os.path.abspath(EMS), "model.xml"]) + list(args)
        with open(os.path.join(p, "console.txt"), "w") as f:
            subprocess.run(cmd, cwd=p, stdout=f, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:])
