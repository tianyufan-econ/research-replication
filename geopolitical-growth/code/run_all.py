#!/usr/bin/env python3
"""
run_all.py
==========
Run every script of the short replication package in order and collect the
reported numbers into output/numbers.csv.

Usage (from code/):
    python3 run_all.py              # all scripts
    python3 run_all.py 01 07        # only the scripts whose names start with these prefixes

Each script runs in its own interpreter with a fixed string-hash seed, single-threaded
BLAS and a fixed file date, so a rerun that changes nothing writes identical files.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent

SCRIPTS = [
    ("01_baseline.py", "Figure 6"),
    ("02_sustained.py", "Figure 7 and the sustained-path intervals"),
    ("03_partners.py", "Figure 8"),
    ("04_components.py", "Figure 9"),
    ("05_robustness.py", "Figure 10"),
    ("06_placebo.py", "Figure 11"),
    ("07_transformations_iv.py", "Figure 12"),
    ("08_channels.py", "Figure 13"),
    ("09_democracy.py", "Figures 14 and 15"),
    ("10_accounting.py", "Figures 16 and 17, Table 4"),
]

ENV = dict(os.environ, PYTHONHASHSEED="0", MPLBACKEND="Agg", SOURCE_DATE_EPOCH="946684800",
           PYTHONDONTWRITEBYTECODE="1",
           **{v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})


def main():
    prefixes = sys.argv[1:]
    todo = [s for s in SCRIPTS if not prefixes or s[0].startswith(tuple(prefixes))]
    t_all, failed = time.time(), []
    for script, what in todo:
        print(f"\n=== {script}: {what} ===", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, str(CODE / script)], cwd=CODE, env=ENV)
        print(f"--- {script}: {'ok' if rc == 0 else f'FAILED ({rc})'} in {time.time() - t0:.0f}s", flush=True)
        if rc:
            failed.append(script)

    import pandas as pd
    files = sorted((CODE.parent / "output" / "numbers").glob("*.csv"))
    if files:
        pd.concat([pd.read_csv(f) for f in files]).to_csv(CODE.parent / "output" / "numbers.csv", index=False)
        print(f"\nCollected {len(files)} number files into output/numbers.csv")
    print(f"=== done in {(time.time() - t_all) / 60:.1f} min" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
