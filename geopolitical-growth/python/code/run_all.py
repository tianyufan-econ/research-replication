#!/usr/bin/env python3
"""
run_all.py
==========
Master runner for the Geopolitical Growth replication package.

Executes scripts 01-10 sequentially, printing timing for each script
and a total elapsed time at the end. Errors in any single script are
caught and reported, and execution continues to the next script.

IMPORTANT: Scripts 02 (transitory) and 06 (placebo) use bootstrap
with 500 iterations each and are SLOW (~15 min each). Total runtime
for the full package is approximately 8-10 minutes on Apple M5 Pro.

Usage:
  cd replication/python/code
  python run_all.py
"""

import sys
import time
import importlib
import traceback
from pathlib import Path

# Ensure the code directory is on sys.path so imports work
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

# ── Script registry (in execution order) ─────────────────────────────
SCRIPTS = [
    ("01_baseline",         "Baseline LP-IRFs (Fig 6a-b)"),
    ("02_transitory",       "Transitory & Permanent IRFs (Fig 7a-b) [SLOW: ~15 min]"),
    ("03_decomposition",    "Component Decomposition (Fig 8a-b)"),
    ("04_robustness",       "Robustness Checks (Fig 11a-b)"),
    ("05_symmetry",         "Partner & Temporal Symmetry (Fig 10a-b)"),
    ("06_placebo",          "Placebo Randomization Tests (Fig 12a-b) [SLOW: ~15 min]"),
    ("07_iv_verbal",        "IV: Verbal Conflicts (Fig 13a)"),
    ("08_iv_leadership",    "IV: Leadership Changes (Fig 13b)"),
    ("09_channels",         "Channel Variables (Fig 14a-b)"),
    ("10_growth_accounting", "Growth Accounting (Fig 17-18)"),
]


def run_script(module_name, description):
    """Import and run a script's main() function. Returns elapsed seconds."""
    print(f"\n{'=' * 72}")
    print(f"  {module_name}: {description}")
    print(f"{'=' * 72}")

    t0 = time.time()
    try:
        # Import (or re-import) the module
        if module_name in sys.modules:
            mod = importlib.reload(sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)

        # Call main()
        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"  WARNING: {module_name} has no main() function.")
    except SystemExit as e:
        # Some scripts call sys.exit on missing data
        print(f"  Script exited: {e}")
    except Exception:
        print(f"\n  ERROR in {module_name}:")
        traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n  [{module_name}] finished in {elapsed:.1f}s")
    return elapsed


def main():
    print("=" * 72)
    print("  GEOPOLITICAL GROWTH -- REPLICATION PACKAGE")
    print("  Running all scripts (01-10)")
    print("=" * 72)
    print()
    print("WARNING: Bootstrap scripts (02_transitory, 06_placebo) are slow.")
    print("         Total estimated runtime: ~8-10 minutes on Apple M5 Pro.")
    print()

    total_t0 = time.time()
    timings = []

    for module_name, description in SCRIPTS:
        elapsed = run_script(module_name, description)
        timings.append((module_name, elapsed))

    total_elapsed = time.time() - total_t0

    # Summary
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for name, t in timings:
        status = "OK" if t < 1800 else "SLOW"
        print(f"  {name:25s}  {t:8.1f}s  ({status})")
    print(f"  {'TOTAL':25s}  {total_elapsed:8.1f}s  ({total_elapsed/60:.1f} min)")
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
