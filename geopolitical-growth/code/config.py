"""
config.py
=========
Paths, sample restriction and reporting scale for the short replication package.

Every script reads one file, data/panel.csv, built by the full package
(code/11_tables_numbers/build_short_panel.py). The environment variable
SHORT_PANEL points the scripts at another copy of the panel.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]            # short_replication/
CODE = ROOT / "code"
PANEL = Path(os.environ.get("SHORT_PANEL", ROOT / "data" / "panel.csv"))
OUT = ROOT / "output"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
CACHE = OUT / "cache"
NUMBERS = OUT / "numbers"


def ensure_dirs():
    for d in (FIGURES, TABLES, CACHE, NUMBERS):
        d.mkdir(parents=True, exist_ok=True)


# ── Sample and reporting scale ────────────────────────────────────────
MAJOR_NATIONS = [
    "ARG", "AUS", "BEL", "BRA", "CAN", "CHE", "CHN", "DEU", "DNK", "ESP",
    "FRA", "GBR", "IDN", "IND", "ITA", "JPN", "KOR", "MEX", "NLD", "POL",
    "RUS", "SAU", "TUR", "USA",
]

MIN_YEXT_YEARS = 40  # long-coverage sample: countries with >= 40 years of y_ext

# Coefficients on the alignment index are reported per one standard deviation of
# the index (its SD over all 12,545 country-years). Estimation stays in index units.
GEO_SD = 0.14069170857195315


def restrict_balanced_sample(df, y_var="y_ext", min_years=MIN_YEXT_YEARS):
    """Restrict the panel to countries with >= min_years of non-missing y_var."""
    yr_counts = df.dropna(subset=[y_var]).groupby("country_code")["year"].nunique()
    keep = yr_counts[yr_counts >= min_years].index
    n_before = df["country_code"].nunique()
    out = df[df["country_code"].isin(keep)].copy()
    print(f"  Long-coverage sample: {len(keep)}/{n_before} countries "
          f"(>={min_years}yr {y_var})")
    return out


def read_csv_fallback(path, **kwargs):
    """Read a CSV as utf-8, falling back to latin-1.

    The panel is parsed exactly (float_precision="round_trip"): panel.csv stores every
    value with all its digits, so each value equals the one the full package uses.
    """
    import pandas as pd
    if Path(path) == PANEL:
        kwargs.setdefault("float_precision", "round_trip")
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="latin-1", encoding_errors="replace", **kwargs)


# ── Output helpers ────────────────────────────────────────────────────
def savefig(fig, filename, **kwargs):
    """Save a figure to output/figures/ (the file names follow the paper's figure numbers)."""
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("bbox_inches", "tight")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, **kwargs)
    print(f"  Saved: output/figures/{filename}")


def save_numbers(script, rows):
    """Write the numbers a script reports to output/numbers/<script>.csv.

    rows: dicts with key, value, paper (where the number appears in the paper)
    and full_ref (<full-package script>:<key> of the same number in the full
    package's output/numbers/). run_all.py collects them into output/numbers.csv.
    """
    import pandas as pd
    NUMBERS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ["key", "value", "paper", "full_ref"]:
        if col not in df:
            df[col] = ""
    df.insert(0, "script", Path(script).name)
    out = NUMBERS / f"{Path(script).stem}.csv"
    df[["script", "key", "value", "paper", "full_ref"]].to_csv(out, index=False)
    print(f"  Numbers: output/numbers/{out.name} ({len(df)} rows)")
