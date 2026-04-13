"""
config.py
=========
Simplified configuration for the Geopolitical Growth replication package.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # replication/python/
DATA = ROOT.parent / "data"                            # replication/data/
PANEL = DATA / "panel.csv"
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
CACHE = OUTPUT / "cache"


def ensure_dirs():
    """Create output directories if they do not exist."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)


# ── LP defaults ──────────────────────────────────────────────────────────
LP_DEFAULTS = dict(
    y_var="y_ext",
    shock_var="geo_relation_dyn",
    y_lags=4,
    shock_lags=4,
    horizon_start=-10,
    horizon_end=25,
    confidence_level=0.95,
)

# ── 24 major nations (GDP-weighting partners) ────────────────────────────
MAJOR_NATIONS = [
    "ARG", "AUS", "BEL", "BRA", "CAN", "CHE", "CHN", "DEU",
    "DNK", "ESP", "FRA", "GBR", "IDN", "IND", "ITA", "JPN",
    "KOR", "MEX", "NLD", "POL", "RUS", "SAU", "TUR", "USA",
]

# ── Sample restriction ───────────────────────────────────────────────────
MIN_YEXT_YEARS = 40


def restrict_balanced_sample(df, y_var="y_ext", min_years=40):
    """Keep countries with >= min_years non-null y_var observations."""
    counts = df.groupby("country_code")[y_var].apply(lambda s: s.notna().sum())
    keep = counts[counts >= min_years].index
    return df[df["country_code"].isin(keep)].copy()


# ── I/O helpers ──────────────────────────────────────────────────────────
def savefig(fig, directory, filename):
    """Save figure to *directory/filename* (always saves, no PAPER_ONLY filter)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {directory / filename}")


def read_csv_fallback(path):
    """Read CSV trying utf-8 first, then latin-1."""
    import pandas as pd
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")
