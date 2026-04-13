"""
09_channels.py
==============
Fig 14a: Channel variables panel A (2x2 grid)
  - yPWT (Real GDP per Capita, PWT)
  - unrest_new (Domestic Unrest)
  - csh_i (Investment Share, IRF x100 for pp)
  - log_k (Log Capital Stock, computed from rnna)

Fig 14b: Channel variables panel B (2x2 grid)
  - irr (Internal Rate of Return, IRF x100 for pp)
  - log_tfp (Log TFP, computed from rtfpna)
  - trade (Trade Share, IRF x100 for pp)
  - log_hc (Human Capital, computed from hc)

Each variable: LP with geo_relation_dyn as shock, 4 lags of dep var +
4 lags of geo, country + region-year FE, DK SE, h = -10 .. 25.
Per-variable balanced sample (drop countries missing that variable).
Show N countries in panel title.

Source: 03_extensions/lp_covariates_reg.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS

from config import PANEL, FIGURES, CACHE, ensure_dirs, savefig, \
    read_csv_fallback, restrict_balanced_sample
import plot_style as PS
PS.apply_theme()
ensure_dirs()

# ── Configuration ────────────────────────────────────────────────────
Y_VAR     = "y_ext"
SHOCK     = "geo_relation_dyn"
DEP_LAGS  = 4
GEO_LAGS  = 4
YEXT_LAGS = 4
H_NEG     = 10
H_POS     = 25
LP_RANGE  = range(-H_NEG, H_POS + 1)
MIN_COUNTRY_OBS = 1

# Channel definitions: (column, title, needs_transform)
CHANNELS_A = [
    ("yPWT",       "Real GDP per Capita (PWT)",  False),
    ("unrest_new", "Domestic Unrest",             False),
    ("csh_i",      "Investment Share of GDP",     False),
    ("log_k",      "Log Capital Stock",           True),
]
CHANNELS_B = [
    ("irr",        "Internal Rate of Return",     False),
    ("log_tfp",    "Log TFP",                     True),
    ("trade",      "Trade Share of GDP",          False),
    ("log_hc",     "Human Capital Index",         True),
]

# Scale factors: multiply IRF/SE by 100 for share-of-GDP variables
SCALE_FACTORS = {"csh_i": 100, "trade": 100, "irr": 100}

# Channel-specific y-axis labels
CHANNEL_YLABELS = {
    "yPWT":       "Real GDP per Capita, PWT (\u00d7100)",
    "unrest_new": "Domestic Unrest Index",
    "csh_i":      "Investment Share (pp)",
    "log_k":      "Capital Stock (\u00d7100)",
    "irr":        "Internal Rate of Return (pp)",
    "log_tfp":    "TFP (\u00d7100)",
    "trade":      "Trade Openness (pp)",
    "log_hc":     "Human Capital Index (\u00d7100)",
}


# ── Data helpers ─────────────────────────────────────────────────────

def add_transforms(df):
    """Generate log_k, log_tfp, log_hc from PWT variables."""
    df = df.copy()
    if "log_k" not in df.columns and "rnna" in df.columns:
        df["log_k"] = np.log(df["rnna"].replace(0, np.nan)) * 100
    if "log_tfp" not in df.columns and "rtfpna" in df.columns:
        df["log_tfp"] = np.log(df["rtfpna"].replace(0, np.nan)) * 100
    if "log_hc" not in df.columns and "hc" in df.columns:
        df["log_hc"] = np.log(df["hc"].replace(0, np.nan)) * 100
    # Scale trade from percentage to decimal
    if "trade" in df.columns:
        df["trade"] = df["trade"] / 100
    return df


def load_and_prepare():
    """Load panel, add transforms, lags, FE columns."""
    df = read_csv_fallback(PANEL)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)

    df["country_idx"] = pd.factorize(df["country_code"])[0]
    if "region" in df.columns:
        df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    else:
        df["region_year"] = "r1_" + df["year"].astype(str)

    df = add_transforms(df)

    # y_ext lags
    for l in range(1, YEXT_LAGS + 1):
        df[f"y_ext_lag{l}"] = df.groupby("country_idx")[Y_VAR].shift(l)
    # geo lags
    for l in range(1, GEO_LAGS + 1):
        df[f"{SHOCK}_lag{l}"] = df.groupby("country_idx")[SHOCK].shift(l)

    print(f"Panel: {df.shape}, {df['country_idx'].nunique()} countries")
    return df


# ── Consistent-country sample (per variable) ────────────────────────

def get_consistent_countries(df, dep_var):
    """Countries with enough obs at both extreme horizons for dep_var."""
    dep_lag_terms = [f"{dep_var}_lag{l}" for l in range(1, DEP_LAGS + 1)]
    geo_lag_terms = [f"{SHOCK}_lag{l}" for l in range(1, GEO_LAGS + 1)]
    yext_lag_terms = [f"y_ext_lag{l}" for l in range(1, YEXT_LAGS + 1)]

    valid = set()
    for cc in df["country_code"].unique():
        cdf = df[df["country_code"] == cc]
        ok = True
        for h in (-H_NEG, H_POS):
            tmp = cdf.copy()
            tmp["dep_shift"] = tmp.groupby("country_idx")[dep_var].shift(-h)
            need = (["dep_shift", SHOCK] + dep_lag_terms + geo_lag_terms
                    + yext_lag_terms + ["region_year"])
            if tmp.dropna(subset=need).shape[0] < MIN_COUNTRY_OBS:
                ok = False
                break
        if ok:
            valid.add(cc)
    return valid


# ── LP estimation ────────────────────────────────────────────────────

def run_lp(df, dep_var, valid_countries):
    """LP-IRF of dep_var to SHOCK with DK SEs, balanced country sample."""
    dfs = df[df["country_code"].isin(valid_countries)].copy()

    dep_lag_terms = [f"{dep_var}_lag{l}" for l in range(1, DEP_LAGS + 1)]
    geo_lag_terms = [f"{SHOCK}_lag{l}" for l in range(1, GEO_LAGS + 1)]
    yext_lag_terms = [f"y_ext_lag{l}" for l in range(1, YEXT_LAGS + 1)]

    horizons = list(LP_RANGE)
    irf = np.full(len(horizons), np.nan)
    se  = np.full(len(horizons), np.nan)

    for idx, h in enumerate(horizons):
        tmp = dfs.copy()
        tmp["dep_shift"] = tmp.groupby("country_idx")[dep_var].shift(-h)
        X_cols = [SHOCK] + dep_lag_terms + geo_lag_terms + yext_lag_terms
        need = ["dep_shift"] + X_cols + ["region_year", "country_idx", "year"]
        sub = tmp.dropna(subset=need)
        if sub.shape[0] < 50:
            continue
        try:
            sp = sub.set_index(["country_idx", "year"])
            model = PanelOLS(
                sp["dep_shift"], sp[X_cols],
                entity_effects=True, time_effects=False,
                other_effects=sp["region_year"],
            )
            res = model.fit(cov_type="kernel")
            irf[idx] = res.params.get(SHOCK, np.nan)
            se[idx]  = res.std_errors.get(SHOCK, np.nan)
        except Exception as e:
            print(f"  LP error ({dep_var}, h={h}): {e}")

    return horizons, irf, se


# ── Plotting ─────────────────────────────────────────────────────────

def plot_single_irf(h, irf, se, title, ax, color, ylabel=None):
    """Plot IRF with 95% CI on a given axes."""
    h = np.asarray(h)
    vm = ~np.isnan(irf)
    if vm.any():
        ax.plot(h[vm], irf[vm], color=color, lw=2, marker='o')
        ci_vm = vm & ~np.isnan(se)
        if ci_vm.any():
            lo = irf[ci_vm] - 1.96 * se[ci_vm]
            hi = irf[ci_vm] + 1.96 * se[ci_vm]
            ax.fill_between(h[ci_vm], lo, hi, color=color, alpha=PS.CI_ALPHA)

    ax.axhline(0, **PS.HLINE_KW)
    ax.axvline(0, **PS.VLINE_KW)
    ax.set_xlabel(PS.IRF_XLABEL, fontsize=13)
    ax.set_ylabel(ylabel or PS.IRF_YLABEL_GDP, fontsize=13)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, axis='y')
    ax.set_xlim(-H_NEG, H_POS)


def plot_grid(results, channels, fname, color_offset=0):
    """2x2 grid of channel IRFs."""
    colors = sns.color_palette("deep")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, ((col, title, _), (h, irf, se, n_c)) in enumerate(zip(channels, results)):
        ax = axes[i // 2, i % 2]
        sc = SCALE_FACTORS.get(col, 1)
        title_n = f"{title}\n(N countries: {n_c})"
        ylabel = CHANNEL_YLABELS.get(col, PS.IRF_YLABEL_GDP)
        plot_single_irf(h, irf * sc, se * sc, title_n, ax,
                        colors[(i + color_offset) % len(colors)], ylabel=ylabel)
        ax.set_xlim(-H_NEG, H_POS)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    savefig(fig, FIGURES, fname)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Fig 14a-b: Channel Variables IRFs")
    print("=" * 70)

    df = load_and_prepare()

    for panel_label, channels, fname, color_offset in [
        ("A", CHANNELS_A, "Fig14A_channels_output_stability.pdf", 0),
        ("B", CHANNELS_B, "Fig14B_channels_fundamentals.pdf", 4),
    ]:
        print(f"\n--- Panel {panel_label} ---")
        results = []
        for col, title, _ in channels:
            print(f"  Processing {col}: {title}")
            # Create dep var lags
            for l in range(1, DEP_LAGS + 1):
                lag_col = f"{col}_lag{l}"
                if lag_col not in df.columns:
                    df[lag_col] = df.groupby("country_idx")[col].shift(l)

            valid = get_consistent_countries(df, col)
            print(f"    Balanced sample: {len(valid)} countries")

            h, irf, se = run_lp(df, col, valid)
            results.append((h, irf, se, len(valid)))

            # Cache individual channel
            sc = SCALE_FACTORS.get(col, 1)
            cache_df = pd.DataFrame({
                "horizon": h, "coef": irf * sc, "se": se * sc,
                "label": col,
            })
            cache_df.to_csv(CACHE / f"channel_{col}.csv", index=False)

        plot_grid(results, channels, fname, color_offset=color_offset)

    print("\nDone.")


if __name__ == "__main__":
    main()
