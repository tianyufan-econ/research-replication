#!/usr/bin/env python3
"""
03_decomposition.py — Component Decomposition (Replication)
============================================================
Decomposes the geopolitical alignment index into three components:
  Economic (A), Diplomatic (B), Security (C+D)

Produces two figures:
  Fig 8a: Horse-race LP — all 3 components entered simultaneously
  Fig 8b: Residualized — each component orthogonalized to the other two

Source: 07_decomposition/component_decomposition.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from linearmodels.panel import PanelOLS

from config import PANEL, FIGURES, ensure_dirs, savefig, read_csv_fallback, restrict_balanced_sample
import plot_style as PS

PS.apply_theme()
ensure_dirs()

# ── Style (from plot_style.py, matching working code) ────────────────────
Z95 = stats.norm.ppf(0.975)

# ── Component definitions ────────────────────────────────────────────────
COMPONENTS = ["econ", "diplo", "security"]
COMP_COLS = {
    "econ": "econ_relation_dyn",
    "diplo": "diplo_relation_dyn",
    "security": "security_relation_dyn",
}
COMP_LABELS = {"econ": "Economic", "diplo": "Diplomatic", "security": "Security"}
COMP_COLORS = {"econ": PS.C_BLUE, "diplo": PS.C_ORANGE, "security": PS.C_GREEN}

# ── LP parameters ────────────────────────────────────────────────────────
H_START, H_END = 0, 25
Y_LAGS = 4
GEO_LAGS = 4


# =====================================================================
# Data Preparation
# =====================================================================

def load_panel():
    """Load and prepare the panel with component scores."""
    print("Loading panel ...")
    df = read_csv_fallback(PANEL)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)
    df = restrict_balanced_sample(df)
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

    # Index variables
    df["country_idx"] = pd.factorize(df["country_code"])[0]
    df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)

    # y_ext lags
    for lag in range(1, Y_LAGS + 1):
        df[f"y_ext_lag{lag}"] = df.groupby("country_code")["y_ext"].shift(lag)

    # Component lags
    for comp in COMPONENTS:
        col = COMP_COLS[comp]
        for lag in range(1, GEO_LAGS + 1):
            df[f"{col}_lag{lag}"] = df.groupby("country_code")[col].shift(lag)

    # Verify
    for comp in COMPONENTS:
        col = COMP_COLS[comp]
        s = df[col].dropna()
        print(f"  {col}: n={len(s):,}  mean={s.mean():.4f}  sd={s.std():.4f}")

    return df


# =====================================================================
# LP Estimation
# =====================================================================

def run_lp(df, regressors, y_var="y_ext"):
    """
    LP-IRF with Driscoll-Kraay SEs, country + region-year FE.

    Returns (h_vals, results_dict, n_obs).
    results_dict maps each regressor to {"coef": array, "se": array}.
    """
    h_vals = np.arange(H_START, H_END + 1)
    n_h = len(h_vals)
    results = {v: {"coef": np.full(n_h, np.nan),
                    "se": np.full(n_h, np.nan)} for v in regressors}
    n_obs = np.full(n_h, np.nan)

    for i, h in enumerate(h_vals):
        tmp = df.copy()
        tmp["_dep"] = tmp.groupby("country_idx")[y_var].shift(-h)
        sub = tmp.dropna(subset=["_dep"] + regressors + ["region_year"])
        if len(sub) < 50:
            continue

        sp = sub.set_index(["country_idx", "year"])
        try:
            mod = PanelOLS(
                sp["_dep"], sp[regressors],
                entity_effects=True,
                time_effects=False,
                other_effects=sp[["region_year"]],
            )
            fit = mod.fit(cov_type="kernel")
            n_obs[i] = fit.nobs
            for v in regressors:
                results[v]["coef"][i] = fit.params[v]
                results[v]["se"][i] = fit.std_errors[v]
        except Exception:
            pass

    return h_vals, results, n_obs


# =====================================================================
# Residualization
# =====================================================================

def residualize(panel):
    """
    Regress each component on the other two + country FE + year FE.
    Return panel with residual columns and their lags.
    """
    print("Residualizing components ...")
    comp_cols = [COMP_COLS[c] for c in COMPONENTS]
    sub = panel.dropna(subset=comp_cols).copy()

    # Country and year dummies
    c_dum = pd.get_dummies(sub["country_code"], prefix="c", drop_first=True, dtype=float)
    y_dum = pd.get_dummies(sub["year"], prefix="yr", drop_first=True, dtype=float)

    for target in comp_cols:
        others = [c for c in comp_cols if c != target]
        X = np.column_stack([np.ones(len(sub)),
                             sub[others].values,
                             c_dum.values,
                             y_dum.values])
        y_vec = sub[target].values
        beta = np.linalg.lstsq(X, y_vec, rcond=None)[0]
        sub[f"{target}_resid"] = y_vec - X @ beta

    resid_cols = [f"{c}_resid" for c in comp_cols]
    panel = panel.merge(sub[["country_code", "year"] + resid_cols],
                        on=["country_code", "year"], how="left")

    # Lags for residuals
    for comp in COMPONENTS:
        col = f"{COMP_COLS[comp]}_resid"
        for lag in range(1, GEO_LAGS + 1):
            panel[f"{col}_lag{lag}"] = panel.groupby("country_code")[col].shift(lag)

    for comp in COMPONENTS:
        col = f"{COMP_COLS[comp]}_resid"
        s = panel[col].dropna()
        print(f"  {col}: n={len(s):,}  sd={s.std():.4f}")

    return panel


# =====================================================================
# Plotting
# =====================================================================

def _ci_band(ax, h, coef, se, color, alpha=PS.CI_ALPHA):
    """Draw CI fill + dashed bounds (matches working code _ci_band)."""
    lo, hi = coef - Z95 * se, coef + Z95 * se
    v = ~np.isnan(coef)
    ax.fill_between(h[v], lo[v], hi[v], color=color, alpha=alpha)
    ax.plot(h[v], lo[v], color=color, ls='--',
            lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
    ax.plot(h[v], hi[v], color=color, ls='--',
            lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)


def plot_components(h, data, fname, show_ci=True, ylabel=PS.IRF_YLABEL_GDP):
    """Overlay IRFs for each component (matches working plot_components)."""
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    for comp in COMPONENTS:
        c, lab = COMP_COLORS[comp], COMP_LABELS[comp]
        coef, se = data[comp]['coef'], data[comp]['se']
        v = ~np.isnan(coef)
        ax.plot(h[v], coef[v], color=c, lw=2.5, label=lab)
        if show_ci:
            _ci_band(ax, h, coef, se, c)
    PS.style_irf_ax(ax, ylabel=ylabel)
    fig.tight_layout()
    savefig(fig, FIGURES, fname)
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 65)
    print("  COMPONENT DECOMPOSITION (REPLICATION)")
    print("=" * 65)

    # Load data
    panel = load_panel()

    # ── Horse race (all three components simultaneously) ────────────
    print("\nHorse race LP (all three components jointly) ...")
    comp_cols_list = [COMP_COLS[c] for c in COMPONENTS]
    regressors = list(comp_cols_list)
    regressors += [f"y_ext_lag{i}" for i in range(1, Y_LAGS + 1)]
    for col in comp_cols_list:
        regressors += [f"{col}_lag{i}" for i in range(1, GEO_LAGS + 1)]

    h, res_hr, nobs_hr = run_lp(panel, regressors)
    horse_race = {comp: res_hr[COMP_COLS[comp]] for comp in COMPONENTS}

    for comp in COMPONENTS:
        coef = horse_race[comp]["coef"]
        se = horse_race[comp]["se"]
        sig = np.nansum(np.abs(coef / se) > 1.96)
        print(f"  {COMP_LABELS[comp]}: {sig} significant horizons")

    # Fig 8a
    plot_components(h, horse_race, "Fig8A_component_horserace.pdf")

    # ── Residualized ────────────────────────────────────────────────
    panel = residualize(panel)
    print("\nResidual LP (each component orthogonalized) ...")
    residualized = {}
    for comp in COMPONENTS:
        col = f"{COMP_COLS[comp]}_resid"
        regs = ([col]
                + [f"y_ext_lag{i}" for i in range(1, Y_LAGS + 1)]
                + [f"{col}_lag{i}" for i in range(1, GEO_LAGS + 1)])
        _, r, _ = run_lp(panel, regs)
        residualized[comp] = r[col]
        sig = np.nansum(np.abs(r[col]["coef"] / r[col]["se"]) > 1.96)
        print(f"  {COMP_LABELS[comp]} (residualized): {sig} significant horizons")

    # Fig 8b
    plot_components(h, residualized, "Fig8B_component_residualized.pdf")

    print("\nDone. Figures saved to:", FIGURES)


if __name__ == "__main__":
    main()
