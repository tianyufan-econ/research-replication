"""
01_baseline.py
==============
Replication script for baseline results (Figures 6a, 6b).

Produces two figures:
  Fig 6a: Self-IRF of geo_relation_dyn (h=0..25)
  Fig 6b: GDP IRF — balanced vs unbalanced panel (h=-10..25)

Self-contained: loads data, estimates, and plots in one file.
"""

# ── Imports ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (PANEL, FIGURES, ensure_dirs, savefig,
                    read_csv_fallback, restrict_balanced_sample, LP_DEFAULTS)
from lp_utils import prepare_panel, lp_irf
import plot_style as PS

PS.apply_theme()
ensure_dirs()

SEABORN_BLUE = sns.color_palette("deep")[0]
SEABORN_ORANGE = sns.color_palette("deep")[1]


# ── Helpers ──────────────────────────────────────────────────────────────

def plot_irf(h_vals, coef, se, label, color, fname, ylabel,
             ci_label=PS.LEGEND_CI_DK, z=1.96):
    """Plot a single IRF with confidence band and save to FIGURES."""
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    ci_lo = coef - z * se
    ci_hi = coef + z * se
    valid = ~np.isnan(coef)
    ax.plot(h_vals[valid], coef[valid], color=color, ls="-", lw=2.5,
            marker="o", ms=6, label=label)
    ci_ok = valid & ~np.isnan(ci_lo) & ~np.isnan(ci_hi)
    if ci_ok.any():
        ax.fill_between(h_vals[ci_ok], ci_lo[ci_ok], ci_hi[ci_ok],
                        color=color, alpha=PS.CI_ALPHA, label=ci_label)
        ax.plot(h_vals[ci_ok], ci_lo[ci_ok], color=color,
                ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
        ax.plot(h_vals[ci_ok], ci_hi[ci_ok], color=color,
                ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
    ax.axhline(0, **PS.HLINE_KW)
    ax.axvline(0, **PS.VLINE_KW)
    ax.set_xlabel(PS.IRF_XLABEL, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    savefig(fig, FIGURES, fname)
    plt.close(fig)




# ── Main ─────────────────────────────────────────────────────────────────

def main():
    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    print("Loading panel ...")
    df_full = read_csv_fallback(PANEL)
    df_full["country_code"] = df_full["country_code"].astype(str)
    df_full["year"] = df_full["year"].astype(int)
    print(f"  Panel shape: {df_full.shape}")

    shock = "geo_relation_dyn"
    y_lags = LP_DEFAULTS["y_lags"]
    shock_lags = LP_DEFAULTS["shock_lags"]

    # ==================================================================
    # Fig 6a  — Self-IRF of geo_relation_dyn  (h = 0 .. 25)
    # ==================================================================
    print("\n--- Fig 6a: Self-IRF of geo_relation_dyn ---")
    df_bal = restrict_balanced_sample(df_full.copy())
    df_bal = prepare_panel(df_bal, shock_var=shock,
                           y_lags=y_lags, shock_lags=shock_lags)
    print(f"  Balanced countries: {df_bal['country_code'].nunique()}")

    res_self = lp_irf(df_bal, shock_var=shock, y_var=shock,
                      horizon_range=range(0, 26),
                      y_lags=y_lags, shock_lags=shock_lags)
    plot_irf(res_self["h_vals"], res_self["coef"], res_self["se"],
             PS.LEGEND_POINT, SEABORN_BLUE,
             "Fig6A_self_irf.pdf",
             PS.IRF_YLABEL_GEO)

    # ==================================================================
    # Fig 6b  — GDP IRF: balanced vs unbalanced  (h = -10 .. 25)
    # ==================================================================
    print("\n--- Fig 6b: GDP IRF balanced vs unbalanced ---")
    horizon_range = range(-10, 26)

    # Balanced
    res_bal = lp_irf(df_bal, shock_var=shock, y_var="y_ext",
                     horizon_range=horizon_range,
                     y_lags=y_lags, shock_lags=shock_lags)

    # Unbalanced (all countries)
    df_unbal = prepare_panel(df_full.copy(), shock_var=shock,
                             y_lags=y_lags, shock_lags=shock_lags)
    print(f"  Unbalanced countries: {df_unbal['country_code'].nunique()}")
    res_unbal = lp_irf(df_unbal, shock_var=shock, y_var="y_ext",
                       horizon_range=horizon_range,
                       y_lags=y_lags, shock_lags=shock_lags)

    z = 1.96
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    # Balanced (solid + CI)
    h_b, c_b, s_b = res_bal["h_vals"], res_bal["coef"], res_bal["se"]
    v = ~np.isnan(c_b)
    ax.plot(h_b[v], c_b[v], color=SEABORN_BLUE, ls="-", lw=2.5,
            marker="o", ms=6, label="Balanced Panel")
    ci_lo, ci_hi = c_b - z * s_b, c_b + z * s_b
    ci_ok = v & ~np.isnan(ci_lo) & ~np.isnan(ci_hi)
    ax.fill_between(h_b[ci_ok], ci_lo[ci_ok], ci_hi[ci_ok],
                    color=SEABORN_BLUE, alpha=PS.CI_ALPHA, label=PS.LEGEND_CI_DK)
    ax.plot(h_b[ci_ok], ci_lo[ci_ok], color=SEABORN_BLUE,
            ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
    ax.plot(h_b[ci_ok], ci_hi[ci_ok], color=SEABORN_BLUE,
            ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
    # Unbalanced (dashed)
    h_u, c_u = res_unbal["h_vals"], res_unbal["coef"]
    vu = ~np.isnan(c_u)
    ax.plot(h_u[vu], c_u[vu], color=SEABORN_ORANGE, ls="--", lw=2.0,
            marker="s", ms=5, label="Unbalanced Panel")
    ax.axhline(0, **PS.HLINE_KW)
    ax.axvline(0, **PS.VLINE_KW)
    ax.set_xlabel(PS.IRF_XLABEL, fontsize=12)
    ax.set_ylabel(PS.IRF_YLABEL_GDP, fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    savefig(fig, FIGURES, "Fig6B_gdp_irf.pdf")
    plt.close(fig)

    print("\n=== 01_baseline.py complete ===")


if __name__ == "__main__":
    main()
