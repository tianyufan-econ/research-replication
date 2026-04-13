#!/usr/bin/env python3
"""
05_symmetry.py — Partner Symmetry and Temporal Stability (Replication)
======================================================================
Two figures:

  Fig 10a: Partner symmetry — joint LP with geo_relation_dyn_us and
           geo_relation_dyn_exclus as two shock variables.
           h=-10..25, country + region-year FE, Driscoll-Kraay SEs.

  Fig 10b: Temporal stability — split panel into 1960-1989 and 1990-2019
           sub-periods. Run baseline LP on each. Overlay both.

Source: 02_baseline/lp_us_decomp_reg.py + 05_robustness/robustness_additional.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS

from config import PANEL, FIGURES, ensure_dirs, savefig, read_csv_fallback, restrict_balanced_sample
from lp_utils import prepare_panel, lp_irf, lp_irf_joint
import plot_style as PS

PS.apply_theme()
ensure_dirs()

# ── Colors (matching working plot_symmetry.ipynb) ────────────────────────
C1_COLOR = PS.C_BLUE       # Component 1 (US / 1960-89): blue, solid
C2_COLOR = PS.C_RED        # Component 2 (non-US / 1990-2019): red, dashed
REF_COLOR = '0.45'         # medium gray for baseline reference

# ── LP parameters ────────────────────────────────────────────────────────
Y_LAGS = 4
GEO_LAGS = 4
H_START = -10
H_END = 25


# =====================================================================
# Data Loading
# =====================================================================

def load_panel():
    """Load and prepare the panel."""
    print("Loading panel ...")
    df = read_csv_fallback(PANEL)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)
    df = restrict_balanced_sample(df)
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    return df


# =====================================================================
# Plotting Helpers
# =====================================================================

def _plot_component(ax, h, coef, se, color, label, ls='-', marker='o', ms=5,
                    lw=2.2, show_ci=True, ci_alpha=0.12):
    """Plot one IRF component with optional CI band.
    Matches working plot_symmetry.ipynb _plot_component."""
    valid = ~np.isnan(coef)
    if not valid.any():
        return

    ax.plot(h[valid], coef[valid], color=color, ls=ls, lw=lw,
            marker=marker, ms=ms, label=label, zorder=3)

    if show_ci:
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        ci_v = valid & ~np.isnan(ci_lo) & ~np.isnan(ci_hi)
        if ci_v.any():
            ax.fill_between(h[ci_v], ci_lo[ci_v], ci_hi[ci_v],
                            color=color, alpha=ci_alpha, zorder=1)


def _plot_reference(ax, h, coef, color, label):
    """Plot baseline reference line (thin, no CI, no markers).
    Matches working plot_symmetry.ipynb _plot_reference."""
    valid = ~np.isnan(coef)
    if valid.any():
        ax.plot(h[valid], coef[valid], color=color, ls='--', lw=1.8,
                marker='', alpha=0.7, label=label, zorder=2)


# =====================================================================
# Fig 10a: Partner Symmetry
# =====================================================================

def fig_partner_symmetry(raw):
    """
    Joint LP with geo_relation_dyn_us and geo_relation_dyn_exclus.
    Both entered as shock variables in the same regression.
    """
    print("\n--- Fig 10a: Partner Symmetry ---")

    shock_vars = ["geo_relation_dyn_us", "geo_relation_dyn_exclus"]

    # Verify columns exist
    for sv in shock_vars:
        if sv not in raw.columns:
            print(f"ERROR: {sv} not in panel. Skipping partner symmetry.")
            return
        print(f"  {sv}: n={raw[sv].notna().sum():,}  sd={raw[sv].std():.4f}")

    # Prepare panel with lags for both shock variables
    df = prepare_panel(raw, y_var="y_ext", shock_var="geo_relation_dyn_us",
                       y_lags=Y_LAGS, shock_lags=GEO_LAGS,
                       extra_shock_vars=["geo_relation_dyn_exclus"])

    # Run joint LP
    results = lp_irf_joint(
        df, shock_vars=shock_vars, y_var="y_ext",
        horizon_range=range(H_START, H_END + 1),
        y_lags=Y_LAGS, shock_lags=GEO_LAGS,
        fe="region_year", cov_type="kernel",
    )

    # Also run baseline (full geo_relation_dyn) for reference line
    df_base = prepare_panel(raw, y_var="y_ext", shock_var="geo_relation_dyn",
                            y_lags=Y_LAGS, shock_lags=GEO_LAGS)
    res_base = lp_irf(df_base, shock_var="geo_relation_dyn", y_var="y_ext",
                      horizon_range=range(0, H_END + 1),
                      y_lags=Y_LAGS, shock_lags=GEO_LAGS,
                      fe="region_year", cov_type="kernel")

    # Plot (matching working plot_symmetry.ipynb)
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)

    # US component (blue, solid, with CI)
    r_us = results["geo_relation_dyn_us"]
    h = r_us["h_vals"]
    # Restrict to h >= 0 for consistency with temporal panel
    h0_mask = h >= 0
    _plot_component(ax, h[h0_mask], r_us["coef"][h0_mask], r_us["se"][h0_mask],
                    C1_COLOR, label='Alignment with US',
                    ls='-', marker='o', ms=5)

    # Non-US component (red, dashed, with CI)
    r_ex = results["geo_relation_dyn_exclus"]
    _plot_component(ax, h[h0_mask], r_ex["coef"][h0_mask], r_ex["se"][h0_mask],
                    C2_COLOR, label='Alignment Excl. US',
                    ls='--', marker='s', ms=5)

    # Baseline reference (gray, thin dashed, no CI)
    h_b = res_base["h_vals"]
    _plot_reference(ax, h_b, res_base["coef"], REF_COLOR,
                    label='Aggregate (Baseline)')

    PS.style_irf_ax(ax, ylabel=PS.IRF_YLABEL_GDP)
    fig.tight_layout()
    savefig(fig, FIGURES, "Fig10A_symmetry_partner.pdf")
    plt.close(fig)


# =====================================================================
# Fig 10b: Temporal Stability
# =====================================================================

def fig_temporal_stability(raw):
    """
    Split panel into 1960-1989 and 1990-2019.
    Run baseline LP on each sub-period (h=0..10).
    Overlay both on same figure.
    """
    print("\n--- Fig 10b: Temporal Stability ---")

    sub_horizon = range(0, 11)  # shorter horizon for sub-periods

    # Period 1: 1960-1989
    print("  1960-1989 ...")
    df_p1 = raw[raw["year"].between(1960, 1989)].copy()
    df_p1 = prepare_panel(df_p1, y_var="y_ext", shock_var="geo_relation_dyn",
                          y_lags=Y_LAGS, shock_lags=GEO_LAGS)
    res_p1 = lp_irf(df_p1, shock_var="geo_relation_dyn", y_var="y_ext",
                    horizon_range=sub_horizon, y_lags=Y_LAGS, shock_lags=GEO_LAGS,
                    fe="region_year", cov_type="kernel")

    # Period 2: 1990-2019
    print("  1990-2019 ...")
    df_p2 = raw[raw["year"].between(1990, 2019)].copy()
    df_p2 = prepare_panel(df_p2, y_var="y_ext", shock_var="geo_relation_dyn",
                          y_lags=Y_LAGS, shock_lags=GEO_LAGS)
    res_p2 = lp_irf(df_p2, shock_var="geo_relation_dyn", y_var="y_ext",
                    horizon_range=sub_horizon, y_lags=Y_LAGS, shock_lags=GEO_LAGS,
                    fe="region_year", cov_type="kernel")

    # Full sample baseline for reference
    df_full = prepare_panel(raw, y_var="y_ext", shock_var="geo_relation_dyn",
                            y_lags=Y_LAGS, shock_lags=GEO_LAGS)
    res_full = lp_irf(df_full, shock_var="geo_relation_dyn", y_var="y_ext",
                      horizon_range=sub_horizon, y_lags=Y_LAGS, shock_lags=GEO_LAGS,
                      fe="region_year", cov_type="kernel")

    # Plot (matching working plot_symmetry.ipynb)
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)

    _plot_component(ax, res_p1["h_vals"], res_p1["coef"], res_p1["se"],
                    C1_COLOR, label='1960\u20131989',
                    ls='-', marker='o', ms=5)
    _plot_component(ax, res_p2["h_vals"], res_p2["coef"], res_p2["se"],
                    C2_COLOR, label='1990\u20132019',
                    ls='--', marker='s', ms=5)
    _plot_reference(ax, res_full["h_vals"], res_full["coef"], REF_COLOR,
                    label='Full Sample')

    PS.style_irf_ax(ax, ylabel=PS.IRF_YLABEL_GDP)
    fig.tight_layout()
    savefig(fig, FIGURES, "Fig10B_symmetry_temporal.pdf")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 65)
    print("  SYMMETRY: PARTNER DECOMPOSITION + TEMPORAL STABILITY (REPLICATION)")
    print("=" * 65)

    raw = load_panel()
    print(f"Panel: {len(raw):,} rows, {raw['country_code'].nunique()} countries, "
          f"years {raw['year'].min()}-{raw['year'].max()}")

    fig_partner_symmetry(raw)
    fig_temporal_stability(raw)

    print("\nDone. Figures saved to:", FIGURES)


if __name__ == "__main__":
    main()
