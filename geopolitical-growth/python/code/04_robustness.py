#!/usr/bin/env python3
"""
04_robustness.py — Fixed Effects and Progressive Controls Robustness (Replication)
==================================================================================
Two figures:

  Fig 11a: Five FE specifications (baseline = region-year with CI, others dashed):
    1. Region-Year FE (baseline)
    2. Year FE only
    3. Initial GDP Quintile x Year FE
    4. Region x Current Regime x Year FE
    5. Region x Initial Regime x Year FE

  Fig 11b: Progressive controls on a common sample (~109 countries):
    1. Baseline (no extra controls)
    2. + Trade Lags (4 lags)
    3. + Unrest Lags (4 lags)
    4. + War Exposure Lags (4 lags of 5 war variables = 20 terms)
    5. + Institution Lags (4 lags of 5 V-Dem variables = 20 terms)

Source: 02_baseline/lp_robust_fe_reg.py + 05_robustness/combined_controls.py +
        05_robustness/war_controls.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS

from config import PANEL, FIGURES, ensure_dirs, savefig, read_csv_fallback, restrict_balanced_sample
import plot_style as PS

PS.apply_theme()
ensure_dirs()

# ── Colors (matching working code) ──────────────────────────────────────
SEABORN_BLUE = sns.color_palette("deep")[0]
VIRIDIS = sns.color_palette("deep", n_colors=4)  # sequential for FE overlay specs

# ── LP parameters ────────────────────────────────────────────────────────
Y_VAR = "y_ext"
SHOCK_VAR = "geo_relation_dyn"
Y_LAGS = 4
GEO_LAGS = 4
H_START = -10
H_END = 25
MIN_COUNTRY_OBS = 10

# ── War control variables ────────────────────────────────────────────────
WAR_VARS = [
    "war_site_onset_all",
    "war_site_caspop_all",
    "war_trade_caspop_weighted_all",
    "war_prox_caspop_weighted_all",
    "war_exposure_site_count_all",
]

# ── V-Dem institution variables ──────────────────────────────────────────
INSTITUTION_VARS = [
    "v2x_polyarchy",
    "v2x_liberal",
    "v2x_partip",
    "v2xdl_delib",
    "v2x_egal",
]

# Democracy threshold for current regime classification
DEMOCRACY_THRESHOLD = 0.5


# =====================================================================
# Data Preparation
# =====================================================================

def load_and_prepare():
    """Load panel, add lags, construct FE columns."""
    print("Loading panel ...")
    df = read_csv_fallback(PANEL)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)
    df = restrict_balanced_sample(df)
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

    # Index variables
    df["country_idx"] = pd.factorize(df["country_code"])[0]

    # ── FE columns ───────────────────────────────────────────────────
    # 1. Region-Year
    df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)

    # 2. Year string (for year-only FE)
    df["year_str"] = df["year"].astype(str)

    # 3. Initial GDP Quintile x Year
    if "gdp_quintile" in df.columns:
        df["iniGDP_yr"] = df["gdp_quintile"].astype(str) + "_" + df["year"].astype(str)
    else:
        print("WARNING: gdp_quintile not found, constructing from initial y_ext")
        first_y = df.groupby("country_code")["y_ext"].first()
        quintile = pd.qcut(first_y, 5, labels=False, duplicates="drop")
        quintile_map = quintile.to_dict()
        df["gdp_quintile"] = df["country_code"].map(quintile_map)
        df["iniGDP_yr"] = df["gdp_quintile"].astype(str) + "_" + df["year"].astype(str)

    # 4. Region x Current Regime x Year
    if "v2x_polyarchy" in df.columns:
        df["current_regime"] = (df["v2x_polyarchy"] >= DEMOCRACY_THRESHOLD).astype(int)
    else:
        df["current_regime"] = 0
    df["regRegYr"] = (df["region"].astype(str) + "_" +
                      df["current_regime"].astype(str) + "_" +
                      df["year"].astype(str))

    # 5. Region x Initial Regime x Year
    if "InitReg" in df.columns:
        df["regIniregYr"] = (df["region"].astype(str) + "_" +
                             df["InitReg"].astype(str) + "_" +
                             df["year"].astype(str))
    else:
        # Construct from first observed v2x_polyarchy
        first_dem = df.groupby("country_code")["v2x_polyarchy"].first()
        init_reg = (first_dem >= DEMOCRACY_THRESHOLD).astype(int)
        df["InitReg"] = df["country_code"].map(init_reg.to_dict())
        df["regIniregYr"] = (df["region"].astype(str) + "_" +
                             df["InitReg"].astype(str) + "_" +
                             df["year"].astype(str))

    # ── Lags ─────────────────────────────────────────────────────────
    for lag in range(1, Y_LAGS + 1):
        df[f"{Y_VAR}_lag{lag}"] = df.groupby("country_idx")[Y_VAR].shift(lag)
    for lag in range(1, GEO_LAGS + 1):
        df[f"{SHOCK_VAR}_lag{lag}"] = df.groupby("country_idx")[SHOCK_VAR].shift(lag)

    # Control variable lags (for Fig 11b)
    control_vars = ["trade", "unrest_new"] + WAR_VARS + INSTITUTION_VARS
    for col in control_vars:
        if col not in df.columns:
            df[col] = np.nan
    for col in control_vars:
        for lag in range(1, 5):
            df[f"{col}_lag{lag}"] = df.groupby("country_idx")[col].shift(lag)

    # Fill war controls NaN with 0 (no war = 0 exposure)
    war_cols = [c for c in df.columns if c.startswith("war_")]
    df[war_cols] = df[war_cols].fillna(0.0)

    n = df["country_code"].nunique()
    print(f"Panel: {len(df):,} rows, {n} countries, "
          f"years {df['year'].min()}-{df['year'].max()}")
    return df


# =====================================================================
# LP Estimation
# =====================================================================

def lp_irf_balanced(df, valid_countries, fixed_effect="region_year",
                    extra_controls=None):
    """
    LP-IRF with Driscoll-Kraay SEs on a common country sample.

    Returns DataFrame with columns: horizon, coef, se, ci_lo, ci_hi, n_obs.
    """
    y_lags = [f"{Y_VAR}_lag{i}" for i in range(1, Y_LAGS + 1)]
    shock_lags = [f"{SHOCK_VAR}_lag{i}" for i in range(1, GEO_LAGS + 1)]
    x_cols = [SHOCK_VAR] + y_lags + shock_lags
    if extra_controls:
        x_cols = x_cols + extra_controls

    sample = df[df["country_code"].isin(valid_countries)].copy()
    records = []

    for h in range(H_START, H_END + 1):
        tmp = sample.copy()
        tmp["_dep"] = tmp.groupby("country_idx")[Y_VAR].shift(-h)
        required = ["_dep", fixed_effect, "country_idx", "year"] + x_cols
        sub = tmp.dropna(subset=required).copy()

        if sub.empty or sub.shape[0] < len(x_cols) + 50:
            records.append(dict(horizon=h, coef=np.nan, se=np.nan,
                                ci_lo=np.nan, ci_hi=np.nan, n_obs=0))
            continue

        panel = sub.set_index(["country_idx", "year"])
        y = panel["_dep"]
        X = panel[x_cols].copy()

        # Drop columns with zero variance (can happen with some FE configs)
        keep = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
        if SHOCK_VAR not in keep:
            records.append(dict(horizon=h, coef=np.nan, se=np.nan,
                                ci_lo=np.nan, ci_hi=np.nan, n_obs=len(sub)))
            continue
        X = X[keep]

        try:
            model = PanelOLS(
                dependent=y, exog=X,
                entity_effects=True,
                time_effects=False,
                other_effects=panel[fixed_effect],
                drop_absorbed=True,
                check_rank=False,
            )
            res = model.fit(cov_type="kernel")
            coef = res.params.get(SHOCK_VAR, np.nan)
            se = res.std_errors.get(SHOCK_VAR, np.nan)
        except Exception as exc:
            print(f"  LP error h={h}: {exc}")
            coef, se = np.nan, np.nan

        records.append(dict(
            horizon=h, coef=coef, se=se,
            ci_lo=coef - 1.96 * se if pd.notna(se) else np.nan,
            ci_hi=coef + 1.96 * se if pd.notna(se) else np.nan,
            n_obs=int(len(sub)),
        ))

    return pd.DataFrame.from_records(records)


def get_consistent_country_sample(df, extra_controls=None):
    """Find countries with enough data at the horizon endpoints."""
    y_lags = [f"{Y_VAR}_lag{i}" for i in range(1, Y_LAGS + 1)]
    shock_lags = [f"{SHOCK_VAR}_lag{i}" for i in range(1, GEO_LAGS + 1)]
    controls = extra_controls or []
    horizon_edges = [H_START, H_END]
    valid = set()

    for cc, cdf in df.groupby("country_code", sort=False):
        ok = True
        for h in horizon_edges:
            tmp = cdf.copy()
            tmp["_dep"] = tmp.groupby("country_idx")[Y_VAR].shift(-h)
            req = (["_dep", SHOCK_VAR, "region_year"]
                   + y_lags + shock_lags + controls)
            sub = tmp.dropna(subset=req)
            if len(sub) < MIN_COUNTRY_OBS:
                ok = False
                break
        if ok:
            valid.add(cc)

    print(f"  Common sample: {len(valid)} / {df['country_code'].nunique()} countries")
    return valid


# =====================================================================
# Plotting
# =====================================================================

def plot_fe_specs(irf_data, fname, confidence_level=0.95):
    """Plot original IRF with CI and additional IRFs as dashed lines.
    Matches working lp_robust_fe_reg.py plot_combined_irf."""
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)

    # Plot original specification (region_year) with CI
    h_vals = irf_data[0]["h"]
    irf = irf_data[0]["coef"]
    se = irf_data[0]["se"]
    label = irf_data[0]["label"]
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    lower_ci = irf - z_score * se
    upper_ci = irf + z_score * se

    valid_mask = ~np.isnan(irf)
    if valid_mask.any():
        ax.plot(h_vals[valid_mask], irf[valid_mask],
                label=label, color=SEABORN_BLUE,
                linestyle='-', linewidth=2.5, marker='o', markersize=6)

        ci_valid_mask = ~(np.isnan(lower_ci) | np.isnan(upper_ci)) & valid_mask
        if ci_valid_mask.any():
            ax.fill_between(h_vals[ci_valid_mask],
                           lower_ci[ci_valid_mask],
                           upper_ci[ci_valid_mask],
                           color=SEABORN_BLUE, alpha=PS.CI_ALPHA,
                           label=f"{int(confidence_level*100)}% CI (Region-Year)")
            ax.plot(h_vals[ci_valid_mask], lower_ci[ci_valid_mask],
                   color=SEABORN_BLUE, linestyle='--', linewidth=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
            ax.plot(h_vals[ci_valid_mask], upper_ci[ci_valid_mask],
                   color=SEABORN_BLUE, linestyle='--', linewidth=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)

    # Plot additional specifications as dashed lines
    for idx, spec in enumerate(irf_data[1:], start=1):
        h2, irf2 = spec["h"], spec["coef"]
        label2 = spec["label"]
        color = VIRIDIS[idx - 1]
        valid_mask2 = ~np.isnan(irf2)
        if valid_mask2.any():
            ax.plot(h2[valid_mask2], irf2[valid_mask2],
                    label=label2, color=color,
                    linestyle='--', linewidth=2, marker='s', markersize=5)

    # Styling
    ax.axhline(0, **PS.HLINE_KW)
    ax.axvline(0, **PS.VLINE_KW)
    ax.set_xlabel(PS.IRF_XLABEL, fontsize=12)
    ax.set_ylabel(PS.IRF_YLABEL_GDP, fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    fig.tight_layout()
    savefig(fig, FIGURES, fname)
    plt.close(fig)


def plot_progressive_controls(results, fname):
    """Plot the baseline with CI and all control overlays as dashed lines.
    Matches working combined_controls.py plot_results."""
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)

    baseline = results["Baseline"]
    baseline_mask = baseline["coef"].notna()
    ax.plot(
        baseline.loc[baseline_mask, "horizon"],
        baseline.loc[baseline_mask, "coef"],
        color=PS.C_BLUE,
        lw=2.5,
        ls="-",
        marker="o",
        ms=6,
        label="Baseline",
    )

    ci_mask = baseline["ci_lo"].notna() & baseline["ci_hi"].notna() & baseline_mask
    ax.fill_between(
        baseline.loc[ci_mask, "horizon"],
        baseline.loc[ci_mask, "ci_lo"],
        baseline.loc[ci_mask, "ci_hi"],
        color=PS.C_BLUE,
        alpha=PS.CI_ALPHA,
        label=PS.LEGEND_CI_DK,
    )
    ax.plot(
        baseline.loc[ci_mask, "horizon"],
        baseline.loc[ci_mask, "ci_lo"],
        color=PS.C_BLUE,
        ls="--",
        lw=PS.CI_BOUND_LW,
        alpha=PS.CI_BOUND_ALPHA,
    )
    ax.plot(
        baseline.loc[ci_mask, "horizon"],
        baseline.loc[ci_mask, "ci_hi"],
        color=PS.C_BLUE,
        ls="--",
        lw=PS.CI_BOUND_LW,
        alpha=PS.CI_BOUND_ALPHA,
    )

    overlay_specs = [
        ("+ Trade Lags",        PS.C_ORANGE),
        ("+ Unrest Lags",       PS.C_GREEN),
        ("+ War Exposure Lags", PS.C_RED),
        ("+ Institution Lags",  sns.color_palette("deep")[4]),
    ]

    for label, color in overlay_specs:
        if label not in results:
            continue
        res = results[label]
        valid = res["coef"].notna()
        ax.plot(
            res.loc[valid, "horizon"],
            res.loc[valid, "coef"],
            color=color,
            lw=2.0,
            ls="--",
            marker="s",
            ms=5,
            label=label,
        )

    PS.style_irf_ax(ax, ylabel=PS.IRF_YLABEL_GDP)
    fig.tight_layout()
    savefig(fig, FIGURES, fname)
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 72)
    print("  ROBUSTNESS: FE SPECIFICATIONS + PROGRESSIVE CONTROLS (REPLICATION)")
    print("=" * 72)

    df = load_and_prepare()

    # ==================================================================
    # FIG 11a: Five FE Specifications
    # ==================================================================
    print("\n--- Fig 11a: Fixed Effects Robustness ---")
    valid_all = set(df["country_code"].unique())

    fe_specs = [
        ("region_year", "Region-Year FE"),
        ("year_str", "Year FE"),
        ("iniGDP_yr", "Initial GDP-Year FE"),
        ("regRegYr", "Region-Regime-Year FE"),
        ("regIniregYr", "Region-Initial Regime-Year FE"),
    ]

    irf_data = []
    for fe_col, label in fe_specs:
        print(f"  Estimating: {label} ...")
        res = lp_irf_balanced(df, valid_all, fixed_effect=fe_col)
        h = res["horizon"].values
        coef = res["coef"].values
        se = res["se"].values if fe_col == "region_year" else None
        irf_data.append(dict(h=h, coef=coef, se=se, label=label))

    plot_fe_specs(irf_data, "Fig11A_fe_robustness.pdf")

    # ==================================================================
    # FIG 11b: Progressive Controls
    # ==================================================================
    print("\n--- Fig 11b: Progressive Controls ---")

    trade_lags = [f"trade_lag{i}" for i in range(1, 5)]
    unrest_lags = [f"unrest_new_lag{i}" for i in range(1, 5)]
    war_lags = [f"{c}_lag{l}" for c in WAR_VARS for l in range(1, 5)]
    inst_lags = [f"{v}_lag{i}" for v in INSTITUTION_VARS for i in range(1, 5)]

    # Progressive accumulation
    specifications = [
        ("Baseline", []),
        ("+ Trade Lags", trade_lags),
        ("+ Unrest Lags", trade_lags + unrest_lags),
        ("+ War Exposure Lags", trade_lags + unrest_lags + war_lags),
        ("+ Institution Lags", trade_lags + unrest_lags + war_lags + inst_lags),
    ]

    # Common sample: countries with all controls non-null
    all_controls = trade_lags + unrest_lags + war_lags + inst_lags
    valid_countries = get_consistent_country_sample(df, extra_controls=all_controls)
    if not valid_countries:
        print("ERROR: No valid countries for common sample.")
        return

    results = {}
    for label, controls in specifications:
        print(f"  Estimating: {label} ...")
        res = lp_irf_balanced(df, valid_countries, extra_controls=controls)
        results[label] = res
        v = res["coef"].notna()
        print(f"    valid horizons: {int(v.sum())}, "
              f"obs range: {res.loc[v, 'n_obs'].min():.0f}-{res.loc[v, 'n_obs'].max():.0f}")

    plot_progressive_controls(results, "Fig11B_control_robustness.pdf")

    print("\nDone. Figures saved to:", FIGURES)


if __name__ == "__main__":
    main()
