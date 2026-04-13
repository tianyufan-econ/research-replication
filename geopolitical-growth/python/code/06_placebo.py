#!/usr/bin/env python3
"""
06_placebo.py — Placebo / Randomization Tests (Replication)
============================================================
Two placebo exercises, 500 iterations each (~15 min total):

  Design A (within-region-year reassignment):
    For each iteration, randomly shuffle geo_relation_dyn across countries
    WITHIN each region-year cell. Re-run baseline LP. Store IRF (h=0..25).

  Design B (future-year timing):
    For each iteration, replace each country's shock with its own shock from
    8-15 years in the future (randomly drawn offset per country). Re-run LP.

Produces four figures:
  - Fig12A_placebo_region_year.pdf  (IRF spaghetti)
  - Fig12B_placebo_future_timing.pdf  (IRF spaghetti)
  - placebo_A_within_region_year_reassignment_hist.pdf  (histogram)
  - placebo_B_future_year_timing_reassignment_hist.pdf  (histogram)

Also saves cache to output/cache/placebo_500iter.pkl.

Source: 06_placebo/placebo_core.py + placebo_run.py + placebo_plot.ipynb
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, linalg

from config import PANEL, FIGURES, CACHE, ensure_dirs, savefig, read_csv_fallback, restrict_balanced_sample

try:
    import pyhdfe
    HAS_PYHDFE = True
except ImportError:
    HAS_PYHDFE = False
    print("WARNING: pyhdfe not installed. Using PanelOLS for placebo draws (slower).")

from lp_utils import prepare_panel, lp_irf
import plot_style as PS

PS.apply_theme()
ensure_dirs()

# ── Colors (matching working placebo_plot.ipynb) ─────────────────────────
COLORS = {
    "actual":  PS.C_BLUE,
    "placebo": PS.C_ORANGE,
}

# ── Configuration ────────────────────────────────────────────────────────
Y_VAR = "y_ext"
SHOCK_VAR = "geo_relation_dyn"
Y_LAGS = 4
SHOCK_LAGS = 4
H_START = 0
H_END = 25
LEAD_MIN = 8
LEAD_MAX = 15
N_ITER = 500
SEED = 20260331
SUMMARY_H = (0, 10)  # horizons for summary statistic


# =====================================================================
# Data Loading
# =====================================================================

def load_panel():
    """Load the minimal panel needed for placebo exercises."""
    print("Loading panel ...")
    df = read_csv_fallback(PANEL)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)
    df = restrict_balanced_sample(df)
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    df["country_idx"] = pd.factorize(df["country_code"])[0]
    if "region" in df.columns:
        df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    else:
        df["region_year"] = "all_" + df["year"].astype(str)
    return df


# =====================================================================
# Placebo Shock Generators
# =====================================================================

def make_region_year_shuffle(df, rng):
    """Shuffle shock across countries within each region-year cell."""
    out = df.copy()
    placebo = np.empty(len(out), dtype=float)
    for idx_arr in out.groupby("region_year", sort=False).indices.values():
        idx_arr = np.asarray(idx_arr)
        vals = out.iloc[idx_arr][SHOCK_VAR].to_numpy(copy=True)
        rng.shuffle(vals)
        placebo[idx_arr] = vals
    out["shock_placebo"] = placebo
    return out


def make_random_future_lead(df, rng, max_base_year):
    """Assign each country's shock to a random future year 8-15 years ahead."""
    out = df.copy()
    out["shock_placebo"] = np.nan
    for _, chunk in out.groupby("country_code", sort=False):
        lead = int(rng.integers(LEAD_MIN, LEAD_MAX + 1))
        shifted = chunk[SHOCK_VAR].shift(-lead).to_numpy()
        out.loc[chunk.index, "shock_placebo"] = shifted
    out.loc[out["year"] > max_base_year, "shock_placebo"] = np.nan
    return out


# =====================================================================
# Fast LP (pyhdfe) for placebo draws
# =====================================================================

def lp_irf_fast(df, shock_var="shock_placebo"):
    """
    Fast LP-IRF using pyhdfe for FE absorption.
    Point estimates only (no standard errors).
    Falls back to PanelOLS if pyhdfe unavailable.
    """
    y_lag_cols = [f"{Y_VAR}_lag{i}" for i in range(1, Y_LAGS + 1)]
    shock_lag_cols = [f"{shock_var}_lag{i}" for i in range(1, SHOCK_LAGS + 1)]

    h_vals = list(range(H_START, H_END + 1))
    n_h = len(h_vals)
    coef = np.full(n_h, np.nan)
    X_cols = [shock_var] + y_lag_cols + shock_lag_cols

    for idx, h in enumerate(h_vals):
        tmp = df.copy()
        tmp["dep_var"] = tmp.groupby("country_idx")[Y_VAR].shift(-h)
        required = ["dep_var"] + X_cols + ["region_year", "country_idx"]
        sub = tmp.dropna(subset=required)

        if sub.shape[0] < len(X_cols) + 50:
            continue

        try:
            if HAS_PYHDFE:
                fe_ids = np.column_stack([
                    pd.factorize(sub["country_idx"])[0],
                    pd.factorize(sub["region_year"])[0],
                ])
                algo = pyhdfe.create(fe_ids)
                data = sub[["dep_var"] + X_cols].values.astype(np.float64)
                resid = algo.residualize(data)
                y_r = resid[:, 0]
                X_r = resid[:, 1:]
                beta = linalg.solve(X_r.T @ X_r, X_r.T @ y_r)
                coef[idx] = beta[0]
            else:
                from linearmodels.panel import PanelOLS
                sp = sub.set_index(["country_idx", "year"])
                mod = PanelOLS(sp["dep_var"], sp[X_cols],
                               entity_effects=True, time_effects=False,
                               other_effects=sp["region_year"])
                res = mod.fit(cov_type="unadjusted")
                coef[idx] = res.params.get(shock_var, np.nan)
        except Exception:
            continue

    return coef


def prepare_for_fast_lp(df, shock_var="shock_placebo"):
    """Add lags for the placebo shock variable."""
    out = df.copy()
    # y_ext lags (may already exist)
    for l in range(1, Y_LAGS + 1):
        col = f"{Y_VAR}_lag{l}"
        if col not in out.columns:
            out[col] = out.groupby("country_idx")[Y_VAR].shift(l)
    # Shock lags
    for l in range(1, SHOCK_LAGS + 1):
        out[f"{shock_var}_lag{l}"] = out.groupby("country_idx")[shock_var].shift(l)
    return out


# =====================================================================
# Run Placebo Exercises
# =====================================================================

def run_placebo_A(raw, rng_seed):
    """Design A: within-region-year reassignment."""
    print("\n=== Placebo A: Within-Region-Year Reassignment ===")
    rng = np.random.default_rng(rng_seed)

    # Actual IRF
    df_actual = prepare_panel(raw, y_var=Y_VAR, shock_var=SHOCK_VAR,
                              y_lags=Y_LAGS, shock_lags=SHOCK_LAGS)
    actual = lp_irf(df_actual, shock_var=SHOCK_VAR, y_var=Y_VAR,
                    horizon_range=range(H_START, H_END + 1),
                    y_lags=Y_LAGS, shock_lags=SHOCK_LAGS,
                    fe="region_year", cov_type="kernel")

    # Placebo draws
    n_h = H_END - H_START + 1
    placebo_draws = np.full((N_ITER, n_h), np.nan)
    for b in range(N_ITER):
        draw_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        praw = make_region_year_shuffle(raw, draw_rng)
        praw = prepare_for_fast_lp(praw, "shock_placebo")
        placebo_draws[b, :] = lp_irf_fast(praw, "shock_placebo")
        if (b + 1) % 50 == 0 or b == 0 or (b + 1) == N_ITER:
            print(f"  completed {b + 1}/{N_ITER}")

    return actual, placebo_draws


def run_placebo_B(raw, rng_seed):
    """Design B: future-year timing reassignment."""
    print("\n=== Placebo B: Future-Year Timing Reassignment ===")
    rng = np.random.default_rng(rng_seed)

    # Trim actual to years where future leads are defined
    max_base_year = int(raw["year"].max() - LEAD_MAX)
    raw_actual = raw.copy()
    raw_actual["shock_timing_actual"] = raw_actual[SHOCK_VAR].where(
        raw_actual["year"] <= max_base_year, np.nan)

    df_actual = prepare_panel(raw_actual, y_var=Y_VAR,
                              shock_var="shock_timing_actual",
                              y_lags=Y_LAGS, shock_lags=SHOCK_LAGS)
    actual = lp_irf(df_actual, shock_var="shock_timing_actual", y_var=Y_VAR,
                    horizon_range=range(H_START, H_END + 1),
                    y_lags=Y_LAGS, shock_lags=SHOCK_LAGS,
                    fe="region_year", cov_type="kernel")

    # Placebo draws
    n_h = H_END - H_START + 1
    placebo_draws = np.full((N_ITER, n_h), np.nan)
    for b in range(N_ITER):
        draw_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        praw = make_random_future_lead(raw, draw_rng, max_base_year)
        praw = prepare_for_fast_lp(praw, "shock_placebo")
        placebo_draws[b, :] = lp_irf_fast(praw, "shock_placebo")
        if (b + 1) % 50 == 0 or b == 0 or (b + 1) == N_ITER:
            print(f"  completed {b + 1}/{N_ITER}")

    return actual, placebo_draws


# =====================================================================
# Plotting
# =====================================================================

def plot_irf_spaghetti(actual, placebo_draws, title_stub, fname):
    """
    IRF placebo figure: actual estimate + DK CI overlaid on placebo
    percentile band + median. Matches working placebo_plot.ipynb.
    """
    color_p = COLORS["placebo"]
    color_a = COLORS["actual"]

    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    h = actual["h_vals"]

    # Placebo 5th-95th percentile band
    p05 = np.nanpercentile(placebo_draws, 5, axis=0)
    p50 = np.nanpercentile(placebo_draws, 50, axis=0)
    p95 = np.nanpercentile(placebo_draws, 95, axis=0)

    ax.fill_between(
        h, p05, p95,
        color=color_p, alpha=PS.CI_ALPHA, zorder=1,
    )
    ax.plot(h, p05, color=color_p, ls="--",
            lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA, zorder=2)
    ax.plot(h, p95, color=color_p, ls="--",
            lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA, zorder=2)

    # Placebo median
    placebo_line, = ax.plot(
        h, p50, color=color_p, lw=2.5, marker="o", markersize=5.5,
        label="Placebo median", zorder=3,
    )

    # Actual estimate DK CI band
    coef = actual["coef"]
    se = actual["se"]
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se

    ax.fill_between(
        h, ci_lo, ci_hi,
        color=color_a, alpha=0.18, zorder=4,
    )
    ax.plot(h, ci_lo, color=color_a, ls="--",
            lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA, zorder=5)
    ax.plot(h, ci_hi, color=color_a, ls="--",
            lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA, zorder=5)

    # Actual point estimate
    actual_line, = ax.plot(
        h, coef, color=color_a, lw=2.8, marker="o", markersize=6.0,
        label="Actual estimate", zorder=6,
    )

    # Axis styling
    ax.axhline(0, **PS.HLINE_KW)
    ax.axvline(0, **PS.VLINE_KW)
    ax.set_xlim(-0.25, H_END)
    ax.set_xticks(np.arange(0, H_END + 1, 5))
    ax.set_xlabel(PS.IRF_XLABEL, fontsize=12)
    ax.set_ylabel(PS.IRF_YLABEL_GDP, fontsize=12)
    ax.grid(True, alpha=0.25)

    # Legend with just two line handles
    ax.legend(
        handles=[actual_line, placebo_line],
        frameon=False, fontsize=10, loc="upper right",
    )
    plt.setp(ax.get_legend().get_lines(), linewidth=2.8)

    fig.tight_layout()
    savefig(fig, FIGURES, fname)
    plt.close(fig)


def plot_histogram(actual, placebo_draws, title_stub, fname):
    """
    Histogram of placebo average responses vs actual.
    Matches working placebo_plot.ipynb plot_placebo_hist.
    """
    # Compute average IRF over summary horizons
    h_min, h_max = SUMMARY_H
    h_idx = np.arange(h_min - H_START, h_max - H_START + 1)
    h_idx = h_idx[h_idx < placebo_draws.shape[1]]

    placebo_avgs = np.nanmean(placebo_draws[:, h_idx], axis=1)
    actual_avg = np.nanmean(actual["coef"][h_idx])

    vals = placebo_avgs[np.isfinite(placebo_avgs)]

    # Randomization p-value (right-tail)
    extreme = int(np.sum(vals >= actual_avg))
    n_iter = len(vals)
    p_val = (extreme + 1.0) / (n_iter + 1.0)

    # Full x-range: include actual value with padding
    x_lo = min(float(np.nanmin(vals)), actual_avg) - 0.5
    x_hi = max(float(np.nanmax(vals)), actual_avg) + 0.5

    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    ax.hist(vals, bins=40, density=True, color=COLORS["placebo"], alpha=0.45,
            edgecolor="none", label="Placebo Distribution")

    # Actual value as a solid vertical line
    p_text = f"p={p_val:.3f}" if extreme > 0 else f"p<{1/n_iter:.3f}"
    ax.axvline(actual_avg, color=COLORS["actual"], lw=2.4,
               label=f"Actual ({p_text})")

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(f"Average IRF (h={h_min} to {h_max})", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    sns.despine(ax=ax)

    fig.tight_layout()
    savefig(fig, FIGURES, fname)
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    t0 = time.time()
    print("=" * 65)
    print(f"  PLACEBO TESTS ({N_ITER} iterations each) (REPLICATION)")
    print("=" * 65)

    raw = load_panel()
    print(f"Panel: {len(raw):,} rows, {raw['country_code'].nunique()} countries, "
          f"years {raw['year'].min()}-{raw['year'].max()}")

    # Add y_ext lags to raw (shared across exercises)
    for l in range(1, Y_LAGS + 1):
        col = f"{Y_VAR}_lag{l}"
        if col not in raw.columns:
            raw[col] = raw.groupby("country_idx")[Y_VAR].shift(l)

    # Run exercises
    actual_A, draws_A = run_placebo_A(raw, SEED)
    actual_B, draws_B = run_placebo_B(raw, SEED + 1000)

    # Plot IRF spaghetti
    plot_irf_spaghetti(actual_A, draws_A,
                       "Within-Region-Year Reassignment",
                       "Fig12A_placebo_region_year.pdf")
    plot_irf_spaghetti(actual_B, draws_B,
                       "Future-Year Timing Reassignment",
                       "Fig12B_placebo_future_timing.pdf")

    # Save cache
    cache = {
        "actual_A": actual_A,
        "draws_A": draws_A,
        "actual_B": actual_B,
        "draws_B": draws_B,
        "n_iter": N_ITER,
        "seed": SEED,
    }
    cache_path = CACHE / "placebo_500iter.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nCache saved: {cache_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
