"""
06_placebo.py
=============
Figure 11: placebo tests for the GDP response. (a) The index is reassigned across
countries within region-year cells. (b) Each country's index is replaced by its value
8-15 years ahead (one random lead per country and draw), on base years 1960-2009.

Each of the 500 draws per exercise re-estimates the baseline local projection
(h = 0..25, long-coverage sample; point estimates, with the country and region-year
fixed effects absorbed by pyhdfe). The figure overlays the placebo median and
5th-95th percentiles on the realized estimate and its 95% Driscoll-Kraay band; in
panel (b) the realized estimate uses the same 1960-2009 base years. Coefficients are
reported per one standard deviation of the index.

Full-package version: code/06_placebo/placebo_run.py with placebo_core.py (draws) and
placebo_plot.py (figures).
Outputs: output/figures/Fig11{a,b}_*.pdf, output/cache/placebo_irf_summary.csv,
         placebo_irf_draws_long.csv, output/numbers/06_placebo.csv
Usage:   python3 06_placebo.py [--n-iter N]   (default 500, as in the paper; the first
         N draws of a shorter run equal the first N of the full run)
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel.utility import AbsorbingEffectWarning

import plot_style as PS
from config import CACHE, GEO_SD, PANEL, ensure_dirs, read_csv_fallback, restrict_balanced_sample, save_numbers, savefig
from lp_utils import lp_irf, lp_irf_fast, prepare_panel

ensure_dirs()
PS.apply_theme()
warnings.simplefilter("ignore", AbsorbingEffectWarning)

SHOCK = "geo_relation_dyn"
H = range(0, 26)
LEAD_MIN, LEAD_MAX = 8, 15
SEED, N_ITER = 20260331, 500
EXPERIMENTS = {  # key: (label, seed offset)
    "region_year_shuffle": ("Placebo A. Within-Region-Year Reassignment", 0),
    "future_timing": ("Placebo B. Future-Year Timing Reassignment", 1000),
}
FULL = "06_placebo/placebo_plot.py"


def load():
    """Long-coverage sample, sorted by country and year."""
    df = read_csv_fallback(PANEL, usecols=["country_code", "year", "region", "y_ext", SHOCK], low_memory=False)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = restrict_balanced_sample(df)
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    df["country_idx"] = pd.factorize(df["country_code"])[0]
    df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    return df


# ── Placebo assignments ───────────────────────────────────────────────
def region_year_shuffle(df, rng):
    """Shuffle the index across countries within region-year cells."""
    out = df.copy()
    placebo = np.empty(len(out), dtype=float)
    for idx in out.groupby("region_year", sort=False).indices.values():
        idx = np.asarray(idx)
        vals = out.iloc[idx][SHOCK].to_numpy(copy=True)
        rng.shuffle(vals)
        placebo[idx] = vals
    out["shock_placebo"] = placebo
    return out


def random_future_lead(df, rng, max_base_year):
    """Replace each country's index by its value a random 8-15 years ahead."""
    out = df.copy()
    out["shock_placebo"] = np.nan
    for _, chunk in out.groupby("country_code", sort=False):
        lead = int(rng.integers(LEAD_MIN, LEAD_MAX + 1))
        out.loc[chunk.index, "shock_placebo"] = chunk[SHOCK].shift(-lead).to_numpy()
    out.loc[out["year"] > max_base_year, "shock_placebo"] = np.nan
    return out


def run(raw, key, n_iter):
    """Realized estimate (Driscoll-Kraay) and n_iter placebo draws for one exercise."""
    label, offset = EXPERIMENTS[key]
    print(f"\n=== {label} ===")
    rng = np.random.default_rng(SEED + offset)
    actual, shock = raw.copy(), SHOCK
    if key == "future_timing":            # common base-year support 1960-2009 under every lead
        max_base_year = int(raw["year"].max() - LEAD_MAX)
        shock = "shock_timing_actual"
        actual[shock] = actual[SHOCK].where(actual["year"] <= max_base_year, np.nan)
    act = lp_irf(prepare_panel(actual, shock_var=shock), shock_var=shock, horizon_range=H,
                 fe="region_year", cov_type="kernel")

    draws = np.full((n_iter, len(H)), np.nan)
    for b in range(n_iter):
        draw_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        placebo = (random_future_lead(raw.copy(), draw_rng, max_base_year) if key == "future_timing"
                   else region_year_shuffle(raw.copy(), draw_rng))
        draws[b, :] = lp_irf_fast(prepare_panel(placebo, shock_var="shock_placebo"), shock_var="shock_placebo",
                                  horizon_range=H, fe="region_year")["coef"]
        if (b + 1) % 25 == 0 or b == 0 or (b + 1) == n_iter:
            print(f"  completed {b + 1}/{n_iter} draws", flush=True)

    summary = pd.DataFrame({"horizon": act["h_vals"], "actual_coef": act["coef"], "actual_ci_lo": act["ci_lower"],
                            "actual_ci_hi": act["ci_upper"], "actual_n_obs": act["nobs"]})
    summary["placebo_p05"] = np.nanpercentile(draws, 5, axis=0)
    summary["placebo_p95"] = np.nanpercentile(draws, 95, axis=0)
    summary["placebo_p50"] = np.nanpercentile(draws, 50, axis=0)
    summary["experiment"], summary["label"] = key, label

    long = pd.DataFrame(draws, columns=list(H))
    long["experiment"], long["label"], long["iter"] = key, label, np.arange(1, n_iter + 1)
    long = long.melt(id_vars=["experiment", "label", "iter"], var_name="horizon", value_name="coef")
    return summary, long


# ── Figure 11 ─────────────────────────────────────────────────────────
def plot_placebo_irf(df):
    """Realized estimate and DK band over the placebo 5th-95th percentile band and median."""
    actual_c, placebo_c = PS.C_BLUE, PS.C_ORANGE
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    h = df["horizon"].to_numpy()

    ax.fill_between(h, df["placebo_p05"].to_numpy(), df["placebo_p95"].to_numpy(),
                    color=placebo_c, alpha=PS.CI_ALPHA, zorder=1)
    for col in ("placebo_p05", "placebo_p95"):
        ax.plot(h, df[col].to_numpy(), color=placebo_c, ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA, zorder=2)
    placebo_line, = ax.plot(h, df["placebo_p50"].to_numpy(), color=placebo_c, lw=2.5, marker="o", markersize=5.5,
                            label="Placebo median", zorder=3)

    ax.fill_between(h, df["actual_ci_lo"].to_numpy(), df["actual_ci_hi"].to_numpy(),
                    color=actual_c, alpha=0.18, zorder=4)
    for col in ("actual_ci_lo", "actual_ci_hi"):
        ax.plot(h, df[col].to_numpy(), color=actual_c, ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA, zorder=5)
    actual_line, = ax.plot(h, df["actual_coef"].to_numpy(), color=actual_c, lw=2.8, marker="o", markersize=6.0,
                           label="Baseline estimate", zorder=6)

    ax.axhline(0, **PS.HLINE_KW)
    ax.axvline(0, **PS.VLINE_KW)
    ax.set_xlim(-0.25, H[-1])
    ax.set_xticks(np.arange(0, H[-1] + 1, 5))
    ax.set_xlabel(PS.IRF_XLABEL, fontsize=12)
    ax.set_ylabel(PS.IRF_YLABEL_GDP, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(handles=[actual_line, placebo_line], frameon=False, fontsize=10, loc="upper right")
    plt.setp(ax.get_legend().get_lines(), linewidth=2.8)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=N_ITER, help=f"placebo draws per exercise (default {N_ITER})")
    n_iter = ap.parse_args().n_iter

    raw = load()
    out = [run(raw, key, n_iter) for key in EXPERIMENTS]
    pd.concat([o[0] for o in out], ignore_index=True).to_csv(CACHE / "placebo_irf_summary.csv", index=False)
    pd.concat([o[1] for o in out], ignore_index=True).to_csv(CACHE / "placebo_irf_draws_long.csv", index=False)

    # The figures are drawn from the saved summary, as in the full package, then put per one SD.
    irf = pd.read_csv(CACHE / "placebo_irf_summary.csv")
    irf[["actual_coef", "actual_ci_lo", "actual_ci_hi", "placebo_p05", "placebo_p95", "placebo_p50"]] *= GEO_SD
    for key, name in [("region_year_shuffle", "Fig11a_placebo_region_year_reassignment.pdf"),
                      ("future_timing", "Fig11b_placebo_future_year_timing.pdf")]:
        fig = plot_placebo_irf(irf.loc[irf["experiment"] == key].copy())
        savefig(fig, name)
        plt.close(fig)

    a = irf.loc[irf["experiment"] == "region_year_shuffle"].set_index("horizon")
    b = irf.loc[irf["experiment"] == "future_timing"].set_index("horizon")
    sec = "Section 3.3.4, Figure 11"
    numbers = [
        {"key": "placebo_A_n_iter", "value": n_iter, "paper": f"{sec} (note)"},
        {"key": "placebo_A_max_abs_median_irf", "value": a["placebo_p50"].abs().max(), "paper": f"{sec}(a)"},
        {"key": "placebo_B_n_iter", "value": n_iter, "paper": f"{sec} (note)"},
        {"key": "placebo_B_max_abs_median_irf_h0_h5", "value": b.loc[0:5, "placebo_p50"].abs().max(),
         "paper": f"{sec}(b)"},
        {"key": "placebo_B_median_irf_h9", "value": b.loc[9, "placebo_p50"], "paper": f"{sec}(b)"},
    ]
    for r in numbers:
        r["full_ref"] = f"{FULL}:{r['key']}"
    save_numbers(__file__, numbers)


if __name__ == "__main__":
    main()
