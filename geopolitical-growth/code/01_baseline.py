"""
01_baseline.py
==============
Figure 6: (a) persistence of the alignment index, (b) GDP response, long-coverage
sample against all countries.

Local projections of y_{c,t+h} (panel b) or p_{c,t+h} (panel a) on p_{ct}, with four
lags of GDP and of the index, country and region-year fixed effects, and
Driscoll-Kraay standard errors (calendar year as the time index). Coefficients are
reported per one standard deviation of the index.

Full-package version: code/02_baseline/lp_baseline_reg.py (Parts 1 and 2).
Outputs: output/figures/Fig06{a,b}_*.pdf, output/cache/baseline_irf.csv, self_irf.csv,
         output/numbers/01_baseline.csv
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels.panel import PanelOLS

import plot_style as PS
from config import CACHE, GEO_SD, PANEL, ensure_dirs, read_csv_fallback, restrict_balanced_sample, save_numbers, savefig

ensure_dirs()
PS.apply_theme()
warnings.filterwarnings("ignore")

BLUE, ORANGE = sns.color_palette("deep")[0], sns.color_palette("deep")[1]
Z = 1.96
LAGS = 4
FULL = "02_baseline/lp_baseline_reg.py"


def load():
    df = read_csv_fallback(PANEL)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)
    return df


def add_lags(df, id_col):
    """Country index, region-year cell and four lags of GDP and of the index."""
    df = df.copy()
    df["country_idx"] = pd.factorize(df[id_col])[0]
    df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    for l in range(1, LAGS + 1):
        df[f"y_ext_lag{l}"] = df.groupby("country_idx")["y_ext"].shift(l)
        df[f"geo_relation_dyn_lag{l}"] = df.groupby("country_idx")["geo_relation_dyn"].shift(l)
    return df


def lp(df, dep, horizons, id_col="country_code", fe="region_year"):
    """Coefficient on geo_relation_dyn and its Driscoll-Kraay SE at each horizon."""
    shock = "geo_relation_dyn"
    X_cols = [shock] + [f"y_ext_lag{i}" for i in range(1, LAGS + 1)] + [f"{shock}_lag{i}" for i in range(1, LAGS + 1)]
    h_vals = np.array(list(horizons))
    coef, se = np.full(len(h_vals), np.nan), np.full(len(h_vals), np.nan)
    for i, h in enumerate(h_vals):
        tmp = df.copy()
        tmp["dep_var"] = tmp.groupby("country_idx")[dep].shift(-h)
        sub = tmp.dropna(subset=["dep_var"] + X_cols + [id_col, fe, "country_idx", "year"])
        if sub.shape[0] < len(X_cols) + sub[id_col].nunique() + sub[fe].nunique() + 10:
            continue
        sp = sub.set_index(["country_idx", "year"])
        res = PanelOLS(sp["dep_var"], sp[X_cols], entity_effects=True, time_effects=False,
                       other_effects=sp[fe]).fit(cov_type="kernel")
        coef[i], se[i] = res.params[shock], res.std_errors[shock]
    return h_vals, coef, se


def draw_band(ax, h, c, s, color):
    lo, hi = c - Z * s, c + Z * s
    ok = ~np.isnan(c) & ~np.isnan(lo) & ~np.isnan(hi)
    ax.fill_between(h[ok], lo[ok], hi[ok], color=color, alpha=PS.CI_ALPHA, label=PS.LEGEND_CI_DK)
    for b in (lo, hi):
        ax.plot(h[ok], b[ok], color=color, ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
    return lo, hi


def finish(ax, ylabel):
    ax.axhline(0, **PS.HLINE_KW)
    ax.axvline(0, **PS.VLINE_KW)
    ax.set_xlabel(PS.IRF_XLABEL, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)


def gdp_response(df_full, numbers):
    """Figure 6(b): GDP response, long-coverage sample (148 countries) and all countries."""
    H = range(-10, 26)
    h_u, c_u, s_u = lp(add_lags(df_full, "country_code"), "y_ext", H)
    df_bal = restrict_balanced_sample(df_full.copy())
    h_b, c_b, s_b = lp(add_lags(df_bal, "country_code"), "y_ext", H)

    rows = [{"horizon": h, "coef": c, "se": s, "ci_lo": c - Z * s, "ci_hi": c + Z * s, "spec_label": lab}
            for lab, hh, cc, ss in [("balanced", h_b, c_b, s_b), ("unbalanced", h_u, c_u, s_u)]
            for h, c, s in zip(hh, cc, ss)]
    pd.DataFrame(rows).to_csv(CACHE / "baseline_irf.csv", index=False)

    c_b, s_b, c_u = c_b * GEO_SD, s_b * GEO_SD, c_u * GEO_SD          # per one SD of the index
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    ok = ~np.isnan(c_b)
    ax.plot(h_b[ok], c_b[ok], color=BLUE, ls="-", lw=2.5, marker="o", ms=6, label="Long-Coverage Sample")
    lo, hi = draw_band(ax, h_b, c_b, s_b, BLUE)
    ok_u = ~np.isnan(c_u)
    ax.plot(h_u[ok_u], c_u[ok_u], color=ORANGE, ls="--", lw=2.0, marker="s", ms=5, label="Unbalanced Panel")
    finish(ax, PS.IRF_YLABEL_GDP)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    savefig(fig, "Fig06b_gdp_response.pdf")
    plt.close(fig)

    b, l, u = dict(zip(h_b, c_b)), dict(zip(h_b, lo)), dict(zip(h_b, hi))
    post = h_b >= 0
    h_peak = int(h_b[post][np.nanargmax(c_b[post])])
    pre = h_b < 0
    sec = "Section 3.2, Figure 6(b)"
    numbers += [
        {"key": "baseline_coef_h0", "value": b[0], "paper": sec},
        {"key": "baseline_peak_coef", "value": b[h_peak], "paper": sec},
        {"key": "baseline_peak_h", "value": h_peak, "paper": "Introduction and Section 3.2"},
        {"key": "baseline_coef_h20", "value": b[20], "paper": sec},
        {"key": "baseline_ci_lo_h20", "value": l[20], "paper": "Introduction"},
        {"key": "baseline_ci_hi_h20", "value": u[20], "paper": "Introduction"},
        {"key": "baseline_n_pre_horizons_ci_excl_zero", "value": int(np.sum(pre & ((lo > 0) | (hi < 0)))),
         "paper": "Section 3.5"},
        {"key": "n_countries_long_coverage", "value": df_bal["country_code"].nunique(), "paper": "Section 3.1"},
    ]


def persistence(df_full, numbers):
    """Figure 6(a): response of the index to its own change, h = 0..20 (panel keyed on country and regYr)."""
    df = df_full.copy()
    df["country"] = df["country"].astype(str)
    df["regYr"] = df["regYr"].astype(str)
    df = restrict_balanced_sample(df, y_var="y_ext")
    h, c, s = lp(add_lags(df, "country"), "geo_relation_dyn", range(0, 21), id_col="country", fe="regYr")
    pd.DataFrame({"horizon": h, "coef": c, "se": s, "ci_lo": c - Z * s, "ci_hi": c + Z * s,
                  "spec_label": "self_irf_dk"}).to_csv(CACHE / "self_irf.csv", index=False)

    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    ok = ~np.isnan(c)
    ax.plot(h[ok], c[ok], label=PS.LEGEND_POINT, color=BLUE, linestyle="-", linewidth=2.5, marker="o", markersize=6)
    draw_band(ax, h, c, s, BLUE)
    finish(ax, "Geopolitical Alignment Index")
    ax.set_xticks(np.arange(0, int(np.nanmax(h)) + 1, 5))
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    savefig(fig, "Fig06a_persistence.pdf")
    plt.close(fig)

    d = dict(zip(h, c))
    sec = "Section 3.2, Figure 6(a)"
    numbers += [{"key": f"self_irf_h{k}", "value": d[k], "paper": sec} for k in (4, 10, 14, 20)]
    numbers.append({"key": "self_irf_all_negative_h15_20", "value": int(all(d[k] < 0 for k in range(15, 21))),
                    "paper": sec})


def main():
    df = load()
    numbers = []
    gdp_response(df, numbers)
    persistence(df, numbers)
    for r in numbers:
        r["full_ref"] = f"{FULL}:{r['key']}"
    save_numbers(__file__, numbers)


if __name__ == "__main__":
    main()
