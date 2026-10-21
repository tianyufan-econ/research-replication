"""
08_channels.py
==============
Figure 13: growth correlates of the alignment index. (a) PWT GDP per capita, domestic
unrest, investment share and capital stock; (b) internal rate of return, TFP, trade
openness and human capital.

Local projection of each outcome m_{c,t+h} on p_{ct} with four lags of the outcome, of
GDP and of the index, country and region-year fixed effects, and Driscoll-Kraay standard
errors. Each outcome uses the countries with data at both extreme horizons (h = -10 and
h = 25). The panels plot h = 0..25 per one standard deviation of the index.

Full-package version: code/03_extensions/lp_covariates_reg.py (estimates, Figure 13
outcomes) and code/03_extensions_plot/plot_channels_main.py (figure).
Outputs: output/figures/Fig13{a,b}_*.pdf, output/cache/channel_{outcome}.csv,
         output/numbers/08_channels.csv
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels.panel import PanelOLS
from matplotlib.ticker import MaxNLocator, MultipleLocator

import plot_style as PS
from figure_textsize import savefig_aspect, wrap_long_ylabels
from config import CACHE, GEO_SD, PANEL, ensure_dirs, read_csv_fallback, save_numbers, savefig

ensure_dirs()
PS.apply_theme()
warnings.filterwarnings("ignore")

LAGS = 4
H = range(-10, 26)
FULL = "03_extensions/lp_covariates_reg.py"

# (outcome, panel title); log_k, log_tfp and log_hc are 100 x log of rnna, rtfpna and hc
CHANNELS = [
    ("yPWT", "Log Real GDP per Capita (PWT)"),
    ("unrest_new", "Domestic Unrest"),
    ("csh_i", "Investment as Share of GDP"),
    ("log_k", "Log Capital Stock"),
    ("irr", "Internal Rate of Return"),
    ("log_tfp", "Log TFP"),
    ("trade", "Trade as Share of GDP"),
    ("log_hc", "Log Human Capital Index"),
]
SCALE = {"csh_i": 100, "trade": 100, "irr": 100}     # shares (0-1) reported in percentage points
FIGSIZE = (7.3, 5.2)
ASPECT = {"a": 1000.057 / 693.805, "b": 1000.336 / 693.805}   # page aspect of the paper's files


def load():
    """Panel with country index, region-year cell, the log outcomes and the lags of GDP and the index."""
    df = read_csv_fallback(PANEL)
    df["country_idx"] = pd.factorize(df["country_code"])[0]
    df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    for col, src in [("log_k", "rnna"), ("log_tfp", "rtfpna"), ("log_hc", "hc")]:
        df[col] = np.log(df[src].replace(0, np.nan)) * 100
    df["trade"] = df["trade"] / 100                    # percent of GDP -> share
    for l in range(1, LAGS + 1):
        df[f"y_ext_lag{l}"] = df.groupby("country_idx")["y_ext"].shift(l)
        df[f"geo_relation_dyn_lag{l}"] = df.groupby("country_idx")["geo_relation_dyn"].shift(l)
    return df


def lp_channel(df, dep):
    """LP of dep on geo_relation_dyn over the countries with data at h = -10 and h = 25."""
    shock = "geo_relation_dyn"
    for l in range(1, LAGS + 1):
        df[f"{dep}_lag{l}"] = df.groupby("country_idx")[dep].shift(l)
    X_cols = ([shock] + [f"{dep}_lag{l}" for l in range(1, LAGS + 1)]
              + [f"{shock}_lag{l}" for l in range(1, LAGS + 1)] + [f"y_ext_lag{l}" for l in range(1, LAGS + 1)])

    def complete(d, h):
        d = d.copy()
        d["dep_shift"] = d.groupby("country_idx")[dep].shift(-h)
        return d.dropna(subset=["dep_shift"] + X_cols + ["country_code", "region_year", "country_idx", "year"])

    valid = set.intersection(*(set(complete(df, h)["country_code"]) for h in (min(H), max(H))))
    sample = df[df["country_code"].isin(valid)].copy()
    h_vals = np.array(list(H))
    coef, se = np.full(len(h_vals), np.nan), np.full(len(h_vals), np.nan)
    nobs, n_ctry = [], []
    for i, h in enumerate(h_vals):
        sub = complete(sample, h)
        nobs.append(sub.shape[0])
        n_ctry.append(sub["country_code"].nunique())
        if sub.shape[0] < 50:
            continue
        sp = sub.set_index(["country_idx", "year"])
        res = PanelOLS(sp["dep_shift"], sp[X_cols], entity_effects=True, time_effects=False,
                       other_effects=sp["region_year"]).fit(cov_type="kernel")
        coef[i], se[i] = res.params[shock], res.std_errors[shock]
    return h_vals, coef, se, nobs, n_ctry, len(valid)


def estimate():
    """Estimate each outcome and cache its path in plotted units (per unit of the index)."""
    df = load()
    for col, _ in CHANNELS:
        h, c, s, nobs, n_ctry, n_sample = lp_channel(df, col)
        k = SCALE.get(col, 1)
        pd.DataFrame({"horizon": h, "coef": c * k, "se": s * k, "ci_lo": (c - 1.96 * s) * k,
                      "ci_hi": (c + 1.96 * s) * k, "label": col, "nobs": nobs, "n_countries": n_ctry,
                      "n_countries_sample": n_sample}).to_csv(CACHE / f"channel_{col}.csv", index=False)
        print(f"  {col:12} countries: {n_sample}")


def read_cache(col):
    """Cached path per one SD of the index, indexed by horizon."""
    d = pd.read_csv(CACHE / f"channel_{col}.csv").sort_values("horizon")
    for f in ("coef", "se", "ci_lo", "ci_hi"):
        d[f] = d[f] * GEO_SD
    return d


def plot_panel(series, tag, filename, colors):
    """One 2x2 panel of Figure 13, h = 0..25."""
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    for ax, (ci, col, title) in zip(axes.flat, series):
        d = read_cache(col)
        d = d.loc[d["horizon"].between(0, 25)]
        h = d["horizon"].to_numpy()
        ax.plot(h, d["coef"], color=colors[ci], lw=2, marker="o")
        ax.fill_between(h, d["ci_lo"], d["ci_hi"], color=colors[ci], alpha=PS.CI_ALPHA)
        ax.axhline(0, **PS.HLINE_KW)
        ax.axvline(0, **PS.VLINE_KW)
        ax.set_xlim(0, 25)
        ax.set_xlabel(PS.IRF_XLABEL, fontsize=12.5)
        ax.set_ylabel(PS.CHANNEL_YLABELS.get(col, PS.IRF_YLABEL_GDP), fontsize=12.5)
        ax.set_title(f"{title}\n(N countries: {int(d['n_countries_sample'].iloc[0])})", fontsize=12.5, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12.5)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    fig.tight_layout()
    print("  Wrapped y-labels:", wrap_long_ylabels(fig, axes.flat))
    fig.tight_layout()
    savefig_aspect(fig, filename, ASPECT[tag])
    plt.close(fig)


def numbers():
    """Numbers quoted in Section 4.1, from the cached paths per one SD."""
    c = {col: read_cache(col).set_index("horizon") for col, _ in CHANNELS}
    sec, note = "Section 4.1", "Figure 13 (panel titles and note)"
    rows = [{"key": f"n_countries_{col}", "value": int(c[col].n_countries_sample.iloc[0]), "paper": note}
            for col, _ in CHANNELS]
    t = c["trade"]
    rows.append({"key": "trade_horizons_fewer_countries", "value": len(t.index[t.n_countries < t.n_countries_sample]),
                 "paper": "Figure 13 note"})
    hc = c["log_hc"].loc[1:25]
    rows += [
        {"key": "yPWT_h15", "value": c["yPWT"].coef[15], "paper": f"{sec}, Figure 13(a)"},
        {"key": "unrest_h0", "value": c["unrest_new"].coef[0], "paper": f"{sec}, Figure 13(a)"},
        {"key": "irr_h1", "value": c["irr"].coef[1], "paper": f"{sec}, Figure 13(b)"},
        {"key": "irr_h2", "value": c["irr"].coef[2], "paper": f"{sec}, Figure 13(b)"},
        {"key": "irr_ci_hi_max_h21_25", "value": c["irr"].ci_hi.loc[21:25].max(), "paper": f"{sec}, Figure 13(b)"},
        {"key": "csh_i_h1", "value": c["csh_i"].coef[1], "paper": f"{sec}, Figure 13(a)"},
        {"key": "csh_i_h2", "value": c["csh_i"].coef[2], "paper": f"{sec}, Figure 13(a)"},
        {"key": "log_k_h8_12_min", "value": c["log_k"].coef.loc[8:12].min(), "paper": f"{sec}, Figure 13(a)"},
        {"key": "log_k_h8_12_max", "value": c["log_k"].coef.loc[8:12].max(), "paper": f"{sec}, Figure 13(a)"},
        {"key": "log_tfp_h4", "value": c["log_tfp"].coef[4], "paper": f"{sec}, Figure 13(b)"},
        {"key": "log_tfp_min_h2_10", "value": c["log_tfp"].coef.loc[2:10].min(), "paper": f"{sec}, Figure 13(b)"},
        {"key": "trade_h0", "value": c["trade"].coef[0], "paper": f"{sec}, Figure 13(b)"},
        {"key": "trade_max_h16_25", "value": c["trade"].coef.loc[16:25].max(), "paper": f"{sec}, Figure 13(b)"},
        {"key": "log_hc_significant_post_impact", "value": int(((hc.ci_lo > 0) | (hc.ci_hi < 0)).sum()),
         "paper": f"{sec}, Figure 13(b)"},
    ]
    for r in rows:
        r["full_ref"] = f"{FULL}:{r['key']}"
    return rows


def main():
    estimate()
    colors = sns.color_palette("deep")
    indexed = [(i, col, title) for i, (col, title) in enumerate(CHANNELS)]
    plot_panel(indexed[:4], "a", "Fig13a_output_stability.pdf", colors)
    plot_panel(indexed[4:], "b", "Fig13b_growth_fundamentals.pdf", colors)
    save_numbers(__file__, numbers())


if __name__ == "__main__":
    main()
