"""
09_democracy.py
===============
Figure 14: bilateral alignment with eight major nations after democratization.
Figure 15: GDP on alignment and democracy, separate against joint specifications.

Figure 14: local projection of the bilateral score S_{cj,t+h} with major nation j on the
ANRR democracy indicator, four lags of the score, country and year fixed effects, all
available country-years, h = -10..15 (plotted from h = 0).
Figure 15: local projections of ANRR GDP (y) on the alignment index and democracy, four
lags of GDP, country and year fixed effects, long-coverage sample, h = 0..15:
(a) alignment, separate vs joint with democracy (both with four alignment lags);
(b) democracy, separate (no alignment lags, as ANRR) vs joint (with four alignment lags).
Driscoll-Kraay standard errors throughout. Alignment coefficients are per one SD of the
index; democracy coefficients are per a zero-to-one change.

Full-package version: code/03_extensions/democracy_bilateral_reg.py (Figure 14) and
code/03_extensions/democracy_horserace_reg.py (Figure 15, year fixed effects).
Outputs: output/figures/Fig14{a,b}_*.pdf, Fig15{a,b}_*.pdf,
         output/cache/bilateral_partner_irf.csv, horserace_irf.csv,
         output/numbers/09_democracy.csv
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels.panel import PanelOLS
from matplotlib.ticker import MultipleLocator

import plot_style as PS
from figure_textsize import savefig_aspect, wrap_long_ylabels
from config import CACHE, GEO_SD, PANEL, ensure_dirs, read_csv_fallback, restrict_balanced_sample, save_numbers, savefig

ensure_dirs()
PS.apply_theme()
warnings.filterwarnings("ignore")

LAGS = 4
DEEP = sns.color_palette("deep")
FULL_BILAT = "03_extensions/democracy_bilateral_reg.py"
FULL_HR = "03_extensions/democracy_horserace_reg.py"
NAMES = {"USA": "United States", "GBR": "United Kingdom", "DEU": "Germany", "FRA": "France",
         "RUS": "Russia", "CHN": "China", "IND": "India", "JPN": "Japan"}


def load():
    df = read_csv_fallback(PANEL)
    df["country_code"] = df["country_code"].astype(str)
    return df


# ── Figure 14: bilateral reorientation ────────────────────────────────
def lp_bilateral(df, score, horizons=range(-10, 16)):
    """Coefficient on demANRR in LPs of the bilateral score (country and year FE)."""
    df = df.copy()
    df["country_idx"] = pd.factorize(df["country_code"])[0]
    df["y_ext"] = df[score]
    lags = [f"y_ext_lag{l}" for l in range(1, LAGS + 1)]
    for l in range(1, LAGS + 1):
        df[f"y_ext_lag{l}"] = df.groupby("country_idx")["y_ext"].shift(l)
    h_vals = np.array(list(horizons))
    coef, se = np.full(len(h_vals), np.nan), np.full(len(h_vals), np.nan)
    nobs, n_ctry = np.zeros(len(h_vals), int), np.zeros(len(h_vals), int)
    for i, h in enumerate(h_vals):
        tmp = df.copy()
        tmp["y_ext_dep"] = tmp.groupby("country_idx")["y_ext"].shift(-h)
        sub = tmp.dropna(subset=["y_ext_dep", "demANRR"] + lags + ["country_code", "year", "country_idx"])
        nobs[i], n_ctry[i] = sub.shape[0], sub["country_code"].nunique()
        if sub.shape[0] < len(lags) + sub["country_code"].nunique() + sub["year"].nunique() + 10:
            continue
        sp = sub.set_index(["country_idx", "year"])
        res = PanelOLS(sp["y_ext_dep"], sp[["demANRR"] + lags], entity_effects=True, time_effects=True).fit(cov_type="kernel")
        coef[i], se[i] = res.params["demANRR"], res.std_errors["demANRR"]
    return h_vals, coef, se, nobs, n_ctry


def bilateral(df):
    """Figure 14: one row of four partners per panel, h = 0..15."""
    paths = []
    for partners, filename in [(["USA", "GBR", "DEU", "FRA"], "Fig14a_democratic_partners.pdf"),
                               (["RUS", "CHN", "IND", "JPN"], "Fig14b_other_partners.pdf")]:
        fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.5), sharex=True, sharey=True)
        for i, cc in enumerate(partners):
            h, c, s, nobs, n_ctry = lp_bilateral(df, f"geo_score_dyn_{cc}")
            paths.append(pd.DataFrame({"partner": cc, "horizon": h, "coef": c, "se": s, "ci_lo": c - 1.96 * s,
                                       "ci_hi": c + 1.96 * s, "nobs": nobs, "n_countries": n_ctry}))
            post, ax = h >= 0, axes[i]
            ax.plot(h[post], c[post], label=f"Geopolitical Alignment Score {cc}", color=DEEP[0], linewidth=2.2, marker="o")
            ax.fill_between(h[post], (c - 1.96 * s)[post], (c + 1.96 * s)[post], color=DEEP[0], alpha=0.25)
            ax.axhline(0, **PS.HLINE_KW)
            ax.set_xlabel(PS.IRF_XLABEL, fontsize=11.5)
            if i == 0:
                ax.set_ylabel(PS.IRF_YLABEL_GEO, fontsize=11.5)
            ax.tick_params(axis="both", labelsize=11)
            ax.set_title(NAMES[cc], fontsize=13, fontweight="bold")
            ax.set_xlim(0, 15)
            ax.set_xticks([0, 5, 10, 15])
            ax.set_ylim(-0.08, 0.12)
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            sns.despine(ax=ax)
        fig.tight_layout(pad=0.5, w_pad=1.0)
        print("  Wrapped y-labels:", wrap_long_ylabels(fig, fig.axes))
        fig.tight_layout(pad=0.5, w_pad=1.0)
        savefig_aspect(fig, filename, 13.0 / 3.5)
        plt.close(fig)
    paths = pd.concat(paths, ignore_index=True)
    paths.to_csv(CACHE / "bilateral_partner_irf.csv", index=False)
    return paths


# ── Figure 15: separate against joint specifications ──────────────────
def lp_gdp(df, shocks, geo_lags, horizons=range(16)):
    """Coefficients on shocks in LPs of y with four lags of y (and of the index if geo_lags), country and year FE."""
    lags = [f"y_lag{l}" for l in range(1, LAGS + 1)]
    glags = [f"geo_relation_dyn_lag{l}" for l in range(1, LAGS + 1)] if geo_lags else []
    h_vals = np.array(list(horizons))
    coef = {s: np.full(len(h_vals), np.nan) for s in shocks}
    se = {s: np.full(len(h_vals), np.nan) for s in shocks}
    nobs, n_ctry = np.zeros(len(h_vals), int), np.zeros(len(h_vals), int)
    for i, h in enumerate(h_vals):
        tmp = df.copy()
        tmp["y_dep"] = tmp.groupby("country_idx")["y"].shift(-h)
        sub = tmp.dropna(subset=["y_dep"] + list(shocks) + lags + glags + ["country_code", "country_idx", "year"])
        nobs[i], n_ctry[i] = sub.shape[0], sub["country_code"].nunique()
        if sub.shape[0] < len(shocks) + len(lags) + len(glags) + sub["country_code"].nunique() + sub["year"].nunique() + 10:
            continue
        sp = sub.set_index(["country_idx", "year"])
        res = PanelOLS(sp["y_dep"], sp[list(shocks) + lags + glags], entity_effects=True,
                       time_effects=True).fit(cov_type="kernel")
        for s in shocks:
            coef[s][i], se[s][i] = res.params[s], res.std_errors[s]
    return {s: (h_vals, coef[s], se[s], nobs, n_ctry) for s in shocks}


def plot_comparison(sep, joint, shock, filename, legend_loc):
    """Separate (solid) against joint (dashed) specification."""
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    h_u, c_u, s_u = sep[:3]
    h_j, c_j, s_j = joint[:3]
    if shock == "geo_relation_dyn":                    # per one SD of the index
        c_u, s_u, c_j, s_j = c_u * GEO_SD, s_u * GEO_SD, c_j * GEO_SD, s_j * GEO_SD
    color = DEEP[0] if shock == "geo_relation_dyn" else DEEP[1]
    for h, c, s, label, ls, alpha in [(h_u, c_u, s_u, "Separate Specification", "-", 0.3),
                                      (h_j, c_j, s_j, "Joint Specification", "--", 0.15)]:
        ax.plot(h, c, label=label, color=color, linestyle=ls, linewidth=2.5, marker=None)
        ax.fill_between(h, c - 1.96 * s, c + 1.96 * s, color=color, alpha=alpha)
    ax.axhline(0, **PS.HLINE_KW)
    ax.axvline(0, **PS.VLINE_KW)
    ax.set_xlabel(PS.IRF_XLABEL, fontsize=12)
    ax.set_ylabel(PS.IRF_YLABEL_GDP, fontsize=12)
    ax.set_title("")
    ax.legend(frameon=False, fontsize=10, loc=legend_loc)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    fig.tight_layout(pad=0.5)
    savefig(fig, filename)
    plt.close(fig)


def horse_race(df):
    """Figure 15 on the long-coverage sample."""
    df = restrict_balanced_sample(df)
    df["country_idx"] = pd.factorize(df["country_code"])[0]
    for l in range(1, LAGS + 1):
        df[f"y_lag{l}"] = df.groupby("country_idx")["y"].shift(l)
        df[f"geo_relation_dyn_lag{l}"] = df.groupby("country_idx")["geo_relation_dyn"].shift(l)
    joint = lp_gdp(df, ("geo_relation_dyn", "demANRR"), geo_lags=True)
    paths = {("joint", "geo_relation_dyn"): joint["geo_relation_dyn"],
             ("joint", "demANRR"): joint["demANRR"],
             ("separate", "geo_relation_dyn"): lp_gdp(df, ("geo_relation_dyn",), geo_lags=True)["geo_relation_dyn"],
             ("separate", "demANRR"): lp_gdp(df, ("demANRR",), geo_lags=False)["demANRR"]}
    plot_comparison(paths["separate", "geo_relation_dyn"], paths["joint", "geo_relation_dyn"],
                    "geo_relation_dyn", "Fig15a_alignment_and_democracy.pdf", "upper left")
    plot_comparison(paths["separate", "demANRR"], paths["joint", "demANRR"],
                    "demANRR", "Fig15b_democracy_and_alignment.pdf", "lower right")
    hr = pd.concat([pd.DataFrame({"fe": "year", "spec": spec, "shock": s, "horizon": h, "coef": c, "se": se,
                                  "ci_lo": c - 1.96 * se, "ci_hi": c + 1.96 * se, "nobs": nobs, "n_countries": n})
                    for (spec, s), (h, c, se, nobs, n) in paths.items()], ignore_index=True)
    hr.to_csv(CACHE / "horserace_irf.csv", index=False)
    return hr


# ── Numbers quoted in Section 4.2 ─────────────────────────────────────
def numbers(paths, hr):
    p = paths.set_index(["partner", "horizon"])["coef"]
    post = paths[paths.horizon >= 0]
    peak = lambda cc, h1=15: post[(post.partner == cc) & (post.horizon <= h1)]["coef"].max()
    a, b = "Section 4.2, Figure 14(a)", "Section 4.2, Figure 14(b)"
    rows = [{"key": "usa_h0", "value": p["USA", 0], "paper": a},
            *[{"key": f"usa_h{h}", "value": p["USA", h], "paper": a} for h in (7, 8, 9, 10, 15)],
            *[{"key": f"{cc.lower()}_peak_h0_5", "value": peak(cc, 5), "paper": a} for cc in ("GBR", "DEU", "FRA")],
            *[{"key": f"rus_h{h}", "value": p["RUS", h], "paper": b} for h in (0, *range(10, 16))],
            {"key": "chn_peak", "value": peak("CHN"), "paper": b},
            {"key": "ind_max", "value": peak("IND"), "paper": b},
            {"key": "jpn_max", "value": peak("JPN"), "paper": b},
            {"key": "n_countries_h0", "value": int(paths[paths.horizon == 0]["n_countries"].max()),
             "paper": "Figure 14 note"}]
    for r in rows:
        r["full_ref"] = f"{FULL_BILAT}:{r['key']}"

    path = lambda spec, s: hr[(hr.spec == spec) & (hr.shock == s)].set_index("horizon")["coef"]
    ug, jg = path("separate", "geo_relation_dyn") * GEO_SD, path("joint", "geo_relation_dyn") * GEO_SD
    ud, jd = path("separate", "demANRR"), path("joint", "demANRR")
    a, b = "Section 4.2, Figure 15(a)", "Section 4.2, Figure 15(b)"
    hr_rows = [{"key": "fig15a_separate_peak", "value": ug.max(), "paper": a},
               {"key": "fig15a_joint_peak", "value": jg.max(), "paper": a},
               {"key": "fig15a_joint_h15", "value": jg[15], "paper": a},
               {"key": "fig15b_separate_h0", "value": ud[0], "paper": b},
               {"key": "fig15b_separate_peak", "value": ud.max(), "paper": b},
               {"key": "fig15b_separate_h15", "value": ud[15], "paper": b},
               {"key": "fig15b_joint_peak", "value": jd.max(), "paper": b},
               {"key": "fig15b_joint_h15", "value": jd[15], "paper": b},
               {"key": "fig15b_joint_below_separate", "value": int((jd < ud).sum()), "paper": b}]
    for r in hr_rows:
        r["full_ref"] = f"{FULL_HR}:{r['key']}"
    return rows + hr_rows


def main():
    df = load()
    paths = bilateral(df)
    hr = horse_race(df)
    save_numbers(__file__, numbers(paths, hr))


if __name__ == "__main__":
    main()
