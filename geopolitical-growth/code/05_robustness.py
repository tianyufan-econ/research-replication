"""
05_robustness.py
================
Figure 10: robustness to alternative specifications. (a) alternative fixed effects,
(b) cumulatively added controls on a common 98-country sample.

Both panels are the baseline local projection (four lags of GDP and of the index,
country fixed effects, Driscoll-Kraay standard errors) on the long-coverage sample.
(a) replaces the region-year effects by year, initial GDP quintile-year, region-current
regime-year or region-initial regime-year effects; rows with a missing cell code drop out.
(b) keeps region-year effects and adds, cumulatively, four lags of growth fundamentals,
of unrest and five war-exposure measures, and of five V-Dem indices, then the current
values of all of them. All five paths of (b) use identical observations: the countries
with at least 10 usable observations at h = -10 and h = 25 under the fullest
specification and, within them, the country-years where every control is observed.
Coefficients are reported per one standard deviation of the index.

Full-package versions: code/02_baseline/lp_robust_fe_reg.py (panel a) and
code/05_robustness/combined_controls.py with its helper war_controls.py (panel b).
Outputs: output/figures/Fig10{a,b}_*.pdf, output/cache/robust_fe_specs_irf.csv,
         lp_combined_control_specs.csv, output/numbers/05_robustness.csv
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

BLUE = sns.color_palette("deep")[0]
DEEP4 = sns.color_palette("deep", n_colors=4)
Z = 1.96
LAGS = 4
H = range(-10, 26)
SHOCK = "geo_relation_dyn"
BASE_X = [SHOCK] + [f"y_ext_lag{i}" for i in range(1, LAGS + 1)] + [f"{SHOCK}_lag{i}" for i in range(1, LAGS + 1)]
FULL_FE = "02_baseline/lp_robust_fe_reg.py"
FULL_CTRL = "05_robustness/combined_controls.py"

FE_SPECS = [("region_year", "Region-Year FE"), ("year_str", "Year FE"), ("iniGDP_yr", "Initial GDP-Year FE"),
            ("regRegYr", "Region-Regime-Year FE"), ("regIniregYr", "Region-Initial Regime-Year FE")]
FE_LEGEND = {"Region-Year FE": "Region-Year FE (Baseline)",
             "Initial GDP-Year FE": "Initial GDP Quintile-Year FE",
             "Region-Regime-Year FE": "Region-Current Regime-Year FE"}

GROWTH_FUNDAMENTALS = ["trade", "logpop", "pop014share", "pop1564share", "csh_i", "log_k", "log_hc"]
WAR_VARS = ["war_site_onset_all", "war_site_caspop_all", "war_trade_caspop_weighted_all",
            "war_prox_caspop_weighted_all", "war_exposure_site_count_all"]
INSTITUTION_VARS = ["v2x_polyarchy", "v2x_liberal", "v2x_partip", "v2xdl_delib", "v2x_egal"]


def load():
    df = read_csv_fallback(PANEL, low_memory=False)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)
    return df


def add_lags(df, cols):
    """Four within-country lags of each column."""
    for c in cols:
        for l in range(1, LAGS + 1):
            df[f"{c}_lag{l}"] = df.groupby("country_idx")[c].shift(l)
    return df


# ── Panel (a): alternative fixed effects ──────────────────────────────
def lp_fe(df, fe):
    """Coefficient, Driscoll-Kraay SE and sample size at each horizon, absorbing country and `fe` effects."""
    h_vals = np.array(list(H))
    coef, se = np.full(len(h_vals), np.nan), np.full(len(h_vals), np.nan)
    n_obs, n_countries = [], []
    for i, h in enumerate(h_vals):
        tmp = df.copy()
        tmp["dep_var"] = tmp.groupby("country_idx")["y_ext"].shift(-h)
        sub = tmp.dropna(subset=["dep_var"] + BASE_X + ["country_code", fe, "country_idx", "year"])
        n_obs.append(sub.shape[0])
        n_countries.append(sub["country_code"].nunique())
        if sub.shape[0] < len(BASE_X) + sub["country_code"].nunique() + sub[fe].nunique() + 10:
            continue
        sp = sub.set_index(["country_idx", "year"])
        res = PanelOLS(sp["dep_var"], sp[BASE_X], entity_effects=True, time_effects=False,
                       other_effects=sp[fe]).fit(cov_type="kernel")
        coef[i], se[i] = res.params[SHOCK], res.std_errors[SHOCK]
    return h_vals, coef, se, np.array(n_obs), np.array(n_countries)


def plot_fe(specs):
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    base = specs[0]
    h, c = base["h"], base["coef"]
    lo, hi = c - Z * base["se"], c + Z * base["se"]
    ok = ~np.isnan(c) & (h >= 0)
    ax.plot(h[ok], c[ok], label=FE_LEGEND[base["label"]], color=BLUE, linestyle="-", linewidth=2.5,
            marker="o", markersize=6)
    ci = ~(np.isnan(lo) | np.isnan(hi)) & ok
    ax.fill_between(h[ci], lo[ci], hi[ci], color=BLUE, alpha=PS.CI_ALPHA, label="95% CI (Region-Year)")
    for b in (lo, hi):
        ax.plot(h[ci], b[ci], color=BLUE, linestyle="--", linewidth=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
    for k, s in enumerate(specs[1:]):
        ok = ~np.isnan(s["coef"]) & (s["h"] >= 0)
        ax.plot(s["h"][ok], s["coef"][ok], label=FE_LEGEND.get(s["label"], s["label"]), color=DEEP4[k],
                linestyle="--", linewidth=2, marker="s", markersize=5)
    ax.axhline(0, **PS.HLINE_KW)
    ax.axvline(0, **PS.VLINE_KW)
    ax.set_xlabel("Horizon (years)", fontsize=12)
    ax.set_ylabel(PS.IRF_YLABEL_GDP, fontsize=12)
    ax.set_xlim(0, 25)
    ax.set_xticks(np.arange(0, 26, 5))
    ax.legend(frameon=False, fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    savefig(fig, "Fig10a_fixed_effects.pdf")
    plt.close(fig)


def fixed_effects(df_full, numbers):
    """Figure 10(a): five fixed-effects structures on the 148-country long-coverage sample."""
    n_quintile = df_full.dropna(subset=["iniGDP_yr"])["country_code"].nunique()
    df = restrict_balanced_sample(df_full.copy())
    df["country_idx"] = pd.factorize(df["country_code"])[0]
    df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    df["year_str"] = df["year"].astype(str)
    df = add_lags(df, ["y_ext", SHOCK])

    specs = []
    for fe, label in FE_SPECS:
        h, c, s, n, nc = lp_fe(df, fe)
        specs.append({"label": label, "h": h, "coef": c, "se": s, "n_obs": n, "n_countries": nc})
    pd.DataFrame([{"horizon": h, "coef": c, "se": s if sp["label"] == "Region-Year FE" else np.nan,
                   "spec_label": sp["label"], "n_obs": n, "n_countries": nc}
                  for sp in specs
                  for h, c, s, n, nc in zip(sp["h"], sp["coef"], sp["se"], sp["n_obs"], sp["n_countries"])]
                 ).to_csv(CACHE / "robust_fe_specs_irf.csv", index=False)

    for sp in specs:                                                   # per one SD of the index
        sp["coef"], sp["se"] = sp["coef"] * GEO_SD, sp["se"] * GEO_SD
    plot_fe(specs)

    for sp in specs:
        post = sp["h"] >= 0
        tag = sp["label"].replace(" ", "_").replace("-", "_")
        numbers += [
            {"key": f"fe_peak_{tag}", "value": np.nanmax(sp["coef"][post]), "paper": "Section 3.3.3, Figure 10(a)"},
            {"key": f"fe_n_countries_{tag}", "value": int(np.max(sp["n_countries"])), "paper": "Figure 10 note"},
            {"key": f"fe_n_obs_h0_{tag}", "value": int(sp["n_obs"][list(sp["h"]).index(0)]),
             "paper": "Figure 10 note (effective sample at h = 0)"},
        ]
    numbers.append({"key": "n_countries_1960_gdp_quintile", "value": n_quintile,
                    "paper": "Section 3.3.3, footnote on initial GDP quintiles"})
    for r in numbers:
        r["full_ref"] = f"{FULL_FE}:{r['key']}"


# ── Panel (b): cumulatively added controls ────────────────────────────
def controls_panel(df_full):
    """Long-coverage panel with war exposure filled with 0 and four lags of every control."""
    df = restrict_balanced_sample(df_full.copy())
    df = df[df["year"] >= 1960].copy()
    df[WAR_VARS] = df[WAR_VARS].fillna(0.0)                            # country-years without war records
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    df["country_idx"] = pd.factorize(df["country_code"])[0]
    df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    df = add_lags(df, ["y_ext", SHOCK] + WAR_VARS)
    for col in ["trade", "unrest_new", "logpop", "pop014share", "pop1564share", "csh_i", "rnna", "hc"] + INSTITUTION_VARS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["log_k"] = np.log(df["rnna"].where(df["rnna"] > 0)) * 100
    df["log_hc"] = np.log(df["hc"].where(df["hc"] > 0)) * 100
    return add_lags(df, GROWTH_FUNDAMENTALS + ["unrest_new"] + INSTITUTION_VARS)


def common_countries(df, controls):
    """Countries with >= 10 usable observations at both h = -10 and h = 25."""
    ok = None
    for h in (H[0], H[-1]):
        dep = df.groupby("country_idx")["y_ext"].shift(-h)
        usable = dep.notna() & df[["region_year"] + BASE_X + controls].notna().all(axis=1)
        enough = usable.groupby(df["country_code"]).sum() >= 10
        ok = enough if ok is None else ok & enough
    return set(ok.index[ok])


def lp_controls(df, controls):
    """LP with country and region-year effects; constant regressors are dropped at each horizon."""
    x_cols = BASE_X + controls
    rows = []
    for h in H:
        tmp = df.copy()
        tmp["_dep"] = tmp.groupby("country_idx")["y_ext"].shift(-h)
        sub = tmp.dropna(subset=["_dep", "region_year", "country_idx", "year"] + x_cols)
        panel = sub.set_index(["country_idx", "year"])
        X = panel[x_cols]
        X = X[[c for c in X.columns if X[c].nunique(dropna=False) > 1]]
        res = PanelOLS(dependent=panel["_dep"], exog=X, entity_effects=True, time_effects=False,
                       other_effects=panel["region_year"], drop_absorbed=True, check_rank=False).fit(cov_type="kernel")
        c, s = res.params[SHOCK], res.std_errors[SHOCK]
        rows.append({"horizon": h, "coef": c, "se": s, "ci_lo": c - 1.96 * s, "ci_hi": c + 1.96 * s,
                     "n_obs": int(len(sub)), "n_countries": int(sub["country_code"].nunique())})
    return pd.DataFrame(rows)


def plot_controls(results):
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    base = results["Baseline"]
    ok = base["coef"].notna() & base["horizon"].ge(0)
    ax.plot(base.loc[ok, "horizon"], base.loc[ok, "coef"], color=PS.C_BLUE, lw=2.5, ls="-", marker="o", ms=6,
            label="Baseline")
    ci = base["ci_lo"].notna() & base["ci_hi"].notna() & ok
    ax.fill_between(base.loc[ci, "horizon"], base.loc[ci, "ci_lo"], base.loc[ci, "ci_hi"], color=PS.C_BLUE,
                    alpha=PS.CI_ALPHA, label=PS.LEGEND_CI_DK)
    for b in ("ci_lo", "ci_hi"):
        ax.plot(base.loc[ci, "horizon"], base.loc[ci, b], color=PS.C_BLUE, ls="--", lw=PS.CI_BOUND_LW,
                alpha=PS.CI_BOUND_ALPHA)
    for label, color, ls in [("+ Growth Fundamentals", PS.C_ORANGE, "--"),
                             ("+ Unrest and War Exposure", PS.C_GREEN, "--"),
                             ("+ Institutions", PS.C_RED, "--"),
                             ("+ Current Values", sns.color_palette("deep")[4], ":")]:
        res = results[label]
        ok = res["coef"].notna() & res["horizon"].ge(0)
        ax.plot(res.loc[ok, "horizon"], res.loc[ok, "coef"], color=color, lw=2.0, ls=ls, marker="s", ms=5,
                label=label)
    PS.style_irf_ax(ax, xlabel=PS.IRF_XLABEL, ylabel=PS.IRF_YLABEL_GDP)
    ax.set_xlim(0, 25)
    ax.set_xticks(np.arange(0, 26, 5))
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    savefig(fig, "Fig10b_additional_controls.pdf")
    plt.close(fig)


def added_controls(df_full, numbers):
    """Figure 10(b): baseline and four cumulative control sets on identical observations."""
    df = controls_panel(df_full)
    lags = lambda cols: [f"{c}_lag{l}" for c in cols for l in range(1, LAGS + 1)]
    fund, conflict, inst = lags(GROWTH_FUNDAMENTALS), lags(["unrest_new"]) + lags(WAR_VARS), lags(INSTITUTION_VARS)
    current = GROWTH_FUNDAMENTALS + ["unrest_new"] + WAR_VARS + INSTITUTION_VARS
    specs = [("Baseline", []),
             ("+ Growth Fundamentals", fund),
             ("+ Unrest and War Exposure", fund + conflict),
             ("+ Institutions", fund + conflict + inst),
             ("+ Current Values", fund + conflict + inst + current)]
    full = specs[-1][1]

    # identical observations: the shock is set to missing where any control is missing
    df.loc[~df[full].notna().all(axis=1), SHOCK] = np.nan
    keep = common_countries(df, full)
    sample = df[df["country_code"].isin(keep)].copy()
    results = {label: lp_controls(sample, ctrl) for label, ctrl in specs}
    pd.concat([r.assign(specification=label) for label, r in results.items()], ignore_index=True
              ).to_csv(CACHE / "lp_combined_control_specs.csv", index=False)

    for r in results.values():                                         # per one SD of the index
        r[["coef", "se", "ci_lo", "ci_hi"]] = r[["coef", "se", "ci_lo", "ci_hi"]] * GEO_SD
    plot_controls(results)

    sec = "Section 3.3.3, Figure 10(b)"
    rows = [{"key": "n_countries_common_sample", "value": len(keep), "paper": "Section 3.3.3 and Figure 10 note"}]
    for label, r in results.items():
        rows.append({"key": f"peak_coef[{label}]", "value": r.loc[r["horizon"].ge(0), "coef"].max(), "paper": sec})
    cur = results["+ Current Values"].set_index("horizon")["coef"]
    rows += [{"key": f"current_values_h{h}", "value": cur[h], "paper": sec} for h in (5, 10)]
    for r in rows:
        r["full_ref"] = f"{FULL_CTRL}:{r['key']}"
    numbers += rows


def main():
    df = load()
    numbers = []
    fixed_effects(df, numbers)
    added_controls(df, numbers)
    save_numbers(__file__, numbers)


if __name__ == "__main__":
    main()
