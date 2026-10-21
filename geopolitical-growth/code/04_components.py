"""
04_components.py
================
Figure 9: (a) GDP response to the three event-domain components of the index entered
jointly, (b) the baseline against the index rebuilt from formal acts only.

Panel (a): one local projection with the economic (category A), diplomatic-political (B)
and security (C and D) components entered jointly, with four lags of GDP and of each
component. A footnote check adds the multilateral-governance and other-events component
(E and F) as a fourth regressor. Panel (b): the baseline local projection with the index replaced by the
formal-acts index, scored by the model (solid) or at the median Goldstein value of each
act's CAMEO code (dashed). Both panels: long-coverage sample, country and region-year
fixed effects, Driscoll-Kraay standard errors, horizons 0..25, coefficients per one SD
of the aggregate index (the components share its scale).

Full-package version: code/07_decomposition/component_decomposition.py (panel a; its
individual and residualized specifications are not in the paper) and
code/07_decomposition/formal_acts.py (panel b; its strict-set and assault-event variants
are not in the paper).
Outputs: output/figures/Fig09{a,b}_*.pdf, output/cache/component_irf.csv,
         component_irf_fourway.csv, formal_acts_irf.csv, output/numbers/04_components.csv
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from linearmodels.panel import PanelOLS

import plot_style as PS
from config import CACHE, GEO_SD, PANEL, ensure_dirs, read_csv_fallback, restrict_balanced_sample, save_numbers, savefig
from lp_utils import lp_irf, prepare_panel

ensure_dirs()
PS.apply_theme()
warnings.filterwarnings("ignore")

H = np.arange(0, 26)
LAGS = 4
Z_95 = stats.norm.ppf(0.975)
COMPONENTS = ["econ", "diplo", "security"]
COMP_LABELS = {"econ": "Economic", "diplo": "Diplomatic", "security": "Security"}
OTHER = "other"      # multilateral governance (E) and other events (F): footnote check only
COMP_COLORS = {"econ": PS.C_BLUE, "diplo": PS.C_ORANGE, "security": PS.C_GREEN}
FORMAL = ["geo_relation_dyn", "formal_relation_dyn", "formal_llm_relation_dyn"]
FULL_A = "07_decomposition/component_decomposition.py"
FULL_B = "07_decomposition/formal_acts.py"


def load():
    df = read_csv_fallback(PANEL, low_memory=False)
    df["country_code"] = df["country_code"].astype(str)
    return df


# ── Panel (a): event-domain components ────────────────────────────────
def component_panel(df):
    """Long-coverage panel with region-year cells and lags (component_decomposition.prepare_panel)."""
    panel = restrict_balanced_sample(df).reset_index(drop=True)
    panel["region_year"] = panel["region"].astype(str) + "_" + panel["year"].astype(str)
    panel["country_idx"] = pd.factorize(panel["country_code"])[0]
    panel.sort_values(["country_code", "year"], inplace=True)
    for lag in range(1, LAGS + 1):
        panel[f"y_ext_lag{lag}"] = panel.groupby("country_code")["y_ext"].shift(lag)
    for comp in COMPONENTS + [OTHER]:
        col = f"{comp}_relation_dyn"
        for lag in range(1, LAGS + 1):
            panel[f"{col}_lag{lag}"] = panel.groupby("country_code")[col].shift(lag)
    return panel


def horse_race(panel, comps=COMPONENTS):
    """Joint LP of y_{t+h} on the components (component_decomposition.run_lp)."""
    comp_cols = [f"{c}_relation_dyn" for c in comps]
    regs = comp_cols + [f"y_ext_lag{i}" for i in range(1, LAGS + 1)] + \
        [f"{c}_lag{i}" for c in comp_cols for i in range(1, LAGS + 1)]
    res = {c: {"coef": np.full(len(H), np.nan), "se": np.full(len(H), np.nan)} for c in comps}
    n_obs = np.full(len(H), np.nan)
    for i, h in enumerate(H):
        tmp = panel.copy()
        tmp["_dep"] = tmp.groupby("country_idx")["y_ext"].shift(-h)
        sub = tmp.dropna(subset=["_dep"] + regs + ["region_year"])
        sp = sub.set_index(["country_idx", "year"])
        fit = PanelOLS(sp["_dep"], sp[regs], entity_effects=True, time_effects=False,
                       other_effects=sp[["region_year"]]).fit(cov_type="kernel")
        n_obs[i] = fit.nobs
        for c in comps:
            res[c]["coef"][i] = fit.params[f"{c}_relation_dyn"]
            res[c]["se"][i] = fit.std_errors[f"{c}_relation_dyn"]
    return res, n_obs


def components(df, numbers):
    panel = component_panel(df)
    res, n_obs = horse_race(panel)
    full = pd.concat([pd.DataFrame({"spec": "horse_race", "component": c, "horizon": H,
                                    "coef": res[c]["coef"], "se": res[c]["se"],
                                    "ci_lo": res[c]["coef"] - Z_95 * res[c]["se"],
                                    "ci_hi": res[c]["coef"] + Z_95 * res[c]["se"], "n_obs": n_obs})
                      for c in COMPONENTS], ignore_index=True)
    full.to_csv(CACHE / "component_irf.csv", index=False)

    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    for c in COMPONENTS:
        coef, se = res[c]["coef"] * GEO_SD, res[c]["se"] * GEO_SD         # per one SD of the index
        color = COMP_COLORS[c]
        ax.plot(H, coef, color=color, lw=2.5, label=COMP_LABELS[c])
        lo, hi = coef - Z_95 * se, coef + Z_95 * se
        ax.fill_between(H, lo, hi, color=color, alpha=PS.CI_ALPHA)
        for b in (lo, hi):
            ax.plot(H, b, color=color, ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
    PS.style_irf_ax(ax, ylabel=PS.IRF_YLABEL_GDP)
    savefig(fig, "Fig09a_components.pdf")
    plt.close(fig)

    # Numbers (Section 3.3.2): correlations on the long-coverage panel, coefficients per SD
    cols = ["geo_relation_dyn"] + [f"{c}_relation_dyn" for c in COMPONENTS]
    corr = panel[cols].corr()
    keys = [(f"corr[Full Index,{COMP_LABELS[c]}]", corr.loc["geo_relation_dyn", f"{c}_relation_dyn"])
            for c in COMPONENTS]
    keys += [(f"corr[{COMP_LABELS[a]},{COMP_LABELS[b]}]", corr.loc[f"{a}_relation_dyn", f"{b}_relation_dyn"])
             for a, b in [("econ", "diplo"), ("econ", "security"), ("diplo", "security")]]
    numbers += [{"key": k, "value": v, "paper": "Section 3.3.2", "full_ref": f"{FULL_A}:{k}"} for k, v in keys]

    sd = full.copy()
    for f in ("coef", "se", "ci_lo", "ci_hi"):
        sd[f] = sd[f] * GEO_SD
    hr = {c: sd[sd["component"] == c].set_index("horizon") for c in COMPONENTS}
    keys = [("horse_race_econ_peak_h0_h5", hr["econ"].loc[0:5, "coef"].max()),
            ("horse_race_diplo_min_h5_h12", hr["diplo"].loc[5:12, "coef"].min()),
            ("horse_race_diplo_max_h5_h12", hr["diplo"].loc[5:12, "coef"].max()),
            ("horse_race_security_h10", hr["security"].loc[10, "coef"])]
    keys += [(f"horse_race_{c}_sig_horizons", int((hr[c]["ci_lo"] > 0).sum())) for c in COMPONENTS]
    numbers += [{"key": k, "value": v, "paper": "Section 3.3.2, Figure 9(a)", "full_ref": f"{FULL_A}:{k}"}
                for k, v in keys]

    # Footnote check: adding the E+F component as a fourth regressor leaves the three paths unchanged
    res4, n4 = horse_race(panel, COMPONENTS + [OTHER])
    four = pd.concat([pd.DataFrame({"component": c, "horizon": H, "coef": res4[c]["coef"], "se": res4[c]["se"],
                                    "n_obs": n4}) for c in COMPONENTS + [OTHER]], ignore_index=True)
    four.to_csv(CACHE / "component_irf_fourway.csv", index=False)
    change = max(np.nanmax(np.abs(res4[c]["coef"] - res[c]["coef"]) * GEO_SD) for c in COMPONENTS)
    keys = [("fourway_max_abs_change", change)]
    keys += [(f"fourway_peak[{COMP_LABELS[c]}]", np.nanmax(res4[c]["coef"] * GEO_SD)) for c in COMPONENTS]
    o = res4[OTHER]
    keys.append(("fourway_other_n_pos_sig", int(np.sum(o["coef"] - Z_95 * o["se"] > 0))))
    numbers += [{"key": k, "value": v, "paper": "Section 3.3.2, footnote on the fourth component",
                 "full_ref": f"{FULL_A}:{k}"} for k, v in keys]


# ── Panel (b): formal acts ────────────────────────────────────────────
def formal_acts(df, numbers):
    pooled = df.dropna(subset=["geo_relation_dyn"])
    corr = {v: pooled[v].corr(pooled["geo_relation_dyn"]) for v in FORMAL}
    bal = restrict_balanced_sample(df)
    rows = []
    for v in FORMAL:
        r = lp_irf(prepare_panel(bal, shock_var=v), shock_var=v, horizon_range=range(0, 26))
        rows.append(pd.DataFrame({"spec": v, "h": r["h_vals"], "coef": r["coef"] * GEO_SD,
                                  "se": r["se"] * GEO_SD, "ci_lo": r["ci_lower"] * GEO_SD,
                                  "ci_hi": r["ci_upper"] * GEO_SD, "nobs": r["nobs"]}))
    irf = pd.concat(rows, ignore_index=True)                   # per one SD of the index
    irf.to_csv(CACHE / "formal_acts_irf.csv", index=False)

    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    for v, color, label in [("geo_relation_dyn", PS.C_BLUE, "Baseline (Aggregate Index)"),
                            ("formal_llm_relation_dyn", PS.C_ORANGE, "Formal Acts")]:
        d = irf[irf["spec"] == v]
        ax.plot(d["h"], d["coef"], color=color, lw=2.5, label=label)
        ax.fill_between(d["h"], d["ci_lo"], d["ci_hi"], color=color, alpha=PS.CI_ALPHA)
        for b in ("ci_lo", "ci_hi"):
            ax.plot(d["h"], d[b], color=color, ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
    d = irf[irf["spec"] == "formal_relation_dyn"]
    ax.plot(d["h"], d["coef"], color=PS.C_GREEN, lw=2.2, ls="--", label="Formal Acts, Code-Fixed Scores")
    PS.style_irf_ax(ax, ylabel=PS.IRF_YLABEL_GDP)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    savefig(fig, "Fig09b_formal_acts.pdf")
    plt.close(fig)

    p = {v: irf[irf["spec"] == v].set_index("h") for v in FORMAL}
    b, m, f = p["geo_relation_dyn"], p["formal_llm_relation_dyn"], p["formal_relation_dyn"]
    h15 = range(0, 16)

    def inside(x):
        return int(((x.loc[h15, "coef"] >= b.loc[h15, "ci_lo"]) & (x.loc[h15, "coef"] <= b.loc[h15, "ci_hi"])).sum())

    sec = "Section 3.3.2, Figure 9(b)"
    keys = [("corr_formal_llm_baseline", corr["formal_llm_relation_dyn"], "Section 3.3.2"),
            ("corr_formal_fixed_baseline", corr["formal_relation_dyn"], "Section 3.3.2"),
            ("formal_llm_inside_baseline_ci_h0_15", inside(m), sec),
            ("formal_fixed_inside_baseline_ci_h0_15", inside(f), sec),
            ("formal_llm_peak", m["coef"].max(), sec),
            ("formal_llm_peak_h", int(m["coef"].idxmax()), sec),
            ("formal_fixed_peak", f["coef"].max(), sec),
            ("baseline_peak", b["coef"].max(), sec),
            ("formal_fixed_max_gap", (f["coef"] - m["coef"]).abs().max(), sec)]
    numbers += [{"key": k, "value": v, "paper": s, "full_ref": f"{FULL_B}:{k}"} for k, v, s in keys]


def main():
    df = load()
    numbers = []
    components(df, numbers)
    formal_acts(df, numbers)
    save_numbers(__file__, numbers)


if __name__ == "__main__":
    main()
