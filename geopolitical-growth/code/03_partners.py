"""
03_partners.py
==============
Figure 8: (a) GDP response by partner group, (b) GDP response in the two historical
periods 1960-1989 and 1990-2019.

Panel (a): one local projection with the three additive partner components of the
index entered jointly, geo_relation_dyn = US + other Western + other major nations
(other Western = geo_relation_dyn_western - geo_relation_dyn_us; geo_relation_dyn_exclus
is aggregate-minus-US and is not used), with four lags of GDP and of each component.
The univariate baseline is the gray reference line.
Panel (b): the baseline local projection on each 30-year subperiod. Lags are built on
the full long-coverage panel, then the estimation rows are restricted to the subperiod.
Both panels: long-coverage sample, country and region-year fixed effects,
Driscoll-Kraay standard errors, horizons 0..10, coefficients per one SD of the
aggregate index.

Full-package version: code/02_baseline/lp_partner_3way_reg.py (panel a; it estimates
h = -15..25 and plots h = 0..10, the horizons estimated here) and
code/02_baseline_plot/plot_symmetry.py (panel b, which estimates the subperiod IRFs).
Outputs: output/figures/Fig08{a,b}_*.pdf, output/cache/partner_3way_irf.csv,
         subperiod_irf.csv, output/numbers/03_partners.csv
"""

import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

import plot_style as PS
from config import CACHE, GEO_SD, PANEL, ensure_dirs, read_csv_fallback, restrict_balanced_sample, save_numbers, savefig
from lp_utils import lp_irf, lp_irf_joint, prepare_panel

ensure_dirs()
PS.apply_theme()
warnings.filterwarnings("ignore")

H = range(0, 11)                      # horizons plotted in both panels
LAGS = 4
REF_COLOR = "0.45"                    # gray reference line
THREE = ["geo_relation_dyn_us", "geo_relation_dyn_westexus", "geo_relation_dyn_nonwestern"]
STYLE = {
    "geo_relation_dyn_us":         dict(color=PS.C_BLUE,   ls="--", marker="o", label="United States"),
    "geo_relation_dyn_westexus":   dict(color=PS.C_ORANGE, ls="--", marker="s", label="Other Western nations"),
    "geo_relation_dyn_nonwestern": dict(color=PS.C_RED,    ls="-.", marker="^", label="Other major nations"),
}
FULL_A = "02_baseline/lp_partner_3way_reg.py"
FULL_B = "02_baseline_plot/plot_symmetry.py"


def per_sd(res, fields=("coef", "se", "ci_lo", "ci_hi", "ci_lower", "ci_upper")):
    """Copy of an LP result with coefficient-type fields per one SD of the index (as the full config.per_sd)."""
    out = res.copy()
    for f in fields:
        if f in out:
            out[f] = out[f] * GEO_SD if hasattr(out[f], "mul") else np.asarray(out[f], dtype=float) * GEO_SD
    return out


def load():
    df = read_csv_fallback(PANEL, low_memory=False)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)
    return restrict_balanced_sample(df)


# ── Panel (a): partner groups ─────────────────────────────────────────
def partners(df, numbers):
    df = df.copy()
    df["geo_relation_dyn_westexus"] = df["geo_relation_dyn_western"] - df["geo_relation_dyn_us"]
    prep = prepare_panel(df, y_var="y_ext", shock_var=THREE[0], y_lags=LAGS, shock_lags=LAGS,
                         extra_shock_vars=THREE[1:] + ["geo_relation_dyn"])
    res3 = lp_irf_joint(prep, THREE, y_var="y_ext", horizon_range=H, y_lags=LAGS, shock_lags=LAGS,
                        fe="region_year", cov_type="kernel")
    base = lp_irf(prep, shock_var="geo_relation_dyn", y_var="y_ext", horizon_range=H, y_lags=LAGS,
                  shock_lags=LAGS, fe="region_year", cov_type="kernel")

    rows = [{"spec": spec, "group": sv, "horizon": int(h), "coef": r["coef"][i], "se": r["se"][i],
             "ci_lo": r["ci_lower"][i], "ci_hi": r["ci_upper"][i]}
            for spec, sv, r in [("3way", sv, res3[sv]) for sv in THREE] + [("baseline", "geo_relation_dyn", base)]
            for i, h in enumerate(r["h_vals"])]
    pd.DataFrame(rows).to_csv(CACHE / "partner_3way_irf.csv", index=False)

    res3 = {sv: per_sd(res3[sv]) for sv in THREE}
    base = per_sd(base)
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    for sv in THREE:
        r, st = res3[sv], STYLE[sv]
        ax.plot(r["h_vals"], r["coef"], color=st["color"], ls=st["ls"], lw=2.2,
                marker=st["marker"], ms=5, label=st["label"], zorder=3)
        ax.fill_between(r["h_vals"], r["ci_lower"], r["ci_upper"], color=st["color"], alpha=0.12, zorder=1)
    ax.plot(base["h_vals"], base["coef"], color=REF_COLOR, ls="--", lw=1.8,
            marker="", alpha=0.7, label="Baseline (Aggregate Index)", zorder=2)
    ax.set_xlim(H.start - 0.3, H.stop - 1 + 0.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    PS.style_irf_ax(ax, ylabel=PS.IRF_YLABEL_GDP)
    fig.tight_layout()
    savefig(fig, "Fig08a_partners.pdf")
    plt.close(fig)

    sec = "Section 3.3.1, Figure 8(a)"
    for sv in THREE:
        r = res3[sv]
        k = int(r["coef"].argmax())
        numbers += [
            {"key": f"peak_coef_{sv}", "value": r["coef"][k], "paper": sec, "full_ref": f"{FULL_A}:peak_coef_{sv}"},
            {"key": f"peak_h_{sv}", "value": int(r["h_vals"][k]), "paper": sec, "full_ref": f"{FULL_A}:peak_h_{sv}"},
        ]


# ── Panel (b): historical periods ─────────────────────────────────────
def lp_period(df, shock="geo_relation_dyn"):
    """Baseline LP at h = 0..10 on the rows of df (plot_symmetry.lp_estimate)."""
    x_cols = [shock] + [f"y_ext_lag{i}" for i in range(1, LAGS + 1)] + [f"{shock}_lag{i}" for i in range(1, LAGS + 1)]
    records = []
    for h in H:
        tmp = df.copy()
        tmp["_dep"] = tmp.groupby("country_idx")["y_ext"].shift(-h)
        sub = tmp.dropna(subset=["_dep"] + x_cols + ["region_year"])
        panel = sub.set_index(["country_idx", "year"])
        res = PanelOLS(dependent=panel["_dep"], exog=panel[x_cols], entity_effects=True, time_effects=False,
                       other_effects=panel["region_year"]).fit(cov_type="kernel")
        coef, se = res.params[shock], res.std_errors[shock]
        records.append(dict(horizon=h, coef=coef, se=se, ci_lo=coef - 1.96 * se, ci_hi=coef + 1.96 * se,
                            n_obs=int(res.nobs)))
    return pd.DataFrame(records)


def plot_path(ax, d, color, label, ls, marker):
    h, coef, se = d["horizon"].values, d["coef"].values, d["se"].values
    ax.plot(h, coef, color=color, ls=ls, lw=2.2, marker=marker, ms=5, label=label, zorder=3)
    ax.fill_between(h, coef - 1.96 * se, coef + 1.96 * se, color=color, alpha=0.12, zorder=1)


def periods(df, numbers):
    # The panel's own country_idx is the entity index; lags are built on the full long-coverage panel.
    df = df.copy()
    df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    for l in range(1, LAGS + 1):
        df[f"y_ext_lag{l}"] = df.groupby("country_idx")["y_ext"].shift(l)
    for l in range(1, LAGS + 1):
        df[f"geo_relation_dyn_lag{l}"] = df.groupby("country_idx")["geo_relation_dyn"].shift(l)

    parts = []
    for label, sub in [("full_sample", df), ("1960_1989", df[df["year"].between(1960, 1989)].copy()),
                       ("1990_2019", df[df["year"].between(1990, 2019)].copy())]:
        r = lp_period(sub)
        r["spec_label"] = label
        parts.append(r)
    irf = pd.concat(parts, ignore_index=True)
    irf.to_csv(CACHE / "subperiod_irf.csv", index=False)

    irf = per_sd(irf)
    full, p1, p2 = (irf[irf["spec_label"] == s].sort_values("horizon") for s in ("full_sample", "1960_1989", "1990_2019"))
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    plot_path(ax, p1, PS.C_BLUE, "1960–1989", "-", "o")
    plot_path(ax, p2, PS.C_RED, "1990–2019", "--", "s")
    ax.plot(full["horizon"].values, full["coef"].values, color=REF_COLOR, ls="--", lw=1.8,
            marker="", alpha=0.7, label="Full Sample", zorder=2)
    PS.style_irf_ax(ax, ylabel=PS.IRF_YLABEL_GDP)
    fig.tight_layout()
    savefig(fig, "Fig08b_periods.pdf")
    plt.close(fig)

    for spec, d in [("1960_1989", p1), ("1990_2019", p2)]:
        numbers += [
            {"key": f"subperiod_peak_{spec}", "value": d["coef"].max(), "paper": "Section 3.3.1, Figure 8(b)",
             "full_ref": f"{FULL_B}:subperiod_peak_{spec}"},
            {"key": f"subperiod_n_obs_h0_{spec}", "value": int(d.loc[d["horizon"] == 0, "n_obs"].iloc[0]),
             "paper": "Section 3.3.1, footnote to Figure 8(b)", "full_ref": f"{FULL_B}:subperiod_n_obs_h0_{spec}"},
        ]


def main():
    df = load()
    numbers = []
    partners(df, numbers)
    periods(df, numbers)
    save_numbers(__file__, numbers)


if __name__ == "__main__":
    main()
