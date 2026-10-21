"""
07_transformations_iv.py
========================
Figure 12: robustness to bundled domestic transformations, long-coverage sample.

(a) Baseline local projection against two exclusion rules: dropping country-years
    classified as domestic transformations (historical codebook), and dropping V-Dem
    covered country-years within three years of a change in regime type (V-Dem path
    on covered observations only).
(b) LP-IV instrumenting the index with the leadership-change instrument z_leader
    (bilateral shift S_{t+1} - S_{t-1} around deaths in office and close elections in
    the major nations), with four lags of the instrument; baseline (148 countries) and
    with four lags of trade, unrest, war exposure and five V-Dem indices (109-country
    common sample). Driscoll-Kraay SEs from the 2SLS residual.

All paths: four lags of GDP and of the index, country and region-year fixed effects,
coefficients per one standard deviation of the index.

Full-package version: code/05_robustness/era_fe_robustness_vdem.py (panel a),
code/09_iv/lp_leader_iv_reg.py and 09_iv/plot_iv_paper.py (panel b, window tp1_tm1).
Outputs: output/figures/Fig12{a,b}_*.pdf, output/cache/lp_era_fe_specs_vdem.csv,
         lp_era_fe_vdem_complete_coverage.csv, iv_irf.csv, iv_irf_inst.csv,
         output/numbers/07_transformations_iv.csv
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from matplotlib.ticker import MaxNLocator
from scipy import stats

import plot_style as PS
from config import CACHE, GEO_SD, PANEL, ensure_dirs, read_csv_fallback, restrict_balanced_sample, save_numbers, savefig
from lp_utils import prepare_panel, save_irf

ensure_dirs()
PS.apply_theme()
warnings.filterwarnings("ignore")

Y, SHOCK, INST = "y_ext", "geo_relation_dyn", "z_leader"
LAGS = 4
H = range(-10, 26)
MIN_COUNTRY_OBS = 10
Z95 = stats.norm.ppf(0.975)
HIST, VDEM = "Transformation-Year Exclusion", "V-Dem Exclusion"
VDEM_W = "vdem_window_3"
WAR_COLS = ["war_site_onset_all", "war_site_caspop_all", "war_trade_caspop_weighted_all",
            "war_prox_caspop_weighted_all", "war_exposure_site_count_all"]
INSTITUTIONS = ["v2x_polyarchy", "v2x_liberal", "v2x_partip", "v2xdl_delib", "v2x_egal"]
ERA, IV, WIN = ("05_robustness/era_fe_robustness_vdem.py", "09_iv/lp_leader_iv_reg.py",
                "09_iv/window_robustness_table.py")

Y_LAGS = [f"{Y}_lag{l}" for l in range(1, LAGS + 1)]
GEO_LAGS = [f"{SHOCK}_lag{l}" for l in range(1, LAGS + 1)]
IV_LAGS = [f"{INST}_lag{l}" for l in range(1, LAGS + 1)]


def load():
    df = read_csv_fallback(PANEL)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)
    return df


def common_countries(df, extra_controls=()):
    """Countries with >= 10 usable observations at both horizon endpoints (h = -10 and 25)."""
    valid = set()
    for code, g in df.groupby("country_code", sort=False):
        ok = True
        for h in (H.start, H.stop - 1):
            tmp = g.copy()
            tmp["_dep"] = tmp.groupby("country_idx")[Y].shift(-h)
            if len(tmp.dropna(subset=["_dep", SHOCK, "region_year"] + Y_LAGS + GEO_LAGS + list(extra_controls))) \
                    < MIN_COUNTRY_OBS:
                ok = False
                break
        if ok:
            valid.add(code)
    print(f"  Common country sample: {len(valid)} / {df['country_code'].nunique()} countries")
    return valid


def per_sd(res):
    out = res.copy()
    for f in ("coef", "se", "ci_lo", "ci_hi"):
        out[f] = out[f] * GEO_SD
    return out


# ── Panel (a): excluding domestic transformation episodes ─────────────
def era_panel(df):
    """Long-coverage sample with the era and V-Dem flags (era_type missing = stable, V-Dem missing kept)."""
    df = restrict_balanced_sample(df.copy())
    df["era_type"] = df["era_type"].fillna("stable")
    for c in ("vdem_event", "vdem_window_1", "vdem_window_3", "vdem_window_5"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    df["country_idx"] = pd.factorize(df["country_code"])[0]
    df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    for l in range(1, LAGS + 1):
        df[f"{Y}_lag{l}"] = df.groupby("country_idx")[Y].shift(l)
        df[f"{SHOCK}_lag{l}"] = df.groupby("country_idx")[SHOCK].shift(l)
    return df


def lp_era(df, countries, mode):
    """mode: baseline, transform_year (stable years only) or vdem_year (V-Dem covered and unflagged)."""
    x_cols = [SHOCK] + Y_LAGS + GEO_LAGS
    sample = df[df["country_code"].isin(countries)].copy()
    rec = []
    for h in H:
        tmp = sample.copy()
        tmp["_dep"] = tmp.groupby("country_idx")[Y].shift(-h)
        req = ["_dep", "region_year", "country_idx", "year"] + x_cols
        req += {"transform_year": ["era_type"], "vdem_year": [VDEM_W]}.get(mode, [])
        sub = tmp.dropna(subset=req).copy()
        if mode == "transform_year":
            sub = sub[sub["era_type"] == "stable"].copy()
        if mode == "vdem_year":
            sub = sub[sub[VDEM_W] == 0].copy()
        if sub.empty:
            rec.append({"horizon": h, "coef": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                        "n_obs": 0, "n_countries": 0})
            continue
        panel = sub.set_index(["country_idx", "year"])
        X = panel[x_cols].copy()
        keep = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
        if SHOCK not in keep:
            rec.append({"horizon": h, "coef": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                        "n_obs": int(len(sub)), "n_countries": int(sub["country_code"].nunique())})
            continue
        try:
            r = PanelOLS(panel["_dep"], X[keep], entity_effects=True, time_effects=False,
                         other_effects=panel["region_year"], drop_absorbed=True,
                         check_rank=False).fit(cov_type="kernel")
            coef, se = r.params.get(SHOCK, np.nan), r.std_errors.get(SHOCK, np.nan)
        except Exception as exc:
            print(f"LP err h={h} ({mode}): {exc}")
            coef, se = np.nan, np.nan
        rec.append({"horizon": h, "coef": coef, "se": se,
                    "ci_lo": coef - 1.96 * se if pd.notna(se) else np.nan,
                    "ci_hi": coef + 1.96 * se if pd.notna(se) else np.nan,
                    "n_obs": int(len(sub)), "n_countries": int(sub["country_code"].nunique())})
    return pd.DataFrame.from_records(rec)


def plot_series(ax, res, color, label, lw, marker, ms):
    """Solid line, CI fill and dashed CI bounds for one specification."""
    mask = res["coef"].notna()
    ax.plot(res.loc[mask, "horizon"], res.loc[mask, "coef"], color=color, lw=lw, ls="-", marker=marker, ms=ms,
            label=label)
    ci = res["ci_lo"].notna() & res["ci_hi"].notna() & mask
    if ci.any():
        ax.fill_between(res.loc[ci, "horizon"], res.loc[ci, "ci_lo"], res.loc[ci, "ci_hi"], color=color,
                        alpha=PS.CI_ALPHA)
        for b in ("ci_lo", "ci_hi"):
            ax.plot(res.loc[ci, "horizon"], res.loc[ci, b], color=color, ls="--", lw=PS.CI_BOUND_LW,
                    alpha=PS.CI_BOUND_ALPHA)


def transformations(df_full, numbers):
    """Figure 12(a), and the fixed 124-country V-Dem comparison quoted in its footnote."""
    df = era_panel(df_full)
    valid = common_countries(df)
    specs = [("Baseline", "baseline"), (HIST, "transform_year"), (VDEM, "vdem_year")]
    res = {lab: lp_era(df, valid, mode) for lab, mode in specs}
    long = pd.concat([r.assign(specification=lab) for lab, r in res.items()], ignore_index=True)
    long.to_csv(CACHE / "lp_era_fe_specs_vdem.csv", index=False)

    r = {k: per_sd(v) for k, v in res.items()}                       # per one SD of the index
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    plot_series(ax, r["Baseline"], PS.C_BLUE, "Baseline", lw=2.5, marker="o", ms=6)
    plot_series(ax, r[HIST], PS.C_ORANGE, HIST, lw=2.0, marker="s", ms=5)
    v = r[VDEM]["coef"].notna()
    ax.plot(r[VDEM].loc[v, "horizon"], r[VDEM].loc[v, "coef"], color=PS.C_RED, lw=2.0, ls="--", marker="^", ms=5,
            label=VDEM)
    PS.style_irf_ax(ax, ylabel=PS.IRF_YLABEL_GDP)
    ax.set_xlim(-0.5, 25.5)                                          # horizons 0--25
    fig.tight_layout()
    savefig(fig, "Fig12a_transformation_exclusion.pdf")
    plt.close(fig)

    # Countries observed in V-Dem in every year 1960-2024 (Section 3.4.1, footnote)
    complete = set(df.groupby("country_code")[VDEM_W].apply(lambda s: s.notna().all()).loc[lambda s: s].index) & valid
    cov = pd.concat([lp_era(df, complete, mode).assign(specification=lab) for lab, mode in
                     [("Baseline (Complete V-Dem Coverage)", "baseline"),
                      ("V-Dem Exclusion (Complete Coverage)", "vdem_year")]], ignore_index=True)
    cov.to_csv(CACHE / "lp_era_fe_vdem_complete_coverage.csv", index=False)

    specs, cov = per_sd(long), per_sd(cov)
    at = lambda d, lab, h, col="coef": d.loc[(d["specification"] == lab) & (d["horizon"] == h), col].iloc[0]
    base = specs[specs["specification"] == "Baseline"].set_index("horizon")["coef"]
    hist = specs[specs["specification"] == HIST].set_index("horizon")["coef"]
    sec = "Section 3.4.1, Figure 12(a)"
    numbers += [
        {"key": "coef_h5[Baseline]", "value": at(specs, "Baseline", 5), "paper": sec},
        {"key": f"coef_h5[{HIST}]", "value": at(specs, HIST, 5), "paper": sec},
        {"key": f"coef_h5[{VDEM}]", "value": at(specs, VDEM, 5), "paper": sec},
        {"key": f"n_countries_h5[{VDEM}]", "value": at(specs, VDEM, 5, "n_countries"), "paper": sec},
        {"key": "n_countries[Baseline]", "value": specs.loc[specs["specification"] == "Baseline", "n_countries"].max(),
         "paper": "Figure 12 note"},
        {"key": "n_horizons_h0_h20_transformation_ratio_ge_0.6", "value": int(((hist / base).loc[0:20] >= 0.6).sum()),
         "paper": "Introduction"},
        {"key": "coef_h5[Baseline (Complete V-Dem Coverage)]",
         "value": at(cov, "Baseline (Complete V-Dem Coverage)", 5), "paper": "Section 3.4.1, footnote"},
        {"key": "coef_h5[V-Dem Exclusion (Complete Coverage)]",
         "value": at(cov, "V-Dem Exclusion (Complete Coverage)", 5), "paper": "Section 3.4.1, footnote"},
        {"key": "n_countries[complete V-Dem coverage]", "value": cov["n_countries"].max(),
         "paper": "Section 3.4.1, footnote"},
    ]
    for row in numbers[-9:]:
        row["full_ref"] = f"{ERA}:{row['key']}"


# ── Panel (b): leadership transitions in major nations ────────────────
def lp_iv_2sls(df, extra_controls=()):
    """LP-IV by manual 2SLS; the SE comes from a refit on y - b*(p - p_hat), i.e. the 2SLS residual."""
    ctrl = Y_LAGS + GEO_LAGS + IV_LAGS + list(extra_controls)
    hs = list(H)
    coefs, ses, nobs = (np.full(len(hs), np.nan) for _ in range(3))
    for i, h in enumerate(hs):
        tmp = df.copy()
        tmp["dep"] = tmp.groupby("country_idx")[Y].shift(-h)
        sub = tmp.dropna(subset=["dep", SHOCK, INST] + ctrl + ["region_year", "country_idx", "year"])
        if sub.shape[0] < 50:
            continue
        try:
            sp = sub.set_index(["country_idx", "year"])
            fs = PanelOLS(sp[SHOCK], sp[[INST] + ctrl], entity_effects=True, time_effects=False,
                          other_effects=sp["region_year"]).fit(cov_type="kernel")
            sr = sp.reset_index()
            sr["shock_hat"] = fs.fitted_values.values
            sp2 = sr.set_index(["country_idx", "year"])
            ss_X = sp2[["shock_hat"] + ctrl]
            ss = PanelOLS(sp2["dep"], ss_X, entity_effects=True, time_effects=False,
                          other_effects=sp2["region_year"]).fit(cov_type="kernel")
            b = ss.params["shock_hat"]
            sp2["dep_adj"] = sp2["dep"] - b * (sp2[SHOCK] - sp2["shock_hat"])
            iv = PanelOLS(sp2["dep_adj"], ss_X, entity_effects=True, time_effects=False,
                          other_effects=sp2["region_year"]).fit(cov_type="kernel")
            coefs[i], ses[i], nobs[i] = b, iv.std_errors.get("shock_hat", np.nan), ss.nobs
        except Exception as e:
            print(f"  LP-IV h={h}: {e}")
    return {"h_vals": np.array(hs, dtype=float), "coef": coefs, "se": ses,
            "ci_lower": coefs - Z95 * ses, "ci_upper": coefs + Z95 * ses, "nobs": nobs}


def first_stage_F(df):
    """First-stage F (t^2 of the instrument) at h = 0, with four lags of GDP, the index and the instrument."""
    ctrl = Y_LAGS + GEO_LAGS + IV_LAGS
    sp = df.dropna(subset=[SHOCK, INST] + ctrl + ["region_year", "country_idx", "year"]).set_index(["country_idx", "year"])
    res = PanelOLS(sp[SHOCK], sp[[INST] + ctrl], entity_effects=True, time_effects=False,
                   other_effects=sp["region_year"]).fit(cov_type="kernel")
    return float((res.params[INST] / res.std_errors[INST]) ** 2)


def leadership_iv(df_full, numbers):
    """Figure 12(b): LP-IV baseline (148 countries, 95% band) and with full controls (109 countries)."""
    df_base = prepare_panel(restrict_balanced_sample(df_full.copy()), y_var=Y, shock_var=SHOCK,
                            y_lags=LAGS, shock_lags=LAGS, extra_shock_vars=[INST])
    iv = lp_iv_2sls(df_base)
    save_irf(iv, CACHE / "iv_irf.csv")

    df_war = df_full.copy()
    df_war[WAR_COLS] = df_war[WAR_COLS].fillna(0.0)                  # war exposure: missing = 0
    df_war = prepare_panel(restrict_balanced_sample(df_war), y_var=Y, shock_var=SHOCK,
                           y_lags=LAGS, shock_lags=LAGS, extra_shock_vars=[INST])
    controls = ["trade", "unrest_new"] + WAR_COLS + INSTITUTIONS
    for var in controls:
        for l in range(1, LAGS + 1):
            df_war[f"{var}_lag{l}"] = df_war.groupby("country_idx")[var].shift(l)
    all_controls = [f"{v}_lag{l}" for v in controls for l in range(1, LAGS + 1)]
    valid = common_countries(df_war, all_controls)
    iv_inst = lp_iv_2sls(df_war[df_war["country_code"].isin(valid)].copy(), all_controls)
    save_irf(iv_inst, CACHE / "iv_irf_inst.csv")

    # Plot from the caches, as the full package does (per one SD of the index)
    base = pd.read_csv(CACHE / "iv_irf.csv").sort_values("h")
    full = pd.read_csv(CACHE / "iv_irf_inst.csv").sort_values("h")
    h, coef, se = base["h"].values, base["coef"].values * GEO_SD, base["se"].values * GEO_SD
    h_f, coef_f = full["h"].values, full["coef"].values * GEO_SD
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    m = (h >= 0) & (h <= 20)
    lo, hi = coef - 1.96 * se, coef + 1.96 * se
    ax.fill_between(h[m], lo[m], hi[m], color=PS.C_BLUE, alpha=PS.CI_ALPHA, label="95% CI (Driscoll-Kraay)", zorder=1)
    for b in (lo, hi):
        ax.plot(h[m], b[m], color=PS.C_BLUE, ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
    ax.plot(h[m], coef[m], color=PS.C_BLUE, ls="-", lw=2.5, marker="o", ms=6, label="LP-IV Baseline", zorder=3)
    m_f = (h_f >= 0) & (h_f <= 20)
    ax.plot(h_f[m_f], coef_f[m_f], color=PS.C_ORANGE, ls="--", lw=2.0, marker="s", ms=5,
            label="LP-IV + Full Controls", zorder=3)
    PS.style_irf_ax(ax, ylabel=PS.IRF_YLABEL_GDP)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    savefig(fig, "Fig12b_leadership_iv.pdf")
    plt.close(fig)

    c_iv, c_full = dict(zip(iv["h_vals"], iv["coef"] * GEO_SD)), dict(zip(iv_inst["h_vals"], iv_inst["coef"] * GEO_SD))
    sec = "Section 3.4.2, Figure 12(b)"
    rows = [{"key": "n_countries_baseline", "value": df_base["country_code"].nunique(), "paper": "Figure 12 note"},
            {"key": "n_countries_common", "value": len(valid), "paper": "Figure 12 note"}]
    rows += [{"key": f"iv_baseline_h{k}", "value": c_iv[k], "paper": sec} for k in (3, 4, 5, 6, 9, 10, 11, 12, 13, 14)]
    rows += [{"key": f"iv_full_controls_h{k}", "value": c_full[k], "paper": sec} for k in (2, 3, 4, 5, 13, 14)]
    sig = ((iv["ci_lower"] > 0) | (iv["ci_upper"] < 0)) & (iv["h_vals"] >= 0) & (iv["h_vals"] <= 20)
    rows.append({"key": "iv_baseline_sig_horizons", "value": ",".join(str(int(x)) for x in iv["h_vals"][sig]),
                 "paper": "Section 3.4.2 (horizons whose 95% interval excludes zero)"})
    for r in rows:
        r["full_ref"] = f"{IV}:{r['key']}"
    rows.append({"key": "tp1_tm1_F_h0", "value": first_stage_F(df_base), "paper": "Section 3.4.2",
                 "full_ref": f"{WIN}:tp1_tm1_F_h0"})
    numbers += rows


def main():
    df = load()
    numbers = []
    transformations(df, numbers)
    leadership_iv(df, numbers)
    save_numbers(__file__, numbers)


if __name__ == "__main__":
    main()
