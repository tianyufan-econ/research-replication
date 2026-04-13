"""
07_iv_verbal.py
===============
Fig 13a: LP-IV using non-economic verbal conflicts as instrument for
         geopolitical alignment.

Instrument: geo_noecon_conflict_relation_dyn
Method:     Manual 2SLS with Driscoll-Kraay standard errors
FE:         Country + region-year
Horizons:   h = 0 .. 20  (estimation uses -10 .. 25 for balanced sample)

Plot: baseline IV (blue solid with 95% CI) + full progressive controls
      overlay (orange dashed). Both estimated on a common sample that
      has all control variables non-missing.

Source scripts:
  09_iv/lp_verbal_iv_reg.py
  09_iv/combined_lpiv_controls.py
  09_iv/plot_iv_paper.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from linearmodels.panel import PanelOLS

from config import PANEL, FIGURES, CACHE, ensure_dirs, savefig, \
    read_csv_fallback, restrict_balanced_sample
import plot_style as PS
PS.apply_theme()
ensure_dirs()

# ── Configuration ────────────────────────────────────────────────────
Y_VAR      = "y_ext"
SHOCK      = "geo_relation_dyn"
INSTRUMENT = "geo_noecon_conflict_relation_dyn"
Y_LAG      = 4
GEO_LAG    = 4
IV_LAG     = 4
H_START    = -10
H_END      = 25
H_PLOT_START = 0
H_PLOT_END   = 20
Z95        = 1.96
MIN_OBS_PER_COUNTRY = 10

# Progressive control variables
INSTITUTION_VARS = [
    "v2x_polyarchy", "v2x_liberal", "v2x_partip",
    "v2xdl_delib", "v2x_egal",
]

# ── Data helpers ─────────────────────────────────────────────────────

def load_and_prepare():
    """Load panel, restrict to balanced sample, add lags for IV estimation."""
    df = read_csv_fallback(PANEL)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)
    df = restrict_balanced_sample(df)

    df["country_idx"] = pd.factorize(df["country_code"])[0]
    if "region" in df.columns:
        df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    else:
        df["region_year"] = "r1_" + df["year"].astype(str)

    # Lags: y_ext, geo, instrument
    for l in range(1, Y_LAG + 1):
        df[f"y_ext_lag{l}"] = df.groupby("country_idx")[Y_VAR].shift(l)
    for l in range(1, GEO_LAG + 1):
        df[f"geo_lag{l}"] = df.groupby("country_idx")[SHOCK].shift(l)
    for l in range(1, IV_LAG + 1):
        df[f"iv_lag{l}"] = df.groupby("country_idx")[INSTRUMENT].shift(l)

    # Lags for progressive controls
    for var in ["trade", "unrest_new"] + INSTITUTION_VARS:
        if var not in df.columns:
            df[var] = np.nan
        for l in range(1, 5):
            df[f"{var}_lag{l}"] = df.groupby("country_idx")[var].shift(l)

    # War control lags (if available)
    war_cols = [c for c in df.columns if c.startswith("war_") and "_lag" not in c]
    for var in war_cols:
        for l in range(1, 5):
            col = f"{var}_lag{l}"
            if col not in df.columns:
                df[col] = df.groupby("country_idx")[var].shift(l)

    print(f"Panel: {df.shape}, {df['country_idx'].nunique()} countries")
    return df


def _base_ctrl():
    """Baseline control columns: y lags + geo lags + instrument lags."""
    return ([f"y_ext_lag{l}" for l in range(1, Y_LAG + 1)]
            + [f"geo_lag{l}" for l in range(1, GEO_LAG + 1)]
            + [f"iv_lag{l}" for l in range(1, IV_LAG + 1)])


def _full_extra_ctrl(df):
    """Build full progressive control list: trade + unrest + war + institutions."""
    trade_lags = [f"trade_lag{i}" for i in range(1, 5)]
    unrest_lags = [f"unrest_new_lag{i}" for i in range(1, 5)]

    # War lags: find war_ columns that have _lag suffix
    war_lag_cols = sorted([c for c in df.columns
                           if c.startswith("war_") and "_lag" in c])
    inst_lags = [f"{v}_lag{i}" for v in INSTITUTION_VARS for i in range(1, 5)]

    return trade_lags + unrest_lags + war_lag_cols + inst_lags


# ── Consistent-country sample ───────────────────────────────────────

def get_consistent_countries(df, extra_ctrl=None):
    """Return countries with enough obs at both extreme horizons."""
    ctrl = _base_ctrl() + (extra_ctrl or [])
    valid = set()
    for cc in df["country_code"].unique():
        cdf = df[df["country_code"] == cc]
        ok = True
        for h in (H_START, H_END):
            tmp = cdf.copy()
            tmp["dep"] = tmp.groupby("country_idx")[Y_VAR].shift(-h)
            need = ["dep", SHOCK, INSTRUMENT] + ctrl + ["region_year"]
            if tmp.dropna(subset=need).shape[0] < MIN_OBS_PER_COUNTRY:
                ok = False
                break
        if ok:
            valid.add(cc)
    print(f"Consistent countries: {len(valid)}/{df['country_code'].nunique()}")
    return valid


# ── LP-IV (manual 2SLS with DK SE) ──────────────────────────────────

def lp_iv(df, valid_countries, extra_ctrl=None):
    """
    Manual two-stage LP-IV across horizons H_START..H_END.

    First  stage: SHOCK(t)      ~ INSTRUMENT(t) + controls + FE
    Second stage: Y_VAR(t+h)    ~ fitted_SHOCK(t) + controls + FE
    """
    ctrl = _base_ctrl() + (extra_ctrl or [])
    dfs = df[df["country_code"].isin(valid_countries)].copy()
    horizons = list(range(H_START, H_END + 1))
    coefs = np.full(len(horizons), np.nan)
    ses   = np.full(len(horizons), np.nan)

    for idx, h in enumerate(horizons):
        tmp = dfs.copy()
        tmp["dep"] = tmp.groupby("country_idx")[Y_VAR].shift(-h)
        need = ["dep", SHOCK, INSTRUMENT] + ctrl + ["region_year", "country_idx", "year"]
        sub = tmp.dropna(subset=need)
        if sub.shape[0] < 50:
            continue
        try:
            sp = sub.set_index(["country_idx", "year"])

            # First stage
            fs_X = sp[[INSTRUMENT] + ctrl]
            fs_keep = [c for c in fs_X.columns if fs_X[c].nunique(dropna=False) > 1]
            if INSTRUMENT not in fs_keep:
                continue
            fs_X = fs_X[fs_keep]
            fs_mod = PanelOLS(sp[SHOCK], fs_X,
                              entity_effects=True, time_effects=False,
                              other_effects=sp["region_year"],
                              drop_absorbed=True, check_rank=False)
            fs_res = fs_mod.fit(cov_type="kernel")
            fitted = np.asarray(fs_res.fitted_values).reshape(-1)

            # Second stage
            sr = sp.reset_index()
            sr["shock_hat"] = fitted
            sp2 = sr.set_index(["country_idx", "year"])
            ss_X = sp2[["shock_hat"] + ctrl]
            ss_keep = [c for c in ss_X.columns if ss_X[c].nunique(dropna=False) > 1]
            if "shock_hat" not in ss_keep:
                continue
            ss_X = ss_X[ss_keep]
            ss_mod = PanelOLS(sp2["dep"], ss_X,
                              entity_effects=True, time_effects=False,
                              other_effects=sp2["region_year"],
                              drop_absorbed=True, check_rank=False)
            ss_res = ss_mod.fit(cov_type="kernel")
            coefs[idx] = ss_res.params.get("shock_hat", np.nan)
            ses[idx]   = ss_res.std_errors.get("shock_hat", np.nan)
        except Exception as e:
            print(f"  LP-IV h={h}: {e}")

    return np.array(horizons, dtype=float), coefs, ses


# ── Plotting ─────────────────────────────────────────────────────────

def plot_iv_panel(h_base, coef_base, se_base, h_full, coef_full, fname):
    """Baseline with 95% CI + full-controls overlay, h=0..20."""
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    mask_b = (h_base >= H_PLOT_START) & (h_base <= H_PLOT_END)
    mask_f = (h_full >= H_PLOT_START) & (h_full <= H_PLOT_END)

    hb, cb, sb = h_base[mask_b], coef_base[mask_b], se_base[mask_b]
    hf, cf = h_full[mask_f], coef_full[mask_f]

    # 95% CI band
    ci_lo = cb - Z95 * sb
    ci_hi = cb + Z95 * sb
    ax.fill_between(hb, ci_lo, ci_hi,
                    color=PS.C_BLUE, alpha=PS.CI_ALPHA,
                    label="95% CI (Driscoll-Kraay)", zorder=1)
    # CI bound lines
    ax.plot(hb, ci_lo, color=PS.C_BLUE, ls="--",
            lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
    ax.plot(hb, ci_hi, color=PS.C_BLUE, ls="--",
            lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)

    # Baseline
    ax.plot(hb, cb, color=PS.C_BLUE, ls="-", lw=2.5,
            marker="o", ms=6, label="LP-IV Baseline", zorder=3)

    # Full controls overlay
    ax.plot(hf, cf, color=PS.C_ORANGE, ls="--",
            lw=2.0, marker="s", ms=5, label="LP-IV + Full Controls", zorder=3)

    PS.style_irf_ax(ax, ylabel=PS.IRF_YLABEL_GDP)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    savefig(fig, FIGURES, fname)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Fig 13a: LP-IV (Verbal Conflicts Instrument)")
    print("=" * 65)

    df = load_and_prepare()

    # Build full extra controls list
    full_extra = _full_extra_ctrl(df)
    # Filter to columns that actually exist
    full_extra = [c for c in full_extra if c in df.columns]

    # Baseline IV: balanced sample (148 countries, no extra controls)
    valid_base = get_consistent_countries(df)
    if not valid_base:
        print("ERROR: No valid countries for baseline. Check data.")
        return

    # Full controls IV: common sample (~109 countries with all controls)
    valid_full = get_consistent_countries(df, extra_ctrl=full_extra)
    if not valid_full:
        print("ERROR: No valid countries for full controls. Check data.")
        return

    print(f"\n  Baseline sample: {len(valid_base)} countries")
    print(f"  Full controls sample: {len(valid_full)} countries")

    # Baseline IV (balanced sample, no extra controls)
    print("\n--- Baseline IV ---")
    h_b, c_b, s_b = lp_iv(df, valid_base)

    # Full controls IV (common sample)
    print("\n--- IV + Full Controls ---")
    h_f, c_f, s_f = lp_iv(df, valid_full, extra_ctrl=full_extra)

    # Cache
    for label, h, c, s in [("baseline", h_b, c_b, s_b),
                            ("full_controls", h_f, c_f, s_f)]:
        cache_df = pd.DataFrame({"horizon": h, "coef": c, "se": s})
        cache_df.to_csv(CACHE / f"iv_verbal_{label}.csv", index=False)

    # Plot Fig 13a
    plot_iv_panel(h_b, c_b, s_b, h_f, c_f, "Fig13A_iv_verbal_conflicts.pdf")
    print("\nDone.")


if __name__ == "__main__":
    main()
