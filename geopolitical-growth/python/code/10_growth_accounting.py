"""
10_growth_accounting.py
=======================
Fig 17: Decade boxplots of geopolitical growth effects
  Panel a: Contemporaneous (within-decade) effects
  Panel b: Long-run (cumulative) effects

Fig 18: Period scatter (1960-1990 vs 1991-2024)
  X-axis: geo growth effects 1960-1990
  Y-axis: geo growth effects 1991-2024
  Selective country labels (~33 key countries), 45-degree line,
  region colors with size differentiation.

Pipeline:
  1. Estimate transitory + permanent IRFs via LP
  2. Compute counterfactual GDP paths using auxiliary-shock decomposition
  3. Calculate decade-by-decade contemporaneous and long-run effects
  4. Plot boxplots and period scatter

Requires `adjustText` package for label placement in Fig 18.

Source scripts:
  03_extensions/growth_accounting_reg.py
  03_extensions_plot/plot_growth_accounting.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None
    print("WARNING: adjustText not installed. Labels in Fig 18 may overlap.")
    print("  Install with: pip install adjustText")

from config import PANEL, FIGURES, CACHE, ensure_dirs, savefig, \
    read_csv_fallback, restrict_balanced_sample
import plot_style as PS
PS.apply_theme()
ensure_dirs()

# ── Configuration ────────────────────────────────────────────────────
H = 25
Y_LAG = 4
GEO_LAG = 4

# Region mapping
REGION_NAMES = {
    "LAC": "Latin America & Caribbean", "AFR": "Africa",
    "ECA": "Europe & Central Asia",     "INL": "Developed",
    "SAS": "South Asia",                "MNA": "Middle East & North Africa",
    "EAP": "East Asia & Pacific",
}
REGION_ORDER = [
    "Middle East & North Africa", "Africa", "Europe & Central Asia",
    "Developed", "Latin America & Caribbean", "South Asia", "East Asia & Pacific",
]
_DEEP = sns.color_palette("deep")
REGION_COLORS = {
    "Middle East & North Africa": _DEEP[0],
    "Africa":                     _DEEP[1],
    "Europe & Central Asia":      _DEEP[2],
    "Developed":                  _DEEP[3],
    "Latin America & Caribbean":  _DEEP[4],
    "South Asia":                 _DEEP[5],
    "East Asia & Pacific":        _DEEP[6],
}

# Countries to label in Fig 18
LABEL_COUNTRIES = {
    "ZAF", "EST", "LVA", "LTU", "CHL", "GEO",
    "VEN", "NIC", "CHN", "RUS", "BLR", "USA", "SGP",
    "DEU", "GBR", "FRA", "JPN", "IND", "BRA", "KOR",
    "IRN", "PRK", "UKR", "SAU", "TUR",
    "LBY", "AFG", "SYR", "ERI", "CUB",
    "HUN", "QAT", "MMR",
}


# ── Data preparation ─────────────────────────────────────────────────

def prepare_panel(df):
    """Add country_idx, region_year, and lags."""
    df = df.copy()
    df["country_idx"] = pd.factorize(df["country_code"])[0]
    if "region" in df.columns:
        df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    else:
        df["region_year"] = "r1_" + df["year"].astype(str)
    df = df.sort_values(["country_idx", "year"])
    for l in range(1, Y_LAG + 1):
        df[f"y_ext_lag{l}"] = df.groupby("country_idx")["y_ext"].shift(l)
    for l in range(1, GEO_LAG + 1):
        df[f"geo_relation_dyn_lag{l}"] = df.groupby("country_idx")["geo_relation_dyn"].shift(l)
    return df


# ── LP-IRF estimation (OLS with C() FE) ─────────────────────────────

def lp_irf_simple(df, shock_var, dep_var, horizon_range):
    """Estimate LP-IRF via statsmodels OLS with country + region-year FE."""
    y_lags = [f"y_ext_lag{i}" for i in range(1, Y_LAG + 1)]
    geo_lags = [f"geo_relation_dyn_lag{i}" for i in range(1, GEO_LAG + 1)]
    h_vals = list(horizon_range)
    irf = np.full(len(h_vals), np.nan)

    for idx, h in enumerate(h_vals):
        tmp = df.copy()
        tmp["dep_var"] = tmp.groupby("country_idx")[dep_var].shift(-h)
        cols = ["dep_var", shock_var] + y_lags + geo_lags + ["country_idx", "region_year"]
        sub = tmp.dropna(subset=cols)
        if sub.shape[0] < 100:
            continue
        fmla = f"dep_var ~ {shock_var} + " + " + ".join(y_lags + geo_lags) + \
               " + C(country_idx) + C(region_year)"
        try:
            res = smf.ols(fmla, data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["country_idx"]})
            irf[idx] = res.params.get(shock_var, np.nan)
        except Exception as e:
            print(f"  LP error h={h}, dep={dep_var}: {e}")

    return np.array(h_vals), irf


# ── Auxiliary-shock decomposition ────────────────────────────────────

def compute_auxiliary_shocks(phi_p, H_val):
    """Compute auxiliary shocks for transitory response."""
    phi_matrix = np.eye(H_val + 1)
    for i in range(1, H_val + 1):
        for j in range(i):
            phi_matrix[i, j] = phi_p[i - j] if i - j < len(phi_p) else 0
    try:
        return np.linalg.inv(phi_matrix) @ np.array(
            [1] + [0] * H_val, dtype=float)
    except np.linalg.LinAlgError:
        return np.zeros(H_val + 1)


# ── Decade calculations ─────────────────────────────────────────────

def calculate_decade_changes(df):
    """Within-decade changes in geo_relation_dyn for each country."""
    decades = [(1960, 1969), (1970, 1979), (1980, 1989),
               (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2024)]
    rows = []
    for ci, cdf in df.groupby("country_idx"):
        cc = cdf["country_code"].iloc[0]
        for s, e in decades:
            sd = cdf[cdf["year"] == s]
            ed = cdf[cdf["year"] == e]
            if sd.empty or ed.empty:
                continue
            sg = sd["geo_relation_dyn"].iloc[0]
            eg = ed["geo_relation_dyn"].iloc[0]
            if np.isnan(sg) or np.isnan(eg):
                continue
            label = f"{s}s" if e - s == 9 else f"{s}-{e}"
            rows.append({"country_code": cc, "decade": label,
                         "geo_change": eg - sg})
    return pd.DataFrame(rows)


def calculate_contemporaneous_effects(df, alpha_transitory, decades):
    """Contemporaneous geopolitical effects within each decade."""
    rows = []
    for ci, cdf in df.groupby("country_idx"):
        cc = cdf["country_code"].iloc[0]
        cdf = cdf.sort_values("year")
        for s, e in decades:
            ddf = cdf[(cdf["year"] >= s) & (cdf["year"] <= e)]
            if ddf.empty:
                continue
            s_data = ddf[ddf["year"] == s]
            if s_data.empty:
                continue
            init_geo = s_data["geo_relation_dyn"].iloc[0]
            if np.isnan(init_geo):
                continue
            effect = 0
            for _, row in ddf.iterrows():
                diff = row["geo_relation_dyn"] - init_geo
                horizon = e - int(row["year"])
                if not np.isnan(diff) and horizon < len(alpha_transitory):
                    effect += alpha_transitory[horizon] * diff
            label = f"{s}s" if e - s == 9 else f"{s}-{e}"
            rows.append({"country_code": cc, "decade": label,
                         "contemporaneous_effect": effect})
    return pd.DataFrame(rows)


def compute_counterfactual_gdp(df, alpha_transitory):
    """Compute delta_y_geo for each country-year."""
    df = df.copy()
    median_geo = df.groupby("year")["geo_relation_dyn"].median().reset_index()
    median_geo.columns = ["year", "median_geo"]
    df = df.merge(median_geo, on="year", how="left")
    df["gross_change"] = df["median_geo"] - df["geo_relation_dyn"]

    df["delta_y_cf"] = np.nan
    for ci, group in df.groupby("country_idx"):
        gc = group["gross_change"].fillna(0).values
        valid = group["gross_change"].notna().values
        dcf = np.full(len(gc), np.nan)
        for t in range(len(gc)):
            if not valid[t]:
                continue
            eff = 0
            for hh in range(H + 1):
                if t - hh >= 0 and not np.isnan(alpha_transitory[hh]):
                    eff += alpha_transitory[hh] * gc[t - hh]
            dcf[t] = eff
        df.loc[group.index, "delta_y_cf"] = dcf

    df["delta_y_geo"] = -df["delta_y_cf"]
    return df


# ── Plotting: Fig 17 (boxplots) ─────────────────────────────────────

def plot_decade_boxplots(longrun_df, contemp_df):
    """Two separate boxplot figures: contemporaneous and long-run."""
    decade_order = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020-2024"]
    palette = sns.color_palette("deep")

    # Panel a: Contemporaneous
    fig1, ax1 = plt.subplots(figsize=(12, 7.2))
    sns.boxplot(data=contemp_df, x="decade", y="contemporaneous_effect",
                palette=palette, ax=ax1, order=decade_order)
    sns.stripplot(data=contemp_df, x="decade", y="contemporaneous_effect",
                  color="black", alpha=0.3, size=4, ax=ax1, order=decade_order)
    ax1.axhline(0, **PS.HLINE_KW)
    ax1.set_xlabel("Decade", fontsize=14)
    ax1.set_ylabel("Contemporaneous Geopolitical Effect on Log GDP per Capita (\u00d7100)",
                    fontsize=14)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")
    sns.despine(ax=ax1)
    fig1.tight_layout()
    savefig(fig1, FIGURES, "Fig17A_decade_contemporaneous.pdf")
    plt.close(fig1)

    # Panel b: Long-run
    fig2, ax2 = plt.subplots(figsize=(12, 7.2))
    sns.boxplot(data=longrun_df, x="decade", y="longrun_effect",
                palette=palette, ax=ax2, order=decade_order)
    sns.stripplot(data=longrun_df, x="decade", y="longrun_effect",
                  color="black", alpha=0.3, size=4, ax=ax2, order=decade_order)
    ax2.axhline(0, **PS.HLINE_KW)
    ax2.set_xlabel("Decade", fontsize=14)
    ax2.set_ylabel("Long-run Geopolitical Effect on Log GDP per Capita (\u00d7100)",
                    fontsize=14)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")
    sns.despine(ax=ax2)
    fig2.tight_layout()
    savefig(fig2, FIGURES, "Fig17B_decade_longrun.pdf")
    plt.close(fig2)


# ── Plotting: Fig 18 (period scatter) ───────────────────────────────

def plot_period_scatter(cf_df):
    """Period comparison scatter (1960-1990 vs 1991-2024)."""
    # Calculate changes for two periods
    d1 = cf_df[cf_df["year"].isin([1960, 1990])].pivot_table(
        index="country_code", columns="year", values="delta_y_geo").reset_index()
    d1.columns = ["country_code", "dy_1960", "dy_1990"]
    d1["change_cw"] = d1["dy_1990"] - d1["dy_1960"]

    d2 = cf_df[cf_df["year"].isin([1991, 2024])].pivot_table(
        index="country_code", columns="year", values="delta_y_geo").reset_index()
    d2.columns = ["country_code", "dy_1991", "dy_2024"]
    d2["change_pcw"] = d2["dy_2024"] - d2["dy_1991"]

    pc = d1[["country_code", "change_cw"]].merge(
        d2[["country_code", "change_pcw"]], on="country_code", how="inner")
    region_info = cf_df[["country_code", "region"]].drop_duplicates()
    pc = pc.merge(region_info, on="country_code", how="left")
    pc["region_full"] = pc["region"].map(REGION_NAMES).fillna(pc["region"])

    fig, ax = plt.subplots(figsize=(12, 7.2))

    # Axis limits
    xp, yp = 3, 3
    xmin, xmax = pc["change_cw"].min() - xp, pc["change_cw"].max() + xp
    ymin, ymax = pc["change_pcw"].min() - yp, pc["change_pcw"].max() + yp
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # 45-degree line
    dmin, dmax = min(xmin, ymin), max(xmax, ymax)
    ax.plot([dmin, dmax], [dmin, dmax],
            color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)

    # Two-layer scatter
    pc["labeled"] = pc["country_code"].isin(LABEL_COUNTRIES)
    unlabeled = pc[~pc["labeled"]]
    labeled = pc[pc["labeled"]]

    # Layer 1: unlabeled (small, transparent)
    for region in REGION_ORDER:
        sub = unlabeled[unlabeled["region_full"] == region]
        if len(sub) > 0:
            ax.scatter(sub["change_cw"], sub["change_pcw"],
                       c=REGION_COLORS[region], s=45, alpha=0.25,
                       edgecolor='none', zorder=2)

    # Layer 2: labeled (larger, opaque)
    for region in REGION_ORDER:
        sub = labeled[labeled["region_full"] == region]
        if len(sub) > 0:
            ax.scatter(sub["change_cw"], sub["change_pcw"],
                       c=REGION_COLORS[region], s=130, alpha=0.9,
                       edgecolor='white', linewidth=1.3, zorder=3,
                       label=region)

    # Layer 3: text labels
    texts = []
    for _, row in labeled.iterrows():
        cc = str(row["country_code"])[:3]
        texts.append(ax.text(row["change_cw"], row["change_pcw"], cc,
                             fontsize=10.5, ha="center", va="center",
                             fontweight="medium", alpha=0.9, zorder=4))

    if adjust_text is not None and texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray",
                                   alpha=0.4, lw=0.5),
                    expand=(1.5, 1.5), force_text=(0.8, 0.8),
                    only_move="xy")

    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel("Geopolitical Growth Effects (1960\u20131990)", fontsize=14)
    ax.set_ylabel("Geopolitical Growth Effects (1991\u20132024)", fontsize=14)
    ax.legend(frameon=False, fontsize=10, loc="lower right",
              markerscale=0.8, handletextpad=0.3)
    ax.grid(True, ls="--", alpha=0.3)
    sns.despine(ax=ax)
    fig.tight_layout()
    savefig(fig, FIGURES, "Fig18_growth_scatter.pdf")
    plt.close(fig)
    print(f"  Labeled: {len(labeled)}, Unlabeled: {len(unlabeled)}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Fig 17-18: Growth Accounting")
    print("=" * 70)

    # Load data
    df = read_csv_fallback(PANEL)
    for col in ["country_code"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["year"] = df["year"].astype(int)
    for col in ["y_ext", "geo_relation_dyn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Balanced sample for LP estimation
    df_bal = restrict_balanced_sample(df)
    df_bal = df_bal[(df_bal["year"] >= 1960) & (df_bal["year"] <= 2024)]
    df_bal = prepare_panel(df_bal)
    print(f"Balanced panel: {df_bal.shape}, {df_bal['country_code'].nunique()} countries")

    # Step 1: Estimate IRFs
    print("\n--- Estimating IRFs ---")
    _, phi_p = lp_irf_simple(df_bal, "geo_relation_dyn", "geo_relation_dyn",
                             range(0, H + 1))
    _, alpha = lp_irf_simple(df_bal, "geo_relation_dyn", "y_ext",
                             range(0, H + 1))

    if np.isnan(phi_p).all() or np.isnan(alpha).all():
        print("ERROR: IRF estimation failed.")
        return

    # Transitory and permanent IRFs
    p_shock = compute_auxiliary_shocks(phi_p, H)
    P_mat = np.zeros((H + 1, H + 1))
    for i in range(H + 1):
        for j in range(i + 1):
            P_mat[i, j] = p_shock[i - j] if i - j < len(p_shock) else 0
    alpha_transitory = P_mat @ alpha
    alpha_permanent = np.cumsum(alpha_transitory)
    longrun_effect = alpha_permanent[H]
    print(f"Long-run effect at H={H}: {longrun_effect:.4f}")

    # Step 2: Counterfactual GDP (full sample)
    print("\n--- Counterfactual GDP ---")
    df_full = df[(df["year"] >= 1960) & (df["year"] <= 2024)].copy()
    df_full = prepare_panel(df_full)
    cf_df = compute_counterfactual_gdp(df_full, alpha_transitory)

    # Step 3: Decade effects
    print("\n--- Decade effects ---")
    decades = [(1960, 1969), (1970, 1979), (1980, 1989),
               (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2024)]
    decade_changes = calculate_decade_changes(df_full)
    decade_changes["longrun_effect"] = decade_changes["geo_change"] * longrun_effect
    contemp_effects = calculate_contemporaneous_effects(
        df_full, alpha_transitory, decades)

    # Cache
    cf_result = cf_df[["country_code", "year", "y_ext", "delta_y_geo", "region"]].dropna(
        subset=["delta_y_geo"])
    cf_result.to_csv(CACHE / "counterfactual_gdp_results_2024.csv", index=False)
    decade_changes.to_csv(CACHE / "decade_longrun_effects_2024.csv", index=False)
    contemp_effects.to_csv(CACHE / "decade_contemporaneous_effects_2024.csv", index=False)

    irf_df = pd.DataFrame({
        "horizon": np.arange(H + 1),
        "permanent_irf": alpha_permanent,
        "transitory_irf": alpha_transitory,
        "longrun_effect": longrun_effect,
    })
    irf_df.to_csv(CACHE / "irf_permanent_transitory_2024.csv", index=False)

    # Step 4: Plots
    print("\n--- Plotting ---")
    plot_decade_boxplots(decade_changes, contemp_effects)
    plot_period_scatter(cf_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
