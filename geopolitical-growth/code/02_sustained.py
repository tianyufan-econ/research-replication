"""
02_sustained.py
===============
Figure 7: GDP paths associated with (a) a transitory and (b) a sustained alignment
path, with 95% bands from 500 country-block bootstrap draws, and the two footnote
checks on the 25-year sustained-path coefficient.

Local projections of p_{c,t+h} and y_{c,t+h} on p_{ct} (h = 0..25) with four lags of
GDP and of the index, country and region-year fixed effects, long-coverage sample.
The persistence coefficients give the auxiliary shocks that hold the index at
(1, 0, ..., 0); applied to the GDP coefficients they give the transitory path
(alpha tilde), whose cumulative sum is the sustained path. Bands: 2.5th and 97.5th
percentiles of 500 country-block bootstrap draws (seed 20252026 + b). Footnote: a
delta-method interval from the stacked influence functions clustered by country and
by base year, and a split-panel jackknife (Dhaene and Jochmans 2015). Paths are
reported per one standard deviation of the index.

Full-package version: code/02_baseline/lp_transitory_bootstrap.py (--run-bootstrap,
figures and interval), lp_transitory_reg.py (point estimates) and
sustained_path_inference.py (footnote).
Outputs: output/figures/Fig07{a,b}_*.pdf,
         output/cache/transitory_original_estimates.csv,
         output/cache/transitory_bootstrap_{alpha_tilde,cumulative}_500iter.csv,
         output/cache/sustained_path_inference.csv, output/numbers/02_sustained.csv
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhdfe
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import linalg

import plot_style as PS
from config import CACHE, GEO_SD, PANEL, ensure_dirs, read_csv_fallback, restrict_balanced_sample, save_numbers, savefig

ensure_dirs()
PS.apply_theme()
warnings.filterwarnings("ignore")

BLUE = sns.color_palette("deep")[0]
H = 25
LAGS = 4
SHOCK = "geo_relation_dyn"
X_COLS = [SHOCK] + [f"y_ext_lag{i}" for i in range(1, LAGS + 1)] + [f"{SHOCK}_lag{i}" for i in range(1, LAGS + 1)]
N_BOOT, SEED, MIN_VALID = 500, 20252026, 100
ALPHA = 0.025                                   # 95% percentile band
FULL = "02_baseline/"


def load():
    df = read_csv_fallback(PANEL, low_memory=False)
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)
    return df


def add_lags(df):
    """Country index, region-year cell and four lags of GDP and of the index."""
    df = df.copy()
    df["country_idx"] = pd.factorize(df["country_code"])[0]
    df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    for l in range(1, LAGS + 1):
        df[f"y_ext_lag{l}"] = df.groupby("country_idx")["y_ext"].shift(l)
        df[f"{SHOCK}_lag{l}"] = df.groupby("country_idx")[SHOCK].shift(l)
    return df


# ── Point estimates and bootstrap draws (lp_transitory_bootstrap.py) ──
def lp_irf(df, dep):
    """Coefficients on the index at h = 0..H, OLS with country and region-year dummies."""
    irf = np.full(H + 1, np.nan)
    for h in range(H + 1):
        tmp = df.copy()
        tmp["dep_var"] = tmp.groupby("country_idx")[dep].shift(-h)
        sub = tmp.dropna(subset=["dep_var"] + X_COLS + ["country_idx", "region_year"])
        if sub.shape[0] < len(X_COLS) + sub["country_idx"].nunique() + sub["region_year"].nunique() + 10:
            continue
        fmla = f"dep_var ~ {' + '.join(X_COLS)} + C(country_idx) + C(region_year)"
        res = smf.ols(fmla, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["country_idx"]})
        irf[h] = res.params.get(SHOCK, np.nan)
    return np.arange(H + 1), irf


def lp_irf_fast(df, dep):
    """The same coefficients with the fixed effects absorbed by pyhdfe (bootstrap draws)."""
    irf = np.full(H + 1, np.nan)
    for h in range(H + 1):
        tmp = df.copy()
        tmp["dep_var"] = tmp.groupby("country_idx")[dep].shift(-h)
        sub = tmp.dropna(subset=["dep_var"] + X_COLS + ["country_idx", "region_year"])
        if sub.shape[0] < len(X_COLS) + 50:
            continue
        try:
            fe_ids = np.column_stack([pd.factorize(sub["country_idx"])[0], pd.factorize(sub["region_year"])[0]])
            resid = pyhdfe.create(fe_ids).residualize(sub[["dep_var"] + X_COLS].values.astype(np.float64))
            y_r, X_r = resid[:, 0], resid[:, 1:]
            irf[h] = linalg.solve(X_r.T @ X_r, X_r.T @ y_r)[0]
        except Exception:
            continue
    return irf


def counterfactual(phi_p, alpha):
    """Transitory path alpha tilde = P^shock alpha and the sustained path, its cumulative sum."""
    phi = np.eye(H + 1)
    for i in range(1, H + 1):
        for j in range(i):
            phi[i, j] = phi_p[i - j]
    e0 = np.zeros(H + 1)
    e0[0] = 1
    try:
        p_shock = np.linalg.inv(phi) @ e0
    except np.linalg.LinAlgError:
        p_shock = np.zeros(H + 1)
    P = np.zeros((H + 1, H + 1))
    for i in range(H + 1):
        for j in range(i + 1):
            P[i, j] = p_shock[i - j]
    alpha_tilde = P @ alpha
    return alpha_tilde, np.cumsum(alpha_tilde)


def resample(df, seed):
    """Country-block bootstrap sample: countries drawn with replacement, each draw a new country."""
    np.random.seed(seed)
    ids = df["country_idx"].unique()
    drawn = np.random.choice(ids, size=len(ids), replace=True)
    parts = []
    for new_idx, c in enumerate(drawn):
        part = df[df["country_idx"] == c].copy()
        part["country_idx"] = new_idx
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def bootstrap(df):
    """500 draws of the transitory and sustained paths."""
    draws_a, draws_c = [], []
    for b in range(N_BOOT):
        try:
            df_b = resample(df, SEED + b)
            phi_b, alpha_b = lp_irf_fast(df_b, SHOCK), lp_irf_fast(df_b, "y_ext")
            if np.isnan(phi_b).all() or np.isnan(alpha_b).all():
                continue
            a, c = counterfactual(phi_b, alpha_b)
            draws_a.append(a)
            draws_c.append(c)
        except Exception:
            continue
        if (b + 1) % 100 == 0:
            print(f"  bootstrap draw {b + 1}/{N_BOOT}", flush=True)
    return np.array(draws_a), np.array(draws_c)


def band(draws):
    valid = draws[~np.isnan(draws).all(axis=1)]
    if len(valid) < MIN_VALID:
        return None
    return np.percentile(valid, ALPHA * 100, axis=0), np.percentile(valid, (1 - ALPHA) * 100, axis=0)


def plot_path(h, irf, ci, label, ylabel, fname):
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    ok = ~np.isnan(irf)
    ax.plot(h[ok], irf[ok], label=label, color=BLUE, linestyle="-", linewidth=2.5, marker="o", markersize=6)
    if ci is not None:
        lo, hi = ci
        ok_ci = ~(np.isnan(lo) | np.isnan(hi)) & ok
        if ok_ci.any():
            ax.fill_between(h[ok_ci], lo[ok_ci], hi[ok_ci], color=BLUE, alpha=PS.CI_ALPHA,
                            label=PS.LEGEND_CI_BOOT_COUNTRY)
            for bnd in (lo, hi):
                ax.plot(h[ok_ci], bnd[ok_ci], color=BLUE, linestyle="--", linewidth=PS.CI_BOUND_LW,
                        alpha=PS.CI_BOUND_ALPHA)
    ax.axhline(0, **PS.HLINE_KW)
    ax.axvline(0, **PS.VLINE_KW)
    ax.set_xlabel(PS.IRF_XLABEL, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    savefig(fig, fname)
    plt.close(fig)


def paths(df_full, numbers):
    """Figure 7(a)-(b): point paths, bootstrap bands, and the sustained-path interval."""
    df = add_lags(restrict_balanced_sample(df_full.copy()))
    h, phi_p = lp_irf(df, SHOCK)
    _, alpha = lp_irf(df, "y_ext")
    alpha_tilde, cumulative = counterfactual(phi_p, alpha)
    draws_a, draws_c = bootstrap(df)
    ci_a, ci_c = band(draws_a), band(draws_c)

    pd.DataFrame({"horizon": h, "alpha_tilde": alpha_tilde, "cumulative_irf": cumulative,
                  "alpha_tilde_lower": ci_a[0], "alpha_tilde_upper": ci_a[1],
                  "cumulative_lower": ci_c[0], "cumulative_upper": ci_c[1]}).to_csv(
        CACHE / "transitory_original_estimates.csv", index=False)
    for name, draws in (("alpha_tilde", draws_a), ("cumulative", draws_c)):
        out = pd.DataFrame(draws, columns=[f"h_{k}" for k in h])
        out["iteration"] = range(len(draws))
        out.to_csv(CACHE / f"transitory_bootstrap_{name}_{N_BOOT}iter.csv", index=False)

    # Report per one SD of the index (caches stay per unit)
    a_sd, c_sd = alpha_tilde * GEO_SD, cumulative * GEO_SD
    ci_a, ci_c = (ci_a[0] * GEO_SD, ci_a[1] * GEO_SD), (ci_c[0] * GEO_SD, ci_c[1] * GEO_SD)
    plot_path(h, a_sd, ci_a, "Counterfactual GDP Response", PS.IRF_YLABEL_GDP, "Fig07a_transitory.pdf")
    plot_path(h, c_sd, ci_c, "Cumulative GDP Response", PS.IRF_YLABEL_CUM, "Fig07b_sustained.pdf")

    sec = "Section 3.2, Economic Magnitude"
    numbers += [
        {"key": "transitory_h0", "value": a_sd[0], "paper": "Section 3.2, Figure 7(a)",
         "full_ref": f"{FULL}lp_transitory_reg.py:transitory_h0"},
        {"key": "transitory_h1", "value": a_sd[1], "paper": "Section 3.2, Figure 7(a)",
         "full_ref": f"{FULL}lp_transitory_reg.py:transitory_h1"},
        {"key": "sustained_h25", "value": c_sd[H], "paper": "Section 3.2, Figure 7(b) and Economic Magnitude",
         "full_ref": f"{FULL}lp_transitory_reg.py:sustained_h25"},
        {"key": "sustained_h25_ci_lo", "value": ci_c[0][H], "paper": sec,
         "full_ref": f"{FULL}lp_transitory_bootstrap.py:sustained_h25_ci_lo"},
        {"key": "sustained_h25_ci_hi", "value": ci_c[1][H], "paper": sec,
         "full_ref": f"{FULL}lp_transitory_bootstrap.py:sustained_h25_ci_hi"},
        {"key": "sigma_pooled", "value": GEO_SD, "paper": sec,
         "full_ref": f"{FULL}lp_transitory_bootstrap.py:sigma_pooled"},
    ]
    return c_sd[H]


# ── Footnote: two-way clustered interval and split-panel jackknife ────
#    (sustained_path_inference.py)
def inference_panel(df_full):
    df = restrict_balanced_sample(df_full.copy()).sort_values(["country_code", "year"]).reset_index(drop=True)
    df["cidx"] = pd.factorize(df["country_code"])[0]
    df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    g = df.groupby("cidx")
    for l in range(1, LAGS + 1):
        df[f"y_ext_lag{l}"] = g["y_ext"].shift(l)
        df[f"{SHOCK}_lag{l}"] = g[SHOCK].shift(l)
    for h in range(H + 1):
        df[f"y_ext_f{h}"] = g["y_ext"].shift(-h)
        df[f"{SHOCK}_f{h}"] = g[SHOCK].shift(-h)
    return df


def fe_fit(sub, dep):
    """Coefficient on the index with country and region-year FE, and its influence function."""
    ids = np.column_stack([pd.factorize(sub["cidx"])[0], pd.factorize(sub["region_year"])[0]])
    r = pyhdfe.create(ids, drop_singletons=False).residualize(sub[[dep] + X_COLS].to_numpy(float))
    y, Xm = r[:, 0], r[:, 1:]
    A_inv = np.linalg.inv(Xm.T @ Xm)
    b = A_inv @ Xm.T @ y
    psi = (Xm * (y - Xm @ b)[:, None]) @ A_inv.T
    return b[0], pd.Series(psi[:, 0], index=sub.index)


def sustained(theta):
    """theta = (rho_1..rho_H, alpha_0..alpha_H) -> sustained-path coefficient at H (per unit)."""
    rho, alpha = np.r_[1.0, theta[:H]], theta[H:]
    R = np.eye(H + 1)
    for i in range(1, H + 1):
        for j in range(i):
            R[i, j] = rho[i - j]
    p_shock = np.linalg.solve(R, np.eye(H + 1)[:, 0])
    P = np.zeros((H + 1, H + 1))
    for i in range(H + 1):
        for j in range(i + 1):
            P[i, j] = p_shock[i - j]
    return np.cumsum(P @ alpha)[H]


def inference(df_full, s25_baseline, numbers):
    df = inference_panel(df_full)
    specs = [(SHOCK, h) for h in range(1, H + 1)] + [("y_ext", h) for h in range(H + 1)]
    theta, infl, theta_j = [], [], []
    for dep, h in specs:
        col = f"{dep}_f{h}"
        sub = df.dropna(subset=[col] + X_COLS + ["region_year"])
        b, psi = fe_fit(sub, col)
        mid = np.median(np.unique(sub["year"]))
        b1, _ = fe_fit(sub[sub["year"] <= mid], col)
        b2, _ = fe_fit(sub[sub["year"] > mid], col)
        theta.append(b)
        infl.append(psi)
        theta_j.append(2 * b - 0.5 * (b1 + b2))
    theta, theta_j = np.array(theta), np.array(theta_j)

    s25 = sustained(theta)
    if abs(s25 * GEO_SD - s25_baseline) > 1e-6:
        raise RuntimeError(f"Sustained path {s25 * GEO_SD:.6f} does not reproduce the baseline {s25_baseline:.6f}")

    # Joint covariance: two-way clustered (country, base year) on the stacked influence functions
    P = pd.concat(infl, axis=1).fillna(0.0)
    country, year = df.loc[P.index, "cidx"].to_numpy(), df.loc[P.index, "year"].to_numpy()
    P = P.to_numpy()
    meat = lambda g: (lambda G: G.T @ G)(pd.DataFrame(P).groupby(g).sum().to_numpy())
    V = meat(country) + meat(year) - P.T @ P
    grad = np.array([(sustained(theta + 1e-6 * e) - sustained(theta - 1e-6 * e)) / 2e-6 for e in np.eye(len(theta))])
    se = float(np.sqrt(grad @ V @ grad))
    s25_j = sustained(theta_j)

    pd.DataFrame({"coefficient": [f"rho_{h}" for h in range(1, H + 1)] + [f"alpha_{h}" for h in range(H + 1)],
                  "full": theta, "split_panel_jackknife": theta_j}).to_csv(CACHE / "sustained_path_inference.csv",
                                                                         index=False)
    sec, ref = "Section 3.2, footnote to Economic Magnitude", f"{FULL}sustained_path_inference.py"
    numbers += [
        {"key": "sustained_h25_ci_lo_twoway", "value": (s25 - 1.96 * se) * GEO_SD, "paper": sec,
         "full_ref": f"{ref}:sustained_h25_ci_lo_twoway"},
        {"key": "sustained_h25_ci_hi_twoway", "value": (s25 + 1.96 * se) * GEO_SD, "paper": sec,
         "full_ref": f"{ref}:sustained_h25_ci_hi_twoway"},
        {"key": "sustained_h25_spj", "value": s25_j * GEO_SD, "paper": sec, "full_ref": f"{ref}:sustained_h25_spj"},
    ]


def main():
    df = load()
    numbers = []
    s25 = paths(df, numbers)
    inference(df, s25, numbers)
    save_numbers(__file__, numbers)


if __name__ == "__main__":
    main()
