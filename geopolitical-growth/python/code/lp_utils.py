"""
lp_utils.py
===========
Shared Local Projection IRF estimation functions for the Geopolitical Growth project.

Extracted from [02a]lp_baseline.ipynb. Every regression script imports from here
instead of re-implementing the LP estimation loop.

Usage:
    from lp_utils import lp_irf, lp_irf_joint, prepare_panel, save_irf, load_irf
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats
from scipy import linalg
import pyhdfe


# ── Data preparation ──────────────────────────────────────────────────

def prepare_panel(df, y_var="y_ext", shock_var="geo_relation_dyn",
                  y_lags=4, shock_lags=4, extra_shock_vars=None):
    """
    Add country index, region-year FE column, and lags.

    Parameters
    ----------
    df : DataFrame with columns [country_code, year, region, y_var, shock_var]
    y_var : str — outcome variable name
    shock_var : str — main shock variable name
    y_lags : int — number of lags for outcome
    shock_lags : int — number of lags for shock
    extra_shock_vars : list of str — additional shock variables to lag

    Returns
    -------
    DataFrame with added lag columns, country_idx, region_year
    """
    df = df.copy()
    df["country_idx"] = pd.factorize(df["country_code"])[0]

    if "region" in df.columns:
        df["region_year"] = df["region"].astype(str) + "_" + df["year"].astype(str)
    else:
        df["region_year"] = "region1_" + df["year"].astype(str)

    # Outcome lags
    for l in range(1, y_lags + 1):
        df[f"{y_var}_lag{l}"] = df.groupby("country_idx")[y_var].shift(l)

    # Shock lags
    for l in range(1, shock_lags + 1):
        df[f"{shock_var}_lag{l}"] = df.groupby("country_idx")[shock_var].shift(l)

    # Extra shock variable lags
    if extra_shock_vars:
        for var in extra_shock_vars:
            for l in range(1, shock_lags + 1):
                df[f"{var}_lag{l}"] = df.groupby("country_idx")[var].shift(l)

    return df


# ── Core LP-IRF estimation ───────────────────────────────────────────

def lp_irf(df, shock_var="geo_relation_dyn", y_var="y_ext",
           horizon_range=range(-10, 26), y_lags=4, shock_lags=4,
           fe="region_year", extra_controls=None, cov_type="kernel"):
    """
    Univariate Local Projection IRF with Driscoll-Kraay or clustered SEs.

    Parameters
    ----------
    df : DataFrame — prepared panel (output of prepare_panel)
    shock_var : str — treatment variable
    y_var : str — outcome variable
    horizon_range : iterable of int — horizons to estimate
    y_lags : int — number of outcome lags included
    shock_lags : int — number of shock lags included
    fe : str — "region_year" (default), "year", or None for entity-only
    extra_controls : list of str — additional control variable names
    cov_type : str — "kernel" (Driscoll-Kraay) or "clustered"

    Returns
    -------
    dict with keys: h_vals, coef, se, ci_lower, ci_upper, nobs
    """
    y_lag_cols = [f"{y_var}_lag{i}" for i in range(1, y_lags + 1)]
    shock_lag_cols = [f"{shock_var}_lag{i}" for i in range(1, shock_lags + 1)]

    h_vals = list(horizon_range)
    n_h = len(h_vals)
    coef = np.full(n_h, np.nan)
    se = np.full(n_h, np.nan)
    nobs = np.full(n_h, np.nan)

    X_cols = list(dict.fromkeys([shock_var] + y_lag_cols + shock_lag_cols))
    if extra_controls:
        X_cols = list(dict.fromkeys(X_cols + extra_controls))

    required = X_cols + ["country_idx", "year", y_var]
    if fe == "region_year":
        required.append("region_year")

    for idx, h in enumerate(h_vals):
        tmp = df.copy()
        tmp["dep_var"] = tmp.groupby("country_idx")[y_var].shift(-h)

        sub = tmp.dropna(subset=["dep_var"] + X_cols + (["region_year"] if fe == "region_year" else []))

        if sub.shape[0] < len(X_cols) + 50:
            continue

        sub_panel = sub.set_index(["country_idx", "year"])
        y = sub_panel["dep_var"]
        X = sub_panel[X_cols]

        try:
            if fe == "region_year":
                model = PanelOLS(
                    y, X,
                    entity_effects=True,
                    time_effects=False,
                    other_effects=sub_panel["region_year"],
                )
            elif fe == "year":
                model = PanelOLS(
                    y, X,
                    entity_effects=True,
                    time_effects=True,
                )
            else:
                model = PanelOLS(
                    y, X,
                    entity_effects=True,
                    time_effects=False,
                )

            res = model.fit(cov_type=cov_type)
            coef[idx] = res.params.get(shock_var, np.nan)
            se[idx] = res.std_errors.get(shock_var, np.nan)
            nobs[idx] = res.nobs
        except Exception as e:
            print(f"LP error h={h}: {e}")

    z = stats.norm.ppf(0.975)
    return {
        "h_vals": np.array(h_vals),
        "coef": coef,
        "se": se,
        "ci_lower": coef - z * se,
        "ci_upper": coef + z * se,
        "nobs": nobs,
    }


def lp_irf_joint(df, shock_vars, y_var="y_ext",
                 horizon_range=range(-10, 26), y_lags=4, shock_lags=4,
                 fe="region_year", cov_type="kernel"):
    """
    Joint Local Projection IRF with multiple shock variables.

    Returns a dict mapping each shock_var to its IRF results dict.
    """
    y_lag_cols = [f"{y_var}_lag{i}" for i in range(1, y_lags + 1)]

    # Build all shock lag columns
    all_shock_lag_cols = []
    for sv in shock_vars:
        for l in range(1, shock_lags + 1):
            all_shock_lag_cols.append(f"{sv}_lag{l}")

    X_cols = list(shock_vars) + y_lag_cols + all_shock_lag_cols
    h_vals = list(horizon_range)
    n_h = len(h_vals)

    results = {}
    for sv in shock_vars:
        results[sv] = {
            "h_vals": np.array(h_vals),
            "coef": np.full(n_h, np.nan),
            "se": np.full(n_h, np.nan),
            "nobs": np.full(n_h, np.nan),
        }

    for idx, h in enumerate(h_vals):
        tmp = df.copy()
        tmp["dep_var"] = tmp.groupby("country_idx")[y_var].shift(-h)

        sub = tmp.dropna(subset=["dep_var"] + X_cols + (["region_year"] if fe == "region_year" else []))
        if sub.shape[0] < len(X_cols) + 50:
            continue

        sub_panel = sub.set_index(["country_idx", "year"])
        y = sub_panel["dep_var"]
        X = sub_panel[X_cols]

        try:
            if fe == "region_year":
                model = PanelOLS(
                    y, X,
                    entity_effects=True,
                    time_effects=False,
                    other_effects=sub_panel["region_year"],
                )
            else:
                model = PanelOLS(
                    y, X,
                    entity_effects=True,
                    time_effects=True,
                )

            res = model.fit(cov_type=cov_type)
            for sv in shock_vars:
                results[sv]["coef"][idx] = res.params.get(sv, np.nan)
                results[sv]["se"][idx] = res.std_errors.get(sv, np.nan)
                results[sv]["nobs"][idx] = res.nobs
        except Exception as e:
            print(f"Joint LP error h={h}: {e}")

    z = stats.norm.ppf(0.975)
    for sv in shock_vars:
        results[sv]["ci_lower"] = results[sv]["coef"] - z * results[sv]["se"]
        results[sv]["ci_upper"] = results[sv]["coef"] + z * results[sv]["se"]

    return results


# ── Fast LP-IRF via pyhdfe (bootstrap/placebo iterations) ────────────

def lp_irf_fast(df, shock_var="geo_relation_dyn", y_var="y_ext",
                horizon_range=range(-10, 26), y_lags=4, shock_lags=4,
                fe="region_year", extra_controls=None):
    """
    Fast LP-IRF using pyhdfe for FE absorption.
    Point estimates only -- no standard errors.
    Drop-in replacement for lp_irf when SEs are not needed
    (e.g., bootstrap iterations, placebo draws).

    Returns dict with keys: h_vals, coef (se/ci_lower/ci_upper are NaN).
    """
    y_lag_cols = [f"{y_var}_lag{i}" for i in range(1, y_lags + 1)]
    shock_lag_cols = [f"{shock_var}_lag{i}" for i in range(1, shock_lags + 1)]

    h_vals = list(horizon_range)
    n_h = len(h_vals)
    coef = np.full(n_h, np.nan)
    se = np.full(n_h, np.nan)
    nobs = np.full(n_h, np.nan)

    X_cols = list(dict.fromkeys([shock_var] + y_lag_cols + shock_lag_cols))
    if extra_controls:
        X_cols = list(dict.fromkeys(X_cols + extra_controls))

    for idx, h in enumerate(h_vals):
        tmp = df.copy()
        tmp["dep_var"] = tmp.groupby("country_idx")[y_var].shift(-h)

        required = ["dep_var"] + X_cols
        if fe == "region_year":
            required.append("region_year")
        required.append("country_idx")

        sub = tmp.dropna(subset=required)

        if sub.shape[0] < len(X_cols) + 50:
            continue

        try:
            # Build FE ids
            if fe == "region_year":
                fe_ids = np.column_stack([
                    pd.factorize(sub["country_idx"])[0],
                    pd.factorize(sub["region_year"])[0],
                ])
            elif fe == "year":
                fe_ids = np.column_stack([
                    pd.factorize(sub["country_idx"])[0],
                    pd.factorize(sub["year"])[0],
                ])
            else:
                # Entity-only
                fe_ids = pd.factorize(sub["country_idx"])[0].reshape(-1, 1)

            algo = pyhdfe.create(fe_ids)
            data = sub[["dep_var"] + X_cols].values.astype(np.float64)
            resid = algo.residualize(data)

            y_r = resid[:, 0]
            X_r = resid[:, 1:]

            beta = linalg.solve(X_r.T @ X_r, X_r.T @ y_r)
            coef[idx] = beta[0]  # shock_var is first column
            nobs[idx] = sub.shape[0]
        except Exception:
            continue

    return {
        "h_vals": np.array(h_vals),
        "coef": coef,
        "se": se,
        "ci_lower": np.full(n_h, np.nan),
        "ci_upper": np.full(n_h, np.nan),
        "nobs": nobs,
    }


# ── Caching helpers ───────────────────────────────────────────────────

def save_irf(result, path):
    """Save IRF result dict as CSV."""
    df = pd.DataFrame({
        "h": result["h_vals"],
        "coef": result["coef"],
        "se": result["se"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
    })
    if "nobs" in result:
        df["nobs"] = result["nobs"]
    df.to_csv(path, index=False)
    print(f"Saved IRF: {path}")


def load_irf(path):
    """Load cached IRF result from CSV."""
    df = pd.read_csv(path)
    result = {
        "h_vals": df["h"].values,
        "coef": df["coef"].values,
        "se": df["se"].values,
        "ci_lower": df["ci_lower"].values,
        "ci_upper": df["ci_upper"].values,
    }
    if "nobs" in df.columns:
        result["nobs"] = df["nobs"].values
    return result


# ── Bootstrap utilities ────────────────────────────────────────────────

def bootstrap_resample_countries(df, random_seed=None):
    """
    Bootstrap resample by countries (cluster bootstrap).

    Resamples countries with replacement, preserving the full time series
    for each drawn country. Assigns new country identifiers to avoid
    duplicate index issues.

    Parameters
    ----------
    df : DataFrame with 'country_code' column
    random_seed : int or None

    Returns
    -------
    DataFrame — bootstrapped sample with reassigned country codes
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    unique_countries = df["country_code"].unique()
    n_countries = len(unique_countries)
    drawn = np.random.choice(unique_countries, size=n_countries, replace=True)

    parts = []
    for i, c in enumerate(drawn):
        chunk = df[df["country_code"] == c].copy()
        chunk["country_code"] = f"boot_{i}"
        parts.append(chunk)

    return pd.concat(parts, ignore_index=True)


def load_panel(panel_path=None):
    """Load the main panel dataset."""
    if panel_path is None:
        from config import PANEL
        panel_path = PANEL
    from config import read_csv_fallback
    df = read_csv_fallback(panel_path)
    if "country_code" in df:
        df["country_code"] = df["country_code"].astype(str)
    if "year" in df:
        df["year"] = df["year"].astype(int)
    return df
