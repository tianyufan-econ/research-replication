"""
02_transitory.py
================
Replication script for transitory/permanent IRF decomposition (Figures 7a, 7b).

Produces two figures + bootstrap cache:
  Fig 7a: Transitory counterfactual GDP IRF  (with bootstrap CI)
  Fig 7b: Permanent cumulative GDP IRF       (with bootstrap CI)

Methodology
-----------
1. Estimate the direct GDP IRF alpha_h  (geo_relation_dyn -> y_ext, h=0..25).
2. Estimate the self-IRF rho_h          (geo_relation_dyn -> geo_relation_dyn).
3. Compute auxiliary shocks that produce a purely transitory (1,0,0,...) path
   for geo_relation_dyn, given the self-persistence encoded in rho_h.
   Specifically, construct the lower-triangular Phi matrix from rho_h,
   invert it, and multiply by the transitory impulse vector [1, 0, ..., 0].
4. The transitory counterfactual GDP IRF is:
      alpha_tilde = P_shock @ alpha
   where P_shock is the lower-triangular convolution matrix built from the
   auxiliary shocks.
5. The permanent counterfactual is the cumulative sum of alpha_tilde.
6. Bootstrap: resample countries with replacement (500 iterations), re-
   estimate alpha and rho on each bootstrap sample, recompute alpha_tilde
   and cumulative, then take 2.5th/97.5th percentile CIs.

Self-contained: loads data, estimates, bootstraps, and plots in one file.
"""

# ── Imports ──────────────────────────────────────────────────────────────
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import (PANEL, FIGURES, CACHE, ensure_dirs, savefig,
                    read_csv_fallback, restrict_balanced_sample, LP_DEFAULTS)
from lp_utils import (prepare_panel, lp_irf, lp_irf_fast,
                       bootstrap_resample_countries)
import plot_style as PS

PS.apply_theme()
ensure_dirs()

SEABORN_BLUE = sns.color_palette("deep")[0]

# ── Bootstrap configuration ─────────────────────────────────────────────
N_BOOTSTRAP = 500
CONFIDENCE_LEVEL = 0.95
ALPHA_LEVEL = (1 - CONFIDENCE_LEVEL) / 2
RANDOM_SEED = 20252026
H = 25  # maximum horizon


# ── Counterfactual construction ──────────────────────────────────────────

def construct_phi_matrix(phi_p, H):
    """
    Build (H+1) x (H+1) lower-triangular Phi^p matrix from the self-IRF.

    Phi[i, j] = phi_p[i-j]  for i >= j, else 0.
    Phi[0, 0] = phi_p[0] (impact = 1 by construction).
    """
    n = H + 1
    Phi = np.eye(n)
    for i in range(1, n):
        for j in range(i):
            gap = i - j
            Phi[i, j] = phi_p[gap] if gap < len(phi_p) else 0.0
    return Phi


def compute_auxiliary_shocks(phi_p, H):
    """
    Compute the auxiliary shock sequence that produces a purely
    transitory (1, 0, 0, ..., 0) path for geo_relation_dyn.

    p_shock = Phi^{-1} @ [1, 0, ..., 0]
    """
    Phi = construct_phi_matrix(phi_p, H)
    target = np.zeros(H + 1)
    target[0] = 1.0
    try:
        p_shock = np.linalg.inv(Phi) @ target
    except np.linalg.LinAlgError:
        p_shock = np.zeros(H + 1)
    return p_shock


def construct_p_shock_matrix(p_shock, H):
    """
    Build (H+1) x (H+1) lower-triangular P_shock convolution matrix.

    P_shock[i, j] = p_shock[i-j]  for i >= j, else 0.
    """
    n = H + 1
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            gap = i - j
            P[i, j] = p_shock[gap] if gap < len(p_shock) else 0.0
    return P


def compute_alpha_tilde(P_shock, alpha):
    """Counterfactual transitory GDP IRF: alpha_tilde = P_shock @ alpha."""
    return P_shock @ alpha


def compute_cumulative(alpha_tilde):
    """Permanent counterfactual: cumulative sum of alpha_tilde."""
    return np.cumsum(alpha_tilde)


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_irf_with_ci(h_vals, irf, ci, label, color_key, fname, ylabel):
    """Plot IRF with bootstrap confidence band."""
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    valid = ~np.isnan(irf)
    ax.plot(h_vals[valid], irf[valid], label=label,
            color=SEABORN_BLUE, ls="-", lw=2.5, marker="o", ms=6)

    if ci is not None:
        lo, hi = ci
        ci_ok = ~(np.isnan(lo) | np.isnan(hi)) & valid
        if ci_ok.any():
            ax.fill_between(h_vals[ci_ok], lo[ci_ok], hi[ci_ok],
                            color=SEABORN_BLUE, alpha=PS.CI_ALPHA,
                            label=PS.LEGEND_CI_BOOT_COUNTRY)
            ax.plot(h_vals[ci_ok], lo[ci_ok], color=SEABORN_BLUE,
                    ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)
            ax.plot(h_vals[ci_ok], hi[ci_ok], color=SEABORN_BLUE,
                    ls="--", lw=PS.CI_BOUND_LW, alpha=PS.CI_BOUND_ALPHA)

    ax.axhline(0, **PS.HLINE_KW)
    ax.axvline(0, **PS.VLINE_KW)
    ax.set_xlabel(PS.IRF_XLABEL, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    savefig(fig, FIGURES, fname)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    shock = "geo_relation_dyn"
    y_lags = LP_DEFAULTS["y_lags"]
    shock_lags = LP_DEFAULTS["shock_lags"]
    horizons = range(0, H + 1)

    # ------------------------------------------------------------------
    # Load and prepare data  (balanced sample)
    # ------------------------------------------------------------------
    print("Loading panel ...")
    df_full = read_csv_fallback(PANEL)
    df_full["country_code"] = df_full["country_code"].astype(str)
    df_full["year"] = df_full["year"].astype(int)
    df = restrict_balanced_sample(df_full)
    df = prepare_panel(df, shock_var=shock,
                       y_lags=y_lags, shock_lags=shock_lags)
    print(f"  Balanced: {df['country_code'].nunique()} countries, "
          f"{df.shape[0]} obs")

    # ------------------------------------------------------------------
    # Original estimates
    # ------------------------------------------------------------------
    print("\nEstimating original IRFs ...")
    res_rho = lp_irf(df, shock_var=shock, y_var=shock,
                     horizon_range=horizons,
                     y_lags=y_lags, shock_lags=shock_lags)
    res_alpha = lp_irf(df, shock_var=shock, y_var="y_ext",
                       horizon_range=horizons,
                       y_lags=y_lags, shock_lags=shock_lags)

    h_vals = res_rho["h_vals"]
    rho_orig = res_rho["coef"]
    alpha_orig = res_alpha["coef"]

    if np.isnan(rho_orig).all() or np.isnan(alpha_orig).all():
        print("CRITICAL: original IRF estimation returned all NaN. Exiting.")
        return

    # Counterfactual
    p_shock_orig = compute_auxiliary_shocks(rho_orig, H)
    P_shock_orig = construct_p_shock_matrix(p_shock_orig, H)
    alpha_tilde_orig = compute_alpha_tilde(P_shock_orig, alpha_orig)
    cumulative_orig = compute_cumulative(alpha_tilde_orig)

    print(f"  Peak transitory |alpha_tilde|: "
          f"{np.nanmax(np.abs(alpha_tilde_orig)):.4f}")
    print(f"  Peak cumulative |cum|: "
          f"{np.nanmax(np.abs(cumulative_orig)):.4f}")

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    print(f"\nBootstrap ({N_BOOTSTRAP} iterations) ...")
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    boot_alpha_tilde = []
    boot_cumulative = []
    n_ok = 0

    for b in tqdm(range(N_BOOTSTRAP), desc="Bootstrap"):
        seed_b = (RANDOM_SEED + b) if RANDOM_SEED is not None else None
        df_b = bootstrap_resample_countries(df, random_seed=seed_b)
        # Re-prepare after resampling (new country_code identifiers)
        df_b = prepare_panel(df_b, shock_var=shock,
                             y_lags=y_lags, shock_lags=shock_lags)
        try:
            rho_b = lp_irf_fast(df_b, shock_var=shock, y_var=shock,
                                horizon_range=horizons,
                                y_lags=y_lags, shock_lags=shock_lags)["coef"]
            alpha_b = lp_irf_fast(df_b, shock_var=shock, y_var="y_ext",
                                  horizon_range=horizons,
                                  y_lags=y_lags, shock_lags=shock_lags)["coef"]

            if np.isnan(rho_b).all() or np.isnan(alpha_b).all():
                continue

            ps_b = compute_auxiliary_shocks(rho_b, H)
            Ps_b = construct_p_shock_matrix(ps_b, H)
            at_b = compute_alpha_tilde(Ps_b, alpha_b)
            cm_b = compute_cumulative(at_b)

            boot_alpha_tilde.append(at_b)
            boot_cumulative.append(cm_b)
            n_ok += 1
        except Exception:
            continue

    print(f"  Successful: {n_ok}/{N_BOOTSTRAP} "
          f"({n_ok / N_BOOTSTRAP * 100:.1f}%)")

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------
    alpha_tilde_ci = None
    cumulative_ci = None

    if n_ok >= 50:
        arr_at = np.array(boot_alpha_tilde)
        arr_cm = np.array(boot_cumulative)

        at_lo = np.percentile(arr_at, ALPHA_LEVEL * 100, axis=0)
        at_hi = np.percentile(arr_at, (1 - ALPHA_LEVEL) * 100, axis=0)
        alpha_tilde_ci = (at_lo, at_hi)

        cm_lo = np.percentile(arr_cm, ALPHA_LEVEL * 100, axis=0)
        cm_hi = np.percentile(arr_cm, (1 - ALPHA_LEVEL) * 100, axis=0)
        cumulative_ci = (cm_lo, cm_hi)
    else:
        print("  WARNING: too few valid iterations for CIs.")

    # ------------------------------------------------------------------
    # Save bootstrap cache
    # ------------------------------------------------------------------
    cache_path = CACHE / f"bootstrap_transitory_{N_BOOTSTRAP}iter.pkl"
    results = {
        "h_vals": h_vals,
        "alpha_tilde_orig": alpha_tilde_orig,
        "cumulative_orig": cumulative_orig,
        "bootstrap_alpha_tilde": np.array(boot_alpha_tilde) if boot_alpha_tilde else None,
        "bootstrap_cumulative": np.array(boot_cumulative) if boot_cumulative else None,
        "alpha_tilde_ci": alpha_tilde_ci,
        "cumulative_ci": cumulative_ci,
        "metadata": {
            "n_bootstrap_requested": N_BOOTSTRAP,
            "n_bootstrap_successful": n_ok,
            "confidence_level": CONFIDENCE_LEVEL,
            "horizon": H,
            "random_seed": RANDOM_SEED,
            "timestamp": pd.Timestamp.now().isoformat(),
        },
    }
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Bootstrap cache saved: {cache_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nPlotting ...")

    # Fig 7a — Transitory
    plot_irf_with_ci(h_vals, alpha_tilde_orig, alpha_tilde_ci,
                     "Counterfactual GDP Response", "irf",
                     "Fig7A_transitory_irf.pdf",
                     PS.IRF_YLABEL_GDP)

    # Fig 7b — Permanent (cumulative)
    plot_irf_with_ci(h_vals, cumulative_orig, cumulative_ci,
                     "Cumulative GDP Response", "cumulative",
                     "Fig7B_permanent_irf.pdf",
                     PS.IRF_YLABEL_CUM)

    print("\n=== 02_transitory.py complete ===")


if __name__ == "__main__":
    main()
