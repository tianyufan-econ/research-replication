"""
plot_style.py
=============
Publication-ready figure style constants and helpers for the
Geopolitical Growth empirics project.

Usage in every notebook
-----------------------
    import sys
    sys.path.insert(0, str(ROOT / "code" / "python" / "v2"))
    import plot_style as PS
    PS.apply_theme()

All plotting functions then reference PS.* constants so that
a single edit here propagates to every figure.
"""

import seaborn as sns
import matplotlib.pyplot as plt

# ── Global theme ───────────────────────────────────────────────────────────
FONT_SCALE = 1.1   # 1.2 was too large at 0.48-textwidth subfigure size


def apply_theme():
    """Call once at the top of each notebook."""
    sns.set_theme(style="whitegrid", font_scale=FONT_SCALE, context="paper")


# ── Color palette ──────────────────────────────────────────────────────────
_PAL     = sns.color_palette("deep")
C_BLUE   = _PAL[0]   # main estimate / primary series
C_ORANGE = _PAL[1]   # overlay / secondary series
C_GREEN  = _PAL[2]   # third series (bootstrap)
C_RED    = _PAL[3]   # fourth series (wild bootstrap)


# ── Axis labels — IRF x-axis ───────────────────────────────────────────────
IRF_XLABEL = "Years Relative to Shock"


# ── Axis labels — IRF y-axis ───────────────────────────────────────────────
IRF_YLABEL_GDP        = "Log GDP per Capita (×100)"
IRF_YLABEL_CUM        = "Cumulative Log GDP per Capita (×100)"
IRF_YLABEL_GEO        = "Geopolitical Alignment Score"
IRF_YLABEL_FIRST_STAGE = "Geopolitical Alignment Score"   # first-stage IRF
IRF_YLABEL_DEM        = "Democracy Index"

# Channel-specific ylabels for [03a] and [03c]
CHANNEL_YLABELS = {
    # Panel A — output & stability
    "rgdpna"         : "Real GDP per Capita (×100)",
    "unrest_new"     : "Domestic Unrest Index",
    "csh_i"          : "Investment Share (pp)",
    "rkna"           : "Capital Stock (×100)",
    # Panel B — growth fundamentals
    "irr"            : "Internal Rate of Return (pp)",
    "ctfp"           : "TFP (×100)",
    "trade_open"     : "Trade Openness (pp)",
    "hc"             : "Human Capital Index (×100)",
    "log_hc"         : "Human Capital Index (×100)",
    # Appendix Panel A — market & education
    "market_lib"     : "Market Reform Index",
    "gov_exp"        : "Tax-to-GDP Ratio (pp)",
    "school_enroll"  : "School Enrollment Rate (pp)",
    # Appendix Panel B — employment & consumption
    "emp_pop"        : "Employment-to-Population Ratio (pp)",
    "lab_share"      : "Labor Share (pp)",
    "consumption"    : "Consumption Share (pp)",
    "absorption"     : "Absorption Share (pp)",
    # Democracy
    "dem_ANRR"       : "Democracy Index (ANRR)",
    "polity2"        : "Polity II Score",
    # Geo relation
    "geo_relation_dyn" : "Geopolitical Alignment Score",
    "geo_score"        : "Geopolitical Alignment Score",
    # [03a] covariate channel names (PWT/notebook-specific column names)
    "yPWT"               : "Real GDP per Capita, PWT (×100)",
    "log_k"              : "Capital Stock (×100)",
    "log_tfp"            : "TFP (×100)",
    "trade"              : "Trade Openness (pp)",
    # [03a] appendix channels
    "marketref"          : "Market Reform Index",
    "lgov"               : "Tax-to-GDP Ratio (×100)",
    "lprienr"            : "Primary School Enrollment (×100)",
    "lsecenr"            : "Secondary School Enrollment (×100)",
    "emp_pop_ratio"      : "Employment-to-Population Ratio (pp)",
    "labsh"              : "Labor Share (pp)",
    "real_consumption_pc": "Consumption per Capita (×100)",
    "real_absorption_pc" : "Absorption per Capita (×100)",
}


# ── Legend labels ──────────────────────────────────────────────────────────
LEGEND_POINT            = "Point Estimate"
LEGEND_CI_DK            = "95% CI (Driscoll-Kraay)"
LEGEND_CI_BOOT_COUNTRY  = "95% CI (Country Bootstrap)"
LEGEND_CI_BOOT_WILD     = "95% CI (Wild Bootstrap)"
LEGEND_CI_AR_BOOT       = "95% CI (AR Bootstrap)"
LEGEND_BASELINE         = "Aggregate Effect (Baseline)"


# ── Reference line styling ─────────────────────────────────────────────────
HLINE_KW = dict(ls="--", color="black", lw=0.8, alpha=0.5)   # zero line
VLINE_KW = dict(ls=":",  color="gray",  lw=1.0, alpha=0.7)   # shock at h=0


# ── Confidence band ────────────────────────────────────────────────────────
CI_ALPHA       = 0.15   # fill_between alpha
CI_BOUND_LW    = 1.5    # dashed CI bound line width
CI_BOUND_ALPHA = 0.7    # dashed CI bound line alpha


# ── Figure sizes ───────────────────────────────────────────────────────────
FIGSIZE_PAIR   = (7.5, 4.5)    # paired IRF subfigures (0.48\textwidth each)
FIGSIZE_SINGLE = (9.0, 5.0)    # single full-width IRF panel
FIGSIZE_WIDE   = (14.0, 5.0)   # two side-by-side panels in one figure object
FIGSIZE_GRID22 = (12.0, 9.0)   # 2×2 subplot grid (lag robustness)


# ── Helper: apply standard styling to an IRF axes object ──────────────────
def style_irf_ax(ax, xlabel=None, ylabel=None, legend=True, fontsize=12):
    """
    Apply publication-standard styling to an IRF matplotlib Axes.

    Parameters
    ----------
    ax      : matplotlib.axes.Axes
    xlabel  : str, optional  (default: IRF_XLABEL)
    ylabel  : str, optional
    legend  : bool           (default: True — calls ax.legend())
    fontsize: int            (default: 12)
    """
    ax.axhline(0, **HLINE_KW)
    ax.axvline(0, **VLINE_KW)
    ax.set_xlabel(xlabel or IRF_XLABEL, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if legend:
        ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
