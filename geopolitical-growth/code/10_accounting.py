"""
10_accounting.py
================
Figures 16 and 17 and Table 4: geopolitical growth accounting and the scale of the index.

The persistence and GDP coefficients (h = 0..25; country and region-year fixed effects, four
lags of GDP and of the index, long-coverage sample) give the transitory path, the GDP response
to a one-period unit change in the index, and its cumulative sum, the sustained path. Applied
to each country's deviation from the annual median index, with deviations before 1960 set to
their 1960 value, the transitory path gives the implied GDP contribution of alignment (all
countries; log GDP per capita x 100).
  Figure 16  within-period and 25-year contributions of each period's realignment
  Figure 17  (a) change in the implied contribution over 1960-1990 and 1991-2024,
             (b) implied contribution in 2024 against log GDP per capita
  Table 4    historical endpoint changes of the index, in index units and SDs
The strip-plot jitter of Figure 16 is the jitter recorded from the paper's files (panel
columns fig16_jitter_*); the labels of Figure 17 are placed by accounting_labels.py.

Full-package version: code/03_extensions/growth_accounting_reg.py (paths, contributions,
Figure 16), code/03_extensions_plot/plot_growth_accounting.py with label_layout_big.py
(Figure 17) and code/03_extensions/episode_table.py (Table 4). episode_table.py reads the
sustained path of 02_baseline/lp_transitory_reg.py, the same estimate on the same sample
(the two caches are bit-identical), so the paths are estimated once here.
Outputs: output/figures/Fig16{a,b}_*.pdf, Fig17{a,b}_*.pdf, output/tables/Table4_episodes.tex,
         output/cache/{irf_permanent_transitory,counterfactual_gdp_results,
         decade_contemporaneous_effects,decade_longrun_effects}_2024.csv,
         period_changes_by_country.csv, geo_contribution_2024_by_country.csv,
         episode_percentiles.csv, episode_named.csv, output/numbers/10_accounting.csv
"""

import warnings
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

import plot_style as PS
from figure_textsize import savefig_aspect
from accounting_labels import place_labels
from config import (CACHE, MAJOR_NATIONS, PANEL, TABLES, ensure_dirs, read_csv_fallback,
                    restrict_balanced_sample, save_numbers, savefig)

ensure_dirs()
PS.apply_theme()
warnings.filterwarnings("ignore")

H = 25
LAGS = 4
FULL_GA = "03_extensions/growth_accounting_reg.py"
FULL_PLOT = "03_extensions_plot/plot_growth_accounting.py"
FULL_EP = "03_extensions/episode_table.py"


def load():
    df = read_csv_fallback(PANEL, low_memory=False)
    for col in ["country", "country_code"]:
        df[col] = df[col].astype(str)
    df["year"] = df["year"].astype(int)
    for col in ["y_ext", "geo_relation_dyn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════
# 1. Transitory and sustained paths (growth_accounting_reg.py, step 1)
# ══════════════════════════════════════════════════════════════════════
def prepare_panel(df, y_ext_lag_order, geo_lag_order):
    """Add country_idx, region_year, and lags for y_ext and geo_relation_dyn."""
    df = df.copy()
    df['country_idx'] = pd.factorize(df['country_code'])[0]
    df['region_year'] = df['region'].astype(str) + '_' + df['year'].astype(str)
    df = df.sort_values(['country_idx', 'year'])
    for l in range(1, y_ext_lag_order + 1):
        df[f'y_ext_lag{l}'] = df.groupby('country_idx')['y_ext'].shift(l)
    for l in range(1, geo_lag_order + 1):
        df[f'geo_relation_dyn_lag{l}'] = df.groupby('country_idx')['geo_relation_dyn'].shift(l)
    return df


def lp_irf(df, shock_var, dependent_var, horizon_range, y_ext_lag_order, geo_lag_order):
    """Coefficient on shock_var at each horizon, OLS with country and region-year dummies."""
    y_lags = [f'y_ext_lag{i}' for i in range(1, y_ext_lag_order + 1)]
    geo_lags = [f'geo_relation_dyn_lag{i}' for i in range(1, geo_lag_order + 1)]
    h_vals = list(horizon_range)
    irf = np.full(len(h_vals), np.nan)
    for idx, h in enumerate(h_vals):
        tmp = df.copy()
        tmp['dep_var'] = tmp.groupby('country_idx')[dependent_var].shift(-h)
        cols = ['dep_var', shock_var] + y_lags + geo_lags + ['country_idx', 'region_year']
        sub = tmp.dropna(subset=cols)
        min_obs = len([shock_var]) + len(y_lags) + len(geo_lags) + sub['country_idx'].nunique() + sub['region_year'].nunique() + 10
        if sub.shape[0] < min_obs:
            continue
        fmla = f'dep_var ~ {shock_var}'
        if y_lags or geo_lags:
            fmla += ' + ' + ' + '.join(y_lags + geo_lags)
        fmla += ' + C(country_idx) + C(region_year)'
        try:
            res = smf.ols(fmla, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['country_idx']})
            irf[idx] = res.params.get(shock_var, np.nan)
        except Exception as e:
            print(f"LP error h={h}, dependent={dependent_var}: {e}")
    return np.array(h_vals), irf


def construct_phi_matrix(phi_p, H):
    """Phi^p: lower triangular with the persistence coefficients."""
    phi_matrix = np.eye(H + 1)
    for i in range(1, H + 1):
        for j in range(i):
            phi_matrix[i, j] = phi_p[i - j] if i - j < len(phi_p) else 0
    return phi_matrix


def compute_auxiliary_shocks(phi_p, H):
    """Auxiliary shocks that hold the index at (1, 0, ..., 0)."""
    phi_matrix = construct_phi_matrix(phi_p, H)
    transitory_response = np.zeros(H + 1)
    transitory_response[0] = 1
    try:
        return np.linalg.inv(phi_matrix) @ transitory_response
    except np.linalg.LinAlgError:
        print("Error: Phi matrix is singular, cannot compute inverse.")
        return np.zeros(H + 1)


def construct_p_shock_matrix(p_shock, H):
    """P^shock: lower triangular with the auxiliary shocks."""
    P_shock = np.zeros((H + 1, H + 1))
    for i in range(H + 1):
        for j in range(i + 1):
            P_shock[i, j] = p_shock[i - j] if i - j < len(p_shock) else 0
    return P_shock


def response_paths(df):
    """Transitory and sustained (cumulative) paths per unit of the index, long-coverage sample."""
    df_balanced = restrict_balanced_sample(df)
    df_balanced = df_balanced[(df_balanced['year'] >= 1960) & (df_balanced['year'] <= 2024)]
    df_balanced = prepare_panel(df_balanced, LAGS, LAGS)
    h_vals, phi_p = lp_irf(df_balanced, 'geo_relation_dyn', 'geo_relation_dyn', range(0, H + 1), LAGS, LAGS)
    _, alpha = lp_irf(df_balanced, 'geo_relation_dyn', 'y_ext', range(0, H + 1), LAGS, LAGS)
    if np.isnan(phi_p).all() or np.isnan(alpha).all():
        raise RuntimeError("IRF estimation failed")
    P_shock = construct_p_shock_matrix(compute_auxiliary_shocks(phi_p, H), H)
    alpha_transitory = P_shock @ alpha
    alpha_permanent = np.cumsum(alpha_transitory)
    longrun_effect = alpha_permanent[H] if len(alpha_permanent) > H else alpha_permanent[-1]
    print(f"  Long-run effect (sustained path at h={H}): {longrun_effect:.4f}")
    pd.DataFrame({'horizon': h_vals, 'permanent_irf': alpha_permanent, 'transitory_irf': alpha_transitory,
                  'longrun_effect': longrun_effect}).to_csv(CACHE / 'irf_permanent_transitory_2024.csv', index=False)
    return alpha_transitory, longrun_effect


# ══════════════════════════════════════════════════════════════════════
# 2. Counterfactual GDP and period contributions (growth_accounting_reg.py, steps 2-3)
# ══════════════════════════════════════════════════════════════════════
DECADES = [(1960, 1969), (1970, 1979), (1980, 1989), (1990, 1999), (2000, 2009), (2010, 2019),
           (2020, 2024)]  # partial last period
DECADE_ORDER = ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020-2024']


def decade_label(start_year, end_year):
    return f"{start_year}s" if end_year - start_year == 9 else f"{start_year}-{end_year}"


def calculate_decade_changes(df):
    """Change in the index over each period, by country."""
    decade_changes = []
    for country_idx, country_df in df.groupby('country_idx'):
        country_code = country_df['country_code'].iloc[0]
        for start_year, end_year in DECADES:
            start_data = country_df[country_df['year'] == start_year]
            end_data = country_df[country_df['year'] == end_year]
            if not start_data.empty and not end_data.empty:
                start_geo = start_data['geo_relation_dyn'].iloc[0]
                end_geo = end_data['geo_relation_dyn'].iloc[0]
                if not np.isnan(start_geo) and not np.isnan(end_geo):
                    decade_changes.append({
                        'country_code': country_code, 'country_idx': country_idx,
                        'decade': decade_label(start_year, end_year), 'start_year': start_year,
                        'end_year': end_year, 'geo_change': end_geo - start_geo,
                        'start_geo': start_geo, 'end_geo': end_geo})
    return pd.DataFrame(decade_changes)


def calculate_contemporaneous_effects(df, alpha_transitory, decades):
    """Within-period contribution: the transitory path applied to the changes realized in the period."""
    contemp_effects = []
    for country_idx, country_df in df.groupby('country_idx'):
        country_code = country_df['country_code'].iloc[0]
        country_df = country_df.sort_values('year')
        for start_year, end_year in decades:
            decade_df = country_df[(country_df['year'] >= start_year) & (country_df['year'] <= end_year)]
            if len(decade_df) == 0:
                continue
            start_data = decade_df[decade_df['year'] == start_year]
            if start_data.empty:
                continue
            initial_geo = start_data['geo_relation_dyn'].iloc[0]
            if np.isnan(initial_geo):
                continue
            decade_df = decade_df.copy()
            decade_df['geo_diff_from_initial'] = decade_df['geo_relation_dyn'] - initial_geo
            contemp_effect = 0
            for year in decade_df['year'].values:
                geo_diff = decade_df[decade_df['year'] == year]['geo_diff_from_initial'].iloc[0]
                if not np.isnan(geo_diff):
                    horizon = end_year - year
                    if horizon < len(alpha_transitory):
                        contemp_effect += alpha_transitory[horizon] * geo_diff
            contemp_effects.append({
                'country_code': country_code, 'country_idx': country_idx,
                'decade': decade_label(start_year, end_year), 'start_year': start_year,
                'end_year': end_year, 'contemporaneous_effect': contemp_effect})
    return pd.DataFrame(contemp_effects)


def accounting(df, alpha_transitory, longrun_effect):
    """Implied contributions for all countries, 1960-2024."""
    df = df[(df['year'] >= 1960) & (df['year'] <= 2024)]
    df = prepare_panel(df, LAGS, LAGS)
    print(f"  Full sample: {df['country_code'].nunique()} countries")

    df['delta_y'] = np.nan
    for country_idx, group in df.groupby('country_idx'):
        group_valid = group.dropna(subset=['y_ext']).copy()
        if not group_valid.empty:
            df.loc[group_valid.index, 'delta_y'] = group_valid['y_ext'] - group_valid['y_ext'].iloc[0]

    median_geo = df.groupby('year')['geo_relation_dyn'].median().reset_index()
    median_geo.columns = ['year', 'median_geo']
    df = df.merge(median_geo, on='year', how='left')
    df['gross_change'] = df['median_geo'] - df['geo_relation_dyn']

    df['delta_y_cf'] = np.nan
    for country_idx, group in df.groupby('country_idx'):
        gross_changes = group['gross_change'].copy()
        valid_mask = gross_changes.notna()
        if gross_changes.empty or not valid_mask.any():
            continue
        delta_y_cf = np.full(len(gross_changes), np.nan)
        gross_changes_array = gross_changes.fillna(0).values
        # Years before the first observation (1960) carry the 1960 deviation, so every year has a
        # full 26-term kernel and a constant deviation implies a constant contribution
        first = int(np.argmax(valid_mask.values))
        initial = gross_changes_array[first]
        for t in range(len(gross_changes)):
            if not valid_mask.iloc[t]:
                continue
            effect = 0
            for h in range(H + 1):
                if not np.isnan(alpha_transitory[h]):
                    effect += alpha_transitory[h] * (gross_changes_array[t - h] if t - h >= first else initial)
            delta_y_cf[t] = effect
        df.loc[group.index, 'delta_y_cf'] = delta_y_cf
    df['delta_y_geo'] = -df['delta_y_cf']
    df['y_cf'] = df['y_ext'] + df['delta_y_cf']

    decade_changes = calculate_decade_changes(df)
    decade_changes['longrun_effect'] = decade_changes['geo_change'] * longrun_effect
    contemp_effects = calculate_contemporaneous_effects(df, alpha_transitory, DECADES)

    result = df[['country_code', 'year', 'y_ext', 'delta_y', 'delta_y_geo', 'y_cf', 'region']].copy()
    result.dropna(subset=['delta_y_geo'], inplace=True)
    result.to_csv(CACHE / 'counterfactual_gdp_results_2024.csv', index=False)
    decade_changes.to_csv(CACHE / 'decade_longrun_effects_2024.csv', index=False)
    contemp_effects.to_csv(CACHE / 'decade_contemporaneous_effects_2024.csv', index=False)


# ══════════════════════════════════════════════════════════════════════
# 3. Figure 16: period boxplots (growth_accounting_reg.py, plot_decade_boxplots)
# ══════════════════════════════════════════════════════════════════════
SEABORN_PALETTE = sns.color_palette("deep")
ASPECT_16 = 855.42788 / 509.72599   # page aspect of the paper's files


@contextmanager
def _recorded_jitter(groups):
    """Feed the recorded jitter through seaborn's (unseeded) np.random.uniform, one period at a time."""
    queue = [np.asarray(g, dtype=float) for g in groups]
    orig = np.random.uniform
    rng = np.random.default_rng(0)

    def uniform(low=0.0, high=1.0, size=None):
        a = queue.pop(0) if queue else None
        if a is None or size != len(a):
            print(f"  WARNING: recorded jitter does not match the data ({size} dots); using a seeded jitter")
            return rng.uniform(low, high, size)
        return a.copy()

    np.random.uniform = uniform
    try:
        yield
    finally:
        np.random.uniform = orig


def recorded_jitter(panel, data, col):
    """The recorded jitter of each period's dots, in plotting order (panel column col)."""
    j = panel.set_index(['country_code', 'year'])[col]
    return [j.loc[pd.MultiIndex.from_arrays([sub['country_code'], sub['start_year']])].to_numpy()
            for sub in (data[data['decade'] == dec] for dec in DECADE_ORDER)]


def plot_boxplot(data, ycol, ylabel, jitter, filename):
    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    sns.boxplot(data=data, x='decade', y=ycol, palette=SEABORN_PALETTE, ax=ax, order=DECADE_ORDER)
    with _recorded_jitter(jitter):
        sns.stripplot(data=data, x='decade', y=ycol, color='black', alpha=0.3, size=4, ax=ax, order=DECADE_ORDER)
    ax.axhline(0, **PS.HLINE_KW)
    ax.set_xlabel("Period", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    sns.despine(ax=ax)
    ax.tick_params(axis='both', labelsize=10.5)
    fig.tight_layout()
    savefig_aspect(fig, filename, ASPECT_16)
    plt.close(fig)


def figure16(panel, numbers):
    longrun_df = pd.read_csv(CACHE / 'decade_longrun_effects_2024.csv')
    contemp_df = pd.read_csv(CACHE / 'decade_contemporaneous_effects_2024.csv')
    plot_boxplot(contemp_df, 'contemporaneous_effect',
                 "Implied Within-Period Contribution\nto Log GDP per Capita (×100)",
                 recorded_jitter(panel, contemp_df, 'fig16_jitter_within'), "Fig16a_within_period.pdf")
    plot_boxplot(longrun_df, 'longrun_effect',
                 "Implied 25-Year Contribution\nto Log GDP per Capita (×100)",
                 recorded_jitter(panel, longrun_df, 'fig16_jitter_25yr'), "Fig16b_25_year.pdf")

    for name, data, col, letter in [("within", contemp_df, 'contemporaneous_effect', "a"),
                                    ("25yr", longrun_df, 'longrun_effect', "b")]:
        med = data.groupby('decade')[col].median()
        for dec in DECADE_ORDER:
            intro = name == "25yr" and dec in ("1980s", "2010s")
            numbers.append({"key": f"median_{name}_{dec}", "value": med[dec], "full_ref": f"{FULL_GA}:median_{name}_{dec}",
                            "paper": ("Introduction; " if intro else "") + f"Section 4.3, Figure 16({letter})"})


# ══════════════════════════════════════════════════════════════════════
# 4. Figure 17: cross-country accounting (plot_growth_accounting.py)
# ══════════════════════════════════════════════════════════════════════
ASPECT_17 = 856.1479 / 509.84455   # page aspect of the paper's files
REGION_NAMES = {
    'LAC': 'Latin America & Caribbean', 'AFR': 'Africa',
    'ECA': 'Europe & Central Asia',     'INL': 'Developed',
    'SAS': 'South Asia',                'MNA': 'Middle East & North Africa',
    'EAP': 'East Asia & Pacific',
}
REGION_ORDER = [
    'Middle East & North Africa', 'Africa', 'Europe & Central Asia',
    'Developed', 'Latin America & Caribbean', 'South Asia', 'East Asia & Pacific',
]
_DEEP = sns.color_palette("deep")
REGION_COLORS = {region: _DEEP[k] for k, region in enumerate(REGION_ORDER)}

LABEL_COUNTRIES_17A = {
    'ZAF', 'EST', 'LVA', 'LTU', 'CHL', 'GEO',            # discussed in text (reversals)
    'VEN', 'NIC', 'CHN', 'HUN',                          # discussed in text (deterioration)
    'RUS', 'BLR', 'USA', 'SGP',                          # discussed in text (other)
    'DEU', 'GBR', 'FRA', 'JPN', 'IND', 'BRA', 'KOR',     # major economies
    'IRN', 'PRK', 'UKR', 'SAU', 'TUR',                   # geopolitically important
    'LBY', 'AFG', 'SYR', 'ERI', 'CUB', 'MOZ', 'DOM',     # outliers
    'QAT', 'MMR',
}
LABEL_COUNTRIES_17B = {
    'RUS', 'BLR', 'CHN',                                 # discussed in text (largest negative)
    'USA', 'ISR', 'CAN',                                 # developed-economy negative
    'SGP', 'LAO', 'MNG', 'MOZ',                          # beneficiaries
    'DEU', 'GBR', 'FRA', 'JPN', 'IND', 'BRA', 'KOR',     # major economies
    'IRN', 'UKR', 'SAU', 'TUR',                          # geopolitically important
    'NIC', 'VEN', 'HUN', 'NOR', 'QAT', 'ZAF',            # outliers
    'EST', 'SYR', 'PRK', 'CUB',
}


def _plot_scatter_layered(ax, data, xcol, ycol, label_set):
    """Faded region-coloured dots, bold labelled dots and their labels (placed later)."""
    data = data.copy()
    data['labeled'] = data['country_code'].isin(label_set)
    unlabeled = data[~data['labeled']]
    labeled = data[data['labeled']]
    for region in REGION_ORDER:
        subset = unlabeled[unlabeled['region_full'] == region]
        if len(subset) > 0:
            ax.scatter(subset[xcol], subset[ycol], color=REGION_COLORS[region], s=45, alpha=0.25,
                       edgecolor='none', zorder=2)
    for region in REGION_ORDER:
        subset = labeled[labeled['region_full'] == region]
        if len(subset) > 0:
            ax.scatter(subset[xcol], subset[ycol], color=REGION_COLORS[region], s=130, alpha=0.9,
                       edgecolor='white', linewidth=1.3, zorder=3, label=region)
    texts = []
    for _, row in labeled.iterrows():
        cc = str(row.get('country_code', ''))[:3]
        texts.append(ax.text(row[xcol], row[ycol], cc, fontsize=10.5, ha='center', va='center',
                             fontweight='medium', alpha=0.9, zorder=4))
    return texts, labeled, unlabeled


def figure17a():
    """Change in the implied contribution, 1960-1990 against 1991-2024."""
    df = pd.read_csv(CACHE / 'counterfactual_gdp_results_2024.csv')
    d1 = df[df['year'].isin([1960, 1990])].pivot_table(
        index='country_code', columns='year', values='delta_y_geo').reset_index()
    d1.columns = ['country_code', 'dy_1960', 'dy_1990']
    d1['change_cw'] = d1['dy_1990'] - d1['dy_1960']
    d2 = df[df['year'].isin([1991, 2024])].pivot_table(
        index='country_code', columns='year', values='delta_y_geo').reset_index()
    d2.columns = ['country_code', 'dy_1991', 'dy_2024']
    d2['change_pcw'] = d2['dy_2024'] - d2['dy_1991']
    pc = d1[['country_code', 'change_cw']].merge(d2[['country_code', 'change_pcw']], on='country_code', how='inner')
    pc = pc.merge(df[['country_code', 'region']].drop_duplicates(), on='country_code', how='left')
    pc['region_full'] = pc['region'].map(REGION_NAMES).fillna(pc['region'])

    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    x_pad, y_pad = 3, 3
    x_min, x_max = pc['change_cw'].min() - x_pad, pc['change_cw'].max() + x_pad
    y_min, y_max = pc['change_pcw'].min() - y_pad, pc['change_pcw'].max() + y_pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    diag_min, diag_max = min(x_min, y_min), max(x_max, y_max)       # 45-degree line
    ax.plot([diag_min, diag_max], [diag_min, diag_max], color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    texts, labeled, unlabeled = _plot_scatter_layered(ax, pc, 'change_cw', 'change_pcw', LABEL_COUNTRIES_17A)
    ax.set_xlabel("Change in Implied GDP Contribution (1960–1990)", fontsize=13)
    ax.set_ylabel("Change in Implied GDP Contribution (1991–2024)", fontsize=13)
    ax.tick_params(axis='both', labelsize=12.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    leg = ax.legend(frameon=False, fontsize=12, loc='upper right', markerscale=0.8, handletextpad=0.3)
    place_labels(fig, ax, texts, labeled[['change_cw', 'change_pcw']].to_numpy(), leg, 'geo_growth_period_scatter')
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
    ax.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    savefig_aspect(fig, "Fig17a_period_scatter.pdf", ASPECT_17)
    plt.close(fig)
    pc.drop(columns='region_full').to_csv(CACHE / 'period_changes_by_country.csv', index=False)
    return pc


def figure17b(df_original):
    """Implied contribution in 2024 against log GDP per capita."""
    longrun_effect = pd.read_csv(CACHE / 'irf_permanent_transitory_2024.csv')['longrun_effect'].iloc[0]
    year = 2024
    df_year = df_original[df_original['year'] == year].copy()
    df_year = df_year.dropna(subset=['geo_relation_dyn', 'y_ext'])
    median_geo = df_year['geo_relation_dyn'].median()
    df_year['geo_deviation'] = df_year['geo_relation_dyn'] - median_geo
    df_year['geo_contribution'] = df_year['geo_deviation'] * longrun_effect
    df_year['gdp_per_capita'] = df_year['y_ext'] / 100
    df_year['region_full'] = df_year['region'].map(REGION_NAMES).fillna(df_year['region'])

    fig, ax = plt.subplots(figsize=PS.FIGSIZE_PAIR)
    texts, labeled, unlabeled = _plot_scatter_layered(ax, df_year, 'gdp_per_capita', 'geo_contribution',
                                                      LABEL_COUNTRIES_17B)
    ax.set_xlabel(f'Log GDP per Capita, {year}', fontsize=13)
    ax.set_ylabel('Implied Contribution to\nLog GDP per Capita (×100)', fontsize=13)
    ax.tick_params(axis='both', labelsize=12.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    leg = ax.legend(frameon=False, fontsize=12, loc='lower left', markerscale=0.8, handletextpad=0.3)
    place_labels(fig, ax, texts, labeled[['gdp_per_capita', 'geo_contribution']].to_numpy(), leg,
                 f'geo_contribution_{year}_cross_country')
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig_aspect(fig, "Fig17b_contribution_2024.pdf", ASPECT_17)
    plt.close(fig)
    out = df_year[['country_code', 'region', 'geo_relation_dyn', 'geo_deviation', 'geo_contribution',
                   'gdp_per_capita']].sort_values('geo_contribution')
    out.to_csv(CACHE / f'geo_contribution_{year}_by_country.csv', index=False)
    return out, median_geo, longrun_effect


def figure17_numbers(pc, c24, median_geo, longrun_effect, numbers):
    """Numbers quoted in Section 4.3 (and the Introduction) about Figure 17."""
    x, y = pc.set_index('country_code')['change_cw'], pc.set_index('country_code')['change_pcw']
    a, intro_a = "Section 4.3, Figure 17(a)", "Introduction; Section 4.3, Figure 17(a)"
    rows = [("fig17a_x_min", x.min(), intro_a), ("fig17a_x_max", x.max(), intro_a),
            ("fig17a_y_min", y.min(), a), ("fig17a_y_max", y.max(), intro_a), ("fig17a_n_countries", len(pc), a)]
    for name, m in [("pp", (x > 0) & (y > 0)), ("np", (x < 0) & (y > 0)), ("nn", (x < 0) & (y < 0)), ("pn", (x > 0) & (y < 0))]:
        rows.append((f"fig17a_quadrant_{name}", int(m.sum()), a))
    for cc in ('ZAF', 'EST', 'LVA', 'LTU', 'CHL', 'GEO', 'VEN', 'NIC', 'CHN', 'HUN'):
        rows += [(f"fig17a_{cc}_1960_1990", x[cc], a), (f"fig17a_{cc}_1991_2024", y[cc], a)]
    rev = y - x
    rows += [("fig17a_largest_reversal", rev.max(), a), ("fig17a_largest_negative_reversal", rev.min(), a),
             ("fig17a_corr_xy", np.corrcoef(x, y)[0, 1], a)]

    c = c24.set_index('country_code')['geo_contribution']
    b, intro_b = "Section 4.3, Figure 17(b)", "Introduction; Section 4.3, Figure 17(b)"
    rows += [("fig17b_min", c.min(), intro_b), ("fig17b_max", c.max(), intro_b), ("fig17b_n_countries", len(c), b),
             ("fig17b_median_index_2024", median_geo, b), ("fig17b_longrun_effect", longrun_effect, b)]
    order = c.sort_values()
    rows += [(f"fig17b_bottom{k}_{cc}", order[cc], b) for k, cc in enumerate(order.index[:5], 1)]
    rows += [(f"fig17b_top{k}_{cc}", order[cc], b) for k, cc in enumerate(order.index[::-1][:5], 1)]
    rows += [(f"fig17b_{cc}", c[cc], b) for cc in ('ISR', 'CAN', 'USA')]
    major = c.index.isin(MAJOR_NATIONS)
    rows += [("fig17b_mean_major", c[major].mean(), b), ("fig17b_mean_other", c[~major].mean(), b)]
    g = c24.set_index('country_code')['gdp_per_capita']
    rows += [("fig17b_spearman_gdp", stats.spearmanr(g, c)[0], b), ("fig17b_pearson_gdp", np.corrcoef(g, c)[0, 1], b)]
    numbers += [{"key": k, "value": v, "paper": p, "full_ref": f"{FULL_PLOT}:{k}"} for k, v, p in rows]


# ══════════════════════════════════════════════════════════════════════
# 5. Table 4: historical realignments on the index scale (episode_table.py)
# ══════════════════════════════════════════════════════════════════════
# Named episodes in table order, left column then right column: (iso3, label, year_start, year_end)
NAMED = [
    ("ZAF", "South Africa: apartheid ends", 1986, 1996),
    ("IRN", "Iran: Islamic Revolution", 1977, 1983),
    ("POL", "Poland: post-communist transition", 1988, 1998),
    ("CHN", "China: reform and opening", 1976, 1986),
    ("IRQ", "Iraq: Gulf War and sanctions", 1989, 1994),
    ("VNM", "Vietnam: normalization with the US", 1990, 2000),
    ("RUS", "Russia: deterioration through Crimea", 2005, 2015),
    ("IDN", "Indonesia: post-Suharto reorientation", 1996, 2004),
    ("EGY", "Egypt: Camp David-era realignment", 1973, 1981),
]


def years(y0, y1):
    """1986--96, 1990--2000 (the end year is abbreviated within a century)."""
    return f"{y0}--{y1 % 100:02d}" if y0 // 100 == y1 // 100 else f"{y0}--{y1}"


def write_table(named, sigma, pct, n_countries, y_range):
    iqr = pct[75] - pct[25]
    cells = [("Interquartile shift (25th$\\to$75th)", f"{iqr:.3f} ({iqr / sigma:.2f})")]
    for r in named.itertuples():
        cells.append((f"{r.episode} ({years(r.y0, r.y1)})", f"${r.delta_units:+.2f}$ (${r.delta_SD:+.1f}$)"))
    half = len(cells) // 2
    rows = [f"{a} & {b} & {c} & {d} \\\\" for (a, b), (c, d) in zip(cells[:half], cells[half:])]
    lines = [
        "% Historical endpoint changes used only to illustrate the index scale.",
        f"% Values are computed from the complete {n_countries}-country, {y_range[0]}--{y_range[1]} panel.",
        "\\begin{table}[!htb]",
        "\\centering",
        "\\caption{Historical Realignments on the Index Scale}",
        "\\label{tab:episodes_1sd}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lclc}",
        "\\toprule",
        "Benchmark or episode & $\\Delta$ index (SD) & Benchmark or episode & $\\Delta$ index (SD) \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}}",
        "\\note{$\\Delta$ is the end-year minus start-year value for each selected window. Parentheses report "
        "the change in standard deviations, using the full-sample pooled "
        f"$\\sigma={sigma:.3f}$. The interquartile benchmark is the shift from the 25th percentile "
        f"({pct[25]:.3f}) to the 75th percentile ({pct[75]:.3f}). Episode labels provide historical context, "
        "and examples are chosen for recognizability rather than as a ranking.}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def table4(df, numbers):
    perm_per_unit = pd.read_csv(CACHE / 'irf_permanent_transitory_2024.csv').set_index("horizon").loc[25, "permanent_irf"]
    g = df[["country_code", "year", "geo_relation_dyn"]].dropna(subset=["geo_relation_dyn"])
    s = g.geo_relation_dyn
    sigma = s.std()                       # pooled SD
    print(f"  Pooled sample: {len(s):,} country-years, SD = {sigma:.6f}; 1-SD sustained = {sigma * perm_per_unit:.2f}")

    pct = {p: np.percentile(s, p) for p in [10, 25, 50, 75, 90]}
    iqr = pct[75] - pct[25]
    perc_rows = [{"pctile": p, "index_level": v, "sd_from_median": (v - pct[50]) / sigma,
                  "impl_25yr_logpts": np.nan} for p, v in pct.items()]
    perc_rows.append({"pctile": "IQR shift (25->75)", "index_level": iqr, "sd_from_median": iqr / sigma,
                      "impl_25yr_logpts": iqr * perm_per_unit})
    pd.DataFrame(perc_rows).to_csv(CACHE / "episode_percentiles.csv", index=False)

    rows = []
    for cc, nm, y0, y1 in NAMED:
        ser = g[g.country_code == cc].set_index("year").geo_relation_dyn
        s0, s1 = ser.get(y0, np.nan), ser.get(y1, np.nan)
        d = s1 - s0
        rows.append({"episode": nm, "iso3": cc, "y0": y0, "y1": y1, "S_start": s0, "S_end": s1,
                     "delta_units": d, "delta_SD": d / sigma, "impl_25yr_logpts": d * perm_per_unit})
    named = pd.DataFrame(rows)
    named.to_csv(CACHE / "episode_named.csv", index=False)

    (TABLES / "Table4_episodes.tex").write_text(
        write_table(named, sigma, pct, g.country_code.nunique(), (g.year.min(), g.year.max())))
    print("  Saved: output/tables/Table4_episodes.tex")

    s24, s32, t4 = "Section 2.4", "Section 3.2", "Table 4"
    rows = [("sigma", sigma, f"{s24} and {s32}"), ("p25", pct[25], f"{s32}; {t4} note"),
            ("p75", pct[75], f"{s32}; {t4} note"), ("median", pct[50], s32), ("min", s.min(), s32),
            ("max", s.max(), s32), ("iqr", iqr, f"{s24}, {t4}"), ("iqr_sd", iqr / sigma, f"{s24}, {t4}"),
            ("sd_permanent25_logpts", sigma * perm_per_unit, s32),
            ("iqr_permanent25_logpts", iqr * perm_per_unit, s32)]
    for r in named.itertuples():
        rows.append((f"{r.iso3}_{r.y0}_{r.y1}_delta", r.delta_units, t4))
        rows.append((f"{r.iso3}_{r.y0}_{r.y1}_sd", r.delta_SD,
                     t4 + (f"; {s24}" if r.iso3 in ("RUS", "IDN", "ZAF", "IRN", "IRQ") else "")
                        + (f"; {s32}" if r.iso3 in ("RUS", "VNM") else "")))
    numbers += [{"key": k, "value": v, "paper": p, "full_ref": f"{FULL_EP}:{k}"} for k, v, p in rows]


def main():
    df = load()
    numbers = []
    alpha_transitory, longrun_effect = response_paths(df)
    accounting(df, alpha_transitory, longrun_effect)
    figure16(df, numbers)
    pc = figure17a()
    c24, median_geo, longrun_effect = figure17b(df)
    figure17_numbers(pc, c24, median_geo, longrun_effect, numbers)
    table4(df, numbers)
    save_numbers(__file__, numbers)


if __name__ == "__main__":
    main()
