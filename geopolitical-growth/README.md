# Short Replication Package

**Paper:** "Measuring Geopolitical Alignment and Economic Growth"
**Author:** Tianyu Fan (Yale University)
**Date:** April 2026

---

## Overview

This package reproduces the main empirical results from the paper using a single analysis-ready panel dataset. It provides both Python and Stata code so that core results can be verified in either language.

The package covers 19 main-text regression figures (Figures 6--14, 17--18) spanning baseline local projections, transitory/permanent shock decomposition, component analysis, robustness checks, placebo tests, instrumental variables, channel analysis, and growth accounting. Descriptive and measurement figures (Figures 1--5), democracy analysis (Figures 15--16), dynamic panel estimates (Figure 19), and event-based specifications (Figure 20) are excluded from this submission-stage package. A full archival package including the LLM event-compilation pipeline, raw data construction, and all appendix results will be prepared upon publication.

**Python** reproduces all 19 figures. **Stata** independently verifies 10 figures covering the core specifications (baseline, decomposition, symmetry, robustness, and both IV strategies).

---

## Directory Structure

```
replication/
  README.md                          This file
  PLAN.md                            Detailed construction plan
  data/
    panel.csv                        Analysis-ready panel (35 variables, 12,545 obs)
  python/
    code/
      config.py                      Paths, LP defaults, sample restrictions
      lp_utils.py                    Core LP estimation engine
      plot_style.py                  Publication figure styling
      01_baseline.py                 Fig 6a-b: Self-IRF + GDP IRF
      02_transitory.py               Fig 7a-b: Transitory/permanent decomposition
      03_decomposition.py            Fig 8a-b: Component horse-race + residualized
      04_robustness.py               Fig 11a-b: FE robustness + progressive controls
      05_symmetry.py                 Fig 10a-b: Partner + temporal symmetry
      06_placebo.py                  Fig 12a-b: Placebo randomization tests
      07_iv_verbal.py                Fig 13a: IV with verbal conflicts
      08_iv_leadership.py            Fig 13b: IV with leadership changes
      09_channels.py                 Fig 14a-b: Channel variables (2 panels)
      10_growth_accounting.py        Fig 17a-b, 18: Growth accounting + scatter
      run_all.py                     Master runner (executes 01-10 sequentially)
    output/
      figures/                       19 PDF figures (FigXY_name.pdf)
      cache/                         Bootstrap results + intermediate CSV
  stata/
    code/
      master.do                      Load data, construct lags/FE, run all scripts
      01_baseline.do                 Fig 6a-b
      02_decomposition.do            Fig 8a-b
      03_symmetry.do                 Fig 10a-b
      04_robustness.do               Fig 11a-b
      05_iv.do                       Fig 13a: IV verbal conflicts
      06_iv_leadership.do            Fig 13b: IV leadership changes
    output/
      figures/                       10 PDF figures
      estimates/                     CSV coefficient estimates
```

---

## Data

### Panel Description

The file `data/panel.csv` is a country-year panel containing 35 pre-computed variables for 193 countries over 1960--2024 (12,545 observations). All variables needed for every specification in the package are included in this single file. No additional data files are required.

After applying the balanced-sample restriction (countries with at least 40 years of non-missing GDP data), the estimation sample contains 148 countries and 9,620 observations. This is the sample used for all baseline specifications. The progressive-controls specifications use a common subsample of approximately 109 countries with complete data on all control variables.

### Variable Dictionary

**Identifiers**

| Variable | Description |
|----------|-------------|
| `country_code` | ISO 3166-1 alpha-3 country code |
| `country_name` | Country name |
| `year` | Year (1960--2024) |
| `region` | World Bank region classification |

**Outcome Variables**

| Variable | Description | Source |
|----------|-------------|--------|
| `y_ext` | Log GDP per capita x 100 (constant USD, Acemoglu et al. 2019, chained with WDI) | Acemoglu et al. (2019) + WDI |
| `yPWT` | Log real GDP per capita x 100 | Penn World Table 10.01 |

**Geopolitical Alignment Index**

| Variable | Description |
|----------|-------------|
| `geo_relation_dyn` | GDP-weighted dynamic geopolitical alignment index (main shock variable) |
| `geo_relation_dyn_us` | Alignment with the United States |
| `geo_relation_dyn_exclus` | Alignment excluding the United States |
| `geo_relation_dyn_western` | Alignment with Western democracies |
| `geo_relation_dyn_nonwestern` | Alignment with non-Western powers |

**Component Decomposition (pre-computed)**

| Variable | Description |
|----------|-------------|
| `econ_relation_dyn` | GDP-weighted economic relations component (CAMEO category A) |
| `diplo_relation_dyn` | GDP-weighted diplomatic and political component (category B) |
| `security_relation_dyn` | GDP-weighted security and territorial component (categories C+D) |

**Instrumental Variables**

| Variable | Description |
|----------|-------------|
| `geo_noecon_conflict_relation_dyn` | GDP-weighted non-economic verbal conflict index |
| `z_leader` | Leadership-change instrument: GDP-weighted bilateral shifts around 25 deaths in office and 42 close election turnovers (S_{t+1} - S_{t-1} window) |

**Control Variables**

| Variable | Description | Source |
|----------|-------------|--------|
| `trade` | Trade as share of GDP | WDI |
| `unrest_new` | Domestic political unrest index | CNTS 2024 |
| `v2x_polyarchy` | Electoral democracy index | V-Dem v14 |
| `v2x_liberal` | Liberal democracy index | V-Dem v14 |
| `v2x_partip` | Participatory democracy index | V-Dem v14 |
| `v2xdl_delib` | Deliberative democracy index | V-Dem v14 |
| `v2x_egal` | Egalitarian democracy index | V-Dem v14 |
| `war_site_onset_all` | Domestic war-site onset indicator | Federle et al. (2026) |
| `war_site_caspop_all` | Domestic casualty-to-population ratio | Federle et al. (2026) |
| `war_trade_caspop_weighted_all` | Trade-weighted foreign war casualty exposure | Federle et al. (2026) |
| `war_prox_caspop_weighted_all` | Proximity-weighted foreign war casualty exposure | Federle et al. (2026) |
| `war_exposure_site_count_all` | Count of active foreign war sites | Federle et al. (2026) |

**Channel Variables**

| Variable | Description | Source |
|----------|-------------|--------|
| `csh_i` | Investment share of GDP | PWT 11.0 |
| `rnna` | Real capital stock (for log transformation) | PWT 11.0 |
| `rtfpna` | Total factor productivity (for log transformation) | PWT 11.0 |
| `hc` | Human capital index (for log transformation) | PWT 11.0 |
| `irr` | Internal rate of return | PWT 11.0 |

**Fixed Effects Helpers**

| Variable | Description |
|----------|-------------|
| `gdp_quintile` | Initial GDP quintile (Maddison 1960 estimates) |
| `InitReg` | Initial political regime classification |

### Notes on Variable Construction

- War control variables are filled with 0 for country-years with no recorded war activity (missing = no exposure, not missing data).
- The leadership instrument `z_leader` is filled with 0 for country-years without leadership-change events (zero = no instrument shock).
- Component scores are GDP-weighted from bilateral dyad-level CAMEO category scores, aggregated to the country-year level using the same 24-nation GDP-weighting scheme as the main index.
- Log transformations (`log_k`, `log_tfp`, `log_hc`) are computed at runtime by the channel scripts from the raw PWT variables (`rnna`, `rtfpna`, `hc`), multiplied by 100 for log-point interpretation.

---

## Python Package

### Requirements

Python 3.10 or later with the following packages:

```
numpy
pandas
matplotlib
seaborn
scipy
statsmodels
linearmodels        # PanelOLS with Driscoll-Kraay SEs
tqdm                # Bootstrap progress bars
adjustText          # Scatter plot label placement (growth accounting only)
```

Install all dependencies:
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels linearmodels tqdm adjustText
```

### Quick Start

```bash
cd replication/python/code
python run_all.py
```

This executes all 10 scripts sequentially. Scripts 02 (transitory/permanent bootstrap) and 06 (placebo randomization) each run 500 iterations. Total runtime is approximately 8--10 minutes on Apple M5 Pro. All other scripts complete in under 1 minute each.

Output figures are saved to `replication/python/output/figures/` as PDF files. Bootstrap results and intermediate estimates are cached in `replication/python/output/cache/`.

Each script can also be run independently:
```bash
python 01_baseline.py       # Fast (~5 seconds)
python 02_transitory.py     # Slow (~3 minutes, bootstrap)
```

### Econometric Specification

All local projection estimates follow the baseline specification described in the paper:

- **Dependent variable:** `y_ext` (log GDP per capita x 100)
- **Shock variable:** `geo_relation_dyn` (GDP-weighted dynamic geopolitical alignment index)
- **Lags:** 4 lags of the dependent variable and 4 lags of the shock variable
- **Fixed effects:** Country fixed effects and region-year fixed effects
- **Standard errors:** Driscoll-Kraay (HAC, robust to cross-sectional dependence and serial correlation)
- **Horizons:** h = -10 to +25 years (baseline); varies by specification
- **Sample:** Balanced panel of 148 countries with at least 40 years of GDP data

### Figure-to-Script Mapping

| Figure | Content | Script | Output |
|--------|---------|--------|--------|
| 6a | Self-IRF of geopolitical alignment (h=0..30) | `01_baseline.py` | `Fig6A_self_irf.pdf` |
| 6b | GDP IRF: balanced vs unbalanced (h=-10..25) | `01_baseline.py` | `Fig6B_gdp_irf.pdf` |
| 7a | Transitory shock response (500 bootstrap) | `02_transitory.py` | `Fig7A_transitory_irf.pdf` |
| 7b | Permanent shock cumulative response | `02_transitory.py` | `Fig7B_permanent_irf.pdf` |
| 8a | Component horse-race (joint specification) | `03_decomposition.py` | `Fig8A_component_horserace.pdf` |
| 8b | Orthogonal component variation | `03_decomposition.py` | `Fig8B_component_residualized.pdf` |
| 10a | Partner symmetry: US vs non-US | `05_symmetry.py` | `Fig10A_symmetry_partner.pdf` |
| 10b | Temporal stability: 1960--1989 vs 1990--2019 | `05_symmetry.py` | `Fig10B_symmetry_temporal.pdf` |
| 11a | Alternative fixed effects (5 specifications) | `04_robustness.py` | `Fig11A_fe_robustness.pdf` |
| 11b | Progressive controls (109 common-sample countries) | `04_robustness.py` | `Fig11B_control_robustness.pdf` |
| 12a | Placebo: within-region-year reassignment (500 draws) | `06_placebo.py` | `Fig12A_placebo_region_year.pdf` |
| 12b | Placebo: future-year timing (500 draws, 8--15yr ahead) | `06_placebo.py` | `Fig12B_placebo_future_timing.pdf` |
| 13a | IV: non-economic verbal conflicts | `07_iv_verbal.py` | `Fig13A_iv_verbal_conflicts.pdf` |
| 13b | IV: leadership changes (25 deaths + 42 elections) | `08_iv_leadership.py` | `Fig13B_iv_leadership_changes.pdf` |
| 14a | Channels: output and stability (4 variables) | `09_channels.py` | `Fig14A_channels_output_stability.pdf` |
| 14b | Channels: growth fundamentals (4 variables) | `09_channels.py` | `Fig14B_channels_fundamentals.pdf` |
| 17a | Decade contemporaneous growth effects | `10_growth_accounting.py` | `Fig17A_decade_contemporaneous.pdf` |
| 17b | Decade long-run growth effects | `10_growth_accounting.py` | `Fig17B_decade_longrun.pdf` |
| 18 | Cross-country growth scatter (1960--90 vs 1991--2024) | `10_growth_accounting.py` | `Fig18_growth_scatter.pdf` |

---

## Stata Package

### Requirements

Stata 17 or later with the following packages:

```
reghdfe         # High-dimensional fixed effects absorption
ftools          # Required by reghdfe
```

Install from SSC:
```stata
ssc install reghdfe
ssc install ftools
```

### Quick Start

```stata
cd replication/stata/code
do master.do
```

This loads the panel, constructs all lag and fixed-effect variables, then executes scripts 01--06 sequentially. Total runtime is approximately 40 seconds.

Output figures are saved to `replication/stata/output/figures/` as PDF files. Coefficient estimates are exported as CSV to `replication/stata/output/estimates/` for numerical cross-verification with the Python output.

### Implementation Notes

The Stata code independently verifies the Python results using `reghdfe` for fixed-effect absorption. Two methodological differences from the Python implementation are documented:

1. **Standard errors.** Python uses Driscoll-Kraay standard errors (`cov_type='kernel'` in `linearmodels.PanelOLS`), which are robust to both cross-sectional dependence and serial correlation. Stata uses country-clustered standard errors (`cluster(country_code)` in `reghdfe`), which allow for arbitrary within-country correlation but assume independence across countries. Point estimates are identical; confidence intervals will differ slightly.

2. **IV estimation.** Both implementations use manual two-stage least squares (first-stage `reghdfe` to predict, second-stage `reghdfe` on fitted values). This matches the Python approach of manual 2SLS via `PanelOLS`. Standard errors in the second stage are clustered (Stata) or Driscoll-Kraay (Python).

War control variable names are shortened in Stata to comply with the 32-character variable name limit (`war_trade_caspop_weighted_all` becomes `war_trade_wt`, etc.). The underlying data is identical.

### Figure-to-Script Mapping

| Figure | Content | Script | Output |
|--------|---------|--------|--------|
| 6a | Self-IRF of geopolitical alignment | `01_baseline.do` | `Fig6A_self_irf.pdf` |
| 6b | GDP IRF: balanced vs unbalanced | `01_baseline.do` | `Fig6B_gdp_irf.pdf` |
| 8a | Component horse-race | `02_decomposition.do` | `Fig8A_component_horserace.pdf` |
| 8b | Orthogonal component variation | `02_decomposition.do` | `Fig8B_component_residualized.pdf` |
| 10a | Partner symmetry: US vs non-US | `03_symmetry.do` | `Fig10A_symmetry_partner.pdf` |
| 10b | Temporal stability | `03_symmetry.do` | `Fig10B_symmetry_temporal.pdf` |
| 11a | Alternative fixed effects | `04_robustness.do` | `Fig11A_fe_robustness.pdf` |
| 11b | Progressive controls | `04_robustness.do` | `Fig11B_control_robustness.pdf` |
| 13a | IV: verbal conflicts | `05_iv.do` | `Fig13A_iv_verbal_conflicts.pdf` |
| 13b | IV: leadership changes | `06_iv_leadership.do` | `Fig13B_iv_leadership_changes.pdf` |

---

## Scope Note

This submission-stage package covers the main regression results (Figures 6--14, 17--18). The following are excluded and will be provided in a full archival package upon publication:

- **LLM event-compilation pipeline** (Gemini API prompts and raw event processing)
- **Raw data construction** (WDI, PWT, ATOP, GSDB, V-Dem, Archigos, war controls)
- **Descriptive and measurement figures** (Figures 1--5: case studies, topology, distribution, model robustness, pipeline validation)
- **Democracy analysis** (Figures 15--16: bilateral responses and horse-race)
- **Dynamic panel estimates** (Figure 19: AR(4) specification)
- **Event-based specifications** (Figure 20: unsmoothed geopolitical scores)
- **All appendix figures and tables**

---

## Contact

Tianyu Fan
Department of Economics, Yale University
tianyu.fan@yale.edu
