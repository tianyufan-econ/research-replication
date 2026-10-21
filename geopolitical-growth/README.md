# Short replication package: "Measuring Geopolitical Alignment and Economic Growth"

Tianyu Fan, Princeton University. October 2026.

## Overview

This package reproduces the main-text regression results of the paper from one country-year panel:
- Figures 6–17;
- Table 4;
- the numbers the text quotes from them.

Every estimate equals the one in the full replication package, and every figure and the Table 4 file are byte-identical to the paper's.

The full package also builds the event database with large language models, constructs the alignment index from the bilateral events, and produces the descriptive exhibits (Figures 1–5, Tables 1–3) and every appendix exhibit. It will be deposited with the journal on publication.

A full run takes about 10 minutes on a 2025 Apple laptop, most of it the bootstrap and the placebo draws.

## Contents

```
README.md               this file
requirements.txt        Python packages and the versions used
data/
  panel.csv             country-year panel: 12,545 rows (193 countries, 1960–2024), 61 columns
  panel_columns.csv     every column with its source and description
code/
  run_all.py            runs the ten scripts in order and collects output/numbers.csv
  config.py             paths, long-coverage sample, reporting scale
  lp_utils.py           local-projection helpers
  plot_style.py         figure style
  figure_textsize.py    saving helpers for the multi-panel figures
  accounting_labels.py  label placement for Figure 17
  01_baseline.py … 10_accounting.py
output/
  figures/              FigNNx_*.pdf (NN = paper figure, x = panel)
  tables/               Table4_episodes.tex
  numbers.csv           every reported number (also output/numbers/<script>.csv)
  cache/                the estimates behind each figure
```

## Data

`data/panel.csv` is written by the full package (`code/11_tables_numbers/build_short_panel.py`). Its columns are copied unchanged from the full package's constructed data. `data/panel_columns.csv` lists each column with its source.
- **GDP per capita:** Acemoglu et al. (2019), extended with WDI (2024 release). Also PWT 11.0.
- **Geopolitical alignment:** the paper's index and its components, from the LLM-compiled event database (Gemini 2.5 Pro, 1950–2024).
- **Controls and channels:**
  - PWT 11.0, WDI, V-Dem v16;
  - the democracy indicator of Acemoglu et al. (2019), extended;
  - war exposure from Federle et al. (2026);
  - domestic unrest from CNTS 2024. Only the derived indicator is included, because the CNTS file itself is licensed.
- **Instrument and episodes:**
  - the leadership-change instrument (bilateral shifts around 25 deaths in office and 42 close-election turnovers in the major nations);
  - LLM-coded domestic transformation periods;
  - V-Dem regime transitions.
- **Plotting input, not data:** the two `fig16_jitter_*` columns hold the recorded horizontal jitter of Figure 16's dots. seaborn draws that jitter at random, and the full package replays the recorded values.

The long-coverage estimation sample keeps the 148 countries with at least 40 years of GDP data.

## Requirements and instructions

Python with the packages in `requirements.txt` (`pip install -r requirements.txt`). The outputs were produced with Python 3.14.8 on macOS (arm64). linearmodels must be 7.1 or later, because 7.0 computes the two-way within transform incorrectly when there are many controls.

```bash
cd code
python3 run_all.py            # all ten scripts
python3 run_all.py 02 07      # only the scripts whose names start with 02 or 07
```

`run_all.py` fixes the string-hash seed, runs single-threaded and fixes the file dates, so a rerun writes identical files. A script run on its own (`python3 01_baseline.py`) gives the same numbers.

The bootstrap (Figure 7) and the placebo (Figure 11) redraw 500 samples each, with fixed seeds.

## Specification

The baseline is a local projection of log GDP per capita (×100) at horizon h on the current alignment index, with:
- four lags of GDP and of the index;
- country and region-year fixed effects;
- Driscoll–Kraay standard errors, with the calendar year as the time index.

Coefficients on the alignment index are reported per one standard deviation of the index (0.141).

## Figures, tables and programs

| Paper | Output (`output/`) | Script (`code/`) |
|---|---|---|
| Figure 6 | `figures/Fig06a_persistence.pdf`, `Fig06b_gdp_response.pdf` | `01_baseline.py` |
| Figure 7 | `figures/Fig07a_transitory.pdf`, `Fig07b_sustained.pdf` | `02_sustained.py` |
| Figure 8 | `figures/Fig08a_partners.pdf`, `Fig08b_periods.pdf` | `03_partners.py` |
| Figure 9 | `figures/Fig09a_components.pdf`, `Fig09b_formal_acts.pdf` | `04_components.py` |
| Figure 10 | `figures/Fig10a_fixed_effects.pdf`, `Fig10b_additional_controls.pdf` | `05_robustness.py` |
| Figure 11 | `figures/Fig11a_placebo_region_year_reassignment.pdf`, `Fig11b_placebo_future_year_timing.pdf` | `06_placebo.py` |
| Figure 12 | `figures/Fig12a_transformation_exclusion.pdf`, `Fig12b_leadership_iv.pdf` | `07_transformations_iv.py` |
| Figure 13 | `figures/Fig13a_output_stability.pdf`, `Fig13b_growth_fundamentals.pdf` | `08_channels.py` |
| Figure 14 | `figures/Fig14a_democratic_partners.pdf`, `Fig14b_other_partners.pdf` | `09_democracy.py` |
| Figure 15 | `figures/Fig15a_alignment_and_democracy.pdf`, `Fig15b_democracy_and_alignment.pdf` | `09_democracy.py` |
| Figure 16 | `figures/Fig16a_within_period.pdf`, `Fig16b_25_year.pdf` | `10_accounting.py` |
| Figure 17 | `figures/Fig17a_period_scatter.pdf`, `Fig17b_contribution_2024.pdf` | `10_accounting.py` |
| Table 4 | `tables/Table4_episodes.tex` | `10_accounting.py` |

`04_components.py` also runs the footnote check that adds the multilateral-governance and other-events component (categories E and F) as a fourth regressor.

`02_sustained.py` also computes the two checks in the footnote on the 25-year sustained path:
- the delta-method interval clustered by country and by year;
- the split-panel jackknife.

## Numbers

`output/numbers.csv` has 265 rows, with these columns:
- `script`, `key` and `value`;
- `paper`: where the number appears in the paper;
- `full_ref`: the same number's location in the full package's `output/numbers/` (`<script>:<key>`).

The numbers cover the estimates the text quotes from Figures 6–17 and Table 4. Some text numbers come from other parts of the full package and are not here:
- event and sample counts of the database;
- the transformation-period counts;
- the sustained-path mappings within the exclusion samples;
- the narrow-window IV.

## How the package was checked

The full package's `code/11_tables_numbers/check_short_package.py` compares this package with the full package after both have run:
- **Panel:** every column of `panel.csv` round-trips exactly. The panel is parsed with exact float parsing.
- **Estimates:** every plotted series equals the full package's estimate to within 1e-8. The bootstrap and placebo draws are identical.
- **Numbers:** every row of `numbers.csv` equals its `full_ref` value.
- **Files:** every figure and the Table 4 file are byte-identical to the full package's, which are the files in the paper.

Every check is also matched against an inventory of 733 expected checks, so a series, number or file that goes missing fails. On the final run (21 October 2026), all 443 estimate series, 265 numbers and 25 files (24 figures and Table 4) passed.

## Changes from the April 2026 version

- Figures follow the paper's current numbering.
- **Removed:**
  - the non-economic verbal-conflict IV;
  - the residualized-components panel;
  - the Stata cross-check.
- **Added:**
  - the three-way partner split;
  - the formal-acts check;
  - the transformation-period exclusions;
  - the democracy figures (14–15);
  - the 2024 cross-country contributions;
  - Table 4;
  - the sustained-path interval checks.
- **Data updates:**
  - CAMEO quad-class correction;
  - World Bank 2024 GDP weights;
  - PWT 11.0 TFP;
  - V-Dem v16;
  - coefficients per one SD of the index.
