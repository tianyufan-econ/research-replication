/*
master.do
=========
Master script for the Geopolitical Growth Stata replication package.
Loads panel.csv, constructs lags and FE variables, then runs all scripts.

Usage:
  cd replication/stata/code
  do master.do
*/

clear all
set more off
set matsize 5000

* ── Required packages (install if missing) ───────────────────────────
* Run these once before first use:
*   ssc install reghdfe
*   ssc install ftools
*   ssc install grstyle    (optional, for figure styling)
foreach pkg in reghdfe ftools {
    cap which `pkg'
    if _rc {
        di as err "ERROR: `pkg' not installed. Run: ssc install `pkg'"
        error 198
    }
}

* ── Paths ────────────────────────────────────────────────────────────
global root   ".."
global data   "../../data"
global output "${root}/output"
global figures "${output}/figures"
global estimates "${output}/estimates"

cap mkdir "${output}"
cap mkdir "${figures}"
cap mkdir "${estimates}"

* ── Graph scheme (mimic Python style) ────────────────────────────────
set scheme s2color
cap which grstyle
if _rc == 0 {
    grstyle init
    grstyle set plain, horizontal grid
    grstyle set color Set1
    grstyle set legend 4, nobox
}
else {
    di as txt "  Note: grstyle not installed. Using default scheme."
}

* ── Load panel ───────────────────────────────────────────────────────
di as txt "Loading panel..."
import delimited "${data}/panel.csv", clear varnames(1) encoding(utf-8)
di as txt "  Rows: `=_N'  Variables: `=c(k)'"

* ── Encode country for xtset ─────────────────────────────────────────
encode country_code, gen(cid)
xtset cid year
di as txt "  Countries: `=r(imax)'"

* ── Balanced sample: keep countries with >= 40 years of y_ext ────────
bysort cid: egen n_yext = count(y_ext)
qui keep if n_yext >= 40
drop n_yext
qui levelsof country_code, local(cc)
local n_countries : word count `cc'
di as txt "  Balanced sample: `=_N' obs, `n_countries' countries"

* ── Construct lags ───────────────────────────────────────────────────
di as txt "Constructing lags..."

* Core: y_ext and geo_relation_dyn
forvalues l = 1/4 {
    qui by cid: gen y_ext_lag`l' = L`l'.y_ext
    qui by cid: gen geo_lag`l' = L`l'.geo_relation_dyn
}

* Components
foreach v in econ_relation_dyn diplo_relation_dyn security_relation_dyn {
    forvalues l = 1/4 {
        qui by cid: gen `v'_lag`l' = L`l'.`v'
    }
}

* Partner decomposition
foreach v in geo_relation_dyn_us geo_relation_dyn_exclus {
    forvalues l = 1/4 {
        qui by cid: gen `v'_lag`l' = L`l'.`v'
    }
}

* IV instruments
forvalues l = 1/4 {
    qui by cid: gen iv_lag`l' = L`l'.geo_noecon_conflict_relation_dyn
    qui by cid: gen z_leader_lag`l' = L`l'.z_leader
}

* Controls: trade and unrest
forvalues l = 1/4 {
    qui by cid: gen trade_lag`l' = L`l'.trade
    qui by cid: gen unrest_new_lag`l' = L`l'.unrest_new
}

* War controls (fill NaN with 0, rename long names for 32-char limit)
foreach v of varlist war_* {
    qui replace `v' = 0 if mi(`v')
}
rename war_trade_caspop_weighted_all war_trade_wt
rename war_prox_caspop_weighted_all  war_prox_wt
rename war_exposure_site_count_all   war_expo_ct

foreach v in war_site_onset_all war_site_caspop_all ///
             war_trade_wt war_prox_wt war_expo_ct {
    forvalues l = 1/4 {
        qui by cid: gen `v'_lag`l' = L`l'.`v'
    }
}

* V-Dem institutions
foreach v in v2x_polyarchy v2x_liberal v2x_partip v2xdl_delib v2x_egal {
    forvalues l = 1/4 {
        qui by cid: gen `v'_lag`l' = L`l'.`v'
    }
}

* ── Construct FE group variables ─────────────────────────────────────
di as txt "Constructing FE variables..."

* region-year FE
egen region_year_id = group(region year)

* initial GDP quintile x year
egen iniGDP_yr_id = group(gdp_quintile year)

* region x current regime x year
gen current_regime = cond(mi(v2x_polyarchy), 0, v2x_polyarchy >= 0.5)
egen regRegYr_id = group(region current_regime year)

* region x initial regime x year
egen regIniregYr_id = group(region initreg year)

di as txt "  FE variables created: region_year_id iniGDP_yr_id regRegYr_id regIniregYr_id"

* ── Define base control locals ───────────────────────────────────────
global base_controls "y_ext_lag1 y_ext_lag2 y_ext_lag3 y_ext_lag4 geo_lag1 geo_lag2 geo_lag3 geo_lag4"

* ── Save prepared panel ──────────────────────────────────────────────
tempfile panel
save `panel'
di as txt "Panel prepared and saved to tempfile."
di as txt "  Obs: `=_N'  Vars: `=c(k)'"

* ── Run analysis scripts ─────────────────────────────────────────────
di ""
di as txt "============================================================"
di as txt "  RUNNING REPLICATION SCRIPTS"
di as txt "============================================================"

foreach script in 01_baseline 02_decomposition 03_symmetry 04_robustness 05_iv 06_iv_leadership {
    di ""
di as txt "--- `script'.do ---"
    timer clear 1
    timer on 1
    cap noi do `script'.do
    timer off 1
    qui timer list 1
    di as txt "  [`script'] finished in `=round(r(t1), 0.1)'s"
    use `panel', clear
}

di ""
di as txt "============================================================"
di as txt "  ALL SCRIPTS COMPLETE"
di as txt "============================================================"
