/*
04_robustness.do
================
Robustness checks for baseline LP: alternative FE specs and progressive controls.

Produces:
  Fig 11a  Fig11A_fe_robustness.pdf        — 5 FE specifications (h=-10..25)
  Fig 11b  Fig11B_control_robustness.pdf   — progressive controls, common sample

Assumes master.do has loaded panel, created lags/FE variables, set globals.
Requires: reghdfe (ssc install reghdfe, replace)
*/

timer clear 1
timer on 1

* ══════════════════════════════════════════════════════════════════════
* Setup
* ══════════════════════════════════════════════════════════════════════

local H_START = -10
local H_END   = 25
local Z       = 1.96

* Colours (approximate seaborn deep palette)
local c_base    "navy"
local c_orange  "orange"
local c_green   "forest_green"
local c_red     "cranberry"
local c_purple  "purple"

* ══════════════════════════════════════════════════════════════════════
* Fig 11a — Alternative Fixed Effects Specifications
* ══════════════════════════════════════════════════════════════════════

di ""
di as txt "============================================================"
di as txt "  Fig 11a: Alternative FE specifications"
di as txt "============================================================"

* --- storage dataset ---
tempfile fe_results
postfile _fe_res h coef se spec using `fe_results'

forvalues s = 1/5 {
    forvalues h = `H_START'/`H_END' {

        * Generate dependent variable: F(h).y_ext for h>=0, L(|h|).y_ext for h<0
        cap drop _dep
        if `h' >= 0 {
            qui by cid: gen _dep = F`h'.y_ext
        }
        else {
            local absh = abs(`h')
            qui by cid: gen _dep = L`absh'.y_ext
        }

        * Run reghdfe with the appropriate absorb() spec
        cap {
            if `s' == 1 {
                qui reghdfe _dep geo_relation_dyn $base_controls, ///
                    absorb(cid region_year_id) ///
                    cluster(country_code) keepsingletons
            }
            else if `s' == 2 {
                qui reghdfe _dep geo_relation_dyn $base_controls, ///
                    absorb(cid year) ///
                    cluster(country_code) keepsingletons
            }
            else if `s' == 3 {
                qui reghdfe _dep geo_relation_dyn $base_controls, ///
                    absorb(cid iniGDP_yr_id) ///
                    cluster(country_code) keepsingletons
            }
            else if `s' == 4 {
                qui reghdfe _dep geo_relation_dyn $base_controls, ///
                    absorb(cid regRegYr_id) ///
                    cluster(country_code) keepsingletons
            }
            else if `s' == 5 {
                qui reghdfe _dep geo_relation_dyn $base_controls, ///
                    absorb(cid regIniregYr_id) ///
                    cluster(country_code) keepsingletons
            }
        }

        if _rc == 0 {
            post _fe_res (`h') (_b[geo_relation_dyn]) (_se[geo_relation_dyn]) (`s')
        }
        else {
            post _fe_res (`h') (.) (.) (`s')
        }
    }
    local spec_names `" "Region-Year FE" "Year FE" "Initial GDP-Year FE" "Region-Regime-Year FE" "Region-Initial Regime-Year FE" "'
    local sname : word `s' of `spec_names'
    di as txt "  Spec `s' (`sname') complete"
}

postclose _fe_res

* --- Save estimates CSV ---
preserve
use `fe_results', clear
gen ci_lo = coef - `Z' * se
gen ci_hi = coef + `Z' * se
export delimited using "${estimates}/fig11a_fe_specs.csv", replace
di as txt "  Estimates saved -> ${estimates}/fig11a_fe_specs.csv"
restore

* --- Plot Fig 11a ---
preserve
use `fe_results', clear

gen ci_lo = coef - `Z' * se
gen ci_hi = coef + `Z' * se

* Reshape to wide: one row per horizon, separate vars per spec
drop ci_lo ci_hi
reshape wide coef se, i(h) j(spec)

* Reconstruct CIs for baseline (spec 1)
gen ci_lo_1 = coef1 - `Z' * se1
gen ci_hi_1 = coef1 + `Z' * se1

twoway ///
    (rarea ci_lo_1 ci_hi_1 h, color(`c_base'%15) lwidth(none)) ///
    (connected coef1 h, lcolor(`c_base') mcolor(`c_base') ///
        lwidth(medthick) msymbol(O) msize(small) ///
        lpattern(solid)) ///
    (connected coef2 h, lcolor(`c_orange') mcolor(`c_orange') ///
        lwidth(medium) msymbol(S) msize(vsmall) ///
        lpattern(dash)) ///
    (connected coef3 h, lcolor(`c_green') mcolor(`c_green') ///
        lwidth(medium) msymbol(T) msize(vsmall) ///
        lpattern(dash)) ///
    (connected coef4 h, lcolor(`c_red') mcolor(`c_red') ///
        lwidth(medium) msymbol(D) msize(vsmall) ///
        lpattern(dash)) ///
    (connected coef5 h, lcolor(`c_purple') mcolor(`c_purple') ///
        lwidth(medium) msymbol(+) msize(small) ///
        lpattern(dash)) ///
    , yline(0, lcolor(gs10) lpattern(dash)) ///
      xline(0, lcolor(gs10) lpattern(dot)) ///
      ytitle("Log GDP per Capita (x100)", size(medium)) ///
      xtitle("Horizon (years)", size(medium)) ///
      legend(order( ///
        2 "Region-Year FE (baseline)" ///
        1 "95% CI" ///
        3 "Year FE" ///
        4 "Initial GDP-Year FE" ///
        5 "Region-Regime-Year FE" ///
        6 "Region-Initial Regime-Year FE") ///
        cols(2) size(small) region(lstyle(none))) ///
      xlabel(`H_START'(5)`H_END') ///
      graphregion(color(white)) plotregion(color(white)) ///
      title("") ///
      scheme(s2color) ///
      name(fig11a, replace)

graph export "${figures}/Fig11A_fe_robustness.pdf", replace as(pdf)
di as txt "  Figure saved -> ${figures}/Fig11A_fe_robustness.pdf"
restore


* ══════════════════════════════════════════════════════════════════════
* Fig 11b — Progressive Controls (common sample)
* ══════════════════════════════════════════════════════════════════════

di ""
di as txt "============================================================"
di as txt "  Fig 11b: Progressive controls (common sample)"
di as txt "============================================================"

* --- Define control blocks ---
local trade_lags    trade_lag1 trade_lag2 trade_lag3 trade_lag4
local unrest_lags   unrest_new_lag1 unrest_new_lag2 unrest_new_lag3 unrest_new_lag4

local war_lags ///
    war_site_onset_all_lag1 war_site_onset_all_lag2 ///
    war_site_onset_all_lag3 war_site_onset_all_lag4 ///
    war_site_caspop_all_lag1 war_site_caspop_all_lag2 ///
    war_site_caspop_all_lag3 war_site_caspop_all_lag4 ///
    war_trade_wt_lag1 war_trade_wt_lag2 ///
    war_trade_wt_lag3 war_trade_wt_lag4 ///
    war_prox_wt_lag1 war_prox_wt_lag2 ///
    war_prox_wt_lag3 war_prox_wt_lag4 ///
    war_expo_ct_lag1 war_expo_ct_lag2 ///
    war_expo_ct_lag3 war_expo_ct_lag4

local inst_lags ///
    v2x_polyarchy_lag1 v2x_polyarchy_lag2 v2x_polyarchy_lag3 v2x_polyarchy_lag4 ///
    v2x_liberal_lag1 v2x_liberal_lag2 v2x_liberal_lag3 v2x_liberal_lag4 ///
    v2x_partip_lag1 v2x_partip_lag2 v2x_partip_lag3 v2x_partip_lag4 ///
    v2xdl_delib_lag1 v2xdl_delib_lag2 v2xdl_delib_lag3 v2xdl_delib_lag4 ///
    v2x_egal_lag1 v2x_egal_lag2 v2x_egal_lag3 v2x_egal_lag4

local all_controls `trade_lags' `unrest_lags' `war_lags' `inst_lags'

* --- Determine common sample ---
* Countries with non-missing data for ALL controls at both extreme horizons

cap drop _dep_m10
cap drop _dep_p25
qui by cid: gen _dep_m10 = L10.y_ext
qui by cid: gen _dep_p25 = F25.y_ext

gen _common = !mi(_dep_m10) & !mi(_dep_p25) & !mi(geo_relation_dyn) & !mi(y_ext)
foreach v of local all_controls {
    qui replace _common = 0 if mi(`v')
}
foreach v in $base_controls {
    qui replace _common = 0 if mi(`v')
}

* Count valid observations per country (match Python: MIN_OBS_PER_COUNTRY >= 10)
bysort cid: egen _nvalid = total(_common)
gen common_sample = (_nvalid >= 10)
drop _dep_m10 _dep_p25 _common _nvalid

qui count if common_sample == 1
local n_common = r(N)
qui tab cid if common_sample == 1
local n_countries_common = r(r)
di as txt "  Common sample: `n_common' obs, `n_countries_common' countries"

* --- Progressive specs ---
* Spec 1: Baseline (no extra controls)
* Spec 2: + trade lags
* Spec 3: + unrest lags
* Spec 4: + war lags (20 terms)
* Spec 5: + institution lags (20 terms)

tempfile ctrl_results
postfile _ctrl_res h coef se spec using `ctrl_results'

forvalues s = 1/5 {
    * Build the control list for this spec
    local extra ""
    if `s' >= 2  local extra `trade_lags'
    if `s' >= 3  local extra `extra' `unrest_lags'
    if `s' >= 4  local extra `extra' `war_lags'
    if `s' >= 5  local extra `extra' `inst_lags'

    forvalues h = `H_START'/`H_END' {
        cap drop _dep
        if `h' >= 0 {
            qui by cid: gen _dep = F`h'.y_ext if common_sample == 1
        }
        else {
            local absh = abs(`h')
            qui by cid: gen _dep = L`absh'.y_ext if common_sample == 1
        }

        cap {
            qui reghdfe _dep geo_relation_dyn $base_controls `extra' ///
                if common_sample == 1, ///
                absorb(cid region_year_id) ///
                cluster(country_code) keepsingletons
        }

        if _rc == 0 {
            post _ctrl_res (`h') (_b[geo_relation_dyn]) (_se[geo_relation_dyn]) (`s')
        }
        else {
            post _ctrl_res (`h') (.) (.) (`s')
        }
    }

    local spec_labels `" "Baseline" "+ Trade Lags" "+ Unrest Lags" "+ War Exposure Lags" "+ Institution Lags" "'
    local slabel : word `s' of `spec_labels'
    di as txt "  Spec `s' (`slabel') complete"
}

postclose _ctrl_res

* --- Save estimates CSV ---
preserve
use `ctrl_results', clear
gen ci_lo = coef - `Z' * se
gen ci_hi = coef + `Z' * se
export delimited using "${estimates}/fig11b_progressive_controls.csv", replace
di as txt "  Estimates saved -> ${estimates}/fig11b_progressive_controls.csv"
restore

* --- Plot Fig 11b ---
preserve
use `ctrl_results', clear

reshape wide coef se, i(h) j(spec)

gen ci_lo_1 = coef1 - `Z' * se1
gen ci_hi_1 = coef1 + `Z' * se1

twoway ///
    (rarea ci_lo_1 ci_hi_1 h, color(`c_base'%15) lwidth(none)) ///
    (connected coef1 h, lcolor(`c_base') mcolor(`c_base') ///
        lwidth(medthick) msymbol(O) msize(small) ///
        lpattern(solid)) ///
    (connected coef2 h, lcolor(`c_orange') mcolor(`c_orange') ///
        lwidth(medium) msymbol(S) msize(vsmall) ///
        lpattern(dash)) ///
    (connected coef3 h, lcolor(`c_green') mcolor(`c_green') ///
        lwidth(medium) msymbol(T) msize(vsmall) ///
        lpattern(dash)) ///
    (connected coef4 h, lcolor(`c_red') mcolor(`c_red') ///
        lwidth(medium) msymbol(D) msize(vsmall) ///
        lpattern(dash)) ///
    (connected coef5 h, lcolor(`c_purple') mcolor(`c_purple') ///
        lwidth(medium) msymbol(+) msize(small) ///
        lpattern(dash)) ///
    , yline(0, lcolor(gs10) lpattern(dash)) ///
      xline(0, lcolor(gs10) lpattern(dot)) ///
      ytitle("Log GDP per Capita (x100)", size(medium)) ///
      xtitle("Horizon (years)", size(medium)) ///
      legend(order( ///
        2 "Baseline" ///
        1 "95% CI" ///
        3 "+ Trade Lags" ///
        4 "+ Unrest Lags" ///
        5 "+ War Exposure Lags" ///
        6 "+ Institution Lags") ///
        cols(2) size(small) region(lstyle(none))) ///
      xlabel(`H_START'(5)`H_END') ///
      graphregion(color(white)) plotregion(color(white)) ///
      title("") ///
      scheme(s2color) ///
      name(fig11b, replace)

graph export "${figures}/Fig11B_control_robustness.pdf", replace as(pdf)
di as txt "  Figure saved -> ${figures}/Fig11B_control_robustness.pdf"
restore

* Cleanup
cap drop common_sample

timer off 1
qui timer list 1
di as txt _n "04_robustness.do complete in `=round(r(t1), 0.1)'s"
