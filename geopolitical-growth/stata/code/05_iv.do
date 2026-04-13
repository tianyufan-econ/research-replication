/*
05_iv.do
========
LP-IV estimation: instrument geopolitical alignment with non-economic
verbal conflict scores, then overlay with full progressive controls.

Produces:
  Fig 13a  Fig13A_iv_verbal_conflicts.pdf  — baseline IV + full controls (h=0..20)

Assumes master.do has loaded panel, created lags/FE variables, set globals.
Requires: reghdfe (ssc install reghdfe, replace)
          ivreghdfe (ssc install ivreghdfe, replace)  [optional; falls back to manual 2SLS]
*/

timer clear 1
timer on 1

* ══════════════════════════════════════════════════════════════════════
* Setup
* ══════════════════════════════════════════════════════════════════════

local H_START = 0
local H_END   = 20
local Z       = 1.96

* IV-augmented base controls (add instrument lags)
local iv_controls $base_controls iv_lag1 iv_lag2 iv_lag3 iv_lag4

* Colours
local c_base   "navy"
local c_orange "orange"

* Use manual 2SLS (reghdfe first stage + predict + reghdfe second stage)
di as txt "Using manual 2SLS via reghdfe"


* ══════════════════════════════════════════════════════════════════════
* Panel A — Baseline LP-IV (h=0..20)
* ══════════════════════════════════════════════════════════════════════

di ""
di as txt "============================================================"
di as txt "  Fig 13a: LP-IV baseline + full controls"
di as txt "============================================================"

* --- Define progressive control blocks (same as 04_robustness.do) ---
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

local full_extra `trade_lags' `unrest_lags' `war_lags' `inst_lags'

* --- Determine common sample for IV ---
* Match Python: require >= 10 valid obs at both h=-10 and h=25
* with all variables (dep, shock, instrument, all controls) non-missing

cap drop _dep_m10
cap drop _dep_p25
qui by cid: gen _dep_m10 = L10.y_ext
qui by cid: gen _dep_p25 = F25.y_ext

gen _iv_valid = !mi(_dep_m10) & !mi(_dep_p25) & !mi(geo_relation_dyn) ///
    & !mi(geo_noecon_conflict_relation_dyn) & !mi(y_ext)

foreach v of local iv_controls {
    qui replace _iv_valid = 0 if mi(`v')
}
foreach v of local full_extra {
    qui replace _iv_valid = 0 if mi(`v')
}

bysort cid: egen _nvalid = total(_iv_valid)
gen iv_common_sample = (_nvalid >= 10)
drop _dep_m10 _dep_p25 _iv_valid _nvalid

qui count if iv_common_sample == 1
local n_iv_common = r(N)
qui tab cid if iv_common_sample == 1
local n_iv_countries = r(r)
di as txt "  IV common sample: `n_iv_common' obs, `n_iv_countries' countries"


* ══════════════════════════════════════════════════════════════════════
* Estimate both specs: baseline IV and full-controls IV
* ══════════════════════════════════════════════════════════════════════

tempfile iv_results
postfile _iv_res h coef se spec using `iv_results'

forvalues s = 1/2 {

    * Control list: spec 1 = baseline (iv_controls only), spec 2 = + full
    * Spec 1 uses full balanced sample (148 countries)
    * Spec 2 uses common sample with all controls (109 countries)
    local extra ""
    local sample_cond ""
    local sample_and ""
    if `s' == 2 {
        local extra `full_extra'
        local sample_cond "if iv_common_sample == 1"
        local sample_and "& iv_common_sample == 1"
    }

    forvalues h = `H_START'/`H_END' {

        cap drop _dep
        qui by cid: gen _dep = F`h'.y_ext `sample_cond'

        local success = 0

        * --- Manual 2SLS via reghdfe ---
        {
            cap drop _shock_hat
            * First stage (restricted to non-missing _dep, matching Python)
            cap qui reghdfe geo_relation_dyn geo_noecon_conflict_relation_dyn ///
                `iv_controls' `extra' ///
                if !mi(_dep) `sample_and', ///
                absorb(cid region_year_id) ///
                cluster(country_code) keepsingletons
            if _rc == 0 {
                qui predict _shock_hat
                * Second stage
                cap qui reghdfe _dep _shock_hat `iv_controls' `extra' ///
                    `sample_cond', ///
                    absorb(cid region_year_id) ///
                    cluster(country_code) keepsingletons
                if _rc == 0 {
                    post _iv_res (`h') (_b[_shock_hat]) (_se[_shock_hat]) (`s')
                    local success = 1
                }
            }
            if `success' == 0 {
                post _iv_res (`h') (.) (.) (`s')
            }
            cap drop _shock_hat
        }
    }

    if `s' == 1  di as txt "  Baseline IV complete"
    if `s' == 2  di as txt "  Full controls IV complete"
}

postclose _iv_res

* --- Save estimates CSV ---
preserve
use `iv_results', clear
gen ci_lo = coef - `Z' * se
gen ci_hi = coef + `Z' * se
gen spec_label = "Baseline" if spec == 1
replace spec_label = "Full Controls" if spec == 2
export delimited using "${estimates}/fig13a_iv_verbal.csv", replace
di as txt "  Estimates saved -> ${estimates}/fig13a_iv_verbal.csv"
restore


* ══════════════════════════════════════════════════════════════════════
* Plot Fig 13a
* ══════════════════════════════════════════════════════════════════════

preserve
use `iv_results', clear

gen ci_lo = coef - `Z' * se
gen ci_hi = coef + `Z' * se

reshape wide coef se ci_lo ci_hi, i(h) j(spec)

* Baseline CI
rename ci_lo1 ci_lo_base
rename ci_hi1 ci_hi_base

twoway ///
    (rarea ci_lo_base ci_hi_base h, color(`c_base'%15) lwidth(none)) ///
    (connected coef1 h, lcolor(`c_base') mcolor(`c_base') ///
        lwidth(medthick) msymbol(O) msize(small) ///
        lpattern(solid)) ///
    (connected coef2 h, lcolor(`c_orange') mcolor(`c_orange') ///
        lwidth(medium) msymbol(S) msize(vsmall) ///
        lpattern(dash)) ///
    , yline(0, lcolor(gs10) lpattern(dash)) ///
      xline(0, lcolor(gs10) lpattern(dot)) ///
      ytitle("Log GDP per Capita (x100)", size(medium)) ///
      xtitle("Horizon (years)", size(medium)) ///
      legend(order( ///
        2 "Baseline (IV)" ///
        1 "95% CI (Clustered)" ///
        3 "+ Full Controls") ///
        cols(1) size(small) region(lstyle(none))) ///
      xlabel(`H_START'(5)`H_END') ///
      graphregion(color(white)) plotregion(color(white)) ///
      title("") ///
      scheme(s2color) ///
      name(fig13a, replace)

graph export "${figures}/Fig13A_iv_verbal_conflicts.pdf", replace as(pdf)
di as txt "  Figure saved -> ${figures}/Fig13A_iv_verbal_conflicts.pdf"
restore


* ══════════════════════════════════════════════════════════════════════
* First-stage diagnostics (informational)
* ══════════════════════════════════════════════════════════════════════

di ""
di as txt "--- First-stage F-statistics ---"

tempfile fs_results
postfile _fs_res h fstat nobs using `fs_results'

forvalues h = `H_START'/`H_END' {
    cap drop _dep_fs
    qui by cid: gen _dep_fs = F`h'.geo_relation_dyn if iv_common_sample == 1

    cap {
        qui reghdfe _dep_fs geo_noecon_conflict_relation_dyn ///
            `iv_controls' ///
            if iv_common_sample == 1, ///
            absorb(cid region_year_id) ///
            cluster(country_code) keepsingletons
    }

    if _rc == 0 {
        local fval = (_b[geo_noecon_conflict_relation_dyn] / ///
                      _se[geo_noecon_conflict_relation_dyn])^2
        post _fs_res (`h') (`fval') (e(N))
        if `h' == 0 | `h' == 5 | `h' == 10 | `h' == 15 | `h' == 20 {
            di as txt "  h=`h': F = `=round(`fval', 0.1)',  N = `=e(N)'"
        }
    }
    else {
        post _fs_res (`h') (.) (.)
    }
}

postclose _fs_res

* Save first-stage stats
preserve
use `fs_results', clear
export delimited using "${estimates}/fig13a_first_stage_fstats.csv", replace
di as txt "  First-stage F-stats saved -> ${estimates}/fig13a_first_stage_fstats.csv"
restore

* Cleanup
cap drop iv_common_sample

timer off 1
qui timer list 1
di as txt _n "05_iv.do complete in `=round(r(t1), 0.1)'s"
