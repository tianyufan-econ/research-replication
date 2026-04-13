/*
06_iv_leadership.do
===================
LP-IV estimation using leadership changes (z_leader) as instrument.

Produces:
  Fig 13b  Fig13B_iv_leadership_changes.pdf  — baseline IV + full controls (h=0..20)

Instrument: z_leader (pre-computed GDP-weighted bilateral shifts around
            25 deaths in office + 42 close election turnovers)

Spec 1 (Baseline): 148 balanced countries, iv_controls only
Spec 2 (Full Controls): ~109 common-sample countries, + progressive controls

Uses manual 2SLS via reghdfe (first stage predict, second stage on fitted).
*/

set more off

local H_START = 0
local H_END   = 20
local Z       = 1.96

* IV-augmented base controls (add 4 lags of z_leader instrument)
local iv_controls $base_controls z_leader_lag1 z_leader_lag2 z_leader_lag3 z_leader_lag4

* Colours
local c_base   "navy"
local c_orange "orange"

di ""
di as txt "============================================================"
di as txt "  Fig 13b: LP-IV Leadership Changes"
di as txt "============================================================"

* --- Define progressive control blocks ---
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

* --- Determine common sample for full-controls spec ---
* Match Python: require >= 10 valid obs at both h=-10 and h=25
* with all variables (dep, shock, instrument, all controls) non-missing

cap drop _dep_m10
cap drop _dep_p25
qui by cid: gen _dep_m10 = L10.y_ext
qui by cid: gen _dep_p25 = F25.y_ext

gen _ldr_valid = !mi(_dep_m10) & !mi(_dep_p25) & !mi(geo_relation_dyn) ///
    & !mi(z_leader) & !mi(y_ext)

foreach v of local iv_controls {
    qui replace _ldr_valid = 0 if mi(`v')
}
foreach v of local full_extra {
    qui replace _ldr_valid = 0 if mi(`v')
}

bysort cid: egen _nvalid = total(_ldr_valid)
gen ldr_common_sample = (_nvalid >= 10)
drop _dep_m10 _dep_p25 _ldr_valid _nvalid

qui count if ldr_common_sample == 1
local n_common = r(N)
qui tab cid if ldr_common_sample == 1
local n_countries = r(r)
di as txt "  Common sample: `n_common' obs, `n_countries' countries"


* ══════════════════════════════════════════════════════════════════════
* Estimate both specs via manual 2SLS
* Spec 1: baseline (full balanced, 148 countries)
* Spec 2: full controls (common sample, ~109 countries)
* ══════════════════════════════════════════════════════════════════════

tempfile ldr_results
postfile _ldr_res h coef se spec using `ldr_results'

forvalues s = 1/2 {

    local extra ""
    local sample_cond ""
    local sample_and ""
    if `s' == 2 {
        local extra `full_extra'
        local sample_cond "if ldr_common_sample == 1"
        local sample_and "& ldr_common_sample == 1"
    }

    forvalues h = `H_START'/`H_END' {

        cap drop _dep
        qui by cid: gen _dep = F`h'.y_ext `sample_cond'

        local success = 0

        * Manual 2SLS
        cap drop _shock_hat
        * First stage: restricted to non-missing _dep (matching Python)
        cap qui reghdfe geo_relation_dyn z_leader ///
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
                post _ldr_res (`h') (_b[_shock_hat]) (_se[_shock_hat]) (`s')
                local success = 1
            }
        }
        if `success' == 0 {
            post _ldr_res (`h') (.) (.) (`s')
        }
        cap drop _shock_hat
    }

    if `s' == 1  di as txt "  Baseline IV complete"
    if `s' == 2  di as txt "  Full controls IV complete"
}

postclose _ldr_res

* --- Save estimates CSV ---
preserve
use `ldr_results', clear
gen ci_lo = coef - `Z' * se
gen ci_hi = coef + `Z' * se
gen spec_label = "Baseline" if spec == 1
replace spec_label = "Full Controls" if spec == 2
export delimited using "${estimates}/fig13b_iv_leadership.csv", replace
di as txt "  Estimates saved -> ${estimates}/fig13b_iv_leadership.csv"
restore


* ══════════════════════════════════════════════════════════════════════
* Plot Fig 13b
* ══════════════════════════════════════════════════════════════════════

preserve
use `ldr_results', clear

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
      xtitle("Years Relative to Shock", size(medium)) ///
      legend(order(2 "Baseline (IV)" 1 "95% CI (Clustered)" 3 "+ Full Controls") ///
        rows(1) pos(6) ring(1) size(small) region(lcolor(none))) ///
      xlabel(`H_START'(5)`H_END') ///
      graphregion(color(white)) plotregion(color(white)) ///
      name(fig13b, replace)

graph export "${figures}/Fig13B_iv_leadership_changes.pdf", replace as(pdf)
di as txt "  Figure saved -> ${figures}/Fig13B_iv_leadership_changes.pdf"
restore

* Cleanup
cap drop ldr_common_sample

di ""
di as txt "  06_iv_leadership.do complete."
