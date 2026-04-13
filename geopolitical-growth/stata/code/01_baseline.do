/*
01_baseline.do
==============
Baseline local projection IRFs (Figures 6a and 6b).

  Fig 6a — Self-IRF: geo_relation_dyn → geo_relation_dyn (h = 0..25)
  Fig 6b — GDP IRF:  geo_relation_dyn → y_ext (h = -10..25), balanced vs unbalanced

Assumes master.do has already:
  - loaded panel, created lags, FE variables
  - set xtset cid year
  - defined $base_controls, $figures, $estimates
*/

set more off

di ""
di as txt "============================================================"
di as txt "  01_baseline.do — Baseline LP IRFs (Figs 6a, 6b)"
di as txt "============================================================"

* ═════════════════════════════════════════════════════════════════════════
* Figure 6a — Self-IRF: geo_relation_dyn → geo_relation_dyn (h = 0..25)
* ═════════════════════════════════════════════════════════════════════════

di ""
di as txt "--- Fig 6a: Self-IRF (h = 0..25) ---"

local h_min = 0
local h_max = 25

tempfile res6a
postfile _res6a horizon coef se ci_lo ci_hi using `res6a'

forvalues h = `h_min'/`h_max' {
    cap drop _dep
    qui by cid: gen _dep = F`h'.geo_relation_dyn

    cap qui reghdfe _dep geo_relation_dyn $base_controls, ///
        absorb(cid region_year_id) cluster(country_code) keepsingletons
    if _rc == 0 {
        local b  = _b[geo_relation_dyn]
        local s  = _se[geo_relation_dyn]
        local lo = `b' - 1.96 * `s'
        local hi = `b' + 1.96 * `s'
        post _res6a (`h') (`b') (`s') (`lo') (`hi')
    }
    cap drop _dep
}

postclose _res6a

* ── Export to CSV and plot ──
preserve
use `res6a', clear
export delimited using "${estimates}/fig6a_self_irf.csv", replace
di as txt "  Saved: ${estimates}/fig6a_self_irf.csv"

* ── Plot ──
twoway (rarea ci_lo ci_hi horizon, color("31 119 180%30") lwidth(none)) ///
       (connected coef horizon, lcolor("31 119 180") lwidth(medthick) ///
            mcolor("31 119 180") msymbol(O) msize(small) lpattern(solid)), ///
    yline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    xline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    ytitle("Geopolitical Alignment Score", size(medium)) ///
    xtitle("Years Relative to Shock", size(medium)) ///
    legend(order(2 "Point Estimate" 1 "95% CI") ///
           rows(1) pos(6) ring(1) size(small) region(lcolor(none))) ///
    xlabel(0(5)25) ///
    graphregion(color(white)) plotregion(color(white)) ///
    name(fig6a, replace)
graph export "${figures}/Fig6A_self_irf.pdf", as(pdf) replace
di as txt "  Saved: ${figures}/Fig6A_self_irf.pdf"
restore


* ═════════════════════════════════════════════════════════════════════════
* Figure 6b — GDP IRF: geo_relation_dyn → y_ext (h = -10..25)
*             Balanced sample (with CI) + Unbalanced overlay
* ═════════════════════════════════════════════════════════════════════════

di ""
di as txt "--- Fig 6b: GDP IRF (h = -10..25) ---"

local h_min = -10
local h_max = 25

* ── Balanced panel (current sample from master.do) ──
di as txt "  Running balanced panel..."

tempfile res6b_bal
postfile _res6b_bal horizon coef se ci_lo ci_hi using `res6b_bal'

forvalues h = `h_min'/`h_max' {
    cap drop _dep

    * Negative horizons use L operator, positive use F
    if `h' < 0 {
        local absh = abs(`h')
        qui by cid: gen _dep = L`absh'.y_ext
    }
    else {
        qui by cid: gen _dep = F`h'.y_ext
    }

    cap qui reghdfe _dep geo_relation_dyn $base_controls, ///
        absorb(cid region_year_id) cluster(country_code) keepsingletons
    if _rc == 0 {
        local b  = _b[geo_relation_dyn]
        local s  = _se[geo_relation_dyn]
        local lo = `b' - 1.96 * `s'
        local hi = `b' + 1.96 * `s'
        post _res6b_bal (`h') (`b') (`s') (`lo') (`hi')
    }
    cap drop _dep
}

postclose _res6b_bal

* ── Unbalanced panel ──
* Reload full panel without balanced restriction and re-create lags/FE
di as txt "  Running unbalanced panel..."

tempfile res6b_unbal
postfile _res6b_unb horizon coef se using `res6b_unbal'

preserve
    * Load full panel (without balanced restriction)
    qui import delimited "${data}/panel.csv", clear varnames(1) encoding(utf-8)
    qui encode country_code, gen(cid_ub)
    qui xtset cid_ub year

    * Construct lags for unbalanced sample
    forvalues l = 1/4 {
        qui by cid_ub: gen y_ext_lag`l' = L`l'.y_ext
        qui by cid_ub: gen geo_lag`l'   = L`l'.geo_relation_dyn
    }
    qui egen region_year_id_ub = group(region year)

    local ub_controls "y_ext_lag1 y_ext_lag2 y_ext_lag3 y_ext_lag4 geo_lag1 geo_lag2 geo_lag3 geo_lag4"

    forvalues h = `h_min'/`h_max' {
        cap drop _dep

        if `h' < 0 {
            local absh = abs(`h')
            qui by cid_ub: gen _dep = L`absh'.y_ext
        }
        else {
            qui by cid_ub: gen _dep = F`h'.y_ext
        }

        cap qui reghdfe _dep geo_relation_dyn `ub_controls', ///
            absorb(cid_ub region_year_id_ub) cluster(country_code) keepsingletons
        if _rc == 0 {
            local b  = _b[geo_relation_dyn]
            local s  = _se[geo_relation_dyn]
            post _res6b_unb (`h') (`b') (`s')
        }
        cap drop _dep
    }
restore

postclose _res6b_unb

* ── Merge balanced and unbalanced results, export CSV and plot ──
preserve
use `res6b_bal', clear
rename coef  bal_coef
rename se    bal_se
rename ci_lo bal_ci_lo
rename ci_hi bal_ci_hi

tempfile bal_tmp
save `bal_tmp'

use `res6b_unbal', clear
rename coef unbal_coef
rename se   unbal_se

merge 1:1 horizon using `bal_tmp', nogenerate

export delimited using "${estimates}/fig6b_gdp_irf.csv", replace
di as txt "  Saved: ${estimates}/fig6b_gdp_irf.csv"

* ── Plot ──
twoway (rarea bal_ci_lo bal_ci_hi horizon, color("31 119 180%30") lwidth(none)) ///
       (connected bal_coef horizon, lcolor("31 119 180") lwidth(medthick) ///
            mcolor("31 119 180") msymbol(O) msize(small) lpattern(solid)) ///
       (connected unbal_coef horizon, lcolor("255 127 14") lwidth(medthick) ///
            mcolor("255 127 14") msymbol(S) msize(small) lpattern(dash)), ///
    yline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    xline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    ytitle("Log GDP per Capita (x100)", size(medium)) ///
    xtitle("Years Relative to Shock", size(medium)) ///
    legend(order(2 "Balanced" 3 "Unbalanced" 1 "95% CI") ///
           rows(1) pos(6) ring(1) size(small) region(lcolor(none))) ///
    xlabel(-10(5)25) ///
    graphregion(color(white)) plotregion(color(white)) ///
    name(fig6b, replace)
graph export "${figures}/Fig6B_gdp_irf.pdf", as(pdf) replace
di as txt "  Saved: ${figures}/Fig6B_gdp_irf.pdf"
restore

di ""
di as txt "  01_baseline.do complete."
