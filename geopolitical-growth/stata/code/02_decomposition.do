/*
02_decomposition.do
===================
Component decomposition LP IRFs (Figures 8a and 8b).

Three components:
  Economic (econ_relation_dyn)     — category A
  Diplomatic (diplo_relation_dyn)  — category B
  Security (security_relation_dyn) — categories C+D

  Fig 8a — Horse-race: all 3 components jointly in one regression (h = 0..25)
  Fig 8b — Residualized: each component orthogonalized to others (h = 0..25)

Assumes master.do has already:
  - loaded panel, created component lags, FE variables
  - set xtset cid year
  - defined $base_controls, $figures, $estimates
*/

set more off

di ""
di as txt "============================================================"
di as txt "  02_decomposition.do — Component Decomposition (Figs 8a, 8b)"
di as txt "============================================================"

local h_min = 0
local h_max = 25

* Component variable names
local comp1 "econ_relation_dyn"
local comp2 "diplo_relation_dyn"
local comp3 "security_relation_dyn"

* Component lag controls (4 lags of each component)
local comp_lags ///
    econ_relation_dyn_lag1 econ_relation_dyn_lag2 ///
    econ_relation_dyn_lag3 econ_relation_dyn_lag4 ///
    diplo_relation_dyn_lag1 diplo_relation_dyn_lag2 ///
    diplo_relation_dyn_lag3 diplo_relation_dyn_lag4 ///
    security_relation_dyn_lag1 security_relation_dyn_lag2 ///
    security_relation_dyn_lag3 security_relation_dyn_lag4

* y_ext lag controls
local y_lags "y_ext_lag1 y_ext_lag2 y_ext_lag3 y_ext_lag4"


* ═════════════════════════════════════════════════════════════════════════
* Figure 8a — Horse-race: all 3 components jointly (h = 0..25)
* ═════════════════════════════════════════════════════════════════════════

di ""
di as txt "--- Fig 8a: Horse-race (joint regression, h = 0..25) ---"

* component encoding: 1=econ, 2=diplo, 3=security
tempfile res8a
postfile _res8a horizon coef se ci_lo ci_hi component using `res8a'

forvalues h = `h_min'/`h_max' {
    cap drop _dep
    qui by cid: gen _dep = F`h'.y_ext

    cap qui reghdfe _dep `comp1' `comp2' `comp3' `y_lags' `comp_lags', ///
        absorb(cid region_year_id) cluster(country_code) keepsingletons

    if _rc == 0 {
        * Economic (component 1)
        local b  = _b[`comp1']
        local s  = _se[`comp1']
        post _res8a (`h') (`b') (`s') (`b' - 1.96 * `s') (`b' + 1.96 * `s') (1)

        * Diplomatic (component 2)
        local b  = _b[`comp2']
        local s  = _se[`comp2']
        post _res8a (`h') (`b') (`s') (`b' - 1.96 * `s') (`b' + 1.96 * `s') (2)

        * Security (component 3)
        local b  = _b[`comp3']
        local s  = _se[`comp3']
        post _res8a (`h') (`b') (`s') (`b' - 1.96 * `s') (`b' + 1.96 * `s') (3)
    }
    cap drop _dep
}

postclose _res8a

* ── Export to CSV ──
preserve
use `res8a', clear
export delimited using "${estimates}/fig8a_horserace.csv", replace
di as txt "  Saved: ${estimates}/fig8a_horserace.csv"

* ── Plot: reshape wide by component, then overlay ──
reshape wide coef se ci_lo ci_hi, i(horizon) j(component)

rename coef1 hr_econ_coef
rename coef2 hr_diplo_coef
rename coef3 hr_security_coef
rename ci_lo1 hr_econ_ci_lo
rename ci_lo2 hr_diplo_ci_lo
rename ci_lo3 hr_security_ci_lo
rename ci_hi1 hr_econ_ci_hi
rename ci_hi2 hr_diplo_ci_hi
rename ci_hi3 hr_security_ci_hi

* ── Plot: 3 colored lines with CI bands ──
* Colors: blue=Economic, orange=Diplomatic, green=Security
twoway (rarea hr_econ_ci_lo hr_econ_ci_hi horizon, ///
            color("31 119 180%20") lwidth(none)) ///
       (rarea hr_diplo_ci_lo hr_diplo_ci_hi horizon, ///
            color("255 127 14%20") lwidth(none)) ///
       (rarea hr_security_ci_lo hr_security_ci_hi horizon, ///
            color("44 160 44%20") lwidth(none)) ///
       (connected hr_econ_coef horizon, lcolor("31 119 180") ///
            mcolor("31 119 180") msymbol(O) msize(small) ///
            lwidth(medthick) lpattern(solid)) ///
       (connected hr_diplo_coef horizon, lcolor("255 127 14") ///
            mcolor("255 127 14") msymbol(S) msize(small) ///
            lwidth(medthick) lpattern(solid)) ///
       (connected hr_security_coef horizon, lcolor("44 160 44") ///
            mcolor("44 160 44") msymbol(T) msize(small) ///
            lwidth(medthick) lpattern(solid)), ///
    yline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    xline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    ytitle("Log GDP per Capita (x100)", size(medium)) ///
    xtitle("Years Relative to Shock", size(medium)) ///
    legend(order(4 "Economic" 5 "Diplomatic" 6 "Security") ///
           rows(1) pos(6) ring(1) size(small) region(lcolor(none))) ///
    graphregion(color(white)) plotregion(color(white)) ///
    name(fig8a, replace)
graph export "${figures}/Fig8A_component_horserace.pdf", as(pdf) replace
di as txt "  Saved: ${figures}/Fig8A_component_horserace.pdf"
restore


* ═════════════════════════════════════════════════════════════════════════
* Figure 8b — Residualized components (h = 0..25)
*
* Step 1: Residualize each component on the other two + country FE + year FE
* Step 2: Generate 4 lags of each residual
* Step 3: Run separate LP for each residualized component
* ═════════════════════════════════════════════════════════════════════════

di ""
di as txt "--- Fig 8b: Residualized (orthogonal) components (h = 0..25) ---"

* ── Step 1: Residualize ──
di as txt "  Residualizing components (absorbing country + year FE)..."

* Economic residual: regress econ on diplo + security, absorb(cid year)
cap drop econ_resid
qui reghdfe `comp1' `comp2' `comp3', absorb(cid year) resid(econ_resid)

* Diplomatic residual: regress diplo on econ + security, absorb(cid year)
cap drop diplo_resid
qui reghdfe `comp2' `comp1' `comp3', absorb(cid year) resid(diplo_resid)

* Security residual: regress security on econ + diplo, absorb(cid year)
cap drop security_resid
qui reghdfe `comp3' `comp1' `comp2', absorb(cid year) resid(security_resid)

* ── Step 2: Generate 4 lags of each residual ──
di as txt "  Generating residual lags..."
foreach comp in econ diplo security {
    forvalues l = 1/4 {
        cap drop `comp'_resid_lag`l'
        qui by cid: gen `comp'_resid_lag`l' = L`l'.`comp'_resid
    }
}

* ── Step 3: Run LP for each residualized component separately ──
* component encoding: 1=econ, 2=diplo, 3=security
tempfile res8b
postfile _res8b horizon coef se ci_lo ci_hi component using `res8b'

local comp_id = 0
foreach comp in econ diplo security {
    local comp_id = `comp_id' + 1
    di as txt "  Running LP for `comp' (residualized)..."
    local resid_var "`comp'_resid"
    local resid_lags "`comp'_resid_lag1 `comp'_resid_lag2 `comp'_resid_lag3 `comp'_resid_lag4"

    forvalues h = `h_min'/`h_max' {
        cap drop _dep
        qui by cid: gen _dep = F`h'.y_ext

        cap qui reghdfe _dep `resid_var' `y_lags' `resid_lags', ///
            absorb(cid region_year_id) cluster(country_code) keepsingletons

        if _rc == 0 {
            local b  = _b[`resid_var']
            local s  = _se[`resid_var']
            local lo = `b' - 1.96 * `s'
            local hi = `b' + 1.96 * `s'
            post _res8b (`h') (`b') (`s') (`lo') (`hi') (`comp_id')
        }
        cap drop _dep
    }
}

postclose _res8b

* ── Export to CSV ──
preserve
use `res8b', clear
export delimited using "${estimates}/fig8b_residualized.csv", replace
di as txt "  Saved: ${estimates}/fig8b_residualized.csv"

* ── Plot: reshape wide by component, then overlay ──
reshape wide coef se ci_lo ci_hi, i(horizon) j(component)

rename coef1 re_econ_coef
rename coef2 re_diplo_coef
rename coef3 re_security_coef
rename ci_lo1 re_econ_ci_lo
rename ci_lo2 re_diplo_ci_lo
rename ci_lo3 re_security_ci_lo
rename ci_hi1 re_econ_ci_hi
rename ci_hi2 re_diplo_ci_hi
rename ci_hi3 re_security_ci_hi

* ── Plot: 3 colored lines with CI bands ──
twoway (rarea re_econ_ci_lo re_econ_ci_hi horizon, ///
            color("31 119 180%20") lwidth(none)) ///
       (rarea re_diplo_ci_lo re_diplo_ci_hi horizon, ///
            color("255 127 14%20") lwidth(none)) ///
       (rarea re_security_ci_lo re_security_ci_hi horizon, ///
            color("44 160 44%20") lwidth(none)) ///
       (connected re_econ_coef horizon, lcolor("31 119 180") ///
            mcolor("31 119 180") msymbol(O) msize(small) ///
            lwidth(medthick) lpattern(solid)) ///
       (connected re_diplo_coef horizon, lcolor("255 127 14") ///
            mcolor("255 127 14") msymbol(S) msize(small) ///
            lwidth(medthick) lpattern(solid)) ///
       (connected re_security_coef horizon, lcolor("44 160 44") ///
            mcolor("44 160 44") msymbol(T) msize(small) ///
            lwidth(medthick) lpattern(solid)), ///
    yline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    xline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    ytitle("Log GDP per Capita (x100)", size(medium)) ///
    xtitle("Years Relative to Shock", size(medium)) ///
    legend(order(4 "Economic" 5 "Diplomatic" 6 "Security") ///
           rows(1) pos(6) ring(1) size(small) region(lcolor(none))) ///
    graphregion(color(white)) plotregion(color(white)) ///
    name(fig8b, replace)
graph export "${figures}/Fig8B_component_residualized.pdf", as(pdf) replace
di as txt "  Saved: ${figures}/Fig8B_component_residualized.pdf"
restore

* Clean up residual variables
cap drop econ_resid diplo_resid security_resid
cap drop *_resid_lag?

di ""
di as txt "  02_decomposition.do complete."
