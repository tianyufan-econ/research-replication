/*
03_symmetry.do
==============
Partner and temporal symmetry LP IRFs (Figures 10a and 10b).

  Fig 10a — Partner symmetry: US vs non-US geopolitical alignment → y_ext (h = 0..25)
             Joint regression with geo_relation_dyn_us and geo_relation_dyn_exclus
  Fig 10b — Temporal stability: Cold War (1960-1989) vs Post-Cold War (1990-2019) (h = 0..10)
             Baseline LP run on two subperiods separately

Assumes master.do has already:
  - loaded panel, created partner-decomposition lags, FE variables
  - set xtset cid year
  - defined $base_controls, $figures, $estimates
*/

set more off

di ""
di as txt "============================================================"
di as txt "  03_symmetry.do — Partner & Temporal Symmetry (Figs 10a, 10b)"
di as txt "============================================================"


* ═════════════════════════════════════════════════════════════════════════
* Figure 10a — Partner symmetry: US vs non-US (h = 0..25)
*
* Joint regression including both geo_relation_dyn_us and
* geo_relation_dyn_exclus with their respective 4 lags + y_ext lags.
* ═════════════════════════════════════════════════════════════════════════

di ""
di as txt "--- Fig 10a: Partner symmetry (US vs non-US, h = 0..25) ---"

local h_min = 0
local h_max = 25

* Partner-specific lag controls
local partner_controls ///
    geo_relation_dyn_us_lag1 geo_relation_dyn_us_lag2 ///
    geo_relation_dyn_us_lag3 geo_relation_dyn_us_lag4 ///
    geo_relation_dyn_exclus_lag1 geo_relation_dyn_exclus_lag2 ///
    geo_relation_dyn_exclus_lag3 geo_relation_dyn_exclus_lag4

local y_lags "y_ext_lag1 y_ext_lag2 y_ext_lag3 y_ext_lag4"

* spec encoding: 1=US, 2=exclUS
tempfile res10a
postfile _res10a horizon coef se ci_lo ci_hi spec using `res10a'

forvalues h = `h_min'/`h_max' {
    cap drop _dep
    qui by cid: gen _dep = F`h'.y_ext

    cap qui reghdfe _dep geo_relation_dyn_us geo_relation_dyn_exclus ///
        `y_lags' `partner_controls', ///
        absorb(cid region_year_id) cluster(country_code) keepsingletons

    if _rc == 0 {
        * US (spec 1)
        local b  = _b[geo_relation_dyn_us]
        local s  = _se[geo_relation_dyn_us]
        post _res10a (`h') (`b') (`s') (`b' - 1.96 * `s') (`b' + 1.96 * `s') (1)

        * Excl-US (spec 2)
        local b  = _b[geo_relation_dyn_exclus]
        local s  = _se[geo_relation_dyn_exclus]
        post _res10a (`h') (`b') (`s') (`b' - 1.96 * `s') (`b' + 1.96 * `s') (2)
    }
    cap drop _dep
}

postclose _res10a

* ── Export to CSV and plot ──
preserve
use `res10a', clear
export delimited using "${estimates}/fig10a_partner_symmetry.csv", replace
di as txt "  Saved: ${estimates}/fig10a_partner_symmetry.csv"

* Reshape wide for plotting
reshape wide coef se ci_lo ci_hi, i(horizon) j(spec)

rename coef1  us_coef
rename coef2  exclus_coef
rename ci_lo1 us_ci_lo
rename ci_lo2 exclus_ci_lo
rename ci_hi1 us_ci_hi
rename ci_hi2 exclus_ci_hi

* ── Plot: blue = US, red = non-US, both with CI bands ──
twoway (rarea us_ci_lo us_ci_hi horizon, ///
            color("31 119 180%20") lwidth(none)) ///
       (rarea exclus_ci_lo exclus_ci_hi horizon, ///
            color("214 39 40%20") lwidth(none)) ///
       (connected us_coef horizon, lcolor("31 119 180") ///
            mcolor("31 119 180") msymbol(O) msize(small) ///
            lwidth(medthick) lpattern(solid)) ///
       (connected exclus_coef horizon, lcolor("214 39 40") ///
            mcolor("214 39 40") msymbol(S) msize(small) ///
            lwidth(medthick) lpattern(solid)), ///
    yline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    xline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    ytitle("Log GDP per Capita (x100)", size(medium)) ///
    xtitle("Years Relative to Shock", size(medium)) ///
    legend(order(3 "Alignment with US" 4 "Alignment Excl. US") ///
           rows(1) pos(6) ring(1) size(small) region(lcolor(none))) ///
    graphregion(color(white)) plotregion(color(white)) ///
    name(fig10a, replace)
graph export "${figures}/Fig10A_symmetry_partner.pdf", as(pdf) replace
di as txt "  Saved: ${figures}/Fig10A_symmetry_partner.pdf"
restore


* ═════════════════════════════════════════════════════════════════════════
* Figure 10b — Temporal stability: Cold War vs Post-Cold War (h = 0..10)
*
* Run baseline LP specification on two subperiods:
*   Cold War:      year >= 1960 & year <= 1989
*   Post-Cold War: year >= 1990 & year <= 2019
* ═════════════════════════════════════════════════════════════════════════

di ""
di as txt "--- Fig 10b: Temporal stability (Cold War vs Post-Cold War, h = 0..10) ---"

local h_min_t = 0
local h_max_t = 10

* spec encoding: 1=cold war, 2=post-cold war
tempfile res10b
postfile _res10b horizon coef se ci_lo ci_hi spec using `res10b'

* ── Cold War: 1960-1989 ──
di as txt "  Running Cold War subperiod (1960-1989)..."
preserve
    qui keep if year >= 1960 & year <= 1989

    * Rebuild lags within subperiod (match Python: subset first, then lag)
    drop y_ext_lag1 y_ext_lag2 y_ext_lag3 y_ext_lag4
    drop geo_lag1 geo_lag2 geo_lag3 geo_lag4
    xtset cid year
    forvalues l = 1/4 {
        qui by cid: gen y_ext_lag`l' = L`l'.y_ext
        qui by cid: gen geo_lag`l' = L`l'.geo_relation_dyn
    }

    forvalues h = `h_min_t'/`h_max_t' {
        cap drop _dep
        qui by cid: gen _dep = F`h'.y_ext

        cap qui reghdfe _dep geo_relation_dyn $base_controls, ///
            absorb(cid region_year_id) cluster(country_code) keepsingletons

        if _rc == 0 {
            local b  = _b[geo_relation_dyn]
            local s  = _se[geo_relation_dyn]
            post _res10b (`h') (`b') (`s') (`b' - 1.96 * `s') (`b' + 1.96 * `s') (1)
        }
        cap drop _dep
    }
restore

* ── Post-Cold War: 1990-2019 ──
di as txt "  Running Post-Cold War subperiod (1990-2019)..."
preserve
    qui keep if year >= 1990 & year <= 2019

    * Rebuild lags within subperiod (match Python: subset first, then lag)
    drop y_ext_lag1 y_ext_lag2 y_ext_lag3 y_ext_lag4
    drop geo_lag1 geo_lag2 geo_lag3 geo_lag4
    xtset cid year
    forvalues l = 1/4 {
        qui by cid: gen y_ext_lag`l' = L`l'.y_ext
        qui by cid: gen geo_lag`l' = L`l'.geo_relation_dyn
    }

    forvalues h = `h_min_t'/`h_max_t' {
        cap drop _dep
        qui by cid: gen _dep = F`h'.y_ext

        cap qui reghdfe _dep geo_relation_dyn $base_controls, ///
            absorb(cid region_year_id) cluster(country_code) keepsingletons

        if _rc == 0 {
            local b  = _b[geo_relation_dyn]
            local s  = _se[geo_relation_dyn]
            post _res10b (`h') (`b') (`s') (`b' - 1.96 * `s') (`b' + 1.96 * `s') (2)
        }
        cap drop _dep
    }
restore

postclose _res10b

* ── Export to CSV and plot ──
preserve
use `res10b', clear
export delimited using "${estimates}/fig10b_temporal_stability.csv", replace
di as txt "  Saved: ${estimates}/fig10b_temporal_stability.csv"

* Reshape wide for plotting
reshape wide coef se ci_lo ci_hi, i(horizon) j(spec)

rename coef1  cold_coef
rename coef2  post_coef
rename ci_lo1 cold_ci_lo
rename ci_lo2 post_ci_lo
rename ci_hi1 cold_ci_hi
rename ci_hi2 post_ci_hi

* ── Plot: blue = Cold War, orange = Post-Cold War, both with CI bands ──
twoway (rarea cold_ci_lo cold_ci_hi horizon, ///
            color("31 119 180%20") lwidth(none)) ///
       (rarea post_ci_lo post_ci_hi horizon, ///
            color("255 127 14%20") lwidth(none)) ///
       (connected cold_coef horizon, lcolor("31 119 180") ///
            mcolor("31 119 180") msymbol(O) msize(small) ///
            lwidth(medthick) lpattern(solid)) ///
       (connected post_coef horizon, lcolor("255 127 14") ///
            mcolor("255 127 14") msymbol(S) msize(small) ///
            lwidth(medthick) lpattern(dash)), ///
    yline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    xline(0, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    ytitle("Log GDP per Capita (x100)", size(medium)) ///
    xtitle("Years Relative to Shock", size(medium)) ///
    legend(order(3 "Cold War (1960-1989)" 4 "Post-Cold War (1990-2019)") ///
           rows(1) pos(6) ring(1) size(small) region(lcolor(none))) ///
    graphregion(color(white)) plotregion(color(white)) ///
    name(fig10b, replace)
graph export "${figures}/Fig10B_symmetry_temporal.pdf", as(pdf) replace
di as txt "  Saved: ${figures}/Fig10B_symmetry_temporal.pdf"
restore

di ""
di as txt "  03_symmetry.do complete."
