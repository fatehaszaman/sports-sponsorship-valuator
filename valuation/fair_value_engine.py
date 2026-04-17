"""
valuation/fair_value_engine.py — Sponsorship Fair Value Engine.

Business summary
----------------
The master valuation model. Takes three independent signals — exposure-based
CPM value, comparable deals regression, and audience demographic premium — and
blends them into a single fair value estimate using a configurable weight vector.

On top of the blended value, it computes:
  - Exclusivity premium: paying for category exclusivity (the only bank, the
    only airline, etc.) commands a structural uplift.
  - Multi-year NPV: the present value of committing to N years at an assumed
    growth rate, discounted at a WACC proxy — the same DCF mechanics used in
    corporate finance, applied to sponsorship cash flows.
  - Scenario analysis: bear/base/bull case values with explicit assumption sets.
  - Sensitivity table: how NPV changes as discount rate varies 5%→15% and
    growth rate varies 2%→8% — a 5×4 grid surfacing model sensitivity.
  - ROI projection: revenue lift, brand awareness lift, customer acquisition.

DCF Formula
-----------
For a deal with annual value V growing at rate g, discounted at rate r:

    NPV = Σ_{t=1}^{T}  V × (1 + g)^(t-1)
                        ──────────────────
                            (1 + r)^t

Which simplifies (for g ≠ r) to the Gordon-Growth-style closed form:

    NPV = V × (1 - ((1+g)/(1+r))^T) / (r - g)

Both the loop form (explicit, auditable) and the closed form are implemented.
The loop form is used in scenario analysis so each year's cash flow is visible.

Developer notes
---------------
The engine is intentionally designed to be a pure function machine: all methods
are deterministic given the same inputs. Randomness lives in monte_carlo.py.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_GROWTH_RATE,
    EXCLUSIVITY_PREMIUMS,
    VALUATION_WEIGHTS,
    PSYCH_WEIGHT,
    PHYSICAL_WEIGHT,
    ATHLETE_TOTAL_WEIGHT,
    PSYCH_WEIGHT_SAFE,
    PHYSICAL_WEIGHT_SAFE,
    ATHLETE_TOTAL_WEIGHT_SAFE,
    BRAND_SAFETY_SENSITIVE_BRANDS,
)
from data.sports_data import BRAND_PROFILES, SPORTS_PROPERTIES
from schemas.models import ROIProjection, SensitivityTable, ValuationResult
from valuation.audience_model import AudienceDemographicModel
from valuation.comparable_deals import ComparableDealsAnalyzer
from valuation.exposure_model import ExposureValueModel


# ── Scenario Assumption Sets ──────────────────────────────────────────────────

_SCENARIO_ASSUMPTIONS: dict[str, dict] = {
    "bear": {
        "label":          "Bear Case",
        "description":    "Viewership −20%, engagement −25%, no growth, high discount",
        "exposure_scale": 0.75,   # scale factor on exposure value
        "comp_scale":     0.85,   # market comps compress in bad markets
        "growth_rate":    0.00,
        "discount_rate":  0.14,
        "audience_scale": 0.90,
    },
    "base": {
        "label":          "Base Case",
        "description":    "Consensus expectations; config defaults",
        "exposure_scale": 1.00,
        "comp_scale":     1.00,
        "growth_rate":    DEFAULT_GROWTH_RATE,
        "discount_rate":  DEFAULT_DISCOUNT_RATE,
        "audience_scale": 1.00,
    },
    "bull": {
        "label":          "Bull Case",
        "description":    "Viewership +15%, engagement +20%, 8% growth, low discount",
        "exposure_scale": 1.18,
        "comp_scale":     1.12,
        "growth_rate":    0.08,
        "discount_rate":  0.07,
        "audience_scale": 1.10,
    },
}

# ── Brand Revenue / Awareness Lift Parameters (hardcoded research baselines) ──
# Sources: IEG Sponsorship Report, Harvard Business School case studies,
#          McKinsey "Value of Sponsorship" (2019).
_BRAND_AWARENESS_BASELINE: dict[str, float] = {
    "emirates": 0.78,  "nike": 0.95, "adidas": 0.91, "pepsi": 0.92,
    "heineken": 0.84,  "rolex": 0.80, "crypto_com": 0.45, "qatar_airways": 0.68,
    "visa": 0.88,      "oracle": 0.72, "samsung": 0.89, "mastercard": 0.87,
    "aramco": 0.35,    "etihad": 0.62, "spotify": 0.82,
}

# Revenue lift per $10m of sports sponsorship spend (as fraction of annual revenue)
# Varies by category — airlines and telecoms see more direct lift than FMCG.
_REVENUE_LIFT_PER_10M: dict[str, float] = {
    "airlines":      0.0018,   # 0.18% of annual revenue per $10m spend
    "telecoms":      0.0015,
    "sportswear":    0.0025,
    "soft_drinks":   0.0012,
    "beer_alcohol":  0.0014,
    "luxury_watches":0.0020,
    "financial_svcs":0.0010,
    "crypto_fintech":0.0022,
    "energy_oil":    0.0008,
    "insurance":     0.0009,
    "default":       0.0013,
}


class SponsorshipFairValueEngine:
    """
    Master valuation engine blending exposure, comps, and demographic signals.

    Business summary
    ----------------
    This is the top-level model a sports finance analyst would use. It takes a
    brand name, a property name, and an asset type, and returns a comprehensive
    valuation package including:
      - Weighted fair value (USD m/yr)
      - Full DCF / NPV for a multi-year deal
      - Bear / base / bull scenario values
      - NPV sensitivity table across discount rates and growth rates
      - ROI projection for the sponsoring brand

    Usage
    -----
    >>> engine = SponsorshipFairValueEngine()
    >>> result = engine.intrinsic_value("emirates", "manchester_city", "jersey")
    >>> print(f"Fair value: ${result.final_fair_value_usd_m:.1f}m / year")
    >>> sens = engine.sensitivity_table("emirates", "manchester_city", 5)
    """

    def __init__(self) -> None:
        self._exposure   = ExposureValueModel()
        self._audience   = AudienceDemographicModel()
        self._comps      = ComparableDealsAnalyzer()
        self._comps.fit()                     # train OLS regression on init
        self._weights    = VALUATION_WEIGHTS

    # ── Primary Valuation ───────────────────────────────────────────────────

    def intrinsic_value(
        self,
        brand_key: str,
        property_key: str,
        asset_type: str = "jersey",
        deal_years: int = 5,
        is_exclusive: bool = False,
        discount_rate: Optional[float] = None,
        growth_rate:   Optional[float] = None,
    ) -> ValuationResult:
        """
        Full sponsorship valuation combining all signals.

        The blending formula is:

            fair_value = Σ_i  weight_i × signal_i

        where signals are:
          - exposure_value:  bottom-up CPM impression model
          - comp_implied:    OLS regression on comparable deals
          - audience_premium: demographic fit adjustment (additive, can be negative)
          - social_signal:   social amplification-derived value

        Exclusivity premium is added on top of the blended value:
            final = weighted_fair_value + exclusivity_premium

        Parameters
        ----------
        brand_key : str
            Brand key from BRAND_PROFILES.
        property_key : str
            Property key from SPORTS_PROPERTIES + _PROPERTY_DEMOGRAPHICS.
        asset_type : str
            Sponsorship asset type (jersey, stadium_naming, etc.).
        deal_years : int
            Multi-year deal term for NPV calculation.
        is_exclusive : bool
            Whether category exclusivity is sought.
        discount_rate : float, optional
            WACC proxy. Defaults to config.DEFAULT_DISCOUNT_RATE.
        growth_rate : float, optional
            Annual sponsorship value growth rate. Defaults to DEFAULT_GROWTH_RATE.

        Returns
        -------
        ValuationResult
            Fully validated Pydantic schema with all valuation components.
        """
        dr = discount_rate if discount_rate is not None else DEFAULT_DISCOUNT_RATE
        gr = growth_rate   if growth_rate   is not None else DEFAULT_GROWTH_RATE

        brand    = BRAND_PROFILES.get(brand_key)
        property_ = SPORTS_PROPERTIES.get(property_key)

        if brand is None:
            raise ValueError(f"Unknown brand '{brand_key}'.")
        if property_ is None:
            raise ValueError(f"Unknown property '{property_key}'.")

        # ── Signal 1: Exposure (bottom-up CPM model) ─────────────────────────
        exp_usd   = self._exposure.total_exposure_value(property_key, asset_type)
        exp_usd_m = exp_usd / 1_000_000

        # ── Signal 2: Comparable Deals (OLS regression) ──────────────────────
        comp_usd_m = self._comps.regression_for_property(
            property_.name, deal_type=asset_type, is_exclusive=is_exclusive
        )

        # ── Signal 3: Audience Demographic Premium ───────────────────────────
        # Audience premium = deviation of the demographic-adjusted value
        # from the simple exposure value.
        # audience_premium = exp_usd_m × (multiplier - 1.0)
        # This can be negative (bad demographic fit discounts the base value).
        fit_mult      = self._audience.premium_multiplier(brand_key, property_key)
        aud_prem_usd_m = exp_usd_m * (fit_mult - 1.0)

        # ── Signal 4: Social Amplification Signal ────────────────────────────
        # Approximated from social followers and engagement rate.
        prop = SPORTS_PROPERTIES[property_key]
        social_score     = prop.social_followers_m * prop.avg_engagement_rate / 100.0
        social_usd_m     = social_score * 2.5  # $2.5m per unit of social score

        # ── Weighted Blend ────────────────────────────────────────────────────
        # weighted_fair_value = Σ weight_i × signal_i
        # Note: audience_premium is already an adjustment ON TOP of exposure,
        # so it uses exposure as base and blends in the delta.
        w = self._weights
        weighted_fv = (
            w["exposure_value"]   * exp_usd_m +
            w["comp_implied"]     * comp_usd_m +
            w["audience_premium"] * (exp_usd_m + aud_prem_usd_m) +   # adjusted exposure
            w["social_signal"]    * social_usd_m
        )

        # ── Exclusivity Premium ───────────────────────────────────────────────
        excl_prem = 0.0
        if is_exclusive:
            excl_prem = self.exclusivity_premium(brand.category, property_key) * weighted_fv

        final_fv = weighted_fv + excl_prem

        # ── Multi-Year NPV (DCF) ──────────────────────────────────────────────
        npv = self.multi_year_npv(final_fv, deal_years, gr, dr)

        # ── Scenario Valuations ───────────────────────────────────────────────
        bear_val = self._scenario_value(exp_usd_m, comp_usd_m, aud_prem_usd_m, social_usd_m, "bear", is_exclusive, brand.category, property_key)
        bull_val = self._scenario_value(exp_usd_m, comp_usd_m, aud_prem_usd_m, social_usd_m, "bull", is_exclusive, brand.category, property_key)

        return ValuationResult(
            property_key=property_key,
            brand_key=brand_key,
            asset_type=asset_type,
            exposure_value_usd_m=round(exp_usd_m, 2),
            comp_implied_value_usd_m=round(comp_usd_m, 2),
            audience_premium_usd_m=round(aud_prem_usd_m, 2),
            social_signal_usd_m=round(social_usd_m, 2),
            weighted_fair_value_usd_m=round(weighted_fv, 2),
            exclusivity_premium_usd_m=round(excl_prem, 2),
            final_fair_value_usd_m=round(final_fv, 2),
            deal_years=deal_years,
            discount_rate=dr,
            growth_rate=gr,
            npv_usd_m=round(npv, 2),
            bear_case_usd_m=round(bear_val, 2),
            base_case_usd_m=round(final_fv, 2),
            bull_case_usd_m=round(bull_val, 2),
        )

    # ── DCF / NPV Model ─────────────────────────────────────────────────────

    def multi_year_npv(
        self,
        annual_value: float,
        years: int,
        growth_rate: float = DEFAULT_GROWTH_RATE,
        discount_rate: float = DEFAULT_DISCOUNT_RATE,
    ) -> float:
        """
        Net present value of a multi-year sponsorship deal.

        DCF Formula (explicit year-by-year loop form):
        -----------------------------------------------
        Each year t ∈ {1, …, T}, the deal delivers value:

            CF_t = annual_value × (1 + growth_rate)^(t-1)

        The present value of CF_t, discounted at rate r:

            PV_t = CF_t / (1 + discount_rate)^t

        Total NPV:
            NPV = Σ_{t=1}^{T} PV_t

        Closed-form equivalent (Gordon Growth, for g ≠ r):
            NPV = V × [1 − ((1+g)/(1+r))^T] / (r − g)

        Both are implemented and cross-checked in unit tests.

        Parameters
        ----------
        annual_value : float
            Year-1 annual sponsorship value in USD millions.
        years : int
            Deal term in years.
        growth_rate : float
            Annual growth rate of deal value (as decimal, e.g. 0.05).
        discount_rate : float
            Discount rate / WACC proxy (as decimal, e.g. 0.10).

        Returns
        -------
        float
            NPV in USD millions.
        """
        # ── Explicit DCF loop (auditable, year-by-year) ──────────────────────
        npv = 0.0
        for t in range(1, years + 1):
            cf_t = annual_value * (1 + growth_rate) ** (t - 1)   # growing cash flow
            pv_t = cf_t / (1 + discount_rate) ** t                # discounted to t=0
            npv += pv_t

        return npv

    def multi_year_npv_closed_form(
        self,
        annual_value: float,
        years: int,
        growth_rate: float = DEFAULT_GROWTH_RATE,
        discount_rate: float = DEFAULT_DISCOUNT_RATE,
    ) -> float:
        """
        Closed-form NPV using Gordon Growth formula.

        For g ≠ r:
            NPV = V × [1 − ((1+g)/(1+r))^T] / (r − g)

        For g = r (edge case):
            NPV = V × T / (1 + r)

        Used as cross-check against the loop form.
        """
        if abs(growth_rate - discount_rate) < 1e-9:
            # Special case: g = r
            return annual_value * years / (1 + discount_rate)
        ratio = (1 + growth_rate) / (1 + discount_rate)
        npv   = annual_value * (1 - ratio ** years) / (discount_rate - growth_rate)
        return npv

    # ── Scenario Analysis ───────────────────────────────────────────────────

    def scenario_analysis(
        self,
        brand_key: str,
        property_key: str,
        asset_type: str = "jersey",
        deal_years: int = 5,
        is_exclusive: bool = False,
    ) -> dict:
        """
        Run bear / base / bull scenarios with explicit assumption sets.

        Returns
        -------
        dict
            {scenario_name: {annual_value, npv, assumptions}} for each scenario.
        """
        brand    = BRAND_PROFILES[brand_key]
        property_ = SPORTS_PROPERTIES[property_key]

        exp_usd_m  = self._exposure.total_exposure_value(property_key, asset_type) / 1_000_000
        comp_usd_m = self._comps.regression_for_property(property_.name, deal_type=asset_type)
        fit_mult   = self._audience.premium_multiplier(brand_key, property_key)
        aud_prem   = exp_usd_m * (fit_mult - 1.0)
        prop       = SPORTS_PROPERTIES[property_key]
        social_usd_m = prop.social_followers_m * prop.avg_engagement_rate / 100.0 * 2.5

        results = {}
        for sc_name, sc in _SCENARIO_ASSUMPTIONS.items():
            ann_val = self._scenario_value(
                exp_usd_m, comp_usd_m, aud_prem, social_usd_m,
                sc_name, is_exclusive, brand.category, property_key
            )
            npv = self.multi_year_npv(ann_val, deal_years, sc["growth_rate"], sc["discount_rate"])
            results[sc_name] = {
                "label":          sc["label"],
                "description":    sc["description"],
                "annual_value_usd_m": round(ann_val, 2),
                "npv_usd_m":      round(npv, 2),
                "assumptions":    {k: v for k, v in sc.items() if k not in ("label", "description")},
            }
        return results

    def _scenario_value(
        self,
        exp_usd_m: float,
        comp_usd_m: float,
        aud_prem_usd_m: float,
        social_usd_m: float,
        scenario: str,
        is_exclusive: bool,
        brand_category: str,
        property_key: str,
    ) -> float:
        """Compute blended annual fair value under a named scenario."""
        sc = _SCENARIO_ASSUMPTIONS[scenario]
        w  = self._weights

        scaled_exp    = exp_usd_m    * sc["exposure_scale"]
        scaled_comp   = comp_usd_m   * sc["comp_scale"]
        scaled_aud    = (exp_usd_m + aud_prem_usd_m) * sc["audience_scale"]
        scaled_social = social_usd_m * sc["exposure_scale"]

        weighted = (
            w["exposure_value"]   * scaled_exp +
            w["comp_implied"]     * scaled_comp +
            w["audience_premium"] * scaled_aud +
            w["social_signal"]    * scaled_social
        )

        if is_exclusive:
            excl = self.exclusivity_premium(brand_category, property_key)
            weighted += excl * weighted

        return weighted

    # ── Sensitivity Table ───────────────────────────────────────────────────

    def sensitivity_table(
        self,
        brand_key: str,
        property_key: str,
        deal_years: int = 5,
        asset_type: str = "jersey",
        discount_rates: Optional[list[float]] = None,
        growth_rates: Optional[list[float]] = None,
    ) -> SensitivityTable:
        """
        Compute NPV sensitivity across a grid of (discount_rate, growth_rate) pairs.

        Business summary
        ----------------
        This is the "what if" table every CFO wants to see. It shows how the
        deal's NPV changes as our macro assumptions shift. High sensitivity
        (wide NPV range) → risky deal. Low sensitivity → robust valuation.

        Grid defaults
        -------------
        discount_rates : [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
        growth_rates   : [0.02, 0.04, 0.06, 0.08]

        Implementation
        --------------
        Uses numpy broadcasting to compute the full (D × G) NPV matrix
        in a single vectorized operation:

            For each (r, g) pair:
            t_vec  = np.arange(1, T+1)
            cf_vec = V × (1+g)^(t_vec - 1)      # growing cash flows
            pv_vec = cf_vec / (1+r)^t_vec        # discounted cash flows
            NPV    = pv_vec.sum()

        The outer loop over (r, g) pairs can be fully vectorized using
        2-D broadcasting, which is done below.

        Returns
        -------
        SensitivityTable
            Pydantic schema with npv_matrix[i][j] = NPV at dr[i], gr[j].
        """
        dr_list = discount_rates or [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
        gr_list = growth_rates   or [0.02, 0.04, 0.06, 0.08]

        # Get the base annual fair value
        result    = self.intrinsic_value(brand_key, property_key, asset_type, deal_years)
        base_val  = result.final_fair_value_usd_m

        # ── Vectorized NPV grid computation ───────────────────────────────────
        # t_vec shape: (T,) — years 1 through T
        t_vec = np.arange(1, deal_years + 1, dtype=float)  # shape (T,)

        # dr_arr shape: (D, 1, 1), gr_arr shape: (1, G, 1), t_arr shape: (1, 1, T)
        dr_arr = np.array(dr_list)[:, np.newaxis, np.newaxis]   # (D, 1, 1)
        gr_arr = np.array(gr_list)[np.newaxis, :, np.newaxis]   # (1, G, 1)
        t_arr  = t_vec[np.newaxis, np.newaxis, :]               # (1, 1, T)

        # Cash flow at each (r, g, t) combination — broadcasted
        cf_cube = base_val * (1 + gr_arr) ** (t_arr - 1)        # (D, G, T)
        pv_cube = cf_cube / (1 + dr_arr) ** t_arr               # (D, G, T)
        npv_mat = pv_cube.sum(axis=2)                            # (D, G) — sum over T

        return SensitivityTable(
            property_key=property_key,
            brand_key=brand_key,
            annual_base_value_usd_m=base_val,
            deal_years=deal_years,
            discount_rates=dr_list,
            growth_rates=gr_list,
            npv_matrix=[[round(npv_mat[i, j], 2) for j in range(len(gr_list))]
                        for i in range(len(dr_list))],
        )

    def print_sensitivity_table(
        self,
        brand_key: str,
        property_key: str,
        deal_years: int = 5,
    ) -> None:
        """Print a formatted sensitivity table to stdout."""
        st = self.sensitivity_table(brand_key, property_key, deal_years)
        gr_labels = [f"g={r:.0%}" for r in st.growth_rates]
        dr_labels = [f"r={r:.0%}" for r in st.discount_rates]

        col_w = 12
        header = f"{'NPV (USD m)':>{col_w}} " + "  ".join(f"{g:>{col_w}}" for g in gr_labels)
        print(f"\n  NPV Sensitivity Table — {brand_key} × {property_key} "
              f"({deal_years}yr deal, base value ${st.annual_base_value_usd_m:.1f}m/yr)")
        print("  " + "─" * len(header))
        print("  " + header)
        print("  " + "─" * len(header))
        for dr_label, row in zip(dr_labels, st.npv_matrix):
            row_str = "  ".join(f"{v:>{col_w}.1f}" for v in row)
            print(f"  {dr_label:>{col_w}} {row_str}")
        print("  " + "─" * len(header))

    # ── Exclusivity Premium ─────────────────────────────────────────────────

    def exclusivity_premium(self, brand_category: str, property_key: str) -> float:
        """
        Category exclusivity premium as a fraction of base value.

        Being the ONLY airline sponsor of Man City commands a structural
        premium — competitors cannot reach that audience via this vehicle.
        Premiums vary by category and property's commercial density.

        Returns
        -------
        float
            Premium fraction (e.g. 0.25 = 25% on top of base value).
        """
        base_prem = EXCLUSIVITY_PREMIUMS.get(brand_category, EXCLUSIVITY_PREMIUMS["default"])
        # Top-tier properties command higher exclusivity premiums (more competitive bidding)
        reach_lookup = {
            "manchester_city": 1.10, "manchester_united": 1.20, "real_madrid": 1.25,
            "barcelona": 1.22, "liverpool": 1.15, "nfl": 1.30, "formula_1": 1.28,
            "red_bull_racing": 1.18, "la_lakers": 1.15, "nba": 1.25,
            "dallas_cowboys": 1.20, "super_bowl": 1.40,
        }
        property_factor = reach_lookup.get(property_key, 1.0)
        return round(base_prem * property_factor, 4)

    # ── ROI Projection ──────────────────────────────────────────────────────

    # ── Athlete Endorsement Value ────────────────────────────────────────────

    def athlete_endorsement_value(
        self,
        athlete_key: str,
        brand_key: str,
        brand_safety_sensitive: Optional[bool] = None,
    ) -> dict:
        """
        Compute the fair endorsement value for an athlete-brand pairing.

        Business summary
        ----------------
        Athlete endorsement value combines two dimensions:

          - Physical/performance score: on-field achievement and global recognition.
            Elite athletes generate enormous media coverage regardless of personal
            conduct. This is the primary commercial driver.

          - Psychological/reputation score: brand safety, character, public
            sentiment. Acts as a risk-adjustment rather than a primary driver.

        Standard weighting formula (config.PSYCH_WEIGHT / PHYSICAL_WEIGHT)
        --------------------------------------------------------------------
        Uses constants imported from config.py:

            PSYCH_WEIGHT    = 1.0   # reputation / character
            PHYSICAL_WEIGHT = 1.5   # on-field performance / athletic achievement
            TOTAL_WEIGHT    = 2.5

            athlete_score = (
                (reputation_score * PSYCH_WEIGHT) +
                (performance_score * PHYSICAL_WEIGHT)
            ) / TOTAL_WEIGHT

        Rationale: sponsors primarily buy athletic visibility and achievement.
        A world-class athlete with some controversy (reputation=50/100) still
        delivers substantial value; brand risk discount is real but not
        disqualifying for most sponsors. The 1.5:1.0 ratio reflects that
        sponsors accept some reputational risk for elite athletic association.

        Brand-safety-sensitive override (config.PSYCH_WEIGHT_SAFE)
        -------------------------------------------------------------
        Conservative brands (Rolex, Visa, Mastercard) flip the weights to
        PSYCH_WEIGHT=1.5, PHYSICAL_WEIGHT=1.0, making reputation the primary
        driver. A single controversy can materially damage these brands\'s core
        promise of trustworthiness or prestige.

            athlete_score_safe = (
                (reputation_score * PSYCH_WEIGHT_SAFE) +
                (performance_score * PHYSICAL_WEIGHT_SAFE)
            ) / TOTAL_WEIGHT_SAFE

        Auto-detected for brands in config.BRAND_SAFETY_SENSITIVE_BRANDS;
        manually overridable via brand_safety_sensitive param.
        """
        from data.sports_data import ATHLETE_PROFILES

        athlete = ATHLETE_PROFILES.get(athlete_key)
        brand   = BRAND_PROFILES.get(brand_key)
        if athlete is None:
            raise ValueError(f"Unknown athlete '{athlete_key}'.")
        if brand is None:
            raise ValueError(f"Unknown brand '{brand_key}'.")

        # ── Determine weight regime ───────────────────────────────────────────
        if brand_safety_sensitive is None:
            brand_safety_sensitive = brand_key in BRAND_SAFETY_SENSITIVE_BRANDS

        if brand_safety_sensitive:
            pw  = PSYCH_WEIGHT_SAFE      # 1.5 — reputation is primary driver
            phw = PHYSICAL_WEIGHT_SAFE   # 1.0
            tw  = ATHLETE_TOTAL_WEIGHT_SAFE
        else:
            pw  = PSYCH_WEIGHT           # 1.0
            phw = PHYSICAL_WEIGHT        # 1.5 — performance is primary driver
            tw  = ATHLETE_TOTAL_WEIGHT

        # ── Score inputs (both scaled to 0–100) ──────────────────────────────
        # Reputation: inverted controversiality (0–10 → 0–100; high = brand-safe)
        reputation_score  = max(0.0, (10.0 - athlete.controversiality_score) * 10.0)
        # Performance: composite of on-field rating (60%) + global recognition (40%)
        performance_score = (
            athlete.on_field_rating * 0.6 +
            athlete.global_recognition_score * 0.4
        ) * 10.0

        # ── Core weighted formula ─────────────────────────────────────────────
        #   score = (reputation_score × PSYCH_WEIGHT
        #            + performance_score × PHYSICAL_WEIGHT) / TOTAL_WEIGHT
        weighted_score = (
            (reputation_score * pw) + (performance_score * phw)
        ) / tw

        # ── Social reach value ────────────────────────────────────────────────
        total_followers_m = (
            athlete.instagram_followers_m + athlete.twitter_followers_m +
            athlete.tiktok_followers_m    + athlete.youtube_subscribers_m
        )
        eff_impressions = (
            total_followers_m * 1_000_000 *
            (athlete.avg_engagement_rate / 100.0) *
            1.20 * 52   # organic amplification × ~52 brand posts/year
        )
        social_reach_usd_m = (eff_impressions / 1_000) * 8.75 / 1_000_000  # $8.75 CPM

        # ── Demographic fit ───────────────────────────────────────────────────
        _athlete_to_property = {
            "cristiano_ronaldo": "manchester_united",
            "lionel_messi":      "barcelona",
            "lebron_james":      "la_lakers",
            "naomi_osaka":       "wimbledon",
            "max_verstappen":    "red_bull_racing",
            "serena_williams":   "wimbledon",
            "neymar":            "psg",
            "lewis_hamilton":    "formula_1",
            "stephen_curry":     "golden_state_warriors",
            "kylian_mbappe":     "real_madrid",
        }
        prop_key = _athlete_to_property.get(athlete_key, "formula_1")
        try:
            demo_fit = self._audience.demographic_fit_score(brand_key, prop_key)
        except ValueError:
            demo_fit = 0.65

        # ── Fair endorsement value ────────────────────────────────────────────
        quality_scale  = weighted_score / 100.0
        fame_mult      = athlete.global_recognition_score / 10.0
        endorsement_fv = social_reach_usd_m * quality_scale * demo_fit * fame_mult

        # Risk adjustment: brand-safety-sensitive brands apply heavier discount
        # for controversiality above 5.0 (scale of 0–10)
        risk_adjustment = 1.0
        if brand_safety_sensitive and athlete.controversiality_score > 5.0:
            excess = athlete.controversiality_score - 5.0
            risk_adjustment = max(0.40, 1.0 - (excess * 0.10))
            endorsement_fv *= risk_adjustment

        return {
            "athlete":                      athlete.name,
            "brand":                        brand.name,
            "brand_safety_sensitive":        brand_safety_sensitive,
            "reputation_score":             round(reputation_score, 1),
            "performance_score":            round(performance_score, 1),
            "psych_weight":                 pw,
            "physical_weight":              phw,
            "total_weight":                 tw,
            "weighted_athlete_score":       round(weighted_score, 1),
            "demographic_fit":              round(demo_fit, 4),
            "social_reach_value_usd_m":     round(social_reach_usd_m, 2),
            "endorsement_fair_value_usd_m": round(endorsement_fv, 2),
            "annual_value_usd_m":           round(endorsement_fv, 2),
            "risk_adjustment":              round(risk_adjustment, 4),
            "formula": (
                f"({reputation_score:.1f} × {pw} [psych]) + "
                f"({performance_score:.1f} × {phw} [physical]) "
                f"/ {tw} = {weighted_score:.1f} / 100"
            ),
        }


    def roi_projection(
        self,
        brand_key: str,
        deal_value_usd_m: float,
        property_key: str,
        deal_years: int = 5,
        is_exclusive: bool = False,
    ) -> ROIProjection:
        """
        Project the sponsoring brand's financial return from the deal.

        Methodology
        -----------
        Revenue lift:
          lift_pct_per_year = _REVENUE_LIFT_PER_10M[category] × (deal_value / 10)
          cumulative_lift = annual_revenue × lift_pct_per_year × years × time_decay

        Brand awareness:
          awareness_lift_pp = (1 - current_awareness) × fit_score × 0.12 × deal_years
          (max 20 pp over deal term)

        Customer acquisition:
          new_customers = revenue_lift / avg_customer_value (industry default: $150/customer)

        Parameters
        ----------
        brand_key : str
        deal_value_usd_m : float
            Proposed annual deal value.
        property_key : str
        deal_years : int

        Returns
        -------
        ROIProjection
            Full ROI model output.
        """
        brand   = BRAND_PROFILES[brand_key]
        fit     = self._audience.demographic_fit_score(brand_key, property_key)

        # Revenue lift model
        lpm     = _REVENUE_LIFT_PER_10M.get(brand.category, _REVENUE_LIFT_PER_10M["default"])
        ann_lift_pct  = lpm * (deal_value_usd_m / 10.0)
        # Time-decay: later years generate less incremental lift (saturation)
        total_lift    = 0.0
        for t in range(1, deal_years + 1):
            decay = 1.0 / (1.0 + 0.08 * (t - 1))
            total_lift += ann_lift_pct * decay
        rev_lift_usd_m = brand.annual_revenue_usd_b * 1_000 * total_lift * fit

        # Awareness lift
        current_awareness = _BRAND_AWARENESS_BASELINE.get(brand_key, 0.70)
        awareness_lift_pp = (1 - current_awareness) * fit * 0.12 * deal_years
        awareness_lift_pp = min(awareness_lift_pp, 20.0)

        # Consideration lift (roughly 0.6× awareness lift in practice)
        consideration_lift_pp = awareness_lift_pp * 0.60

        # Customer acquisition
        avg_customer_value = 150.0  # USD (industry default)
        total_deal_cost    = deal_value_usd_m * deal_years
        new_customers      = int((rev_lift_usd_m * 1_000_000) / avg_customer_value)

        # ROI metrics
        gross_roi = rev_lift_usd_m / (deal_value_usd_m * deal_years) if deal_value_usd_m > 0 else 0.0
        # Payback period: when does cumulative revenue lift = cumulative cost?
        annual_lift_avg   = rev_lift_usd_m / deal_years
        payback           = (deal_value_usd_m / annual_lift_avg) if annual_lift_avg > 0 else float("inf")

        # Break-even CPM
        total_impressions_bn = (
            SPORTS_PROPERTIES[property_key].social_followers_m * 1_000_000 *
            SPORTS_PROPERTIES[property_key].avg_engagement_rate / 100.0 *
            1_200 * deal_years / 1e9
        )
        break_even_cpm = (deal_value_usd_m * 1e9) / max(total_impressions_bn * 1e9, 1)

        return ROIProjection(
            brand_key=brand_key,
            property_key=property_key,
            deal_value_usd_m=deal_value_usd_m,
            deal_years=deal_years,
            estimated_revenue_lift_usd_m=round(rev_lift_usd_m, 2),
            revenue_lift_pct=round(total_lift * fit * 100, 3),
            awareness_lift_pct=round(awareness_lift_pp, 2),
            consideration_lift_pct=round(consideration_lift_pp, 2),
            new_customers_acquired=new_customers,
            cost_per_acquisition_usd=round((deal_value_usd_m * deal_years * 1e6) / max(new_customers, 1), 2),
            gross_roi_multiple=round(gross_roi, 3),
            payback_period_years=round(min(payback, 99.0), 2),
            break_even_cpm_usd=round(break_even_cpm, 2),
        )
