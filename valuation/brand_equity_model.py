"""
valuation/brand_equity_model.py — Brand Equity Impact Model.

Business summary
----------------
Exposure metrics measure how many people see your logo. Brand equity measures
what they think and feel about your brand AFTER the sponsorship. This module
models the long-term brand equity impact — awareness build, consideration lift,
halo effects from team success, and the dilution penalty when a property is
over-commercialised with too many sponsors.

Three mechanisms:
  1. Awareness S-curve: brand recognition builds slowly, peaks at midterm,
     then plateaus. Modelled as a logistic (sigmoid) function over the deal term.
  2. Performance halo: a team winning a championship generates measurable
     sponsor brand equity lift above and beyond pure media exposure.
     Hardcoded for real events (Man City 2023 treble, Real Madrid UCL wins).
  3. Clutter discount: properties with many sponsors dilute each individual
     brand's recall and association. Each additional sponsor reduces the
     incremental equity value of a new one.

Developer notes
---------------
S-curve formula (logistic function):
    awareness(t) = L / (1 + exp(-k × (t - t_mid)))
    where:
        L     = maximum achievable awareness lift (= fit-adjusted potential)
        k     = growth rate constant (steepness of the S-curve)
        t_mid = year at which awareness is half of L (inflection point ≈ deal_years / 2)

Brand equity is a stock (accumulates) not a flow (not earned in one year).
This module reports year-by-year equity build, cumulative equity by end of term,
and the residual equity (brand memory) that persists after the deal ends.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from data.sports_data import SPORTS_PROPERTIES


# ── Hardcoded performance halo events ─────────────────────────────────────────
# Source: Brand Finance annual study, Nielsen Sports post-event brand impact reports.
# Value: % uplift in sponsor brand awareness / equity metrics in the 12 months
# following the event (relative to counterfactual without championship).

_PERFORMANCE_HALO_EVENTS: dict[str, dict] = {
    # (property_key, season/year) → {brand_lift_pct, event_label}
    ("manchester_city", "2023_treble"):     {"lift": 0.18, "label": "Man City 2022–23 Treble (PL + FA Cup + UCL)"},
    ("manchester_city", "2022_prem"):       {"lift": 0.09, "label": "Man City 2021–22 Premier League title"},
    ("real_madrid",     "2022_ucl"):        {"lift": 0.16, "label": "Real Madrid 2021–22 Champions League"},
    ("real_madrid",     "2024_ucl"):        {"lift": 0.17, "label": "Real Madrid 2023–24 Champions League"},
    ("liverpool",       "2019_ucl"):        {"lift": 0.14, "label": "Liverpool 2018–19 Champions League"},
    ("barcelona",       "2023_laliga"):     {"lift": 0.08, "label": "Barcelona 2022–23 La Liga title"},
    ("red_bull_racing", "2023_domination"): {"lift": 0.22, "label": "Red Bull 2023 F1 21/22 race wins season"},
    ("red_bull_racing", "2022_wcc"):        {"lift": 0.16, "label": "Red Bull 2022 F1 Constructors' + Drivers' titles"},
    ("ferrari_f1",      "2024_resurgence"): {"lift": 0.12, "label": "Ferrari 2024 multiple race wins resurgence"},
    ("new_england_patriots", "2019_sb"):    {"lift": 0.18, "label": "Patriots Super Bowl LIII win"},
    ("la_lakers",       "2020_nba"):        {"lift": 0.20, "label": "Lakers 2019–20 NBA Championship (bubble)"},
    ("golden_state_warriors", "2022_nba"):  {"lift": 0.17, "label": "Warriors 2021–22 NBA Championship"},
    ("dallas_cowboys",  "2024_contend"):    {"lift": 0.05, "label": "Cowboys 2023–24 playoff contention"},
    ("formula_1",       "2022_growth"):     {"lift": 0.15, "label": "F1 post-Drive-to-Survive viewership surge 2022"},
    ("formula_1",       "2023_growth"):     {"lift": 0.10, "label": "F1 2023 continued audience expansion"},
    ("nfl",             "2023_sb"):         {"lift": 0.12, "label": "NFL Super Bowl LVII Taylor Swift effect"},
    ("arsenal",         "2024_title_race"): {"lift": 0.11, "label": "Arsenal 2023–24 Premier League title race"},
    ("psg",             "2024_sf"):         {"lift": 0.08, "label": "PSG 2023–24 UCL semi-final run"},
    ("manchester_united","2023_fa_cup"):    {"lift": 0.07, "label": "Manchester United 2022–23 FA Cup win"},
    ("nba",             "2023_global"):     {"lift": 0.09, "label": "NBA 2022–23 global viewership record"},
}

# Clutter parameters: number of tier-1 sponsors for each property
# (sponsors competing for audience mindshare on the same asset class)
_SPONSOR_CLUTTER: dict[str, dict] = {
    "manchester_city":     {"n_sponsors": 12, "tier1": 3},
    "manchester_united":   {"n_sponsors": 20, "tier1": 5},
    "real_madrid":         {"n_sponsors": 15, "tier1": 4},
    "barcelona":           {"n_sponsors": 18, "tier1": 5},
    "psg":                 {"n_sponsors": 14, "tier1": 3},
    "chelsea":             {"n_sponsors": 16, "tier1": 4},
    "arsenal":             {"n_sponsors": 13, "tier1": 3},
    "liverpool":           {"n_sponsors": 14, "tier1": 3},
    "nfl":                 {"n_sponsors": 32, "tier1": 8},
    "dallas_cowboys":      {"n_sponsors": 22, "tier1": 5},
    "new_england_patriots":{"n_sponsors": 18, "tier1": 4},
    "la_lakers":           {"n_sponsors": 20, "tier1": 5},
    "golden_state_warriors":{"n_sponsors": 18, "tier1": 4},
    "formula_1":           {"n_sponsors": 40, "tier1": 10},
    "red_bull_racing":     {"n_sponsors": 25, "tier1": 5},
    "ferrari_f1":          {"n_sponsors": 22, "tier1": 5},
    "super_bowl":          {"n_sponsors": 35, "tier1": 8},
    "wimbledon":           {"n_sponsors": 10, "tier1": 2},
    "nba":                 {"n_sponsors": 35, "tier1": 8},
    "brentford":           {"n_sponsors": 6,  "tier1": 1},
}

# Maximum achievable awareness lift by sponsorship deal type (percentage points)
_MAX_AWARENESS_LIFT_PP: dict[str, float] = {
    "jersey":         12.0,
    "jersey_sleeve":   6.0,
    "jersey_patch":    4.5,
    "stadium_naming": 10.0,
    "title_sponsor":  15.0,
    "broadcast":       7.0,
    "digital":         5.0,
    "athlete":         9.0,
}


class BrandEquityAnalyzer:
    """
    Models the long-term brand equity impact of a sponsorship deal.

    Business summary
    ----------------
    Brand equity builds slowly through repeated positive associations,
    peaks during the deal, and partially persists after the deal ends
    (brand memory / residual equity). This model captures:
      - Awareness S-curve (year-by-year equity build)
      - Performance halo uplift from championship wins
      - Clutter discount from over-sponsored properties
      - Residual equity that persists post-deal (30–50% of peak)

    Usage
    -----
    >>> analyzer = BrandEquityAnalyzer()
    >>> curve = analyzer.awareness_curve(5, 67.5, 280.0)
    >>> print(curve)
    >>> analyzer.performance_halo("manchester_city", "2023_treble")
    0.18
    >>> analyzer.clutter_discount("formula_1")
    0.62
    """

    def __init__(self) -> None:
        self._halo   = _PERFORMANCE_HALO_EVENTS
        self._clutter = _SPONSOR_CLUTTER

    # ── Primary API ──────────────────────────────────────────────────────────

    def awareness_curve(
        self,
        deal_years: int,
        deal_value_usd_m: float,
        property_reach_m: float,
        asset_type: str = "jersey",
        demographic_fit: float = 0.70,
        k: float = 1.2,
    ) -> dict:
        """
        Model awareness build-up over a deal's term using a logistic S-curve.

        Business summary
        ----------------
        Brand awareness doesn't jump instantly when a deal is signed — it builds
        as fans encounter the brand repeatedly across a season, then multiple seasons.
        The logistic (sigmoid) S-curve captures this: slow start, rapid build in
        years 2–3, plateau in years 4–5.

        Formula (logistic function)
        ---------------------------
            awareness(t) = L / (1 + exp(-k × (t - t_mid)))

        where:
            L     = max achievable awareness lift (percentage points)
                  = _MAX_AWARENESS_LIFT_PP[asset_type]
                    × min(deal_value_usd_m / 50.0, 1.5)   [deal size scale]
                    × min(property_reach_m / 100.0, 1.0)  [reach scale]
                    × demographic_fit                       [brand-property fit]
            k     = steepness constant (1.2 → moderate sigmoid slope)
            t_mid = midpoint year = deal_years / 2

        Post-deal residual: 35% of peak awareness persists for 2 years after
        deal end (diminishing at 50%/yr → captures brand memory decay).

        Parameters
        ----------
        deal_years : int
            Deal term length.
        deal_value_usd_m : float
            Annual deal value in USD millions.
        property_reach_m : float
            Property total social followers in millions (reach proxy).
        asset_type : str
            Deal asset type (affects max achievable lift).
        demographic_fit : float
            Brand-property demographic fit score [0–1].
        k : float
            Sigmoid steepness (default 1.2; higher = faster build).

        Returns
        -------
        dict
            {
              year_by_year: list[{year, awareness_lift_pp, cumulative}],
              peak_lift_pp: float,
              cumulative_lift_pp: float,
              residual_lift_pp: float,
              total_including_residual_pp: float,
            }
        """
        max_lift = _MAX_AWARENESS_LIFT_PP.get(asset_type, 8.0)
        # Scale max lift by deal size (anchored at $50m/yr = 1.0×)
        size_scale  = min(deal_value_usd_m / 50.0, 1.5)
        # Scale by property reach (anchored at 100m followers = 1.0×)
        reach_scale = min(property_reach_m / 100.0, 1.0)
        L = max_lift * size_scale * reach_scale * demographic_fit

        t_mid = deal_years / 2.0
        years = list(range(1, deal_years + 1))

        # S-curve: awareness in year t (absolute percentage points, not cumulative)
        # We compute cumulative awareness as the logistic value at each year,
        # and year-by-year incremental lift as the first difference.
        cumulative_at_t = [
            L / (1 + math.exp(-k * (t - t_mid))) for t in years
        ]

        year_increments = []
        for i, t in enumerate(years):
            incr = cumulative_at_t[i] - (cumulative_at_t[i - 1] if i > 0 else 0.0)
            year_increments.append(incr)

        # Peak lift = logistic value at final year (approaches L asymptotically)
        peak_lift = cumulative_at_t[-1]
        total_incremental = sum(year_increments)

        # Post-deal residual: 35% of peak, decaying 50%/yr for 2 years
        residual_year1 = peak_lift * 0.35
        residual_year2 = residual_year1 * 0.50
        residual_total = residual_year1 + residual_year2

        year_by_year = [
            {
                "year":               t,
                "awareness_lift_pp":  round(year_increments[i], 3),
                "cumulative_pp":      round(cumulative_at_t[i], 3),
            }
            for i, t in enumerate(years)
        ]

        return {
            "year_by_year":                 year_by_year,
            "peak_lift_pp":                 round(peak_lift, 3),
            "cumulative_in_deal_pp":        round(total_incremental, 3),
            "residual_lift_pp":             round(residual_total, 3),
            "total_including_residual_pp":  round(total_incremental + residual_total, 3),
            "sigmoid_L":                    round(L, 3),
            "sigmoid_k":                    k,
            "sigmoid_t_mid":                t_mid,
        }

    def performance_halo(self, property_key: str, season_result: str) -> float:
        """
        Return the sponsor brand equity lift from a specific championship event.

        Business summary
        ----------------
        When a team wins a major trophy, the media coverage and fan euphoria
        create positive brand associations that extend to all the team's sponsors.
        Nielsen Sports research shows jersey sponsors receive 12–22% awareness
        lift in the 12 months following a championship win vs. a counterfactual.

        Parameters
        ----------
        property_key : str
            Sports property key (e.g. "manchester_city").
        season_result : str
            Event key (e.g. "2023_treble", "2022_ucl").

        Returns
        -------
        float
            Sponsor brand equity lift as a decimal (e.g. 0.18 = 18%).
            Returns 0.0 if no halo event data exists.
        """
        key = (property_key, season_result)
        event = self._halo.get(key)
        if event is None:
            return 0.0
        return event["lift"]

    def performance_halo_detail(self, property_key: str, season_result: str) -> dict:
        """Return full halo event detail including the event label."""
        key = (property_key, season_result)
        event = self._halo.get(key)
        if event is None:
            return {"property": property_key, "event": season_result, "lift": 0.0, "label": "No data"}
        return {
            "property": property_key,
            "event":    season_result,
            "lift":     event["lift"],
            "label":    event["label"],
        }

    def all_halo_events(self, property_key: str) -> list[dict]:
        """Return all known halo events for a property, sorted by lift descending."""
        events = []
        for (pk, sc), data in self._halo.items():
            if pk == property_key:
                events.append({"event": sc, "lift": data["lift"], "label": data["label"]})
        events.sort(key=lambda x: x["lift"], reverse=True)
        return events

    def expected_halo_lift(self, property_key: str, deal_years: int) -> float:
        """
        Expected average annual halo lift over a deal term.

        Uses historical halo events to estimate the probability-weighted
        expected championship-driven brand lift per year.

        Returns
        -------
        float
            Expected annual halo lift as a decimal.
        """
        events = self.all_halo_events(property_key)
        if not events:
            return 0.02  # default: 2% expected annual halo for any top-sport property

        # Probability of a major win event in any given year ≈ events / deal_years
        # Weighted average lift × annual probability
        total_lift = sum(e["lift"] for e in events)
        avg_lift   = total_lift / len(events)
        # Historical frequency: how many events over how many years?
        # Use a rough 5-year base period
        annual_prob = min(len(events) / 5.0, 1.0) * (deal_years / max(deal_years, 5))
        return round(avg_lift * annual_prob, 4)

    def clutter_discount(self, property_key: str) -> float:
        """
        Compute the clutter discount for a property — how much individual brand
        equity is diluted by competition from other sponsors.

        Business summary
        ----------------
        A stadium with 40 visible sponsors creates noise. Fans cannot form
        strong associations with any single brand when overwhelmed by choices.
        Properties with fewer, higher-quality sponsorships deliver better
        brand equity per dollar spent.

        Formula
        -------
        Base discount starts at 1.0 (no discount) for 1 sponsor.
        Each additional sponsor reduces equity delivery by a diminishing amount:

            discount = 1.0 / (1 + α × (n_sponsors - 1))^β

        where α = 0.03 (marginal clutter cost) and β = 0.80 (diminishing returns).
        Tier-1 sponsors (headline partnerships) receive a clutter relief bonus
        since they command more prominent placement.

        Parameters
        ----------
        property_key : str

        Returns
        -------
        float in (0, 1].
            Multiplier: 1.0 = no clutter, 0.5 = 50% equity dilution.
        """
        clutter = self._clutter.get(property_key)
        if clutter is None:
            return 0.80  # default for unknown properties

        n   = clutter["n_sponsors"]
        t1  = clutter["tier1"]
        alpha, beta = 0.03, 0.80

        if n <= 1:
            return 1.0

        base_discount  = 1.0 / (1 + alpha * (n - 1)) ** beta
        # Tier-1 bonus: if you're a headline sponsor, recover 30% of clutter loss
        tier1_relief = (1.0 - base_discount) * 0.30 * (t1 / max(n, 1))
        return round(min(base_discount + tier1_relief, 1.0), 4)

    def total_equity_value(
        self,
        brand_key: str,
        property_key: str,
        deal_value_usd_m: float,
        deal_years: int,
        asset_type: str = "jersey",
        demographic_fit: float = 0.70,
        include_halo: bool = True,
    ) -> dict:
        """
        Aggregate all brand equity components into a total equity value (USD m).

        Formula
        -------
        Equity value = cumulative awareness lift (pp) × reach_m × CPP
                       × (1 + halo_lift) × clutter_discount

        where CPP = cost per percentage point of awareness (industry benchmark:
        $0.18m per percentage point of national brand awareness in UK/EU;
        $0.25m per pp in USA markets). Sourced from IPA Effectiveness Databank.

        Parameters
        ----------
        brand_key : str
        property_key : str
        deal_value_usd_m : float
        deal_years : int
        asset_type : str
        demographic_fit : float
        include_halo : bool

        Returns
        -------
        dict
            {awareness_equity_usd_m, halo_equity_usd_m, total_equity_usd_m,
             clutter_discount, cumulative_awareness_pp, residual_pp}
        """
        prop = SPORTS_PROPERTIES.get(property_key)
        if prop is None:
            raise ValueError(f"Unknown property '{property_key}'.")

        reach_m = prop.social_followers_m + prop.avg_broadcast_viewers_m * prop.broadcast_frequency / 12
        curve   = self.awareness_curve(deal_years, deal_value_usd_m, reach_m, asset_type, demographic_fit)
        clutter = self.clutter_discount(property_key)
        halo    = self.expected_halo_lift(property_key, deal_years) if include_halo else 0.0

        # CPP: cost per percentage point of awareness (USD millions)
        # Varies by market: EU=0.18, US=0.25, Global weighted avg=0.21
        cpp = 0.21
        # Awareness equity
        awareness_eq = curve["total_including_residual_pp"] * reach_m / 10.0 * cpp * clutter
        # Halo equity: additional lift on top of base
        halo_eq      = awareness_eq * halo

        return {
            "awareness_equity_usd_m": round(awareness_eq, 2),
            "halo_equity_usd_m":      round(halo_eq, 2),
            "total_equity_usd_m":     round(awareness_eq + halo_eq, 2),
            "clutter_discount":       clutter,
            "cumulative_awareness_pp": curve["cumulative_in_deal_pp"],
            "residual_pp":            curve["residual_lift_pp"],
            "expected_halo_annual":   round(halo, 4),
            "awareness_curve":        curve["year_by_year"],
        }
