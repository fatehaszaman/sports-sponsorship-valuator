"""
valuation/comparable_deals.py — Comparable Deals Analyzer with OLS Regression.

Business summary
----------------
The "comps" approach asks: what did similar deals actually trade for? By
searching our database of 47 real historical deals and finding the most
similar ones, we derive a market-implied fair value — the same methodology
used in M&A by investment banks to sanity-check DCF models.

This module goes further than simple median comparables: it implements a
multiple linear regression (OLS) fitted on the deal database so we can
predict fair value for any combination of input features, including
property reach, audience quality, exclusivity, and deal type.

Regression methodology
-----------------------
We solve the classic OLS normal equations in matrix form:

    β = (XᵀX)⁻¹ Xᵀy

where:
    X ∈ ℝ^(n × k)  design matrix (n deals, k features + intercept column)
    y ∈ ℝ^n        target vector (annual_value_usd_m)
    β ∈ ℝ^k        coefficient vector

Features:
  [0] property_reach_score  — composite of viewership + social followers (0–10)
  [1] audience_quality_score — AQI from audience_model (0–10)
  [2] is_exclusive           — 1 if category-exclusive deal, else 0
  [3] deal_type_encoded      — jersey=1, stadium_naming=2, title_sponsor=3,
                               broadcast=4, athlete=0.5, jersey_patch=0.8
  [4] 1                      — intercept (bias term)

This is implemented with raw numpy linear algebra — no sklearn dependency —
to make the math explicit and auditable.

Developer notes
---------------
- numpy.linalg.lstsq is used instead of explicit inversion for numerical
  stability when XᵀX is near-singular (pseudoinverse / Moore-Penrose).
- R² and adjusted R² are computed manually to show the formula.
- Training set excludes athlete deals by default (different value drivers).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from data.sports_data import HISTORICAL_DEALS, SponsorshipDeal
from schemas.models import DealComparable, RegressionModel


# ── Deal type encoding (for regression feature) ────────────────────────────────
_DEAL_TYPE_ENCODING: dict[str, float] = {
    "athlete":        0.5,
    "jersey_patch":   0.8,
    "jersey":         1.0,
    "jersey_sleeve":  0.9,
    "broadcast":      1.2,
    "stadium_naming": 2.0,
    "title_sponsor":  1.8,
}

# ── Property reach scores: composite of broadcast viewership + social presence ──
# Manually calibrated 0–10 scale; could be derived from ExposureValueModel.
_PROPERTY_REACH_SCORES: dict[str, float] = {
    "Manchester City FC":         7.2,
    "Manchester United FC":       8.8,
    "Real Madrid CF":             9.5,
    "FC Barcelona":               9.2,
    "Paris Saint-Germain FC":     7.8,
    "Chelsea FC":                 6.8,
    "Arsenal FC":                 6.5,
    "Liverpool FC":               8.2,
    "NFL (League)":               9.8,
    "Dallas Cowboys":             8.5,
    "New England Patriots":       7.9,
    "Los Angeles Lakers":         8.8,
    "Golden State Warriors":      8.4,
    "Formula 1 (Series)":         9.6,
    "Red Bull Racing (F1)":       8.9,
    "Scuderia Ferrari (F1)":      8.5,
    "Super Bowl (Event)":         10.0,
    "Wimbledon Championships":    6.2,
    "NBA (League)":               9.0,
    "Brentford FC":               3.5,
    # Athlete reach scores
    "Cristiano Ronaldo":          9.9,
    "Lionel Messi":               9.8,
    "LeBron James":               9.5,
    "Naomi Osaka":                7.5,
    "Max Verstappen":             7.8,
    "Serena Williams":            9.0,
    "Neymar Jr.":                 9.2,
    "Lewis Hamilton":             8.8,
    "Stephen Curry":              8.5,
    "Kylian Mbappé":              9.0,
}

# Audience quality scores (from AudienceDemographicModel, precomputed)
_PROPERTY_AQI: dict[str, float] = {
    "Manchester City FC":         6.80,
    "Manchester United FC":       7.10,
    "Real Madrid CF":             7.50,
    "FC Barcelona":               7.30,
    "Paris Saint-Germain FC":     7.20,
    "Chelsea FC":                 6.60,
    "Arsenal FC":                 6.70,
    "Liverpool FC":               6.90,
    "NFL (League)":               7.40,
    "Dallas Cowboys":             7.80,
    "New England Patriots":       8.10,
    "Los Angeles Lakers":         7.60,
    "Golden State Warriors":      8.00,
    "Formula 1 (Series)":         8.60,
    "Red Bull Racing (F1)":       8.20,
    "Scuderia Ferrari (F1)":      9.10,
    "Super Bowl (Event)":         8.30,
    "Wimbledon Championships":    8.80,
    "NBA (League)":               7.50,
    "Brentford FC":               5.20,
    "Cristiano Ronaldo":          7.80,
    "Lionel Messi":               7.90,
    "LeBron James":               7.70,
    "Naomi Osaka":                7.20,
    "Max Verstappen":             8.00,
    "Serena Williams":            7.80,
    "Neymar Jr.":                 7.40,
    "Lewis Hamilton":             8.10,
    "Stephen Curry":              7.80,
    "Kylian Mbappé":              7.90,
}


def _deal_to_features(deal: SponsorshipDeal) -> Optional[np.ndarray]:
    """
    Convert a raw SponsorshipDeal to a feature vector for regression.

    Returns None if the deal lacks enough information for feature construction.
    Feature order: [reach, aqi, is_exclusive, deal_type_enc, 1.0 (intercept)]
    """
    reach = _PROPERTY_REACH_SCORES.get(deal.property_name)
    aqi   = _PROPERTY_AQI.get(deal.property_name)
    if reach is None or aqi is None:
        return None
    enc = _DEAL_TYPE_ENCODING.get(deal.deal_type, 1.0)
    return np.array([reach, aqi, float(deal.is_exclusive), enc, 1.0])


class ComparableDealsAnalyzer:
    """
    Market-comparables analyzer for sports sponsorship deals.

    Business summary
    ----------------
    Two valuation signals are produced:

      1. Median comps: find the N most-similar historical deals and report
         the median annual value. Simple, intuitive, transparent.

      2. OLS regression: fit a linear model on all deal data and use it to
         predict value for any input feature combination. More precise, but
         requires understanding the regression coefficients.

    The regression is implemented in explicit matrix algebra form:

        β = (XᵀX)⁻¹Xᵀy

    making the math visible and auditable, with no hidden sklearn transforms.

    Developer notes
    ---------------
    - `fit()` must be called before `regression_implied_value()`.
    - The fitted model is stored as `self.model_` (a RegressionModel schema).
    - Athlete deals are excluded from regression training by default since
      athlete endorsement value drivers differ materially from property deals.

    Usage
    -----
    >>> analyzer = ComparableDealsAnalyzer()
    >>> analyzer.fit()
    >>> analyzer.find_comps("Manchester City FC", "jersey")
    [{'deal_id': 'PL004', 'property': 'Manchester United FC', ...}, ...]
    >>> analyzer.implied_value_from_comps("Manchester City FC", "jersey")
    52.5
    >>> analyzer.regression_implied_value(reach=7.2, aqi=6.8, is_exclusive=False, deal_type="jersey")
    48.3
    """

    def __init__(self, exclude_athlete_deals: bool = True) -> None:
        self._deals = HISTORICAL_DEALS
        self._exclude_athletes = exclude_athlete_deals
        self.model_: Optional[RegressionModel] = None
        self._beta: Optional[np.ndarray] = None

    # ── Data Prep ───────────────────────────────────────────────────────────

    def _training_deals(self) -> list[SponsorshipDeal]:
        """Return deals eligible for regression training."""
        deals = self._deals
        if self._exclude_athletes:
            deals = [d for d in deals if d.deal_type != "athlete"]
        return deals

    def as_dataframe(self) -> pd.DataFrame:
        """Return all deals as a tidy pandas DataFrame."""
        rows = []
        for d in self._deals:
            rows.append({
                "deal_id":             d.deal_id,
                "property_name":       d.property_name,
                "brand_name":          d.brand_name,
                "deal_type":           d.deal_type,
                "annual_value_usd_m":  d.annual_value_usd_m,
                "year_signed":         d.year_signed,
                "deal_duration_years": d.deal_duration_years,
                "is_exclusive":        d.is_exclusive,
                "sport":               d.sport,
                "league":              d.league,
                "reach_score":         _PROPERTY_REACH_SCORES.get(d.property_name),
                "aqi":                 _PROPERTY_AQI.get(d.property_name),
                "deal_type_enc":       _DEAL_TYPE_ENCODING.get(d.deal_type, 1.0),
            })
        return pd.DataFrame(rows)

    # ── OLS Regression Training ─────────────────────────────────────────────

    def fit(self, exclude_athlete_deals: Optional[bool] = None) -> "RegressionModel":
        """
        Fit OLS regression on the deal database.

        Explicit matrix algebra implementation:

            Design matrix  X ∈ ℝ^(n × 5)
            Target vector  y ∈ ℝ^n   (annual_value_usd_m)

            Normal equations:  β = (XᵀX)⁻¹ Xᵀy

        Solved via numpy.linalg.lstsq (numerically stable pseudoinverse)
        instead of explicit matrix inversion to handle near-singular cases.

        Parameters
        ----------
        exclude_athlete_deals : bool, optional
            Override instance setting.

        Returns
        -------
        RegressionModel
            Schema object with coefficients, R², RMSE.
        """
        if exclude_athlete_deals is not None:
            self._exclude_athletes = exclude_athlete_deals

        training = self._training_deals()

        # Build design matrix X and target y
        rows_X, rows_y = [], []
        for deal in training:
            feat = _deal_to_features(deal)
            if feat is None:
                continue
            rows_X.append(feat)
            rows_y.append(deal.annual_value_usd_m)

        X = np.array(rows_X, dtype=float)   # shape (n, 5)
        y = np.array(rows_y, dtype=float)   # shape (n,)
        n, k = X.shape

        # ── OLS Normal Equations: β = (XᵀX)⁻¹Xᵀy ──────────────────────────
        # numpy.linalg.lstsq uses SVD-based pseudoinverse for stability.
        # Equivalent to: beta = np.linalg.inv(X.T @ X) @ X.T @ y
        # but numerically safer when XᵀX is ill-conditioned.
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        self._beta = beta

        # ── Goodness-of-fit metrics ──────────────────────────────────────────
        y_hat      = X @ beta                          # fitted values
        ss_res     = float(np.sum((y - y_hat) ** 2))  # residual sum of squares
        ss_tot     = float(np.sum((y - y.mean()) ** 2))
        r_squared  = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj_r2     = 1.0 - (1 - r_squared) * (n - 1) / (n - k - 1)
        rmse       = math.sqrt(ss_res / n)

        year_min = min(d.year_signed for d in training)
        year_max = max(d.year_signed for d in training)

        self.model_ = RegressionModel(
            coefficients=beta.tolist(),
            feature_names=["reach_score", "audience_quality", "is_exclusive", "deal_type_enc", "intercept"],
            r_squared=round(r_squared, 4),
            adj_r_squared=round(adj_r2, 4),
            n_observations=n,
            rmse_usd_m=round(rmse, 2),
            training_year_range=(year_min, year_max),
        )
        return self.model_

    def regression_implied_value(
        self,
        reach: float,
        aqi: float,
        is_exclusive: bool,
        deal_type: str,
    ) -> float:
        """
        Predict annual deal value (USD m) from the fitted OLS model.

        Feature vector: [reach, aqi, is_exclusive, deal_type_enc, 1.0]
        Prediction:     ŷ = Xβ  (dot product with fitted coefficients)

        Parameters
        ----------
        reach : float
            Property reach score (0–10).
        aqi : float
            Audience quality index (0–10).
        is_exclusive : bool
            Category exclusivity flag.
        deal_type : str
            One of the DEAL_TYPE_ENCODING keys.

        Returns
        -------
        float
            Predicted annual value in USD millions.

        Raises
        ------
        RuntimeError
            If fit() has not been called first.
        """
        if self._beta is None:
            raise RuntimeError("Model not fitted. Call fit() before regression_implied_value().")
        enc = _DEAL_TYPE_ENCODING.get(deal_type, 1.0)
        x   = np.array([reach, aqi, float(is_exclusive), enc, 1.0])
        pred = float(np.dot(x, self._beta))
        return round(max(pred, 0.0), 2)

    def regression_for_property(
        self, property_name: str, deal_type: str = "jersey", is_exclusive: bool = False
    ) -> float:
        """Convenience wrapper: predict value for a named property."""
        if self._beta is None:
            self.fit()
        reach = _PROPERTY_REACH_SCORES.get(property_name, 5.0)
        aqi   = _PROPERTY_AQI.get(property_name, 6.0)
        return self.regression_implied_value(reach, aqi, is_exclusive, deal_type)

    def print_regression_summary(self) -> None:
        """Print a readable regression summary table."""
        if self.model_ is None:
            print("Model not fitted. Call fit() first.")
            return
        m = self.model_
        print("\n═══ OLS Regression Summary: β = (XᵀX)⁻¹Xᵀy ═══")
        print(f"  Observations:  {m.n_observations}")
        print(f"  R²:            {m.r_squared:.4f}")
        print(f"  Adj. R²:       {m.adj_r_squared:.4f}")
        print(f"  RMSE:          ${m.rmse_usd_m:.1f}m")
        print(f"  Training data: {m.training_year_range[0]}–{m.training_year_range[1]}")
        print("\n  Coefficients:")
        for name, coef in zip(m.feature_names, m.coefficients):
            print(f"    {name:<25} {coef:+.4f}")
        print()

    # ── Comparable Deal Search ──────────────────────────────────────────────

    def find_comps(
        self,
        property_name: str,
        asset_type: str,
        n: int = 5,
        year_floor: int = 2010,
    ) -> list[dict]:
        """
        Find the N most-comparable historical deals for a given property + asset type.

        Similarity scoring
        ------------------
        Each historical deal is scored against the query on:
          - Same sport (+3 pts)
          - Same asset/deal type (+2 pts)
          - Similar reach score (1 / (1 + |reach_diff|) * 2 pts)
          - Similar AQI (1 / (1 + |aqi_diff|) * 1.5 pts)
          - Recent deal (year ≥ year_floor, +0.5 pts)

        Parameters
        ----------
        property_name : str
            Display name matching the deals database (e.g. "Manchester City FC").
        asset_type : str
            Deal type to match (jersey, stadium_naming, etc.).
        n : int
            Number of comps to return.
        year_floor : int
            Minimum year for a deal to be considered relevant.

        Returns
        -------
        list[dict]
            Sorted by similarity score descending.
        """
        target_reach = _PROPERTY_REACH_SCORES.get(property_name, 5.0)
        target_aqi   = _PROPERTY_AQI.get(property_name, 6.0)

        # Infer sport from property name
        from data.sports_data import SPORTS_PROPERTIES
        sport_lookup = {v.name: v.sport for v in SPORTS_PROPERTIES.values()}
        target_sport = sport_lookup.get(property_name, "")

        scored = []
        for deal in self._deals:
            if deal.property_name == property_name:
                continue  # exclude self-comparables
            if deal.year_signed < year_floor:
                continue

            score = 0.0
            if deal.sport == target_sport:
                score += 3.0
            if deal.deal_type == asset_type:
                score += 2.0

            d_reach = _PROPERTY_REACH_SCORES.get(deal.property_name, 5.0)
            d_aqi   = _PROPERTY_AQI.get(deal.property_name, 6.0)
            score  += 2.0 / (1.0 + abs(d_reach - target_reach))
            score  += 1.5 / (1.0 + abs(d_aqi - target_aqi))

            if deal.year_signed >= year_floor:
                score += 0.5

            scored.append((score, deal))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:n]

        result = []
        for sim_score, deal in top:
            result.append({
                "deal_id":             deal.deal_id,
                "property":            deal.property_name,
                "brand":               deal.brand_name,
                "deal_type":           deal.deal_type,
                "annual_value_usd_m":  deal.annual_value_usd_m,
                "year_signed":         deal.year_signed,
                "duration_years":      deal.deal_duration_years,
                "is_exclusive":        deal.is_exclusive,
                "sport":               deal.sport,
                "similarity_score":    round(sim_score, 3),
                "notes":               deal.notes,
            })
        return result

    def implied_value_from_comps(
        self,
        property_name: str,
        asset_type: str,
        n: int = 5,
    ) -> float:
        """
        Median-implied annual value from the N most similar historical deals.

        Returns
        -------
        float
            Median annual value in USD millions.
        """
        comps = self.find_comps(property_name, asset_type, n=n)
        if not comps:
            return 0.0
        values = [c["annual_value_usd_m"] for c in comps]
        return round(float(np.median(values)), 2)

    def deal_premium_discount(
        self,
        proposed_value_usd_m: float,
        property_name: str,
        asset_type: str,
        n: int = 5,
    ) -> dict:
        """
        Classify a proposed deal value vs. comparable market deals.

        Returns
        -------
        dict
            {verdict, proposed, comp_median, pct_vs_comps}
            verdict: "FAIR" (within ±10%), "+X% PREMIUM", or "-X% DISCOUNT"
        """
        comps  = self.find_comps(property_name, asset_type, n=n)
        median = self.implied_value_from_comps(property_name, asset_type, n=n)

        if median == 0:
            return {"verdict": "INSUFFICIENT_DATA", "proposed": proposed_value_usd_m,
                    "comp_median": 0.0, "pct_vs_comps": None, "comps": comps}

        pct = (proposed_value_usd_m - median) / median * 100
        if abs(pct) <= 10:
            verdict = "FAIR"
        elif pct > 0:
            verdict = f"+{pct:.1f}% PREMIUM"
        else:
            verdict = f"{pct:.1f}% DISCOUNT"

        return {
            "verdict":       verdict,
            "proposed":      proposed_value_usd_m,
            "comp_median":   median,
            "pct_vs_comps":  round(pct, 1),
            "comps":         comps,
        }

    def deals_in_range(
        self, min_value: float, max_value: float, deal_type: Optional[str] = None
    ) -> list[dict]:
        """Return deals with annual value between min_value and max_value (USD m)."""
        results = []
        for d in self._deals:
            if d.annual_value_usd_m < min_value or d.annual_value_usd_m > max_value:
                continue
            if deal_type and d.deal_type != deal_type:
                continue
            results.append({
                "deal_id":  d.deal_id,
                "property": d.property_name,
                "brand":    d.brand_name,
                "type":     d.deal_type,
                "value":    d.annual_value_usd_m,
                "year":     d.year_signed,
            })
        results.sort(key=lambda x: x["value"], reverse=True)
        return results

    # ── Value Tier Classification ───────────────────────────────────────────

    @staticmethod
    def value_tier(annual_value_usd_m: float) -> str:
        """
        Classify a deal into a value tier.

        Tiers
        -----
        TIER_1  < $10m/yr
        TIER_2  $10–30m/yr
        TIER_3  $30–60m/yr
        TIER_4  $60–100m/yr
        TIER_5  $100m+/yr
        """
        if annual_value_usd_m < 10:
            return "TIER_1 (<$10m)"
        elif annual_value_usd_m < 30:
            return "TIER_2 ($10–30m)"
        elif annual_value_usd_m < 60:
            return "TIER_3 ($30–60m)"
        elif annual_value_usd_m < 100:
            return "TIER_4 ($60–100m)"
        else:
            return "TIER_5 ($100m+)"

    @staticmethod
    def duration_tier(years: int) -> str:
        """Classify deal duration: SHORT (1-2yr), MEDIUM (3-5yr), LONG (6+yr)."""
        if years <= 2:
            return "SHORT (1–2yr)"
        elif years <= 5:
            return "MEDIUM (3–5yr)"
        else:
            return "LONG (6+yr)"
