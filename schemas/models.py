"""
schemas/models.py — Pydantic domain models for the Sports Sponsorship Valuator.

Business summary
----------------
Every object that flows through the valuation pipeline has a schema defined here.
These schemas serve two purposes:

  1. Runtime validation — Pydantic raises clear errors if a caller passes nonsense
     values (e.g. a negative CPM, or an engagement rate > 100 %).
  2. Documentation contract — field descriptions are the canonical source of truth
     for what each value means, its units, and its expected range.

Developer notes
---------------
All monetary fields are in USD millions (usd_m) unless suffixed otherwise.
All fractions (rates, weights, percentages expressed as decimals) are in [0, 1].
All timestamps use ISO-8601 strings to stay JSON-serializable without extra deps.

Pydantic v2 is required (model_validator / field_validator syntax).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class AssetType(str, Enum):
    JERSEY              = "jersey"
    JERSEY_SLEEVE       = "jersey_sleeve"
    JERSEY_PATCH        = "jersey_patch"
    STADIUM_NAMING      = "stadium_naming"
    TITLE_SPONSOR       = "title_sponsor"
    BROADCAST           = "broadcast"
    DIGITAL             = "digital"
    ATHLETE             = "athlete"


class DealType(str, Enum):
    JERSEY        = "jersey"
    STADIUM_NAMING = "stadium_naming"
    TITLE_SPONSOR  = "title_sponsor"
    BROADCAST      = "broadcast"
    ATHLETE        = "athlete"
    JERSEY_PATCH   = "jersey_patch"


class Sport(str, Enum):
    FOOTBALL        = "football"
    AMERICAN_FOOTBALL = "american_football"
    BASKETBALL      = "basketball"
    MOTORSPORT      = "motorsport"
    TENNIS          = "tennis"
    GOLF            = "golf"
    RUGBY           = "rugby"


class Scenario(str, Enum):
    BEAR = "bear"
    BASE = "base"
    BULL = "bull"


# ── Core Input Schemas ────────────────────────────────────────────────────────

class SponsorshipAsset(BaseModel):
    """
    Business summary
    ----------------
    Represents a single sponsorable asset (the "thing" being sold):
    e.g. the front-of-jersey on Manchester City for 5 years.

    All exposure metrics are annual figures.
    """

    property_key: str = Field(
        ...,
        description="Internal property identifier matching data/sports_data.py keys.",
        examples=["manchester_city", "red_bull_racing"],
    )
    asset_type: AssetType = Field(
        ...,
        description="Category of sponsorship placement.",
    )
    annual_broadcast_viewers_m: float = Field(
        ...,
        ge=0,
        description="Average millions of viewers per broadcast event.",
    )
    broadcast_frequency: int = Field(
        ...,
        ge=0,
        description="Number of broadcast events per year.",
    )
    social_followers_m: float = Field(
        ...,
        ge=0,
        description="Total cross-platform social followers in millions.",
    )
    avg_engagement_rate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Average social engagement rate as a decimal (e.g. 0.032 = 3.2%).",
    )
    annual_attendance: int = Field(
        ...,
        ge=0,
        description="Total in-venue attendees per year.",
    )
    international_fanbase_pct: float = Field(
        ...,
        ge=0,
        le=1,
        description="Fraction of fanbase outside the home country.",
    )
    brand_value_usd_m: float = Field(
        ...,
        ge=0,
        description="Estimated brand/franchise value in USD millions (Deloitte / Forbes).",
    )
    sport: Sport
    league: str

    @field_validator("avg_engagement_rate")
    @classmethod
    def engagement_rate_sanity(cls, v: float) -> float:
        if v > 0.30:
            raise ValueError(
                f"avg_engagement_rate {v:.2%} exceeds 30 % — likely entered as percentage "
                f"rather than decimal. Divide by 100."
            )
        return v


class DealComparable(BaseModel):
    """
    Business summary
    ----------------
    A single historical sponsorship deal used as a market reference point.
    The comparable deals database powers both the median-implied value and
    the multiple linear regression valuation.
    """

    deal_id: str = Field(..., description="Unique identifier for audit traceability.")
    property_name: str
    brand_name: str
    deal_type: DealType
    annual_value_usd_m: float = Field(..., ge=0, description="Annual deal value in USD millions.")
    year_signed: int = Field(..., ge=1990, le=2025)
    deal_duration_years: int = Field(..., ge=1, le=30)
    is_exclusive: bool = Field(..., description="Category exclusivity granted to sponsor.")
    sport: Sport
    league: str
    notes: str = ""

    # Derived feature fields (populated by ComparableDealsAnalyzer)
    property_reach_score: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description="Composite reach score (0–10) computed from viewership + social.",
    )
    audience_quality_score: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description="Weighted demographic quality index for this property.",
    )
    deal_type_encoded: Optional[int] = Field(
        None,
        description="Integer encoding of deal_type for regression features.",
    )

    @field_validator("annual_value_usd_m")
    @classmethod
    def value_not_absurd(cls, v: float) -> float:
        if v > 2000:
            raise ValueError(
                f"annual_value_usd_m={v} exceeds $2 bn/yr — confirm this is correct."
            )
        return v


class ValuationResult(BaseModel):
    """
    Business summary
    ----------------
    The output of the SponsorshipFairValueEngine for a single brand–property pair.
    Contains the weighted fair value, the signal decomposition, and the NPV
    of the deal if the brand were to sign for `years` years.

    Developer notes
    ---------------
    weighted_fair_value = Σ(signal_i × weight_i) where weights sum to 1.0.
    See config.VALUATION_WEIGHTS for the current weight vector.
    """

    property_key: str
    brand_key: str
    asset_type: AssetType

    # Individual signal valuations (all in USD millions / year)
    exposure_value_usd_m: float = Field(..., ge=0, description="Bottom-up CPM exposure model value.")
    comp_implied_value_usd_m: float = Field(..., ge=0, description="Regression-implied value from comps.")
    audience_premium_usd_m: float = Field(..., description="Demographic fit adjustment (can be negative).")
    social_signal_usd_m: float = Field(..., ge=0, description="Social amplification signal value.")

    # Blended output
    weighted_fair_value_usd_m: float = Field(..., ge=0, description="Weighted blend of all signals.")
    exclusivity_premium_usd_m: float = Field(
        0.0, description="Add-on for category exclusivity, if applicable."
    )
    final_fair_value_usd_m: float = Field(
        ..., ge=0, description="weighted_fair_value + exclusivity_premium."
    )

    # Multi-year NPV
    deal_years: int = Field(..., ge=1)
    discount_rate: float = Field(..., ge=0, le=1)
    growth_rate: float = Field(..., ge=-0.5, le=0.5)
    npv_usd_m: float = Field(..., description="Present value of the multi-year deal.")

    # Scenario valuations
    bear_case_usd_m: float
    base_case_usd_m: float
    bull_case_usd_m: float

    @model_validator(mode="after")
    def final_value_equals_weighted_plus_exclusivity(self) -> "ValuationResult":
        expected = self.weighted_fair_value_usd_m + self.exclusivity_premium_usd_m
        if abs(self.final_fair_value_usd_m - expected) > 0.01:
            raise ValueError(
                f"final_fair_value_usd_m ({self.final_fair_value_usd_m:.2f}) must equal "
                f"weighted_fair_value_usd_m + exclusivity_premium_usd_m ({expected:.2f})."
            )
        return self


class ROIProjection(BaseModel):
    """
    Business summary
    ----------------
    Forward-looking projection of a sponsorship deal's financial return for
    the sponsoring brand. Translates media exposure value into revenue lift,
    brand equity gains, and customer acquisition estimates.

    These are model projections, not guarantees.
    """

    brand_key: str
    property_key: str
    deal_value_usd_m: float = Field(..., ge=0)
    deal_years: int = Field(..., ge=1)

    # Revenue impact
    estimated_revenue_lift_usd_m: float = Field(
        ..., description="Estimated cumulative incremental revenue over deal term."
    )
    revenue_lift_pct: float = Field(..., description="Revenue lift as % of brand annual revenue.")

    # Brand equity
    awareness_lift_pct: float = Field(
        ..., ge=0, description="Estimated brand awareness increase in target demo (percentage points)."
    )
    consideration_lift_pct: float = Field(
        ..., ge=0, description="Estimated brand consideration increase (percentage points)."
    )

    # Customer acquisition
    new_customers_acquired: int = Field(..., ge=0)
    cost_per_acquisition_usd: float = Field(..., ge=0)

    # ROI metrics
    gross_roi_multiple: float = Field(
        ..., description="(Revenue lift) / (Deal cost). >1.0 means positive return."
    )
    payback_period_years: float = Field(
        ..., ge=0, description="Years until cumulative revenue lift exceeds deal cost."
    )
    break_even_cpm_usd: float = Field(
        ..., ge=0, description="CPM required for the deal to break even on revenue."
    )


class MonteCarloResult(BaseModel):
    """
    Business summary
    ----------------
    Output of a Monte Carlo simulation run. The 10,000-trial simulation samples
    uncertainty in viewership, engagement, and team performance to produce a
    distribution of deal fair values rather than a single point estimate.

    The P10/P50/P90 range represents the 10th, 50th, and 90th percentile of
    simulated values — analogous to a confidence interval.
    """

    property_key: str
    brand_key: str
    asset_type: AssetType
    n_simulations: int = Field(..., ge=100)

    p10_usd_m: float = Field(..., description="10th percentile simulated fair value (downside scenario).")
    p25_usd_m: float = Field(..., description="25th percentile.")
    p50_usd_m: float = Field(..., description="Median simulated fair value (central estimate).")
    p75_usd_m: float = Field(..., description="75th percentile.")
    p90_usd_m: float = Field(..., description="90th percentile simulated fair value (upside scenario).")

    mean_usd_m: float = Field(..., description="Mean of simulation distribution.")
    std_usd_m: float = Field(..., description="Standard deviation of simulation distribution.")
    skewness: float = Field(..., description="Skewness of the distribution (+ = right tail).")
    var_95_usd_m: float = Field(
        ..., description="Value-at-Risk at 95% confidence: worst 5% of outcomes."
    )

    # Input uncertainty assumptions
    viewership_cv: float = Field(..., description="Coefficient of variation for viewership input.")
    engagement_cv: float = Field(..., description="Coefficient of variation for engagement input.")
    performance_cv: float = Field(..., description="Coefficient of variation for team performance input.")


class BacktestRecord(BaseModel):
    """
    Business summary
    ----------------
    A single entry in the backtesting report. Compares the model's implied fair
    value (using historical inputs from the year the deal was signed) against
    the actual deal value. Used to measure model accuracy over known outcomes.
    """

    deal_id: str
    property_name: str
    brand_name: str
    deal_type: DealType
    year_signed: int

    actual_annual_value_usd_m: float = Field(..., ge=0)
    model_implied_value_usd_m: float = Field(..., ge=0)
    absolute_error_usd_m: float = Field(..., ge=0)
    percentage_error: float = Field(..., description="(model - actual) / actual. Positive = overpriced.")
    verdict: str = Field(
        ...,
        description="'FAIR' (within ±15%), 'OVERPRICED' (model < actual - 15%), or 'UNDERPRICED' (model > actual + 15%).",
    )

    @model_validator(mode="after")
    def compute_absolute_error(self) -> "BacktestRecord":
        expected_abs = abs(self.model_implied_value_usd_m - self.actual_annual_value_usd_m)
        if abs(self.absolute_error_usd_m - expected_abs) > 0.001:
            raise ValueError(
                f"absolute_error_usd_m ({self.absolute_error_usd_m:.3f}) does not match "
                f"|model - actual| ({expected_abs:.3f})."
            )
        return self


class SensitivityTable(BaseModel):
    """
    Business summary
    ----------------
    Shows how a deal's NPV changes across a grid of (discount_rate, growth_rate) pairs.
    Useful for understanding how sensitive the valuation is to macroeconomic assumptions.
    """

    property_key: str
    brand_key: str
    annual_base_value_usd_m: float
    deal_years: int

    # Axes
    discount_rates: list[float] = Field(..., description="List of discount rates tested (e.g. [0.05, ..., 0.15]).")
    growth_rates: list[float] = Field(..., description="List of growth rates tested.")

    # Results matrix: npv_matrix[i][j] = NPV at discount_rates[i], growth_rates[j]
    npv_matrix: list[list[float]] = Field(
        ...,
        description="2-D list of NPV values in USD millions. Rows = discount rates, cols = growth rates.",
    )


class RegressionModel(BaseModel):
    """
    Developer notes
    ---------------
    Stores the fitted OLS regression coefficients and diagnostics.
    β = (XᵀX)⁻¹Xᵀy — computed in numpy matrix form.

    Features in order:
      [0] property_reach_score  (0–10)
      [1] audience_quality_score (0–10)
      [2] is_exclusive           (0 or 1)
      [3] deal_type_encoded      (integer)
      [4] intercept              (bias term, always 1 in design matrix)
    """

    coefficients: list[float] = Field(
        ...,
        description="OLS coefficients [β_reach, β_audience, β_exclusive, β_deal_type, β_intercept].",
    )
    feature_names: list[str]
    r_squared: float = Field(..., ge=0, le=1)
    adj_r_squared: float
    n_observations: int = Field(..., ge=1)
    rmse_usd_m: float = Field(..., ge=0, description="Root mean squared error in USD millions.")
    training_year_range: tuple[int, int] = Field(
        ..., description="(min_year, max_year) of training data."
    )
