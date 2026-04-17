"""
valuation/monte_carlo_valuation.py — Monte Carlo Sponsorship Valuation.

Business summary
----------------
A single point estimate of deal fair value is misleading: it obscures the
range of outcomes driven by uncertainty in viewership trends, engagement rates,
and team performance. This module runs 10,000 simulation trials, drawing
key inputs from probability distributions calibrated to historical volatility,
and returns the full distribution of simulated fair values.

The P10/P50/P90 output is the sponsorship equivalent of a confidence interval:
  - P10: only 10% of simulated outcomes are worse than this — downside scenario
  - P50: median expectation — the central estimate
  - P90: only 10% of simulated outcomes are better than this — upside scenario

Monte Carlo methodology
-----------------------
For each trial t ∈ {1, …, N_SIMULATIONS}:

  1. Sample viewership multiplier:   m_v ~ LogNormal(μ=0, σ=viewership_cv)
  2. Sample engagement multiplier:   m_e ~ LogNormal(μ=0, σ=engagement_cv)
  3. Sample performance multiplier:  m_p ~ LogNormal(μ=0, σ=performance_cv)
     (team performance affects broadcast prominence; μ=0 → geometric mean = 1)

  4. Compute trial exposure value:
       exp_t = base_exposure × m_v × m_e × m_p

  5. Compute trial comp value:
       comp_t = base_comp × m_c   where m_c ~ LogNormal(μ=0, σ=0.15)

  6. Blend:
       value_t = w_exp × exp_t + w_comp × comp_t + w_aud × aud_adj

  Vectorized: all N trials computed as numpy array operations (no Python loop).

Why LogNormal?
--------------
Viewership and engagement cannot go below zero and historically exhibit
right-skewed distributions (occasional viral events create outlier highs).
LogNormal is the industry standard for modelling multiplicative risk factors.
If X ~ LogNormal(0, σ), then E[X] = exp(σ²/2) ≈ 1 for small σ — meaning
the distribution is centred near the base value, with σ controlling spread.

Developer notes
---------------
- All N simulations are run as element-wise numpy array operations — zero
  Python-level loops over trials.
- numpy.random.default_rng(seed) is used for reproducible simulations.
- VaR is computed as the 5th percentile of the loss distribution, consistent
  with standard financial risk reporting (95% confidence level).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from config import VALUATION_WEIGHTS
from data.sports_data import SPORTS_PROPERTIES
from schemas.models import MonteCarloResult
from valuation.audience_model import AudienceDemographicModel
from valuation.comparable_deals import ComparableDealsAnalyzer
from valuation.exposure_model import ExposureValueModel


# ── Default simulation parameters ────────────────────────────────────────────

N_SIMULATIONS: int = 10_000
DEFAULT_SEED:  int = 42

# Coefficients of variation for each uncertain input
# CV = std / mean; represents relative uncertainty.
# Calibrated from Nielsen Sports historical viewership variance data.
DEFAULT_VIEWERSHIP_CV:   float = 0.22   # ±22% viewership volatility
DEFAULT_ENGAGEMENT_CV:   float = 0.28   # ±28% engagement rate volatility
DEFAULT_PERFORMANCE_CV:  float = 0.18   # ±18% team performance effect


class MonteCarloValuator:
    """
    10,000-trial Monte Carlo simulation for sponsorship deal fair value.

    Business summary
    ----------------
    Instead of a single point estimate, this produces a distribution of
    possible deal fair values. The sponsor can then make an informed decision:
      - If P10 > proposed deal cost → low-risk; proceed.
      - If P50 ≈ proposed deal cost → fairly priced; negotiate standard terms.
      - If P90 < proposed deal cost → high-risk overpay; renegotiate or pass.

    Developer notes
    ---------------
    All N_SIMULATIONS trials are computed as vectorized numpy operations:
      - Multiplicative noise drawn from LogNormal distributions.
      - Blending weights applied as scalar multiplies on length-N arrays.
      - Statistics (percentiles, mean, std, skewness) computed with numpy.

    Usage
    -----
    >>> mc = MonteCarloValuator()
    >>> result = mc.simulate("emirates", "manchester_city", "jersey")
    >>> print(f"P50: ${result.p50_usd_m:.1f}m  P10: ${result.p10_usd_m:.1f}m  P90: ${result.p90_usd_m:.1f}m")
    >>> mc.print_distribution_summary(result)
    """

    def __init__(
        self,
        n_simulations: int = N_SIMULATIONS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._n       = n_simulations
        self._rng     = np.random.default_rng(seed)
        self._exp     = ExposureValueModel()
        self._aud     = AudienceDemographicModel()
        self._comps   = ComparableDealsAnalyzer()
        self._comps.fit()
        self._weights = VALUATION_WEIGHTS

    # ── Primary API ─────────────────────────────────────────────────────────

    def simulate(
        self,
        brand_key: str,
        property_key: str,
        asset_type: str = "jersey",
        is_exclusive: bool = False,
        viewership_cv: float = DEFAULT_VIEWERSHIP_CV,
        engagement_cv: float = DEFAULT_ENGAGEMENT_CV,
        performance_cv: float = DEFAULT_PERFORMANCE_CV,
    ) -> MonteCarloResult:
        """
        Run N_SIMULATIONS Monte Carlo trials and return the value distribution.

        Simulation algorithm (fully vectorized)
        ----------------------------------------
        Step 1 — Deterministic base values (scalars):
            base_exp  = ExposureValueModel.total_exposure_value(...)  [USD m]
            base_comp = ComparableDealsAnalyzer.regression_for_property(...)
            aud_adj   = base_exp × (audience_multiplier - 1.0)

        Step 2 — Sample N multiplicative noise vectors from LogNormal:
            LogNormal(μ, σ) parameterised so that geometric mean = 1:
            μ = -σ²/2  (ensures E[X] ≈ 1 for all σ)

            m_viewership  ~ LogNormal(-σ_v²/2, σ_v)   shape (N,)
            m_engagement  ~ LogNormal(-σ_e²/2, σ_e)   shape (N,)
            m_performance ~ LogNormal(-σ_p²/2, σ_p)   shape (N,)
            m_comps       ~ LogNormal(-0.15²/2, 0.15)  shape (N,)

        Step 3 — Compute trial exposure values (element-wise):
            exp_trials  = base_exp  × m_viewership × m_engagement × m_performance

        Step 4 — Compute trial comp values:
            comp_trials = base_comp × m_comps

        Step 5 — Blend using VALUATION_WEIGHTS:
            aud_trials  = (base_exp + aud_adj) × m_viewership  [audience-adjusted]
            soc_trials  = social_usd_m         × m_engagement

            value_trials = (
                w_exp  × exp_trials  +
                w_comp × comp_trials +
                w_aud  × aud_trials  +
                w_soc  × soc_trials
            )

        Step 6 — Apply exclusivity premium (scalar multiply if is_exclusive):
            value_trials += excl_prem × value_trials

        Step 7 — Compute distribution statistics on value_trials vector.

        Parameters
        ----------
        brand_key : str
        property_key : str
        asset_type : str
        is_exclusive : bool
        viewership_cv : float
            Coefficient of variation for viewership noise (e.g. 0.22 = 22%).
        engagement_cv : float
            Coefficient of variation for engagement noise.
        performance_cv : float
            Coefficient of variation for team-performance noise.

        Returns
        -------
        MonteCarloResult
            Full distribution statistics, P10/P25/P50/P75/P90, VaR.
        """
        from data.sports_data import BRAND_PROFILES
        from valuation.fair_value_engine import SponsorshipFairValueEngine

        brand    = BRAND_PROFILES[brand_key]
        property_ = SPORTS_PROPERTIES[property_key]

        # ── Step 1: Deterministic base values ──────────────────────────────
        base_exp_usd_m   = self._exp.total_exposure_value(property_key, asset_type) / 1_000_000
        base_comp_usd_m  = self._comps.regression_for_property(
            property_.name, deal_type=asset_type, is_exclusive=is_exclusive
        )
        fit_mult         = self._aud.premium_multiplier(brand_key, property_key)
        aud_adj_usd_m    = base_exp_usd_m * (fit_mult - 1.0)

        prop = SPORTS_PROPERTIES[property_key]
        social_usd_m = prop.social_followers_m * prop.avg_engagement_rate / 100.0 * 2.5

        # Exclusivity premium fraction
        excl_prem = 0.0
        if is_exclusive:
            from config import EXCLUSIVITY_PREMIUMS
            excl_prem = EXCLUSIVITY_PREMIUMS.get(brand.category, 0.15)

        # ── Step 2: Sample noise vectors (vectorized, shape N) ──────────────
        # LogNormal with μ = -σ²/2 ensures geometric mean = 1 (unbiased)
        def _lognormal(cv: float) -> np.ndarray:
            """Sample N LogNormal multipliers with mean ≈ 1 and CV = cv."""
            sigma = cv
            mu    = -0.5 * sigma ** 2
            return self._rng.lognormal(mean=mu, sigma=sigma, size=self._n)

        m_viewership  = _lognormal(viewership_cv)    # shape (N,)
        m_engagement  = _lognormal(engagement_cv)    # shape (N,)
        m_performance = _lognormal(performance_cv)   # shape (N,)
        m_comps       = _lognormal(0.15)             # comps market noise

        # ── Step 3 & 4: Trial values (element-wise array ops, no loops) ─────
        exp_trials  = base_exp_usd_m  * m_viewership * m_engagement * m_performance
        comp_trials = base_comp_usd_m * m_comps

        # Audience adjustment mirrors viewership noise (correlated: if viewership
        # is up, the audience quality premium scales proportionally)
        aud_trials  = (base_exp_usd_m + aud_adj_usd_m) * m_viewership
        soc_trials  = social_usd_m * m_engagement

        # ── Step 5: Blend ───────────────────────────────────────────────────
        w = self._weights
        value_trials = (
            w["exposure_value"]   * exp_trials  +
            w["comp_implied"]     * comp_trials +
            w["audience_premium"] * aud_trials  +
            w["social_signal"]    * soc_trials
        )

        # ── Step 6: Exclusivity premium ─────────────────────────────────────
        if excl_prem > 0:
            value_trials = value_trials * (1.0 + excl_prem)

        # Clip at zero (deal value cannot be negative)
        value_trials = np.maximum(value_trials, 0.0)

        # ── Step 7: Distribution statistics ─────────────────────────────────
        percentiles = np.percentile(value_trials, [10, 25, 50, 75, 90])
        mean_val    = float(value_trials.mean())
        std_val     = float(value_trials.std())

        # Skewness: E[(X - μ)³] / σ³  (Pearson's moment coefficient)
        skewness = float(
            np.mean((value_trials - mean_val) ** 3) / (std_val ** 3)
        ) if std_val > 0 else 0.0

        # VaR at 95%: the value below which the worst 5% of outcomes fall
        var_95 = float(np.percentile(value_trials, 5))

        return MonteCarloResult(
            property_key=property_key,
            brand_key=brand_key,
            asset_type=asset_type,
            n_simulations=self._n,
            p10_usd_m=round(float(percentiles[0]), 2),
            p25_usd_m=round(float(percentiles[1]), 2),
            p50_usd_m=round(float(percentiles[2]), 2),
            p75_usd_m=round(float(percentiles[3]), 2),
            p90_usd_m=round(float(percentiles[4]), 2),
            mean_usd_m=round(mean_val, 2),
            std_usd_m=round(std_val, 2),
            skewness=round(skewness, 4),
            var_95_usd_m=round(var_95, 2),
            viewership_cv=viewership_cv,
            engagement_cv=engagement_cv,
            performance_cv=performance_cv,
        )

    def simulate_batch(
        self,
        brand_key: str,
        property_keys: list[str],
        asset_type: str = "jersey",
    ) -> list[MonteCarloResult]:
        """
        Run simulations for a brand across multiple properties.

        Returns
        -------
        list[MonteCarloResult]
            Sorted by P50 descending.
        """
        results = []
        for pk in property_keys:
            try:
                r = self.simulate(brand_key, pk, asset_type)
                results.append(r)
            except (ValueError, KeyError):
                continue
        results.sort(key=lambda r: r.p50_usd_m, reverse=True)
        return results

    def convergence_check(
        self,
        brand_key: str,
        property_key: str,
        trial_counts: Optional[list[int]] = None,
    ) -> "pd.DataFrame":
        """
        Verify that the simulation converges: check P50 stability as N grows.

        As N → ∞, the P50 should converge to a stable value.
        This method runs the simulation at multiple N values and shows how
        estimates stabilise, demonstrating the law of large numbers.

        Parameters
        ----------
        trial_counts : list[int], optional
            N values to test. Default: [100, 500, 1000, 5000, 10000].

        Returns
        -------
        pd.DataFrame
            Columns: n_trials, p10, p50, p90, std, pct_spread (P90-P10)/P50.
        """
        import pandas as pd

        counts = trial_counts or [100, 500, 1_000, 5_000, 10_000]
        rows   = []
        for n in counts:
            mc = MonteCarloValuator(n_simulations=n, seed=DEFAULT_SEED)
            r  = mc.simulate(brand_key, property_key)
            spread = (r.p90_usd_m - r.p10_usd_m) / r.p50_usd_m * 100 if r.p50_usd_m else 0
            rows.append({
                "n_trials":   n,
                "p10_usd_m":  r.p10_usd_m,
                "p50_usd_m":  r.p50_usd_m,
                "p90_usd_m":  r.p90_usd_m,
                "std_usd_m":  r.std_usd_m,
                "pct_spread": round(spread, 1),
            })
        return pd.DataFrame(rows)

    def sensitivity_to_cv(
        self,
        brand_key: str,
        property_key: str,
        cv_values: Optional[list[float]] = None,
    ) -> "pd.DataFrame":
        """
        Show how P10/P90 spread changes as input uncertainty (CV) increases.

        Useful for stress-testing: "what if viewership uncertainty doubles?"

        Returns
        -------
        pd.DataFrame
            Columns: viewership_cv, engagement_cv, p10, p50, p90, spread_pct.
        """
        import pandas as pd

        cvs  = cv_values or [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        rows = []
        for cv in cvs:
            r = self.simulate(brand_key, property_key, viewership_cv=cv, engagement_cv=cv)
            spread = (r.p90_usd_m - r.p10_usd_m) / r.p50_usd_m * 100 if r.p50_usd_m else 0
            rows.append({
                "input_cv":   cv,
                "p10_usd_m":  r.p10_usd_m,
                "p50_usd_m":  r.p50_usd_m,
                "p90_usd_m":  r.p90_usd_m,
                "spread_pct": round(spread, 1),
            })
        return pd.DataFrame(rows)

    # ── Reporting Helpers ────────────────────────────────────────────────────

    def print_distribution_summary(self, result: MonteCarloResult) -> None:
        """Print a formatted ASCII distribution summary to stdout."""
        print(f"\n  Monte Carlo Valuation — {result.property_key} × {result.brand_key}")
        print(f"  Asset type: {result.asset_type}  |  N = {result.n_simulations:,} simulations")
        print(f"  Input CVs — viewership: {result.viewership_cv:.0%}  "
              f"engagement: {result.engagement_cv:.0%}  "
              f"performance: {result.performance_cv:.0%}")
        print()
        print(f"  {'P10 (downside)':>20}:  ${result.p10_usd_m:>7.1f}m / yr")
        print(f"  {'P25':>20}:  ${result.p25_usd_m:>7.1f}m / yr")
        print(f"  {'P50 (median)':>20}:  ${result.p50_usd_m:>7.1f}m / yr")
        print(f"  {'P75':>20}:  ${result.p75_usd_m:>7.1f}m / yr")
        print(f"  {'P90 (upside)':>20}:  ${result.p90_usd_m:>7.1f}m / yr")
        print(f"  {'Mean':>20}:  ${result.mean_usd_m:>7.1f}m / yr")
        print(f"  {'Std Dev':>20}:  ${result.std_usd_m:>7.1f}m / yr")
        print(f"  {'Skewness':>20}:  {result.skewness:>7.3f}")
        spread = (result.p90_usd_m - result.p10_usd_m) / result.p50_usd_m * 100
        print(f"  {'P10–P90 Spread':>20}:  {spread:>6.1f}%")
        print(f"  {'VaR (95%)':>20}:  ${result.var_95_usd_m:>7.1f}m / yr")

        # ASCII bar chart
        max_bar = 30
        max_val = result.p90_usd_m
        labels = [
            ("P10", result.p10_usd_m),
            ("P25", result.p25_usd_m),
            ("P50", result.p50_usd_m),
            ("P75", result.p75_usd_m),
            ("P90", result.p90_usd_m),
        ]
        print("\n  Distribution bar chart:")
        for label, val in labels:
            bar_len = int(val / max_val * max_bar) if max_val > 0 else 0
            bar = "█" * bar_len
            print(f"  {label:>4}  {bar:<{max_bar}} ${val:.1f}m")
        print()
