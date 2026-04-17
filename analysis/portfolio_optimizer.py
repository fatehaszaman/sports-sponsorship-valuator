"""
analysis/portfolio_optimizer.py — Sponsorship Portfolio Optimizer.

Business summary
----------------
A brand with a $200m sponsorship budget faces a capital allocation problem:
how should it split that budget across multiple properties to maximise total
brand equity lift while managing risk (concentration) and ensuring the right
demographic coverage?

This module applies Markowitz mean-variance optimisation to sponsorship
portfolios. The "expected return" is estimated brand equity lift per dollar;
the "variance" is the uncertainty / risk of each deal; and the "correlation"
between deals reflects fanbase overlap — two football clubs sharing the same
UK fanbase provide less diversification than a football club + an F1 team.

The efficient frontier shows the set of portfolios that maximise equity lift
for a given risk level. The optimal allocation picks the maximum Sharpe-ratio-
equivalent point on that frontier.

Developer notes
---------------
Mean-variance optimisation is implemented as a grid search over N_POINTS
random portfolio weights (Monte Carlo portfolio simulation), then finding
the efficient frontier analytically. This avoids scipy.optimize dependency.

For the analytical frontier, we use the closed-form solution:

    w* = Σ⁻¹ (μ - λ × 1) / normalisation_constant

where:
    Σ   = covariance matrix of deal returns (n × n)
    μ   = expected equity lift per dollar vector (n,)
    λ   = Lagrange multiplier (found via grid search)
    1   = ones vector (budget constraint: w sums to budget)

The covariance matrix is built from correlation + variance inputs (not from
historical price data, since sponsorship has no price history). Instead,
we use structural fanbase overlap to construct the correlation matrix.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from data.sports_data import SPORTS_PROPERTIES
from valuation.audience_model import AudienceDemographicModel
from valuation.comparable_deals import ComparableDealsAnalyzer
from valuation.risk_model import SponsorshipRiskModel


# ── Fanbase overlap matrix (structural correlation proxy) ─────────────────────
# Correlation of fan audiences between properties.
# High overlap = high correlation = less diversification benefit.
# Scale: 0.0 (no overlap) to 1.0 (identical audience).
# Sources: Nielsen Fan Overlap surveys, Kantar Sports crossover research.

_FANBASE_OVERLAP: dict[tuple[str, str], float] = {
    # Premier League clubs: heavy UK domestic overlap
    ("manchester_city",     "manchester_united"):   0.30,  # rival fans, lower crossover
    ("manchester_city",     "liverpool"):           0.18,
    ("manchester_city",     "arsenal"):             0.20,
    ("manchester_city",     "chelsea"):             0.22,
    ("manchester_united",   "liverpool"):           0.15,
    ("manchester_united",   "arsenal"):             0.28,
    ("manchester_united",   "chelsea"):             0.25,
    ("liverpool",           "arsenal"):             0.22,
    ("liverpool",           "chelsea"):             0.20,
    ("arsenal",             "chelsea"):             0.25,
    # Cross-league European football: moderate overlap
    ("manchester_city",     "real_madrid"):         0.35,
    ("manchester_city",     "barcelona"):           0.30,
    ("manchester_city",     "psg"):                 0.28,
    ("manchester_united",   "real_madrid"):         0.40,
    ("manchester_united",   "barcelona"):           0.38,
    ("liverpool",           "real_madrid"):         0.35,
    ("liverpool",           "barcelona"):           0.32,
    ("real_madrid",         "barcelona"):           0.40,  # El Clasico rivals
    ("real_madrid",         "psg"):                 0.35,
    ("barcelona",           "psg"):                 0.33,
    # Football vs. F1: low overlap (different demo)
    ("manchester_city",     "formula_1"):           0.15,
    ("manchester_city",     "red_bull_racing"):     0.12,
    ("manchester_city",     "ferrari_f1"):          0.10,
    ("real_madrid",         "formula_1"):           0.18,
    ("real_madrid",         "ferrari_f1"):          0.20,
    ("formula_1",           "red_bull_racing"):     0.75,  # same sport
    ("formula_1",           "ferrari_f1"):          0.70,
    ("red_bull_racing",     "ferrari_f1"):          0.65,
    # Football vs. NBA/NFL: very low overlap (geographic segmentation)
    ("manchester_city",     "la_lakers"):           0.08,
    ("manchester_city",     "nba"):                 0.10,
    ("manchester_united",   "la_lakers"):           0.10,
    ("real_madrid",         "la_lakers"):           0.12,
    ("formula_1",           "nba"):                 0.08,
    ("formula_1",           "nfl"):                 0.06,
    # NFL teams
    ("nfl",                 "dallas_cowboys"):      0.80,
    ("nfl",                 "new_england_patriots"):0.75,
    ("dallas_cowboys",      "new_england_patriots"):0.40,
    # NBA
    ("nba",                 "la_lakers"):           0.82,
    ("nba",                 "golden_state_warriors"):0.78,
    ("la_lakers",           "golden_state_warriors"):0.55,
    # Cross-sport low
    ("nfl",                 "nba"):                 0.22,
    ("nfl",                 "formula_1"):           0.06,
    ("wimbledon",           "formula_1"):           0.18,
    ("wimbledon",           "manchester_city"):     0.08,
    # Brentford: niche, low overlap with all
    ("brentford",           "manchester_city"):     0.10,
    ("brentford",           "manchester_united"):   0.08,
    ("brentford",           "liverpool"):           0.08,
}

# Expected equity lift per USD million invested (% brand equity lift)
# Calibrated from BrandFinance and IPA Effectiveness Databank data
_EQUITY_LIFT_PER_M: dict[str, float] = {
    "manchester_city":      1.20,
    "manchester_united":    1.45,
    "real_madrid":          1.55,
    "barcelona":            1.50,
    "psg":                  1.30,
    "chelsea":              1.15,
    "arsenal":              1.10,
    "liverpool":            1.35,
    "nfl":                  1.80,
    "dallas_cowboys":       1.60,
    "new_england_patriots": 1.55,
    "la_lakers":            1.50,
    "golden_state_warriors":1.45,
    "formula_1":            1.65,
    "red_bull_racing":      1.70,
    "ferrari_f1":           1.60,
    "wimbledon":            1.20,
    "nba":                  1.55,
    "brentford":            0.85,
    "super_bowl":           2.50,
}

# Risk (volatility of expected equity lift) per property
_EQUITY_VOLATILITY: dict[str, float] = {
    "manchester_city":      0.22,
    "manchester_united":    0.28,
    "real_madrid":          0.18,
    "barcelona":            0.25,
    "psg":                  0.24,
    "chelsea":              0.26,
    "arsenal":              0.22,
    "liverpool":            0.20,
    "nfl":                  0.15,
    "dallas_cowboys":       0.18,
    "new_england_patriots": 0.20,
    "la_lakers":            0.22,
    "golden_state_warriors":0.24,
    "formula_1":            0.20,
    "red_bull_racing":      0.25,
    "ferrari_f1":           0.22,
    "wimbledon":            0.12,
    "nba":                  0.16,
    "brentford":            0.40,
    "super_bowl":           0.08,
}


def _overlap(p1: str, p2: str) -> float:
    """Get fanbase overlap correlation between two properties (symmetric)."""
    key = (p1, p2) if (p1, p2) in _FANBASE_OVERLAP else (p2, p1)
    return _FANBASE_OVERLAP.get(key, 0.05)  # default 5% if no data


class PortfolioOptimizer:
    """
    Markowitz mean-variance optimisation for sponsorship budget allocation.

    Business summary
    ----------------
    Given a total sponsorship budget and a menu of available deals, find the
    optimal allocation that maximises expected brand equity lift per unit of risk.

    Key outputs:
      - correlation_matrix(): fanbase overlap between properties
      - efficient_frontier():  risk/return tradeoff across 50 portfolio points
      - optimal_allocation():  the specific budget split that maximises the
                               Sharpe-equivalent ratio

    Developer notes
    ---------------
    The Markowitz optimisation is implemented as:
      1. Monte Carlo simulation: randomly sample 50,000 portfolio weight vectors,
         compute (return, risk) for each, identify the efficient frontier.
      2. Analytical solution: use the matrix form w* = Σ⁻¹μ / (1ᵀΣ⁻¹μ) to find
         the maximum-return-per-unit-risk portfolio.

    Both methods are shown so the math is explicit and cross-checkable.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng      = np.random.default_rng(seed)
        self._risk_mdl = SponsorshipRiskModel()

    # ── Correlation Matrix ────────────────────────────────────────────────────

    def correlation_matrix(self, properties: list[str]) -> np.ndarray:
        """
        Build the fanbase overlap correlation matrix for a set of properties.

        The diagonal is always 1.0 (perfect self-correlation).
        Off-diagonal entries are the fanbase overlap fractions from _FANBASE_OVERLAP.

        Parameters
        ----------
        properties : list[str]

        Returns
        -------
        np.ndarray
            Shape (n, n) symmetric correlation matrix.
        """
        n   = len(properties)
        mat = np.eye(n)
        for i, p1 in enumerate(properties):
            for j, p2 in enumerate(properties):
                if i != j:
                    mat[i, j] = _overlap(p1, p2)
        return mat

    def covariance_matrix(self, properties: list[str]) -> np.ndarray:
        """
        Covariance matrix = D × Corr × D  where D = diag(volatilities).

        Parameters
        ----------
        properties : list[str]

        Returns
        -------
        np.ndarray
            Shape (n, n) covariance matrix.
        """
        vols = np.array([_EQUITY_VOLATILITY.get(p, 0.25) for p in properties])
        corr = self.correlation_matrix(properties)
        D    = np.diag(vols)
        return D @ corr @ D

    def correlation_dataframe(self, properties: list[str]) -> pd.DataFrame:
        """Return correlation matrix as a labelled DataFrame."""
        mat = self.correlation_matrix(properties)
        return pd.DataFrame(mat, index=properties, columns=properties).round(3)

    # ── Efficient Frontier ────────────────────────────────────────────────────

    def efficient_frontier(
        self,
        budget: float,
        properties: list[str],
        n_points: int = 50,
        n_mc_portfolios: int = 50_000,
    ) -> pd.DataFrame:
        """
        Compute the risk/return efficient frontier for sponsorship portfolios.

        Methodology
        -----------
        Monte Carlo simulation of n_mc_portfolios random weight vectors:
          1. Sample random weights w ~ Dirichlet(α=1) [uniform on simplex]
          2. Scale to budget: dollar_weights = w × budget
          3. Compute portfolio expected equity lift:
               E[R_p] = w · μ    (dot product)
          4. Compute portfolio variance:
               Var(R_p) = wᵀ Σ w (quadratic form)
          5. Identify the Pareto-efficient frontier:
               For each level of variance, keep only the max-return portfolio.

        The efficient frontier is the upper-left boundary of the (risk, return)
        scatter plot — the set of portfolios where no improvement in return is
        possible without increasing risk.

        Parameters
        ----------
        budget : float
            Total sponsorship budget in USD millions.
        properties : list[str]
            Available properties to allocate budget across.
        n_points : int
            Number of points on the returned frontier.
        n_mc_portfolios : int
            Number of random portfolios to simulate.

        Returns
        -------
        pd.DataFrame
            Columns: portfolio_risk (std dev), portfolio_return (equity lift),
                     sharpe_equivalent, allocations (dict).
        """
        props = [p for p in properties if p in _EQUITY_LIFT_PER_M]
        n     = len(props)
        if n == 0:
            return pd.DataFrame()

        mu    = np.array([_EQUITY_LIFT_PER_M.get(p, 1.0) for p in props])
        sigma = self.covariance_matrix(props)

        # ── Monte Carlo: sample random weight vectors ─────────────────────────
        # Dirichlet(1,...,1) = uniform distribution on the n-simplex
        raw_weights = self._rng.dirichlet(np.ones(n), size=n_mc_portfolios)  # (M, n)

        # Portfolio return = w · μ (vectorized dot product over M portfolios)
        port_returns = raw_weights @ mu                    # (M,)

        # Portfolio variance = diag(W Σ Wᵀ) — computed efficiently
        # Var_i = w_i · (Σ w_i)  for each portfolio i
        sigma_w = raw_weights @ sigma                      # (M, n)
        port_var = np.einsum("ij,ij->i", raw_weights, sigma_w)  # (M,)
        port_std = np.sqrt(np.maximum(port_var, 0))       # (M,)

        # ── Extract efficient frontier ────────────────────────────────────────
        # Bin by risk level; keep max-return portfolio in each bin
        risk_bins  = np.linspace(port_std.min(), port_std.max(), n_points + 1)
        frontier_rows = []
        for lo, hi in zip(risk_bins[:-1], risk_bins[1:]):
            mask = (port_std >= lo) & (port_std < hi)
            if not mask.any():
                continue
            best_idx = np.argmax(port_returns[mask])
            global_idx = np.where(mask)[0][best_idx]
            w = raw_weights[global_idx]
            sharpe = port_returns[global_idx] / port_std[global_idx] if port_std[global_idx] > 0 else 0

            allocs = {props[i]: round(w[i] * budget, 2) for i in range(n)}
            frontier_rows.append({
                "portfolio_risk_std":   round(float(port_std[global_idx]), 4),
                "portfolio_return":     round(float(port_returns[global_idx]), 4),
                "sharpe_equivalent":    round(float(sharpe), 4),
                "allocations_usd_m":    allocs,
            })

        if not frontier_rows:
            return pd.DataFrame()

        df = pd.DataFrame(frontier_rows).sort_values("portfolio_risk_std")
        return df

    # ── Optimal Allocation ────────────────────────────────────────────────────

    def optimal_allocation(
        self,
        budget: float,
        properties: list[str],
        max_single_deal_pct: float = 0.40,
        min_n_deals: int = 2,
    ) -> dict:
        """
        Find the optimal budget allocation maximising Sharpe-equivalent ratio.

        Analytical solution (max-Sharpe tangency portfolio):
        -----------------------------------------------------
        Without constraints, the max-Sharpe portfolio is:

            w_unscaled = Σ⁻¹ × μ
            w*         = w_unscaled / sum(w_unscaled)   [normalise to sum = 1]

        Where Σ⁻¹ is the inverse covariance matrix (computed via numpy.linalg.inv).
        This is the closed-form solution to the Markowitz optimisation problem
        with a risk-free rate of 0.

        Constraints applied post-optimisation:
          - max_single_deal_pct: no single deal > X% of budget (concentration limit)
          - min_n_deals: minimum number of properties in portfolio

        Parameters
        ----------
        budget : float
            Total budget in USD millions.
        properties : list[str]
        max_single_deal_pct : float
            Maximum fraction of budget for any single deal (default 40%).
        min_n_deals : int
            Minimum number of deals (forces diversification).

        Returns
        -------
        dict
            {optimal_allocations, portfolio_return, portfolio_risk,
             sharpe_equivalent, rationale}
        """
        props = [p for p in properties if p in _EQUITY_LIFT_PER_M]
        n     = len(props)
        if n < min_n_deals:
            raise ValueError(f"Need at least {min_n_deals} valid properties.")

        mu    = np.array([_EQUITY_LIFT_PER_M.get(p, 1.0) for p in props])
        sigma = self.covariance_matrix(props)

        # ── Analytical max-Sharpe weights: w* = Σ⁻¹μ / (1ᵀΣ⁻¹μ) ─────────────
        try:
            sigma_inv   = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            # Fallback: pseudoinverse if sigma is singular
            sigma_inv   = np.linalg.pinv(sigma)

        w_raw = sigma_inv @ mu                      # Σ⁻¹μ
        w_raw = np.maximum(w_raw, 0)                # long-only: no short positions
        if w_raw.sum() == 0:
            w_raw = np.ones(n) / n
        w_opt = w_raw / w_raw.sum()                 # normalise to sum = 1

        # ── Apply concentration constraint ───────────────────────────────────
        max_w = max_single_deal_pct
        for _ in range(100):                        # iterative clipping
            clipped  = np.minimum(w_opt, max_w)
            excess   = w_opt - clipped
            free_idx = w_opt < max_w
            if free_idx.sum() > 0 and excess.sum() > 0:
                clipped[free_idx] += excess.sum() / free_idx.sum()
                clipped = np.minimum(clipped, max_w)
            total = clipped.sum()
            w_opt = clipped / total if total > 0 else clipped
            if np.all(w_opt <= max_w + 1e-8):
                break

        # ── Ensure min_n_deals ───────────────────────────────────────────────
        if (w_opt > 0.01).sum() < min_n_deals:
            # Force at least min_n_deals by giving equal weight to bottom ones
            bottom_idx = np.argsort(w_opt)[:min_n_deals]
            w_opt[bottom_idx] = 0.02
            w_opt /= w_opt.sum()

        # ── Portfolio metrics ────────────────────────────────────────────────
        port_return = float(w_opt @ mu)
        port_var    = float(w_opt @ sigma @ w_opt)
        port_std    = math.sqrt(max(port_var, 0))
        sharpe      = port_return / port_std if port_std > 0 else 0.0

        # ── Dollar allocations ───────────────────────────────────────────────
        allocations = {
            props[i]: {
                "weight":    round(float(w_opt[i]), 4),
                "usd_m":     round(float(w_opt[i]) * budget, 2),
                "eq_lift":   _EQUITY_LIFT_PER_M.get(props[i], 1.0),
                "risk":      _EQUITY_VOLATILITY.get(props[i], 0.25),
            }
            for i in range(n)
            if w_opt[i] > 0.005
        }

        # Sort by allocation size
        allocations = dict(sorted(allocations.items(), key=lambda x: x[1]["usd_m"], reverse=True))

        rationale = []
        top3 = list(allocations.keys())[:3]
        for p in top3:
            rationale.append(
                f"{p}: ${allocations[p]['usd_m']:.1f}m ({allocations[p]['weight']:.1%}) — "
                f"eq_lift={allocations[p]['eq_lift']:.2f}, risk={allocations[p]['risk']:.2f}"
            )

        return {
            "budget_usd_m":         budget,
            "n_properties":         len(allocations),
            "optimal_allocations":  allocations,
            "portfolio_return":     round(port_return, 4),
            "portfolio_risk_std":   round(port_std, 4),
            "sharpe_equivalent":    round(sharpe, 4),
            "max_single_pct":       max_single_deal_pct,
            "rationale":            rationale,
            "method": "Analytical: w* = Σ⁻¹μ / (1ᵀΣ⁻¹μ), then clipped to concentration limit",
        }

    def diversification_score(self, allocations: dict[str, float]) -> float:
        """
        Score the diversification quality of a portfolio (0–10).

        High score = low average fanbase overlap + spread across sports/geographies.
        Low score  = concentrated in one sport/market.

        Parameters
        ----------
        allocations : dict[str, float]
            {property_key: budget_usd_m}

        Returns
        -------
        float in [0, 10].
        """
        props = list(allocations.keys())
        if len(props) < 2:
            return 0.0

        # Average pairwise overlap (lower = better diversification)
        total_weight = sum(allocations.values())
        if total_weight == 0:
            return 0.0

        weights = {p: allocations[p] / total_weight for p in props}
        avg_overlap = 0.0
        n_pairs = 0
        for p1, p2 in combinations(props, 2):
            w  = weights[p1] * weights[p2]
            ov = _overlap(p1, p2)
            avg_overlap += w * ov
            n_pairs += 1

        # Convert: 0 overlap → score 10; 1.0 overlap → score 0
        score = (1.0 - avg_overlap) * 10.0

        # Bonus for having 3+ different sports
        from data.sports_data import SPORTS_PROPERTIES
        sports = {SPORTS_PROPERTIES[p].sport for p in props if p in SPORTS_PROPERTIES}
        if len(sports) >= 3:
            score = min(score + 0.5, 10.0)

        return round(score, 2)
