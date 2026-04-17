"""
valuation/risk_model.py — Sponsorship Deal Risk Model.

Business summary
----------------
Every sponsorship deal carries risks that can impair its value below what the
exposure model predicts. This module quantifies four categories of risk:

  1. Athlete/individual scandal risk: an endorser's public misconduct (doping,
     legal issues, social media controversy) can rapidly destroy brand value.
  2. Team performance risk: relegation or sustained poor results shrink
     broadcast audiences and reduce media value.
  3. Regulatory risk: several jurisdictions are restricting gambling and
     alcohol sponsorship in sport (UK's Gambling Act review, Italy's ban).
  4. Geopolitical risk: state-owned properties (PSG, Manchester City, LIV Golf)
     carry reputational risk for sponsors from certain markets.

VaR methodology
---------------
Value-at-Risk is computed by simulating the loss distribution from each risk
factor and reporting the worst-case annual deal value at a given confidence
level (default 95%). Concretely:

    VaR_95 = deal_value - percentile_5(simulated_impaired_values)

This is the sponsorship equivalent of a financial portfolio VaR calculation.

Scenario matrix
---------------
A 3×3 matrix of (team performance × brand reception) scenarios gives
nine distinct NPV outcomes, helping sponsors visualise their full exposure
before committing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import DEFAULT_DISCOUNT_RATE, DEFAULT_GROWTH_RATE
from data.sports_data import ATHLETE_PROFILES, SPORTS_PROPERTIES


# ── Athlete Reputation Risk Scores (0–10; high = HIGH risk) ──────────────────
# Sourced from public sentiment analysis, legal history, social media monitoring.
# Note: these are RISK scores, not controversy scores — higher = more dangerous.

_ATHLETE_RISK_SCORES: dict[str, dict] = {
    "cristiano_ronaldo": {
        "scandal_risk":        3.5,
        "legal_risk":          2.0,
        "social_media_risk":   3.0,
        "doping_risk":         0.5,
        "overall_risk":        2.8,
        "notes": "Historical Juventus tax case resolved; strong brand management",
    },
    "lionel_messi": {
        "scandal_risk":        1.5,
        "legal_risk":          2.5,
        "social_media_risk":   0.5,
        "doping_risk":         0.5,
        "overall_risk":        1.5,
        "notes": "Spanish tax conviction (2016, resolved); squeaky-clean image since",
    },
    "lebron_james": {
        "scandal_risk":        2.0,
        "legal_risk":          0.5,
        "social_media_risk":   4.0,
        "doping_risk":         0.5,
        "overall_risk":        2.2,
        "notes": "Political commentary risk for certain brand categories",
    },
    "naomi_osaka": {
        "scandal_risk":        1.5,
        "legal_risk":          0.0,
        "social_media_risk":   2.5,
        "doping_risk":         0.0,
        "overall_risk":        1.5,
        "notes": "Mental health advocacy; some PR complexity but generally positive",
    },
    "max_verstappen": {
        "scandal_risk":        2.0,
        "legal_risk":          0.5,
        "social_media_risk":   3.5,
        "doping_risk":         0.0,
        "overall_risk":        2.0,
        "notes": "Aggressive on-track style; Dutch directness in interviews",
    },
    "serena_williams": {
        "scandal_risk":        2.0,
        "legal_risk":          0.0,
        "social_media_risk":   2.0,
        "doping_risk":         0.0,
        "overall_risk":        1.5,
        "notes": "Retired; umpire controversies resolved; business pivot positive",
    },
    "neymar": {
        "scandal_risk":        6.5,
        "legal_risk":          5.0,
        "social_media_risk":   5.5,
        "doping_risk":         1.0,
        "overall_risk":        5.8,
        "notes": "Rape allegations (acquitted), tax fraud, nightclub controversies",
    },
    "lewis_hamilton": {
        "scandal_risk":        1.5,
        "legal_risk":          0.5,
        "social_media_risk":   2.0,
        "doping_risk":         0.0,
        "overall_risk":        1.3,
        "notes": "Activism-driven brand; minimal legal risk; strong sponsor record",
    },
    "stephen_curry": {
        "scandal_risk":        0.5,
        "legal_risk":          0.0,
        "social_media_risk":   0.5,
        "doping_risk":         0.0,
        "overall_risk":        0.5,
        "notes": "Widely regarded as the safest endorser in professional sport",
    },
    "kylian_mbappe": {
        "scandal_risk":        2.0,
        "legal_risk":          1.5,
        "social_media_risk":   2.5,
        "doping_risk":         0.0,
        "overall_risk":        1.8,
        "notes": "PSG contract saga; France NT internal tensions; managed well",
    },
}

# ── Property-level risk parameters ───────────────────────────────────────────

_PROPERTY_RISKS: dict[str, dict] = {
    "manchester_city": {
        "relegation_risk":     0.02,  # probability per year
        "viewership_decline":  0.05,  # annual probability of >15% viewership drop
        "regulatory_risk":     0.10,  # probability of sponsorship category ban (betting, gambling)
        "geopolitical_risk":   0.20,  # UAE state ownership controversy risk
        "financial_risk":      0.15,  # FFP / PSR rule breach risk
        "notes": "Abu Dhabi ownership creates reputational risk in certain markets (EU human rights legislation)",
    },
    "manchester_united": {
        "relegation_risk":     0.03,
        "viewership_decline":  0.08,
        "regulatory_risk":     0.08,
        "geopolitical_risk":   0.05,
        "financial_risk":      0.10,
        "notes": "Post-Glazer sale uncertainty; performance decline risk",
    },
    "real_madrid": {
        "relegation_risk":     0.01,
        "viewership_decline":  0.03,
        "regulatory_risk":     0.06,
        "geopolitical_risk":   0.04,
        "financial_risk":      0.05,
        "notes": "Consistently highest brand value; low structural risk",
    },
    "barcelona": {
        "relegation_risk":     0.01,
        "viewership_decline":  0.04,
        "regulatory_risk":     0.07,
        "geopolitical_risk":   0.03,
        "financial_risk":      0.25,  # Significant financial crisis / LaLiga lever rules
        "notes": "Debt crisis ongoing; LaLiga economic controls constrain roster spending",
    },
    "psg": {
        "relegation_risk":     0.02,
        "viewership_decline":  0.05,
        "regulatory_risk":     0.08,
        "geopolitical_risk":   0.22,  # Qatar state ownership
        "financial_risk":      0.12,
        "notes": "Qatar state backing limits financial risk but creates geopolitical exposure",
    },
    "chelsea": {
        "relegation_risk":     0.04,
        "viewership_decline":  0.08,
        "regulatory_risk":     0.08,
        "geopolitical_risk":   0.08,  # American ownership, less controversial
        "financial_risk":      0.18,  # PSR breaches
        "notes": "Clegg ownership transition; PSR/FFP compliance risk",
    },
    "arsenal": {
        "relegation_risk":     0.03,
        "viewership_decline":  0.06,
        "regulatory_risk":     0.07,
        "geopolitical_risk":   0.04,
        "financial_risk":      0.08,
        "notes": "Stable Kroenke ownership; PSR compliant; performance risk manageable",
    },
    "liverpool": {
        "relegation_risk":     0.02,
        "viewership_decline":  0.04,
        "regulatory_risk":     0.07,
        "geopolitical_risk":   0.04,
        "financial_risk":      0.06,
        "notes": "FSG ownership stable; strong commercial infrastructure",
    },
    "nfl": {
        "relegation_risk":     0.00,
        "viewership_decline":  0.05,
        "regulatory_risk":     0.12,
        "geopolitical_risk":   0.06,
        "financial_risk":      0.03,
        "notes": "Political controversy (anthem protests); gambling regulation expansion",
    },
    "dallas_cowboys": {
        "relegation_risk":     0.00,
        "viewership_decline":  0.04,
        "regulatory_risk":     0.10,
        "geopolitical_risk":   0.05,
        "financial_risk":      0.03,
        "notes": "Jerry Jones polarisation risk; consistently highest franchise value",
    },
    "formula_1": {
        "relegation_risk":     0.00,
        "viewership_decline":  0.08,
        "regulatory_risk":     0.10,
        "geopolitical_risk":   0.15,  # Saudi, Abu Dhabi, Las Vegas race controversies
        "financial_risk":      0.05,
        "notes": "Post-peak viewership risk after Drive to Survive boom; new markets carry political risk",
    },
    "red_bull_racing": {
        "relegation_risk":     0.00,
        "viewership_decline":  0.08,
        "regulatory_risk":     0.10,
        "geopolitical_risk":   0.08,
        "financial_risk":      0.10,  # RB F1 budget cap violations
        "notes": "Budget cap breach sanctions (2022); Horner controversy (2024)",
    },
    "ferrari_f1": {
        "relegation_risk":     0.00,
        "viewership_decline":  0.08,
        "regulatory_risk":     0.10,
        "geopolitical_risk":   0.04,
        "financial_risk":      0.05,
        "notes": "Brand-safe legacy; low geopolitical risk; on-track performance risk",
    },
    "la_lakers": {
        "relegation_risk":     0.00,
        "viewership_decline":  0.06,
        "regulatory_risk":     0.08,
        "geopolitical_risk":   0.05,
        "financial_risk":      0.04,
        "notes": "LeBron aging risk; roster transition may reduce viewership",
    },
    "golden_state_warriors": {
        "relegation_risk":     0.00,
        "viewership_decline":  0.07,
        "regulatory_risk":     0.08,
        "geopolitical_risk":   0.04,
        "financial_risk":      0.04,
        "notes": "Dynasty-end transition; Curry aging risk",
    },
    "wimbledon": {
        "relegation_risk":     0.00,
        "viewership_decline":  0.04,
        "regulatory_risk":     0.05,
        "geopolitical_risk":   0.02,
        "financial_risk":      0.02,
        "notes": "Institutionally stable; Russia ban controversy manageable",
    },
    "nba": {
        "relegation_risk":     0.00,
        "viewership_decline":  0.06,
        "regulatory_risk":     0.09,
        "geopolitical_risk":   0.10,  # China market tensions
        "financial_risk":      0.03,
        "notes": "China-USA geopolitical tension constrains Asian market value",
    },
    "brentford": {
        "relegation_risk":     0.25,  # newly promoted club, significant risk
        "viewership_decline":  0.15,
        "regulatory_risk":     0.06,
        "geopolitical_risk":   0.02,
        "financial_risk":      0.08,
        "notes": "Relegation risk is the primary value impairment factor",
    },
    "new_england_patriots": {
        "relegation_risk":     0.00,
        "viewership_decline":  0.06,
        "regulatory_risk":     0.10,
        "geopolitical_risk":   0.04,
        "financial_risk":      0.03,
        "notes": "Post-Belichick era uncertainty; viewership risk without sustained winning",
    },
}


class SponsorshipRiskModel:
    """
    Quantifies risks that can impair sponsorship deal value below fair value.

    Business summary
    ----------------
    The fair value engine gives you the expected value. This model gives you
    the distribution of bad outcomes. Key outputs:

      - value_at_risk(): worst-case annual value at 95% confidence
      - scenario_matrix(): 9-cell grid of performance × brand scenarios
      - deal_stress_test(): NPV impact of named shock events

    Developer notes
    ---------------
    VaR simulation draws 10,000 samples from each risk factor's distribution
    (Bernoulli for binary events, Beta for continuous impairments) and computes
    the combined impaired value as a product of independent multipliers.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng       = np.random.default_rng(seed)
        self._prop_risks = _PROPERTY_RISKS
        self._ath_risks  = _ATHLETE_RISK_SCORES

    # ── VaR ─────────────────────────────────────────────────────────────────

    def value_at_risk(
        self,
        property_key: str,
        annual_deal_value_usd_m: float,
        confidence: float = 0.95,
        n_simulations: int = 10_000,
    ) -> dict:
        """
        VaR-style worst-case deal value at given confidence level.

        Simulation methodology
        ----------------------
        For each trial, multiply the deal value by a product of risk multipliers:

            value_t = V × m_relegation × m_viewership × m_regulatory × m_geo × m_financial

        where each m_i is drawn as:
            - Binary event (relegation, regulatory): Bernoulli(p) → if event occurs,
              m = Beta(α, β) loss multiplier (e.g. relegation → lose 60–80% of value)
            - Continuous (viewership decline): m ~ Beta(α, β) with mean close to 1

        VaR = V - percentile_{1-confidence}(value_t)
              = value lost in worst (1-confidence)% of scenarios

        Parameters
        ----------
        property_key : str
        annual_deal_value_usd_m : float
        confidence : float
            VaR confidence level (default 0.95 = 95%).
        n_simulations : int

        Returns
        -------
        dict
            {var_usd_m, cvar_usd_m, worst_pct, impaired_p50, original_value}
        """
        risks = self._prop_risks.get(property_key, {
            "relegation_risk": 0.03, "viewership_decline": 0.06,
            "regulatory_risk": 0.08, "geopolitical_risk": 0.05, "financial_risk": 0.05,
        })

        V = annual_deal_value_usd_m
        N = n_simulations

        # ── Sample risk multipliers (vectorized) ─────────────────────────────
        # Relegation / competition risk: binary event → value drops 65–85%
        rel_event = self._rng.binomial(1, risks["relegation_risk"], N).astype(float)
        rel_mult  = np.where(rel_event, self._rng.beta(2.5, 1.5, N) * 0.35, 1.0)
        # Note: Beta(2.5,1.5) ≈ [0.5,0.9]; × 0.35 → multiplier [0.18,0.32]

        # Viewership decline: continuous, mostly near 1.0 with left tail
        view_mult = self._rng.beta(8, 2, N)   # mean ≈ 0.80, can dip to 0.5

        # Regulatory risk: binary → forced value impairment if category banned
        reg_event = self._rng.binomial(1, risks["regulatory_risk"], N).astype(float)
        reg_mult  = np.where(reg_event, self._rng.beta(3, 2, N) * 0.50, 1.0)

        # Geopolitical risk: continuous impairment in affected markets
        geo_scale = risks["geopolitical_risk"]
        geo_mult  = self._rng.beta(
            max(1.0, 10 * (1 - geo_scale)),
            max(1.0, 10 * geo_scale),
            N,
        )

        # Financial (FFP/PSR breach) risk: partial value loss
        fin_event = self._rng.binomial(1, risks["financial_risk"], N).astype(float)
        fin_mult  = np.where(fin_event, self._rng.beta(5, 2, N) * 0.80, 1.0)

        # Combined impaired value
        combined_mult   = rel_mult * view_mult * reg_mult * geo_mult * fin_mult
        impaired_values = V * combined_mult

        # VaR and CVaR
        pctile = (1 - confidence) * 100
        var_threshold   = float(np.percentile(impaired_values, pctile))
        var_usd_m       = V - var_threshold

        # CVaR (Expected Shortfall): mean of worst (1-confidence)% outcomes
        tail_vals = impaired_values[impaired_values <= var_threshold]
        cvar_usd_m = V - float(tail_vals.mean()) if len(tail_vals) > 0 else var_usd_m

        return {
            "property":             property_key,
            "original_value_usd_m": round(V, 2),
            "var_usd_m":            round(var_usd_m, 2),
            "cvar_usd_m":           round(cvar_usd_m, 2),
            "impaired_p50_usd_m":   round(float(np.median(impaired_values)), 2),
            "worst_pct_loss":       round(var_usd_m / V * 100, 1) if V > 0 else 0.0,
            "confidence":           confidence,
            "n_simulations":        N,
            "risk_factors":         risks,
        }

    # ── Scenario Matrix ──────────────────────────────────────────────────────

    def scenario_matrix(
        self,
        property_key: str,
        brand_key: str,
        base_annual_value_usd_m: float,
        deal_years: int = 5,
        discount_rate: float = DEFAULT_DISCOUNT_RATE,
    ) -> pd.DataFrame:
        """
        3×3 scenario matrix: team performance × brand reception → NPV.

        Scenarios
        ---------
        Team performance (rows):
          POOR:    team performs badly — relegation, early exits, viewership drops
          NEUTRAL: expected performance — base case
          STRONG:  team wins major trophy — halo effect, viewership surge

        Brand reception (columns):
          WEAK:    sponsor ad recall low, product recall mediocre, no viral moments
          BASE:    expected brand impact per model
          STRONG:  high brand association, viral campaigns, strong recall metrics

        Each cell computes NPV under the combined scenario assumptions.

        Returns
        -------
        pd.DataFrame
            3×3 NPV matrix in USD millions.
            Index:   ["poor_perf", "neutral_perf", "strong_perf"]
            Columns: ["weak_brand", "base_brand",  "strong_brand"]
        """
        # Scale factors per scenario cell (team_perf × brand_reception)
        _PERF_SCALES   = {"poor_perf": 0.60, "neutral_perf": 1.00, "strong_perf": 1.22}
        _BRAND_SCALES  = {"weak_brand": 0.75, "base_brand": 1.00,  "strong_brand": 1.18}
        _PERF_GROWTH   = {"poor_perf": -0.02, "neutral_perf": DEFAULT_GROWTH_RATE, "strong_perf": 0.08}
        _BRAND_DR      = {"weak_brand": 0.12, "base_brand": DEFAULT_DISCOUNT_RATE, "strong_brand": 0.08}

        perf_keys  = ["poor_perf",  "neutral_perf",  "strong_perf"]
        brand_keys = ["weak_brand", "base_brand",    "strong_brand"]

        def _npv(annual_val: float, gr: float, dr: float) -> float:
            """DCF NPV loop."""
            total = 0.0
            for t in range(1, deal_years + 1):
                total += annual_val * (1 + gr) ** (t - 1) / (1 + dr) ** t
            return round(total, 2)

        data = {}
        for bk in brand_keys:
            col = []
            for pk in perf_keys:
                adj_val = base_annual_value_usd_m * _PERF_SCALES[pk] * _BRAND_SCALES[bk]
                npv_val = _npv(adj_val, _PERF_GROWTH[pk], _BRAND_DR[bk])
                col.append(npv_val)
            data[bk] = col

        df = pd.DataFrame(data, index=perf_keys)
        df.index.name   = "Team Performance →"
        df.columns.name = "Brand Reception ↓"
        return df

    # ── Stress Testing ───────────────────────────────────────────────────────

    def deal_stress_test(
        self,
        property_key: str,
        annual_deal_value_usd_m: float,
        deal_years: int,
        shocks: Optional[list[dict]] = None,
        discount_rate: float = DEFAULT_DISCOUNT_RATE,
    ) -> dict:
        """
        Apply named shock events and compute NPV impact.

        Each shock has a name, a value impairment fraction, and a year offset.

        Built-in named shocks
        ---------------------
        "team_relegated"         → -65% value from year of shock onwards
        "athlete_scandal"        → -25% value in shock year, -10% for 2 subsequent years
        "viewership_drops_30pct" → -30% value from shock year onwards
        "regulatory_ban"         → -100% value (deal terminated) from shock year
        "ownership_controversy"  → -15% value for 2 years post-shock
        "major_trophy_won"       → +20% value in shock year, +8% following year
        "rival_sponsor_scandal"  → +10% halo from competitor brand's reputational damage

        Parameters
        ----------
        property_key : str
        annual_deal_value_usd_m : float
            Base annual deal value.
        deal_years : int
        shocks : list[dict], optional
            Each shock: {"name": str, "year": int}
            "year" is 1-indexed (year 1 = first year of deal).
        discount_rate : float

        Returns
        -------
        dict
            {base_npv, stressed_npv, npv_delta, npv_delta_pct, year_by_year_cashflows}
        """
        _SHOCK_LIBRARY = {
            "team_relegated":          {"type": "permanent",  "scale": 0.35, "duration": 99},
            "athlete_scandal":         {"type": "temporary",  "scale": 0.75, "duration": 1, "recovery": 0.90},
            "viewership_drops_30pct":  {"type": "permanent",  "scale": 0.70, "duration": 99},
            "regulatory_ban":          {"type": "permanent",  "scale": 0.00, "duration": 99},
            "ownership_controversy":   {"type": "temporary",  "scale": 0.85, "duration": 2, "recovery": 1.00},
            "major_trophy_won":        {"type": "temporary",  "scale": 1.20, "duration": 1, "recovery": 1.08},
            "rival_sponsor_scandal":   {"type": "temporary",  "scale": 1.10, "duration": 2, "recovery": 1.00},
        }

        shocks = shocks or []

        # Build year-by-year cash flow array
        cfs_base     = np.array([annual_deal_value_usd_m * (1 + DEFAULT_GROWTH_RATE) ** (t) for t in range(deal_years)])
        cfs_stressed = cfs_base.copy()

        applied_shocks = []
        for shock in shocks:
            name = shock.get("name", "")
            yr   = shock.get("year", 1) - 1  # 0-indexed
            s    = _SHOCK_LIBRARY.get(name)
            if s is None or yr < 0 or yr >= deal_years:
                continue

            dur = min(s["duration"], deal_years - yr)
            scale = s["scale"]
            applied_shocks.append({"name": name, "year": yr + 1, "scale": scale})

            for i in range(yr, yr + dur):
                if i < deal_years:
                    cfs_stressed[i] *= scale
                    if s["type"] == "temporary" and i > yr:
                        recovery = s.get("recovery", 1.0)
                        cfs_stressed[i] *= recovery

        # NPV calculation
        discount_factors = np.array([(1 + discount_rate) ** -(t + 1) for t in range(deal_years)])
        base_npv     = float((cfs_base     * discount_factors).sum())
        stressed_npv = float((cfs_stressed * discount_factors).sum())

        return {
            "property":             property_key,
            "base_npv_usd_m":       round(base_npv, 2),
            "stressed_npv_usd_m":   round(stressed_npv, 2),
            "npv_delta_usd_m":      round(stressed_npv - base_npv, 2),
            "npv_delta_pct":        round((stressed_npv - base_npv) / base_npv * 100, 1) if base_npv > 0 else 0.0,
            "applied_shocks":       applied_shocks,
            "year_by_year_base":    [round(v, 2) for v in cfs_base.tolist()],
            "year_by_year_stressed":[round(v, 2) for v in cfs_stressed.tolist()],
            "discount_rate":        discount_rate,
        }

    def athlete_risk_score(self, athlete_key: str) -> dict:
        """Return full risk profile for an athlete."""
        return self._ath_risks.get(athlete_key, {"overall_risk": 5.0, "notes": "No data"})

    def property_risk_summary(self, property_key: str) -> dict:
        """Return composite risk summary for a property."""
        risks = self._prop_risks.get(property_key, {})
        if not risks:
            return {"overall_risk_score": 5.0, "risk_level": "MEDIUM"}
        factors = ["relegation_risk", "viewership_decline", "regulatory_risk",
                   "geopolitical_risk", "financial_risk"]
        weights = [0.30, 0.20, 0.20, 0.20, 0.10]
        score   = sum(risks.get(f, 0.05) * w for f, w in zip(factors, weights)) * 10
        level   = "LOW" if score < 2.5 else "MEDIUM" if score < 5.0 else "HIGH"
        return {
            "property":            property_key,
            "overall_risk_score":  round(score, 2),
            "risk_level":          level,
            "risk_factors":        {k: risks.get(k, 0.0) for k in factors},
            "notes":               risks.get("notes", ""),
        }
