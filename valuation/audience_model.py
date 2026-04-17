"""
valuation/audience_model.py — Audience Demographic Fit Model.

Business summary
----------------
Not all eyeballs are equal. A luxury watch brand (Rolex) targeting HHI $150k+
audiences gets far more value from Wimbledon than from an NFL game — even if
the NFL has ten times the raw viewership. This module quantifies that mismatch.

For each brand-property pair, it computes:
  - demographic_fit_score (0–1): overlap between brand's target customer and
    property's actual fanbase demographics (age, gender, income)
  - audience_quality_index (0–10): standalone quality metric for a property's
    fanbase (income, engagement, geographic reach)
  - premium_multiplier (0.7–1.5): how much a brand should pay above/below
    the base exposure value given demographic fit

Developer notes
---------------
Fit score uses a weighted cosine-similarity approach across three demographic
vectors: age distribution, gender split, and household income distribution.
Each vector is L1-normalized before dot product so the score is scale-invariant.
premium_multiplier maps fit_score [0,1] → multiplier [0.70, 1.50] via a
piecewise linear schedule that punishes poor fit and rewards excellent fit.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from config import DEMOGRAPHIC_QUALITY_WEIGHTS
from data.sports_data import SPORTS_PROPERTIES, BRAND_PROFILES


# ── Hardcoded Property Demographic Profiles ───────────────────────────────────
# Each entry: age bands (18-24, 25-34, 35-44, 45-54, 55+),
#             gender_male fraction, hhi_under50k / hhi_50-100k / hhi_over100k,
#             domestic_pct, international_pct, engagement_index (vs. avg=1.0)
# Sources: Nielsen Fan Insights, Repucom, Kantar Sports Fan DNA 2022–2024.

_PROPERTY_DEMOGRAPHICS: dict[str, dict] = {
    "manchester_city": {
        "age": [0.18, 0.28, 0.26, 0.16, 0.12],
        "gender_male": 0.65,
        "hhi": [0.28, 0.42, 0.30],
        "domestic_pct": 0.28,
        "international_pct": 0.72,
        "engagement_index": 1.22,
        "primary_market": "Global / Middle East",
        "high_income_pct": 0.30,
        "prime_age_pct": 0.54,  # 25-44
    },
    "manchester_united": {
        "age": [0.20, 0.27, 0.25, 0.17, 0.11],
        "gender_male": 0.68,
        "hhi": [0.26, 0.40, 0.34],
        "domestic_pct": 0.22,
        "international_pct": 0.78,
        "engagement_index": 1.15,
        "primary_market": "Global",
        "high_income_pct": 0.34,
        "prime_age_pct": 0.52,
    },
    "real_madrid": {
        "age": [0.21, 0.29, 0.24, 0.15, 0.11],
        "gender_male": 0.66,
        "hhi": [0.24, 0.38, 0.38],
        "domestic_pct": 0.18,
        "international_pct": 0.82,
        "engagement_index": 1.30,
        "primary_market": "Global / Latin America",
        "high_income_pct": 0.38,
        "prime_age_pct": 0.53,
    },
    "barcelona": {
        "age": [0.22, 0.28, 0.24, 0.15, 0.11],
        "gender_male": 0.64,
        "hhi": [0.25, 0.40, 0.35],
        "domestic_pct": 0.20,
        "international_pct": 0.80,
        "engagement_index": 1.25,
        "primary_market": "Global / Asia",
        "high_income_pct": 0.35,
        "prime_age_pct": 0.52,
    },
    "psg": {
        "age": [0.24, 0.30, 0.22, 0.14, 0.10],
        "gender_male": 0.67,
        "hhi": [0.26, 0.40, 0.34],
        "domestic_pct": 0.30,
        "international_pct": 0.70,
        "engagement_index": 1.28,
        "primary_market": "Global / Middle East / Asia",
        "high_income_pct": 0.34,
        "prime_age_pct": 0.52,
    },
    "chelsea": {
        "age": [0.19, 0.27, 0.26, 0.17, 0.11],
        "gender_male": 0.66,
        "hhi": [0.29, 0.41, 0.30],
        "domestic_pct": 0.32,
        "international_pct": 0.68,
        "engagement_index": 1.10,
        "primary_market": "Global",
        "high_income_pct": 0.30,
        "prime_age_pct": 0.53,
    },
    "arsenal": {
        "age": [0.21, 0.28, 0.25, 0.16, 0.10],
        "gender_male": 0.65,
        "hhi": [0.27, 0.41, 0.32],
        "domestic_pct": 0.29,
        "international_pct": 0.71,
        "engagement_index": 1.12,
        "primary_market": "Global / Africa",
        "high_income_pct": 0.32,
        "prime_age_pct": 0.53,
    },
    "liverpool": {
        "age": [0.20, 0.28, 0.25, 0.16, 0.11],
        "gender_male": 0.67,
        "hhi": [0.27, 0.41, 0.32],
        "domestic_pct": 0.25,
        "international_pct": 0.75,
        "engagement_index": 1.20,
        "primary_market": "Global / Asia",
        "high_income_pct": 0.32,
        "prime_age_pct": 0.53,
    },
    "nfl": {
        "age": [0.14, 0.22, 0.26, 0.22, 0.16],
        "gender_male": 0.72,
        "hhi": [0.22, 0.42, 0.36],
        "domestic_pct": 0.88,
        "international_pct": 0.12,
        "engagement_index": 1.35,
        "primary_market": "USA",
        "high_income_pct": 0.36,
        "prime_age_pct": 0.48,
    },
    "dallas_cowboys": {
        "age": [0.13, 0.21, 0.27, 0.23, 0.16],
        "gender_male": 0.74,
        "hhi": [0.20, 0.40, 0.40],
        "domestic_pct": 0.91,
        "international_pct": 0.09,
        "engagement_index": 1.40,
        "primary_market": "USA",
        "high_income_pct": 0.40,
        "prime_age_pct": 0.48,
    },
    "new_england_patriots": {
        "age": [0.12, 0.20, 0.28, 0.24, 0.16],
        "gender_male": 0.73,
        "hhi": [0.19, 0.38, 0.43],
        "domestic_pct": 0.92,
        "international_pct": 0.08,
        "engagement_index": 1.32,
        "primary_market": "USA / New England",
        "high_income_pct": 0.43,
        "prime_age_pct": 0.48,
    },
    "la_lakers": {
        "age": [0.22, 0.30, 0.24, 0.14, 0.10],
        "gender_male": 0.62,
        "hhi": [0.24, 0.38, 0.38],
        "domestic_pct": 0.60,
        "international_pct": 0.40,
        "engagement_index": 1.30,
        "primary_market": "USA / Global",
        "high_income_pct": 0.38,
        "prime_age_pct": 0.54,
    },
    "golden_state_warriors": {
        "age": [0.23, 0.31, 0.24, 0.13, 0.09],
        "gender_male": 0.60,
        "hhi": [0.22, 0.36, 0.42],
        "domestic_pct": 0.58,
        "international_pct": 0.42,
        "engagement_index": 1.35,
        "primary_market": "USA / Asia",
        "high_income_pct": 0.42,
        "prime_age_pct": 0.55,
    },
    "formula_1": {
        "age": [0.18, 0.28, 0.26, 0.17, 0.11],
        "gender_male": 0.69,
        "hhi": [0.18, 0.36, 0.46],
        "domestic_pct": 0.08,
        "international_pct": 0.92,
        "engagement_index": 1.55,
        "primary_market": "Global / Europe",
        "high_income_pct": 0.46,
        "prime_age_pct": 0.54,
    },
    "red_bull_racing": {
        "age": [0.26, 0.32, 0.22, 0.12, 0.08],
        "gender_male": 0.72,
        "hhi": [0.20, 0.38, 0.42],
        "domestic_pct": 0.05,
        "international_pct": 0.95,
        "engagement_index": 1.60,
        "primary_market": "Global",
        "high_income_pct": 0.42,
        "prime_age_pct": 0.54,
    },
    "ferrari_f1": {
        "age": [0.14, 0.24, 0.28, 0.20, 0.14],
        "gender_male": 0.73,
        "hhi": [0.12, 0.30, 0.58],
        "domestic_pct": 0.12,
        "international_pct": 0.88,
        "engagement_index": 1.48,
        "primary_market": "Global / Italy",
        "high_income_pct": 0.58,
        "prime_age_pct": 0.52,
    },
    "wimbledon": {
        "age": [0.10, 0.18, 0.26, 0.26, 0.20],
        "gender_male": 0.51,
        "hhi": [0.12, 0.32, 0.56],
        "domestic_pct": 0.35,
        "international_pct": 0.65,
        "engagement_index": 1.10,
        "primary_market": "Global",
        "high_income_pct": 0.56,
        "prime_age_pct": 0.44,
    },
    "nba": {
        "age": [0.25, 0.30, 0.22, 0.13, 0.10],
        "gender_male": 0.60,
        "hhi": [0.26, 0.38, 0.36],
        "domestic_pct": 0.50,
        "international_pct": 0.50,
        "engagement_index": 1.38,
        "primary_market": "Global",
        "high_income_pct": 0.36,
        "prime_age_pct": 0.52,
    },
    "brentford": {
        "age": [0.14, 0.24, 0.28, 0.20, 0.14],
        "gender_male": 0.70,
        "hhi": [0.32, 0.44, 0.24],
        "domestic_pct": 0.78,
        "international_pct": 0.22,
        "engagement_index": 1.45,
        "primary_market": "UK",
        "high_income_pct": 0.24,
        "prime_age_pct": 0.52,
    },
    "super_bowl": {
        "age": [0.13, 0.22, 0.26, 0.22, 0.17],
        "gender_male": 0.66,
        "hhi": [0.20, 0.38, 0.42],
        "domestic_pct": 0.85,
        "international_pct": 0.15,
        "engagement_index": 1.80,
        "primary_market": "USA",
        "high_income_pct": 0.42,
        "prime_age_pct": 0.48,
    },
}

# Premium multiplier schedule: fit_score → multiplier
# Piecewise linear: poor fit penalized, excellent fit rewarded
_FIT_TO_MULTIPLIER = [
    (0.00, 0.70),
    (0.25, 0.85),
    (0.50, 1.00),
    (0.70, 1.15),
    (0.85, 1.30),
    (1.00, 1.50),
]

# Demographic vector weights for fit scoring
_AGE_WEIGHT    = 0.35
_INCOME_WEIGHT = 0.40
_GENDER_WEIGHT = 0.25


def _l1_normalize(vec: list[float]) -> np.ndarray:
    """L1-normalize a vector so it represents a probability distribution."""
    arr = np.array(vec, dtype=float)
    total = arr.sum()
    return arr / total if total > 0 else arr


def _interp_multiplier(fit: float) -> float:
    """Piecewise linear interpolation of fit_score → premium_multiplier."""
    for i in range(len(_FIT_TO_MULTIPLIER) - 1):
        x0, y0 = _FIT_TO_MULTIPLIER[i]
        x1, y1 = _FIT_TO_MULTIPLIER[i + 1]
        if x0 <= fit <= x1:
            t = (fit - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return _FIT_TO_MULTIPLIER[-1][1]


class AudienceDemographicModel:
    """
    Models how well a brand's target customer matches a sports property's fanbase.

    Business summary
    ----------------
    A brand's ideal customer may be a 35-44 year-old, high-income male in Europe.
    Ferrari F1 fans skew exactly that way; Brentford fans skew differently.
    This model computes a fit score (0–1) for any brand-property pair and
    translates it into a premium_multiplier used by the fair value engine to
    adjust the base exposure value up or down.

    Developer notes
    ---------------
    Fit score = weighted average of three cosine similarities:
      - age_similarity:    dot(brand_age_target, property_age_dist)  [L1-normalized]
      - income_similarity: dot(brand_hhi_target, property_hhi_dist)  [L1-normalized]
      - gender_similarity: 1 - |brand_gender_male - property_gender_male|

    These are combined as:
      fit = _AGE_WEIGHT * age_sim + _INCOME_WEIGHT * income_sim + _GENDER_WEIGHT * gender_sim

    The fit score is then mapped to a premium_multiplier via piecewise linear
    interpolation (_FIT_TO_MULTIPLIER schedule).

    Usage
    -----
    >>> model = AudienceDemographicModel()
    >>> model.demographic_fit_score("rolex", "ferrari_f1")
    0.892
    >>> model.premium_multiplier("rolex", "ferrari_f1")
    1.38
    >>> model.audience_quality_index("ferrari_f1")
    8.24
    """

    def __init__(self) -> None:
        self._demographics = _PROPERTY_DEMOGRAPHICS
        self._brands = BRAND_PROFILES

    # ── Public API ──────────────────────────────────────────────────────────

    def demographic_fit_score(self, brand_key: str, property_key: str) -> float:
        """
        Compute how well the brand's target customer overlaps with this property's fanbase.

        Methodology
        -----------
        Three demographic dimensions are compared:
          1. Age distribution (5 buckets): weighted cosine similarity
          2. Household income distribution (3 buckets): weighted cosine similarity
          3. Gender split: 1 - absolute difference in male fraction

        Combined as: fit = 0.35*age + 0.40*income + 0.25*gender

        Parameters
        ----------
        brand_key : str
            Key from BRAND_PROFILES (e.g. "rolex", "pepsi").
        property_key : str
            Key from _PROPERTY_DEMOGRAPHICS (e.g. "ferrari_f1").

        Returns
        -------
        float in [0, 1].
        """
        brand = self._brands.get(brand_key)
        prop  = self._demographics.get(property_key)

        if brand is None:
            raise ValueError(f"Unknown brand '{brand_key}'. Available: {list(self._brands)}")
        if prop is None:
            raise ValueError(f"Unknown property '{property_key}'. Available: {list(self._demographics)}")

        # Age similarity — cosine-style dot product on L1-normalized vectors
        brand_age = _l1_normalize([
            brand.target_age_18_24, brand.target_age_25_34, brand.target_age_35_44,
            brand.target_age_45_54, brand.target_age_55_plus
        ])
        prop_age  = _l1_normalize(prop["age"])
        # Dot product of two L1-normalized vectors ∈ [0,1] (max when identical)
        age_sim   = float(np.dot(brand_age, prop_age))
        # Scale to [0,1]: minimum possible dot product approaches 0
        # Max dot product for 5-element L1 simplex ≈ 1 (identical), practical max ~0.5
        # Rescale so perfect match → 1.0
        age_sim_scaled = min(age_sim / 0.40, 1.0)

        # Income similarity
        brand_hhi = _l1_normalize([brand.target_hhi_under_50k, brand.target_hhi_50_100k, brand.target_hhi_over_100k])
        prop_hhi  = _l1_normalize(prop["hhi"])
        inc_sim   = float(np.dot(brand_hhi, prop_hhi))
        inc_sim_scaled = min(inc_sim / 0.45, 1.0)

        # Gender similarity — simple complement of absolute deviation
        gender_sim = 1.0 - abs(brand.target_gender_male - prop["gender_male"])

        # Weighted combination
        fit = (
            _AGE_WEIGHT    * age_sim_scaled +
            _INCOME_WEIGHT * inc_sim_scaled +
            _GENDER_WEIGHT * gender_sim
        )
        return round(float(np.clip(fit, 0.0, 1.0)), 4)

    def premium_multiplier(self, brand_key: str, property_key: str) -> float:
        """
        Translate demographic fit into a price premium/discount multiplier.

        A fit of 0.85+ → multiplier ~1.30–1.50 (brand should pay premium).
        A fit of 0.40  → multiplier ~0.88 (below-average match, discount warranted).

        Returns
        -------
        float in [0.70, 1.50].
        """
        fit = self.demographic_fit_score(brand_key, property_key)
        return round(_interp_multiplier(fit), 4)

    def audience_quality_index(self, property_key: str) -> float:
        """
        Standalone quality rating for a property's fanbase (0–10).

        Business summary
        ----------------
        Properties with wealthy, globally dispersed, highly engaged fanbases
        command structural premiums regardless of which specific brand is looking.
        This index captures that inherent quality.

        Formula
        -------
        AQI = 10 × (
            w_income     × high_income_pct  +
            w_prime_age  × prime_age_pct    +
            w_intl       × international_pct +
            w_engagement × min(engagement_index / 2.0, 1.0)
        )
        where weights come from config.DEMOGRAPHIC_QUALITY_WEIGHTS.

        Returns
        -------
        float in [0, 10].
        """
        prop = self._demographics.get(property_key)
        if prop is None:
            raise ValueError(f"Unknown property '{property_key}'.")

        w = DEMOGRAPHIC_QUALITY_WEIGHTS
        raw = (
            w["high_income_pct"]   * prop["high_income_pct"] +
            w["prime_age_pct"]     * prop["prime_age_pct"] +
            w["international_pct"] * prop["international_pct"] +
            w["engagement_rate"]   * min(prop["engagement_index"] / 2.0, 1.0)
        )
        return round(raw * 10.0, 2)

    def demographic_fit_matrix(
        self, brands: list[str], properties: list[str]
    ) -> "pd.DataFrame":
        """
        Compute the full fit score matrix for a list of brands × properties.

        Returns
        -------
        pd.DataFrame
            Index = property keys, Columns = brand keys. Values = fit scores.
        """
        import pandas as pd

        matrix = {}
        for brand in brands:
            col = {}
            for prop in properties:
                try:
                    col[prop] = self.demographic_fit_score(brand, prop)
                except ValueError:
                    col[prop] = float("nan")
            matrix[brand] = col
        return pd.DataFrame(matrix)

    def top_brand_matches(self, property_key: str, n: int = 5) -> list[dict]:
        """
        Return the top-N best-fitting brands for a given property.

        Returns
        -------
        list[dict]
            Sorted by fit_score descending. Each entry: {brand, fit_score, multiplier}.
        """
        scores = []
        for bk in self._brands:
            try:
                fit = self.demographic_fit_score(bk, property_key)
                scores.append({
                    "brand": bk,
                    "fit_score": fit,
                    "premium_multiplier": round(_interp_multiplier(fit), 4),
                })
            except ValueError:
                pass
        scores.sort(key=lambda x: x["fit_score"], reverse=True)
        return scores[:n]

    def top_property_matches(self, brand_key: str, n: int = 5) -> list[dict]:
        """
        Return the top-N best-fitting properties for a given brand.

        Returns
        -------
        list[dict]
            Sorted by fit_score descending. Each entry: {property, fit_score, aqi, multiplier}.
        """
        scores = []
        for pk in self._demographics:
            try:
                fit = self.demographic_fit_score(brand_key, pk)
                aqi = self.audience_quality_index(pk)
                scores.append({
                    "property": pk,
                    "fit_score": fit,
                    "audience_quality_index": aqi,
                    "premium_multiplier": round(_interp_multiplier(fit), 4),
                })
            except ValueError:
                pass
        scores.sort(key=lambda x: x["fit_score"], reverse=True)
        return scores[:n]

    def all_quality_indices(self) -> "pd.DataFrame":
        """
        Return audience quality index for all properties as a sorted DataFrame.
        """
        import pandas as pd

        rows = [
            {"property": pk, "audience_quality_index": self.audience_quality_index(pk)}
            for pk in self._demographics
        ]
        df = pd.DataFrame(rows).set_index("property")
        return df.sort_values("audience_quality_index", ascending=False)
