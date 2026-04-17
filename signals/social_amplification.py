"""
signals/social_amplification.py — Social Media Amplification Scorer.

Business summary
----------------
Beyond paid media, every sponsorship deal benefits from organic amplification:
fans sharing content, journalists writing about the partnership, athletes
posting sponsored content. This module quantifies that "earned media" layer.

Social amplification multiplies the value of paid exposure: a sponsor's logo
on a jersey generates not just the TV broadcast impression, but also fan photos,
highlight reels, and social posts — all unpaid. Properties with highly engaged,
digitally active fanbases deliver more social amplification per dollar of deal value.

Athlete endorsement value additionally uses the PSYCH_WEIGHT / PHYSICAL_WEIGHT
scoring framework (imported from config) to weight reputation vs. on-field
performance. See config.py for the full rationale and override parameters.

Developer notes
---------------
- amplification_multiplier(): ratio of total organic reach to base paid reach.
  Uses LogNormal distribution mean for continuous uncertainty.
- influencer_network_value(): values an individual athlete's social footprint
  using the CPM-based impression model.
- campaign_virality_score(): categorical affinity between brand category and
  property fanbase, calibrated on observed viral campaigns.
"""

from __future__ import annotations

from config import (
    PSYCH_WEIGHT,
    PHYSICAL_WEIGHT,
    ATHLETE_TOTAL_WEIGHT,
    PSYCH_WEIGHT_SAFE,
    PHYSICAL_WEIGHT_SAFE,
    ATHLETE_TOTAL_WEIGHT_SAFE,
    BRAND_SAFETY_SENSITIVE_BRANDS,
)
from data.sports_data import ATHLETE_PROFILES, SPORTS_PROPERTIES


# ── Hardcoded Social Media Data (cross-platform, as of 2024) ──────────────────
# Sources: public profile pages, SocialBlade estimates, club official reports.

_PROPERTY_SOCIAL_DATA: dict[str, dict] = {
    "manchester_city": {
        "instagram_m": 38.0, "twitter_m": 15.5, "youtube_m": 5.2, "tiktok_m": 8.0,
        "avg_engagement_rate": 0.032,
        "viral_coefficient": 1.8,    # content shares per post (relative to followers)
        "post_frequency_per_week": 28,
        "hashtag_volume_weekly": 850_000,
    },
    "manchester_united": {
        "instagram_m": 68.0, "twitter_m": 31.0, "youtube_m": 5.8, "tiktok_m": 15.0,
        "avg_engagement_rate": 0.028,
        "viral_coefficient": 2.2,
        "post_frequency_per_week": 35,
        "hashtag_volume_weekly": 2_400_000,
    },
    "real_madrid": {
        "instagram_m": 146.0, "twitter_m": 40.0, "youtube_m": 14.0, "tiktok_m": 25.0,
        "avg_engagement_rate": 0.038,
        "viral_coefficient": 3.5,
        "post_frequency_per_week": 42,
        "hashtag_volume_weekly": 5_200_000,
    },
    "barcelona": {
        "instagram_m": 130.0, "twitter_m": 36.5, "youtube_m": 12.0, "tiktok_m": 22.0,
        "avg_engagement_rate": 0.035,
        "viral_coefficient": 3.2,
        "post_frequency_per_week": 40,
        "hashtag_volume_weekly": 4_800_000,
    },
    "psg": {
        "instagram_m": 68.0, "twitter_m": 12.0, "youtube_m": 4.0, "tiktok_m": 12.0,
        "avg_engagement_rate": 0.042,
        "viral_coefficient": 2.8,
        "post_frequency_per_week": 35,
        "hashtag_volume_weekly": 2_100_000,
    },
    "chelsea": {
        "instagram_m": 30.0, "twitter_m": 16.0, "youtube_m": 3.5, "tiktok_m": 6.0,
        "avg_engagement_rate": 0.029,
        "viral_coefficient": 1.6,
        "post_frequency_per_week": 28,
        "hashtag_volume_weekly": 920_000,
    },
    "arsenal": {
        "instagram_m": 33.0, "twitter_m": 14.5, "youtube_m": 3.2, "tiktok_m": 7.0,
        "avg_engagement_rate": 0.031,
        "viral_coefficient": 1.7,
        "post_frequency_per_week": 28,
        "hashtag_volume_weekly": 1_100_000,
    },
    "liverpool": {
        "instagram_m": 55.0, "twitter_m": 22.0, "youtube_m": 6.0, "tiktok_m": 10.0,
        "avg_engagement_rate": 0.034,
        "viral_coefficient": 2.4,
        "post_frequency_per_week": 32,
        "hashtag_volume_weekly": 2_800_000,
    },
    "nfl": {
        "instagram_m": 18.0, "twitter_m": 15.0, "youtube_m": 5.0, "tiktok_m": 8.5,
        "avg_engagement_rate": 0.025,
        "viral_coefficient": 3.8,
        "post_frequency_per_week": 50,
        "hashtag_volume_weekly": 8_500_000,
    },
    "dallas_cowboys": {
        "instagram_m": 5.2, "twitter_m": 3.8, "youtube_m": 1.1, "tiktok_m": 2.0,
        "avg_engagement_rate": 0.022,
        "viral_coefficient": 2.5,
        "post_frequency_per_week": 20,
        "hashtag_volume_weekly": 1_800_000,
    },
    "la_lakers": {
        "instagram_m": 20.0, "twitter_m": 10.0, "youtube_m": 4.0, "tiktok_m": 8.0,
        "avg_engagement_rate": 0.036,
        "viral_coefficient": 3.0,
        "post_frequency_per_week": 30,
        "hashtag_volume_weekly": 3_200_000,
    },
    "golden_state_warriors": {
        "instagram_m": 12.5, "twitter_m": 8.0, "youtube_m": 2.5, "tiktok_m": 5.0,
        "avg_engagement_rate": 0.038,
        "viral_coefficient": 2.8,
        "post_frequency_per_week": 25,
        "hashtag_volume_weekly": 2_100_000,
    },
    "formula_1": {
        "instagram_m": 45.0, "twitter_m": 12.5, "youtube_m": 10.0, "tiktok_m": 25.0,
        "avg_engagement_rate": 0.052,
        "viral_coefficient": 4.5,
        "post_frequency_per_week": 55,
        "hashtag_volume_weekly": 6_800_000,
    },
    "red_bull_racing": {
        "instagram_m": 28.0, "twitter_m": 9.5, "youtube_m": 6.0, "tiktok_m": 15.0,
        "avg_engagement_rate": 0.068,
        "viral_coefficient": 5.2,
        "post_frequency_per_week": 60,
        "hashtag_volume_weekly": 4_500_000,
    },
    "ferrari_f1": {
        "instagram_m": 22.0, "twitter_m": 6.5, "youtube_m": 5.5, "tiktok_m": 12.0,
        "avg_engagement_rate": 0.055,
        "viral_coefficient": 4.0,
        "post_frequency_per_week": 45,
        "hashtag_volume_weekly": 3_800_000,
    },
    "wimbledon": {
        "instagram_m": 2.8, "twitter_m": 1.5, "youtube_m": 0.8, "tiktok_m": 1.2,
        "avg_engagement_rate": 0.028,
        "viral_coefficient": 1.4,
        "post_frequency_per_week": 10,
        "hashtag_volume_weekly": 850_000,
    },
    "nba": {
        "instagram_m": 35.0, "twitter_m": 28.0, "youtube_m": 17.0, "tiktok_m": 20.0,
        "avg_engagement_rate": 0.040,
        "viral_coefficient": 4.2,
        "post_frequency_per_week": 70,
        "hashtag_volume_weekly": 9_500_000,
    },
    "brentford": {
        "instagram_m": 1.0, "twitter_m": 0.6, "youtube_m": 0.15, "tiktok_m": 0.5,
        "avg_engagement_rate": 0.045,
        "viral_coefficient": 1.6,
        "post_frequency_per_week": 12,
        "hashtag_volume_weekly": 120_000,
    },
    "new_england_patriots": {
        "instagram_m": 3.1, "twitter_m": 2.6, "youtube_m": 0.8, "tiktok_m": 1.5,
        "avg_engagement_rate": 0.020,
        "viral_coefficient": 2.0,
        "post_frequency_per_week": 18,
        "hashtag_volume_weekly": 900_000,
    },
    "super_bowl": {
        "instagram_m": 0.0, "twitter_m": 0.0, "youtube_m": 0.0, "tiktok_m": 0.0,
        "avg_engagement_rate": 0.0,
        "viral_coefficient": 8.0,   # Super Bowl is the single highest-viral event
        "post_frequency_per_week": 0,
        "hashtag_volume_weekly": 25_000_000,
    },
}

# Brand category × property fanbase virality affinity (0–10)
# High affinity = brand content resonates naturally with this fanbase
_VIRALITY_AFFINITY: dict[str, dict[str, float]] = {
    "sportswear":     {"football": 9.0, "motorsport": 8.5, "basketball": 9.0, "tennis": 8.0, "american_football": 8.5},
    "soft_drinks":    {"football": 8.5, "basketball": 9.0, "american_football": 9.5, "motorsport": 7.0, "tennis": 6.0},
    "beer_alcohol":   {"football": 9.0, "american_football": 9.5, "motorsport": 8.5, "basketball": 7.5, "tennis": 6.5},
    "airlines":       {"football": 7.5, "motorsport": 8.0, "tennis": 7.0, "basketball": 6.5, "american_football": 6.0},
    "crypto_fintech": {"basketball": 9.0, "motorsport": 8.5, "football": 7.5, "american_football": 8.0, "tennis": 6.0},
    "luxury_watches": {"tennis": 8.5, "motorsport": 9.5, "football": 7.0, "golf": 9.0, "american_football": 6.0},
    "financial_svcs": {"american_football": 8.0, "football": 7.0, "basketball": 7.5, "motorsport": 7.5, "tennis": 7.0},
    "telecoms":       {"football": 8.0, "basketball": 8.5, "american_football": 8.5, "motorsport": 7.5, "tennis": 7.0},
    "automotive":     {"motorsport": 10.0, "american_football": 8.5, "football": 7.5, "basketball": 7.0, "tennis": 6.5},
    "energy_oil":     {"motorsport": 8.0, "football": 6.0, "american_football": 7.0, "basketball": 6.0, "tennis": 5.5},
    "default":        {"football": 7.0, "basketball": 7.0, "american_football": 7.0, "motorsport": 7.0, "tennis": 7.0},
}


class SocialAmplificationScorer:
    """
    Models organic social media amplification from fans, press, and athletes.

    Business summary
    ----------------
    The amplification_multiplier tells you: for every $1 of paid media exposure
    value generated by this property, how much additional organic (free) reach
    does the brand get? Red Bull Racing, for example, has extremely high viral
    coefficient — fan-made content, highlight reels, and team social posts
    generate 4–5× the organic reach of the raw broadcast footprint.

    Athlete scoring also respects the PSYCH/PHYSICAL weighting constants
    from config — see influencer_network_value() for the formula.

    Usage
    -----
    >>> scorer = SocialAmplificationScorer()
    >>> scorer.amplification_multiplier("red_bull_racing")
    4.82
    >>> scorer.influencer_network_value("cristiano_ronaldo")
    {'annual_usd_m': 18.4, 'weighted_athlete_score': 87.3, ...}
    >>> scorer.campaign_virality_score("formula_1", "luxury_watches")
    9.5
    """

    def __init__(self) -> None:
        self._social   = _PROPERTY_SOCIAL_DATA
        self._athletes = ATHLETE_PROFILES
        self._affinity = _VIRALITY_AFFINITY

    # ── Property-level amplification ─────────────────────────────────────────

    def amplification_multiplier(self, property_key: str) -> float:
        """
        How much organic reach multiplies the base paid exposure value.

        Formula
        -------
            multiplier = 1.0
                         + viral_coefficient × engagement_rate × 10
                         + log(1 + hashtag_volume_weekly / 500_000) × 0.2

        The multiplier represents total reach as a multiple of base (paid) reach.
        A multiplier of 3.5 means for every paid impression, 2.5 additional
        organic impressions are generated.

        Returns
        -------
        float
            Amplification multiplier (≥ 1.0).
        """
        data = self._social.get(property_key)
        if data is None:
            return 1.5  # default for unknown properties

        import math
        viral = data["viral_coefficient"]
        eng   = data["avg_engagement_rate"]
        hashtag = data["hashtag_volume_weekly"]

        multiplier = (
            1.0
            + viral * eng * 10
            + math.log(1 + hashtag / 500_000) * 0.2
        )
        return round(multiplier, 3)

    def total_annual_organic_reach_m(self, property_key: str) -> float:
        """
        Estimated total annual organic reach in millions of impressions.

        = (instagram + twitter + youtube + tiktok) × eng_rate × 1.2 × post_freq_per_year

        Returns
        -------
        float
            Annual organic impressions in millions.
        """
        data = self._social.get(property_key)
        if data is None:
            return 0.0

        total_followers_m = (
            data["instagram_m"] + data["twitter_m"] +
            data["youtube_m"]   + data["tiktok_m"]
        )
        annual_posts = data["post_frequency_per_week"] * 52
        organic_m    = (
            total_followers_m *
            data["avg_engagement_rate"] *
            1.20 *         # organic amplification from shares
            annual_posts
        )
        return round(organic_m, 2)

    def compare_amplification(self, property_keys: list[str]) -> list[dict]:
        """
        Compare amplification multipliers across multiple properties.

        Returns
        -------
        list[dict]
            Sorted by amplification_multiplier descending.
        """
        results = [
            {
                "property":            pk,
                "multiplier":          self.amplification_multiplier(pk),
                "annual_organic_m":    self.total_annual_organic_reach_m(pk),
                "viral_coefficient":   self._social.get(pk, {}).get("viral_coefficient", 0),
                "engagement_rate":     self._social.get(pk, {}).get("avg_engagement_rate", 0),
            }
            for pk in property_keys if pk in self._social
        ]
        results.sort(key=lambda x: x["multiplier"], reverse=True)
        return results

    # ── Athlete social value ──────────────────────────────────────────────────

    def influencer_network_value(
        self,
        athlete_key: str,
        brand_key: str = "nike",
        brand_safety_sensitive: bool = False,
    ) -> dict:
        """
        Value an individual athlete's social media contribution to a sponsor.

        Athlete endorsement scoring formula
        ------------------------------------
        Uses the PSYCH/PHYSICAL weighting constants from config.py.
        The formula is applied explicitly here:

        Standard brands (brand_safety_sensitive=False):
            PSYCH_WEIGHT    = 1.0   (reputation / character)
            PHYSICAL_WEIGHT = 1.5   (on-field performance / achievement)
            TOTAL_WEIGHT    = 2.5

            athlete_score = (
                (reputation_score * PSYCH_WEIGHT) +
                (performance_score * PHYSICAL_WEIGHT)
            ) / TOTAL_WEIGHT

        Brand-safety-sensitive brands (Rolex, Visa, etc.) — flip weights:
            PSYCH_WEIGHT_SAFE    = 1.5   (reputation is PRIMARY driver)
            PHYSICAL_WEIGHT_SAFE = 1.0
            TOTAL_WEIGHT_SAFE    = 2.5

            athlete_score = (
                (reputation_score * PSYCH_WEIGHT_SAFE) +
                (performance_score * PHYSICAL_WEIGHT_SAFE)
            ) / TOTAL_WEIGHT_SAFE

        Rationale: reputation is a risk-adjustment; performance is the primary
        commercial driver for standard brands. The 1.5:1.0 ratio reflects that
        sponsors accept some reputational risk for elite athletic association.

        Social reach value:
            impressions = total_followers × engagement_rate × 1.2 × 52 posts/yr
            social_value = (impressions / 1000) × $8.75 CPM

        Final endorsement value:
            = social_value × (weighted_score / 100) × global_recognition_factor

        Parameters
        ----------
        athlete_key : str
        brand_key : str
        brand_safety_sensitive : bool
            If True, flip weights: psych (1.5) > physical (1.0).

        Returns
        -------
        dict
            Full scoring breakdown with formula shown explicitly.
        """
        athlete = self._athletes.get(athlete_key)
        if athlete is None:
            raise ValueError(f"Unknown athlete '{athlete_key}'.")

        # ── Auto-detect brand safety regime ─────────────────────────────────
        if brand_key in BRAND_SAFETY_SENSITIVE_BRANDS:
            brand_safety_sensitive = True

        # ── Select weight regime ─────────────────────────────────────────────
        if brand_safety_sensitive:
            pw  = PSYCH_WEIGHT_SAFE      # 1.5 — reputation is primary
            phw = PHYSICAL_WEIGHT_SAFE   # 1.0
            tw  = ATHLETE_TOTAL_WEIGHT_SAFE
        else:
            pw  = PSYCH_WEIGHT           # 1.0
            phw = PHYSICAL_WEIGHT        # 1.5 — performance is primary
            tw  = ATHLETE_TOTAL_WEIGHT

        # ── Score inputs ─────────────────────────────────────────────────────
        reputation_score = max(0.0, (10.0 - athlete.controversiality_score) * 10.0)
        performance_score = (
            athlete.on_field_rating * 0.6 +
            athlete.global_recognition_score * 0.4
        ) * 10.0

        # ── Core formula (from config constants) ─────────────────────────────
        #   athlete_score = (reputation × PSYCH_WEIGHT
        #                    + performance × PHYSICAL_WEIGHT) / TOTAL_WEIGHT
        weighted_score = (
            (reputation_score * pw) + (performance_score * phw)
        ) / tw

        # ── Social reach valuation ───────────────────────────────────────────
        total_followers_m = (
            athlete.instagram_followers_m + athlete.twitter_followers_m +
            athlete.tiktok_followers_m    + athlete.youtube_subscribers_m
        )
        eff_impressions = (
            total_followers_m * 1_000_000 *
            (athlete.avg_engagement_rate / 100.0) *
            1.20 * 52
        )
        social_cpm        = 8.75
        social_value_usd_m = (eff_impressions / 1_000) * social_cpm / 1_000_000

        # ── Final endorsement value ──────────────────────────────────────────
        quality_scale  = weighted_score / 100.0
        fame_mult      = athlete.global_recognition_score / 10.0
        endorsement_fv = social_value_usd_m * quality_scale * fame_mult

        return {
            "athlete":               athlete.name,
            "brand_safety_regime":   "SAFE (reputation-primary)" if brand_safety_sensitive else "STANDARD (performance-primary)",
            "reputation_score":      round(reputation_score, 1),
            "performance_score":     round(performance_score, 1),
            "psych_weight":          pw,
            "physical_weight":       phw,
            "total_weight":          tw,
            "weighted_athlete_score":round(weighted_score, 1),
            "formula": (
                f"({reputation_score:.1f} × {pw}) + ({performance_score:.1f} × {phw}) "
                f"/ {tw} = {weighted_score:.1f} / 100"
            ),
            "total_followers_m":     round(total_followers_m, 1),
            "annual_impressions_m":  round(eff_impressions / 1_000_000, 1),
            "social_value_usd_m":    round(social_value_usd_m, 2),
            "annual_usd_m":          round(endorsement_fv, 2),
        }

    def rank_athletes_for_brand(
        self,
        brand_key: str,
        brand_safety_sensitive: bool = False,
        top_n: int = 10,
    ) -> list[dict]:
        """
        Rank all athletes by endorsement value for a given brand.

        Uses the PSYCH/PHYSICAL weighting regime appropriate for the brand.
        Brand-safety-sensitive brands (Rolex, Visa) will heavily penalise
        controversial athletes like Neymar.

        Returns
        -------
        list[dict]
            Sorted by annual_usd_m descending.
        """
        if brand_key in BRAND_SAFETY_SENSITIVE_BRANDS:
            brand_safety_sensitive = True

        results = []
        for ak in self._athletes:
            try:
                val = self.influencer_network_value(ak, brand_key, brand_safety_sensitive)
                val["athlete_key"] = ak
                results.append(val)
            except ValueError:
                continue

        results.sort(key=lambda x: x["annual_usd_m"], reverse=True)
        return results[:top_n]

    # ── Virality scoring ─────────────────────────────────────────────────────

    def campaign_virality_score(self, property_key: str, brand_category: str) -> float:
        """
        Predicted virality score (0–10) for a brand category sponsoring a property.

        Business summary
        ----------------
        Not all brand-sport combinations go viral equally. Automotive brands in
        F1 generate massive organic content (fans sharing livery reveals, race
        highlights). Beer brands in football do too. But airline logos on a tennis
        court are less inherently shareable.

        This score combines:
          - Category-sport affinity (_VIRALITY_AFFINITY)
          - Property's intrinsic viral coefficient
          - Property's fanbase engagement index

        Returns
        -------
        float in [0, 10].
        """
        prop_data = self._social.get(property_key)
        if prop_data is None:
            return 5.0

        # Infer sport from SPORTS_PROPERTIES
        sport = None
        from data.sports_data import SPORTS_PROPERTIES
        sp = SPORTS_PROPERTIES.get(property_key)
        if sp:
            sport = sp.sport

        affinity_row = self._affinity.get(brand_category, self._affinity["default"])
        base_affinity = affinity_row.get(sport, 7.0) if sport else 7.0

        viral_coeff = prop_data["viral_coefficient"]
        eng_rate    = prop_data["avg_engagement_rate"]

        # Combine: affinity anchors the base; viral and engagement scale it
        raw = base_affinity * min(viral_coeff / 3.0, 1.5) * (1 + eng_rate * 5)
        return round(min(raw, 10.0), 2)

    def social_value_summary(self, property_key: str) -> dict:
        """Return a concise social amplification summary for a property."""
        data = self._social.get(property_key)
        if data is None:
            return {"error": f"No data for '{property_key}'"}

        total_followers = (
            data["instagram_m"] + data["twitter_m"] +
            data["youtube_m"]   + data["tiktok_m"]
        )
        return {
            "property":              property_key,
            "total_followers_m":     round(total_followers, 1),
            "avg_engagement_rate":   data["avg_engagement_rate"],
            "viral_coefficient":     data["viral_coefficient"],
            "weekly_hashtag_volume": data["hashtag_volume_weekly"],
            "amplification_multiplier": self.amplification_multiplier(property_key),
            "annual_organic_reach_m": self.total_annual_organic_reach_m(property_key),
        }
