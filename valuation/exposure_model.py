"""
valuation/exposure_model.py — Sponsorship Exposure Value Model.

Business summary
----------------
This module answers the question: "What is the fair media/exposure value of putting
our logo on this sports property?"

It models four exposure channels — in-stadium, broadcast, digital/social, and
jersey/kit — and aggregates them into an annual USD exposure value using
impression-based CPM pricing. Think of it as a bottom-up ad-rate card for
sports sponsorship.

The `batch_score_all_properties()` method scores every property simultaneously
using vectorized pandas/numpy operations, producing a ranked comparison table
in milliseconds.

Developer notes
---------------
- CPM benchmarks and visibility coefficients live in config.py (centralised).
- Per-property exposure parameters (_PROPERTY_EXPOSURE_PARAMS) are hardcoded from
  public club media guides, Kantar Sports reports, and Nielsen Sports benchmarks.
- `batch_score_all_properties()` constructs numpy arrays for each channel and
  computes all impressions + CPM products as element-wise array operations —
  no Python for-loops over properties.

Methodology references
  - Nielsen Sports Global Sponsorship Benchmarks 2023
  - SportsPro Media "The True Value of Sports Sponsorship" (2022)
  - Repucom (now Nielsen) jersey value methodology
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import CPM_BENCHMARKS, VISIBILITY_COEFFICIENTS
from data.sports_data import SPORTS_PROPERTIES


# ── Per-property hardcoded exposure parameters ────────────────────────────────
# Values sourced from public club media guides, Kantar Sports, and industry reports.

_PROPERTY_EXPOSURE_PARAMS: dict[str, dict] = {
    "manchester_city": {
        "in_stadium": {"attendance": 830_000, "visibility_coefficient": 0.85, "dwell_time_hours": 3.5, "cpm_override": None},
        "broadcast": {"avg_viewership_m": 2.1, "broadcast_frequency": 38, "avg_screen_time_seconds": 28, "cpm_override": None, "global_reach_factor": 1.35},
        "digital_social": {"follower_count_m": 68.0, "engagement_rate": 0.032, "post_frequency_per_year": 1200, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 3420, "camera_captures_per_minute": 4.2, "jersey_cpm_override": None, "global_reach_factor": 1.40},
    },
    "manchester_united": {
        "in_stadium": {"attendance": 870_000, "visibility_coefficient": 0.90, "dwell_time_hours": 3.5, "cpm_override": None},
        "broadcast": {"avg_viewership_m": 3.5, "broadcast_frequency": 38, "avg_screen_time_seconds": 32, "cpm_override": None, "global_reach_factor": 1.65},
        "digital_social": {"follower_count_m": 162.0, "engagement_rate": 0.028, "post_frequency_per_year": 1400, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 3420, "camera_captures_per_minute": 5.8, "jersey_cpm_override": None, "global_reach_factor": 1.80},
    },
    "real_madrid": {
        "in_stadium": {"attendance": 750_000, "visibility_coefficient": 0.88, "dwell_time_hours": 3.2, "cpm_override": None},
        "broadcast": {"avg_viewership_m": 4.2, "broadcast_frequency": 38, "avg_screen_time_seconds": 34, "cpm_override": None, "global_reach_factor": 1.75},
        "digital_social": {"follower_count_m": 320.0, "engagement_rate": 0.038, "post_frequency_per_year": 1500, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 3420, "camera_captures_per_minute": 6.5, "jersey_cpm_override": None, "global_reach_factor": 1.90},
    },
    "barcelona": {
        "in_stadium": {"attendance": 820_000, "visibility_coefficient": 0.87, "dwell_time_hours": 3.2, "cpm_override": None},
        "broadcast": {"avg_viewership_m": 3.9, "broadcast_frequency": 38, "avg_screen_time_seconds": 32, "cpm_override": None, "global_reach_factor": 1.70},
        "digital_social": {"follower_count_m": 295.0, "engagement_rate": 0.035, "post_frequency_per_year": 1500, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 3420, "camera_captures_per_minute": 6.2, "jersey_cpm_override": None, "global_reach_factor": 1.85},
    },
    "psg": {
        "in_stadium": {"attendance": 530_000, "visibility_coefficient": 0.82, "dwell_time_hours": 3.0, "cpm_override": None},
        "broadcast": {"avg_viewership_m": 1.4, "broadcast_frequency": 38, "avg_screen_time_seconds": 24, "cpm_override": None, "global_reach_factor": 1.30},
        "digital_social": {"follower_count_m": 120.0, "engagement_rate": 0.042, "post_frequency_per_year": 1300, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 3420, "camera_captures_per_minute": 4.5, "jersey_cpm_override": None, "global_reach_factor": 1.45},
    },
    "chelsea": {
        "in_stadium": {"attendance": 740_000, "visibility_coefficient": 0.80, "dwell_time_hours": 3.5, "cpm_override": None},
        "broadcast": {"avg_viewership_m": 2.0, "broadcast_frequency": 38, "avg_screen_time_seconds": 26, "cpm_override": None, "global_reach_factor": 1.25},
        "digital_social": {"follower_count_m": 68.0, "engagement_rate": 0.029, "post_frequency_per_year": 1100, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 3420, "camera_captures_per_minute": 4.0, "jersey_cpm_override": None, "global_reach_factor": 1.30},
    },
    "arsenal": {
        "in_stadium": {"attendance": 820_000, "visibility_coefficient": 0.83, "dwell_time_hours": 3.5, "cpm_override": None},
        "broadcast": {"avg_viewership_m": 1.9, "broadcast_frequency": 38, "avg_screen_time_seconds": 26, "cpm_override": None, "global_reach_factor": 1.22},
        "digital_social": {"follower_count_m": 65.0, "engagement_rate": 0.031, "post_frequency_per_year": 1100, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 3420, "camera_captures_per_minute": 4.1, "jersey_cpm_override": None, "global_reach_factor": 1.28},
    },
    "liverpool": {
        "in_stadium": {"attendance": 850_000, "visibility_coefficient": 0.87, "dwell_time_hours": 3.5, "cpm_override": None},
        "broadcast": {"avg_viewership_m": 2.8, "broadcast_frequency": 38, "avg_screen_time_seconds": 30, "cpm_override": None, "global_reach_factor": 1.50},
        "digital_social": {"follower_count_m": 115.0, "engagement_rate": 0.034, "post_frequency_per_year": 1300, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 3420, "camera_captures_per_minute": 5.2, "jersey_cpm_override": None, "global_reach_factor": 1.55},
    },
    "nfl": {
        "in_stadium": {"attendance": 17_500_000, "visibility_coefficient": 0.88, "dwell_time_hours": 4.0, "cpm_override": 22.0},
        "broadcast": {"avg_viewership_m": 17.5, "broadcast_frequency": 285, "avg_screen_time_seconds": 30, "cpm_override": 55.0, "global_reach_factor": 1.0},
        "digital_social": {"follower_count_m": 42.0, "engagement_rate": 0.025, "post_frequency_per_year": 2000, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 1024, "camera_captures_per_minute": 8.0, "jersey_cpm_override": 120.0, "global_reach_factor": 1.0},
    },
    "dallas_cowboys": {
        "in_stadium": {"attendance": 720_000, "visibility_coefficient": 0.92, "dwell_time_hours": 4.0, "cpm_override": 22.0},
        "broadcast": {"avg_viewership_m": 23.0, "broadcast_frequency": 16, "avg_screen_time_seconds": 35, "cpm_override": 55.0, "global_reach_factor": 1.05},
        "digital_social": {"follower_count_m": 12.0, "engagement_rate": 0.022, "post_frequency_per_year": 800, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 960, "camera_captures_per_minute": 9.5, "jersey_cpm_override": 120.0, "global_reach_factor": 1.05},
    },
    "new_england_patriots": {
        "in_stadium": {"attendance": 530_000, "visibility_coefficient": 0.88, "dwell_time_hours": 4.0, "cpm_override": 22.0},
        "broadcast": {"avg_viewership_m": 19.5, "broadcast_frequency": 16, "avg_screen_time_seconds": 32, "cpm_override": 55.0, "global_reach_factor": 1.02},
        "digital_social": {"follower_count_m": 7.5, "engagement_rate": 0.020, "post_frequency_per_year": 700, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 960, "camera_captures_per_minute": 8.5, "jersey_cpm_override": 120.0, "global_reach_factor": 1.02},
    },
    "la_lakers": {
        "in_stadium": {"attendance": 780_000, "visibility_coefficient": 0.88, "dwell_time_hours": 2.5, "cpm_override": 20.0},
        "broadcast": {"avg_viewership_m": 2.8, "broadcast_frequency": 82, "avg_screen_time_seconds": 22, "cpm_override": 42.0, "global_reach_factor": 1.45},
        "digital_social": {"follower_count_m": 42.0, "engagement_rate": 0.036, "post_frequency_per_year": 1200, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 3936, "camera_captures_per_minute": 6.0, "jersey_cpm_override": None, "global_reach_factor": 1.40},
    },
    "golden_state_warriors": {
        "in_stadium": {"attendance": 750_000, "visibility_coefficient": 0.85, "dwell_time_hours": 2.5, "cpm_override": 20.0},
        "broadcast": {"avg_viewership_m": 2.6, "broadcast_frequency": 82, "avg_screen_time_seconds": 22, "cpm_override": 42.0, "global_reach_factor": 1.40},
        "digital_social": {"follower_count_m": 28.0, "engagement_rate": 0.038, "post_frequency_per_year": 1100, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 3936, "camera_captures_per_minute": 5.8, "jersey_cpm_override": None, "global_reach_factor": 1.38},
    },
    "formula_1": {
        "in_stadium": {"attendance": 6_500_000, "visibility_coefficient": 0.75, "dwell_time_hours": 6.0, "cpm_override": 16.0},
        "broadcast": {"avg_viewership_m": 70.0, "broadcast_frequency": 23, "avg_screen_time_seconds": 40, "cpm_override": 38.0, "global_reach_factor": 2.0},
        "digital_social": {"follower_count_m": 83.0, "engagement_rate": 0.052, "post_frequency_per_year": 2000, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 1380, "camera_captures_per_minute": 12.0, "jersey_cpm_override": 110.0, "global_reach_factor": 2.0},
    },
    "red_bull_racing": {
        "in_stadium": {"attendance": 1_500_000, "visibility_coefficient": 0.72, "dwell_time_hours": 6.0, "cpm_override": 16.0},
        "broadcast": {"avg_viewership_m": 55.0, "broadcast_frequency": 23, "avg_screen_time_seconds": 42, "cpm_override": 38.0, "global_reach_factor": 1.90},
        "digital_social": {"follower_count_m": 52.0, "engagement_rate": 0.068, "post_frequency_per_year": 2500, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 1380, "camera_captures_per_minute": 14.0, "jersey_cpm_override": 110.0, "global_reach_factor": 1.90},
    },
    "ferrari_f1": {
        "in_stadium": {"attendance": 1_200_000, "visibility_coefficient": 0.72, "dwell_time_hours": 6.0, "cpm_override": 16.0},
        "broadcast": {"avg_viewership_m": 48.0, "broadcast_frequency": 23, "avg_screen_time_seconds": 40, "cpm_override": 38.0, "global_reach_factor": 1.80},
        "digital_social": {"follower_count_m": 38.0, "engagement_rate": 0.055, "post_frequency_per_year": 2000, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 1380, "camera_captures_per_minute": 13.0, "jersey_cpm_override": 110.0, "global_reach_factor": 1.80},
    },
    "super_bowl": {
        "in_stadium": {"attendance": 70_000, "visibility_coefficient": 0.95, "dwell_time_hours": 5.0, "cpm_override": 50.0},
        "broadcast": {"avg_viewership_m": 115.0, "broadcast_frequency": 1, "avg_screen_time_seconds": 30, "cpm_override": 180.0, "global_reach_factor": 1.0},
        "digital_social": {"follower_count_m": 0.0, "engagement_rate": 0.0, "post_frequency_per_year": 0, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 60, "camera_captures_per_minute": 15.0, "jersey_cpm_override": 200.0, "global_reach_factor": 1.0},
    },
    "wimbledon": {
        "in_stadium": {"attendance": 500_000, "visibility_coefficient": 0.80, "dwell_time_hours": 5.0, "cpm_override": None},
        "broadcast": {"avg_viewership_m": 4.5, "broadcast_frequency": 14, "avg_screen_time_seconds": 20, "cpm_override": 45.0, "global_reach_factor": 1.60},
        "digital_social": {"follower_count_m": 5.0, "engagement_rate": 0.028, "post_frequency_per_year": 400, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 4200, "camera_captures_per_minute": 3.0, "jersey_cpm_override": None, "global_reach_factor": 1.60},
    },
    "nba": {
        "in_stadium": {"attendance": 22_000_000, "visibility_coefficient": 0.85, "dwell_time_hours": 2.5, "cpm_override": 20.0},
        "broadcast": {"avg_viewership_m": 1.6, "broadcast_frequency": 1230, "avg_screen_time_seconds": 22, "cpm_override": 42.0, "global_reach_factor": 1.30},
        "digital_social": {"follower_count_m": 75.0, "engagement_rate": 0.040, "post_frequency_per_year": 3000, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 59040, "camera_captures_per_minute": 6.0, "jersey_cpm_override": None, "global_reach_factor": 1.30},
    },
    "brentford": {
        "in_stadium": {"attendance": 210_000, "visibility_coefficient": 0.75, "dwell_time_hours": 3.5, "cpm_override": None},
        "broadcast": {"avg_viewership_m": 0.85, "broadcast_frequency": 38, "avg_screen_time_seconds": 18, "cpm_override": None, "global_reach_factor": 1.05},
        "digital_social": {"follower_count_m": 2.2, "engagement_rate": 0.045, "post_frequency_per_year": 700, "cpm_override": None},
        "jersey_kit": {"minutes_played_per_year": 3420, "camera_captures_per_minute": 2.5, "jersey_cpm_override": None, "global_reach_factor": 1.05},
    },
}

# Asset type → channels that apply + visibility coefficient key
_ASSET_CHANNEL_MAP: dict[str, dict] = {
    "jersey":          {"channels": ["in_stadium", "broadcast", "jersey_kit"], "visibility_key": "jersey_front"},
    "jersey_sleeve":   {"channels": ["in_stadium", "broadcast", "jersey_kit"], "visibility_key": "jersey_sleeve"},
    "jersey_patch":    {"channels": ["in_stadium", "broadcast", "jersey_kit"], "visibility_key": "jersey_sleeve"},
    "stadium_naming":  {"channels": ["in_stadium", "broadcast", "digital_social"], "visibility_key": "stadium_naming"},
    "title_sponsor":   {"channels": ["in_stadium", "broadcast", "digital_social"], "visibility_key": "broadcast_title"},
    "broadcast":       {"channels": ["broadcast", "digital_social"], "visibility_key": "broadcast_break"},
    "digital":         {"channels": ["digital_social"], "visibility_key": "digital_post"},
    "athlete":         {"channels": ["digital_social", "broadcast"], "visibility_key": "digital_post"},
}


@dataclass
class ChannelValuation:
    """Valuation result for a single exposure channel."""
    channel: str
    impressions: float
    cpm: float
    visibility_coefficient: float
    raw_value_usd: float
    adjusted_value_usd: float


class ExposureValueModel:
    """
    Bottom-up sponsorship exposure valuation model.

    Business summary
    ----------------
    Aggregates annual impression-based exposure across in-stadium, broadcast,
    digital/social, and jersey/kit channels. Each channel is independently
    valued using channel-specific CPMs adjusted for visibility coefficients
    and global reach factors.

    The key public methods are:

      - total_exposure_value(property, asset_type) → float (USD/yr)
      - exposure_breakdown(property) → dict (channel fractions)
      - batch_score_all_properties() → pd.DataFrame (all properties, vectorized)
      - compare_properties(properties, asset_type) → list[dict]

    Usage
    -----
    >>> model = ExposureValueModel()
    >>> model.total_exposure_value("manchester_city", "jersey")
    45_823_400.0
    >>> df = model.batch_score_all_properties()
    >>> df.sort_values("jersey_usd_m", ascending=False).head()
    """

    def __init__(self) -> None:
        self._params = _PROPERTY_EXPOSURE_PARAMS
        self._asset_map = _ASSET_CHANNEL_MAP

    # ── Primary Public API ──────────────────────────────────────────────────

    def total_exposure_value(self, property_name: str, asset_type: str = "jersey") -> float:
        """
        Annual total exposure value in USD for a single property + asset type.

        Parameters
        ----------
        property_name : str
            Key from SPORTS_PROPERTIES (e.g. "manchester_city").
        asset_type : str
            Sponsorship asset type — one of: jersey, jersey_sleeve, jersey_patch,
            stadium_naming, title_sponsor, broadcast, digital, athlete.

        Returns
        -------
        float
            Total annual exposure value in USD.
        """
        breakdown = self._compute_channel_valuations(property_name, asset_type)
        return sum(ch.adjusted_value_usd for ch in breakdown.values())

    def exposure_breakdown(self, property_name: str, asset_type: str = "jersey") -> dict[str, float]:
        """
        Exposure value split by channel as fractions (sum to 1.0).
        Useful for pie chart / breakdown visualisations.

        Returns
        -------
        dict
            Channel → fraction of total value.
        """
        valuations = self._compute_channel_valuations(property_name, asset_type)
        total = sum(v.adjusted_value_usd for v in valuations.values())
        if total == 0:
            return {ch: 0.0 for ch in valuations}
        return {ch: round(v.adjusted_value_usd / total, 4) for ch, v in valuations.items()}

    def channel_detail(
        self, property_name: str, asset_type: str = "jersey"
    ) -> dict[str, ChannelValuation]:
        """Full channel-level valuation detail for inspection and reporting."""
        return self._compute_channel_valuations(property_name, asset_type)

    def all_asset_types_value(self, property_name: str) -> dict[str, float]:
        """Return the total exposure value for every asset type for one property."""
        return {
            asset_type: self.total_exposure_value(property_name, asset_type)
            for asset_type in self._asset_map
        }

    def compare_properties(
        self,
        properties: list[str],
        asset_type: str = "jersey",
    ) -> list[dict]:
        """
        Compare exposure values across multiple properties.

        Returns
        -------
        list[dict]
            Sorted by total_value_usd descending.
        """
        results = []
        for prop in properties:
            if prop not in self._params:
                continue
            total = self.total_exposure_value(prop, asset_type)
            breakdown = self.exposure_breakdown(prop, asset_type)
            results.append({"property": prop, "total_value_usd": total, "breakdown": breakdown})
        results.sort(key=lambda x: x["total_value_usd"], reverse=True)
        return results

    # ── Vectorized Batch Scoring ────────────────────────────────────────────

    def batch_score_all_properties(self) -> pd.DataFrame:
        """
        Score ALL properties simultaneously using vectorized numpy/pandas operations.

        Business summary
        ----------------
        Produces a ranked DataFrame with exposure values for every asset type
        across every known property — useful for league-wide benchmarking and
        portfolio analysis.

        Developer notes
        ---------------
        Implementation uses numpy array operations rather than Python loops
        over properties. For each channel, a numpy vector of length N (properties)
        is constructed and the CPM formula is applied as an element-wise product.
        This approach scales to thousands of properties with negligible overhead.

        Vectorized formula per channel
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        In-stadium:
            impressions = attendance ⊙ vis_coeff ⊙ (dwell / 2.0)
            value       = (impressions / 1000) ⊙ cpm ⊙ vis_asset_coeff

        Broadcast:
            impressions = viewers_m × 1e6 ⊙ freq ⊙ (screen_sec / 30)
            value       = (impressions / 1000) ⊙ cpm ⊙ reach ⊙ vis_asset_coeff

        Digital/Social:
            impressions = followers_m × 1e6 ⊙ eng_rate ⊙ 1.2 ⊙ freq
            value       = (impressions / 1000) ⊙ cpm ⊙ vis_asset_coeff

        Jersey/Kit:
            captures    = min_played ⊙ captures_per_min
            impressions = captures ⊙ 100_000 ⊙ reach
            value       = (impressions / 1000) ⊙ cpm ⊙ vis_asset_coeff

        ⊙ denotes element-wise (Hadamard) product.

        Returns
        -------
        pd.DataFrame
            Index: property name
            Columns: jersey_usd_m, stadium_naming_usd_m, title_sponsor_usd_m,
                     broadcast_usd_m, digital_usd_m, in_stadium_usd_m,
                     broadcast_usd_m_raw (channel), digital_usd_m_raw (channel),
                     jersey_kit_usd_m (channel), total_usd_m (jersey asset),
                     plus decomposed channel columns for the jersey asset type.
        """
        props = list(self._params.keys())
        n = len(props)

        # ── Build raw parameter arrays ──────────────────────────────────────
        attendance      = np.array([self._params[p]["in_stadium"]["attendance"]           for p in props], dtype=float)
        vis_coeff_stad  = np.array([self._params[p]["in_stadium"]["visibility_coefficient"] for p in props], dtype=float)
        dwell           = np.array([self._params[p]["in_stadium"]["dwell_time_hours"]      for p in props], dtype=float)
        cpm_stad        = np.array([self._params[p]["in_stadium"].get("cpm_override") or CPM_BENCHMARKS["in_stadium"] for p in props], dtype=float)

        viewers_m       = np.array([self._params[p]["broadcast"]["avg_viewership_m"]       for p in props], dtype=float)
        bcast_freq      = np.array([self._params[p]["broadcast"]["broadcast_frequency"]    for p in props], dtype=float)
        screen_sec      = np.array([self._params[p]["broadcast"]["avg_screen_time_seconds"] for p in props], dtype=float)
        cpm_bcast       = np.array([self._params[p]["broadcast"].get("cpm_override") or CPM_BENCHMARKS["broadcast_sports"] for p in props], dtype=float)
        reach_bcast     = np.array([self._params[p]["broadcast"].get("global_reach_factor", 1.0) for p in props], dtype=float)

        followers_m     = np.array([self._params[p]["digital_social"]["follower_count_m"]  for p in props], dtype=float)
        eng_rate        = np.array([self._params[p]["digital_social"]["engagement_rate"]   for p in props], dtype=float)
        post_freq       = np.array([self._params[p]["digital_social"]["post_frequency_per_year"] for p in props], dtype=float)
        cpm_dig         = np.array([self._params[p]["digital_social"].get("cpm_override") or CPM_BENCHMARKS["digital_social"] for p in props], dtype=float)

        min_played      = np.array([self._params[p]["jersey_kit"]["minutes_played_per_year"]    for p in props], dtype=float)
        cam_captures    = np.array([self._params[p]["jersey_kit"]["camera_captures_per_minute"] for p in props], dtype=float)
        cpm_jersey      = np.array([self._params[p]["jersey_kit"].get("jersey_cpm_override") or CPM_BENCHMARKS["jersey_kit"] for p in props], dtype=float)
        reach_jersey    = np.array([self._params[p]["jersey_kit"].get("global_reach_factor", 1.0) for p in props], dtype=float)

        # ── Vectorized channel computations ─────────────────────────────────
        # Each formula operates on all N properties simultaneously.

        # In-stadium impressions & value
        impr_stad   = attendance * vis_coeff_stad * (dwell / 2.0)
        val_stad    = (impr_stad / 1_000) * cpm_stad                    # raw, pre-asset-coeff

        # Broadcast impressions & value
        impr_bcast  = viewers_m * 1_000_000 * bcast_freq * (screen_sec / 30.0)
        val_bcast   = (impr_bcast / 1_000) * cpm_bcast * reach_bcast

        # Digital/social impressions & value
        impr_dig    = followers_m * 1_000_000 * eng_rate * 1.20 * post_freq
        val_dig     = (impr_dig / 1_000) * cpm_dig

        # Jersey/kit impressions & value
        captures    = min_played * cam_captures
        impr_jersey = captures * 100_000 * reach_jersey
        val_jersey  = (impr_jersey / 1_000) * cpm_jersey

        # ── Apply asset-type visibility coefficients ─────────────────────────
        def _asset_total(vis_key: str, channels: list[str]) -> np.ndarray:
            vc = VISIBILITY_COEFFICIENTS.get(vis_key, 0.5)
            total = np.zeros(n)
            if "in_stadium"    in channels: total += val_stad   * vc
            if "broadcast"     in channels: total += val_bcast  * vc
            if "digital_social" in channels: total += val_dig   * vc
            if "jersey_kit"    in channels: total += val_jersey * vc
            return total

        jersey_vals        = _asset_total("jersey_front",   ["in_stadium", "broadcast", "jersey_kit"])
        sleeve_vals        = _asset_total("jersey_sleeve",  ["in_stadium", "broadcast", "jersey_kit"])
        stadium_vals       = _asset_total("stadium_naming", ["in_stadium", "broadcast", "digital_social"])
        title_vals         = _asset_total("broadcast_title",["in_stadium", "broadcast", "digital_social"])
        broadcast_vals     = _asset_total("broadcast_break",["broadcast", "digital_social"])
        digital_vals       = _asset_total("digital_post",   ["digital_social"])

        # ── Assemble DataFrame ───────────────────────────────────────────────
        df = pd.DataFrame(
            {
                "property":              props,
                # Channel-level raw values (pre-asset-coeff, in USD)
                "ch_in_stadium_usd":     val_stad,
                "ch_broadcast_usd":      val_bcast,
                "ch_digital_social_usd": val_dig,
                "ch_jersey_kit_usd":     val_jersey,
                # Asset-level totals (in USD millions)
                "jersey_usd_m":          jersey_vals          / 1_000_000,
                "jersey_sleeve_usd_m":   sleeve_vals          / 1_000_000,
                "stadium_naming_usd_m":  stadium_vals         / 1_000_000,
                "title_sponsor_usd_m":   title_vals           / 1_000_000,
                "broadcast_usd_m":       broadcast_vals       / 1_000_000,
                "digital_usd_m":         digital_vals         / 1_000_000,
                # Total impressions across all channels
                "total_impressions_bn":  (impr_stad + impr_bcast + impr_dig + impr_jersey) / 1e9,
            }
        ).set_index("property")

        # Round to 2 dp for readability
        df = df.round(2)

        # Add rank columns
        df["jersey_rank"]       = df["jersey_usd_m"].rank(ascending=False).astype(int)
        df["total_impr_rank"]   = df["total_impressions_bn"].rank(ascending=False).astype(int)

        return df.sort_values("jersey_usd_m", ascending=False)

    # ── Internal Computation ────────────────────────────────────────────────

    def _compute_channel_valuations(
        self, property_name: str, asset_type: str
    ) -> dict[str, ChannelValuation]:
        """Core single-property channel valuation. Returns per-channel breakdown."""
        if property_name not in self._params:
            raise ValueError(f"Unknown property '{property_name}'. Available: {list(self._params)}")
        if asset_type not in self._asset_map:
            raise ValueError(f"Unknown asset type '{asset_type}'. Available: {list(self._asset_map)}")

        params    = self._params[property_name]
        asset_info = self._asset_map[asset_type]
        vis_coeff  = VISIBILITY_COEFFICIENTS.get(asset_info["visibility_key"], 0.5)

        valuations: dict[str, ChannelValuation] = {}

        for channel in asset_info["channels"]:
            if channel not in params:
                continue
            p = params[channel]

            if channel == "in_stadium":
                cpm         = p.get("cpm_override") or CPM_BENCHMARKS["in_stadium"]
                dwell_f     = p["dwell_time_hours"] / 2.0
                impressions = p["attendance"] * p["visibility_coefficient"] * dwell_f
                raw_value   = (impressions / 1_000) * cpm
                adjusted    = raw_value * vis_coeff

            elif channel == "broadcast":
                cpm         = p.get("cpm_override") or CPM_BENCHMARKS["broadcast_sports"]
                reach       = p.get("global_reach_factor", 1.0)
                screen_f    = p["avg_screen_time_seconds"] / 30.0
                impressions = p["avg_viewership_m"] * 1_000_000 * p["broadcast_frequency"] * screen_f
                raw_value   = (impressions / 1_000) * cpm * reach
                adjusted    = raw_value * vis_coeff

            elif channel == "digital_social":
                cpm         = p.get("cpm_override") or CPM_BENCHMARKS["digital_social"]
                eff_reach   = p["follower_count_m"] * 1_000_000 * p["engagement_rate"] * 1.20
                impressions = eff_reach * p["post_frequency_per_year"]
                raw_value   = (impressions / 1_000) * cpm
                adjusted    = raw_value * vis_coeff

            elif channel == "jersey_kit":
                cpm         = p.get("jersey_cpm_override") or CPM_BENCHMARKS["jersey_kit"]
                reach       = p.get("global_reach_factor", 1.0)
                captures    = p["minutes_played_per_year"] * p["camera_captures_per_minute"]
                impressions = captures * 100_000 * reach
                raw_value   = (impressions / 1_000) * cpm
                adjusted    = raw_value * vis_coeff

            else:
                continue

            valuations[channel] = ChannelValuation(
                channel=channel,
                impressions=impressions,
                cpm=cpm,
                visibility_coefficient=vis_coeff,
                raw_value_usd=raw_value,
                adjusted_value_usd=adjusted,
            )

        return valuations
