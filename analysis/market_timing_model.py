"""
analysis/market_timing_model.py — Market Timing Analyzer for Sponsorship Deals.

Business summary
----------------
Timing is as important as price. Signing a sponsorship deal with a club that
is about to win the Champions League (before the market prices in that success)
is a tremendous buy. Signing a deal with a club entering a rebuilding phase at
peak-cycle valuations is a poor use of capital.

This module answers three questions:
  1. timing_score(): Is NOW a good time to sign a deal with this property?
     (0.0 = terrible time, 1.0 = perfect time)
  2. renewal_recommendation(): Should the incumbent sponsor renew, renegotiate,
     or exit when their deal expires?
  3. auction_dynamics(): If competing brands are bidding, what is the right
     bid strategy and what premium above fair value should you expect to pay?

The timing_score combines:
  - Commercial momentum: recent viewership trend, recent on-field performance
  - Market cycle positioning: are we in a buyer's or seller's market globally?
  - Competitive bidding intensity: how many known competitors are at the table?
  - Valuation relative to historical comps: is the property priced at a premium?
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd

from data.sports_data import SPORTS_PROPERTIES


# ── Property Commercial Momentum Data ────────────────────────────────────────
# Momentum score (−5 to +5) capturing:
#   - viewership trend (positive = growing audience)
#   - on-field performance trend (positive = improving results)
#   - commercial pipeline (positive = expanding fanbase, new markets)
# Sign convention: positive = property momentum benefits buyer (good time to BUY),
#                  negative = seller's market (property has leverage)
# Sources: Deloitte Annual Review of Football Finance, SportsPro 50, Forbes valuations.

_PROPERTY_MOMENTUM: dict[str, dict] = {
    "manchester_city": {
        "viewership_trend":      +1.5,  # stable European audience post-treble
        "performance_trend":     +2.0,  # consistent top performer
        "commercial_pipeline":   +1.0,
        "momentum_score":        +1.8,
        "market_phase":          "seller",  # club has leverage; premium-priced
        "narrative":             "Post-treble peak; club has pricing power",
    },
    "manchester_united": {
        "viewership_trend":      -1.0,  # audience declining with poor results
        "performance_trend":     -2.5,  # worst decade in modern era
        "commercial_pipeline":   +0.5,  # INEOS rebuild narrative
        "momentum_score":        -1.0,
        "market_phase":          "buyer",   # sponsors have negotiating leverage
        "narrative":             "Performance nadir; INEOS rebuild = long-term buy opportunity",
    },
    "real_madrid": {
        "viewership_trend":      +2.0,
        "performance_trend":     +2.5,  # back-to-back UCL
        "commercial_pipeline":   +2.0,  # Bernabeu renovation, Mbappe arrival
        "momentum_score":        +2.2,
        "market_phase":          "seller",
        "narrative":             "Mbappe era beginning; structural premium warranted",
    },
    "barcelona": {
        "viewership_trend":      +0.5,
        "performance_trend":     -0.5,
        "commercial_pipeline":   -1.0,  # financial crisis / lever constraints
        "momentum_score":        -0.3,
        "market_phase":          "buyer",
        "narrative":             "Financial crisis creates buyer's market; Spotify got a good deal in 2022",
    },
    "psg": {
        "viewership_trend":      +1.0,
        "performance_trend":     +0.5,
        "commercial_pipeline":   -0.5,  # post-Mbappe departure
        "momentum_score":        +0.3,
        "market_phase":          "neutral",
        "narrative":             "Post-Mbappe transition; valuation uncertainty",
    },
    "chelsea": {
        "viewership_trend":      -0.5,
        "performance_trend":     -2.0,
        "commercial_pipeline":   -0.5,
        "momentum_score":        -1.0,
        "market_phase":          "buyer",
        "narrative":             "Clegg era instability; mid-table commercial risk",
    },
    "arsenal": {
        "viewership_trend":      +2.0,
        "performance_trend":     +2.5,  # title challengers after 20-year drought
        "commercial_pipeline":   +1.5,
        "momentum_score":        +2.0,
        "market_phase":          "seller",
        "narrative":             "Arsenal resurgence; window to lock in pre-peak pricing",
    },
    "liverpool": {
        "viewership_trend":      +1.5,
        "performance_trend":     +1.0,
        "commercial_pipeline":   +1.5,
        "momentum_score":        +1.3,
        "market_phase":          "neutral",
        "narrative":             "Slot era beginning; medium-term momentum building",
    },
    "formula_1": {
        "viewership_trend":      +1.0,
        "performance_trend":     +0.0,  # sport-level, not team
        "commercial_pipeline":   +3.0,  # Las Vegas, Miami, new markets
        "momentum_score":        +1.8,
        "market_phase":          "seller",
        "narrative":             "Drive to Survive plateau but new market expansion ongoing",
    },
    "red_bull_racing": {
        "viewership_trend":      +1.5,
        "performance_trend":     +2.5,  # dominant champions
        "commercial_pipeline":   +1.0,
        "momentum_score":        +1.7,
        "market_phase":          "seller",
        "narrative":             "Verstappen dominance; premium pricing period",
    },
    "ferrari_f1": {
        "viewership_trend":      +1.5,
        "performance_trend":     +2.0,  # resurgent 2024 form with Hamilton
        "commercial_pipeline":   +2.5,  # Hamilton generates massive commercial interest
        "momentum_score":        +2.0,
        "market_phase":          "seller",
        "narrative":             "Hamilton era creating massive commercial opportunity; ACT NOW",
    },
    "nfl": {
        "viewership_trend":      +2.0,
        "performance_trend":     0.0,
        "commercial_pipeline":   +2.0,  # Taylor Swift effect, international games
        "momentum_score":        +2.0,
        "market_phase":          "seller",
        "narrative":             "Record viewership; premium market; no discount available",
    },
    "dallas_cowboys": {
        "viewership_trend":      +1.5,
        "performance_trend":     -0.5,
        "commercial_pipeline":   +1.0,
        "momentum_score":        +0.7,
        "market_phase":          "neutral",
        "narrative":             "Brand value vs. performance disconnect; stable market",
    },
    "la_lakers": {
        "viewership_trend":      -0.5,
        "performance_trend":     +0.5,
        "commercial_pipeline":   -0.5,  # LeBron exit imminent
        "momentum_score":        -0.2,
        "market_phase":          "buyer",
        "narrative":             "Pre-LeBron exit: negotiate before full rebuilding uncertainty",
    },
    "golden_state_warriors": {
        "viewership_trend":      -1.0,
        "performance_trend":     -1.5,
        "commercial_pipeline":   -0.5,
        "momentum_score":        -1.0,
        "market_phase":          "buyer",
        "narrative":             "Dynasty waning; 3-5 year rebuild; good entry valuation",
    },
    "wimbledon": {
        "viewership_trend":      +0.5,
        "performance_trend":     0.0,
        "commercial_pipeline":   +0.5,
        "momentum_score":        +0.5,
        "market_phase":          "neutral",
        "narrative":             "Stable heritage institution; no timing edge either way",
    },
    "nba": {
        "viewership_trend":      +1.0,
        "performance_trend":     0.0,
        "commercial_pipeline":   +1.5,
        "momentum_score":        +1.2,
        "market_phase":          "neutral",
        "narrative":             "International growth offset by domestic linear TV decline",
    },
    "brentford": {
        "viewership_trend":      +0.5,
        "performance_trend":     -1.0,
        "commercial_pipeline":   +0.5,
        "momentum_score":        -0.2,
        "market_phase":          "buyer",
        "narrative":             "Relegation risk creates value entry for long-term brand play",
    },
    "new_england_patriots": {
        "viewership_trend":      -0.5,
        "performance_trend":     -2.0,  # post-Belichick/Brady era
        "commercial_pipeline":   -0.5,
        "momentum_score":        -1.0,
        "market_phase":          "buyer",
        "narrative":             "Dynasty post-peak; commercial value will normalise lower",
    },
}

# Global sponsorship market cycle indicator
# +1.0 = strongly seller's market (brands competing; prices elevated)
# -1.0 = strongly buyer's market (recession/belt-tightening; brands cautious)
# Source: IEG Sponsorship Report, PwC Sports Outlook.
_GLOBAL_MARKET_CYCLE: dict[int, float] = {
    2019: +0.3,
    2020: -0.8,  # COVID: brands cut sponsorship
    2021: -0.1,  # Recovery
    2022: +0.5,  # Post-COVID boom
    2023: +0.6,
    2024: +0.4,
    2025: +0.3,
    2026: +0.2,
}


class MarketTimingAnalyzer:
    """
    Assesses whether now is a good time to sign a sponsorship deal.

    Business summary
    ----------------
    Timing affects price, terms, and value capture. A brand that signs a deal
    just before a property enters a growth phase locks in pre-peak pricing; one
    that signs at the peak of a property's commercial cycle overpays.

    The timing_score (0–1) combines property momentum, global market cycle,
    and competitive bidding pressure into a single "buy now" signal.

    Usage
    -----
    >>> analyzer = MarketTimingAnalyzer()
    >>> analyzer.timing_score("barcelona", "2022-06-01")
    0.82  # Barcelona in financial crisis = excellent buyer's market in 2022
    >>> analyzer.timing_score("real_madrid", "2024-06-01")
    0.35  # Madrid post-UCL = seller's market, poor time to sign
    >>> analyzer.renewal_recommendation({"property": "manchester_city", "expiry_year": 2026, ...})
    """

    def __init__(self) -> None:
        self._momentum     = _PROPERTY_MOMENTUM
        self._market_cycle = _GLOBAL_MARKET_CYCLE

    # ── Timing Score ──────────────────────────────────────────────────────────

    def timing_score(
        self,
        property_key: str,
        current_date: Optional[str] = None,
        competing_bidders: int = 0,
    ) -> float:
        """
        Score for signing a deal NOW with this property (0.0–1.0).

        Higher score = better time to sign (for the buyer / sponsoring brand).

        Components
        ----------
        1. Property momentum score → maps momentum to [0, 1]:
           Poor momentum = good for buyer (property needs revenue) → high score
           Great momentum = seller's market (property has leverage) → lower score

        2. Global market cycle (0–1):
           Buyer's market globally → high score; seller's market → low score

        3. Competitive pressure penalty:
           Each known competing bidder reduces the timing score by 0.05

        Formula
        -------
            raw = 0.5 × momentum_component + 0.3 × market_component + 0.2 × competition_component
            timing_score = clip(raw, 0.05, 0.95)

        Parameters
        ----------
        property_key : str
        current_date : str, optional
            ISO 8601 date string (e.g. "2024-06-01"). Defaults to today.
        competing_bidders : int
            Number of known competing brands in the bidding process.

        Returns
        -------
        float in [0.05, 0.95].
        """
        data = self._momentum.get(property_key, {})
        if not data:
            return 0.50  # no data = neutral

        # Component 1: Momentum (inverted for buyer perspective)
        # Momentum [-5, +5] → mapped to [0, 1]
        # Negative momentum (property struggling) = good for buyer → score near 1.0
        momentum = data["momentum_score"]
        momentum_component = ((-momentum + 5.0) / 10.0)   # invert and normalise

        # Component 2: Global market cycle
        if current_date:
            year = datetime.strptime(current_date, "%Y-%m-%d").year
        else:
            year = date.today().year
        cycle_raw = self._market_cycle.get(year, 0.0)
        # cycle_raw: +1 = seller's market → bad for buyer; -1 = buyer's market → good
        market_component = (-cycle_raw + 1.0) / 2.0       # invert and normalise to [0,1]

        # Component 3: Competitive pressure
        competition_component = max(0.0, 1.0 - competing_bidders * 0.12)

        raw = (
            0.50 * momentum_component +
            0.30 * market_component +
            0.20 * competition_component
        )
        return round(float(np.clip(raw, 0.05, 0.95)), 4)

    def timing_verdict(self, score: float) -> str:
        """Convert timing score to a human-readable verdict."""
        if score >= 0.75:
            return "STRONG BUY — excellent buyer's market conditions"
        elif score >= 0.60:
            return "BUY — favourable timing"
        elif score >= 0.45:
            return "NEUTRAL — standard market conditions; negotiate hard"
        elif score >= 0.30:
            return "WAIT — seller has leverage; consider delaying or renegotiating"
        else:
            return "AVOID — seller's market peak; significant premium risk"

    def market_phase(self, property_key: str) -> dict:
        """Return detailed market phase analysis for a property."""
        data = self._momentum.get(property_key, {})
        score = self.timing_score(property_key)
        return {
            "property":       property_key,
            "market_phase":   data.get("market_phase", "unknown"),
            "momentum_score": data.get("momentum_score", 0.0),
            "timing_score":   score,
            "verdict":        self.timing_verdict(score),
            "narrative":      data.get("narrative", "No narrative data"),
        }

    # ── Renewal Recommendation ────────────────────────────────────────────────

    def renewal_recommendation(
        self,
        current_deal: dict,
        current_date: Optional[str] = None,
    ) -> dict:
        """
        Advise whether to renew early, wait, renegotiate, or exit.

        Parameters
        ----------
        current_deal : dict
            Required keys: property_key, annual_value_usd_m, expiry_year,
                           brand_key (optional), deal_type (optional).
        current_date : str, optional
            ISO 8601 date string.

        Returns
        -------
        dict
            {recommendation, rationale, suggested_action, urgency_score}
        """
        property_key = current_deal["property_key"]
        current_value = current_deal["annual_value_usd_m"]
        expiry_year   = current_deal["expiry_year"]

        timing   = self.timing_score(property_key, current_date)
        momentum = self._momentum.get(property_key, {})
        mphase   = momentum.get("market_phase", "neutral")

        if current_date:
            current_year = datetime.strptime(current_date, "%Y-%m-%d").year
        else:
            current_year = date.today().year
        years_to_expiry = expiry_year - current_year

        # Decision logic
        if timing >= 0.65 and years_to_expiry >= 2:
            # Good buyer's market AND time to act
            rec = "RENEW EARLY"
            rationale = (
                f"{property_key} is in a buyer's market (timing={timing:.2f}). "
                f"Lock in current pricing before momentum shifts. "
                f"{momentum.get('narrative', '')}"
            )
            urgency = 0.85
        elif timing < 0.40 and years_to_expiry >= 2:
            # Seller's market — don't extend early, let it run out
            rec = "WAIT / RENEGOTIATE AT EXPIRY"
            rationale = (
                f"{property_key} currently in seller's market (timing={timing:.2f}). "
                f"Do not renew early — risk overpaying at peak. "
                f"Renegotiate when deal expires in {years_to_expiry}yr. "
                f"{momentum.get('narrative', '')}"
            )
            urgency = 0.25
        elif years_to_expiry <= 1:
            # Expiring soon regardless of market
            rec = "RENEW OR EXIT — URGENT"
            rationale = (
                f"Deal expires in {years_to_expiry} year(s). Decision required imminently. "
                f"Timing score: {timing:.2f} ({self.timing_verdict(timing)}). "
                f"{momentum.get('narrative', '')}"
            )
            urgency = 0.95
        elif mphase == "buyer":
            rec = "RENEW — NEGOTIATE HARD"
            rationale = (
                f"Property in buyer's market phase. Push for reduced rates or added value. "
                f"{momentum.get('narrative', '')}"
            )
            urgency = 0.65
        else:
            rec = "MAINTAIN CURRENT DEAL"
            rationale = (
                f"Neutral conditions. Honour current terms; reassess at renewal window. "
                f"{momentum.get('narrative', '')}"
            )
            urgency = 0.45

        return {
            "property":          property_key,
            "current_value_usd_m": current_value,
            "expiry_year":       expiry_year,
            "years_to_expiry":   years_to_expiry,
            "timing_score":      timing,
            "market_phase":      mphase,
            "recommendation":    rec,
            "rationale":         rationale,
            "urgency_score":     urgency,
        }

    # ── Auction Dynamics ──────────────────────────────────────────────────────

    def auction_dynamics(
        self,
        property_key: str,
        known_competitors: list[str],
        base_fair_value_usd_m: float,
        is_exclusive: bool = False,
    ) -> dict:
        """
        Model auction dynamics and recommend a bid strategy.

        Business summary
        ----------------
        When multiple brands compete for the same deal, the selling property
        runs an implicit auction. The winning bid is typically driven by:
          - Number of serious bidders (more bidders → higher clearing price)
          - Strategic value of the asset to each bidder (exclusivity matters)
          - Information asymmetry (how much does each bidder know about others?)

        Auction premium model
        ---------------------
        Based on auction theory (Vickrey-Clarke-Groves), in a first-price
        sealed-bid auction with N bidders drawing values uniformly from [V, V+σ]:

            Expected clearing price ≈ V × (1 + (N-1)/(N+1) × CV)

        where CV is the coefficient of variation in perceived value.
        In sponsorship, CV ≈ 0.15–0.25 (brands have similar but not identical valuations).

        Parameters
        ----------
        property_key : str
        known_competitors : list[str]
            Brand keys of known competing bidders.
        base_fair_value_usd_m : float
            Model-implied fair value.
        is_exclusive : bool

        Returns
        -------
        dict
            {recommended_bid, bid_ceiling, auction_premium_pct, strategy, win_probability}
        """
        n_competitors = len(known_competitors)
        timing = self.timing_score(property_key, competing_bidders=n_competitors)

        # Auction premium: based on number of bidders
        # CV of perceived value ≈ 0.20 in sports sponsorship markets
        cv_val = 0.20
        if n_competitors == 0:
            auction_premium = 0.0
            win_prob = 1.0
        else:
            # Expected auction premium = (N-1)/(N+1) × CV × base_value
            n = n_competitors + 1  # include ourselves
            auction_premium = (n - 1) / (n + 1) * cv_val
            # Win probability in a symmetric auction: 1/N
            win_prob = 1.0 / n

        # Exclusivity premium (if competing for exclusive rights, stakes higher)
        excl_adjustment = 0.12 if is_exclusive else 0.0

        # Recommended bid: fair value + auction premium + exclusivity
        recommended_bid = base_fair_value_usd_m * (1 + auction_premium + excl_adjustment)

        # Bid ceiling: max we should pay (based on timing and strategic value)
        # In buyer's market, ceiling is closer to fair value; seller's market = higher
        ceiling_scale = 1.0 + (auction_premium * 2.0) + excl_adjustment
        bid_ceiling = base_fair_value_usd_m * ceiling_scale

        # Strategy
        if n_competitors == 0:
            strategy = "SOLE BIDDER — open with fair value, hold firm. No need to overpay."
        elif n_competitors == 1:
            strategy = ("TWO-HORSE RACE — bid 15–20% above fair value. "
                        "Consider adding non-monetary value (activation, media, services).")
        elif n_competitors <= 3:
            strategy = ("COMPETITIVE AUCTION — bid at recommended level. "
                        "Focus on non-price differentiation: commitment period, activation budget. "
                        "Consider requesting BAFO (Best And Final Offer) round.")
        else:
            strategy = ("CROWDED AUCTION — assess strategic value carefully. "
                        "Over-paying is high risk. Consider walking away if bid ceiling is reached. "
                        "Identify if any competitor has stronger strategic rationale.")

        return {
            "property":             property_key,
            "base_fair_value_usd_m": base_fair_value_usd_m,
            "n_known_competitors":  n_competitors,
            "known_competitors":    known_competitors,
            "auction_premium_pct":  round(auction_premium * 100, 1),
            "recommended_bid_usd_m": round(recommended_bid, 2),
            "bid_ceiling_usd_m":    round(bid_ceiling, 2),
            "win_probability":      round(win_prob, 3),
            "timing_score":         timing,
            "is_exclusive":         is_exclusive,
            "strategy":             strategy,
        }

    def compare_timing_across_properties(
        self, property_keys: list[str], current_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare timing scores across multiple properties.

        Returns
        -------
        pd.DataFrame
            Index: property_key. Columns: timing_score, market_phase, verdict.
        """
        rows = []
        for pk in property_keys:
            score   = self.timing_score(pk, current_date)
            phase   = self._momentum.get(pk, {}).get("market_phase", "unknown")
            verdict = self.timing_verdict(score)
            rows.append({
                "property":     pk,
                "timing_score": score,
                "market_phase": phase,
                "verdict":      verdict,
            })
        df = pd.DataFrame(rows).set_index("property")
        return df.sort_values("timing_score", ascending=False)
