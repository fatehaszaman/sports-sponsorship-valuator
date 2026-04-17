"""
config.py — Central configuration for Sports Sponsorship Valuator.

Loads environment variables and defines global constants used across
valuation models.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN", "")

# ── Valuation Defaults ────────────────────────────────────────────────────────
# Discount rate for NPV calculations (WACC proxy)
DEFAULT_DISCOUNT_RATE: float = float(os.getenv("DEFAULT_DISCOUNT_RATE", "0.10"))

# Sponsorship market growth rate assumption
DEFAULT_GROWTH_RATE: float = float(os.getenv("DEFAULT_GROWTH_RATE", "0.05"))

# Output currency
DEFAULT_CURRENCY: str = os.getenv("DEFAULT_CURRENCY", "USD")

# ── CPM Benchmarks (Cost Per Mille, in USD) ───────────────────────────────────
# Source: Nielsen Sports, Sponsorship Research International benchmarks
CPM_BENCHMARKS = {
    "in_stadium": 18.50,       # per 1,000 in-stadium impressions
    "broadcast_sports": 42.00, # per 1,000 broadcast viewers (live sports premium)
    "digital_social": 8.75,    # per 1,000 social media impressions
    "jersey_kit": 95.00,       # per 1,000 jersey logo impressions (premium placement)
    "stadium_naming": 12.00,   # per 1,000 ambient stadium impressions
    "press_conference": 28.00, # per 1,000 press/media impressions
}

# ── Exposure Coefficients ─────────────────────────────────────────────────────
# Visibility coefficients: how prominently a sponsor appears in each context
VISIBILITY_COEFFICIENTS = {
    "jersey_front": 1.00,      # primary shirt sponsor, chest
    "jersey_sleeve": 0.45,     # sleeve sponsor
    "kit_shorts": 0.20,        # shorts sponsor
    "stadium_pitch_side": 0.65,# perimeter LED boards
    "stadium_naming": 0.80,    # stadium name rights
    "broadcast_title": 0.90,   # broadcast title sponsor (e.g. "The Pepsi Super Bowl")
    "broadcast_break": 0.55,   # commercial breaks during broadcast
    "digital_post": 1.00,      # dedicated social post
    "digital_mention": 0.30,   # brand mention in post
}

# ── Exclusivity Premiums ──────────────────────────────────────────────────────
# Premium multipliers for category exclusivity (no competing brand in same category)
EXCLUSIVITY_PREMIUMS = {
    "beer_alcohol":    0.35,   # 35% premium for exclusive alcohol rights
    "telecoms":        0.30,
    "automotive":      0.28,
    "financial_svcs":  0.25,
    "airlines":        0.22,
    "fast_food":       0.20,
    "soft_drinks":     0.18,
    "sportswear":      0.40,   # highest — kit deal = exclusive sportswear
    "crypto_fintech":  0.15,
    "luxury_watches":  0.20,
    "energy_oil":      0.25,
    "insurance":       0.18,
    "default":         0.15,
}

# ── NewsAPI Settings ──────────────────────────────────────────────────────────
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
NEWSAPI_MAX_RESULTS = 100
NEWSAPI_LOOKBACK_DAYS = 30

# ── Twitter API v2 Settings ───────────────────────────────────────────────────
TWITTER_API_BASE_URL = "https://api.twitter.com/2"
TWITTER_SEARCH_MAX_RESULTS = 100
TWITTER_LOOKBACK_DAYS = 7

# ── Weighting Scheme for Fair Value Engine ────────────────────────────────────
# How much weight to give each valuation signal (must sum to 1.0)
VALUATION_WEIGHTS = {
    "exposure_value":    0.40,  # bottom-up exposure model
    "comp_implied":      0.35,  # comparable deal analysis
    "audience_premium":  0.15,  # audience demographic fit adjustment
    "social_signal":     0.10,  # social amplification signal
}

# ── Demographic Quality Weights ───────────────────────────────────────────────
# How much each demographic factor contributes to audience quality index
DEMOGRAPHIC_QUALITY_WEIGHTS = {
    "high_income_pct":   0.35,  # % HHI > $100k
    "prime_age_pct":     0.30,  # % aged 25-44 (peak purchasing power)
    "international_pct": 0.20,  # % international fanbase (global brand relevance)
    "engagement_rate":   0.15,  # relative fan engagement vs. league average
}

# ── Athlete Endorsement Value Weighting ───────────────────────────────────────
# Weighting: athletic performance (1.5) outweighs reputation/character (1.0).
# Rationale: sponsors primarily buy athletic visibility and achievement.
# Reputation/character is a risk-adjustment factor — a scandal (reputation=20/100)
# on a world-class athlete still carries significant commercial value, but the
# brand risk discount is meaningful. The 1.5:1.0 ratio reflects that sponsors
# accept some reputational risk for elite athletic association.
#
# Exception: family/luxury brands weight reputation higher — add an override:
# brand_safety_sensitive: bool — if True, use PSYCH_WEIGHT=1.5, PHYSICAL_WEIGHT=1.0
# This flip is appropriate for Rolex, Disney, family-oriented consumer products,
# or any brand where a sponsee scandal would materially damage the brand's core
# promise (trustworthiness, prestige, family values).

PSYCH_WEIGHT: float = 1.0    # reputation, character, brand-safety score weight
PHYSICAL_WEIGHT: float = 1.5  # on-field performance, athletic achievement weight
ATHLETE_TOTAL_WEIGHT: float = PSYCH_WEIGHT + PHYSICAL_WEIGHT  # 2.5

# Brand-safety-sensitive override (flips the ratio: reputation > performance)
PSYCH_WEIGHT_SAFE: float = 1.5    # used when brand_safety_sensitive=True
PHYSICAL_WEIGHT_SAFE: float = 1.0  # used when brand_safety_sensitive=True
ATHLETE_TOTAL_WEIGHT_SAFE: float = PSYCH_WEIGHT_SAFE + PHYSICAL_WEIGHT_SAFE  # 2.5

# Brands that should use the brand-safety-sensitive weighting by default
BRAND_SAFETY_SENSITIVE_BRANDS: set = {
    "rolex",       # luxury prestige — scandal destroys brand cachet
    "disney",      # family audience — zero tolerance for controversy
    "lego",        # children's product
    "mastercard",  # broad consumer trust essential
    "visa",        # broad consumer trust essential
}
