# Sports Sponsorship Valuator

[Algorithm guide: pseudocode, time complexity, and memory](docs/ALGORITHM_GUIDE.md).

An exploratory sponsorship-valuation codebase in Python (~7,600 lines across 14 modules) covering multi-channel CPM exposure, audience and brand-equity scoring, comparable-deal regression over 47 historical sponsorship transactions, Monte Carlo valuation, and analytic Markowitz portfolio optimization.

**Status: exploratory module suite.** Individual analytical modules (`valuation/`, `analysis/`, `signals/`, `schemas/`) are implemented; `tests/`, `backtest/`, `examples/`, and `reports/` are package scaffolding without a unified CLI or test runner yet. Values and parameters are illustrative, not empirically calibrated against proprietary rights-holder loss data.

## Implemented modules

| Module | File | What it models |
|---|---|---|
| Exposure valuation | `valuation/exposure_model.py` | Broadcast, streaming, social, and in-venue impression valuation using channel-specific CPMs, viewability, and clutter adjustments |
| Audience alignment | `valuation/audience_model.py` | Demographic overlap, purchase-intent weighting, and regional reach multipliers |
| Brand equity | `valuation/brand_equity_model.py` | Recall, favorability lift, and long-term brand-transfer heuristics |
| Comparable deals | `valuation/comparable_deals.py` | 47 historical sponsorship deals with OLS regression and similarity scoring |
| Monte Carlo engine | `valuation/monte_carlo_valuation.py` | Seeded scenario simulation over impression delivery, CPM drift, and renewal probability |
| Risk & brand safety | `valuation/risk_model.py` | Controversy, performance downturn, and category-exclusivity risk adjustments |
| Fair value synthesis | `valuation/fair_value_engine.py` | Combines exposure, comparables, and Monte Carlo percentiles into a composite valuation band |
| Portfolio optimizer | `analysis/portfolio_optimizer.py` | Closed-form Markowitz mean-variance allocation across sponsorship assets (NumPy only, no SciPy) |
| Market timing | `analysis/market_timing_model.py` | Contract cycle and league-expansion timing heuristics |
| Social amplification | `signals/social_amplification.py` | Engagement-weighted social lift estimator |

## Quick check

```bash
git clone https://github.com/fatehaszaman/sports-sponsorship-valuator.git
cd sports-sponsorship-valuator
pip install -r requirements.txt
python3 -c "from valuation.comparable_deals import ComparableDealsAnalyzer; print(len(ComparableDealsAnalyzer().deals))"
```
