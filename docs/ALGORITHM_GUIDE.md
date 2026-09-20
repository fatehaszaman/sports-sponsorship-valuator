# Algorithm guide

This card follows the numerical simulation stage after base valuation inputs
have been resolved. It distinguishes simulated dispersion from evidence that
the underlying valuation assumptions are accurate.

## Sponsorship-value distribution

Implementation: [`MonteCarloValuator.simulate`](../valuation/monte_carlo_valuation.py).

```text
# Sponsorship Monte Carlo / Multiplicative Scenarios
# Input: resolved valuation components, weights, noise settings, N trials
# Output: fixed-size distribution summary; not the full trial array
# Time: O(N) array operations + Q(N) percentile work
# Memory: O(N) temporary trial arrays; O(1) returned summary
# Q(N): library dependent; O(N log N) under a conservative sorting model

RESOLVE base exposure, comparable, audience, and social components
FOR each of four noise sources:
    sigma = supplied setting
    mu = -sigma^2 / 2
    DRAW N lognormal multipliers
COMPUTE exposure trials using three multiplier arrays
COMPUTE comparable, audience, and social trials with their assigned multipliers
BLEND trial components with configured weights
APPLY any exclusivity premium; floor values at zero
COMPUTE percentiles, mean, standard deviation, and skewness
RETURN rounded summary including the fifth value percentile
```

Why: a fixed number of N-element arrays and masks are retained together.
Vectorized operations reduce Python overhead, not the O(N) trial storage.
Costs of finding comparable deals and obtaining base inputs are separate from
this conditional simulation-stage bound.

## Important interpretation details

- **Noise setting:** the code assigns the parameter named `cv` directly to
  log-space sigma. The actual coefficient of variation is
  `sqrt(exp(sigma^2)-1)`, not exactly sigma. Treat the name as an approximation.
- **Mean versus median:** `mu = -sigma^2/2` gives theoretical arithmetic mean
  one for a lognormal multiplier. Its median/geometric mean is `exp(mu)`,
  not one; a finite sample's average will fluctuate.
- **Risk field:** `var_95_usd_m` is the fifth percentile of simulated value.
  It is not automatically a 95% quantile of financial loss; a loss definition
  and reference value would be needed for that interpretation.
- **Reproducibility:** the seeded generator is stateful. Repeated calls on one
  instance advance its state; reset/recreate it to replay identical draws.
- **Boundaries:** use a positive trial count and finite, meaningful inputs.
  More trials reduce sampling noise, not parameter or model uncertainty.

These notes document existing behavior without changing formulas or claiming
calibration.
