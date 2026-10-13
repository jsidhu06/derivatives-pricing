# Changelog

## 0.3.0

This release adds Monte Carlo standard errors and implied-volatility
diagnostics, changes the default PDE grid domain, and refines the handling of
a number of edge cases: dividend and monitoring dates that coincide, very
short or very long maturities on the PDE grid, correlated simulation when
assets have different dates, and curve extrapolation. Prices for standard
contracts on the analytical and binomial engines are unchanged; PDE prices
move within discretisation error because of the new default domain.

### API changes

Calls that need updating:

- **`CorrelationContext`** takes a required `time_grid`, and each process in a
  context is built with `SimulationConfig(time_grid=context.time_grid)`. The
  new `shared_time_grid(...)` helper builds the grid.
- **`DiscountCurve.flat(rate)`** no longer takes `end_time`; the curve always
  spans 100 years. The argument did not affect prices.

Combinations that now raise `UnsupportedFeatureError`, each outside what the
method supports:

- `control_variate_european` with `PDE_FD` or `MONTE_CARLO` when a discrete
  dividend falls within the option's life. The control variate's analytical
  leg uses the escrowed-dividend model and these engines use the spot-drop
  model, so the two legs are not comparable. `BINOMIAL` is unaffected.
- The analytical engine for a discretely monitored barrier when the spot is
  already beyond the barrier. The closed form applies on the live side only;
  the binomial, PDE and Monte Carlo engines price this case.
- Numerical theta for a process in a `CorrelationContext`, since it would
  move the pricing date off the shared grid. Pathwise and likelihood-ratio
  theta are available.

### Added

- `OptionValuation.std_error()`: the Monte Carlo standard error of the
  present value, with antithetic pairs treated as one sample.
- `ImpliedVolResult.well_determined` and the `min_vega` argument, which
  identify quotes whose price barely depends on volatility (far out of the
  money, or at intrinsic value).
- `shared_time_grid(...)` and `contract_event_dates(...)` for building one
  time grid across correlated assets and contracts.
- Warnings when projected SOR does not converge, when a PDE boundary is
  within 3 standard deviations of the spot or forward, and when the grid has
  fewer than 3 nodes per standard deviation.
- Tests of the measured convergence order of the binomial, PDE and Monte
  Carlo engines, and of American puts against Longstaff and Schwartz (2001),
  Table 1.

### Changed

- **PDE domain.** `PDEParams.smax_mult` defaults to `None`: the domain is
  sized from the spot, the forward and `sigma * sqrt(T)`, which keeps the grid
  resolved for very short maturities and wide enough for long-dated,
  high-volatility options. Passing a number keeps the previous fixed span.
- **Implied-volatility tolerance.** `tol` is an absolute tolerance on the
  implied volatility for every method. For Newton-Raphson and bisection it was
  previously a price tolerance, which is loose for far out-of-the-money
  quotes.
- **Default implied-volatility solver.** Newton-Raphson for `ANALYTICAL`;
  Brent for `BINOMIAL` and `PDE_FD`, which need no vega.
- **`PricingMethod.ANALYTICAL`** is the new name for `PricingMethod.BSM`,
  which remains as an alias.
- **`ImpliedVolResult.function_evaluations`** counts every pricing, including
  numerical-vega bumps; `iterations` is the root finder's own count.
- **Cash dividends together with a dividend yield.** Cash dividends are
  discounted at `r - q`, QuantLib's convention, so the escrowed and spot-drop
  dividend models share one forward. Results with only a yield or only cash
  dividends are unchanged.
- **Discount-curve extrapolation.** Beyond the last knot the last forward
  rate continues, and before a first knot later than zero the first zero rate
  applies back to the origin. Previously the discount factor was held flat on
  both sides.

### Edge cases

- **Correlated assets with different dates.** When one asset's grid had a
  date the others lacked, such as an ex-dividend date, the assets' random
  draws were no longer aligned beyond that date. All assets in a context now
  share one grid, so the correlation holds throughout.
- **Barrier fixing on an ex-dividend date.** The PDE knock-out observed the
  barrier at the cum-dividend price on such a date. It now observes it at the
  ex-dividend price, consistent with the direct knock-in PDE solve and the
  other engines.
- **Rebate when the first hit is at the final observation.** For a discrete
  knock-out with an at-hit rebate, the PDE engine set the knocked-out nodes at
  the final observation to zero, and American Monte Carlo paid nothing on
  paths first hit at the last step. Both now pay the rebate.
- **Numerical vega below a 1% volatility** uses a forward difference, since
  the down bump would be negative. Implied volatility relies on this for
  quotes at intrinsic value.
- **Implied volatility on a steep curve.** The binomial volatility floor uses
  the tree's per-step forward rates.
- **Implied-volatility results that did not converge** are no longer reported
  as well determined.

### Documentation

- Each theta method's rate convention on a sloped curve is documented.
- Each engine's discrete-dividend model is named in the README and the
  parameter docs.
