# derivatives-pricing

[![CI](https://github.com/jsidhu06/derivatives-pricing/actions/workflows/ci.yml/badge.svg)](https://github.com/jsidhu06/derivatives-pricing/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jsidhu06/derivatives-pricing/branch/main/graph/badge.svg)](https://codecov.io/gh/jsidhu06/derivatives-pricing)
[![PyPI version](https://img.shields.io/pypi/v/derivatives_pricing)](https://pypi.org/project/derivatives-pricing/)
[![Python versions](https://img.shields.io/pypi/pyversions/derivatives_pricing)](https://pypi.org/project/derivatives-pricing/)

A Python package for options pricing and Greeks computation, with a unified API
across analytical, binomial tree, finite difference and Monte Carlo methods.

Built for teaching, research, and production-adjacent workflows.

---

## Features

### Pricing Coverage

| | Vanilla | Asian | Barrier | Custom |
|---|:---:|:---:|:---:|:---:|
| BSM | E | E | E | — |
| Binomial | E/A | E/A | E/A | E/A |
| PDE_FD | E/A | — | E/A | E/A |
| Monte Carlo | E/A | E/A | E/A | E/A |

E = European, A = American

**Method details:** BSM uses closed-form Black-Scholes-Merton; Binomial uses Cox-Ross-Rubinstein trees;
PDE_FD supports implicit, explicit, and Crank-Nicolson finite difference schemes;
Monte Carlo uses Longstaff-Schwartz for American-style exercise.
Asian analytical pricing uses Turnbull-Wakeman (arithmetic) and Kemna-Vorst (geometric),
with Hull averaging on binomial trees. Barrier pricing supports continuous and discrete monitoring,
knock-in and knock-out structures, and rebates.

---

### Additional Capabilities

- **Greeks** — analytical, tree, grid, pathwise, likelihood-ratio, and numerical bump-and-revalue (delta, gamma, vega, theta, rho)
- **Implied volatility** — Newton-Raphson, bisection, and Brent solvers with arbitrage-bounds checking
- **Stochastic processes** — Geometric Brownian Motion, Jump Diffusion (Merton), Square-Root Diffusion (CIR)
- **Discount curves** — log-linear interpolation on arbitrary term structures; deterministic time-varying forward rate and dividend curves
- **Discrete dividends** — supported across all pricing methods
- **Barrier options** — continuous and discrete monitoring, knock-in/knock-out, rebates (at-hit and at-expiry)
- **Control variates** — European analytical control variates for American pricing variance reduction
- **Custom payoffs** — user-defined payoff functions via `PayoffSpec`

---

## Why derivatives-pricing?

- Consistent API across analytical, tree, finite difference, and Monte Carlo methods  
- Designed for transparency and clarity of implementation  
- Includes vanilla, Asian, barrier, and custom-payoff workflows behind the same facade  
- Suitable for teaching, experimentation, research, and production-adjacent workflows  
- Extensible architecture for new models and payoffs  

---

## Installation

Install from PyPI:

```bash
pip install derivatives-pricing
# or pip install derivatives-pricing[numba] for optional PDE_FD solver acceleration
```

### Beta feature — barrier options
Barrier option pricing is currently
available as a pre-release. To install:
```bash
pip install --pre -U derivatives-pricing
```

For development:

```bash
pip install -r requirements/dev.txt   
pip install -e . --no-deps                                                                         
pip install -e ".[jupyter]"           # optional, for running notebooks
```

Requires Python 3.10 – 3.13

## Quick Start

```python
import datetime as dt

import derivatives_pricing as dp

pricing_date = dt.datetime(2025, 1, 1)
maturity = dt.datetime(2025, 7, 1)

dc = dp.DiscountCurve.flat(rate=0.05, end_time=1.0)
md = dp.MarketData(
    pricing_date=pricing_date,
    discount_curve=dc,
    currency="USD",
)

underlying = dp.UnderlyingData(
    initial_value=100.0,
    volatility=0.20,
    market_data=md,
)

spec = dp.VanillaSpec(
    option_type=dp.OptionType.CALL,
    exercise_type=dp.ExerciseType.EUROPEAN,
    strike=105.0,
    maturity=maturity,
)

val = dp.OptionValuation(
    underlying=underlying,
    spec=spec,
    pricing_method=dp.PricingMethod.BSM,
)
print(f"{'PV:':<8} {val.present_value():>10.4f}")
print(f"{'Delta:':<8} {val.delta():>10.4f}")
```

## Examples & Tutorials

The repo includes two companion directories:

- **`examples/`** — concise notebooks showing how to call the public API for each feature
  (European and American vanilla options, Asians, barriers, Greeks, jump diffusion, discount curves).
- **`tutorials/`** — deeper walkthroughs that teach the theory behind each pricing method
  (BSM, binomial trees, finite differences, Monte Carlo, Asian averaging, barrier pricing).
  Tutorials may access private/internal classes for demonstration purposes.

## Tests

```bash
pytest -q -n auto --runslow
```

## Project Structure

```
src/derivatives_pricing/
├── enums.py                  # OptionType, ExerciseType, PricingMethod, …
├── exceptions.py             # Custom exception hierarchy
├── market_environment.py     # MarketData, CorrelationContext
├── rates.py                  # DiscountCurve (log-linear interpolation)
├── stochastic_processes.py   # GBM, JDProcess, SRD, PathSimulation
├── utils.py                  # Day-count, forward price, put-call parity
├── valuation/
│   ├── asian_analytical.py   # Turnbull-Wakeman, Kemna-Vorst
│   ├── barrier_analytical.py # Analytical barrier pricing
│   ├── binomial.py           # Cox-Ross-Rubinstein tree
│   ├── bsm.py                # Closed-form Black-Scholes-Merton
│   ├── contracts.py          # VanillaSpec, BarrierSpec, PayoffSpec, AsianSpec
│   ├── core.py               # OptionValuation facade, UnderlyingData
│   ├── implied_volatility.py # IV solver
│   ├── monte_carlo.py        # Monte Carlo with Longstaff-Schwartz and barrier pricing
│   ├── params.py             # MonteCarloParams, BinomialParams, PDEParams
│   └── pde.py                # Finite difference (implicit, explicit, Crank-Nicolson, barriers)
tests/                        # Test suite
examples/                     # API usage notebooks
tutorials/                    # Theory deep-dive notebooks
```

## Roadmap

Planned: stochastic volatility models.

Found a bug or have a feature request? [Open an issue](https://github.com/jsidhu06/derivatives-pricing/issues).

## Disclaimer

This pricing library is provided as-is. Users are responsible for
independently verifying any results before relying on them.

## License

This repository uses a **dual-license model**.

| Component | License | SPDX |
|-----------|---------|------|
| `src/derivatives_pricing/` | [MIT](https://opensource.org/licenses/MIT) | `MIT` |
| `examples/` | [MIT](https://opensource.org/licenses/MIT) | `MIT` |
| `tutorials/` | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | `CC-BY-NC-SA-4.0` |

**Library code and examples** are released under the **MIT License** and may be
freely used, modified, and redistributed.

The **PyPI distribution** contains the MIT-licensed library package. Tutorial
materials are separately hosted in the GitHub repository and are not part of
the published package.

**Tutorial notebooks** in `tutorials/` are licensed under **Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)**. You may share and
adapt these materials for **non-commercial purposes with attribution**. See the
[full license text](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
for details.