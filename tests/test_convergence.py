"""Convergence-order tests.

Each test fits the log-log slope of the error against the discretisation
parameter, so a regression to a lower order fails even when every
individual price is still within a loose absolute tolerance.
"""

import datetime as dt

import numpy as np
import pytest

from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.enums import ExerciseType, OptionType, PricingMethod
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.stochastic_processes import GBMParams, GBMProcess, SimulationConfig
from derivatives_pricing.valuation import (
    BinomialParams,
    MonteCarloParams,
    OptionValuation,
    PDEParams,
    UnderlyingData,
    VanillaSpec,
)
from helpers import PRICING_DATE

_MATURITY = dt.datetime(2026, 1, 1)
_MARKET = MarketData(PRICING_DATE, DiscountCurve.flat(0.05), currency="USD")
_UNDERLYING = UnderlyingData(initial_value=100.0, volatility=0.20, market_data=_MARKET)

pytestmark = pytest.mark.slow


def _spec(option_type: OptionType, strike: float) -> VanillaSpec:
    return VanillaSpec(
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=_MATURITY,
    )


def _exact(option_type: OptionType, strike: float) -> float:
    return OptionValuation(
        _UNDERLYING, _spec(option_type, strike), PricingMethod.ANALYTICAL
    ).present_value()


def _log_log_slope(x: list[float], y: list[float]) -> float:
    """Least-squares slope of log(y) against log(x)."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_crr_tree_converges_at_first_order(option_type):
    """At the money with an even step count the strike sits on a node and the
    CRR error is a clean c / N. Away from the money the coefficient
    oscillates with the strike's position between nodes, which would make a
    fitted slope meaningless."""
    exact = _exact(option_type, 100.0)
    steps = [100, 200, 400, 800, 1600]
    errors = [
        abs(
            OptionValuation(
                _UNDERLYING,
                _spec(option_type, 100.0),
                PricingMethod.BINOMIAL,
                params=BinomialParams(num_steps=n),
            ).present_value()
            - exact
        )
        for n in steps
    ]
    slope = _log_log_slope(steps, errors)
    # Expected order -1. Flatter fails a sub-linear or plateauing error (a
    # wrong reference or a fixed bias); steeper means the fit is invalid.
    assert -1.1 < slope < -0.9, (errors, slope)


@pytest.mark.parametrize(
    "option_type,strike",
    [
        (OptionType.CALL, 100.0),
        (OptionType.PUT, 100.0),
        (OptionType.PUT, 80.0),
        (OptionType.CALL, 120.0),
    ],
)
def test_crank_nicolson_pde_converges_at_second_order(option_type, strike):
    """Uniform spot grid on [0, 400] with spot_steps = time_steps = n: the spot
    and the strike are grid nodes at every n, so the payoff kink adds no
    oscillating interpolation error and Rannacher-smoothed Crank-Nicolson is
    O(h^2 + dt^2). With the default fitted domain the strike falls between
    nodes and the error is second order only on average."""
    exact = _exact(option_type, strike)
    sizes = [200, 400, 800, 1600]
    errors = [
        abs(
            OptionValuation(
                _UNDERLYING,
                _spec(option_type, strike),
                PricingMethod.PDE_FD,
                params=PDEParams(smax_mult=4.0, spot_steps=n, time_steps=n),
            ).present_value()
            - exact
        )
        for n in sizes
    ]
    slope = _log_log_slope(sizes, errors)
    # Expected order -2. Flatter fails a first- or mixed-order scheme (fully
    # implicit gives -1.4 to -1.7) or an error floor such as domain
    # truncation; steeper means the error crosses zero inside the range.
    assert -2.2 < slope < -1.8, (errors, slope)


def test_monte_carlo_error_falls_as_one_over_root_paths():
    """The reported standard error must scale as N^(-1/2), and the price must
    stay within 4 standard errors of the closed form at every N. Plain
    sampling (no antithetic pairs or moment matching) keeps the estimator an
    average of independent draws."""
    exact = _exact(OptionType.CALL, 100.0)
    paths = [10_000, 40_000, 160_000, 640_000]
    std_errors = []
    for n in paths:
        process = GBMProcess(
            _MARKET,
            GBMParams(initial_value=100.0, volatility=0.20),
            SimulationConfig(
                paths=n, num_steps=4, end_date=_MATURITY, antithetic=False, moment_matching=False
            ),
        )
        mc = OptionValuation(
            process,
            _spec(OptionType.CALL, 100.0),
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        )
        se = mc.std_error()
        assert se is not None
        assert abs(mc.present_value() - exact) < 4.0 * se, (n, mc.present_value(), se)
        std_errors.append(se)
    slope = _log_log_slope(paths, std_errors)
    # Expected order -1/2 (central limit theorem). Flatter fails dependent
    # samples (e.g. reused random streams) or a wrong SE formula.
    assert -0.55 < slope < -0.45, (std_errors, slope)
