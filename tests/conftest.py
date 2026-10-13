"""Shared pytest fixtures for derivatives_pricing tests."""

import datetime as dt

import pytest

from derivatives_pricing.enums import (
    ExerciseType,
    OptionType,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.valuation import (
    BinomialParams,
    MonteCarloParams,
    PDEParams,
    VanillaSpec,
    UnderlyingData,
)

from helpers import (
    PRICING_DATE,
    MATURITY,
    CURRENCY,
    SPOT,
    STRIKE,
    RATE,
    VOL,
)


# ---------------------------------------------------------------------------
# Standard pricing-method parameters
# ---------------------------------------------------------------------------

BINOM_PARAMS = BinomialParams(num_steps=500)
MC_PARAMS = MonteCarloParams(random_seed=42)
PDE_PARAMS = PDEParams(spot_steps=200, time_steps=200, max_iter=20_000)


@pytest.fixture()
def pricing_date() -> dt.datetime:
    return PRICING_DATE


@pytest.fixture()
def maturity() -> dt.datetime:
    return MATURITY


@pytest.fixture()
def currency() -> str:
    return CURRENCY


@pytest.fixture()
def risk_free_rate() -> float:
    return RATE


@pytest.fixture()
def vol() -> float:
    return VOL


@pytest.fixture()
def spot() -> float:
    return SPOT


@pytest.fixture()
def strike() -> float:
    return STRIKE


# ---------------------------------------------------------------------------
# Curve / Market Data
# ---------------------------------------------------------------------------


@pytest.fixture()
def discount_curve(
    pricing_date: dt.datetime, maturity: dt.datetime, risk_free_rate: float
) -> DiscountCurve:
    """Flat discount curve over [pricing_date, maturity]."""
    return DiscountCurve.flat(risk_free_rate)


@pytest.fixture()
def market_data(
    pricing_date: dt.datetime, discount_curve: DiscountCurve, currency: str
) -> MarketData:
    return MarketData(pricing_date, discount_curve, currency=currency)


@pytest.fixture()
def underlying_data(market_data: MarketData) -> UnderlyingData:
    """ATM underlying with no dividends."""
    return UnderlyingData(
        initial_value=SPOT,
        volatility=VOL,
        market_data=market_data,
    )


# ---------------------------------------------------------------------------
# Option specs
# ---------------------------------------------------------------------------


@pytest.fixture()
def euro_call_spec(strike: float, maturity: dt.datetime, currency: str) -> VanillaSpec:
    return VanillaSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency=currency,
    )


@pytest.fixture()
def euro_put_spec(strike: float, maturity: dt.datetime, currency: str) -> VanillaSpec:
    return VanillaSpec(
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency=currency,
    )


# ---------------------------------------------------------------------------
# Common discrete dividends
# ---------------------------------------------------------------------------


@pytest.fixture()
def standard_discrete_dividends(pricing_date: dt.datetime) -> list[tuple[dt.datetime, float]]:
    """Two semi-annual dividends of 0.50 each."""
    return [
        (pricing_date + dt.timedelta(days=90), 0.50),
        (pricing_date + dt.timedelta(days=270), 0.50),
    ]


# ---------------------------------------------------------------------------
# Skip slow tests by default
# -------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
