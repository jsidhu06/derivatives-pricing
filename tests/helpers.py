import datetime as dt
from typing import Any, Sequence

import numpy as np

from derivatives_pricing.enums import DayCountConvention, ExerciseType, OptionType
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.stochastic_processes import GBMParams, GBMProcess, SimulationConfig
from derivatives_pricing.valuation import OptionValuation, UnderlyingData, VanillaSpec


# ---------------------------------------------------------------------------
# Scalar constants (canonical source — re-exported by conftest.py)
# ---------------------------------------------------------------------------

PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)
CURRENCY = "USD"
SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
VOL = 0.20


def flat_curve(
    pricing_date: dt.datetime,
    maturity: dt.datetime,
    rate: float,
) -> DiscountCurve:
    """Flat curve at ``rate``.

    ``pricing_date`` and ``maturity`` are unused: they once sized the curve,
    and are kept so the existing call sites need not change.
    """
    return DiscountCurve.flat(rate)


_CURVE = flat_curve(PRICING_DATE, MATURITY, RATE)
_MD = MarketData(
    PRICING_DATE,
    _CURVE,
    currency=CURRENCY,
    day_count_convention=DayCountConvention.ACT_365F,
)


def market_data(
    *,
    pricing_date: dt.datetime,
    discount_curve: DiscountCurve,
    currency: str = CURRENCY,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> MarketData:
    return MarketData(
        pricing_date,
        discount_curve,
        currency=currency,
        day_count_convention=day_count_convention,
    )


def underlying(
    *,
    initial_value: float = SPOT,
    volatility: float = VOL,
    market_data: MarketData = _MD,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    dividend_curve: DiscountCurve | None = None,
) -> UnderlyingData:
    """Build UnderlyingData; signature mirrors the production dataclass."""
    return UnderlyingData(
        initial_value=initial_value,
        volatility=volatility,
        market_data=market_data,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def spec(
    option_type: OptionType = OptionType.CALL,
    exercise: ExerciseType = ExerciseType.EUROPEAN,
    strike: float = STRIKE,
    maturity: dt.datetime = MATURITY,
    currency: str = CURRENCY,
) -> VanillaSpec:
    return VanillaSpec(
        option_type=option_type,
        exercise_type=exercise,
        strike=strike,
        maturity=maturity,
        currency=currency,
    )


def pv(underlying: Any, spec: Any, method: Any, **kw: Any) -> float:
    """Shortcut: build an OptionValuation and return its present value."""

    return OptionValuation(underlying, spec, method, **kw).present_value()


def gbm(
    *,
    market_data: MarketData,
    process_params: GBMParams,
    sim_config: SimulationConfig,
    random_seed: int | None = None,
) -> GBMProcess:
    """Build GBMProcess; signature mirrors constructor plus optional eager simulate."""
    process = GBMProcess(market_data, process_params, sim_config)
    if random_seed is not None:
        process.simulate(random_seed=random_seed)
    return process


def make_vanilla_spec(
    *,
    strike: float,
    maturity: dt.datetime,
    option_type: OptionType,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    currency: str = CURRENCY,
) -> VanillaSpec:
    """Backward-compatible wrapper around ``spec``."""
    return spec(
        option_type=option_type,
        exercise=exercise_type,
        strike=strike,
        maturity=maturity,
        currency=currency,
    )


def assert_greeks_close(
    *,
    lhs: dict[str, float],
    rhs: dict[str, float | None],
    tols: dict[str, float],
    log_prefix: str,
    lhs_name: str,
    rhs_name: str,
    atol: float = 0.0,
    skip_missing_rhs: bool = False,
    logger: Any = None,
) -> None:
    """Log and assert per-greek closeness with per-greek tolerances."""
    for greek, tol in tols.items():
        rhs_val = rhs[greek]
        if rhs_val is None and skip_missing_rhs:
            if logger is not None:
                logger.info("%s %s unavailable in %s; skipping", log_prefix, greek, rhs_name)
            continue

        assert rhs_val is not None
        if logger is not None:
            logger.info(
                "%s %s | %s=%.6f %s=%.6f",
                log_prefix,
                greek,
                lhs_name,
                lhs[greek],
                rhs_name,
                rhs_val,
            )
        assert np.isclose(lhs[greek], rhs_val, rtol=tol, atol=atol), (
            f"{greek}: {lhs_name}={lhs[greek]:.6f} vs {rhs_name}={rhs_val:.6f}"
        )
