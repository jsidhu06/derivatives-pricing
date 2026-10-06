"""Tests for forward price and put-call parity utilities."""

import datetime as dt
import numpy as np

from derivatives_pricing.enums import OptionType, ExerciseType, PricingMethod
from derivatives_pricing.market_environment import MarketData
from helpers import flat_curve
from derivatives_pricing.valuation import VanillaSpec, UnderlyingData, OptionValuation
from derivatives_pricing.utils import (
    calculate_year_fraction,
    forward_price,
    put_call_parity_rhs,
    put_call_parity_gap,
)


def test_forward_price_continuous_dividend_curve():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    ttm = calculate_year_fraction(pricing_date, maturity)
    spot = 100.0
    r = 0.05
    q = 0.02
    r_curve = flat_curve(pricing_date, maturity, r)
    q_curve = flat_curve(pricing_date, maturity, q)

    fwd = forward_price(
        spot=spot,
        pricing_date=pricing_date,
        maturity=maturity,
        discount_curve=r_curve,
        dividend_curve=q_curve,
    )

    expected = spot * np.exp((r - q) * ttm)
    assert np.isclose(fwd, expected)


def test_forward_price_discrete_dividends():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    ttm = calculate_year_fraction(pricing_date, maturity)
    spot = 100.0
    r = 0.05
    dividends = [(dt.datetime(2025, 7, 1), 0.5)]
    curve = flat_curve(pricing_date, maturity, r)

    fwd = forward_price(
        spot=spot,
        pricing_date=pricing_date,
        maturity=maturity,
        discount_curve=curve,
        discrete_dividends=dividends,
    )

    t_div = calculate_year_fraction(pricing_date, dividends[0][0])
    pv_div = 0.5 * np.exp(-r * t_div)
    expected = (spot - pv_div) * np.exp(r * ttm)
    assert np.isclose(fwd, expected)


def test_put_call_parity_rhs_and_gap():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    spot = 100.0
    strike = 100.0
    r = 0.05
    curve = flat_curve(pricing_date, maturity, r)

    rhs = put_call_parity_rhs(
        spot=spot,
        strike=strike,
        pricing_date=pricing_date,
        maturity=maturity,
        discount_curve=curve,
    )

    call_price = 10.0
    put_price = call_price - rhs

    gap = put_call_parity_gap(
        call_price=call_price,
        put_price=put_price,
        spot=spot,
        strike=strike,
        pricing_date=pricing_date,
        maturity=maturity,
        discount_curve=curve,
    )

    assert np.isclose(gap, 0.0)


def test_put_call_parity_bsm_no_dividend():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    spot = 100.0
    strike = 100.0
    r = 0.05

    curve = flat_curve(pricing_date, maturity, r)
    market_data = MarketData(pricing_date, curve, currency="USD")

    underlying_call = UnderlyingData(
        initial_value=spot,
        volatility=0.2,
        market_data=market_data,
    )
    underlying_put = UnderlyingData(
        initial_value=spot,
        volatility=0.2,
        market_data=market_data,
    )

    call_spec = VanillaSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency="USD",
    )
    put_spec = VanillaSpec(
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency="USD",
    )

    call_price = OptionValuation(
        underlying=underlying_call,
        spec=call_spec,
        pricing_method=PricingMethod.ANALYTICAL,
    ).present_value()

    put_price = OptionValuation(
        underlying=underlying_put,
        spec=put_spec,
        pricing_method=PricingMethod.ANALYTICAL,
    ).present_value()

    rhs = put_call_parity_rhs(
        spot=spot,
        strike=strike,
        pricing_date=pricing_date,
        maturity=maturity,
        discount_curve=curve,
    )

    assert np.isclose(call_price - put_price, rhs, rtol=1e-10)


def test_put_call_parity_bsm_with_dividend_curve():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    spot = 100.0
    strike = 105.0
    r = 0.04
    q = 0.02

    curve = flat_curve(pricing_date, maturity, r)
    q_curve = flat_curve(pricing_date, maturity, q)
    market_data = MarketData(pricing_date, curve, currency="USD")
    underlying_call = UnderlyingData(
        initial_value=spot,
        volatility=0.25,
        market_data=market_data,
        dividend_curve=q_curve,
    )
    underlying_put = UnderlyingData(
        initial_value=spot,
        volatility=0.25,
        market_data=market_data,
        dividend_curve=q_curve,
    )

    call_spec = VanillaSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency="USD",
    )
    put_spec = VanillaSpec(
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency="USD",
    )

    call_price = OptionValuation(
        underlying=underlying_call,
        spec=call_spec,
        pricing_method=PricingMethod.ANALYTICAL,
    ).present_value()

    put_price = OptionValuation(
        underlying=underlying_put,
        spec=put_spec,
        pricing_method=PricingMethod.ANALYTICAL,
    ).present_value()

    rhs = put_call_parity_rhs(
        spot=spot,
        strike=strike,
        pricing_date=pricing_date,
        maturity=maturity,
        discount_curve=curve,
        dividend_curve=q_curve,
    )

    assert np.isclose(call_price - put_price, rhs, rtol=1e-10)
