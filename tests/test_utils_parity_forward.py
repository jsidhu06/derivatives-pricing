"""Tests for forward price and put-call parity utilities."""

import datetime as dt
import numpy as np
import pytest

from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.enums import OptionType, ExerciseType, PricingMethod
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.valuation import VanillaSpec, UnderlyingData, OptionValuation
from derivatives_pricing.utils import (
    calculate_year_fraction,
    forward_price,
    pv_discrete_dividends,
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
    r_curve = DiscountCurve.flat(r)
    q_curve = DiscountCurve.flat(q)

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
    curve = DiscountCurve.flat(r)

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


def test_forward_price_yield_and_discrete_dividends_discounts_at_carry():
    """With a yield, each cash dividend is discounted at r - q (QuantLib)."""
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    ttm = calculate_year_fraction(pricing_date, maturity)
    spot, r, q, amount = 100.0, 0.05, 0.03, 2.0
    ex_date = dt.datetime(2025, 7, 1)
    r_curve = DiscountCurve.flat(r)
    q_curve = DiscountCurve.flat(q)

    with pytest.warns(UserWarning, match="Both dividend_curve and discrete_dividends"):
        fwd = forward_price(
            spot=spot,
            pricing_date=pricing_date,
            maturity=maturity,
            discount_curve=r_curve,
            dividend_curve=q_curve,
            discrete_dividends=[(ex_date, amount)],
        )

    t_div = calculate_year_fraction(pricing_date, ex_date)
    pv_div = amount * np.exp(-(r - q) * t_div)
    expected = (spot - pv_div) * np.exp((r - q) * ttm)
    assert np.isclose(fwd, expected, rtol=1e-12)
    # Spot-drop view: grow S to the ex-date, drop D, grow to maturity.
    spot_drop = (spot * np.exp((r - q) * t_div) - amount) * np.exp((r - q) * (ttm - t_div))
    assert np.isclose(fwd, spot_drop, rtol=1e-12)


def test_pv_discrete_dividends_from_intermediate_start_discounts_at_carry():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    r, q = 0.05, 0.03
    start = dt.datetime(2025, 4, 1)
    ex_date = dt.datetime(2025, 10, 1)
    pv = pv_discrete_dividends(
        [(ex_date, 1.0)],
        curve_date=pricing_date,
        end_date=maturity,
        discount_curve=DiscountCurve.flat(r),
        start_date=start,
        dividend_curve=DiscountCurve.flat(q),
    )
    tau = calculate_year_fraction(start, ex_date)
    assert np.isclose(pv, np.exp(-(r - q) * tau), rtol=1e-12)


def test_put_call_parity_rhs_and_gap():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    spot = 100.0
    strike = 100.0
    r = 0.05
    curve = DiscountCurve.flat(r)

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

    curve = DiscountCurve.flat(r)
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

    curve = DiscountCurve.flat(r)
    q_curve = DiscountCurve.flat(q)
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
