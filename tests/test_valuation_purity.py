# tests/test_valuation_purity.py
"""Purity / non-mutation tests for OptionValuation.

These tests assert that:
- present_value() does not mutate UnderlyingData (or curves)
- greeks (delta/gamma/vega) may bump parameters internally, but must restore state

Rationale
---------
Users will commonly reuse one UnderlyingData instance across multiple valuations/methods.
The library should guarantee that pricing/greeks do not leave surprising side-effects behind.
"""

import datetime as dt

import pytest

from derivatives_pricing.enums import (
    ExerciseType,
    OptionType,
    PricingMethod,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from helpers import flat_curve
from derivatives_pricing.valuation import VanillaSpec, OptionValuation, UnderlyingData
from derivatives_pricing.valuation import BinomialParams


def _make_ud(
    *,
    spot: float = 100.0,
    vol: float = 0.20,
    pricing_date: dt.datetime,
    discount_curve: DiscountCurve,
    currency: str = "USD",
    dividend_curve: DiscountCurve | None = None,
) -> UnderlyingData:
    market_data = MarketData(pricing_date, discount_curve, currency=currency)
    return UnderlyingData(
        initial_value=spot,
        volatility=vol,
        market_data=market_data,
        dividend_curve=dividend_curve,
    )


def _make_spec(
    *,
    option_type: OptionType,
    strike: float,
    maturity: dt.datetime,
    currency: str = "USD",
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
) -> VanillaSpec:
    return VanillaSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=maturity,
        currency=currency,
    )


def _snapshot_ud(ud: UnderlyingData) -> dict:
    """Capture the fields we expect not to change after pricing/greeks."""
    # Add to this if UnderlyingData grows.
    return {
        "initial_value": ud.initial_value,
        "volatility": ud.volatility,
        "pricing_date": ud.pricing_date,
        # identity + rate (curve object should remain same & rate unchanged)
        "discount_curve_id": id(ud.discount_curve),
        "discount_curve_rate": float(ud.discount_curve.flat_rate),
        "dividend_curve_id": id(ud.dividend_curve) if ud.dividend_curve is not None else None,
        "dividend_curve_rate": (
            float(ud.dividend_curve.flat_rate) if ud.dividend_curve is not None else None
        ),
    }


@pytest.fixture()
def common_setup():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    rate = 0.05
    curve = flat_curve(pricing_date, maturity, rate)
    return pricing_date, maturity, curve


class TestValuationPurityPresentValue:
    def test_present_value_does_not_mutate_underlyingdata_across_methods(self, common_setup):
        """Reusing the same UnderlyingData across pricing methods should be safe.

        This test intentionally reuses *one* UnderlyingData instance to match typical user behaviour.
        """
        pricing_date, maturity, csr = common_setup

        ud = _make_ud(
            spot=100.0,
            vol=0.20,
            pricing_date=pricing_date,
            discount_curve=csr,
            dividend_curve=flat_curve(pricing_date, maturity, 0.02),
        )
        spec = _make_spec(option_type=OptionType.CALL, strike=100.0, maturity=maturity)

        baseline = _snapshot_ud(ud)

        # BSM pricing
        bsm = OptionValuation(ud, spec, PricingMethod.ANALYTICAL)
        _ = bsm.present_value()
        assert _snapshot_ud(ud) == baseline, "present_value() (BSM) mutated UnderlyingData"

        # Binomial pricing (re-using same ud)
        tree = OptionValuation(
            ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=2000)
        )
        _ = tree.present_value()
        assert _snapshot_ud(ud) == baseline, "present_value() (Binomial) mutated UnderlyingData"


class TestValuationPurityGreeks:
    def test_delta_restores_underlying_spot(self, common_setup):
        pricing_date, maturity, csr = common_setup

        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=csr)
        spec = _make_spec(option_type=OptionType.CALL, strike=100.0, maturity=maturity)
        val = OptionValuation(ud, spec, PricingMethod.ANALYTICAL)

        baseline = _snapshot_ud(ud)
        _ = val.delta()  # may bump internally
        assert _snapshot_ud(ud) == baseline, "delta() did not restore UnderlyingData state"

    def test_gamma_restores_underlying_spot(self, common_setup):
        pricing_date, maturity, csr = common_setup

        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=csr)
        spec = _make_spec(option_type=OptionType.CALL, strike=100.0, maturity=maturity)
        val = OptionValuation(ud, spec, PricingMethod.ANALYTICAL)

        baseline = _snapshot_ud(ud)
        _ = val.gamma()  # may bump internally
        assert _snapshot_ud(ud) == baseline, "gamma() did not restore UnderlyingData state"

    def test_vega_restores_underlying_volatility(self, common_setup):
        pricing_date, maturity, csr = common_setup

        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=csr)
        spec = _make_spec(option_type=OptionType.CALL, strike=100.0, maturity=maturity)
        val = OptionValuation(ud, spec, PricingMethod.ANALYTICAL)

        baseline = _snapshot_ud(ud)
        _ = val.vega()  # may bump internally
        assert _snapshot_ud(ud) == baseline, "vega() did not restore UnderlyingData state"

    def test_binomial_greeks_do_not_mutate_underlyingdata(self, common_setup):
        """Binomial greeks are still numerical bumps; ensure restoration works there too."""
        pricing_date, maturity, csr = common_setup

        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=csr)
        spec = _make_spec(option_type=OptionType.CALL, strike=100.0, maturity=maturity)
        val = OptionValuation(
            ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=2000)
        )

        baseline = _snapshot_ud(ud)

        _ = val.delta()
        assert _snapshot_ud(ud) == baseline, "Binomial delta() did not restore UnderlyingData state"

        _ = val.gamma()
        assert _snapshot_ud(ud) == baseline, "Binomial gamma() did not restore UnderlyingData state"

        _ = val.vega()
        assert _snapshot_ud(ud) == baseline, "Binomial vega() did not restore UnderlyingData state"


class TestUnderlyingDataReplace:
    """Test the immutable replace pattern on UnderlyingData."""

    def test_replace_volatility(self, common_setup):
        pricing_date, maturity, curve = common_setup
        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=curve)
        ud2 = ud.replace(volatility=0.30)
        assert ud2.volatility == 0.30
        assert ud.volatility == 0.20

    def test_replace_initial_value(self, common_setup):
        pricing_date, maturity, curve = common_setup
        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=curve)
        ud2 = ud.replace(initial_value=110.0)
        assert ud2.initial_value == 110.0
        assert ud.initial_value == 100.0

    def test_replace_preserves_other_fields(self, common_setup):
        pricing_date, maturity, curve = common_setup
        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=curve)
        ud2 = ud.replace(volatility=0.30)
        assert ud2.initial_value == ud.initial_value
        assert ud2.market_data is ud.market_data
        assert ud2.dividend_curve is ud.dividend_curve
        assert ud2.discrete_dividends == ud.discrete_dividends

    def test_replace_returns_new_instance(self, common_setup):
        pricing_date, maturity, curve = common_setup
        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=curve)
        ud2 = ud.replace(volatility=0.20)
        assert ud is not ud2
