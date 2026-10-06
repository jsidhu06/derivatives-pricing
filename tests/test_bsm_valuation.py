"""Tests for Black-Scholes-Merton valuation implementation."""

import datetime as dt

import numpy as np

from derivatives_pricing.enums import OptionType, PricingMethod
from conftest import (
    PRICING_DATE,
    MATURITY,
    RATE,
    SPOT,
    STRIKE,
)
from helpers import flat_curve, underlying, pv, spec
from derivatives_pricing.valuation import (
    VanillaSpec,
    OptionValuation,
    UnderlyingData,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BSM reference values (S=100, K=100, r=0.05, σ=0.20, T=1, no divs)
# ---------------------------------------------------------------------------
_BSM_ATM_CALL = 10.4506
_BSM_ATM_PUT = 5.5735
_BSM_ITM_CALL_110 = 17.6630  # S=110
_BSM_OTM_PUT_110 = 2.7859  # S=110


def _bsm(ud: UnderlyingData, sp: VanillaSpec) -> float:
    return pv(ud, sp, PricingMethod.ANALYTICAL)


class TestBSMValuation:
    """Tests for Black-Scholes-Merton valuation implementation."""

    def test_bsm_call_option_atm(self):
        """BSM ATM call price matches closed-form Black-Scholes."""
        result = _bsm(underlying(), spec())
        assert np.isclose(result, _BSM_ATM_CALL, rtol=1e-4)

    def test_bsm_discrete_dividends_reduce_call_price(self):
        """Discrete dividends should reduce European call price (all else equal)."""
        sp = spec()
        pv_no_div = _bsm(underlying(), sp)
        pv_div = _bsm(
            underlying(discrete_dividends=[(PRICING_DATE + dt.timedelta(days=180), 1.0)]),
            sp,
        )
        assert pv_div < pv_no_div

    def test_bsm_put_option_atm(self):
        """BSM ATM put price matches closed-form Black-Scholes."""
        result = _bsm(underlying(), spec(OptionType.PUT))
        assert np.isclose(result, _BSM_ATM_PUT, rtol=1e-4)

    def test_bsm_call_itm(self):
        """BSM ITM call: exceeds no-arbitrage lower bound and matches reference."""
        result = _bsm(underlying(initial_value=110.0), spec())
        lower_bound = 110.0 - STRIKE * np.exp(-RATE * 1.0)
        assert result > lower_bound
        assert np.isclose(result, _BSM_ITM_CALL_110, rtol=1e-4)

    def test_bsm_put_otm(self):
        """BSM OTM put: positive time value, matches reference."""
        result = _bsm(underlying(initial_value=110.0), spec(OptionType.PUT))
        assert np.isclose(result, _BSM_OTM_PUT_110, rtol=1e-4)

    def test_bsm_with_dividend_curve(self):
        """Test BSM pricing with dividend curve."""
        q_curve = flat_curve(PRICING_DATE, MATURITY, 0.03)
        sp = spec()
        pv_no_div = _bsm(underlying(), sp)
        pv_with_div = _bsm(underlying(dividend_curve=q_curve), sp)
        assert pv_with_div < pv_no_div

    def test_bsm_call_put_parity(self):
        """Test BSM call-put parity: C - P = S*exp(-q*T) - K*exp(-r*T)."""
        ud = underlying()
        call_price = _bsm(ud, spec(OptionType.CALL))
        put_price = _bsm(underlying(), spec(OptionType.PUT))

        T = 1.0
        parity_rhs = SPOT * np.exp(-0.0 * T) - STRIKE * np.exp(-RATE * T)
        assert np.isclose(call_price - put_price, parity_rhs, rtol=1e-10)

    def test_bsm_present_value_returns_float(self):
        """Test BSM present_value returns a scalar float."""
        valuation = OptionValuation(underlying(), spec(), PricingMethod.ANALYTICAL)
        result = valuation.present_value()
        assert isinstance(result, float)
