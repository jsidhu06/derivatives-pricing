"""Tests for discrete dividend consistency across pricing engines."""

import dataclasses
import datetime as dt

import numpy as np
import pytest

from derivatives_pricing.enums import ExerciseType, OptionType, PricingMethod
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from helpers import (
    market_data,
    underlying,
    spec,
    PRICING_DATE,
)
from derivatives_pricing.utils import put_call_parity_rhs, pv_discrete_dividends
from derivatives_pricing.valuation import OptionValuation, UnderlyingData
from derivatives_pricing.valuation.params import BinomialParams, MonteCarloParams, PDEParams

# Scenario: Hull-style discrete dividend case
_SPOT = 52.0
_STRIKE = 50.0
_VOL = 0.4
_RATE = 0.1
_MATURITY = PRICING_DATE + dt.timedelta(days=365)
_DIVS = [
    (PRICING_DATE + dt.timedelta(days=90), 0.5),
    (PRICING_DATE + dt.timedelta(days=270), 0.5),
]


def _build_case(
    *,
    discount_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: list[tuple[dt.datetime, float]] | None = None,
):
    divs = _DIVS if discrete_dividends is None else discrete_dividends
    rate_curve = discount_curve if discount_curve is not None else DiscountCurve.flat(_RATE)
    md = market_data(
        pricing_date=PRICING_DATE,
        discount_curve=rate_curve,
    )
    sp = spec(OptionType.PUT, ExerciseType.EUROPEAN, strike=_STRIKE, maturity=_MATURITY)
    ud = underlying(
        initial_value=_SPOT,
        volatility=_VOL,
        market_data=md,
        dividend_curve=dividend_curve,
        discrete_dividends=divs,
    )

    sim_config = SimulationConfig(paths=200_000, num_steps=52, end_date=_MATURITY)
    gbm_params = GBMParams(
        initial_value=_SPOT,
        volatility=_VOL,
        dividend_curve=dividend_curve,
        discrete_dividends=divs,
    )
    gbm = GBMProcess(md, gbm_params, sim_config)

    return md, sp, ud, gbm


def test_discrete_dividend_engine_consistency():
    md, sp, ud, gbm = _build_case()

    # MC vs PDE (jump model vs PDE jump condition)
    mc_val = OptionValuation(
        gbm, sp, PricingMethod.MONTE_CARLO, params=MonteCarloParams(random_seed=42)
    )
    mc_pv = mc_val.present_value()

    div_dates = [_DIVS[0][0], _DIVS[1][0]]
    assert all(div_date in mc_val.underlying.observation_dates for div_date in div_dates)
    assert all(div_date in mc_val.underlying.time_grid for div_date in div_dates)
    pde_val = OptionValuation(
        ud, sp, PricingMethod.PDE_FD, params=PDEParams(spot_steps=140, time_steps=140)
    )
    pde_pv = pde_val.present_value()

    assert np.isclose(mc_pv, pde_pv, rtol=0.01)

    # Binomial vs BSM (prepaid-forward adjustment)
    bsm_pv = OptionValuation(ud, sp, PricingMethod.ANALYTICAL).present_value()
    binom_pv = OptionValuation(
        ud, sp, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=1500)
    ).present_value()

    assert np.isclose(bsm_pv, binom_pv, rtol=0.01)

    # Volatility-adjusted BSM/Binomial should be closer to MC/PDE
    pv_divs = pv_discrete_dividends(
        dividends=_DIVS,
        curve_date=md.pricing_date,
        end_date=sp.maturity,
        discount_curve=md.discount_curve,
    )
    vol_multiplier = ud.initial_value / (ud.initial_value - pv_divs)
    adjusted_ud = ud.replace(volatility=ud.volatility * vol_multiplier)

    bsm_adj = OptionValuation(adjusted_ud, sp, PricingMethod.ANALYTICAL).present_value()
    binom_adj = OptionValuation(
        adjusted_ud, sp, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=1500)
    ).present_value()

    assert np.isclose(mc_pv, bsm_adj, rtol=0.02)
    assert np.isclose(pde_pv, binom_adj, rtol=0.02)


def test_mixed_yield_and_cash_dividends_put_call_parity_all_models():
    """Escrowed and spot-drop engines price off the same forward with a yield.

    Larger dividends and yield than the Hull case make discounting cash
    dividends at r rather than r - q a ~0.1 parity error, far above the
    PDE's discretisation error.
    """
    divs = [
        (PRICING_DATE + dt.timedelta(days=90), 2.0),
        (PRICING_DATE + dt.timedelta(days=270), 2.0),
    ]
    q_curve = DiscountCurve.flat(0.06)
    with pytest.warns(UserWarning, match="both dividend_curve and discrete_dividends"):
        md, put_spec, ud, _ = _build_case(dividend_curve=q_curve, discrete_dividends=divs)
    call_spec = dataclasses.replace(put_spec, option_type=OptionType.CALL)
    rhs = put_call_parity_rhs(
        spot=_SPOT,
        strike=_STRIKE,
        pricing_date=PRICING_DATE,
        maturity=_MATURITY,
        discount_curve=md.discount_curve,
        dividend_curve=q_curve,
        discrete_dividends=divs,
    )

    cases = [
        (PricingMethod.ANALYTICAL, None, 1e-10),
        (PricingMethod.BINOMIAL, BinomialParams(num_steps=500), 1e-8),
        (PricingMethod.PDE_FD, PDEParams(spot_steps=200, time_steps=200), 5e-3),
    ]
    for method, params, atol in cases:
        call = OptionValuation(ud, call_spec, method, params=params).present_value()
        put = OptionValuation(ud, put_spec, method, params=params).present_value()
        assert np.isclose(call - put, rhs, rtol=0.0, atol=atol), (method, call - put, rhs)


def test_binomial_accepts_discrete_dividends_with_nonflat_curve():
    nonflat_curve = DiscountCurve(
        times=np.array([0.0, 0.5, 1.0]),
        dfs=np.array([1.0, np.exp(-0.03 * 0.5), np.exp(-0.06 * 1.0)]),
    )
    md = MarketData(PRICING_DATE, nonflat_curve, currency="USD")
    ud = UnderlyingData(
        initial_value=_SPOT,
        volatility=_VOL,
        market_data=md,
        discrete_dividends=_DIVS,
    )
    sp = spec(OptionType.PUT, ExerciseType.EUROPEAN, strike=_STRIKE, maturity=_MATURITY)

    OptionValuation(
        ud, sp, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=400)
    ).present_value()
