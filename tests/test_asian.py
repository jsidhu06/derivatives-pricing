"""Tests for Asian option pricing — analytical, Monte Carlo, and binomial.

Covers:
- Geometric: Kemna-Vorst (1990) exact closed-form
- Arithmetic: Turnbull-Wakeman (1991) moment-matching (Hull §26.13)
- Seasoned Asians (Hull K* adjustment)
- Binomial Hull tree, Binomial MC
- American Asian (Longstaff-Schwartz LSM)
- Fixing-dates and economic-property tests
"""

import datetime as dt
from dataclasses import dataclass, replace as dc_replace
import logging
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from derivatives_pricing.enums import (
    AsianAveraging,
    ExerciseType,
    OptionType,
    PricingMethod,
    GreekCalculationMethod,
)
from derivatives_pricing.exceptions import UnsupportedFeatureError, ValidationError
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from helpers import (
    flat_curve,
    market_data,
    underlying,
)
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from derivatives_pricing.utils import pv_discrete_dividends
from derivatives_pricing.valuation import VanillaSpec, OptionValuation, UnderlyingData
from derivatives_pricing.valuation.asian_analytical import (
    _asian_arithmetic_analytical,
    _asian_geometric_analytical,
)
from derivatives_pricing.valuation.binomial import _BinomialAsianValuation
from derivatives_pricing.valuation.core import AsianSpec
from derivatives_pricing.valuation.params import BinomialParams, MonteCarloParams


logger = logging.getLogger(__name__)


PRICING_DATE = dt.datetime(2025, 1, 1)
CURRENCY = "USD"
MC_PATHS = 200_000
MC_SEED = 42
NUM_STEPS = 60
NUM_OBSERVATIONS = NUM_STEPS + 1
BINOM_STEPS = 60
ASIAN_TREE_AVERAGES = 100
DEFAULT_SHORT_RATE = 0.03


def _market_data(
    maturity: dt.datetime,
    discount_curve: DiscountCurve | None = None,
) -> MarketData:
    curve = (
        discount_curve
        if discount_curve is not None
        else flat_curve(PRICING_DATE, maturity, DEFAULT_SHORT_RATE)
    )
    return market_data(
        pricing_date=PRICING_DATE,
        discount_curve=curve,
        currency=CURRENCY,
    )


def _flat_dividend_curve(dividend_yield: float, maturity: dt.datetime) -> DiscountCurve | None:
    if dividend_yield == 0.0:
        return None
    return flat_curve(PRICING_DATE, maturity, dividend_yield)


def _underlying(
    *,
    spot: float,
    vol: float,
    maturity: dt.datetime,
    discount_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
) -> UnderlyingData:
    rate_curve = (
        discount_curve
        if discount_curve is not None
        else flat_curve(PRICING_DATE, maturity, DEFAULT_SHORT_RATE)
    )
    return underlying(
        initial_value=spot,
        volatility=vol,
        market_data=_market_data(maturity, discount_curve=rate_curve),
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def _gbm_underlying(
    *,
    spot: float,
    vol: float,
    maturity: dt.datetime,
    paths: int,
    num_steps: int,
    discount_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
) -> GBMProcess:
    rate_curve = (
        discount_curve
        if discount_curve is not None
        else flat_curve(PRICING_DATE, maturity, DEFAULT_SHORT_RATE)
    )
    sim_config = SimulationConfig(
        paths=paths,
        num_steps=num_steps,
        end_date=maturity,
    )
    gbm_params = GBMParams(
        initial_value=spot,
        volatility=vol,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )
    return GBMProcess(
        _market_data(maturity, discount_curve=rate_curve),
        gbm_params,
        sim_config,
    )


def _asian_spec(
    *,
    strike: float,
    maturity: dt.datetime,
    option_type: OptionType,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    averaging: AsianAveraging = AsianAveraging.ARITHMETIC,
    num_observations: int | None = NUM_OBSERVATIONS,
    averaging_start: dt.datetime | None = None,
    observed_average: float | None = None,
    observed_count: int | None = None,
) -> AsianSpec:
    return AsianSpec(
        averaging=averaging,
        option_type=option_type,
        strike=strike,
        maturity=maturity,
        currency=CURRENCY,
        exercise_type=exercise_type,
        num_observations=num_observations,
        averaging_start=averaging_start,
        observed_average=observed_average,
        observed_count=observed_count,
    )


@dataclass(frozen=True, slots=True)
class _ThreeMethodCase:
    spot: float
    strike: float
    vol: float
    short_rate: float
    dividend_yield: float
    days: int
    option_type: OptionType
    spec_observations: int
    mc_grid_steps: int
    binom_mc_steps: int
    hull_steps: int
    averaging_start_days: int | None
    observed_average: float | None
    observed_count: int | None
    mc_paths: int
    rtol: float


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _ThreeMethodCase(
                spot=100.0,
                strike=100.0,
                vol=0.20,
                short_rate=0.03,
                dividend_yield=0.00,
                days=365,
                option_type=OptionType.CALL,
                spec_observations=61,
                mc_grid_steps=60,
                binom_mc_steps=60,
                hull_steps=60,
                averaging_start_days=None,
                observed_average=None,
                observed_count=None,
                mc_paths=120_000,
                rtol=0.03,
            ),
            id="fresh_atm_balanced",
        ),
        pytest.param(
            _ThreeMethodCase(
                spot=110.0,
                strike=100.0,
                vol=0.25,
                short_rate=0.01,
                dividend_yield=0.015,
                days=270,
                option_type=OptionType.PUT,
                spec_observations=37,
                mc_grid_steps=72,
                binom_mc_steps=72,
                hull_steps=72,
                averaging_start_days=None,
                observed_average=None,
                observed_count=None,
                mc_paths=100_000,
                rtol=0.03,
            ),
            id="fresh_put_div_decoupled_steps",
        ),
        pytest.param(
            _ThreeMethodCase(
                spot=95.0,
                strike=90.0,
                vol=0.30,
                short_rate=0.05,
                dividend_yield=0.01,
                days=540,
                option_type=OptionType.CALL,
                spec_observations=49,
                mc_grid_steps=80,
                binom_mc_steps=80,
                hull_steps=80,
                averaging_start_days=90,
                observed_average=None,
                observed_count=None,
                mc_paths=100_000,
                rtol=0.03,
            ),
            id="forward_start_fresh",
        ),
        pytest.param(
            _ThreeMethodCase(
                spot=100.0,
                strike=100.0,
                vol=0.20,
                short_rate=0.03,
                dividend_yield=0.01,
                days=365,
                option_type=OptionType.CALL,
                spec_observations=61,
                mc_grid_steps=60,
                binom_mc_steps=60,
                hull_steps=60,
                averaging_start_days=None,
                observed_average=102.0,
                observed_count=5,
                mc_paths=120_000,
                rtol=0.03,
            ),
            id="seasoned_fresh_window",
        ),
        pytest.param(
            _ThreeMethodCase(
                spot=100.0,
                strike=100.0,
                vol=0.20,
                short_rate=0.03,
                dividend_yield=0.01,
                days=365,
                option_type=OptionType.CALL,
                spec_observations=61,
                mc_grid_steps=60,
                binom_mc_steps=60,
                hull_steps=60,
                averaging_start_days=60,
                observed_average=102.0,
                observed_count=5,
                mc_paths=120_000,
                rtol=0.03,
            ),
            id="seasoned_forward_start",
        ),
    ],
)
def test_asian_three_method_convergence_across_schedule_variants(case: _ThreeMethodCase):
    """Cross-check Asian prices across stochastic MC, binomial MC, and Hull tree.

    This test intentionally decouples contract schedule resolution (AsianSpec)
    from pricing-engine numerical resolution (MC/binomial step counts) and
    covers fresh, forward-start, seasoned, and forward-start seasoned cases.
    """
    maturity = PRICING_DATE + dt.timedelta(days=case.days)
    averaging_start = (
        None
        if case.averaging_start_days is None
        else PRICING_DATE + dt.timedelta(days=case.averaging_start_days)
    )
    spec = _asian_spec(
        strike=case.strike,
        maturity=maturity,
        option_type=case.option_type,
        num_observations=case.spec_observations,
        averaging_start=averaging_start,
        observed_average=case.observed_average,
        observed_count=case.observed_count,
    )
    q_curve = _flat_dividend_curve(case.dividend_yield, maturity)

    mc_underlying = _gbm_underlying(
        spot=case.spot,
        vol=case.vol,
        discount_curve=flat_curve(PRICING_DATE, maturity, case.short_rate),
        dividend_curve=q_curve,
        maturity=maturity,
        paths=case.mc_paths,
        num_steps=case.mc_grid_steps,
    )
    mc_pv = OptionValuation(
        mc_underlying,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    binom_underlying = _underlying(
        spot=case.spot,
        vol=case.vol,
        discount_curve=flat_curve(PRICING_DATE, maturity, case.short_rate),
        dividend_curve=q_curve,
        maturity=maturity,
    )

    binom_mc_pv = OptionValuation(
        binom_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=case.binom_mc_steps,
            mc_paths=case.mc_paths,
            random_seed=MC_SEED,
        ),
    ).present_value()

    hull_pv = OptionValuation(
        binom_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=case.hull_steps,
            asian_tree_averages=2 * case.hull_steps,
        ),
    ).present_value()

    logger.info(
        "Asian 3-method %s S=%.2f K=%.2f vol=%.2f r=%.2f q=%.2f days=%d "
        "spec_observations=%d mc_grid=%d binom_mc=%d hull_steps=%d avg_start=%s n1=%s\n"
        "MC=%.6f Hull=%.6f BinomMC=%.6f",
        case.option_type.value,
        case.spot,
        case.strike,
        case.vol,
        case.short_rate,
        case.dividend_yield,
        case.days,
        case.spec_observations,
        case.mc_grid_steps,
        case.binom_mc_steps,
        case.hull_steps,
        case.averaging_start_days,
        case.observed_count,
        mc_pv,
        hull_pv,
        binom_mc_pv,
    )

    assert np.isclose(mc_pv, hull_pv, rtol=case.rtol)
    assert np.isclose(mc_pv, binom_mc_pv, rtol=case.rtol)
    assert np.isclose(binom_mc_pv, hull_pv, rtol=case.rtol)


@pytest.mark.parametrize(
    "div_days,div_amt,rtol_mc_adj,mc_paths",
    [
        ([90, 270], 0.5, 0.03, 100_000),
        ([30, 330], 0.5, 0.05, 50_000),
        ([60, 300], 1.0, 0.06, 50_000),
        ([30, 120, 210, 330], 0.25, 0.05, 50_000),
    ],
)
def test_asian_discrete_dividends_binomial_hull_vs_mc(div_days, div_amt, rtol_mc_adj, mc_paths):
    spot = 52.0
    strike = 50.0
    vol = 0.4
    short_rate = 0.1
    days = 365
    maturity = PRICING_DATE + dt.timedelta(days=days)
    divs = [(PRICING_DATE + dt.timedelta(days=day), div_amt) for day in div_days]

    spec = _asian_spec(strike=strike, maturity=maturity, option_type=OptionType.CALL)

    mc_underlying = _gbm_underlying(
        spot=spot,
        vol=vol,
        discount_curve=flat_curve(PRICING_DATE, maturity, short_rate),
        discrete_dividends=divs,
        maturity=maturity,
        paths=mc_paths,
        num_steps=NUM_STEPS,
    )
    mc_pv = OptionValuation(
        mc_underlying,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    binom_underlying = _underlying(
        spot=spot,
        vol=vol,
        discount_curve=flat_curve(PRICING_DATE, maturity, short_rate),
        discrete_dividends=divs,
        maturity=maturity,
    )

    binom_mc_pv = OptionValuation(
        binom_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS * 2,
            mc_paths=MC_PATHS,
            random_seed=MC_SEED,
        ),
    ).present_value()

    hull_pv = OptionValuation(
        binom_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    pv_divs = pv_discrete_dividends(
        dividends=divs,
        curve_date=PRICING_DATE,
        end_date=maturity,
        discount_curve=binom_underlying.discount_curve,
    )
    vol_multiplier = spot / (spot - pv_divs)
    adjusted_underlying = binom_underlying.replace(
        volatility=binom_underlying.volatility * vol_multiplier
    )
    adjusted_hull_pv = OptionValuation(
        adjusted_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    logger.info(
        "Asian disc-div S=%.2f K=%.2f vol=%.2f r=%.2f divs=%d\n"
        "MC=%.6f Hull=%.6f BinomMC=%.6f HullAdj=%.6f",
        spot,
        strike,
        vol,
        short_rate,
        len(divs),
        mc_pv,
        hull_pv,
        binom_mc_pv,
        adjusted_hull_pv,
    )

    assert np.isclose(binom_mc_pv, hull_pv, rtol=0.02)
    assert np.isclose(mc_pv, adjusted_hull_pv, rtol=rtol_mc_adj)


@pytest.mark.parametrize(
    "spot,strike,vol,short_rate,days,option_type",
    [
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.CALL),
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.PUT),
        (110.0, 100.0, 0.25, 0.01, 270, OptionType.CALL),
        (110.0, 100.0, 0.25, 0.01, 270, OptionType.PUT),
    ],
)
def test_asian_american_at_least_european_hull(spot, strike, vol, short_rate, days, option_type):
    maturity = PRICING_DATE + dt.timedelta(days=days)
    euro_spec = _asian_spec(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
    )
    amer_spec = _asian_spec(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        exercise_type=ExerciseType.AMERICAN,
    )

    binom_underlying = _underlying(
        spot=spot,
        vol=vol,
        discount_curve=flat_curve(PRICING_DATE, maturity, short_rate),
        dividend_curve=None,
        maturity=maturity,
    )

    euro_pv = OptionValuation(
        binom_underlying,
        euro_spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    amer_pv = OptionValuation(
        binom_underlying,
        amer_spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    assert amer_pv >= euro_pv - 1e-8


# ---------------------------------------------------------------------------
# Geometric Averaging Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spot,strike,vol,short_rate,days,option_type",
    [
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.CALL),
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.PUT),
        (110.0, 100.0, 0.25, 0.01, 270, OptionType.CALL),
        (95.0, 90.0, 0.3, 0.05, 540, OptionType.CALL),
        (105.0, 110.0, 0.18, 0.02, 180, OptionType.PUT),
    ],
)
def test_geometric_asian_mc_positive_value(spot, strike, vol, short_rate, days, option_type):
    """Geometric Asian options should produce positive option values."""
    maturity = PRICING_DATE + dt.timedelta(days=days)
    spec = _asian_spec(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        averaging=AsianAveraging.GEOMETRIC,
    )

    mc_underlying = _gbm_underlying(
        spot=spot,
        vol=vol,
        discount_curve=flat_curve(PRICING_DATE, maturity, short_rate),
        maturity=maturity,
        paths=MC_PATHS,
        num_steps=NUM_STEPS,
    )
    pv = OptionValuation(
        mc_underlying,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    assert pv > 0.0


@pytest.mark.parametrize(
    "spot,strike,vol,short_rate,days,option_type",
    [
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.CALL),
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.PUT),
        (110.0, 100.0, 0.25, 0.01, 270, OptionType.CALL),
    ],
)
def test_geometric_leq_arithmetic_asian(spot, strike, vol, short_rate, days, option_type):
    """Geometric average ≤ arithmetic average ⟹ geometric call ≤ arithmetic call
    (and geometric put ≥ arithmetic put for the same strike).

    For calls: geometric payoff ≤ arithmetic payoff (AM-GM inequality).
    For puts:  geometric payoff ≥ arithmetic payoff.

    In expectation (hence PV), the same ordering holds.
    """
    maturity = PRICING_DATE + dt.timedelta(days=days)

    arith_spec = _asian_spec(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        averaging=AsianAveraging.ARITHMETIC,
    )
    geom_spec = _asian_spec(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        averaging=AsianAveraging.GEOMETRIC,
    )

    mc_underlying = _gbm_underlying(
        spot=spot,
        vol=vol,
        discount_curve=flat_curve(PRICING_DATE, maturity, short_rate),
        maturity=maturity,
        paths=MC_PATHS,
        num_steps=NUM_STEPS,
    )

    arith_pv = OptionValuation(
        mc_underlying,
        arith_spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    geom_pv = OptionValuation(
        mc_underlying,
        geom_spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    if option_type is OptionType.CALL:
        # Geometric call ≤ arithmetic call (AM-GM)
        assert geom_pv <= arith_pv + 1e-8
    else:
        # Geometric put ≥ arithmetic put (AM-GM)
        assert geom_pv >= arith_pv - 1e-8


@pytest.mark.parametrize(
    "spot,strike,vol,short_rate,days,option_type",
    [
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.CALL),
        (110.0, 100.0, 0.25, 0.01, 270, OptionType.CALL),
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.PUT),
    ],
)
def test_geometric_asian_binomial_hull_close_to_mc(
    spot, strike, vol, short_rate, days, option_type
):
    """Binomial Hull tree with geometric averaging should match MC geometric."""
    maturity = PRICING_DATE + dt.timedelta(days=days)
    spec = _asian_spec(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        averaging=AsianAveraging.GEOMETRIC,
    )

    mc_underlying = _gbm_underlying(
        spot=spot,
        vol=vol,
        discount_curve=flat_curve(PRICING_DATE, maturity, short_rate),
        maturity=maturity,
        paths=MC_PATHS,
        num_steps=NUM_STEPS,
    )
    mc_pv = OptionValuation(
        mc_underlying,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    binom_underlying = _underlying(
        spot=spot,
        vol=vol,
        discount_curve=flat_curve(PRICING_DATE, maturity, short_rate),
        maturity=maturity,
    )
    hull_pv = OptionValuation(
        binom_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    logger.info(
        "Geometric Asian %s S=%.2f K=%.2f | MC=%.6f Hull=%.6f",
        option_type.value,
        spot,
        strike,
        mc_pv,
        hull_pv,
    )
    assert np.isclose(mc_pv, hull_pv, rtol=0.03)


# ---------------------------------------------------------------------------
# American Asian MC (Longstaff-Schwartz) Tests
# ---------------------------------------------------------------------------

# Each scenario: (spot, strike, vol, option_type, averaging, rate_spec, div_spec)
# rate_spec: float → flat rate; dict → DiscountCurve.from_forwards kwargs
# div_spec: None, ("continuous", yield), or ("discrete", [(days_offset, amount), ...])
_AMERICAN_ASIAN_SCENARIOS = [
    # Baseline: flat rate, no divs — original option_type × averaging combos
    pytest.param(
        100, 100, 0.20, OptionType.CALL, AsianAveraging.ARITHMETIC, 0.03, None, id="atm_call_arith"
    ),
    pytest.param(
        100, 100, 0.20, OptionType.PUT, AsianAveraging.ARITHMETIC, 0.03, None, id="atm_put_arith"
    ),
    pytest.param(
        100, 100, 0.20, OptionType.CALL, AsianAveraging.GEOMETRIC, 0.03, None, id="atm_call_geom"
    ),
    pytest.param(
        100, 100, 0.20, OptionType.PUT, AsianAveraging.GEOMETRIC, 0.03, None, id="atm_put_geom"
    ),
    # Continuous dividend yield
    pytest.param(
        100,
        100,
        0.20,
        OptionType.CALL,
        AsianAveraging.ARITHMETIC,
        0.03,
        ("continuous", 0.02),
        id="call_arith_cont_div",
    ),
    pytest.param(
        100,
        100,
        0.20,
        OptionType.PUT,
        AsianAveraging.GEOMETRIC,
        0.03,
        ("continuous", 0.03),
        id="put_geom_cont_div",
    ),
    # Discrete dividends
    pytest.param(
        100,
        100,
        0.20,
        OptionType.PUT,
        AsianAveraging.ARITHMETIC,
        0.03,
        ("discrete", [(91, 2.0), (274, 2.0)]),
        id="put_arith_disc_div",
    ),
    pytest.param(
        100,
        100,
        0.20,
        OptionType.CALL,
        AsianAveraging.GEOMETRIC,
        0.03,
        ("discrete", [(182, 3.0)]),
        id="call_geom_disc_div",
    ),
    # OTM, higher vol
    pytest.param(
        100,
        110,
        0.35,
        OptionType.CALL,
        AsianAveraging.ARITHMETIC,
        0.03,
        None,
        id="otm_call_high_vol",
    ),
    # ITM
    pytest.param(
        100, 90, 0.25, OptionType.PUT, AsianAveraging.GEOMETRIC, 0.03, None, id="itm_put_geom"
    ),
    # Non-flat rate curve
    pytest.param(
        100,
        100,
        0.20,
        OptionType.CALL,
        AsianAveraging.ARITHMETIC,
        {"times": np.array([0.0, 0.5, 1.0]), "forwards": np.array([0.02, 0.05])},
        None,
        id="call_arith_non_flat_rate",
    ),
    # Non-flat rate + continuous div
    pytest.param(
        100,
        100,
        0.20,
        OptionType.PUT,
        AsianAveraging.ARITHMETIC,
        {"times": np.array([0.0, 0.5, 1.0]), "forwards": np.array([0.03, 0.06])},
        ("continuous", 0.02),
        id="put_arith_non_flat_cont_div",
    ),
]


class TestAmericanAsianMC:
    """Tests for Longstaff-Schwartz American Asian option pricing."""

    SPOT = 100.0
    STRIKE = 100.0
    VOL = 0.2
    SHORT_RATE = 0.03
    DAYS = 365
    PATHS = 100_000
    SEED = 42
    NUM_STEPS = 60

    @property
    def maturity(self) -> dt.datetime:
        return PRICING_DATE + dt.timedelta(days=self.DAYS)

    # --- scenario helpers ---

    def _build_rate_curve(self, rate_spec) -> DiscountCurve:
        if isinstance(rate_spec, (int, float)):
            return flat_curve(PRICING_DATE, self.maturity, rate_spec)
        return DiscountCurve.from_forwards(**rate_spec)

    def _parse_div_spec(
        self, div_spec
    ) -> tuple[DiscountCurve | None, Sequence[tuple[dt.datetime, float]] | None]:
        if div_spec is None:
            return None, None
        kind, val = div_spec
        if kind == "continuous":
            return _flat_dividend_curve(val, self.maturity), None
        # "discrete": val is [(days_offset, amount), ...]
        divs = [(PRICING_DATE + dt.timedelta(days=d), amt) for d, amt in val]
        return None, divs

    def _mc_underlying(
        self,
        *,
        spot: float | None = None,
        vol: float | None = None,
        discount_curve: DiscountCurve | None = None,
        maturity: dt.datetime | None = None,
        paths: int | None = None,
        dividend_curve: DiscountCurve | None = None,
    ) -> GBMProcess:
        return _gbm_underlying(
            spot=spot or self.SPOT,
            vol=vol or self.VOL,
            discount_curve=discount_curve,
            maturity=maturity or self.maturity,
            paths=paths or self.PATHS,
            num_steps=self.NUM_STEPS,
            dividend_curve=dividend_curve,
        )

    def _price(
        self,
        option_type: OptionType,
        exercise_type: ExerciseType,
        averaging: AsianAveraging = AsianAveraging.ARITHMETIC,
        *,
        spot: float | None = None,
        strike: float | None = None,
        vol: float | None = None,
        discount_curve: DiscountCurve | None = None,
        maturity: dt.datetime | None = None,
        paths: int | None = None,
        seed: int | None = None,
        dividend_curve: DiscountCurve | None = None,
    ) -> float:
        mat = maturity or self.maturity
        spec = _asian_spec(
            strike=strike or self.STRIKE,
            maturity=mat,
            option_type=option_type,
            exercise_type=exercise_type,
            averaging=averaging,
        )
        underlying = self._mc_underlying(
            spot=spot,
            vol=vol,
            discount_curve=discount_curve,
            maturity=mat,
            paths=paths,
            dividend_curve=dividend_curve,
        )
        return OptionValuation(
            underlying,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=seed or self.SEED),
        ).present_value()

    # -- American >= European for puts (early exercise premium) --

    @pytest.mark.parametrize("option_type", [OptionType.PUT])
    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    def test_american_geq_european_put(self, option_type, averaging):
        """American Asian put >= European Asian put (early exercise premium)."""
        euro = self._price(option_type, ExerciseType.EUROPEAN, averaging)
        amer = self._price(option_type, ExerciseType.AMERICAN, averaging)
        assert amer >= euro - 1e-6, (
            f"American ({amer:.6f}) < European ({euro:.6f}) for {averaging.value} {option_type.value}"
        )

    # -- American >= European for calls with dividends --

    def test_american_geq_european_call_with_dividends(self):
        """American Asian call >= European when dividends present."""
        q_curve = flat_curve(PRICING_DATE, self.maturity, 0.04)
        euro = self._price(OptionType.CALL, ExerciseType.EUROPEAN, dividend_curve=q_curve)
        amer = self._price(OptionType.CALL, ExerciseType.AMERICAN, dividend_curve=q_curve)
        assert amer >= euro - 1e-6

    # -- Positive prices --

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    def test_positive_price(self, option_type, averaging):
        pv = self._price(option_type, ExerciseType.AMERICAN, averaging)
        assert pv > 0.0

    # -- Price increases with volatility --

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_price_increases_with_vol(self, option_type):
        low = self._price(option_type, ExerciseType.AMERICAN, vol=0.15)
        high = self._price(option_type, ExerciseType.AMERICAN, vol=0.35)
        assert high > low

    # -- MC American vs Binomial Hull American --

    @pytest.mark.parametrize(
        "spot,strike,vol,option_type,averaging,rate_spec,div_spec",
        _AMERICAN_ASIAN_SCENARIOS,
    )
    def test_mc_american_close_to_hull_american(
        self, spot, strike, vol, option_type, averaging, rate_spec, div_spec
    ):
        """MC LSM American Asian should be close to Hull binomial American Asian."""
        mat = self.maturity
        rate_curve = self._build_rate_curve(rate_spec)
        div_curve, disc_divs = self._parse_div_spec(div_spec)
        md = MarketData(PRICING_DATE, rate_curve, currency=CURRENCY)

        spec = _asian_spec(
            strike=strike,
            maturity=mat,
            option_type=option_type,
            exercise_type=ExerciseType.AMERICAN,
            averaging=averaging,
        )

        # MC American (GBMProcess)
        gbm_params = GBMParams(
            initial_value=spot,
            volatility=vol,
            dividend_curve=div_curve,
            discrete_dividends=disc_divs,
        )
        sim_config = SimulationConfig(
            paths=self.PATHS,
            num_steps=self.NUM_STEPS,
            end_date=mat,
        )
        mc_underlying = GBMProcess(md, gbm_params, sim_config)
        mc_pv = OptionValuation(
            mc_underlying,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=self.SEED),
        ).present_value()

        # Hull binomial American (UnderlyingData)
        # Escrowed-dividend trees diffuse S* = S - PV(divs) with flat vol σ,
        # but the true lognormal vol of S* is higher: σ_adj = σ · S / S*.
        binom_vol = vol
        if disc_divs:
            pv_divs = pv_discrete_dividends(
                disc_divs,
                curve_date=PRICING_DATE,
                end_date=mat,
                discount_curve=rate_curve,
            )
            binom_vol = vol * spot / (spot - pv_divs)

        binom_underlying = UnderlyingData(
            initial_value=spot,
            volatility=binom_vol,
            market_data=md,
            dividend_curve=div_curve,
            discrete_dividends=disc_divs,
        )
        hull_pv = OptionValuation(
            binom_underlying,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(
                num_steps=NUM_STEPS,
                asian_tree_averages=ASIAN_TREE_AVERAGES,
            ),
        ).present_value()

        logger.info(
            "American Asian %s %s K=%.0f vol=%.2f (binom_vol=%.4f) | MC=%.6f Hull=%.6f",
            averaging.value,
            option_type.value,
            strike,
            vol,
            binom_vol,
            mc_pv,
            hull_pv,
        )
        assert np.isclose(mc_pv, hull_pv, rtol=0.03), (
            f"MC ({mc_pv:.6f}) vs Hull ({hull_pv:.6f}) too far apart"
        )

    # -- private solver returns expected shapes --

    def test_impl_solve_returns_correct_shapes(self):
        """Access the private solver directly to verify MC Asian output shapes."""
        mat = self.maturity
        spec = _asian_spec(
            strike=self.STRIKE,
            maturity=mat,
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
        )
        n_paths = 1000
        underlying = _gbm_underlying(
            spot=self.SPOT,
            vol=self.VOL,
            discount_curve=flat_curve(PRICING_DATE, self.maturity, self.SHORT_RATE),
            maturity=mat,
            paths=n_paths,
            num_steps=self.NUM_STEPS,
        )
        ov = OptionValuation(
            underlying,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=self.SEED),
        )
        avg_paths, running_avg, intrinsic = ov._impl.solve()
        assert avg_paths.shape == running_avg.shape == intrinsic.shape
        assert avg_paths.shape[1] == n_paths
        # All intrinsic values should be non-negative
        assert np.all(intrinsic >= 0.0)

    # -- BSM American Asian raises --

    def test_bsm_american_asian_raises(self):
        """BSM does not support American exercise for Asian options."""
        spec = _asian_spec(
            strike=self.STRIKE,
            maturity=self.maturity,
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
        )
        binom_underlying = _underlying(
            spot=self.SPOT,
            vol=self.VOL,
            discount_curve=flat_curve(PRICING_DATE, self.maturity, self.SHORT_RATE),
            maturity=self.maturity,
        )
        with pytest.raises(ValidationError, match="AMERICAN.*BSM|BSM.*AMERICAN"):
            OptionValuation(
                binom_underlying,
                spec,
                PricingMethod.BSM,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Fixing-dates & economic-property tests
# ═══════════════════════════════════════════════════════════════════════════

_FD_SPOT = 100.0
_FD_STRIKE = 100.0
_FD_VOL = 0.20
_FD_RATE = 0.05
_FD_MATURITY = PRICING_DATE + dt.timedelta(days=365)
_FD_PATHS = 200_000
_FD_SEED = 42
_FD_STEPS = 60


def _fd_market_data(
    r_curve: DiscountCurve | None = None,
) -> MarketData:
    curve = r_curve if r_curve is not None else flat_curve(PRICING_DATE, _FD_MATURITY, _FD_RATE)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY)


def _fd_gbm(
    *,
    dividend_curve: DiscountCurve | None = None,
) -> GBMProcess:
    md = _fd_market_data()
    params = GBMParams(
        initial_value=_FD_SPOT,
        volatility=_FD_VOL,
        dividend_curve=dividend_curve,
    )
    sim_cfg = SimulationConfig(
        paths=_FD_PATHS,
        end_date=_FD_MATURITY,
        num_steps=_FD_STEPS,
    )
    return GBMProcess(md, params, sim_cfg)


# Monthly fixing dates used as baseline
_MONTHLY_FIXINGS = tuple(
    dt.datetime(2025, m, 1) if m <= 12 else dt.datetime(2026, m - 12, 1) for m in range(2, 14)
)


def _fd_asian_pv(
    option_type: OptionType,
    averaging: AsianAveraging = AsianAveraging.ARITHMETIC,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    fixing_dates: tuple[dt.datetime, ...] = _MONTHLY_FIXINGS,
    dividend_curve: DiscountCurve | None = None,
    seed: int = _FD_SEED,
) -> float:
    """Helper: MC Asian PV using explicit fixing dates."""
    spec = AsianSpec(
        averaging=averaging,
        option_type=option_type,
        strike=_FD_STRIKE,
        maturity=_FD_MATURITY,
        currency=CURRENCY,
        exercise_type=exercise_type,
        fixing_dates=fixing_dates,
    )
    gbm = _fd_gbm(dividend_curve=dividend_curve)
    return OptionValuation(
        gbm,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=seed),
    ).present_value()


class TestDividendEffectOnAsians:
    """Dividends should reduce call value and increase put value."""

    DIV_YIELD = 0.04

    @property
    def div_curve(self) -> DiscountCurve:
        return flat_curve(PRICING_DATE, _FD_MATURITY, self.DIV_YIELD)

    def test_dividends_reduce_asian_call(self):
        base = _fd_asian_pv(OptionType.CALL)
        with_div = _fd_asian_pv(OptionType.CALL, dividend_curve=self.div_curve)
        assert with_div < base, (
            f"Dividend call ({with_div:.4f}) should be < no-div call ({base:.4f})"
        )

    def test_dividends_increase_asian_put(self):
        base = _fd_asian_pv(OptionType.PUT)
        with_div = _fd_asian_pv(OptionType.PUT, dividend_curve=self.div_curve)
        assert with_div > base, f"Dividend put ({with_div:.4f}) should be > no-div put ({base:.4f})"

    def test_dividends_reduce_geometric_call(self):
        base = _fd_asian_pv(OptionType.CALL, averaging=AsianAveraging.GEOMETRIC)
        with_div = _fd_asian_pv(
            OptionType.CALL,
            averaging=AsianAveraging.GEOMETRIC,
            dividend_curve=self.div_curve,
        )
        assert with_div < base

    def test_dividends_increase_geometric_put(self):
        base = _fd_asian_pv(OptionType.PUT, averaging=AsianAveraging.GEOMETRIC)
        with_div = _fd_asian_pv(
            OptionType.PUT,
            averaging=AsianAveraging.GEOMETRIC,
            dividend_curve=self.div_curve,
        )
        assert with_div > base


class TestAmericanAsianPremium:
    """American Asians should have non-negative early-exercise premium."""

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    def test_american_geq_european_with_fixing_dates(self, option_type, averaging):
        """American Asian PV >= European Asian PV (using fixing dates)."""
        euro = _fd_asian_pv(option_type, averaging, ExerciseType.EUROPEAN)
        amer = _fd_asian_pv(option_type, averaging, ExerciseType.AMERICAN)
        assert amer >= euro - 1e-6, (
            f"American ({amer:.6f}) < European ({euro:.6f}) for {averaging.value} {option_type.value}"
        )


class TestFixingDatesValidation:
    """Validation rules for fixing_dates on AsianSpec."""

    def test_empty_fixing_dates_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            AsianSpec(
                averaging=AsianAveraging.ARITHMETIC,
                option_type=OptionType.CALL,
                strike=100.0,
                maturity=_FD_MATURITY,
                fixing_dates=(),
                exercise_type=ExerciseType.EUROPEAN,
            )

    def test_unsorted_fixing_dates_raises(self):
        with pytest.raises(ValidationError, match="ascending"):
            AsianSpec(
                averaging=AsianAveraging.ARITHMETIC,
                option_type=OptionType.CALL,
                strike=100.0,
                maturity=_FD_MATURITY,
                fixing_dates=(
                    dt.datetime(2025, 6, 1),
                    dt.datetime(2025, 3, 1),
                ),
                exercise_type=ExerciseType.EUROPEAN,
            )

    def test_fixing_dates_beyond_maturity_raises(self):
        with pytest.raises(ValidationError, match="maturity"):
            AsianSpec(
                averaging=AsianAveraging.ARITHMETIC,
                option_type=OptionType.CALL,
                strike=100.0,
                maturity=_FD_MATURITY,
                fixing_dates=(
                    dt.datetime(2025, 6, 1),
                    dt.datetime(2026, 6, 1),  # past maturity
                ),
                exercise_type=ExerciseType.EUROPEAN,
            )

    def test_fixing_dates_before_averaging_start_raises(self):
        with pytest.raises(ValidationError, match="averaging_start"):
            AsianSpec(
                averaging=AsianAveraging.ARITHMETIC,
                option_type=OptionType.CALL,
                strike=100.0,
                maturity=_FD_MATURITY,
                averaging_start=dt.datetime(2025, 4, 1),
                fixing_dates=(
                    dt.datetime(2025, 3, 1),  # before averaging_start
                    dt.datetime(2025, 6, 1),
                ),
                exercise_type=ExerciseType.EUROPEAN,
            )

    def test_valid_fixing_dates_accepted(self):
        """No error raised for well-formed fixing dates."""
        spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=100.0,
            maturity=_FD_MATURITY,
            fixing_dates=_MONTHLY_FIXINGS,
            exercise_type=ExerciseType.EUROPEAN,
        )
        assert spec.fixing_dates == _MONTHLY_FIXINGS


# ---------------------------------------------------------------------------
# N=1 should reproduce vanilla BSM exactly
# ---------------------------------------------------------------------------


class TestSmallObservationCounts:
    """With M=2 observations the average is over {S₀, S(T)}.

    Averaging reduces effective variance, so the Asian price is strictly
    less than the vanilla BSM price for both calls and puts.
    """

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,option_type",
        [
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.CALL),
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.PUT),
            (110, 90, 0.30, 0.03, 0.02, 180, OptionType.CALL),
            (90, 110, 0.25, 0.08, 0.01, 540, OptionType.PUT),
        ],
    )
    def test_m2_less_than_bsm(self, spot, strike, vol, r, q, days, option_type):
        maturity = PRICING_DATE + dt.timedelta(days=days)
        und = _underlying(
            spot=spot,
            vol=vol,
            discount_curve=flat_curve(PRICING_DATE, maturity, r),
            maturity=maturity,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )

        asian_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                option_type=option_type,
                num_observations=2,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        vanilla_spec = VanillaSpec(
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            strike=strike,
            maturity=maturity,
            currency=CURRENCY,
        )
        bsm_pv = OptionValuation(
            und,
            vanilla_spec,
            PricingMethod.BSM,
        ).present_value()

        assert asian_pv < bsm_pv, f"M=2 Asian={asian_pv:.8f} should be < BSM={bsm_pv:.8f}"
        assert asian_pv > 0.0


# ---------------------------------------------------------------------------
# Put-call parity for geometric Asians
# ---------------------------------------------------------------------------


class TestGeometricAsianPutCallParity:
    """For European geometric Asians: C - P = e^{-rT}(E[G] - K)."""

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,num_obs",
        [
            (100, 100, 0.20, 0.05, 0.00, 365, 12),
            (100, 100, 0.20, 0.05, 0.02, 365, 52),
            (110, 90, 0.30, 0.03, 0.00, 180, 6),
            (90, 110, 0.25, 0.08, 0.01, 540, 30),
        ],
    )
    def test_put_call_parity(self, spot, strike, vol, r, q, days, num_obs):
        maturity = PRICING_DATE + dt.timedelta(days=days)
        und = _underlying(
            spot=spot,
            vol=vol,
            discount_curve=flat_curve(PRICING_DATE, maturity, r),
            maturity=maturity,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )

        call_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=num_obs,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        put_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                option_type=OptionType.PUT,
                num_observations=num_obs,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # E[G] via the formula internals (M observations, N=M-1 intervals)
        T = days / 365.0
        M = num_obs
        N = M - 1
        delta = T / N
        t_bar = N * delta / 2.0
        M1 = np.log(spot) + (r - q - 0.5 * vol**2) * t_bar
        M2 = vol**2 * (delta * N * (2 * N + 1) / (6.0 * M))
        F_G = np.exp(M1 + 0.5 * M2)
        df = np.exp(-r * T)

        parity_rhs = df * (F_G - strike)
        assert np.isclose(call_pv - put_pv, parity_rhs, atol=1e-10), (
            f"C-P={call_pv - put_pv:.10f} vs df*(F_G-K)={parity_rhs:.10f}"
        )


# ---------------------------------------------------------------------------
# Analytical vs Monte Carlo convergence
# ---------------------------------------------------------------------------


class TestAnalyticalVsMC:
    """Analytical geometric Asian should match MC geometric within sampling noise."""

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,option_type",
        [
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.CALL),
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.PUT),
            (110, 100, 0.25, 0.03, 0.00, 270, OptionType.CALL),
            (100, 100, 0.20, 0.05, 0.02, 365, OptionType.CALL),
            (95, 90, 0.30, 0.05, 0.01, 540, OptionType.PUT),
        ],
    )
    def test_analytical_close_to_mc(self, spot, strike, vol, r, q, days, option_type):
        maturity = PRICING_DATE + dt.timedelta(days=days)
        num_observations = 61

        # Analytical
        und = _underlying(
            spot=spot,
            vol=vol,
            discount_curve=flat_curve(PRICING_DATE, maturity, r),
            maturity=maturity,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )
        analytical_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                option_type=option_type,
                num_observations=num_observations,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # MC geometric (same number of steps so observations match approximately)
        mc_und = _gbm_underlying(
            spot=spot,
            vol=vol,
            discount_curve=flat_curve(PRICING_DATE, maturity, r),
            maturity=maturity,
            paths=300_000,
            num_steps=num_observations - 1,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )
        mc_pv = OptionValuation(
            mc_und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                option_type=option_type,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        ).present_value()

        logger.info(
            "Geom Asian %s S=%.0f K=%.0f analytical=%.6f MC=%.6f",
            option_type.value,
            spot,
            strike,
            analytical_pv,
            mc_pv,
        )
        # Both analytical and MC include S₀ in the average (M observations).
        # Tolerance absorbs MC sampling noise.
        assert np.isclose(analytical_pv, mc_pv, rtol=0.02), (
            f"analytical={analytical_pv:.6f} MC={mc_pv:.6f}"
        )


# ---------------------------------------------------------------------------
# Monotonicity and ordering properties
# ---------------------------------------------------------------------------


class TestGeometricAsianProperties:
    """Sanity checks on the analytical price."""

    def test_positive_price(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(
            spot=100,
            vol=0.2,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )
        pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()
        assert pv > 0.0

    def test_call_increases_with_spot(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        pvs = []
        for spot in (90, 100, 110):
            und = _underlying(
                spot=spot,
                vol=0.2,
                discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
                maturity=maturity,
            )
            pv = OptionValuation(
                und,
                _asian_spec(
                    strike=100,
                    maturity=maturity,
                    option_type=OptionType.CALL,
                    num_observations=12,
                    averaging=AsianAveraging.GEOMETRIC,
                ),
                PricingMethod.BSM,
            ).present_value()
            pvs.append(pv)
        assert pvs[0] < pvs[1] < pvs[2]

    def test_put_decreases_with_spot(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        pvs = []
        for spot in (90, 100, 110):
            und = _underlying(
                spot=spot,
                vol=0.2,
                discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
                maturity=maturity,
            )
            pv = OptionValuation(
                und,
                _asian_spec(
                    strike=100,
                    maturity=maturity,
                    option_type=OptionType.PUT,
                    num_observations=12,
                    averaging=AsianAveraging.GEOMETRIC,
                ),
                PricingMethod.BSM,
            ).present_value()
            pvs.append(pv)
        assert pvs[0] > pvs[1] > pvs[2]

    def test_more_observations_increases_effective_variance(self):
        """With S₀ included, M₂ = σ²T·(2N+1)/(6(N+1)) is increasing in N.

        The known S₀ observation contributes zero variance; adding more
        future observations dilutes its weight, raising the effective vol
        of the geometric average toward σ/√3.  ATM call price therefore
        increases with the number of observations.
        """
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(
            spot=100,
            vol=0.3,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )
        pv_4 = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=4,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()
        pv_252 = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=252,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()
        assert pv_252 > pv_4

    def test_geometric_call_leq_vanilla_bsm(self):
        """Geometric average call ≤ vanilla European call (averaging reduces variance)."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(
            spot=100,
            vol=0.25,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )

        geom_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=52,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        vanilla_spec = VanillaSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100,
            maturity=maturity,
            currency=CURRENCY,
        )
        vanilla_pv = OptionValuation(
            und,
            vanilla_spec,
            PricingMethod.BSM,
        ).present_value()

        assert geom_pv < vanilla_pv

    def test_geometric_put_leq_vanilla_bsm(self):
        """Geometric average put ≤ vanilla European put (averaging reduces variance)."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(
            spot=100,
            vol=0.25,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )

        geom_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.PUT,
                num_observations=52,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        vanilla_spec = VanillaSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100,
            maturity=maturity,
            currency=CURRENCY,
        )
        vanilla_pv = OptionValuation(
            und,
            vanilla_spec,
            PricingMethod.BSM,
        ).present_value()

        assert geom_pv < vanilla_pv


# ---------------------------------------------------------------------------
# Continuous limit: N → ∞ gives σ_a = σ/√3
# ---------------------------------------------------------------------------


class TestContinuousLimit:
    """For large N the discrete formula should approach the continuous result."""

    def test_large_n_approaches_continuous(self):
        T, S0, K, sigma, r, q = 1.0, 100.0, 100.0, 0.25, 0.05, 0.02

        # Continuous closed form: use σ_a = σ/√3, adjusted growth rate
        t_bar_cont = T / 2.0
        M1_cont = np.log(S0) + (r - q - 0.5 * sigma**2) * t_bar_cont
        M2_cont = sigma**2 * T / 3.0
        F_G = np.exp(M1_cont + 0.5 * M2_cont)
        vol_sqrt = np.sqrt(M2_cont)
        d1 = (np.log(F_G / K) + 0.5 * M2_cont) / vol_sqrt
        d2 = d1 - vol_sqrt
        continuous_call = np.exp(-r * T) * (F_G * norm.cdf(d1) - K * norm.cdf(d2))

        # Discrete with N=10000
        N = 10_000
        t = np.linspace(0, T, N)
        F = S0 * np.exp((r - q) * t)
        discrete_call = _asian_geometric_analytical(
            strike=K,
            volatility=sigma,
            discount_factor_T=np.exp(-r * T),
            forward_prices=F,
            observation_times=t,
            option_type=OptionType.CALL,
        )

        assert np.isclose(discrete_call, continuous_call, rtol=1e-4)


# ---------------------------------------------------------------------------
# Dividend yield support
# ---------------------------------------------------------------------------


class TestDividendYield:
    """Continuous dividend yield should reduce call value, increase put value."""

    def test_dividend_reduces_call(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        pv_no_q = OptionValuation(
            _underlying(
                spot=100,
                vol=0.2,
                discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
                maturity=maturity,
            ),
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        pv_with_q = OptionValuation(
            _underlying(
                spot=100,
                vol=0.2,
                discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
                maturity=maturity,
                dividend_curve=_flat_dividend_curve(0.03, maturity),
            ),
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert pv_with_q < pv_no_q


# ---------------------------------------------------------------------------
# Seasoned Asian (observed_average / observed_count — Hull K* adjustment)
# ---------------------------------------------------------------------------


class TestSeasonedAsian:
    """Test Hull's K* strike-adjustment for seasoned Asian options.

    When n₁ fixings have already been observed with average S̄, the payoff is:

        max((n₁·S̄ + n₂·S_avg) / (n₁+n₂) − K, 0)

    This equals ``(n₂/(n₁+n₂)) · max(S_avg − K*, 0)`` with:

        K* = ((n₁+n₂)/n₂) · K  −  (n₁/n₂) · S̄

    See Hull, "Options, Futures, and Other Derivatives", Section 26.13.
    """

    SPOT = 50.0
    VOL = 0.40
    RATE = 0.10
    MATURITY = PRICING_DATE + dt.timedelta(days=365)

    def _ud(self) -> UnderlyingData:
        return _underlying(
            spot=self.SPOT,
            vol=self.VOL,
            discount_curve=flat_curve(PRICING_DATE, self.MATURITY, self.RATE),
            maturity=self.MATURITY,
        )

    # ── Validation ────────────────────────────────────────────────────────

    def test_observed_average_requires_observed_count(self):
        with pytest.raises(Exception, match="observed_average and observed_count"):
            AsianSpec(
                averaging=AsianAveraging.ARITHMETIC,
                option_type=OptionType.CALL,
                strike=50.0,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_observations=5,
                observed_average=52.0,
                exercise_type=ExerciseType.EUROPEAN,
            )

    def test_observed_count_requires_observed_average(self):
        with pytest.raises(Exception, match="observed_average and observed_count"):
            AsianSpec(
                averaging=AsianAveraging.ARITHMETIC,
                option_type=OptionType.CALL,
                strike=50.0,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_observations=5,
                observed_count=6,
                exercise_type=ExerciseType.EUROPEAN,
            )

    def test_observed_average_must_be_positive(self):
        with pytest.raises(Exception, match="observed_average must be > 0"):
            AsianSpec(
                averaging=AsianAveraging.ARITHMETIC,
                option_type=OptionType.CALL,
                strike=50.0,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_observations=5,
                observed_average=-1.0,
                observed_count=6,
                exercise_type=ExerciseType.EUROPEAN,
            )

    def test_observed_count_must_be_positive_int(self):
        with pytest.raises(Exception, match="observed_count must be a positive integer"):
            AsianSpec(
                averaging=AsianAveraging.ARITHMETIC,
                option_type=OptionType.CALL,
                strike=50.0,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_observations=5,
                observed_average=52.0,
                observed_count=0,
                exercise_type=ExerciseType.EUROPEAN,
            )

    # ── K* > 0: reduces to fresh Asian ───────────────────────────────────

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_seasoned_matches_manual_k_star(self, option_type):
        """Seasoned PV should equal scale * fresh_PV(K=K*) when K* > 0.

        Only arithmetic averaging: Hull's K* decomposition is invalid for
        geometric averaging (see test_geometric_seasoned_raises_on_bsm).
        """
        averaging = AsianAveraging.ARITHMETIC
        n1, S_bar, K = 6, 52.0, 50.0
        n2 = 6
        n_total = n1 + n2

        K_star = (n_total / n2) * K - (n1 / n2) * S_bar
        assert K_star > 0, "This test requires K* > 0"
        scale = n2 / n_total

        # Manual: price fresh Asian with K*
        fresh_spec = AsianSpec(
            averaging=averaging,
            option_type=option_type,
            strike=K_star,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=n2,
            exercise_type=ExerciseType.EUROPEAN,
        )
        fresh_pv = OptionValuation(
            self._ud(),
            fresh_spec,
            PricingMethod.BSM,
        ).present_value()

        # Library seasoned
        seasoned_spec = AsianSpec(
            averaging=averaging,
            option_type=option_type,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=n2,
            observed_average=S_bar,
            observed_count=n1,
            exercise_type=ExerciseType.EUROPEAN,
        )
        seasoned_pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        assert np.isclose(seasoned_pv, scale * fresh_pv, rtol=1e-12)

    def test_geometric_seasoned_raises_on_bsm(self):
        """Geometric seasoned on BSM raises (K* decomposition is arithmetic-only)."""
        seasoned_spec = AsianSpec(
            averaging=AsianAveraging.GEOMETRIC,
            option_type=OptionType.CALL,
            strike=50.0,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=5,
            observed_average=52.0,
            observed_count=6,
            exercise_type=ExerciseType.EUROPEAN,
        )
        with pytest.raises(UnsupportedFeatureError, match="arithmetic averaging"):
            OptionValuation(
                self._ud(),
                seasoned_spec,
                PricingMethod.BSM,
            ).present_value()

    # ── K* <= 0: certain exercise (call → forward, put → 0) ─────────────

    def test_k_star_negative_call_is_forward(self):
        """When K* <= 0, the call is certain to be exercised."""
        n1, S_bar, K = 6, 120.0, 50.0
        n2 = 6
        n_total = n1 + n2
        K_star = (n_total / n2) * K - (n1 / n2) * S_bar
        assert K_star < 0, "This test requires K* < 0"

        seasoned_spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=n2,
            observed_average=S_bar,
            observed_count=n1,
            exercise_type=ExerciseType.EUROPEAN,
        )
        pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        # Must be strictly positive
        assert pv > 0.0

        # Manual: scale * (disc_M1 - K* * df)
        ttm = 1.0
        df = np.exp(-self.RATE * ttm)
        zero_spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=0.0,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=n2,
            exercise_type=ExerciseType.EUROPEAN,
        )
        disc_M1 = OptionValuation(
            self._ud(),
            zero_spec,
            PricingMethod.BSM,
        ).present_value()
        scale = n2 / n_total
        expected = scale * (disc_M1 - K_star * df)
        assert np.isclose(pv, expected, rtol=1e-12)

    def test_k_star_negative_put_is_zero(self):
        """When K* <= 0, a put is worthless (average > 0 > K*)."""
        seasoned_spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.PUT,
            strike=50.0,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=5,
            observed_average=120.0,
            observed_count=6,
            exercise_type=ExerciseType.EUROPEAN,
        )
        pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        assert pv == 0.0

    # ── Monotonicity and sanity ──────────────────────────────────────────

    def test_seasoned_less_than_fresh_when_atm(self):
        """A seasoned ATM call (S̄ = S₀ = K) should be worth less than a fresh one.

        With S̄ = K, some of the averaging period has elapsed at-the-money,
        leaving the remaining average with less optionality.
        """
        K = self.SPOT  # ATM
        n2 = 6
        n1 = 6

        fresh_spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=n2,
            exercise_type=ExerciseType.EUROPEAN,
        )
        fresh_pv = OptionValuation(
            self._ud(),
            fresh_spec,
            PricingMethod.BSM,
        ).present_value()

        seasoned_spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=n2,
            observed_average=K,  # ATM: S̄ = K
            observed_count=n1,
            exercise_type=ExerciseType.EUROPEAN,
        )
        seasoned_pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        assert seasoned_pv < fresh_pv

    def test_higher_observed_average_increases_call_value(self):
        """A higher observed average should increase the seasoned call value."""
        n2 = 6
        n1 = 6

        def _seasoned_call(s_bar: float) -> float:
            spec = AsianSpec(
                averaging=AsianAveraging.ARITHMETIC,
                option_type=OptionType.CALL,
                strike=self.SPOT,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_observations=n2,
                observed_average=s_bar,
                observed_count=n1,
                exercise_type=ExerciseType.EUROPEAN,
            )
            return OptionValuation(
                self._ud(),
                spec,
                PricingMethod.BSM,
            ).present_value()

        assert _seasoned_call(55.0) > _seasoned_call(50.0) > _seasoned_call(45.0)

    def test_higher_observed_average_decreases_put_value(self):
        """A higher observed average should decrease the seasoned put value."""
        n2 = 6
        n1 = 6

        def _seasoned_put(s_bar: float) -> float:
            spec = AsianSpec(
                averaging=AsianAveraging.ARITHMETIC,
                option_type=OptionType.PUT,
                strike=self.SPOT,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_observations=n2,
                observed_average=s_bar,
                observed_count=n1,
                exercise_type=ExerciseType.EUROPEAN,
            )
            return OptionValuation(
                self._ud(),
                spec,
                PricingMethod.BSM,
            ).present_value()

        assert _seasoned_put(45.0) > _seasoned_put(50.0) > _seasoned_put(55.0)

    def test_more_observed_reduces_optionality(self):
        """More past observations (with S̄ = K) → less optionality → lower call value."""
        n2 = 6
        K = self.SPOT

        def _seasoned_call(n1: int) -> float:
            spec = AsianSpec(
                averaging=AsianAveraging.ARITHMETIC,
                option_type=OptionType.CALL,
                strike=K,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_observations=n2,
                observed_average=K,
                observed_count=n1,
                exercise_type=ExerciseType.EUROPEAN,
            )
            return OptionValuation(
                self._ud(),
                spec,
                PricingMethod.BSM,
            ).present_value()

        # More past ATM observations → the average is "pinned" closer to K
        assert _seasoned_call(1) > _seasoned_call(6) > _seasoned_call(50)

    # ── Cross-engine consistency (Binomial / MC) ─────────────────────────

    def test_binomial_seasoned_matches_analytical(self):
        """Binomial Hull tree seasoned Asian should converge to analytical."""
        n1, S_bar, K = 6, 52.0, 50.0
        n2 = 61

        seasoned_spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=n2,
            observed_average=S_bar,
            observed_count=n1,
            exercise_type=ExerciseType.EUROPEAN,
        )

        bsm_pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        binom_pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=n2 - 1, asian_tree_averages=100),
        ).present_value()

        assert np.isclose(binom_pv, bsm_pv, rtol=0.01), (
            f"Binomial={binom_pv:.6f} vs BSM={bsm_pv:.6f}"
        )

    def test_mc_seasoned_matches_analytical(self):
        """MC seasoned Asian should converge to analytical."""
        n1, S_bar, K = 6, 52.0, 50.0
        n2 = 61

        seasoned_spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=n2,
            observed_average=S_bar,
            observed_count=n1,
            exercise_type=ExerciseType.EUROPEAN,
        )

        bsm_pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        gbm = _gbm_underlying(
            spot=self.SPOT,
            vol=self.VOL,
            discount_curve=flat_curve(PRICING_DATE, self.MATURITY, self.RATE),
            maturity=self.MATURITY,
            paths=200_000,
            num_steps=n2 - 1,
        )
        mc_pv = OptionValuation(
            gbm,
            seasoned_spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        ).present_value()

        assert np.isclose(mc_pv, bsm_pv, rtol=0.01), f"MC={mc_pv:.6f} vs BSM={bsm_pv:.6f}"

    def test_dividend_increases_put(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        pv_no_q = OptionValuation(
            _underlying(
                spot=100,
                vol=0.2,
                discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
                maturity=maturity,
            ),
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.PUT,
                num_observations=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        pv_with_q = OptionValuation(
            _underlying(
                spot=100,
                vol=0.2,
                discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
                maturity=maturity,
                dividend_curve=_flat_dividend_curve(0.03, maturity),
            ),
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.PUT,
                num_observations=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert pv_with_q > pv_no_q


class TestSeasonedBinomialVsMC:
    """Seasoned Asian: Hull binomial tree (running-average folding) vs MC.

    The Hull tree folds n₁ past observations directly into node running
    averages, so American exercise decisions use the correct full-period
    average at each node — no K* decomposition.

    Arithmetic tests also cross-check against BSM (K*) for European
    exercise, where the decomposition IS exact.
    """

    SPOT = 100.0
    VOL = 0.20
    RATE = 0.05
    MATURITY = PRICING_DATE + dt.timedelta(days=365)
    N1 = 6
    S_BAR = 102.0
    STRIKE = 100.0
    NUM_STEPS = 60
    MC_PATHS = 200_000
    SEED = 42
    TREE_AVERAGES = 100

    def _ud(self, *, vol: float | None = None) -> UnderlyingData:
        return _underlying(
            spot=self.SPOT,
            vol=vol or self.VOL,
            discount_curve=flat_curve(PRICING_DATE, self.MATURITY, self.RATE),
            maturity=self.MATURITY,
        )

    def _gbm(self, *, vol: float | None = None) -> GBMProcess:
        return _gbm_underlying(
            spot=self.SPOT,
            vol=vol or self.VOL,
            discount_curve=flat_curve(PRICING_DATE, self.MATURITY, self.RATE),
            maturity=self.MATURITY,
            paths=self.MC_PATHS,
            num_steps=self.NUM_STEPS,
        )

    def _seasoned_spec(
        self,
        option_type: OptionType,
        averaging: AsianAveraging,
        exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    ) -> AsianSpec:
        return AsianSpec(
            averaging=averaging,
            option_type=option_type,
            strike=self.STRIKE,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=self.NUM_STEPS + 1,
            exercise_type=exercise_type,
            observed_average=self.S_BAR,
            observed_count=self.N1,
        )

    # ── European: binomial vs MC ─────────────────────────────────────────

    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_european_binomial_vs_mc(self, averaging, option_type):
        """Seasoned European binomial should match MC within noise."""
        spec = self._seasoned_spec(option_type, averaging)

        binom_pv = OptionValuation(
            self._ud(),
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(
                num_steps=self.NUM_STEPS,
                asian_tree_averages=self.TREE_AVERAGES,
            ),
        ).present_value()

        mc_pv = OptionValuation(
            self._gbm(),
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=self.SEED),
        ).present_value()

        logger.info(
            "Seasoned European %s %s | Binom=%.6f MC=%.6f",
            averaging.value,
            option_type.value,
            binom_pv,
            mc_pv,
        )
        assert np.isclose(binom_pv, mc_pv, rtol=0.02), f"Binom={binom_pv:.6f} vs MC={mc_pv:.6f}"

    # ── European arithmetic: binomial vs BSM (K* is exact for European) ──

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_european_arithmetic_binomial_vs_bsm(self, option_type):
        """Seasoned arithmetic European: binomial ≈ BSM K* (both exact for European)."""
        spec = self._seasoned_spec(option_type, AsianAveraging.ARITHMETIC)

        bsm_pv = OptionValuation(
            self._ud(),
            spec,
            PricingMethod.BSM,
        ).present_value()

        binom_pv = OptionValuation(
            self._ud(),
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(
                num_steps=self.NUM_STEPS,
                asian_tree_averages=self.TREE_AVERAGES,
            ),
        ).present_value()

        logger.info(
            "Seasoned European Arithmetic %s | BSM=%.6f Binom=%.6f",
            option_type.value,
            bsm_pv,
            binom_pv,
        )
        assert np.isclose(binom_pv, bsm_pv, rtol=0.01), f"BSM={bsm_pv:.6f} vs Binom={binom_pv:.6f}"

    # ── American: binomial vs MC ─────────────────────────────────────────

    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_american_binomial_vs_mc(self, averaging, option_type):
        """Seasoned American binomial should match MC LSM within noise."""
        spec = self._seasoned_spec(option_type, averaging, ExerciseType.AMERICAN)

        binom_pv = OptionValuation(
            self._ud(),
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(
                num_steps=self.NUM_STEPS,
                asian_tree_averages=self.TREE_AVERAGES,
            ),
        ).present_value()

        mc_pv = OptionValuation(
            self._gbm(),
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=self.SEED),
        ).present_value()

        logger.info(
            "Seasoned American %s %s | Binom=%.6f MC=%.6f",
            averaging.value,
            option_type.value,
            binom_pv,
            mc_pv,
        )
        assert np.isclose(binom_pv, mc_pv, rtol=0.03), f"Binom={binom_pv:.6f} vs MC={mc_pv:.6f}"

    # ── American >= European (early exercise premium) ────────────────────

    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    def test_american_geq_european_seasoned_put(self, averaging):
        """American seasoned Asian put >= European (early exercise premium)."""
        euro_spec = self._seasoned_spec(OptionType.PUT, averaging, ExerciseType.EUROPEAN)
        amer_spec = self._seasoned_spec(OptionType.PUT, averaging, ExerciseType.AMERICAN)

        euro_pv = OptionValuation(
            self._ud(),
            euro_spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(
                num_steps=self.NUM_STEPS,
                asian_tree_averages=self.TREE_AVERAGES,
            ),
        ).present_value()

        amer_pv = OptionValuation(
            self._ud(),
            amer_spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(
                num_steps=self.NUM_STEPS,
                asian_tree_averages=self.TREE_AVERAGES,
            ),
        ).present_value()

        assert amer_pv >= euro_pv - 1e-6, f"American ({amer_pv:.6f}) < European ({euro_pv:.6f})"


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------


class TestValidation:
    """Error handling for analytical Asian pricing."""

    def test_arithmetic_bsm_returns_positive_price(self):
        """Arithmetic averaging is now supported via Turnbull-Wakeman."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(
            spot=100,
            vol=0.2,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )
        pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=12,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()
        assert pv > 0.0

    def test_missing_num_observations_raises(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(
            spot=100,
            vol=0.2,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )
        with pytest.raises(Exception, match="num_observations"):
            OptionValuation(
                und,
                _asian_spec(
                    strike=100,
                    maturity=maturity,
                    option_type=OptionType.CALL,
                    num_observations=None,
                    averaging=AsianAveraging.GEOMETRIC,
                ),
                PricingMethod.BSM,
            )

    def test_invalid_num_observations_on_spec(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        with pytest.raises(Exception, match="num_observations"):
            AsianSpec(
                averaging=AsianAveraging.GEOMETRIC,
                option_type=OptionType.CALL,
                strike=100,
                maturity=maturity,
                currency=CURRENCY,
                num_observations=1,
                exercise_type=ExerciseType.EUROPEAN,
            )

    def test_pure_function_validation(self):
        # observation_times must have >= 2 entries
        with pytest.raises(Exception, match="observation_times"):
            _asian_geometric_analytical(
                strike=100,
                volatility=0.2,
                discount_factor_T=np.exp(-0.05),
                forward_prices=np.array([100.0]),
                observation_times=np.array([0.5]),
                option_type=OptionType.CALL,
            )
        with pytest.raises(Exception, match="volatility"):
            _asian_geometric_analytical(
                strike=100,
                volatility=-0.2,
                discount_factor_T=np.exp(-0.05),
                forward_prices=np.array([100.0, 102.0]),
                observation_times=np.array([0.5, 1.0]),
                option_type=OptionType.CALL,
            )
        # observation_times must have >= 2 entries (single element)
        with pytest.raises(Exception, match="observation_times"):
            _asian_geometric_analytical(
                strike=100,
                volatility=0.2,
                discount_factor_T=np.exp(-0.05),
                forward_prices=np.array([100.0]),
                observation_times=np.array([1.0]),
                option_type=OptionType.CALL,
            )


# ---------------------------------------------------------------------------
# Averaging start in the future
# ---------------------------------------------------------------------------


class TestAveragingStart:
    """When averaging_start is after pricing_date, observations span [t_s, T]."""

    def test_later_start_changes_price(self):
        """Pushing the averaging window forward should change the price."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        avg_start = PRICING_DATE + dt.timedelta(days=90)
        und = _underlying(
            spot=100,
            vol=0.2,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )

        pv_full = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        pv_late = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=12,
                averaging_start=avg_start,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # Shorter averaging window → less variance reduction → more expensive call
        assert pv_late > pv_full

    def test_averaging_start_put_call_parity(self):
        """Put-call parity holds even with a deferred averaging start."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        avg_start = PRICING_DATE + dt.timedelta(days=60)
        und = _underlying(
            spot=100,
            vol=0.2,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )

        call_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=10,
                averaging_start=avg_start,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        put_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.PUT,
                num_observations=10,
                averaging_start=avg_start,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # Compute E[G] (M observations, N=M-1 intervals)
        T = 365.0 / 365.0
        M = 10
        N = M - 1
        r, q, sigma = 0.05, 0.0, 0.2
        t_s = 60.0 / 365.0
        delta = (T - t_s) / N
        t_bar = t_s + N * delta / 2.0
        M1 = np.log(100) + (r - q - 0.5 * sigma**2) * t_bar
        M2 = sigma**2 * (t_s + delta * N * (2 * N + 1) / (6.0 * M))
        F_G = np.exp(M1 + 0.5 * M2)
        df = np.exp(-r * T)

        assert np.isclose(call_pv - put_pv, df * (F_G - 100), atol=1e-10)


# ---------------------------------------------------------------------------
# 4-method comparison: Analytical vs Binomial-MC vs Binomial-Hull vs MC
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "spot,strike,vol,r,q,days,option_type",
    [
        (100, 100, 0.20, 0.05, 0.00, 365, OptionType.CALL),
        (100, 100, 0.20, 0.05, 0.00, 365, OptionType.PUT),
        (50, 50, 0.40, 0.10, 0.00, 365, OptionType.CALL),  # Hull Example 26.3
        (110, 100, 0.25, 0.03, 0.02, 270, OptionType.CALL),
        (95, 90, 0.30, 0.05, 0.01, 540, OptionType.PUT),
        (105, 110, 0.18, 0.02, 0.00, 180, OptionType.PUT),
    ],
)
def test_geometric_asian_four_method_comparison(
    spot,
    strike,
    vol,
    r,
    q,
    days,
    option_type,
):
    """Compare geometric Asian European prices across 4 methods.

    1. Analytical (BSM) — Kemna-Vorst closed-form
    2. Binomial MC — binomial tree with MC path sampling for Asian payoff
    3. Binomial Hull — Hull's tree-average interpolation method
    4. Monte Carlo — direct GBM path simulation

    This is a diagnostic/comparison test; it logs all four prices
    and asserts they are broadly consistent (within MC noise).
    """
    maturity = PRICING_DATE + dt.timedelta(days=days)

    # --- 1. Analytical (BSM) ---
    und_determ = _underlying(
        spot=spot,
        vol=vol,
        discount_curve=flat_curve(PRICING_DATE, maturity, r),
        maturity=maturity,
        dividend_curve=_flat_dividend_curve(q, maturity),
    )
    analytical_pv = OptionValuation(
        und_determ,
        _asian_spec(
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            num_observations=NUM_OBSERVATIONS,
            averaging=AsianAveraging.GEOMETRIC,
        ),
        PricingMethod.BSM,
    ).present_value()

    # --- 2. Binomial MC ---
    binom_mc_pv = OptionValuation(
        und_determ,
        _asian_spec(
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            averaging=AsianAveraging.GEOMETRIC,
        ),
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS * 2,
            mc_paths=MC_PATHS,
            random_seed=MC_SEED,
        ),
    ).present_value()

    # --- 3. Binomial Hull tree averages ---
    hull_pv = OptionValuation(
        und_determ,
        _asian_spec(
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            averaging=AsianAveraging.GEOMETRIC,
        ),
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=BINOM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    # --- 4. Monte Carlo ---
    mc_und = _gbm_underlying(
        spot=spot,
        vol=vol,
        discount_curve=flat_curve(PRICING_DATE, maturity, r),
        maturity=maturity,
        paths=MC_PATHS,
        num_steps=NUM_STEPS,
        dividend_curve=_flat_dividend_curve(q, maturity),
    )
    mc_pv = OptionValuation(
        mc_und,
        _asian_spec(
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            averaging=AsianAveraging.GEOMETRIC,
        ),
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    logger.info(
        "Geometric Asian %s S=%.0f K=%.0f vol=%.2f r=%.2f q=%.2f days=%d\n"
        "  Analytical=%.6f  BinomMC=%.6f  Hull=%.6f  MC=%.6f",
        option_type.value,
        spot,
        strike,
        vol,
        r,
        q,
        days,
        analytical_pv,
        binom_mc_pv,
        hull_pv,
        mc_pv,
    )

    # Broad consistency: all four should be within 5% of each other
    prices = [analytical_pv, binom_mc_pv, hull_pv, mc_pv]
    mid = np.mean(prices)
    for label, pv in zip(["Analytical", "BinomMC", "Hull", "MC"], prices):
        assert np.isclose(pv, mid, rtol=0.02), f"{label}={pv:.6f} deviates from mean={mid:.6f}"


# ===========================================================================
# ARITHMETIC AVERAGE — Turnbull-Wakeman / Hull §26.13
# ===========================================================================


# ---------------------------------------------------------------------------
# Hull Example 26.3 verification
# ---------------------------------------------------------------------------


class TestHullExample26_3:
    """Verify against Hull Example 26.3 (pp. 626-627).

    Hull's example uses observations at T/m, 2T/m, ..., T (i.e. does NOT
    include S₀ at time 0).  Our convention includes S₀, so using
    ``num_observations = m + 1`` gives that extra t0 observation. We test both
    the raw moments against Hull's
    numbers (using his convention) and the pipeline with our S₀ convention.
    """

    def test_hull_continuous_moments(self):
        """Verify M₁, M₂ formulas against Hull's continuous-average closed forms.

        Hull continuous formulas (r=0.1, q=0, σ=0.4, T=1, S₀=50):
            M₁ = [exp((r-q)T) - 1] / [(r-q)T] · S₀ = 52.59
            M₂ = 2922.76 (from equation 26.4)
        """
        S0, r, q, sigma, T = 50.0, 0.1, 0.0, 0.4, 1.0

        # Hull's continuous M₁
        hull_M1 = (np.exp((r - q) * T) - 1) / ((r - q) * T) * S0

        # Hull's continuous M₂ (equation 26.4)
        hull_M2 = 2 * np.exp((2 * (r - q) + sigma**2) * T) * S0**2 / (
            (r - q + sigma**2) * (2 * (r - q) + sigma**2) * T**2
        ) + 2 * S0**2 / ((r - q) * T**2) * (
            1 / (2 * (r - q) + sigma**2) - np.exp((r - q) * T) / (r - q + sigma**2)
        )

        assert np.isclose(hull_M1, 52.59, atol=0.01), f"M1={hull_M1}"
        assert np.isclose(hull_M2, 2922.76, rtol=5e-4), f"M2={hull_M2}"

        # Verify our discrete formula approaches the continuous limit
        N_large = 10_000
        M = N_large + 1
        delta = T / N_large
        t = np.arange(M, dtype=float) * delta
        F = S0 * np.exp((r - q) * t)
        our_M1 = np.mean(F)
        F_cumrev = np.cumsum(F[::-1])[::-1]
        our_M2 = np.sum(F * np.exp(sigma**2 * t) * (2 * F_cumrev - F)) / M**2

        assert np.isclose(our_M1, hull_M1, rtol=1e-4), f"our M1={our_M1} vs hull {hull_M1}"
        assert np.isclose(our_M2, hull_M2, rtol=1e-3), f"our M2={our_M2} vs hull {hull_M2}"

    def test_hull_continuous_price(self):
        """Hull continuous average price: 5.62 for a call."""
        S0, K, r, q, sigma, T = 50.0, 50.0, 0.1, 0.0, 0.4, 1.0

        # Large N approximates continuous average
        N = 10_000
        t = np.linspace(0, T, N)
        F = S0 * np.exp((r - q) * t)
        price = _asian_arithmetic_analytical(
            strike=K,
            time_to_maturity=T,
            volatility=sigma,
            discount_factor_T=np.exp(-r * T),
            forward_prices=F,
            observation_times=t,
            option_type=OptionType.CALL,
        )
        assert np.isclose(price, 5.62, atol=0.02), f"price={price:.4f} expected ~5.62"

    def test_hull_discrete_observations(self):
        """Hull's discrete prices: 12 obs → 6.00, 52 obs → 5.70, 250 obs → 5.63.

        Hull uses m observations at T/m, 2T/m, ..., T (no S₀).
        Our pricer with ``averaging_start = T/m`` and ``num_observations = m``
        places M = m observations at exactly those times, matching Hull.
        """
        S0, K, r, q, sigma, T = 50.0, 50.0, 0.1, 0.0, 0.4, 1.0

        for m, hull_price in [(12, 6.00), (52, 5.70), (250, 5.63)]:
            # averaging_start = T/m skips S₀, giving m obs at T/m, ..., T
            t_s = T / m
            t = t_s + np.arange(m, dtype=float) * (T - t_s) / (m - 1)
            F = S0 * np.exp((r - q) * t)
            price = _asian_arithmetic_analytical(
                strike=K,
                time_to_maturity=T,
                volatility=sigma,
                discount_factor_T=np.exp(-r * T),
                forward_prices=F,
                observation_times=t,
                option_type=OptionType.CALL,
            )

            logger.info(
                "Hull 26.3: m=%d price=%.4f (expected %.2f)",
                m,
                price,
                hull_price,
            )
            assert np.isclose(price, hull_price, atol=0.02), (
                f"m={m}: price={price:.4f} expected {hull_price:.2f}"
            )


# ---------------------------------------------------------------------------
# Arithmetic: put-call parity
# ---------------------------------------------------------------------------


class TestArithmeticPutCallParity:
    """For European arithmetic Asians: C - P = e^{-rT}(M₁ - K)."""

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,num_observations",
        [
            (100, 100, 0.20, 0.05, 0.00, 365, 12),
            (100, 100, 0.25, 0.05, 0.02, 365, 52),
            (50, 50, 0.40, 0.10, 0.00, 365, 30),
            (110, 90, 0.30, 0.03, 0.00, 180, 6),
        ],
    )
    def test_put_call_parity(self, spot, strike, vol, r, q, days, num_observations):
        maturity = PRICING_DATE + dt.timedelta(days=days)
        und = _underlying(
            spot=spot,
            vol=vol,
            discount_curve=flat_curve(PRICING_DATE, maturity, r),
            maturity=maturity,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )

        call_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=num_observations,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        put_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                option_type=OptionType.PUT,
                num_observations=num_observations,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # E[S_avg] = M₁ via forward prices
        T = days / 365.0
        M = num_observations
        N = M - 1
        delta = T / N
        t = np.arange(M, dtype=float) * delta
        F = spot * np.exp((r - q) * t)
        M1 = np.mean(F)
        df = np.exp(-r * T)

        parity_rhs = df * (M1 - strike)
        assert np.isclose(call_pv - put_pv, parity_rhs, atol=1e-10), (
            f"C-P={call_pv - put_pv:.10f} vs df*(M1-K)={parity_rhs:.10f}"
        )


# ---------------------------------------------------------------------------
# Arithmetic vs MC convergence
# ---------------------------------------------------------------------------


class TestArithmeticVsMC:
    """Turnbull-Wakeman approximation should be close to MC arithmetic average."""

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,option_type",
        [
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.CALL),
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.PUT),
            (50, 50, 0.40, 0.10, 0.00, 365, OptionType.CALL),
            (110, 100, 0.25, 0.03, 0.02, 270, OptionType.CALL),
            (95, 90, 0.30, 0.05, 0.01, 540, OptionType.PUT),
        ],
    )
    def test_analytical_close_to_mc(self, spot, strike, vol, r, q, days, option_type):
        maturity = PRICING_DATE + dt.timedelta(days=days)
        num_observations = 61

        # Turnbull-Wakeman analytical
        und = _underlying(
            spot=spot,
            vol=vol,
            discount_curve=flat_curve(PRICING_DATE, maturity, r),
            maturity=maturity,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )
        analytical_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                option_type=option_type,
                num_observations=num_observations,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # MC arithmetic
        mc_und = _gbm_underlying(
            spot=spot,
            vol=vol,
            discount_curve=flat_curve(PRICING_DATE, maturity, r),
            maturity=maturity,
            paths=300_000,
            num_steps=num_observations - 1,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )
        mc_pv = OptionValuation(
            mc_und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                option_type=option_type,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        ).present_value()

        logger.info(
            "Arith Asian %s S=%.0f K=%.0f analytical=%.6f MC=%.6f",
            option_type.value,
            spot,
            strike,
            analytical_pv,
            mc_pv,
        )
        # Turnbull-Wakeman is an approximation; allow slightly wider tolerance
        assert np.isclose(analytical_pv, mc_pv, rtol=0.03), (
            f"analytical={analytical_pv:.6f} MC={mc_pv:.6f}"
        )


# ---------------------------------------------------------------------------
# Arithmetic properties
# ---------------------------------------------------------------------------


class TestArithmeticProperties:
    """Sanity checks on arithmetic analytical prices."""

    def test_arithmetic_geq_geometric_call(self):
        """Arithmetic call ≥ geometric call (AM-GM: E[arith avg] ≥ E[geom avg])."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(
            spot=100,
            vol=0.25,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )

        arith_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=52,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        geom_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=52,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert arith_pv >= geom_pv - 1e-10, f"arith call={arith_pv:.6f} < geom call={geom_pv:.6f}"

    def test_arithmetic_leq_geometric_put(self):
        """Arithmetic put ≤ geometric put (AM-GM: higher avg lowers put value)."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(
            spot=100,
            vol=0.25,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )

        arith_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.PUT,
                num_observations=52,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        geom_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.PUT,
                num_observations=52,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert arith_pv <= geom_pv + 1e-10, f"arith put={arith_pv:.6f} > geom put={geom_pv:.6f}"

    def test_arithmetic_less_than_vanilla(self):
        """Arithmetic Asian call < vanilla European call (averaging effect)."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(
            spot=100,
            vol=0.25,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )

        arith_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=52,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        vanilla_pv = OptionValuation(
            und,
            VanillaSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100,
                maturity=maturity,
                currency=CURRENCY,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert arith_pv < vanilla_pv

    def test_positive_price(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(
            spot=100,
            vol=0.2,
            discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
            maturity=maturity,
        )
        for option_type in (OptionType.CALL, OptionType.PUT):
            pv = OptionValuation(
                und,
                _asian_spec(
                    strike=100,
                    maturity=maturity,
                    option_type=option_type,
                    num_observations=12,
                    averaging=AsianAveraging.ARITHMETIC,
                ),
                PricingMethod.BSM,
            ).present_value()
            assert pv > 0.0

    def test_dividend_reduces_call(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        pv_no_q = OptionValuation(
            _underlying(
                spot=100,
                vol=0.2,
                discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
                maturity=maturity,
            ),
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=12,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        pv_with_q = OptionValuation(
            _underlying(
                spot=100,
                vol=0.2,
                discount_curve=flat_curve(PRICING_DATE, maturity, 0.05),
                maturity=maturity,
                dividend_curve=_flat_dividend_curve(0.03, maturity),
            ),
            _asian_spec(
                strike=100,
                maturity=maturity,
                option_type=OptionType.CALL,
                num_observations=12,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert pv_with_q < pv_no_q


# ---------------------------------------------------------------------------
# Asian theta edge cases (fixing_date near pricing_date)
# ---------------------------------------------------------------------------


class TestAsianThetaFixingAtPricingDate:
    """Theta when the first fixing coincides with pricing_date.

    When pricing_date is bumped forward by 1 day the first fixing becomes a
    past observation.  The library should produce a seasoned spec with
    observed_average = S₀ and compute theta correctly (Case 1).

    When a fixing falls strictly between pricing_date and bumped_date
    (intra-day), UnsupportedFeatureError must be raised (Case 2).
    """

    SPOT = 100.0
    STRIKE = 100.0
    VOL = 0.20
    RATE = 0.05
    MATURITY = PRICING_DATE + dt.timedelta(days=365)

    FIXINGS_WITH_T0 = (
        PRICING_DATE,
        PRICING_DATE + dt.timedelta(days=45),
        PRICING_DATE + dt.timedelta(days=105),
        PRICING_DATE + dt.timedelta(days=165),
        PRICING_DATE + dt.timedelta(days=225),
        PRICING_DATE + dt.timedelta(days=285),
        PRICING_DATE + dt.timedelta(days=345),
    )

    FIXINGS_NO_T0 = FIXINGS_WITH_T0[1:]

    def _gbm(self) -> GBMProcess:
        return _gbm_underlying(
            spot=self.SPOT,
            vol=self.VOL,
            maturity=self.MATURITY,
            paths=MC_PATHS,
            num_steps=NUM_STEPS,
            discount_curve=flat_curve(PRICING_DATE, self.MATURITY, self.RATE),
        )

    # ── Case 1: fixing == pricing_date → theta returns a float ──────────

    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_theta_case1_returns_finite(self, averaging, option_type):
        """Theta should be finite and negative for ATM options."""
        spec = AsianSpec(
            averaging=averaging,
            option_type=option_type,
            strike=self.STRIKE,
            maturity=self.MATURITY,
            currency=CURRENCY,
            fixing_dates=self.FIXINGS_WITH_T0,
            exercise_type=ExerciseType.EUROPEAN,
        )
        ov = OptionValuation(
            self._gbm(),
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=MC_SEED),
        )
        theta = ov.theta()
        assert np.isfinite(theta)
        # ATM options should have negative theta (time decay)
        assert theta < 0.0

    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    def test_theta_case1_magnitude_vs_no_t0(self, averaging):
        """Theta with t0 fixing should be smaller in magnitude than without.

        Locking in S₀ = K removes optionality from the first fixing, so
        the remaining average has less time-value sensitivity.
        """
        spec_with_t0 = AsianSpec(
            averaging=averaging,
            option_type=OptionType.PUT,
            strike=self.STRIKE,
            maturity=self.MATURITY,
            currency=CURRENCY,
            fixing_dates=self.FIXINGS_WITH_T0,
            exercise_type=ExerciseType.EUROPEAN,
        )
        spec_no_t0 = AsianSpec(
            averaging=averaging,
            option_type=OptionType.PUT,
            strike=self.STRIKE,
            maturity=self.MATURITY,
            currency=CURRENCY,
            fixing_dates=self.FIXINGS_NO_T0,
            exercise_type=ExerciseType.EUROPEAN,
        )
        gbm = self._gbm()
        mc = MonteCarloParams(random_seed=MC_SEED)
        theta_with = OptionValuation(
            gbm, spec_with_t0, PricingMethod.MONTE_CARLO, params=mc
        ).theta()
        theta_no = OptionValuation(gbm, spec_no_t0, PricingMethod.MONTE_CARLO, params=mc).theta()

        assert abs(theta_with) < abs(theta_no)

    # ── Case 2: intra-day fixing → UnsupportedFeatureError ──────────────

    def test_theta_case2_intraday_raises(self):
        """If a fixing falls between pricing_date and bumped_date, raise."""
        # Place a fixing at pricing_date + 12 hours (would need intra-day)
        intraday_fixing = PRICING_DATE + dt.timedelta(hours=12)
        spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.PUT,
            strike=self.STRIKE,
            maturity=self.MATURITY,
            currency=CURRENCY,
            fixing_dates=(intraday_fixing,) + self.FIXINGS_NO_T0,
            exercise_type=ExerciseType.EUROPEAN,
        )
        ov = OptionValuation(
            self._gbm(),
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=MC_SEED),
        )
        with pytest.raises(UnsupportedFeatureError, match="intra-day"):
            ov.theta()

    def test_theta_num_observations_magnitude_vs_equivalent_no_t0_fixing(self):
        """Implicit num_observations schedule should have smaller |theta| than no-t0 equivalent.

        A num_observations schedule includes pricing_date as the first observation,
        equivalent to an explicit fixing_dates schedule with a t0 fixing.
        Removing that first fixing should increase time-value sensitivity.
        """
        num_observations = 52
        explicit_fixings = tuple(
            pd.date_range(
                PRICING_DATE,
                self.MATURITY,
                periods=num_observations,
            ).to_pydatetime()
        )

        spec_with_t0 = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.PUT,
            strike=self.STRIKE,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_observations=num_observations,
            exercise_type=ExerciseType.EUROPEAN,
        )
        spec_no_t0 = dc_replace(
            spec_with_t0, fixing_dates=explicit_fixings[1:], num_observations=None
        )

        gbm = self._gbm()
        mc = MonteCarloParams(random_seed=MC_SEED)

        theta_with_t0 = OptionValuation(
            gbm,
            spec_with_t0,
            PricingMethod.MONTE_CARLO,
            params=mc,
        ).theta(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        theta_no_t0 = OptionValuation(
            gbm,
            spec_no_t0,
            PricingMethod.MONTE_CARLO,
            params=mc,
        ).theta(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        assert abs(theta_with_t0) < abs(theta_no_t0)

    def test_theta_num_observations_intraday_raises(self):
        """Implicit schedules with sub-daily spacing should raise for 1-day theta bumps."""
        gbm = self._gbm()
        spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.PUT,
            strike=self.STRIKE,
            maturity=gbm.pricing_date + dt.timedelta(days=30),
            currency=CURRENCY,
            # ~0.5-day spacing -> introduces fixings in (t, t+1d)
            num_observations=60,
            exercise_type=ExerciseType.EUROPEAN,
        )
        ov = OptionValuation(
            gbm,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=MC_SEED),
        )
        with pytest.raises(UnsupportedFeatureError, match="intra-day"):
            ov.theta(greek_calc_method=GreekCalculationMethod.NUMERICAL)


# ---------------------------------------------------------------------------
# MC seasoned Asian (observed_average/observed_count handled in MC engine)
# ---------------------------------------------------------------------------


class TestMCSeasonedAsian:
    """MC engine correctly incorporates past observations into the average.

    The MC engine should handle both arithmetic and geometric seasoned
    Asians directly, bypassing the Hull K* interceptor which only works
    for arithmetic.
    """

    SPOT = 100.0
    VOL = 0.20
    RATE = 0.05
    MATURITY = PRICING_DATE + dt.timedelta(days=365)

    def _gbm(self, paths: int = MC_PATHS) -> GBMProcess:
        return _gbm_underlying(
            spot=self.SPOT,
            vol=self.VOL,
            maturity=self.MATURITY,
            paths=paths,
            num_steps=NUM_STEPS,
            discount_curve=flat_curve(PRICING_DATE, self.MATURITY, self.RATE),
        )

    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_seasoned_mc_positive_pv(self, averaging, option_type):
        """A seasoned Asian with ATM observed average should have positive PV."""
        fixings = tuple(PRICING_DATE + dt.timedelta(days=d) for d in range(30, 361, 30))
        spec = AsianSpec(
            averaging=averaging,
            option_type=option_type,
            strike=self.SPOT,
            maturity=self.MATURITY,
            currency=CURRENCY,
            fixing_dates=fixings,
            observed_average=self.SPOT,
            observed_count=3,
            exercise_type=ExerciseType.EUROPEAN,
        )
        pv = OptionValuation(
            self._gbm(),
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=MC_SEED),
        ).present_value()
        assert pv > 0.0

    def test_geometric_seasoned_vs_full_mc(self):
        """Seasoned geometric MC should agree with a full simulation.

        Simulate all 7 fixings, then compare against a seasoned version
        where the first fixing (at S₀) is pre-observed.
        """
        fixings_all = (
            PRICING_DATE,
            PRICING_DATE + dt.timedelta(days=60),
            PRICING_DATE + dt.timedelta(days=120),
            PRICING_DATE + dt.timedelta(days=180),
            PRICING_DATE + dt.timedelta(days=240),
            PRICING_DATE + dt.timedelta(days=300),
            PRICING_DATE + dt.timedelta(days=360),
        )
        future_fixings = fixings_all[1:]

        # Full 7-fixing geometric Asian put
        full_spec = AsianSpec(
            averaging=AsianAveraging.GEOMETRIC,
            option_type=OptionType.PUT,
            strike=self.SPOT,
            maturity=self.MATURITY,
            currency=CURRENCY,
            fixing_dates=fixings_all,
            exercise_type=ExerciseType.EUROPEAN,
        )
        full_pv = OptionValuation(
            self._gbm(paths=500_000),
            full_spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=MC_SEED),
        ).present_value()

        # Seasoned: first fixing observed at S₀
        seasoned_spec = AsianSpec(
            averaging=AsianAveraging.GEOMETRIC,
            option_type=OptionType.PUT,
            strike=self.SPOT,
            maturity=self.MATURITY,
            currency=CURRENCY,
            fixing_dates=future_fixings,
            observed_average=self.SPOT,
            observed_count=1,
            exercise_type=ExerciseType.EUROPEAN,
        )
        seasoned_pv = OptionValuation(
            self._gbm(paths=500_000),
            seasoned_spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=MC_SEED),
        ).present_value()

        # Both should agree to machine precision: same seed → same time_grid
        # (PRICING_DATE is already grid[0]) → same paths → same fixing
        # indices → identical averaging data.  The only difference is
        # floating-point ordering of log-sums, giving ~1 ULP of drift.
        assert np.isclose(full_pv, seasoned_pv, rtol=1e-12)


# ---------------------------------------------------------------------------
# Hull Asian binomial helper tests (Figure 27.3, S=50, K=50, r=10%, σ=40%)
# ---------------------------------------------------------------------------


class TestBinomialAsianHullHelpers:
    """Focused tests for helper methods used by Hull Asian backward induction."""

    def test_compute_ordering_bounds_matches_hull_example_xyz_nodes(self):
        # Hull Ch. 27 example parameters (Figure 27.3)
        s0 = 50.0
        u = 1.0936
        d = 0.9144
        num_steps = 20

        i_idx = np.arange(num_steps + 1)[:, None]
        t_idx = np.arange(num_steps + 1)[None, :]
        spot_lattice = s0 * (u ** (t_idx - i_idx)) * (d**i_idx)

        observation_indices = np.arange(num_steps + 1, dtype=int)
        avg_min, avg_max = _BinomialAsianValuation._compute_ordering_bounds(
            spot_lattice,
            num_steps,
            observation_indices,
        )

        # Node X: t=4, row=2; Node Y: t=5, row=2; Node Z: t=5, row=3
        assert np.isclose(avg_min[2, 4], 46.65, atol=0.01)
        assert np.isclose(avg_max[2, 4], 53.83, atol=0.01)
        assert np.isclose(avg_min[2, 5], 47.99, atol=0.01)
        assert np.isclose(avg_max[2, 5], 57.39, atol=0.01)
        assert np.isclose(avg_min[3, 5], 43.88, atol=0.01)
        assert np.isclose(avg_max[3, 5], 52.48, atol=0.01)

        # Representative 4-point average grids used in the figure.
        x_grid = np.linspace(avg_min[2, 4], avg_max[2, 4], 4)
        y_grid = np.linspace(avg_min[2, 5], avg_max[2, 5], 4)
        z_grid = np.linspace(avg_min[3, 5], avg_max[3, 5], 4)

        assert np.allclose(x_grid, np.array([46.65, 49.04, 51.44, 53.83]), atol=0.01)
        assert np.allclose(y_grid, np.array([47.99, 51.12, 54.26, 57.39]), atol=0.01)
        assert np.allclose(z_grid, np.array([43.88, 46.75, 49.61, 52.48]), atol=0.01)

    def test_interp_child_values_matches_hull_example_at_node_x(self):
        # Hull Figure 27.3 interpolation at node X (t=0.20 -> 0.25).
        k = 4
        num_steps = 20
        t_idx = 4
        rows = np.array([2], dtype=int)

        avg_grid = np.zeros((k, num_steps + 1, num_steps + 1), dtype=float)
        values = np.zeros((k, num_steps + 1, num_steps + 1), dtype=float)

        avg_grid[:, 2, 5] = np.array([47.99, 51.12, 54.26, 57.39])
        values[:, 2, 5] = np.array([7.575, 8.101, 8.635, 9.178])

        avg_grid[:, 3, 5] = np.array([43.88, 46.75, 49.61, 52.48])
        values[:, 3, 5] = np.array([3.430, 3.750, 4.079, 4.416])

        x_grid = np.array([46.65, 49.04, 51.44, 53.83])
        s_up = 54.68
        s_down = 45.72
        n_obs_so_far = 5

        avg_up = ((n_obs_so_far * x_grid + s_up) / (n_obs_so_far + 1))[:, None]
        avg_down = ((n_obs_so_far * x_grid + s_down) / (n_obs_so_far + 1))[:, None]

        v_up, v_down = _BinomialAsianValuation._interp_child_values(
            avg_up=avg_up,
            avg_down=avg_down,
            avg_grid=avg_grid,
            values=values,
            rows=rows,
            t=t_idx,
        )

        assert np.isclose(v_up[2, 0], 8.247, atol=1.0e-3)
        assert np.isclose(v_down[2, 0], 4.182, atol=1.0e-3)

        p = 0.5056
        discount = np.exp(-0.1 * 0.05)
        x_values = discount * (p * v_up[:, 0] + (1.0 - p) * v_down[:, 0])
        assert np.allclose(x_values, np.array([5.642, 5.923, 6.206, 6.492]), atol=1.0e-3)


# ---------------------------------------------------------------------------
# Hull Asian binomial tree integration tests (S=50, K=50, r=10%, σ=40%, T=1)
# ---------------------------------------------------------------------------


def _hull_asian_underlying() -> UnderlyingData:
    """Build UnderlyingData for Hull's Asian option example."""
    curve_r = flat_curve(PRICING_DATE, PRICING_DATE + dt.timedelta(days=365), 0.10)
    curve_q = flat_curve(PRICING_DATE, PRICING_DATE + dt.timedelta(days=365), 0.0)
    md = market_data(pricing_date=PRICING_DATE, discount_curve=curve_r)
    return underlying(
        initial_value=50.0,
        volatility=0.40,
        market_data=md,
        dividend_curve=curve_q,
    )


class TestBinomialAsianHullTreePricing:
    """Integration tests: Hull Asian binomial tree prices (arithmetic call).

    Hull Ch. 27 Figure 27.3: S₀=50, K=50, r=10%, σ=40%, T=1 year.
    """

    @pytest.mark.parametrize(
        "exercise_type, num_steps, tree_averages, expected_pv",
        [
            pytest.param(ExerciseType.EUROPEAN, 20, 4, 7.17, id="european_20steps_4avg"),
            pytest.param(ExerciseType.AMERICAN, 20, 4, 7.77, id="american_20steps_4avg"),
            pytest.param(ExerciseType.EUROPEAN, 60, 100, 5.58, id="european_60steps_100avg"),
            pytest.param(ExerciseType.AMERICAN, 60, 100, 6.17, id="american_60steps_100avg"),
        ],
    )
    def test_hull_asian_call(
        self,
        exercise_type: ExerciseType,
        num_steps: int,
        tree_averages: int,
        expected_pv: float,
    ):
        ud = _hull_asian_underlying()
        asian_spec = AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=50.0,
            maturity=PRICING_DATE + dt.timedelta(days=365),
            num_observations=num_steps + 1,
            currency=CURRENCY,
            exercise_type=exercise_type,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            params = BinomialParams(num_steps=num_steps, asian_tree_averages=tree_averages)
        price = OptionValuation(ud, asian_spec, PricingMethod.BINOMIAL, params).present_value()
        assert np.isclose(price, expected_pv, atol=0.01)


class TestBinomialAsianExerciseTypeRouting:
    """Verify binomial Asian MC rejects Americans and Hull mode accepts both."""

    BINOM_STEPS = 20

    @staticmethod
    def _make_spec(exercise_type: ExerciseType) -> AsianSpec:
        return AsianSpec(
            averaging=AsianAveraging.ARITHMETIC,
            option_type=OptionType.CALL,
            strike=50.0,
            maturity=PRICING_DATE + dt.timedelta(days=365),
            num_observations=21,
            currency=CURRENCY,
            exercise_type=exercise_type,
        )

    def test_mc_european_prices(self):
        """Binomial MC mode returns a finite price for European Asians."""
        ud = _hull_asian_underlying()
        spec = self._make_spec(ExerciseType.EUROPEAN)
        params = BinomialParams(num_steps=self.BINOM_STEPS, mc_paths=10_000, random_seed=42)
        pv = OptionValuation(ud, spec, PricingMethod.BINOMIAL, params).present_value()
        assert np.isfinite(pv) and pv > 0.0

    def test_mc_american_raises(self):
        """Binomial MC mode raises UnsupportedFeatureError for American Asians."""
        ud = _hull_asian_underlying()
        spec = self._make_spec(ExerciseType.AMERICAN)
        params = BinomialParams(num_steps=self.BINOM_STEPS, mc_paths=10_000, random_seed=42)
        with pytest.raises(
            UnsupportedFeatureError, match="American early exercise is not supported"
        ):
            OptionValuation(ud, spec, PricingMethod.BINOMIAL, params).present_value()

    @pytest.mark.parametrize("exercise_type", [ExerciseType.EUROPEAN, ExerciseType.AMERICAN])
    def test_hull_mode_accepts_both(self, exercise_type: ExerciseType):
        """Hull representative-average mode works for both European and American."""
        ud = _hull_asian_underlying()
        spec = self._make_spec(exercise_type)
        tree_avg = 2 * self.BINOM_STEPS
        params = BinomialParams(num_steps=self.BINOM_STEPS, asian_tree_averages=tree_avg)
        pv = OptionValuation(ud, spec, PricingMethod.BINOMIAL, params).present_value()
        assert np.isfinite(pv) and pv > 0.0
