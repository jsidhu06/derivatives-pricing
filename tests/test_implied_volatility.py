"""Tests for implied volatility solver."""

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pytest

from derivatives_pricing.exceptions import (
    ConvergenceError,
    UnsupportedFeatureError,
    ValidationError,
)
from derivatives_pricing.enums import (
    AsianAveraging,
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    ExerciseType,
    ImpliedVolMethod,
    OptionType,
    PricingMethod,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from helpers import (
    market_data,
    underlying,
    make_vanilla_spec,
)
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from derivatives_pricing.valuation import (
    AsianSpec,
    BarrierSpec,
    ImpliedVolResult,
    PayoffSpec,
    VanillaSpec,
    OptionValuation,
    implied_volatility,
)
from derivatives_pricing.valuation.params import BinomialParams, PDEParams


PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)
RATE = 0.05
_DEFAULT_RATE_CURVE = DiscountCurve.flat(RATE)


def _build_valuation(
    *,
    option_type: OptionType,
    spot: float = 100.0,
    strike: float = 100.0,
    vol: float = 0.2,
    discount_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
) -> OptionValuation:
    rate_curve = discount_curve if discount_curve is not None else _DEFAULT_RATE_CURVE
    market_data_obj = market_data(
        pricing_date=PRICING_DATE,
        discount_curve=rate_curve,
    )
    underlying_data = underlying(
        initial_value=spot,
        volatility=vol,
        market_data=market_data_obj,
        dividend_curve=dividend_curve,
    )
    spec = make_vanilla_spec(
        strike=strike,
        maturity=MATURITY,
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
    )
    return OptionValuation(
        underlying=underlying_data,
        spec=spec,
        pricing_method=PricingMethod.ANALYTICAL,
    )


def _build_discrete_dividend_valuation(
    *,
    option_type: OptionType,
    vol: float,
    spot: float = 52.0,
    strike: float = 50.0,
    discount_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: list[tuple[dt.datetime, float]] | None = None,
) -> OptionValuation:
    pricing_date = PRICING_DATE
    maturity = MATURITY
    divs = (
        [
            (pricing_date + dt.timedelta(days=90), 0.5),
            (pricing_date + dt.timedelta(days=270), 0.5),
        ]
        if discrete_dividends is None
        else discrete_dividends
    )
    rate_curve = discount_curve if discount_curve is not None else _DEFAULT_RATE_CURVE
    market_data_obj = market_data(
        pricing_date=pricing_date,
        discount_curve=rate_curve,
    )
    underlying_data = underlying(
        initial_value=spot,
        volatility=vol,
        market_data=market_data_obj,
        dividend_curve=dividend_curve,
        discrete_dividends=divs,
    )
    spec = make_vanilla_spec(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
    )
    return OptionValuation(
        underlying=underlying_data,
        spec=spec,
        pricing_method=PricingMethod.ANALYTICAL,
    )


def _build_binomial_valuation(
    *,
    option_type: OptionType,
    exercise_type: ExerciseType,
    spot: float = 100.0,
    strike: float = 100.0,
    vol: float = 0.2,
    num_steps: int = 400,
    discount_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
) -> OptionValuation:
    rate_curve = discount_curve if discount_curve is not None else _DEFAULT_RATE_CURVE
    market_data_obj = market_data(
        pricing_date=PRICING_DATE,
        discount_curve=rate_curve,
    )
    underlying_data = underlying(
        initial_value=spot,
        volatility=vol,
        market_data=market_data_obj,
        dividend_curve=dividend_curve,
    )
    spec = make_vanilla_spec(
        strike=strike,
        maturity=MATURITY,
        option_type=option_type,
        exercise_type=exercise_type,
    )
    return OptionValuation(
        underlying=underlying_data,
        spec=spec,
        pricing_method=PricingMethod.BINOMIAL,
        params=BinomialParams(num_steps=num_steps),
    )


@pytest.mark.parametrize("method", [ImpliedVolMethod.NEWTON_RAPHSON, ImpliedVolMethod.BISECTION])
def test_implied_volatility_recovers_call_vol(method: ImpliedVolMethod):
    true_vol = 0.25
    initial_vol = 0.12
    pricing_valuation = _build_valuation(option_type=OptionType.CALL, vol=true_vol)
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_valuation(option_type=OptionType.CALL, vol=initial_vol)
    result = implied_volatility(target_price, solver_valuation, method=method)

    assert isinstance(result, ImpliedVolResult)
    assert result.converged
    assert abs(result.implied_vol - 0.25) < 1.0e-6


def test_implied_volatility_recovers_put_vol_with_brentq():
    true_vol = 0.18
    initial_vol = 0.35
    pricing_valuation = _build_valuation(option_type=OptionType.PUT, vol=true_vol)
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_valuation(option_type=OptionType.PUT, vol=initial_vol)
    result = implied_volatility(target_price, solver_valuation, method=ImpliedVolMethod.BRENTQ)

    assert result.converged
    assert abs(result.implied_vol - 0.18) < 1.0e-6


def test_implied_volatility_rejects_out_of_bounds_price():
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    target_price = valuation.present_value() + 1000.0

    with pytest.raises(ValidationError, match="outside no-arbitrage bounds"):
        implied_volatility(target_price, valuation)


def test_implied_volatility_rejects_monte_carlo():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    curve = DiscountCurve.flat(0.05)
    market_data = MarketData(pricing_date, curve, currency="USD")
    sim_config = SimulationConfig(
        paths=5_000,
        num_steps=50,
        end_date=maturity,
    )
    gbm_params = GBMParams(
        initial_value=100.0,
        volatility=0.2,
    )
    underlying = GBMProcess(market_data, gbm_params, sim_config)
    spec = VanillaSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=maturity,
        currency="USD",
    )
    valuation = OptionValuation(
        underlying=underlying,
        spec=spec,
        pricing_method=PricingMethod.MONTE_CARLO,
    )
    target_price = valuation.present_value()

    with pytest.raises(UnsupportedFeatureError, match="pricing methods"):
        implied_volatility(target_price, valuation)


def test_implied_volatility_with_dividend_curve():
    true_vol = 0.3
    initial_vol = 0.1
    q_curve = DiscountCurve.flat(0.02)
    pricing_valuation = _build_valuation(
        option_type=OptionType.CALL, vol=true_vol, dividend_curve=q_curve
    )
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_valuation(
        option_type=OptionType.CALL, vol=initial_vol, dividend_curve=q_curve
    )
    result = implied_volatility(target_price, solver_valuation)

    assert result.converged
    assert np.isclose(result.implied_vol, 0.3, atol=1.0e-6)


def test_implied_volatility_with_discrete_dividends():
    true_vol = 0.4
    initial_vol = 0.15
    pricing_valuation = _build_discrete_dividend_valuation(option_type=OptionType.PUT, vol=true_vol)
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_discrete_dividend_valuation(
        option_type=OptionType.PUT, vol=initial_vol
    )
    result = implied_volatility(target_price, solver_valuation)

    assert result.converged
    assert np.isclose(result.implied_vol, true_vol, atol=1.0e-6)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_implied_volatility_recovers_american_binomial(option_type: OptionType):
    true_vol = 0.3
    initial_vol = 0.12
    dividend_curve = DiscountCurve.flat(0.02) if option_type is OptionType.CALL else None

    pricing_valuation = _build_binomial_valuation(
        option_type=option_type,
        exercise_type=ExerciseType.AMERICAN,
        vol=true_vol,
        dividend_curve=dividend_curve,
    )
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_binomial_valuation(
        option_type=option_type,
        exercise_type=ExerciseType.AMERICAN,
        vol=initial_vol,
        dividend_curve=dividend_curve,
    )
    result = implied_volatility(
        target_price,
        solver_valuation,
        method=ImpliedVolMethod.BISECTION,
        tol=1.0e-6,
    )

    assert result.converged
    assert np.isclose(result.implied_vol, true_vol, atol=5.0e-3)


@pytest.mark.parametrize("method", [ImpliedVolMethod.NEWTON_RAPHSON, ImpliedVolMethod.BRENTQ])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_implied_volatility_extreme_high_vol(option_type, method):
    """A 300% vol target is recovered to tol and reprices."""
    true_vol = 3.0
    target_price = _build_valuation(option_type=option_type, vol=true_vol).present_value()
    solver_valuation = _build_valuation(option_type=option_type, vol=0.2)
    result = implied_volatility(
        target_price,
        solver_valuation,
        method=method,
        vol_bounds=(1.0e-6, 5.0),
        tol=1.0e-8,
        max_iter=200,
    )
    assert result.converged
    assert np.isclose(result.implied_vol, true_vol, atol=1.0e-7)


@pytest.mark.parametrize("method", list(ImpliedVolMethod))
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("true_vol", [1.0e-3, 2.0e-3])
def test_implied_volatility_flags_vols_the_price_cannot_determine(option_type, true_vol, method):
    """At a 0.1-0.2% vol an at-the-spot option is at its forward intrinsic value.

    Vega is negligible, so a wide range of vols reprice the target to 1e-7
    (the old price-tolerance solvers returned e.g. 0.0067 for 0.001 and
    called it converged). The root finder still converges and the result
    reprices, but it is flagged ``well_determined=False`` with a warning.
    """
    target_price = _build_valuation(option_type=option_type, vol=true_vol).present_value()
    solver_valuation = _build_valuation(option_type=option_type, vol=0.2)
    with pytest.warns(UserWarning, match="not determined by the price"):
        result = implied_volatility(
            target_price,
            solver_valuation,
            method=method,
            vol_bounds=(1.0e-6, 5.0),
            tol=1.0e-8,
            max_iter=200,
        )
    assert result.converged
    assert not result.well_determined
    repriced = _build_valuation(option_type=option_type, vol=result.implied_vol).present_value()
    assert np.isclose(repriced, target_price, atol=1.0e-7)


@pytest.mark.parametrize("method", list(ImpliedVolMethod))
def test_implied_volatility_low_vol_at_the_forward_is_well_determined(method):
    """With the strike at the forward, vega stays ~0.4 S sqrt(T) even at 0.2% vol.

    So a tiny vol is fully determined by the price and must be recovered to tol.
    """
    probe = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    forward = 100.0 / float(probe.discount_curve.df(probe._maturity_year_fraction()))
    true_vol = 2.0e-3
    target_price = _build_valuation(
        option_type=OptionType.CALL, strike=forward, vol=true_vol
    ).present_value()
    solver_valuation = _build_valuation(option_type=OptionType.CALL, strike=forward, vol=0.2)
    result = implied_volatility(target_price, solver_valuation, method=method, tol=1.0e-10)
    assert result.converged
    assert np.isclose(result.implied_vol, true_vol, atol=1.0e-9)


def test_implied_volatility_far_otm_agrees_across_methods():
    """K=130, 1 month: vega is small but resolvable; every method agrees to tol.

    The old price tolerance let Newton and bisection stop ~5e-6 vol away.
    """
    md = market_data(pricing_date=PRICING_DATE, discount_curve=_DEFAULT_RATE_CURVE)
    maturity = PRICING_DATE + dt.timedelta(days=30)
    spec = make_vanilla_spec(option_type=OptionType.CALL, strike=130.0, maturity=maturity)

    def valuation(vol: float) -> OptionValuation:
        return OptionValuation(
            underlying(initial_value=100.0, volatility=vol, market_data=md),
            spec,
            PricingMethod.ANALYTICAL,
        )

    target_price = valuation(0.2).present_value()
    for method in ImpliedVolMethod:
        result = implied_volatility(target_price, valuation(0.5), method=method)
        assert result.converged, method
        assert np.isclose(result.implied_vol, 0.2, atol=1.0e-8), method


def test_implied_volatility_brent_reports_non_convergence_instead_of_raising():
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    target_price = _build_valuation(option_type=OptionType.CALL, vol=0.35).present_value()
    result = implied_volatility(
        target_price, valuation, method=ImpliedVolMethod.BRENTQ, tol=1.0e-14, max_iter=3
    )
    assert not result.converged
    # An unconverged vol is not a usable answer, however steep the price.
    assert not result.well_determined


def test_implied_volatility_rejects_non_positive_tolerances():
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    with pytest.raises(ValidationError, match="tol must be positive"):
        implied_volatility(10.0, valuation, tol=0.0)
    with pytest.raises(ValidationError, match="min_vega must be >= 0"):
        implied_volatility(10.0, valuation, min_vega=-1.0)


def test_implied_volatility_rejects_call_price_above_spot():
    """European call cannot exceed spot under no-arbitrage bounds."""
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    with pytest.raises(ValidationError, match="outside no-arbitrage bounds"):
        implied_volatility(valuation.underlying.initial_value + 1.0, valuation)


def test_implied_volatility_rejects_put_price_above_discounted_strike():
    """European put upper bound is discounted strike."""
    valuation = _build_valuation(option_type=OptionType.PUT, vol=0.2)
    ttm = (valuation.maturity - valuation.pricing_date).days / 365.0
    upper = valuation.strike * np.exp(-0.05 * ttm)
    with pytest.raises(ValidationError, match="outside no-arbitrage bounds"):
        implied_volatility(float(upper + 1.0), valuation)


@pytest.mark.parametrize("method", [ImpliedVolMethod.BISECTION, ImpliedVolMethod.BRENTQ])
def test_implied_volatility_converges_near_upper_bound(method):
    """Solver should converge when true volatility is near the upper bound."""
    true_vol = 4.9
    pricing_valuation = _build_valuation(option_type=OptionType.PUT, vol=true_vol)
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_valuation(option_type=OptionType.PUT, vol=0.2)
    result = implied_volatility(
        target_price,
        solver_valuation,
        method=method,
        vol_bounds=(1.0e-6, 5.0),
        tol=1.0e-8,
        max_iter=300,
    )
    assert result.converged
    assert np.isclose(result.implied_vol, true_vol, rtol=2.0e-2, atol=1.0e-3)


def test_implied_volatility_rejects_non_positive_target_price():
    """Zero target price is rejected by validation/no-arbitrage guards."""
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    with pytest.raises(ValidationError, match="outside no-arbitrage bounds|non-negative"):
        implied_volatility(0.0, valuation)


@dataclass
class _AsianIVCase:
    option_type: OptionType
    exercise_type: ExerciseType
    averaging: AsianAveraging
    true_vol: float
    initial_vol: float
    num_steps: int
    discrete_divs: list[tuple[dt.datetime, float]] | None = None
    dividend_curve: DiscountCurve | None = None


_NONFLAT_DIV_CURVE = DiscountCurve.from_zero_rates(
    times=np.array([0.0, 0.25, 0.5, 1.0]),
    zero_rates=np.array([0.01, 0.012, 0.015, 0.02]),
)

_ASIAN_IV_CASES = [
    pytest.param(
        _AsianIVCase(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            averaging=AsianAveraging.ARITHMETIC,
            true_vol=0.25,
            initial_vol=0.40,
            num_steps=50,
        ),
        id="european-call-arithmetic-flat",
    ),
    pytest.param(
        _AsianIVCase(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            averaging=AsianAveraging.GEOMETRIC,
            true_vol=0.30,
            initial_vol=0.10,
            num_steps=50,
            discrete_divs=[
                (PRICING_DATE + dt.timedelta(days=90), 1.0),
                (PRICING_DATE + dt.timedelta(days=270), 1.0),
            ],
        ),
        id="european-put-geometric-discrete-divs",
    ),
    pytest.param(
        _AsianIVCase(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            averaging=AsianAveraging.ARITHMETIC,
            true_vol=0.35,
            initial_vol=0.15,
            num_steps=50,
            dividend_curve=_NONFLAT_DIV_CURVE,
        ),
        id="american-call-arithmetic-nonflat-div-curve",
    ),
    pytest.param(
        _AsianIVCase(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.AMERICAN,
            averaging=AsianAveraging.ARITHMETIC,
            true_vol=0.20,
            initial_vol=0.45,
            num_steps=50,
            discrete_divs=[
                (PRICING_DATE + dt.timedelta(days=120), 1.5),
                (PRICING_DATE + dt.timedelta(days=300), 1.5),
            ],
        ),
        id="american-put-arithmetic-discrete-divs",
    ),
]


@pytest.mark.parametrize("case", _ASIAN_IV_CASES)
def test_implied_volatility_asian_binomial(case: _AsianIVCase):
    """IV solver recovers volatility for Asian options priced via binomial."""
    md = market_data(pricing_date=PRICING_DATE, discount_curve=_DEFAULT_RATE_CURVE)
    ud = underlying(
        initial_value=100.0,
        volatility=case.true_vol,
        market_data=md,
        dividend_curve=case.dividend_curve,
        discrete_dividends=case.discrete_divs,
    )
    asian_spec = AsianSpec(
        averaging=case.averaging,
        option_type=case.option_type,
        exercise_type=case.exercise_type,
        strike=100.0,
        maturity=MATURITY,
        num_observations=case.num_steps,
    )
    params = BinomialParams(num_steps=case.num_steps, asian_tree_averages=2 * case.num_steps)

    pricing_val = OptionValuation(ud, asian_spec, PricingMethod.BINOMIAL, params=params)
    target_price = pricing_val.present_value()

    solver_val = OptionValuation(
        ud.replace(volatility=case.initial_vol), asian_spec, PricingMethod.BINOMIAL, params=params
    )
    result = implied_volatility(
        target_price,
        solver_val,
        method=ImpliedVolMethod.BRENTQ,
        initial_vol=case.initial_vol,
        vol_bounds=(0.05, 5.0),
        tol=1.0e-6,
    )

    assert result.converged
    assert np.isclose(result.implied_vol, case.true_vol, atol=5.0e-3)


def test_implied_volatility_rejects_barrier_spec():
    """Barrier specs are blocked: P(sigma) is not guaranteed to be monotonic in sigma."""
    market_data_obj = market_data(pricing_date=PRICING_DATE, discount_curve=_DEFAULT_RATE_CURVE)
    underlying_data = underlying(
        initial_value=100.0,
        volatility=0.2,
        market_data=market_data_obj,
    )
    barrier_spec = BarrierSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=MATURITY,
        barrier=85.0,
        direction=BarrierDirection.DOWN,
        action=BarrierAction.OUT,
        monitoring=BarrierMonitoring.CONTINUOUS,
    )
    valuation = OptionValuation(
        underlying=underlying_data,
        spec=barrier_spec,
        pricing_method=PricingMethod.ANALYTICAL,
    )
    target_price = valuation.present_value()

    with pytest.raises(
        UnsupportedFeatureError,
        match=r"not supported for barrier specs",
    ):
        implied_volatility(target_price, valuation)


def test_implied_volatility_rejects_payoff_spec():
    """PayoffSpec is blocked: P(sigma) monotonicity depends on the
    user-supplied payoff function and isn't guaranteed in general."""
    market_data_obj = market_data(pricing_date=PRICING_DATE, discount_curve=_DEFAULT_RATE_CURVE)
    underlying_data = underlying(
        initial_value=100.0,
        volatility=0.2,
        market_data=market_data_obj,
    )
    payoff_spec = PayoffSpec(
        exercise_type=ExerciseType.EUROPEAN,
        maturity=MATURITY,
        payoff_fn=lambda S: np.where(S > 100.0, 1.0, 0.0),
        currency="USD",
    )
    valuation = OptionValuation(
        underlying=underlying_data,
        spec=payoff_spec,
        pricing_method=PricingMethod.BINOMIAL,
    )
    target_price = valuation.present_value()

    with pytest.raises(
        UnsupportedFeatureError,
        match=r"not supported for PayoffSpec",
    ):
        implied_volatility(target_price, valuation)


# ---------------------------------------------------------------------------
# Solver diagnostics, defaults and robustness
# ---------------------------------------------------------------------------


def _american_put(method: PricingMethod, vol: float, curve: DiscountCurve | None = None):
    rate_curve = curve if curve is not None else _DEFAULT_RATE_CURVE
    params = (
        BinomialParams(num_steps=200)
        if method is PricingMethod.BINOMIAL
        else PDEParams(spot_steps=80, time_steps=100)
    )
    return OptionValuation(
        underlying(
            initial_value=100.0,
            volatility=vol,
            market_data=market_data(pricing_date=PRICING_DATE, discount_curve=rate_curve),
        ),
        make_vanilla_spec(
            strike=110.0,
            maturity=MATURITY,
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.AMERICAN,
        ),
        method,
        params=params,
    )


def test_result_reports_pricing_calls_and_well_determined():
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    target = _build_valuation(option_type=OptionType.CALL, vol=0.35).present_value()
    result = implied_volatility(target, valuation)
    assert result.converged and result.well_determined
    # Bracketing prices both ends; each Newton step prices once more.
    assert result.function_evaluations >= result.iterations + 2


def test_quote_at_intrinsic_value_is_not_well_determined():
    """Regression: a deep in-the-money American put is worth its intrinsic value
    at every small volatility, so the solve ends near zero vol. The conditioning
    check then bumped the vol down by 1% for a numerical vega and raised on the
    negative volatility."""
    valuation = _american_put(PricingMethod.BINOMIAL, 0.3)  # K = 110, S = 100
    with pytest.warns(UserWarning, match="is not determined by the price"):
        result = implied_volatility(10.0, valuation)
    assert result.implied_vol < 0.01
    assert not result.well_determined


@pytest.mark.filterwarnings("ignore:PDE grid has", "ignore:PDE domain boundary is")
@pytest.mark.parametrize("engine", [PricingMethod.BINOMIAL, PricingMethod.PDE_FD])
def test_numerical_engines_default_to_brent(engine):
    """Brent needs no vega; Newton's numerical vega costs 2 extra pricings a step."""
    target = _american_put(engine, 0.3).present_value()
    default = implied_volatility(target, _american_put(engine, 0.2))
    newton = implied_volatility(
        target, _american_put(engine, 0.2), method=ImpliedVolMethod.NEWTON_RAPHSON
    )
    assert default.converged and newton.converged
    assert np.isclose(default.implied_vol, 0.3, atol=1.0e-6)
    assert default.function_evaluations < newton.function_evaluations


def test_root_on_a_bracket_end_returns_without_solving():
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    target = _build_valuation(option_type=OptionType.CALL, vol=0.4).present_value()
    result = implied_volatility(target, valuation, vol_bounds=(0.1, 0.4))
    assert result.converged
    assert result.implied_vol == 0.4
    assert result.iterations == 0


def test_bracket_failure_reports_the_interval_and_residuals():
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    target = _build_valuation(option_type=OptionType.CALL, vol=3.0).present_value()
    with pytest.raises(ConvergenceError, match=r"vol in \[.*\] gives price residuals"):
        implied_volatility(target, valuation, vol_bounds=(0.01, 0.02))


def test_tree_vol_floor_uses_the_steepest_step_of_a_term_curve():
    """The CRR floor is max_i |r_i - q_i| sqrt(dt), not the average zero rate.

    On a steep curve the late steps' forward rate is far above the average,
    so bracketing at the old floor priced an inadmissible tree and raised
    ArbitrageViolationError.
    """
    steep = DiscountCurve.from_zero_rates(np.array([0.25, 0.5, 1.0]), np.array([0.01, 0.10, 0.30]))
    target = _american_put(PricingMethod.BINOMIAL, 0.2, steep).present_value()
    result = implied_volatility(
        target, _american_put(PricingMethod.BINOMIAL, 0.2, steep), vol_bounds=(1.0e-6, 5.0)
    )
    assert result.converged


# The vol sweep and bracketing reach far outside one grid's comfort zone; the
# PDE says so, as it should.
@pytest.mark.filterwarnings("ignore:PDE grid has", "ignore:PDE domain boundary is")
@pytest.mark.parametrize("engine", [PricingMethod.BINOMIAL, PricingMethod.PDE_FD])
def test_numerical_engine_price_is_monotone_in_vol(engine):
    """The root finders assume price is non-decreasing in vol; check the discretised engines.

    Flat (not decreasing) at low vol, where this ITM American put is
    worth exactly its intrinsic value 10.
    """
    vols = np.linspace(0.05, 1.0, 40)
    prices = np.array([_american_put(engine, v).present_value() for v in vols])
    assert np.all(np.diff(prices) >= 0.0)
    assert np.all(np.diff(prices[prices > 10.0 + 1e-9]) > 0.0)


@pytest.mark.parametrize("max_iter", [0, -1, 2.5, True])
def test_implied_volatility_rejects_bad_max_iter(max_iter):
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    with pytest.raises(ValidationError, match="max_iter must be a positive integer"):
        implied_volatility(10.0, valuation, max_iter=max_iter)


@pytest.mark.parametrize("initial_vol", [0.0, -0.2, float("nan"), float("inf")])
def test_implied_volatility_rejects_bad_initial_vol(initial_vol):
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    with pytest.raises(ValidationError, match="initial_vol must be finite and positive"):
        implied_volatility(10.0, valuation, initial_vol=initial_vol)
