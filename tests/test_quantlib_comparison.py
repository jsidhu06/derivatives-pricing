"""Compare option present values between derivatives_pricing and QuantLib."""

from __future__ import annotations
import datetime as dt
import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pytest

from derivatives_pricing.enums import (
    AsianAveraging,
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    DayCountConvention,
    ExerciseType,
    OptionType,
    PricingMethod,
    RebateTiming,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from helpers import flat_curve
from derivatives_pricing.valuation import (
    AsianSpec,
    BarrierSpec,
    VanillaSpec,
    OptionValuation,
    UnderlyingData,
    as_underlying_data,
)
from derivatives_pricing.stochastic_processes import GBMParams, GBMProcess, SimulationConfig
from derivatives_pricing.valuation.params import BinomialParams, MonteCarloParams, PDEParams
from derivatives_pricing.utils import calculate_year_fraction, pv_discrete_dividends

if TYPE_CHECKING:
    import QuantLib as ql_typing

ql = pytest.importorskip("QuantLib")


logger = logging.getLogger(__name__)


PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = PRICING_DATE + dt.timedelta(days=365)
RISK_FREE = 0.1
VOL = 0.4
CURRENCY = "USD"


def _ql_dcc(dcc: DayCountConvention):
    """Return the QuantLib DayCounter matching a DP DayCountConvention."""
    if dcc is DayCountConvention.ACT_360:
        return ql.Actual360()
    return ql.Actual365Fixed()


PDE_CFG = PDEParams(spot_steps=200, time_steps=200, max_iter=20_000)
BINOM_CFG = BinomialParams(num_steps=500)
MC_CFG = MonteCarloParams(random_seed=42, deg=3)


def _market_data(
    r_curve: DiscountCurve | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> MarketData:
    curve = r_curve if r_curve is not None else flat_curve(PRICING_DATE, MATURITY, RISK_FREE)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY, day_count_convention=dcc)


def _ql_curve_from_times(
    *,
    times: np.ndarray,
    dfs: np.ndarray,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> ql_typing.YieldTermStructureHandle:
    denom = 360.0 if dcc is DayCountConvention.ACT_360 else 365.0
    day_count = _ql_dcc(dcc)
    ql.Settings.instance().evaluationDate = ql.Date(
        PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year
    )
    dates = [ql.Settings.instance().evaluationDate]
    for t in times[1:]:
        days = int(round(float(t) * denom))
        dates.append(ql.Settings.instance().evaluationDate + days)
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, list(dfs), day_count))


def _spec(
    *,
    strike: float,
    option_type: OptionType,
    exercise_type: ExerciseType = ExerciseType.AMERICAN,
) -> VanillaSpec:
    return VanillaSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


def _quantlib_dividend_schedule(
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
) -> ql_typing.DividendVector:
    if discrete_dividends:
        div_dates = []
        dividends = []
        for ex_date, amount in discrete_dividends:
            if PRICING_DATE <= ex_date <= MATURITY:
                div_dates.append(ql.Date(ex_date.day, ex_date.month, ex_date.year))
                dividends.append(float(amount))
        return ql.DividendVector(div_dates, dividends)
    return ql.DividendVector([], [])


def _quantlib_american_with_curves(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    rf_curve: ql_typing.YieldTermStructureHandle,
    div_curve: ql_typing.YieldTermStructureHandle,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
    grid_points: int = 200,
    time_steps: int = 400,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    ql.Settings.instance().evaluationDate = ql.Date(
        PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year
    )

    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(
            ql.Settings.instance().evaluationDate,
            ql.TARGET(),
            float(VOL),
            _ql_dcc(dcc),
        )
    )

    if option_type is OptionType.PUT:
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(strike))
    else:
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(strike))

    exercise = ql.AmericanExercise(ql.Settings.instance().evaluationDate, ql_maturity)
    option = ql.VanillaOption(payoff, exercise)

    dividend_schedule = _quantlib_dividend_schedule(discrete_dividends)

    process = ql.BlackScholesMertonProcess(spot_handle, div_curve, rf_curve, vol_handle)

    engine = ql.FdBlackScholesVanillaEngine(
        process,
        dividend_schedule,
        int(grid_points),
        int(time_steps),
    )
    option.setPricingEngine(engine)
    return float(option.NPV())


def _quantlib_american(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    dividend_yield: float,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
    grid_points: int = 200,
    time_steps: int = 400,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    ql.Settings.instance().evaluationDate = ql.Date(
        PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year
    )
    day_count = _ql_dcc(dcc)
    rf_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(
            ql.Settings.instance().evaluationDate,
            float(RISK_FREE),
            day_count,
        )
    )
    div_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(
            ql.Settings.instance().evaluationDate,
            float(dividend_yield),
            day_count,
        )
    )
    return _quantlib_american_with_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        rf_curve=rf_curve,
        div_curve=div_curve,
        discrete_dividends=discrete_dividends,
        grid_points=grid_points,
        time_steps=time_steps,
        dcc=dcc,
    )


def _quantlib_american_curves(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    r_times: np.ndarray,
    r_dfs: np.ndarray,
    q_times: np.ndarray,
    q_dfs: np.ndarray,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    grid_points: int = 200,
    time_steps: int = 400,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    rf_curve = _ql_curve_from_times(times=r_times, dfs=r_dfs, dcc=dcc)
    div_curve = _ql_curve_from_times(times=q_times, dfs=q_dfs, dcc=dcc)
    return _quantlib_american_with_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        rf_curve=rf_curve,
        div_curve=div_curve,
        discrete_dividends=discrete_dividends,
        grid_points=grid_points,
        time_steps=time_steps,
        dcc=dcc,
    )


def _pde_fd_american(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    r_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    ud = UnderlyingData(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(r_curve, dcc=dcc),
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )
    spec = _spec(strike=strike, option_type=option_type)
    return OptionValuation(ud, spec, PricingMethod.PDE_FD, PDE_CFG).present_value()


# ═══════════════════════════════════════════════════════════════════════════
# Method equivalence — all pricing methods + QuantLib benchmark
# ═══════════════════════════════════════════════════════════════════════════

# Section-local constants (vol=0.2, rate=0.05 — different from the
# American FD section which uses VOL=0.4, RISK_FREE=0.1).
_ME_VOL = 0.2
_ME_RATE = 0.05
_ME_PDE = PDEParams(spot_steps=400, time_steps=400, max_iter=20_000)
_ME_BINOM = BinomialParams(num_steps=1500)
_ME_MC_EU = MonteCarloParams(random_seed=42)
_ME_MC_AM = MonteCarloParams(random_seed=42, deg=3)


def _nonflat_r_curve() -> DiscountCurve:
    times = np.array([0.0, 0.25, 0.5, 1.0])
    forwards = np.array([0.03, 0.05, 0.04])
    return DiscountCurve.from_forwards(times=times, forwards=forwards)


def _me_market_data(
    r_curve: DiscountCurve | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> MarketData:
    if r_curve is None:
        ttm = calculate_year_fraction(PRICING_DATE, MATURITY, dcc)
        r_curve = DiscountCurve.flat(_ME_RATE, end_time=ttm)
    return MarketData(PRICING_DATE, r_curve, currency=CURRENCY, day_count_convention=dcc)


def _underlying(
    *,
    spot: float,
    r_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> UnderlyingData:
    return UnderlyingData(
        initial_value=spot,
        volatility=_ME_VOL,
        market_data=_me_market_data(r_curve, dcc=dcc),
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def _gbm(
    *,
    spot: float,
    r_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    paths: int = 200_000,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> GBMProcess:
    sim_config = SimulationConfig(
        paths=paths,
        num_steps=52,
        end_date=MATURITY,
    )
    params = GBMParams(
        initial_value=spot,
        volatility=_ME_VOL,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )
    return GBMProcess(_me_market_data(r_curve, dcc=dcc), params, sim_config)


def _ql_price(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    exercise_type: ExerciseType,
    rf_handle: "ql_typing.YieldTermStructureHandle",
    div_handle: "ql_typing.YieldTermStructureHandle",
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    """QuantLib vanilla pricing with vol=_ME_VOL and configurable curves."""
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)

    spot_h = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, ql.TARGET(), float(_ME_VOL), _ql_dcc(dcc))
    )
    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(ql_type, float(strike))
    process = ql.BlackScholesMertonProcess(spot_h, div_handle, rf_handle, vol_h)
    dividend_schedule = _quantlib_dividend_schedule(discrete_dividends)

    if exercise_type is ExerciseType.EUROPEAN:
        exercise = ql.EuropeanExercise(ql_maturity)
    else:
        exercise = ql.AmericanExercise(eval_date, ql_maturity)

    option = ql.VanillaOption(payoff, exercise)

    if not discrete_dividends and exercise_type is ExerciseType.EUROPEAN:
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    else:
        option.setPricingEngine(
            ql.FdBlackScholesVanillaEngine(process, dividend_schedule, 200, 400)
        )
    return float(option.NPV())


def _ql_flat_handles(
    rf_rate: float = _ME_RATE,
    div_yield: float = 0.0,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> tuple:
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    day_count = _ql_dcc(dcc)
    rf_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, float(rf_rate), day_count))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, float(div_yield), day_count))
    return rf_h, div_h


def _dp_price(
    underlying: UnderlyingData | GBMProcess,
    spec: VanillaSpec | AsianSpec | BarrierSpec,
    method: PricingMethod,
    params: BinomialParams | MonteCarloParams | PDEParams | None = None,
) -> float:
    return OptionValuation(underlying, spec, method, params=params).present_value()


# ── European vanilla — BSM / PDE / Binomial / MC / QuantLib ────────────


@pytest.mark.parametrize(
    "spot,strike,option_type,dividend_yield,dcc",
    [
        (90.0, 100.0, OptionType.CALL, 0.03, DayCountConvention.ACT_365F),
        (110.0, 100.0, OptionType.CALL, 0.02, DayCountConvention.ACT_360),
    ],
    ids=["otm_call_ACT365F", "itm_call_ACT360"],
)
def test_european_vanilla_all_methods_vs_quantlib(spot, strike, option_type, dividend_yield, dcc):
    """European vanilla: BSM, PDE, Binomial, MC all match QuantLib analytical."""
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY, dcc)
    q_curve = DiscountCurve.flat(dividend_yield, end_time=ttm)
    ud = _underlying(spot=spot, dividend_curve=q_curve, dcc=dcc)
    gbm = _gbm(spot=spot, dividend_curve=q_curve, dcc=dcc)
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.EUROPEAN)

    bsm_pv = _dp_price(ud, spec, PricingMethod.BSM)
    pde_pv = _dp_price(ud, spec, PricingMethod.PDE_FD, _ME_PDE)
    binom_pv = _dp_price(ud, spec, PricingMethod.BINOMIAL, _ME_BINOM)
    mc_pv = _dp_price(gbm, spec, PricingMethod.MONTE_CARLO, _ME_MC_EU)

    rf_h, div_h = _ql_flat_handles(div_yield=dividend_yield, dcc=dcc)
    ql_pv = _ql_price(
        spot=spot,
        strike=strike,
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        rf_handle=rf_h,
        div_handle=div_h,
        dcc=dcc,
    )

    logger.info(
        "European %s S=%.0f K=%.0f q=%.2f dcc=%s | BSM=%.6f PDE=%.6f Binom=%.6f MC=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        dividend_yield,
        dcc.value,
        bsm_pv,
        pde_pv,
        binom_pv,
        mc_pv,
        ql_pv,
    )

    assert np.isclose(bsm_pv, ql_pv, rtol=1e-8), f"BSM {bsm_pv:.6f} vs QL {ql_pv:.6f}"
    assert np.isclose(pde_pv, ql_pv, rtol=2e-3, atol=1e-4), f"PDE {pde_pv:.6f} vs QL {ql_pv:.6f}"
    assert np.isclose(binom_pv, ql_pv, rtol=2e-3, atol=1e-4), (
        f"Binom {binom_pv:.6f} vs QL {ql_pv:.6f}"
    )
    assert np.isclose(mc_pv, ql_pv, rtol=0.01, atol=1e-3), f"MC {mc_pv:.6f} vs QL {ql_pv:.6f}"


# ── American vanilla — PDE / Binomial / MC / QuantLib ──────────────────


@pytest.mark.parametrize(
    "spot,strike,option_type,dividend_yield,dcc",
    [
        (90.0, 100.0, OptionType.CALL, 0.03, DayCountConvention.ACT_365F),
        (110.0, 100.0, OptionType.CALL, 0.02, DayCountConvention.ACT_360),
    ],
    ids=["otm_call_ACT365F", "itm_call_ACT360"],
)
def test_american_vanilla_all_methods_vs_quantlib(spot, strike, option_type, dividend_yield, dcc):
    """American vanilla: PDE, Binomial, MC all match QuantLib FD."""
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY, dcc)
    q_curve = DiscountCurve.flat(dividend_yield, end_time=ttm)
    ud = _underlying(spot=spot, dividend_curve=q_curve, dcc=dcc)
    gbm = _gbm(spot=spot, dividend_curve=q_curve, dcc=dcc)
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.AMERICAN)

    pde_pv = _dp_price(ud, spec, PricingMethod.PDE_FD, _ME_PDE)
    binom_pv = _dp_price(ud, spec, PricingMethod.BINOMIAL, _ME_BINOM)
    mc_pv = _dp_price(gbm, spec, PricingMethod.MONTE_CARLO, _ME_MC_AM)

    rf_h, div_h = _ql_flat_handles(div_yield=dividend_yield, dcc=dcc)
    ql_pv = _ql_price(
        spot=spot,
        strike=strike,
        option_type=option_type,
        exercise_type=ExerciseType.AMERICAN,
        rf_handle=rf_h,
        div_handle=div_h,
        dcc=dcc,
    )

    logger.info(
        "American %s S=%.0f K=%.0f q=%.2f dcc=%s | PDE=%.6f Binom=%.6f MC=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        dividend_yield,
        dcc.value,
        pde_pv,
        binom_pv,
        mc_pv,
        ql_pv,
    )

    assert np.isclose(pde_pv, ql_pv, rtol=5e-3, atol=1e-4), f"PDE {pde_pv:.6f} vs QL {ql_pv:.6f}"
    assert np.isclose(binom_pv, ql_pv, rtol=5e-3, atol=1e-4), (
        f"Binom {binom_pv:.6f} vs QL {ql_pv:.6f}"
    )
    assert np.isclose(mc_pv, ql_pv, rtol=0.015, atol=1e-3), f"MC {mc_pv:.6f} vs QL {ql_pv:.6f}"


# ── Discrete-dividend European — all methods + QuantLib ────────────────


@pytest.mark.parametrize(
    "r_curve",
    [
        flat_curve(PRICING_DATE, MATURITY, _ME_RATE),
        _nonflat_r_curve(),
    ],
    ids=["flat", "nonflat"],
)
def test_discrete_div_european_vs_quantlib(r_curve):
    """Discrete divs European: PDE/MC align, BSM/Binomial align, all close to QuantLib."""
    spot = 52.0
    strike = 50.0
    divs = [
        (PRICING_DATE + dt.timedelta(days=90), 0.5),
        (PRICING_DATE + dt.timedelta(days=270), 0.5),
    ]

    ud = _underlying(spot=spot, r_curve=r_curve, discrete_dividends=divs)
    gbm = _gbm(spot=spot, r_curve=r_curve, discrete_dividends=divs, paths=200_000)
    spec = _spec(strike=strike, option_type=OptionType.PUT, exercise_type=ExerciseType.EUROPEAN)

    pde_pv = _dp_price(ud, spec, PricingMethod.PDE_FD, _ME_PDE)
    mc_pv = _dp_price(gbm, spec, PricingMethod.MONTE_CARLO, _ME_MC_EU)
    bsm_pv = _dp_price(ud, spec, PricingMethod.BSM)
    binom_pv = _dp_price(ud, spec, PricingMethod.BINOMIAL, _ME_BINOM)

    # Vol-adjusted BSM/Binomial cross-check
    pv_divs = pv_discrete_dividends(
        dividends=divs,
        curve_date=ud.pricing_date,
        end_date=spec.maturity,
        discount_curve=r_curve,
    )
    vol_multiplier = ud.initial_value / (ud.initial_value - pv_divs)
    adjusted_ud = ud.replace(volatility=ud.volatility * vol_multiplier)
    bsm_adj = _dp_price(adjusted_ud, spec, PricingMethod.BSM)
    binom_adj = _dp_price(adjusted_ud, spec, PricingMethod.BINOMIAL, _ME_BINOM)

    # QuantLib FD European with discrete dividends
    rf_h = _ql_curve_from_times(times=r_curve.times, dfs=r_curve.dfs)
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, ql.Actual365Fixed()))
    ql_pv = _ql_price(
        spot=spot,
        strike=strike,
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        rf_handle=rf_h,
        div_handle=div_h,
        discrete_dividends=divs,
    )

    logger.info(
        "Disc-div EU PUT | PDE=%.6f MC=%.6f BSM=%.6f Binom=%.6f QL=%.6f",
        pde_pv,
        mc_pv,
        bsm_pv,
        binom_pv,
        ql_pv,
    )

    # PDE/MC agree with each other (MC noise dominates)
    assert np.isclose(pde_pv, mc_pv, rtol=0.015, atol=1e-3)
    assert np.isclose(pde_pv, ql_pv, rtol=5e-3, atol=1e-4), f"PDE {pde_pv:.6f} vs QL {ql_pv:.6f}"

    # BSM/Binomial agree with each other (CRR converges well for European)
    assert np.isclose(bsm_pv, binom_pv, rtol=2e-3, atol=1e-4)
    assert np.isclose(bsm_adj, binom_adj, rtol=2e-3, atol=1e-4)

    # Vol-adjusted prices close to PDE/MC (escrow vs Hull-Fisher approximation
    # produces a genuine small model gap; keep wider than pure discretization)
    assert np.isclose(pde_pv, bsm_adj, rtol=0.015, atol=1e-3)
    assert np.isclose(mc_pv, binom_adj, rtol=0.015, atol=1e-3)


# ── Discrete-dividend American — PDE / MC / QuantLib ───────────────────


@pytest.mark.parametrize(
    "spot,strike",
    [
        (90.0, 100.0),
        (110.0, 100.0),
    ],
)
@pytest.mark.parametrize(
    "r_curve",
    [
        flat_curve(PRICING_DATE, MATURITY, _ME_RATE),
        _nonflat_r_curve(),
    ],
    ids=["flat", "nonflat"],
)
def test_discrete_div_american_vs_quantlib(spot, strike, r_curve):
    """American discrete dividend: PDE, MC, and QuantLib all align."""
    divs = [
        (PRICING_DATE + dt.timedelta(days=120), 0.6),
        (PRICING_DATE + dt.timedelta(days=240), 0.6),
    ]
    ud = _underlying(spot=spot, r_curve=r_curve, discrete_dividends=divs)
    gbm = _gbm(spot=spot, r_curve=r_curve, discrete_dividends=divs, paths=60_000)
    spec = _spec(strike=strike, option_type=OptionType.PUT, exercise_type=ExerciseType.AMERICAN)

    pde_pv = _dp_price(ud, spec, PricingMethod.PDE_FD, _ME_PDE)
    mc_pv = _dp_price(gbm, spec, PricingMethod.MONTE_CARLO, _ME_MC_AM)

    # QuantLib FD with discrete dividends
    rf_h = _ql_curve_from_times(times=r_curve.times, dfs=r_curve.dfs)
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, ql.Actual365Fixed()))
    ql_pv = _ql_price(
        spot=spot,
        strike=strike,
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.AMERICAN,
        rf_handle=rf_h,
        div_handle=div_h,
        discrete_dividends=divs,
    )

    logger.info(
        "Disc-div AM PUT S=%.0f K=%.0f | PDE=%.6f MC=%.6f QL=%.6f",
        spot,
        strike,
        pde_pv,
        mc_pv,
        ql_pv,
    )

    assert np.isclose(pde_pv, mc_pv, rtol=0.015, atol=1e-3)
    assert np.isclose(pde_pv, ql_pv, rtol=5e-3, atol=1e-4), f"PDE {pde_pv:.6f} vs QL {ql_pv:.6f}"
    assert np.isclose(mc_pv, ql_pv, rtol=0.015, atol=1e-3), f"MC {mc_pv:.6f} vs QL {ql_pv:.6f}"


@pytest.mark.parametrize(
    "spot,strike,option_type",
    [
        (52.0, 50.0, OptionType.PUT),
        (52.0, 50.0, OptionType.CALL),
    ],
)
def test_american_fd_vs_quantlib_nonflat_rate_with_discrete_divs(spot, strike, option_type):
    r_times = np.array([0.0, 0.25, 0.5, 1.0])
    r_forwards = np.array([0.03, 0.05, 0.04])
    r_curve = DiscountCurve.from_forwards(times=r_times, forwards=r_forwards)

    q_times = np.array([0.0, 1.0])
    q_dfs = np.array([1.0, 1.0])

    divs = [
        (PRICING_DATE + dt.timedelta(days=90), 0.5),
        (PRICING_DATE + dt.timedelta(days=270), 0.5),
    ]

    pde = _pde_fd_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_curve=r_curve,
        discrete_dividends=divs,
    )
    ql_price = _quantlib_american_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_times=r_curve.times,
        r_dfs=r_curve.dfs,
        q_times=q_times,
        q_dfs=q_dfs,
        discrete_dividends=divs,
    )

    logger.info(
        "Forward-curve + discrete-div American %s S=%.2f K=%.2f | PDE=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        pde,
        ql_price,
    )

    assert np.isclose(pde, ql_price, rtol=5e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Helpers for boundary-dividend comparison tests
# ---------------------------------------------------------------------------


def _quantlib_european(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    """QuantLib European price using AnalyticDividendEuropeanEngine (Merton approach)."""
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)

    day_count = _ql_dcc(dcc)
    rf_curve = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, float(RISK_FREE), day_count))
    div_curve = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, day_count))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, ql.TARGET(), float(VOL), day_count)
    )

    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(ql_type, float(strike))
    exercise = ql.EuropeanExercise(ql_maturity)
    option = ql.VanillaOption(payoff, exercise)

    dividend_schedule = _quantlib_dividend_schedule(discrete_dividends)
    process = ql.BlackScholesMertonProcess(spot_handle, div_curve, rf_curve, vol_handle)
    engine = ql.AnalyticDividendEuropeanEngine(process, dividend_schedule)
    option.setPricingEngine(engine)
    return float(option.NPV())


def _bsm_european(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    ud = UnderlyingData(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(dcc=dcc),
        discrete_dividends=discrete_dividends,
    )
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.EUROPEAN)
    return OptionValuation(ud, spec, PricingMethod.BSM).present_value()


def _binomial_european(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    ud = UnderlyingData(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(dcc=dcc),
        discrete_dividends=discrete_dividends,
    )
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.EUROPEAN)
    return OptionValuation(ud, spec, PricingMethod.BINOMIAL, BINOM_CFG).present_value()


def _mc_american(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    md = _market_data(dcc=dcc)
    gbm_params = GBMParams(
        initial_value=spot,
        volatility=VOL,
        discrete_dividends=discrete_dividends,
    )
    sim_config = SimulationConfig(
        paths=100_000,
        end_date=MATURITY,
        num_steps=200,
    )
    gbm = GBMProcess(md, gbm_params, sim_config)
    spec = _spec(strike=strike, option_type=option_type)
    return _dp_price(gbm, spec, PricingMethod.MONTE_CARLO, MC_CFG)


# ---------------------------------------------------------------------------
# Boundary-dividend tests: discrete dividends on pricing date / maturity / both
# ---------------------------------------------------------------------------

_BOUNDARY_DIV_CASES = [
    pytest.param(
        [
            (PRICING_DATE + dt.timedelta(days=90), 0.75),
            (PRICING_DATE + dt.timedelta(days=270), 0.75),
        ],
        id="interior",
    ),
    pytest.param(
        [
            (PRICING_DATE, 0.75),
            (PRICING_DATE + dt.timedelta(days=180), 0.75),
        ],
        id="on_pricing_date",
    ),
    pytest.param(
        [
            (PRICING_DATE + dt.timedelta(days=180), 0.75),
            (MATURITY, 0.75),
        ],
        id="on_maturity",
    ),
    pytest.param(
        [
            (PRICING_DATE, 0.75),
            (MATURITY, 0.75),
        ],
        id="on_both_boundaries",
    ),
]


@pytest.mark.parametrize("option_type", [OptionType.PUT, OptionType.CALL])
@pytest.mark.parametrize("divs", _BOUNDARY_DIV_CASES)
def test_european_discrete_div_boundary_vs_quantlib(divs, option_type):
    """BSM and Binomial European prices match QuantLib with boundary dividends."""
    spot, strike = 52.0, 50.0

    ql_price = _quantlib_european(
        spot=spot,
        strike=strike,
        option_type=option_type,
        discrete_dividends=divs,
    )
    bsm_price = _bsm_european(
        spot=spot,
        strike=strike,
        option_type=option_type,
        discrete_dividends=divs,
    )
    binom_price = _binomial_european(
        spot=spot,
        strike=strike,
        option_type=option_type,
        discrete_dividends=divs,
    )

    logger.info(
        "European %s | QL=%.6f BSM=%.6f Binom=%.6f",
        option_type.value,
        ql_price,
        bsm_price,
        binom_price,
    )

    assert np.isclose(bsm_price, ql_price, rtol=1e-4), f"BSM {bsm_price:.6f} vs QL {ql_price:.6f}"
    assert np.isclose(binom_price, ql_price, rtol=0.01), (
        f"Binomial {binom_price:.6f} vs QL {ql_price:.6f}"
    )


@pytest.mark.parametrize("option_type", [OptionType.PUT, OptionType.CALL])
@pytest.mark.parametrize("divs", _BOUNDARY_DIV_CASES)
def test_american_discrete_div_boundary_vs_quantlib(divs, option_type):
    """PDE and MC American prices match QuantLib with boundary dividends."""
    spot, strike = 52.0, 50.0

    ql_price = _quantlib_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        dividend_yield=0.0,
        discrete_dividends=divs,
    )
    pde_price = _pde_fd_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        discrete_dividends=divs,
    )
    mc_price = _mc_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        discrete_dividends=divs,
    )

    logger.info(
        "American %s | QL=%.6f PDE=%.6f MC=%.6f",
        option_type.value,
        ql_price,
        pde_price,
        mc_price,
    )

    assert np.isclose(pde_price, ql_price, rtol=0.01), f"PDE {pde_price:.6f} vs QL {ql_price:.6f}"
    assert np.isclose(mc_price, ql_price, rtol=0.015), f"MC {mc_price:.6f} vs QL {ql_price:.6f}"


# ── European with forward curves — BSM / PDE / Binomial / MC / QL ─────


@pytest.mark.parametrize(
    "spot,strike,option_type,r_times,r_forwards,q_times,q_forwards",
    [
        (
            52.0,
            50.0,
            OptionType.PUT,
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.03, 0.05, 0.04]),
            np.array([0.0, 1.0]),
            np.array([0.0]),
        ),
        (
            60.0,
            55.0,
            OptionType.CALL,
            np.array([0.0, 1.0]),
            np.array([0.04]),
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.00, 0.02, 0.04]),
        ),
        (
            52.0,
            50.0,
            OptionType.PUT,
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.03, 0.05, 0.04]),
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.01, 0.02, 0.00]),
        ),
    ],
)
def test_european_forward_curves_vs_quantlib(
    spot,
    strike,
    option_type,
    r_times,
    r_forwards,
    q_times,
    q_forwards,
):
    """European BSM, PDE, Binomial, MC, and QuantLib agree under forward curves."""
    r_curve = DiscountCurve.from_forwards(times=r_times, forwards=r_forwards)
    q_curve = DiscountCurve.from_forwards(times=q_times, forwards=q_forwards)

    ud = _underlying(spot=spot, r_curve=r_curve, dividend_curve=q_curve)
    gbm = _gbm(spot=spot, r_curve=r_curve, dividend_curve=q_curve, paths=150_000)
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.EUROPEAN)

    bsm_pv = _dp_price(ud, spec, PricingMethod.BSM)
    pde_pv = _dp_price(ud, spec, PricingMethod.PDE_FD, _ME_PDE)
    binom_pv = _dp_price(ud, spec, PricingMethod.BINOMIAL, _ME_BINOM)
    mc_pv = _dp_price(gbm, spec, PricingMethod.MONTE_CARLO, _ME_MC_EU)

    # QuantLib analytical European with forward curves
    rf_h = _ql_curve_from_times(times=r_curve.times, dfs=r_curve.dfs)
    div_h = _ql_curve_from_times(times=q_curve.times, dfs=q_curve.dfs)
    ql_pv = _ql_price(
        spot=spot,
        strike=strike,
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        rf_handle=rf_h,
        div_handle=div_h,
    )

    logger.info(
        "Forward-curve EU %s S=%.0f K=%.0f | BSM=%.6f PDE=%.6f Binom=%.6f MC=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        bsm_pv,
        pde_pv,
        binom_pv,
        mc_pv,
        ql_pv,
    )

    assert np.isclose(bsm_pv, ql_pv, rtol=1e-4), f"BSM {bsm_pv:.6f} vs QL {ql_pv:.6f}"
    assert np.isclose(pde_pv, ql_pv, rtol=0.01), f"PDE {pde_pv:.6f} vs QL {ql_pv:.6f}"
    assert np.isclose(binom_pv, ql_pv, rtol=0.01), f"Binom {binom_pv:.6f} vs QL {ql_pv:.6f}"
    assert np.isclose(mc_pv, ql_pv, rtol=0.01), f"MC {mc_pv:.6f} vs QL {ql_pv:.6f}"


# ── American with forward curves — PDE / Binomial / MC / QL ───────────


@pytest.mark.parametrize(
    "spot,strike,option_type,r_times,r_forwards,q_times,q_forwards",
    [
        (
            52.0,
            50.0,
            OptionType.PUT,
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.03, 0.05, 0.04]),
            np.array([0.0, 1.0]),
            np.array([0.0]),
        ),
        (
            60.0,
            55.0,
            OptionType.CALL,
            np.array([0.0, 1.0]),
            np.array([0.04]),
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.00, 0.02, 0.04]),
        ),
        (
            52.0,
            50.0,
            OptionType.PUT,
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.03, 0.05, 0.04]),
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.01, 0.02, 0.00]),
        ),
    ],
)
def test_american_forward_curves_vs_quantlib(
    spot,
    strike,
    option_type,
    r_times,
    r_forwards,
    q_times,
    q_forwards,
):
    """American PDE, Binomial, MC, and QuantLib agree under forward curves."""
    r_curve = DiscountCurve.from_forwards(times=r_times, forwards=r_forwards)
    q_curve = DiscountCurve.from_forwards(times=q_times, forwards=q_forwards)

    ud = _underlying(spot=spot, r_curve=r_curve, dividend_curve=q_curve)
    gbm = _gbm(spot=spot, r_curve=r_curve, dividend_curve=q_curve, paths=150_000)
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.AMERICAN)

    pde_pv = _dp_price(ud, spec, PricingMethod.PDE_FD, _ME_PDE)
    binom_pv = _dp_price(ud, spec, PricingMethod.BINOMIAL, _ME_BINOM)
    mc_pv = _dp_price(gbm, spec, PricingMethod.MONTE_CARLO, _ME_MC_AM)

    # QuantLib FD American with forward curves
    rf_h = _ql_curve_from_times(times=r_curve.times, dfs=r_curve.dfs)
    div_h = _ql_curve_from_times(times=q_curve.times, dfs=q_curve.dfs)
    ql_pv = _ql_price(
        spot=spot,
        strike=strike,
        option_type=option_type,
        exercise_type=ExerciseType.AMERICAN,
        rf_handle=rf_h,
        div_handle=div_h,
    )

    logger.info(
        "Forward-curve AM %s S=%.0f K=%.0f | PDE=%.6f Binom=%.6f MC=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        pde_pv,
        binom_pv,
        mc_pv,
        ql_pv,
    )

    assert np.isclose(pde_pv, ql_pv, rtol=0.01), f"PDE {pde_pv:.6f} vs QL {ql_pv:.6f}"
    assert np.isclose(binom_pv, ql_pv, rtol=0.01), f"Binom {binom_pv:.6f} vs QL {ql_pv:.6f}"
    assert np.isclose(mc_pv, ql_pv, rtol=0.01), f"MC {mc_pv:.6f} vs QL {ql_pv:.6f}"


# ═══════════════════════════════════════════════════════════════════════════
# Asian option comparison — discrete fixing dates vs QuantLib
# ═══════════════════════════════════════════════════════════════════════════

# Shared Asian constants
_ASIAN_SPOT = 100.0
_ASIAN_STRIKE = 100.0
_ASIAN_VOL = 0.20
_ASIAN_RATE = 0.05
_ASIAN_MATURITY = PRICING_DATE + dt.timedelta(days=365)
_ASIAN_MC_PATHS = 500_000
_ASIAN_MC_SEED = 42
_ASIAN_NUM_STEPS = 60  # dense grid for MC simulation


def _dt_to_ql(d: dt.datetime) -> "ql_typing.Date":
    return ql.Date(d.day, d.month, d.year)


def _ql_asian_process(
    *,
    spot: float = _ASIAN_SPOT,
    vol: float = _ASIAN_VOL,
    rf_rate: float | None = None,
    rf_handle: "ql_typing.YieldTermStructureHandle | None" = None,
    div_rate: float = 0.0,
    div_handle: "ql_typing.YieldTermStructureHandle | None" = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> "ql_typing.BlackScholesMertonProcess":
    eval_date = _dt_to_ql(PRICING_DATE)
    ql.Settings.instance().evaluationDate = eval_date
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    ql_dc = _ql_dcc(dcc)
    if rf_handle is None:
        rf_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date, rf_rate if rf_rate is not None else _ASIAN_RATE, ql_dc)
        )
    if div_handle is None:
        div_handle = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, div_rate, ql_dc))
    vol_h = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, ql.TARGET(), vol, ql_dc))
    return ql.BlackScholesMertonProcess(spot_h, div_handle, rf_handle, vol_h)


def _ql_discrete_asian_price(
    *,
    fixing_dates_ql: list,
    option_type_ql: int,
    averaging_ql,
    engine_factory,
    strike: float = _ASIAN_STRIKE,
    spot: float = _ASIAN_SPOT,
    vol: float = _ASIAN_VOL,
    rf_rate: float | None = None,
    rf_handle: "ql_typing.YieldTermStructureHandle | None" = None,
    div_rate: float = 0.0,
    div_handle: "ql_typing.YieldTermStructureHandle | None" = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    process = _ql_asian_process(
        spot=spot,
        vol=vol,
        rf_rate=rf_rate,
        rf_handle=rf_handle,
        div_rate=div_rate,
        div_handle=div_handle,
        dcc=dcc,
    )
    payoff = ql.PlainVanillaPayoff(option_type_ql, strike)
    exercise = ql.EuropeanExercise(_dt_to_ql(_ASIAN_MATURITY))
    opt = ql.DiscreteAveragingAsianOption(averaging_ql, fixing_dates_ql, payoff, exercise)
    opt.setPricingEngine(engine_factory(process))
    return opt.NPV()


def _dp_asian_market_data(
    r_curve: DiscountCurve | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> MarketData:
    ttm = calculate_year_fraction(PRICING_DATE, _ASIAN_MATURITY, dcc)
    curve = r_curve if r_curve is not None else DiscountCurve.flat(_ASIAN_RATE, end_time=ttm)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY, day_count_convention=dcc)


def _dp_asian_gbm(
    *,
    fixing_dates: tuple[dt.datetime, ...],
    spot: float = _ASIAN_SPOT,
    vol: float = _ASIAN_VOL,
    r_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> GBMProcess:
    md = _dp_asian_market_data(r_curve, dcc=dcc)
    params = GBMParams(initial_value=spot, volatility=vol, dividend_curve=dividend_curve)
    sim_cfg = SimulationConfig(
        paths=_ASIAN_MC_PATHS,
        end_date=_ASIAN_MATURITY,
        num_steps=_ASIAN_NUM_STEPS,
    )
    return GBMProcess(md, params, sim_cfg)


# ── Test 1: Monthly fixings — arithmetic call ───────────────────────────

_MONTHLY_FIXINGS = tuple(
    dt.datetime(2025, m, 1) if m <= 12 else dt.datetime(2026, m - 12, 1) for m in range(2, 14)
)


def test_asian_arithmetic_call_monthly_vs_quantlib():
    """Monthly-fixing arithmetic Asian call: our MC vs QuantLib TW analytic."""

    ql_fixings = [_dt_to_ql(d) for d in _MONTHLY_FIXINGS]

    ql_tw = _ql_discrete_asian_price(
        fixing_dates_ql=ql_fixings,
        option_type_ql=ql.Option.Call,
        averaging_ql=ql.Average.Arithmetic,
        engine_factory=lambda p: ql.TurnbullWakemanAsianEngine(p),
        dcc=DayCountConvention.ACT_360,
    )

    spec = AsianSpec(
        averaging=AsianAveraging.ARITHMETIC,
        option_type=OptionType.CALL,
        strike=_ASIAN_STRIKE,
        maturity=_ASIAN_MATURITY,
        currency=CURRENCY,
        fixing_dates=_MONTHLY_FIXINGS,
        exercise_type=ExerciseType.EUROPEAN,
    )
    gbm = _dp_asian_gbm(fixing_dates=_MONTHLY_FIXINGS, dcc=DayCountConvention.ACT_360)
    dp_mc = OptionValuation(
        gbm,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=_ASIAN_MC_SEED),
    ).present_value()
    dp_analytical = OptionValuation(
        as_underlying_data(gbm), spec, PricingMethod.BSM
    ).present_value()

    logger.info(
        "Asian arith call monthly | QL_TW=%.6f DP_MC=%.6f DP_AN=%.6f",
        ql_tw,
        dp_mc,
        dp_analytical,
    )
    assert np.isclose(dp_mc, ql_tw, rtol=0.015), f"DP_MC {dp_mc:.6f} vs QL {ql_tw:.6f}"
    assert np.isclose(dp_analytical, ql_tw, rtol=0.005), (
        f"DP_AN {dp_analytical:.6f} vs QL {ql_tw:.6f}"
    )


# ── Test 2: Quarterly fixings — geometric put ───────────────────────────

_QUARTERLY_FIXINGS = (
    dt.datetime(2025, 4, 1),
    dt.datetime(2025, 7, 1),
    dt.datetime(2025, 10, 1),
    dt.datetime(2026, 1, 1),
)


def test_asian_geometric_put_quarterly_vs_quantlib():
    """Quarterly-fixing geometric Asian put: our MC vs QuantLib analytic."""
    ql_fixings = [_dt_to_ql(d) for d in _QUARTERLY_FIXINGS]

    ql_analytic = _ql_discrete_asian_price(
        fixing_dates_ql=ql_fixings,
        option_type_ql=ql.Option.Put,
        averaging_ql=ql.Average.Geometric,
        engine_factory=lambda p: ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(p),
        dcc=DayCountConvention.ACT_365F,
    )

    spec = AsianSpec(
        averaging=AsianAveraging.GEOMETRIC,
        option_type=OptionType.PUT,
        strike=_ASIAN_STRIKE,
        maturity=_ASIAN_MATURITY,
        currency=CURRENCY,
        fixing_dates=_QUARTERLY_FIXINGS,
        exercise_type=ExerciseType.EUROPEAN,
    )
    gbm = _dp_asian_gbm(fixing_dates=_QUARTERLY_FIXINGS, dcc=DayCountConvention.ACT_365F)
    dp_mc = OptionValuation(
        gbm,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=_ASIAN_MC_SEED),
    ).present_value()
    dp_analytical = OptionValuation(
        as_underlying_data(gbm), spec, PricingMethod.BSM
    ).present_value()

    logger.info(
        "Asian geom put quarterly | QL_analytic=%.6f DP_MC=%.6f DP_AN=%.6f",
        ql_analytic,
        dp_mc,
        dp_analytical,
    )
    assert np.isclose(dp_mc, ql_analytic, rtol=0.015), f"DP_MC {dp_mc:.6f} vs QL {ql_analytic:.6f}"
    assert np.isclose(dp_analytical, ql_analytic, rtol=0.005), (
        f"DP_AN {dp_analytical:.6f} vs QL {ql_analytic:.6f}"
    )


# ── Test 3: Non-flat rate and dividend curves — arithmetic put ──────────

_BIMONTHLY_FIXINGS = tuple(
    dt.datetime(2025, m, 15) for m in range(2, 13, 2)
)  # Feb, Apr, Jun, Aug, Oct, Dec — 6 dates


def _nonflat_asian_curves() -> tuple[DiscountCurve, DiscountCurve]:
    """Build the non-flat rate and dividend curves for Asian non-flat tests."""
    ttm = calculate_year_fraction(PRICING_DATE, _ASIAN_MATURITY)
    r_times = np.array([0.0, 0.25, 0.5, ttm])
    r_forwards = np.array([0.03, 0.05, 0.04])
    q_times = np.array([0.0, 0.25, 0.5, ttm])
    q_forwards = np.array([0.01, 0.02, 0.00])
    return (
        DiscountCurve.from_forwards(times=r_times, forwards=r_forwards),
        DiscountCurve.from_forwards(times=q_times, forwards=q_forwards),
    )


def test_asian_arithmetic_put_nonflat_curves_vs_quantlib():
    """Arithmetic Asian put with non-flat rate/div curves: our MC vs QuantLib TW."""
    r_curve, q_curve = _nonflat_asian_curves()

    rf_handle = _ql_curve_from_times(times=r_curve.times, dfs=r_curve.dfs)
    div_handle = _ql_curve_from_times(times=q_curve.times, dfs=q_curve.dfs)

    ql_fixings = [_dt_to_ql(d) for d in _BIMONTHLY_FIXINGS]

    ql_tw = _ql_discrete_asian_price(
        fixing_dates_ql=ql_fixings,
        option_type_ql=ql.Option.Put,
        averaging_ql=ql.Average.Arithmetic,
        engine_factory=lambda p: ql.TurnbullWakemanAsianEngine(p),
        rf_handle=rf_handle,
        div_handle=div_handle,
    )

    spec = AsianSpec(
        averaging=AsianAveraging.ARITHMETIC,
        option_type=OptionType.PUT,
        strike=_ASIAN_STRIKE,
        maturity=_ASIAN_MATURITY,
        currency=CURRENCY,
        fixing_dates=_BIMONTHLY_FIXINGS,
        exercise_type=ExerciseType.EUROPEAN,
    )
    gbm = _dp_asian_gbm(
        fixing_dates=_BIMONTHLY_FIXINGS,
        r_curve=r_curve,
        dividend_curve=q_curve,
    )
    dp_mc = OptionValuation(
        gbm,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=_ASIAN_MC_SEED),
    ).present_value()
    dp_analytical = OptionValuation(
        as_underlying_data(gbm), spec, PricingMethod.BSM
    ).present_value()

    logger.info(
        "Asian arith put nonflat curves | QL_TW=%.6f DP_MC=%.6f DP_AN=%.6f",
        ql_tw,
        dp_mc,
        dp_analytical,
    )
    assert np.isclose(dp_mc, ql_tw, rtol=0.015), f"DP_MC {dp_mc:.6f} vs QL {ql_tw:.6f}"
    assert np.isclose(dp_analytical, ql_tw, rtol=0.005), (
        f"DP_AN {dp_analytical:.6f} vs QL {ql_tw:.6f}"
    )


def test_asian_geometric_put_nonflat_curves_vs_quantlib():
    """Geometric Asian put with non-flat rate/div curves: our MC vs QuantLib MC.

    QL's AnalyticDiscreteGeometricAveragePriceAsianEngine appears to single
    flat-equivalent rates internally, so prices diverge with non-flat curves.
    Benchmark against QL's own MC engine instead.
    """
    r_curve, q_curve = _nonflat_asian_curves()

    rf_handle = _ql_curve_from_times(times=r_curve.times, dfs=r_curve.dfs)
    div_handle = _ql_curve_from_times(times=q_curve.times, dfs=q_curve.dfs)

    ql_fixings = [_dt_to_ql(d) for d in _BIMONTHLY_FIXINGS]

    ql_mc = _ql_discrete_asian_price(
        fixing_dates_ql=ql_fixings,
        option_type_ql=ql.Option.Put,
        averaging_ql=ql.Average.Geometric,
        engine_factory=lambda p: ql.MCDiscreteGeometricAPEngine(
            p, "pseudorandom", requiredSamples=500_000, seed=42
        ),
        rf_handle=rf_handle,
        div_handle=div_handle,
    )

    spec = AsianSpec(
        averaging=AsianAveraging.GEOMETRIC,
        option_type=OptionType.PUT,
        strike=_ASIAN_STRIKE,
        maturity=_ASIAN_MATURITY,
        currency=CURRENCY,
        fixing_dates=_BIMONTHLY_FIXINGS,
        exercise_type=ExerciseType.EUROPEAN,
    )
    gbm = _dp_asian_gbm(
        fixing_dates=_BIMONTHLY_FIXINGS,
        r_curve=r_curve,
        dividend_curve=q_curve,
    )
    dp_mc = OptionValuation(
        gbm,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=_ASIAN_MC_SEED),
    ).present_value()
    dp_analytical = OptionValuation(
        as_underlying_data(gbm), spec, PricingMethod.BSM
    ).present_value()

    logger.info(
        "Asian geom put nonflat curves | QL_MC=%.6f DP_MC=%.6f DP_AN=%.6f",
        ql_mc,
        dp_mc,
        dp_analytical,
    )
    assert np.isclose(dp_mc, ql_mc, rtol=0.015), f"DP_MC {dp_mc:.6f} vs QL {ql_mc:.6f}"
    # QL benchmark here is MC (500K paths), so use wider tolerance
    assert np.isclose(dp_analytical, ql_mc, rtol=0.015), (
        f"DP_AN {dp_analytical:.6f} vs QL_MC {ql_mc:.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Barrier options — DP analytical vs QuantLib analytical
# ═══════════════════════════════════════════════════════════════════════════

_BARRIER_SPOT = 100.0
_BARRIER_VOL = 0.25
_BARRIER_RATE = 0.05
_BARRIER_DIV = 0.02
_BARRIER_MATURITY = dt.datetime(2026, 1, 1)

_QL_BARRIER_TYPE = {
    (BarrierDirection.DOWN, BarrierAction.IN): ql.Barrier.DownIn,
    (BarrierDirection.DOWN, BarrierAction.OUT): ql.Barrier.DownOut,
    (BarrierDirection.UP, BarrierAction.IN): ql.Barrier.UpIn,
    (BarrierDirection.UP, BarrierAction.OUT): ql.Barrier.UpOut,
}


def _barrier_nonflat_curves() -> tuple[DiscountCurve, DiscountCurve]:
    """Non-flat rate and dividend curves for barrier tests."""
    ttm = calculate_year_fraction(PRICING_DATE, _BARRIER_MATURITY)
    r_times = np.array([0.0, 0.25, 0.5, ttm])
    r_forwards = np.array([0.03, 0.06, 0.04])
    q_times = np.array([0.0, 0.25, 0.5, ttm])
    q_forwards = np.array([0.01, 0.03, 0.005])
    return (
        DiscountCurve.from_forwards(times=r_times, forwards=r_forwards),
        DiscountCurve.from_forwards(times=q_times, forwards=q_forwards),
    )


def _ql_barrier_price(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    rebate: float = 0.0,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
) -> float:
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    ql_dc = ql.Actual365Fixed()

    spot_h = ql.QuoteHandle(ql.SimpleQuote(_BARRIER_SPOT))

    if r_curve is not None:
        rf_h = _ql_curve_from_times(times=r_curve.times, dfs=r_curve.dfs)
    else:
        rf_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, _BARRIER_RATE, ql_dc))

    if q_curve is not None:
        div_h = _ql_curve_from_times(times=q_curve.times, dfs=q_curve.dfs)
    else:
        div_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, _BARRIER_DIV, ql_dc))

    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, ql.TARGET(), _BARRIER_VOL, ql_dc)
    )
    proc = ql.BlackScholesMertonProcess(spot_h, div_h, rf_h, vol_h)

    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    exercise = ql.EuropeanExercise(
        ql.Date(_BARRIER_MATURITY.day, _BARRIER_MATURITY.month, _BARRIER_MATURITY.year)
    )
    bt = _QL_BARRIER_TYPE[(direction, action)]
    opt = ql.BarrierOption(bt, barrier, rebate, payoff, exercise)
    opt.setPricingEngine(ql.AnalyticBarrierEngine(proc))
    return opt.NPV()


def _ql_barrier_binomial_price(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    rebate: float = 0.0,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
    binom_steps: int = 400,
) -> float:
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    ql_dc = ql.Actual365Fixed()

    spot_h = ql.QuoteHandle(ql.SimpleQuote(_BARRIER_SPOT))

    if r_curve is not None:
        rf_h = _ql_curve_from_times(times=r_curve.times, dfs=r_curve.dfs)
    else:
        rf_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, _BARRIER_RATE, ql_dc))

    if q_curve is not None:
        div_h = _ql_curve_from_times(times=q_curve.times, dfs=q_curve.dfs)
    else:
        div_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, _BARRIER_DIV, ql_dc))

    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, ql.TARGET(), _BARRIER_VOL, ql_dc)
    )
    proc = ql.BlackScholesMertonProcess(spot_h, div_h, rf_h, vol_h)

    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    ql_maturity = ql.Date(_BARRIER_MATURITY.day, _BARRIER_MATURITY.month, _BARRIER_MATURITY.year)
    if exercise_type is ExerciseType.EUROPEAN:
        exercise = ql.EuropeanExercise(ql_maturity)
    else:
        exercise = ql.AmericanExercise(eval_date, ql_maturity)
    bt = _QL_BARRIER_TYPE[(direction, action)]
    opt = ql.BarrierOption(bt, barrier, rebate, payoff, exercise)
    opt.setPricingEngine(ql.BinomialCRRBarrierEngine(proc, binom_steps))
    return opt.NPV()


def _ql_barrier_fd_price(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    rebate: float = 0.0,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
    grid_points: int = 800,
    time_steps: int = 800,
) -> float:
    """QL FdBlackScholesBarrierEngine — uses time-varying rates like our PDE."""
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    ql_dc = ql.Actual365Fixed()

    spot_h = ql.QuoteHandle(ql.SimpleQuote(_BARRIER_SPOT))

    if r_curve is not None:
        rf_h = _ql_curve_from_times(times=r_curve.times, dfs=r_curve.dfs)
    else:
        rf_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, _BARRIER_RATE, ql_dc))

    if q_curve is not None:
        div_h = _ql_curve_from_times(times=q_curve.times, dfs=q_curve.dfs)
    else:
        div_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, _BARRIER_DIV, ql_dc))

    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, ql.TARGET(), _BARRIER_VOL, ql_dc)
    )
    proc = ql.BlackScholesMertonProcess(spot_h, div_h, rf_h, vol_h)

    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    exercise = ql.EuropeanExercise(
        ql.Date(_BARRIER_MATURITY.day, _BARRIER_MATURITY.month, _BARRIER_MATURITY.year)
    )
    bt = _QL_BARRIER_TYPE[(direction, action)]
    opt = ql.BarrierOption(bt, barrier, rebate, payoff, exercise)
    opt.setPricingEngine(ql.FdBlackScholesBarrierEngine(proc, time_steps, grid_points))
    return opt.NPV()


def _dp_barrier_analytical_price(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    rebate: float = 0.0,
    rebate_timing: RebateTiming = RebateTiming.AT_HIT,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
) -> float:
    ud = _barrier_underlying_data(r_curve=r_curve, q_curve=q_curve)
    spec = _barrier_spec(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.EUROPEAN,
        rebate=rebate,
        rebate_timing=rebate_timing,
    )
    return _dp_price(ud, spec, PricingMethod.BSM)


_BARRIER_BINOM_CFG = BinomialParams(num_steps=1000)


def _barrier_resolved_curves(
    *,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
) -> tuple[DiscountCurve, DiscountCurve]:
    ttm = calculate_year_fraction(PRICING_DATE, _BARRIER_MATURITY)
    rc = r_curve if r_curve is not None else DiscountCurve.flat(_BARRIER_RATE, end_time=ttm)
    qc = q_curve if q_curve is not None else DiscountCurve.flat(_BARRIER_DIV, end_time=ttm)
    return rc, qc


def _barrier_underlying_data(
    *,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
) -> UnderlyingData:
    rc, qc = _barrier_resolved_curves(r_curve=r_curve, q_curve=q_curve)
    md = MarketData(PRICING_DATE, rc, currency=CURRENCY)
    return UnderlyingData(
        initial_value=_BARRIER_SPOT,
        volatility=_BARRIER_VOL,
        market_data=md,
        dividend_curve=qc,
    )


def _barrier_gbm(
    *,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
) -> GBMProcess:
    rc, qc = _barrier_resolved_curves(r_curve=r_curve, q_curve=q_curve)
    md = MarketData(PRICING_DATE, rc, currency=CURRENCY)
    return GBMProcess(
        md,
        GBMParams(initial_value=_BARRIER_SPOT, volatility=_BARRIER_VOL, dividend_curve=qc),
        SimulationConfig(
            paths=_BARRIER_MC_PATHS, end_date=_BARRIER_MATURITY, num_steps=_BARRIER_MC_STEPS
        ),
    )


def _barrier_spec(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    rebate: float = 0.0,
    rebate_timing: RebateTiming = RebateTiming.AT_HIT,
    monitoring: BarrierMonitoring = BarrierMonitoring.CONTINUOUS,
) -> BarrierSpec:
    return BarrierSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=_BARRIER_MATURITY,
        barrier=barrier,
        direction=direction,
        action=action,
        monitoring=monitoring,
        rebate=rebate,
        rebate_timing=rebate_timing,
    )


def _dp_barrier_pde_price(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    rebate: float = 0.0,
    rebate_timing: RebateTiming = RebateTiming.AT_HIT,
    monitoring: BarrierMonitoring = BarrierMonitoring.CONTINUOUS,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
) -> float:
    ud = _barrier_underlying_data(r_curve=r_curve, q_curve=q_curve)
    spec = _barrier_spec(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=exercise_type,
        rebate=rebate,
        rebate_timing=rebate_timing,
        monitoring=monitoring,
    )
    return _dp_price(ud, spec, PricingMethod.PDE_FD)


def _dp_barrier_binomial_price(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    rebate: float = 0.0,
    rebate_timing: RebateTiming = RebateTiming.AT_HIT,
    monitoring: BarrierMonitoring = BarrierMonitoring.CONTINUOUS,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
) -> float:
    ud = _barrier_underlying_data(r_curve=r_curve, q_curve=q_curve)
    spec = _barrier_spec(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=exercise_type,
        rebate=rebate,
        rebate_timing=rebate_timing,
        monitoring=monitoring,
    )
    return _dp_price(ud, spec, PricingMethod.BINOMIAL)


_BARRIER_MC_PATHS = 150_000
_BARRIER_MC_STEPS = 200
_BARRIER_MC_CFG = MonteCarloParams(random_seed=42, deg=3, barrier_aware_basis=True)


def _dp_barrier_mc_price(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    rebate: float = 0.0,
    rebate_timing: RebateTiming = RebateTiming.AT_HIT,
    r_curve: DiscountCurve | None = None,
    q_curve: DiscountCurve | None = None,
) -> float:
    gbm = _barrier_gbm(r_curve=r_curve, q_curve=q_curve)
    spec = _barrier_spec(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=exercise_type,
        rebate=rebate,
        rebate_timing=rebate_timing,
        monitoring=BarrierMonitoring.CONTINUOUS,
    )
    return _dp_price(gbm, spec, PricingMethod.MONTE_CARLO, _BARRIER_MC_CFG)


def _ql_barrier_american_price(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    rebate: float = 0.0,
    binom_steps: int = _BARRIER_BINOM_CFG.num_steps,
) -> float:
    """American barrier via QL BinomialCRRBarrierEngine."""
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    ql_dc = ql.Actual365Fixed()

    spot_h = ql.QuoteHandle(ql.SimpleQuote(_BARRIER_SPOT))
    rf_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, _BARRIER_RATE, ql_dc))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, _BARRIER_DIV, ql_dc))
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, ql.TARGET(), _BARRIER_VOL, ql_dc)
    )
    proc = ql.BlackScholesMertonProcess(spot_h, div_h, rf_h, vol_h)

    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    ql_maturity = ql.Date(_BARRIER_MATURITY.day, _BARRIER_MATURITY.month, _BARRIER_MATURITY.year)
    exercise = ql.AmericanExercise(eval_date, ql_maturity)
    bt = _QL_BARRIER_TYPE[(direction, action)]
    opt = ql.BarrierOption(bt, barrier, rebate, payoff, exercise)
    opt.setPricingEngine(ql.BinomialCRRBarrierEngine(proc, binom_steps))
    return opt.NPV()


# 8 scenarios covering all 4 barrier types × call/put, mixed flat/non-flat curves
_BARRIER_SCENARIOS = [
    # Flat curves
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        0.0,
        None,
        None,
        id="down_out_call_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.CALL,
        105.0,
        115.0,
        0.0,
        None,
        None,
        id="up_in_call_flat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        85.0,
        0.0,
        None,
        None,
        id="down_in_put_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        95.0,
        125.0,
        0.0,
        None,
        None,
        id="up_out_put_flat",
    ),
    # Non-flat curves
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.PUT,
        105.0,
        90.0,
        0.0,
        *_barrier_nonflat_curves(),
        id="down_out_put_nonflat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        120.0,
        0.0,
        *_barrier_nonflat_curves(),
        id="up_out_call_nonflat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.CALL,
        95.0,
        80.0,
        0.0,
        *_barrier_nonflat_curves(),
        id="down_in_call_nonflat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        110.0,
        0.0,
        *_barrier_nonflat_curves(),
        id="up_in_put_nonflat",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier,rebate,r_curve,q_curve",
    _BARRIER_SCENARIOS,
)
def test_barrier_european_vs_quantlib(
    direction, action, option_type, strike, barrier, rebate, r_curve, q_curve
):
    """European barrier option: DP analytical, PDE, MC, and binomial vs QuantLib.

    For flat curves, PDE is compared against QL analytical (constant-rate).
    For non-flat curves, PDE is compared against QL FdBlackScholesBarrierEngine
    (time-varying rates) since the analytical formula assumes constant rates.
    """
    dp_analytical = _dp_barrier_analytical_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
        r_curve=r_curve,
        q_curve=q_curve,
    )
    dp_pde = _dp_barrier_pde_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        r_curve=r_curve,
        q_curve=q_curve,
    )
    dp_mc = _dp_barrier_mc_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        r_curve=r_curve,
        q_curve=q_curve,
    )
    dp_bn = _dp_barrier_binomial_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.EUROPEAN,
        rebate=rebate,
        r_curve=r_curve,
        q_curve=q_curve,
    )
    ql_analytical = _ql_barrier_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
        r_curve=r_curve,
        q_curve=q_curve,
    )
    assert np.isclose(dp_analytical, ql_analytical, rtol=1e-10), (
        f"DP_AN {dp_analytical:.6f} vs QL_AN {ql_analytical:.6f}"
    )

    # For non-flat curves, compare DP PDE/MC/binomial against QL FD
    # engine instead of analytical, since the latter assumes constant rates.
    if r_curve is not None:
        ql_fd = _ql_barrier_fd_price(
            direction=direction,
            action=action,
            barrier=barrier,
            option_type=option_type,
            strike=strike,
            rebate=rebate,
            r_curve=r_curve,
            q_curve=q_curve,
        )
        logger.info(
            "Barrier %s-%s %s K=%.0f H=%.0f | DP_AN=%.6f DP_PDE=%.6f DP_MC=%.6f DP_BN=%.6f QL_AN=%.6f QL_FD=%.6f",
            direction.value,
            action.value,
            option_type.value,
            strike,
            barrier,
            dp_analytical,
            dp_pde,
            dp_mc,
            dp_bn,
            ql_analytical,
            ql_fd,
        )
        assert np.isclose(dp_pde, ql_fd, rtol=8e-3, atol=1e-3), (
            f"DP_PDE {dp_pde:.6f} vs QL_FD {ql_fd:.6f}"
        )
        assert np.isclose(dp_mc, ql_fd, rtol=0.015, atol=2e-3), (
            f"DP_MC {dp_mc:.6f} vs QL_FD {ql_fd:.6f}"
        )
        assert np.isclose(dp_bn, ql_fd, rtol=0.012, atol=2e-3), (
            f"DP_BN {dp_bn:.6f} vs QL_FD {ql_fd:.6f}"
        )
    else:
        logger.info(
            "Barrier %s-%s %s K=%.0f H=%.0f | DP_AN=%.6f DP_PDE=%.6f DP_MC=%.6f DP_BN=%.6f QL_AN=%.6f",
            direction.value,
            action.value,
            option_type.value,
            strike,
            barrier,
            dp_analytical,
            dp_pde,
            dp_mc,
            dp_bn,
            ql_analytical,
        )
        assert np.isclose(dp_pde, ql_analytical, rtol=1e-4, atol=1e-4), (
            f"DP_PDE {dp_pde:.6f} vs QL_AN {ql_analytical:.6f}"
        )
        assert np.isclose(dp_mc, ql_analytical, rtol=0.015, atol=1e-3), (
            f"DP_MC {dp_mc:.6f} vs QL_AN {ql_analytical:.6f}"
        )
        assert np.isclose(dp_bn, ql_analytical, rtol=1e-3, atol=1e-4), (
            f"DP_BN {dp_bn:.6f} vs QL_AN {ql_analytical:.6f}"
        )


# Rebate tests: KO at-hit (matches QL), KI at-expiry (known small difference)
_BARRIER_REBATE_SCENARIOS = [
    # KO at-hit — should match QL exactly
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        3.0,
        RebateTiming.AT_HIT,
        None,
        None,
        id="down_out_call_rebate_hit_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        95.0,
        125.0,
        3.0,
        RebateTiming.AT_HIT,
        *_barrier_nonflat_curves(),
        id="up_out_put_rebate_hit_nonflat",
    ),
    # KI at-expiry — known ~0.3% difference vs QL
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        85.0,
        3.0,
        RebateTiming.AT_EXPIRY,
        None,
        None,
        id="down_in_put_rebate_expiry_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.CALL,
        105.0,
        115.0,
        3.0,
        RebateTiming.AT_EXPIRY,
        *_barrier_nonflat_curves(),
        id="up_in_call_rebate_expiry_nonflat",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier,rebate,rebate_timing,r_curve,q_curve",
    _BARRIER_REBATE_SCENARIOS,
)
def test_barrier_rebate_european_vs_quantlib(
    direction,
    action,
    option_type,
    strike,
    barrier,
    rebate,
    rebate_timing,
    r_curve,
    q_curve,
):
    """Barrier rebate pricing: analytical, PDE, MC, and binomial vs QuantLib."""
    dp_analytical = _dp_barrier_analytical_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
        rebate_timing=rebate_timing,
        r_curve=r_curve,
        q_curve=q_curve,
    )
    dp_pde = _dp_barrier_pde_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
        rebate_timing=rebate_timing,
        r_curve=r_curve,
        q_curve=q_curve,
    )
    dp_mc = _dp_barrier_mc_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
        rebate_timing=rebate_timing,
        r_curve=r_curve,
        q_curve=q_curve,
    )
    dp_bn = _dp_barrier_binomial_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.EUROPEAN,
        rebate=rebate,
        rebate_timing=rebate_timing,
        r_curve=r_curve,
        q_curve=q_curve,
    )
    ql_analytical = _ql_barrier_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
        r_curve=r_curve,
        q_curve=q_curve,
    )
    # TODO: Switch ql to FD engine for non-flat curves
    assert np.isclose(dp_analytical, ql_analytical, rtol=1e-10), (
        f"DP_AN {dp_analytical:.6f} vs QL_AN {ql_analytical:.6f}"
    )
    assert np.isclose(dp_pde, ql_analytical, rtol=3e-3, atol=1e-4), (
        f"DP_PDE {dp_pde:.6f} vs QL_AN {ql_analytical:.6f}"
    )
    assert np.isclose(dp_mc, ql_analytical, rtol=0.015, atol=1e-3), (
        f"DP_MC {dp_mc:.6f} vs QL_AN {ql_analytical:.6f}"
    )
    assert np.isclose(dp_bn, ql_analytical, rtol=3e-3, atol=1e-4), (
        f"DP_BN {dp_bn:.6f} vs QL_AN {ql_analytical:.6f}"
    )

    logger.info(
        "Barrier rebate %s-%s %s K=%.0f H=%.0f R=%.1f | DP_AN=%.6f DP_PDE=%.6f DP_MC=%.6f DP_BN=%.6f QL_AN=%.6f",
        direction.value,
        action.value,
        option_type.value,
        strike,
        barrier,
        rebate,
        dp_analytical,
        dp_pde,
        dp_mc,
        dp_bn,
        ql_analytical,
    )


_BARRIER_BINOMIAL_EUROPEAN_SCENARIOS = [
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        0.0,
        RebateTiming.AT_HIT,
        id="binom_eu_down_out_call",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        115.0,
        0.0,
        RebateTiming.AT_EXPIRY,
        id="binom_eu_up_in_put",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        3.0,
        RebateTiming.AT_HIT,
        id="binom_eu_down_out_call_rebate",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        115.0,
        3.0,
        RebateTiming.AT_EXPIRY,
        id="binom_eu_up_in_put_rebate",
    ),
]


@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier,rebate,rebate_timing",
    _BARRIER_BINOMIAL_EUROPEAN_SCENARIOS,
)
def test_barrier_binomial_european_vs_quantlib(
    direction,
    action,
    option_type,
    strike,
    barrier,
    rebate,
    rebate_timing,
):
    dp_bn = _dp_barrier_binomial_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.EUROPEAN,
        rebate=rebate,
        rebate_timing=rebate_timing,
    )
    ql_bn = _ql_barrier_binomial_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
        binom_steps=_BARRIER_BINOM_CFG.num_steps,
    )
    assert np.isclose(dp_bn, ql_bn, rtol=0.003), f"DP_BN {dp_bn:.6f} vs QL_BN {ql_bn:.6f}"


_BARRIER_BINOMIAL_AMERICAN_SCENARIOS = [
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.PUT,
        100.0,
        85.0,
        0.0,
        RebateTiming.AT_HIT,
        id="binom_am_down_out_put",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        120.0,
        0.0,
        RebateTiming.AT_EXPIRY,
        id="binom_am_up_in_put",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        100.0,
        120.0,
        5.0,
        RebateTiming.AT_HIT,
        id="binom_am_up_out_put_rebate",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        120.0,
        5.0,
        RebateTiming.AT_EXPIRY,
        id="binom_am_up_in_put_rebate",
    ),
]


@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier,rebate,rebate_timing",
    _BARRIER_BINOMIAL_AMERICAN_SCENARIOS,
)
def test_barrier_binomial_american_vs_quantlib(
    direction,
    action,
    option_type,
    strike,
    barrier,
    rebate,
    rebate_timing,
):
    dp_bn = _dp_barrier_binomial_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.AMERICAN,
        rebate=rebate,
        rebate_timing=rebate_timing,
    )
    ql_bn = _ql_barrier_american_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
    )
    assert np.isclose(dp_bn, ql_bn, rtol=0.01), f"DP_BN {dp_bn:.6f} vs QL_BN {ql_bn:.6f}"


# American barrier KO — DP PDE vs QL BinomialCRR
_BARRIER_AMERICAN_KO_SCENARIOS = [
    # Flat curves — KO only (American KI not supported)
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        0.0,
        id="am_down_out_call_flat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.PUT,
        100.0,
        85.0,
        0.0,
        id="am_down_out_put_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        120.0,
        0.0,
        id="am_up_out_call_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        100.0,
        120.0,
        0.0,
        id="am_up_out_put_flat",
    ),
    # With rebate (AT_HIT)
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        5.0,
        id="am_down_out_call_rebate",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        100.0,
        120.0,
        5.0,
        id="am_up_out_put_rebate",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier,rebate",
    _BARRIER_AMERICAN_KO_SCENARIOS,
)
def test_barrier_american_ko_vs_quantlib(
    direction,
    action,
    option_type,
    strike,
    barrier,
    rebate,
):
    """American KO barrier: DP PDE, MC, and binomial vs QL BinomialCRR."""
    dp_pde = _dp_barrier_pde_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.AMERICAN,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_HIT,
    )
    dp_mc = _dp_barrier_mc_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.AMERICAN,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_HIT,
    )
    dp_bn = _dp_barrier_binomial_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.AMERICAN,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_HIT,
    )
    ql_bn = _ql_barrier_american_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
    )
    logger.info(
        "American KO %s-%s %s K=%.0f H=%.0f R=%.1f | DP_PDE=%.6f DP_MC=%.6f DP_BN=%.6f QL_BN=%.6f",
        direction.value,
        action.value,
        option_type.value,
        strike,
        barrier,
        rebate,
        dp_pde,
        dp_mc,
        dp_bn,
        ql_bn,
    )
    # PDE rtol stays at 0.02 for DOWN-OUT put / UP-OUT call scenarios where
    # the KO barrier truncates the in-the-money payoff region. Both DP and
    # QL CRR binomials (~1000 steps) are biased *low* in this regime, they
    # converge to the PDE engine as binomial steps are increased.
    assert np.isclose(dp_pde, ql_bn, rtol=0.02, atol=1e-3), (
        f"DP_PDE {dp_pde:.6f} vs QL_BN {ql_bn:.6f}"
    )
    assert np.isclose(dp_bn, ql_bn, rtol=6e-3, atol=1e-3), f"DP_BN {dp_bn:.6f} vs QL_BN {ql_bn:.6f}"
    # LSM shows larger downward bias when the KO barrier cuts into the payoff region.
    mc_tol = (
        0.08
        if (
            (direction is BarrierDirection.UP and option_type is OptionType.CALL)
            or (direction is BarrierDirection.DOWN and option_type is OptionType.PUT)
        )
        else 0.03
    )
    if mc_tol == 0.08:
        logger.info(
            "American KO MC tolerance widened to %.2f%% because LSM downward bias is expected when "
            "the KO barrier cuts into the payoff region.",
            mc_tol * 100,
        )
    assert np.isclose(dp_mc, ql_bn, rtol=mc_tol), f"DP_MC {dp_mc:.6f} vs QL_BN {ql_bn:.6f}"


# ── American knock-in barrier: PDE (two-surface) vs QL BinomialCRR ────────

_BARRIER_AMERICAN_KI_SCENARIOS = [
    # Down-and-in
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.CALL,
        100.0,
        85.0,
        0.0,
        id="am_down_in_call_flat",
    ),
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        85.0,
        0.0,
        id="am_down_in_put_flat",
    ),
    # Up-and-in
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.CALL,
        100.0,
        120.0,
        0.0,
        id="am_up_in_call_flat",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        120.0,
        0.0,
        id="am_up_in_put_flat",
    ),
    # With rebate (AT_EXPIRY for KI)
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.IN,
        OptionType.CALL,
        100.0,
        85.0,
        5.0,
        id="am_down_in_call_rebate",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.PUT,
        100.0,
        120.0,
        5.0,
        id="am_up_in_put_rebate",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier,rebate",
    _BARRIER_AMERICAN_KI_SCENARIOS,
)
def test_barrier_american_ki_vs_quantlib(
    direction,
    action,
    option_type,
    strike,
    barrier,
    rebate,
):
    """American KI barrier: DP PDE, MC, and binomial vs QL BinomialCRR."""
    dp_pde = _dp_barrier_pde_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.AMERICAN,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )
    dp_mc = _dp_barrier_mc_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.AMERICAN,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )
    dp_bn = _dp_barrier_binomial_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.AMERICAN,
        rebate=rebate,
        rebate_timing=RebateTiming.AT_EXPIRY,
    )

    ql_bn = _ql_barrier_american_price(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        rebate=rebate,
    )
    logger.info(
        "American KI %s-%s %s K=%.0f H=%.0f R=%.1f | DP_PDE=%.6f DP_MC=%.6f DP_BN=%.6f QL_BN=%.6f",
        direction.value,
        action.value,
        option_type.value,
        strike,
        barrier,
        rebate,
        dp_pde,
        dp_mc,
        dp_bn,
        ql_bn,
    )
    assert np.isclose(dp_pde, ql_bn, rtol=3e-3, atol=1e-3), (
        f"DP_PDE {dp_pde:.6f} vs QL_BN {ql_bn:.6f}"
    )
    assert np.isclose(dp_bn, ql_bn, rtol=3e-3, atol=1e-3), f"DP_BN {dp_bn:.6f} vs QL_BN {ql_bn:.6f}"
    assert np.isclose(dp_mc, ql_bn, rtol=0.02, atol=2e-3), f"DP_MC {dp_mc:.6f} vs QL_BN {ql_bn:.6f}"


# ── European barrier + discrete dividends: DP PDE vs QL FD ───────────────
# BSM can't handle discrete dividends (escrow-adjusted formulas don't apply
# to path-dependent barriers). Binomial and PDE both can. Here we
# cross-validate DP PDE_FD against QL FdBlackScholesBarrierEngine with a
# DividendVector — the only external reference that handles the combination.


def _ql_barrier_fd_with_discrete_divs(
    *,
    direction: BarrierDirection,
    action: BarrierAction,
    barrier: float,
    option_type: OptionType,
    strike: float,
    discrete_dividends: Sequence[tuple[dt.datetime, float]],
    spot: float = _BARRIER_SPOT,
    grid_points: int = 800,
    time_steps: int = 800,
) -> float:
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    ql_dc = ql.Actual365Fixed()
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    rf_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, _BARRIER_RATE, ql_dc))
    # No continuous div yield — all dividends are cash-discrete here.
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, ql_dc))
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, ql.TARGET(), _BARRIER_VOL, ql_dc)
    )
    proc = ql.BlackScholesMertonProcess(spot_h, div_h, rf_h, vol_h)

    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    exercise = ql.EuropeanExercise(
        ql.Date(_BARRIER_MATURITY.day, _BARRIER_MATURITY.month, _BARRIER_MATURITY.year)
    )
    bt = _QL_BARRIER_TYPE[(direction, action)]
    opt = ql.BarrierOption(bt, barrier, 0.0, payoff, exercise)

    div_vector = _quantlib_dividend_schedule(discrete_dividends)
    opt.setPricingEngine(ql.FdBlackScholesBarrierEngine(proc, div_vector, time_steps, grid_points))
    return opt.NPV()


_BARRIER_DISCRETE_DIV_SCENARIOS = [
    pytest.param(
        BarrierDirection.DOWN,
        BarrierAction.OUT,
        OptionType.CALL,
        100.0,
        85.0,
        id="down_out_call",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.IN,
        OptionType.CALL,
        100.0,
        120.0,
        id="up_in_call",
    ),
    pytest.param(
        BarrierDirection.UP,
        BarrierAction.OUT,
        OptionType.PUT,
        100.0,
        120.0,
        id="up_out_put",
    ),
]


@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier",
    _BARRIER_DISCRETE_DIV_SCENARIOS,
)
def test_barrier_european_discrete_divs_pde_vs_quantlib(
    direction, action, option_type, strike, barrier
):
    """European barrier + discrete cash dividends: DP PDE_FD vs QL FD.

    The interaction between the dividend jump and the barrier boundary is
    non-trivial — post-ex spots can land on the knockout side of the
    barrier purely from the discrete jump. This test catches bugs in the
    `_apply_dividend_jump` coupling with the barrier boundary conditions.
    """
    discrete_divs = [
        (PRICING_DATE + dt.timedelta(days=90), 2.0),
        (PRICING_DATE + dt.timedelta(days=270), 2.0),
    ]

    ttm = calculate_year_fraction(PRICING_DATE, _BARRIER_MATURITY)
    rc = DiscountCurve.flat(_BARRIER_RATE, end_time=ttm)
    md = MarketData(PRICING_DATE, rc, currency=CURRENCY)
    ud = UnderlyingData(
        initial_value=_BARRIER_SPOT,
        volatility=_BARRIER_VOL,
        market_data=md,
        discrete_dividends=discrete_divs,
    )
    spec = _barrier_spec(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.EUROPEAN,
    )
    dp_pde = OptionValuation(ud, spec, PricingMethod.PDE_FD).present_value()

    ql_fd = _ql_barrier_fd_with_discrete_divs(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        discrete_dividends=discrete_divs,
    )
    logger.info(
        "Barrier discrete-div %s-%s %s K=%.0f H=%.0f | DP_PDE=%.6f QL_FD=%.6f",
        direction.value,
        action.value,
        option_type.value,
        strike,
        barrier,
        dp_pde,
        ql_fd,
    )

    assert np.isclose(dp_pde, ql_fd, rtol=0.01, atol=1e-3), (
        f"DP_PDE {dp_pde:.6f} vs QL_FD {ql_fd:.6f}"
    )


@pytest.mark.parametrize(
    "direction,action,option_type,strike,barrier",
    _BARRIER_DISCRETE_DIV_SCENARIOS,
)
def test_barrier_european_boundary_date_discrete_divs_pde_vs_quantlib(
    direction, action, option_type, strike, barrier
):
    """European barrier + discrete dividends on pricing date AND maturity date.

    Boundary-date dividends exercise two distinct DP code paths not hit by
    the in-between-only scenario above:

    * Pricing-date dividend: the input spot is cum-dividend, so DP prices
      at ``S0 - D_0`` by interpolating the final PDE surface at that
      ex-dividend spot (``pricing_div`` branch in ``_fd_barrier_core``).

    * Maturity-date dividend: the terminal payoff is evaluated at the
      ex-dividend spot at maturity; DP applies a dividend jump to the
      terminal value array before the first backward time step
      (``mat_div`` branch, popped out of the tau grid).

    QL's ``FdBlackScholesBarrierEngine`` does not accept a ``DividendVector``
    entry on the evaluation date — empirically it raises
    ``RuntimeError: theta not provided`` for up-in barriers whenever any
    dividend sits on the pricing date.  We work around this via the
    equivalence:

        (S0 cum-div,  schedule including D_0 at t=0)
          == (S0 − D_0 ex-div,  schedule with D_0 stripped)

    DP sees the cum-div form and applies ``pricing_div`` internally at the
    final interpolation step; QL sees the shifted-spot form so it never
    needs a pricing-date dividend.  Both price the same economic problem.
    """
    pricing_div = 1.5
    mid_div = 2.0
    maturity_div = 1.0
    discrete_divs = [
        (PRICING_DATE, pricing_div),
        (PRICING_DATE + dt.timedelta(days=180), mid_div),
        (_BARRIER_MATURITY, maturity_div),
    ]

    ttm = calculate_year_fraction(PRICING_DATE, _BARRIER_MATURITY)
    rc = DiscountCurve.flat(_BARRIER_RATE, end_time=ttm)
    md = MarketData(PRICING_DATE, rc, currency=CURRENCY)
    ud = UnderlyingData(
        initial_value=_BARRIER_SPOT,
        volatility=_BARRIER_VOL,
        market_data=md,
        discrete_dividends=discrete_divs,
    )
    spec = _barrier_spec(
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        exercise_type=ExerciseType.EUROPEAN,
    )
    dp_pde = OptionValuation(ud, spec, PricingMethod.PDE_FD).present_value()

    # QL workaround: shift spot by pricing-date div, strip that entry from schedule.
    ql_spot = _BARRIER_SPOT - pricing_div
    ql_divs = [(d, a) for d, a in discrete_divs if d != PRICING_DATE]
    ql_fd = _ql_barrier_fd_with_discrete_divs(
        spot=ql_spot,
        direction=direction,
        action=action,
        barrier=barrier,
        option_type=option_type,
        strike=strike,
        discrete_dividends=ql_divs,
    )
    logger.info(
        "Barrier boundary-div %s-%s %s K=%.0f H=%.0f | DP_PDE=%.6f QL_FD=%.6f",
        direction.value,
        action.value,
        option_type.value,
        strike,
        barrier,
        dp_pde,
        ql_fd,
    )

    assert np.isclose(dp_pde, ql_fd, rtol=0.01, atol=1e-3), (
        f"DP_PDE {dp_pde:.6f} vs QL_FD {ql_fd:.6f}"
    )
