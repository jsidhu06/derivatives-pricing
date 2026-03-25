"""Compare Greeks between derivatives_pricing and QuantLib for reference.

Sections
--------
1. Vanilla European Greeks: DP engines vs QuantLib (broad scenarios)
2. Vanilla American Greeks: DP engines vs QuantLib FD (broad scenarios)
3. European Asian MC numerical Greeks vs QuantLib
"""

from __future__ import annotations

from collections.abc import Sequence
import datetime as dt
import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from derivatives_pricing.enums import (
    AsianAveraging,
    DayCountConvention,
    ExerciseType,
    OptionType,
    PricingMethod,
)
from derivatives_pricing.market_environment import MarketData
from derivatives_pricing.rates import DiscountCurve
from helpers import (
    assert_greeks_close,
    market_data,
    make_vanilla_spec,
    underlying,
)
from derivatives_pricing.valuation import (
    AsianSpec,
    VanillaSpec,
    OptionValuation,
    UnderlyingData,
)
from derivatives_pricing.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from derivatives_pricing.valuation.params import (
    BinomialParams,
    MonteCarloParams,
    PDEParams,
)
from derivatives_pricing.utils import calculate_year_fraction

if TYPE_CHECKING:
    import QuantLib as ql_typing

pytestmark = pytest.mark.slow

ql = pytest.importorskip("QuantLib")

logger = logging.getLogger(__name__)


def _ql_dcc(dcc: DayCountConvention):
    """Map a DP DayCountConvention to the corresponding QuantLib DayCounter."""
    if dcc is DayCountConvention.ACT_360:
        return ql.Actual360()
    return ql.Actual365Fixed()


# Convention mapping: QL → DP
# delta, gamma: same
# vega: QL per 100% vol → /100 to match DP per 1 vol-pt
# theta: QL per year → /365 to match DP per day
# rho: QL per 100% rate → /100 to match DP per 1%

_QL_SCALE = {"delta": 1.0, "gamma": 1.0, "vega": 1 / 100, "theta": 1 / 365, "rho": 1 / 100}

# ── Shared constants ────────────────────────────────────────────────────

PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = PRICING_DATE + dt.timedelta(days=365)
RISK_FREE = 0.05
VOL = 0.20
CURRENCY = "USD"

BINOM_CFG = BinomialParams(num_steps=1500)
PDE_CFG = PDEParams(spot_steps=200, time_steps=200, max_iter=20_000)
MC_CFG = MonteCarloParams(random_seed=42)

# ── derivatives_pricing helpers ─────────────────────────────────────────


def _market_data(
    discount_curve: DiscountCurve | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> MarketData:
    if discount_curve is None:
        ttm = calculate_year_fraction(PRICING_DATE, MATURITY, dcc)
        discount_curve = DiscountCurve.flat(RISK_FREE, end_time=ttm)
    return market_data(
        pricing_date=PRICING_DATE,
        discount_curve=discount_curve,
        currency=CURRENCY,
        day_count_convention=dcc,
    )


def _spec(
    *,
    strike: float,
    option_type: OptionType,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
) -> VanillaSpec:
    return make_vanilla_spec(
        strike=strike,
        maturity=MATURITY,
        option_type=option_type,
        exercise_type=exercise_type,
        currency=CURRENCY,
    )


def _underlying(
    *,
    spot: float,
    risk_free_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> UnderlyingData:
    return underlying(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(discount_curve=risk_free_curve, dcc=dcc),
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def _gbm(
    *,
    spot: float,
    risk_free_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    paths: int = 500_000,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> GBMProcess:
    return GBMProcess(
        _market_data(discount_curve=risk_free_curve, dcc=dcc),
        GBMParams(
            initial_value=spot,
            volatility=VOL,
            dividend_curve=dividend_curve,
            discrete_dividends=discrete_dividends,
        ),
        SimulationConfig(
            paths=paths,
            num_steps=52,
            end_date=MATURITY,
        ),
    )


# ── QuantLib helpers ────────────────────────────────────────────────────


def _ql_setup() -> "ql_typing.Date":
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    return eval_date


def _ql_curve_handle_from_discount_curve(
    curve: DiscountCurve,
    *,
    eval_date: "ql_typing.Date",
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> "ql_typing.YieldTermStructureHandle":
    ql_dc = _ql_dcc(dcc)
    denom = 360.0 if dcc is DayCountConvention.ACT_360 else 365.0
    dates = [eval_date]
    for t in curve.times[1:]:
        dates.append(eval_date + int(round(float(t) * denom)))
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, list(curve.dfs), ql_dc))


def _ql_dividend_vector(
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
) -> "ql_typing.DividendVector":
    if not discrete_dividends:
        return ql.DividendVector([], [])
    dates: list[ql_typing.Date] = []
    amounts: list[float] = []
    for ex_date, amount in discrete_dividends:
        if PRICING_DATE <= ex_date <= MATURITY:
            dates.append(ql.Date(ex_date.day, ex_date.month, ex_date.year))
            amounts.append(float(amount))
    return ql.DividendVector(dates, amounts)


def _ql_process(
    eval_date: "ql_typing.Date",
    *,
    spot: float,
    dividend_yield: float = 0.0,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> "ql_typing.BlackScholesMertonProcess":
    """BSM process with flat rate, dividend yield, and vol."""
    ql_dc = _ql_dcc(dcc)
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, dividend_yield, ql_dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, RISK_FREE, ql_dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, ql.TARGET(), VOL, ql_dc)),
    )


def _ql_european_option(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    risk_free_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> "ql_typing.VanillaOption":
    """QuantLib European with AnalyticEuropeanEngine (flat or term-structured curves)."""
    eval_date = _ql_setup()
    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)
    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    ql_dc = _ql_dcc(dcc)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql_type, strike),
        ql.EuropeanExercise(ql_maturity),
    )

    if risk_free_curve is None and dividend_curve is None:
        process = _ql_process(eval_date, spot=spot, dividend_yield=dividend_yield, dcc=dcc)
    else:
        rf_handle = (
            _ql_curve_handle_from_discount_curve(risk_free_curve, eval_date=eval_date, dcc=dcc)
            if risk_free_curve is not None
            else ql.YieldTermStructureHandle(ql.FlatForward(eval_date, RISK_FREE, ql_dc))
        )
        div_handle = (
            _ql_curve_handle_from_discount_curve(dividend_curve, eval_date=eval_date, dcc=dcc)
            if dividend_curve is not None
            else ql.YieldTermStructureHandle(ql.FlatForward(eval_date, dividend_yield, ql_dc))
        )
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(spot)),
            div_handle,
            rf_handle,
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, ql.TARGET(), VOL, ql_dc)),
        )

    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option


def _ql_american_fd_option(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    risk_free_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    grid_points: int = 200,
    time_steps: int = 400,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> "ql_typing.VanillaOption":
    """QuantLib American with FdBlackScholesVanillaEngine."""
    eval_date = _ql_setup()
    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)
    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    ql_dc = _ql_dcc(dcc)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql_type, strike),
        ql.AmericanExercise(eval_date, ql_maturity),
    )
    if risk_free_curve is None and dividend_curve is None:
        process = _ql_process(eval_date, spot=spot, dividend_yield=dividend_yield, dcc=dcc)
    else:
        rf_handle = (
            _ql_curve_handle_from_discount_curve(risk_free_curve, eval_date=eval_date, dcc=dcc)
            if risk_free_curve is not None
            else ql.YieldTermStructureHandle(ql.FlatForward(eval_date, RISK_FREE, ql_dc))
        )
        div_handle = (
            _ql_curve_handle_from_discount_curve(dividend_curve, eval_date=eval_date, dcc=dcc)
            if dividend_curve is not None
            else ql.YieldTermStructureHandle(ql.FlatForward(eval_date, dividend_yield, ql_dc))
        )
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(spot)),
            div_handle,
            rf_handle,
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, ql.TARGET(), VOL, ql_dc)),
        )

    engine = ql.FdBlackScholesVanillaEngine(
        process,
        _ql_dividend_vector(discrete_dividends),
        grid_points,
        time_steps,
    )
    option.setPricingEngine(engine)
    return option


def _ql_greek(option: "ql_typing.VanillaOption", greek: str) -> float | None:
    """Return a QuantLib greek when available, else ``None``."""
    try:
        return float(getattr(option, greek)())
    except RuntimeError:
        return None


def _ql_scaled_greeks(
    option: "ql_typing.VanillaOption",
    *,
    allow_missing: bool,
) -> dict[str, float | None]:
    """Return QuantLib Greeks scaled to derivatives_pricing conventions."""
    values: dict[str, float | None] = {}
    for greek, scale in _QL_SCALE.items():
        val = _ql_greek(option, greek)
        if val is None and allow_missing:
            values[greek] = None
        else:
            values[greek] = None if val is None else val * scale
    return values


# Convention conversions (QuantLib → derivatives_pricing):
#   vega:  QL returns d(V)/d(σ);  DP returns d(V)/d(σ)/100  (per 1 vol-pt)
#   theta: QL returns d(V)/d(t) per year; DP returns per calendar day (/365)
#   rho:   QL returns d(V)/d(r);  DP returns d(V)/d(r)/100 (per 1% rate)


# ═══════════════════════════════════════════════════════════════════════
# 1. Vanilla European Greeks: DP engines vs QuantLib (broad scenarios)
# ═══════════════════════════════════════════════════════════════════════

_EU_VANILLA_SCENARIOS = [
    pytest.param(
        100.0,
        100.0,
        OptionType.CALL,
        "flat",
        "none",
        DayCountConvention.ACT_365F,
        id="atm_call_flat_ACT365F",
    ),
    pytest.param(
        100.0,
        95.0,
        OptionType.PUT,
        "flat",
        "flat",
        DayCountConvention.ACT_360,
        id="itm_put_flat_div_ACT360",
    ),
    pytest.param(
        90.0,
        100.0,
        OptionType.CALL,
        "nonflat",
        "nonflat",
        DayCountConvention.ACT_360,
        id="otm_call_nonflat_ACT360",
    ),
    pytest.param(
        110.0,
        100.0,
        OptionType.PUT,
        "nonflat",
        "none",
        DayCountConvention.ACT_365F,
        id="otm_put_nonflat_ACT365F",
    ),
]


def _resolve_curve(
    kind: str,
    *,
    is_dividend: bool,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> DiscountCurve | None:
    if kind == "none":
        return None
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY, dcc)
    if kind == "flat":
        rate = 0.03 if is_dividend else RISK_FREE
        return DiscountCurve.flat(rate, end_time=ttm)
    if kind == "nonflat":
        forwards = np.array([0.01, 0.02, 0.015]) if is_dividend else np.array([0.03, 0.05, 0.06])
        return DiscountCurve.from_forwards(times=np.array([0.0, 0.25, 0.5, ttm]), forwards=forwards)
    raise ValueError(f"unsupported curve kind: {kind}")


@pytest.mark.parametrize("spot,strike,option_type,rate_kind,div_kind,dcc", _EU_VANILLA_SCENARIOS)
@pytest.mark.parametrize(
    "engine,tols",
    [
        (
            PricingMethod.BSM,
            {"delta": 1e-4, "gamma": 1e-4, "vega": 1e-4, "theta": 1e-4, "rho": 1e-4},
        ),
        (
            PricingMethod.BINOMIAL,
            {"delta": 0.01, "gamma": 0.03, "vega": 0.08, "theta": 0.30, "rho": 0.08},
        ),
        (
            PricingMethod.PDE_FD,
            {"delta": 0.02, "gamma": 0.05, "vega": 0.10, "theta": 0.30, "rho": 0.08},
        ),
        (
            PricingMethod.MONTE_CARLO,
            {"delta": 0.03, "gamma": 0.10, "vega": 0.05, "theta": 0.10, "rho": 0.10},
        ),
    ],
)
def test_vanilla_european_greeks_vs_quantlib(
    spot, strike, option_type, rate_kind, div_kind, dcc, engine, tols
):
    """DP vanilla European Greeks align with QuantLib across flat/non-flat curve scenarios."""
    r_curve = _resolve_curve(rate_kind, is_dividend=False, dcc=dcc)
    q_curve = _resolve_curve(div_kind, is_dividend=True, dcc=dcc)

    underlying = (
        _gbm(spot=spot, risk_free_curve=r_curve, dividend_curve=q_curve, dcc=dcc)
        if engine is PricingMethod.MONTE_CARLO
        else _underlying(spot=spot, risk_free_curve=r_curve, dividend_curve=q_curve, dcc=dcc)
    )
    params = (
        MC_CFG
        if engine is PricingMethod.MONTE_CARLO
        else BINOM_CFG
        if engine is PricingMethod.BINOMIAL
        else PDE_CFG
        if engine is PricingMethod.PDE_FD
        else None
    )

    ov = OptionValuation(
        underlying,
        _spec(strike=strike, option_type=option_type),
        engine,
        params=params,
    )
    ql_opt = _ql_european_option(
        spot=spot,
        strike=strike,
        option_type=option_type,
        risk_free_curve=r_curve,
        dividend_curve=q_curve,
        dcc=dcc,
    )

    ql_values = _ql_scaled_greeks(ql_opt, allow_missing=False)
    dp_values = {
        "delta": ov.delta(),
        "gamma": ov.gamma(),
        "vega": ov.vega(),
        "theta": ov.theta(),
        "rho": ov.rho(),
    }

    assert_greeks_close(
        lhs=dp_values,
        rhs=ql_values,
        tols=tols,
        log_prefix=f"{engine.name} EU {option_type.value} S={spot:.0f} K={strike:.0f} {dcc.value}",
        lhs_name="DP",
        rhs_name="QL",
        skip_missing_rhs=False,
        logger=logger,
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. Vanilla American Greeks: DP engines vs QuantLib FD (broad scenarios)
# ═══════════════════════════════════════════════════════════════════════

_AM_VANILLA_SCENARIOS = [
    pytest.param(
        90.0,
        100.0,
        OptionType.PUT,
        "flat",
        "none",
        None,
        DayCountConvention.ACT_365F,
        id="itm_put_flat_ACT365F",
    ),
    pytest.param(
        110.0,
        100.0,
        OptionType.CALL,
        "flat",
        "flat",
        None,
        DayCountConvention.ACT_360,
        id="itm_call_flat_div_ACT360",
    ),
    pytest.param(
        100.0,
        100.0,
        OptionType.PUT,
        "nonflat",
        "nonflat",
        None,
        DayCountConvention.ACT_360,
        id="atm_put_nonflat_ACT360",
    ),
    pytest.param(
        100.0,
        100.0,
        OptionType.PUT,
        "nonflat",
        "none",
        [
            (PRICING_DATE + dt.timedelta(days=90), 0.50),
            (PRICING_DATE + dt.timedelta(days=270), 0.50),
        ],
        DayCountConvention.ACT_365F,
        id="atm_put_nonflat_discrete_ACT365F",
    ),
]


@pytest.mark.parametrize(
    "spot,strike,option_type,rate_kind,div_kind,discrete_dividends,dcc", _AM_VANILLA_SCENARIOS
)
@pytest.mark.parametrize(
    "engine,tols",
    [
        (
            PricingMethod.BINOMIAL,
            {"delta": 0.03, "gamma": 0.08, "vega": 0.12, "theta": 0.35, "rho": 0.12},
        ),
        (
            PricingMethod.PDE_FD,
            {"delta": 0.03, "gamma": 0.08, "vega": 0.15, "theta": 0.35, "rho": 0.15},
        ),
    ],
)
def test_vanilla_american_greeks_vs_quantlib(
    spot,
    strike,
    option_type,
    rate_kind,
    div_kind,
    discrete_dividends,
    dcc,
    engine,
    tols,
):
    """DP vanilla American Greeks align with QuantLib FD for broad curve/dividend scenarios."""
    r_curve = _resolve_curve(rate_kind, is_dividend=False, dcc=dcc)
    q_curve = _resolve_curve(div_kind, is_dividend=True, dcc=dcc)

    ov = OptionValuation(
        _underlying(
            spot=spot,
            risk_free_curve=r_curve,
            dividend_curve=q_curve,
            discrete_dividends=discrete_dividends,
            dcc=dcc,
        ),
        _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.AMERICAN),
        engine,
        params=BINOM_CFG if engine is PricingMethod.BINOMIAL else PDE_CFG,
    )
    ql_opt = _ql_american_fd_option(
        spot=spot,
        strike=strike,
        option_type=option_type,
        risk_free_curve=r_curve,
        dividend_curve=q_curve,
        discrete_dividends=discrete_dividends,
        dcc=dcc,
    )

    dp_values = {
        "delta": ov.delta(),
        "gamma": ov.gamma(),
        "vega": ov.vega(),
        "theta": ov.theta(),
        "rho": ov.rho(),
    }
    ql_values = _ql_scaled_greeks(ql_opt, allow_missing=True)
    assert_greeks_close(
        lhs=dp_values,
        rhs=ql_values,
        tols=tols,
        log_prefix=f"{engine.name} AM {option_type.value} S={spot:.0f} K={strike:.0f} {dcc.value}",
        lhs_name="DP",
        rhs_name="QL",
        skip_missing_rhs=True,
        atol=1e-4,
        logger=logger,
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. European Asian MC numerical Greeks vs QuantLib
# ═══════════════════════════════════════════════════════════════════════

_ASIAN_FIXINGS = tuple(
    dt.datetime(2025, m, 1) if m <= 12 else dt.datetime(2026, m - 12, 1) for m in range(2, 14)
)


def _dt_to_ql(d: dt.datetime) -> "ql_typing.Date":
    return ql.Date(d.day, d.month, d.year)


def _ql_asian_greeks(
    *,
    option_type: OptionType,
    averaging: AsianAveraging,
    strike: float,
    spot: float,
    vol: float,
    risk_free_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> dict[str, float | None]:
    """Price a European Asian via QuantLib and return available Greeks.

    Geometric analytic engine provides all 5 Greeks.
    TW arithmetic engine provides delta and gamma only.
    """
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    ql_dc = _ql_dcc(dcc)

    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    rf_h = _ql_curve_handle_from_discount_curve(risk_free_curve, eval_date=eval_date, dcc=dcc)
    div_h = (
        _ql_curve_handle_from_discount_curve(dividend_curve, eval_date=eval_date, dcc=dcc)
        if dividend_curve is not None
        else ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, ql_dc))
    )
    vol_h = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, ql.TARGET(), vol, ql_dc))
    proc = ql.BlackScholesMertonProcess(spot_h, div_h, rf_h, vol_h)

    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    exercise = ql.EuropeanExercise(_dt_to_ql(MATURITY))
    ql_fixings = [_dt_to_ql(d) for d in _ASIAN_FIXINGS]

    if averaging is AsianAveraging.GEOMETRIC:
        ql_avg = ql.Average.Geometric
        engine = ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(proc)
    else:
        ql_avg = ql.Average.Arithmetic
        engine = ql.TurnbullWakemanAsianEngine(proc)

    opt = ql.DiscreteAveragingAsianOption(ql_avg, ql_fixings, payoff, exercise)
    opt.setPricingEngine(engine)

    result: dict[str, float | None] = {
        "npv": opt.NPV(),
        "delta": opt.delta(),
        "gamma": opt.gamma(),
    }
    # TW engine does not provide vega/theta/rho
    for greek in ("vega", "theta", "rho"):
        try:
            result[greek] = getattr(opt, greek)()
        except RuntimeError:
            result[greek] = None
    return result


def _dp_asian_mc_greeks(
    *,
    option_type: OptionType,
    averaging: AsianAveraging,
    strike: float,
    spot: float,
    vol: float,
    risk_free_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> dict[str, float]:
    """Build our MC Asian and compute numerical Greeks."""
    md = MarketData(PRICING_DATE, risk_free_curve, currency=CURRENCY, day_count_convention=dcc)
    params = GBMParams(initial_value=spot, volatility=vol, dividend_curve=dividend_curve)
    sim_cfg = SimulationConfig(
        paths=150_000,
        end_date=MATURITY,
        num_steps=30,
    )
    process = GBMProcess(md, params, sim_cfg)
    spec = AsianSpec(
        averaging=averaging,
        option_type=option_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
        fixing_dates=_ASIAN_FIXINGS,
        exercise_type=ExerciseType.EUROPEAN,
    )
    ov = OptionValuation(
        process,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=42),
    )
    return {
        "npv": ov.present_value(),
        "delta": ov.delta(),
        "gamma": ov.gamma(),
        "vega": ov.vega(),
        "theta": ov.theta(),
        "rho": ov.rho(),
    }


def _dp_asian_analytical_greeks(
    *,
    option_type: OptionType,
    averaging: AsianAveraging,
    strike: float,
    spot: float,
    vol: float,
    risk_free_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dcc: DayCountConvention = DayCountConvention.ACT_365F,
) -> dict[str, float]:
    """Build our analytical (BSM) Asian and compute numerical Greeks."""
    md = MarketData(PRICING_DATE, risk_free_curve, currency=CURRENCY, day_count_convention=dcc)
    underlying = UnderlyingData(
        initial_value=spot,
        volatility=vol,
        market_data=md,
        dividend_curve=dividend_curve,
    )
    spec = AsianSpec(
        averaging=averaging,
        option_type=option_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
        fixing_dates=_ASIAN_FIXINGS,
        exercise_type=ExerciseType.EUROPEAN,
    )
    ov = OptionValuation(underlying, spec, PricingMethod.BSM)
    return {
        "npv": ov.present_value(),
        "delta": ov.delta(),
        "gamma": ov.gamma(),
        "vega": ov.vega(),
        "theta": ov.theta(),
        "rho": ov.rho(),
    }


_ASIAN_GREEK_SCENARIOS = [
    # Geometric (all 5 Greeks available from QL analytic engine)
    pytest.param(
        100,
        100,
        0.20,
        OptionType.CALL,
        AsianAveraging.GEOMETRIC,
        "flat",
        "none",
        DayCountConvention.ACT_365F,
        id="geom_call_atm_flat_ACT365F",
    ),
    pytest.param(
        100,
        90,
        0.25,
        OptionType.PUT,
        AsianAveraging.GEOMETRIC,
        "nonflat",
        "flat",
        DayCountConvention.ACT_360,
        id="geom_put_itm_nonflat_ACT360",
    ),
    # Arithmetic (TW engine: delta/gamma only)
    pytest.param(
        100,
        110,
        0.30,
        OptionType.CALL,
        AsianAveraging.ARITHMETIC,
        "flat",
        "flat",
        DayCountConvention.ACT_360,
        id="arith_call_otm_flat_div_ACT360",
    ),
    pytest.param(
        100,
        100,
        0.20,
        OptionType.PUT,
        AsianAveraging.ARITHMETIC,
        "nonflat",
        "none",
        DayCountConvention.ACT_365F,
        id="arith_put_atm_nonflat_ACT365F",
    ),
]


@pytest.mark.parametrize(
    "spot,strike,vol,option_type,averaging,rate_kind,div_kind,dcc",
    _ASIAN_GREEK_SCENARIOS,
)
def test_asian_mc_greeks_vs_quantlib(
    spot,
    strike,
    vol,
    option_type,
    averaging,
    rate_kind,
    div_kind,
    dcc,
):
    """European Asian MC numerical Greeks vs QuantLib analytic/TW Greeks."""
    r_curve = _resolve_curve(rate_kind, is_dividend=False, dcc=dcc)
    q_curve = _resolve_curve(div_kind, is_dividend=True, dcc=dcc)
    assert r_curve is not None

    ql_greeks = _ql_asian_greeks(
        option_type=option_type,
        averaging=averaging,
        strike=strike,
        spot=spot,
        vol=vol,
        risk_free_curve=r_curve,
        dividend_curve=q_curve,
        dcc=dcc,
    )
    dp_greeks = _dp_asian_mc_greeks(
        option_type=option_type,
        averaging=averaging,
        strike=strike,
        spot=spot,
        vol=vol,
        risk_free_curve=r_curve,
        dividend_curve=q_curve,
        dcc=dcc,
    )
    dp_an_greeks = _dp_asian_analytical_greeks(
        option_type=option_type,
        averaging=averaging,
        strike=strike,
        spot=spot,
        vol=vol,
        risk_free_curve=r_curve,
        dividend_curve=q_curve,
        dcc=dcc,
    )

    # Tolerances: MC numerical bump-and-revalue vs analytic/TW
    mc_tols = {"delta": 0.03, "gamma": 0.10, "vega": 0.05, "theta": 0.10, "rho": 0.10}
    # Analytical bump-and-revalue vs QL analytic/TW.  Match MC tolerances —
    # QL's own geometric analytic engine has limitations with non-flat curves
    an_tols = {"delta": 0.03, "gamma": 0.10, "vega": 0.05, "theta": 0.10, "rho": 0.10}

    for greek in ("delta", "gamma", "vega", "theta", "rho"):
        ql_val = ql_greeks[greek]
        if ql_val is None:
            continue  # TW engine doesn't provide this greek
        dp_val = dp_greeks[greek]
        dp_an_val = dp_an_greeks[greek]
        ql_scaled = ql_val * _QL_SCALE[greek]
        logger.info(
            "Asian %s %s %s S=%.0f K=%.0f | DP_MC=%.6f DP_AN=%.6f QL=%.6f",
            averaging.value,
            option_type.value,
            greek,
            spot,
            strike,
            dp_val,
            dp_an_val,
            ql_scaled,
        )
        assert np.isclose(dp_val, ql_scaled, rtol=mc_tols[greek], atol=1e-4), (
            f"{greek}: DP_MC {dp_val:.6f} vs QL {ql_scaled:.6f}"
        )
        assert np.isclose(dp_an_val, ql_scaled, rtol=an_tols[greek], atol=1e-4), (
            f"{greek}: DP_AN {dp_an_val:.6f} vs QL {ql_scaled:.6f}"
        )
