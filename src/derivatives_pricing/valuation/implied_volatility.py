"""Implied volatility solvers for European and American options."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging

import numpy as np
from scipy import optimize

from ..enums import (
    ExerciseType,
    ImpliedVolMethod,
    OptionType,
    PricingMethod,
    GreekCalculationMethod,
)
from ..utils import calculate_year_fraction, pv_discrete_dividends, log_timing
from ..exceptions import (
    ConfigurationError,
    ConvergenceError,
    UnsupportedFeatureError,
    ValidationError,
)
from .contracts import BarrierSpec, PayoffSpec
from .core import OptionValuation, UnderlyingData
from .params import BinomialParams


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    """Result container for implied volatility calculation."""

    implied_vol: float
    iterations: int
    converged: bool


def _adjusted_spot_and_dividend_df(valuation: OptionValuation) -> tuple[float, float]:
    """Return dividend-adjusted spot and continuous-dividend discount factor.

    Parameters
    ----------
    valuation
        Valuation object containing underlying, dates, and discount curves.

    Returns
    -------
    tuple[float, float]
        ``(spot_adjusted_for_discrete_dividends, df_q)``.
    """
    spot = float(valuation.underlying.initial_value)
    dividend_curve = valuation.underlying.dividend_curve
    discrete_dividends = valuation.underlying.discrete_dividends
    if not discrete_dividends:
        if dividend_curve is None:
            return spot, 1.0
        time_to_maturity = calculate_year_fraction(
            valuation.pricing_date,
            valuation.maturity,
            day_count_convention=valuation.day_count_convention,
        )
        return spot, float(dividend_curve.df(time_to_maturity))
    pv_divs = pv_discrete_dividends(
        discrete_dividends,
        curve_date=valuation.pricing_date,
        end_date=valuation.maturity,
        discount_curve=valuation.discount_curve,
        day_count_convention=valuation.day_count_convention,
    )
    if dividend_curve is None:
        df_q = 1.0
    else:
        time_to_maturity = calculate_year_fraction(
            valuation.pricing_date,
            valuation.maturity,
            day_count_convention=valuation.day_count_convention,
        )
        df_q = float(dividend_curve.df(time_to_maturity))
    return max(spot - pv_divs, 0.0), df_q


def _price_bounds(valuation: OptionValuation) -> tuple[float, float]:
    """Compute no-arbitrage lower/upper bounds for option price.

    Parameters
    ----------
    valuation
        Option valuation describing contract style and market inputs.

    Returns
    -------
    tuple[float, float]
        Price interval ``(lower, upper)``.
    """
    time_to_maturity = calculate_year_fraction(
        valuation.pricing_date,
        valuation.maturity,
        day_count_convention=valuation.day_count_convention,
    )
    if time_to_maturity <= 0:
        raise ValidationError("Option maturity must be after pricing date.")

    strike = float(valuation.strike)

    if valuation.exercise_type is ExerciseType.AMERICAN:
        spot = float(valuation.underlying.initial_value)
        if valuation.option_type is OptionType.CALL:
            lower = max(0.0, spot - strike)
            upper = spot
        else:  # PUT
            lower = max(0.0, strike - spot)
            upper = strike
        return lower, upper

    spot, df_q = _adjusted_spot_and_dividend_df(valuation)
    df_r = float(valuation.discount_curve.df(time_to_maturity))

    if valuation.option_type is OptionType.CALL:
        lower = max(0.0, spot * df_q - strike * df_r)
        upper = spot * df_q
    else:  # PUT
        lower = max(0.0, strike * df_r - spot * df_q)
        upper = strike * df_r

    return lower, upper


def _valuation_with_vol(valuation: OptionValuation, vol: float) -> OptionValuation:
    """Clone valuation with bumped volatility."""
    if not isinstance(valuation.underlying, UnderlyingData):
        raise ConfigurationError("Implied volatility requires UnderlyingData (not PathSimulation).")

    bumped_underlying = valuation.underlying.replace(volatility=float(vol))
    return OptionValuation(
        underlying=bumped_underlying,
        spec=valuation.spec,
        pricing_method=valuation.pricing_method,
        params=valuation.params,
    )


def _non_bsm_initial_guess(
    valuation: OptionValuation,
    target_price: float,
    low: float,
    high: float,
) -> float:
    """Heuristic initial volatility for non-BSM implied-vol solves."""
    spot = float(valuation.underlying.initial_value)
    strike = float(valuation.strike)
    time_to_maturity = calculate_year_fraction(
        valuation.pricing_date,
        valuation.maturity,
        day_count_convention=valuation.day_count_convention,
    )
    time_sqrt = np.sqrt(max(time_to_maturity, 1.0e-8))

    if valuation.option_type is OptionType.CALL:
        intrinsic = max(spot - strike, 0.0)
    else:
        intrinsic = max(strike - spot, 0.0)

    extrinsic = max(target_price - intrinsic, 0.0)
    scale = max(spot, strike, 1.0)
    rough = 0.2 + 0.8 * (extrinsic / scale) / time_sqrt
    return float(np.clip(rough, low, high))


def _bracket_volatility(
    *,
    f: Callable[[float], float],
    low: float,
    high: float,
    max_expansions: int = 6,
) -> tuple[float, float, float, float]:
    """Expand volatility interval until target residual is bracketed."""
    f_low = f(low)
    f_high = f(high)

    if f_low > 0:
        for _ in range(max_expansions):
            low = max(low / 2.0, 1.0e-12)
            f_low = f(low)
            if f_low <= 0:
                break

    if f_high < 0:
        for _ in range(max_expansions):
            high *= 2.0
            f_high = f(high)
            if f_high >= 0:
                break

    return low, high, f_low, f_high


def _newton_raphson(
    *,
    f: Callable[[float], float],
    vega: Callable[[float], float],
    low: float,
    high: float,
    initial: float,
    tol: float,
    max_iter: int,
) -> ImpliedVolResult:
    """Run safeguarded Newton-Raphson updates for implied volatility."""
    vol = float(initial)
    iterations = 0

    for i in range(max_iter):
        iterations = i + 1
        diff = f(vol)
        if abs(diff) <= tol:
            return ImpliedVolResult(implied_vol=vol, iterations=iterations, converged=True)

        slope = vega(vol)
        if slope <= 0 or not np.isfinite(slope):
            break

        step = diff / slope
        candidate = vol - step

        if not np.isfinite(candidate) or candidate <= low or candidate >= high:
            candidate = 0.5 * (low + high)

        if diff > 0:
            high = min(high, vol)
        else:
            low = max(low, vol)

        if abs(high - low) <= tol:
            return ImpliedVolResult(implied_vol=candidate, iterations=iterations, converged=True)

        vol = candidate

    return ImpliedVolResult(implied_vol=vol, iterations=iterations, converged=False)


def _bisection(
    *,
    f: Callable[[float], float],
    low: float,
    high: float,
    tol: float,
    max_iter: int,
) -> ImpliedVolResult:
    """Run bisection on the implied-vol residual function."""
    f_low = f(low)
    f_high = f(high)

    if f_low > 0 or f_high < 0:
        raise ConvergenceError("Price not bracketed by vol_bounds; adjust bounds.")

    vol = 0.5 * (low + high)
    for i in range(max_iter):
        f_mid = f(vol)
        if abs(f_mid) <= tol or abs(high - low) <= tol:
            return ImpliedVolResult(implied_vol=vol, iterations=i + 1, converged=True)
        if f_mid > 0:
            high = vol
        else:
            low = vol
        vol = 0.5 * (low + high)

    return ImpliedVolResult(implied_vol=vol, iterations=max_iter, converged=False)


def implied_volatility(
    target_price: float,
    valuation: OptionValuation,
    method: ImpliedVolMethod = ImpliedVolMethod.NEWTON_RAPHSON,
    *,
    initial_vol: float | None = None,
    vol_bounds: tuple[float, float] = (1.0e-6, 5.0),
    tol: float = 1.0e-8,
    max_iter: int = 100,
    log_timings: bool = False,
) -> ImpliedVolResult:
    """Solve for implied volatility using BSM, binomial, or PDE pricing.

    Parameters
    ----------
    target_price
        Observed option price per unit (not multiplied by contract size).
    valuation
        Option valuation configured for BSM (European) or
        BINOMIAL/PDE_FD (European/American).
    method
        Root-finding method.
    initial_vol
        Optional initial annualized volatility guess. If ``None``, a
        model-specific default/heuristic is used.
    vol_bounds
        Lower/upper bounds for the volatility search interval.
    tol
        Absolute tolerance for the pricing residual.
    max_iter
        Maximum number of iterations for iterative solvers.
    log_timings
        When ``True``, emit timing logs for the solver section.

    Notes
    -----
    Vega is expected in per-1% volatility terms. The solver rescales vega to a
    per-1.0 volatility derivative when computing Newton updates.

    Returns
    -------
    ImpliedVolResult
        Solver output including implied volatility, iteration count, and
        convergence status.
    """

    if not isinstance(valuation, OptionValuation):
        raise ConfigurationError("valuation must be an OptionValuation instance")
    if isinstance(valuation.spec, PayoffSpec):
        raise UnsupportedFeatureError(
            "Implied volatility is not supported for PayoffSpec: "
            "P(sigma) monotonicity depends on the user-supplied payoff "
            "function and cannot be guaranteed in general."
        )
    if isinstance(valuation.spec, BarrierSpec):
        raise UnsupportedFeatureError(
            "Implied volatility is not supported for barrier specs: "
            "P(sigma) is not guaranteed to be monotonic in sigma."
        )
    if valuation.pricing_method is PricingMethod.BSM:
        if valuation.exercise_type is not ExerciseType.EUROPEAN:
            raise UnsupportedFeatureError("BSM implied volatility supports European options only.")
    elif valuation.pricing_method not in (PricingMethod.BINOMIAL, PricingMethod.PDE_FD):
        raise UnsupportedFeatureError(
            "Implied volatility supports BSM, BINOMIAL, or PDE_FD pricing methods only."
        )

    if not np.isfinite(target_price):
        raise ValidationError("target_price must be finite")
    if target_price < 0:
        raise ValidationError("target_price must be non-negative")

    low, high = vol_bounds
    if low <= 0 or high <= 0 or low >= high:
        raise ValidationError("vol_bounds must be positive and satisfy low < high")

    if valuation.pricing_method is PricingMethod.BINOMIAL and isinstance(
        valuation.params, BinomialParams
    ):
        time_to_maturity = calculate_year_fraction(
            valuation.pricing_date,
            valuation.maturity,
            day_count_convention=valuation.day_count_convention,
        )
        dt = time_to_maturity / max(int(valuation.params.num_steps), 1)
        df_r = float(valuation.discount_curve.df(time_to_maturity))
        r = -np.log(df_r) / time_to_maturity
        dividend_curve = valuation.underlying.dividend_curve
        if dividend_curve is None:
            q = 0.0
        else:
            df_q = float(dividend_curve.df(time_to_maturity))
            q = -np.log(df_q) / time_to_maturity
        drift = r - q
        sigma_min = abs(drift) * np.sqrt(max(dt, 1.0e-12))
        low = max(low, sigma_min * 1.01)

    min_price, max_price = _price_bounds(valuation)
    if target_price < min_price - tol or target_price > max_price + tol:
        raise ValidationError("target_price is outside no-arbitrage bounds for the provided inputs")

    def price_at(vol: float) -> float:
        return _valuation_with_vol(valuation, vol).present_value()

    def f(vol: float) -> float:
        return price_at(vol) - target_price

    def vega_at(vol: float) -> float:
        val = _valuation_with_vol(valuation, vol)
        greek_method = (
            GreekCalculationMethod.ANALYTICAL
            if valuation.pricing_method is PricingMethod.BSM
            else GreekCalculationMethod.NUMERICAL
        )
        vega = val.vega(greek_calc_method=greek_method)
        return float(vega) * 100.0  # convert from per-1% to per-1.0 volatility

    if initial_vol is not None:
        initial = float(initial_vol)
    elif valuation.pricing_method is PricingMethod.BSM:
        initial = float(valuation.underlying.volatility or 0.2)
    else:
        initial = _non_bsm_initial_guess(valuation, target_price, low, high)

    initial = max(low, min(high, initial))

    low, high, f_low, f_high = _bracket_volatility(f=f, low=low, high=high)
    if f_low > 0 or f_high < 0:
        raise ConvergenceError("Price not bracketed by vol_bounds; adjust bounds.")

    with log_timing(logger, "Implied vol solver", log_timings):
        if method is ImpliedVolMethod.NEWTON_RAPHSON:
            result = _newton_raphson(
                f=f,
                vega=vega_at,
                low=low,
                high=high,
                initial=initial,
                tol=tol,
                max_iter=max_iter,
            )
            if not result.converged:
                result = _bisection(f=f, low=low, high=high, tol=tol, max_iter=max_iter)
        elif method is ImpliedVolMethod.BISECTION:
            result = _bisection(f=f, low=low, high=high, tol=tol, max_iter=max_iter)
        elif method is ImpliedVolMethod.BRENTQ:
            implied, r = optimize.brentq(f, low, high, xtol=tol, maxiter=max_iter, full_output=True)
            result = ImpliedVolResult(
                implied_vol=float(implied),
                iterations=r.iterations,
                converged=True,
            )
        else:
            raise UnsupportedFeatureError(
                f"Implied vol method '{method.value}' is not implemented in this solver."
            )

    logger.debug(
        "Implied vol method=%s converged=%s iterations=%d implied_vol=%.6g",
        method.value,
        result.converged,
        result.iterations,
        result.implied_vol,
    )

    return result
