"""Implied volatility solvers for European and American options."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import warnings

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
from .binomial import _crr_min_volatility


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    """Result container for implied volatility calculation.

    Attributes
    ----------
    implied_vol
        Solved volatility (annualised).
    iterations
        Iterations of the final root finder (Newton steps, bisection
        halvings or Brent iterations); 0 when the root sat on a bracket end.
    converged
        True when the root finder located the volatility to within ``tol``.
    function_evaluations
        Model pricings used in total: bracketing, solving and the final
        conditioning check, each numerical-vega bump counted. For tree and
        PDE engines this, not ``iterations``, measures the cost.
    well_determined
        True when vega at the solution is at least ``min_vega``, i.e. the
        price actually pins the volatility down. A converged but not
        well-determined result reprices the target, yet small rounding in
        the price would move it a lot (far out of the money, near intrinsic).
    """

    implied_vol: float
    iterations: int
    converged: bool
    function_evaluations: int = 0
    well_determined: bool = True


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
        dividend_curve=dividend_curve,
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
    """Try to widen ``[low, high]`` until the residual changes sign across it.

    Halves ``low`` (floored at 1e-12) while the price there is still above the
    target, doubles ``high`` while it is still below, at most
    ``max_expansions`` times each. The returned interval may still fail to
    bracket the root; the caller checks the residual signs.
    """
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
    residual_and_vega: Callable[[float], tuple[float, float]],
    low: float,
    high: float,
    initial: float,
    tol: float,
    max_iter: int,
) -> ImpliedVolResult:
    """Safeguarded Newton-Raphson on the price residual, converging in vol.

    Price is increasing in vol, so each residual's sign shrinks the bracket
    ``[low, high]`` around the root. A Newton step that leaves the bracket
    (or a non-positive vega) is replaced by the bracket midpoint. Converged
    when the Newton step or the bracket is within ``tol`` in vol.
    """
    vol = float(initial)
    for i in range(max_iter):
        iterations = i + 1
        diff, slope = residual_and_vega(vol)
        if diff == 0.0:
            return ImpliedVolResult(implied_vol=vol, iterations=iterations, converged=True)
        if diff > 0:
            high = min(high, vol)
        else:
            low = max(low, vol)

        candidate = float("nan")
        if slope > 0 and np.isfinite(slope):
            step = diff / slope
            candidate = vol - step
            if abs(step) <= tol and low <= candidate <= high:
                return ImpliedVolResult(
                    implied_vol=candidate, iterations=iterations, converged=True
                )
        if high - low <= tol:
            return ImpliedVolResult(
                implied_vol=0.5 * (low + high), iterations=iterations, converged=True
            )
        if not np.isfinite(candidate) or candidate <= low or candidate >= high:
            candidate = 0.5 * (low + high)
        vol = candidate

    return ImpliedVolResult(implied_vol=vol, iterations=max_iter, converged=False)


def _bisection(
    *,
    f: Callable[[float], float],
    low: float,
    high: float,
    tol: float,
    max_iter: int,
) -> ImpliedVolResult:
    """Bisection on the price residual; converged once the vol bracket is within ``tol``."""
    for i in range(max_iter):
        vol = 0.5 * (low + high)
        if high - low <= tol:
            return ImpliedVolResult(implied_vol=vol, iterations=i, converged=True)
        f_mid = f(vol)
        if f_mid == 0.0:
            return ImpliedVolResult(implied_vol=vol, iterations=i + 1, converged=True)
        if f_mid > 0:
            high = vol
        else:
            low = vol

    vol = 0.5 * (low + high)
    return ImpliedVolResult(implied_vol=vol, iterations=max_iter, converged=high - low <= tol)


def implied_volatility(
    target_price: float,
    valuation: OptionValuation,
    method: ImpliedVolMethod | None = None,
    *,
    initial_vol: float | None = None,
    vol_bounds: tuple[float, float] = (1.0e-6, 5.0),
    tol: float = 1.0e-8,
    max_iter: int = 100,
    min_vega: float | None = None,
    log_timings: bool = False,
) -> ImpliedVolResult:
    """Solve for implied volatility using ANALYTICAL, BINOMIAL, or PDE_FD pricing.

    Parameters
    ----------
    target_price
        Observed option price per unit (not multiplied by contract size).
    valuation
        Option valuation configured for ANALYTICAL (European) or
        BINOMIAL/PDE_FD (European/American).
    method
        Root-finding method. ``None`` (default) picks by engine: safeguarded
        Newton-Raphson for ANALYTICAL (closed-form vega is cheap), Brent for
        BINOMIAL and PDE_FD (no vega needed; a numerical vega costs two extra
        pricings per Newton step, so Brent is ~30% faster in wall time).
    initial_vol
        Starting point for Newton-Raphson. If ``None``: the valuation's own
        volatility for ANALYTICAL, an extrinsic-value heuristic otherwise.
        Ignored by the bracketing methods.
    vol_bounds
        Initial search interval for the volatility, widened (halving the
        lower bound, doubling the upper, up to 6 times each) until it
        brackets the target.
    tol
        Absolute tolerance on the implied volatility (annualised vol units,
        e.g. ``1e-8`` = 1e-6 vol points), the same for every method: Newton
        stops once its step or its safeguarding bracket is within ``tol``,
        bisection and Brent once the bracket is. A price tolerance would
        not do: the vol error is the price error divided by vega, so the
        same price tolerance is loose far from the money and tight near it.
    max_iter
        Maximum number of iterations for iterative solvers.
    min_vega
        Smallest vega (price change per unit of vol, i.e. per 100 vol
        points) at the solution for which the price is taken to determine
        the vol. Below it the result has ``well_determined=False`` and a
        warning is issued. Defaults to ``1e-6 * max(spot, strike)``; pass
        ``0`` to disable.
    log_timings
        When ``True``, emit timing logs for the solver section.

    Returns
    -------
    ImpliedVolResult
        ``converged`` reports the root finder; ``well_determined`` reports
        whether the price identifies the vol. Use the vol only when both
        are true.
    """
    _validate_problem(valuation, target_price, vol_bounds, tol, max_iter, initial_vol, min_vega)
    low, high = vol_bounds
    if valuation.pricing_method is PricingMethod.BINOMIAL:
        # Below this the CRR tree's no-arbitrage condition fails at some step.
        low = max(low, 1.01 * _crr_min_volatility(valuation, valuation.underlying.dividend_curve))
        if low >= high:
            raise ValidationError(
                f"vol_bounds upper {high} is below the CRR tree's minimum admissible "
                f"volatility {low:.6g}; raise it or use more num_steps."
            )

    min_price, max_price = _price_bounds(valuation)
    # Rounding slack only: a quote outside the bounds has no implied vol.
    bound_slack = 1.0e-12 * max(1.0, max_price)
    if target_price < min_price - bound_slack or target_price > max_price + bound_slack:
        raise ValidationError("target_price is outside no-arbitrage bounds for the provided inputs")

    if method is None:
        method = (
            ImpliedVolMethod.NEWTON_RAPHSON
            if valuation.pricing_method is PricingMethod.ANALYTICAL
            else ImpliedVolMethod.BRENTQ
        )
    numerical_vega = valuation.pricing_method is not PricingMethod.ANALYTICAL
    vega_method = (
        GreekCalculationMethod.NUMERICAL if numerical_vega else GreekCalculationMethod.ANALYTICAL
    )
    pricings = 0

    def f(vol: float) -> float:
        nonlocal pricings
        pricings += 1
        return _valuation_with_vol(valuation, vol).present_value() - target_price

    def vega_of(val: OptionValuation) -> float:
        # Per-1% vega rescaled to per-1.0 vol (dPrice/dsigma). A numerical
        # vega reprices twice (central bump).
        nonlocal pricings
        pricings += 2 if numerical_vega else 0
        return float(val.vega(greek_calc_method=vega_method)) * 100.0

    def residual_and_vega(vol: float) -> tuple[float, float]:
        # One valuation per vol: its memoised PV is reused by numerical vega.
        nonlocal pricings
        pricings += 1
        val = _valuation_with_vol(valuation, vol)
        return val.present_value() - target_price, vega_of(val)

    low, high, f_low, f_high = _bracket_volatility(f=f, low=low, high=high)
    if f_low > 0 or f_high < 0:
        raise ConvergenceError(
            f"Could not bracket the implied volatility: after widening, vol in "
            f"[{low:.6g}, {high:.6g}] gives price residuals [{f_low:.6g}, {f_high:.6g}] "
            "(need <= 0 at the low end and >= 0 at the high end). Widen vol_bounds."
        )

    with log_timing(logger, "Implied vol solver", log_timings):
        if f_low == 0.0 or f_high == 0.0:
            # The root sits on a bracket end: nothing to solve.
            vol = low if f_low == 0.0 else high
            result = ImpliedVolResult(implied_vol=vol, iterations=0, converged=True)
        else:
            result = _solve_root(
                method=method,
                f=f,
                residual_and_vega=residual_and_vega,
                low=low,
                high=high,
                initial=_initial_vol(valuation, target_price, initial_vol, low, high),
                tol=tol,
                max_iter=max_iter,
            )

    well_determined = True
    if result.converged:
        well_determined = _is_well_determined(valuation, result.implied_vol, vega_of, min_vega)
    result = ImpliedVolResult(
        implied_vol=result.implied_vol,
        iterations=result.iterations,
        converged=result.converged,
        function_evaluations=pricings,
        well_determined=well_determined,
    )

    logger.debug(
        "Implied vol method=%s converged=%s well_determined=%s iterations=%d "
        "function_evaluations=%d implied_vol=%.6g",
        method.value,
        result.converged,
        result.well_determined,
        result.iterations,
        result.function_evaluations,
        result.implied_vol,
    )
    return result


def _validate_problem(
    valuation: OptionValuation,
    target_price: float,
    vol_bounds: tuple[float, float],
    tol: float,
    max_iter: int,
    initial_vol: float | None,
    min_vega: float | None,
) -> None:
    """Reject unsupported contracts/engines and invalid solver settings."""
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
    if valuation.pricing_method is PricingMethod.ANALYTICAL:
        if valuation.exercise_type is not ExerciseType.EUROPEAN:
            raise UnsupportedFeatureError(
                "ANALYTICAL implied volatility supports European options only."
            )
    elif valuation.pricing_method not in (PricingMethod.BINOMIAL, PricingMethod.PDE_FD):
        raise UnsupportedFeatureError(
            "Implied volatility supports ANALYTICAL, BINOMIAL, or PDE_FD pricing methods only."
        )
    if not np.isfinite(target_price):
        raise ValidationError("target_price must be finite")
    if target_price < 0:
        raise ValidationError("target_price must be non-negative")
    low, high = vol_bounds
    if low <= 0 or high <= 0 or low >= high:
        raise ValidationError("vol_bounds must be positive and satisfy low < high")
    if not tol > 0:
        raise ValidationError(f"tol must be positive, got {tol}")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValidationError(f"max_iter must be a positive integer, got {max_iter!r}")
    if initial_vol is not None and not (np.isfinite(initial_vol) and initial_vol > 0):
        raise ValidationError(f"initial_vol must be finite and positive, got {initial_vol}")
    if min_vega is not None and not min_vega >= 0:
        raise ValidationError(f"min_vega must be >= 0, got {min_vega}")


def _initial_vol(
    valuation: OptionValuation,
    target_price: float,
    initial_vol: float | None,
    low: float,
    high: float,
) -> float:
    """Newton starting point inside the final bracket."""
    if initial_vol is not None:
        initial = float(initial_vol)
    elif valuation.pricing_method is PricingMethod.ANALYTICAL:
        initial = float(valuation.underlying.volatility or 0.2)
    else:
        initial = _non_bsm_initial_guess(valuation, target_price, low, high)
    return max(low, min(high, initial))


def _solve_root(
    *,
    method: ImpliedVolMethod,
    f: Callable[[float], float],
    residual_and_vega: Callable[[float], tuple[float, float]],
    low: float,
    high: float,
    initial: float,
    tol: float,
    max_iter: int,
) -> ImpliedVolResult:
    """Run the chosen root finder on a bracket with a strict sign change."""
    if method is ImpliedVolMethod.NEWTON_RAPHSON:
        result = _newton_raphson(
            residual_and_vega=residual_and_vega,
            low=low,
            high=high,
            initial=initial,
            tol=tol,
            max_iter=max_iter,
        )
        if result.converged:
            return result
        return _bisection(f=f, low=low, high=high, tol=tol, max_iter=max_iter)
    if method is ImpliedVolMethod.BISECTION:
        return _bisection(f=f, low=low, high=high, tol=tol, max_iter=max_iter)
    if method is ImpliedVolMethod.BRENTQ:
        implied, r = optimize.brentq(
            f, low, high, xtol=tol, maxiter=max_iter, full_output=True, disp=False
        )
        return ImpliedVolResult(
            implied_vol=float(implied), iterations=int(r.iterations), converged=bool(r.converged)
        )
    raise UnsupportedFeatureError(
        f"Implied vol method '{method.value}' is not implemented in this solver."
    )


def _is_well_determined(
    valuation: OptionValuation,
    vol: float,
    vega_of: Callable[[OptionValuation], float],
    min_vega: float | None,
) -> bool:
    """Whether vega at the solution is large enough for the price to fix the vol."""
    floor = (
        1.0e-6 * max(float(valuation.underlying.initial_value), float(valuation.strike))
        if min_vega is None
        else min_vega
    )
    vega = vega_of(_valuation_with_vol(valuation, vol))
    if vega >= floor:
        return True
    warnings.warn(
        f"Implied vol {vol:.6g} is not determined by the price: vega {vega:.3g} per unit vol "
        f"is below min_vega={floor:.3g}, so rounding in the price is a large vol error "
        "(typical far out of the money or near intrinsic). Returning well_determined=False.",
        stacklevel=3,
    )
    return False
