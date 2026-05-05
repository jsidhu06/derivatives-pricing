"""Helper functions and classes for derivatives valuation."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from math import comb
from typing import TYPE_CHECKING
import time
import numpy as np

from .enums import DayCountConvention, OptionType
from .exceptions import ArbitrageViolationError, ConfigurationError, ValidationError

if TYPE_CHECKING:
    from .rates import DiscountCurve

__all__ = [
    "log_timing",
    "validate_naive_datetime",
    "coerce_positive_float",
    "calculate_year_fraction",
    "pv_discrete_dividends",
    "forward_price",
    "put_call_parity_rhs",
    "put_call_parity_gap",
    "binomial_pmf",
    "expected_binomial",
    "expected_binomial_payoff",
]

SECONDS_IN_DAY = 86400


def validate_naive_datetime(
    value: datetime,
    field_name: str,
    *,
    type_error_cls: type[Exception] = ValidationError,
) -> None:
    """Validate that a datetime field is timezone-naive."""
    if not isinstance(value, datetime):
        raise type_error_cls(f"{field_name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is not None:
        raise ValidationError(f"{field_name} must be timezone-naive datetime")


def coerce_positive_float(
    value: object,
    *,
    name: str,
    strict: bool = True,
    allow_none: bool = False,
) -> float:
    """Coerce ``value`` to ``float`` and validate it is finite and positive.

    Centralises the float-coercion / finite / sign-check pattern used by
    dataclasses (``UnderlyingData``, ``GBMParams``, ``BarrierSpec``,
    etc.) so that all fields raise a consistent ``ConfigurationError`` for
    non-numeric input and ``ValidationError`` for non-finite or wrong-sign
    input.

    Parameters
    ----------
    value
        The user-supplied value. May be int, float, or anything ``float()``
        can accept.
    name
        Human-readable field name used in error messages
        (e.g. ``"UnderlyingData.initial_value"``).
    strict
        If True (default) require ``value > 0``; if False require
        ``value >= 0``.  Use ``strict=False`` for fields where zero is a
        meaningful boundary value (volatility, rebate, strike).
    allow_none
        If True, ``None`` returns ``None`` (caller handles default).
        Otherwise raise ``ValidationError``.

    Returns
    -------
    float
        The coerced value.

    Raises
    ------
    ConfigurationError
        If ``value`` cannot be coerced to ``float``.
    ValidationError
        If ``value`` is None (and ``allow_none`` is False), non-finite,
        or violates the sign constraint.
    """
    if value is None:
        if allow_none:
            return None  # type: ignore[return-value]
        raise ValidationError(f"{name} must be provided")
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be numeric") from exc
    if not np.isfinite(x):
        raise ValidationError(f"{name} must be finite")
    if strict and x <= 0.0:
        raise ValidationError(f"{name} must be > 0, got {x}")
    if not strict and x < 0.0:
        raise ValidationError(f"{name} must be >= 0, got {x}")
    return x


@contextmanager
def log_timing(logger, label: str, enabled: bool) -> Iterator[None]:
    """Context manager that logs elapsed time for a code block.

    Parameters
    ----------
    logger
        Logger instance used for debug timing output.
    label
        Label included in the log message.
    enabled
        When ``True``, timing is measured and emitted at ``DEBUG`` level.
    """
    if not enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.debug("Timing %s: %.6fs", label, elapsed)


def _day_count_30_360_us(start_date: datetime, end_date: datetime) -> float:
    """30/360 (US) day-count fraction between two dates."""
    y1, m1, d1 = start_date.year, start_date.month, start_date.day
    y2, m2, d2 = end_date.year, end_date.month, end_date.day

    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 in (30, 31):
        d2 = 30

    return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0


def calculate_year_fraction(
    start_date,
    end_date,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    """Calculate year fraction between two dates.

    Parameters
    ----------
    start_date
        Start date.
    end_date
        End date.
    day_count_convention
        Day-count basis. Supported values are ``ACT_365F``, ``ACT_360``,
        ``ACT_365_25``, and ``THIRTY_360_US``.

    Returns
    -------
    float
        Year fraction between ``start_date`` and ``end_date``.

    Examples
    --------
    >>> from datetime import datetime
    >>> start = datetime(2025, 1, 1)
    >>> end = datetime(2025, 1, 2)
    >>> calculate_year_fraction(start, end)  # doctest: +SKIP
    0.00273972...
    """
    validate_naive_datetime(start_date, "start_date")
    validate_naive_datetime(end_date, "end_date")

    if day_count_convention is DayCountConvention.THIRTY_360_US:
        return _day_count_30_360_us(start_date, end_date)
    if day_count_convention is DayCountConvention.ACT_360:
        denom = 360.0
    elif day_count_convention is DayCountConvention.ACT_365_25:
        denom = 365.25
    elif day_count_convention is DayCountConvention.ACT_365F:
        denom = 365.0
    else:
        raise ValidationError(f"Unsupported day_count_convention: {day_count_convention}")

    delta_days = (end_date - start_date).total_seconds() / SECONDS_IN_DAY
    year_fraction = delta_days / denom
    return year_fraction


def pv_discrete_dividends(
    dividends: Sequence[tuple[datetime, float]],
    curve_date: datetime,
    end_date: datetime,
    discount_curve: DiscountCurve,
    start_date: datetime | None = None,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
    include_start: bool = True,
) -> float:
    """Present value of discrete cash dividends in a date range.

    Parameters
    ----------
    include_start : bool
        When True (default), the range is *closed* on the left:
        ``start_date <= ex_date <= end_date``.  This is the correct
        convention when computing the clean spot (QuantLib-compatible:
        the input spot is treated as cum-dividend).

        When False the range is *open* on the left:
        ``start_date < ex_date <= end_date``.  Use this for the
        escrowed-dividend add-back at intermediate tree nodes, where a
        dividend going ex at exactly *t* has already left the stock
        price.
    """
    if not dividends:
        return 0.0

    start_date = curve_date if start_date is None else start_date
    t_start = calculate_year_fraction(curve_date, start_date, day_count_convention)
    df_start = float(discount_curve.df(t_start))

    pv = 0.0
    for ex_date, amount in dividends:
        if include_start:
            in_range = start_date <= ex_date <= end_date
        else:
            in_range = start_date < ex_date <= end_date
        if in_range:
            t_ex = calculate_year_fraction(curve_date, ex_date, day_count_convention)
            df_ex = float(discount_curve.df(t_ex))
            df = df_ex / df_start
            pv += float(amount) * df
    return float(pv)


def forward_price(
    *,
    spot: float,
    pricing_date: datetime,
    maturity: datetime,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: list[tuple[datetime, float]] | None = None,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    """Compute the no-arbitrage forward price for an equity-style underlying.

    Supports continuous dividend curve, discrete dividends, or both.  When both
    are provided the continuous yield enters the cost-of-carry and the PV of
    discrete dividends is subtracted from spot.

    Parameters
    ----------
    spot
        Spot price at ``pricing_date``.
    pricing_date
        Curve/valuation anchor date.
    maturity
        Forward maturity date.
    discount_curve
        Risk-free discount curve.
    dividend_curve
        Optional continuous dividend discount curve.
    discrete_dividends
        Optional discrete dividends as ``(ex_date, amount)`` pairs.
    day_count_convention
        Day-count convention used for year-fraction conversion.

    Returns
    -------
    float
        No-arbitrage forward price.
    """
    if dividend_curve is not None and discrete_dividends:
        warnings.warn(
            "Both dividend_curve and discrete_dividends provided to forward_price. "
            "The continuous yield will enter the cost-of-carry and discrete "
            "dividends will be subtracted from spot.",
            stacklevel=2,
        )

    t = calculate_year_fraction(pricing_date, maturity, day_count_convention)
    if t <= 0:
        raise ValidationError("maturity must be after pricing_date")

    spot = float(spot)

    # PV of discrete dividends (0 if none)
    pv_divs = 0.0
    if discrete_dividends:
        pv_divs = pv_discrete_dividends(
            discrete_dividends,
            curve_date=pricing_date,
            end_date=maturity,
            discount_curve=discount_curve,
            day_count_convention=day_count_convention,
        )

    # Continuous dividend yield discount factor (1 if none)
    df_q = 1.0 if dividend_curve is None else float(dividend_curve.df(t))
    df_r = float(discount_curve.df(t))

    fwd = (spot * df_q - pv_divs) / df_r
    return float(fwd)


def put_call_parity_rhs(
    *,
    spot: float,
    strike: float,
    pricing_date: datetime,
    maturity: datetime,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: list[tuple[datetime, float]] | None = None,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    """Compute the RHS of put-call parity for European options.

    Returns C - P implied by no-arbitrage (i.e., PV(F - K)).
    """
    t = calculate_year_fraction(pricing_date, maturity, day_count_convention)
    if t <= 0:
        raise ValidationError("maturity must be after pricing_date")

    fwd = forward_price(
        spot=spot,
        pricing_date=pricing_date,
        maturity=maturity,
        discount_curve=discount_curve,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
        day_count_convention=day_count_convention,
    )
    df_r = float(discount_curve.df(t))
    return float(df_r * (fwd - float(strike)))


def put_call_parity_gap(
    *,
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    pricing_date: datetime,
    maturity: datetime,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: list[tuple[datetime, float]] | None = None,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    """Return call-put parity residual: (C - P) - RHS."""
    rhs = put_call_parity_rhs(
        spot=spot,
        strike=strike,
        pricing_date=pricing_date,
        maturity=maturity,
        discount_curve=discount_curve,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
        day_count_convention=day_count_convention,
    )
    return float(call_price - put_price - rhs)


def binomial_pmf(k: np.ndarray | int, n: int, p: float) -> np.ndarray:
    """Binomial(n, p) probability mass function.

    Parameters
    ----------
    k
        Success count(s). Can be an integer or an integer array.
    n
        Number of trials, must be non-negative.
    p
        Success probability in ``[0, 1]``.
    """
    if n < 0:
        raise ValidationError("n must be >= 0")

    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValidationError("p must be in [0, 1]")

    k_arr = np.asarray(k, dtype=int)
    out = np.zeros_like(k_arr, dtype=float)

    in_support = (k_arr >= 0) & (k_arr <= n)
    if not np.any(in_support):
        return out

    ks = k_arr[in_support]
    combs = np.array([comb(n, int(kk)) for kk in ks], dtype=float)
    out[in_support] = combs * (p**ks) * ((1.0 - p) ** (n - ks))
    return out


def expected_binomial(
    n: int,
    p: float,
    f: Callable[[np.ndarray], np.ndarray],
) -> float:
    r"""Compute $\mathbb{E}[f(K)]$ where $K \sim \text{Binomial}(n, p)$.

    This is a convenience wrapper around the explicit sum
    $\sum_{k=0}^n \binom{n}{k} p^k (1-p)^{n-k} f(k)$.

    Parameters
    ----------
    n
        Number of Bernoulli trials.
    p
        Success probability.
    f
        Vectorized function evaluated on ``k = 0, ..., n``.

    Returns
    -------
    float
        Expected value under the ``Binomial(n, p)`` distribution.
    """
    if n < 0:
        raise ValidationError("n must be >= 0")
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValidationError("p must be in [0, 1]")

    ks = np.arange(n + 1)
    pmf = binomial_pmf(ks, n=n, p=p)
    vals = np.asarray(f(ks), dtype=float)
    if vals.shape != ks.shape:
        raise ValidationError("f(k) must return an array with same shape as k")
    return float(np.dot(pmf, vals))


def expected_binomial_payoff(
    *,
    S0: float,
    n: int,
    T: float,
    option_type: OptionType,
    K: float,
    r: float,
    q: float,
    u: float,
) -> float:
    r"""Expected vanilla payoff under a binomial terminal distribution.

    Parameters
    ----------
    S0
        Initial spot.
    n
        Number of binomial time steps.
    T
        Time to maturity in years.
    option_type
        Option payoff direction (call or put).
    K
        Strike price.
    r
        Continuously-compounded risk-free rate.
    q
        Continuously-compounded dividend yield.
    u
        Up multiplier per step. Down multiplier is set to ``d = 1 / u``.

    Notes
    -----
    In a CRR-style model, the terminal spot is $S_T(k) = S_0 u^k d^{n-k}$
    with risk-neutral probability
    $p = (e^{(r-q)\Delta t} - d) / (u - d)$.
    This function returns the expected terminal payoff (not discounted).
    """
    if option_type not in (OptionType.CALL, OptionType.PUT):
        raise ValidationError("option_type must be OptionType.CALL or OptionType.PUT")

    if n < 1:
        raise ValidationError("n must be >= 1")
    if T <= 0:
        raise ValidationError("T must be positive")

    S0 = float(S0)
    K = float(K)
    r = float(r)
    q = float(q)

    u = float(u)
    if u <= 0:
        raise ValidationError("u must be positive")
    d = 1.0 / u

    def payoff_from_k(ks: np.ndarray) -> np.ndarray:
        ST = S0 * (u**ks) * (d ** (n - ks))
        if option_type is OptionType.CALL:
            return np.maximum(ST - K, 0.0)
        return np.maximum(K - ST, 0.0)

    delta_t = T / n
    p = (np.exp((r - q) * delta_t) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ArbitrageViolationError("risk-neutral probability p must be in [0, 1]")

    return expected_binomial(n=n, p=p, f=payoff_from_k)
