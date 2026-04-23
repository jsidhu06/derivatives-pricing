"""Analytical barrier option valuation.

Implements closed-form barrier pricing for all 8 standard barrier types:
{up, down} × {in, out} × {call, put}.

Supports:
- Continuous monitoring (default)
- Discrete monitoring via Broadie-Glasserman-Kou continuity correction
- Non-flat rate and dividend term structures
- Rebates: paid at hit (knock-out) or at expiry (knock-in / knock-out)

Current scope
-------------
- European exercise only
- Single barrier (no double-barrier)
- GBM dynamics (no jumps or stochastic volatility)

References
----------
Reiner, E. and Rubinstein, M. (1991). "Breaking Down the Barriers",
    *Risk*, 4(8), 28–35.
Hull, J. C. *Options, Futures, and Other Derivatives*, Section 26.9.
Broadie, M., Glasserman, P. and Kou, S. (1997). "A Continuity Correction
    for Discrete Barrier Options", *Mathematical Finance*, 7(4), 325–349.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from ..enums import (
    BarrierAction,
    BarrierDirection,
    BarrierMonitoring,
    OptionType,
    PricingMethod,
    RebateTiming,
)
from ..exceptions import UnsupportedFeatureError
from .contracts import VanillaSpec

if TYPE_CHECKING:
    from .contracts import BarrierSpec
    from .core import OptionValuation, UnderlyingData


# ── Shared helpers ──────────────────────────────────────────────────


def _is_triggered(
    spot: float,
    barrier: float,
    direction: BarrierDirection,
) -> bool:
    """Return ``True`` if the barrier is triggered.

    Parameters
    ----------
    spot
        Current spot price.
    barrier
        Barrier level.
    direction
        UP or DOWN.

    Returns
    -------
    bool
        ``True`` if the barrier is triggered.
    """
    if direction is BarrierDirection.UP:
        return spot >= barrier
    return spot <= barrier


_BG_BETA = 0.5826  # Broadie-Glasserman-Kou constant


def _broadie_glasserman_adjustment(
    barrier: float,
    sigma: float,
    T: float,
    m: int,
    direction: BarrierDirection,
) -> float:
    """Apply Broadie-Glasserman-Kou continuity correction to a barrier level.

    For discrete monitoring with *m* equally spaced observations at
    ``t_i = i · T/m`` for ``i = 1, ..., m`` (i.e. the pricing date ``t = 0``
    is excluded; the set of observation points is ``(0, T]``), the corrected
    barrier is:

        H_adj = H · exp(±β · σ · √(T/m))

    where β ≈ 0.5826 and the sign is ``+`` for UP (shift barrier up) and
    ``-`` for DOWN (shift barrier down).  This makes the discrete-monitoring
    price converge to the continuous-monitoring price at the adjusted level.

    Parameters
    ----------
    barrier
        Original barrier level H.
    sigma
        Annualised volatility σ.
    T
        Time to maturity in years.
    m
        Number of equally spaced monitoring observations.
    direction
        UP or DOWN.

    Returns
    -------
    float
        Adjusted barrier level.
    """
    sign = 1.0 if direction is BarrierDirection.UP else -1.0
    return barrier * np.exp(sign * _BG_BETA * sigma * np.sqrt(T / m))


# ── Rebate pricing ────────────────────────────────────────────────


def _rebate_knock_out_at_hit(
    R: float,
    S: float,
    H: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    direction: BarrierDirection,
) -> float:
    """PV of a rebate paid instantly when a knock-out barrier is hit.

    Uses the one-touch cash-or-nothing digital formula
    (Reiner-Rubinstein / Haug).

    Parameters
    ----------
    R : float
        Rebate amount.
    S, H : float
        Spot and barrier.
    r, q : float
        Risk-free rate and dividend yield (flat-equivalent).
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    direction : BarrierDirection
        UP or DOWN.

    Returns
    -------
    float
        Present value of the at-hit rebate.
    """
    if R == 0.0:
        return 0.0

    sigma2 = sigma**2
    a = (r - q - sigma2 / 2.0) / sigma2
    b = np.sqrt(a**2 + 2.0 * r / sigma2)
    sigma_sqrt_T = sigma * np.sqrt(T)

    eta = 1.0 if direction is BarrierDirection.DOWN else -1.0
    log_HS = np.log(H / S)

    z1 = log_HS / sigma_sqrt_T + b * sigma_sqrt_T
    z2 = log_HS / sigma_sqrt_T - b * sigma_sqrt_T

    return R * ((H / S) ** (a + b) * norm.cdf(eta * z1) + (H / S) ** (a - b) * norm.cdf(eta * z2))


def _rebate_knock_out_at_expiry(
    R: float,
    S: float,
    H: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    df_r: float,
    direction: BarrierDirection,
) -> float:
    """PV of a rebate paid at expiry when a knock-out barrier is hit.

    R · e^{-rT} · P(τ_B ≤ T).

    Parameters
    ----------
    R : float
        Rebate amount.
    S, H : float
        Spot and barrier.
    r, q : float
        Risk-free rate and dividend yield (flat-equivalent).
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    df_r : float
        Discount factor at maturity.
    direction : BarrierDirection
        UP or DOWN.

    Returns
    -------
    float
        Present value of the at-expiry knock-out rebate.
    """
    if R == 0.0:
        return 0.0

    sigma2 = sigma**2
    # Log-spot (risk-neutral) drift μ = r − q − σ²/2.  The first-passage
    # probability formula uses this drift directly
    mu = r - q - sigma2 / 2.0
    sigma_sqrt_T = sigma * np.sqrt(T)

    eta = 1.0 if direction is BarrierDirection.DOWN else -1.0
    log_HS = np.log(H / S)

    # Reflection-principle first-passage probability of a Brownian
    # motion with drift: P(barrier hit on [0, T]) =
    #     N(η·a) + (H/S)^{2μ/σ²} · N(η·b)
    # where a = (ln(H/S) − μT)/σ√T  and  b = (ln(H/S) + μT)/σ√T.
    a = (log_HS - mu * T) / sigma_sqrt_T
    b = (log_HS + mu * T) / sigma_sqrt_T
    hit_prob = norm.cdf(eta * a) + (H / S) ** (2.0 * mu / sigma2) * norm.cdf(eta * b)

    return R * df_r * hit_prob


def _rebate_knock_in_at_expiry(
    R: float,
    S: float,
    H: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    df_r: float,
    direction: BarrierDirection,
) -> float:
    """PV of a rebate paid at expiry if a knock-in barrier is never hit.

    No-touch: R · e^{-rT} · (1 − P(τ_B ≤ T)).
    """
    if R == 0.0:
        return 0.0

    hit_pv = _rebate_knock_out_at_expiry(R, S, H, r, q, sigma, T, df_r, direction)
    return R * df_r - hit_pv


# ── Reiner-Rubinstein building-block terms ────────────────────────


def _barrier_formula_terms(
    S: float,
    K: float,
    H: float,
    sigma: float,
    T: float,
    df_r: float,
    df_q: float,
    phi: float,
    eta: float,
    lam: float,
) -> tuple[float, float, float, float]:
    """Compute the four Reiner-Rubinstein building-block terms A, B, C, D.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike.
    H : float
        Barrier level.
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    df_r : float
        Discount factor e^{-rT}.
    df_q : float
        Dividend discount factor e^{-qT}.
    phi : float
        +1 for call, −1 for put.
    eta : float
        +1 for down barrier, −1 for up barrier.
    lam : float
        λ = (r − q + σ²/2) / σ².

    Returns
    -------
    tuple[float, float, float, float]
        Terms (A, B, C, D).
    """
    sigma_sqrt_T = sigma * np.sqrt(T)

    d1 = np.log(S / K) / sigma_sqrt_T + lam * sigma_sqrt_T
    x1 = np.log(S / H) / sigma_sqrt_T + lam * sigma_sqrt_T
    y = np.log(H**2 / (S * K)) / sigma_sqrt_T + lam * sigma_sqrt_T
    y1 = np.log(H / S) / sigma_sqrt_T + lam * sigma_sqrt_T

    # Term A: vanilla-like
    A = phi * S * df_q * norm.cdf(phi * d1) - phi * K * df_r * norm.cdf(phi * (d1 - sigma_sqrt_T))

    # Term B: truncated at barrier
    B = phi * S * df_q * norm.cdf(phi * x1) - phi * K * df_r * norm.cdf(phi * (x1 - sigma_sqrt_T))

    # Term C: reflected vanilla
    C = phi * S * df_q * (H / S) ** (2.0 * lam) * norm.cdf(eta * y) - phi * K * df_r * (H / S) ** (
        2.0 * lam - 2.0
    ) * norm.cdf(eta * (y - sigma_sqrt_T))

    # Term D: reflected truncated
    D = phi * S * df_q * (H / S) ** (2.0 * lam) * norm.cdf(eta * y1) - phi * K * df_r * (H / S) ** (
        2.0 * lam - 2.0
    ) * norm.cdf(eta * (y1 - sigma_sqrt_T))

    return A, B, C, D


def _barrier_price_no_rebate(
    S: float,
    K: float,
    H: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    df_r: float,
    df_q: float,
    option_type: OptionType,
    direction: BarrierDirection,
    action: BarrierAction,
) -> float:
    """Analytical no-rebate barrier option price from the Reiner-Rubinstein formulas.

    Dispatches to the correct A/B/C/D combination based on the 16-case table.

    Parameters
    ----------
    S, K, H : float
        Spot, strike, barrier.
    r, q : float
        Risk-free rate and dividend yield (flat-equivalent).
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    df_r : float
        Discount factor at maturity.
    df_q : float
        Dividend discount factor at maturity.
    option_type : OptionType
        CALL or PUT.
    direction : BarrierDirection
        UP or DOWN.
    action : BarrierAction
        IN or OUT.

    Returns
    -------
    float
        Barrier option price (may be negative before flooring, caller should floor).
    """
    phi = 1.0 if option_type is OptionType.CALL else -1.0
    eta = 1.0 if direction is BarrierDirection.DOWN else -1.0
    sigma2 = sigma**2
    lam = (r - q + sigma2 / 2.0) / sigma2

    A, B, C, D = _barrier_formula_terms(S, K, H, sigma, T, df_r, df_q, phi, eta, lam)

    is_call = option_type is OptionType.CALL
    is_down = direction is BarrierDirection.DOWN
    is_in = action is BarrierAction.IN

    if is_down and is_in and is_call:
        # Down-and-in call
        return C if H <= K else A - B + D

    if is_down and not is_in and is_call:
        # Down-and-out call
        return A - C if H <= K else B - D

    if not is_down and is_in and is_call:
        # Up-and-in call
        return A if H <= K else B - C + D

    if not is_down and not is_in and is_call:
        # Up-and-out call
        return 0.0 if H <= K else A - B + C - D

    if is_down and is_in and not is_call:
        # Down-and-in put
        return B - C + D if H <= K else A

    if is_down and not is_in and not is_call:
        # Down-and-out put
        return A - B + C - D if H <= K else 0.0

    if not is_down and is_in and not is_call:
        # Up-and-in put
        return A - B + D if H <= K else C

    # Up-and-out put
    return B - D if H <= K else A - C


# ── Engine class ──────────────────────────────────────────────────


class _AnalyticalBarrierValuation:
    """Analytical barrier option valuation engine.

    Dispatched by ``OptionValuation`` when ``spec`` is :class:`BarrierSpec` and
    ``pricing_method`` is ``BSM``.
    """

    def __init__(self, valuation_ctx: OptionValuation) -> None:
        self.valuation_ctx = valuation_ctx
        self.underlying: UnderlyingData = valuation_ctx.underlying  # type: ignore[assignment]
        self.spec: BarrierSpec = valuation_ctx.spec  # type: ignore[assignment]

        if self.underlying.discrete_dividends:
            raise UnsupportedFeatureError(
                "Analytical barrier formula does not support discrete dividends. Use PDE_FD "
                "or MONTE_CARLO."
            )

    def solve(self) -> float:
        """Return the analytical barrier option value."""
        return self.present_value()

    def theta(self) -> float:
        r"""Barrier theta via the Black-Scholes PDE identity.

        .. math::

            \Theta = r V - (r - q) S \Delta - \tfrac{1}{2} \sigma^{2} S^{2} \Gamma

        ``V`` is the closed-form barrier price; ``Δ`` and ``Γ`` come from
        central-difference bump-and-revalue around the same closed-form
        evaluator (routed through :attr:`valuation_ctx` so repeated calls
        hit the OV-level cache).  The identity is exact in the
        continuation region (triggered-at-inception cases already
        short-circuit in :meth:`present_value`) and delivers better
        accuracy than a naive forward-difference time bump.

        Returned per **calendar day**
        """
        ctx = self.valuation_ctx
        underlying = self.underlying

        S = float(underlying.initial_value)
        sigma = float(underlying.volatility)
        sigma2 = sigma**2
        T = ctx._maturity_year_fraction()

        df_r = float(ctx.discount_curve.df(T))
        r = -np.log(df_r) / T
        dividend_curve = underlying.dividend_curve
        df_q = float(dividend_curve.df(T)) if dividend_curve is not None else 1.0
        q = -np.log(df_q) / T

        V = ctx.present_value()
        delta = ctx.delta()
        gamma = ctx.gamma()

        theta_annual = r * V - (r - q) * S * delta - 0.5 * sigma2 * S * S * gamma
        return float(theta_annual / 365.0)

    def present_value(self) -> float:
        """Compute the analytical barrier option price."""
        spec = self.spec
        underlying = self.underlying
        ctx = self.valuation_ctx

        S = float(underlying.initial_value)
        K = float(spec.strike)
        H = float(spec.barrier)
        sigma = float(underlying.volatility)

        T = self.valuation_ctx._maturity_year_fraction()

        # Discount factors from curves
        df_r = float(ctx.discount_curve.df(T))
        dividend_curve = underlying.dividend_curve
        df_q = float(dividend_curve.df(T)) if dividend_curve is not None else 1.0

        # Flat-equivalent rates
        r = -np.log(df_r) / T
        q = -np.log(df_q) / T

        # ── Check if barrier already triggered at inception ──
        if ctx._barrier_triggered_at_inception():
            return self._triggered_at_inception(df_r, spec)

        # ── Discrete monitoring: Broadie-Glasserman adjustment ──
        if spec.monitoring is BarrierMonitoring.DISCRETE:
            if spec.num_observations is None:
                raise UnsupportedFeatureError(
                    "Analytical barrier pricing with DISCRETE monitoring requires "
                    "num_observations (equally spaced). Use BINOMIAL, PDE_FD or MONTE_CARLO for "
                    "explicit monitoring_dates."
                )
            H = _broadie_glasserman_adjustment(H, sigma, T, spec.num_observations, spec.direction)

        # ── No-rebate barrier value ──
        value = _barrier_price_no_rebate(
            S,
            K,
            H,
            r,
            q,
            sigma,
            T,
            df_r,
            df_q,
            spec.option_type,
            spec.direction,
            spec.action,
        )

        # ── Add rebate leg ──
        if spec.rebate > 0.0:
            value += self._rebate_pv(
                spec.rebate,
                S,
                H,
                r,
                q,
                sigma,
                T,
                df_r,
                spec.action,
                spec.direction,
                spec.rebate_timing,
            )

        return max(value, 0.0)

    @staticmethod
    def _rebate_pv(
        R: float,
        S: float,
        H: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        df_r: float,
        action: BarrierAction,
        direction: BarrierDirection,
        timing: RebateTiming,
    ) -> float:
        """Compute the rebate component's present value."""
        if action is BarrierAction.OUT:
            if timing is RebateTiming.AT_HIT:
                return _rebate_knock_out_at_hit(R, S, H, r, q, sigma, T, direction)
            return _rebate_knock_out_at_expiry(R, S, H, r, q, sigma, T, df_r, direction)

        # Knock-in: rebate paid at expiry if barrier is never hit
        return _rebate_knock_in_at_expiry(R, S, H, r, q, sigma, T, df_r, direction)

    def _triggered_at_inception(
        self,
        df_r: float,
        spec: BarrierSpec,
    ) -> float:
        """Handle the case where the barrier is already breached at time zero."""
        if spec.action is BarrierAction.OUT:
            # Knocked out immediately — return PV of rebate only
            if spec.rebate == 0.0:
                return 0.0
            if spec.rebate_timing is RebateTiming.AT_HIT:
                return spec.rebate  # paid immediately
            return spec.rebate * df_r  # paid at expiry

        # Knock-in: already activated → price as vanilla
        vanilla_spec = VanillaSpec(
            option_type=spec.option_type,
            exercise_type=spec.exercise_type,
            strike=spec.strike,
            maturity=spec.maturity,
            currency=spec.currency,
            contract_size=spec.contract_size,
        )
        from .core import OptionValuation

        return OptionValuation(
            underlying=self.underlying,
            spec=vanilla_spec,
            pricing_method=PricingMethod.BSM,
        ).present_value()
